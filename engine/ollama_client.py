"""Ollama connector with graceful degradation.

The knowledge base must stay usable when Ollama is not running: retrieval and
citations still work, only answer synthesis is unavailable. Every entry point
therefore reports status rather than raising, and :meth:`OllamaConnector.health`
is cheap enough to call on every UI refresh.

Reasoning models (qwen3.x, deepseek-r1, ...) emit ``<think>`` blocks that would
otherwise stream into the chat pane and stall time-to-first-token. When
``strip_reasoning_tokens`` is set, those spans are filtered out of the stream.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

from engine.config import Settings

LOGGER = logging.getLogger(__name__)

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


class OllamaUnavailableError(RuntimeError):
    """Raised when a generation is attempted but the Ollama service is down."""


@dataclass(slots=True)
class OllamaStatus:
    """Result of a health probe against the Ollama daemon."""

    available: bool
    models: list[str] = field(default_factory=list)
    active_model: str = ""
    error: str = ""

    @property
    def model_installed(self) -> bool:
        return bool(self.active_model) and self.active_model in self.models

    def describe(self) -> str:
        """Return a one-line human-readable summary."""
        if not self.available:
            return f"Ollama unavailable ({self.error or 'not running'})"
        if not self.models:
            return "Ollama running, but no models installed (try: ollama pull llama3.2:3b)"
        return f"Ollama ready - model {self.active_model}"


class OllamaConnector:
    """Wraps the ``ollama`` Python client with health checks and streaming."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = None
        self._resolved_model: str | None = None

    # ------------------------------------------------------------ lifecycle
    def _get_client(self):
        """Lazily construct the Ollama client (import is not free)."""
        if self._client is None:
            try:
                import ollama
            except ImportError as exc:  # pragma: no cover
                raise OllamaUnavailableError(f"ollama package not installed: {exc}") from exc
            self._client = ollama.Client(host=self._settings.ollama_host)
        return self._client

    def list_models(self) -> list[str]:
        """Return the model names installed on the local daemon."""
        client = self._get_client()
        payload = client.list()
        models = getattr(payload, "models", None)
        if models is None and isinstance(payload, dict):
            models = payload.get("models", [])
        names: list[str] = []
        for entry in models or []:
            name = getattr(entry, "model", None) or getattr(entry, "name", None)
            if name is None and isinstance(entry, dict):
                name = entry.get("model") or entry.get("name")
            if name:
                names.append(str(name))
        return names

    def health(self) -> OllamaStatus:
        """Probe the daemon and resolve which model will be used.

        Never raises — a down service is reported, not thrown.
        """
        try:
            models = self.list_models()
        except Exception as exc:
            return OllamaStatus(available=False, error=str(exc).split("\n")[0][:200])

        active = self.resolve_model(models)
        return OllamaStatus(available=True, models=models, active_model=active)

    def resolve_model(self, models: list[str] | None = None) -> str:
        """Pick the model to use, falling back when the configured one is absent.

        Preference order: the configured model, then each entry of
        ``ollama_fallback_models`` that is installed, then any installed model.
        """
        if models is None:
            try:
                models = self.list_models()
            except Exception:
                return self._settings.ollama_model

        configured = self._settings.ollama_model
        if configured in models:
            return configured

        # Tolerate a missing ':latest' suffix on either side.
        bare = {m.split(":")[0]: m for m in models}
        if configured.split(":")[0] in bare and ":" not in configured:
            return bare[configured.split(":")[0]]

        for candidate in self._settings.ollama_fallback_models:
            if candidate in models:
                LOGGER.warning("Model %s not installed; falling back to %s", configured, candidate)
                return candidate

        if models:
            LOGGER.warning("Model %s not installed; falling back to %s", configured, models[0])
            return models[0]
        return configured

    def pull(self, model: str) -> Iterator[str]:
        """Pull a model, yielding human-readable progress lines."""
        client = self._get_client()
        try:
            for update in client.pull(model, stream=True):
                status = getattr(update, "status", None)
                if status is None and isinstance(update, dict):
                    status = update.get("status")
                if status:
                    yield str(status)
        except Exception as exc:
            raise OllamaUnavailableError(f"pull of {model!r} failed: {exc}") from exc

    # ----------------------------------------------------------- generation
    def stream_chat(self, system_prompt: str, user_prompt: str, *, model: str | None = None) -> Iterator[str]:
        """Stream an answer token by token.

        Args:
            system_prompt: System role content.
            user_prompt: User role content (question plus retrieved context).
            model: Override the resolved model.

        Yields:
            Text fragments with any reasoning blocks removed.

        Raises:
            OllamaUnavailableError: If the daemon is unreachable mid-stream.
        """
        client = self._get_client()
        target = model or self.resolve_model()

        options = {
            "temperature": self._settings.ollama_temperature,
            "num_ctx": self._settings.ollama_num_ctx,
            "num_predict": self._settings.ollama_num_predict,
        }

        try:
            stream = client.chat(
                model=target,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=True,
                options=options,
                keep_alive=self._settings.ollama_keep_alive,
            )
            filt = _ThinkFilter(enabled=self._settings.strip_reasoning_tokens)
            for part in stream:
                message = getattr(part, "message", None)
                if message is None and isinstance(part, dict):
                    message = part.get("message")
                content = getattr(message, "content", None)
                if content is None and isinstance(message, dict):
                    content = message.get("content")
                if not content:
                    continue
                emitted = filt.feed(str(content))
                if emitted:
                    yield emitted
            tail = filt.flush()
            if tail:
                yield tail
        except OllamaUnavailableError:
            raise
        except Exception as exc:
            raise OllamaUnavailableError(
                f"generation failed on model {target!r}: {str(exc).splitlines()[0]}"
            ) from exc

    def complete(self, system_prompt: str, user_prompt: str, *, model: str | None = None) -> str:
        """Non-streaming convenience wrapper around :meth:`stream_chat`."""
        return "".join(self.stream_chat(system_prompt, user_prompt, model=model))


class _ThinkFilter:
    """Incrementally strips ``<think>...</think>`` spans from a token stream."""

    def __init__(self, *, enabled: bool) -> None:
        self._enabled = enabled
        self._buffer = ""
        self._in_think = False

    def feed(self, text: str) -> str:
        """Consume a fragment and return the part safe to display."""
        if not self._enabled:
            return text

        self._buffer += text
        out: list[str] = []

        while self._buffer:
            if self._in_think:
                close = self._buffer.find(_THINK_CLOSE)
                if close == -1:
                    # Keep only enough to recognise a split closing tag.
                    self._buffer = self._buffer[-len(_THINK_CLOSE) :]
                    break
                self._buffer = self._buffer[close + len(_THINK_CLOSE) :]
                self._in_think = False
                continue

            open_at = self._buffer.find(_THINK_OPEN)
            if open_at == -1:
                # Hold back a possible partial opening tag at the tail.
                keep = len(_THINK_OPEN) - 1
                if len(self._buffer) > keep:
                    out.append(self._buffer[:-keep])
                    self._buffer = self._buffer[-keep:]
                break
            out.append(self._buffer[:open_at])
            self._buffer = self._buffer[open_at + len(_THINK_OPEN) :]
            self._in_think = True

        return "".join(out)

    def flush(self) -> str:
        """Return any buffered text once the stream ends."""
        if not self._enabled:
            return ""
        tail = "" if self._in_think else self._buffer
        self._buffer = ""
        return tail
