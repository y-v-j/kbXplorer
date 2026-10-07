#!/usr/bin/env python3
"""Command-line entry point for the local research knowledge base.

Subcommands
-----------
``ingest``   Scan the corpus and index new or changed papers.
``watch``    Ingest, then keep watching the folder for changes.
``query``    Ask one question and print a cited answer (no TUI).
``tui``      Launch the dual-pane terminal interface (default).
``status``   Show index statistics and backend health.
``doctor``   Diagnose the environment: Ollama, models, memory, disk.
``reindex``  Rebuild the vector store from scratch.
``reset``    Delete the index (and optionally extracted figures).

Run ``python main.py <subcommand> --help`` for per-command options.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path

from engine.config import PROJECT_ROOT, Settings, configure_logging, load_settings
from engine.locking import IngestLockBusy

LOGGER = logging.getLogger("main")

#: Progress callback signature: (stage, message, current, total).
#: Mirrors ingest.ProgressCallback without importing it, so `--help` does
#: not pay the cost of loading ChromaDB.
ProgressCallback = Callable[[str, str, int, int], None]

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INTERRUPTED = 130


# --------------------------------------------------------------------- utils
def _human_bytes(count: float) -> str:
    """Format a byte count for display."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if count < 1024 or unit == "TB":
            return f"{count:.1f} {unit}"
        count /= 1024
    return f"{count:.1f} TB"


def _progress_printer() -> ProgressCallback:
    """Return a progress callback that prints a single updating line."""
    state = {"last": 0.0}

    def report(stage: str, message: str, current: int, total: int) -> None:
        if stage == "ingest":
            now = time.monotonic()
            if now - state["last"] < 0.1 and current != total:
                return
            state["last"] = now
            bar_width = 28
            filled = int(bar_width * current / total) if total else 0
            bar = "█" * filled + "░" * (bar_width - filled)
            label = message[:44].ljust(44)
            print(f"\r  [{bar}] {current:>4}/{total} {label}", end="", flush=True)
        elif stage == "complete":
            print(f"\r  {' ' * 88}\r  {message}")
        elif stage in {"scan", "recover"}:
            print(f"  {message}")

    return report


# ------------------------------------------------------------------ commands
def cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    """Index new or changed papers."""
    from ingest import ReindexSelector, build_service

    selector = ReindexSelector(pattern=args.only)
    service, _store, manifest = build_service(settings, _progress_printer())
    try:
        print(f"Corpus: {settings.pdf_dir}")
        if args.dry_run:
            report = service.sync(force=args.force, selector=selector, dry_run=True)
            if not report.planned:
                print("  Nothing to index — everything is up to date.")
                return EXIT_OK
            print(f"  Would index {len(report.planned)} document(s):")
            for name in report.planned[:60]:
                print(f"    - {name}")
            if len(report.planned) > 60:
                print(f"    … and {len(report.planned) - 60} more")
            return EXIT_OK
        report = service.sync(force=args.force, prune=not args.no_prune, selector=selector)
        if report.errors:
            print("\n  Problems:")
            for name, message in report.errors[:20]:
                print(f"    - {name}: {message}")
        return EXIT_OK if report.failed == 0 else EXIT_ERROR
    finally:
        manifest.close()


def cmd_watch(args: argparse.Namespace, settings: Settings) -> int:
    """Ingest, then watch the folder until interrupted."""
    from ingest import build_service
    from watcher import FolderWatcher

    service, _store, manifest = build_service(settings, _progress_printer())

    def announce(event: str, message: str) -> None:
        print(f"  [{event}] {message}")

    watcher = FolderWatcher(settings, service, announce)
    try:
        watcher.start(initial_sync=not args.no_initial_sync)
        print("\nWatching for changes. Press Ctrl+C to stop.")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping…")
        return EXIT_INTERRUPTED
    finally:
        watcher.stop()
        manifest.close()


def cmd_query(args: argparse.Namespace, settings: Settings) -> int:
    """Answer one question on stdout."""
    from engine.ollama_client import OllamaConnector
    from engine.rag_engine import RagEngine
    from engine.vector_store import VectorStore

    store = VectorStore(settings)
    connector = OllamaConnector(settings)
    engine = RagEngine(settings, store, connector)

    question = " ".join(args.question).strip()
    if not question:
        print("No question supplied.", file=sys.stderr)
        return EXIT_ERROR

    print(f"\n\033[1mQ:\033[0m {question}\n")
    result = None
    try:
        for event in engine.answer_stream(question):
            if event.kind == "token":
                print(event.text, end="", flush=True)
            elif event.kind == "done":
                result = event.result
            elif event.kind == "error":
                print(f"\nError: {event.text}", file=sys.stderr)
                return EXIT_ERROR
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return EXIT_INTERRUPTED

    print("\n")
    if result is None:
        return EXIT_ERROR

    if result.citations:
        print("\033[1mSources\033[0m")
        for index, citation in enumerate(result.citations, start=1):
            mark = "\033[32m✓\033[0m" if citation.verified else "\033[33m?\033[0m"
            print(f"  {mark} {index}. {citation.label()}  [{citation.content_type}]")
            if citation.image_path:
                print(f"      image: {citation.image_path}")

    for warning in result.warnings:
        print(f"\033[33m!\033[0m {warning}")

    print(
        f"\n{result.elapsed_seconds:.1f}s · {result.model or 'no model'} · "
        f"{len(result.verified_citations)}/{len(result.citations)} citations verified"
    )
    return EXIT_OK


def cmd_tui(args: argparse.Namespace, settings: Settings) -> int:
    """Launch the dual-pane Textual interface."""
    from engine.ollama_client import OllamaConnector
    from engine.rag_engine import RagEngine
    from ingest import build_service
    from tui.app import KnowledgeBaseApp

    from engine.registry import Registry

    configure_logging(settings, quiet=True)
    registry = Registry.load(settings)
    if args.kb:
        registry.set_active(args.kb)
    service, store, manifest = build_service(settings)
    connector = OllamaConnector(settings)
    engine = RagEngine(settings, store, connector)
    app = KnowledgeBaseApp(settings, engine, connector, store, service, registry, manifest)

    watcher = None
    if args.watch:
        from watcher import FolderWatcher

        watcher = FolderWatcher(settings, service, app.on_watcher_event)
        watcher.start(initial_sync=False)

    try:
        app.run()
    finally:
        if watcher is not None:
            watcher.stop()
        # The app may have swapped in a different manifest via Ctrl+K.
        current = getattr(app, "_manifest", manifest) or manifest
        try:
            current.close()
        except Exception:  # pragma: no cover - best effort
            LOGGER.debug("Manifest close failed", exc_info=True)
    return EXIT_OK


def cmd_status(args: argparse.Namespace, settings: Settings) -> int:
    """Print index and backend status."""
    from engine.ollama_client import OllamaConnector
    from ingest import build_service, discover_pdfs

    service, _store, manifest = build_service(settings)
    try:
        status = service.status()
        on_disk = len(discover_pdfs(settings))

        print("\n\033[1mCorpus\033[0m")
        print(f"  directory         {status['pdf_dir']}")
        print(f"  PDFs on disk      {on_disk}")
        print(f"  documents indexed {status['documents_indexed']}  ({status['pages']} pages)")

        print("\n\033[1mIndex\033[0m")
        print(f"  passage vectors   {status['vectors_chunks']}")
        print(f"  figure vectors    {status['vectors_figures']}")
        print(f"  tables extracted  {status['tables']}")
        print(f"  embedding         {status['embedding']}")
        print(f"  chroma path       {settings.chroma_dir}")
        print(f"  by status         {status['by_status']}")

        images = list(settings.images_dir.glob("*")) if settings.images_dir.is_dir() else []
        size = sum(p.stat().st_size for p in images if p.is_file())
        print(f"  extracted images  {len(images)} ({_human_bytes(size)})")

        print("\n\033[1mLLM\033[0m")
        health = OllamaConnector(settings).health()
        print(f"  {health.describe()}")
        if health.available and health.models:
            print(f"  installed         {', '.join(health.models)}")

        failed = manifest.by_status("failed")
        if failed:
            print(f"\n\033[33mFailed documents ({len(failed)})\033[0m")
            for record in failed[:15]:
                print(f"  - {record.document_name}: {record.error[:90]}")
        print()
        return EXIT_OK
    finally:
        manifest.close()


def cmd_doctor(args: argparse.Namespace, settings: Settings) -> int:
    """Diagnose the runtime environment."""
    print("\n\033[1mEnvironment check\033[0m\n")
    ok = True

    print(f"  python            {sys.version.split()[0]}")
    if sys.version_info < (3, 11):
        print("    \033[31m✗\033[0m Python 3.12 is recommended")
        ok = False

    for module in ("pymupdf", "pdfplumber", "chromadb", "textual", "ollama", "watchdog", "onnxruntime"):
        try:
            imported = __import__(module)
            version = getattr(imported, "__version__", "?")
            print(f"  \033[32m✓\033[0m {module:<16} {version}")
        except ImportError:
            print(f"  \033[31m✗\033[0m {module:<16} NOT INSTALLED")
            ok = False

    print()
    for label, path in (
        ("corpus", settings.pdf_dir),
        ("chroma", settings.chroma_dir),
        ("images", settings.images_dir),
        ("state", settings.state_dir),
    ):
        mark = "\033[32m✓\033[0m" if path.exists() else "\033[33m?\033[0m"
        print(f"  {mark} {label:<16} {path}")

    usage = shutil.disk_usage(settings.chroma_dir if settings.chroma_dir.exists() else PROJECT_ROOT)
    print(f"\n  disk free         {_human_bytes(usage.free)} of {_human_bytes(usage.total)}")
    if usage.free < 5 * 1024**3:
        print("    \033[33m!\033[0m Less than 5 GB free")

    try:
        meminfo = dict(
            (parts[0].rstrip(":"), int(parts[1]))
            for line in Path("/proc/meminfo").read_text().splitlines()
            if (parts := line.split())
        )
        total = meminfo.get("MemTotal", 0) * 1024
        available = meminfo.get("MemAvailable", 0) * 1024
        swap_total = meminfo.get("SwapTotal", 0) * 1024
        swap_free = meminfo.get("SwapFree", 0) * 1024
        print(f"  RAM available     {_human_bytes(available)} of {_human_bytes(total)}")
        if swap_total:
            print(f"  swap free         {_human_bytes(swap_free)} of {_human_bytes(swap_total)}")
        if available < 3 * 1024**3:
            print(
                "    \033[33m!\033[0m Under 3 GB free. A 3B model needs ~2.5 GB; close "
                "memory-heavy apps or answers will be very slow."
            )
        if swap_total and swap_free < swap_total * 0.05:
            print(
                "    \033[31m✗\033[0m Swap is essentially exhausted. The system is "
                "thrashing and generation will take minutes per answer."
            )
            ok = False
    except OSError:
        pass

    print("\n\033[1mOllama\033[0m")
    from engine.ollama_client import OllamaConnector

    health = OllamaConnector(settings).health()
    print(f"  {health.describe()}")
    if health.available:
        print(f"  installed         {', '.join(health.models) or 'none'}")
        if not health.model_installed:
            print(f"    \033[33m!\033[0m Configured model {settings.ollama_model!r} is not installed")
            print(f"      Run: ollama pull {settings.ollama_model}")
    else:
        print("    \033[33m!\033[0m Start it with: ollama serve")
        ok = False

    print(f"\n{'All checks passed.' if ok else 'Some checks need attention.'}\n")
    return EXIT_OK if ok else EXIT_ERROR


def cmd_reindex(args: argparse.Namespace, settings: Settings) -> int:
    """Re-ingest documents that are already indexed.

    Use this after changing a parser setting, upgrading the parser, or when a
    paper was indexed badly. Selectors narrow the work so a single bad document
    does not cost a full corpus rebuild.
    """
    from ingest import ReindexSelector, build_service

    selector = ReindexSelector(
        pattern=args.only,
        failed_only=args.failed,
        stale_only=args.stale,
    )
    service, store, manifest = build_service(settings, _progress_printer())
    try:
        print(f"Corpus:   {settings.pdf_dir}")
        print(f"Selection: {selector.describe()}")

        if args.dry_run:
            report = service.sync(force=True, selector=selector, dry_run=True)
            if not report.planned:
                print("\n  Nothing to re-ingest.")
                return EXIT_OK
            print(f"\n  Would re-ingest {len(report.planned)} document(s):")
            for name in report.planned[:60]:
                print(f"    - {name}")
            if len(report.planned) > 60:
                print(f"    … and {len(report.planned) - 60} more")
            print("\n  Re-run without --dry-run to apply.")
            return EXIT_OK

        if args.rebuild:
            if selector.is_narrowed:
                print("\n--rebuild drops the whole store and cannot be combined "
                      "with a selector.", file=sys.stderr)
                return EXIT_ERROR
            if not args.yes:
                reply = input("Drop the entire vector store and rebuild? [y/N] ").strip().lower()
                if reply not in {"y", "yes"}:
                    print("Cancelled.")
                    return EXIT_OK
            print("Clearing vector store…")
            store.reset()
            for record in manifest.all_documents():
                manifest.delete(record.doc_uid)

        report = service.sync(force=True, selector=selector)
        if report.errors:
            print("\n  Problems:")
            for name, message in report.errors[:20]:
                print(f"    - {name}: {message}")
        return EXIT_OK if report.failed == 0 else EXIT_ERROR
    finally:
        manifest.close()


def cmd_kb(args: argparse.Namespace, settings: Settings) -> int:
    """Manage the set of registered knowledge bases."""
    from engine.registry import KnowledgeBaseRef, Registry, RegistryError

    registry = Registry.load(settings)
    action = args.kb_action

    if action == "list":
        print(f"\nRegistered knowledge bases ({len(registry)}):\n")
        for ref in registry.all():
            mark = "\033[32m●\033[0m" if ref.name == registry.active_name else " "
            missing = "" if ref.exists() else "  \033[31m(corpus folder missing)\033[0m"
            print(f"  {mark} \033[1m{ref.name}\033[0m{missing}")
            print(f"      corpus   {ref.pdf_dir}  ({ref.pdf_count()} PDFs)")
            print(f"      vectors  {ref.chroma_dir}")
            if ref.description:
                print(f"      note     {ref.description}")
            print()
        print("  Switch with: python main.py kb use <name>   (or Ctrl+K in the TUI)\n")
        return EXIT_OK

    if action == "add":
        pdf_dir = args.pdf_dir_arg.expanduser()
        if not pdf_dir.is_dir():
            print(f"\n\033[31mError:\033[0m corpus directory does not exist: {pdf_dir}",
                  file=sys.stderr)
            return EXIT_ERROR
        ref = KnowledgeBaseRef.derived(args.name, pdf_dir, args.description or "")
        try:
            registry.add(ref, replace_existing=args.force)
        except RegistryError as exc:
            print(f"\n\033[31mError:\033[0m {exc}", file=sys.stderr)
            return EXIT_ERROR
        registry.save()
        print(f"\nAdded knowledge base '\033[1m{ref.name}\033[0m'")
        print(f"  corpus   {ref.pdf_dir}  ({ref.pdf_count()} PDFs)")
        print(f"  data     {ref.chroma_dir.parent}")
        print(f"\nIndex it with:  python main.py --kb {ref.name} ingest\n")
        return EXIT_OK

    if action == "remove":
        try:
            ref = registry.remove(args.name)
        except RegistryError as exc:
            print(f"\n\033[31mError:\033[0m {exc}", file=sys.stderr)
            return EXIT_ERROR
        registry.save()
        print(f"\nUnregistered '{ref.name}'. No files were deleted.")
        print(f"  corpus kept at  {ref.pdf_dir}")
        print(f"  index kept at   {ref.chroma_dir.parent}")
        print("  Delete the index directory by hand if you no longer want it.\n")
        return EXIT_OK

    if action == "use":
        try:
            ref = registry.set_active(args.name)
        except RegistryError as exc:
            print(f"\n\033[31mError:\033[0m {exc}", file=sys.stderr)
            return EXIT_ERROR
        registry.save()
        print(f"\nActive knowledge base is now '\033[1m{ref.name}\033[0m' ({ref.pdf_dir})\n")
        return EXIT_OK

    print("Specify one of: list, add, remove, use", file=sys.stderr)
    return EXIT_ERROR


def cmd_reset(args: argparse.Namespace, settings: Settings) -> int:
    """Delete the index, and optionally the extracted figures."""
    targets = [settings.chroma_dir, settings.state_dir]
    if args.images:
        targets.append(settings.images_dir)

    print("This will delete:")
    for path in targets:
        print(f"  - {path}")
    if not args.yes:
        reply = input("Proceed? [y/N] ").strip().lower()
        if reply not in {"y", "yes"}:
            print("Cancelled.")
            return EXIT_OK

    for path in targets:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
            print(f"  removed {path}")
    settings.ensure_directories()
    print("Reset complete.")
    return EXIT_OK


# ------------------------------------------------------------------- parsing
def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Local, citation-grounded knowledge base over a folder of PDF papers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py ingest              # index new or changed papers\n"
            "  python main.py tui                 # launch the dual-pane interface\n"
            "  python main.py tui --watch         # …and auto-index folder changes\n"
            '  python main.py query "what sample size did the study use?"\n'
            "  python main.py doctor              # diagnose the environment\n"
            "  python main.py kb list             # show knowledge bases\n"
            '  python main.py kb add genomics --pdf-dir ~/papers/genomics\n'
            "  python main.py --kb genomics ingest\n"
            "  python main.py reindex --stale     # re-parse after a parser upgrade\n"
            '  python main.py reindex --only "study_a*"   # re-parse one paper\n'
        ),
    )
    parser.add_argument("--kb", metavar="NAME", help="Use a specific knowledge base for this command")
    parser.add_argument("--pdf-dir", type=Path, help="Override the corpus directory")
    parser.add_argument("--model", help="Override the Ollama model")
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING, ERROR")
    sub = parser.add_subparsers(dest="command")

    p_ingest = sub.add_parser("ingest", help="Index new or changed papers")
    p_ingest.add_argument("--force", action="store_true", help="Re-index even unchanged files")
    p_ingest.add_argument("--no-prune", action="store_true", help="Keep vectors for deleted files")
    p_ingest.add_argument("--only", metavar="PATTERN", help="Limit to filenames matching a glob or substring")
    p_ingest.add_argument("--dry-run", action="store_true", help="Show what would be indexed, change nothing")
    p_ingest.set_defaults(func=cmd_ingest)

    p_watch = sub.add_parser("watch", help="Ingest, then watch the folder")
    p_watch.add_argument("--no-initial-sync", action="store_true", help="Skip the startup scan")
    p_watch.set_defaults(func=cmd_watch)

    p_query = sub.add_parser("query", help="Ask one question and print the answer")
    p_query.add_argument("question", nargs="+", help="The question to ask")
    p_query.set_defaults(func=cmd_query)

    p_tui = sub.add_parser("tui", help="Launch the dual-pane interface (default)")
    p_tui.add_argument("--watch", action="store_true", help="Also watch the folder for changes")
    p_tui.set_defaults(func=cmd_tui)

    sub.add_parser("status", help="Show index statistics").set_defaults(func=cmd_status)
    sub.add_parser("doctor", help="Diagnose the environment").set_defaults(func=cmd_doctor)
    p_reindex = sub.add_parser(
        "reindex",
        help="Re-ingest already-indexed papers (after a parser or setting change)",
        description=(
            "Re-ingest documents that are already indexed. Without a selector "
            "every document is re-parsed in place; selectors narrow the work. "
            "Selectors combine with OR."
        ),
    )
    p_reindex.add_argument("--only", metavar="PATTERN", help="Limit to filenames matching a glob or substring")
    p_reindex.add_argument("--failed", action="store_true", help="Limit to documents that previously failed")
    p_reindex.add_argument(
        "--stale", action="store_true",
        help="Limit to documents parsed by a different PARSER_VERSION",
    )
    p_reindex.add_argument("--dry-run", action="store_true", help="List what would be re-ingested, change nothing")
    p_reindex.add_argument(
        "--rebuild", action="store_true",
        help="Drop the entire vector store first (needed after an embedding-backend change)",
    )
    p_reindex.add_argument("--yes", action="store_true", help="Do not prompt for --rebuild confirmation")
    p_reindex.set_defaults(func=cmd_reindex)

    p_kb = sub.add_parser("kb", help="Manage multiple knowledge bases")
    kb_sub = p_kb.add_subparsers(dest="kb_action")
    kb_sub.add_parser("list", help="List registered knowledge bases")
    p_kb_add = kb_sub.add_parser("add", help="Register a new knowledge base")
    p_kb_add.add_argument("name", help="Short name, e.g. genomics")
    p_kb_add.add_argument("--pdf-dir", dest="pdf_dir_arg", type=Path, required=True,
                          metavar="PATH", help="Folder holding that corpus's PDFs")
    p_kb_add.add_argument("--description", help="Optional note shown in the picker")
    p_kb_add.add_argument("--force", action="store_true", help="Overwrite an existing entry")
    p_kb_remove = kb_sub.add_parser("remove", help="Unregister (does not delete files)")
    p_kb_remove.add_argument("name")
    p_kb_use = kb_sub.add_parser("use", help="Set the default knowledge base")
    p_kb_use.add_argument("name")
    p_kb.set_defaults(func=cmd_kb)

    p_reset = sub.add_parser("reset", help="Delete the index")
    p_reset.add_argument("--images", action="store_true", help="Also delete extracted figures")
    p_reset.add_argument("--yes", action="store_true", help="Do not prompt for confirmation")
    p_reset.set_defaults(func=cmd_reset)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    settings = load_settings()

    # Resolve the active knowledge base before any other path override.
    if args.command != "kb":
        from engine.registry import Registry, RegistryError

        registry = Registry.load(settings)
        try:
            ref = registry.get(args.kb) if args.kb else registry.active
        except RegistryError as exc:
            print(f"\n\033[31mError:\033[0m {exc}", file=sys.stderr)
            return EXIT_ERROR
        settings = ref.apply(settings)

    if args.pdf_dir:
        settings.pdf_dir = args.pdf_dir.expanduser().resolve()
    if args.model:
        settings.ollama_model = args.model
    if args.log_level:
        settings.log_level = args.log_level
    settings.ensure_directories()

    if args.command != "tui":
        configure_logging(settings)

    handler = getattr(args, "func", None)
    if handler is None:
        args.watch = False
        return cmd_tui(args, settings)

    try:
        return handler(args, settings)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return EXIT_INTERRUPTED
    except IngestLockBusy as exc:
        print(f"\n\033[33mBusy:\033[0m {exc}", file=sys.stderr)
        return EXIT_ERROR
    except Exception as exc:
        LOGGER.exception("Command failed")
        print(f"\n\033[31mError:\033[0m {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
