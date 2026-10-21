// Midnight Ink GitHub Pages theme: turns the rendered README into an app-style page.
// Without JavaScript the page still reads fine; this only adds structure.
(function () {
  "use strict";
  var main = document.getElementById("content");
  if (!main) return;

  // 1. Hero = everything before the first h2; each h2 starts a card (accent colours cycle like the app's cards)
  var nodes = Array.prototype.slice.call(main.childNodes);
  var hero = document.createElement("div");
  hero.className = "hero";
  var current = hero, n = 0;
  main.insertBefore(hero, main.firstChild);
  nodes.forEach(function (node) {
    if (node.nodeType === 1 && node.tagName === "H2") {
      current = document.createElement("section");
      current.className = "card c" + ((n++ % 5) + 1);
      if (node.id) current.setAttribute("aria-labelledby", node.id);
      main.appendChild(current);
    }
    current.appendChild(node);
  });

  // 2. Badges row, hero screenshot and screenshot groups
  hero.querySelectorAll("p").forEach(function (p) {
    var imgs = p.querySelectorAll("img");
    if (imgs.length > 1 && Array.prototype.every.call(imgs, function (i) { return /shields\.io/.test(i.src); })) {
      p.className = "badges";
    }
  });
  var first = hero.querySelector("img:not([src*='shields.io'])");
  if (first) first.classList.add("shot-hero");
  main.querySelectorAll(".card p").forEach(function (p) {
    var imgs = p.querySelectorAll("img");
    if (imgs.length > 1) p.classList.add("shots");
    imgs.forEach(function (i) { i.classList.add("shot"); });
  });

  // 3. Sidebar from the h2s; the README's own "Contents" card is hidden on wide screens
  var toc = document.getElementById("toc");
  var links = [];
  if (toc) {
    var title = document.createElement("div");
    title.className = "toc-title";
    title.textContent = "On this page";
    toc.appendChild(title);
    main.querySelectorAll("section.card").forEach(function (sec) {
      var h = sec.querySelector("h2");
      if (!h || !h.id) return;
      if (/^(table of )?contents$/i.test(h.textContent.trim())) { sec.classList.add("toc-section"); return; }
      var a = document.createElement("a");
      a.href = "#" + h.id;
      a.textContent = h.textContent;
      a.style.setProperty("--c", getComputedStyle(sec).getPropertyValue("--c"));
      toc.appendChild(a);
      links.push({ a: a, sec: sec });
    });
  }
  if ("IntersectionObserver" in window && links.length) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (!e.isIntersecting) return;
        links.forEach(function (l) { l.a.classList.toggle("active", l.sec === e.target); });
      });
    }, { rootMargin: "-30% 0px -60% 0px" });
    links.forEach(function (l) { io.observe(l.sec); });
  }

  // 4. Heading anchors
  main.querySelectorAll("h2[id], h3[id]").forEach(function (h) {
    var a = document.createElement("a");
    a.className = "anchor";
    a.href = "#" + h.id;
    a.setAttribute("aria-label", "Link to this section");
    a.textContent = "#";
    h.appendChild(a);
  });

  // 5. Tables scroll sideways on small screens; screenshot tables lose their empty header row
  main.querySelectorAll("table").forEach(function (t) {
    var ths = t.querySelectorAll("th");
    if (ths.length && Array.prototype.every.call(ths, function (th) { return !th.textContent.trim(); })) {
      t.tHead.style.display = "none";
      t.classList.add("gallery");
    }
    t.querySelectorAll("td img").forEach(function (img) { img.classList.add("shot"); });
    var w = document.createElement("div");
    w.className = "table-wrap";
    t.parentNode.insertBefore(w, t);
    w.appendChild(t);
  });

  // 6. Copy buttons on code blocks
  main.querySelectorAll("div.highlighter-rouge").forEach(function (block) {
    var code = block.querySelector("code");
    if (!code || !navigator.clipboard) return;
    var b = document.createElement("button");
    b.type = "button";
    b.className = "copy";
    b.textContent = "copy";
    b.addEventListener("click", function () {
      navigator.clipboard.writeText(code.innerText.replace(/\n$/, "")).then(function () {
        b.textContent = "copied";
        b.classList.add("done");
        setTimeout(function () { b.textContent = "copy"; b.classList.remove("done"); }, 1500);
      });
    });
    block.appendChild(b);
  });

  // 7. Links to Markdown files that are not part of the site open on GitHub instead
  //    (pages Jekyll built were already rewritten to .html by jekyll-relative-links)
  var repo = document.body.getAttribute("data-repo");
  var branch = document.body.getAttribute("data-branch") || "main";
  var base = document.body.getAttribute("data-base") || "";
  if (repo) {
    main.querySelectorAll("a[href]").forEach(function (a) {
      var href = a.getAttribute("href");
      if (/^[a-z]+:|^#|^\/\//i.test(href) || !/\.(md|markdown)(#|$)/i.test(href)) return;
      var url = new URL(href, location.href);
      var path = url.pathname.replace(new RegExp("^" + base.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "/"), "");
      a.href = repo + "/blob/" + branch + "/" + path + url.hash;
    });
  }

  // 8. Click a screenshot to see it full size
  main.addEventListener("click", function (ev) {
    var img = ev.target.closest("img.shot, img.shot-hero");
    if (!img) return;
    var box = document.createElement("figure");
    box.className = "lightbox";
    box.setAttribute("role", "dialog");
    box.setAttribute("aria-label", img.alt || "Screenshot");
    box.innerHTML = '<img alt=""><figcaption></figcaption>';
    box.querySelector("img").src = img.src;
    box.querySelector("img").alt = img.alt;
    box.querySelector("figcaption").textContent = img.alt;
    function close() { box.remove(); document.removeEventListener("keydown", onKey); }
    function onKey(e) { if (e.key === "Escape") close(); }
    box.addEventListener("click", close);
    document.addEventListener("keydown", onKey);
    document.body.appendChild(box);
  });
})();
