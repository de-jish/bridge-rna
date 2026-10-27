/* The guided tour: chapters, slides, and opening once on a first visit.
 *
 * `bridge_rna.guide.build_guide` renders one closed native <dialog> on the
 * shell, beside the header and view slots, and no callback ever touches it.
 * This file owns every attribute that changes on it, and Dash sets none of
 * them, so the two never disagree:
 *
 *   - `data-current` on the current chapter and slide (CSS shows only those,
 *     with `display: none` for the rest, which also takes them out of the
 *     accessibility tree);
 *   - `aria-selected` and `tabindex` on the chapter tabs;
 *   - the counter text, the dots, and Back/Next's disabled state;
 *   - each screenshot's `src`, set from `data-src` when its slide is shown.
 *
 * `showModal()` gives the rest for free: the page behind goes inert, focus is
 * held in the dialog, and Esc closes it. Focus returns to the "How it works"
 * button on close - looked up again at that moment, since the header is
 * rebuilt on every route change and the button that opened the guide may be
 * gone.
 *
 * First visit: the guide opens once, for the chapter of the page you landed
 * on, after that page has actually finished drawing - the boot skeleton gone
 * and, on the map, the first figure in place. It gives up quietly if the page
 * is not ready within a bounded wait, or if the reader starts using the page
 * first, and it records the visit only once it has really opened. A blocked
 * localStorage means it never opens by itself, which is the safe failure.
 * Nor does it open by itself in an automated browser (`navigator.webdriver`).
 */
(function () {
  "use strict";

  var SEEN = "bridge-rna-guide-seen";
  var READY_POLL_MS = 250;
  var READY_LIMIT_MS = 30000;

  var chapter = null;
  var index = 0;

  function dialog() { return document.getElementById("guide"); }
  function chapters() { return dialog().querySelectorAll(".guide-chapter"); }
  function tabs() { return dialog().querySelectorAll(".guide-tab"); }
  function slides() { return chapter ? chapter.querySelectorAll(".guide-slide") : []; }

  function chapterForPath() {
    var path = window.location.pathname.replace(/[/]+$/, "");
    return path.endsWith("/map") ? "map" : "retrieve";
  }

  function selectChapter(key, focusTab) {
    var all = chapters();
    for (var i = 0; i < all.length; i++) {
      var on = all[i].getAttribute("data-chapter") === key;
      if (on) { all[i].setAttribute("data-current", ""); chapter = all[i]; }
      else all[i].removeAttribute("data-current");
    }
    var t = tabs();
    for (var j = 0; j < t.length; j++) {
      var sel = t[j].getAttribute("data-guide-chapter") === key;
      t[j].setAttribute("aria-selected", sel ? "true" : "false");
      t[j].tabIndex = sel ? 0 : -1;
      if (sel && focusTab) t[j].focus();
    }
    var dots = dialog().querySelector(".guide-dots");
    dots.textContent = "";
    for (var k = 0; k < slides().length; k++) {
      dots.appendChild(document.createElement("span"));
    }
    show(0);
  }

  function show(i) {
    var list = slides();
    if (!list.length) return;
    index = Math.max(0, Math.min(i, list.length - 1));
    for (var n = 0; n < list.length; n++) {
      if (n === index) list[n].setAttribute("data-current", "");
      else list[n].removeAttribute("data-current");
    }
    load(list[index]);
    if (index + 1 < list.length) load(list[index + 1]);
    var d = dialog();
    var dots = d.querySelectorAll(".guide-dots span");
    for (var m = 0; m < dots.length; m++) {
      dots[m].className = m === index ? "is-current" : "";
    }
    var title = list[index].querySelector(".guide-slide-title");
    d.querySelector(".guide-counter").textContent =
      "Step " + (index + 1) + " of " + list.length +
      (title ? ": " + title.textContent : "");
    d.querySelector(".guide-back").disabled = index === 0;
    var next = d.querySelector(".guide-next");
    var last = index === list.length - 1;
    next.textContent = last ? "Done" : "Next";
    next.setAttribute("data-done", last ? "true" : "false");
  }

  function load(slide) {
    var img = slide.querySelector("img[data-src]");
    if (img && !img.getAttribute("src")) img.src = img.getAttribute("data-src");
  }

  function open(key) {
    var d = dialog();
    if (!d || d.open) return false;
    selectChapter(key || chapterForPath(), false);
    try { d.showModal(); } catch (e) { return false; }
    d.querySelector(".guide-next").focus();
    return true;
  }

  function close() {
    var d = dialog();
    if (d && d.open) d.close();
  }

  function restoreFocus() {
    var opener = document.querySelector("[data-guide-open]");
    if (opener) opener.focus({ preventScroll: true });
  }

  // Delegated, because the header that holds the opener is rebuilt on every
  // route change and the dialog's own buttons are Dash nodes.
  document.addEventListener("click", function (event) {
    var target = event.target;
    if (!(target instanceof Element)) return;
    if (target.closest("[data-guide-open]")) {
      cancelAutoOpen();
      open();
      return;
    }
    var d = dialog();
    if (!d || !d.open || !d.contains(target)) {
      // A click on the backdrop lands on the dialog element itself.
      return;
    }
    if (target === d) { close(); return; }
    if (target.closest(".guide-close")) { close(); return; }
    var tab = target.closest(".guide-tab");
    if (tab) { selectChapter(tab.getAttribute("data-guide-chapter"), false); return; }
    if (target.closest(".guide-back")) { show(index - 1); return; }
    var next = target.closest(".guide-next");
    if (next) {
      if (next.getAttribute("data-done") === "true") close();
      else show(index + 1);
    }
  });

  document.addEventListener("keydown", function (event) {
    var d = dialog();
    if (!d || !d.open) return;
    var onTab = event.target instanceof Element && event.target.closest(".guide-tab");
    if (onTab && (event.key === "ArrowLeft" || event.key === "ArrowRight")) {
      // Chapter tabs follow the tabs pattern: arrows move between them.
      var t = tabs();
      var at = Array.prototype.indexOf.call(t, onTab);
      var to = (at + (event.key === "ArrowRight" ? 1 : t.length - 1)) % t.length;
      event.preventDefault();
      selectChapter(t[to].getAttribute("data-guide-chapter"), true);
      return;
    }
    if (event.key === "ArrowRight") { event.preventDefault(); show(index + 1); }
    else if (event.key === "ArrowLeft") { event.preventDefault(); show(index - 1); }
  });

  // `close` fires for Esc, the close button, Done and the backdrop alike.
  document.addEventListener("close", function (event) {
    if (event.target === dialog()) restoreFocus();
  }, true);

  // --- First visit ----------------------------------------------------------

  var pending = null;

  function seen() {
    try { return window.localStorage.getItem(SEEN) !== null; }
    catch (e) { return true; }
  }

  function remember() {
    try { window.localStorage.setItem(SEEN, new Date().toISOString()); }
    catch (e) { /* nothing to remember with; it will not open by itself */ }
  }

  function pageReady() {
    if (!dialog() || document.querySelector("._dash-loading")) return false;
    if (chapterForPath() !== "map") {
      return !!document.getElementById("results-panel");
    }
    var skeleton = document.getElementById("map-skeleton");
    if (skeleton) return skeleton.classList.contains("is-done");
    // No skeleton yet is not "no map": Dash can drop its loading node a beat
    // before the view mounts. Only the explanation the shell draws when this
    // machine has no map cache counts as ready without a figure.
    return !!document.getElementById("map-unavailable");
  }

  function cancelAutoOpen() {
    if (pending === null) return;
    window.clearInterval(pending);
    pending = null;
    document.removeEventListener("pointerdown", cancelAutoOpen, true);
    document.removeEventListener("keydown", cancelAutoOpen, true);
  }

  function scheduleAutoOpen() {
    // An automated browser - the test suites, the screenshot scripts - is not
    // a first-time reader, and a modal opening over it would make the page it
    // is driving inert.
    if (seen() || navigator.webdriver) return;
    var started = Date.now();
    document.addEventListener("pointerdown", cancelAutoOpen, true);
    document.addEventListener("keydown", cancelAutoOpen, true);
    pending = window.setInterval(function () {
      if (Date.now() - started > READY_LIMIT_MS) { cancelAutoOpen(); return; }
      if (!pageReady()) return;
      cancelAutoOpen();
      if (open()) remember();
    }, READY_POLL_MS);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", scheduleAutoOpen);
  } else {
    scheduleAutoOpen();
  }
})();
