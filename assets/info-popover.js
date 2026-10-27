/* The "i" explanations: one popover, opened beside whichever button asked.
 *
 * The server renders two things and never touches either again: a button per
 * explanation (`bridge_rna.help.info`, carrying only `data-info="<key>"`) and
 * one hidden library of every explanation on the shell (`help.library`). This
 * file owns everything that changes:
 *
 *   - the popover itself, a node it creates on <body> - never a Dash node, so
 *     React never reconciles over it and no scroll container can clip it;
 *   - `aria-expanded` and `aria-controls` on the button that opened it. Dash
 *     never sets them, so a re-render cannot fight this file over them.
 *
 * It never reparents, rewrites or removes a node Dash rendered. Opening an
 * explanation posts nothing to the server.
 *
 * The popover is a non-modal dialog: focus moves into it, Esc or the close
 * button returns focus to the button, and tabbing out of it closes it rather
 * than wandering to the end of the page where it lives. It closes on any
 * scroll or resize instead of chasing its button, and on an outside click.
 * It also closes when its button leaves the page, which is what a Dash
 * re-render of the button's parent looks like from here.
 */
(function () {
  "use strict";

  var GAP = 6;
  var MARGIN = 8;
  var pop = null;
  var trigger = null;
  var watcher = null;

  function ensure() {
    if (pop) return pop;
    pop = document.createElement("div");
    pop.id = "info-popover";
    pop.className = "info-popover";
    pop.setAttribute("role", "dialog");
    pop.setAttribute("aria-modal", "false");
    pop.setAttribute("aria-labelledby", "info-popover-title");
    pop.tabIndex = -1;
    pop.hidden = true;
    document.body.appendChild(pop);
    return pop;
  }

  function entryFor(key) {
    var library = document.getElementById("info-library");
    if (!library) return null;
    var nodes = library.querySelectorAll("[data-info-entry]");
    for (var i = 0; i < nodes.length; i++) {
      if (nodes[i].getAttribute("data-info-entry") === key) return nodes[i];
    }
    return null;
  }

  function fill(source) {
    var p = ensure();
    p.textContent = "";
    var close = document.createElement("button");
    close.type = "button";
    close.className = "info-pop-close";
    close.setAttribute("aria-label", "Close");
    p.appendChild(close);
    for (var i = 0; i < source.children.length; i++) {
      p.appendChild(source.children[i].cloneNode(true));
    }
    var title = p.querySelector(".info-title");
    if (title) title.id = "info-popover-title";
    // A code block wider than the popover scrolls; give it a tab stop so a
    // keyboard can scroll it too.
    var blocks = p.querySelectorAll("pre");
    for (var j = 0; j < blocks.length; j++) blocks[j].tabIndex = 0;
  }

  function place(button) {
    var p = pop;
    var r = button.getBoundingClientRect();
    var vw = document.documentElement.clientWidth;
    var vh = window.innerHeight;
    p.style.maxHeight = "";
    p.style.left = "0px";
    p.style.top = "0px";
    var width = p.offsetWidth;
    var natural = p.scrollHeight;
    var below = vh - r.bottom - GAP - MARGIN;
    var above = r.top - GAP - MARGIN;
    var onBelow = below >= Math.min(natural, 260) || below >= above;
    var room = Math.max(140, onBelow ? below : above);
    var height = Math.min(natural, room);
    p.style.maxHeight = room + "px";
    // Start the card a little left of the button so its edge reads as
    // belonging to it, then keep it inside the viewport.
    var left = Math.min(Math.max(MARGIN, r.left - 12), vw - width - MARGIN);
    var top = onBelow ? r.bottom + GAP : r.top - GAP - height;
    p.style.left = Math.max(MARGIN, left) + "px";
    p.style.top = Math.max(MARGIN, top) + "px";
    p.setAttribute("data-side", onBelow ? "below" : "above");
  }

  function open(button) {
    var key = button.getAttribute("data-info");
    var source = entryFor(key);
    if (!source) return;
    if (trigger === button && pop && !pop.hidden) {
      close(true);
      return;
    }
    close(false);
    fill(source);
    pop.setAttribute("data-key", key);
    pop.hidden = false;
    place(button);
    trigger = button;
    button.setAttribute("aria-expanded", "true");
    button.setAttribute("aria-controls", "info-popover");
    pop.focus({ preventScroll: true });
    watch();
  }

  function close(restoreFocus) {
    if (!pop || pop.hidden) return;
    pop.hidden = true;
    var button = trigger;
    trigger = null;
    if (watcher) { watcher.disconnect(); watcher = null; }
    if (!button) return;
    button.setAttribute("aria-expanded", "false");
    if (restoreFocus && document.contains(button)) {
      button.focus({ preventScroll: true });
    }
  }

  // While open, and only while open: if Dash re-renders the button away, the
  // popover is pointing at nothing and closes.
  function watch() {
    if (!window.MutationObserver) return;
    watcher = new MutationObserver(function () {
      if (trigger && !document.contains(trigger)) close(false);
    });
    watcher.observe(document.body, { childList: true, subtree: true });
  }

  function focusables() {
    return pop.querySelectorAll('button, a[href], [tabindex="0"]');
  }

  // Capture phase, so a component that stops propagation cannot keep the
  // popover from closing, and so a click on an "i" inside a clickable row or
  // card does not also activate the row.
  document.addEventListener("click", function (event) {
    var target = event.target;
    if (!(target instanceof Element)) return;
    var button = target.closest(".info-btn[data-info]");
    if (button) {
      event.preventDefault();
      event.stopPropagation();
      open(button);
      return;
    }
    if (!pop || pop.hidden) return;
    if (target.closest(".info-pop-close")) {
      close(true);
    } else if (!pop.contains(target)) {
      close(false);
    }
  }, true);

  document.addEventListener("keydown", function (event) {
    if (!pop || pop.hidden) return;
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      close(true);
      return;
    }
    if (event.key !== "Tab") return;
    var items = focusables();
    var first = items[0];
    var last = items[items.length - 1];
    var active = document.activeElement;
    var leaving = event.shiftKey
      ? (active === pop || active === first)
      : (active === last || !pop.contains(active));
    if (leaving) {
      event.preventDefault();
      close(true);
    }
  }, true);

  window.addEventListener("scroll", function (event) {
    if (pop && !pop.hidden && !(event.target instanceof Node && pop.contains(event.target))) {
      close(false);
    }
  }, true);

  window.addEventListener("resize", function () { close(false); });
})();
