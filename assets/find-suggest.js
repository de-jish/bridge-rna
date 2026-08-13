/* Keyboard, pointer and open/closed behaviour for the "Find a study" combobox.
 *
 * This is the map's only JavaScript, and it exists because Dash cannot express
 * the one thing an autocomplete is made of: a keystroke. `dcc.Input` publishes
 * `value`, `n_submit` and `n_blur`, and none of those is Up, Down or Escape.
 * Everything that *can* be done in Python is: the server decides what the
 * suggestions are and what selecting one does, and this file never invents a
 * suggestion, never resolves an identifier, and never writes into a store.
 *
 * The division of labour is the design, and it is what keeps the two halves
 * from fighting over one property:
 *
 *   the server owns WHAT IS IN the list  - `offer_suggestions` renders the rows
 *   this file owns WHETHER IT IS OPEN    - one class, `is-closed`, on the group
 *
 * Those are independent facts with one owner each, and `map.css` composes them
 * into a single `display`: a list with no rows is hidden by `:empty`, and an
 * open list the reader has dismissed is hidden by the class. No callback writes
 * a `style` here, and this file never removes a row.
 *
 * Selection goes through Dash, not around it. A row is a real component with a
 * pattern-matching id, so a mouse click, a tap and Enter all end in the same
 * `n_clicks` and the same server callback - Enter is implemented as
 * `activeRow.click()` for exactly that reason. Nothing here pokes a value into
 * the input: React would not see it, and there would be two ways to commit a
 * search that could drift apart.
 */
(function () {
  "use strict";

  var GROUP = "find-group";
  var INPUT = "find-input";
  var LIST = "find-suggestions";
  var CLOSED = "is-closed";
  var ACTIVE = "is-active";
  var OPTION = '[role="option"][id]';

  function group() { return document.getElementById(GROUP); }
  function input() { return document.getElementById(INPUT); }
  function list() { return document.getElementById(LIST); }

  function options() {
    var l = list();
    return l ? Array.prototype.slice.call(l.querySelectorAll(OPTION)) : [];
  }

  /* The combobox semantics live here rather than in the layout because
   * `dcc.Input` declares no `aria-*` or `role` wildcard in Dash 4 - passing one
   * raises when the layout is built. That is the right home regardless: three
   * of these five attributes are live state, and only the code that opens,
   * closes and moves through the list can keep them true. */
  function ensureAria() {
    var box = input();
    if (!box || box.getAttribute("role") === "combobox") { return; }
    box.setAttribute("role", "combobox");
    box.setAttribute("aria-autocomplete", "list");
    box.setAttribute("aria-controls", LIST);
    box.setAttribute("aria-expanded", "false");
  }

  function isClosed() {
    var g = group();
    return !g || g.classList.contains(CLOSED);
  }

  /* `aria-expanded` has to describe what a reader can actually reach, which is
   * rows AND not-dismissed - the same conjunction the stylesheet draws. */
  function syncExpanded() {
    var box = input();
    if (box) {
      box.setAttribute("aria-expanded",
        (!isClosed() && options().length > 0) ? "true" : "false");
    }
  }

  function open() {
    var g = group();
    if (g) { g.classList.remove(CLOSED); }
    syncExpanded();
  }

  function close() {
    var g = group();
    if (g) { g.classList.add(CLOSED); }
    clearActive();
    syncExpanded();
  }

  function clearActive() {
    options().forEach(function (el) {
      el.classList.remove(ACTIVE);
      el.setAttribute("aria-selected", "false");
    });
    var box = input();
    if (box) { box.removeAttribute("aria-activedescendant"); }
  }

  function activeIndex(opts) {
    for (var i = 0; i < opts.length; i++) {
      if (opts[i].classList.contains(ACTIVE)) { return i; }
    }
    return -1;
  }

  function setActive(opts, index) {
    clearActive();
    var el = opts[index];
    if (!el) { return; }
    el.classList.add(ACTIVE);
    el.setAttribute("aria-selected", "true");
    var box = input();
    if (box && el.id) { box.setAttribute("aria-activedescendant", el.id); }
    /* `nearest` scrolls only when the row is actually out of view, and it walks
     * every scrollable ancestor - which matters because the list is inline in a
     * rail that scrolls too, so a row can be inside the list's window and still
     * below the fold of the page. */
    if (el.scrollIntoView) { el.scrollIntoView({ block: "nearest" }); }
  }

  function move(delta) {
    var opts = options();
    if (!opts.length) { return; }
    /* Down on a dismissed list re-opens it, which is the standard escape from
     * having pressed Escape and changed your mind without retyping. */
    if (isClosed()) { open(); }
    var current = activeIndex(opts);
    var next = current < 0
      ? (delta > 0 ? 0 : opts.length - 1)
      : (current + delta + opts.length) % opts.length;
    setActive(opts, next);
  }

  /* The rows are replaced wholesale on every keystroke, and a replaced row
   * takes its `is-active` class with it but leaves `aria-activedescendant`
   * pointing at an id that no longer exists. One observer, re-pointed whenever
   * the element identity changes, because the router unmounts this whole view.
   */
  var observed = null;
  var observer = new MutationObserver(function () {
    clearActive();
    syncExpanded();
  });

  function watchList() {
    ensureAria();
    var l = list();
    if (l && l !== observed) {
      observer.disconnect();
      observer.observe(l, { childList: true });
      observed = l;
    }
  }

  function onInput(e) {
    if (!e.target || e.target.id !== INPUT) { return; }
    watchList();
    /* Typing always re-opens: the reader is asking a new question, so a
     * dismissal of the previous answer should not outlive it. */
    open();
    clearActive();
  }

  function onKeyDown(e) {
    if (!e.target || e.target.id !== INPUT) { return; }
    watchList();
    var opts = options();
    var active = opts[activeIndex(opts)];

    if (e.key === "ArrowDown") {
      e.preventDefault();          // the caret must not jump to the line end
      move(1);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      move(-1);
    } else if (e.key === "Enter") {
      if (active && !isClosed()) {
        /* Stop the event here, in the capture phase, so Dash's own Enter
         * handler never sees it: `n_submit` would commit the half-typed text
         * the reader is in the middle of replacing. With no row active it is
         * deliberately left alone, and Enter means what it always did. */
        e.preventDefault();
        e.stopPropagation();
        active.click();
        close();
      }
    } else if (e.key === "Escape") {
      /* Swallowed, so it cannot reach anything else on the page that treats
       * Escape as "close me" - and so a first Escape closes the list rather
       * than clearing the box, which is the ARIA combobox contract. */
      e.preventDefault();
      e.stopPropagation();
      close();
    } else if (e.key === "Tab") {
      close();                     // leaving the control closes it; focus moves on
    }
  }

  function onMouseDown(e) {
    var l = list();
    if (l && l.contains(e.target)) {
      /* Keep focus in the text field. Without this the mousedown blurs the
       * input, `n_blur` commits whatever fragment is in the box, and the click
       * that follows commits the real identifier a moment later - two searches
       * for one gesture, the first of them wrong. This is also what makes a tap
       * work, since a touch synthesises this mousedown before the blur. */
      e.preventDefault();
      return;
    }
    var g = group();
    if (g && !g.contains(e.target)) { close(); }
  }

  function onClick(e) {
    var l = list();
    if (l && l.contains(e.target)) {
      /* The row's own Dash callback still runs - nothing is stopped here. The
       * list is closed because the question has been answered; the server is
       * about to fill the box with the chosen identifier, and a programmatic
       * value change fires no `input` event, so this stays closed. */
      close();
    }
  }

  function onFocusIn(e) {
    if (!e.target || e.target.id !== INPUT) { return; }
    watchList();
    if (e.target.value) { open(); }
  }

  document.addEventListener("input", onInput, true);
  document.addEventListener("keydown", onKeyDown, true);
  document.addEventListener("mousedown", onMouseDown, true);
  document.addEventListener("click", onClick, true);
  document.addEventListener("focusin", onFocusIn, true);
})();
