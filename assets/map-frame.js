/* Keep a framed 2-D map at one scale while its canvas changes size.
 *
 * The axes are scale-anchored, so Plotly fits a requested window by widening
 * one of its ranges - and writes the widened range back into the layout. A
 * resize then starts from that range rather than from the window that was
 * asked for, so what the map showed depended on whether the figure or the
 * resize landed first. Fit results docks the evidence drawer while the framed
 * figure is still being built: when the figure won, the docked map kept the
 * full-width range and showed the frame 1.4x further out than it needed to.
 * `constrain: "domain"` cannot prevent it, because Plotly drops the domain
 * branch for any range zoomed in from the one it first constrained.
 *
 * The server states the window it framed as `layout.meta.viewport`. This holds
 * that window centred at the scale it was first fitted at, and never at a
 * smaller one than showing all of it needs: a wider canvas reveals more around
 * the frame, a narrower one zooms out exactly as far as containing it requires.
 * So closing the drawer is not a zoom, and docking it cannot hide a result.
 *
 * A framed figure arrives with `constrain: "range"` for the same reason: in
 * domain mode a relayout to a held window resets Plotly's reference range, and
 * the next wider canvas is letterboxed - the frame shown inside empty bands,
 * its neighbours clipped at an invisible edge.
 *
 * Two moments need it. After a resize, the held window is put back with one
 * relayout. When the server restates a frame already held, which every drawer
 * toggle does, the figure is handed to `Plotly.react` with the held window in
 * place of the stored one, so Plotly paints the frame once instead of
 * painting its own fit first and having it taken back.
 *
 * It changes no points, sampling or retrieval state, and it never writes
 * `viewport-store`: its relayout carries whole-range keys, which the server
 * reads as "unchanged", so the stored frame stays the one that was framed.
 */
(function () {
  "use strict";
  const TOLERANCE = 1e-6;
  let graph = null;
  let attached = false;
  // {viewport: [x0, x1, y0, y1], scale: data units per pixel}, or null when
  // the map is unframed and Plotly's own autorange is already a pure function
  // of the canvas.
  let frame = null;

  function served() {
    const meta = graph.layout && graph.layout.meta;
    const vp = meta && meta.viewport;
    return Array.isArray(vp) && vp.length === 4 ? vp.map(Number) : null;
  }

  function plotSize() {
    const size = graph._fullLayout && graph._fullLayout._size;
    return size && size.w > 0 && size.h > 0 ? size : null;
  }

  function fitScale(vp, size) {
    return Math.max((vp[1] - vp[0]) / size.w, (vp[3] - vp[2]) / size.h);
  }

  function same(a, b) {
    return !!a && !!b && a.length === b.length && a.every(function (v, i) {
      return Math.abs(v - b[i]) <= TOLERANCE * Math.max(1, Math.abs(v));
    });
  }

  function adopt(vp) {
    const size = plotSize();
    frame = vp && size ? {viewport: vp, scale: fitScale(vp, size)} : null;
  }

  // The window that holds the frame on a canvas of `size`: centred on the
  // stored window, at its scale, unless the canvas is now too small for that
  // scale to show all of it.
  function heldWindow(size) {
    const vp = frame.viewport;
    frame.scale = Math.max(frame.scale, fitScale(vp, size));
    const cx = (vp[0] + vp[1]) / 2, cy = (vp[2] + vp[3]) / 2;
    const hw = frame.scale * size.w / 2, hh = frame.scale * size.h / 2;
    return [cx - hw, cx + hw, cy - hh, cy + hh];
  }

  function hold() {
    const size = plotSize();
    const axes = graph._fullLayout;
    if (!frame || !size || !axes.xaxis || !axes.yaxis) return;
    const want = heldWindow(size);
    const shown = axes.xaxis.range.concat(axes.yaxis.range).map(Number);
    // Within a tenth of a pixel is the same picture: Plotly rounds the canvas,
    // and chasing that would only repaint it.
    if (shown.every(function (v, i) {
      return Math.abs(v - want[i]) <= 0.1 * frame.scale;
    })) return;
    window.Plotly.relayout(graph, {
      "xaxis.range": [want[0], want[1]],
      "yaxis.range": [want[2], want[3]],
    });
  }

  function react() {
    const vp = served();
    if (!vp) frame = null;
    else if (frame && same(frame.viewport, vp)) hold();
    else adopt(vp);
  }

  // When the server restates the frame already held - closing the drawer,
  // switching arm or focus - its ranges are the stored window, which Plotly
  // would fit to the canvas afresh and paint before `react` could put the held
  // scale back. Handing it the held window instead paints the frame once. A
  // new frame passes through untouched and is adopted after Plotly fits it.
  function restated(figure) {
    const layout = figure && figure.layout;
    const meta = layout && layout.meta;
    const vp = meta && Array.isArray(meta.viewport) ? meta.viewport.map(Number) : null;
    // The canvas Plotly will lay this out in is the one it already has: a
    // react does not re-measure the container, the debounced resize does.
    const size = plotSize();
    if (!frame || !vp || !size || !same(frame.viewport, vp)) return figure;
    const want = heldWindow(size);
    return Object.assign({}, figure, {layout: Object.assign({}, layout, {
      xaxis: Object.assign({}, layout.xaxis, {range: [want[0], want[1]]}),
      yaxis: Object.assign({}, layout.yaxis, {range: [want[2], want[3]]}),
    })});
  }

  function wrapReact() {
    const Plotly = window.Plotly;
    if (!Plotly || !Plotly.react || Plotly.react.holdsMapFrame) return;
    const react = Plotly.react;
    const wrapped = function (gd, figure) {
      const args = Array.prototype.slice.call(arguments);
      if (gd === graph) args[1] = restated(figure);
      return react.apply(this, args);
    };
    wrapped.holdsMapFrame = true;
    Plotly.react = wrapped;
  }

  function relayout(event) {
    if (!event) return;
    if (event.autosize) {
      hold();
    } else if ("xaxis.range[0]" in event && "xaxis.range[1]" in event &&
               "yaxis.range[0]" in event && "yaxis.range[1]" in event) {
      // The reader panned or zoomed: what is on screen is the new frame, and
      // the server will store exactly this window a round trip from now.
      const x = [event["xaxis.range[0]"], event["xaxis.range[1]"]].map(Number);
      const y = [event["yaxis.range[0]"], event["yaxis.range[1]"]].map(Number);
      adopt([Math.min(...x), Math.max(...x), Math.min(...y), Math.max(...y)]);
    } else if (event["xaxis.autorange"] || event["yaxis.autorange"]) {
      frame = null;
    }
  }

  function connect() {
    const next = document.querySelector("#manifold-graph .js-plotly-plot");
    if (next === graph && (attached || !next || !next.on)) return;
    if (graph && graph.removeListener) {
      graph.removeListener("plotly_react", react);
      graph.removeListener("plotly_relayout", relayout);
    }
    graph = next;
    attached = false;
    frame = null;
    if (graph && graph.on) {
      wrapReact();
      graph.on("plotly_react", react);
      graph.on("plotly_relayout", relayout);
      attached = true;
      if (graph._fullLayout) react();
    }
  }

  // Dash replaces the route, and with it the graph. Reconnect to the new one.
  new MutationObserver(connect).observe(document.documentElement,
    {childList: true, subtree: true});
  connect();
})();
