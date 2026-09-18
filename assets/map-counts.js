/* Refine the server's initial counts using Plotly's final visible axes.
 * Scale constraints and responsive resizing can expand the requested ranges.
 * This readout never changes points, sampling, axes, or retrieval state.
 */
(function () {
  "use strict";
  let graph = null;
  let attached = false;
  let frame = null;

  function update() {
    frame = null;
    const next = document.querySelector("#manifold-graph .js-plotly-plot");
    if (next !== graph || (next && !attached && next.on)) {
      if (graph && graph.removeListener) {
        graph.removeListener("plotly_afterplot", schedule);
        graph.removeListener("plotly_relayout", schedule);
      }
      graph = next;
      attached = false;
      if (graph && graph.on) {
        graph.on("plotly_afterplot", schedule);
        graph.on("plotly_relayout", schedule);
        attached = true;
      }
    }
    if (!graph || !graph._fullData || !graph._fullLayout) return;
    const axes = graph._fullLayout;
    const counts = {archs4: 0, osdr: 0};
    for (const trace of graph._fullData) {
      const corpus = trace.meta && trace.meta.corpus;
      if (!(corpus in counts) || trace.visible === false ||
          trace.visible === "legendonly") continue;
      const x = trace.x || [], y = trace.y || [];
      if (trace.type === "scatter3d") {
        counts[corpus] += x.length;
        continue;
      }
      if (!axes.xaxis || !axes.yaxis) continue;
      const xr = axes.xaxis.range, yr = axes.yaxis.range;
      const xmin = Math.min(...xr), xmax = Math.max(...xr);
      const ymin = Math.min(...yr), ymax = Math.max(...yr);
      for (let i = 0; i < x.length; i++) {
        if (x[i] >= xmin && x[i] <= xmax && y[i] >= ymin && y[i] <= ymax) {
          counts[corpus]++;
        }
      }
    }
    for (const corpus of Object.keys(counts)) {
      const value = document.querySelector(
        '#plot-badges [data-corpus-count="' + corpus + '"] b');
      const text = counts[corpus].toLocaleString("en-US");
      if (value && value.textContent !== text) value.textContent = text;
    }
  }

  function schedule() {
    if (frame === null) frame = requestAnimationFrame(update);
  }

  // Dash replaces both the route and badge children. Reconnect to a new graph
  // and refresh newly rendered badges, coalescing mutations into one frame.
  new MutationObserver(function (records) {
    const next = document.querySelector("#manifold-graph .js-plotly-plot");
    if (next !== graph || (next && !attached) ||
        records.some(record => record.target.id === "plot-badges")) schedule();
  }).observe(document.documentElement,
    {childList: true, subtree: true});
  schedule();
})();
