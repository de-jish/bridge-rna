/* Local emphasis for the single-query retrieval network. Existing Plotly click
 * events still drive Dash's inspector; no ranking, layout or payload changes. */
(function () {
  "use strict";
  let graph = null;
  let nodes = null;
  let selected = null;
  let hovered = null;
  let frame = null;

  function schedule() {
    if (frame === null) frame = requestAnimationFrame(paint);
  }

  function paint() {
    frame = null;
    if (!graph || !nodes) return;
    const id = hovered || selected;
    // Style the existing SVG instead of restyling Plotly on hover. Restyle
    // redraws can interrupt a click in flight and rebuild the hover layer.
    // Scientific trace data and clean PNG exports retain their base styling.
    graph.querySelectorAll(".scatterlayer .trace").forEach(function (element) {
      const trace = element.__data__ && element.__data__[0]?.trace;
      if (!trace || !trace.meta) return;
      if (trace.meta.network_edge) {
        element.classList.toggle("network-edge-active", trace.meta.network_edge.includes(id));
      }
      if (!trace.meta.network_nodes) return;
      element.querySelectorAll(".point").forEach(function (point) {
        const datum = trace.customdata[point.__data__.i];
        point.classList.toggle("network-node-active", datum[1] === id);
        point.classList.toggle("network-query", datum[0] === "query");
      });
    });
  }

  function pointId(event) {
    const point = event.points && event.points[0];
    return point && point.customdata ? point.customdata[1] : null;
  }
  function hover(event) { hovered = pointId(event); schedule(); }
  function unhover() { hovered = null; schedule(); }
  function click(event) { selected = pointId(event); schedule(); }
  function afterplot() {
    const next = (graph.data || []).find(t => t.meta && t.meta.network_nodes) || null;
    if (next !== nodes) {
      nodes = next;
      selected = hovered = null;
    }
    schedule();
  }
  function connect() {
    const next = document.querySelector("#network-graph .js-plotly-plot");
    if (next === graph || (next && !next.on)) return;
    if (graph && graph.removeListener) {
      graph.removeListener("plotly_hover", hover);
      graph.removeListener("plotly_unhover", unhover);
      graph.removeListener("plotly_click", click);
      graph.removeListener("plotly_afterplot", afterplot);
    }
    graph = next;
    nodes = null;
    selected = hovered = null;
    if (graph) {
      graph.on("plotly_hover", hover);
      graph.on("plotly_unhover", unhover);
      graph.on("plotly_click", click);
      graph.on("plotly_afterplot", afterplot);
      afterplot();
    }
  }
  new MutationObserver(connect).observe(document.documentElement,
    {childList: true, subtree: true});
  connect();
})();
