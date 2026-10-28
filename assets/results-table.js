/* The match table's browser half. Everything here exists to keep a click from
 * posting the whole retrieval back to the server: at k=1,000 hits-store is
 * 385 KB of JSON per cohort, and the inspector or the AI summary needs a few
 * rows of it. Which rows are in view is decided in Python (bridge_rna/results.py)
 * and published as `results-visible-store`; nothing here filters or ranks. */
(function () {
  "use strict";

  // The AI prompt formats at most this many hits (bridge_rna/ai.py), so the
  // request carries no more, and GEO enrichment runs on no more.
  const AI_SUMMARY_MAX_ROWS = 20;

  function hitLists(hits) {
    const a = (hits && hits.hits) || [];
    const b = ((hits && hits.comparison) || {}).hits_b || [];
    return a.concat(b);
  }

  function byGsm(hits) {
    const out = {};
    hitLists(hits).forEach(function (h) {
      // Cohort A's record wins for a shared hit, as the inspector always did.
      if (h && h.gsm && !(h.gsm in out)) out[h.gsm] = h;
    });
    return out;
  }

  window.dash_clientside = Object.assign({}, window.dash_clientside, {
    results: {
      inspectorContext: function (selected, hits) {
        if (!hits) return null;
        const kind = selected && selected.kind;
        const id = selected && selected.node_id;
        let rows = [];
        if (kind === "gsm") {
          const one = byGsm(hits)[id];
          rows = one ? [one] : [];
        } else if (kind === "gse") {
          const seen = {};
          rows = hitLists(hits).filter(function (h) {
            if (!h || h.gse !== id || seen[h.gsm]) return false;
            seen[h.gsm] = true;
            return true;
          });
        }
        const comparison = hits.comparison || {};
        return {
          query: hits.query || null,
          sample_id: hits.sample_id || "",
          query_b: comparison.query_b || null,
          entrez_email: hits.entrez_email || "",
          scoring: hits.scoring || "cosine",
          selected: selected || null,
          rows: rows,
        };
      },

      markSelected: function (selected) {
        const gsm = selected && selected.kind === "gsm" ? selected.node_id : null;
        // After the table has painted, so a re-rendered page is marked too.
        requestAnimationFrame(function () {
          document.querySelectorAll("#results-table-wrap tr.result-row")
            .forEach(function (tr) {
              const on = gsm !== null && tr.dataset.gsm === gsm;
              tr.classList.toggle("is-selected", on);
              if (on) {
                tr.setAttribute("aria-current", "true");
                if (tr.scrollIntoView) tr.scrollIntoView({block: "nearest"});
              } else {
                tr.removeAttribute("aria-current");
              }
            });
        });
        return window.dash_clientside.no_update;
      },

      aiRequest: function (clicks, visible, hits) {
        if (!clicks) return window.dash_clientside.no_update;
        if (!hits) return {stamp: Date.now(), query: null};
        const lookup = byGsm(hits);
        const order = visible || [];
        return {
          stamp: Date.now(),
          query: hits.query || null,
          sample_id: hits.sample_id || "",
          entrez_email: hits.entrez_email || "",
          scoring: hits.scoring || "cosine",
          hits: order.slice(0, AI_SUMMARY_MAX_ROWS)
            .map(function (g) { return lookup[g]; })
            .filter(Boolean),
          total: order.length,
        };
      },
    },
  });

  // A click anywhere on a row selects it, through the row's own button, so the
  // keyboard path (Tab to the accession, Enter) and the mouse path are one
  // path. Clicks on controls, and the end of a text selection, are left alone.
  document.addEventListener("click", function (event) {
    const target = event.target;
    if (!target || !target.closest) return;
    const row = target.closest("#results-table-wrap tr.result-row");
    if (!row || target.closest("button, a, input, select, textarea, summary")) return;
    const selection = window.getSelection && String(window.getSelection());
    if (selection) return;
    const button = row.querySelector("button.result-select");
    if (button) button.click();
  });
})();
