#!/usr/bin/env python3
"""Read-only real-corpus checks for documented comparison evidence.

No discovery claim, enrichment test or result-stability UI is produced.
Run .venv/bin/python precompute/validate_comparisons.py > report.json
"""
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from bridge_rna.comparison_ui import run_documented_comparison
from bridge_rna.retrieval import run_cohort_retrieval_with_neighborhood
from bridge_rna.reviewed import COMPARISONS, resolve_comparison


def validate():
    results = []
    for preset in COMPARISONS:
        t = time.monotonic()
        a, b = resolve_comparison(preset.key)
        payload, *_ = run_documented_comparison(preset.key, 30)
        paired = payload["paired_evidence"]
        assert len({r["gsm"] for r in paired["rows"]}) == len(paired["rows"])
        assert len({r["archs4_index"] for r in paired["rows"]}) == len(paired["rows"])
        maximum_error = 0.
        for field, cohort in (("a", a), ("b", b)):
            _, legacy, _ = run_cohort_retrieval_with_neighborhood(cohort.members, 30)
            rows = sorted((r for r in paired["rows"] if r[f"rank_{field}"] <= 250),
                          key=lambda r: r[f"rank_{field}"])
            # Ties can choose different samples at the old argpartition boundary;
            # compare shared identities numerically and require the full top-30.
            before = dict(zip(legacy.gsm, legacy.score))
            after = {r["gsm"]: r[f"score_{field}"] for r in rows}
            assert set(legacy.iloc[:30].gsm) == set(r["gsm"] for r in rows[:30])
            maximum_error = max(maximum_error, max(abs(after[g] - before[g]) for g in after.keys() & before.keys()))
        assert maximum_error < 1e-6
        sensitivity = {}
        for depth in (30, 100, 250):
            rows = [r for r in paired["rows"] if min(r["rank_a"], r["rank_b"]) <= depth]
            counts = {}
            for r in rows:
                counts[r["gse"]] = counts.get(r["gse"], 0) + 1
            sensitivity[depth] = dict(samples=len(rows), selected_gse_groups=len(counts),
                                      largest_selected_gse_samples=max(counts.values()))
        results.append(dict(comparison=preset.key, members_a=len(a.members), members_b=len(b.members),
                            reference_count=paired["reference_count"],
                            legacy_max_abs_cosine_difference=maximum_error,
                            source_context_available=paired["context"]["available"],
                            depth_sensitivity=sensitivity, seconds=round(time.monotonic() - t, 3)))
    return dict(checks="fixed membership, unique identities, full-index paired scores and legacy agreement",
                interpretation="Exploratory retrieval evidence; not biological validation or independent replication.",
                comparisons=results)


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
