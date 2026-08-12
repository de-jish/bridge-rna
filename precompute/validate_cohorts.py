#!/usr/bin/env python3
"""Does pooling a cohort actually work? Six checks against the real corpus.

Every other candidate in this repository was accepted or rejected on a
measurement against a null of the same form, and this is that measurement for
cohort retrieval. It opens the real 963 MB ARCHS4 memmap and the real 2,108
cached OSDR embeddings; it is not part of the pytest suite, which never touches
either.

    .venv/bin/python precompute/validate_cohorts.py                  # all checks
    .venv/bin/python precompute/validate_cohorts.py --stability      # check 5 only
    .venv/bin/python precompute/validate_cohorts.py --cohorts 40     # smaller sweep

The six checks:

1. **Identity.** A pooled query over one sample must reproduce that sample's
   cached-path result exactly. If it does not, the pooling code is not sitting
   on the path it claims to reuse. *Correctness: a failure exits non-zero.*
2. **Leave-one-out stability.** Full-cohort top-k against each leave-one-out
   top-k, versus member-against-member. The feature's headline number.
3. **A structure-free null.** k random OSDR samples pooled. If a random group is
   as stable as a real cohort, pooling is only averaging and the cohort
   definition is doing nothing. *Correctness: a failure exits non-zero.*
4. **A within-study null.** k random samples from one study, which isolates what
   tissue and arm contribute on top of the batch that same-study membership
   already supplies. Study alone supplies 84% of a cohort's coherence, so this
   is the uncomfortable one.
5. **Stability versus k.** The curve that sets `cohorts.LOW_N_THRESHOLD`, and
   nothing else any more. It used to be pasted into `STABILITY_BY_K` and quoted
   on the rail beside a selected cohort; the app measures this statistic live
   now, per query, so the curve is back to answering only the question it can
   answer honestly - how large a cohort has to be. *Correctness: a knee that no
   longer matches the shipped threshold exits non-zero.*
6. **Normalization.** Spherical against raw mean over every cohort, so "it
   changes almost nothing here" stays a measurement rather than a memory.

**The whole thing is one pass over the memmap.** Every query vector for every
check is built first, stacked into one (n_queries, 512) matrix, and scored in a
single streaming sweep that keeps a running top-k per query. That is the same
technique `validate_artifacts.py --mixing` uses, and it is why several hundred
queries cost one 963 MB read rather than several hundred.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from bridge_rna import cohorts as C  # noqa: E402
from bridge_rna.retrieval import (  # noqa: E402
    _cached_osdr_embeddings,
    _load_archs4_index,
    _topk_cosine_matrix,
)

# Deep enough that the agreement statistic is not dominated by ties, shallow
# enough that it is the number a user actually reads off the screen.
DEPTHS = (5, 20, 100)
MAX_DEPTH = max(DEPTHS)

# The two gates. Both come from docs/design-notes.md#cohort-pooling, which measured pooled
# top-5 leave-one-out agreement at 0.78 against 0.13 for a single member.
MIN_POOLED_STABILITY = 0.60      # check 2: well below 0.78, so noise cannot trip it
MIN_NULL_MARGIN = 0.03           # check 3: a real cohort must beat random by this


# One definition of "how much do two hit lists share", shared with the live
# path. `bridge_rna.cohorts.top_k_agreement` is what the app measures during a
# search and reports in the inspector, so the number a user reads off the screen
# and the number this gate computes over all 212 cohorts are the same statistic
# rather than two that merely resemble each other.
jaccard = C.top_k_agreement


class QueryBatch:
    """Accumulates query vectors, then scores them all in one memmap pass.

    The sweep itself is `retrieval._topk_cosine_matrix`, which is the app's own
    scan. It used to be a second copy living here, and the copy is gone for the
    reason every duplicated source of truth in this repository is gone: the live
    stability measurement now runs the same fan-out this validator does, and two
    implementations of it could disagree while both looked right.
    """

    def __init__(self) -> None:
        self._vectors: list[np.ndarray] = []
        self._names: list[str] = []
        self.results: dict[str, np.ndarray] = {}
        self.scores: dict[str, np.ndarray] = {}

    def add(self, name: str, vec: np.ndarray) -> str:
        self._vectors.append(np.asarray(vec, dtype=np.float32).reshape(-1))
        self._names.append(name)
        return name

    def __len__(self) -> int:
        return len(self._vectors)

    def run(self, k: int = MAX_DEPTH, block_bytes: int = 200_000_000) -> None:
        if not self._vectors:
            return
        Q = np.stack(self._vectors)
        index_vecs, _, _ = _load_archs4_index()
        n = int(index_vecs.shape[0])
        q = Q.shape[0]

        t0 = time.time()

        def tick(done: int, total: int) -> None:
            print(f"\r  scanning {100.0 * done / total:5.1f}%  ({q} queries)",
                  end="", flush=True)

        best_idx, best_score = _topk_cosine_matrix(
            index_vecs, Q, k=min(k, n), block_bytes=block_bytes, progress=tick)
        print(f"\r  scanned {n:,} rows x {q} queries in {time.time() - t0:.1f} s")

        for i, name in enumerate(self._names):
            self.results[name] = best_idx[i]
            self.scores[name] = best_score[i]


def agreement(batch: QueryBatch, a: str, b: str, depth: int) -> float:
    return jaccard(batch.results[a][:depth], batch.results[b][:depth])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohorts", type=int, default=0,
                    help="how many cohorts to sweep, size-stratified. Default 0 "
                         "means every cohort, which is what the shipped curve "
                         "is built from: at two cohorts per size bin the "
                         "per-k figure swings by 0.5 between adjacent sizes "
                         "and is an artifact of which cohorts were drawn.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stability", action="store_true",
                    help="print only the stability-versus-k curve")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    cached = _cached_osdr_embeddings()
    if cached is None:
        print("FAIL: no cached OSDR embeddings. Run precompute/embed_osdr.py.")
        return 2
    vectors, key_to_row = cached
    print(f"Cached OSDR embeddings: {len(vectors):,} x {vectors.shape[1]}")

    all_cohorts = [c for c in C.build_cohorts() if c.size >= C.MIN_COHORT_SIZE]
    sizes = np.array([c.size for c in all_cohorts])
    print(f"Cohorts under the default definition "
          f"({', '.join(C.DEFAULT_FACETS)}): {len(all_cohorts):,} "
          f"across {len({c.study for c in all_cohorts})} studies")
    print(f"  size: median {int(np.median(sizes))}, mean {sizes.mean():.1f}, "
          f"max {sizes.max()}, total members {int(sizes.sum()):,}")

    if args.cohorts <= 0:
        picked = list(all_cohorts)
    else:
        # A size-stratified sample rather than the largest N, so the low-k end -
        # which is exactly what LOW_N_THRESHOLD is about - is represented at all.
        by_size: dict[int, list] = {}
        for c in all_cohorts:
            by_size.setdefault(c.size, []).append(c)
        picked = []
        for size in sorted(by_size):
            group = by_size[size]
            take = max(1, min(len(group), args.cohorts // max(1, len(by_size))))
            idx = rng.choice(len(group), size=take, replace=False)
            picked.extend(group[i] for i in idx)
        picked = picked[:args.cohorts]
    print(f"  sweeping {len(picked)} cohorts, sizes "
          f"{min(c.size for c in picked)} to {max(c.size for c in picked)}\n")

    def rows_for(members) -> np.ndarray:
        return np.stack([vectors[key_to_row[m]] for m in members])

    # ---- build every query vector this run needs, before touching the memmap --
    batch = QueryBatch()
    plan = []                       # (cohort, name_full, [loo names], [member names])

    for n_c, c in enumerate(picked):
        rows = rows_for(c.members)
        full = batch.add(f"c{n_c}:full", C.cohort_query_vector(rows))
        raw_mean = rows.mean(axis=0)
        raw = batch.add(f"c{n_c}:raw", raw_mean / np.linalg.norm(raw_mean))
        loo_names, member_names = [], []
        for j in range(c.size):
            keep = np.delete(np.arange(c.size), j)
            loo_names.append(batch.add(f"c{n_c}:loo{j}",
                                       C.cohort_query_vector(rows[keep])))
            member_names.append(batch.add(f"c{n_c}:m{j}", rows[j]))
        plan.append((c, full, raw, loo_names, member_names))

    # Check 1: a one-member pool against the plain cached vector.
    probe_key = picked[0].members[0]
    probe_vec = vectors[key_to_row[probe_key]]
    identity_pooled = batch.add("identity:pooled",
                                C.cohort_query_vector(probe_vec.reshape(1, -1)))
    identity_plain = batch.add("identity:plain", probe_vec)

    # Checks 3 and 4: the two nulls, matched to the real cohorts' sizes.
    all_keys = list(key_to_row)
    study_of = {}
    meta = C.cohort_metadata()
    for _, r in meta.iterrows():
        study_of[str(r["sample_key"])] = str(r["study"])
    by_study: dict[str, list[str]] = {}
    for key, study in study_of.items():
        by_study.setdefault(study, []).append(key)

    null_plan, within_plan = [], []
    for n_c, c in enumerate(picked):
        k = c.size
        # structure-free: k samples from anywhere in OSDR
        members = [all_keys[i] for i in rng.choice(len(all_keys), size=k, replace=False)]
        r = rows_for(members)
        null_plan.append((k,
                          batch.add(f"n{n_c}:full", C.cohort_query_vector(r)),
                          [batch.add(f"n{n_c}:loo{j}",
                                     C.cohort_query_vector(np.delete(r, j, axis=0)))
                           for j in range(k)]))
        # within-study: k samples from this cohort's own study, ignoring tissue
        # and arm. Skipped when the study cannot supply k samples outside a
        # single cohort, since then the "null" would be the cohort itself.
        pool = by_study.get(c.study, [])
        if len(pool) >= k and len(set(pool) - set(c.members)) >= 1:
            members = [pool[i] for i in rng.choice(len(pool), size=k, replace=False)]
            r = rows_for(members)
            within_plan.append((k,
                                batch.add(f"w{n_c}:full", C.cohort_query_vector(r)),
                                [batch.add(f"w{n_c}:loo{j}",
                                           C.cohort_query_vector(np.delete(r, j, axis=0)))
                                 for j in range(k)]))

    print(f"Scoring {len(batch):,} query vectors in one memmap pass...")
    batch.run()
    print()

    failures: list[str] = []

    # ---- check 1: identity ------------------------------------------------
    #
    # What this can assert, and why it is not "identical top-100".
    #
    # The first version demanded an exact match all the way down and failed, and
    # the reason turned out to be worth recording rather than tuning away.
    # Pooling one sample normalizes it twice - once inside cohort_query_vector,
    # once inside the scan - so the query vector differs from the plain one by
    # 7.45e-9, a single float32 ulp, at cosine 1.0. Scores then differ by at
    # most 1.19e-7, which is float32 epsilon at magnitude 1.
    #
    # That is enough to reorder the list, because the index is denser than
    # float32 can resolve: for this probe the first differing rank is 23, and
    # the score gap between rank 23 and rank 24 there is *exactly 0.0*. The two
    # runs are permuting an exact tie, not disagreeing. Through rank 50 the two
    # lists are the same set; only at 100 does the cutoff itself land inside a
    # tie band and admit a different member.
    #
    # So the gate is: scores agree to float32, and the depth a user actually
    # reads is identical in order. Demanding more would be demanding that
    # float32 have more precision than it has, and it would be the same
    # over-reading of a hairline score gap that this whole feature exists to
    # stop. The measured divergence depth is printed rather than hidden.
    IDENTITY_ORDERED_DEPTH = 20
    pooled_idx = batch.results[identity_pooled]
    plain_idx = batch.results[identity_plain]
    max_score_diff = float(np.abs(batch.scores[identity_pooled]
                                  - batch.scores[identity_plain]).max())
    ordered_ok = np.array_equal(pooled_idx[:IDENTITY_ORDERED_DEPTH],
                                plain_idx[:IDENTITY_ORDERED_DEPTH])
    scores_ok = max_score_diff < 1e-6
    first_diff = next((i for i in range(len(pooled_idx))
                       if pooled_idx[i] != plain_idx[i]), None)
    tie_gap = (float(batch.scores[identity_plain][first_diff]
                     - batch.scores[identity_plain][first_diff + 1])
               if first_diff is not None and first_diff + 1 < len(plain_idx)
               else float("nan"))
    print("1. IDENTITY  a one-sample pool against the plain cached vector")
    print(f"   sample                     {probe_key}")
    print(f"   top-{IDENTITY_ORDERED_DEPTH} identical in order   {ordered_ok}")
    print(f"   max score diff             {max_score_diff:.2e}  "
          f"(float32 eps at |1| is 1.19e-07)")
    print(f"   first differing rank       {first_diff}")
    print(f"   score gap at that rank     {tie_gap:.2e}  "
          f"<- an exact tie, permuted by argpartition")
    if not (ordered_ok and scores_ok):
        failures.append("check 1: pooling one sample does not reproduce the "
                        "cached path")
    print(f"   -> {'PASS' if ordered_ok and scores_ok else 'FAIL'}\n")

    # ---- check 2: leave-one-out stability ---------------------------------
    pooled_stab = {d: [] for d in DEPTHS}
    member_stab = {d: [] for d in DEPTHS}
    per_size: dict[int, list[float]] = {}
    for c, full, _raw, loo_names, member_names in plan:
        for d in DEPTHS:
            vals = [agreement(batch, full, lo, d) for lo in loo_names]
            pooled_stab[d].append(float(np.mean(vals)))
            pairs = [agreement(batch, member_names[i], member_names[j], d)
                     for i in range(c.size) for j in range(i + 1, c.size)]
            if pairs:
                member_stab[d].append(float(np.mean(pairs)))
        per_size.setdefault(c.size, []).append(
            float(np.mean([agreement(batch, full, lo, 5) for lo in loo_names])))

    print("2. LEAVE-ONE-OUT STABILITY  how much a pooled result survives "
          "dropping one animal")
    print(f"   {'depth':>8}  {'pooled':>8}  {'member vs member':>18}  {'gain':>6}")
    for d in DEPTHS:
        p, m = float(np.mean(pooled_stab[d])), float(np.mean(member_stab[d]))
        print(f"   {'top-' + str(d):>8}  {p:8.3f}  {m:18.3f}  {p / max(m, 1e-9):5.1f}x")
    top5_pooled = float(np.mean(pooled_stab[5]))
    if top5_pooled < MIN_POOLED_STABILITY:
        failures.append(f"check 2: pooled top-5 stability {top5_pooled:.3f} is "
                        f"below the {MIN_POOLED_STABILITY} floor")
    print(f"   -> {'PASS' if top5_pooled >= MIN_POOLED_STABILITY else 'FAIL'}\n")

    # ---- checks 3 and 4: the nulls ----------------------------------------
    def null_stability(entries) -> float:
        vals = []
        for _k, full, loo_names in entries:
            vals.append(float(np.mean([agreement(batch, full, lo, 5)
                                       for lo in loo_names])))
        return float(np.mean(vals)) if vals else float("nan")

    random_stab = null_stability(null_plan)
    within_stab = null_stability(within_plan)
    print("3/4. NULLS  is the cohort definition doing anything?")
    print(f"   real cohorts, top-5              {top5_pooled:.3f}")
    print(f"   k random OSDR samples            {random_stab:.3f}   "
          f"(structure-free null)")
    print(f"   k random samples, same study     {within_stab:.3f}   "
          f"(within-study null, n={len(within_plan)})")
    margin = top5_pooled - random_stab
    print(f"   margin over the structure-free null   {margin:+.3f}")
    print(f"   margin over the within-study null     {top5_pooled - within_stab:+.3f}")
    if not (margin >= MIN_NULL_MARGIN):
        failures.append(f"check 3: real cohorts beat the structure-free null by "
                        f"only {margin:+.3f}, under the {MIN_NULL_MARGIN} floor")
    print(f"   -> {'PASS' if margin >= MIN_NULL_MARGIN else 'FAIL'}\n")

    # ---- check 5: stability versus k --------------------------------------
    #
    # Bucketed, not per-k, and that is a correction rather than a convenience.
    # The first run of this script sampled two cohorts per size and produced a
    # "curve" that read 0.38 at k=5 and 0.90 at k=6 - an artifact of which two
    # cohorts were drawn, not a property of size. A confidence number quoted in
    # the UI has to be backed by enough cohorts that it does not swing by half
    # its range when the seed changes, so sizes are pooled into buckets and each
    # bucket reports how many cohorts stand behind it.
    print("5. STABILITY VERSUS COHORT SIZE  the curve behind LOW_N_THRESHOLD")
    buckets = [(2, 2), (3, 3), (4, 4), (5, 6), (7, 9), (10, 14), (15, 10**6)]

    def bucket_label(lo: int, hi: int) -> str:
        return f"{lo}" if lo == hi else (f"{lo}+" if hi > 1000 else f"{lo}-{hi}")

    print(f"   {'k':>7}  {'cohorts':>8}  {'stability':>10}  {'spread (sd)':>11}")
    measured: list[tuple[int, int, list[float]]] = []
    for lo, hi in buckets:
        vals = [v for size, group in per_size.items() if lo <= size <= hi
                for v in group]
        if not vals:
            continue
        measured.append((lo, hi, vals))
        print(f"   {bucket_label(lo, hi):>7}  {len(vals):>8}  "
              f"{float(np.mean(vals)):>10.3f}  {float(np.std(vals)):>11.3f}")

    # A bigger cohort must never be reported as less trustworthy than a smaller
    # one. Adjacent buckets can and do invert - measured, 5-6 scores 0.736 and
    # 7-9 scores 0.696, a 0.04 gap against a within-bucket sd of ~0.18, so the
    # inversion is noise rather than a finding. Merging the two into one bucket
    # is the honest repair: it neither invents monotonicity by clamping nor
    # ships a curve that tells a researcher their larger cohort is worse.
    merged: list[tuple[int, int, list[float]]] = []
    for lo, hi, vals in measured:
        while merged and np.mean(vals) < np.mean(merged[-1][2]):
            plo, _phi, pvals = merged.pop()
            lo, vals = plo, pvals + vals
        merged.append((lo, hi, vals))
    if len(merged) != len(measured):
        print("\n   merged non-monotone adjacent buckets (inversion within "
              "noise):")
        for lo, hi, vals in merged:
            print(f"   {bucket_label(lo, hi):>7}  {len(vals):>8}  "
                  f"{float(np.mean(vals)):>10.3f}  {float(np.std(vals)):>11.3f}")

    # This curve no longer gets pasted anywhere. It used to populate
    # `cohorts.STABILITY_BY_K`, which the rail's cohort card quoted by size -
    # and quoting a population average beside one cohort's name is what got it
    # deleted on 2026-08-06, because the spread inside a bucket is most of the
    # range (a cohort of 7 measuring 0.316 sat in the same bucket as one of 6
    # measuring 0.849). The app measures this statistic live now, on the query
    # that just ran; see docs/design-notes.md#live-stability.
    #
    # What the curve still settles is the one question it can answer honestly:
    # how large does a cohort have to be before its results usually hold up.
    # That is `LOW_N_THRESHOLD`, and the picker uses it to mark a small cohort
    # before a search has run, which is the only moment when size is all that
    # is known. So the check now reports the knee and compares it, rather than
    # printing a dict to copy.
    curve = {lo: float(np.mean(vals)) for lo, _hi, vals in merged}
    knee = next((s for s in sorted(curve) if curve[s] >= C.STABILITY_FLOOR), None)
    print(f"\n   first bucket reaching {C.STABILITY_FLOOR:.2f} stability: "
          f"k >= {knee}")

    # The comparison is only a gate on a **full** sweep, and that restriction is
    # the same correction the bucketing above exists for. A --cohorts sample
    # draws one or two cohorts per size, which is exactly the regime that
    # produced 0.38 at k=5 beside 0.90 at k=6 in the first run of this script.
    # Measured here on a 22-cohort sample the knee lands at 10 rather than 5,
    # from buckets standing on a single cohort each. Failing a build on that
    # would be enforcing the noise the shipped number was chosen to survive.
    full_sweep = len(picked) == len(all_cohorts)
    if not full_sweep:
        print(f"   cohorts.LOW_N_THRESHOLD is {C.LOW_N_THRESHOLD}  -> not "
              f"checked: this is a {len(picked)}-cohort sample, and the knee "
              "is only quotable from every cohort. Re-run without --cohorts.")
    else:
        print(f"   cohorts.LOW_N_THRESHOLD is {C.LOW_N_THRESHOLD}  "
              f"-> {'agrees' if knee == C.LOW_N_THRESHOLD else 'DISAGREES'}")
        if knee is not None and knee != C.LOW_N_THRESHOLD:
            failures.append(
                f"check 5: over all {len(all_cohorts)} cohorts the knee is "
                f"k >= {knee}, but cohorts.LOW_N_THRESHOLD is "
                f"{C.LOW_N_THRESHOLD}. That threshold is what marks a small "
                "cohort in the picker before a search has run.")
    print()

    if args.stability:
        return 0 if not failures else 1

    # ---- check 6: normalization -------------------------------------------
    print("6. NORMALIZATION  spherical mean against raw mean")
    same_top5, cosines = 0, []
    for c, full, raw, _loo, _members in plan:
        rows = rows_for(c.members)
        m = rows.mean(axis=0)
        cosines.append(float(np.dot(C.cohort_query_vector(rows),
                                    m / np.linalg.norm(m))))
        if np.array_equal(batch.results[full][:5], batch.results[raw][:5]):
            same_top5 += 1
    cosines = np.array(cosines)
    print(f"   cos(spherical, raw)   median {np.median(cosines):.7f}, "
          f"worst {cosines.min():.5f}")
    print(f"   identical top-5       {same_top5} of {len(plan)} cohorts")
    print("   -> the two agree almost everywhere here, as expected: within one "
          "cohort")
    print("      the L2-norm spread is ~1.09x. Spherical is kept because it is "
          "the")
    print("      correct estimator when Tissue is unticked and the spread is "
          "3.9x.\n")

    print("=" * 72)
    if failures:
        for f in failures:
            print(f"FAIL  {f}")
        return 1
    print("All checks passed. Cohort pooling is doing what it claims.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
