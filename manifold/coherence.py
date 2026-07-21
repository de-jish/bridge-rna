"""The lasso readout: "are these samples meaningfully related?"

Every statistic here is computed in the original 512-d cosine space, never from
the 2D projection. The lasso only decides *which* points; the pixels never
define the number. This is non-negotiable: UMAP and PCA distances are distorted,
so a statistic read off the plot would be a lie dressed as a measurement.

The readout has five parts (IMPLEMENTATION.md section 7):
  A. Geometric cohesion - mean cosine to the selection centroid, against a null
     of uniform random selections of the same size and corpus composition, plus
     kNN-purity fold-enrichment from the hnswlib index.
  B. Metadata enrichment - one-sided hypergeometric per category, BH-corrected.
  C. Batch-confound guard - is the coherence just one study?
  D. Cross-dataset caution - OSDR/ARCHS4 mixes carry a precision batch effect.
  E. Honest negative - say plainly when a selection looks random.

The null is computed analytically rather than by resampling, from exact
population moments plus a finite-population correction. That is a correctness
requirement, not an optimization: a bootstrap over a cached background pool
converges to the pool's mean while its own spread shrinks with selection size,
which turns a fixed sampling error into a z-bias growing without bound. See
_permutation_null.

Effect sizes (z, fold) travel with every p-value, because at large |S| every p
is ~0 and significance without effect size is false confidence.
"""

from __future__ import annotations

import numpy as np

from . import data

MIN_SELECTION = 8
N_PERM = 1000
KNN_K = 15
SEED = 12345
# A category has to clear this fold-enrichment before the verdict names it as a
# driver. Significance alone does not qualify: at large |S| a 1.05x deviation
# reaches q < 1e-10 on sample size alone.
MIN_DRIVER_FOLD = 1.25

# Two independent routes to "coherent", which measure different things and can
# legitimately disagree. Global cohesion is mean cosine to the selection
# centroid: it asks whether the selection is one tight cloud. kNN purity asks
# whether each point's own neighbourhood lies inside the selection. A lasso over
# several tight but mutually distant groups scores high on the second and low on
# the first. The verdict has to say which one carries the claim, or it reads as
# a contradiction - see _synthesize.
MIN_COHESION_Z = 3.0
MAX_COHESION_P = 0.05
MIN_KNN_PURITY_FOLD = 2.0


def _mean_cos_to_centroid(vecs: np.ndarray) -> float:
    """Mean cosine of a set of unit vectors to their own normalized centroid.

    For unit vectors v_1..v_n with mean m, the statistic collapses to ||m||:

        (1/n) * sum_i v_i . (m/||m||)  =  (1/n) * (n*m) . m/||m||  =  ||m||

    This is an identity, not an approximation, and it is what makes the
    permutation null affordable: a null draw needs only the *sum* of its
    sampled vectors, never the per-point dot products. Valid because every
    vector reaching this function came from data.normalized_vectors().
    """
    if len(vecs) == 0:
        return 0.0
    return float(np.linalg.norm(vecs.mean(axis=0)))


def _arm_null_moments(n_pop: int, mu: np.ndarray, cov: np.ndarray, n_draw: int):
    """Mean and covariance of the sample mean of ``n_draw`` points from a corpus.

    Sampling is *without replacement* from the whole corpus, which is what a
    lasso actually does, so the finite-population correction applies:

        E[m]   = mu
        Cov[m] = (N - n) / (n * (N - 1)) * Sigma

    The correction is what makes the null degenerate when the selection is the
    entire corpus - at n = N there is only one possible draw, so the honest
    z-score is 0. A with-replacement null has no such property.
    """
    if n_draw <= 0:
        return np.zeros_like(mu), np.zeros_like(cov)
    denom = n_draw * max(n_pop - 1, 1)
    factor = max(n_pop - n_draw, 0) / denom
    return mu, cov * factor


def _permutation_null(n_archs4_sel: int, n_osdr_sel: int) -> tuple[np.ndarray, float]:
    """Null distribution of the cohesion statistic, matching dataset composition.

    The statistic is ||mean(V)|| (see _mean_cos_to_centroid), so the null only
    needs the distribution of the sample *mean* under a random selection of the
    same size and corpus composition. That mean is a sum of many bounded terms,
    so its distribution is Gaussian to high accuracy, and its exact first two
    moments follow from the population moments plus the finite-population
    correction. Drawing from that Gaussian is both exact in n and cheap.

    Why not resample a cached pool of background vectors: with-replacement
    draws from a fixed pool converge to that *pool's* mean rather than the
    population's, while the null's spread keeps shrinking like 1/sqrt(n). The
    gap between pool mean and population mean therefore turns into a z-offset
    growing as sqrt(n / pool_size) whose sign is fixed by the pool's random
    seed. Measured on the real corpus proportions that offset reaches |z| > 40,
    which would label random selections as strongly (in)coherent.

    Returns the null statistics and the model variance of the sample mean. The
    variance is returned rather than inferred from the samples because it is
    exactly zero for a selection covering a whole corpus, where a float32
    spread of ~1e-9 over identical draws is indistinguishable from a real but
    very tight null if you only look at the samples.
    """
    rng = np.random.default_rng(SEED)
    total = n_archs4_sel + n_osdr_sel
    if total == 0:
        return np.zeros(N_PERM, dtype=np.float64), 0.0

    n_archs4, n_osdr, _ = data.counts()
    mean = np.zeros(512, dtype=np.float64)
    cov = np.zeros((512, 512), dtype=np.float64)

    if n_archs4_sel > 0:
        mu, sigma = data.population_moments("archs4")
        m, c = _arm_null_moments(n_archs4, mu, sigma, n_archs4_sel)
        mean += (n_archs4_sel / total) * m
        cov += (n_archs4_sel / total) ** 2 * c
    if n_osdr_sel > 0:
        mu, sigma = data.population_moments("osdr")
        m, c = _arm_null_moments(n_osdr, mu, sigma, n_osdr_sel)
        mean += (n_osdr_sel / total) * m
        cov += (n_osdr_sel / total) ** 2 * c

    model_var = float(np.trace(cov))
    draws = _sample_mvn(mean, cov, N_PERM, rng)
    return np.linalg.norm(draws, axis=1), model_var


def _sample_mvn(mean: np.ndarray, cov: np.ndarray, n: int, rng) -> np.ndarray:
    """Draw ``n`` samples from N(mean, cov) via a symmetric eigendecomposition.

    An eigendecomposition rather than a Cholesky because the covariance of
    L2-normalized vectors is singular by construction (the data lie on a sphere)
    and Cholesky would fail; negative eigenvalues from round-off are clipped.
    """
    total_var = float(np.trace(cov))
    if total_var <= 0:
        return np.tile(mean, (n, 1))
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 0.0, None)
    scale = evecs * np.sqrt(evals)
    return mean + rng.standard_normal((n, len(mean))) @ scale.T


def _knn_purity(point_indices: np.ndarray, vecs: np.ndarray | None = None) -> dict | None:
    """Fraction of each point's k nearest neighbours that are also selected.

    Reported as a fold-enrichment over the fraction expected by chance,
    |S|/N. This is the robust companion to the centroid statistic: it survives
    selections that are elongated or multi-lobed, where a single centroid is a
    poor summary and mean-cosine understates real local structure.
    """
    idx = data.hnsw_index()
    if idx is None:
        return None
    _, _, N = data.counts()
    if vecs is None:
        vecs = data.normalized_vectors(point_indices)
    try:
        labels, _ = idx.knn_query(vecs, k=KNN_K + 1)
    except Exception:
        return None

    # Vectorized: drop each point's self-match, then count how many of its
    # remaining neighbours fall inside the selection. A Python loop here costs
    # seconds once a lasso covers tens of thousands of points.
    labels = np.asarray(labels, dtype=np.int64)
    is_self = labels == point_indices[:, None]
    # Keep exactly KNN_K neighbours per row: if the self-match is absent (it can
    # be, for an approximate index), drop the furthest instead.
    drop = np.where(is_self.any(axis=1), is_self.argmax(axis=1), labels.shape[1] - 1)
    keep = np.ones(labels.shape, dtype=bool)
    keep[np.arange(len(labels)), drop] = False
    neighbours = labels[keep].reshape(len(labels), labels.shape[1] - 1)

    selected = np.zeros(N, dtype=bool)
    selected[point_indices] = True
    observed = float(selected[neighbours].mean())
    expected = (len(point_indices) - 1) / (N - 1)
    fold = observed / expected if expected > 0 else float("inf")
    return {"observed": observed, "expected": expected, "fold": fold}


def _bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted q-values."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    # enforce monotonicity from the largest rank down
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.clip(ranked, 0, 1)
    return q


def _enrichment(point_indices: np.ndarray) -> list[dict]:
    from scipy.stats import hypergeom

    n_archs4, n_osdr, N = data.counts()
    pm = data.points_meta()
    is_osdr = point_indices >= n_archs4
    osdr_sel_src = (point_indices[is_osdr] - n_archs4)

    records = []  # (field, category, k, K, n, popN)

    # Species over the datasets actually present in the selection.
    present_datasets = set(pm["dataset"].to_numpy()[point_indices])
    sp_labels_full = data.species_labels()
    pop_mask = np.isin(pm["dataset"].to_numpy(), list(present_datasets))
    pop_species = sp_labels_full[pop_mask]
    sel_species = sp_labels_full[point_indices]
    for cat in np.unique(sel_species):
        k = int((sel_species == cat).sum())
        K = int((pop_species == cat).sum())
        records.append(("species", str(cat), k, K, len(sel_species), len(pop_species)))

    # OSDR categorical fields over the OSDR population (only if OSDR present).
    if len(osdr_sel_src) >= 3:
        for field in data.OSDR_FIELDS:
            vals_full = data.osdr_field_values(field)
            sel_vals = vals_full.iloc[osdr_sel_src]
            n = len(sel_vals)
            popN = len(vals_full)
            counts_sel = sel_vals.value_counts()
            counts_pop = vals_full.value_counts()
            for cat, k in counts_sel.items():
                if str(cat) in ("Unknown", "nan", "None"):
                    continue
                K = int(counts_pop.get(cat, 0))
                records.append((field, str(cat), int(k), K, n, popN))

    if not records:
        return []

    rows = []
    for field, cat, k, K, n, popN in records:
        if K == 0 or n == 0:
            continue
        # P(X >= k) under hypergeometric(popN, K, n)
        p = float(hypergeom.sf(k - 1, popN, K, n))
        fold = (k / n) / (K / popN) if K > 0 else float("inf")
        rows.append({"field": field, "category": cat, "k": k, "K": K,
                     "n": n, "N": popN, "p": p, "fold": fold})

    if not rows:
        return []
    q = _bh_qvalues(np.array([r["p"] for r in rows]))
    for r, qi in zip(rows, q):
        r["q"] = float(qi)
    rows.sort(key=lambda r: (r["q"], -r["fold"]))
    return rows


def analyze_selection(point_indices) -> dict:
    """Full 512-d readout for a lasso selection. Returns a structured dict."""
    point_indices = np.asarray(point_indices, dtype=np.int64)
    point_indices = np.unique(point_indices)
    n_archs4, n_osdr, N = data.counts()
    k = len(point_indices)
    if k < MIN_SELECTION:
        return {"status": "too_small", "n": k, "min": MIN_SELECTION}

    is_osdr = point_indices >= n_archs4
    n_osdr_sel = int(is_osdr.sum())
    n_archs4_sel = int((~is_osdr).sum())

    vecs = data.normalized_vectors(point_indices)
    obs = _mean_cos_to_centroid(vecs)
    null, model_var = _permutation_null(n_archs4_sel, n_osdr_sel)
    null_mean, null_std = float(null.mean()), float(null.std())

    if model_var <= 0.0:
        # The finite-population correction has collapsed the null: the
        # selection is the whole corpus, so there is exactly one draw to
        # compare against and no evidence either way. Reporting a z from a
        # zero-width null would be dividing by round-off.
        z, emp_p = 0.0, 1.0
    else:
        z = (obs - null_mean) / null_std
        emp_p = float((np.sum(null >= obs) + 1) / (N_PERM + 1))

    knn = _knn_purity(point_indices, vecs)
    enrich = _enrichment(point_indices)
    significant = [r for r in enrich if r["q"] < 0.05]

    cross_dataset = n_archs4_sel > 0 and n_osdr_sel > 0
    cohesive_global = z >= MIN_COHESION_Z and emp_p < MAX_COHESION_P
    cohesive_local = knn is not None and knn["fold"] >= MIN_KNN_PURITY_FOLD
    cohesive = cohesive_global or cohesive_local
    has_enrichment = len(significant) > 0

    # Batch-confound guard on OSDR study. The flag asks "is the coherence we are
    # reporting explained by one batch?", so it is only raised when there is
    # coherence to explain - warning that a *random-looking* selection might be
    # batch-driven would be noise dressed as caution.
    batch = None
    if n_osdr_sel >= 3:
        study_vals = data.osdr_field_values("study").iloc[point_indices[is_osdr] - n_archs4]
        top = study_vals.value_counts()
        if len(top):
            top_study = str(top.index[0])
            frac = float(top.iloc[0] / len(study_vals))
            study_is_top_field = bool(significant and significant[0]["field"] == "study")
            batch = {
                "top_study": top_study,
                "fraction": frac,
                "flag": bool((frac > 0.5 or study_is_top_field)
                             and (cohesive or has_enrichment)),
            }

    return {
        "status": "ok",
        "n": k, "n_archs4": n_archs4_sel, "n_osdr": n_osdr_sel,
        "cohesion": {"obs": obs, "null_mean": null_mean, "null_std": null_std,
                     "z": z, "emp_p": emp_p},
        "knn": knn,
        "enrichment": significant[:8],
        "batch": batch,
        "cross_dataset": cross_dataset,
        "cohesive": cohesive,
        "cohesive_global": cohesive_global,
        "cohesive_local": cohesive_local,
        "has_enrichment": has_enrichment,
        "verdict": _synthesize(z, emp_p, knn, significant, batch, cross_dataset,
                               cohesive, has_enrichment,
                               cohesive_global=cohesive_global),
        "verdict_class": _verdict_class(cohesive, significant, batch),
    }


def _fmt_p(p: float) -> str:
    if p <= 1.0 / (N_PERM + 1):
        return f"p<{1.0/(N_PERM+1):.1e}".replace("1.0e", "1e")
    if p < 1e-3:
        return f"p={p:.1e}"
    return f"p={p:.3f}"


def _fmt_q(q: float) -> str:
    return "q<1e-10" if q < 1e-10 else (f"q={q:.0e}" if q < 1e-3 else f"q={q:.3f}")


def _verdict_class(cohesive: bool, significant, batch) -> str:
    """Which color the verdict banner gets.

    Deliberately stricter than "something was significant". A selection with
    weak cohesion whose only enriched category sits at 1.2x has not shown
    anything, and painting that banner green would contradict the sentence
    inside it. Green requires either real cohesion or a driver that clears the
    effect-size floor.
    """
    if batch and batch["flag"]:
        return "warn"
    has_driver = any(r["fold"] >= MIN_DRIVER_FOLD for r in significant)
    if cohesive or has_driver:
        return "good"
    return "null"


CROSS_DATASET_NOTE = (
    "NOTE: selection mixes OSDR and ARCHS4; cross-corpus proximity is "
    "confounded by the fp32-vs-bf16 precision batch effect"
)


def _synthesize(z, emp_p, knn, significant, batch, cross_dataset,
                cohesive, has_enrichment, cohesive_global=None) -> str:
    knn_txt = f", kNN-purity {knn['fold']:.1f}x" if knn else ""
    if cohesive_global is None:  # callers that only know the combined flag
        cohesive_global = z >= MIN_COHESION_Z and emp_p < MAX_COHESION_P

    if not cohesive and not has_enrichment:
        # The honest negative. It still carries the cross-corpus caution,
        # because "no structure" between two differently-embedded corpora is a
        # claim the precision artifact can affect just as much as a positive one.
        parts = [f"This selection resembles a random draw (z={z:.1f}, "
                 f"{_fmt_p(emp_p)}{knn_txt}): cohesion is not significant and "
                 f"nothing enriches at q < 0.05. No coherent structure detected"]
        if cross_dataset:
            parts.append(CROSS_DATASET_NOTE)
        return "; ".join(parts) + "."

    parts = []
    if cohesive and not cohesive_global:
        # The claim rests entirely on kNN purity while the global statistic
        # says the opposite. Printing a bare "Coherent" beside a z of -1.8 and
        # p=0.96 reads as a contradiction, or as a p-value nobody checked. Name
        # which measure carries the verdict and what the disagreement means:
        # tight neighbourhoods, spread apart. That shape is a real finding, not
        # a caveat to bury.
        shape = ("looser overall than a matched random draw" if z < 0
                 else "not significantly tighter overall than a matched random draw")
        local = f" ({knn_txt.lstrip(', ')})" if knn else ""
        parts.append(
            f"Locally coherent{local}, but {shape} (z={z:.1f}, {_fmt_p(emp_p)}): "
            f"several close-knit groups sitting apart from each other rather "
            f"than one cloud"
        )
    elif cohesive:
        parts.append(f"Coherent (z={z:.1f}, {_fmt_p(emp_p)}{knn_txt})")
    else:
        parts.append(f"Weak cohesion (z={z:.1f}, {_fmt_p(emp_p)}{knn_txt})")

    # Only categories with a real effect size get to be called drivers. At
    # |S| in the tens of thousands a 1.05x deviation reaches q < 1e-10 purely
    # on sample size, and naming it as the reason a selection is coherent is
    # exactly the false confidence this readout exists to avoid. Everything
    # significant still appears in the enrichment table with its own fold.
    drivers = [r for r in significant if r["fold"] >= MIN_DRIVER_FOLD]
    if drivers:
        text = "; ".join(
            f"{r['field']}={r['category']} ({r['fold']:.1f}x, {_fmt_q(r['q'])})"
            for r in drivers[:3]
        )
        parts.append(f"driven by {text}")
    elif significant:
        n = len(significant)
        subject = ("1 significant category is a small deviation" if n == 1
                   else f"{n} significant categories are small deviations")
        parts.append(
            f"no feature exceeds {MIN_DRIVER_FOLD:g}x enrichment, so the "
            f"{subject} made detectable by selection size rather than an "
            f"explanation of it")

    if batch and batch["flag"]:
        parts.append(
            f"CAUTION: likely study/batch driven - top study {batch['top_study']} "
            f"is {batch['fraction']*100:.0f}% of the OSDR selection")
    elif batch:
        parts.append(f"low batch confound (top study {batch['fraction']*100:.0f}%)")

    if cross_dataset:
        parts.append(CROSS_DATASET_NOTE)
    return "; ".join(parts) + "."
