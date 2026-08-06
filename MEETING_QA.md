# Meeting Q&A

Answers grounded in the current codebase and its docs (`CLAUDE.md`, `IMPLEMENTATION.md`, `REFERENCE.md`, `progress.md`, `README.md`), plus the actual model code (`generate_archs4_embeddings.py`, `demo_osdr_top5.py`, `precompute/`, `bridge_rna/`, `manifold/`).
A handful of questions are about the team or the meeting itself rather than the code, and those are marked **[needs team input]** rather than guessed at.

Two things the meeting asked about as future possibilities have since been built, so their answers below were rewritten rather than left standing: live file ingestion (retrieval mode `uploaded`) and cohort pooling (retrieval mode `cohort`).
Where an answer changed because the code changed, it says so.

## Questions asked in the meeting

### What metadata is available from the ARCHS4 data enricher?

`precompute/fetch_archs4_meta.py` produces `cache/archs4_metadata.parquet`: 940,455 rows with columns `global_index`, `geo_accession`, `series_id`, `title`, `source_name`, `characteristics`, and a derived canonical `tissue` bucket.
It is sourced from the Maayan Lab sigpy JSON API (`POST https://maayanlab.cloud/sigpy/meta/samplemeta`), not from the ARCHS4 HDF5 files, and it resolves 99.911% of all 940,455 accessions (human 99.851%, mouse 99.982%) across 51,284 distinct GEO series.
The 839 unresolved samples (0.089%) get tissue `Unknown`; they are not GEO withdrawals, they are present in the older v2.5 release and simply absent from the newer v2.latest release the API serves.

### Could we examine whether a spaceflight sample is closer to a cancer-associated region?

Yes as an exploratory visualization: the shared tissue vocabulary already has a "Tumor / cancer" bucket covering 53,117 ARCHS4 samples (5.6%), so coloring the map by tissue and looking at where OSDR points fall relative to that bucket is possible today.
As a quantitative claim it needs more care.
A similar idea was already tried and rejected: a "spaceflight-likeness" axis built from cosine similarity to OSDR flight/ground centroids turned out to correlate r = -0.990 with PC1, which is a transcriptome-concentration axis, not a spaceflight signal, and it lost to random flight/ground relabelings about half the time under a within-study permutation test.
So a naive "distance to the cancer centroid" number would need the same null-testing rigor (a structure-free null of the same form, checked against transcriptome concentration) before being presented as a finding rather than a picture of the projection.

### Was the similarity-score mismatch fixed?

Two distinct issues were found and fixed, both recorded in `progress.md`.
First, the retrieval network's edge width encoded rank rather than similarity score: a min-max rescale drew the thinnest hit at 1.5 px and the thickest at 8 px regardless of the actual scores, so a real spread of 0.0016 looked as dramatic as 0.4 while the legend said "similarity score."
It now maps onto a fixed [0.90, 1.0] domain.
Second, the status banner mislabeled cached-path results as "real demo script output" because it special-cased only the `precomputed` mode; every cached result is now correctly announced as cached.
Separately, and not a bug: the cached and subprocess retrieval paths were verified to produce identical accessions and identical scores to six decimal places, so there was never an actual numerical mismatch between those two paths.

### What is the application built with, and how is it currently run?

Dash and Plotly, in Python.
The serving app's dependency surface is only `dash`, `plotly`, `numpy`, `pandas`, and `pyarrow`; the scientific stack (`torch`, `scikit-learn`, `umap-learn`, `openTSNE`, `pynndescent`) is precompute-only and is never imported at module scope in the serving path.
`app.py` is the single entry point, serving the retrieval view at `/` and the map view at `/map`.
It is run with `/Users/josh/Bridge-RNA/.venv/bin/python app.py` and serves at `http://127.0.0.1:8050`.

### How are the Plotly charts rendered?

The map is one `dcc.Graph` holding `go.Scattergl` (WebGL) traces, layered as an ARCHS4 background plus an OSDR overlay, because browsers cap WebGL contexts at roughly 8-16 and one shared context is the safe budget.
Coordinates travel to the browser as base64 typed arrays under Plotly 6's serialization rather than as JSON number lists, which is what makes shipping all 942,563 points affordable: 0.15 s and 11.3 MB for the whole corpus.
The retrieval view's network graph is built separately in `bridge_rna/figures.py`.

### Which visualizations are currently available?

The retrieval view shows a network graph of the nearest ARCHS4 hits around one query OSDR sample, with a detail inspector panel and an optional LLM summary.
Since the meeting it also pools a whole experimental group into one query and, when a sibling cohort differs by exactly one facet, runs both arms as two independent pooled queries and reports their overlap.
The map draws every pooled member rather than one point for the group, because a pooled query vector is a mean that no projection was fit on, and inventing a coordinate for it would be a lie.
With a comparison on screen the map is carrying four encodings at once (corpus hue, member hue, hit ring shape, and corpus glyph shape), so its floating panel keys all four rather than only color.
The map view shows a single shared 2D or 3D scatter of all 942,563 points (OSDR plus ARCHS4), switchable across three projection methods (PCA, UMAP, t-SNE), colorable by 11 registered fields (tissue, species, flight status, spaceflight arm, strain, sex, genotype, study, habitat, mission duration, diet), with an adjustable point budget and viewport-based level of detail on zoom.
Three earlier features are gone and are not being quietly reintroduced: the lasso selection tool with its 512-d statistical readout, the precomputed density-raster underlay, and an unsupervised k-means cluster color-by that was built, measured, and then deleted because it mostly just redrew the projection's own shape.

### How does the application obtain the OSDR and ARCHS4 data?

OSDR: `precompute/embed_osdr.py` reads the OSDR counts files and metadata, maps mouse genes to human orthologs, reindexes onto the fixed 15,165-gene canonical order, TPM-normalizes and log1p-transforms, then runs the result through `ExpressionPerformer.encode` to produce `cache/osdr_sample_embeddings.float32.npy` and `cache/osdr_metadata.parquet`.
ARCHS4: the 940,455 x 512 embeddings already exist as a precomputed memmap built previously; nothing is re-embedded or re-downloaded.
ARCHS4's per-sample GEO metadata comes from the sigpy JSON API described above, not from any HDF5 download.

### When users change the sidebar controls, is the analysis recomputed?

No.
Every coordinate set (PCA, UMAP, t-SNE, in 2D and 3D) is precomputed offline and cached; changing color-by, projection method, or point budget on the sidebar only selects and re-renders from those cached parquet tables.
The one thing that does recompute live is a stratified re-sample of the ARCHS4 background points when the user zooms, restricted to the new viewport, so zooming reveals finer structure instead of just enlarging sparse dots; that is sampling for display, not a projection recompute.

### Should users eventually be able to rerun PCA, UMAP, or t-SNE?

Not currently supported, and it is a deliberate non-goal so far: the offline/online split exists specifically because these fits are expensive (UMAP is roughly 5-6 minutes per dimensionality on the full corpus, t-SNE's 3D fit alone is about 2.3 hours), which is far outside what an interactive session can absorb.
This is a genuinely open question for the team to weigh rather than something the codebase has already answered: an on-demand *partial* refit (e.g., a smaller subsample, or restricting to a filtered subset) might be tractable where a full corpus refit is not, but nothing like that exists today.

### Could a researcher upload a new RNA-seq sample and search for similar samples?

Yes, and this changed after the meeting: it is now built and shipped as retrieval mode `uploaded`.
A user uploads an OSDR counts file, it is embedded live, and it is scored against the same 940,455-sample ARCHS4 index as every other query.
It is a fourth query-vector source rather than a second pipeline, so the cosine scan, the annotation step, and the map join are the cached path's, reused unchanged, which is what makes an uploaded hit carry exactly the same schema as a precomputed one.
`bridge_rna.retrieval.run_uploaded_retrieval` is the entry point, and the embedding happens in a subprocess (`precompute/embed_upload.py`) because the serving app never imports torch at module scope.
That subprocess reuses the exact preprocessing symbols the corpus itself was built with, and it enforces the gene-digest gate before producing any vector, so a file is either embedded in the same gene order as the corpus or the embed is refused.
It was verified rather than assumed: embedding an OSDR sample's own counts through this path reproduces its precomputed vector at cosine 1.0, with a maximum absolute difference of 0.0.
Input is mouse Ensembl-indexed counts, and a file that maps zero orthologs is rejected rather than embedded into a meaningless vector.
No metadata is required or accepted, because the query vector is a pure function of one counts column, so tissue or flight status could only fill the inspector and never change a hit or a score.
The design doc is `docs/file_ingestion.md`.
The format contract and a real working input are in `examples/README.md` and `examples/osdr_upload_example.csv`.

### Can multiple biological replicates be combined into one representative or population embedding?

Yes, and this also changed after the meeting: it is now built and shipped as retrieval mode `cohort`.
The argument for it turned out to be stronger than the "replicates cluster tightly" reasoning given at the time.
A single-sample top-5 is not a stable measurement: two replicates from the same cage share only 0.161 of their top-5 hits, because the entire top-500 of a 940,455-sample index spans a cosine range comparable to the gap between two animals in the same cage.
Pooling raises leave-one-out top-5 agreement to 0.738, a 4.6x gain, and that is the case for the feature rather than protection against outliers.
A cohort is `(study, tissue, spaceflight arm)`, which is OSDR's own curated factor grouping, and it yields 212 cohorts of two or more members across 70 studies, median size 10 and maximum 38, covering 2,105 of the 2,108 embedded samples.
Study can never be unticked, because random samples drawn from one study already reach 0.9805 mean pairwise cosine against 0.9933 for a real cohort, so pooling across studies would average across the corpus's strongest batch boundary.
Each member is L2-normalized before averaging, which is the maximum-likelihood vMF mean direction and makes ranking by the pooled vector exactly ranking by the unweighted mean of the members' own cosines, one animal one vote.
The confidence readout quotes a measured stability curve as a function of cohort size (0.34, 0.51, 0.55, 0.72, 0.81 and 0.86 at k of 2, 3, 4, 5-9, 10-14 and 15+) rather than a tightness score, because tightness never separated a trustworthy group from an untrustworthy one.
Design and every measurement behind it: `docs/cohort_retrieval.md`.

### Would averaging multiple embeddings erase meaningful biological differences?

It depends on what is being averaged.
Averaging true replicates of the same condition should be safe, since they are near-identical in the embedding space and most of what differs between them is technical noise.
Averaging across genuinely different biological conditions is the documented failure mode: a flight-minus-ground centroid axis, meant to capture "spaceflight-likeness," was tried and rejected precisely because it collapsed real variation into a single number that actually tracked transcriptome concentration (correlating r = -0.990 with PC1) rather than spaceflight biology, and it lost to random relabeling nulls nearly half the time.
So: safe within a replicate group, unsafe across biologically distinct groups.
That is exactly the line the shipped cohort feature draws.
It pools only within one curated experimental group, and its two-arm comparison runs two independent pooled queries and reports their overlap, rather than ever ranking the index against a `centroid(flight) - centroid(ground)` difference vector.
A difference of two means is not a transcriptome, so cosine-ranking an index of profiles against it is a category error, and the corpus-level version of that idea was built and rejected on exactly the evidence described in the previous answer.

### Does inconsistent tissue naming cause samples to be missed or mislabeled?

Yes, and this was a real, measured problem, which is why `manifold/tissue.py` exists at all.
OSDR uses 48 hyper-specific curated terms and ARCHS4's signal lives in 42,754 distinct free-text GEO strings, with no shared vocabulary between them; left alone, "Tissue" would have to be two color-bys, each leaving the other corpus grey.
The canonical-bucket mapper (40 ordered keyword rules into 37 buckets plus "Other" and "Unknown") now covers 100% of OSDR and 90.6% of ARCHS4, with the remaining 9.4% honestly marked "Other" or "Unknown" rather than guessed at.
Two real bugs were caught by ordering and word-boundary mistakes, such as "renal" mis-firing inside "adrenal," and by conflating "Other" with "Unknown," which had been mislabeling HeLa samples until the two were ranked and kept distinct.

### Should the team add t-SNE or other dimensionality-reduction methods?

t-SNE already shipped on 2026-07-23 (openTSNE, perplexity 30, PCA initialization), following a ten-method evaluation run earlier that specifically recommended it if a third method were ever added.
It is the strongest performer of the three on local and biological fidelity (it beats UMAP on both kNN recall and tissue purity in both dimensionalities), but it separates the two corpora more than UMAP does (2.2% shared bins versus 8.9%), which is a real weakness for this app's core question of whether a spaceflight sample sits among Earth samples.
Adding a fourth method is mechanically cheap (one registry entry plus a build stage), but nothing else in that ten-method evaluation scored competitively, so there is no obvious next candidate waiting in the wings.

### Should the presentation use slides, a live demonstration, or both?

**[Needs team input]** This is a presentation-logistics decision, not something derivable from the codebase.

### What happened to the proposed ontology and knowledge-graph work?

**[Needs team input]** No trace of an ontology or knowledge-graph proposal appears anywhere in `progress.md`, `IMPLEMENTATION.md`, `REFERENCE.md`, or `README.md`; if this was discussed, it happened outside what is recorded in this repository's history.

### What is the team's main internship project outside this collaboration?

**[Needs team input]** Not something the codebase or its docs would record.

### What were the presentation logistics?

**[Needs team input]** Same as above; check meeting notes or calendar invites rather than this repo.

## Additional questions people may ask

### What problem does this tool solve?

Two related problems.
Retrieve answers "given one NASA spaceflight RNA-seq sample, what are its closest Earth-based analogs," fast, out of a 940,455-sample GEO index.
Map answers the wider question, "what is the overall shape of that whole embedding space, and where does spaceflight sit inside it," by putting both corpora into one shared, browsable 2D/3D view.

### What exactly does the BERT-style model learn?

`ExpressionPerformer` is a 12-layer transformer trained with a masked-expression-prediction objective: its `forward()` path applies a linear head per gene token to predict masked expression values from the surrounding, unmasked genes.
Through that training it learns co-expression structure across the 15,165-gene panel, i.e. which genes' expression levels predict others and which combinations characterize particular tissues or cell states.
`encode()`, the inference path used everywhere in this app, reuses those same learned per-gene representations and mean-pools them into one general-purpose sample embedding; it does not run the masked-prediction head at all.

### Is each gene represented by 512 values, or is each sample?

Both, at different stages, but only the sample-level vector is ever used downstream.
Internally, each of the 15,165 genes gets its own 512-dimensional hidden vector after the transformer layers run.
The final step averages all 15,165 of those gene vectors together into a single 512-d vector per sample (`h.mean(dim=1)` in `encode()`), and that one 512-d vector per sample is what powers retrieval and the map.

### Why use embeddings instead of comparing raw gene-expression profiles?

A raw profile is 15,165 numbers, noisy, and not directly comparable across samples processed on different hardware or precision.
The learned 512-d embedding compresses that into a space specifically trained to capture co-expression structure, which makes both search and comparison cheaper and, in principle, more robust than a naive distance over 15,165 raw values: a 512-d cosine top-k search over 940,455 rows runs in about half a second.

### How is similarity between samples measured?

Cosine similarity between L2-normalized 512-d embeddings.
`bridge_rna/retrieval.py`'s `_topk_cosine_from_memmap` computes this over the ARCHS4 memmap in 25,000-row chunks, and both UMAP and t-SNE are built on the same cosine metric over the same normalized vectors, so retrieval results and map placement are answering the same geometric question.

### How do you know the nearest neighbors are biologically meaningful?

Several independent checks, not just visual inspection.
The tissue color-by scores 25-NN label purity of 0.8142 against a permuted-label null of 0.0501, and it survives both a batch control and a depth control at 0.7058.
Species is the single clearest structure in the space, at spatial eta-squared 0.985, well clear of the 0.87-0.94 band that 30 arbitrary random directions score just by virtue of UMAP having been fit on those same vectors.
Same-study OSDR replicate pairs neighbor each other at roughly 5,233x their chance rate, which is the expected biological (replicate) signal rather than noise.
UMAP and t-SNE are also scored directly against the exact 512-d nearest-neighbor graph (kNN recall), not judged by eye.

### Are distances in a UMAP plot the same as distances in the original 512-dimensional embedding space?

No.
UMAP (and t-SNE) preserve local neighborhoods, not global distances, so cluster separation and cluster size on the plot are not quantitatively meaningful, and the gap between two visual clusters does not measure how biologically different they are.

### Why do the 2D and 3D plots look different?

For UMAP, the 2D and 3D fits share the same k=30 neighbor graph but are two independent layout optimizations at different output dimensionality, so their global arrangement can differ even though both preserve the same local neighbor structure.
For t-SNE it is more fundamental: 2D uses openTSNE's FIt-SNE interpolation accelerator, while 3D falls back to Barnes-Hut, because that accelerator explicitly refuses more than two output dimensions.
The 2D and 3D t-SNE fits are literally different algorithms, not just two views of one computation, which is also why the 3D t-SNE fit costs about 20x its own 2D fit and dominates the whole build's wall clock.

### Does a visible cluster prove a biological discovery?

No, and this is one of the project's core methodological rules.
Thirty arbitrary random directions in 512-d score spatial eta-squared 0.874 ± 0.025 on this UMAP purely because the UMAP was fit on those same vectors, so anything scoring in the 0.89-0.94 band is indistinguishable from an arbitrary projection.
A visible cluster needs to be checked against a structure-free null of the same form (permuted labels, random directions, or a Voronoi partition, depending on what kind of field it is) and checked for whether it is recoverable from the coordinates themselves or from transcriptome concentration, before it is treated as a finding rather than a picture of the projection.
Only species (0.985) clearly clears that bar today.

### How are batch effects handled?

Made visible, not corrected.
Study and species are offered as explicit color-bys, and the measured cross-corpus batch effect (OSDR pairs sharing neither study nor tissue neighbor each other 54x above chance, attributable to fp32/CPU versus bf16/CUDA inference differences) is documented in the docs rather than corrected with something like Harmony or ComBat.
This was a deliberate final decision, made specifically to avoid a correction algorithm risking the erasure of real biology, and `validate_artifacts.py --mixing` recomputes the number and warns above 50x so the claim stays tied to a measurement anyone can rerun.

### How do you prevent species or tissue from dominating every search?

Nothing actively reweights or filters this today; it is an open caveat rather than a solved problem.
Species is in fact the strongest, cleanest structure in the whole embedding space (eta-squared 0.985), so it will tend to dominate nearest-neighbor results unless a user separately checks the species or tissue color-by to interpret hits in context.
There is no scoped "search only within this species/tissue" option in retrieval yet.

### What happens when metadata is missing?

It is disclosed rather than hidden or guessed at.
The 839 ARCHS4 samples missing GEO metadata (0.089%) get tissue `Unknown`, kept distinct from `Other`.
If the whole `archs4_metadata.parquet` file is absent, the Tissue color-by is shown disabled with the exact command that would enable it, and Species remains the whole-map default.
Whenever a selected color-by field does not describe a corpus, that corpus is drawn as a faint, un-legended context cloud rather than as a colored (or flat grey) category, so absence of data is never presented as though it were data.

### How often will the reference dataset and vector index be updated?

Not something the codebase establishes as a cadence; there is no automated refresh pipeline.
The precompute pipeline (`embed_osdr.py`, `build_projections.py`, `fetch_archs4_meta.py`, `validate_artifacts.py`) is run manually and offline, so an update cadence is a team decision rather than an existing schedule.

### Can the system support models other than this BERT-based model?

In principle yes, since the serving app only ever consumes precomputed 512-d embeddings and a memmap format.
In practice, swapping the encoder is a full re-embedding effort, not a config change: the gene-digest gate, the canonical gene ordering, and the entire ARCHS4 index are all tied to this specific `ExpressionPerformer` checkpoint, so a new model means re-embedding all 940,455 ARCHS4 samples and 2,108 OSDR samples and rebuilding the map cache from scratch.

### How scalable is the tool?

Proven at the current scale of 942,563 points end to end: exact full-corpus PCA runs in about 5 seconds, each UMAP dimensionality takes roughly 5-6 minutes, and the 3D t-SNE fit (the bottleneck) takes about 2.3 hours; the serving app renders all 942,563 points live with a 0.15-second figure serialization.
Most stages scale close to linearly or n log n with corpus size, so a meaningfully larger corpus is plausible without an architecture change, though the 3D t-SNE stage would become an even larger share of build time.

### Why not display all 940,000-plus samples at once?

It already does, by default.
An earlier design assumed that many live WebGL points was out of reach and compensated with a precomputed density-raster underlay; when actually measured, rendering cost turned out to be dominated by resolving one label array over the full corpus rather than by how many points were drawn, so the raster bought smoothness that did not need buying and was removed entirely.
3D still caps at 40,000 points because Plotly's `Scatter3d` has no WebGL fast path comparable to `Scattergl`.

### What are the biggest limitations of the current prototype?

No batch-effect correction (a deliberate design choice, but still a real limitation for direct cross-corpus comparison); UMAP and t-SNE distances are not globally meaningful, which invites over-reading if a viewer is not careful; about 9.4% of ARCHS4 samples carry only "Other" or "Unknown" tissue labels; projections cannot be rerun on demand; retrieval still cannot be scoped to a species or tissue; and there is currently no validated single-number "spaceflight-likeness" metric, since the one attempt at that failed its own null test.
The upload limitation listed here at the time of the meeting is now resolved, and uploaded samples are restricted to mouse Ensembl-indexed counts.
Species also currently dominates nearest-neighbor structure, so results should be interpreted alongside species/tissue context rather than taken as pure spaceflight signal.

### What would make the tool production-ready?

This is a recommendation rather than something the codebase already states.
Reasonable next steps, grounded in what the repo's own "rejected candidates" and "non-goals" sections already flag as open: a scheduled or at least repeatable re-embedding process for new ARCHS4/OSDR releases, a properly null-tested quantitative similarity metric if a continuous score is ever wanted, species/tissue-scoped search filters, and keeping `validate_artifacts.py` and `validate_cohorts.py` as hard gates in any CI process.
The upload pipeline that headed this list at the time of the meeting has since been built.
One item genuinely not on this list is a production web deployment: the app binds to 127.0.0.1 by default and is run locally, and the argument-parser refuses to serve the Werkzeug debugger on any non-loopback interface precisely because that console executes arbitrary code for anyone who can reach the port.

### What is the clearest next experiment or validation test?

Given the project's own established standard, the clearest next test is running whatever new candidate is proposed next (a new color-by, a new similarity metric, a cancer-proximity score) through the same rigor already applied to everything else here: a structure-free null of the same form, tissue/batch/depth controls, and `validate_artifacts.py --quality --compare` against the shipped baseline, before it is presented as a finding.

### What should the audience remember about this project?

Three points worth carrying out of the room: one shared 512-d embedding space now unifies 2,108 NASA spaceflight samples with 940,455 Earth reference samples, so a researcher can instantly find Earth analogs for a spaceflight sample; the map is deliberately "read, not queried," with every color-by and projection choice measured against a null baseline rather than eyeballed; and all the heavy computation (model inference, PCA, UMAP, t-SNE) happens once, offline, so the live app itself stays fast and lightweight.

### How was the BERT model trained on ARCHS4 data?

What is verifiable from the checkpoint is its architecture and inference-time configuration: hidden dim 512, 8 attention heads, 12 layers, feed-forward dim 2048, flash attention, `log1p_tpm` normalization, mask token -10, no species embedding, seed 42.
Its training objective is masked expression prediction, since `forward()` applies a per-gene linear head to predict masked expression values from context.
Details of the training run itself, such as data splits, epoch count, or hyperparameter search, are not documented anywhere in this repository beyond the checkpoint carrying `optimizer_state_dict`, `scheduler_state_dict`, `epoch`, `train_loss`, and `val_loss`; that history is outside what this codebase can answer.

### How is an RNA-seq sample converted into a 512-dimensional embedding?

A fixed-length, canonically-ordered log1p-TPM vector is built for the sample; each gene position gets a learned gene-identity embedding added to a rotary (sinusoidal) embedding of its expression value; the combined per-gene vectors pass through 12 transformer layers; and the resulting per-gene hidden states are mean-pooled across the gene axis into one 512-d vector (`ExpressionPerformer.encode`, `generate_archs4_embeddings.py:187-193`).

### What preprocessing must be applied before a sample enters the model?

For an OSDR mouse sample specifically: read the raw per-sample counts, strip Ensembl version suffixes, map mouse Ensembl IDs to human gene symbols through one-to-one orthologs (summing any duplicates that collapse onto the same human gene), reindex onto the fixed 15,165-gene canonical order (filling any missing gene with 0), TPM-normalize using mouse exon lengths, and finally apply log1p.

### How are missing genes or differently ordered genes handled?

Every sample is reindexed onto one fixed, 15,165-gene canonical order; any gene absent from a given sample's own data is filled with 0 at that position.
The order itself is protected by a hard gate: `canonical_gene_order_digest` must match a pinned SHA256 hash, and the build aborts on mismatch, specifically because a subtly wrong gene order would otherwise produce embeddings that look normal but are scientifically invalid.

### Why was cosine similarity selected?

The docs do not record an explicit comparison against alternative metrics, but the design is consistent throughout: L2-normalizing the 512-d vectors and comparing under cosine makes vector magnitude, which turned out to be a real but separate "transcriptome concentration" signal, irrelevant to the similarity comparison itself.
It is also the same metric used to build both UMAP and t-SNE's neighbor graphs, which keeps retrieval results and map placement consistent with one another.

### How should similarity scores be interpreted?

As a relative ranking among a sample's nearest ARCHS4 neighbors, not as a calibrated probability or a percentage of "how similar."
Measured real-hit score spreads can be as small as 0.0016, and absolute cosine similarities routinely sit at 0.96 or higher even for cross-species matches, which is exactly why the retrieval network's edge-width encoding was fixed to stretch across a realistic [0.90, 1.0] domain rather than a full [0, 1] range: scores should be compared to each other, not read as standalone confidence numbers.

### Does the model distinguish biological signal from technical noise?

Not inherently; the model itself has no explicit noise or batch term.
The application instead separates the two after the fact, by measurement: a 54x tissue-controlled cross-corpus batch effect and a 5,233x same-study replicate effect are both explicitly quantified, and every proposed color-by or similarity metric is tested against a structure-free null before being trusted as biology, which is exactly how the "spaceflight-likeness" axis was caught tracking transcriptome concentration instead of spaceflight.

### Can the embedding explain which genes caused two samples to be considered similar?

Not currently.
Because the sample-level vector is a mean over all 15,165 per-gene hidden states, there is no built-in mechanism that attributes similarity between two specific samples back to individual genes; that would require a separate interpretability step, such as ablating genes or inspecting attention weights, which is not part of the current pipeline.

### Can researchers search within a particular species, tissue, or disease?

Not as a retrieval filter today.
The retrieval view returns the top-k nearest ARCHS4 neighbors by cosine similarity over the whole 940,455-sample index with no species, tissue, or disease constraint applied to the search itself.
On the map, species and tissue can be used as a color-by to visually inspect where results land after the fact, but there is no scoped or filtered nearest-neighbor query.

### How reproducible are the PCA and UMAP visualizations?

PCA is exactly reproducible: it is a deterministic streaming eigendecomposition, verified to agree with scikit-learn's PCA to float64 round-off, with eigenvector signs pinned by a fixed convention specifically so a rebuild cannot silently mirror the map.
UMAP is reproducible bit-for-bit given the same seed (`random_state=42`) and single-threaded neighbor search, but that reproducibility carries a real, accepted cost: it forces a single-threaded layout that runs 4.3-7.5x slower than the parallel path, and a faster multi-threaded neighbor search would break bitwise reproducibility.
This is a deliberate build-time tradeoff, made because the projections are a once-per-corpus offline artifact, not a limitation of the methods themselves.

### How will uploaded research data be secured?

The upload feature now exists, so this answer describes what it actually does rather than deferring.
The posture starts with where the app runs: it binds to 127.0.0.1 by default, so an upload does not cross a network at all in the normal local setup.
An uploaded file is staged rather than merely written.
`bridge_rna.callbacks._upload_dir` owns exactly one temporary directory per process, a session's previous file is unlinked as soon as its next upload arrives, and the deletion is confined to that directory so a malformed or stale client store cannot point it at anything the app did not itself write (`bridge_rna/callbacks.py:265-279`).
A directory abandoned by a killed process is reaped by the next run rather than leaking forever, and the reaping is PID-tagged rather than exit-based on purpose: `atexit` does not run on SIGTERM or SIGKILL, and a signal handler is not available either, because uploads arrive on Dash's request threads while `signal.signal` may only be called from the main one (`bridge_rna/callbacks.py:211-241`).
Before that fix, every upload leaked its counts matrix into the system temp directory permanently.
No metadata is required or accepted alongside the counts, so the feature collects nothing about the sample or the person uploading it.
An upload is capped at 200 MB, which is generous headroom over the few MB a whole-transcriptome counts file actually costs, and the cap is enforced in two places: as Flask's `MAX_CONTENT_LENGTH` so an oversized body fails at the request layer (`app.py:262`), and again in the callback after the base64 decode, where the user is told the actual size and the limit (`bridge_rna/callbacks.py:1004-1007`).
A file that fails any later check, such as mapping zero orthologs, is taken back off disk rather than left staged.

### Can this approach be applied to other omics datasets or foundation models?

The pattern is conceptually general: a fixed-order numeric feature vector goes through a transformer, gets mean-pooled into an embedding, and is then compared by cosine similarity and visualized with UMAP/t-SNE.
Nothing in the current code is actually generic, though: the canonical gene list, the gene-digest gate, the ortholog maps, and the trained checkpoint are all specific to this bulk RNA-seq and ARCHS4 setup.
Applying the approach to another omics modality (proteomics, ATAC-seq) or a different foundation model would mean retraining the encoder and rebuilding the entire embedding index and map cache from scratch; the design pattern transfers, but none of the artifacts do.
