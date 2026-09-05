# Bridge RNA contributor guidance

Read `CLAUDE.md` for architecture, scientific invariants, commands, and the existing visual language.

## Interface copy and feature simplicity

- Before planning, implementing, or reviewing any UI or UX change, load and use both **taste** (`design-taste-frontend`) and **impeccable**. This applies to every agent. Report a missing skill explicitly. Apply both again during final review.
- Preserve the navy header, teal rule, existing type, accessible tokens, and data encodings. Refine the working instrument rather than redesigning it.
- Write plain, precise copy. Name actions and their objects consistently. Avoid promotional claims, repeated instructions, decorative badges, and authored em dashes. Preserve punctuation in scientific notation and source metadata.
- Describe the actual reference collection and query source. Embedding cosine similarity does not establish biological equivalence; projected map distances do not determine retrieval rank.
- Distinguish source annotations from derived tissue categories and AI interpretation. Keep identifiers, source links, provenance, missing-data distinctions, and reproduction settings accessible.
- Give each action one clear entry point. Do not expose graph selection tools without a consumer for the selection. Prefer inline counts over standalone statistic cards; keep result-sensitivity diagnostics optional and their warnings visible.
- Use contextual disclosures for secondary explanations and specialized settings. Remove a capability only after tracing its dependencies and establishing redundancy or lack of research value. Do not call it unused without usage evidence.
- Keep sample identities, metadata, labels, and coordinates aligned. A cosmetic cleanup must not change preprocessing, embeddings, ranking, grouping, or projection algorithms. Document scientific concerns separately.
- Verify representative sample, cohort, upload, map, metadata, and available export workflows in the running app. Check desktop and narrow layouts, review the scientific diff, and preserve unrelated work when committing.

Prefer `rtk`-prefixed shell commands when installed. If it is unavailable, report that and use a transparent command fallback.
