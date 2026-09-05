# Interface simplification, second pass

This is a refinement of the existing navy, teal, and blue scientific interface. Both independent reviewers loaded taste and impeccable. The researcher review is a simulated expert assessment, not evidence of actual usage.

## Changes and rationale

| Before | After | Reason |
| --- | --- | --- |
| Fit results + Explore neighborhood | One Fit results action in 2-D; View results in 3-D | Both opened the same exact-cosine results drawer. The combined control fits only where fitting works. |
| Retrieve Box Select and Lasso Select | Removed from the Plotly toolbar | No callback consumes selectedData. Clicking individual nodes still opens metadata. |
| Bordered cohort-size cards repeating cohort names | Inline counts with A/B role names and color dots | Names are already in the selectors; actual included counts still update after exclusions. |
| Always-expanded Result stability panel | Closed disclosure with low-overlap warnings visible | Leave-one-out ranking sensitivity is useful when interpreting fragile results, but need not dominate every cohort search. |
| Always-expanded executed pooled-member list | Pooled members disclosure | Retains the exact executed membership separately from editable selections. |
| Study ID repeated in two query sections | One source link in Identity | The second row linked to the same record. Study title and other source metadata remain. |
| View 10 matches on map for two top-5 cohorts | View results on map | Shared hits mean the summed count need not equal the number of distinct samples. |

The complete stability measurements remain available: depth, per-arm overlap, single-sample baseline, pooling gain, and the member whose omission changes the result most. This sensitivity measure is distinct from member embedding similarity and from biological equivalence. The computations are unchanged.

## Deliberately preserved

Sample, cohort, comparison and uploaded-count queries; normalized pooling and exclusions; exact cosine retrieval; the fixed top-250 evidence neighborhood distinct from requested hits; map projections and settings; study/sample focus; metadata and source links; optional enrichment; AI interpretation with its caveats; and existing PNG exports. Leading-study summaries remain useful for a quick overview, while the Studies tab supports full inspection and focus. Metadata prefetch settings remain because they affect when records are fetched.

## Review and verification

Taste informed preservation of the incumbent visual identity and removal of decorative containers. Impeccable's distill and operate guidance informed action consolidation, inline counts, and native disclosures. Final review checks keyboard access, visible warnings, narrow layouts, and scientific invariants rather than treating a smaller interface as sufficient evidence of quality.

- 490 automated tests passed, including independent warning cases for cohorts A, B, and both.
- 202 main browser checks, 185 cohort browser checks, and 70 upload browser checks passed on the real corpus. Uploads were embedded twice through the real model, matched catalog results, and recovered after invalid inputs.
- Focused browser checks passed for source links, AI-response invalidation, PNG export, native disclosures, and desktop/tablet/mobile layout.
- The real-data join check confirmed the query and all 10 retrieved sample identities against map point indices.
- Both reviewers independently inspected the revised interface. The final pass fixed disclosure arrow states and contained the Plotly canvas during panel resizing.
- The CSS detector's remaining findings concern existing error/status accents and an existing meter transition; no new design-pattern findings were introduced.

Baseline and revised real-app screenshots and logs are available locally under `.lavish/simplification/`.

## Remaining concerns

This pass does not claim that specialized features are unused. The member checklist's cosine annotations refer to the original cohort geometry; the result stability diagnostic measures retrieval overlap instead. Those meanings should remain explicit if that advanced display is revised. This task changes no scientific algorithms or stored metadata. The duplicate-count map link was a presentation ambiguity, not a ranking or identity error.
