# Interface simplification, second pass

This is a refinement of the existing navy, teal, and blue scientific interface. Both independent reviewers loaded taste and impeccable. The researcher review is a simulated expert assessment, not evidence of actual usage.

## Changes and rationale

| Before | After | Reason |
| --- | --- | --- |
| Fit results + Explore neighborhood | One Fit results action in 2-D; View results in 3-D | Both opened the same exact-cosine results drawer. The combined control fits only where fitting works. |
| Retrieve Box Select and Lasso Select | Removed from the Plotly toolbar | No callback consumes selectedData. Clicking individual nodes still opens metadata. |
| Bordered cohort-size cards repeating cohort names | Inline counts with A/B role names and color dots | Names are already in the selectors; actual included counts still update after exclusions. |
| Result stability panel and diagnostics | Removed entirely | Explicit product decision to remove this feature. No hidden panel, warnings, payload, or result-removal calculation remains. |
| Always-expanded executed pooled-member list | Pooled members disclosure | Retains the exact executed membership separately from editable selections. |
| Study ID repeated in two query sections | One source link in Identity | The second row linked to the same record. Study title and other source metadata remain. |
| View 10 matches on map for two top-5 cohorts | View results on map | Shared hits mean the summed count need not equal the number of distinct samples. |

The Result stability feature is retired, including its panel, warning flags, callback, payload keys, CSS, measurement class, and leave-one-out query construction. The scientific design history and independently used offline validation remain. Per-member embedding cosine still supports reviewing sample exclusions, and A/B Jaccard overlap still describes the two independently retrieved hit sets.

Removing diagnostics required care with numerical behavior. A naive one-vector scan changes floating-point accumulation and can reorder near-tied matches. The pooled scorer therefore retains matrix multiplication using an inert zero column, discarded before ranking. Only the pooled biological query is scored and returned. This compatibility padding preserves existing results without calculating the retired diagnostic. On the installed NumPy/BLAS runtime, 25 real cohorts covering all 22 available cohort sizes (2–38 members) preserved top-30 hits, all 250 evidence neighbors, ordering, scores, metadata, and member rows bit-for-bit against the legacy pooled-row calculation. A focused fixture regression checks the same numerical contract.

## Deliberately preserved

Sample, cohort, comparison and uploaded-count queries; normalized pooling and exclusions; exact cosine retrieval; the fixed top-250 evidence neighborhood distinct from requested hits; map projections and settings; study/sample focus; metadata and source links; optional enrichment; AI interpretation with its caveats; and existing PNG exports. Leading-study summaries remain useful for a quick overview, while the Studies tab supports full inspection and focus. Metadata prefetch settings remain because they affect when records are fetched.

## Review and verification

Taste informed preservation of the incumbent visual identity and removal of decorative containers. Impeccable's distill and operate guidance informed action consolidation, inline counts, native membership disclosures, and complete removal of retired controls. Final review checks keyboard access, narrow layouts, and scientific invariants rather than treating a smaller interface as sufficient evidence of quality.

- 469 automated tests passed, covering removal of the panel and payload, preserved pooled ranking, and the core workflows.
- 202 main browser checks and 70 upload browser checks passed on the real corpus. Uploads were embedded twice through the real model, matched catalog results, and recovered after invalid inputs. 171 cohort browser checks passed for the final feature set, including explicit absence of the retired panel and both arms' diagnostic payloads.
- Focused browser checks passed for source links, AI-response invalidation, PNG export, native disclosures, and desktop/tablet/mobile layout.
- The real-data join check confirmed the query and all 10 retrieved sample identities against map point indices.
- Both reviewers independently inspected the revised interface. The final pass fixed disclosure arrow states and contained the Plotly canvas during panel resizing.
- The CSS detector's remaining findings concern existing error/status accents; the retired diagnostic meter styles are now removed. No new design-pattern findings were introduced.

Baseline and revised real-app screenshots and logs are available locally under `.lavish/simplification/`.

## Remaining concerns

This pass does not claim that specialized features are unused. The member checklist's cosine annotations refer to the original cohort geometry. That meaning should remain explicit if the advanced member display is revised. Pooling and retrieval semantics and stored metadata are preserved. Numerical compatibility tests specifically cover removal of the auxiliary diagnostic queries. The duplicate-count map link was a presentation ambiguity, not a ranking or identity error. At phone widths, some Plotly cohort labels still clip inside the network canvas; the inspector retains full identities.
