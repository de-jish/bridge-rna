# Interface cleanup

This cleanup followed two independent reviews: language and interface clarity,
and a simulated space biosciences researcher walkthrough. Their findings are
implementation observations and expert judgments, not evidence of actual usage.

## Product scope verified

The running app and repository support mouse OSDR sample, cohort, and uploaded
counts queries; precomputed or live ExpressionPerformer embeddings; exact cosine
retrieval against the ARCHS4 index; a relationship network; and shared projected
maps with metadata inspection. OSDR includes flight and ground-control samples.
The local corpus contains 2,108 OSDR and 940,455 ARCHS4 samples. The index is a
specific reference collection, not every terrestrial transcriptome.

## Changes and rationale

| Before | After | Reason |
| --- | --- | --- |
| “NASA spaceflight transcriptomes, against all of Earth's” | “Compare mouse OSDR RNA-seq with ARCHS4” | State the actual query and reference scope. |
| “Top-k neighbors” | “Number of matches” | Name the quantity the control sets. |
| “AI hypothesis” and “Beta” | Collapsed “AI summary” with an interpretation caveat | Keep optional generation available without competing with source metadata. |
| “Frame it” / “Frame the retrieval” | “Fit study” / “Fit results” | Name the object affected by the view action. |
| “ARCHS4 live” | “ARCHS4 shown” | Describe displayed points rather than the rendering mechanism. |
| Unlabeled score badge | “Cosine” plus the existing score | Identify the metric without changing precision or values. |
| Em dash for missing information | “Not recorded” or “Unavailable” | Make absence explicit; source punctuation is preserved. |

Removed repeated starting instructions, decorative network/AI panel dots, the
Beta/Optional badges, the visible Search controls and Retrieval headings, and the
nested selected-sample card treatment. The selected sample's full name and
metadata remain visible. The inspector now explains selection once, and the
canvas carries the initial search instruction. Status space is reserved for
actual search outcomes and errors.

Moved recorded projection settings into a native disclosure directly under
Projection, retaining all dimension-aware values and availability checks. The AI
disclosure keeps generation, provider setup/error messages, and output available.
Metadata enrichment remains opt-in under Metadata options.

Made existing recognized OSDR/GEO identifiers clickable in the inspector,
including cohort studies. Unknown identifiers and uploaded sample identities
remain plain text. Links retain the identifier's spelling and target the source
record; uploaded files are not represented as OSDR studies. Shared query labels
say “Query sample” or “Query” so uploads do not imply OSDR provenance.

## Scientific integrity and separate concerns

- No changes to preprocessing, canonical gene order, embeddings, cosine scoring,
  ranking, cohort arithmetic, tissue rules, or projected coordinates.
- Preserved sample and cohort queries, member exclusions, independent cohort
  comparison, measured stability, metadata enrichment, network/map navigation,
  neighborhood Overview/Studies/Samples, source metadata, all map settings,
  coverage counts, and the existing PNG image download.
- Tissue help now identifies standardized categories derived from source text.
  Other means no rule matched; Unknown means no usable tissue text. Category
  counts and their original denominators are unchanged.
- Neighborhood study groups take an example sample title from their members.
  Previously this could be mistaken for the study title. The display now labels
  it “Example sample”; aggregation is unchanged. Fetching actual study titles
  would be separate work.
- AI output could survive a new retrieval. Its callback now clears output and
  status when results change; an older in-flight response is superseded. This is
  a result-context fix, not a change to generation or scientific computation.
- There is no existing tabular result export. Adding one requires a separate
  specification for scores, provenance, and reproducibility fields.

## Design and verification

Both reviewers and the implementer loaded taste (`design-taste-frontend`) and
impeccable. Taste's preservation guidance kept the navy header, teal rule,
typography, and data encodings. Impeccable's Operate, Distill, Clarify, and Polish
guidance informed the task hierarchy, disclosures, precise labels, and bounded
desktop/mobile review. No new dependency, animation, or design system was added.

Baseline: 483 pytest tests passed. Focused regression tests first reproduced the
missing source links and stale AI context. The revised suite passes 485 tests.
Real-corpus identity checks pass for the query and ten retrieved hits. Existing
browser suites cover core retrieval, cohort comparison/stability, upload
embedding, map joins, projection settings, neighborhood interaction, and error
states: 201 main, 180 cohort, and 70 upload checks passed. `tests/e2e_cleanup_check.py` adds keyboard disclosure, source access,
late AI-response invalidation, existing image download, and narrow-screen checks.
It controls only the AI response to reproduce a race; it does not validate the
scientific quality of a generated summary.

The broad mechanical design scan reported eight existing stylesheet patterns:
status/cohort accent borders, a panel accent edge, and width transitions on
existing indicators. These are outside the changed rules; semantic status and
cohort markings were retained. The visual reviews found no new layout defects.

Remaining product uncertainties: the value of AI interpretation and the best
default point density need researcher feedback. Neither review supplies usage
evidence for removing a specialized feature. The existing long stacked mobile
workflow remains; a navigation redesign was outside this cleanup.
