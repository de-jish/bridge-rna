# Example upload: what a counts file has to look like

`osdr_upload_example.csv` is a real, working input for the Retrieve view's **Or upload a sample** control.
It is two sample columns lifted verbatim from OSD-100's unnormalized counts matrix (`GLDS-100_rna_seq_Unnormalized_Counts.csv`), one spaceflight eye and one ground control, with all 55,536 gene rows intact.
Upload it, pick a column, click **Embed & search uploaded sample**, and you get the same network graph, inspector, and optional summary that the sample picker produces.

## The format, in one screenful

```csv
gene_id,Mmus_C57-6J_EYE_FLT_Rep1_M23,Mmus_C57-6J_EYE_GC_Rep1_M33
ENSMUSG00000000001,4064,3310
ENSMUSG00000000003,0,0
ENSMUSG00000000028,144,98
ENSMUSG00000000031,1603,1006
ENSMUSG00000000037,183,132
...
```

- **Column 1 is mouse Ensembl gene IDs.** `ENSMUSG...`. A version suffix is fine, `ENSMUSG00000000001.5` is stripped before the lookup.
- **Every other column is one sample.** The header cell is the sample's name, and it is what the column picker offers you. One column is fine; the picker only appears when there is a choice to make.
- **Cells are raw counts.** Unnormalized integers straight out of STAR/RSEM. Do not pre-normalize: the pipeline does TPM itself, and handing it TPM or CPM would normalize twice.
- **Row order does not matter** and neither does completeness. The file is reindexed onto the 15,165 canonical genes and anything missing is filled with 0.
- **The header cell of column 1 is ignored.** `gene_id` here, empty in the OSDR originals, both work.

## What is required and what is not

**Required:** the counts matrix. That is the entire input.

**Not required, and not currently accepted:** any metadata at all.
No tissue, no flight-vs-ground, no strain, sex, duration, or OSD accession.
The 512-d query vector is a pure function of one counts column, so no metadata you could supply would move a single retrieved hit or change a single similarity score.
Metadata would only fill in the inspector's Biology rows and give the summary model something to reason about, both of which are blank for an uploaded sample today.

## The rules that will reject a file

Each of these fails with one clean line in the status banner rather than producing a wrong answer quietly.

| Input | What happens |
| --- | --- |
| Mouse Ensembl IDs (`ENSMUSG...`) | Accepted. This is the contract. |
| Human Ensembl IDs (`ENSG...`) | **Rejected.** Zero genes map through the mouse-to-human ortholog table. OSDR spaceflight data is *Mus musculus* and the preprocessing maps into human ortholog space, so a human matrix is not silently mis-mapped. |
| Gene symbols (`Actb`, `Gapdh`) | **Rejected**, same reason: the ortholog table is keyed on Ensembl IDs. |
| No sample columns (one column total) | **Rejected.** "Expected mouse Ensembl gene IDs in the first column and one or more sample columns." |
| Not a parseable table | **Rejected** with the parser's reason. |
| Over 200 MB | **Rejected** before anything is read. The cap is `MAX_UPLOAD_BYTES`. |
| A gene order that is not the canonical one | **Aborted before any vector exists.** This is invariant 1, the gene-digest gate, and it is the one failure that would otherwise be invisible. |

## Accepted variations

- `.csv` (comma) and `.tsv` / `.txt` (tab).
- Gzipped: `.csv.gz`, `.tsv.gz`. Pandas decompresses by suffix.
- Version-suffixed IDs, as above.
- Duplicate gene IDs, which are summed after the ortholog mapping rather than being an error.
- Extra sample columns you do not intend to query. Only the column you pick is embedded.

## Why you can trust an uploaded score against a catalog one

`precompute/embed_upload.py` does not re-implement preprocessing.
It imports the exact symbols the corpus build uses through `manifold/bridge_rna.py`, runs the same ortholog map, the same canonical reindex, the same TPM and `log1p`, and the same `model.encode(x, None, normalize=False)`.
`tests/test_upload_ingestion.py::test_live_upload_embedding_matches_the_cached_corpus_vector` uploads a catalog sample's own counts and checks the live vector reproduces its precomputed one, measured at cosine 1.0 and max absolute difference 0.0.
That is what makes a cosine of 0.83 against a GEO sample mean the same thing whichever path produced it.

Design notes: [`docs/design-notes.md`](docs/design-notes.md#file-ingestion).
Browser-level verification of this file: `tests/e2e_upload_check.py`.
