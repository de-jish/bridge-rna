# The key on the map, and describing both pooled queries

**Status: built, run against the real corpus, and tested, 2026-08-06.**
This is the implementation document for a copy pass over both views plus one real design change: the map now keys every mark it draws, and a two-arm cohort comparison describes both of its pooled queries instead of one.

| | |
| --- | --- |
| Encodings the map draws at once | **4** - corpus hue, member fill hue, hit ring shape, corpus glyph shape |
| Encodings it explained before this | **1** |
| Defects found on the way and fixed | **5** - two in the map as it stood, three in this change, all found by audit or review rather than by a test |
| Sentences removed from the interface | **7** |
| Tests | 307 unit tests, 242 browser checks across 3 suites |

## 1. What was asked for

Seven pieces of copy to delete, one to reword, American spelling in the map, a description for the second pooled cohort, and this:

> I want the key/legend of the mapping to be much more clear, especially when 2 different sets of vectors are pooled. The UI has to be clean and easy to follow.

The first six are edits.
The last two are the design work, and they turned out to be the same problem seen from the two views: **a comparison runs two pooled queries, and the interface consistently described only one of them.**

## 2. The copy that went, and why each one earned its deletion

Every removal below leaves the fact it carried recorded somewhere that outlives microcopy - the docs, or the code comment beside the thing it describes.
Deleting a sentence from a rail is not deleting a decision.

| removed | where it was | why it goes |
| --- | --- | --- |
| "Not a difference vector." | cohort compare hint | Defines the feature by what it is not, to a reader who never suspected it was. The claim is load-bearing and stays in `CLAUDE.md` and `docs/cohort_retrieval.md` §4. |
| "Nothing is dropped for you." | cohort member list | Reassurance against a fear the interface never raised. "Untick a sample to leave it out of the pool" already says the pool is yours. |
| "Adds roughly two seconds per hit. Off, a search is local and instant, and abstracts are fetched for a hit when you open it or when the AI hypothesis needs them." | metadata enrichment | Three lines of rail explaining a control that sits inside a disclosure already labelled **Optional** and defaults to off. |
| "Colours all 942,563 points." | color-by coverage | Restates a full bar in words. The readout exists to answer "why is most of my map not colored?", which a whole-map field never raises - so it now says nothing and the bar carries it. |
| "Both corpora folded onto one anatomical vocabulary, so a liver in GEO and a NASA liver share a colour." | Tissue color-by hint | A paragraph of method under a control that had already answered the question; the menu says "whole map" and the legend names every bucket. |
| "One glyph per sample; zoom re-samples the visible window." | ARCHS4 point budget | Describes the mechanism of a control whose four pills already read 100k / 250k / 500k / All. |
| "Each glyph is a pooled member; the query itself is a mean of them and has no position here." | map retrieval key | Replaced by a key row that *says* `pooled member` beside the glyph it names. Showing beats telling. |
| "Hits are ranked by cosine distance in 512 dimensions. This map is a projection into two or three of them and does not preserve those distances, so how far a hit sits from the query here is not its rank, and no line is drawn between them." | map retrieval caveat | Four clauses to protect against one misreading. |

The caveat's last sentence was kept and reworded, because it had to stand without the paragraph that used to precede it:

> **Hover a hit for its rank in the search and its rank on the map. The two disagree: a projection cannot preserve 512-dimensional distances.**

That is the whole honest claim in two sentences: there are two orderings, they differ, and here is where to see both.
It said "a projection of 512 dimensions into two" first, which §5 records as one of the three things this change got wrong.

`tests/test_app.py::test_the_removed_copy_stays_removed` pins every one of them.
Prose regresses silently - nothing else in the suite would notice a paragraph coming back, and several of these had already outlived the thing they described.

### Spelling

The app now spells `color` throughout, in strings, comments and identifiers, and `tests/test_app.py::test_the_app_spells_color_the_american_way` keeps it that way.
This was drift rather than a choice: the package is `colorby.py`, the control is `#color-by`, the functions are `color_for_index` and `covers_corpus`, and the one private function spelled the other way, `_colour_plan`, is now `_color_plan`.
Only the map was asked for, but the retrieval view had one user-visible instance left - `", colour = which cohort retrieved it"` in the comparison legend strip - which would have put both spellings on screen for the same two-cohort search. Everything else there was comments and one local variable named `colour` sitting inside `{"color": colour}`.

## 3. The map key

### The problem, stated exactly

With a two-cohort comparison drawn, the map carries four encodings at once:

| channel | what it distinguishes | where it was explained |
| --- | --- | --- |
| corpus glyph **hue** | tissue, species, flight arm, … | the floating legend |
| member fill **hue** | which cohort a pooled sample belongs to | a swatch on the rail, 800 px from the glyph |
| hit ring **shape** | which cohort retrieved a hit | **nowhere** |
| corpus glyph **shape** | ARCHS4 circle versus OSDR diamond | **nowhere** |

The audit that produced that table is in §7.
The third row is the worst of them: in a comparison the hits outnumber the members and are what the whole feature is about, and their channel had no key at all.
A viewer saw two sizes of white ring and two colors of star with nothing on screen tying them together.

### What was built

**One panel on the plot, three sections, ordered by how transient each is.**

```
┌──────────────────────────────┐
│ POOLED MEMBERS               │   only when two cohorts are drawn
│  ★  Liver · Space Flight   8 │   teal star
│  ★  Liver · Ground Control 6 │   gold star
│ RETRIEVED HITS               │
│  ○  Liver · Space Flight  10 │   white circle ring
│  □  Liver · Ground Control 10│   white square ring
│  ▣  retrieved by both      3 │   ring inside a square
├──────────────────────────────┤
│ COLOR · TISSUE               │   the only section that scrolls
│  [ filter categories…      ] │
│  ■  Blood / immune   155,761 │
│  …                           │
├──────────────────────────────┤
│  ●  ARCHS4                   │   shape, not hue
│  ◆  OSDR                     │
└──────────────────────────────┘
```

The retrieval is on top because it is what the reader asked for a moment ago and what will be gone next search.
The color list is the standing state of the map and is the only part that can grow, so it is the only part that scrolls.
The corpus shapes are a footnote that never changes.

**A comparison is grouped by role, not by cohort, and that is the whole design.**
The encoding has two factors - which cohort, and member versus hit - and they do not use the same channel.
Listing each cohort's two marks together buries that.
Listing the two member rows adjacent and the two hit rows adjacent shows it: one pair differs only in hue, the next differs only in shape, and the reader sees each channel vary with the other held fixed.
No sentence has to assert the rule, which is what makes it readable at a glance rather than after a paragraph.

**A single query does not pay for the comparison.**
A plain search gets two rows, no headings and no cohort names: with one query on screen there is nothing for a name or a group to distinguish it from.
An uploaded sample gets one row, because it was never embedded into this map, has no coordinate, and draws no member mark - a row for it would key a glyph that is not there.

**Marks are drawn as marks.**
Naming a square ring in words, in a panel that could simply draw one, is the key doing less than it could.
Shapes are CSS; **hues come from `manifold/theme.py` through an inline style**, which removes the last place a cohort color was written twice.
The rail's old swatches mirrored `RETRIEVAL_QUERY` and `RETRIEVAL_QUERY_B` into `map.css` by hand, because Plotly cannot read a CSS variable - two spellings of one hex, and a drifted swatch would have labelled the wrong cohort.
`test_the_map_key_reads_its_hues_from_the_theme` asserts both that the key carries the theme's values and that neither hex appears in `map.css` again.

**The key follows the plot into 3-D.**
`Scatter3d` rejects `star` outright, so a member draws as a diamond there and the key draws a diamond too.
A key that kept showing a star would assert a mark 3-D does not draw, which is worse than the silence it replaced.
The hit shapes are deliberately unchanged between 2-D and 3-D; that is why shape, not hue, carries cohort identity in the first place.

**An unticked arm keeps both of its rows**, receded, with `hidden` where the count would be.
Dropping them would leave a gold glyph on screen a moment later with nothing to look it up in, and would make the key disagree with the ticks that produced it.
The `retrieved by both` row leaves, because with one arm hidden the doubled mark cannot exist.

### What the rail gave up

The rail's two-cohort key - a swatch per cohort and a paragraph on what a ring inside a square means - is gone.
The ticks directly above it already carry each cohort's name, so the rail was naming them twice and explaining glyphs nowhere near them.
What is left is one line of the control's own feedback:

- *"**Liver · Space Flight** and its 5 nearest ARCHS4 neighbors, drawn where they sit in the space."*
- *"Both arms drawn, **10** hits, **2** of them retrieved by both."*
- *"**Liver · Space Flight** only. Tick both to see which samples they share."*

This supersedes the placement recorded in `docs/cohort_retrieval.md` §9 ("What the user controls"), and the reason it supersedes it is the reason that section gave for putting it there: put the fact where the misreading happens.
The misreading happens at the glyph.

### What was deliberately not done

**The plot badges stay.** Two of the three designs proposed folding them into the key. They answer a different question - *what is drawn right now*, which changes on every zoom - where the key answers *what does this mark mean*. A key that changed on zoom would be a worse key.

**The halo gets no row.** It is always concentric with a member mark, carries no value independent of it, and its alpha varies with cohort size, so a row would have to explain a quantity that means nothing alone.

**Residual swatches are not dimmed to match their glyphs.** "Other" is drawn at 0.26 opacity and 82% size on the plot while its legend swatch is full strength. Matching them was considered and dropped: the recession is a deliberate ranking of informative categories over uninformative ones, and a dimmed swatch reads as a disabled row.

**The context cloud still gets no legend row**, per invariant 5. One design proposed a subtitle for it inside the key; the coverage readout and the plot badge already carry that state, and both say the points are scenery rather than a category.

## 4. Two pooled queries, two descriptions

Arming a comparison runs a **second** independent pooled query.
The rail described the size and stability of the first and said nothing at all about the second, while both the network figure and the map gave it a color and drew it.

That matters for a specific reason rather than for symmetry: cohort B can easily be the smaller and shakier arm, and `STABILITY_BY_K` is a function of size.
An overlap of 0.25 between a 12-animal cohort at 0.81 and a 2-animal cohort at 0.34 means something different from the same 0.25 between two cohorts of twelve.
The number that decides how much of the overlap to believe was not on screen.

**Each pooled query now gets a card under its own picker**: the selected cohort under the Cohort dropdown, the sibling under "Compare against".
That is the rail's standing rule - the fact that qualifies a control sits directly under that control - and it keeps the member ticks, which belong to cohort A alone, next to cohort A's card.

Each card gains a role line only when there are two: `● COHORT A` and `● COHORT B · differs by spaceflight arm`, with a left rule in the same hue.
A lone cohort gets no letter, because with no second arm on screen there is nothing to distinguish it from.
The contrast facet is stated once, on the second card, because it is a property of the pair.

The dot and the rule are `--accent-teal` and `--accent-warm`, which **are** `GRAPH_THEME["cohort_a"]` and `["cohort_b"]`, so a card and the star it describes in the network figure cannot disagree.

### The one thing that cannot be made consistent, and what carries it instead

Cohort B is `#d9791b` in the retrieval network and `#ffc233` on the map, and this is not fixable by choosing better.
`#ffc233` on white measures about 1.8:1, unusable in the retrieval view; `#d9791b` sits 0.3 dE from `CATEGORICAL[3]` under deuteranopia, unusable on the map's navy canvas next to eleven tissue hues.
`manifold/theme.py` records both measurements.

So **the binding across the two views is the cohort's name, not its hue**, and both surfaces now print the name next to their own mark: the card, the network's band labels, the map's tick, and the map's key rows.
Cohort A survives the trip unchanged at `#0bab9f`, which is what made the difference look like a different arm rather than a different surface.

## 5. Three defects this change introduced, and what caught them

An adversarial review of the finished branch raised 27 candidates, of which these three survived a refutation pass. None was visible to any of the 307 tests or the 242 browser checks, because all three were *statements* rather than crashes.

**The single-query key ignored the show/hide tick.** `retrieval_key_children`'s comparison branch reads `roles` and recedes a hidden arm; the single-query branch never read it. So unticking "Show it on the map" for a plain search took the star and the rings off the plot while the key went on reading "the query sample 1 / retrieved hit 5" - the exact failure the count rule exists to prevent, in the commonest state the map has. The rail sentence beside it had the same gap, under a docstring this change had just rewritten to claim it read the ticks.

**A hit retrieved by both cohorts was counted twice.** `hit_points` is a concatenation across the arms, so the rail said "10 hits, 2 of them retrieved by both" for the same comparison whose banner on the other view said "share 2 of 8 retrieved samples". Two surfaces, one search, two numbers, and a subset relation that holds under neither reading. Both now count distinct samples, and the badge says "samples" rather than "hits" so the unit is unambiguous.

**The reworded caveat said "two" while you could be looking at three.** "a projection of 512 dimensions into two" is a static string inside a group that stays on screen in 3-D. The copy it replaced read "into two or three of them", correct in both. It is now "a projection cannot preserve 512-dimensional distances", which is true of any projection and needs no dimensionality to check itself against - the alternative being to make one hint dims-aware for one word.

Each is now pinned by a test, including a browser check that unticks a single query and asserts the plot, the key and the rail all agree.

## 6. Two defects found on the way

Neither was in scope. Both were found by auditing what the map draws against what anything on screen explains, and both make a claim the code already made in prose actually true.

**3-D silently deleted the OSDR overlay's visual identity.**
`_scatter`'s `Scatter3d` branch passed no `symbol` and hard-coded `line=dict(width=0)`, so `OSDR_SYMBOL="diamond"` and `OSDR_OUTLINE="#ffffff"` were both discarded.
In 2-D the overlay is a white-ringed diamond at size 8.5; in 3-D it arrived as a plain circle at 4.25 with no ring, in the same palette hue as the 940,455 glyphs beneath it.
Only hover could tell 2,108 spaceflight samples from the corpus they sit in, which is the one thing `theme.py` says this map may not do, and `render.py`'s own module docstring asserted the opposite.
`test_osdr_markers_are_visually_distinct_from_the_cloud` never caught it because it only ever ran `("pca", "2d")`.

Both channels survive the trip: `diamond` is in `Scatter3d`'s eight-symbol vocabulary and gl-scatter3d honours `marker.line` as a per-point border.
This was also a **prerequisite** for the corpus key: a key asserting a diamond that 3-D does not draw is worse than the silence it replaced.

**A hit retrieved by both cohorts named only one of them on hover.**
Two traces sit at the identical coordinate and Plotly resolves exactly one tooltip per position, so building each cohort's rows from its own hit list alone made one arm's rank and cosine unreachable - for the very points a comparison exists to show.
`render.py` justified dropping the rank numerals in a comparison on the grounds that "the hover says strictly more (a shared hit names both cohorts, both ranks and both cosines)".
No code produced that tooltip.

A pre-pass now indexes every drawn point by the cohorts that retrieved it, and each arm carries its own map rank, because a shared hit is one coordinate with two nearest members and therefore two valid answers.
**The marks stay emergent.** A hit is still drawn twice because it is in two hit lists; nothing computes an intersection in order to draw. Only the hover reads across the arms.

## 7. Every mark the map can draw

The audit this work was built on. Each row is what the mark is, its 2-D and 3-D symbols, and where a viewer can decode it **now**.

| mark | 2-D | 3-D | keyed by |
| --- | --- | --- | --- |
| ARCHS4 corpus glyph | circle 3.4 | circle 1.7 | color legend + corpus key |
| residual (Other / Unknown) | circle 2.8 @ 0.26 | same | color legend + corpus key |
| ARCHS4 context cloud | circle 2.6 @ 0.35 | same | coverage readout + plot badge (no legend row, invariant 5) |
| OSDR overlay | diamond 8.5, white ring | **diamond 4.25, white ring** (was a plain circle) | corpus key |
| query halo | circle-open 46 / 32 | half size | nothing, deliberately (§3) |
| pooled member, cohort A | star, teal | diamond, teal | retrieval key |
| pooled member, cohort B | star, gold | diamond, gold | retrieval key |
| hit ring, cohort A | circle-open 20, white | circle-open 10 | retrieval key |
| hit ring, cohort B | square-open 27, white | square-open 13.5 | retrieval key |
| retrieved by both | both of the above | both | retrieval key |
| rank numerals | text above the ring | same | hover |

Two things in that table are known and unchanged.
`OSDR_HIGHLIGHT` is currently unreachable - every `ColorBy` has OSDR in its scope and `covers()` only ever strips ARCHS4 - and is kept as the defensive branch for a future ARCHS4-only field.
`RETRIEVAL_MAX_NUMERALS` is 25 while the top-k slider goes to 30, so at k > 25 the last rings carry no numeral; the ring and the hover both still work, and numbering past 25 was measured to be unreadable overlapping text.

## 8. Testing

**`tests/test_app.py`, 10 new tests, and `tests/test_cohorts.py`, 4 more.**
The key's structure for each of the five states (plain search, pooled cohort, upload, comparison, one arm hidden), the 3-D symbol substitution, the corpus key following the Layers ticks, that every glyph shape has a stylesheet rule, that the key reads its hues from `theme`, and the two copy guards.

The glyph-rule test exists because `test_every_classname_used_in_python_exists_in_some_stylesheet` cannot catch these: the class is built by string interpolation from the shape name, so it never appears in a `className` literal, and a shape with no rule renders as an empty 14 px box - a key row pointing at nothing.

**`tests/e2e_check.py`** gains the corpus key, its response to the Layers ticks, and the silent whole-map coverage readout.

**`tests/e2e_cohort_check.py`** gains both cohort cards with their role labels and the contrast facet, the role-grouped key with its shapes and its two hues, the hidden-arm rows, the 3-D diamond substitution, the OSDR overlay keeping its symbol and ring in 3-D, and a shared hit's hover naming both arms.

**One trap, recorded.**
A browser check that navigates with `page.goto("/map")` loses `hits-store` and sees an empty map. Use the in-app link; it is also the real user path.

**One bug the tests did not catch, and now do.**
The divider between the retrieval key and the color list was written as `.bm-key:not(:last-child)`, which can never fire - each key is the only child of its own slot div, so it is always a last child. It was rewritten as a modifier class, and then landed on only one of the two return statements, because the comparison branch returns from a different indent level than the single-query branch. Both key tests now assert the modifier is present. Neither failure was visible to any assertion; both were visible in a screenshot.
