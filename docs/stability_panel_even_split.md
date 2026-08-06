# Splitting the stability panel's two cohort sections evenly

**Status: in progress, 2026-08-06.**
A follow-on to `docs/live_stability.md`, which built the panel this document re-lays-out.
Section 7 of that document, "Making two measurements fit on screen at once", is the direct ancestor: it fixed a 644 px panel into a 389 px box by cutting what each block says.
This document is what happened next, which is that the cut was not quite enough.

## 1. The defect, measured

Driving the real app in Chromium against study OSD-100, cohort `left eye · Ground Control` compared against `left eye · Space Flight` (differs by spaceflight arm), top-k 5, cohort A and cohort B do not get the same treatment.

| viewport | `#stability-panel` client height | its content height | cohort B's "Moves it most" row is clipped by |
| --- | ---: | ---: | ---: |
| 1680 x 1050 | 447 px | 456 px | **7.8 px** |
| 1600 x 1000 | 445 px | 456 px | **9.9 px** |
| 1512 x 982 | 444 px | 456 px | **10.7 px** |
| 1440 x 900 | 441 px | 456 px | **14.2 px** |
| 1280 x 800 | 389 px | 456 px | **65.6 px** |

At every viewport tested, cohort A's block is complete and cohort B's is not.
The row that goes is the one naming the animal whose absence moves the result furthest, which is the only actionable line in the block.

The two blocks are also unequal before any clipping: cohort A measures 148.3 px and cohort B measures 160.7 px.
That 12.4 px is worth decomposing, because the obvious culprit is not one of the terms and the real shape of it is the argument for the fix.

| term | px |
| --- | ---: |
| the separator box that exists on the second block only (`padding-top: 12` + `border-top: 1`) | **+13.00** |
| cohort A's scale sentence wrapping to two lines where cohort B's fits on one | **-15.94** |
| cohort B's 28-character member key wrapping where cohort A's 27-character key does not | **+15.28** |
| | **+12.34** |

The `differs by spaceflight arm` phrase costs nothing: at 322 px it sits on cohort B's role line beside the letter and adds no height at all.
What is left is one structural offset and **two content terms that happened to nearly cancel**, each swinging between roughly -16 px and +29 px depending on how a given cohort's sentence and member name wrap.
So the layout was not stably asymmetric, it was *metastable*: the gap between the two arms was a property of the strings in them, and it moved from one search to the next.
That is the case for an arrangement in which the two arms are the same size because of how they are laid out rather than because of what they happen to contain.

The panel is not the only thing starved.
In the same measurements `#details-panel` had 506 px of content and was given between 118 px and 310 px, so the inspector below is scrolling hard while the panel above it holds 447 px to say two numbers.

## 2. Why the shipped guard did not catch it

`tests/e2e_cohort_check.py` already asserts that both arms fit, and it passes:

```python
box = stab.bounding_box() or {}
last = stab.locator(".stability-cohort").last.bounding_box() or {}
c.ok(bool(box) and bool(last)
     and last["y"] + last["height"] <= box["y"] + box["height"] + 1,
     ...)
```

`bounding_box()` returns the **border box**, so the comparison allows the last block to run into the panel's own 20 px bottom padding and stop 1 px short of its border.
At 1600 x 1000 the last block's bottom is 9.9 px below the content box and 10.1 px above the border box, so the assertion is satisfied by the exact margin that hides the failure.
The panel is also a scroll container, and the check never reads `scrollHeight` against `clientHeight`, so 11 px of unreachable content is invisible to it.

That is the same class of error `docs/live_stability.md` section 1 describes for `STABILITY_BY_K`: an assertion that is true about a thing adjacent to the one that matters. The guard is tightened as part of this change.

## 3. What "split evenly" has to mean

The request was "can these 2 sections in the cohort feature be split evenly UI wise", against a screenshot of the panel with cohort A complete and cohort B truncated mid-block.

Any fix has to satisfy three things at once, and the measurements above are why:

- **The two arms get the same amount of the panel.** Not approximately: a comparison exists to be read as a comparison, and one arm rendered complete beside one arm rendered partial tells the reader the first is the finding and the second is a footnote.
- **Neither arm is clipped at any viewport the app is used at**, down to 1280 x 800.
- **The panel stops starving the inspector below it.** 447 px for two numbers, against 506 px of definition, members and estimator prose compressed into 264 px, is the wrong division of a 934 px column.

## 4. What shipped: two even columns with their rows aligned

The two arms sit side by side in equal columns, and their rows line up across the gap.

```
Result stability
Measured on this query: how much of these 5 hits comes back when any
one pooled sample is dropped, averaged over all of them.
The two arms differ by spaceflight arm.

● COHORT A                      ● COHORT B
left eye · Ground Control       left eye · Space Flight
0.89  of 5                      0.94  of 5
━━━━━━━━━━━━━━━━━━━━━━━╌╌       ━━━━━━━━━━━━━━━━━━━━━━━━╌
6 pooled, and one alone         6 pooled, and one alone
overlaps another by 0.32,       overlaps another by 0.60,
a 2.8x gain.                    a 1.6x gain.
- - - - - - - - - - - - -       - - - - - - - - - - - - -
MOVES IT MOST      0.67         MOVES IT MOST      0.67
Mmus_C57-6J_EYE_GC_Rep1_M33     Mmus_C57-6J_EYE_FLT_Rep4_M26
```

Measured on the real app, same query as section 1:

| | before | after |
| --- | ---: | ---: |
| `#stability-panel` content height | 456 px | **354 px** |
| internal overflow at 1600 x 1000 | 11 px | **0 px** |
| internal overflow at 1280 x 800 | 67 px | **0 px** |
| cohort A block | 148.3 px | **155 px wide, equal height** |
| cohort B block | 160.7 px | **155 px wide, equal height** |
| `#details-panel` height at 1600 x 1000 | 264 px | **355 px** |
| `#details-panel` height at 1280 x 800 | 120 px (its floor) | **155 px** |

Five things about it are load-bearing.

**The alignment is `subgrid`, and the rows are assigned by class rather than by child order.**
Each arm is a grid whose rows are the pair grid's own, so the name row, the number row, the member row and the flag row line up across both columns however many lines any of them wraps to.
That is what puts 0.89 on the same baseline as 0.94 instead of 160 px below it, which is the comparison the panel exists to support.
Rows are addressed by class because either of the last two can be missing from either arm, and counting children would let cohort B's flag land in the row holding cohort A's member name.
Where `subgrid` is unsupported the declaration is dropped and each arm falls back to its own four-row grid: the columns stay even and only the cross-column baselines go.

**The tracks are `minmax(0, 1fr)`, never `1fr`.**
`1fr` is `minmax(auto, 1fr)`, which floors a track at its min-content width.
Sample keys are mono and run to 39 characters across the corpus (`Mmus_C57-6J_LVR_RR1_BSL_noERCC_Rep5_M10`; median 26, p95 37, over all 2,108), and any unbreakable run would push its column wider than its twin, silently undoing the even split the rule exists to make.

**"differs by *facet*" moved from cohort B's role line to the panel header.**
It is the one fact in the panel that belongs to neither arm: it describes the pair.
Hanging it under B's letter made B's name start a line below A's, which is precisely the ragged edge that even columns exist to remove.
This is the panel's own existing rule - `docs/live_stability.md` section 7, "what is shared is said once" - applied to the one line that had escaped it.

**The "moves it most" row is always drawn, and says so when there is nobody to name.**
`cohorts.weakest_member` returns `None` when every member's absence moves the list equally far, and that is an answer rather than a gap.
It now reads "Moves it most / every member equally", in the text face rather than the mono one, because it is prose and not an accession.
An absent row and a clipped row look identical on screen, and a clipped row on this exact line is the defect being fixed, so silence was not available as an answer here.

**The low-stability flag is deliberately *not* equalized.**
It appears under the shaky arm only, leaving a gap under the healthy one.
Equalizing it would need either a blank cell, which is the empty promise the missing "moves it most" row just stopped being, or an "above 70%" counterpart badge - and a pass mark for a healthy cohort is exactly the grade `R̄` was deleted for being (`docs/cohort_pooling.md`).
A caution that appears only when there is something to be cautious about is working correctly.

The three-line weakest row is a consequence rather than a choice.
Label, score and a 27-character key shared one baseline while the panel was a single 322 px column; in a 155 px one they cannot, since the label and value are fixed width and left about 29 px for the key, which - because it wraps rather than truncates, deliberately - broke one character per line into a 400 px column.
Label and score now share a line and the key sits beneath them.

### The flex fix underneath it

Making the panel shorter was not sufficient, and the residue is instructive.
With the columns in place the panel still lost 3 px at 1680 x 1050 and 11 px at 1280 x 800, which is enough to cut the descenders off the last line of a sample key.

The cause was `.details-panel { flex: 1 20 auto }`.
With a content basis that panel asks for the ~506 px its definition, member list and estimator prose measure - a height it will never get and does not need, since it scrolls by design - so every layout pass began in overflow and ended by taking that overflow back off both panels in proportion.
`flex-shrink: 20` made the details panel's share large; it never made the stability panel's share zero.

`.details-panel` is now `flex: 1 1 0`: it claims the leftovers instead of claiming its content and giving it back, so there is no overflow to divide.
The degradation path is unchanged and now falls out of the same rule rather than from a tuned constant - a zero-basis item contributes nothing to shrink, so on a column too short for all three the details panel stops at its 120 px floor and the stability panel is the one that scrolls internally.

`.stability-panel`'s `max-height: 65%` stays, demoted to the backstop it was always meant to be.
At 1600 x 1000 the panel asks for 354 px of an allowed 607 px, so it never binds; it still earns its place on a genuinely short window, where two flagged arms and long labels reach about 409 px.

**`flex: 1 1 0` is right only while the column's height is fixed, and getting that wrong regressed every width below 1180 px.**
There the app grid collapses to one column and the document scrolls, so the inspector's height comes from its contents - and an item with a zero basis contributes nothing to that height.
The column therefore sized itself to the other two panels and left the details panel sitting on its 120 px floor, scrolling internally with **372 px hidden at 900 px wide and 388 px at 390 px**, where with a content basis it had stood at its full 491 px and let the page scroll.
Measured both ways against the running app rather than reasoned about.
The `@media (max-width: 1180px)` block now restores `flex: 1 1 auto` and `overflow: visible`, alongside the two rules already there that lift the caps on `.stability-panel` and `.ai-panel` for exactly the same reason: once the document scrolls, a panel should be as tall as what it holds.

### The narrowest phones

The label and the score of the "moves it most" row are 97.8 px and 26.4 px, so with the 8 px gap the pair needs 132.2 px on one line.
A column is half the page less 78 px of padding, border and gutter, so the two stop fitting below about 342 px of page width - reached only by the narrowest phones, and measured at 320 px, where a column is 121 px and cohort A's score sat against cohort B's label.
`.stability-weakest-label` is now `flex: 0 1 auto` with `min-width: 0` rather than `flex: none`, so it wraps to two lines instead, identically in both columns.
Stacking the arms under a breakpoint was the alternative and was rejected: it buys the same fix by making the phone the one place the two arms cannot be compared, and the even split survives 320 px without it.

## 5. Alternatives considered and rejected

Three layouts were specified in full and judged on information design, fidelity to this repository's recorded decisions, and implementation risk.
All three judges ranked the aligned comparison first.

**Whole blocks side by side, without shared row baselines.**
The same two columns, but each arm self-contained, its rows falling where its own content puts them.
Rejected because it gives up the thing the columns were for: with A's note wrapping to three lines and B's to two, the two headline numbers stop sharing a baseline and the reader is back to comparing across a vertical offset, just a smaller one.
Its own specification also proposed holding the columns even by padding each arm to a fixed five rows with inert spacer divs, which reintroduces positional counting - the failure mode that `subgrid` plus class-addressed rows exists to avoid.

**Keep the stack and equalize the two blocks.**
The conservative option: same vertical arrangement, identical row structure in both arms, equal heights.
Rejected because it does not improve the reading the panel exists to support - the two headline numbers stay about 180 px apart with five intervening rows, so comparing them still costs memory rather than a glance - and because it resolves the space problem in the wrong direction, growing the panel to about 511 px and dropping the details panel to about 200 px against a details panel already hiding 244 px of its 506.
It is also not what was asked for.
Its diagnosis of the flex bug was correct and is kept; so is its argument that a vanished row and a clipped row look the same, which is why "every member equally" is now printed.

**Stack the two columns again below 680 px, or below 360 px.**
Rejected on measurement: at a 390 px phone width the two columns are 156 px each, which is *wider* than the 155 px they get on a 1600 px desktop, because the inspector goes full-bleed when the app grid collapses to one column.
The one width where anything did break was 320 px, and letting the row's label wrap fixes it without a breakpoint.
A breakpoint would have made the phone the only place the two arms cannot be compared.

**Give cohort B's meter the warm hue its dot carries.**
Rejected because the meter's fill already encodes something else: it turns amber when the measurement is below `STABILITY_FLOOR`.
Two meanings on one channel would make a healthy cohort B indistinguishable from a flagged cohort A.
Identity stays bound to the dot and the name, which is the rule `docs/map_key.md` settled for the same reason.

## 6. How it is tested

In pytest, against the component tree rather than a screenshot:

- Two arms are wrapped in one `.stability-pair.is-pair`; a lone cohort gets the wrapper without the modifier, because a single block in a two-column grid is a half-empty table.
- Every row of an arm is a direct child carrying its own class, checked on a comparison where arm A has a member to name and arm B has a flag instead - the mirror-image case that positional row assignment would get wrong.
- The weakest row is present on both arms whatever either measured, and says "every member equally" with no score when there is nobody to name.
- The facet is stated once, in the header, and the string appears exactly once in the whole panel.

In the browser, against the real corpus, in `tests/e2e_cohort_check.py`:

- The panel does not scroll internally (`scrollHeight - clientHeight <= 1`) and the last arm's last row sits inside the panel's **content** box, not its padding.
- The two arms start on the same line and are the same height, both within 1 px.
- The two headline numbers share a baseline.
- The two arms are laid out as one even pair rather than stacked.

Across ten payload shapes rendered through the shipping `build_stability_panel` and measured in Chromium at 1700 px and at 390 px - both arms named, one arm with no weakest member, neither arm with one, one flagged, both flagged, the corpus's longest sample key in both columns, long cohort labels at top-k 50, a zero baseline, and a lone cohort with and without a named member - every case has equal column widths, equal block heights, aligned names, aligned numbers, and no horizontal overflow.
