"""The "i" explanations quote the code that runs, and stay reachable.

`bridge_rna/help.py` shows real lines of source beside each explanation. These
tests are what keep "real" true: an excerpt is a sequence of blocks separated
by `...`, each block must appear in its cited file as consecutive lines
(indentation aside), and the blocks must appear in the order quoted. Checking
each line on its own was rejected because it lets an excerpt stitch unrelated
lines into code that never existed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bridge_rna import help as info_help

ROOT = Path(__file__).resolve().parents[1]


def _blocks(lines: tuple[str, ...]) -> list[list[str]]:
    blocks: list[list[str]] = [[]]
    for line in lines:
        if line.strip() == "...":
            blocks.append([])
        else:
            blocks[-1].append(line.strip())
    return [b for b in blocks if b]


def _find_block(source: list[str], block: list[str], start: int) -> int:
    """Index just past the first occurrence of `block` at or after `start`."""
    n = len(block)
    for i in range(start, len(source) - n + 1):
        if source[i:i + n] == block:
            return i + n
    return -1


EXCERPTS = [(key, excerpt) for key, entry in info_help.ENTRIES.items()
            for excerpt in entry.code]


@pytest.mark.parametrize("key,excerpt", EXCERPTS,
                         ids=[f"{k}:{e.path}" for k, e in EXCERPTS])
def test_every_excerpt_is_contiguous_ordered_source(key, excerpt):
    path = ROOT / excerpt.path
    assert path.is_file(), f"{key} cites a missing file: {excerpt.path}"
    source = [line.strip() for line in path.read_text().splitlines()]
    position = 0
    for block in _blocks(excerpt.lines):
        found = _find_block(source, block, position)
        assert found >= 0, (f"{key}: these lines are not consecutive in "
                            f"{excerpt.path} after line {position}:\n"
                            + "\n".join(block))
        position = found


def test_excerpts_are_short_and_well_formed():
    for key, entry in info_help.ENTRIES.items():
        assert entry.title and entry.body, key
        for excerpt in entry.code:
            assert 1 <= len(excerpt.lines) <= 8, key
            assert excerpt.lines[0].strip() != "..." and excerpt.lines[-1].strip() != "...", key


def test_explanations_are_plain_and_short():
    for key, entry in info_help.ENTRIES.items():
        words = sum(len(p.split()) for p in entry.body)
        assert words <= 110, f"{key} runs to {words} words"
        text = " ".join(entry.body) + entry.title
        assert "—" not in text, f"{key} uses an em dash"


def test_the_button_names_its_feature_and_carries_only_a_key():
    button = info_help.info("cohort")
    assert button.type == "button"
    props = button.to_plotly_json()["props"]
    assert props["aria-label"] == "About Cohort"
    assert props["aria-haspopup"] == "dialog"
    assert props["data-info"] == "cohort"
    assert "id" not in props, "a key may appear twice on a page; ids would collide"


def test_the_library_holds_every_entry_once_and_starts_hidden():
    library = info_help.library()
    assert library.id == "info-library" and library.hidden is True
    keys = [c.to_plotly_json()["props"]["data-info-entry"] for c in library.children]
    assert keys == list(info_help.ENTRIES)


def _walk(node):
    yield node
    children = getattr(node, "children", None)
    if children is None or isinstance(children, str):
        return
    for child in children if isinstance(children, (list, tuple)) else [children]:
        yield from _walk(child)


def _info_keys(tree) -> list[str]:
    keys = []
    for node in _walk(tree):
        props = getattr(node, "to_plotly_json", None)
        if props is None:
            continue
        key = getattr(node, "data-info", None)
        if key is None:
            key = node.to_plotly_json()["props"].get("data-info")
        if key:
            keys.append(key)
    return keys


def test_every_entry_is_placed_somewhere_and_every_placement_is_known():
    """An entry nobody can open is dead text, and a button naming a missing
    key would throw at render. Both views are walked as built."""
    from bridge_rna import layout as rna_layout
    from bridge_rna import results_view
    from manifold import layout as map_layout

    placed = set(_info_keys(rna_layout.build_view()))
    placed |= set(_info_keys(results_view.build_results_panel()))
    placed |= set(_info_keys(map_layout.build_view()))
    # The inspector is drawn by a callback from the selected row, so its keys
    # are read from the one module that builds it rather than from a layout.
    import re
    placed |= set(re.findall(r'info="([a-z-]+)"',
                             (ROOT / "bridge_rna" / "panels.py").read_text()))
    assert placed <= set(info_help.ENTRIES)
    assert set(info_help.ENTRIES) <= placed, sorted(set(info_help.ENTRIES) - placed)


def test_an_info_button_is_never_inside_a_label():
    """A button inside a <label> or <summary> is activated by clicking the
    label text, and its name is folded into the control's name."""
    from dash import html

    from bridge_rna import layout as rna_layout
    from manifold import layout as map_layout

    for tree in (rna_layout.build_view(), map_layout.build_view()):
        for node in _walk(tree):
            if isinstance(node, (html.Label, html.Summary)):
                assert not _info_keys(node.children if not isinstance(node.children, str) else []), node
