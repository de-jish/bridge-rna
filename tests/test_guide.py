"""The guided tour: its slides, its screenshots, and where it lives.

The screenshots are real captures (`tests/guide_screenshots.py`); these tests
check that every slide has one, that it is the size the dialog reserves, and
that the deploy allowlist carries exactly those files.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

from bridge_rna import guide

ROOT = Path(__file__).resolve().parents[1]


def _webp_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data[:4] == b"RIFF" and data[8:12] == b"WEBP", f"{path.name} is not WebP"
    chunk = data[12:16]
    if chunk == b"VP8 ":
        w, h = struct.unpack("<HH", data[26:30])
        return w & 0x3FFF, h & 0x3FFF
    if chunk == b"VP8L":
        bits = int.from_bytes(data[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if chunk == b"VP8X":
        return (int.from_bytes(data[24:27], "little") + 1,
                int.from_bytes(data[27:30], "little") + 1)
    raise AssertionError(f"{path.name}: unknown WebP chunk {chunk!r}")


@pytest.mark.parametrize("slide", guide.SLIDES, ids=[s.key for s in guide.SLIDES])
def test_every_slide_has_its_screenshot_at_the_reserved_size(slide):
    path = ROOT / slide.image
    assert path.is_file(), f"run tests/guide_screenshots.py: {slide.image} is missing"
    assert _webp_size(path) == (guide.SHOT_WIDTH, guide.SHOT_HEIGHT)
    assert path.stat().st_size < 400_000, f"{slide.image} is unexpectedly large"


def test_no_orphan_screenshots():
    on_disk = {str(p.relative_to(ROOT)) for p in (ROOT / "assets" / "guide").glob("*")}
    assert on_disk == {s.image for s in guide.SLIDES}


def test_slide_text_is_plain_and_short():
    keys = [s.key for s in guide.SLIDES]
    assert len(keys) == len(set(keys))
    for s in guide.SLIDES:
        assert s.alt and s.title and s.body
        assert len(s.body.split()) <= 50, s.key
        assert "—" not in s.title + s.body + s.alt, s.key


def test_the_chapters_open_on_the_routes_they_describe():
    assert [c.key for c in guide.CHAPTERS] == ["retrieve", "map"]
    import app as shell

    assert [c.route for c in guide.CHAPTERS] == [r["path"] for r in shell.ROUTES]


def test_the_dialog_is_one_native_dialog_with_a_tab_per_chapter():
    from dash import html

    dialog = guide.build_guide()
    assert isinstance(dialog, html.Dialog) and dialog.id == "guide"
    text = str(dialog.to_plotly_json())
    for chapter in guide.CHAPTERS:
        assert f"guide-tab-{chapter.key}" in text
        assert f"guide-chapter-{chapter.key}" in text
    # Screenshots load when their slide is shown, not with every page view.
    assert "'src'" not in text and "data-src" in text
    # Nothing the script owns is set by the server.
    assert "aria-selected" not in text and "data-current" not in text


def test_the_guide_and_the_explanations_live_on_the_shell_outside_both_slots():
    """The header and the view are replaced on every route change. The guide
    and the explanation library must not be inside either, or navigating with
    the guide open would destroy it."""
    import app as shell

    built = shell.build_app()
    with built.server.test_request_context("/map"):
        layout = built.layout()
    ids = [getattr(c, "id", None) for c in layout.children]
    assert "guide" in ids and "info-library" in ids
    assert ids.index("app-header-slot") < ids.index("guide")


def test_the_header_opens_the_guide_on_both_routes():
    import app as shell

    for route in shell.ROUTES:
        header = str(shell.header(route["key"]).to_plotly_json())
        assert "data-guide-open" in header and "How it works" in header


def test_the_deploy_allowlist_carries_exactly_the_screenshots():
    sys.path.insert(0, str(ROOT / "deploy"))
    import common

    images = {s.image for s in guide.SLIDES}
    listed = set(json.loads((ROOT / "deploy" / "runtime-files.json").read_text()))
    assert images <= listed
    assert {n for n in common.STATIC_ASSETS if n.startswith("assets/guide/")} == images
    for name in images:
        assert common.runtime_path(name)
    assert not common.runtime_path("assets/guide/other.webp")


def test_the_scripts_own_what_changes_and_stop_moving_under_reduced_motion():
    css = (ROOT / "assets" / "03-help.css").read_text()
    reduced = css[css.index("@media (prefers-reduced-motion: reduce)"):]
    for selector in (".info-popover", ".guide[open]", ".guide-slide"):
        assert selector in reduced
    js = (ROOT / "assets" / "guide.js").read_text()
    assert "showModal" in js and "localStorage" in js and "try" in js
    pop = (ROOT / "assets" / "info-popover.js").read_text()
    assert "document.body.appendChild" in pop and "innerHTML" not in pop
