"""Real-corpus comparison checks and reproducible PR screenshots.

Run against an already running app, optionally with BRIDGE_RNA_STUDY_CONTEXT_DB.
    .venv/bin/python tests/e2e_comparisons.py --url http://127.0.0.1:8066
"""
import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright, expect


def check(url, out):
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1680, "height": 1150})
        errors, runs, results = [], [], []
        page.on("pageerror", lambda e: errors.append(str(e)))

        def response(response):
            if response.request.method != "POST":
                return
            data = response.request.post_data_json or {}
            if "comparison-search-button.n_clicks" in data.get("changedPropIds", []):
                runs.append(data)
                result = response.json().get("response", {}).get("hits-store", {}).get("data")
                if result:
                    results.append(result)

        page.on("response", response)
        page.goto(url)
        expect(page.locator("#sample-preview")).to_contain_text("Mmus")
        expect(page.locator("#search-status")).to_be_empty()
        assert not runs
        page.locator("#mode-tab-comparison").click()
        expect(page.locator("#comparison-search-button")).to_be_visible()
        expect(page.locator("#study-group")).not_to_be_visible()
        expect(page.locator("#metadata-options")).not_to_be_visible()
        expect(page.locator("#reviewed-comparison-notes")).to_contain_text("5 flight and 3 ground")
        page.locator("#reviewed-comparison-notes summary").click()
        expect(page.locator("#reviewed-comparison-notes")).to_contain_text("Mmus_BAL-TAL_LRTN_GC_Rep3_G9")
        page.screenshot(path=str(out / "comparison-definition.png"))
        page.locator("#reviewed-comparison-notes summary").click()
        page.locator("#comparison-search-button").click()
        expect(page.locator("#comparison-evidence-body")).to_contain_text("GSM6661548", timeout=60000)
        payload = results[-1]
        rows = payload["paired_evidence"]["rows"]
        first = next(r for r in rows if r["gsm"] == "GSM6661548")
        assert (first["rank_a"], first["rank_b"]) == (1, 1)
        assert len(payload["member_ids"]) == 5
        assert len(payload["comparison"]["member_ids"]) == 3
        assert len(payload["neighborhood"]["hits"]) == 250
        assert len(payload["comparison"]["neighborhood_b"]["hits"]) == 250
        assert len({r["gsm"] for r in rows}) == len(rows)
        expect(page.locator("#network-panel")).not_to_be_visible()
        assert page.locator(".comparison-table tbody tr").count() == 30
        page.screenshot(path=str(out / "comparison-evidence-desktop.png"))

        # Native disclosures retain source text, distinct from derived categories.
        method = page.locator("#comparison-evidence-body > details").first
        method.locator("summary").first.focus()
        page.keyboard.press("Enter")
        expect(method).to_contain_text("not necessarily independent")
        method.locator("summary").first.click()
        sample = page.locator(".comparison-table tbody tr").first.locator("details")
        sample.locator("summary").click()
        expect(sample).to_contain_text("Source characteristics")
        expect(sample).to_contain_text("Derived tissue category")
        expect(sample.locator("a").first).to_have_attribute("href", "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE216208")
        sample.locator("summary").click()

        for depth in (100, 250, 30):
            page.locator("#comparison-depth").get_by_text(str(depth), exact=True).click()
            expect(page.locator(".comparison-table tbody tr")).to_have_count(
                min(30, sum(min(r["rank_a"], r["rank_b"]) <= depth for r in rows)), timeout=30000)
            expect(page.locator("#comparison-page-label")).to_contain_text(
                f"of {sum(min(r['rank_a'], r['rank_b']) <= depth for r in rows)}", timeout=30000)
        page.locator("#comparison-next").click()
        expect(page.locator("#comparison-page-label")).to_contain_text("31–45 of 45")
        expect(page.locator(".comparison-table tbody tr")).to_have_count(15, timeout=30000)
        page.locator("#comparison-previous").click()
        expect(page.locator(".comparison-table tbody tr")).to_have_count(30, timeout=30000)
        assert len(runs) == 1, "changing display depth or page must not rerun retrieval"
        page.locator("#results-view").get_by_text("Network", exact=True).click()
        expect(page.locator("#network-panel")).to_be_visible()
        page.screenshot(path=str(out / "comparison-network.png"))
        page.locator("#results-view").get_by_text("Paired evidence", exact=True).click()
        expect(page.locator("#network-panel")).not_to_be_visible()
        expect(page.locator("#comparison-evidence-panel")).to_be_visible()
        for width in (768, 393):
            page.set_viewport_size({"width": width, "height": 900})
            page.wait_for_function("document.documentElement.scrollWidth <= innerWidth", timeout=15000)
            # The table is deliberately horizontally scrollable, without page overflow.
            region = page.locator(".comparison-table-wrap").first
            region.evaluate("e => e.scrollLeft = e.scrollWidth")
            assert region.evaluate("e => e.scrollLeft >= 0")
            if width == 393:
                region.evaluate("e => e.scrollLeft = 0")
                page.locator("#comparison-evidence-panel").scroll_into_view_if_needed()
                page.screenshot(path=str(out / "comparison-evidence-mobile.png"))
        page.set_viewport_size({"width": 1680, "height": 1150})

        # Both additional fixed manifests work without changing the study picker.
        for label, key in (("OSD-100 · left eye", "osd-100-eye"),
                           ("OSD-104 · Soleus-both sides", "osd-104-soleus")):
            dropdown = page.locator("#reviewed-comparison")
            dropdown.click()
            page.get_by_role("option", name=label, exact=True).click()
            page.locator("#comparison-search-button").click()
            expect(page.locator("#comparison-evidence-body h2")).to_contain_text(label, timeout=60000)
            assert results[-1]["reviewed_comparison"] == key
            assert len(results[-1]["member_ids"]) == 6

        page.locator("#see-on-map").click()
        expect(page.locator("#frame-retrieval")).to_be_visible(timeout=60000)
        page.wait_for_function("""() => {
          const g = document.querySelector('#manifold-graph .js-plotly-plot');
          return g && g._fullData && g._fullData.reduce((n,t)=>n+(t.x?.length||0),0)>900000;
        }""", timeout=90000)
        print("PASS: fixed manifests, exact paired evidence, source disclosures, depth, responsive layout and map handoff")

        # An ordinary sample retrieval clears the old table and restores the network.
        page.get_by_role("link", name="Retrieve", exact=True).click()
        expect(page.locator("#sample-preview")).to_contain_text("Mmus")
        page.locator("#search-button").click()
        expect(page.locator("#search-status")).to_contain_text("Retrieved", timeout=60000)
        expect(page.locator("#comparison-results-switch")).not_to_be_visible()
        expect(page.locator("#comparison-evidence-body")).to_be_empty()
        expect(page.locator("#network-panel")).to_be_visible()
        assert not errors, errors
        browser.close()
        print("PASS: sample retrieval clears comparison evidence; no browser errors")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8066")
    ap.add_argument("--out", type=Path, default=Path("docs/images/comparisons"))
    args = ap.parse_args()
    check(args.url, args.out)
