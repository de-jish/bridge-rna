#!/usr/bin/env python3
"""Check disclosure access, source links, and stale AI response handling.

Run against an already running real-corpus app:
    .venv/bin/python tests/e2e_cleanup_check.py --url http://127.0.0.1:8050

Only AI generation responses are controlled, to reproduce a delayed response
without depending on a model provider. Retrieval and metadata use the real app.
"""
import argparse
import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright, expect


async def check(url: str, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="chrome")
        page = await browser.new_page(viewport={"width": 1680, "height": 1150})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        ai_started, release_ai = asyncio.Event(), asyncio.Event()
        cleared = asyncio.Event()

        async def handle_request(route):
            payload = route.request.post_data_json
            if ("ai-summary-output" in payload.get("output", "")
                    and "ai-summary-button.n_clicks" in payload.get("changedPropIds", [])):
                ai_started.set()
                await release_ai.wait()
                await route.fulfill(content_type="application/json", body=json.dumps({
                    "multi": True,
                    "response": {"ai-summary-output": {"children": "Old query interpretation"},
                                 "ai-summary-status": {"children": ""}},
                }))
            else:
                await route.continue_()

        def response_received(response):
            if response.request.method != "POST":
                return
            payload = response.request.post_data_json or {}
            if ("ai-summary-output" in payload.get("output", "")
                    and "hits-store.data" in payload.get("changedPropIds", [])):
                cleared.set()

        await page.route("**/_dash-update-component", handle_request)
        page.on("response", response_received)
        await page.goto(url)
        await expect(page.locator("#sample-preview")).to_contain_text("Mmus")
        await expect(page.locator("#search-status")).to_be_empty()
        ai = page.locator(".ai-panel")
        assert await ai.get_attribute("open") is None
        await expect(page.locator("#ai-summary-button")).not_to_be_visible()
        await ai.locator("summary").focus()
        await page.keyboard.press("Enter")
        await expect(page.locator("#ai-summary-button")).to_be_visible()
        await ai.locator("summary").click()
        await page.screenshot(path=str(out / "after-retrieve.png"))

        await page.locator("#search-button").click()
        await expect(page.locator("#search-status")).to_contain_text("Retrieved", timeout=60000)
        await expect(page.locator("#details-panel a[title*='OSDR']").first).to_be_visible(timeout=60000)
        await asyncio.wait_for(cleared.wait(), 30)
        await page.screenshot(path=str(out / "after-results.png"))

        # The late response must not restore an interpretation for old results.
        await ai.locator("summary").click()
        await page.locator("#ai-summary-button").click()
        await asyncio.wait_for(ai_started.wait(), 10)
        cleared.clear()
        await page.locator("#search-button").click()
        await asyncio.wait_for(cleared.wait(), 60)
        release_ai.set()
        await page.wait_for_timeout(1500)
        await expect(page.locator("#ai-summary-output")).to_be_empty()
        await expect(page.locator("#ai-summary-status")).to_be_empty()
        print("PASS: keyboard disclosure, source link, late AI response invalidation", flush=True)
        await ai.locator("summary").click()

        async with page.expect_download() as download_info:
            await page.locator("#network-graph").get_by_role(
                "button", name="Download plot as a PNG").click()
        download = await download_info.value
        await download.save_as(str(out / "network-export.png"))
        assert (out / "network-export.png").stat().st_size > 1000
        print("PASS: existing network PNG export", flush=True)
        await expect(page.get_by_text("Snapshot succeeded", exact=False).first).not_to_be_visible(
            timeout=20000)

        await page.locator("#see-on-map").click()
        await expect(page.locator("#frame-retrieval")).to_be_visible(timeout=60000)
        await page.wait_for_function("""() => {
            const gd = document.querySelector('#manifold-graph .js-plotly-plot');
            return gd && gd._fullData && gd._fullData.reduce(
                (n, t) => n + ((t.x && t.x.length) || 0), 0) > 900000;
        }""", timeout=90000)
        details = page.locator(".bm-method-details")
        assert await details.get_attribute("open") is None
        await details.locator("summary").focus()
        await page.keyboard.press("Enter")
        await expect(page.locator("#method-params")).to_contain_text("n_neighbors", timeout=60000)
        await details.locator("summary").click()
        await page.locator("#projection-label").click()
        await page.screenshot(path=str(out / "after-map.png"))
        await page.locator("#frame-retrieval").click()
        await expect(page.locator("#neighborhood-drawer")).to_be_visible()
        await expect(page.locator("#neighborhood-body")).to_contain_text("GEO studies")
        await page.wait_for_function("""() => {
            const gd = document.querySelector('#manifold-graph .js-plotly-plot');
            return gd && (gd._fullData || []).some(
                t => t.name === '512-D evidence neighbor' && t.x.length === 250);
        }""", timeout=90000)
        await page.screenshot(path=str(out / "after-neighborhood.png"))
        for width in (768, 393):
            await page.set_viewport_size({"width": width, "height": 900})
            await page.wait_for_timeout(800)
            assert await page.evaluate(
                "document.documentElement.scrollWidth <= window.innerWidth")
            await page.screenshot(path=str(out / f"after-map-{width}.png"), full_page=True)
        assert not errors, errors
        print("PASS: projection disclosure, neighborhood, tablet/mobile overflow, console", flush=True)
        await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8050")
    parser.add_argument("--out", type=Path, default=Path(".lavish/cleanup"))
    args = parser.parse_args()
    asyncio.run(check(args.url, args.out))
