#!/usr/bin/env python3
"""Real-data results checks against an already running local/staging app."""
import argparse
import csv
import io
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def run(url, output):
    output.mkdir(parents=True, exist_ok=True)
    checks = []
    states = {}
    errors = []
    searches = []

    def check(condition, name):
        assert condition, name
        checks.append(name)
        print('OK', name, flush=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={'width':1440, 'height':1000}, accept_downloads=True)
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('console', lambda message: errors.append(message.text) if message.type == 'error' else None)

        responses = []
        def response_seen(response):
            responses.append(response)
        page.on('response', response_seen)

        def drain_responses():
            # Read response bodies outside event handlers: Playwright's sync
            # handlers can re-enter one another while a body is still pending.
            while responses:
                response = responses.pop(0)
                if '_dash-update-component' not in response.url:
                    continue
                if response.status >= 400:
                    errors.append(f'HTTP {response.status}')
                    continue
                if response.status == 204:
                    continue
                changed = response.json().get('response', {})
                states.update(changed)
                if 'hits-store' in changed and changed['hits-store'].get('data'):
                    searches.append(changed['hits-store']['data'])

        def choose(component, label, exact=True):
            page.locator('#'+component).click()
            menu = page.locator('.dash-dropdown-content')
            menu.wait_for()
            menu.get_by_text(label, exact=exact).first.click()

        def count(n, total=30):
            page.wait_for_function(
                'text => document.querySelector("#results-count")?.textContent === text',
                arg=f'{n} of {total} matches shown', timeout=30000)
            drain_responses()

        def download():
            with page.expect_download() as pending:
                page.locator('#results-export').click()
            return list(csv.DictReader(io.StringIO(Path(pending.value.path()).read_text())))

        def graph_matches(rows):
            page.wait_for_function('''expected => {
                const g = document.querySelector('#network-graph .js-plotly-plot');
                const ids = new Set((g?._fullData || []).flatMap(t =>
                    (t.customdata || []).filter(d => d[0] === 'gsm').map(d => d[1])));
                return JSON.stringify([...ids].sort()) === JSON.stringify(expected.sort());
            }''', arg=[r['gsm'] for r in rows], timeout=15000)

        page.goto(url)
        page.locator('#sample-dropdown').wait_for()
        page.wait_for_function('document.querySelector("#sample-dropdown-value")?.textContent.includes("Mmus")')
        page.locator('[role=slider]').first.press('End')
        page.locator('#search-button').click()
        count(30)
        original = states['results-rows-store']['data']
        check(len(original) == 30 and len({r['gsm'] for r in original}) == 30, '30 unique enriched matches')
        raw = searches[-1]['hits']
        check([(r['gsm'],r['score']) for r in original] == [(r['gsm'],r['score']) for r in raw],
              'unfiltered identities, scores and order equal original retrieval')
        check(all(r.get('metadata') for r in original), 'real metadata loaded for every returned identity')
        exported = download()
        check([r['gsm'] for r in exported] == [r['gsm'] for r in original], 'unfiltered CSV order equals ranked list')
        check([float(r['score']) for r in exported] == [r['score'] for r in original], 'CSV scores retain full precision')
        page.locator('#results-controls > details > summary').click()
        eye = [r for r in original if r['metadata']['organ'] == 'Eye']
        check(0 < len(eye) < 30, 'real query includes known and unknown organ annotations')
        choose('results-organ', 'Eye')
        count(len(eye))
        graph_matches(eye)
        check(states['results-rows-store']['data'] == eye, 'organ filter preserves original ranks and scores')
        exported = download()
        check([r['gsm'] for r in exported] == [r['gsm'] for r in eye], 'filtered CSV equals visible matches')
        all_studies = {s['gse'] for r in original for s in r['metadata']['studies']}
        eye_studies = {s['gse'] for r in eye for s in r['metadata']['studies']}
        incompatible = sorted(all_studies-eye_studies)[0]
        choose('results-study', incompatible)
        count(0)
        page.wait_for_function("document.querySelector('#network-graph')?.textContent.includes('No matches meet')")
        check(page.locator('#results-export').is_disabled(), 'empty state disables CSV and explains reset')
        page.locator('#results-reset').click()
        count(30)
        graph_matches(original)
        check(states['results-rows-store']['data'] == original, 'reset restores identical ordered rows')
        choose('results-organ', 'Unknown')
        unknown = [r for r in original if r['metadata']['organ'] == 'Unknown']
        count(len(unknown))
        check(states['results-rows-store']['data'] == unknown, 'Unknown is an explicit selectable value')
        page.locator('#results-reset').click()
        count(30)
        treatment = next(r['metadata']['treatment'] for r in original if r['metadata']['treatment'] != 'Unknown')
        choose('results-treatment', treatment)
        treated = [r for r in original if r['metadata']['treatment'] == treatment]
        count(len(treated))
        check([r['gsm'] for r in download()] == [r['gsm'] for r in treated], 'treatment filter and CSV agree')
        label = states['results-sample']['options'][0]['label']
        choose('results-sample', label)
        page.wait_for_function("document.querySelector('#details-panel')?.textContent.includes('Source annotations')")
        detail = page.locator('#details-panel')
        check('Derived tissue' in detail.inner_text(), 'derived tissue remains separate from organ')
        summary = detail.get_by_text('Metadata sources and study context', exact=True)
        summary.focus(); summary.press('Enter')
        page.wait_for_function("document.querySelector('.metadata-details details')?.open")
        check('2773286356289' in detail.inner_text(), 'keyboard disclosure exposes pinned provenance')
        check('Gemma' in detail.inner_text(), 'treatment source is attributed')
        check(len(searches) == 1, 'filters, selection, reset and CSV never rerun retrieval')
        page.locator('#results-controls > details > summary').click()
        for width in (1440, 390, 320):
            page.set_viewport_size({'width':width,'height':1000})
            page.wait_for_timeout(400)
            check(page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1'),
                  f'{width}px layout has no horizontal overflow')
            check(page.evaluate('''() => {
                const r = s => document.querySelector(s).getBoundingClientRect();
                return r('.graph-wrap').bottom <= r('.panel--canvas').bottom
                    && r('.panel--profile').top >= r('.graph-wrap').bottom;
            }'''), f'{width}px network and profile do not overlap')
            page.screenshot(path=str(output/f'metadata-{width}.png'), full_page=True)
        page.set_viewport_size({'width':1440,'height':1000})
        page.locator('#search-button').click()
        count(30)
        check(states['results-treatment']['value'] is None, 'new search clears prior filters')
        page.locator('#see-on-map').click()
        page.locator('#manifold-graph').wait_for()
        page.wait_for_function("document.querySelector('#manifold-graph .js-plotly-plot')?._fullData?.length > 0",timeout=60000)
        check('/map' in page.url, 'full retrieval remains available on the map')
        page.wait_for_load_state('networkidle')
        drain_responses()
        page.goto(url)
        page.locator('#mode-tab-cohort').click()
        page.locator('#mode-panel-cohort').wait_for(state='visible')
        choose('study-dropdown', 'OSD-137')
        page.wait_for_timeout(500)
        choose('cohort-compare-dropdown', 'differs by', exact=False)
        page.locator('.cohort-card.is-b').wait_for()
        page.locator('#cohort-search-button').click()
        page.wait_for_function("document.querySelector('#results-order-note')?.textContent.includes('A rank order')")
        drain_responses()
        combined = states['results-rows-store']['data']
        total = len(combined)
        count(total, total)
        check(len({r['gsm'] for r in combined}) == total, 'comparison has one row per sample')
        exported = download()
        check([r['gsm'] for r in exported] == [r['gsm'] for r in combined], 'comparison CSV uses displayed union order')
        check(all((not r['score'] or float(r['score']) == original['score']) and
                  (not r['score_b'] or float(r['score_b']) == original['score_b'])
                  for r, original in zip(exported, combined)), 'comparison CSV preserves both arm scores')
        page.locator('#results-controls > details > summary').click()
        study = combined[0]['metadata']['studies'][0]['gse']
        choose('results-study', study)
        subset = [r for r in combined if study in [s['gse'] for s in r['metadata']['studies']]]
        count(len(subset), total)
        graph_matches(subset)
        check([r['gsm'] for r in download()] == [r['gsm'] for r in subset], 'comparison study filter, network and CSV agree')
        page.locator('#results-reset').click()
        count(total, total)
        check(states['results-rows-store']['data'] == combined, 'comparison reset preserves both rankings')
        page.wait_for_load_state('networkidle')
        drain_responses()
        check(not errors, f'no browser errors: {errors}')
        page.remove_listener('response', response_seen)
        browser.close()
    (output/'verification.json').write_text(json.dumps({'checks':checks,'passed':len(checks)}, indent=2)+'\n')
    print(f'ALL {len(checks)} METADATA E2E CHECKS PASSED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8066')
    parser.add_argument('--output', type=Path, default=Path('.lavish/metadata-upgrade'))
    args = parser.parse_args()
    run(args.url, args.output)
