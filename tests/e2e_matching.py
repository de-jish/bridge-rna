"""Exercise real matching corrections and capture PR evidence from the live app."""
import argparse
from pathlib import Path
from playwright.sync_api import sync_playwright, expect


def check(url, out):
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel='chrome')
        page = browser.new_page(viewport={'width': 1680, 'height': 1150})
        errors, payloads = [], []
        page.on('pageerror', lambda e: errors.append(str(e)))

        def response(res):
            if res.request.method != 'POST':
                return
            data = res.request.post_data_json or {}
            if 'search-button.n_clicks' in data.get('changedPropIds', []):
                result = res.json().get('response', {}).get('hits-store', {}).get('data')
                if result:
                    payloads.append(result)

        page.on('response', response)
        page.goto(url)
        expect(page.locator('#sample-preview')).to_contain_text('Mmus')
        expect(page.locator('#matching-method')).to_contain_text('Original cosine')
        page.locator('#search-button').click()
        expect(page.locator('#search-status')).to_contain_text('Retrieved 5 hits', timeout=60000)
        assert payloads[-1].get('matching') is None
        original = [r['gsm'] for r in payloads[-1]['hits']]
        page.screenshot(path=str(out/'original-cosine.png'))

        for method, label in [('background', 'Remove shared background (experimental)'),
                               ('whiten', 'Regularized whitening (experimental)')]:
            page.locator('#matching-method').click()
            page.locator('.dash-dropdown-content').get_by_text(label, exact=True).click()
            previous = len(payloads)
            page.locator('#search-button').click()
            expect(page.locator('#search-status')).to_contain_text(label.lower(), timeout=60000)
            expect(page.locator('#matching-evidence')).to_contain_text(label)
            assert len(payloads) == previous + 1
            result = payloads[-1]
            assert result['matching']['method'] == method
            assert result['matching']['reference_count'] == 940455
            assert len(result['neighborhood']['hits']) == 250
            assert all(r['matching_method'] == method for r in result['hits'])
            assert [r['gsm'] for r in result['hits']] != original
            assert all(-1 <= r['score'] <= 1 and r['raw_rank'] >= 1 for r in result['hits'])
            expect(page.locator('#matching-evidence tbody tr')).to_have_count(12)
            expect(page.locator('#matching-evidence a').first).to_have_attribute('href',
                'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc='+result['hits'][0]['gsm'])
            page.screenshot(path=str(out/(method+'-desktop.png')))

        disclosure = page.locator('.matching-distribution summary')
        disclosure.focus()
        page.keyboard.press('Enter')
        expect(page.locator('.matching-settings')).to_be_visible()
        expect(page.locator('#matching-evidence')).to_contain_text('50000')
        disclosure.click()
        # Hit inspector must distinguish adjusted and original geometry.
        from e2e_check import GSM_NODE_JS
        page.wait_for_timeout(600)  # Plotly's responsive redraw follows the disclosure resize.
        point = page.evaluate(GSM_NODE_JS)
        page.mouse.click(point['x'], point['y'])
        expect(page.locator('#details-panel')).to_contain_text('Adjusted cosine')
        expect(page.locator('#details-panel')).to_contain_text('Original cosine')

        # Evidence scrolls inside its own table on narrow screens, not the page.
        page.set_viewport_size({'width': 393, 'height': 900})
        page.locator('#matching-evidence').scroll_into_view_if_needed()
        page.wait_for_function('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path=str(out/'whiten-mobile.png'))

        # Fixed comparisons and cohorts retain their original workflows.
        page.set_viewport_size({'width': 1680, 'height': 1150})
        page.locator('#mode-tab-cohort').click()
        expect(page.locator('#matching-method')).not_to_be_visible()
        expect(page.locator('#matching-evidence')).to_be_empty()
        page.locator('#mode-tab-sample').click()
        expect(page.locator('#matching-evidence')).to_contain_text('Regularized whitening')
        page.locator('#see-on-map').click()
        expect(page.locator('#manifold-graph')).to_be_visible(timeout=60000)
        page.wait_for_function("""() => {
            const g = document.querySelector('#manifold-graph .js-plotly-plot');
            return g && g.data && JSON.stringify(g.data).includes('adjusted cosine (whiten)');
        }""", timeout=60000)
        page.locator('#frame-retrieval').click()
        expect(page.locator('#neighborhood-heading')).to_be_visible(timeout=30000)
        page.wait_for_function("""() => {
            const g = document.querySelector('#manifold-graph .js-plotly-plot');
            return g && g.data && JSON.stringify(g.data).includes('adjusted rank');
        }""", timeout=60000)
        page.get_by_role('link', name='Retrieve', exact=True).click()
        expect(page.locator('#matching-method')).to_be_visible()
        page.locator('#matching-method').click()
        page.locator('.dash-dropdown-content').get_by_text('Original cosine', exact=True).click()
        with page.expect_response(lambda res: res.request.method == 'POST'
                and 'search-button.n_clicks' in (res.request.post_data_json or {}).get('changedPropIds', []), timeout=60000):
            page.locator('#search-button').click()
        expect(page.locator('#matching-evidence')).to_be_empty(timeout=60000)
        assert [r['gsm'] for r in payloads[-1]['hits']] == original
        assert not errors, errors
        browser.close()
    print('Matching browser checks passed: both corrections, baseline restoration, provenance, inspector, narrow layout, mode isolation.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8068')
    parser.add_argument('--out', type=Path, default=Path('docs/images/matching'))
    args = parser.parse_args()
    check(args.url, args.out)
