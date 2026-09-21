"""Focused browser acceptance for the layout-preserving NASA refinement."""
from pathlib import Path
import json
import sys
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests'))
from e2e_check import GSM_NODE_JS, NETWORK_READY_JS
OUT = ROOT / '.lavish/nasa-refinement'
OUT.mkdir(exist_ok=True)
checks = []

def check(value, label):
    assert value, label
    checks.append(label)
    print('PASS', label, flush=True)

def shot(page, name):
    page.mouse.move(0, 0)
    page.wait_for_timeout(350)
    page.screenshot(path=str(OUT / (name + '.png')), full_page=False)

def geometry(page):
    return page.evaluate("""() => Object.fromEntries(['.app-header','.sidebar','.workspace','.inspector','#network-graph','#search-button'].map(s=>{let r=document.querySelector(s).getBoundingClientRect();return [s,{x:r.x,y:r.y,w:r.width,h:r.height}]}))""")

with sync_playwright() as pw:
    browser = pw.chromium.launch(args=['--no-proxy-server'])
    page = browser.new_page(viewport={'width':1440,'height':1000}, device_scale_factor=1)
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto('http://127.0.0.1:8065', wait_until='domcontentloaded')
    page.wait_for_selector('#sample-preview .sample-preview-name')
    page.evaluate('document.fonts.ready')
    check(page.locator('.sample-preview-name').is_visible(), 'sample metadata visible before search')
    check(not page.locator('.exploration-heading,#expand-plot,#prototype-results').count(), 'original workspace without prototype structural additions')
    baseline = json.loads((OUT / 'baseline-geometry.json').read_text())
    measured = {'desktop_empty': geometry(page)}
    for selector, rect in baseline['desktop_empty'].items():
        check(all(abs(rect[k]-measured['desktop_empty'][selector][k]) <= 2 for k in rect), 'original geometry: '+selector)
    shot(page, 'after-empty')
    page.wait_for_timeout(1000)  # Let initial Dash callbacks finish before keyboard input.
    page.locator('#search-button').focus()
    check(page.locator('#search-button').evaluate("e=>getComputedStyle(e).outlineStyle") == 'solid', 'visible keyboard focus on Search')
    page.keyboard.press('Enter')
    page.wait_for_function(NETWORK_READY_JS)
    check('Retrieved 5' in page.locator('#search-status').inner_text(), 'real five-hit retrieval')
    check(page.locator('#network-graph').evaluate("e=>e.querySelector('.js-plotly-plot')._fullData.at(-1).customdata.some(d=>d[1]==='GSM6431263')"), 'same representative result identity')
    measured['desktop_populated'] = geometry(page)
    shot(page, 'after-populated')
    point = page.evaluate(GSM_NODE_JS)
    page.mouse.click(point['x'], point['y'])
    page.wait_for_function("document.querySelector('#details-panel').textContent.includes('GSM6431263')", timeout=60000)
    check(True, 'graph click opens original metadata inspector')
    shot(page, 'after-details')
    with page.expect_download() as download:
        page.locator('#network-graph .modebar-btn[data-title="Download plot as a PNG"]').click(force=True)
    check(download.value.suggested_filename.endswith('.png'), 'network PNG export')
    page.locator('.plotly-notifier .notifier-close').evaluate_all('els=>els.forEach(e=>e.click())')
    slider = page.locator('#topk-slider [role=slider]').first
    slider.focus(); page.keyboard.press('ArrowRight')
    check(slider.get_attribute('aria-valuenow') == '6', 'keyboard retrieval-depth control')
    page.keyboard.press('ArrowLeft')
    for width, height in [(1280,800),(390,844)]:
        page.set_viewport_size({'width':width,'height':height});page.wait_for_timeout(800)
        check(page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'), f'{width}px retrieval has no horizontal overflow')
        check(page.locator('#network-graph').evaluate("e=>Math.abs(e.getBoundingClientRect().height-e.querySelector('.js-plotly-plot')._fullLayout.height)<3"), f'{width}px plot resizes into its container')
        if width == 1280:
            check(page.locator('#search-button').bounding_box()['y']+page.locator('#search-button').bounding_box()['height'] < height, 'Search accessible without scrolling at laptop viewport')
        shot(page, f'after-retrieve-{width}')
    page.set_viewport_size({'width':1440,'height':1000})
    page.locator('#mode-tab-cohort').click()
    page.locator('#study-dropdown').click()
    page.locator('.dash-dropdown-content').get_by_text('OSD-137', exact=True).click()
    page.wait_for_timeout(1000)
    page.locator('.cohort-members-summary').click()
    check(page.locator('.member-list label').count() >= 2, 'cohort members remain accessible')
    page.locator('.cohort-members-summary').click()
    page.locator('#cohort-compare-dropdown').click()
    page.locator('.dash-dropdown-content').get_by_text('differs by').first.click()
    page.locator('#cohort-search-button').click()
    page.wait_for_function("document.querySelector('#search-status').textContent.includes('Jaccard')", timeout=60000)
    check(True, 'real two-cohort comparison')
    shot(page, 'after-cohort')
    page.locator('#mode-tab-upload').click()
    expect(page.locator('#upload-search-button')).to_be_disabled()
    check(True, 'upload search disabled until valid input')
    page.set_input_files('#upload-counts input[type=file]', {'name':'invalid.csv','mimeType':'text/csv','buffer':b''})
    page.wait_for_selector('#upload-preview .status-error')
    check(page.locator('#upload-search-button').is_disabled(), 'invalid upload cannot run retrieval')
    shot(page, 'after-upload-error')
    page.set_input_files('#upload-counts input[type=file]', str(ROOT / 'examples/osdr_upload_example.csv'))
    page.wait_for_function("document.querySelector('#upload-preview').textContent.includes('2 sample')", timeout=30000)
    check(page.locator('#upload-sample-column').is_visible(), 'valid upload exposes sample selection')
    page.locator('#upload-search-button').click()
    page.wait_for_function("document.querySelector('#search-status').textContent.includes('uploaded counts matrix live')", timeout=120000)
    check(True, 'real upload embedding and retrieval')
    shot(page, 'after-upload')
    page.locator('#see-on-map').click()
    page.wait_for_selector('#manifold-graph .js-plotly-plot', timeout=60000)
    page.wait_for_function("document.querySelector('#manifold-graph .js-plotly-plot')._fullData?.length>0", timeout=60000)
    page.wait_for_timeout(1800)
    check(True, 'retrieval-to-Map navigation')
    shot(page, 'after-map-upload')
    page.locator('#dims label').filter(has_text='3D').click()
    page.wait_for_function("document.querySelector('#manifold-graph .js-plotly-plot')._fullData.some(t=>t.type==='scatter3d')", timeout=60000)
    check(True, 'Map 3D control')
    page.locator('#dims label').filter(has_text='2D').click()
    page.wait_for_function("document.querySelector('#manifold-graph .js-plotly-plot')._fullData.some(t=>t.type==='scattergl')", timeout=60000)
    page.locator('#color-by').click()
    check(page.locator('.dash-dropdown-content').is_visible(), 'Map color selector opens')
    page.keyboard.press('Escape')
    for width, height in [(1280,800),(390,844)]:
        page.set_viewport_size({'width':width,'height':height});page.wait_for_timeout(1200)
        check(page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'), f'{width}px Map has no horizontal overflow')
        shot(page, f'after-map-{width}')
    page.get_by_role('link', name='Retrieve', exact=True).click()
    page.wait_for_selector('#sample-preview .sample-preview-name')
    check(True, 'return to original Retrieve view')
    # Capture the same representative Map data in a fresh session. Returning
    # from Map has a pre-existing empty-upload enabled-state issue, tracked in
    # README; keep first-load and navigation coverage distinct.
    reference = browser.new_page(viewport={'width':1440,'height':1000})
    reference.goto('http://127.0.0.1:8065')
    reference.wait_for_selector('#sample-preview .sample-preview-name')
    reference.locator('#search-button').click()
    reference.wait_for_function("document.querySelector('#search-status').textContent.includes('Retrieved 5')")
    reference.locator('#see-on-map').click()
    reference.wait_for_function("document.querySelector('#manifold-graph .js-plotly-plot')?._fullData?.length>0", timeout=60000)
    reference.wait_for_timeout(1800)
    shot(reference, 'after-map')
    reference.close()
    check(not errors, 'no browser runtime errors: '+str(errors))
    (OUT / 'verification.json').write_text(json.dumps({'checks':checks,'errors':errors,'geometry':measured}, indent=2))
    browser.close()
print(f'{len(checks)} browser checks passed', flush=True)
