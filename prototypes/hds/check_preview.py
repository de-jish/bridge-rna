"""Focused, real-browser acceptance checks for this throwaway preview."""
from pathlib import Path
import json
import sys
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'.lavish'
OUT.mkdir(exist_ok=True)
checks=[]
def check(value,label):
    assert value,label
    checks.append(label)
    print('PASS',label,flush=True)

with sync_playwright() as pw:
    browser=pw.chromium.launch(args=['--no-proxy-server'])
    page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
    errors=[]
    page.on('pageerror',lambda error:errors.append(str(error)))
    page.goto('http://127.0.0.1:8065/?variant=b',wait_until='domcontentloaded')
    page.wait_for_selector('.result-row',timeout=60000)
    original=page.locator('#result-rows').inner_text()
    check('GSM6431263' in original and '0.9970' in original,'real representative query and scores')
    check(page.evaluate("document.fonts.check('14px \"Public Sans Web\"')"),'HDS body font loaded')
    for variant in ['a','b','c']:
        page.locator(f'button[data-variant={variant}]').click()
        page.wait_for_timeout(500)
        check(page.locator('#result-rows').inner_text()==original,variant+' switching preserves results')
        check(page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),variant+' desktop has no horizontal overflow')
        fit=page.evaluate("""() => {const p=document.querySelector('#network-graph .js-plotly-plot'),w=document.querySelector('.graph-wrap');return {plot:p.getBoundingClientRect().height,wrap:w.getBoundingClientRect().height}}""")
        check(abs(fit['plot']-fit['wrap'])<3,variant+' chart fits canvas '+str(fit))
        page.evaluate('window.scrollTo(0,0)')
        page.screenshot(path=str(OUT/f'hds-{variant}.png'),full_page=False)
        page.screenshot(path=str(OUT/f'hds-{variant}-full.png'),full_page=True)
        page.locator('#expand-plot').click()
        check(page.evaluate('document.documentElement.dataset.expanded')=='true',variant+' expand plot')
        page.keyboard.press('Escape')
        check(page.evaluate('document.documentElement.dataset.expanded')=='false',variant+' close expanded plot by keyboard')
        if variant=='a':
            page.locator('#toggle-query').click()
            check(page.locator('#query-controls').is_visible(),'A query drawer opens')
            page.keyboard.press('Escape')
            check(not page.locator('#query-controls').is_visible(),'A query drawer closes')
        for state in ['loading','empty','error']:
            page.locator('#preview-state').select_option(state)
            check(page.locator('#preview-state-panel').is_visible(),variant+' '+state+' example')
            page.get_by_role('button',name='Return to live data').click()
        page.set_viewport_size({'width':390,'height':844})
        page.wait_for_timeout(500)
        check(page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),variant+' narrow layout has no horizontal overflow')
        page.screenshot(path=str(OUT/f'hds-{variant}-mobile.png'),full_page=True)
        page.set_viewport_size({'width':1440,'height':1000})
    page.locator('button[data-variant=b]').click()
    page.locator('.result-row').first.click()
    page.wait_for_function("document.querySelector('#details-panel').textContent.includes('GSM6431263')",timeout=60000)
    check(True,'result selection opens production metadata inspector')
    page.locator('button[data-variant=a]').click()
    check('GSM6431263' in page.locator('#details-panel').inner_text(),'switching preserves selected result')
    page.locator('button[data-variant=b]').click()
    slider=page.locator('#topk-slider [role=slider]').first
    slider.focus();page.keyboard.press('ArrowRight')
    check(slider.get_attribute('aria-valuenow')=='6','keyboard changes retrieval depth')
    page.locator('#search-button').click()
    page.wait_for_function("document.querySelectorAll('.result-row').length===6",timeout=60000)
    check(True,'search retrieves selected depth')
    page.locator('#mode-tab-cohort').click()
    page.locator('#study-dropdown').click()
    page.locator('.dash-dropdown-content').get_by_text('OSD-137',exact=True).click()
    page.wait_for_timeout(1200)
    page.locator('.cohort-members-summary').click()
    check(page.locator('.member-list label').count()>=2,'cohort members and exclusions are available')
    page.locator('#cohort-compare-dropdown').click()
    page.locator('.dash-dropdown-content').get_by_text('differs by').first.click()
    page.wait_for_timeout(500)
    page.locator('#cohort-search-button').click()
    page.wait_for_function("document.querySelector('#search-status').textContent.includes('Jaccard')",timeout=60000)
    check('across A/B' in page.locator('#result-count').inner_text(),'independent A/B comparison represented in result list')
    page.locator('#mode-tab-upload').click()
    page.set_input_files('#upload-counts input[type=file]',str(ROOT/'examples/osdr_upload_example.csv'))
    page.wait_for_function("document.querySelector('#upload-preview').textContent.includes('2 sample')",timeout=30000)
    check(page.locator('#upload-sample-column').is_visible(),'upload parsing exposes sample columns')
    page.locator('#upload-search-button').click()
    page.wait_for_function("document.querySelector('#search-status').textContent.includes('uploaded counts matrix live')",timeout=120000)
    check(True,'real upload embedding and retrieval completes')
    with page.expect_download() as download:
        page.locator('#network-graph .modebar-btn[data-title="Download plot as a PNG"]').click()
    check(download.value.suggested_filename.endswith('.png'),'Plotly PNG export works')
    page.locator('#see-on-map').click()
    page.wait_for_selector('#manifold-graph .js-plotly-plot',timeout=60000)
    check(True,'map navigation retains live workflow')
    page.screenshot(path=str(OUT/'map-check.png'))
    page.get_by_role('link',name='Retrieve',exact=True).click()
    page.wait_for_selector('#prototype-results',timeout=30000)
    check(True,'returning from map retains prototype composition')
    check(not errors,'no browser runtime errors: '+str(errors))
    (OUT/'verification.json').write_text(json.dumps({'checks':checks,'errors':errors},indent=2))
    browser.close()
print(f'{len(checks)} browser checks passed',flush=True)
