"""Capture settled current-view screenshots and inspect presentation states."""
from pathlib import Path
import json
from playwright.sync_api import sync_playwright

OUT=Path(__file__).resolve().parents[2]/'.lavish/nasa-cohesion'
checks=[]
def check(value,label):
    assert value,label
    checks.append(label)
    print('PASS',label,flush=True)
def shot(page,name):
    page.mouse.move(0,0);page.wait_for_timeout(350)
    page.screenshot(path=str(OUT/(name+'.png')))

with sync_playwright() as pw:
    browser=pw.chromium.launch(args=['--no-proxy-server'])
    page=browser.new_page(viewport={'width':1440,'height':1000})
    page.goto('http://127.0.0.1:8065');page.wait_for_selector('#sample-preview .sample-preview-name')
    page.wait_for_timeout(700)
    check(page.locator('.dash-dropdown-trigger').first.evaluate("e=>getComputedStyle(e).borderTopWidth")=='0px','dropdown has one border owner')
    check(page.locator('.dash-dropdown').first.evaluate("e=>getComputedStyle(e).borderTopWidth")=='1px','dropdown outer border is visible')
    shot(page,'after-empty')
    button=page.locator('#search-button')
    button.hover();page.wait_for_timeout(200)
    check(button.evaluate("e=>getComputedStyle(e).backgroundColor")=='rgb(182, 1, 9)','primary hover')
    page.mouse.down();page.wait_for_timeout(200)
    check(button.evaluate("e=>getComputedStyle(e).backgroundColor")=='rgb(139, 10, 3)','primary active')
    page.mouse.up()
    page.wait_for_function("document.querySelector('#search-status').textContent.includes('Retrieved 5')")
    page.wait_for_function("document.querySelector('#details-panel').textContent.includes('Mmus_C57-6J_EYE_FLT_Rep1_M23')")
    check(page.evaluate("document.querySelector('#network-graph .js-plotly-plot')._fullData.at(-1).marker.opacity") == 1,'network markers match solid legend')
    colors=page.evaluate("document.querySelector('#network-graph .js-plotly-plot')._fullData.at(-1).marker.color")
    check(set(colors)=={'#0bab9f','#2b7fff','#d9791b'},'network scientific colors preserved')
    for width,height in [(1440,1000),(1280,800)]:
        page.set_viewport_size({'width':width,'height':height});page.wait_for_timeout(900)
        shot(page,f'after-retrieve-{width}')
    page.set_viewport_size({'width':1440,'height':1000});page.locator('#see-on-map').click()
    page.wait_for_function("document.querySelector('#manifold-graph .js-plotly-plot')?._fullData?.length>0",timeout=60000)
    page.wait_for_timeout(1200)
    check(page.evaluate("document.querySelector('#manifold-graph .js-plotly-plot')._fullLayout.hoverlabel.bgcolor")=='#17171b','explicit neutral Map hover theme')
    for width,height in [(1440,1000),(1280,800)]:
        page.set_viewport_size({'width':width,'height':height});page.wait_for_timeout(1100)
        shot(page,f'after-map-{width}')
    page.set_viewport_size({'width':1440,'height':1000})
    # Delay genuine responses to expose the existing loading state, without
    # changing callback outputs or presenting fake scientific results.
    cdp=page.context.new_cdp_session(page)
    cdp.send('Network.enable')
    cdp.send('Network.emulateNetworkConditions',{'offline':False,'latency':1400,'downloadThroughput':-1,'uploadThroughput':-1})
    page.locator('#method label').filter(has_text='PCA').click()
    page.wait_for_selector('.dash-spinner',state='visible',timeout=15000)
    check(True,'real Map loading indicator appears during a pending callback')
    shot(page,'after-map-loading')
    cdp.send('Network.emulateNetworkConditions',{'offline':False,'latency':0,'downloadThroughput':-1,'uploadThroughput':-1})
    page.wait_for_selector('.dash-spinner',state='hidden',timeout=60000)
    check(True,'Map loading completes')
    (OUT/'state-checks.json').write_text(json.dumps(checks,indent=2))
    browser.close()
