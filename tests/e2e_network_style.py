"""Real-data acceptance for the focused NASA network/Color by refinement."""
import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright, expect

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=8066)
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '.lavish/network-refinement'
OUT.mkdir(parents=True, exist_ok=True)
checks = []


def check(value, label):
    assert value, label
    checks.append(label)
    print('PASS', label, flush=True)


def point(page, kind, which=0):
    return page.evaluate('''([kind, which]) => {
      const g = document.querySelector('#network-graph .js-plotly-plot');
      const t = g._fullData.find(t => t.meta?.network_nodes);
      const i = t.customdata.map((d,i)=>d[0]===kind?i:-1).filter(i=>i>=0)[which];
      const r = g.getBoundingClientRect(), l = g._fullLayout;
      return {id:t.customdata[i][1], x:r.x+l.xaxis._offset+l.xaxis.l2p(t.x[i]),
        y:r.y+l.yaxis._offset+l.yaxis.l2p(t.y[i])};
    }''', [kind, which])


def state(page):
    return page.evaluate('''() => {
      return [...document.querySelectorAll('#network-graph .scatterlayer .trace')]
        .filter(e=>e.__data__?.[0]?.trace?.meta?.network_edge).map(e=>{
          const style=getComputedStyle(e.querySelector('.js-line'));
          return {ids:e.__data__[0].trace.meta.network_edge,
            width:parseFloat(style.strokeWidth), color:style.stroke};
        });
    }''')


def shot(page, name):
    page.screenshot(path=str(OUT / (name + '.png')))


with sync_playwright() as pw:
    browser = pw.chromium.launch(args=['--no-proxy-server'])
    page = browser.new_page(viewport={'width':1440, 'height':1000}, device_scale_factor=1)
    errors = []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.goto(f'http://127.0.0.1:{args.port}')
    page.wait_for_selector('#sample-preview .sample-preview-name')
    page.locator('#search-button').click()
    page.wait_for_function("document.querySelector('#search-status').textContent.includes('Retrieved 5')")
    page.wait_for_timeout(600)
    colors = page.evaluate('''() => {
      const g=document.querySelector('#network-graph .js-plotly-plot');
      const n=g.data.find(t=>t.meta?.network_nodes);
      return n.customdata.map((d,i)=>[d[0],d[0]==='query'?n.marker.line.color[i]:n.marker.color[i],n.marker.symbol[i]]);
    }''')
    check(['query','#d83933','circle'] in colors, 'real query uses a red ring')
    check(all(c[1:] == ['#0b3d91','circle'] for c in colors if c[0]=='gsm'), 'GSM nodes use dark-blue circles')
    check(all(c[1:] == ['#58585b','diamond'] for c in colors if c[0]=='gse'), 'GSE nodes use graphite diamonds')
    for selector, prop, expected in [('.legend-swatch--query','borderTopColor','rgb(216, 57, 51)'),
                                    ('.legend-swatch--circle','backgroundColor','rgb(11, 61, 145)'),
                                    ('.legend-swatch--diamond','backgroundColor','rgb(88, 88, 91)')]:
        check(page.locator(selector).evaluate('(e,p)=>getComputedStyle(e)[p]',prop)==expected, 'legend matches '+selector)
    geometry = page.evaluate("() => Object.fromEntries(['.app-header','.sidebar','.workspace','.inspector','#network-graph','#search-button'].map(s=>{let r=document.querySelector(s).getBoundingClientRect();return [s,{x:r.x,y:r.y,w:r.width,h:r.height}]}))")
    before = json.loads((OUT/'before-geometry.json').read_text())
    check(all(abs(v-before[s][k])<1 for s,rect in geometry.items() for k,v in rect.items()), 'workspace geometry unchanged')
    check(all(e['color']=='rgb(133, 133, 137)' and e['width']==3 for e in state(page)), 'default edges are neutral and uniform')
    shot(page, 'after-1440')
    gsm = point(page, 'gsm')
    page.mouse.move(gsm['x'],gsm['y']);page.wait_for_timeout(400)
    check(all((e['width']==4.5)==(gsm['id'] in e['ids']) for e in state(page)), 'hover emphasizes only connected edges')
    check(gsm['id'] in page.locator('#network-graph .hoverlayer').text_content(), 'hover retains sample identifiers')
    shot(page, 'hover')
    page.mouse.click(gsm['x'],gsm['y']);page.mouse.move(0,0)
    page.wait_for_function("id => document.querySelector('#details-panel').textContent.includes(id)", arg=gsm['id'], timeout=60000)
    page.wait_for_timeout(400)
    check(any(e['width']==4.5 for e in state(page)), 'inspection emphasis remains after mouse leaves')
    other=point(page,'gsm',1)
    page.mouse.move(other['x'],other['y']);page.wait_for_timeout(400)
    check(all((e['width']==4.5)==(other['id'] in e['ids']) for e in state(page)), 'another hover temporarily emphasizes its relationships')
    page.mouse.move(0,0);page.wait_for_timeout(400)
    check(all((e['width']==4.5)==(gsm['id'] in e['ids']) for e in state(page)), 'unhover restores inspected relationships')
    shot(page, 'inspected')
    query=point(page,'query')
    page.mouse.move(query['x'],query['y'])
    expect(page.locator('#network-graph .hoverlayer')).to_contain_text(query['id'])
    page.mouse.click(query['x'],query['y']);page.mouse.move(0,0)
    page.wait_for_function("document.querySelector('#details-panel').textContent.includes('OSD-100')")
    check(True, 'outlined query remains clickable and opens metadata')
    study=point(page,'gse')
    page.mouse.move(study['x'],study['y'])
    expect(page.locator('#network-graph .hoverlayer')).to_contain_text(study['id'])
    page.mouse.click(study['x'],study['y']);page.mouse.move(0,0)
    page.wait_for_function("id=>document.querySelector('#details-panel').textContent.includes(id)",arg=study['id'],timeout=60000)
    check(True, 'study diamond opens study metadata')
    with page.expect_download() as download:
        page.locator('#network-graph .modebar-btn[data-title="Download plot as a PNG"]').click(force=True)
    check(download.value.suggested_filename.endswith('.png'), 'PNG export remains available')
    page.locator('#search-button').click();page.wait_for_timeout(1500)
    check(all(e['width']==3 for e in state(page)), 'new retrieval clears visual inspection emphasis')
    for width,height in [(1280,800),(390,844)]:
        page.set_viewport_size({'width':width,'height':height});page.wait_for_timeout(700)
        check(page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'), f'{width}px retrieval has no horizontal overflow')
        check(page.locator('#network-graph').evaluate("e=>Math.abs(e.getBoundingClientRect().height-e.querySelector('.js-plotly-plot')._fullLayout.height)<3"),f'{width}px graph resizes correctly')
        shot(page,f'after-{width}')
    page.set_viewport_size({'width':1440,'height':1000})
    page.locator('#see-on-map').click()
    page.wait_for_function("document.querySelector('#manifold-graph .js-plotly-plot')?._fullData?.length>0",timeout=60000)
    page.wait_for_timeout(1200)
    page.locator('#color-by').click()
    dropdown=page.locator('.dash-dropdown-content')
    expect(dropdown.get_by_role('option')).to_have_count(2)
    expect(dropdown).to_contain_text('Tissue')
    expect(dropdown).to_contain_text('Species')
    check(dropdown.locator('input:not([type=radio]):not([type=checkbox])').count()==0,'Color by menu has no search input')
    check('Tissue' in dropdown.inner_text() and 'Species' in dropdown.inner_text(),'both Map coloring choices remain')
    shot(page,'map-menu')
    page.keyboard.press('ArrowDown');page.keyboard.press('Enter')
    page.wait_for_function("document.querySelector('#color-by').textContent.includes('Species')")
    page.wait_for_function("document.querySelector('#legend-title').textContent.includes('Species')",timeout=60000)
    check(True,'keyboard selects Species and updates the Map legend')
    page.locator('#color-by').click();dropdown.get_by_role('option').filter(has_text='Tissue').click()
    page.wait_for_function("document.querySelector('#legend-title').textContent.includes('Tissue')",timeout=60000)
    check(True,'Tissue coloring still updates the Map')
    page.locator('.bm-legend-search input').fill('Liver')
    page.wait_for_function("document.querySelector('.bm-legend-list').textContent.trim().startsWith('Liver')")
    check(True,'tissue legend filter is retained')
    page.locator('.bm-legend-search input').fill('')
    page.locator('#find-input').fill('GSE143281')
    page.wait_for_selector('.bm-suggest-row');page.keyboard.press('ArrowDown');page.keyboard.press('Enter')
    page.wait_for_function("document.querySelector('#find-status').textContent.includes('GSE143281')")
    check(True,'study search is retained')
    for width,height in [(1280,800),(390,844)]:
        page.set_viewport_size({'width':width,'height':height});page.wait_for_timeout(700)
        check(page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),f'{width}px Map has no horizontal overflow')
    page.set_viewport_size({'width':1440,'height':1000})
    page.get_by_role('link',name='Retrieve',exact=True).click()
    page.wait_for_selector('#search-button');page.locator('#search-button').click()
    page.wait_for_function("document.querySelector('#search-status').textContent.includes('Retrieved 5')")
    gsm=point(page,'gsm');page.mouse.move(gsm['x'],gsm['y']);page.wait_for_timeout(400)
    check(any(e['width']==4.5 for e in state(page)), 'hover reconnects after route navigation')
    check(not errors,'no browser runtime errors: '+str(errors))
    (OUT/'verification.json').write_text(json.dumps({'checks':checks,'errors':errors},indent=2))
    browser.close()
print(f'{len(checks)} browser checks passed')
