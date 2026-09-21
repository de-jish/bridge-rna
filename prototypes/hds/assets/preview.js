/* Prototype-only layout controls. Dash remains the owner of all scientific state. */
(() => {
  const root = document.documentElement;
  const names = ['a','b','c'];
  const initial = new URL(location.href).searchParams.get('variant');
  root.dataset.variant = names.includes(initial) ? initial : 'b';
  root.dataset.previewState = 'live';
  const resize = () => {
    document.querySelectorAll('.js-plotly-plot').forEach(plot => {
      if (window.Plotly) window.Plotly.Plots.resize(plot);
    });
  };
  const sync = () => {
    document.querySelectorAll('[data-variant]').forEach(el => {
      if (el.tagName === 'BUTTON') el.setAttribute('aria-pressed',String(el.dataset.variant===root.dataset.variant));
    });
  };
  function changeVariant(key) {
    root.dataset.variant=key; root.dataset.queryOpen='false';
    const url=new URL(location.href); url.searchParams.set('variant',key);
    history.replaceState(null,'',url); sync();
    document.getElementById('toggle-query')?.setAttribute('aria-expanded','false');
    requestAnimationFrame(resize);
  }
  function setState(value) {
    root.dataset.previewState=value;
    const panel=document.getElementById('preview-state-panel');
    if (!panel) return;
    panel.replaceChildren();
    if (value==='live') return;
    const messages={loading:['Retrieving samples','A search is in progress. Your query settings are retained.'],empty:['No retrieved samples yet','Choose a sample, cohort, or counts file, then run a search.'],error:['The search could not complete','The query is retained. Review the status details and try again.']};
    const tag=document.createElement('span');tag.className='preview-state-tag';tag.textContent='UI state example · no computation simulated';
    const title=document.createElement('h2');title.textContent=messages[value][0];
    const copy=document.createElement('p');copy.textContent=messages[value][1];
    const back=document.createElement('button');back.className='usa-button usa-button--secondary';back.textContent='Return to live data';back.onclick=()=>{setState('live');document.getElementById('preview-state').value='live';document.getElementById('preview-state').focus();};
    panel.append(tag,title,copy,back);
  }
  document.addEventListener('change',event=>{if(event.target.id==='preview-state')setState(event.target.value);});
  document.addEventListener('click',event=>{
    const variant=event.target.closest('button[data-variant]');if(variant){changeVariant(variant.dataset.variant);return;}
    const toggle=event.target.closest('#toggle-query');
    if(toggle){const open=root.dataset.queryOpen!=='true';root.dataset.queryOpen=String(open);toggle.setAttribute('aria-expanded',String(open));if(open)document.querySelector('#query-controls button')?.focus();resize();}
    const expand=event.target.closest('#expand-plot');
    if(expand){const open=root.dataset.expanded!=='true';root.dataset.expanded=String(open);expand.textContent=open?'Close expanded plot':'Expand plot';expand.setAttribute('aria-expanded',String(open));resize();}
    // Direct guided navigation without a required Next/Back sequence.
    const anchor=event.target.closest('.guided-nav a');
    if(anchor){event.preventDefault();const target=document.querySelector(anchor.getAttribute('href'));target?.scrollIntoView({behavior:'auto',block:'start'});target?.setAttribute('tabindex','-1');target?.focus({preventScroll:true});}
  });
  document.addEventListener('keydown',event=>{
    if(event.key==='Escape'){
      if(root.dataset.expanded==='true')document.getElementById('expand-plot')?.click();
      if(root.dataset.queryOpen==='true'){root.dataset.queryOpen='false';const toggle=document.getElementById('toggle-query');toggle?.setAttribute('aria-expanded','false');toggle?.focus();}
    }
    // Arrow shortcuts apply only while the prototype switcher has focus.
    if(event.target.closest('.prototype-bar nav')&&['ArrowLeft','ArrowRight'].includes(event.key)){
      event.preventDefault();const index=names.indexOf(root.dataset.variant);changeVariant(names[(index+(event.key==='ArrowRight'?1:2))%3]);document.querySelector(`button[data-variant=${root.dataset.variant}]`).focus();
    }
  });
  new MutationObserver(sync).observe(document.body,{childList:true,subtree:true});
  window.addEventListener('resize',resize);
})();
