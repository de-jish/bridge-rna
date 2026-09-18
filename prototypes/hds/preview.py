"""Three throwaway HDS layouts around the unchanged Bridge RNA application.

Run: /Users/josh/Bridge-RNA/.venv/bin/python prototypes/hds/preview.py
The layouts change composition only. Scientific callbacks are registered by app.py.
"""
from pathlib import Path
import sys
import json
import re
import textwrap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from dash import Input, Output, State, ALL, ctx, html, dcc, no_update
from flask import send_from_directory
import app as production
from bridge_rna import callbacks as callbacks, figures, layout

# Presentation-only theme adapter. Preserve every scientific mark and coordinate.
figures.GRAPH_THEME.update(font_sans="Public Sans Web, sans-serif",
                          font_mono="DM Mono, monospace", text_primary="#17171b",
                          text_secondary="#58585b", grid="#e3e3e3")
def themed(builder):
    def wrapped(*args, **kwargs):
        fig = builder(*args, **kwargs)
        for trace in fig.data:
            if trace.text is not None and trace.customdata is not None:
                trace.text = ["<br>".join(textwrap.wrap(str(label),width=22))
                              if row[0] in ("query","query_b") else label
                              for label,row in zip(trace.text,trace.customdata)]
        fig.update_layout(font=dict(family="Public Sans Web, sans-serif", size=13, color="#17171b"),
                          hoverlabel=dict(bgcolor="#14294a", bordercolor="#14294a",
                                          font=dict(family="Public Sans Web, sans-serif",size=13,color="white")),
                          legend=dict(font=dict(family="Public Sans Web, sans-serif",size=12)),
                          paper_bgcolor="white", plot_bgcolor="white", autosize=True, height=None)
        fig.update_layout(margin=dict(t=36))
        return fig
    return wrapped
for name in ("build_network_figure", "build_comparison_figure", "_empty_network_figure"):
    setattr(callbacks, name, themed(getattr(callbacks, name)))

app = production.build_app()
serve_original = app.layout

def walk(node):
    if hasattr(node, "to_plotly_json"):
        yield node
        children = getattr(node, "children", None)
        for child in children if isinstance(children, (list, tuple)) else [children]:
            yield from walk(child)

def button(label, id=None, **props):
    return html.Button(label, **({"id":id} if id else {}),
                       className="usa-button usa-button--outline", **props)

def toolbar():
    return html.Div(className="prototype-bar", children=[
        html.Span("Design preview", className="prototype-label"),
        html.Nav( **{"aria-label":"Prototype direction"}, children=[
            html.Button([html.Span(key.upper(),className="variant-key"), name],
                        **{"data-variant":key, "aria-pressed":"true" if key=="b" else "false"})
            for key,name in [("a","Visualization first"),("b","Research workspace"),("c","Guided exploration")]
        ]),
        html.Label(["Preview state", html.Select(id="preview-state",children=[
            html.Option(name,value=value) for value,name in [("live","Live data"),("loading","Loading example"),("empty","Empty example"),("error","Error example")]
        ])]),
        html.A("Compare designs",href="/review/",target="_blank")
    ])

def results():
    return html.Section(id="prototype-results",className="prototype-results",children=[
        html.Div(className="results-heading",children=[html.H2("Retrieved samples"),html.Span(id="result-count")]),
        html.P("Ranked by embedding cosine similarity",className="results-caption"),
        html.Div(id="result-rows"),
    ])

def adapt_tree(root):
    nodes = list(walk(root))
    for n in nodes:
        ident = getattr(n,"id",None)
        cls = getattr(n,"className","") or ""
        if cls == "app-brand-mark":
            n.children = html.Img(src="/prototype-assets/nasa.svg", alt="NASA")
        if cls == "app-title": n.children="Bridge RNA"
        if cls == "app-subtitle": n.children="Space Biosciences · Ames Research Center"
        if cls == "app-header-meta":
            n.children=html.Span(f"{layout.ARCHS4_SAMPLE_COUNT:,} ARCHS4 · {layout.ELIGIBLE_OSDR_COUNT:,} OSDR samples",className="collection-count")
        if cls == "sidebar":
            n.id="query-controls"
            n.children.insert(0,html.Div(className="section-intro",children=[html.Span("1",className="step-number"),html.H2("Choose a query")]))
        if ident == "mode-panel-sample":
            n.children = [html.Details([html.Summary("Sample metadata"), child],className="sample-disclosure")
                          if getattr(child,"id",None)=="sample-preview" else child for child in n.children]
        if cls == "sidebar-title": n.children="Query settings"
        if cls == "workspace":
            n.id="network-section"
            n.children.append(results())
        if cls == "inspector":
            n.id="inspection-section"
            n.children.insert(0,html.Div(className="section-intro",children=[html.Span("3",className="step-number"),html.H2("Inspect & compare")]))
        if cls == "panel-header":
            n.children.append(html.Div(className="plot-actions",children=[button("Expand plot",id="expand-plot")]))
        if ident=="hits-store": n.storage_type="memory"
        if ident=="sample-dropdown":
            # Make the shared representative query available to the normal initial
            # search callback, before the study callback populates the full picker.
            n.options=[{"label":layout.default_sample_id.split("|",1)[-1],"value":layout.default_sample_id}]
            n.value=layout.default_sample_id
        if ident=="network-graph":
            n.figure=callbacks._empty_network_figure()
            n.config={"displaylogo":False,"responsive":True,"displayModeBar":True,
                      "modeBarButtonsToRemove":["select2d","lasso2d"],
                      "toImageButtonOptions":{"format":"png","filename":"bridge-rna-network"}}
        # Let actual HDS button classes style actions, without legacy button rules.
        if cls == "btn-primary": n.className="usa-button usa-button--secondary query-action"
        if cls == "btn-secondary": n.className="usa-button usa-button--outline query-action"
    return root

original_view = layout.build_view
layout.build_view = lambda: adapt_tree(original_view())
original_header = production.header
production.header = lambda active: adapt_tree(original_header(active))

def serve_layout():
    shell = serve_original()
    shell.id = "prototype-shell"
    for node in walk(shell):
        if getattr(node, "id", None) == "hits-store": node.storage_type = "memory"
    shell.children.insert(0,toolbar())
    shell.children.insert(3,html.Div(className="exploration-heading",children=[
        html.Div([html.H1("Explore expression similarities"),
                  html.P("Compare a spaceflight query with terrestrial gene expression samples.")]),
        button("Query settings",id="toggle-query",**{"aria-expanded":"false","aria-controls":"query-controls"})
    ]))
    shell.children.insert(4,html.Div(className="query-context",children=[
        html.Div(id="query-context-text"),
        html.Span("Cosine similarity does not establish biological equivalence.",className="science-note")
    ]))
    shell.children.insert(5,html.Nav(className="guided-nav",**{"aria-label":"Exploration steps"},children=[
        html.A([html.Span(str(i),className="step-number"),label],href="#"+target)
        for i,label,target in [(1,"Choose a query","query-controls"),(2,"Explore the network","network-section"),(3,"Inspect & compare","inspection-section")]
    ]))
    shell.children.append(html.Div(id="preview-state-panel",role="status",**{"aria-live":"polite"}))
    return shell

app.layout=serve_layout
app.index_string=re.sub(r'        <link[^\n]+\n', '', production.INDEX_STRING)
app.index_string=app.index_string.replace("{%css%}",
    '<link rel="stylesheet" href="/hds/css/hds.min.css">{%css%}'
    '<link rel="stylesheet" href="/prototype-assets/adapter.css">'
    '<script src="/prototype-assets/preview.js" defer></script>')

@app.server.route("/hds/<path:path>")
def hds_assets(path):
    return send_from_directory(HERE / "node_modules/@nasa-hds/core/dist",path)

@app.server.route("/prototype-assets/<path:path>")
def prototype_assets(path):
    return send_from_directory(HERE / "assets",path)

@app.server.route("/review/")
def review():
    return send_from_directory(ROOT / ".lavish", "hds-comparison.html")

@app.server.route("/review/<path:path>")
def review_assets(path):
    return send_from_directory(ROOT / ".lavish",path)

@app.callback(Output("result-rows","children"),Output("result-count","children"),
              Output("query-context-text","children"),Input("hits-store","data"),
              Input("selected-node-store","data"))
def show_results(payload, selected):
    if not payload:
        return html.P("Choose a query and run Search to retrieve samples.",className="result-empty"),"", "No executed query"
    compare=payload.get("comparison") or {}
    # Separate arm ranks/scores. A/B pooling and ranking remain production-owned.
    rows=[]
    for arm,hits in [("A",payload.get("hits",[])),("B",compare.get("hits_b",[]))]:
        for i,hit in enumerate(hits):
            gsm=hit["gsm"]
            score=hit.get("cosine",hit.get("score",hit.get("similarity",0)))
            rows.append(html.Button(id={"type":"prototype-hit","key":arm+gsm},n_clicks=0,
                className="result-row"+(" selected" if selected and selected.get("node_id")==gsm else ""),
                **{"aria-label":f"Inspect {gsm}, rank {i+1}"+(f", cohort {arm}" if compare else ""),
                   "aria-pressed":"true" if selected and selected.get("node_id")==gsm else "false"},
                children=[html.Span((arm+" · " if compare else "")+str(i+1),className="rank"),
                          html.Div([html.Strong(gsm),html.Span(hit.get("title") or "Source title unavailable",className="hit-title")]),
                          html.Span(hit.get("gse") or "Not annotated",className="hit-study"),
                          html.Span(f"{float(score):.4f}",className="hit-score")]))
    query=payload.get("query") or {}
    label=query.get("sample_name") or payload.get("sample_id")
    context=[html.Strong("Executed query: "),html.Span(label),html.Span(" · "+payload.get("mode", ""),className="source-mode")]
    return rows, f"{len(rows)}"+(" across A/B" if compare else " matches"), context

@app.callback(Output("selected-node-store","data",allow_duplicate=True),
              Input({"type":"prototype-hit","key":ALL},"n_clicks"),prevent_initial_call=True)
def select_result(clicks):
    if not any(clicks) or not isinstance(ctx.triggered_id,dict): return no_update
    return {"kind":"gsm","node_id":ctx.triggered_id["key"][1:]}

if __name__=="__main__":
    print("HDS design preview: http://127.0.0.1:8065/?variant=b",flush=True)
    app.run(host="127.0.0.1",port=8065,debug=False,dev_tools_ui=False)
