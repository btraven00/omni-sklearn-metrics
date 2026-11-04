#!/usr/bin/env python3
"""Build the catalog website: one static page of metric cards, plus catalog.json.

  python site/build.py _site        # writes _site/index.html, _site/catalog.json, _site/.nojekyll

Reads catalog.yaml (the same file run.py executes) and pixi.toml (for the sklearn version the
docs links point at). Standard library and PyYAML only.
"""

import hashlib
import json
import re
import subprocess
import sys
from html import escape
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
REPO = "https://github.com/btraven00/omni-sklearn-metrics"

# A metric's family is the set of data types it reads.
FAMILIES = [
    ({"partition"}, "Clustering agreement", "Two partitions of the same cells, e.g. clusters against a reference."),
    ({"embedding", "partition"}, "Cluster separation", "How well an embedding separates its own clusters; no ground truth."),
    ({"labels"}, "Classification", "Predicted labels against true labels, e.g. cell-type annotation."),
    ({"labels", "scores"}, "Class probabilities", "True labels against per-class probabilities."),
    ({"values"}, "Regression", "A number per cell against the true number, e.g. pseudotime."),
]


def family(entry):
    types = set(entry["args"].values())
    return next(name for kinds, name, _ in FAMILIES if kinds == types)


def sklearn_version():
    m = re.search(r'scikit-learn\s*=\s*"(\d+\.\d+)', (ROOT / "pixi.toml").read_text())
    return m.group(1) if m else "stable"


def fmt_range(r):
    lo, hi = ("−∞" if r[0] is None else f"{r[0]:g}"), ("∞" if r[1] is None else f"{r[1]:g}")
    return f"[{lo}, {hi}]".replace("-", "−")


def card(e, version):
    fn = e["call"].rpartition(".")[2]
    docs = f"https://scikit-learn.org/{version}/modules/generated/sklearn.metrics.{fn}.html"
    up = e["higher_is_better"]
    args = "".join(f'<li><code>{escape(a)}</code><button type="button" class="type" data-type="{escape(t)}" '
                   f'title="Show the metrics that read {escape(t)}">{escape(t)}</button></li>' for a, t in e["args"].items())
    rows = [("Reads", f'<ul class="args">{args}</ul>'),
            ("Range", f'{escape(fmt_range(e["range"]))} · {"higher" if up else "lower"} is better')]
    if e.get("kwargs"):
        rows.append(("Fixed", " ".join(f"<code>{escape(k)}={escape(str(v))}</code>" for k, v in e["kwargs"].items())))
    if e.get("requires"):
        rows.append(("Requires", " ".join(f"<code>{escape(a)}: {escape(', '.join(r))}</code>" for a, r in e["requires"].items())))
    hay = " ".join([e["id"], e["name"], fn, family(e), *e["args"], *e["args"].values()]).lower()
    return f"""<article class="card" data-family="{escape(family(e))}" data-types="{escape(' '.join(sorted(set(e['args'].values()))))}" data-hay="{escape(hay)}">
  <div class="top"><h3>{escape(e["name"])}</h3><span class="badge mono" title="catalog id">{escape(e["id"])}</span></div>
  <a class="call mono" href="{docs}" target="_blank" rel="noopener">sklearn.metrics.{escape(fn)}</a>
  <dl>{"".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows)}</dl>
  <div class="foot"><span class="pill">{escape(family(e))}</span>
    <span class="foot-r"><button type="button" class="use" data-use="{escape(e["id"])}">Use in a plan</button>
    <span class="dir {"up" if up else "down"}" title="{"higher" if up else "lower"} is better">{"↑" if up else "↓"}</span></span></div>
</article>"""


def commit():
    """The commit the site is built from: what a plan should pin."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "main"


def plan_entry(e, sha_commit):
    """The module entry for a plan stage, ready to paste."""
    bind = ",".join(f"{a}=<input id>" for a in e["args"])
    return (f'- id: {e["id"]}\n'
            f'  name: "{e["name"]}"\n'
            f'  software_environment: sklearn\n'
            f'  repository: {{url: {REPO}, commit: "{sha_commit}", entrypoint: run}}\n'
            f'  parameters:\n'
            f'    - metric: "{e["id"]}"\n'
            f'      bind: "{bind}"')


def page(doc, sha, version):
    metrics, types, excluded = doc["metrics"], doc["types"], doc["excluded"]
    n_fn = len({m["call"] for m in metrics})
    chips = lambda name, values: "".join(
        f'<button type="button" class="chip" data-filter="{name}" data-value="{escape(v)}" aria-pressed="false">{escape(v)}</button>' for v in values)
    fam_counts = {name: sum(family(m) == name for m in metrics) for _, name, _ in FAMILIES}
    type_rows = "".join(
        f"<tr><td><code>{escape(t)}</code></td><td>{escape(spec.get('description', ''))}</td>"
        f"<td class='mono'>{escape(str(spec['columns']))}</td><td class='mono'>{escape(', '.join(spec['rules']) or '—')}</td></tr>"
        for t, spec in types.items())
    excl_rows = "".join(f"<tr><td><code>{escape(fn)}</code></td><td>{escape(why)}</td></tr>" for fn, why in sorted(excluded.items()))
    pin = commit()
    data = {m["id"]: {"name": m["name"], "family": family(m),
                      "args": [{"arg": a, "type": t, "about": types[t].get("description", "")} for a, t in m["args"].items()],
                      "requires": m.get("requires", {}), "entry": plan_entry(m, pin)} for m in metrics}
    data_json = json.dumps(data).replace("</", "<\\/")
    fam_list = "".join(f"<li><b>{escape(name)}</b> <span class='muted'>({fam_counts[name]})</span> {escape(desc)}</li>" for _, name, desc in FAMILIES)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>omni-sklearn-metrics</title>
<meta name="description" content="{len(metrics)} scikit-learn metrics as a declarative catalog for omnibenchmark.">
<style>
:root {{ --bg:#fafafa; --card:#ffffff; --line:#e4e4e7; --line-hi:#d4d4d8; --fg:#18181b; --muted:#71717a; --soft:#f4f4f5;
  --accent:#2563eb; --up:#15803d; --down:#b45309; color-scheme: light;
  --sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; --mono: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#09090b; --card:#18181b; --line:#27272a; --line-hi:#3f3f46; --fg:#f4f4f5; --muted:#a1a1aa;
  --soft:#27272a; --accent:#60a5fa; --up:#4ade80; --down:#fbbf24; color-scheme: dark; }} }}
* {{ box-sizing: border-box; }}
[hidden] {{ display: none !important; }}  /* .card sets display:flex, which would beat the hidden attribute */
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 var(--sans); }}
.wrap {{ max-width:1180px; margin:0 auto; padding:32px max(16px, env(safe-area-inset-right)) 64px max(16px, env(safe-area-inset-left)); }}
.mono, code, pre {{ font-family:var(--mono); }}
code {{ font-size:.85em; }}
a {{ color:var(--accent); }}
header h1 {{ margin:0; font-size:28px; letter-spacing:-.02em; }}
header p {{ margin:6px 0 0; color:var(--muted); max-width:70ch; }}
.stats {{ display:flex; flex-wrap:wrap; gap:6px 18px; margin:14px 0 0; padding:0; list-style:none; font-size:13px; color:var(--muted); }}
.stats b {{ color:var(--fg); font-variant-numeric:tabular-nums; }}
.links {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:16px; }}
.links a {{ font-size:13px; text-decoration:none; border:1px solid var(--line); border-radius:8px; padding:5px 10px; background:var(--card); color:var(--fg); }}
.links a:hover {{ border-color:var(--line-hi); }}
.filters {{ position:sticky; top:0; z-index:2; background:var(--bg); padding:14px 0 10px; margin-top:22px; border-bottom:1px solid var(--line); display:grid; gap:8px; }}
.filters input {{ width:100%; font:14px var(--sans); padding:8px 12px; border:1px solid var(--line); border-radius:10px; background:var(--card); color:var(--fg); }}
.row {{ display:flex; flex-wrap:wrap; gap:6px; align-items:center; }}
.row .lbl {{ font-size:12px; color:var(--muted); margin-right:4px; min-width:76px; }}
.chip {{ font:12px var(--sans); border:0; border-radius:999px; padding:3px 10px; background:var(--soft); color:var(--fg); cursor:pointer; }}
.chip[aria-pressed="true"] {{ background:var(--fg); color:var(--bg); }}
.count {{ font-size:12px; color:var(--muted); font-variant-numeric:tabular-nums; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(300px, 1fr)); gap:14px; margin-top:16px; }}
.card {{ display:flex; flex-direction:column; gap:8px; border:1px solid var(--line); border-radius:12px; background:var(--card); padding:18px; min-width:0; }}
.card:hover {{ border-color:var(--line-hi); box-shadow:0 1px 2px rgba(0,0,0,.05); }}
.card .top {{ display:flex; justify-content:space-between; align-items:flex-start; gap:10px; }}
.card h3 {{ margin:0; font-size:15px; letter-spacing:-.01em; }}
.badge {{ flex-shrink:0; font-size:12px; border-radius:6px; padding:1px 8px; background:var(--soft); color:var(--muted); }}
.call {{ font-size:12px; text-decoration:none; overflow-wrap:anywhere; }}
.call:hover {{ text-decoration:underline; }}
dl {{ display:grid; grid-template-columns:max-content minmax(0,1fr); gap:4px 12px; margin:2px 0 0; font-size:13px; }}
dt {{ color:var(--muted); }} dd {{ margin:0; min-width:0; overflow-wrap:anywhere; }}
.args {{ list-style:none; margin:0; padding:0; display:grid; gap:2px; }}
.args li {{ display:flex; gap:6px; align-items:baseline; flex-wrap:wrap; }}
.type {{ font:11px var(--sans); border-radius:999px; padding:0 7px; border:1px solid var(--line-hi); color:var(--muted); background:none; }}
button.type {{ cursor:pointer; }}
button.type:hover, button.type[aria-pressed="true"] {{ border-color:var(--fg); color:var(--fg); }}
details summary {{ font-size:12px; color:var(--muted); cursor:pointer; }}
details pre {{ margin:6px 0 0; font-size:12px; background:var(--soft); border-radius:8px; padding:8px 10px; overflow-x:auto; }}
.foot {{ margin-top:auto; padding-top:6px; display:flex; justify-content:space-between; align-items:center; }}
.pill {{ font-size:11px; border-radius:999px; padding:1px 8px; background:var(--soft); color:var(--muted); }}
.dir {{ font-size:14px; font-weight:600; }} .dir.up {{ color:var(--up); }} .dir.down {{ color:var(--down); }}
.empty {{ display:none; color:var(--muted); padding:24px 0; }}
section.more {{ margin-top:40px; }}
section.more h2 {{ font-size:18px; margin:0 0 4px; letter-spacing:-.01em; }}
section.more p {{ color:var(--muted); margin:0 0 12px; max-width:75ch; font-size:14px; }}
.families {{ margin:0; padding-left:18px; font-size:14px; display:grid; gap:4px; }}
.muted {{ color:var(--muted); }}
.tbl {{ overflow-x:auto; border:1px solid var(--line); border-radius:12px; background:var(--card); }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }}
th, td {{ text-align:left; padding:8px 12px; border-bottom:1px solid var(--line); vertical-align:top; }}
tr:last-child td {{ border-bottom:0; }}
th {{ color:var(--muted); font-weight:500; }}
footer {{ margin-top:40px; font-size:12px; color:var(--muted); }}
.foot-r {{ display:flex; align-items:center; gap:10px; }}
.use {{ font:12px var(--sans); border:1px solid var(--line); border-radius:7px; padding:3px 9px; background:var(--card); color:var(--fg); cursor:pointer; }}
.use:hover {{ border-color:var(--line-hi); background:var(--soft); }}
dialog#use {{ width:min(640px, calc(100% - 32px)); max-height:min(86vh, 860px); padding:0; border:1px solid var(--line); border-radius:14px;
  background:var(--card); color:var(--fg); box-shadow:0 24px 60px rgba(0,0,0,.25); }}
dialog#use::backdrop {{ background:rgba(9,9,11,.5); }}
.dlg-head {{ display:flex; align-items:center; gap:10px; padding:16px 20px 12px; border-bottom:1px solid var(--line); position:sticky; top:0; background:var(--card); }}
.dlg-head h2 {{ margin:0; font-size:17px; letter-spacing:-.01em; flex:1; min-width:0; }}
.close {{ border:0; background:none; color:var(--muted); font-size:16px; cursor:pointer; padding:2px 6px; border-radius:6px; }}
.close:hover {{ background:var(--soft); color:var(--fg); }}
.dlg-body {{ padding:14px 20px 20px; font-size:14px; }}
.dlg-body h3 {{ font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); margin:16px 0 6px; font-weight:600; }}
.dlg-intro, .dlg-note {{ margin:0; color:var(--muted); font-size:13px; }}
.dlg-note {{ margin-top:8px; }}
.bind-list {{ list-style:none; margin:0; padding:0; display:grid; gap:8px; }}
.bind-list li {{ display:grid; gap:2px; padding:8px 10px; border:1px solid var(--line); border-radius:9px; }}
.bind-list .h {{ display:flex; gap:8px; align-items:baseline; flex-wrap:wrap; }}
.bind-list .about {{ font-size:12.5px; color:var(--muted); }}
.bind-list .req {{ font-size:12px; color:var(--down); }}
.snip-head {{ display:flex; align-items:center; justify-content:space-between; }}
.snip-head h3 {{ margin-bottom:6px; }}
.copy {{ font:12px var(--sans); border:1px solid var(--line); border-radius:7px; padding:3px 10px; background:var(--card); color:var(--fg); cursor:pointer; }}
.copy:hover {{ border-color:var(--line-hi); }}
pre.snip {{ margin:0; font-size:12.5px; line-height:1.55; background:var(--soft); border-radius:9px; padding:12px 14px; overflow-x:auto; white-space:pre; }}
button:focus-visible, input:focus-visible, a:focus-visible, summary:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
</style></head>
<body><div class="wrap">
<header>
  <h1>omni-sklearn-metrics</h1>
  <p>Every scalar metric in <code>sklearn.metrics</code> as a catalog entry an omnibenchmark plan can run by id: the function, the data type of each argument, fixed settings, direction and range. One module to run them all.</p>
  <ul class="stats"><li><b>{len(metrics)}</b> metrics</li><li><b>{n_fn}</b> sklearn functions</li><li><b>{len(types)}</b> data types</li>
    <li><b>{len(excluded)}</b> functions excluded, with reasons</li><li>scikit-learn <b>{escape(version)}</b></li></ul>
  <div class="links"><a href="{REPO}">Source on GitHub</a><a href="catalog.json">catalog.json</a><a href="#types">Data types</a><a href="#excluded">Not in the catalog</a></div>
</header>

<div class="filters" role="search">
  <input id="q" type="search" placeholder="Search by name, id, function or argument" aria-label="Search metrics">
  <div class="row"><span class="lbl">Family</span>{chips("family", [n for _, n, _ in FAMILIES])}</div>
  <div class="row"><span class="lbl">Reads</span>{chips("types", list(types))}<span class="count" id="count"></span></div>
</div>

<div class="grid" id="grid">
{chr(10).join(card(m, version) for m in metrics)}
</div>
<p class="empty" id="empty">No metric matches. Clear the search or a filter.</p>

<section class="more"><h2>Families</h2><p>A metric's family is the set of data types it reads.</p><ul class="families">{fam_list}</ul></section>

<section class="more" id="types"><h2>Data types</h2>
<p>Every input is a TSV whose first column is <code>cell_id</code>. Inputs are joined on <code>cell_id</code>; the column count is checked before any data is read, and the rules when a file loads.</p>
<div class="tbl"><table><thead><tr><th>Type</th><th>What it holds</th><th>Columns</th><th>Rules</th></tr></thead><tbody>{type_rows}</tbody></table></div></section>

<section class="more" id="excluded"><h2>Not in the catalog</h2>
<p>Every other function in <code>sklearn.metrics</code>, with the reason. The module's tests fail if the catalog and this list stop covering <code>sklearn.metrics</code> exactly.</p>
<div class="tbl"><table><thead><tr><th>Function</th><th>Why it is not an entry</th></tr></thead><tbody>{excl_rows}</tbody></table></div></section>

<dialog id="use" aria-labelledby="use-title">
  <div class="dlg-head"><h2 id="use-title"></h2><span class="badge mono" id="use-id"></span>
    <button type="button" class="close" id="use-close" aria-label="Close">✕</button></div>
  <div class="dlg-body">
    <p class="dlg-intro">Add this module to a stage that gathers the runs you want to score. Replace each
      <code>&lt;input id&gt;</code> in <code>bind</code> with the id of the plan output that feeds that argument.</p>
    <h3>Arguments</h3>
    <ul class="bind-list" id="use-args"></ul>
    <div class="snip-head"><h3>Module entry</h3><button type="button" class="copy" id="use-copy">Copy</button></div>
    <pre class="mono snip" id="use-entry"></pre>
    <h3>Environment</h3>
    <p class="dlg-note">Declare it once per plan, with <a href="{REPO}/blob/main/envs/sklearn.yml">envs/sklearn.yml</a> from this repository:</p>
    <pre class="mono snip">software_environments:
  sklearn: {{conda: envs/sklearn.yml}}</pre>
    <p class="dlg-note">When an argument gets several files from a gather, the module pairs each pipeline's own files by lineage.
      For replicate-to-replicate agreement, bind both arguments to the same input and add <code>pairing: product</code>.
      Pinned to commit <code class="mono">{escape(pin)}</code>, the one this page was built from.</p>
  </div>
</dialog>
<script type="application/json" id="metric-data">{data_json}</script>

<footer>Built from <code>catalog.yaml</code> (sha256 <span class="mono">{sha[:12]}</span>). <a href="{REPO}">btraven00/omni-sklearn-metrics</a></footer>
</div>
<script>
const q = document.getElementById("q"), cards = [...document.querySelectorAll(".card")];
const active = {{ family: null, types: null }};
function apply() {{
  const words = q.value.toLowerCase().split(/\\s+/).filter(Boolean);
  let shown = 0;
  for (const c of cards) {{
    const ok = words.every(w => c.dataset.hay.includes(w))
      && (!active.family || c.dataset.family === active.family)
      && (!active.types || c.dataset.types.split(" ").includes(active.types));
    c.hidden = !ok; shown += ok;
  }}
  document.getElementById("count").textContent = `${{shown}} of ${{cards.length}}`;
  document.getElementById("empty").style.display = shown ? "none" : "block";
}}
// A filter value toggles: choosing the active one again clears it. The data types on the cards
// set the same filter as the "Reads" chips.
function toggle(f, value) {{
  active[f] = active[f] === value ? null : value;
  document.querySelectorAll(`.chip[data-filter="${{f}}"]`).forEach(x => x.setAttribute("aria-pressed", String(x.dataset.value === active[f])));
  if (f === "types") document.querySelectorAll(".card .type").forEach(x => x.setAttribute("aria-pressed", String(x.dataset.type === active.types)));
  apply();
}}
document.querySelectorAll(".chip").forEach(b => b.addEventListener("click", () => toggle(b.dataset.filter, b.dataset.value)));
document.querySelectorAll(".card .type").forEach(b => b.addEventListener("click", () => toggle("types", b.dataset.type)));
q.addEventListener("input", apply);
apply();

// "Use in a plan": one dialog, filled from the metric's data
const DATA = JSON.parse(document.getElementById("metric-data").textContent), dlg = document.getElementById("use");
const esc = s => String(s).replace(/[&<>"]/g, c => ({{ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }}[c]));
document.querySelectorAll(".use").forEach(b => b.addEventListener("click", () => {{
  const m = DATA[b.dataset.use];
  document.getElementById("use-title").textContent = m.name;
  document.getElementById("use-id").textContent = b.dataset.use;
  document.getElementById("use-args").innerHTML = m.args.map(a => `<li>
    <span class="h"><code>${{esc(a.arg)}}</code><span class="type">${{esc(a.type)}}</span></span>
    <span class="about">${{esc(a.about)}}</span>
    ${{m.requires[a.arg] ? `<span class="req">Requires: ${{esc(m.requires[a.arg].join(", "))}}</span>` : ""}}</li>`).join("");
  document.getElementById("use-entry").textContent = m.entry;
  document.getElementById("use-copy").textContent = "Copy";
  dlg.showModal();
}}));
document.getElementById("use-close").addEventListener("click", () => dlg.close());
dlg.addEventListener("click", e => {{ if (e.target === dlg) dlg.close(); }});
document.getElementById("use-copy").addEventListener("click", e => {{
  const text = document.getElementById("use-entry").textContent;
  navigator.clipboard.writeText(text).then(() => {{ e.target.textContent = "Copied"; }}, () => {{
    const r = document.createRange(); r.selectNodeContents(document.getElementById("use-entry"));
    getSelection().removeAllRanges(); getSelection().addRange(r); e.target.textContent = "Press Ctrl+C";
  }});
}});
</script>
</body></html>
"""


def main(out):
    raw = (ROOT / "catalog.yaml").read_bytes()
    doc, sha, version = yaml.safe_load(raw), hashlib.sha256(raw).hexdigest(), sklearn_version()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(page(doc, sha, version))
    module = {"url": REPO, "commit": commit(), "entrypoint": "run", "software_environment": "sklearn",
              "env_file": "envs/sklearn.yml"}  # what a plan pins; tools such as a plan builder read it
    data = {"sha256": sha, "sklearn": version, "module": module, "types": doc["types"], "excluded": doc["excluded"],
            "metrics": [{**m, "family": family(m)} for m in doc["metrics"]]}
    (out / "catalog.json").write_text(json.dumps(data, indent=1) + "\n")
    (out / ".nojekyll").write_text("")
    print(f"{out}/index.html: {len(doc['metrics'])} metrics, {len(doc['excluded'])} excluded")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "_site")
