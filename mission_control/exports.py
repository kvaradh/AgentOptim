"""Serialize a mission to portable evidence-package formats."""

from __future__ import annotations

import csv
from html import escape
import io
import json
from urllib.parse import urlsplit


def mission_json(mission: dict) -> str:
    from .analysis import validate_mission

    validate_mission(mission)
    return json.dumps(mission, indent=2, ensure_ascii=False) + "\n"


def candidate_csv(mission: dict) -> str:
    from .analysis import export_rows, validate_mission

    validate_mission(mission)
    rows = export_rows(mission["rounds"])
    if not rows:
        return ""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def decision_receipt_html(mission: dict) -> str:
    from .analysis import validate_mission
    from .chemistry import depict_svg

    validate_mission(mission)
    intake = mission["intake"]
    summary = mission["summary"]
    seed = mission["seed"]
    final_id = summary["final_candidate_id"]
    final_round = mission["rounds"][-1]
    final = next(
        candidate
        for candidate in final_round["candidates"]
        if candidate["candidate_id"] == final_id
    )
    seed_depiction = _receipt_depiction(
        depict_svg(seed["smiles"], width=420, height=260)
    )
    final_depiction = _receipt_depiction(
        depict_svg(final["smiles"], width=420, height=260)
    )
    round_rows = "".join(
        _round_receipt(round_value, ledger)
        for round_value, ledger in zip(
            mission["rounds"], mission["decision_ledger"]
        )
    )
    wet_lab_rows = "".join(
        (
            "<tr>"
            f"<td>{step['priority']}</td>"
            f"<td>{escape(step['property'])}</td>"
            f"<td>{escape(step['assay'])}</td>"
            f"<td>{escape(step['decision_question'])}</td>"
            "</tr>"
        )
        for step in mission["wet_lab_plan"]
    )
    source_rows = "".join(
        _source_receipt(source) for source in mission["evidence_library"]
    )
    disclaimers = "".join(
        f"<li>{escape(disclaimer)}</li>" for disclaimer in mission["disclaimers"]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agent Perry Decision Receipt · {escape(final['name'])}</title>
<style>
:root {{ color-scheme: light; --ink:#101827; --muted:#657082; --line:#d9dfeb;
--paper:#f7f5ef; --card:#fff; --blue:#315fdd; --safe:#197a55; --risk:#b94049; }}
* {{ box-sizing:border-box; }} body {{ margin:0; color:var(--ink); background:var(--paper);
font:15px/1.5 Inter,ui-sans-serif,system-ui,sans-serif; }}
main {{ width:min(1120px,calc(100% - 32px)); margin:32px auto 80px; }}
header,.card {{ background:var(--card); border:1px solid var(--line); border-radius:24px;
padding:28px; margin-bottom:18px; }} h1 {{ font-size:clamp(34px,6vw,68px);
line-height:.96; letter-spacing:-.055em; max-width:900px; margin:12px 0 18px; }}
h2 {{ margin:0 0 12px; }} .eyebrow {{ color:var(--blue); text-transform:uppercase;
letter-spacing:.13em; font-size:12px; font-weight:800; }} .grid {{ display:grid;
grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; }}
.metric {{ border:1px solid var(--line); border-radius:16px; padding:16px; }}
.metric strong {{ display:block; font-size:26px; }} code {{ overflow-wrap:anywhere; }}
.structure-comparison {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
.structure-figure {{ margin:0; border:1px solid var(--line); border-radius:20px;
overflow:hidden; background:#fff; }}
.structure-figure figcaption {{ padding:18px 20px; border-top:1px solid var(--line); }}
.structure-figure figcaption span,.structure-figure figcaption strong,
.structure-figure figcaption code {{ display:block; }}
.structure-figure figcaption span {{ color:var(--blue); font-size:11px;
font-weight:800; letter-spacing:.11em; text-transform:uppercase; }}
.structure-figure figcaption strong {{ margin:4px 0; font-size:19px; }}
.structure-figure figcaption code {{ color:var(--muted); font-size:11px; }}
.structure-canvas {{ display:grid; place-items:center; min-height:260px; padding:10px; }}
.structure-canvas svg {{ width:100%; height:auto; max-height:260px; }}
table {{ width:100%; border-collapse:collapse; }} th,td {{ padding:10px;
border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
th {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em; }}
.safe {{ color:var(--safe); }} .risk {{ color:var(--risk); }} .sources li {{ margin:12px 0; }}
.sources span {{ display:block; color:var(--muted); }}
@media(max-width:640px) {{ header,.card {{ padding:20px; border-radius:18px; }}
.structure-comparison {{ grid-template-columns:1fr; }}
table {{ display:block; overflow:auto; }} }}
</style>
</head>
<body>
<main>
<header>
<div class="eyebrow">Auditable decision receipt · {escape(mission['digest'][:12])}</div>
<h1>{escape(final['name'])} advances for {escape(mission['target']['id'])} validation.</h1>
<p>{escape(intake['optimization_hypothesis'])}</p>
<code>{escape(final['smiles'])}</code>
</header>
<section class="card">
<div class="eyebrow">Structure comparison · RDKit 2D</div>
<h2>Starting molecule and proposed lead</h2>
<div class="structure-comparison">
<figure class="structure-figure">
<div class="structure-canvas">{seed_depiction}</div>
<figcaption>
<span>Starting molecule</span>
<strong>{escape(intake['name'])}</strong>
<code>{escape(seed['smiles'])}</code>
</figcaption>
</figure>
<figure class="structure-figure">
<div class="structure-canvas">{final_depiction}</div>
<figcaption>
<span>Proposed molecule · {escape(final_id)}</span>
<strong>{escape(final['name'])}</strong>
<code>{escape(final['smiles'])}</code>
</figcaption>
</figure>
</div>
</section>
<section class="card">
<div class="eyebrow">Mission brief</div>
<h2>Why this target and this seed</h2>
<p><strong>Target.</strong> {escape(mission['target']['name'])} ({escape(mission['target']['id'])}). {escape(intake['therapeutic_intent'])}</p>
<p><strong>Seed rationale.</strong> {escape(intake['seed_rationale'])}</p>
<p><strong>Optimization hypothesis.</strong> {escape(intake['optimization_hypothesis'])}</p>
</section>
<section class="card">
<div class="eyebrow">Outcome</div>
<div class="grid">
<div class="metric"><strong>{summary['candidate_occurrences']}</strong>candidate states</div>
<div class="metric"><strong>{summary['vetoed_occurrences']}</strong>hERG vetoes</div>
<div class="metric"><strong>{summary['unique_molecules']}</strong>unique molecules</div>
<div class="metric"><strong>{final['similarity_seed']:.3f}</strong>seed locality</div>
</div>
</section>
<section class="card">
<div class="eyebrow">Recorded policy</div>
<h2>Five-round decision trail</h2>
<table><thead><tr><th>Round</th><th>Winner</th><th>Gate</th><th>Why</th></tr></thead>
<tbody>{round_rows}</tbody></table>
</section>
<section class="card">
<div class="eyebrow">Next evidence</div>
<h2>Wet-lab validation runway</h2>
<table><thead><tr><th>Priority</th><th>Property</th><th>Assay</th><th>Decision question</th></tr></thead>
<tbody>{wet_lab_rows}</tbody></table>
</section>
<section class="card sources">
<div class="eyebrow">Interpretation sources</div>
<h2>Evidence library</h2><ul>{source_rows}</ul>
</section>
<section class="card"><div class="eyebrow">Knowledge limits</div><ul>{disclaimers}</ul></section>
</main>
</body>
</html>
"""


def _receipt_depiction(svg: str) -> str:
    """Return the RDKit SVG element without its document-level XML declaration."""

    start = svg.find("<svg")
    if start < 0:
        raise ValueError("RDKit depiction is missing its SVG element")
    return svg[start:]


def _round_receipt(round_value: dict, ledger: dict) -> str:
    winner = next(
        candidate
        for candidate in round_value["candidates"]
        if candidate["candidate_id"] == round_value["winner_id"]
    )
    reason = ledger["why_chosen"][0]
    return (
        "<tr>"
        f"<td>R{round_value['round']}</td>"
        f"<td><strong>{escape(winner['name'])}</strong><br>"
        f"<code>{escape(winner['candidate_id'])}</code></td>"
        f"<td class=\"safe\">Clears hERG gate<br>{round_value['veto_count']} vetoed</td>"
        f"<td>{escape(reason)}</td>"
        "</tr>"
    )


def _source_receipt(source: dict) -> str:
    title = escape(source["title"])
    url = _safe_http_url(source.get("url"))
    title_markup = (
        f'<a href="{escape(url, quote=True)}" rel="noopener noreferrer">{title}</a>'
        if url
        else title
    )
    return (
        "<li>"
        f"{title_markup}"
        f"<span>{escape(source['product_effect'])}</span>"
        "</li>"
    )


def _safe_http_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return value
