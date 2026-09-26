"""The demo UI.

    streamlit run frontend/app.py

Built to be read from across a room, in two minutes, by someone who has not
seen it before. Four stops: what we are aiming at, what the agents argued,
what it cost us, where the front moved.

Replays `demo_run.json` by default. The "run live" button exists for a room
that feels safe; the replay exists for every other room.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from core.contract import AXES
from core.loop import HALOPERIDOL, run
from frontend.plots import AXIS_LABELS, molecule_png, pareto_figure

DEFAULT_RUN = Path("demo_run.json")

INK = "#1d3557"
RED = "#c1121f"
DEEP_RED = "#7f0000"
GREEN = "#2d6a4f"
GREY = "#6c757d"

CSS = """
<style>
  .block-container {padding-top: 2rem; max-width: 1400px;}
  h1 {font-size: 2.3rem !important; letter-spacing: -0.02em;}
  .kicker {font-size:0.72rem; letter-spacing:0.14em; color:#6c757d;
           text-transform:uppercase; margin-bottom:0.15rem;}
  .card {border:1px solid rgba(0,0,0,0.10); border-radius:10px;
         padding:0.9rem 1.1rem; background:rgba(0,0,0,0.015);}
  .big {font-size:2.0rem; font-weight:700; line-height:1.1;}
  .smiles {font-family:ui-monospace,Menlo,monospace; font-size:0.74rem;
           color:#6c757d; word-break:break-all;}
  .agentbox {border-left:5px solid; padding:0.55rem 0.9rem; margin-bottom:0.55rem;
             border-radius:0 6px 6px 0;}
  .agentname {font-weight:700; font-size:0.95rem; letter-spacing:0.02em;}
  .verdict {font-size:0.7rem; letter-spacing:0.1em; font-weight:700;
            padding:0.1rem 0.45rem; border-radius:3px; margin-left:0.5rem;}
</style>
"""


def load_run(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def molecule_image(smiles: str, size=(420, 300)):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        return str(molecule_png(smiles, Path(handle.name), size=size))


def delta_arrow(value: float) -> str:
    if value > 0.005:
        return f"<span style='color:{GREEN}'>&#9650; {value:+.2f}</span>"
    if value < -0.005:
        return f"<span style='color:{RED}'>&#9660; {value:+.2f}</span>"
    return f"<span style='color:{GREY}'>&#8212; 0.00</span>"


def bar(label: str, value: float, before: float, highlight: str = INK) -> str:
    """One axis as a labelled bar with the seed value ghosted behind it."""
    return f"""
<div style='margin-bottom:0.55rem'>
  <div style='display:flex;justify-content:space-between;font-size:0.82rem'>
    <span style='font-weight:600'>{label}</span>
    <span><b>{value:.2f}</b> &nbsp;{delta_arrow(value - before)}</span>
  </div>
  <div style='position:relative;height:13px;background:rgba(0,0,0,0.06);
              border-radius:7px;margin-top:0.2rem'>
    <div style='position:absolute;height:13px;width:{before*100:.1f}%;
                background:rgba(0,0,0,0.16);border-radius:7px'></div>
    <div style='position:absolute;height:13px;width:{value*100:.1f}%;
                background:{highlight};border-radius:7px;opacity:0.9'></div>
  </div>
</div>"""


# --------------------------------------------------------------------------
# 1. What we are aiming at
# --------------------------------------------------------------------------

def section_target(run_record: dict, seed: dict, final: dict) -> None:
    st.markdown(
        f"<div class='kicker'>Target</div>"
        f"<div class='big' style='color:{INK}'>DRD2 &mdash; dopamine D2 receptor</div>"
        f"<div style='color:#495057;margin-bottom:0.9rem'>"
        f"The receptor every antipsychotic engages. "
        f"<b>On-target:</b> keep binding it. "
        f"<b>Anti-target:</b> stop binding the hERG cardiac channel."
        f"</div>",
        unsafe_allow_html=True,
    )

    left, mid, right = st.columns([1, 1, 1.15])
    with left:
        st.markdown("<div class='kicker'>Start &mdash; haloperidol</div>",
                    unsafe_allow_html=True)
        st.image(molecule_image(run_record["seed"]))
    with mid:
        st.markdown(f"<div class='kicker'>After {run_record['rounds']} rounds</div>",
                    unsafe_allow_html=True)
        st.image(molecule_image(run_record["final"]["smiles"]))
    with right:
        st.markdown("<div class='kicker'>Grey = where we started</div>",
                    unsafe_allow_html=True)
        st.markdown(
            bar("DRD2 binding", final["affinity"], seed["affinity"], INK)
            + bar("hERG avoidance", final["herg"], seed["herg"], DEEP_RED)
            + bar("Solubility", final["solubility"], seed["solubility"], "#457b9d")
            + bar("Brain penetration", final["bbb"], seed["bbb"], "#457b9d")
            + bar("Synthesisability", final["sa"], seed["sa"], GREY),
            unsafe_allow_html=True,
        )

    headline = (
        f"Held DRD2 binding at {final['affinity']:.2f} "
        f"while moving hERG avoidance {seed['herg']:.2f} → {final['herg']:.2f} "
        f"and solubility {seed['solubility']:.2f} → {final['solubility']:.2f}."
    )
    if run_record.get("affinity_held", True):
        st.success(headline)
    else:
        st.error(
            f"DRD2 binding fell to {final['affinity']:.2f}, below the "
            f"{run_record.get('affinity_floor', 0):.2f} floor. The molecule drifted "
            f"off its target — that is not an optimisation."
        )


# --------------------------------------------------------------------------
# 2. What the agents argued  (the money shot)
# --------------------------------------------------------------------------

STYLE = {
    "support": (GREEN, "rgba(45,106,79,0.06)", "SUPPORTS"),
    "object": (RED, "rgba(193,18,31,0.07)", "OBJECTS"),
    "veto": (DEEP_RED, "rgba(127,0,0,0.11)", "VETO"),
}


def section_transcript(record: dict) -> None:
    decision = record["decision"]

    for review in record["reviews"]:
        colour, background, label = STYLE.get(review["verdict"], (GREY, "none", "?"))
        weight = "700" if review["verdict"] == "veto" else "400"
        st.markdown(
            f"<div class='agentbox' style='border-color:{colour};background:{background}'>"
            f"<span class='agentname' style='color:{colour}'>{review['agent']}</span>"
            f"<span class='verdict' style='background:{colour};color:white'>{label}</span>"
            f"<div style='color:{colour};font-weight:{weight};margin-top:0.3rem;"
            f"font-size:0.93rem;line-height:1.45'>{review['reason']}</div></div>",
            unsafe_allow_html=True,
        )

    if decision["vetoed"]:
        st.markdown(
            f"<div class='agentbox' style='border-color:{DEEP_RED};"
            f"background:rgba(127,0,0,0.11)'>"
            f"<span class='agentname' style='color:{DEEP_RED}'>"
            f"REMOVED FROM CONSIDERATION</span>"
            + "".join(
                f"<div class='smiles' style='color:{DEEP_RED}'>{s}</div>"
                for s in decision["vetoed"]
            )
            + "</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<div class='agentbox' style='border-color:{INK};background:rgba(29,53,87,0.07)'>"
        f"<span class='agentname' style='color:{INK}'>Orchestrator &mdash; decides</span>"
        f"<div style='margin-top:0.3rem;font-size:0.93rem;line-height:1.45'>"
        f"{decision['rationale']}</div></div>",
        unsafe_allow_html=True,
    )

    if decision.get("forced_second_best"):
        st.warning(
            "**The orchestrator took second best.** The highest-scoring molecule "
            "this round was vetoed on hERG and is off the table regardless of its "
            "potency."
        )
    if (decision.get("audit") or {}).get("contradiction"):
        st.error(
            "**Audit flag:** the rationale claims a hERG advantage the chosen "
            "molecule does not have. Shown rather than hidden — an orchestrator "
            "that can reason can also be fluently wrong."
        )


def section_candidates(record: dict) -> None:
    rows = []
    for candidate in record["candidates"]:
        norm = candidate["normalised"]
        vetoed = candidate["smiles"] in record["decision"]["vetoed"]
        rows.append(
            {
                # Short markers: the column is narrow and "VETOED"/"CHOSEN"
                # were rendering truncated mid-word.
                "": "✗" if vetoed
                else ("✓" if candidate["smiles"] == record["decision"]["chosen"]
                      else ""),
                "edit": candidate.get("transform", ""),
                **{AXIS_LABELS[a]: round(norm[a], 2) for a in AXES},
                "SMILES": candidate["smiles"],
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------
# 3 + 4. Where the front moved, and what this isn't
# --------------------------------------------------------------------------

def section_front(run_record: dict) -> None:
    left, right = st.columns([1.3, 1])
    with left:
        st.pyplot(pareto_figure(run_record))
    with right:
        initial = run_record["hypervolume_initial"]
        final = run_record["hypervolume_final"]
        st.markdown(
            f"<div class='card'><div class='kicker'>Pareto hypervolume</div>"
            f"<div class='big' style='color:{INK}'>{initial:.2f} &rarr; {final:.2f}</div>"
            f"<div style='color:{GREY}'>"
            f"{final / initial:.1f}&times; more of the solubility / hERG-safety "
            f"space dominated.</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.markdown(
            "**Not plotted: affinity.** Haloperidol is a marketed drug and scores "
            "exactly **1.00** on the DRD2 oracle — the axis has no headroom, so a "
            "front drawn on it cannot move. It is held as a floor instead."
        )
        prov = run_record["provenance"]
        if prov["any_surrogate"]:
            st.error(
                "**Surrogate oracles in use.** Some axes are physchem heuristics, "
                "not trained models. No accuracy claim attaches to them."
            )
        else:
            st.success(
                f"All five axes are real models — affinity `{prov['affinity']}`, "
                f"ADMET `{prov['solubility']}` (XGBoost on TDC scaffold splits), "
                f"SA `{prov['sa']}`."
            )


def section_caveats() -> None:
    st.markdown(
        """
**What this isn't.** The ADMET models were trained on scaffold splits, so they
were tested on novel chemistry &mdash; but our agents generate molecules further out
of distribution than that. *The predictions get less reliable exactly as the
optimisation gets more interesting*, and nothing in this loop knows when it has
walked off the training manifold.

We score one on-target and one anti-target. That is selectivity against the
off-target that actually ends antipsychotic programmes, but it is **not subtype
selectivity** &mdash; D3, D4 and 5-HT2A are unscored, so nothing here shows we hit
D2 *rather than* its close relatives.

The real version closes the loop on in vitro assay data, and the hard problem
becomes sample efficiency when every data point costs a few hundred dollars and
four weeks.
"""
    )


# --------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="DRD2 lead optimisation", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.header("Demo")
        path = Path(st.text_input("Cached run", str(DEFAULT_RUN)))
        rounds = st.slider("Rounds (live only)", 1, 8, 5)
        st.caption(
            "Replay is the default and costs nothing. Press *Run live* only if the "
            "room feels safe — it calls the oracles and the agent LLM."
        )
        live = st.button("Run live", type="primary")
        st.divider()
        st.caption(
            "**Demo order** — 1 the target, 2 the argument, 3 the front. "
            "Read one veto out loud."
        )

    if live:
        with st.spinner("Running the loop..."):
            run_record = run(seed=HALOPERIDOL, rounds=rounds, verbose=False)
        path.write_text(json.dumps(run_record, indent=2))
        st.session_state["run"] = run_record
    elif "run" not in st.session_state:
        loaded = load_run(path)
        if loaded is None:
            st.warning(
                f"No cached run at `{path}`. Generate one with "
                "`python -m core.loop --out demo_run.json`, or press *Run live*."
            )
            st.stop()
        st.session_state["run"] = loaded

    run_record = st.session_state["run"]
    if not run_record.get("round_records"):
        st.warning("This run has no completed rounds.")
        st.stop()

    seed = run_record["history"][0]["normalised"]
    final = run_record["final"]["normalised"]

    st.title("Multi-agent lead optimisation")
    section_target(run_record, seed, final)

    st.divider()
    st.markdown(
        f"<div class='kicker'>The argument</div>"
        f"<div style='color:#495057;margin-bottom:0.7rem'>Three specialists, one "
        f"objective each. Safety does not negotiate — it removes molecules.</div>",
        unsafe_allow_html=True,
    )
    numbers = [r["round"] for r in run_record["round_records"]]
    veto_rounds = [
        r["round"] for r in run_record["round_records"] if r["decision"]["vetoed"]
    ]
    default_round = veto_rounds[0] if veto_rounds else numbers[-1]
    selected = st.select_slider(
        "Round" + (f"  (vetoes fired in {veto_rounds})" if veto_rounds else ""),
        options=numbers,
        value=default_round,
    )
    record = run_record["round_records"][selected - 1]

    left, right = st.columns([1.15, 1])
    with left:
        section_transcript(record)
    with right:
        st.markdown("<div class='kicker'>Candidates this round</div>",
                    unsafe_allow_html=True)
        section_candidates(record)

    st.divider()
    st.markdown("<div class='kicker'>Did the front move?</div>",
                unsafe_allow_html=True)
    section_front(run_record)

    st.divider()
    with st.expander("What this isn't  —  read this part out loud", expanded=False):
        section_caveats()


if __name__ == "__main__":
    main()
