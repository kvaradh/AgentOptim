"""The demo UI.

    streamlit run frontend/app.py

Replays `demo_run.json` by default. The "run live" button is there for a room
that feels safe; the replay is there for every other room. A timed-out API call
in front of judges costs the demo, and a replay costs nothing.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from core.contract import AXES
from core.loop import HALOPERIDOL, run
from frontend.plots import AXIS_LABELS, molecule_png, pareto_figure, score_bars_figure

DEFAULT_RUN = Path("demo_run.json")

VERDICT_STYLE = {
    "support": ("#2d6a4f", "", ""),
    "object": ("#c1121f", "", ""),
    "veto": ("#7f0000", "**", "**"),
}


def load_run(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def molecule_image(smiles: str):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        return str(molecule_png(smiles, Path(handle.name)))


def provenance_banner(run_record: dict) -> None:
    prov = run_record["provenance"]
    if prov["any_surrogate"]:
        surrogate_axes = [
            axis for axis in AXES
            if "surrogate" in str(prov.get(axis, "")).lower()
        ]
        st.error(
            "**Surrogate oracles in use.** These axes are physchem heuristics, not "
            f"trained models: {', '.join(surrogate_axes)}. Directionally sane, not "
            "calibrated, and carrying no accuracy claim. Do not attach a published "
            "ROC-AUC to anything on this screen."
        )
    else:
        st.success(
            f"Oracles: affinity `{prov['affinity']}`, ADMET `{prov['solubility']}`, "
            f"SA `{prov['sa']}`."
        )


def render_transcript(record: dict) -> None:
    """The agent argument. Objections in red, vetoes in bold red."""
    decision = record["decision"]
    for review in record["reviews"]:
        colour, open_mark, close_mark = VERDICT_STYLE.get(
            review["verdict"], ("#333333", "", "")
        )
        st.markdown(
            f"<div style='border-left:4px solid {colour};padding:0.4rem 0.8rem;"
            f"margin-bottom:0.5rem;background:rgba(0,0,0,0.02)'>"
            f"<span style='color:{colour};font-weight:600'>"
            f"{review['agent']} &middot; {review['verdict'].upper()}</span><br>"
            f"<span style='color:{colour}'>{open_mark}{review['reason']}{close_mark}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    if decision["vetoed"]:
        st.markdown(
            f"<div style='border-left:4px solid #7f0000;padding:0.4rem 0.8rem;"
            f"margin-bottom:0.5rem;background:rgba(193,18,31,0.06)'>"
            f"<b style='color:#7f0000'>REMOVED FROM CONSIDERATION</b><br>"
            + "<br>".join(f"<code>{s}</code>" for s in decision["vetoed"])
            + "</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<div style='border-left:4px solid #1d3557;padding:0.4rem 0.8rem;"
        f"background:rgba(29,53,87,0.06)'>"
        f"<b style='color:#1d3557'>Orchestrator</b><br>{decision['rationale']}</div>",
        unsafe_allow_html=True,
    )
    if decision.get("forced_second_best"):
        st.warning(
            "The orchestrator took second best: the leading candidate was vetoed on hERG."
        )


def main() -> None:
    st.set_page_config(page_title="DRD2 multi-agent lead optimisation", layout="wide")
    st.title("Multi-agent lead optimisation — DRD2")

    with st.sidebar:
        st.header("Run")
        path = Path(st.text_input("Cached run", str(DEFAULT_RUN)))
        rounds = st.slider("Rounds (live only)", 1, 8, 5)
        st.caption(
            "Replay is the default. Press *Run live* only if the room feels safe "
            "— it calls the oracles, and the agent LLM if a key is set."
        )
        live = st.button("Run live", type="primary")

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
    provenance_banner(run_record)

    seed_norm = run_record["history"][0]["normalised"]
    final_norm = run_record["final"]["normalised"]
    columns = st.columns(5)
    for column, axis in zip(columns, AXES):
        column.metric(
            AXIS_LABELS[axis],
            f"{final_norm[axis]:.2f}",
            f"{final_norm[axis] - seed_norm[axis]:+.2f}",
        )
    if not run_record.get("affinity_held", True):
        st.error(
            f"Affinity fell below the programme floor of "
            f"{run_record['affinity_floor']:.2f}. The front moved by spending the "
            f"potency that made the seed a lead — that is not an optimisation."
        )

    round_numbers = [record["round"] for record in run_record["round_records"]]
    if not round_numbers:
        st.warning("This run has no completed rounds.")
        st.stop()
    selected = st.select_slider("Round", options=round_numbers, value=round_numbers[-1])
    record = run_record["round_records"][selected - 1]

    left, right = st.columns([1, 1])
    with left:
        st.subheader(f"Round {selected}")
        st.caption("Incumbent entering this round")
        st.image(molecule_image(record["parent"]["smiles"]))
        st.code(record["parent"]["smiles"], language=None)
        previous = (
            run_record["round_records"][selected - 2]["parent"] if selected > 1 else None
        )
        st.pyplot(score_bars_figure(record["parent"], previous))

    with right:
        st.subheader("Transcript")
        render_transcript(record)

    st.subheader("Candidates this round")
    rows = []
    for candidate in record["candidates"]:
        norm = candidate["normalised"]
        rows.append(
            {
                "vetoed": candidate["smiles"] in record["decision"]["vetoed"],
                "chosen": candidate["smiles"] == record["decision"]["chosen"],
                "edit": candidate.get("transform", ""),
                **{AXIS_LABELS[axis]: round(norm[axis], 3) for axis in AXES},
                "smiles": candidate["smiles"],
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.subheader("Pareto front")
    st.pyplot(pareto_figure(run_record))
    st.caption(
        f"Hypervolume over {' x '.join(run_record['pareto_axes'])}: "
        f"{run_record['hypervolume_initial']:.3f} → "
        f"{run_record['hypervolume_final']:.3f}. "
        f"Affinity {run_record['affinity_seed']:.2f} → "
        f"{run_record['affinity_final']:.2f} "
        f"(floor {run_record['affinity_floor']:.2f}). "
        "Not plotting affinity is deliberate: the seed is a marketed drug and already "
        "sits at the ceiling of the activity oracle, so that axis has no headroom and "
        "a front drawn on it cannot move. It is held as a constraint instead."
    )


if __name__ == "__main__":
    main()
