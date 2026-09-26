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


def target_panel(run_record: dict, seed_norm: dict, final_norm: dict) -> None:
    """The header that says what this run is for, before any number appears.

    Without this the app is five anonymous bars. A viewer should be able to see,
    in one glance and before scrolling: which receptor we are aiming at, which
    molecule we started from, and what we are trying to change about it.
    """
    left, right = st.columns([1, 2])
    with left:
        st.image(molecule_image(run_record["seed"]), caption="seed: haloperidol")
    with right:
        st.markdown(
            f"""
<div style='border:1px solid rgba(29,53,87,0.25);border-radius:8px;
padding:0.9rem 1.1rem;background:rgba(29,53,87,0.04)'>
<div style='font-size:0.78rem;letter-spacing:0.09em;color:#6c757d'>TARGET</div>
<div style='font-size:1.5rem;font-weight:700;color:#1d3557;line-height:1.2'>
DRD2 &mdash; dopamine D2 receptor</div>
<div style='color:#495057;margin-top:0.2rem'>
<code>DRD2</code> / UniProt <code>P14416</code> &middot; class A GPCR &middot;
the receptor every antipsychotic engages</div>
<hr style='margin:0.7rem 0;border:none;border-top:1px solid rgba(0,0,0,0.08)'>
<div style='font-size:0.78rem;letter-spacing:0.09em;color:#6c757d'>OBJECTIVE</div>
<div style='color:#212529'>
Start from <b>haloperidol</b> &mdash; a marketed antipsychotic that binds DRD2
potently but is poorly soluble and carries a <b>QT-prolongation liability</b>.
Edit the molecule to <b>keep DRD2 engagement</b> while <b>pulling it away from
the hERG channel</b> and making it developable.</div>
</div>
""",
            unsafe_allow_html=True,
        )
        delta = final_norm["affinity"] - seed_norm["affinity"]
        floor = run_record.get("affinity_floor", 0.0)
        if run_record.get("affinity_held", True):
            st.success(
                f"DRD2 engagement held: {seed_norm['affinity']:.2f} → "
                f"{final_norm['affinity']:.2f} (floor {floor:.2f}). "
                f"hERG avoidance {seed_norm['herg']:.2f} → {final_norm['herg']:.2f}."
            )
        else:
            st.error(
                f"DRD2 engagement fell to {final_norm['affinity']:.2f}, below the "
                f"{floor:.2f} floor ({delta:+.2f}). The molecule drifted off its "
                f"target — that is not an optimisation."
            )

    st.info(
        "**A note on what 'specificity' means here.** We score DRD2 engagement and "
        "hERG avoidance — one on-target, one anti-target. That is selectivity "
        "against the off-target that actually ends antipsychotic programmes, but it "
        "is not full subtype selectivity: we do **not** yet score D3, D4 or 5-HT2A, "
        "so nothing here shows we are hitting D2 *rather than* its close relatives. "
        "Adding a D3 anti-target is the obvious next axis."
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

    target_panel(run_record, seed_norm, final_norm)

    # Grouped by what each axis is FOR, not just listed. An undifferentiated row
    # of five bars does not tell anyone that this is a receptor-engagement
    # problem with an anti-target attached.
    st.markdown("##### On-target — what we are trying to hit")
    on_target = st.columns([1, 3])
    on_target[0].metric(
        "DRD2 engagement",
        f"{final_norm['affinity']:.2f}",
        f"{final_norm['affinity'] - seed_norm['affinity']:+.2f}",
    )
    on_target[1].caption(
        "Dopamine D2 receptor — the antipsychotic target. Haloperidol binds deep "
        "in an extended pocket formed by TM3/TM5/TM6, anchored by a salt bridge "
        "from its protonated piperidine nitrogen to **Asp114**. That buried "
        "butyrophenone is the subtype-selectivity element, not decoration. "
        "Every edit is scored on whether it keeps this engagement."
    )

    st.markdown("##### Anti-target — what we must not hit")
    anti = st.columns([1, 3])
    anti[0].metric(
        "hERG avoidance",
        f"{final_norm['herg']:.2f}",
        f"{final_norm['herg'] - seed_norm['herg']:+.2f}",
    )
    anti[1].caption(
        "The hERG cardiac potassium channel. **This is the selectivity problem in "
        "miniature:** hERG binds a protonated basic nitrogen plus lipophilic "
        "aromatics via cation-π to **Tyr652** — nearly the same pharmacophore "
        "DRD2 wants. Holding one while dropping the other is the actual "
        "optimisation. The safety agent's veto is a selectivity constraint."
    )

    st.markdown("##### Developability — whether it could ever be a drug")
    dev = st.columns(3)
    for column, axis in zip(dev, ("solubility", "bbb", "sa")):
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
