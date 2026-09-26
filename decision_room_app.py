"""Streamlit shell for the offline-first Agent Perry workspace."""

import streamlit as st

from core.contract import propose
from core.errors import OptimizationHalt, UnsupportedTargetError
from core.llm import client_from_env
from core.runner import load_run, run_optimization
from mission_control.analysis import compile_mission
from mission_control.intake import (
    CompoundResolutionError,
    catalog_names,
    resolve_compound,
)
from mission_control.presentation import build_workspace_html


st.set_page_config(
    page_title="Agent Perry",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
      --mc-ink: #152033;
      --mc-paper: #f4f1e9;
      --mc-line: #dcd8ce;
    }
    [data-testid="stAppViewContainer"] {
      background:
        radial-gradient(circle at 78% -10%, rgba(118, 158, 229, .20), transparent 34rem),
        var(--mc-paper);
      color: var(--mc-ink);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] {
      width: min(1560px, calc(100% - 28px));
      max-width: 1560px;
      padding: 1rem 0 3rem;
    }
    [data-testid="stMainBlockContainer"] h1 {
      color: var(--mc-ink);
      letter-spacing: -.045em;
      font-size: clamp(2rem, 4vw, 3.6rem) !important;
      margin-bottom: .1rem !important;
    }
    [data-testid="stCaptionContainer"] { color: #667085; }
    [data-testid="stWidgetLabel"] p,
    [data-testid="stCheckbox"] p {
      color: var(--mc-ink) !important;
    }
    [data-testid="stForm"] {
      background: rgba(255, 255, 255, .78);
      border: 1px solid var(--mc-line);
      border-radius: 22px;
      box-shadow: 0 18px 55px rgba(41, 52, 76, .07);
      padding: 1rem 1.05rem .55rem;
      backdrop-filter: blur(18px);
    }
    [data-testid="stTextInputRootElement"] {
      background: #fff !important;
      border-color: var(--mc-line) !important;
      border-radius: 12px;
    }
    [data-testid="stTextInputRootElement"] input {
      color: var(--mc-ink) !important;
    }
    .stButton > button, .stFormSubmitButton > button {
      border-radius: 999px;
      color: #fff !important;
      min-height: 2.6rem;
      font-weight: 760;
    }
    .stButton > button *,
    .stFormSubmitButton > button * {
      color: #fff !important;
    }
    .stFormSubmitButton > button[kind="primary"] {
      background: var(--mc-ink);
      border-color: var(--mc-ink);
    }
    .stButton > button[kind="primary"],
    button[data-testid="stBaseButton-primary"] {
      background: var(--mc-ink) !important;
      border-color: var(--mc-ink) !important;
      color: #fff !important;
    }
    iframe {
      border: 0 !important;
      border-radius: 26px;
      box-shadow: 0 28px 90px rgba(35, 48, 77, .11);
      background: #f8f6f0;
    }
    @media (max-width: 640px) {
      [data-testid="stMainBlockContainer"] {
        width: min(100% - 16px, 1560px);
        padding-top: .5rem;
      }
      [data-testid="stForm"] {
        border-radius: 18px;
        padding: .8rem .8rem .4rem;
      }
      iframe { border-radius: 18px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_mission() -> dict:
    return compile_mission(load_run(), resolve_compound("Haloperidol", target="DRD2"))


@st.cache_data(show_spinner=False)
def resolve_intake(query: str, target: str, use_pubchem: bool) -> dict:
    return resolve_compound(query, target=target, use_pubchem=use_pubchem)


def initialize_state() -> None:
    if "mission" not in st.session_state:
        st.session_state["mission"] = cached_mission()
    if "resolved_intake" not in st.session_state:
        st.session_state["resolved_intake"] = resolve_compound(
            "Haloperidol",
            target="DRD2",
        )
    if "mission_source" not in st.session_state:
        st.session_state["mission_source"] = "Bundled offline replay"
    if "query" not in st.session_state:
        st.session_state["query"] = "Haloperidol"
    if "target_query" not in st.session_state:
        st.session_state["target_query"] = "DRD2"


def load_cached_demo(*, reset_query: bool = False) -> None:
    st.session_state["mission"] = cached_mission()
    st.session_state["resolved_intake"] = resolve_compound(
        "Haloperidol",
        target="DRD2",
    )
    st.session_state["mission_source"] = "Bundled offline replay"
    if reset_query:
        st.session_state["query"] = "Haloperidol"
        st.session_state["target_query"] = "DRD2"


def run_live_mission(intake: dict) -> None:
    if intake["target"] != "DRD2":
        raise UnsupportedTargetError(
            f"No {intake['target']} activity model is installed. Resolve and "
            "review the target brief, then add a validated target-specific "
            "scorer before launching optimization."
        )
    client = client_from_env()
    run = run_optimization(
        proposer=propose,
        llm_client=client,
        seed_smiles=intake["canonical_smiles"],
        seed_name=intake["name"],
    )
    st.session_state["mission"] = compile_mission(run, intake)
    st.session_state["mission_source"] = (
        "Live local models with deterministic policy"
        + (" and advisory-only LLM notes" if client is not None else "")
    )


initialize_state()

st.title("Agent Perry")
st.caption(
    "An explainable multi-agent workspace for target-aware molecular "
    "optimization, decision review, and wet-lab planning."
)

with st.form("molecule_intake", border=False):
    input_column, target_column, remote_column, action_column = st.columns(
        [0.40, 0.25, 0.14, 0.21],
        vertical_alignment="bottom",
    )
    with input_column:
        query = st.text_input(
            "Molecule name or SMILES",
            key="query",
            placeholder="Try Haloperidol, Sulpiride, or a valid SMILES",
            help=(
                "Offline catalog: "
                + ", ".join(catalog_names())
                + ". PubChem lookup is explicit and optional."
            ),
        )
    with target_column:
        target_query = st.text_input(
            "Molecular target",
            key="target_query",
            placeholder="DRD2, HTR2A, EGFR…",
            help=(
                "Targets are explicit. This build includes a DRD2 activity "
                "model; other targets require their own validated scorer."
            ),
        )
    with remote_column:
        use_pubchem = st.checkbox(
            "Use PubChem",
            value=False,
            help="Only used after local catalog and SMILES resolution fail.",
        )
    with action_column:
        resolve_clicked = st.form_submit_button(
            "Resolve molecule",
            width="stretch",
        )

button_columns = st.columns([0.28, 0.28, 0.44], vertical_alignment="center")
with button_columns[0]:
    live_clicked = st.button(
        "Launch live mission",
        type="primary",
        width="stretch",
        help="Requires the optional live oracle dependencies.",
    )
with button_columns[1]:
    st.button(
        "Restore judged demo",
        width="stretch",
        on_click=load_cached_demo,
        kwargs={"reset_query": True},
    )
with button_columns[2]:
    st.caption(
        f"Active source: {st.session_state['mission_source']}. "
        "The judged Haloperidol path is fully offline."
    )

if resolve_clicked or live_clicked:
    try:
        intake = resolve_intake(query, target_query, use_pubchem)
        st.session_state["resolved_intake"] = intake
        if (
            resolve_clicked
            and intake["name"] == "Haloperidol"
            and intake["target"] == "DRD2"
            and intake["source"] in {"catalog", "smiles"}
        ):
            load_cached_demo()
        if live_clicked:
            with st.spinner(
                "Running 30 molecular evaluations and the deterministic policy"
            ):
                run_live_mission(intake)
    except CompoundResolutionError as error:
        st.error(str(error))
    except OptimizationHalt as error:
        st.error(
            f"Mission stopped safely ({error.code}): {error}. "
            "No unsupported winner was produced."
        )
    except Exception as error:
        st.error(f"Live mission unavailable: {error}")

mission = st.session_state["mission"]
resolved_intake = st.session_state["resolved_intake"]
if (
    resolved_intake["canonical_smiles"]
    != mission["intake"]["canonical_smiles"]
    or resolved_intake["target"] != mission["target"]["id"]
):
    activity_status = (
        "Its installed activity model is ready for live scoring."
        if resolved_intake["target"] == "DRD2"
        else (
            f"No {resolved_intake['target']} activity model is installed, so "
            "Agent Perry will not fabricate target scores."
        )
    )
    st.info(
        f"Resolved {resolved_intake['name']} to "
        f"{resolved_intake['canonical_smiles']} for target "
        f"{resolved_intake['target']}. Its target rationale is "
        f"{resolved_intake['target_rationale_status']}. {activity_status} "
        "The workspace below remains the validated Haloperidol-DRD2 replay "
        "until a supported mission completes."
    )
st.iframe(
    build_workspace_html(mission),
    width="stretch",
    height="content",
    tab_index=0,
)
