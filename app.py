"""
Week 5 (optional): interactive Streamlit demo for the ADMET property
prediction project. Paste a SMILES string, get predictions for all 5
trained ADMET properties with plain-language interpretation.

Reuses the exact featurization pipeline from Model_Improvement.ipynb
(largest-fragment salt stripping, 2048-bit Morgan fingerprint + 6 RDKit
descriptors) so predictions match what the notebooks report. Models are
the final tuned-XGBoost `best_estimator_` objects from Experiment 3,
saved to models/*.pkl by that notebook's last cell.
"""

import os
from urllib.parse import quote

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from rdkit.DataStructs import ConvertToNumpyArray

MODEL_DIR = "models"
PUBCHEM_HEADERS = {"User-Agent": "admet-property-prediction-demo"}

EXAMPLE_MOLECULES = {
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "Diazepam": "CN1C(=O)CN=C(C2=CC=CC=C2)C2=CC(Cl)=CC=C12",
    "Loperamide": "CCC(CC1=CC=CC=C1)(C1=CC=CC=C1)C(=O)N(C)CCC(O)(C1=CC=CC=C1)C1=CC=CC=C1",
}

# Same 5 properties, same order as the README's results table.
PROPERTIES = ["Solubility", "BBB", "hERG", "CYP3A4", "Clearance"]

MODEL_FILES = {
    "Solubility": "solubility_xgb.pkl",
    "BBB": "bbb_xgb.pkl",
    "hERG": "herg_xgb.pkl",
    "CYP3A4": "cyp3a4_xgb.pkl",
    "Clearance": "clearance_xgb.pkl",
}


# ---------------------------------------------------------------------------
# Featurization -- unchanged from Model_Improvement.ipynb, adapted to a
# single molecule instead of a DataFrame of many.
# ---------------------------------------------------------------------------

def largest_fragment(mol):
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) == 1:
        return mol
    return max(frags, key=lambda m: m.GetNumHeavyAtoms())


def compute_morgan_fp(mol, radius=2, n_bits=2048):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    ConvertToNumpyArray(fp, arr)
    return arr


def featurize_one(mol, radius=2, n_bits=2048):
    fp = compute_morgan_fp(mol, radius=radius, n_bits=n_bits)
    descriptors = {
        "MolWt": Descriptors.MolWt(mol),
        "MolLogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
    }
    fp_cols = [f"fp_{i}" for i in range(n_bits)]
    row = {**dict(zip(fp_cols, fp)), **descriptors}
    return pd.DataFrame([row]), descriptors


# ---------------------------------------------------------------------------
# PubChem name lookup -- lets users search by molecule name when they don't
# have a SMILES string handy, without requiring one (manual paste still works).
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def pubchem_search_names(query, limit=8):
    query = query.strip()
    if len(query) < 2:
        return []
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/autocomplete/compound/{quote(query)}/json"
    try:
        resp = requests.get(url, params={"limit": limit}, headers=PUBCHEM_HEADERS, timeout=5)
        resp.raise_for_status()
        return resp.json().get("dictionary_terms", {}).get("compound", [])
    except requests.RequestException:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def pubchem_name_to_smiles(name):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(name)}/property/SMILES/JSON"
    try:
        resp = requests.get(url, headers=PUBCHEM_HEADERS, timeout=5)
        if not resp.ok:
            return None
        properties = resp.json().get("PropertyTable", {}).get("Properties", [])
        return properties[0]["SMILES"] if properties else None
    except (requests.RequestException, KeyError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@st.cache_resource
def load_models():
    models = {}
    missing = []
    for prop, fname in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            models[prop] = joblib.load(path)
        else:
            missing.append(path)
    return models, missing


# ---------------------------------------------------------------------------
# Plain-language interpretation
# ---------------------------------------------------------------------------

def interpret_solubility(logs):
    # Common LogS (log mol/L) solubility convention (e.g. Delaney/ESOL bins).
    if logs > -1:
        return "Highly soluble", "🟢"
    if logs > -2:
        return "Soluble", "🟢"
    if logs > -3:
        return "Moderately soluble", "🟡"
    if logs > -4:
        return "Poorly soluble", "🟠"
    return "Very poorly soluble", "🔴"


def interpret_binary(prob, positive_label, negative_label, positive_is_bad):
    label = positive_label if prob >= 0.5 else negative_label
    if positive_is_bad:
        icon = "🔴" if prob >= 0.5 else "🟢"
    else:
        icon = "🟢" if prob >= 0.5 else "🟠"
    return label, icon


def interpret_clearance(value):
    # Dataset's own observed range is ~3-150 uL/min/1e6 cells (Clearance_Hepatocyte_AZ).
    # Heuristic tertiles relative to that range, not a clinical cutoff.
    if value < 15:
        return "Low (relative to this dataset's typical 3-150 range)", "🟢"
    if value < 50:
        return "Moderate (relative to this dataset's typical 3-150 range)", "🟡"
    return "High (relative to this dataset's typical 3-150 range)", "🟠"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="ADMET Property Predictor", page_icon="🧪", layout="centered")

st.title("🧪 ADMET Property Predictor")
st.caption(
    "Paste a SMILES string to predict 5 ADMET properties: aqueous solubility, "
    "blood-brain barrier (BBB) permeability, hERG blockade score, CYP3A4 "
    "metabolic inhibition, and hepatocyte clearance."
)

models, missing = load_models()
if missing:
    st.error(
        "Model file(s) not found: " + ", ".join(missing) + ". "
        "Run Model_Improvement.ipynb's last cell to generate them."
    )
    st.stop()

st.subheader("Try an example")
example_cols = st.columns(len(EXAMPLE_MOLECULES))
if "smiles_input" not in st.session_state:
    st.session_state.smiles_input = EXAMPLE_MOLECULES["Aspirin"]

for col, (name, smi) in zip(example_cols, EXAMPLE_MOLECULES.items()):
    if col.button(name):
        st.session_state.smiles_input = smi

st.subheader("Or search by name")
st.caption("Don't have a SMILES string handy? Search PubChem by molecule name instead.")
name_query = st.text_input("Molecule name", placeholder="e.g. aspirin", key="name_query")

if name_query.strip():
    matches = pubchem_search_names(name_query)
    if matches:
        st.caption("Select a match to fill in its SMILES string:")
        match_cols = st.columns(4)
        for i, match_name in enumerate(matches):
            if match_cols[i % 4].button(match_name, key=f"pubchem_match_{i}_{match_name}"):
                smi = pubchem_name_to_smiles(match_name)
                if smi:
                    st.session_state.smiles_input = smi
                else:
                    st.error(f"Couldn't fetch a SMILES string for '{match_name}' from PubChem.")
    else:
        st.caption("No PubChem matches yet -- keep typing, or paste a SMILES string directly below.")

smiles = st.text_input("Or paste a SMILES string directly", key="smiles_input")
predict_clicked = st.button("Predict", type="primary")

if predict_clicked:
    mol_raw = Chem.MolFromSmiles(smiles)
    if mol_raw is None:
        st.error("Couldn't parse that SMILES string. Check the syntax and try again.")
        st.stop()

    mol = largest_fragment(mol_raw)
    X, descriptors = featurize_one(mol)

    col_img, col_desc = st.columns([1, 1.2])
    with col_img:
        img = Draw.MolToImage(mol, size=(300, 300))
        st.image(img, caption="Largest fragment used for prediction")
    with col_desc:
        st.markdown("**Computed descriptors**")
        desc_df = pd.DataFrame(
            {"Value": [f"{v:.2f}" for v in descriptors.values()]},
            index=descriptors.keys(),
        )
        st.table(desc_df)

    st.subheader("Predictions")

    sol_pred = models["Solubility"].predict(X)[0]
    sol_label, sol_icon = interpret_solubility(sol_pred)

    bbb_score = models["BBB"].predict_proba(X)[0, 1]
    bbb_label, bbb_icon = interpret_binary(
        bbb_score,
        "Classifier leans permeable",
        "Classifier leans non-permeable",
        positive_is_bad=False,
    )

    herg_score = models["hERG"].predict_proba(X)[0, 1]
    herg_label, herg_icon = interpret_binary(
        herg_score,
        "Classifier flags potential blocker",
        "Classifier does not flag blocker",
        positive_is_bad=True,
    )

    cyp_score = models["CYP3A4"].predict_proba(X)[0, 1]
    cyp_label, cyp_icon = interpret_binary(
        cyp_score,
        "Classifier leans inhibitor",
        "Classifier leans non-inhibitor",
        positive_is_bad=True,
    )

    clr_pred = models["Clearance"].predict(X)[0]
    clr_label, clr_icon = interpret_clearance(clr_pred)

    rows = [
        ("Solubility (LogS)", f"{sol_pred:.2f}", f"{sol_icon} {sol_label}"),
        ("BBB Permeability", f"{bbb_score:.3f} classifier score", f"{bbb_icon} {bbb_label}"),
        ("hERG Blockade", f"{herg_score:.3f} classifier score", f"{herg_icon} {herg_label}"),
        ("CYP3A4 Inhibition", f"{cyp_score:.3f} classifier score", f"{cyp_icon} {cyp_label}"),
        ("Clearance (uL/min/1e6 cells)", f"{clr_pred:.1f}", f"{clr_icon} {clr_label}"),
    ]
    results_df = pd.DataFrame(rows, columns=["Property", "Model output", "Interpretation"])
    st.table(results_df.set_index("Property"))

    st.caption(
        "Educational project, not for clinical or regulatory use. Classifier "
        "outputs are uncalibrated ranking scores, not probabilities or clinical "
        "risk estimates. Model performance and known limitations are documented "
        "in README.md and NOTES.md."
    )

st.divider()
st.caption(
    "Models: tuned XGBoost per property, scaffold-split trained, from "
    "[admet-property-prediction](https://github.com/tuhinc5203/admet-property-prediction)."
)
