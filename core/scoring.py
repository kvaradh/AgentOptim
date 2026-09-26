from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from rdkit.DataStructs import ConvertToNumpyArray

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPOSITORY_ROOT / "models"
MODEL_FILES = {
    "Solubility": "solubility_xgb.pkl",
    "BBB": "bbb_xgb.pkl",
    "hERG": "herg_xgb.pkl",
}
DESCRIPTOR_NAMES = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
)


def parse_largest_fragment(smiles: str):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    fragments = Chem.GetMolFrags(molecule, asMols=True)
    if len(fragments) == 1:
        return molecule
    return max(fragments, key=lambda fragment: fragment.GetNumHeavyAtoms())


def compute_morgan_fp(molecule, radius: int = 2, n_bits: int = 2048):
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        molecule,
        radius=radius,
        nBits=n_bits,
    )
    values = np.zeros((n_bits,), dtype=int)
    ConvertToNumpyArray(fingerprint, values)
    return values


def featurize_smiles(smiles: str, radius: int = 2, n_bits: int = 2048):
    molecule = parse_largest_fragment(smiles)
    fingerprint = compute_morgan_fp(molecule, radius=radius, n_bits=n_bits)
    descriptors = {
        "MolWt": Descriptors.MolWt(molecule),
        "MolLogP": Descriptors.MolLogP(molecule),
        "TPSA": Descriptors.TPSA(molecule),
        "NumHDonors": Descriptors.NumHDonors(molecule),
        "NumHAcceptors": Descriptors.NumHAcceptors(molecule),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(molecule),
    }
    fingerprint_columns = [f"fp_{index}" for index in range(n_bits)]
    row = {**dict(zip(fingerprint_columns, fingerprint)), **descriptors}
    return pd.DataFrame([row]), descriptors


@lru_cache(maxsize=2)
def load_models(model_dir: str | Path = MODEL_DIR):
    directory = Path(model_dir)
    missing = [
        directory / filename
        for filename in MODEL_FILES.values()
        if not (directory / filename).exists()
    ]
    if missing:
        paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing model files: {paths}")

    return {
        property_name: joblib.load(directory / filename)
        for property_name, filename in MODEL_FILES.items()
    }


def score_admet(smiles: str, models=None):
    active_models = load_models() if models is None else models
    features, descriptors = featurize_smiles(smiles)
    scores = {
        "solubility": float(
            active_models["Solubility"].predict(features)[0]
        ),
        "bbb": float(active_models["BBB"].predict_proba(features)[0, 1]),
        "herg": float(active_models["hERG"].predict_proba(features)[0, 1]),
    }
    return scores, descriptors


def draw_molecule(smiles: str, size=(280, 220)):
    molecule = parse_largest_fragment(smiles)
    return Draw.MolToImage(molecule, size=size)
