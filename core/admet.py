"""ADMET property backends: the inherited XGBoost models, or a labelled surrogate.

Two backends implement the same three-method interface
(`solubility`, `bbb`, `herg`):

  XGBoostBackend    the inherited models in `models/`. Real predictions.
  SurrogateBackend  transparent physchem heuristics. NOT a trained model.

The surrogate exists so that the agent loop, the edit engine and the
frontend can be built and demoed before (or without) the inherited models.
Every score record says which backend produced it, and the frontend prints
that label. This matters: a surrogate number wearing an "0.92 ROC-AUC" claim
in front of judges is the one failure mode of this build that is dishonest
rather than merely broken.

Featurization for the XGBoost path, per the inherited repo:
    2048-bit Morgan fingerprint (radius 2) ++ six descriptors in order
    [MW, LogP, HBD, HBA, RotatableBonds, TPSA]
If the saved models disagree with this, do not reimplement it here -- point
ADMET_FEATURIZER at the inherited repo's own function (see `featurize`).
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

MODELS_DIR = Path(os.environ.get("ADMET_MODELS_DIR", "models"))

MORGAN_RADIUS = 2
MORGAN_BITS = 2048
DESCRIPTOR_ORDER = ("MolWt", "MolLogP", "NumHDonors", "NumHAcceptors",
                    "NumRotatableBonds", "TPSA")

# Filenames tried per task, in order. Extend rather than rename the models.
_MODEL_FILES = {
    "solubility": ("solubility.json", "solubility_xgb.json", "solubility.pkl",
                   "esol.json", "aqsol.json"),
    "bbb": ("bbb.json", "bbb_xgb.json", "bbb.pkl", "bbb_martins.json"),
    "herg": ("herg.json", "herg_xgb.json", "herg.pkl"),
}


def _descriptors(mol) -> list[float]:
    from rdkit.Chem import Descriptors

    return [float(getattr(Descriptors, name)(mol)) for name in DESCRIPTOR_ORDER]


def featurize(smiles: str):
    """2048-bit Morgan(r=2) fingerprint concatenated with six descriptors.

    Set ADMET_FEATURIZER="package.module:function" to delegate to the
    inherited repo's own featurization instead. That override is the
    documented fix when verification against the source notebook fails:
    a feature vector that is merely *plausible* returns numbers that mean
    nothing, and nothing crashes to tell you.
    """
    override = os.environ.get("ADMET_FEATURIZER")
    if override:
        module_name, _, func_name = override.partition(":")
        func = getattr(importlib.import_module(module_name), func_name)
        return func(smiles)

    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")

    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS, fpSize=MORGAN_BITS
    )
    fp = np.asarray(gen.GetFingerprintAsNumPy(mol), dtype=np.float32)
    desc = np.asarray(_descriptors(mol), dtype=np.float32)
    return np.concatenate([fp, desc]).reshape(1, -1)


class SurrogateBackend:
    """Physchem heuristics standing in for the trained ADMET models.

    Directionally sane and monotone in the properties medicinal chemists
    actually trade off -- polarity raises solubility and lowers BBB; lipophilic
    bases with long tails raise hERG risk -- which is enough to make the agents
    genuinely disagree. It is not calibrated and carries no accuracy claim.
    """

    name = "surrogate"
    is_surrogate = True

    def _props(self, smiles: str) -> dict:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"invalid SMILES: {smiles!r}")
        return {
            "logp": Crippen.MolLogP(mol),
            "mw": Descriptors.MolWt(mol),
            "tpsa": rdMolDescriptors.CalcTPSA(mol),
            "hbd": rdMolDescriptors.CalcNumHBD(mol),
            "rotb": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "arom": rdMolDescriptors.CalcNumAromaticRings(mol),
            "basic_n": sum(
                1
                for a in mol.GetAtoms()
                if a.GetSymbol() == "N"
                and not a.GetIsAromatic()
                and a.GetTotalNumHs() + a.GetDegree() <= 3
            ),
        }

    def solubility(self, smiles: str) -> float:
        """logS, in the spirit of the ESOL equation (Delaney 2004)."""
        p = self._props(smiles)
        logs = 0.16 - 0.63 * p["logp"] - 0.0062 * p["mw"] + 0.066 * p["rotb"] - 0.74 * p["arom"]
        return max(-12.0, min(1.5, logs))

    def bbb(self, smiles: str) -> float:
        """P(penetrant). Rewards moderate lipophilicity, punishes polarity."""
        p = self._props(smiles)
        z = (
            1.4
            + 0.55 * min(p["logp"], 5.0)
            - 0.035 * p["tpsa"]
            - 0.004 * max(0.0, p["mw"] - 400.0)
            - 0.45 * p["hbd"]
        )
        return 1.0 / (1.0 + pow(2.718281828459045, -z))

    def herg(self, smiles: str) -> float:
        """P(blockade). The classic pharmacophore: lipophilic + basic amine."""
        p = self._props(smiles)
        z = (
            -3.4
            + 0.62 * p["logp"]
            + 0.85 * min(p["basic_n"], 2)
            + 0.10 * p["arom"]
            + 0.006 * max(0.0, p["mw"] - 250.0)
            - 0.018 * p["tpsa"]
        )
        return 1.0 / (1.0 + pow(2.718281828459045, -z))


class XGBoostBackend:
    """The inherited XGBoost models, loaded from `models/`.

    Constructing this raises if the three models are not all loadable, so a
    half-wired backend can never silently serve surrogate numbers alongside
    real ones.
    """

    name = "xgboost"
    is_surrogate = False

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self._models = {task: self._load(task) for task in _MODEL_FILES}
        meta_path = self.models_dir / "metrics.json"
        self.metrics = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    def _load(self, task: str):
        for filename in _MODEL_FILES[task]:
            path = self.models_dir / filename
            if not path.exists():
                continue
            if path.suffix == ".json":
                import xgboost as xgb

                booster = xgb.Booster()
                booster.load_model(str(path))
                return ("booster", booster)
            import pickle

            with path.open("rb") as handle:
                return ("sklearn", pickle.load(handle))
        raise FileNotFoundError(
            f"no model for {task!r} in {self.models_dir} "
            f"(looked for {', '.join(_MODEL_FILES[task])})"
        )

    def _predict(self, task: str, smiles: str) -> float:
        kind, model = self._models[task]
        features = featurize(smiles)
        if kind == "booster":
            import xgboost as xgb

            return float(model.predict(xgb.DMatrix(features))[0])
        if task == "solubility":  # regression
            return float(model.predict(features)[0])
        return float(model.predict_proba(features)[0][1])

    def solubility(self, smiles: str) -> float:
        return self._predict("solubility", smiles)

    def bbb(self, smiles: str) -> float:
        return self._predict("bbb", smiles)

    def herg(self, smiles: str) -> float:
        return self._predict("herg", smiles)


def get_backend(prefer_real: bool = True):
    """Return the XGBoost backend when the models are present, else surrogate.

    Set ADMET_BACKEND=surrogate to force the surrogate, or
    ADMET_BACKEND=xgboost to fail loudly rather than fall back.
    """
    forced = os.environ.get("ADMET_BACKEND", "").strip().lower()
    if forced == "surrogate":
        return SurrogateBackend()
    if forced == "xgboost":
        return XGBoostBackend()
    if prefer_real:
        try:
            return XGBoostBackend()
        except Exception:
            pass
    return SurrogateBackend()
