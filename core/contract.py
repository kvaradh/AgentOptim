"""The interface contract. Every workstream codes against this module.

Five property axes, one record format, one normalisation. Nothing in here
talks to a model or an LLM -- those live in `core.oracles` and `core.agents`
and are reached through the thin wrappers at the bottom of this file.

Raw units (what `score()` returns):

    affinity    0-1   DRD2 oracle, higher = more potent
    solubility  logS  log10 mol/L, higher = more soluble
    bbb         0-1   P(blood-brain-barrier penetrant), higher = penetrant
    herg        0-1   P(hERG blockade), higher = MORE cardiotoxic
    sa          1-10  synthetic accessibility, LOWER = easier to make

Normalised units (what `normalize()` returns): all five on 0-1 where
higher is always better. `herg` and `sa` therefore invert.
"""

from __future__ import annotations

AXES = ("affinity", "solubility", "bbb", "herg", "sa")

# Normalisation windows for the two axes that are not already 0-1.
# logS: AqSolDB spans about -13..2; -8..0 covers the drug-like band without
# letting one insoluble outlier flatten every bar in the UI.
SOLUBILITY_WINDOW = (-8.0, 0.0)
SA_WINDOW = (1.0, 10.0)

# The safety agent's kill line, in NORMALISED units (so 0.30 here means
# P(hERG blockade) > 0.70). Stated in the contract rather than inside the
# agent so the number is reviewable in one place.
HERG_VETO_THRESHOLD = 0.30


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


def _window(x: float, lo: float, hi: float, invert: bool = False) -> float:
    frac = (float(x) - lo) / (hi - lo)
    return _clip01(1.0 - frac if invert else frac)


def normalize(raw: dict) -> dict:
    """Map a raw score record to 0-1 per axis, higher always better.

    Pure function of `raw`: no model calls, no RDKit. Safe to use on
    fixtures, on cached runs, and in tests.
    """
    missing = [a for a in AXES if a not in raw]
    if missing:
        raise KeyError(f"raw score record missing axes: {missing}")

    return {
        "affinity": _clip01(raw["affinity"]),
        "solubility": _window(raw["solubility"], *SOLUBILITY_WINDOW),
        "bbb": _clip01(raw["bbb"]),
        # hERG inverts: the model predicts blockade, and blockade is bad.
        "herg": _clip01(1.0 - float(raw["herg"])),
        # SA inverts: 1 is trivial to make, 10 is a research project.
        "sa": _window(raw["sa"], *SA_WINDOW, invert=True),
    }


def is_valid(smiles: str) -> bool:
    """RDKit parse + sanitize. False rather than raising, for filtering."""
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    if not isinstance(smiles, str) or not smiles.strip():
        return False
    mol = Chem.MolFromSmiles(smiles)  # MolFromSmiles sanitizes by default
    return mol is not None


def canonical(smiles: str) -> str:
    """Canonical SMILES, so the same molecule is one key everywhere.

    Raises ValueError on invalid input.
    """
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    return Chem.MolToSmiles(mol)


def score(smiles: str) -> dict:
    """Five raw property values for one molecule.

    Raises ValueError on invalid SMILES. The returned record also carries a
    `provenance` key naming the backend behind each number -- see
    `core.oracles`. Callers that only want the five floats can pass the
    record straight to `normalize()`, which ignores extra keys.
    """
    from core.oracles import score as _score

    return _score(smiles)


def propose(parent: str, context: dict) -> list[str]:
    """3-8 valid, distinct SMILES derived from `parent`.

    context = {"round": int, "scores": dict, "agent_notes": list[str]}
    Never returns an empty list; never returns an invalid SMILES.
    """
    from core.edits import propose as _propose

    return _propose(parent, context)
