"""Stable interface shared by the oracle, proposal, agent, and UI layers."""

from functools import lru_cache
import json
import os
from pathlib import Path
import sys

import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem

from .scoring import featurize_smiles, load_models


SCORE_KEYS = ("affinity", "solubility", "bbb", "herg", "sa")
SEED_SMILES = (
    "C1CN(CCC1(C2=CC=C(C=C2)Cl)O)"
    "CCCC(=O)C3=CC=C(C=C3)F"
)
CANDIDATES_PER_ROUND = 6

_REACTION_SMARTS = (
    "[cH:1]>>[c:1]F",
    "[cH:1]>>[c:1]Cl",
    "[cH:1]>>[c:1]O",
    "[cH:1]>>[c:1]OC",
    "[cH:1]>>[c:1]C",
    "[cH:1]>>[c:1]C(=O)O",
    "[cH:1]>>[n:1]",
)

# Known CNS-active molecules are deterministic rescue proposals if graph
# transforms produce too few unique products. They also keep a safety trade-off
# visible in the cached demonstration.
_RESCUE_SMILES = (
    "CCN1CCCC1CNC(=O)C2=C(C=CC(=C2)S(=O)(=O)N)OC",  # sulpiride
    "CN1CCN(CC1)C2=NC3=CC=CC=C3NC4=C2C=C(C=C4)Cl",  # clozapine-like
    "CC1=NC2=C(N1CCN3CCC(CC3)OC)C=CC=C2",  # olanzapine-like
    "CCN(CC)CCNC(=O)C1=CC(=C(C=C1)OC)N",  # metoclopramide
)


def is_valid(smiles: str) -> bool:
    """Return whether RDKit can parse and sanitize the molecule."""
    if not isinstance(smiles, str) or not smiles.strip():
        return False
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return False
    try:
        Chem.SanitizeMol(molecule)
    except (ValueError, RuntimeError):
        return False
    return molecule.GetNumHeavyAtoms() > 0


def score(smiles: str) -> dict[str, float]:
    """Score unitless DRD2 activity, solubility, BBB, hERG, and synthetic access."""
    if not is_valid(smiles):
        raise ValueError(f"Invalid SMILES: {smiles}")

    features, _ = featurize_smiles(smiles)
    models = load_models()
    molecule = Chem.MolFromSmiles(smiles)
    raw = {
        "affinity": float(_get_drd2_oracle()(smiles)),
        "solubility": float(models["Solubility"].predict(features)[0]),
        "bbb": float(models["BBB"].predict_proba(features)[0, 1]),
        "herg": float(models["hERG"].predict_proba(features)[0, 1]),
        "sa": float(_get_sa_scorer()(molecule)),
    }
    if set(raw) != set(SCORE_KEYS):
        raise RuntimeError("Oracle contract returned an unexpected score set")
    if not all(np.isfinite(value) for value in raw.values()):
        raise ValueError("An oracle returned a non-finite score")
    return raw


def normalize(raw: dict[str, float]) -> dict[str, float]:
    """Map all five properties to 0-1 with higher consistently better."""
    missing = set(SCORE_KEYS) - set(raw)
    if missing:
        raise ValueError(f"Missing scores: {', '.join(sorted(missing))}")

    values = {
        "affinity": raw["affinity"],
        # A pragmatic display scale: predicted LogS -6 -> 0, LogS 0 -> 1.
        "solubility": (raw["solubility"] + 6.0) / 6.0,
        "bbb": raw["bbb"],
        # The underlying model predicts blockade probability, so invert it.
        "herg": 1.0 - raw["herg"],
        # RDKit SA convention is approximately 1 (easy) to 10 (difficult).
        "sa": (10.0 - raw["sa"]) / 9.0,
    }
    return {
        key: float(np.clip(values[key], 0.0, 1.0))
        for key in SCORE_KEYS
    }


def propose(parent: str, context: dict) -> list[str]:
    """Generate six deterministic, sanitized proposals from a parent."""
    if not is_valid(parent):
        raise ValueError(f"Invalid parent SMILES: {parent}")

    molecule = Chem.MolFromSmiles(parent)
    parent_canonical = Chem.MolToSmiles(molecule)
    products: list[str] = []
    products_by_reaction: list[list[str]] = []
    seen = {parent_canonical}

    for smarts in _REACTION_SMARTS:
        reaction = AllChem.ReactionFromSmarts(smarts)
        reaction_products = []
        for product_tuple in reaction.RunReactants((molecule,)):
            if not product_tuple:
                continue
            product = product_tuple[0]
            try:
                Chem.SanitizeMol(product)
                canonical = Chem.MolToSmiles(product)
            except (ValueError, RuntimeError):
                continue
            if canonical not in seen and is_valid(canonical):
                products.append(canonical)
                reaction_products.append(canonical)
                seen.add(canonical)
        products_by_reaction.append(reaction_products)

    round_number = int(context.get("round", 1))
    diverse_products = []
    # Fluorination creates visible hERG conflict; hydroxyl and ring-N edits
    # provide plausible counter-proposals from ADME and safety.
    for reaction_index in (0, 2, 6, 3, 4, 5, 1):
        reaction_products = products_by_reaction[reaction_index]
        if reaction_products:
            product_index = (round_number - 1) % len(reaction_products)
            diverse_products.append(reaction_products[product_index])
    diverse_set = set(diverse_products)
    diverse_products.extend(
        product for product in products if product not in diverse_set
    )

    # Elitism keeps the current best in contention, three slots show direct
    # parent edits, and two rescue slots prevent a narrow mutation operator
    # from trapping every candidate behind the hERG safety veto.
    selected = [parent_canonical, *diverse_products[:3]]
    selected_seen = set(selected)
    rescue = list(_RESCUE_SMILES)
    for candidate in rescue + diverse_products[3:]:
        canonical = Chem.MolToSmiles(Chem.MolFromSmiles(candidate))
        if canonical not in selected_seen and is_valid(canonical):
            selected.append(canonical)
            selected_seen.add(canonical)
        if len(selected) == CANDIDATES_PER_ROUND:
            break

    if len(selected) < 3:
        raise RuntimeError("Proposal fallback produced fewer than three molecules")
    return selected[:CANDIDATES_PER_ROUND]


def propose_with_llm(parent: str, context: dict, llm_client) -> list[str]:
    """Validate advisory LLM edits while preserving the parent incumbent."""
    fallback = propose(parent, context)
    parent_canonical = Chem.MolToSmiles(Chem.MolFromSmiles(parent))
    system_prompt = (
        "You edit a parent molecule for DRD2 lead optimization. Suggest six "
        "small molecular edits informed by the scores and agent notes. Do not "
        "claim they are synthesizable. Return strict JSON exactly as "
        '{"proposals": ["SMILES", "..."]}.'
    )
    try:
        response = llm_client(
            system_prompt,
            json.dumps({"parent": parent, "context": context}),
        )
        emitted = response.get("proposals", [])
    except Exception:
        emitted = []

    accepted = [parent_canonical]
    seen = {parent_canonical}
    for candidate in [*emitted, *fallback[1:]]:
        if not isinstance(candidate, str) or not is_valid(candidate):
            continue
        canonical = Chem.MolToSmiles(Chem.MolFromSmiles(candidate))
        if canonical not in seen:
            accepted.append(canonical)
            seen.add(canonical)
        if len(accepted) == CANDIDATES_PER_ROUND:
            break
    return accepted


@lru_cache(maxsize=1)
def _get_drd2_oracle():
    try:
        from tdc import Oracle
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "Live DRD2 scoring requires requirements-oracles.txt. "
            "Cached replay remains available."
        ) from error
    return Oracle(name="DRD2")


@lru_cache(maxsize=1)
def _get_sa_scorer():
    contrib_path = Path(RDConfig.RDContribDir) / "SA_Score"
    if not contrib_path.exists():
        raise RuntimeError(f"RDKit SA_Score was not found at {contrib_path}")
    path_text = os.fspath(contrib_path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
    import sascorer

    return sascorer.calculateScore
