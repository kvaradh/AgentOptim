import json
from pathlib import Path

import pytest
from rdkit import Chem

from core.contract import (
    SCORE_KEYS,
    SEED_SMILES,
    is_valid,
    normalize,
    propose,
    propose_with_llm,
    score,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_normalization_is_bounded_and_inverts_herg():
    raw = {
        "affinity": 0.8,
        "solubility": -3.0,
        "bbb": 0.6,
        "herg": 0.7,
        "sa": 4.0,
    }

    values = normalize(raw)

    assert list(values) == list(SCORE_KEYS)
    assert all(0 <= value <= 1 for value in values.values())
    assert values["solubility"] == 0.5
    assert values["herg"] == pytest.approx(0.3)


def test_invalid_smiles_is_rejected_before_oracle_loading():
    assert is_valid("not-a-smiles") is False
    with pytest.raises(ValueError, match="Invalid SMILES"):
        score("not-a-smiles")


def test_propose_returns_six_unique_sanitized_molecules():
    proposals = propose(
        SEED_SMILES,
        {"round": 1, "scores": {}, "agent_notes": []},
    )

    assert len(proposals) == 6
    assert len(set(proposals)) == 6
    assert all(is_valid(smiles) for smiles in proposals)
    assert Chem.MolToSmiles(Chem.MolFromSmiles(SEED_SMILES)) in proposals


def test_frontend_fixtures_follow_exact_score_contract():
    fixtures = json.loads((REPOSITORY_ROOT / "fixtures.json").read_text())

    assert len(fixtures) == 5
    assert all(set(item["scores"]) == set(SCORE_KEYS) for item in fixtures)
    assert all(is_valid(item["smiles"]) for item in fixtures)


def test_llm_proposals_drop_invalid_values_and_top_up_from_fallback():
    def fake_client(system_prompt, payload):
        return {"proposals": ["not-a-smiles", "CCO", "CCO"]}

    proposals = propose_with_llm(
        SEED_SMILES,
        {"round": 1, "scores": {}, "agent_notes": []},
        fake_client,
    )

    assert len(proposals) == 6
    assert len(set(proposals)) == 6
    assert all(is_valid(smiles) for smiles in proposals)


def test_llm_proposals_cannot_displace_the_parent_incumbent():
    emitted = ["C", "N", "O", "CO", "CN", "C#N"]

    def fake_client(system_prompt, payload):
        return {"proposals": emitted}

    proposals = propose_with_llm(
        SEED_SMILES,
        {"round": 1, "scores": {}, "agent_notes": []},
        fake_client,
    )
    parent = Chem.MolToSmiles(Chem.MolFromSmiles(SEED_SMILES))

    assert proposals[0] == parent
    assert len(proposals) == 6
    assert len(set(proposals)) == 6
