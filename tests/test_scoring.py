import json
from pathlib import Path

import pytest

from core.scoring import (
    DESCRIPTOR_NAMES,
    featurize_smiles,
    load_models,
    score_admet,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = json.loads((REPOSITORY_ROOT / "fixtures.json").read_text())[:3]
EXPECTED = {
    "haloperidol": {
        "solubility": -4.48,
        "bbb": 0.984,
        "herg": 0.865,
    },
    "droperidol": {
        "solubility": -3.38,
        "bbb": 0.997,
        "herg": 0.900,
    },
    "sulpiride": {
        "solubility": -2.43,
        "bbb": 0.741,
        "herg": 0.371,
    },
}


def test_feature_shape_and_exact_column_order():
    features, descriptors = featurize_smiles(CANDIDATES[0]["smiles"])

    assert features.shape == (1, 2054)
    assert list(features.columns[:3]) == ["fp_0", "fp_1", "fp_2"]
    assert list(features.columns[-6:]) == list(DESCRIPTOR_NAMES)
    assert list(descriptors) == list(DESCRIPTOR_NAMES)


def test_invalid_smiles_is_rejected():
    with pytest.raises(ValueError, match="Invalid SMILES"):
        featurize_smiles("not-a-smiles")


@pytest.mark.parametrize(
    "candidate",
    CANDIDATES,
    ids=lambda item: item["name"].lower(),
)
def test_three_inherited_models_match_checked_baseline(candidate):
    scores, _ = score_admet(candidate["smiles"], models=load_models())
    expected = EXPECTED[candidate["name"].lower()]

    assert scores["solubility"] == pytest.approx(
        expected["solubility"],
        abs=0.005,
    )
    assert scores["bbb"] == pytest.approx(expected["bbb"], abs=0.0005)
    assert scores["herg"] == pytest.approx(expected["herg"], abs=0.0005)
