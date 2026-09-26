import io
import json
from urllib.error import URLError

import pytest

from mission_control.intake import (
    CompoundResolutionError,
    PubChemLookupError,
    catalog_names,
    resolve_compound,
)


REQUIRED_FIELDS = {
    "query",
    "source",
    "name",
    "canonical_smiles",
    "formula",
    "molecular_weight",
    "target",
    "therapeutic_intent",
    "seed_rationale",
    "optimization_hypothesis",
    "target_rationale_status",
    "target_rationale",
}


@pytest.mark.parametrize(
    "query, expected_name",
    [
        ("haloperidol", "Haloperidol"),
        ("SULPIRIDE", "Sulpiride"),
        (" Clozapine ", "Clozapine"),
        ("olanzapine", "Olanzapine"),
        ("METOCLOPRAMIDE", "Metoclopramide"),
    ],
)
def test_catalog_resolves_case_insensitively(query, expected_name):
    result = resolve_compound(query)

    assert set(result) == REQUIRED_FIELDS
    assert result["source"] == "catalog"
    assert result["name"] == expected_name
    assert result["target"] == "DRD2"
    assert result["target_rationale_status"] == "supported_reference"
    assert result["formula"]
    assert result["molecular_weight"] > 0


def test_catalog_has_expected_offline_entries():
    assert catalog_names() == (
        "Haloperidol",
        "Sulpiride",
        "Clozapine",
        "Olanzapine",
        "Metoclopramide",
    )


def test_valid_smiles_resolves_locally_without_calling_opener():
    def unexpected_opener(*args, **kwargs):
        raise AssertionError("network opener must not be called for valid SMILES")

    result = resolve_compound("OCC", opener=unexpected_opener)

    assert result["source"] == "smiles"
    assert result["canonical_smiles"] == "CCO"
    assert result["formula"] == "C2H6O"
    assert result["target"] == "DRD2"
    assert result["target_rationale_status"] == "unverified"


def test_explicit_non_drd2_target_is_preserved_without_claiming_support():
    result = resolve_compound("CCO", target="ADORA2A")

    assert result["target"] == "ADORA2A"
    assert result["target_rationale_status"] == "unverified"
    assert "ADORA2A" in result["therapeutic_intent"]
    assert "does not establish target engagement" in result["target_rationale"]


def test_drd2_alias_is_normalized():
    result = resolve_compound("Haloperidol", target="dopamine d2")

    assert result["target"] == "DRD2"
    assert result["target_rationale_status"] == "supported_reference"


def test_unknown_input_requires_explicit_pubchem_opt_in():
    with pytest.raises(CompoundResolutionError, match="explicitly enable PubChem"):
        resolve_compound("not in the catalog")


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def test_pubchem_json_is_parsed_with_injected_opener_and_timeout():
    payload = {
        "PropertyTable": {
            "Properties": [
                {
                    "CID": 702,
                    "Title": "Ethanol",
                    "ConnectivitySMILES": "C(C)O",
                    "MolecularFormula": "C2H6O",
                    "MolecularWeight": "46.07",
                }
            ]
        }
    }
    observed = {}

    def opener(url, *, timeout):
        observed.update(url=url, timeout=timeout)
        return _Response(json.dumps(payload).encode())

    result = resolve_compound(
        "ethyl alcohol",
        use_pubchem=True,
        opener=opener,
        timeout=1.25,
    )

    assert result["source"] == "pubchem"
    assert result["name"] == "Ethanol"
    assert result["canonical_smiles"] == "CCO"
    assert result["formula"] == "C2H6O"
    assert result["molecular_weight"] == pytest.approx(46.07)
    assert result["target_rationale_status"] == "unverified"
    assert "ethyl%20alcohol" in observed["url"]
    assert observed["timeout"] == pytest.approx(1.25)


@pytest.mark.parametrize(
    "opener, message",
    [
        (
            lambda *args, **kwargs: (_ for _ in ()).throw(
                URLError("offline")
            ),
            "PubChem lookup failed",
        ),
        (
            lambda *args, **kwargs: _Response(b"not JSON"),
            "invalid JSON",
        ),
        (
            lambda *args, **kwargs: _Response(
                json.dumps({"PropertyTable": {"Properties": []}}).encode()
            ),
            "no compound properties",
        ),
    ],
)
def test_pubchem_failures_have_clear_errors(opener, message):
    with pytest.raises(PubChemLookupError, match=message):
        resolve_compound("remote only", use_pubchem=True, opener=opener)
