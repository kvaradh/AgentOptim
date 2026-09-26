import math

import pytest

from mission_control import chemistry


def test_identity_and_hash_are_stable_across_equivalent_smiles():
    first = chemistry.molecular_identity("OCC")
    second = chemistry.molecular_identity("CCO")

    assert first == second
    assert first["canonical_smiles"] == "CCO"
    assert first["formula"] == "C2H6O"
    assert first["molecular_weight"] == pytest.approx(46.069, abs=0.001)
    assert first["molecule_id"].startswith("sha256:")
    assert chemistry.molecule_id("C(C)O") == first["molecule_id"]


def test_morgan_tanimoto_similarity_is_bounded_and_reflexive():
    same = chemistry.morgan_tanimoto_similarity("CCO", "OCC")
    different = chemistry.structural_similarity("CCO", "c1ccccc1")

    assert same == pytest.approx(1.0)
    assert 0.0 <= different <= 1.0
    assert different < same


def test_svg_depiction_is_complete():
    svg = chemistry.depict_svg("c1ccccc1O", width=240, height=180)

    assert "<svg" in svg
    assert "</svg>" in svg
    assert "width='240px'" in svg
    assert "height='180px'" in svg


def test_conformer_is_deterministic_finite_and_topologically_valid():
    first = chemistry.generate_conformer("CCO", optimize_uff=True)
    second = chemistry.generate_conformer("OCC", optimize_uff=True)

    assert first["status"] == "success"
    assert first["method"] == "ETKDGv3+UFF"
    assert first["is_3d"] is True
    assert first["optimization_status"] in {"converged", "not_converged"}
    assert first["atoms"]
    assert first["bonds"]

    first_coordinates = [
        (atom["x"], atom["y"], atom["z"]) for atom in first["atoms"]
    ]
    second_coordinates = [
        (atom["x"], atom["y"], atom["z"]) for atom in second["atoms"]
    ]
    assert second_coordinates == pytest.approx(first_coordinates, abs=1e-8)

    atom_count = len(first["atoms"])
    for expected_index, atom in enumerate(first["atoms"]):
        assert atom["index"] == expected_index
        assert all(
            math.isfinite(atom[axis])
            for axis in ("x", "y", "z")
        )
    for bond in first["bonds"]:
        assert 0 <= bond["begin_atom_index"] < atom_count
        assert 0 <= bond["end_atom_index"] < atom_count


def test_conformer_labels_2d_fallback_and_keeps_indices_valid(monkeypatch):
    monkeypatch.setattr(chemistry.AllChem, "EmbedMolecule", lambda *args: -1)

    result = chemistry.generate_conformer("CCO", allow_2d_fallback=True)

    assert result["method"] == "2D"
    assert result["status"] == "fallback"
    assert result["is_3d"] is False
    atom_count = len(result["atoms"])
    assert all(
        0 <= bond["begin_atom_index"] < atom_count
        and 0 <= bond["end_atom_index"] < atom_count
        for bond in result["bonds"]
    )


def test_invalid_smiles_errors_are_clear():
    with pytest.raises(ValueError, match="Invalid SMILES"):
        chemistry.molecular_identity("not a smiles")
