"""Deterministic RDKit utilities used by the mission-control interface."""

from __future__ import annotations

import hashlib
import math

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdDepictor, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D


DEFAULT_CONFORMER_SEED = 0xC0FFEE


def _molecule_from_smiles(smiles: str) -> Chem.Mol:
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a non-empty string")

    molecule = Chem.MolFromSmiles(smiles.strip())
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return molecule


def canonicalize_smiles(smiles: str) -> str:
    """Return RDKit's isomeric canonical SMILES for a valid structure."""

    molecule = _molecule_from_smiles(smiles)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def molecule_id(smiles: str) -> str:
    """Return a stable, structure-derived identifier."""

    canonical_smiles = canonicalize_smiles(smiles)
    digest = hashlib.sha256(canonical_smiles.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def molecular_identity(smiles: str) -> dict[str, str | float]:
    """Return canonical identity and basic formula/mass properties."""

    molecule = _molecule_from_smiles(smiles)
    canonical_smiles = Chem.MolToSmiles(
        molecule, canonical=True, isomericSmiles=True
    )
    return {
        "canonical_smiles": canonical_smiles,
        "formula": rdMolDescriptors.CalcMolFormula(molecule),
        "molecular_weight": float(Descriptors.MolWt(molecule)),
        "molecule_id": (
            "sha256:"
            + hashlib.sha256(canonical_smiles.encode("utf-8")).hexdigest()
        ),
    }


def morgan_tanimoto_similarity(
    first_smiles: str,
    second_smiles: str,
    *,
    radius: int = 2,
    n_bits: int = 2048,
) -> float:
    """Calculate bounded Tanimoto similarity between Morgan fingerprints."""

    if not isinstance(radius, int) or radius < 0:
        raise ValueError("radius must be a non-negative integer")
    if not isinstance(n_bits, int) or n_bits <= 0:
        raise ValueError("n_bits must be a positive integer")

    first = _molecule_from_smiles(first_smiles)
    second = _molecule_from_smiles(second_smiles)
    first_fp = AllChem.GetMorganFingerprintAsBitVect(
        first, radius=radius, nBits=n_bits
    )
    second_fp = AllChem.GetMorganFingerprintAsBitVect(
        second, radius=radius, nBits=n_bits
    )
    similarity = float(DataStructs.TanimotoSimilarity(first_fp, second_fp))
    # RDKit returns a bounded value; clamp tiny floating-point excursions.
    return min(1.0, max(0.0, similarity))


def depict_svg(
    smiles: str,
    *,
    width: int = 360,
    height: int = 260,
) -> str:
    """Render a valid molecule as a self-contained SVG string."""

    if not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive integer")
    if not isinstance(height, int) or height <= 0:
        raise ValueError("height must be a positive integer")

    molecule = _molecule_from_smiles(smiles)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    if "<svg" not in svg or "</svg>" not in svg:
        raise RuntimeError("RDKit did not produce a complete SVG depiction")
    return svg


def generate_conformer(
    smiles: str,
    *,
    optimize_uff: bool = True,
    allow_2d_fallback: bool = True,
    random_seed: int = DEFAULT_CONFORMER_SEED,
) -> dict[str, object]:
    """Generate deterministic ETKDGv3 coordinates and JSON-friendly topology.

    Hydrogens are explicit in the returned atom and bond tables. If 3D
    embedding fails and fallback is enabled, deterministic 2D coordinates are
    returned with ``method="2D"`` and ``status="fallback"``.
    """

    if not isinstance(random_seed, int):
        raise ValueError("random_seed must be an integer")

    input_molecule = _molecule_from_smiles(smiles)
    canonical_smiles = Chem.MolToSmiles(
        input_molecule, canonical=True, isomericSmiles=True
    )
    # Reparse canonical SMILES so equivalent input spellings share atom order
    # as well as the same embedding seed.
    base_molecule = Chem.MolFromSmiles(canonical_smiles)
    molecule = Chem.AddHs(base_molecule)

    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.numThreads = 1
    params.clearConfs = True

    try:
        embed_status = int(AllChem.EmbedMolecule(molecule, params))
    except (RuntimeError, ValueError):
        embed_status = -1

    if embed_status == 0:
        method = "ETKDGv3"
        status = "success"
        optimization_status = "disabled"
        if optimize_uff:
            if AllChem.UFFHasAllMoleculeParams(molecule):
                try:
                    uff_status = int(
                        AllChem.UFFOptimizeMolecule(
                            molecule, confId=0, maxIters=500
                        )
                    )
                except (RuntimeError, ValueError):
                    optimization_status = "failed"
                else:
                    method = "ETKDGv3+UFF"
                    optimization_status = (
                        "converged" if uff_status == 0 else "not_converged"
                    )
            else:
                optimization_status = "unsupported"
    else:
        if not allow_2d_fallback:
            raise RuntimeError("ETKDGv3 could not generate a 3D conformer")
        molecule.RemoveAllConformers()
        rdDepictor.Compute2DCoords(molecule, canonOrient=True, clearConfs=True)
        method = "2D"
        status = "fallback"
        optimization_status = "not_applicable"

    molecule = Chem.RemoveHs(molecule)
    conformer = molecule.GetConformer()
    is_3d = bool(conformer.Is3D()) and method != "2D"
    atoms: list[dict[str, object]] = []
    for atom in molecule.GetAtoms():
        index = atom.GetIdx()
        position = conformer.GetAtomPosition(index)
        coordinates = (float(position.x), float(position.y), float(position.z))
        if not all(math.isfinite(value) for value in coordinates):
            raise RuntimeError("Conformer generation produced non-finite coordinates")
        atoms.append(
            {
                "index": index,
                "element": atom.GetSymbol(),
                "atomic_number": atom.GetAtomicNum(),
                "x": coordinates[0],
                "y": coordinates[1],
                "z": coordinates[2],
            }
        )

    atom_count = len(atoms)
    bonds: list[dict[str, object]] = []
    for bond in molecule.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if not (0 <= begin < atom_count and 0 <= end < atom_count):
            raise RuntimeError("Conformer topology contains an invalid bond index")
        bonds.append(
            {
                "index": bond.GetIdx(),
                "begin_atom_index": begin,
                "end_atom_index": end,
                "order": float(bond.GetBondTypeAsDouble()),
                "type": str(bond.GetBondType()),
            }
        )

    return {
        "canonical_smiles": canonical_smiles,
        "method": method,
        "status": status,
        "optimization_status": optimization_status,
        "is_3d": is_3d,
        "atoms": atoms,
        "bonds": bonds,
    }


# Readable aliases for callers that prefer noun-oriented APIs.
canonical_identity = molecular_identity
structural_similarity = morgan_tanimoto_similarity
svg_depiction = depict_svg
conformer = generate_conformer


__all__ = [
    "DEFAULT_CONFORMER_SEED",
    "canonical_identity",
    "canonicalize_smiles",
    "conformer",
    "depict_svg",
    "generate_conformer",
    "molecular_identity",
    "molecule_id",
    "morgan_tanimoto_similarity",
    "structural_similarity",
    "svg_depiction",
]
