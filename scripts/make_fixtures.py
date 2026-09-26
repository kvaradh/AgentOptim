"""Regenerate fixtures.json.

Run this once the inherited XGBoost models are in `models/` so the fixtures
carry real numbers. Until then it emits surrogate values and says so in the
file, which is what lets the frontend and the agent prompts be built first.

    python -m scripts.make_fixtures
"""

from __future__ import annotations

import json
from pathlib import Path

from core.contract import canonical, normalize
from core.oracles import provenance, score

MOLECULES = (
    ("haloperidol", "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1"),
    ("risperidone", "Cc1nc2n(c(=O)c1CCN1CCC(c3noc4cc(F)ccc34)CC1)CCCC2"),
    ("chlorpromazine", "CN(C)CCCN1c2ccccc2Sc2ccc(Cl)cc21"),
    ("aripiprazole", "Clc1cccc(Cl)c1N1CCN(CCCCOc2ccc3c(c2)CCC(=O)N3)CC1"),
    ("caffeine", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"),
)

OUT_PATH = Path("fixtures.json")


def build() -> dict:
    prov = provenance()
    note = (
        "Five real molecules in the exact format core.contract.score() returns, for "
        "building the frontend and the agent prompts before the oracles are wired."
    )
    if prov["any_surrogate"]:
        note += (
            " Generated with the SURROGATE backend: directionally sane, not calibrated. "
            "Regenerate with scripts/make_fixtures.py once the inherited models are in place."
        )
    fixtures = {"_comment": note, "provenance": prov, "molecules": []}
    for name, smiles in MOLECULES:
        raw = score(smiles)
        fixtures["molecules"].append(
            {
                "name": name,
                "smiles": canonical(smiles),
                "raw": {k: round(v, 4) for k, v in raw.items() if k != "provenance"},
                "normalised": {k: round(v, 4) for k, v in normalize(raw).items()},
            }
        )
    return fixtures


if __name__ == "__main__":
    OUT_PATH.write_text(json.dumps(build(), indent=2))
    print(f"wrote {OUT_PATH} ({len(MOLECULES)} molecules)")
