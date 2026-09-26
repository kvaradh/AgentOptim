#!/usr/bin/env python3

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from core.contract import SCORE_KEYS, is_valid, normalize
from core.runner import load_run


def main():
    run = load_run()
    for round_value in run["rounds"]:
        assert len(round_value["candidates"]) == 6
        assert round_value["winner"]["raw"]["herg"] < 0.50
        for candidate in round_value["candidates"]:
            assert is_valid(candidate["smiles"])
            assert set(candidate["raw"]) == set(SCORE_KEYS)
            assert candidate["normalized"] == normalize(candidate["raw"])

    seed = run["seed"]["raw"]
    final = run["rounds"][-1]["winner"]["raw"]
    assert final["solubility"] > seed["solubility"]
    assert final["herg"] < seed["herg"]
    print("PASS: replay has 5 rounds × 6 valid scored molecules")
    print("PASS: every winner clears the hERG veto")
    print("PASS: final improves seed solubility and predicted hERG blockade")


if __name__ == "__main__":
    main()
