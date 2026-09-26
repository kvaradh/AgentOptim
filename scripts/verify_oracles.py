#!/usr/bin/env python3

import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from core.contract import score


def main():
    fixtures = json.loads((REPOSITORY_ROOT / "fixtures.json").read_text())[:3]
    for fixture in fixtures:
        actual = score(fixture["smiles"])
        for key, tolerance in (("affinity", 0.0001), ("sa", 0.0001)):
            expected = fixture["scores"][key]
            if abs(actual[key] - expected) > tolerance:
                raise AssertionError(
                    f"{fixture['name']} {key} drifted: "
                    f"{actual[key]} != {expected}"
                )
        print(
            f"PASS: {fixture['name']} "
            f"DRD2={actual['affinity']:.3f} SA={actual['sa']:.2f}"
        )


if __name__ == "__main__":
    main()
