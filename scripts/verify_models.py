"""The verification gate. Run this before building anything on the inherited models.

The failure mode this exists to catch: the saved XGBoost models take a feature
vector of a specific shape and column order. Hand them a vector that is merely
plausible -- right length, wrong descriptor order, different Morgan radius, or
counts where the model was trained on bits -- and they return confident floats
that mean nothing. No exception is raised. No test fails. The demo runs, the
numbers move, and every conclusion drawn from them is void.

So this script does not ask "does it run". It asks "does it reproduce a number
somebody already wrote down".

    python -m scripts.verify_models --expected expected_predictions.json

`expected_predictions.json` is the artefact to copy out of the source notebook
before the clock starts:

    {"solubility": {"CCO": -0.77, ...},
     "bbb":        {"CCO":  0.93, ...},
     "herg":       {"CCO":  0.04, ...}}

Exit status is 0 only if every prediction matches within tolerance. If it is
not 0, do not build on these models: point ADMET_FEATURIZER at the source
repo's own featurization function and run this again.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core import admet

DEFAULT_TOLERANCE = 0.02


def check_featurization() -> tuple[bool, str]:
    """Shape and determinism of the feature vector, before any model is loaded."""
    try:
        features = admet.featurize("CCO")
    except Exception as exc:
        return False, f"featurize() raised {type(exc).__name__}: {exc}"

    expected_width = admet.MORGAN_BITS + len(admet.DESCRIPTOR_ORDER)
    width = features.shape[1]
    if width != expected_width:
        return False, (
            f"feature width {width}, expected {expected_width} "
            f"({admet.MORGAN_BITS} fingerprint bits + "
            f"{len(admet.DESCRIPTOR_ORDER)} descriptors). "
            f"If the models were trained on a different layout, set ADMET_FEATURIZER "
            f"rather than editing this."
        )
    if (admet.featurize("CCO") != features).any():
        return False, "featurize() is not deterministic for the same input"
    return True, (
        f"feature vector {width} wide: Morgan r={admet.MORGAN_RADIUS} "
        f"x {admet.MORGAN_BITS} bits ++ {list(admet.DESCRIPTOR_ORDER)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the inherited ADMET models")
    parser.add_argument(
        "--expected",
        type=Path,
        help="JSON of {task: {smiles: value}} copied from the source notebook",
    )
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args()

    print("=" * 72)
    ok, message = check_featurization()
    print(f"[{'PASS' if ok else 'FAIL'}] featurization: {message}")
    if not ok:
        return 1

    try:
        backend = admet.XGBoostBackend()
    except Exception as exc:
        print(f"[FAIL] model loading: {type(exc).__name__}: {exc}")
        print(
            f"\nNo inherited models found in {admet.MODELS_DIR}/. The loop will run on "
            f"the labelled surrogate backend, which is fine for building the agents, "
            f"the edit engine and the frontend -- and is NOT fine to present as a "
            f"trained model with an ROC-AUC attached to it."
        )
        return 1
    print(f"[PASS] model loading: three models from {backend.models_dir}/")
    if backend.metrics:
        print(f"       reported metrics: {backend.metrics}")

    if not args.expected:
        print(
            "\n[SKIP] prediction check: no --expected file given.\n"
            "       Models load and featurization is the documented shape, but nothing\n"
            "       here shows the predictions are the ones the source notebook produced.\n"
            "       This is the check that actually matters. Export three test-set\n"
            "       molecules per task from the notebook and pass --expected."
        )
        return 1

    expected = json.loads(args.expected.read_text())
    failures = 0
    checked = 0
    for task, cases in expected.items():
        method = getattr(backend, task, None)
        if method is None:
            print(f"[FAIL] unknown task {task!r} (expected one of solubility, bbb, herg)")
            failures += 1
            continue
        for smiles, want in cases.items():
            checked += 1
            try:
                got = method(smiles)
            except Exception as exc:
                print(f"[FAIL] {task} {smiles}: raised {type(exc).__name__}: {exc}")
                failures += 1
                continue
            delta = abs(got - float(want))
            if delta <= args.tolerance:
                print(f"[PASS] {task:11s} {smiles:40s} {got:+.4f} (want {float(want):+.4f})")
            else:
                failures += 1
                print(
                    f"[FAIL] {task:11s} {smiles:40s} {got:+.4f} (want {float(want):+.4f}, "
                    f"off by {delta:.4f})"
                )

    print("=" * 72)
    if failures:
        print(
            f"{failures} of {checked} predictions do not match.\n\n"
            f"Do not build on these models. The most likely cause is a featurization\n"
            f"mismatch, not a corrupt model file. Import the source repo's own\n"
            f"featurization instead of reimplementing it:\n\n"
            f"    export ADMET_FEATURIZER='admet_repo.features:featurize_smiles'\n\n"
            f"then run this again."
        )
        return 1
    print(f"All {checked} predictions match within {args.tolerance}. Gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
