#!/usr/bin/env python3

import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from core.runner import load_run
from core.scoring import score_admet
from mission_control.analysis import compile_mission, validate_mission
from mission_control.intake import resolve_compound
from mission_control.presentation import build_workspace_html


def verify_model_path() -> None:
    fixtures = json.loads((REPOSITORY_ROOT / "fixtures.json").read_text())[:3]
    for fixture in fixtures:
        actual, _ = score_admet(fixture["smiles"])
        for key in ("solubility", "bbb", "herg"):
            expected = fixture["scores"][key]
            if abs(actual[key] - expected) > 0.0001:
                raise AssertionError(
                    f"{fixture['name']} {key} drifted: "
                    f"{actual[key]} != {expected}"
                )
    print("PASS: inherited 2054-feature property-model path matches fixtures")


def verify_mission_control() -> None:
    mission = compile_mission(load_run(), resolve_compound("Haloperidol"))
    validate_mission(mission)
    if len(mission["rounds"]) != 5:
        raise AssertionError("Mission does not contain five rounds")
    if any(len(round_value["candidates"]) != 6 for round_value in mission["rounds"]):
        raise AssertionError("A mission round does not contain six candidates")
    if any(
        next(
            candidate
            for candidate in round_value["candidates"]
            if candidate["candidate_id"] == round_value["winner_id"]
        )["vetoed"]
        for round_value in mission["rounds"]
    ):
        raise AssertionError("A mission winner violates the hERG gate")
    if any(
        agent["confidence_score"] is not None
        and not 0 <= agent["confidence_score"] <= 1
        for round_value in mission["rounds"]
        for agent in round_value["agents"]
    ):
        raise AssertionError("An agent confidence score is outside zero to one")
    print("PASS: five-round decision contract, safety gate, and confidence validate")

    workspace = build_workspace_html(mission)
    required = (
        'data-testid="nav-mission"',
        'data-testid="nav-agents"',
        'data-testid="nav-optimize"',
        'data-testid="nav-decisions"',
        'data-testid="nav-body-map"',
        'data-testid="nav-validate"',
        'data-testid="nav-evidence"',
        'data-testid="conformer-canvas"',
        'data-testid="export-receipt"',
        'id="mission-payload"',
    )
    missing = [marker for marker in required if marker not in workspace]
    if missing:
        raise AssertionError(f"Workspace is missing markers: {missing}")
    if "fetch(" in workspace or "XMLHttpRequest" in workspace:
        raise AssertionError("Offline workspace contains a network runtime")
    print("PASS: offline seven-view workspace and embedded exports are present")


def main() -> None:
    verify_model_path()
    verify_mission_control()


if __name__ == "__main__":
    main()
