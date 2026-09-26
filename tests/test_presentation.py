import base64
import copy
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from core.runner import load_run
from mission_control.analysis import compile_mission, validate_mission
from mission_control.exports import candidate_csv, decision_receipt_html, mission_json
from mission_control.intake import resolve_compound
from mission_control.presentation import (
    MISSION_PAYLOAD_BUDGET_BYTES,
    build_workspace_html,
)


PAYLOAD_PATTERN = re.compile(
    r'<script id="(?P<id>[^"]+)" type="application/octet-stream">'
    r"(?P<payload>[A-Za-z0-9+/=]+)</script>"
)


@pytest.fixture(scope="module")
def mission():
    value = compile_mission(load_run(), resolve_compound("Haloperidol"))
    validate_mission(value)
    return value


@pytest.fixture(scope="module")
def workspace_html(mission):
    return build_workspace_html(mission)


def embedded_payloads(html):
    return {
        match.group("id"): base64.b64decode(match.group("payload"), validate=True)
        for match in PAYLOAD_PATTERN.finditer(html)
    }


def test_workspace_validates_before_rendering(mission):
    broken = copy.deepcopy(mission)
    broken["target"]["id"] = "NOT_DRD2"

    with pytest.raises(ValueError, match="DRD2|digest"):
        build_workspace_html(broken)


def test_workspace_is_one_offline_seven_view_spa(workspace_html):
    assert workspace_html.startswith("<!doctype html>")
    assert "Content-Security-Policy" in workspace_html
    for view in (
        "mission",
        "agents",
        "optimize",
        "decisions",
        "body-map",
        "validate",
        "evidence",
    ):
        assert f'data-view-panel="{view}"' in workspace_html
        assert f'data-testid="nav-{view}"' in workspace_html

    assert "<link" not in workspace_html
    assert not re.search(
        r"<(?:script|img|source|iframe)[^>]+\bsrc=[\"']https?://",
        workspace_html,
        re.IGNORECASE,
    )
    assert not re.search(
        r"<link[^>]+\bhref=[\"']https?://",
        workspace_html,
        re.IGNORECASE,
    )
    assert "fetch(" not in workspace_html
    assert "XMLHttpRequest" not in workspace_html
    assert "WebSocket(" not in workspace_html
    assert "innerHTML" not in workspace_html
    assert "insertAdjacentHTML" not in workspace_html


def test_workspace_embeds_exact_base64_mission_and_exports(mission, workspace_html):
    payloads = embedded_payloads(workspace_html)
    assert set(payloads) == {
        "mission-payload",
        "export-json-payload",
        "export-csv-payload",
        "export-receipt-payload",
    }
    decoded_mission = json.loads(payloads["mission-payload"].decode("utf-8"))
    assert decoded_mission == mission
    assert payloads["export-json-payload"].decode("utf-8") == mission_json(mission)
    assert payloads["export-csv-payload"].decode("utf-8") == candidate_csv(mission)
    assert (
        payloads["export-receipt-payload"].decode("utf-8")
        == decision_receipt_html(mission)
    )


def test_initial_mission_payload_stays_within_budget(workspace_html):
    payloads = embedded_payloads(workspace_html)
    assert len(payloads["mission-payload"]) <= MISSION_PAYLOAD_BUDGET_BYTES
    encoded = next(
        match.group("payload")
        for match in PAYLOAD_PATTERN.finditer(workspace_html)
        if match.group("id") == "mission-payload"
    )
    assert len(encoded.encode("ascii")) <= MISSION_PAYLOAD_BUDGET_BYTES


def test_workspace_has_required_scientific_and_audit_content(workspace_html):
    required_labels = (
        "target-rationale-code",
        "Why this seed",
        "Four roles. One inspectable decision.",
        "The Agent Perry loop",
        "The benefit is auditability.",
        "Target rationale",
        "Resolved identity",
        "Optimization hypothesis",
        "Guided replay",
        "Six scored proposals",
        "Heuristic confidence",
        "Complete round ledger",
        "Predicted hERG safety gate",
        "Body Map",
        "Model-derived directional hypotheses",
        "Visual conformer—not docking",
        "Wet-lab runway",
        "Unperformed",
        "plan for ${candidate.candidate_id}",
        "Evidence & exports",
        "Product effect",
        "Knowledge limits",
        "Decision receipt",
    )
    for label in required_labels:
        assert label in workspace_html

    for test_id in (
        "round-rail",
        "replay-controls",
        "body-map",
        "conformer-canvas",
        "export-json",
        "export-csv",
        "export-receipt",
    ):
        assert f'data-testid="{test_id}"' in workspace_html
    assert len(re.findall(r'\sid="conformer-canvas"', workspace_html)) == 1
    assert "Guided 25-second tour of recorded rounds" in workspace_html
    assert "Resume narrative replay" in workspace_html
    assert "▶ Resume" in workspace_html


def test_raw_identity_and_smiles_never_enter_executable_markup(mission, workspace_html):
    assert mission["intake"]["name"] not in workspace_html
    assert mission["intake"]["query"] not in workspace_html
    assert mission["seed"]["smiles"] not in workspace_html
    assert mission["rounds"][0]["candidates"][1]["smiles"] not in workspace_html


def test_malicious_intake_is_only_present_inside_base64_payload():
    attack_name = '</script><img src=x onerror="alert(99173)">'
    attack_query = '"><script>globalThis.pwned=true</script>'
    intake = resolve_compound("Haloperidol")
    intake["name"] = attack_name
    intake["query"] = attack_query
    malicious_mission = compile_mission(load_run(), intake)

    html = build_workspace_html(malicious_mission)
    payload = json.loads(embedded_payloads(html)["mission-payload"].decode("utf-8"))

    assert payload["intake"]["name"] == attack_name
    assert payload["intake"]["query"] == attack_query
    assert attack_name not in html
    assert attack_query not in html
    assert "alert(99173)" not in html
    assert "globalThis.pwned" not in html


def test_workspace_generation_is_deterministic(mission):
    assert build_workspace_html(mission) == build_workspace_html(mission)


def test_workspace_javascript_has_valid_syntax():
    repository = Path(__file__).resolve().parents[1]
    script = repository / "mission_control" / "assets" / "app.js"
    bundled_node = Path(
        "/Applications/ChatGPT.app/Contents/Resources/cua_node/bin/node"
    )
    node = shutil.which("node")
    if node is None and bundled_node.exists():
        node = str(bundled_node)
    if node is None:
        pytest.skip("Node.js is not available for the syntax check")

    completed = subprocess.run(
        [node, "--check", str(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
