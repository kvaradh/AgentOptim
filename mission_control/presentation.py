"""Build the self-contained, offline Agent Perry workspace."""

from __future__ import annotations

import base64
import json
from pathlib import Path

from .analysis import validate_mission
from .exports import candidate_csv, decision_receipt_html, mission_json


ASSET_DIRECTORY = Path(__file__).with_name("assets")
MISSION_PAYLOAD_BUDGET_BYTES = 750 * 1024


def _base64_text(value: str) -> str:
    return base64.b64encode(value.encode("utf-8")).decode("ascii")


def build_workspace_html(mission: dict) -> str:
    """Return one deterministic, self-contained HTML workspace.

    The finalized mission is validated before any serialization. Mission and
    export data are embedded as base64-only text so intake fields, molecule
    names, SMILES, and receipt markup can never enter executable page markup.
    """

    validate_mission(mission)

    template = (ASSET_DIRECTORY / "workspace.html").read_text(encoding="utf-8")
    styles = (ASSET_DIRECTORY / "styles.css").read_text(encoding="utf-8")
    application = (ASSET_DIRECTORY / "app.js").read_text(encoding="utf-8")

    compact_mission = json.dumps(
        mission,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    compact_bytes = compact_mission.encode("utf-8")
    if len(compact_bytes) > MISSION_PAYLOAD_BUDGET_BYTES:
        raise ValueError(
            "Compact mission payload exceeds the 750 KB presentation budget"
        )

    replacements = {
        "__MISSION_PAYLOAD_BASE64__": base64.b64encode(compact_bytes).decode(
            "ascii"
        ),
        "__MISSION_JSON_BASE64__": _base64_text(mission_json(mission)),
        "__CANDIDATE_CSV_BASE64__": _base64_text(candidate_csv(mission)),
        "__DECISION_RECEIPT_BASE64__": _base64_text(
            decision_receipt_html(mission)
        ),
        "/*__WORKSPACE_STYLES__*/": styles,
        "/*__WORKSPACE_APPLICATION__*/": application,
    }
    rendered = template
    for marker, value in replacements.items():
        if rendered.count(marker) != 1:
            raise ValueError(f"Workspace asset marker is missing or repeated: {marker}")
        rendered = rendered.replace(marker, value)
    return rendered
