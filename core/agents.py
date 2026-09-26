"""Strict agent contracts with deterministic offline and optional LLM paths."""

from collections.abc import Callable
import json

from .contract import SCORE_KEYS
from .errors import NoSafeCandidateError


SPECIALIST_KEYS = ("verdict", "reason", "preferred")
ORCHESTRATOR_KEYS = ("chosen", "rationale")
HERG_VETO_THRESHOLD = 0.50

SYSTEM_PROMPTS = {
    "affinity": (
        "You are the Affinity specialist. Care only about DRD2 affinity. "
        "Return strict JSON with verdict, reason, and preferred SMILES."
    ),
    "adme": (
        "You are the ADME specialist. Balance solubility and BBB. Return "
        "strict JSON with verdict, reason, and preferred SMILES."
    ),
    "safety": (
        "You are the Safety specialist. Predicted hERG blockade at or above "
        "0.50 is a non-negotiable veto. Return strict JSON with verdict, "
        "reason, and preferred SMILES."
    ),
    "orchestrator": (
        "You are the Orchestrator. Read the three specialists, remove every "
        "safety-vetoed molecule, then choose one survivor. Return strict JSON "
        "with chosen SMILES and rationale."
    ),
}


def deliberate(
    candidates: list[dict],
    llm_client: Callable[[str, str], dict] | None = None,
) -> dict:
    """Return three specialist JSONs and one orchestrator JSON."""
    if not candidates:
        raise ValueError("Agents need at least one candidate")

    specialists = _offline_specialists(candidates)
    orchestrator = _offline_orchestrator(candidates, specialists)
    advisory = None
    if llm_client is None:
        mode = "deterministic offline agents"
    else:
        try:
            payload = json.dumps(candidates)
            advisory_specialists = {
                role: _validate_specialist(
                    llm_client(SYSTEM_PROMPTS[role], payload),
                    candidates,
                    role,
                )
                for role in ("affinity", "adme", "safety")
            }
            orchestration_payload = json.dumps(
                {"candidates": candidates, "specialists": advisory_specialists}
            )
            advisory_orchestrator = _validate_orchestrator(
                llm_client(
                    SYSTEM_PROMPTS["orchestrator"],
                    orchestration_payload,
                ),
                candidates,
            )
            advisory = {
                "status": "available",
                "specialists": advisory_specialists,
                "orchestrator": advisory_orchestrator,
                "authority": "advisory_only",
            }
        except Exception as error:
            advisory = {
                "status": "unavailable",
                "error_type": type(error).__name__,
                "authority": "advisory_only",
            }
        mode = "deterministic policy with LLM advisory"

    result = {
        "mode": mode,
        "specialists": specialists,
        "orchestrator": orchestrator,
    }
    if advisory is not None:
        result["advisory"] = advisory
    return result


def _offline_specialists(candidates: list[dict]) -> dict:
    affinity_pick = max(
        candidates,
        key=lambda item: (item["normalized"]["affinity"], item["smiles"]),
    )
    adme_pick = max(
        candidates,
        key=lambda item: (
            (
                item["normalized"]["solubility"]
                + item["normalized"]["bbb"]
            )
            / 2,
            item["smiles"],
        ),
    )
    safe = [
        item
        for item in candidates
        if item["raw"]["herg"] < HERG_VETO_THRESHOLD
    ]
    safety_pick = min(
        safe or candidates,
        key=lambda item: (item["raw"]["herg"], item["smiles"]),
    )
    vetoed = [
        item
        for item in candidates
        if item["raw"]["herg"] >= HERG_VETO_THRESHOLD
    ]

    return {
        "affinity": {
            "verdict": "support",
            "reason": (
                "Highest predicted DRD2 affinity in this round "
                f"({affinity_pick['raw']['affinity']:.3f})."
            ),
            "preferred": affinity_pick["smiles"],
        },
        "adme": {
            "verdict": (
                "object"
                if adme_pick["smiles"] != affinity_pick["smiles"]
                else "support"
            ),
            "reason": (
                "Prefers the strongest combined predicted solubility and BBB "
                "profile, even when that conflicts with affinity."
            ),
            "preferred": adme_pick["smiles"],
        },
        "safety": {
            "verdict": "veto" if vetoed else "support",
            "reason": (
                f"Vetoes {len(vetoed)} candidate(s) at or above the 50% "
                "predicted hERG-blockade policy threshold."
                if vetoed
                else "No candidate crosses the predicted hERG veto threshold."
            ),
            "preferred": safety_pick["smiles"],
        },
    }


def _offline_orchestrator(candidates: list[dict], specialists: dict) -> dict:
    chosen = expected_official_candidate(candidates)
    survivors = [item for item in candidates if not is_vetoed(item)]
    affinity_preferred = specialists["affinity"]["preferred"]
    tradeoff = (
        "The affinity specialist's first choice was vetoed, so the "
        "orchestrator selected the strongest surviving trade-off."
        if affinity_preferred not in {item["smiles"] for item in survivors}
        else "The affinity preference survived the safety veto and offered "
        "the strongest overall trade-off."
    )
    return {"chosen": chosen["smiles"], "rationale": tradeoff}


def is_vetoed(candidate: dict) -> bool:
    return candidate["raw"]["herg"] >= HERG_VETO_THRESHOLD


def candidate_utility(candidate: dict) -> float:
    values = candidate["normalized"]
    return (
        0.35 * values["affinity"]
        + 0.25 * values["solubility"]
        + 0.25 * values["bbb"]
        + 0.10 * values["herg"]
        + 0.05 * values["sa"]
    )


def official_pool(candidates: list[dict]) -> list[dict]:
    survivors = [item for item in candidates if not is_vetoed(item)]
    if not survivors:
        raise NoSafeCandidateError(
            "Safety vetoed every proposal; no molecule can advance"
        )
    novel_survivors = [
        item for item in survivors if not item.get("incumbent", False)
    ]
    return novel_survivors or survivors


def expected_official_candidate(candidates: list[dict]) -> dict:
    return max(
        official_pool(candidates),
        key=lambda item: (candidate_utility(item), item["smiles"]),
    )


def _validate_specialist(
    value: dict,
    candidates: list[dict],
    role: str,
) -> dict:
    if set(value) != set(SPECIALIST_KEYS):
        raise ValueError("Specialist response does not match the JSON contract")
    if value["verdict"] not in {"support", "object", "veto"}:
        raise ValueError("Specialist returned an invalid verdict")
    if value["preferred"] not in {item["smiles"] for item in candidates}:
        raise ValueError("Specialist preferred an unknown molecule")
    has_veto = any(
        item["raw"]["herg"] >= HERG_VETO_THRESHOLD
        for item in candidates
    )
    if role == "safety" and has_veto and value["verdict"] != "veto":
        raise ValueError("Safety agent failed to apply the hERG veto")
    return value


def _validate_orchestrator(value: dict, candidates: list[dict]) -> dict:
    if set(value) != set(ORCHESTRATOR_KEYS):
        raise ValueError("Orchestrator response does not match the JSON contract")
    if value["chosen"] not in {item["smiles"] for item in candidates}:
        raise ValueError("Orchestrator chose an unknown molecule")
    chosen = next(
        item for item in candidates if item["smiles"] == value["chosen"]
    )
    if chosen["raw"]["herg"] >= HERG_VETO_THRESHOLD:
        raise ValueError("Orchestrator chose a safety-vetoed molecule")
    return value
