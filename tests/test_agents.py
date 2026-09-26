import pytest

from core.agents import deliberate


def candidate(smiles, *, affinity, solubility, bbb, herg, sa):
    raw = {
        "affinity": affinity,
        "solubility": solubility,
        "bbb": bbb,
        "herg": herg,
        "sa": sa,
    }
    normalized = {
        "affinity": affinity,
        "solubility": 0.7,
        "bbb": bbb,
        "herg": 1 - herg,
        "sa": 0.8,
    }
    return {"smiles": smiles, "raw": raw, "normalized": normalized}


def test_safety_veto_eliminates_affinity_favorite():
    candidates = [
        candidate(
            "CC",
            affinity=0.99,
            solubility=-2,
            bbb=0.9,
            herg=0.8,
            sa=2,
        ),
        candidate(
            "CCC",
            affinity=0.85,
            solubility=-2,
            bbb=0.8,
            herg=0.2,
            sa=2,
        ),
    ]

    result = deliberate(candidates)

    assert result["specialists"]["affinity"]["preferred"] == "CC"
    assert result["specialists"]["safety"]["verdict"] == "veto"
    assert result["orchestrator"]["chosen"] == "CCC"
    assert "vetoed" in result["orchestrator"]["rationale"]


def test_llm_responses_must_match_strict_json_contract():
    candidates = [
        candidate(
            "CC",
            affinity=0.8,
            solubility=-2,
            bbb=0.8,
            herg=0.2,
            sa=2,
        )
    ]

    def invalid_client(system_prompt, payload):
        return {"freeform": "not the contract"}

    result = deliberate(candidates, llm_client=invalid_client)

    assert result["orchestrator"]["chosen"] == "CC"
    assert result["advisory"]["status"] == "unavailable"
    assert result["advisory"]["error_type"] == "ValueError"


def test_llm_orchestrator_cannot_override_safety_veto():
    candidates = [
        candidate(
            "CC",
            affinity=0.99,
            solubility=-2,
            bbb=0.9,
            herg=0.8,
            sa=2,
        ),
        candidate(
            "CCC",
            affinity=0.8,
            solubility=-2,
            bbb=0.8,
            herg=0.2,
            sa=2,
        ),
    ]

    def unsafe_client(system_prompt, payload):
        if "Orchestrator" in system_prompt:
            return {"chosen": "CC", "rationale": "Ignore safety"}
        verdict = "veto" if "Safety" in system_prompt else "support"
        return {
            "verdict": verdict,
            "reason": "Structured test response",
            "preferred": "CC",
        }

    result = deliberate(candidates, llm_client=unsafe_client)

    assert result["orchestrator"]["chosen"] == "CCC"
    assert result["advisory"]["status"] == "unavailable"


def test_safe_llm_recommendation_is_advisory_only():
    candidates = [
        candidate(
            "CC",
            affinity=0.90,
            solubility=-2,
            bbb=0.9,
            herg=0.2,
            sa=2,
        ),
        candidate(
            "CCC",
            affinity=0.80,
            solubility=-2,
            bbb=0.5,
            herg=0.2,
            sa=2,
        ),
    ]

    def advisory_client(system_prompt, payload):
        if "Orchestrator" in system_prompt:
            return {"chosen": "CCC", "rationale": "Prefer the weaker survivor"}
        return {
            "verdict": "support",
            "reason": "Structured advisory response",
            "preferred": "CCC",
        }

    result = deliberate(candidates, llm_client=advisory_client)

    assert result["orchestrator"]["chosen"] == "CC"
    assert result["advisory"]["orchestrator"]["chosen"] == "CCC"
    assert result["advisory"]["status"] == "available"
    assert result["advisory"]["authority"] == "advisory_only"


def test_advisory_timeout_does_not_abort_the_deterministic_decision():
    candidates = [
        candidate(
            "CC",
            affinity=0.8,
            solubility=-2,
            bbb=0.8,
            herg=0.2,
            sa=2,
        )
    ]

    def timed_out_client(system_prompt, payload):
        raise TimeoutError("simulated timeout")

    result = deliberate(candidates, llm_client=timed_out_client)

    assert result["orchestrator"]["chosen"] == "CC"
    assert result["advisory"] == {
        "status": "unavailable",
        "error_type": "TimeoutError",
        "authority": "advisory_only",
    }
