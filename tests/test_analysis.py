import copy
import hashlib
import json

import pytest

from core.contract import normalize
from core.runner import load_run, run_optimization
from mission_control.analysis import (
    _events,
    compile_mission,
    confidence_from_ranking,
    validate_mission,
    weighted_utility,
)
from mission_control.exports import (
    candidate_csv,
    decision_receipt_html,
    mission_json,
)
from mission_control.intake import resolve_compound


@pytest.fixture(scope="module")
def mission():
    return compile_mission(load_run(), resolve_compound("Haloperidol"))


def test_compiler_preserves_the_recorded_mission_contract(mission):
    assert mission["schema_version"] == 3
    assert len(mission["rounds"]) == 5
    assert all(len(round_value["candidates"]) == 6 for round_value in mission["rounds"])
    assert all(
        not next(
            candidate
            for candidate in round_value["candidates"]
            if candidate["candidate_id"] == round_value["winner_id"]
        )["vetoed"]
        for round_value in mission["rounds"]
    )


def test_cached_mission_has_the_expected_safety_and_rollback_story(mission):
    summary = mission["summary"]

    assert summary["candidate_occurrences"] == 30
    assert summary["unique_molecules"] == 19
    assert summary["vetoed_occurrences"] == 13
    assert summary["unique_winners"] == 3
    assert summary["affinity_only_unsafe_rounds"] == 3
    assert summary["adme_only_unsafe_rounds"] == 3
    assert summary["final_seed_similarity"] == pytest.approx(0.12048, abs=1e-5)
    assert [
        item["official_differs_from_weighted"]
        for item in mission["policy_comparison"]
    ] == [False, True, False, True, False]
    assert [
        round_value["round"]
        for round_value in mission["rounds"]
        if any(event["type"] == "rollback" for event in round_value["events"])
    ] == [3, 5]
    round_two = mission["policy_comparison"][1]
    assert round_two["affinity_only"]["score_label"] == "DRD2"
    assert round_two["adme_only"]["score_label"] == "ADME"
    assert round_two["safety_only"]["score_label"] == "hERG safety"
    assert round_two["official"]["score_label"] == "Utility"


def test_rollback_requires_a_previously_official_winner(mission):
    prior_winner_molecule_ids = set()
    rejected_repeat_checked = False

    for round_value in mission["rounds"]:
        agents = {agent["role"]: agent for agent in round_value["agents"]}
        for candidate in round_value["candidates"]:
            was_seen_before = candidate["first_seen"]["round"] < round_value["round"]
            was_prior_winner = (
                candidate["molecule_id"] in prior_winner_molecule_ids
            )
            if was_seen_before and not was_prior_winner and not candidate["incumbent"]:
                probe_events = _events(
                    round_value["round"],
                    round_value["candidates"],
                    agents,
                    candidate["candidate_id"],
                    prior_winner_molecule_ids,
                )
                assert not any(
                    event["type"] == "rollback" for event in probe_events
                )
                rejected_repeat_checked = True
                break
        winner = next(
            candidate
            for candidate in round_value["candidates"]
            if candidate["candidate_id"] == round_value["winner_id"]
        )
        prior_winner_molecule_ids.add(winner["molecule_id"])

    assert rejected_repeat_checked


def test_every_agent_has_a_reproducible_confidence_and_full_ranking(mission):
    for round_value in mission["rounds"]:
        for agent in round_value["agents"]:
            assert agent["confidence_score"] is None or 0 <= agent["confidence_score"] <= 1
            assert agent["confidence_basis"]
            assert len(agent["ranking"]) >= 1
            assert agent["evidence"]
            assert agent["context_sources"]


def test_confidence_handles_ties_and_single_eligible_candidate():
    tied, tied_basis = confidence_from_ranking(
        [
            {"candidate_id": "C1", "score": 0.5, "rank": 1},
            {"candidate_id": "C2", "score": 0.5, "rank": 2},
        ]
    )
    single, single_basis = confidence_from_ranking(
        [{"candidate_id": "C1", "score": 0.5, "rank": 1}]
    )

    assert tied == 0
    assert "tied" in tied_basis
    assert single is None
    assert "fewer than two" in single_basis


def test_body_map_uses_only_supported_properties_and_links_to_validation(mission):
    supported = {"affinity", "bbb", "herg", "solubility"}
    step_ids = {step["id"] for step in mission["wet_lab_plan"]}

    assert {item["system"] for item in mission["body_systems_map"]} == {
        "brain",
        "heart",
        "absorption",
    }
    for item in mission["body_systems_map"]:
        assert set(item["source_properties"]).issubset(supported)
        assert set(item["wet_lab_step_ids"]).issubset(step_ids)
        assert "hypothesis" in item["confidence_label"].lower()


def test_mission_digest_covers_the_entire_payload(mission):
    payload = copy.deepcopy(mission)
    digest = payload.pop("digest")
    expected = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()

    assert digest == expected


def test_hard_gate_is_inclusive_at_point_five(mission):
    broken = copy.deepcopy(mission)
    broken.pop("digest")
    candidate = broken["rounds"][0]["candidates"][4]
    candidate["raw"]["herg"] = 0.5
    candidate["normalized"] = normalize(candidate["raw"])
    candidate["weighted_utility"] = weighted_utility(candidate["normalized"])

    with pytest.raises(ValueError, match="inclusive hERG gate"):
        validate_mission(broken, require_digest=False)


def test_exports_are_complete_and_self_contained(mission):
    json_text = mission_json(mission)
    csv_text = candidate_csv(mission)
    receipt = decision_receipt_html(mission)

    assert json.loads(json_text)["digest"] == mission["digest"]
    assert csv_text.count("\n") == 31
    assert "R5C5" in csv_text
    assert "<!doctype html>" in receipt
    assert mission["digest"][:12] in receipt
    assert "Structure comparison · RDKit 2D" in receipt
    assert "Starting molecule" in receipt
    assert "Proposed molecule · R5C5" in receipt
    assert mission["seed"]["smiles"] in receipt
    assert mission["rounds"][-1]["candidates"][4]["smiles"] in receipt
    assert receipt.count('<div class="structure-canvas"><svg') == 2
    assert "Wet-lab validation runway" in receipt
    assert "<script" not in receipt


def test_receipt_omits_non_http_evidence_links(mission):
    altered = copy.deepcopy(mission)
    altered.pop("digest")
    altered["evidence_library"][0]["url"] = "javascript:alert(1)"
    altered["digest"] = hashlib.sha256(
        json.dumps(
            altered,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()

    receipt = decision_receipt_html(altered)

    assert "javascript:" not in receipt
    assert altered["evidence_library"][0]["title"] in receipt


def test_receipt_regenerates_rdkit_structures_instead_of_trusting_markup(mission):
    altered = copy.deepcopy(mission)
    altered.pop("digest")
    altered["molecules"][altered["seed"]["molecule_id"]]["depiction_svg"] = (
        "<script>alert('not trusted')</script>"
    )
    altered["digest"] = hashlib.sha256(
        json.dumps(
            altered,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()

    receipt = decision_receipt_html(altered)

    assert "<script" not in receipt
    assert receipt.count('<div class="structure-canvas"><svg') == 2


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda value: value["rounds"][0]["candidate_lookup"].update(
                {"R1C1": 5}
            ),
            "Candidate lookup",
        ),
        (
            lambda value: value["rounds"][0]["agents"][0]["ranking"].reverse(),
            "Agent ranking",
        ),
        (
            lambda value: value["decision_ledger"][0]["choice"].update(
                {"winner": "R1C1"}
            ),
            "ledger winner",
        ),
        (lambda value: value["lineage"].pop(), "lineage"),
        (
            lambda value: value["rounds"][1]["parent"].update(
                {"smiles": value["seed"]["smiles"]}
            ),
            "parent",
        ),
        (
            lambda value: value["rounds"][0]["candidates"][1].update(
                {"source_kind": "unknown"}
            ),
            "provenance",
        ),
        (
            lambda value: value["rounds"][0]["candidates"][0].update(
                {
                    "molecule_id": value["rounds"][0]["candidates"][1][
                        "molecule_id"
                    ]
                }
            ),
            "Candidate molecule identity",
        ),
        (
            lambda value: value["seed"].update(
                {"molecule_id": value["rounds"][0]["candidates"][1]["molecule_id"]}
            ),
            "Seed molecule identity",
        ),
    ],
)
def test_schema_mutations_are_rejected(mission, mutate, message):
    broken = copy.deepcopy(mission)
    broken.pop("digest")
    mutate(broken)

    with pytest.raises(ValueError, match=message):
        validate_mission(broken, require_digest=False)


def test_any_payload_mutation_breaks_the_digest(mission):
    broken = copy.deepcopy(mission)
    broken["summary"]["story"] = "A different story"

    with pytest.raises(ValueError, match="digest"):
        validate_mission(broken)


def test_finalized_validation_requires_a_digest(mission):
    broken = copy.deepcopy(mission)
    broken.pop("digest")

    with pytest.raises(ValueError, match="requires a digest"):
        validate_mission(broken)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda value: value["rounds"][0]["candidates"][0].update(
                {"similarity_seed": 2.0}
            ),
            "similarity",
        ),
        (
            lambda value: value["molecules"][
                value["seed"]["molecule_id"]
            ]["conformer"]["bonds"][0].update({"begin_atom_index": -1}),
            "bond index",
        ),
        (
            lambda value: value["decision_ledger"][0]["choice"].update(
                {"margin": 99.0}
            ),
            "ledger margin",
        ),
        (
            lambda value: value["decision_ledger"][0].update(
                {"orchestrator_pool": ["R1C1"]}
            ),
            "orchestrator pool",
        ),
        (
            lambda value: (
                value["rounds"][0]["agents"][-1].update(
                    {"reason": "Fabricated reason"}
                ),
                value["decision_ledger"][0]["why_chosen"].__setitem__(
                    0, "Fabricated reason"
                ),
                value["rounds"][0]["events"][-2].update(
                    {"detail": "Fabricated reason"}
                ),
            ),
            "Agent explanation",
        ),
    ],
)
def test_structural_validation_rejects_invalid_derived_fields(
    mission, mutate, message
):
    broken = copy.deepcopy(mission)
    broken.pop("digest")
    mutate(broken)

    with pytest.raises(ValueError, match=message):
        validate_mission(broken, require_digest=False)


def test_live_and_cached_runs_compile_to_the_same_policy_schema(mission):
    molecules = ("CC", "CCC", "CCCC", "CCO", "CCN", "CCF")

    def proposer(parent, context):
        return [parent, *[item for item in molecules if item != parent]][:6]

    def scorer(smiles):
        index = molecules.index(smiles) if smiles in molecules else 0
        return {
            "affinity": 0.90 - index * 0.04,
            "solubility": -3.0 + index * 0.1,
            "bbb": 0.85 - index * 0.03,
            "herg": 0.7 if index == 0 else 0.2,
            "sa": 2.0 + index * 0.1,
        }

    live_run = run_optimization(
        scorer=scorer,
        proposer=proposer,
        seed_smiles="CO",
        seed_name="Live test seed",
    )
    live_intake = resolve_compound("CO")
    live_mission = compile_mission(live_run, live_intake)

    assert set(live_mission) == set(mission)
    assert live_mission["safety_policy"] == mission["safety_policy"]
    assert all(
        [agent["role"] for agent in round_value["agents"]]
        == ["affinity", "adme", "safety", "orchestrator"]
        for round_value in live_mission["rounds"]
    )
    assert live_mission["target"]["rationale_status"] == "unverified"
    assert all(
        candidate["source_kind"] == "unknown"
        for round_value in live_mission["rounds"]
        for candidate in round_value["candidates"][1:]
    )


def test_same_input_has_identical_science_with_any_advisory_state():
    molecules = ("CC", "CCC", "CCCC", "CCO", "CCN", "CCF")

    def proposer(parent, context):
        return [parent, *[item for item in molecules if item != parent]][:6]

    def scorer(smiles):
        index = molecules.index(smiles) if smiles in molecules else 0
        return {
            "affinity": 0.90 - index * 0.04,
            "solubility": -3.0 + index * 0.1,
            "bbb": 0.85 - index * 0.03,
            "herg": 0.7 if index == 0 else 0.2,
            "sa": 2.0 + index * 0.1,
        }

    def successful_advisory(system_prompt, payload):
        value = json.loads(payload)
        candidates = (
            value["candidates"] if "Orchestrator" in system_prompt else value
        )
        if "Orchestrator" in system_prompt:
            chosen = next(
                candidate["smiles"]
                for candidate in candidates
                if candidate["raw"]["herg"] < 0.5
            )
            return {"chosen": chosen, "rationale": "Advisory only"}
        return {
            "verdict": "veto" if "Safety" in system_prompt else "support",
            "reason": "Advisory only",
            "preferred": candidates[0]["smiles"],
        }

    def timed_out_advisory(system_prompt, payload):
        raise TimeoutError("simulated timeout")

    advisory_clients = (
        None,
        successful_advisory,
        timed_out_advisory,
        lambda system_prompt, payload: {"malformed": True},
    )
    missions = []
    for client in advisory_clients:
        run = run_optimization(
            scorer=scorer,
            proposer=proposer,
            llm_client=client,
            seed_smiles="CO",
            seed_name="Parity seed",
        )
        missions.append(compile_mission(run, resolve_compound("CO")))

    scientific_fields = (
        "rounds",
        "lineage",
        "summary",
        "policy_comparison",
        "body_systems_map",
        "decision_ledger",
        "molecules",
    )
    baseline = missions[0]
    for compared in missions[1:]:
        for field in scientific_fields:
            assert compared[field] == baseline[field]
