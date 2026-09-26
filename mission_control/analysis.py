"""Compile raw optimization rounds into one auditable mission artifact."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

from core.agents import HERG_VETO_THRESHOLD
from core.contract import SCORE_KEYS, normalize
from core.runner import validate_run

from .chemistry import (
    conformer,
    depict_svg,
    molecule_id,
    structural_similarity,
)
from .evidence import EVIDENCE_LIBRARY


MISSION_SCHEMA_VERSION = 3
UTILITY_WEIGHTS = {
    "affinity": 0.35,
    "solubility": 0.25,
    "bbb": 0.25,
    "herg": 0.10,
    "sa": 0.05,
}
DISPLAY_NAMES = {
    "affinity": "Modeled DRD2 activity",
    "solubility": "Predicted solubility",
    "bbb": "Predicted BBB permeability",
    "herg": "Predicted hERG safety",
    "sa": "Synthetic accessibility",
}
ROLE_CONTEXT = {
    "affinity": ["ncats-gpcr", "oecd-qsar"],
    "adme": ["kinetic-solubility", "bbb-assays", "oecd-qsar"],
    "safety": ["ich-herg", "fda-cardiac-protocol", "oecd-qsar"],
    "orchestrator": ["nist-xai", "oecd-qsar"],
}
SLOT_LABELS = {
    1: ("incumbent", "Current incumbent"),
    2: ("direct_edit", "Aromatic fluoro edit"),
    3: ("direct_edit", "Aromatic hydroxyl edit"),
    4: ("direct_edit", "Aromatic nitrogen edit"),
    5: ("rescue_library", "Reference-scaffold rescue"),
    6: ("rescue_library", "Reference-scaffold comparator"),
}
EDIT_HYPOTHESES = {
    "Current incumbent": (
        "Keep the current parent as a control so a new edit advances only if "
        "it earns a better safe DRD2 and developability trade-off."
    ),
    "Aromatic fluoro edit": (
        "Test whether a compact aromatic substitution preserves the modeled "
        "DRD2 signal; judge its solubility, BBB, and hERG effects only from "
        "the recorded model deltas."
    ),
    "Aromatic hydroxyl edit": (
        "Add polarity to test a solubility and hERG-safety hypothesis while "
        "monitoring whether modeled DRD2 activity and BBB permeability fall."
    ),
    "Aromatic nitrogen edit": (
        "Test a heteroaromatic polarity change intended to reduce the local "
        "safety burden while retaining enough modeled DRD2 activity."
    ),
    "Reference-scaffold rescue": (
        "Escape the local scaffold when its hERG liability dominates by "
        "testing a DRD2-reference-like alternative across the same five axes."
    ),
    "Reference-scaffold comparator": (
        "Challenge the local series with a distinct CNS-active comparator to "
        "make high BBB or DRD2 signals compete visibly with hERG and solubility."
    ),
}


def weighted_utility(normalized: dict[str, float]) -> float:
    return sum(UTILITY_WEIGHTS[key] * normalized[key] for key in SCORE_KEYS)


def compile_mission(run: dict, intake: dict | None = None) -> dict:
    validate_run(run)
    mission_intake = deepcopy(intake) if intake is not None else _default_intake(run)
    seed_smiles = run["seed"]["smiles"]
    first_seen: dict[str, dict] = {}
    molecule_library: dict[str, dict] = {}
    rounds = []
    ledgers = []
    prior_winner_molecule_ids: set[str] = set()

    for raw_round in run["rounds"]:
        candidates = []
        for position, raw_candidate in enumerate(raw_round["candidates"], start=1):
            smiles = raw_candidate["smiles"]
            stable_id = molecule_id(smiles)
            first_seen.setdefault(
                stable_id,
                {"round": raw_round["round"], "candidate_id": f"R{raw_round['round']}C{position}"},
            )
            if stable_id not in molecule_library:
                molecule_library[stable_id] = {
                    "molecule_id": stable_id,
                    "name": "",
                    "smiles": smiles,
                    "depiction_svg": depict_svg(smiles),
                    "conformer": conformer(smiles),
                }
            if run.get("proposal_contract") == "deterministic_v1":
                source_kind, edit_label = SLOT_LABELS[position]
            elif position == 1:
                source_kind, edit_label = "incumbent", "Current incumbent"
            else:
                source_kind, edit_label = "unknown", "Unclassified proposal"
            candidate_name = _candidate_name(smiles, source_kind, edit_label)
            if not molecule_library[stable_id]["name"]:
                molecule_library[stable_id]["name"] = candidate_name
            candidate = {
                "candidate_id": f"R{raw_round['round']}C{position}",
                "short_id": f"C{position}",
                "molecule_id": stable_id,
                "name": candidate_name,
                "smiles": smiles,
                "incumbent": bool(raw_candidate["incumbent"]),
                "first_seen": deepcopy(first_seen[stable_id]),
                "source_kind": source_kind,
                "edit_label": edit_label,
                "edit_hypothesis": _edit_hypothesis(edit_label),
                "raw": deepcopy(raw_candidate["raw"]),
                "normalized": deepcopy(raw_candidate["normalized"]),
                "raw_delta_parent": _delta(
                    raw_candidate["raw"], raw_round["parent"]["raw"]
                ),
                "objective_delta_parent": _delta(
                    raw_candidate["normalized"], raw_round["parent"]["normalized"]
                ),
                "similarity_parent": structural_similarity(
                    smiles, raw_round["parent"]["smiles"]
                ),
                "similarity_seed": structural_similarity(smiles, seed_smiles),
                "vetoed": raw_candidate["raw"]["herg"] >= HERG_VETO_THRESHOLD,
                "weighted_utility": weighted_utility(raw_candidate["normalized"]),
                "safe_frontier": False,
                "dominated_by_count": 0,
                "preferred_by": [],
                "selection_flags": [],
            }
            candidates.append(candidate)

        _mark_safe_frontier(candidates)
        policy_choices = _policy_choices(candidates, raw_round)
        agents = _agent_decisions(raw_round, candidates)
        for role, decision in agents.items():
            preferred = decision["preferred_candidate"]
            if preferred:
                _by_id(candidates, preferred)["preferred_by"].append(role)
        for policy, candidate_id in policy_choices.items():
            _by_id(candidates, candidate_id)["selection_flags"].append(policy)

        winner_id = _id_for_smiles(candidates, raw_round["winner"]["smiles"])
        score_provenance = (
            "Bundled cached DRD2 and ADMET model outputs"
            if run.get("execution_mode", "cached") == "cached"
            else "Live DRD2 and ADMET scorer outputs"
        )
        ledger = _decision_ledger(
            raw_round,
            candidates,
            agents,
            policy_choices,
            score_provenance,
        )
        events = _events(
            raw_round["round"],
            candidates,
            agents,
            winner_id,
            prior_winner_molecule_ids,
        )
        ledgers.append(ledger)
        rounds.append(
            {
                "round": raw_round["round"],
                "parent": {
                    "molecule_id": molecule_id(raw_round["parent"]["smiles"]),
                    "smiles": raw_round["parent"]["smiles"],
                },
                "candidates": candidates,
                "candidate_lookup": {
                    candidate["candidate_id"]: index
                    for index, candidate in enumerate(candidates)
                },
                "agents": list(agents.values()),
                "winner_id": winner_id,
                "veto_count": sum(candidate["vetoed"] for candidate in candidates),
                "policy_choices": policy_choices,
                "events": events,
            }
        )
        prior_winner_molecule_ids.add(_by_id(candidates, winner_id)["molecule_id"])

    final_candidate = _by_id(rounds[-1]["candidates"], rounds[-1]["winner_id"])
    mission = {
        "schema_version": MISSION_SCHEMA_VERSION,
        "source_schema_version": run["schema_version"],
        "execution_mode": run.get("execution_mode", "cached"),
        "generated_at": run["generated_at"],
        "intake": mission_intake,
        "target": {
            "id": "DRD2",
            "name": "Dopamine D2 receptor",
            "capability": "Installed single-target DRD2 activity oracle",
            "therapeutic_intent": mission_intake["therapeutic_intent"],
            "rationale_status": mission_intake["target_rationale_status"],
            "rationale": mission_intake["target_rationale"],
            "source_ids": (
                ["haloperidol-pubchem"]
                if mission_intake["name"].casefold() == "haloperidol"
                else []
            ),
        },
        "score_keys": list(SCORE_KEYS),
        "score_labels": deepcopy(DISPLAY_NAMES),
        "utility_weights": deepcopy(UTILITY_WEIGHTS),
        "proposal_contract": run.get("proposal_contract", "unknown"),
        "safety_policy": {
            "property": "raw.herg",
            "operator": ">=",
            "threshold": HERG_VETO_THRESHOLD,
            "meaning": "Predicted hERG-blockade policy gate",
            "mutable": False,
        },
        "seed": {
            **deepcopy(run["seed"]),
            "molecule_id": molecule_id(seed_smiles),
        },
        "molecules": molecule_library,
        "rounds": rounds,
        "lineage": _lineage(rounds),
        "decision_ledger": ledgers,
        "summary": _summary(run, rounds, final_candidate),
        "policy_comparison": _policy_comparison(rounds),
        "body_systems_map": _body_systems_map(run["seed"], final_candidate),
        "wet_lab_plan": wet_lab_plan(final_candidate),
        "evidence_library": [deepcopy(item) for item in EVIDENCE_LIBRARY],
        "disclaimers": [
            "All five property values are model-derived ranking signals, not experimental results.",
            "Confidence is a within-slate decision-margin heuristic, not a probability of correctness.",
            "Similarity is structural locality to a reference molecule, not a true model applicability domain.",
            "The 3D view is a generated visual conformer, not a docking pose or measured structure.",
            "Body-system changes are directional hypotheses, not gene-expression, dose-response, or clinical claims.",
            "External sources guide interpretation and assay selection; they did not test these candidates.",
        ],
    }
    validate_mission(mission, require_digest=False)
    digest_payload = deepcopy(mission)
    mission["digest"] = hashlib.sha256(
        json.dumps(
            digest_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    validate_mission(mission)
    return mission


def confidence_from_ranking(ranking: list[dict]) -> tuple[float | None, str]:
    if len(ranking) < 2:
        return None, "Not estimable because fewer than two candidates were eligible."
    scores = [item["score"] for item in ranking]
    observed_range = max(scores) - min(scores)
    if observed_range <= 0:
        return 0.0, "All eligible candidates tied on this role's score."
    margin = scores[0] - scores[1]
    confidence = min(1.0, max(0.0, margin / observed_range))
    return confidence, (
        f"Top-versus-runner-up margin {margin:.3f}, divided by the "
        f"eligible score range {observed_range:.3f}."
    )


def wet_lab_plan(final_candidate: dict) -> list[dict]:
    candidate_name = final_candidate["name"]
    return [
        {
            "id": "WL-1",
            "priority": 1,
            "property": "Identity and purity",
            "assay": "LC-MS and analytical purity confirmation",
            "decision_question": "Is the tested material the intended compound at suitable purity?",
            "predicted_context": "No model score can establish material identity.",
            "readout": "Identity-consistent mass and reported purity.",
            "confirmation": "Required before interpreting downstream assays.",
            "evidence_ids": [],
        },
        {
            "id": "WL-2",
            "priority": 2,
            "property": "DRD2 binding",
            "assay": "Concentration-response DRD2 binding assay",
            "decision_question": f"Does {candidate_name} retain meaningful DRD2 affinity?",
            "predicted_context": f"Modeled activity score {final_candidate['raw']['affinity']:.3f}.",
            "readout": "Measured affinity with controls and replicate uncertainty.",
            "confirmation": "Confirms or rejects the central target-ranking hypothesis.",
            "evidence_ids": ["ncats-gpcr"],
        },
        {
            "id": "WL-3",
            "priority": 3,
            "property": "DRD2 function",
            "assay": "Gi/cAMP functional response, with beta-arrestin follow-up",
            "decision_question": "What functional potency, efficacy, and signaling behavior follow binding?",
            "predicted_context": "The activity model does not establish receptor function.",
            "readout": "Concentration-response potency and efficacy.",
            "confirmation": "Separates binding from functional pharmacology.",
            "evidence_ids": ["ncats-gpcr"],
        },
        {
            "id": "WL-4",
            "priority": 1,
            "property": "hERG",
            "assay": "Concentration-response hERG patch clamp",
            "decision_question": "Does the candidate clear the program's cardiac safety margin?",
            "predicted_context": (
                f"Predicted blockade score {final_candidate['raw']['herg']:.3f}; "
                f"policy threshold {HERG_VETO_THRESHOLD:.3f}."
            ),
            "readout": "Measured current inhibition across verified concentrations.",
            "confirmation": "Tests the non-negotiable model-based safety gate.",
            "evidence_ids": ["ich-herg", "fda-cardiac-protocol"],
        },
        {
            "id": "WL-5",
            "priority": 2,
            "property": "Solubility",
            "assay": "Defined-pH kinetic solubility, then thermodynamic solubility",
            "decision_question": "Is the modeled solubility improvement experimentally reproducible?",
            "predicted_context": f"Predicted LogS {final_candidate['raw']['solubility']:.3f}.",
            "readout": "Concentration with pH, units, temperature, and time recorded.",
            "confirmation": "Tests developability under declared assay conditions.",
            "evidence_ids": ["kinetic-solubility"],
        },
        {
            "id": "WL-6",
            "priority": 3,
            "property": "BBB passive permeability",
            "assay": "PAMPA-BBB",
            "decision_question": "Does the molecule show useful passive permeability?",
            "predicted_context": f"BBB classifier score {final_candidate['raw']['bbb']:.3f}.",
            "readout": "Passive apparent permeability under controlled conditions.",
            "confirmation": "Tests only the passive component of the BBB hypothesis.",
            "evidence_ids": ["bbb-assays"],
        },
        {
            "id": "WL-7",
            "priority": 3,
            "property": "BBB efflux",
            "assay": "Bidirectional MDCK-MDR1 permeability",
            "decision_question": "Could active efflux limit exposure despite passive permeability?",
            "predicted_context": "The current BBB score does not isolate transporter effects.",
            "readout": "Directional permeability and efflux ratio.",
            "confirmation": "Adds the active-transport question missing from PAMPA.",
            "evidence_ids": ["bbb-assays"],
        },
        {
            "id": "WL-8",
            "priority": 2,
            "property": "Synthetic feasibility",
            "assay": "Medicinal-chemist route and starting-material review",
            "decision_question": "Is there a credible, practical route to test material?",
            "predicted_context": f"SA score {final_candidate['raw']['sa']:.3f}.",
            "readout": "Route risks, precedent, material access, and a go/no-go recommendation.",
            "confirmation": "Replaces a heuristic with expert feasibility evidence.",
            "evidence_ids": ["sa-score"],
        },
    ]


def validate_mission(mission: dict, *, require_digest: bool = True) -> None:
    if mission.get("schema_version") != MISSION_SCHEMA_VERSION:
        raise ValueError("Unsupported mission schema")
    if mission.get("target", {}).get("id") != "DRD2":
        raise ValueError("Mission target must be DRD2")
    if mission.get("execution_mode") not in {"cached", "live"}:
        raise ValueError("Mission execution mode is unsupported")
    if mission.get("score_keys") != list(SCORE_KEYS):
        raise ValueError("Mission score axes do not match the core contract")
    digest = mission.get("digest")
    if digest is None and require_digest:
        raise ValueError("Finalized mission requires a digest")
    if digest is not None:
        payload = deepcopy(mission)
        payload.pop("digest")
        expected_digest = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
        if digest != expected_digest:
            raise ValueError("Mission digest does not match its payload")
    rounds = mission.get("rounds", [])
    if len(rounds) != 5:
        raise ValueError("A mission must contain five rounds")
    evidence_ids = {item["id"] for item in mission["evidence_library"]}
    if not set(mission["target"].get("source_ids", [])).issubset(evidence_ids):
        raise ValueError("Target rationale references unknown evidence")
    molecules = mission.get("molecules", {})
    if not molecules:
        raise ValueError("Mission molecule library is empty")
    for stable_id, record in molecules.items():
        if (
            stable_id != record.get("molecule_id")
            or stable_id != molecule_id(record["smiles"])
        ):
            raise ValueError("Molecule library identity is inconsistent")
        geometry = record["conformer"]
        atom_count = len(geometry["atoms"])
        if not atom_count:
            raise ValueError("Molecule conformer is empty")
        if any(
            not math.isfinite(atom[axis])
            for atom in geometry["atoms"]
            for axis in ("x", "y", "z")
        ):
            raise ValueError("Conformer contains a non-finite coordinate")
        if any(
            not 0 <= bond["begin_atom_index"] < atom_count
            or not 0 <= bond["end_atom_index"] < atom_count
            for bond in geometry["bonds"]
        ):
            raise ValueError("Conformer contains an invalid bond index")
    seed = mission.get("seed", {})
    seed_smiles = seed.get("smiles")
    if (
        not seed_smiles
        or seed.get("molecule_id") != molecule_id(seed_smiles)
        or seed.get("molecule_id") not in molecules
    ):
        raise ValueError("Seed molecule identity is inconsistent")
    expected_parent = seed_smiles
    prior_winner_molecule_ids: set[str] = set()
    for expected_round, round_value in enumerate(rounds, start=1):
        if round_value["round"] != expected_round:
            raise ValueError("Mission rounds are out of sequence")
        if (
            round_value["parent"]["smiles"] != expected_parent
            or round_value["parent"]["molecule_id"]
            != molecule_id(expected_parent)
        ):
            raise ValueError("Mission parent chain is inconsistent")
        candidates = round_value["candidates"]
        if len(candidates) != 6:
            raise ValueError("Every mission round must contain six candidates")
        ids = {candidate["candidate_id"] for candidate in candidates}
        expected_lookup = {
            candidate["candidate_id"]: index
            for index, candidate in enumerate(candidates)
        }
        if round_value["candidate_lookup"] != expected_lookup:
            raise ValueError("Candidate lookup does not match the round slate")
        if round_value["winner_id"] not in ids:
            raise ValueError("Mission winner is not in its round")
        incumbents = [candidate for candidate in candidates if candidate["incumbent"]]
        if len(incumbents) != 1 or incumbents[0]["smiles"] != expected_parent:
            raise ValueError("Mission round does not contain its parent once")
        for candidate in candidates:
            if set(candidate["raw"]) != set(SCORE_KEYS):
                raise ValueError("Candidate raw scores do not match the contract")
            if set(candidate["normalized"]) != set(SCORE_KEYS):
                raise ValueError("Candidate objectives do not match the contract")
            if candidate["normalized"] != normalize(candidate["raw"]):
                raise ValueError("Candidate normalization does not match the contract")
            expected_veto = candidate["raw"]["herg"] >= HERG_VETO_THRESHOLD
            if candidate["vetoed"] is not expected_veto:
                raise ValueError("Candidate veto does not match the inclusive hERG gate")
            if not all(
                math.isfinite(candidate[key])
                for key in ("similarity_parent", "similarity_seed", "weighted_utility")
            ):
                raise ValueError("Candidate analysis contains a non-finite value")
            if not 0 <= candidate["similarity_parent"] <= 1:
                raise ValueError("Parent similarity is outside zero to one")
            if not 0 <= candidate["similarity_seed"] <= 1:
                raise ValueError("Seed similarity is outside zero to one")
            if candidate["molecule_id"] not in molecules:
                raise ValueError("Candidate is missing from the molecule library")
            if candidate["molecule_id"] != molecule_id(candidate["smiles"]):
                raise ValueError("Candidate molecule identity is inconsistent")
            position = int(candidate["short_id"][1:])
            if mission.get("proposal_contract") == "deterministic_v1":
                expected_source, expected_edit = SLOT_LABELS[position]
            elif position == 1:
                expected_source, expected_edit = "incumbent", "Current incumbent"
            else:
                expected_source, expected_edit = "unknown", "Unclassified proposal"
            if (
                candidate["source_kind"] != expected_source
                or candidate["edit_label"] != expected_edit
            ):
                raise ValueError("Candidate provenance is inconsistent")
            if candidate["edit_hypothesis"] != _edit_hypothesis(expected_edit):
                raise ValueError("Candidate edit hypothesis is inconsistent")
            if not math.isclose(
                candidate["weighted_utility"],
                weighted_utility(candidate["normalized"]),
                rel_tol=0,
                abs_tol=1e-12,
            ):
                raise ValueError("Candidate utility does not match the formula")
        winner = _by_id(candidates, round_value["winner_id"])
        if winner["vetoed"]:
            raise ValueError("Official winner violates the hERG gate")
        safe = [candidate for candidate in candidates if not candidate["vetoed"]]
        novel = [candidate for candidate in safe if not candidate["incumbent"]]
        eligible = novel or safe
        expected_winner = _rank("orchestrator", eligible)[0]["candidate_id"]
        if round_value["winner_id"] != expected_winner:
            raise ValueError("Official winner does not match the deterministic policy")
        expected_policies = {
            "affinity_only": _rank("affinity", candidates)[0]["candidate_id"],
            "adme_only": _rank("adme", candidates)[0]["candidate_id"],
            "safety_only": _rank("safety", candidates)[0]["candidate_id"],
            "weighted": _rank("orchestrator", safe)[0]["candidate_id"],
            "official": expected_winner,
        }
        if round_value["policy_choices"] != expected_policies:
            raise ValueError("Round policy choices are inconsistent")
        expected_roles = ("affinity", "adme", "safety", "orchestrator")
        if tuple(agent["role"] for agent in round_value["agents"]) != expected_roles:
            raise ValueError("Round agent roles are incomplete or out of order")
        expected_agents = {}
        affinity_preferred = _rank("affinity", candidates)[0]["candidate_id"]
        for agent in round_value["agents"]:
            pool = eligible if agent["role"] == "orchestrator" else candidates
            expected_ranking = _rank(agent["role"], pool)
            if agent["ranking"] != expected_ranking:
                raise ValueError("Agent ranking does not match the scored slate")
            if agent["preferred_candidate"] != expected_ranking[0]["candidate_id"]:
                raise ValueError("Agent preference does not match its ranking")
            expected_confidence, expected_basis = confidence_from_ranking(
                expected_ranking
            )
            if agent["confidence_score"] != expected_confidence:
                raise ValueError("Agent confidence does not match its ranking")
            if agent["confidence_basis"] != expected_basis:
                raise ValueError("Agent confidence basis is inconsistent")
            confidence = agent["confidence_score"]
            if confidence is not None and not 0 <= confidence <= 1:
                raise ValueError("Agent confidence is outside zero to one")
            if not set(agent["context_sources"]).issubset(evidence_ids):
                raise ValueError("Agent references unknown evidence")
            role = agent["role"]
            preferred = _by_id(candidates, expected_ranking[0]["candidate_id"])
            if role == "orchestrator":
                expected_agent = {
                    "role": role,
                    "verdict": "choose",
                    "preferred_candidate": preferred["candidate_id"],
                    "reason": _orchestrator_reason(
                        preferred, candidates, bool(novel)
                    ),
                    "confidence_score": expected_confidence,
                    "confidence_basis": expected_basis,
                    "ranking": expected_ranking,
                    "evidence": [
                        {
                            "label": "Eligible utility",
                            "value": preferred["weighted_utility"],
                            "formula": (
                                "0.35 DRD2 + 0.25 solubility + 0.25 BBB + "
                                "0.10 hERG safety + 0.05 SA"
                            ),
                        },
                        {
                            "label": "hERG gate",
                            "value": preferred["raw"]["herg"],
                            "threshold": HERG_VETO_THRESHOLD,
                        },
                        {
                            "label": "Exploration policy",
                            "value": (
                                "novel survivor required"
                                if novel
                                else "incumbent allowed"
                            ),
                        },
                    ],
                    "rejections": _orchestrator_rejections(
                        preferred["candidate_id"], candidates, eligible
                    ),
                    "context_sources": deepcopy(
                        ROLE_CONTEXT["orchestrator"]
                    ),
                }
            else:
                expected_agent = {
                    "role": role,
                    "verdict": (
                        "support"
                        if role == "affinity"
                        else "object"
                        if role == "adme"
                        and preferred["candidate_id"] != affinity_preferred
                        else "veto"
                        if role == "safety"
                        and any(candidate["vetoed"] for candidate in candidates)
                        else "support"
                    ),
                    "preferred_candidate": preferred["candidate_id"],
                    "reason": _target_specific_reason(
                        role, preferred, candidates
                    ),
                    "confidence_score": expected_confidence,
                    "confidence_basis": expected_basis,
                    "ranking": expected_ranking,
                    "evidence": _role_evidence(
                        role, preferred, candidates
                    ),
                    "rejections": _role_rejections(
                        role,
                        preferred["candidate_id"],
                        expected_ranking,
                        candidates,
                    ),
                    "context_sources": deepcopy(ROLE_CONTEXT[role]),
                }
            if agent != expected_agent:
                raise ValueError(
                    "Agent explanation does not match the scored policy"
                )
            expected_agents[role] = expected_agent
        expected_events = _events(
            round_value["round"],
            candidates,
            expected_agents,
            expected_winner,
            prior_winner_molecule_ids,
        )
        if round_value["events"] != expected_events:
            raise ValueError("Round events do not match the scored policy")
        prior_winner_molecule_ids.add(winner["molecule_id"])
        expected_parent = winner["smiles"]
    ledgers = mission.get("decision_ledger", [])
    if len(ledgers) != len(rounds):
        raise ValueError("Decision ledger count does not match the mission")
    for round_value, ledger in zip(rounds, ledgers):
        candidates = round_value["candidates"]
        expected_vetoed = [
            candidate["candidate_id"] for candidate in candidates if candidate["vetoed"]
        ]
        expected_survivors = [
            candidate["candidate_id"] for candidate in candidates if not candidate["vetoed"]
        ]
        orchestrator = round_value["agents"][-1]
        expected_pool = [
            item["candidate_id"] for item in orchestrator["ranking"]
        ]
        expected_runner_up = (
            orchestrator["ranking"][1]["candidate_id"]
            if len(orchestrator["ranking"]) > 1
            else None
        )
        expected_margin = (
            orchestrator["ranking"][0]["score"]
            - orchestrator["ranking"][1]["score"]
            if expected_runner_up
            else None
        )
        if ledger["round"] != round_value["round"]:
            raise ValueError("Decision ledger round is inconsistent")
        if ledger["inputs"] != {
            "parent_smiles": round_value["parent"]["smiles"],
            "candidate_ids": [
                candidate["candidate_id"] for candidate in candidates
            ],
            "score_provenance": (
                "Bundled cached DRD2 and ADMET model outputs"
                if mission["execution_mode"] == "cached"
                else "Live DRD2 and ADMET scorer outputs"
            ),
        }:
            raise ValueError("Decision ledger inputs are inconsistent")
        if ledger["specialist_steps"] != round_value["agents"][:3]:
            raise ValueError("Decision ledger specialist steps are inconsistent")
        if ledger["orchestrator_pool"] != expected_pool:
            raise ValueError("Decision ledger orchestrator pool is inconsistent")
        if ledger["choice"]["winner"] != round_value["winner_id"]:
            raise ValueError("Decision ledger winner is inconsistent")
        if ledger["choice"]["runner_up"] != expected_runner_up:
            raise ValueError("Decision ledger runner-up is inconsistent")
        if ledger["choice"]["margin"] != expected_margin:
            raise ValueError("Decision ledger margin is inconsistent")
        if ledger["safety_gate"]["vetoed_ids"] != expected_vetoed:
            raise ValueError("Decision ledger veto list is inconsistent")
        if ledger["safety_gate"]["surviving_ids"] != expected_survivors:
            raise ValueError("Decision ledger survivor list is inconsistent")
        if {
            key: ledger["safety_gate"][key]
            for key in ("property", "operator", "threshold")
        } != {
            "property": "raw.herg",
            "operator": ">=",
            "threshold": HERG_VETO_THRESHOLD,
        }:
            raise ValueError("Decision ledger safety policy is inconsistent")
        if ledger["orchestrator_ranking"] != orchestrator["ranking"]:
            raise ValueError("Decision ledger ranking is inconsistent")
        if ledger["why_chosen"] != [
            orchestrator["reason"],
            "The official choice clears the fixed hERG gate.",
        ]:
            raise ValueError("Decision ledger choice explanation is inconsistent")
        if ledger["why_rejected"] != orchestrator["rejections"]:
            raise ValueError("Decision ledger rejections are inconsistent")
        if ledger["validation"] != {
            "winner_in_candidates": True,
            "winner_not_vetoed": True,
            "agent_contract_present": True,
            "score_axes_exact": True,
        }:
            raise ValueError("Decision ledger validation flags are inconsistent")
    if mission.get("lineage") != _lineage(rounds):
        raise ValueError("Mission lineage is inconsistent")
    if mission.get("policy_comparison") != _policy_comparison(rounds):
        raise ValueError("Mission policy comparison is inconsistent")
    if any(step.get("confirmed") for step in mission["wet_lab_plan"]):
        raise ValueError("Wet-lab plan cannot claim completed confirmation")


def _default_intake(run: dict) -> dict:
    return {
        "query": run["seed"]["name"],
        "source": "catalog",
        "name": run["seed"]["name"],
        "canonical_smiles": run["seed"]["smiles"],
        "formula": "",
        "molecular_weight": None,
        "target": "DRD2",
        "therapeutic_intent": (
            "Prioritize a central DRD2 lead hypothesis while preserving a "
            "credible safety and developability path."
        ),
        "seed_rationale": (
            "Haloperidol is used here as a known DRD2-active reference scaffold "
            "that makes the target objective and hERG liability conflict visible."
        ),
        "optimization_hypothesis": (
            "Retain modeled DRD2 activity, reject predicted hERG liability, "
            "improve solubility, and treat BBB exposure and synthesis as explicit "
            "trade-offs."
        ),
        "target_rationale_status": "supported_reference",
        "target_rationale": (
            "Haloperidol is a known DRD2 antagonist reference. DRD2 is the "
            "installed target capability and a pharmacology objective relevant "
            "to the seed's established mechanism."
        ),
    }


def _delta(current: dict, previous: dict) -> dict:
    return {key: current[key] - previous[key] for key in SCORE_KEYS}


def _candidate_name(smiles: str, source_kind: str, edit_label: str) -> str:
    if "S(N)(=O)=O" in smiles and source_kind == "rescue_library":
        return "Sulpiride-like rescue"
    if "Cl" in smiles and source_kind == "rescue_library":
        return "Tricyclic comparator"
    if source_kind == "incumbent":
        return "Current incumbent"
    return edit_label


def _edit_hypothesis(edit_label: str) -> str:
    return EDIT_HYPOTHESES.get(
        edit_label,
        (
            "Treat this proposal as an unclassified structural hypothesis. "
            "Advance it only from the exact modeled DRD2, ADMET, safety, and "
            "synthetic-accessibility evidence shown."
        ),
    )


def _mark_safe_frontier(candidates: list[dict]) -> None:
    safe = [candidate for candidate in candidates if not candidate["vetoed"]]
    for candidate in safe:
        dominators = [
            other
            for other in safe
            if other is not candidate
            and all(
                other["normalized"][key] >= candidate["normalized"][key]
                for key in SCORE_KEYS
            )
            and any(
                other["normalized"][key] > candidate["normalized"][key]
                for key in SCORE_KEYS
            )
        ]
        candidate["dominated_by_count"] = len(dominators)
        candidate["safe_frontier"] = not dominators


def _role_score(role: str, candidate: dict) -> float:
    if role == "affinity":
        return candidate["normalized"]["affinity"]
    if role == "adme":
        return (
            candidate["normalized"]["solubility"] + candidate["normalized"]["bbb"]
        ) / 2
    if role == "safety":
        return candidate["normalized"]["herg"]
    return candidate["weighted_utility"]


def _rank(role: str, candidates: list[dict]) -> list[dict]:
    if role == "safety":
        ordered = sorted(
            candidates,
            key=lambda item: (-_role_score(role, item), item["smiles"]),
        )
    else:
        ordered = sorted(
            candidates,
            key=lambda item: (_role_score(role, item), item["smiles"]),
            reverse=True,
        )
    return [
        {
            "candidate_id": candidate["candidate_id"],
            "score": _role_score(role, candidate),
            "rank": rank,
        }
        for rank, candidate in enumerate(ordered, start=1)
    ]


def _agent_decisions(raw_round: dict, candidates: list[dict]) -> dict[str, dict]:
    decisions = {}
    for role in ("affinity", "adme", "safety"):
        raw_agent = raw_round["agents"]["specialists"][role]
        ranking = _rank(role, candidates)
        confidence, basis = confidence_from_ranking(ranking)
        preferred_id = _id_for_smiles(candidates, raw_agent["preferred"])
        preferred = _by_id(candidates, preferred_id)
        decisions[role] = {
            "role": role,
            "verdict": raw_agent["verdict"],
            "preferred_candidate": preferred_id,
            "reason": _target_specific_reason(role, preferred, candidates),
            "confidence_score": confidence,
            "confidence_basis": basis,
            "ranking": ranking,
            "evidence": _role_evidence(role, preferred, candidates),
            "rejections": _role_rejections(role, preferred_id, ranking, candidates),
            "context_sources": deepcopy(ROLE_CONTEXT[role]),
        }

    survivors = [candidate for candidate in candidates if not candidate["vetoed"]]
    novel = [candidate for candidate in survivors if not candidate["incumbent"]]
    eligible = novel or survivors
    ranking = _rank("orchestrator", eligible)
    confidence, basis = confidence_from_ranking(ranking)
    chosen_id = _id_for_smiles(
        candidates, raw_round["agents"]["orchestrator"]["chosen"]
    )
    chosen = _by_id(candidates, chosen_id)
    decisions["orchestrator"] = {
        "role": "orchestrator",
        "verdict": "choose",
        "preferred_candidate": chosen_id,
        "reason": _orchestrator_reason(chosen, candidates, bool(novel)),
        "confidence_score": confidence,
        "confidence_basis": basis,
        "ranking": ranking,
        "evidence": [
            {
                "label": "Eligible utility",
                "value": chosen["weighted_utility"],
                "formula": "0.35 DRD2 + 0.25 solubility + 0.25 BBB + 0.10 hERG safety + 0.05 SA",
            },
            {
                "label": "hERG gate",
                "value": chosen["raw"]["herg"],
                "threshold": HERG_VETO_THRESHOLD,
            },
            {
                "label": "Exploration policy",
                "value": "novel survivor required" if novel else "incumbent allowed",
            },
        ],
        "rejections": _orchestrator_rejections(chosen_id, candidates, eligible),
        "context_sources": deepcopy(ROLE_CONTEXT["orchestrator"]),
    }
    return decisions


def _target_specific_reason(role: str, preferred: dict, candidates: list[dict]) -> str:
    if role == "affinity":
        return (
            f"Prefers {preferred['short_id']} because its modeled DRD2 activity "
            f"score is {preferred['normalized']['affinity']:.3f}. This role tests "
            "whether an edit preserves the installed target objective."
        )
    if role == "adme":
        score = _role_score(role, preferred)
        return (
            f"Prefers {preferred['short_id']} on the declared solubility and BBB "
            f"trade-off, with a within-slate mean objective score of {score:.3f}."
        )
    veto_count = sum(candidate["vetoed"] for candidate in candidates)
    return (
        f"Prefers {preferred['short_id']} for the strongest modeled hERG safety "
        f"signal and vetoes {veto_count} candidate(s) at raw hERG score "
        f"{HERG_VETO_THRESHOLD:.3f} or above."
    )


def _orchestrator_reason(
    chosen: dict, candidates: list[dict], novelty_required: bool
) -> str:
    vetoed = sum(candidate["vetoed"] for candidate in candidates)
    exploration = (
        "The policy required a novel safe survivor for this exploration round."
        if novelty_required
        else "No novel safe survivor was available, so the incumbent remained eligible."
    )
    return (
        f"Chooses {chosen['short_id']} after removing {vetoed} hERG-vetoed "
        f"candidate(s). {exploration} The choice has the highest weighted utility "
        "inside that eligible pool."
    )


def _role_evidence(role: str, preferred: dict, candidates: list[dict]) -> list[dict]:
    if role == "affinity":
        return [
            {
                "label": "Modeled DRD2 activity",
                "value": preferred["normalized"]["affinity"],
                "candidate_id": preferred["candidate_id"],
            }
        ]
    if role == "adme":
        return [
            {
                "label": "Solubility objective",
                "value": preferred["normalized"]["solubility"],
                "candidate_id": preferred["candidate_id"],
            },
            {
                "label": "BBB objective",
                "value": preferred["normalized"]["bbb"],
                "candidate_id": preferred["candidate_id"],
            },
        ]
    return [
        {
            "label": "Raw predicted hERG blockade",
            "value": preferred["raw"]["herg"],
            "threshold": HERG_VETO_THRESHOLD,
            "candidate_id": preferred["candidate_id"],
        },
        {
            "label": "Vetoed candidates",
            "value": sum(candidate["vetoed"] for candidate in candidates),
        },
    ]


def _role_rejections(
    role: str,
    preferred_id: str,
    ranking: list[dict],
    candidates: list[dict],
) -> list[dict]:
    ranks = {item["candidate_id"]: item for item in ranking}
    return [
        {
            "candidate_id": candidate["candidate_id"],
            "reason_code": (
                "HERG_VETO"
                if candidate["vetoed"] and role == "safety"
                else "LOWER_ROLE_SCORE"
            ),
            "reason": (
                f"Ranked {ranks[candidate['candidate_id']]['rank']} of 6 on "
                f"the {role} role score."
            ),
        }
        for candidate in candidates
        if candidate["candidate_id"] != preferred_id
    ]


def _orchestrator_rejections(
    chosen_id: str, candidates: list[dict], eligible: list[dict]
) -> list[dict]:
    eligible_ids = {candidate["candidate_id"] for candidate in eligible}
    return [
        {
            "candidate_id": candidate["candidate_id"],
            "reason_code": (
                "HERG_VETO"
                if candidate["vetoed"]
                else "INCUMBENT_EXCLUDED_FOR_EXPLORATION"
                if candidate["candidate_id"] not in eligible_ids
                else "LOWER_ELIGIBLE_UTILITY"
            ),
            "reason": (
                "Crossed the fixed hERG policy threshold."
                if candidate["vetoed"]
                else "Excluded because a novel safe survivor existed."
                if candidate["candidate_id"] not in eligible_ids
                else "Had lower weighted utility inside the eligible pool."
            ),
        }
        for candidate in candidates
        if candidate["candidate_id"] != chosen_id
    ]


def _policy_choices(candidates: list[dict], raw_round: dict) -> dict:
    affinity = _rank("affinity", candidates)[0]["candidate_id"]
    adme = _rank("adme", candidates)[0]["candidate_id"]
    safety = _rank("safety", candidates)[0]["candidate_id"]
    safe = [candidate for candidate in candidates if not candidate["vetoed"]]
    weighted = _rank("orchestrator", safe)[0]["candidate_id"]
    official = _id_for_smiles(candidates, raw_round["winner"]["smiles"])
    return {
        "affinity_only": affinity,
        "adme_only": adme,
        "safety_only": safety,
        "weighted": weighted,
        "official": official,
    }


def _decision_ledger(
    raw_round: dict,
    candidates: list[dict],
    agents: dict[str, dict],
    policy_choices: dict,
    score_provenance: str,
) -> dict:
    survivors = [candidate["candidate_id"] for candidate in candidates if not candidate["vetoed"]]
    official = policy_choices["official"]
    orchestrator = agents["orchestrator"]
    runner_up = (
        orchestrator["ranking"][1]["candidate_id"]
        if len(orchestrator["ranking"]) > 1
        else None
    )
    return {
        "round": raw_round["round"],
        "inputs": {
            "parent_smiles": raw_round["parent"]["smiles"],
            "candidate_ids": [candidate["candidate_id"] for candidate in candidates],
            "score_provenance": score_provenance,
        },
        "specialist_steps": [
            deepcopy(agents[role]) for role in ("affinity", "adme", "safety")
        ],
        "safety_gate": {
            "property": "raw.herg",
            "operator": ">=",
            "threshold": HERG_VETO_THRESHOLD,
            "vetoed_ids": [
                candidate["candidate_id"]
                for candidate in candidates
                if candidate["vetoed"]
            ],
            "surviving_ids": survivors,
        },
        "orchestrator_pool": [
            item["candidate_id"] for item in orchestrator["ranking"]
        ],
        "orchestrator_ranking": deepcopy(orchestrator["ranking"]),
        "choice": {
            "winner": official,
            "runner_up": runner_up,
            "margin": (
                orchestrator["ranking"][0]["score"]
                - orchestrator["ranking"][1]["score"]
                if runner_up
                else None
            ),
        },
        "why_chosen": [
            orchestrator["reason"],
            "The official choice clears the fixed hERG gate.",
        ],
        "why_rejected": deepcopy(orchestrator["rejections"]),
        "validation": {
            "winner_in_candidates": official
            in {candidate["candidate_id"] for candidate in candidates},
            "winner_not_vetoed": not _by_id(candidates, official)["vetoed"],
            "agent_contract_present": all(
                agents[role]["preferred_candidate"]
                for role in ("affinity", "adme", "safety", "orchestrator")
            ),
            "score_axes_exact": all(
                set(candidate["raw"]) == set(SCORE_KEYS) for candidate in candidates
            ),
        },
    }


def _events(
    round_number: int,
    candidates: list[dict],
    agents: dict[str, dict],
    winner_id: str,
    prior_winner_molecule_ids: set[str],
) -> list[dict]:
    events = [
        {
            "type": "round_opened",
            "title": f"Round {round_number} opened",
            "detail": "The incumbent and five alternatives enter the fixed five-objective slate.",
        }
    ]
    events.extend(
        {
            "type": "candidate_scored",
            "candidate_id": candidate["candidate_id"],
            "title": f"{candidate['short_id']} scored",
            "detail": (
                "Candidate crosses the hERG policy."
                if candidate["vetoed"]
                else "Candidate remains eligible at the safety gate."
            ),
        }
        for candidate in candidates
    )
    events.extend(
        {
            "type": "specialist_decision",
            "role": role,
            "candidate_id": agents[role]["preferred_candidate"],
            "title": f"{role.title()} specialist recorded",
            "detail": agents[role]["reason"],
        }
        for role in ("affinity", "adme", "safety")
    )
    events.append(
        {
            "type": "safety_gate",
            "title": "hERG gate applied",
            "detail": (
                f"{sum(candidate['vetoed'] for candidate in candidates)} of "
                "6 candidates removed."
            ),
        }
    )
    winner = _by_id(candidates, winner_id)
    if (
        winner["molecule_id"] in prior_winner_molecule_ids
        and not winner["incumbent"]
    ):
        origin = winner["first_seen"]["candidate_id"]
        events.append(
            {
                "type": "rollback",
                "candidate_id": winner_id,
                "title": f"{winner_id} restores {origin}",
                "detail": (
                    f"The policy returns to the molecule first seen as {origin} "
                    "because it is the highest-utility eligible survivor in "
                    "this round."
                ),
            }
        )
    events.extend(
        [
            {
                "type": "orchestrator_choice",
                "candidate_id": winner_id,
                "title": f"{winner_id} advances",
                "detail": agents["orchestrator"]["reason"],
            },
            {
                "type": "round_closed",
                "candidate_id": winner_id,
                "title": f"Round {round_number} closed",
                "detail": "The recorded winner becomes the next round's parent.",
            },
        ]
    )
    return events


def _lineage(rounds: list[dict]) -> list[dict]:
    molecules: dict[str, dict] = {}
    for round_value in rounds:
        for candidate in round_value["candidates"]:
            record = molecules.setdefault(
                candidate["molecule_id"],
                {
                    "molecule_id": candidate["molecule_id"],
                    "name": candidate["name"],
                    "smiles": candidate["smiles"],
                    "first_seen": deepcopy(candidate["first_seen"]),
                    "occurrences": [],
                    "winner_rounds": [],
                },
            )
            record["occurrences"].append(candidate["candidate_id"])
            if candidate["candidate_id"] == round_value["winner_id"]:
                record["winner_rounds"].append(round_value["round"])
    return list(molecules.values())


def _summary(run: dict, rounds: list[dict], final_candidate: dict) -> dict:
    all_candidates = [
        candidate for round_value in rounds for candidate in round_value["candidates"]
    ]
    winner_molecules = {
        _by_id(round_value["candidates"], round_value["winner_id"])["molecule_id"]
        for round_value in rounds
    }
    affinity_unsafe = sum(
        _by_id(round_value["candidates"], round_value["policy_choices"]["affinity_only"])[
            "vetoed"
        ]
        for round_value in rounds
    )
    adme_unsafe = sum(
        _by_id(round_value["candidates"], round_value["policy_choices"]["adme_only"])[
            "vetoed"
        ]
        for round_value in rounds
    )
    return {
        "candidate_occurrences": len(all_candidates),
        "unique_molecules": len({candidate["molecule_id"] for candidate in all_candidates}),
        "vetoed_occurrences": sum(candidate["vetoed"] for candidate in all_candidates),
        "unique_winners": len(winner_molecules),
        "affinity_only_unsafe_rounds": affinity_unsafe,
        "adme_only_unsafe_rounds": adme_unsafe,
        "final_candidate_id": rounds[-1]["winner_id"],
        "final_molecule_id": final_candidate["molecule_id"],
        "final_seed_similarity": final_candidate["similarity_seed"],
        "controlled_rollbacks": sum(
            len(record["winner_rounds"]) > 1
            for record in _lineage(rounds)
        ),
        "story": (
            "Round 1 performs a safety-motivated scaffold rescue. Later rounds "
            "explore local alternatives and restore the stronger safe candidate "
            "when those excursions do not improve the declared trade-off."
        ),
        "seed_name": run["seed"]["name"],
    }


def _policy_comparison(rounds: list[dict]) -> list[dict]:
    roles = {
        "affinity_only": ("affinity", "DRD2"),
        "adme_only": ("adme", "ADME"),
        "safety_only": ("safety", "hERG safety"),
        "weighted": ("orchestrator", "Utility"),
        "official": ("orchestrator", "Utility"),
    }
    output = []
    for round_value in rounds:
        row = {"round": round_value["round"]}
        for policy, candidate_id in round_value["policy_choices"].items():
            candidate = _by_id(round_value["candidates"], candidate_id)
            role, score_label = roles[policy]
            row[policy] = {
                "candidate_id": candidate_id,
                "vetoed": candidate["vetoed"],
                "weighted_utility": candidate["weighted_utility"],
                "decision_score": _role_score(role, candidate),
                "score_label": score_label,
            }
        row["official_differs_from_weighted"] = (
            round_value["policy_choices"]["official"]
            != round_value["policy_choices"]["weighted"]
        )
        output.append(row)
    return output


def _body_systems_map(seed: dict, final_candidate: dict) -> list[dict]:
    seed_norm = seed["normalized"]
    final_norm = final_candidate["normalized"]
    return [
        {
            "system": "brain",
            "label": "Target engagement and brain exposure hypothesis",
            "source_properties": ["affinity", "bbb"],
            "seed_values": {
                "affinity": seed_norm["affinity"],
                "bbb": seed_norm["bbb"],
            },
            "candidate_values": {
                "affinity": final_norm["affinity"],
                "bbb": final_norm["bbb"],
            },
            "direction": (
                f"DRD2 objective {_signed(final_norm['affinity'] - seed_norm['affinity'])}; "
                f"BBB objective {_signed(final_norm['bbb'] - seed_norm['bbb'])}"
            ),
            "interpretation": (
                "The candidate retains a high modeled DRD2 signal but reduces the "
                "modeled BBB signal relative to the seed. Binding and permeability "
                "must be measured separately."
            ),
            "confidence_label": "Model-derived directional hypothesis",
            "wet_lab_step_ids": ["WL-2", "WL-3", "WL-6", "WL-7"],
        },
        {
            "system": "heart",
            "label": "Cardiac ion-channel risk hypothesis",
            "source_properties": ["herg"],
            "seed_values": {"herg": seed_norm["herg"]},
            "candidate_values": {"herg": final_norm["herg"]},
            "direction": f"hERG safety objective {_signed(final_norm['herg'] - seed_norm['herg'])}",
            "interpretation": (
                "The selected candidate has a stronger modeled hERG safety signal "
                "and clears the program gate. Only electrophysiology can confirm it."
            ),
            "confidence_label": "Model-derived directional hypothesis",
            "wet_lab_step_ids": ["WL-4"],
        },
        {
            "system": "absorption",
            "label": "Solubility-linked absorption hypothesis",
            "source_properties": ["solubility"],
            "seed_values": {"solubility": seed_norm["solubility"]},
            "candidate_values": {"solubility": final_norm["solubility"]},
            "direction": (
                "Solubility objective "
                f"{_signed(final_norm['solubility'] - seed_norm['solubility'])}"
            ),
            "interpretation": (
                "Improved predicted solubility may support exposure, but this view "
                "does not model dose, formulation, metabolism, or bioavailability."
            ),
            "confidence_label": "Model-derived directional hypothesis",
            "wet_lab_step_ids": ["WL-5"],
        },
    ]


def export_rows(rounds: list[dict]) -> list[dict]:
    rows = []
    for round_value in rounds:
        for candidate in round_value["candidates"]:
            rows.append(
                {
                    "round": round_value["round"],
                    "candidate_id": candidate["candidate_id"],
                    "molecule_id": candidate["molecule_id"],
                    "name": candidate["name"],
                    "smiles": candidate["smiles"],
                    "source_kind": candidate["source_kind"],
                    "vetoed": candidate["vetoed"],
                    "official_winner": candidate["candidate_id"]
                    == round_value["winner_id"],
                    "weighted_utility": candidate["weighted_utility"],
                    "similarity_seed": candidate["similarity_seed"],
                    **{
                        f"raw_{key}": candidate["raw"][key] for key in SCORE_KEYS
                    },
                    **{
                        f"objective_{key}": candidate["normalized"][key]
                        for key in SCORE_KEYS
                    },
                }
            )
    return rows


def _id_for_smiles(candidates: list[dict], smiles: str) -> str:
    for candidate in candidates:
        if candidate["smiles"] == smiles:
            return candidate["candidate_id"]
    raise ValueError("Decision references an unknown candidate")


def _by_id(candidates: list[dict], candidate_id: str) -> dict:
    for candidate in candidates:
        if candidate["candidate_id"] == candidate_id:
            return candidate
    raise KeyError(candidate_id)


def _signed(value: float) -> str:
    return f"{value:+.3f}"
