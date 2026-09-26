"""Five-round optimization loop and replay serialization."""

from datetime import datetime, timezone
import json
from pathlib import Path

from rdkit import Chem

from .agents import deliberate, expected_official_candidate
from .contract import (
    CANDIDATES_PER_ROUND,
    SCORE_KEYS,
    SEED_SMILES,
    is_valid,
    normalize,
    propose,
    score,
)
from .errors import ProposalExhaustedError, UnsupportedSeedError


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEMO_RUN_PATH = REPOSITORY_ROOT / "demo_run.json"
SCHEMA_VERSION = 2


def run_optimization(
    *,
    rounds: int = 5,
    candidates_per_round: int = CANDIDATES_PER_ROUND,
    scorer=score,
    proposer=propose,
    llm_client=None,
    seed_smiles: str = SEED_SMILES,
    seed_name: str = "Haloperidol",
) -> dict:
    """Run optimization and return a JSON-serializable record."""
    if rounds != 5:
        raise ValueError("The hackathon contract requires exactly five rounds")
    if candidates_per_round != 6:
        raise ValueError("The hackathon contract requires six candidates")
    if not is_valid(seed_smiles):
        raise UnsupportedSeedError(f"Invalid seed SMILES: {seed_smiles}")
    if not isinstance(seed_name, str) or not seed_name.strip():
        raise ValueError("Seed name must be a non-empty string")

    canonical_seed = Chem.MolToSmiles(Chem.MolFromSmiles(seed_smiles))
    seed_raw = scorer(canonical_seed)
    parent = {
        "smiles": canonical_seed,
        "raw": seed_raw,
        "normalized": normalize(seed_raw),
    }
    result_rounds = []

    for round_number in range(1, rounds + 1):
        try:
            proposed = proposer(
                parent["smiles"],
                {
                    "round": round_number,
                    "scores": parent["raw"],
                    "agent_notes": (
                        []
                        if not result_rounds
                        else [
                            result_rounds[-1]["agents"]["orchestrator"][
                                "rationale"
                            ]
                        ]
                    ),
                },
            )
        except RuntimeError as error:
            raise ProposalExhaustedError(
                f"Round {round_number} could not generate a complete proposal slate"
            ) from error
        if len(proposed) != candidates_per_round:
            raise ProposalExhaustedError(
                f"Round {round_number} returned {len(proposed)} proposals"
            )

        candidates = []
        for smiles in proposed:
            raw = scorer(smiles)
            candidates.append(
                {
                    "smiles": smiles,
                    "raw": raw,
                    "normalized": normalize(raw),
                    "incumbent": smiles == parent["smiles"],
                }
            )

        agents = deliberate(candidates, llm_client=llm_client)
        winner = next(
            item
            for item in candidates
            if item["smiles"] == agents["orchestrator"]["chosen"]
        )
        result_rounds.append(
            {
                "round": round_number,
                "parent": parent,
                "candidates": candidates,
                "agents": agents,
                "winner": winner,
            }
        )
        parent = winner

    run = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target": "DRD2",
        "execution_mode": "live",
        "seed": {
            "name": seed_name.strip(),
            "smiles": canonical_seed,
            "raw": seed_raw,
            "normalized": normalize(seed_raw),
        },
        "score_keys": list(SCORE_KEYS),
        "proposal_contract": (
            "deterministic_v1" if proposer is propose else "unknown"
        ),
        "rounds": result_rounds,
    }
    validate_run(run)
    return run


def save_run(run: dict, path: str | Path = DEMO_RUN_PATH):
    destination = Path(path)
    destination.write_text(json.dumps(run, indent=2) + "\n")


def load_run(path: str | Path = DEMO_RUN_PATH) -> dict:
    run = json.loads(Path(path).read_text())
    validate_run(run)
    return run


def validate_run(run: dict):
    if run.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported optimization replay schema")
    if run.get("target") != "DRD2":
        raise ValueError("The replay target must be DRD2")
    if run.get("execution_mode", "cached") not in {"cached", "live"}:
        raise ValueError("Replay execution mode is unsupported")
    if run.get("score_keys") != list(SCORE_KEYS):
        raise ValueError("Replay score axes do not match the core contract")
    if run.get("proposal_contract", "unknown") not in {
        "deterministic_v1",
        "unknown",
    }:
        raise ValueError("Replay proposal provenance is unsupported")
    rounds = run.get("rounds", [])
    if len(rounds) != 5:
        raise ValueError("Replay must contain exactly five rounds")

    expected_parent = run["seed"]["smiles"]
    for expected_number, round_value in enumerate(rounds, start=1):
        if round_value["round"] != expected_number:
            raise ValueError("Replay rounds are out of sequence")
        if round_value["parent"]["smiles"] != expected_parent:
            raise ValueError("A round does not continue from the prior winner")
        candidates = round_value.get("candidates", [])
        if len(candidates) != 6:
            raise ValueError("Each replay round must have six candidates")
        smiles_values = {item["smiles"] for item in candidates}
        if len(smiles_values) != 6:
            raise ValueError("Each replay round must have six unique candidates")
        incumbents = [item for item in candidates if item.get("incumbent")]
        if len(incumbents) != 1 or incumbents[0]["smiles"] != expected_parent:
            raise ValueError("Each replay round must contain its parent once")
        for candidate in candidates:
            if set(candidate.get("raw", {})) != set(SCORE_KEYS):
                raise ValueError("Candidate raw scores do not match the contract")
            if candidate.get("normalized") != normalize(candidate["raw"]):
                raise ValueError("Candidate normalization does not match the contract")
        winner = round_value["winner"]["smiles"]
        if winner not in smiles_values:
            raise ValueError("Round winner is not one of its candidates")
        if round_value["agents"]["orchestrator"]["chosen"] != winner:
            raise ValueError("Orchestrator and recorded winner disagree")
        recorded_winner = next(
            candidate for candidate in candidates if candidate["smiles"] == winner
        )
        if round_value["winner"] != recorded_winner:
            raise ValueError("Recorded winner payload differs from its candidate")
        expected_winner = expected_official_candidate(candidates)["smiles"]
        if winner != expected_winner:
            raise ValueError("Recorded winner does not match the deterministic policy")
        expected_parent = winner
