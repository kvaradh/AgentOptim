import pytest

from core.errors import NoSafeCandidateError, ProposalExhaustedError
from core.runner import load_run, run_optimization, validate_run


MOLECULES = ("CC", "CCC", "CCCC", "CCO", "CCN", "CCF")


def fake_proposer(parent, context):
    return [parent, *[smiles for smiles in MOLECULES if smiles != parent]][:6]


def fake_scorer(smiles):
    index = MOLECULES.index(smiles) if smiles in MOLECULES else 0
    return {
        "affinity": 0.9 - index * 0.05,
        "solubility": -3.0 + index * 0.1,
        "bbb": 0.9 - index * 0.03,
        "herg": 0.8 if index == 0 else 0.2,
        "sa": 2.0 + index * 0.1,
    }


def test_loop_runs_five_rounds_with_six_candidates_and_chaining():
    run = run_optimization(scorer=fake_scorer, proposer=fake_proposer)

    assert len(run["rounds"]) == 5
    assert all(
        len(round_value["candidates"]) == 6
        for round_value in run["rounds"]
    )
    for prior, current in zip(run["rounds"], run["rounds"][1:]):
        assert current["parent"]["smiles"] == prior["winner"]["smiles"]


def test_loop_accepts_an_explicit_seed_identity():
    seen_parents = []

    def tracking_proposer(parent, context):
        seen_parents.append(parent)
        return list(MOLECULES)

    run = run_optimization(
        scorer=fake_scorer,
        proposer=tracking_proposer,
        seed_smiles="OCC",
        seed_name="Ethanol test seed",
    )

    assert run["seed"]["name"] == "Ethanol test seed"
    assert run["seed"]["smiles"] == "CCO"
    assert seen_parents[0] == "CCO"


def test_loop_rejects_an_invalid_seed_before_scoring():
    def scorer_that_must_not_run(smiles):
        raise AssertionError("invalid seed reached the scorer")

    try:
        run_optimization(
            scorer=scorer_that_must_not_run,
            proposer=fake_proposer,
            seed_smiles="not-a-smiles",
            seed_name="Broken seed",
        )
    except ValueError as error:
        assert "Invalid seed SMILES" in str(error)
    else:
        raise AssertionError("invalid seed was accepted")


def test_loop_returns_a_typed_proposal_exhaustion():
    with pytest.raises(ProposalExhaustedError) as error:
        run_optimization(
            scorer=fake_scorer,
            proposer=lambda parent, context: ["CC", "CCC"],
        )

    assert error.value.code == "proposal_exhausted"


def test_loop_returns_a_typed_no_safe_candidate_outcome():
    def unsafe_scorer(smiles):
        value = fake_scorer(smiles)
        value["herg"] = 0.8
        return value

    with pytest.raises(NoSafeCandidateError) as error:
        run_optimization(scorer=unsafe_scorer, proposer=fake_proposer)

    assert error.value.code == "no_safe_candidate"


def test_replay_validation_rejects_a_safe_non_policy_winner():
    run = run_optimization(scorer=fake_scorer, proposer=fake_proposer)
    round_value = run["rounds"][-1]
    replacement = next(
        candidate
        for candidate in round_value["candidates"]
        if candidate["smiles"] != round_value["winner"]["smiles"]
        and candidate["raw"]["herg"] < 0.5
        and not candidate["incumbent"]
    )
    round_value["winner"] = replacement
    round_value["agents"]["orchestrator"]["chosen"] = replacement["smiles"]

    try:
        validate_run(run)
    except ValueError as error:
        assert "deterministic policy" in str(error)
    else:
        raise AssertionError("non-policy winner was accepted")


def test_cached_run_enforces_safety_and_improves_seed_tradeoffs():
    run = load_run()

    assert all(
        round_value["winner"]["raw"]["herg"] < 0.50
        for round_value in run["rounds"]
    )
    seed = run["seed"]["raw"]
    final = run["rounds"][-1]["winner"]["raw"]
    assert final["solubility"] > seed["solubility"]
    assert final["herg"] < seed["herg"]
    assert final["affinity"] > 0.90
