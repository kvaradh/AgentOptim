"""The ablation: four specialists against one generalist, same seed and oracles.

This is the only comparison in the build that is evidence rather than
demonstration, so it is also the one most able to embarrass you. Run it over
several RNG seeds before quoting a number. A single-seed win on a stochastic
search is not a result, and a judge who has run an experiment will know it.

    python -m scripts.ablation --seeds 8 --rounds 5
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from core.loop import run


def arm(mode: str, seeds: list[int], rounds: int) -> dict:
    records = [
        run(rounds=rounds, mode=mode, rng_seed=seed, verbose=False) for seed in seeds
    ]
    hypervolumes = [r["hypervolume_final"] for r in records]
    affinities = [r["affinity_final"] for r in records]
    return {
        "mode": mode,
        "n": len(records),
        "hypervolume_mean": statistics.fmean(hypervolumes),
        "hypervolume_stdev": statistics.stdev(hypervolumes) if len(records) > 1 else 0.0,
        "hypervolume_values": hypervolumes,
        "affinity_mean": statistics.fmean(affinities),
        "affinity_held_fraction": sum(r["affinity_held"] for r in records) / len(records),
        "herg_violations": sum(
            1
            for r in records
            if r["final"]["normalised"]["herg"] < r["weights"].get("_unused", 0.30)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="specialists vs one generic agent")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("ablation.json"))
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    specialists = arm("specialists", seeds, args.rounds)
    generic = arm("generic", seeds, args.rounds)
    generic_constrained = arm("generic_constrained", seeds, args.rounds)

    print(f"{'arm':14s} {'hypervolume':>22s} {'affinity':>10s} {'held':>6s}")
    for result in (specialists, generic_constrained, generic):
        print(
            f"{result['mode']:14s} "
            f"{result['hypervolume_mean']:10.3f} +/- {result['hypervolume_stdev']:<7.3f} "
            f"{result['affinity_mean']:10.2f} "
            f"{result['affinity_held_fraction']:6.0%}"
        )

    delta = specialists["hypervolume_mean"] - generic["hypervolume_mean"]
    pooled = max(specialists["hypervolume_stdev"], generic["hypervolume_stdev"], 1e-9)
    print(f"\nhypervolume delta {delta:+.3f}  ({delta / pooled:+.2f} pooled sd)")
    if abs(delta) < pooled:
        print(
            "Within one standard deviation across seeds. State it as inconclusive.\n"
            "The defensible claim is the hERG constraint, not the front:"
        )
    print(
        f"affinity floor held: specialists {specialists['affinity_held_fraction']:.0%}, "
        f"generic {generic['affinity_held_fraction']:.0%}"
    )

    ceiling = generic_constrained["hypervolume_mean"]
    print(
        f"\nspecialists vs one agent WITH the same hard constraints: "
        f"{specialists['hypervolume_mean'] - ceiling:+.3f}"
    )
    print(
        "  If that gap is ~0, the architecture's advantage is the constraints, not the\n"
        "  deliberation. Say so before a judge says it for you."
    )

    args.out.write_text(
        json.dumps(
            {"rounds": args.rounds, "seeds": seeds,
             "specialists": specialists,
             "generic_constrained": generic_constrained,
             "generic": generic},
            indent=2,
        )
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
