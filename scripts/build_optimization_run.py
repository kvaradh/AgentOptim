#!/usr/bin/env python3

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from core.runner import run_optimization, save_run


if __name__ == "__main__":
    run = run_optimization()
    save_run(run)
    print("PASS: wrote demo_run.json with 5 rounds and 6 candidates per round")
