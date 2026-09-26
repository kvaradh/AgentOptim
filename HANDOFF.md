# Handoff brief — multi-agent DRD2 lead optimisation

**For:** a coding agent picking this up with unrestricted network and API access.
**Repo:** `kvaradh/AgentOptim`, branch `claude/multi-agent-molecule-opt-b0sbiy`
**Companion repo:** `tuhinc5203/admet-property-prediction` (the trained ADMET models)

Read sections 1–3 before writing any code. They contain four findings that cost
real time to discover and that you will otherwise rediscover the hard way.

---

## 1. What exists and what is actually verified

Working, tested, offline:

| component | file | state |
|---|---|---|
| five-axis contract, normalisation | `core/contract.py` | verified, 54 tests |
| `score()` with per-axis provenance | `core/oracles.py` | verified |
| ADMET backend (real XGBoost) | `core/admet.py` | **verified against source repo** |
| edit engine, 19 SMARTS transforms | `core/edits.py` | verified |
| 3 specialists + orchestrator | `core/agents.py` | **deterministic path only** |
| round loop, Pareto, hypervolume | `core/loop.py` | verified |
| Streamlit UI + headless figures | `frontend/` | smoke-tested on surrogate data |
| model verification gate | `scripts/verify_models.py` | works |
| 3-arm ablation | `scripts/ablation.py` | works, **numbers are stale** |

**Not verified — read this twice:**

- **The LLM code path has never executed.** No API key was available. Everything
  in `core/llm.py`, the `complete_json` calls in `core/agents.py`, and
  `_llm_propose` in `core/edits.py` is written but completely unexercised. Every
  measurement in this repo and in the README came from `AGENT_LLM=off`, i.e.
  three deterministic functions emitting template strings. **Your first job is
  to run this path and fix what breaks.**
- **Affinity is a surrogate.** `SimilarityAffinityOracle` — max Tanimoto to a
  six-ligand panel. PyTDC's DRD2 oracle downloads from `dataverse.harvard.edu`,
  which the previous environment blocked. You can reach it. Wire it in.
- **All ablation numbers predate the real ADMET models.** Re-measure everything
  before quoting any of it.

---

## 2. Four findings you must not rediscover

### 2.1 The featurization order is not the obvious one

The models take a **named pandas DataFrame**, 2054 columns:
`fp_0 … fp_2047`, then six descriptors in this exact order:

```
MolWt, MolLogP, TPSA, NumHDonors, NumHAcceptors, NumRotatableBonds
```

TPSA is **third, not last**. Salts must be stripped first (largest fragment).
`core/admet.py:featurize` already does this and was diffed against
`app.py:featurize_one` in the source repo on five molecules including a salt
form — bit-for-bit identical. **Do not rewrite it.**

The estimators carry `feature_names_in_`, so a wrongly-named DataFrame raises.
A bare numpy array of the right width does **not** — it is silently scored
against mismatched columns and every number downstream is void with nothing
crashing. Always pass the DataFrame.

### 2.2 An absolute hERG veto kills the demo in round 1

The real hERG model scores the seed haloperidol at **P(blockade) = 0.865**.
Risperidone 0.923, chlorpromazine 0.912, aripiprazole 0.909. The model is
correct — these are hERG blockers and haloperidol carries a QT warning.

With an absolute 0.70 line the safety agent vetoes the seed itself, then **0 of
14** first-round analogues. The loop has nothing to select, forever.

`HERG_VETO_MODE = "auto"` in `core/contract.py` fixes this: absolute line while
the lead is clean, **relative to the incumbent** once the lead is already
liable. 7 of those 14 analogues are safer than the parent and 7 are worse, so
the veto discriminates within the series. Keep this behaviour; there is a
regression test for it.

### 2.3 The Pareto front cannot be plotted on affinity

Haloperidol sits at the ceiling of any DRD2 activity oracle. Every edit can only
lose potency, so a front seeded in that corner cannot move — measured flat at
0.877 across five rounds. The front is drawn on **solubility × hERG safety**
(the seed's real liabilities) and affinity is held as a reported programme floor
(`AFFINITY_FLOOR = 0.60`). Re-check this once the real DRD2 oracle is in: it may
score analogues *above* the parent, which the similarity surrogate never can.

### 2.4 The orchestrator does not deliberate — and this is the real problem

`agents.orchestrate()` is `argmax(weighted_sum)` over candidates surviving the
veto and the affinity floor. **It never reads the specialists' `reason` strings.**
Measured consequence: replacing all four agents with one generic agent carrying
the same two constraints gives **bit-identical** output (+0.000 over 12 seeds).

Two further measurements:
- The veto fires ~0.4 molecules/round but in **0 of 20 seeds** removed the
  top-scoring candidate. It only killed molecules already losing.
- Feeding agent objections back into the edit engine made results **worse**
  (0.263 vs 0.301). All four feedback rules are measured in
  `FEEDBACK_RULE` in `core/loop.py`.

**Why:** all three agents read the same table of numbers. Different `max()` calls
over one shared spreadsheet is arithmetic, not deliberation. For agents to argue
there must be **information asymmetry** — each must know something the others
and the score table do not.

This is the gap between what is built and what the team actually wants.

---

## 3. The team's actual intent

> Specialist agents genuinely trained/knowledgeable in their domain, each
> optimising for its own property, arguing to an orchestrator that balances what
> it takes to be a good drug.

Two independent requirements:

- **(a)** each agent genuinely knows its domain
- **(b)** what agents say changes what gets chosen

**(b) is the blocker.** Fixing (a) without (b) buys better-informed narration.

A worked example of the value (a) adds: the number `P(hERG) = 0.865` does not
tell you what to do. Domain knowledge does — *this whole structural class is
liable, so the question is not whether 0.865 is acceptable but whether the
analogue makes an already-known liability worse.* That reasoning produced
finding 2.2. No column in the score table could have.

---

## 4. Setup

```bash
git clone https://github.com/kvaradh/AgentOptim && cd AgentOptim
git checkout claude/multi-agent-molecule-opt-b0sbiy
pip install -r requirements.txt
pip install xgboost joblib pandas scikit-learn anthropic PyTDC

# the trained ADMET models (not committed -- they belong to the other repo)
git clone --depth 1 https://github.com/tuhinc5203/admet-property-prediction /tmp/admet
cp /tmp/admet/models/*.pkl models/

export ANTHROPIC_API_KEY=...
```

`PyTDC` fails to build under current setuptools. Use a venv with
`pip install --upgrade setuptools wheel` first.

Baseline check — must pass before you change anything:

```bash
pytest -q                                          # expect 54 passed
ADMET_BACKEND=xgboost python -m core.loop --rounds 5
```

Environment switches: `AGENT_LLM=off`, `EDIT_ENGINE=mutation`,
`ADMET_BACKEND=xgboost|surrogate`, `AFFINITY_ORACLE=tdc|similarity`,
`FEEDBACK_RULE=weakest|none|disputed|contested-weak`.

---

## 5. Workstreams

Ordering matters. **W0 first, alone.** Then W1/W3/W4 in parallel. W2 after W1.
W5 last.

### W0 — Exercise the LLM path *(do this first, nothing else until it passes)*

**Why first:** everything else assumes this works and nothing has proven it does.

1. Set `ANTHROPIC_API_KEY`. Run `python -m core.loop --rounds 3` with **no**
   `AGENT_LLM=off`.
2. Expect failures in JSON parsing, `preferred` SMILES not matching a candidate
   exactly, timeouts, and rate limits. `core/llm.py:complete_json` swallows
   every exception and returns `None`, which means **failures are currently
   invisible** — they silently degrade to the deterministic path.
3. Add a debug mode that logs why `complete_json` returned `None` instead of
   discarding the reason. You cannot tune prompts you cannot see failing.
4. Verify the LLM verdicts differ from the deterministic ones. If the LLM
   only ever rubber-stamps the deterministic verdict, say so — that is a finding.

**Done when:** a 3-round run completes with LLM agents, and you can state what
fraction of LLM calls succeeded vs. fell back.

**Files:** `core/llm.py`, `core/agents.py`, `core/edits.py`

---

### W1 — Give each specialist private domain knowledge *(parallel)*

**Goal:** create the information asymmetry that makes argument possible. Each
agent gets a tool the others do not have.

Build `core/knowledge.py`:

- **hERG agent:** structural alerts for the classic pharmacophore (basic amine +
  lipophilic aromatic at the characteristic distance), computed basicity/pKa
  proxy, and mitigation strategies (reduce basicity, add polarity, zwitterion).
  Implement as RDKit SMARTS matches, not prose.
- **ADME agent:** CNS MPO score (Wager et al.), P-gp efflux liability proxy,
  and the solubility/BBB tension made explicit.
- **Affinity agent:** DRD2 pharmacophore requirements — which features cannot be
  removed without losing binding.

Each specialist's prompt then carries findings from **its own tool only**. The
orchestrator sees the conclusions, not the tools.

**Done when:** each agent's prompt contains at least one fact about the specific
candidate that is not in the five-number score table, and you can show a case
where two agents disagree about a molecule *on substance*, not just argmax.

**Files:** new `core/knowledge.py`, prompts in `core/agents.py`

---

### W2 — Make deliberation load-bearing *(after W1 — same files)*

**Goal:** fix finding 2.4. The orchestrator must reconcile conflicting
assessments rather than take an argmax.

Candidate mechanisms, in rough order of value:

1. **Agents propose chemistry, not just rankings.** A safety agent that knows
   hERG SAR suggests reducing piperidine basicity or adding an acid to form a
   zwitterion. The 19 fixed transforms cannot invent those. Route agent
   proposals into `edits.propose()`.
2. **Agents report applicability-domain confidence.** "This molecule is outside
   my model's training distribution" is information the bare probability hides,
   and it makes the team's honesty point mechanical rather than spoken.
3. **Orchestrator reconciles rather than sums.** Weights set by the strength of
   argument, or an explicit conflict-resolution step.

**Acceptance test — this is the whole point:** re-run
`scripts/ablation.py`. The `specialists` arm must now differ measurably from
`generic_constrained`. If it still reads +0.000, deliberation is still
narration and you have not fixed it. **Report the number honestly either way.**

**Files:** `core/agents.py`, `core/loop.py`, `core/edits.py`

---

### W3 — Real DRD2 affinity oracle *(parallel)*

1. `from tdc import Oracle; Oracle(name='DRD2')` — downloads from
   `dataverse.harvard.edu`.
2. Wire into `TDCAffinityOracle` in `core/oracles.py` (the class already exists
   and is already the preferred path; it just could not download).
3. **Re-check finding 2.3.** The similarity surrogate decays monotonically so
   affinity could only fall, which made `AFFINITY_FLOOR` a ratchet — only 1.4 of
   6 candidates survived per round and 36% of rounds held the parent. A real
   oracle may score analogues above the parent. Re-measure survivors/round and
   retune `AFFINITY_FLOOR` accordingly.
4. Confirm haloperidol's real DRD2 score. If it is ~1.0, finding 2.3 stands and
   the front must stay off the affinity axis.

**Done when:** `provenance()["affinity"] == "tdc-drd2"` and you report
survivors-per-round before and after.

**Files:** `core/oracles.py`

---

### W4 — Frontend against real data *(parallel)*

The UI was only smoke-tested on surrogate data (HTTP 200, renders).

1. Re-run with real ADMET and check every panel.
2. Surface the veto **mode** — a relative veto needs explaining on screen or it
   looks like an arbitrary threshold. Show the line and which rule set it.
3. Keep the surrogate banner honest: it must fire whenever *any* axis is a
   surrogate, and stop firing once none are.
4. Verify `python -m frontend.plots demo_run.json figures/` still works — that
   is the fallback if Streamlit dies during the demo.

**Files:** `frontend/app.py`, `frontend/plots.py`

---

### W5 — Re-measure everything *(last)*

Every headline number is stale. Re-run on real models and rewrite the README:

```bash
python -m scripts.ablation --seeds 12 --rounds 5
```

Re-measure: the 3-arm comparison, the four `FEEDBACK_RULE` values, how often the
veto removes the top candidate (was 0/20), and survivors-per-filter-stage
(was 6 → 5.6 → 1.4).

**Methodological note:** use ≥12 seeds. A single-seed win on a stochastic search
is not a result. Keep the three ablation arms sharing identical oracles.

---

## 6. Rules for whoever does this

1. **Never let a surrogate borrow a trained model's accuracy claim.** Every
   record carries `provenance`; the CLI warns and the UI shows a red banner. The
   README quotes BBB ROC-AUC 0.920 — that belongs to the XGBoost model, never to
   a fallback. This is the one failure mode here that is dishonest rather than
   merely broken.
2. **Report negative results.** "Deliberation still adds nothing" is a finding,
   not a failure to hide. The most valuable output of this project so far is a
   measured negative result.
3. **Re-run `pytest -q` before every commit.** 54 tests. Several encode
   findings above — particularly that the orchestrator can never select a vetoed
   molecule, and that a liable lead does not veto its own analogues wholesale.
4. **Do not widen scope silently.** If a fix requires changing the contract in
   `core/contract.py`, say so explicitly — every workstream codes against it.
