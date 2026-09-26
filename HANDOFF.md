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

### 3.1 Finding 2.4 is a theorem, not a bug

Given that (i) all agents observe the same score matrix `S`, (ii) each emits an
argmax over one column of `S`, and (iii) the orchestrator computes `w·S`, the
decision is measurable with respect to `S` and the agents' text is
conditionally independent of it. Bit-identical output is forced, not surprising.

The generalisation that should drive every design choice: **an agent earns its
place only if removing it changes either (a) the set of molecules evaluated, or
(b) a decision input not derivable from the scores.**

Supporting literature (see the caveat in §3.3):
- *Debate or Vote* (arXiv 2508.17536) proves debate induces a **martingale** over
  agents' belief trajectories, so debate alone cannot improve expected
  correctness; majority voting accounts for most reported gains across 7
  benchmarks.
- *Why Do Multi-Agent LLM Systems Fail?* (arXiv 2503.13657), 1,600+ traces:
  44.2% of failures are system-design, 32.3% inter-agent misalignment. Not
  fixable by prompt tweaks.
- Single-agent matches or beats multi-agent once **thinking tokens are
  normalised** (arXiv 2604.02460). Most published MAS wins are unnormalised
  test-time compute.
- *LLMs Cannot Self-Correct Reasoning Yet* (arXiv 2310.01798): intrinsic
  self-critique *degrades* accuracy. A critic agent is expected-negative unless
  its critique is a **tool output**, not its own judgement.
- **Personas do not work.** arXiv 2311.10054 (162 personas × 4 model families ×
  2,410 questions): no improvement. Wharton GAIL replication found nine
  statistically significant *negative* effects. "You are a world-class cardiac
  safety expert" buys register, not judgement. Keep one line for readable
  output; never report it as specialisation.

The pattern across systems where multi-agent structure *is* demonstrably
load-bearing: they give agents **different tools or different data**. MT-Mol
(arXiv 2505.20820) partitions 154 RDKit/PubChem tools across five analyst
agents and ablates properly — removing analysts drops albuterol_similarity AUC
0.998→0.750. MultiMol (arXiv 2503.03503) splits a fine-tuned generator from a
literature-retrieval agent. MOLLEO (arXiv 2406.16976) uses the LLM as a genetic
operator with **no deliberation at all** — the cleanest evidence that LLM value
here comes from *proposal*, not *argument*.

### 3.2 The genuine conflict in this target — and it is not in any score column

For haloperidol/DRD2, **the DRD2 pharmacophore and the hERG pharmacophore are
nearly the same pharmacophore.** This is the substantive argument the agents
should be having:

- **DRD2** needs a **protonated basic nitrogen** salt-bridging **Asp114 (3.32)**
  in TM3, plus lipophilic aryls. Haloperidol binds unusually deep, in an
  extended pocket formed by TM3/TM5/TM6 that accommodates the butyrophenone —
  that 4-fluorophenyl ketone is the **subtype-selectivity element**, not
  decoration (DRD2–haloperidol structure, Nat Commun 2020).
- **hERG** blockade needs a **protonated basic nitrogen** making a cation–π with
  **Tyr652**, plus three hydrophobic/aromatic centroids contacting **Phe656**.
  Risk rises jointly with logP and amine pKa.
- **CNS MPO** (Wager et al. 2010) wants **HBD → 0** and **most-basic pKa ≤ 8**,
  with clogP ≤ 3, MW ≤ 360, TPSA 40–90 Å². Threshold for desirability is ≥ 4.

So: DRD2 wants the basic amine. hERG wants it gone. hERG mitigation wants added
polarity and hydroxyls — which CNS MPO penalises as HBDs. Three
literature-backed positions, genuinely mutually incompatible.

Two facts your agents should state on turn one:
- **Haloperidol fails CNS MPO on MW (375.9), clogP (~4) and pKa (~8.3)
  simultaneously.** The seed is a mediocre CNS compound.
- **Haloperidol is a clinically flagged hERG blocker** (Suessbrich, Br J
  Pharmacol 1997, IC50 ≈ 1 µM in oocytes — mammalian values lower; QT warnings
  on label). Your seed molecule *is* the liability. That is the project.

**The resolution, which no score table contains:** the basic nitrogen is not
strictly required for D2 antagonism. Non-basic D2 antagonists exist and have
been purpose-built (Molecules 2023, 28, 4211; CoMFA/MD studies of D2 antagonists
without a protonatable nitrogen). An affinity agent that knows this can say
something an argmax never will. **That is the demo.**

hERG mitigations in the order a chemist would try them: lower clogP; lower amine
pKa (β/γ-fluorination, insert ether O, piperidine→morpholine, ring contraction,
the 5-amino-1,3-dioxanyl motif); add polarity/OH; **zwitterion** (append an acid
to kill the net cation Tyr652 needs); restrict conformation; replace the amine
outright (e.g. hydroxylamine isosteres).

### 3.3 Verification caveat on §3.1–3.2

These findings come from a literature search whose agent **could not fetch the
PDFs** — publisher and preprint domains were blocked by its egress policy. Links
and claims are from search-index summaries, not first-hand reads. The structural
biology and CNS MPO parameters are standard and low-risk. **The quantitative
claims from individual papers are second-hand — verify against the PDF before
putting any of them in a writeup or on a slide.** Treat MultiMol's 82.3% vs
27.5% and PharmAgents' 15.7%→37.9% as unverified.

### 3.4 Your real baseline is Graph-GA, not "one LLM"

PMO (arXiv 2206.12411), 25 algorithms × 23 tasks: most "SOTA" molecular
optimisers fail to beat their predecessors under sample-efficiency constraints,
and **Graph-GA beats more recent methods**. Expect your agent system to lose to
Graph-GA plus your oracles. Report it if it does.

Also: the TDC DRD2 oracle is an ECFP6 Gaussian-kernel SVM (Olivecrona 2017,
ExCAPE-DB). Goal-directed benchmarks on it reward synthetically unrealistic
structures and its out-of-distribution behaviour is poor. Three agents arguing
about that SVM's extrapolations are arguing about its artifacts. Naming this in
the writeup is a strength.

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

### W1 — Information asymmetry + private fact-emitting tools *(parallel)*

Ordered by value-per-hour. W1a alone kills the original critique.

**W1a — Mask the score table per agent (~20 min, best ratio in the project).**
Each agent sees SMILES + **only its own columns**. One dict comprehension in
`core/agents.py`. Immediately the safety agent's veto carries information the
affinity agent provably lacks, so the orchestrator performs aggregation rather
than arithmetic. This is the iAgents (NeurIPS 2024) setting: collaboration is
necessary precisely when each agent accesses only part of the information.

**W1b — One private tool per agent, emitting hard facts (~45 min).**
Build `core/knowledge.py`. Each tool must emit a **fact the agent only has to
report**, never something it must reason *through* — ChemToolAgent (arXiv
2411.07228) found tool-augmented agents underperform their own base LLM when
the agent has to reason through tool output.

- **Safety:** `rdkit.Chem.rdfiltercatalog.FilterCatalog` with BRENK + PAINS +
  NIH → alert names; basic-amine detection; crude pKa/logP hERG risk flag.
- **ADME:** CNS MPO — six desirability functions, five are one-line RDKit
  descriptors; approximate pKa from amine class. Report the 0–6 score and which
  parameters fail.
- **Affinity:** pharmacophore preservation — Murcko scaffold, basic N present?,
  aromatic ring count, ECFP4 Tanimoto to haloperidol.

Do **not** hard-code an N⁺-to-aromatic-centroid distance: it varies up to ~7.2 Å
across D2 antagonists.

**W1c — MMP transformation few-shots for the safety agent (~45 min).**
~20–25 hERG-mitigation transformations from §3.2, written as reaction SMARTS the
existing engine already accepts (piperidine→morpholine, β-fluorination of the
amine, append CH₂COOH for zwitterion, insert ether O in the linker, aryl→
heteroaryl). This is literature expertise **as executable edits** — the only
form of domain knowledge that can change a molecule. Scale later with `mmpdb`.

**Done when:** each agent's prompt contains at least one fact about the specific
candidate absent from the five-number table, and you can show two agents
disagreeing **on substance** — ideally the §3.2 conflict, where ADME wants HBD→0
and safety wants the hydroxyls and acid that mitigate hERG.

**Files:** new `core/knowledge.py`, `core/agents.py`, `core/edits.py`

---

### W2 — Make deliberation load-bearing *(after W1 — same files)*

**Goal:** fix finding 2.4. The orchestrator must reconcile conflicting
assessments rather than take an argmax.

Mechanisms, in measured order of value:

1. **Turn agents from voters into proposers (~45 min). Highest value, full
   stop.** Give each specialist the mutation engine and its own objective; each
   proposes *k* edits to the current lead; the union is the batch. This alone
   makes the ablation come out differently — delete the ADME agent and the
   candidate set *provably* changes. Precedent: MOLLEO uses the LLM as a genetic
   operator with no deliberation and still beats EA/RL/BO baselines, which is
   the evidence that proposal, not argument, is where LLM value lives here.
2. **Replace the weighted sum with Pareto + hard vetoes (~30 min).**
   Non-dominated filter, then per-agent veto with a **machine-checkable reason**
   (BRENK/PAINS hit, CNS MPO < 4, basic nitrogen lost, ECFP4 similarity to
   haloperidol < 0.4), then tie-break. Once a veto is a boolean derived from a
   substructure match, **the orchestrator's output is no longer a function of
   the score matrix and the original critique stops applying on its own terms.**
   Bonus: a weighted sum can only reach Pareto-optimal points on **convex**
   regions of the front — non-convex regions are unreachable by *any* fixed
   weight vector. If you keep it scalar, use the **geometric mean**, which
   penalises any single bad axis.
3. **Agents report applicability-domain confidence.** "This molecule is outside
   my model's training distribution" is information the bare probability hides,
   and it makes the team's honesty point mechanical rather than spoken.

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
(was 6 -> 5.6 -> 1.4).

**Add a fourth arm, and it is the important one.** Run:

  (a) weighted sum only, no agents
  (b) **one agent with ALL the tools**   <- nobody in the literature runs this
  (c) multi-agent, shared table
  (d) multi-agent, asymmetric + proposing

Arm (b) is the one a reviewer will ask for and the one the field under-runs. If
(b) beats (d), say so — that result is more valuable than a demo that hides it.

**Normalise by tokens, or you reproduce the field's central confound.** Anthropic's
own multi-agent system uses ~15x the tokens and token usage explains ~80% of its
performance variance; the equal-token study (arXiv 2604.02460) finds single-agent
matches or beats multi-agent once compute is held equal. An unnormalised
MAS-vs-single comparison is not a comparison.

**Also baseline against Graph-GA** (see 3.4). It is the honest thing to beat.

**Methodological note:** use >=12 seeds. A single-seed win on a stochastic search
is not a result. Keep every ablation arm sharing identical oracles.

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
