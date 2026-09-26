(() => {
  "use strict";

  const byId = (id) => document.getElementById(id);
  const all = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const clean = (value, fallback = "—") => {
    if (value === null || value === undefined || value === "") return fallback;
    return String(value);
  };
  const number = (value, digits = 3) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed.toFixed(digits) : "—";
  };
  const signed = (value, digits = 3) => {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) return "—";
    return `${parsed >= 0 ? "+" : ""}${parsed.toFixed(digits)}`;
  };
  const percent = (value) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? `${(parsed * 100).toFixed(1)}%` : "Not estimable";
  };
  const titleCase = (value) => clean(value).replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());

  function setText(id, value) {
    const node = byId(id);
    if (node) node.textContent = clean(value);
  }

  function node(tag, className, textValue) {
    const created = document.createElement(tag);
    if (className) created.className = className;
    if (textValue !== undefined) created.textContent = clean(textValue);
    return created;
  }

  function appendTextRow(parent, term, description) {
    const wrapper = node("div");
    wrapper.append(node("dt", "", term), node("dd", "", description));
    parent.append(wrapper);
    return wrapper;
  }

  function payloadBytes(id) {
    const source = byId(id);
    if (!source) throw new Error(`Missing embedded payload: ${id}`);
    const encoded = source.textContent.trim();
    const binary = window.atob(encoded);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index);
    }
    return bytes;
  }

  function readMission() {
    const bytes = payloadBytes("mission-payload");
    return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  }

  let mission;
  try {
    mission = readMission();
  } catch (error) {
    const main = byId("workspace-main");
    if (main) {
      const message = node("p", "noscript", "The embedded mission package could not be read.");
      main.prepend(message);
    }
    return;
  }

  const scoreLabels = {
    affinity: "Target activity",
    solubility: "Solubility",
    bbb: "BBB",
    herg: "hERG safety",
    sa: "SA",
  };
  const agentProfiles = {
    affinity: { name: "Ari", initials: "AR", role: "Target activity specialist" },
    adme: { name: "Maya", initials: "MY", role: "ADME specialist" },
    safety: { name: "Cora", initials: "CO", role: "Cardiac safety specialist" },
    orchestrator: { name: "Perry", initials: "PE", role: "Decision orchestrator" },
  };
  const viewNames = new Set(["mission", "agents", "optimize", "decisions", "body-map", "validate", "evidence"]);
  const finalRound = mission.rounds[mission.rounds.length - 1];
  const state = {
    view: "mission",
    round: finalRound.round,
    candidateId: mission.summary.final_candidate_id,
    playing: false,
    playbackStartedAt: 0,
    playbackElapsed: 25000,
    animationFrame: 0,
    depictionUrl: "",
    highlightedAssay: "",
  };

  function currentRound() {
    return mission.rounds.find((item) => item.round === state.round) || finalRound;
  }

  function candidateInRound(roundValue, candidateId) {
    return roundValue.candidates.find((item) => item.candidate_id === candidateId);
  }

  function selectedCandidate() {
    const roundValue = currentRound();
    return candidateInRound(roundValue, state.candidateId)
      || candidateInRound(roundValue, roundValue.winner_id)
      || roundValue.candidates[0];
  }

  function moleculeFor(candidate) {
    return mission.molecules[candidate.molecule_id];
  }

  function announce(message) {
    const toast = byId("toast");
    if (!toast) return;
    toast.textContent = clean(message);
    toast.classList.add("is-visible");
    window.clearTimeout(announce.timer);
    announce.timer = window.setTimeout(() => toast.classList.remove("is-visible"), 2600);
  }

  function showView(name, shouldFocus = false) {
    if (!viewNames.has(name)) return;
    state.view = name;
    all("[data-view-panel]").forEach((panel) => {
      panel.hidden = panel.dataset.viewPanel !== name;
    });
    all("[data-view]").forEach((button) => {
      if (button.dataset.view === name) {
        button.setAttribute("aria-current", "page");
      } else {
        button.removeAttribute("aria-current");
      }
    });
    const select = byId("view-select");
    if (select) select.value = name;
    if (name === "body-map") {
      renderBodyMap();
      window.requestAnimationFrame(resizeConformer);
    }
    if (name === "validate") renderWetLab();
    if (shouldFocus) {
      const main = byId("workspace-main");
      if (main) main.focus({ preventScroll: true });
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  }

  function setRound(roundNumber, chooseWinner = true) {
    const parsed = Number(roundNumber);
    const target = mission.rounds.find((item) => item.round === parsed);
    if (!target) return;
    state.round = parsed;
    if (chooseWinner || !candidateInRound(target, state.candidateId)) {
      state.candidateId = target.winner_id;
    }
    renderOptimize();
    renderDecisions();
    if (state.view === "body-map") renderBodyMap();
  }

  function stateLabels(candidate, roundValue) {
    const labels = [];
    if (candidate.candidate_id === roundValue.winner_id) labels.push(["Official", "official"]);
    if (candidate.vetoed) labels.push(["Veto", "veto"]);
    if (candidate.safe_frontier) labels.push(["Frontier", ""]);
    if (candidate.incumbent) labels.push(["Incumbent", ""]);
    return labels;
  }

  function initializeMission() {
    const intake = mission.intake;
    const target = mission.target;
    const modeLabel = mission.execution_mode === "cached" ? "Cached replay" : "Live scorer record";
    const provenance = mission.execution_mode === "cached"
      ? `Bundled cached ${target.id} activity and ADMET model outputs`
      : `Recorded live ${target.id} activity and ADMET scorer outputs`;

    setText("top-mode", modeLabel);
    setText("top-digest", `Digest ${mission.digest.slice(0, 12)}`);
    setText("mission-target-code", target.id);
    setText("mission-target-title", target.id);
    setText("orbit-target-code", target.id);
    setText("target-rationale-code", target.id);
    setText("body-seed-target", `${target.id} · BBB`);
    setText("body-selected-target", `${target.id} · BBB`);
    setText("mission-intent", target.therapeutic_intent);
    setText("mission-candidate-count", `${mission.summary.candidate_occurrences} candidate states · ${mission.rounds.length} rounds`);
    setText("mission-final-id", `${mission.summary.final_candidate_id} selected`);
    setText("target-rationale-status", titleCase(target.rationale_status));
    setText("target-rationale", target.rationale);
    setText("target-capability", target.capability);
    setText("target-intent", target.therapeutic_intent);
    const targetSources = byId("target-sources");
    if (targetSources) {
      targetSources.replaceChildren();
      (target.source_ids || []).forEach((sourceId) => {
        const source = mission.evidence_library.find((item) => item.id === sourceId);
        if (!source) return;
        const link = node("a", "source-chip", `${source.title} ↗`);
        const url = safeEvidenceUrl(source.url);
        if (!url) return;
        link.href = url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        targetSources.append(link);
      });
    }
    setText("seed-name", intake.name);
    setText("seed-source", titleCase(intake.source));
    setText("seed-rationale", intake.seed_rationale);
    setText("seed-resolved-name", intake.name);
    setText("seed-formula", intake.formula);
    setText("seed-weight", Number.isFinite(Number(intake.molecular_weight)) ? `${Number(intake.molecular_weight).toFixed(2)} Da` : "Not supplied");
    setText("seed-molecule-id", mission.seed.molecule_id.slice(0, 23));
    setText("optimization-hypothesis", intake.optimization_hypothesis);
    setText("launch-state", mission.execution_mode === "cached" ? "Cached mission ready to replay" : "Live mission record ready");
    setText("provenance-mode", modeLabel);
    setText("provenance-detail", provenance);
    setText("generated-at", new Date(mission.generated_at).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }));
    setText("mission-digest", `SHA-256 ${mission.digest.slice(0, 12)}`);
    setText("lineage-story", mission.summary.story);
    setText("evidence-digest", mission.digest);

    const chips = byId("mission-axis-chips");
    if (chips) {
      chips.replaceChildren();
      mission.score_keys.forEach((key) => {
        chips.append(node("span", "", mission.score_labels[key] || scoreLabels[key] || key));
      });
    }
  }

  function buildRoundRail() {
    const rail = byId("round-rail");
    if (!rail) return;
    rail.replaceChildren();
    mission.rounds.forEach((roundValue) => {
      const button = node("button", "round-button");
      button.type = "button";
      button.dataset.testid = `round-${roundValue.round}`;
      button.dataset.round = clean(roundValue.round);
      button.setAttribute("role", "listitem");
      if (roundValue.round === state.round) button.setAttribute("aria-current", "step");
      const winner = candidateInRound(roundValue, roundValue.winner_id);
      button.append(
        node("span", "", `R${roundValue.round}`),
        node("span", "", `${roundValue.veto_count} vetoed · ${winner.short_id} won`),
      );
      button.addEventListener("click", () => {
        stopPlayback();
        setRound(roundValue.round, true);
      });
      rail.append(button);
    });
  }

  function buildAxisList(candidate) {
    const list = node("div", "axis-list");
    mission.score_keys.forEach((key) => {
      const row = node("div", "axis-row");
      const label = node("span", "", scoreLabels[key] || key);
      const track = node("span", "axis-track");
      const fill = node("i");
      fill.style.width = `${Math.max(0, Math.min(1, Number(candidate.normalized[key]))) * 100}%`;
      track.append(fill);
      row.append(label, track, node("span", "", number(candidate.normalized[key])));
      list.append(row);
    });
    return list;
  }

  function renderCandidates() {
    const grid = byId("candidate-grid");
    if (!grid) return;
    grid.replaceChildren();
    const roundValue = currentRound();

    roundValue.candidates.forEach((candidate) => {
      const button = node("button", "candidate-card");
      button.type = "button";
      button.dataset.candidateId = candidate.candidate_id;
      button.dataset.testid = `candidate-${candidate.candidate_id}`;
      button.setAttribute("aria-pressed", candidate.candidate_id === state.candidateId ? "true" : "false");
      if (candidate.vetoed) button.classList.add("is-vetoed");
      if (candidate.candidate_id === roundValue.winner_id) button.classList.add("is-official");

      const head = node("div", "candidate-head");
      const identity = node("span");
      identity.append(
        node("span", "candidate-id", candidate.candidate_id),
        node("span", "candidate-name", candidate.name),
      );
      const chips = node("span", "state-chips");
      stateLabels(candidate, roundValue).forEach(([label, className]) => {
        chips.append(node("span", `state-chip ${className}`.trim(), label));
      });
      head.append(identity, chips);

      const foot = node("span", "candidate-foot");
      foot.append(
        node("span", "", candidate.edit_label),
        node("span", "", `Utility ${number(candidate.weighted_utility)}`),
      );
      button.append(head, buildAxisList(candidate), foot);
      button.addEventListener("click", () => {
        state.candidateId = candidate.candidate_id;
        renderCandidates();
        renderSelection();
      });
      grid.append(button);
    });
  }

  function selectionExplanation(candidate, roundValue) {
    const supporters = roundValue.agents.filter((agent) => agent.preferred_candidate === candidate.candidate_id);
    if (candidate.candidate_id === roundValue.winner_id) {
      return roundValue.agents.find((agent) => agent.role === "orchestrator").reason;
    }
    if (candidate.vetoed) {
      return `Rejected by the fixed hERG gate: raw predicted blockade ${number(candidate.raw.herg)} is at or above ${number(mission.safety_policy.threshold)}.`;
    }
    if (supporters.length) {
      return supporters.map((agent) => agent.reason).join(" ");
    }
    return `Eligible proposal with weighted utility ${number(candidate.weighted_utility)}; it did not lead the official eligible ranking.`;
  }

  function updateDepiction(candidate) {
    const image = byId("candidate-depiction");
    if (!image) return;
    if (state.depictionUrl) URL.revokeObjectURL(state.depictionUrl);
    const molecule = moleculeFor(candidate);
    const blob = new Blob([molecule.depiction_svg], { type: "image/svg+xml;charset=utf-8" });
    state.depictionUrl = URL.createObjectURL(blob);
    image.src = state.depictionUrl;
    image.alt = `RDKit 2D depiction for ${candidate.candidate_id}`;
  }

  function renderSelection() {
    const roundValue = currentRound();
    const candidate = selectedCandidate();
    setText("selection-title", `${candidate.candidate_id} · ${candidate.name}`);
    const labels = stateLabels(candidate, roundValue).map(([label]) => label);
    setText("selection-state", labels.join(" · ") || "Eligible");
    setText("selection-edit", `${candidate.edit_label} · ${titleCase(candidate.source_kind)}`);
    setText("selection-hypothesis", candidate.edit_hypothesis);
    setText("selection-reason", selectionExplanation(candidate, roundValue));
    setText("selection-smiles", candidate.smiles);

    const details = byId("selection-details");
    if (details) {
      details.replaceChildren();
      appendTextRow(details, propertyDisplay("affinity"), number(candidate.raw.affinity));
      appendTextRow(details, "Predicted LogS", number(candidate.raw.solubility));
      appendTextRow(details, "Predicted BBB", number(candidate.raw.bbb));
      appendTextRow(details, "Predicted hERG", number(candidate.raw.herg));
      appendTextRow(details, "SA score", number(candidate.raw.sa));
      appendTextRow(details, "Seed locality", number(candidate.similarity_seed));
    }
    updateDepiction(candidate);
  }

  function renderOptimize() {
    const roundValue = currentRound();
    setText("round-context", `Round ${roundValue.round} slate · parent ${roundValue.parent.molecule_id.slice(7, 15)}`);
    buildRoundRail();
    renderCandidates();
    renderSelection();
    updatePlaybackControls();
  }

  function evidenceValue(item) {
    if (typeof item.value === "number") return number(item.value);
    return clean(item.value);
  }

  function buildAgentCard(agent) {
    const profile = agentProfiles[agent.role] || {
      name: titleCase(agent.role),
      initials: agent.role.slice(0, 2).toUpperCase(),
      role: `${titleCase(agent.role)} specialist`,
    };
    const article = node("article", `agent-card ${agent.role === "orchestrator" ? "orchestrator" : ""}`.trim());
    article.dataset.testid = `agent-${agent.role}`;
    const top = node("div", "agent-top");
    top.append(
      node("span", "agent-avatar", profile.initials),
      node("span", `verdict ${agent.verdict}`, agent.verdict),
    );
    const preferred = node("p", "agent-preference");
    preferred.append(node("span", "", `${profile.name} · ${profile.role} · prefers `), node("strong", "", agent.preferred_candidate));
    const reason = node("p", "agent-reason", agent.reason);

    const confidence = node("div", "confidence");
    const confidenceLine = node("div", "confidence-line");
    confidenceLine.append(node("span", "", "Heuristic confidence"), node("strong", "", percent(agent.confidence_score)));
    const track = node("div", "confidence-track");
    const fill = node("span");
    fill.style.width = `${Math.max(0, Math.min(1, Number(agent.confidence_score) || 0)) * 100}%`;
    track.append(fill);
    confidence.append(confidenceLine, track, node("small", "", agent.confidence_basis));

    const details = node("details", "agent-details");
    details.append(node("summary", "", "Exact evidence, ranking & rejections"));

    const evidenceTitle = node("h3", "", "Exact evidence");
    const evidenceList = node("div", "evidence-values");
    agent.evidence.forEach((item) => {
      const row = node("div", "evidence-row");
      row.append(node("span", "", item.label), node("span", "", evidenceValue(item)), node("span", "", item.threshold !== undefined ? `gate ${number(item.threshold)}` : ""));
      evidenceList.append(row);
    });

    const rankingTitle = node("h3", "", "Recorded ranking");
    const rankingList = node("div", "ranking-list");
    agent.ranking.forEach((item) => {
      const row = node("div", "rank-row");
      row.append(node("span", "", `#${item.rank}`), node("span", "", item.candidate_id), node("span", "", number(item.score)));
      rankingList.append(row);
    });

    const rejectionTitle = node("h3", "", "Why others ranked lower");
    const rejectionList = node("div", "rejection-list");
    agent.rejections.forEach((item) => {
      const row = node("div", "rejection-row");
      row.append(node("span", "", item.candidate_id), node("span", "", `${item.reason_code}: ${item.reason}`));
      rejectionList.append(row);
    });

    const sourceLine = node("p", "microcopy", `Context sources: ${agent.context_sources.join(", ")}`);
    details.append(evidenceTitle, evidenceList, rankingTitle, rankingList, rejectionTitle, rejectionList, sourceLine);
    article.append(top, preferred, reason, confidence, details);
    return article;
  }

  function renderSafetyGate(roundValue, ledger) {
    setText("gate-threshold", `${ledger.safety_gate.operator} ${number(ledger.safety_gate.threshold)}`);
    const outcome = byId("gate-outcome");
    if (!outcome) return;
    outcome.replaceChildren();
    const veto = node("div");
    veto.append(node("span", "", "Vetoed"), node("strong", "", ledger.safety_gate.vetoed_ids.join(", ") || "None"));
    const survivors = node("div");
    survivors.append(node("span", "", "Survived"), node("strong", "", ledger.safety_gate.surviving_ids.join(", ") || "None"));
    const official = node("div");
    official.append(node("span", "", "Official"), node("strong", "", roundValue.winner_id));
    outcome.append(veto, survivors, official);
  }

  function renderPolicyComparison() {
    const container = byId("policy-comparison");
    if (!container) return;
    container.replaceChildren();
    const comparison = mission.policy_comparison.find((item) => item.round === state.round);
    ["affinity_only", "adme_only", "safety_only", "weighted", "official"].forEach((policy) => {
      const result = comparison[policy];
      const row = node("div", "policy-row");
      row.append(
        node("span", "", titleCase(policy)),
        node("strong", "", result.candidate_id),
        node(
          "span",
          result.vetoed ? "verdict veto" : "policy-score",
          `${result.vetoed ? "Vetoed · " : ""}${result.score_label} ${number(result.decision_score)}`,
        ),
      );
      container.append(row);
    });
    if (comparison.official_differs_from_weighted) {
      container.append(node("p", "microcopy", "Official differs from unconstrained weighted choice because the exploration policy governs eligibility."));
    }
  }

  function renderLineage() {
    const container = byId("lineage-path");
    if (!container) return;
    container.replaceChildren();
    mission.rounds.forEach((roundValue, index) => {
      const winner = candidateInRound(roundValue, roundValue.winner_id);
      const restored = roundValue.events.some((event) => event.type === "rollback");
      const label = `${roundValue.winner_id} · ${winner.name}${restored ? " · restored" : ""}`;
      container.append(node("span", "lineage-node", label));
      if (index < mission.rounds.length - 1) container.append(node("span", "lineage-arrow", "→"));
    });
  }

  function renderCompleteLedger() {
    const container = byId("complete-ledger");
    if (!container) return;
    container.replaceChildren();
    mission.decision_ledger.forEach((ledger) => {
      const roundValue = mission.rounds.find((item) => item.round === ledger.round);
      const details = node("details", "ledger-round");
      if (ledger.round === state.round) details.open = true;
      const summary = node("summary");
      summary.append(
        node("strong", "", `Round ${ledger.round} · ${ledger.choice.winner}`),
        node("span", "", `${ledger.safety_gate.vetoed_ids.length} vetoed · margin ${number(ledger.choice.margin)}`),
      );
      const provenance = node("p", "", ledger.inputs.score_provenance);
      const pool = node("p", "", `Eligible ranking: ${ledger.orchestrator_ranking.map((item) => `${item.rank}. ${item.candidate_id} (${number(item.score)})`).join(" · ")}`);
      const chosen = node("p", "", `Why chosen: ${ledger.why_chosen.join(" ")}`);
      const rejected = node("p", "", `Rejections: ${ledger.why_rejected.map((item) => `${item.candidate_id} ${item.reason_code}`).join(" · ")}`);
      const validation = node("p", "", `Validation: ${Object.entries(ledger.validation).map(([key, value]) => `${titleCase(key)} ${value ? "✓" : "✕"}`).join(" · ")}`);
      const policies = node("p", "", `Policies: ${Object.entries(roundValue.policy_choices).map(([key, value]) => `${titleCase(key)} ${value}`).join(" · ")}`);
      const rollback = roundValue.events.find((event) => event.type === "rollback");
      const rollbackLine = rollback
        ? node("p", "rollback-line", `Rollback: ${rollback.detail}`)
        : null;
      details.append(summary, provenance, pool, chosen, rejected, validation, policies);
      if (rollbackLine) details.append(rollbackLine);
      container.append(details);
    });
  }

  function renderDecisions() {
    const select = byId("decision-round-select");
    if (select && select.options.length !== mission.rounds.length) {
      select.replaceChildren();
      mission.rounds.forEach((roundValue) => {
        const option = node("option", "", `Round ${roundValue.round}`);
        option.value = clean(roundValue.round);
        select.append(option);
      });
    }
    if (select) select.value = clean(state.round);

    const roundValue = currentRound();
    const agents = byId("agent-grid");
    if (agents) {
      agents.replaceChildren();
      roundValue.agents.forEach((agent) => agents.append(buildAgentCard(agent)));
    }
    const ledger = mission.decision_ledger.find((item) => item.round === state.round);
    renderSafetyGate(roundValue, ledger);
    renderPolicyComparison();
    renderLineage();
    renderCompleteLedger();
  }

  function wetLabStep(stepId) {
    return mission.wet_lab_plan.find((item) => item.id === stepId);
  }

  function propertyDisplay(key) {
    return mission.score_labels[key] || scoreLabels[key] || titleCase(key);
  }

  function bodyInterpretation(system, candidate) {
    const finalSelected = candidate.molecule_id === mission.summary.final_molecule_id;
    if (finalSelected) return system.interpretation;
    const properties = system.source_properties.map((key) => propertyDisplay(key)).join(" and ");
    return `This comparison shows the selected proposal's ${properties} objectives against the seed. It remains a modeled hypothesis requiring the linked assays.`;
  }

  function buildSystemCard(system, candidate) {
    const article = node("article", "card system-card");
    const head = node("div", "system-card-head");
    const labelWrap = node("div");
    labelWrap.append(node("h2", "", system.label), node("p", "", system.source_properties.map(propertyDisplay).join(" · ")));
    head.append(
      node("span", "system-icon", system.system === "absorption" ? "Gut" : system.system),
      labelWrap,
      node("span", "pill pill-blue", system.confidence_label),
    );

    const deltaGrid = node("div", "delta-grid");
    system.source_properties.forEach((key) => {
      const seedValue = Number(mission.seed.normalized[key]);
      const candidateValue = Number(candidate.normalized[key]);
      const block = node("div");
      block.append(
        node("span", "", propertyDisplay(key)),
        node("strong", "", `${number(seedValue)} → ${number(candidateValue)} · Δ ${signed(candidateValue - seedValue)}`),
      );
      deltaGrid.append(block);
    });

    const links = node("div", "assay-links");
    system.wet_lab_step_ids.forEach((stepId) => {
      const step = wetLabStep(stepId);
      if (!step) return;
      const button = node("button", "text-button", `${step.id} ${step.property} →`);
      button.type = "button";
      button.dataset.wetLabLink = step.id;
      links.append(button);
    });
    article.append(head, deltaGrid, node("p", "", bodyInterpretation(system, candidate)), links);
    return article;
  }

  function renderBodyMap() {
    const candidate = selectedCandidate();
    setText("body-seed-name", mission.seed.name);
    setText("body-selected-name", `${candidate.candidate_id} · ${candidate.name}`);
    const cards = byId("body-system-cards");
    if (cards) {
      cards.replaceChildren();
      mission.body_systems_map.forEach((system) => cards.append(buildSystemCard(system, candidate)));
    }
    const seedSa = Number(mission.seed.normalized.sa);
    const candidateSa = Number(candidate.normalized.sa);
    setText("sa-comparison", `SA objective ${number(seedSa)} → ${number(candidateSa)} · exact Δ ${signed(candidateSa - seedSa)}. This heuristic belongs to route review, not a human organ system.`);

    const conformer = moleculeFor(candidate).conformer;
    setText("conformer-method", conformer.method);
    setText("conformer-status", `${titleCase(conformer.status)} · ${titleCase(conformer.optimization_status)}`);
    conformerState.geometry = conformer;
    resetConformer(false);
    bindWetLabLinks();
  }

  function renderWetLab() {
    const grid = byId("wet-lab-grid");
    if (!grid) return;
    const candidate = selectedCandidate();
    setText(
      "validation-candidate",
      `assays performed · plan for ${candidate.candidate_id}`,
    );
    grid.replaceChildren();
    mission.wet_lab_plan.forEach((step) => {
      let decisionQuestion = step.decision_question;
      let predictedContext = step.predicted_context;
      if (step.id === "WL-2") {
        decisionQuestion = `Does ${candidate.name} retain meaningful ${mission.target.id} activity?`;
        predictedContext = `Modeled activity score ${number(candidate.raw.affinity)}.`;
      } else if (step.id === "WL-4") {
        predictedContext = `Predicted blockade score ${number(candidate.raw.herg)}; policy threshold ${number(mission.safety_policy.threshold)}.`;
      } else if (step.id === "WL-5") {
        predictedContext = `Predicted LogS ${number(candidate.raw.solubility)}.`;
      } else if (step.id === "WL-6") {
        predictedContext = `BBB classifier score ${number(candidate.raw.bbb)}.`;
      } else if (step.id === "WL-8") {
        predictedContext = `SA score ${number(candidate.raw.sa)}.`;
      }
      const article = node("article", "assay-card");
      article.id = `assay-${step.id}`;
      article.dataset.testid = `wet-lab-${step.id}`;
      if (step.id === state.highlightedAssay) article.classList.add("is-highlighted");
      const top = node("div", "assay-top");
      top.append(node("span", "assay-id", step.id), node("span", "status-unperformed", "Unperformed"));
      article.append(
        top,
        node("span", "priority-label", `Priority ${step.priority}`),
        node("h2", "", step.assay),
        node("span", "assay-property", step.property),
        node("p", "assay-question", decisionQuestion),
      );
      const readout = node("div", "readout");
      readout.append(
        node("span", "", "Expected readout"),
        node("p", "", step.readout),
        node("span", "", "Decision link"),
        node("p", "", step.confirmation),
        node("span", "", "Modeled context"),
        node("p", "", predictedContext),
      );
      article.append(readout);
      grid.append(article);
    });
    if (state.highlightedAssay) {
      window.requestAnimationFrame(() => {
        const target = byId(`assay-${state.highlightedAssay}`);
        if (target) target.scrollIntoView({ block: "center", behavior: "smooth" });
      });
    }
  }

  function safeEvidenceUrl(value) {
    try {
      const parsed = new URL(value);
      return parsed.protocol === "https:" || parsed.protocol === "http:" ? parsed.href : "";
    } catch (error) {
      return "";
    }
  }

  function renderEvidence() {
    const list = byId("evidence-list");
    if (list) {
      list.replaceChildren();
      mission.evidence_library.forEach((source) => {
        const article = node("article", "evidence-card");
        article.dataset.testid = `evidence-${source.id}`;
        const effect = node("div", "product-effect");
        effect.append(node("span", "", "Product effect"), node("p", "", source.product_effect));
        article.append(
          node("span", "evidence-topic", source.topic),
          node("h2", "", source.title),
          node("p", "", source.summary),
          effect,
        );
        const url = safeEvidenceUrl(source.url);
        if (url) {
          const link = node("a", "evidence-link", "Open source ↗");
          link.href = url;
          link.target = "_blank";
          link.rel = "noopener noreferrer";
          article.append(link);
        }
        list.append(article);
      });
    }
    const limits = byId("knowledge-limits");
    if (limits) {
      limits.replaceChildren();
      mission.disclaimers.forEach((item) => limits.append(node("li", "", item)));
    }
  }

  function downloadExport(kind) {
    const exports = {
      json: ["export-json-payload", "application/json", `admet-mission-${mission.digest.slice(0, 12)}.json`],
      csv: ["export-csv-payload", "text/csv", `admet-candidates-${mission.digest.slice(0, 12)}.csv`],
      receipt: ["export-receipt-payload", "text/html", `admet-decision-receipt-${mission.digest.slice(0, 12)}.html`],
    };
    const definition = exports[kind];
    if (!definition) return;
    try {
      const bytes = payloadBytes(definition[0]);
      const url = URL.createObjectURL(new Blob([bytes], { type: `${definition[1]};charset=utf-8` }));
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = definition[2];
      anchor.hidden = true;
      document.body.append(anchor);
      anchor.click();
      anchor.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 0);
      announce(`${titleCase(kind)} export prepared from the embedded evidence package.`);
    } catch (error) {
      announce("Export could not be prepared.");
    }
  }

  function bindWetLabLinks() {
    all("[data-wet-lab-link]").forEach((button) => {
      if (button.dataset.bound === "true") return;
      button.dataset.bound = "true";
      button.addEventListener("click", () => {
        state.highlightedAssay = button.dataset.wetLabLink;
        showView("validate", true);
      });
    });
  }

  function updatePlaybackControls() {
    const progress = byId("playback-progress");
    const clamped = Math.max(0, Math.min(25000, state.playbackElapsed));
    const canResume = !state.playing && state.playbackElapsed > 0 && state.playbackElapsed < 25000;
    const actionLabel = state.playing
      ? "Pause narrative replay"
      : (canResume ? "Resume narrative replay" : "Play narrative replay");
    if (progress) progress.style.width = `${(clamped / 25000) * 100}%`;
    setText("playback-status", `${state.playing ? "Playing" : "Paused"} at round ${state.round}`);
    all("[data-replay-action='play']").forEach((button) => {
      const isIcon = button.classList.contains("icon-button");
      button.textContent = isIcon
        ? (state.playing ? "❚❚ Pause" : (canResume ? "▶ Resume" : "▶ Play"))
        : actionLabel;
      button.setAttribute("aria-label", actionLabel);
      button.setAttribute("aria-pressed", state.playing ? "true" : "false");
    });
  }

  function playbackFrame(timestamp) {
    if (!state.playing) return;
    state.playbackElapsed = Math.min(25000, timestamp - state.playbackStartedAt);
    const roundIndex = Math.min(4, Math.floor(state.playbackElapsed / 5000));
    const nextRound = mission.rounds[roundIndex].round;
    if (nextRound !== state.round) setRound(nextRound, true);
    updatePlaybackControls();
    if (state.playbackElapsed >= 25000) {
      state.playing = false;
      state.round = finalRound.round;
      state.candidateId = finalRound.winner_id;
      renderOptimize();
      renderDecisions();
      announce("Narrative replay complete. No scores were recomputed.");
      return;
    }
    state.animationFrame = window.requestAnimationFrame(playbackFrame);
  }

  function startPlayback() {
    if (state.playing) return;
    if (state.playbackElapsed >= 25000 || state.round === finalRound.round) {
      state.playbackElapsed = 0;
      setRound(mission.rounds[0].round, true);
    }
    state.playing = true;
    state.playbackStartedAt = performance.now() - state.playbackElapsed;
    updatePlaybackControls();
    state.animationFrame = window.requestAnimationFrame(playbackFrame);
  }

  function stopPlayback() {
    if (!state.playing) return;
    state.playing = false;
    window.cancelAnimationFrame(state.animationFrame);
    updatePlaybackControls();
  }

  function togglePlayback() {
    if (state.playing) stopPlayback();
    else startPlayback();
  }

  function skipPlayback() {
    const nextRound = Math.min(finalRound.round, state.round + 1);
    state.playbackElapsed = Math.min(25000, nextRound * 5000 - 5000);
    if (state.playing) state.playbackStartedAt = performance.now() - state.playbackElapsed;
    setRound(nextRound, true);
  }

  const conformerState = {
    geometry: null,
    yaw: -0.55,
    pitch: 0.34,
    zoom: 1,
    pointers: new Map(),
    previousPinch: 0,
  };

  const atomColors = {
    C: "#263b53",
    N: "#315ed9",
    O: "#c4584e",
    S: "#b78a20",
    F: "#6f9b52",
    Cl: "#5c8d50",
    Br: "#8b4a37",
    P: "#b36f29",
  };

  function rotatedAtoms() {
    if (!conformerState.geometry) return [];
    const atoms = conformerState.geometry.atoms;
    const center = atoms.reduce((accumulator, atom) => ({
      x: accumulator.x + Number(atom.x) / atoms.length,
      y: accumulator.y + Number(atom.y) / atoms.length,
      z: accumulator.z + Number(atom.z) / atoms.length,
    }), { x: 0, y: 0, z: 0 });
    const cy = Math.cos(conformerState.yaw);
    const sy = Math.sin(conformerState.yaw);
    const cp = Math.cos(conformerState.pitch);
    const sp = Math.sin(conformerState.pitch);
    return atoms.map((atom) => {
      const x = Number(atom.x) - center.x;
      const y = Number(atom.y) - center.y;
      const z = Number(atom.z) - center.z;
      const yawX = x * cy + z * sy;
      const yawZ = -x * sy + z * cy;
      return {
        ...atom,
        rx: yawX,
        ry: y * cp - yawZ * sp,
        rz: y * sp + yawZ * cp,
      };
    });
  }

  function drawConformer() {
    const canvas = byId("conformer-canvas");
    if (!canvas || !conformerState.geometry) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const width = canvas.width / dpr;
    const height = canvas.height / dpr;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, width, height);

    const atoms = rotatedAtoms();
    if (!atoms.length) return;
    const rangeX = Math.max(...atoms.map((atom) => atom.rx)) - Math.min(...atoms.map((atom) => atom.rx)) || 1;
    const rangeY = Math.max(...atoms.map((atom) => atom.ry)) - Math.min(...atoms.map((atom) => atom.ry)) || 1;
    const scale = Math.min(width / (rangeX + 3), height / (rangeY + 3)) * conformerState.zoom;
    const projected = atoms.map((atom) => ({
      ...atom,
      px: width / 2 + atom.rx * scale,
      py: height / 2 - atom.ry * scale,
    }));
    const byIndex = new Map(projected.map((atom) => [atom.index, atom]));
    const bonds = conformerState.geometry.bonds
      .map((bond) => ({
        ...bond,
        start: byIndex.get(bond.begin_atom_index),
        end: byIndex.get(bond.end_atom_index),
      }))
      .filter((bond) => bond.start && bond.end)
      .sort((first, second) => ((first.start.rz + first.end.rz) - (second.start.rz + second.end.rz)));

    bonds.forEach((bond) => {
      const depth = (bond.start.rz + bond.end.rz) / 2;
      context.lineCap = "round";
      context.strokeStyle = depth > 0 ? "rgba(35, 56, 80, 0.82)" : "rgba(75, 94, 115, 0.44)";
      context.lineWidth = Math.max(1.5, scale * 0.045);
      const dx = bond.end.px - bond.start.px;
      const dy = bond.end.py - bond.start.py;
      const length = Math.hypot(dx, dy) || 1;
      const offsetX = (-dy / length) * Math.min(4, scale * 0.06);
      const offsetY = (dx / length) * Math.min(4, scale * 0.06);
      const lineCount = Number(bond.order) >= 1.9 ? 2 : 1;
      for (let line = 0; line < lineCount; line += 1) {
        const direction = lineCount === 1 ? 0 : (line === 0 ? -0.5 : 0.5);
        context.beginPath();
        context.moveTo(bond.start.px + offsetX * direction, bond.start.py + offsetY * direction);
        context.lineTo(bond.end.px + offsetX * direction, bond.end.py + offsetY * direction);
        context.stroke();
      }
    });

    projected.sort((first, second) => first.rz - second.rz).forEach((atom) => {
      const radius = Math.max(4.2, Math.min(10, scale * (atom.element === "C" ? 0.095 : 0.12)));
      context.beginPath();
      context.arc(atom.px, atom.py, radius, 0, Math.PI * 2);
      context.fillStyle = atomColors[atom.element] || "#805a87";
      context.shadowColor = "rgba(16, 37, 63, 0.2)";
      context.shadowBlur = atom.rz > 0 ? 7 : 2;
      context.fill();
      context.shadowBlur = 0;
      if (atom.element !== "C") {
        context.fillStyle = "#ffffff";
        context.font = `700 ${Math.max(7, radius * 1.15)}px ui-sans-serif, system-ui`;
        context.textAlign = "center";
        context.textBaseline = "middle";
        context.fillText(atom.element, atom.px, atom.py + 0.5);
      }
    });
  }

  function resizeConformer() {
    const canvas = byId("conformer-canvas");
    if (!canvas) return;
    const rectangle = canvas.getBoundingClientRect();
    if (rectangle.width < 2 || rectangle.height < 2) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const nextWidth = Math.round(rectangle.width * dpr);
    const nextHeight = Math.round(rectangle.height * dpr);
    if (canvas.width !== nextWidth || canvas.height !== nextHeight) {
      canvas.width = nextWidth;
      canvas.height = nextHeight;
    }
    drawConformer();
  }

  function resetConformer(redraw = true) {
    conformerState.yaw = -0.55;
    conformerState.pitch = 0.34;
    conformerState.zoom = 1;
    if (redraw) drawConformer();
  }

  function bindConformer() {
    const canvas = byId("conformer-canvas");
    if (!canvas) return;
    canvas.tabIndex = 0;
    canvas.addEventListener("pointerdown", (event) => {
      canvas.setPointerCapture(event.pointerId);
      conformerState.pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    });
    canvas.addEventListener("pointermove", (event) => {
      const previous = conformerState.pointers.get(event.pointerId);
      if (!previous) return;
      conformerState.pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
      const points = Array.from(conformerState.pointers.values());
      if (points.length === 1) {
        conformerState.yaw += (event.clientX - previous.x) * 0.012;
        conformerState.pitch = Math.max(-1.45, Math.min(1.45, conformerState.pitch + (event.clientY - previous.y) * 0.012));
      } else if (points.length === 2) {
        const distance = Math.hypot(points[0].x - points[1].x, points[0].y - points[1].y);
        if (conformerState.previousPinch) {
          conformerState.zoom = Math.max(0.45, Math.min(3.6, conformerState.zoom * (distance / conformerState.previousPinch)));
        }
        conformerState.previousPinch = distance;
      }
      drawConformer();
    });
    const release = (event) => {
      conformerState.pointers.delete(event.pointerId);
      if (conformerState.pointers.size < 2) conformerState.previousPinch = 0;
    };
    canvas.addEventListener("pointerup", release);
    canvas.addEventListener("pointercancel", release);
    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      conformerState.zoom = Math.max(0.45, Math.min(3.6, conformerState.zoom * Math.exp(-event.deltaY * 0.0012)));
      drawConformer();
    }, { passive: false });
    canvas.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "+", "-", "="].includes(event.key)) return;
      event.preventDefault();
      if (event.key === "ArrowLeft") conformerState.yaw -= 0.12;
      if (event.key === "ArrowRight") conformerState.yaw += 0.12;
      if (event.key === "ArrowUp") conformerState.pitch -= 0.12;
      if (event.key === "ArrowDown") conformerState.pitch += 0.12;
      if (event.key === "+" || event.key === "=") conformerState.zoom = Math.min(3.6, conformerState.zoom * 1.1);
      if (event.key === "-") conformerState.zoom = Math.max(0.45, conformerState.zoom / 1.1);
      drawConformer();
    });
    const frame = canvas.parentElement;
    if (window.ResizeObserver && frame) {
      const observer = new ResizeObserver(() => resizeConformer());
      observer.observe(frame);
    }
    window.addEventListener("resize", resizeConformer, { passive: true });
  }

  function bindInteractions() {
    all("[data-view]").forEach((button) => {
      button.addEventListener("click", () => showView(button.dataset.view, true));
    });
    all("[data-nav-target]").forEach((control) => {
      control.addEventListener("click", (event) => {
        event.preventDefault();
        showView(control.dataset.navTarget, true);
      });
    });
    const viewSelect = byId("view-select");
    if (viewSelect) viewSelect.addEventListener("change", () => showView(viewSelect.value, true));
    const roundSelect = byId("decision-round-select");
    if (roundSelect) roundSelect.addEventListener("change", () => {
      stopPlayback();
      setRound(roundSelect.value, true);
    });
    all("[data-replay-action='play']").forEach((button) => button.addEventListener("click", togglePlayback));
    const skip = byId("replay-skip");
    if (skip) skip.addEventListener("click", skipPlayback);
    const reset = byId("conformer-reset");
    if (reset) reset.addEventListener("click", () => resetConformer(true));
    all("[data-export]").forEach((button) => button.addEventListener("click", () => downloadExport(button.dataset.export)));
    bindWetLabLinks();
  }

  initializeMission();
  renderOptimize();
  renderDecisions();
  renderBodyMap();
  renderWetLab();
  renderEvidence();
  bindInteractions();
  bindConformer();
  showView("mission");
})();
