# Case Study Content Master Template and Authoring Instructions

**Version:** 1.0 | **Prepared:** 2026-11-06 | **Purpose:** Write detailed, defensible case-study content for industry practice, production engineering, and demanding interviews.

This is an authoring standard and reusable template. It is not a completed case, evidence that a company used a particular design, or a requirement to put every technology into every story. Use it to write one coherent case around one consequential decision. Preserve the 15-section case outline and the seven-section collection INDEX described below.

**Companion:** [Cheat Sheet Content Master Template](CHEAT_SHEET_CONTENT_MASTER_TEMPLATE.md). A case explains a decision in context; a cheat sheet provides the reusable theory, calculations, selection rules, and diagnostics supporting that decision. Link them rather than duplicating large explanations.

**Navigation:** [Authoring contract](#part-a--authoring-contract) | [15-section content template](#part-b--individual-case-content-preserve-sections-115) | [Collection INDEX](#part-c--collection-index-preserve-all-seven-sections) | [Technical and business profiles](#part-d--relevance-profiles-use-only-those-that-change-the-case) | [Workflow and publication checks](#part-e--workflow-source-crosswalk-and-publication-gates).

## How to use this document

1. Read Part A and register the proposed case in the collection INDEX before drafting.
2. Copy the metadata and the 15 headings from Part B into the new case file. Write the actual content under those headings; do not publish these author instructions as case content.
3. Select the relevant technical and business-domain profiles in Part D. Put their requirements into the existing sections, without inventing extra top-level sections.
4. Draft the lesson as a hypothesis and the evaluation plan before writing an outcome narrative. Change the lesson if the evidence disagrees.
5. Complete the interview pack inside Section 14, then finish Section 15 as the final section of the individual case.
6. Apply the checks in Part E and reconcile the collection INDEX with the finished case.

**Meaning of requirement labels:** `MUST` is required for this template; `CONDITIONAL` becomes required when the stated situation applies; `RECOMMENDED` adds useful depth. These are editorial requirements for this library, not external certification criteria. Keep every numbered heading. A conditional subsection may say `N/A — [specific reason]`; uncertainty must say `unknown`, `not tested`, or `not measured`, rather than `N/A`.

## Part A — Authoring contract

### A1. What a comprehensive case must accomplish

A reader should be able to reconstruct the chain:

```text
Business decision -> constraints -> data available at the decision time
  -> baseline and alternatives -> method and design -> validation
  -> action actually taken -> outcome actually supported
  -> failure boundaries -> transferable lesson -> interview defence
```

Every arrow needs evidence or an explicit statement that the connection has not been established. Depth means explaining the mechanism, choices, limitations, and consequences. A long technology list does not make a case comprehensive.

| Reader need | Required content | Insufficient substitute |
|---|---|---|
| Understand the problem | Stakeholder, input, output, decision timing, baseline, business consequence | A title such as “AI in retail” |
| Judge the method | Candidate alternatives, assumptions, selection evidence, invalidating conditions | Naming a fashionable model |
| Reproduce the reasoning | Data grain, cut-off, transformations, parameters, intermediate outputs, artifacts | A polished final diagram |
| Assess production relevance | Interfaces, state, failure handling, ownership, capacity, cost, deployment boundaries | “Deploy with Kubernetes” |
| Trust the outcome | Comparator, metric definition, window, sample, uncertainty, evidence label | An unattributed improvement percentage |
| Transfer the lesson | Mechanism, prerequisites, counterexample, changes needed in another domain | “Better data improves models” |
| Defend it in an interview | Checked answers, numerical work, follow-up constraints, failure diagnosis | Questions without answers |

### A2. Choose a bounded case and an honest source basis

Choose one main decision, not an entire company or industry. Split a large anthology into independent cases when it contains different decision owners, data, evaluation protocols, or outcomes. Supporting decisions can remain inside Section 8 if they serve the same main question.

Select and disclose the source basis:

- **First-hand case:** Use accessible project artifacts and anonymized observations. Describe access limits and redactions without exposing private data or credentials.
- **Published industry case:** Use attributable primary publications, engineering posts, papers, filings, or incident reports. Separate the organization's reported claims from your interpretation.
- **Reproduced case:** Link the public inputs, environment, execution instructions, and results you actually reproduced. Reproduction on a different workload is a separate result.
- **Illustrative design exercise:** Label the setting, workload, figures, and architecture as hypothetical. Present design recommendations and validation plans without inventing deployments or results.
- **Composite teaching case:** Identify it as a composite; explain which elements are sourced and which are constructed. Do not attribute the combined narrative to a named organization.

Existing files in `Train/INTERVIEW_CASES` may inspire a question, mechanism, or interview exercise. They are reference examples only. Their presence does not establish a real incident, measured performance, original source availability, or permission to describe an example as your own project.

### A3. Separate evidence, implementation stage, and publication state

| Evidence label | Meaning | Permitted wording |
|---|---|---|
| `observed` | Direct measurement from the described system or intervention | “In the recorded evaluation window…” with artifact and scope |
| `published` | A named source reports a result | “The organization reports…” with direct primary citation |
| `estimated` | A model or calculation projects a quantity from stated inputs | “Estimated under these assumptions…” |
| `simulated` | A documented simulation or test harness generated the result | “In this simulation…” with configuration and limitations |
| `illustrative` | Constructed teaching values or a hypothetical scenario | “Illustrative; not a company measurement” |
| `not-measured` | No usable outcome measurement exists | “Outcome not yet measured; proposed test is…” |

Keep the archived evidence vocabulary. If a case mixes evidence types, label each claim/table row and explain the mix in the source declaration; one frontmatter label cannot certify every number. Use the headline outcome's evidence label in the INDEX and make any mixed scope visible in its limitation field.

**Implementation stage:** `proposed / pilot / production / scaled / paused / retired / superseded`. This describes the intervention, not document completeness. An illustrative production architecture remains a proposed design, unless the case separately documents a real deployment.

**Document state:** `planned / draft / reviewed / published / needs-review`. This is an editorial field. A published document can describe a proposed intervention with an unmeasured outcome.

**Claim ledger:** Assign stable local identifiers to major claims, decisions, runs, and sources. At minimum, make the primary result traceable from Section 7 to Sections 6, 9, and 11.

| Claim ID | Exact claim and scope | Evidence type | Source/artifact and location | Supports what? | Does not establish | Checked on / trigger |
|---|---|---|---|---|---|---|
| `CLM-01` | `[Population, metric, window]` | `[Label]` | `[Source ID, page/section/run]` | `[Bounded conclusion]` | `[Causality, scale, or portability limit]` | `[Date / event]` |

### A4. Source and currency instructions

- Search for the original evidence before relying on summaries. Prefer company engineering publications for company systems, original papers for research claims, official manuals for APIs, and authoritative publications for applicable rules.
- Record publication/event date, version, access/review date, and the precise claim supported. A publication date is not necessarily the deployment date.
- Verify unstable facts when authoring: model availability, APIs, framework semantics, hardware capabilities, pricing, organizational structure, regulations, and company operating context. Preserve historical facts as historical rather than silently modernizing the case.
- Use a dated price source and currency for cost estimates. Include requests, tokens, storage, compute, egress, retries, review labor, and failed work as applicable. Mark unpriced items explicitly.
- Cite claims where used. Store full source details in Section 5 and cross-reference them elsewhere. A link alone does not prove that the linked text supports the claim.
- If a source is unavailable, record the gap and narrow the claim. Do not reconstruct missing logs, citations, screenshots, or outcomes from confidence or familiarity.
- Company names are research contexts, not evidence. Do not infer proprietary features, model choices, traffic volumes, or results from the company's line of business.

### A5. Scope, depth, and writing quality

Use enough detail to explain and challenge the decision. There is no word-count target that replaces coverage. Split the case when a new topic needs a separate decision contract or makes the main reasoning hard to follow.

- Define abbreviations and domain terms at first use. Use consistent notation, units, time zones, and identifiers.
- Use tables for comparisons and registers; use connected prose to explain why the evidence changes a decision.
- Show relevant formulas and short pseudocode with input/output contracts, rather than dumping implementation code. Link longer artifacts if they exist.
- Label diagram boundaries, state stores, ownership, control paths, and failure paths. Avoid product logos as architectural explanation.
- Write the strongest reasonable alternative, including the current manual/rule-based process. Explain when it wins.
- State unknowns next to the affected conclusion. Avoid a generic limitations paragraph that leaves strong claims elsewhere unqualified.
- For a design exercise, distinguish required validation from completed validation throughout the file. A planned test is not a passed test.

## Part B — Individual case content: preserve Sections 1–15

**Content outline:** [1. Framing](#1-the-one-line-framing), [2. Context](#2-context--background), [3. Business question](#3-the-business-question), [4. Assumptions](#4-assumptions--constraints), [5. Data and method](#5-data-tools--method), [6. Walkthrough](#6-analysis-walkthrough), [7. Finding](#7-the-finding), [8. Design](#8-systemsolution-design), [9. Validation](#9-evaluation--validation), [10. Action](#10-action-taken), [11. Outcomes](#11-outcome--results), [12. Failures](#12-failure-modes--what-went-wrong), [13. Lesson](#13-transferable-lesson), [14. Connections and interview pack](#14-connections--cross-references), [15. Final card](#15-quick-reference-card).

### Case record metadata — place before the case title

```yaml
---
case_id: "CASE-YYYY-NNN"
case_title: "[Decision-focused descriptive title]"
case_version: "1.0"
created: "YYYY-MM-DD"
last_reviewed: null # Set an ISO date after a substantive review.
status: "proposed"
content_status: "draft"
case_type: "analytical"
domain: "[Domain and subdomain]"
primary_pattern: "[Canonical collection pattern]"
secondary_patterns: [] # At most two canonical patterns.
evidence_level: "not-measured"
source_basis: "[first-hand / published / reproduced / illustrative / composite]"
as_of_date: "YYYY-MM-DD"
owner_role: "[Role/team]"
source_index: "[Actual relative path to the collection INDEX]"
review_trigger: "[Specific change that would invalidate a claim or choice]"
---
```

Preserve the archived case types: `analytical / predictive-ml / llm-agentic / streaming-systems / incident / hybrid`. Use `hybrid` only for genuinely integrated methods; describe a specialization such as DSA, CUDA, optimization, or robotics in the domain or a separate local specialization field. Do not force all cases into `hybrid`.

Replace every placeholder in a published case. Use stable IDs and actual relative paths. If a case has not been reviewed, leave `last_reviewed` null and do not mark it reviewed.

### 1. The One-Line Framing

**Required.** State what comes in, what decision or output comes out, for whom, under which constraint, against which baseline.

> Given **[input and unit of analysis]**, decide/predict/build **[output]** for **[stakeholder]** to improve **[business/user outcome]**, subject to **[binding constraint]**, compared with **[current process or baseline]**.

Include:

- Task category and decision frequency: classification, allocation, forecasting, search, generation, dispatch, control, diagnosis, or system operation.
- One primary success condition in plain language; its complete definition belongs in Section 9.
- Primary reusable pattern and no more than two secondary patterns from the INDEX.
- Source/evidence declaration near the opening. If illustrative, say so before the first numerical example.
- A supported surprising finding or decisive tradeoff, if one exists. Do not invent a paradox for storytelling.
- Why the obvious simpler framing misses an important part of the decision.

**Interview cue:** A reader should be able to give a precise 30-second problem statement without listing tools.

**Acceptance:** Input, output, beneficiary, baseline, constraint, and evidence basis can all be identified from this opening.

### 2. Context & Background

**Required.** Explain the business workflow and operating environment that make the problem meaningful.

| Context field | Write explicitly |
|---|---|
| Domain and product | What is being operated or analyzed; relevant domain terminology |
| Workflow | What happens before and after the decision; affected customers and employees |
| Stakeholders | Decision owner, users, engineering/data teams, operations, reviewers |
| Baseline | Existing manual/rules/model/system process and its documented weaknesses |
| Scale | Entity count, events, requests, assets, locations, peak/seasonal conditions, as known |
| Timing | Relevant event dates, business cycle, project stage, source date |
| Geography and boundaries | Regions, language, channels, deployment/data boundaries |
| Prior attempts | Existing investments, constraints, failures, organizational dependencies |
| Applicable requirements | Verified policy/rule source and date, when relevant |

Separate known facts, assumptions, historical conditions, and current verification. If anonymizing a company, explain what was changed and how that limits interpretation. Exclude generic industry history unless it changes the decision.

**Interview cue:** Explain who experiences the failure and why the current process persists despite its limitations.

**Acceptance:** A reader can draw the workflow and identify where the intervention changes it.

### 3. The Business Question

**Required.** Use the stakeholder's actual words when available. If reconstructed, label it a paraphrase; if hypothetical, label it an illustrative stakeholder question. Do not present invented dialogue as a quotation from a real person.

| Decision contract | Fill in |
|---|---|
| Question | `[One or two plain-language sentences]` |
| Decision owner | `[Role with authority to choose, reject, or defer]` |
| Trigger and deadline | `[What changed; why a decision is needed now]` |
| Available actions | `[Concrete alternatives, including no change]` |
| Value and downside | `[Benefit, loss, customer impact, operational consequence]` |
| Objective | `[Quantity to improve and appropriate aggregation]` |
| Hard constraints | `[Limits that disqualify options]` |
| Scope | `[Population, location, channel, workflow, time horizon]` |
| Out of scope | `[Excluded decisions and handoff]` |

Explain the relationship between the business objective and any proxy metric. If they may disagree, name the conflict and evaluation guardrail. For example, more clicks may not mean more profitable completed orders; forecast error alone may not determine inventory service level.

**Interview cue:** Ask which objective is being optimized, what happens with no intervention, and what an expensive false decision looks like.

**Acceptance:** The question can be answered by an action or a justified decision to gather more evidence.

### 4. Assumptions & Constraints

**Required.** Distinguish assumptions to test, externally imposed constraints, and choices under the team's control.

| ID | Assumption/constraint | Type | Evidence or verification plan | Status | Impact if wrong | Fallback | Recheck trigger / owner |
|---|---|---|---|---|---|---|---|
| `A-01` | `[Availability or model condition]` | `[Data/model/business/infrastructure]` | `[Artifact or test]` | `[Verified/unverified/violated]` | `[Decision that fails]` | `[Alternative]` | `[Trigger / role]` |
| `C-01` | `[Binding limit]` | `[Cost/latency/safety/security/etc.]` | `[Requirement source]` | `[Confirmed/review needed]` | `[Consequence]` | `[Redesign/escalation]` | `[Trigger / role]` |

Cover relevant data availability, prediction-time correctness, representativeness, label delay, stationarity, capacity, skills, budget, timeline, dependencies, privacy, access, and physical conditions. Explicitly identify the two or three assumptions most likely to reverse the design or recommendation.

Do not infer legal applicability from geography alone or copy the archive's illustrative regulatory example as a conclusion. Record the actual context and authoritative source if such a determination is material.

**Interview cue:** Change one invalidating assumption and explain how the method, architecture, or decision changes.

**Acceptance:** Each critical assumption has a check and a consequence; each constraint has a source or an explicit scenario assumption.

### 5. Data, Tools & Method

**Required.** Preserve subsections 5a–5e. Make the reasoning reproducible to the extent allowed by source access.

#### 5a. Data Specification

| Field | Required detail |
|---|---|
| Unit of analysis | Entity/event/decision grain; keys; repeated observations |
| Volume and window | Rows/entities/events, dates, time zone, sample and coverage |
| Inputs | Fields/signals, meanings, units, availability timestamps |
| Outcome/label | Exact definition, source, maturity, adjudication, censoring |
| Population and exclusions | Inclusion rules, dropped records, missing segments, selection effects |
| Join semantics | Keys, cardinality, duplicates, point-in-time linkage, reconciliation |
| Splits | Train/validation/test or causal cohorts; entity/time/geographic boundaries |

For non-model systems cases, describe workload traces, message schemas, state snapshots, and correctness oracles instead of inventing a training dataset.

#### 5b. Tools & Stack

Record the tools actually used or explicitly proposed: language, library, platform, storage, messaging, training/serving components, hardware, operating environment, API/model versions, configurations, and ownership. Explain the role of each component. Omit irrelevant tools.

#### 5c. Method Summary

Describe `input -> transformation -> estimation/algorithm -> post-processing -> decision`. Include the simplest credible baseline and two or three plausible alternatives where meaningful. Label whether alternatives were tested, estimated, or only considered.

| Candidate | Evidence considered | Strength | Limitation / avoid when | Chosen/rejected/not tested | Revisit if |
|---|---|---|---|---|---|
| `[Baseline]` | `[Artifact or absence]` | `[Benefit]` | `[Boundary]` | `[Status + reason]` | `[Condition]` |
| `[Chosen approach]` | `[Artifact]` | `[Benefit]` | `[Boundary]` | `[Status + reason]` | `[Condition]` |

Define model objective, constraints, optimizer/solver, feature/representation logic, hyperparameter procedure, stopping condition, threshold selection, and algorithmic invariant as applicable. Explain the mechanism rather than relying on framework names.

#### 5d. Data provenance, quality, exclusions and leakage controls

Document source IDs, owner/access basis, extraction dates, versioned snapshots, transformation lineage, freshness, missingness, skew, duplicates, outliers, permissions, retention, and redactions. Explain what data was unavailable.

For predictive work, audit the earliest time every feature and label could have been known. Fit learned preprocessing on training data within the evaluation procedure; keep the final test data outside tuning. The official [scikit-learn pitfalls guide](https://scikit-learn.org/stable/common_pitfalls.html) illustrates leakage and inconsistent preprocessing; it is a method reference, not evidence for the case's model performance.

#### 5e. Reproducibility and method-selection ledger

Record code revision, dependency versions, configuration, random seeds where relevant, data/model/prompt/index versions, run ID, and artifact locations. Say what can and cannot be reproduced when inputs are restricted. A seed alone does not establish reproducibility across platforms or nondeterministic execution.

| Artifact/source ID | Location | Version/date | Purpose | Accessibility | Reproduction limitation |
|---|---|---|---|---|---|
| `SRC-01` | `[Primary source title/link]` | `[Date/version]` | `[Claim supported]` | `[Public/restricted/unavailable]` | `[What cannot be verified]` |
| `RUN-01` | `[Actual run artifact]` | `[Revision/configuration]` | `[Test or calculation]` | `[Access]` | `[Environment/scope limit]` |

**Interview cue:** Defend the split, label, baseline, and most dangerous join or preprocessing step.

**Acceptance:** Another practitioner can reproduce the logic, identify unavailable evidence, and tell tested decisions from proposals.

### 6. Analysis Walkthrough

**Required.** Open with the decisive supported result or tradeoff, then explain the evidence in a traceable sequence. Preserve intermediate steps; do not jump from raw data to a conclusion.

| Step | Hypothesis or decision | Input snapshot | Method/parameters | Intermediate output | Interpretation | Evidence/status |
|---|---|---|---|---|---|---|
| `1` | `[What is being tested?]` | `[Version/time]` | `[Reproducible operation]` | `[Table/plot/calculation]` | `[What changes?]` | `[Artifact + label]` |

For each significant step, explain why it is necessary, what could go wrong, and how the output is checked. Include one small worked calculation or trace where it makes the mechanism auditable.

**What we tried first and why it did not work:** Required when an approach was actually tested and rejected. Record its rationale, comparable workload, outcome, and stopping reason. Otherwise say `not tested` and discuss it as an alternative, not fabricated project history.

**Sensitivity / Robustness Check:** Required for quantitative conclusions. Vary at least one material specification; choose additional tests by the main uncertainty, not a fixed number of decorative checks.

| Stress/test | Original condition | Alternative | Change in conclusion | Uncertainty / limitation | Decision consequence |
|---|---|---|---|---|---|
| `[Time/population/parameter/cost/load]` | `[Setting]` | `[Variation]` | `[Stable/reversed/inconclusive]` | `[Evidence]` | `[Action]` |

Include negative or inconclusive results when available. Identify the highest-leverage step and one branch where another reasonable choice would change the answer. Separate observation, interpretation, and causal identification.

**Interview cue:** Give the candidate a partial intermediate output and ask what they would check before accepting it.

**Acceptance:** The finding follows from shown work, including the most plausible competing explanation.

### 7. The Finding

**Required.** Answer the business question in one paragraph of roughly three to five sentences. Introduce no new evidence.

1. State the answer or recommended decision.
2. Give supported magnitude and comparator, or say the result is not measured.
3. State uncertainty and evidence scope.
4. Name the main boundary and the next justified action.

Do not call an offline score a production benefit, a modeled saving a realized saving, or an observational association an intervention effect. A rigorous finding may be that no reliable decision is yet supported.

**Interview cue:** Summarize the answer for a business owner, then explain what evidence could reverse it.

**Acceptance:** Every sentence maps back to the walkthrough or evaluation; the paragraph answers Section 3 directly.

### 8. System/Solution Design

**Conditional:** Detailed design is required for built/proposed systems, ML, agents, streaming, infrastructure, and edge/robotics. For a purely analytical case, retain the heading and explain its limited applicability; describe the decision workflow if useful.

#### 8a. Offline / Training / Batch Path

Show data ingestion, validation, transformations, training/optimization where present, evaluation gates, versioned artifacts, registry/handoff, and scheduled updates. Include hardware, partitioning, parallelism, checkpoints, and reproducibility only when relevant. For batch-only systems, explain that the online path is not applicable.

#### 8b. Online / Serving / Real-Time Path

Trace one request, event, or control cycle through authentication, inputs/features/retrieval, computation, state, output/action, logging, and feedback. State timeouts, freshness, authorization, and error behavior at boundaries. Show cold-start, peak-load, and dependency-failure behavior.

At least one labeled input-to-output architecture diagram is required for systems cases. Use both offline and online diagrams if the paths differ. Example shape only:

```text
[Input + timestamp] -> [Validate / authorize] -> [Transform / retrieve]
                               |                         |
                               v                         v
                         [Reject + audit]          [Versioned state]
                                                         |
                                                         v
                         [Metrics / traces] <- [Decision / action gate]
                                                         |
                                                         v
                                              [Output / approved action]
```

Draw the actual design, including feedback, retries, human review, quarantine, or safe-state paths where they exist. A plausible diagram must still be labeled proposed if it has not been implemented.

#### 8c. Key Design Decisions

| Decision ID | Choice | Alternatives | Evidence | Tradeoff | Failure boundary | Revisit trigger |
|---|---|---|---|---|---|---|
| `DEC-01` | `[Selected option]` | `[Credible options]` | `[Measurement or assumption]` | `[Gain and cost]` | `[When choice fails]` | `[Condition]` |

#### 8d. Architecture artifacts and operational readiness

| Area | Questions the content must answer |
|---|---|
| Interfaces | What schemas, versions, error contracts, access boundaries, and idempotency keys apply? |
| Data/state | What consistency, ownership, retention, migration, backup, and reconciliation semantics hold? |
| Reliability | What is the user-facing SLI, target, window, measurement location, and error-budget response? |
| Resilience | How are timeouts, retry budgets, backoff, overload, circuit breaking, and graceful degradation bounded? |
| Recovery | What can be lost or duplicated; what restore/replay tests establish recovery behavior? |
| Observability | Which logs, traces, data-quality signals, alerts, dashboards, and owners support diagnosis? |
| Deployment | What is versioned together; how do shadow/canary/rollout, migration, compatibility, and rollback work? |
| Security | What threats, identities, tenant boundaries, permissions, secrets, and audit trails matter? |
| Economics | What is the cost per useful outcome, capacity envelope, utilization, and sensitivity to peak demand? |
| Human workflow | Who can approve, override, investigate, stop, or recover the system? |

Use user-centered SLI definitions, target windows, and ownership; [Google's SLO workbook](https://sre.google/workbook/implementing-slos/) provides a methodological reference. Select targets for the actual scenario; do not import its examples as universal SLOs.

**Sizing requirements:** Define workload, peaks, bursts, concurrency, state growth, memory, network, storage, and compute bottlenecks that matter. Show units and at least one worked estimate. Distinguish an analytical lower bound, an assumption, a benchmark, and a measured production observation. Do not add component p99 values and label the sum as the end-to-end p99 without a valid basis.

**Interview cue:** Ask what happens if one dependency fails, traffic rises tenfold, state is stale, or rollback meets a changed schema.

**Acceptance:** The reader can follow a successful request and a failed request and explain who owns recovery.

### 9. Evaluation & Validation

**Required.** Make success falsifiable. Separate the intended protocol, the executed protocol, and deviations. For retrospective cases, say when thresholds were actually chosen rather than implying preregistration.

| Evaluation field | Required content |
|---|---|
| Primary metric | Definition/formula, numerator, denominator, unit, population, aggregation, window |
| Baseline | Current process or control, selection logic, measurement dates |
| Pass/fail rule | Threshold, practical significance, owner, timing of selection |
| Guardrails | Relevant cost, latency, reliability, safety, fairness, adoption, or business constraints |
| Test design | Holdout, temporal backtest, experiment, replay, simulation, load test, or review rubric |
| Sample | Size, inclusion, stratification, independence/cluster assumptions, label maturity |
| Uncertainty | Appropriate interval, resampling/model assumptions, power or sensitivity where relevant |
| Validity threat | Most plausible leakage, confound, contamination, selection, or workload mismatch |
| Artifact | Data/configuration/run/rubric/query/log IDs and access limits |
| Decision | Passed, failed, inconclusive, or not executed, with reason |

For experiments, document the randomization unit, exposure/eligibility, interference, sample-ratio checks, stopping rule, and denominator changes as applicable. For observational or quasi-experimental work, state the identification assumptions and falsification checks; avoid causal language without support. For small or dependent samples, choose uncertainty methods appropriate to that design.

For human evaluation, record rubric, sampling, blinded/calibrated review where appropriate, disagreement handling, and adjudication. For automated judges, specify judge/prompt versions, calibration against human labels, repeatability, and known biases.

**Illustrative arithmetic, not an experiment result:** A metric changing from `9,200 / 10,000 = 92%` to `9,500 / 10,000 = 95%` changes by **3 percentage points**, or approximately **3.26% relative to the original rate**. This arithmetic alone establishes neither statistical significance nor causality; both depend on the data and study design.

Apply the relevant profile in Part D for predictive ML, causal work, optimization, LLMs, streams, DSA, accelerators, or robotics. A model-only metric does not establish end-to-end business success.

**Interview cue:** Ask which test could falsify the preferred solution and how the evaluation could accidentally make a bad solution look good.

**Acceptance:** An independent reader can determine what would fail the case, what was actually tested, and what remains unvalidated.

### 10. Action Taken

**Required.** State what the organization actually did, or explicitly state that the case remains a recommendation/exercise.

| Action field | Record |
|---|---|
| Recommendation | `[Specific change or decision]` |
| Approval/status | `[Role, dated record, proposed/approved/rejected/not recorded]` |
| Implementer and handoff | `[Team, artifact, runbook, ownership]` |
| Effective scope | `[Date, region, users, system/data/model version]` |
| Release strategy | `[Pilot/experiment/staged/full/no deployment]` |
| Stop/rollback | `[Trigger, authority, procedure, compatibility limit]` |
| Excluded actions | `[What was deferred or rejected and why]` |

If no action occurred, write `No action taken as of [date]` and explain the blocker or next decision. Use `would` or `proposed` for a design exercise. Do not convert the proposed architecture into a deployment story.

**Interview cue:** Explain what needed organizational approval and how operations would take ownership.

**Acceptance:** Implementation claims have an action record; proposed work remains visibly proposed.

### 11. Outcome & Results

**Required.** Report the supported outcome after the action, or state that it has not been measured. Keep target, offline test, pilot result, production result, and forecast separate.

| Metric | Baseline: value/window/N | Treatment/after: value/window/N | Absolute change | Relative change | Uncertainty | Evidence/source | Limitation |
|---|---|---|---|---|---|---|---|
| `[Primary + unit]` | `[Value or unavailable]` | `[Value or not measured]` | `[Difference]` | `[Defined if baseline permits]` | `[Interval/method or not available]` | `[Label + artifact]` | `[Scope]` |
| `[Guardrail]` | `[Value]` | `[Value]` | `[Change]` | `[Change]` | `[Uncertainty]` | `[Evidence]` | `[Tradeoff]` |

Preserve denominator/population changes explicitly. Relative change is undefined or misleading for some zero, negative, or incompatible baselines; report a meaningful alternative rather than forcing a percentage. Report operational adoption, sustained duration, cost, review effort, and adverse movements where known.

Place a source declaration beside the first results table:

- **Illustrative:** “Figures are constructed teaching values, not measurements of any named organization or evidence of realized impact.”
- **Observed/published:** “Each result is linked to its measurement artifact or named source; reported third-party claims are identified separately.”
- **Mixed:** “Observed, reported, estimated, simulated, and illustrative values are labeled separately and support different conclusions.”

If there is no outcome evidence, retain the section with `Outcome not yet measured`, an explicit measurement plan, owner role, proposed window, and decision trigger. Do not fill the table with hypothetical gains. A limited, failed, or inconclusive outcome is a valid case.

**Interview cue:** Ask what improved, what worsened, and whether the reported change can be attributed to the intervention.

**Acceptance:** Every headline result is scoped, traceable, and consistent with the evidence metadata and actual implementation stage.

### 12. Failure Modes & What Went Wrong

**Required for production and interview use.** The archived outline made this conditional for some analytical cases; this guide strengthens it for the requested use. For analytical cases, cover fragile assumptions, data blind spots, misleading metrics, or decision errors even when there was no incident.

| Failure ID | Trigger and symptom | Detection / discriminating test | Confirmed cause or hypothesis | Impact/blast radius | Containment | Prevention / owner | Verification / limitation |
|---|---|---|---|---|---|---|---|
| `F-01` | `[Observable condition]` | `[Check + expected signal]` | `[Evidence and confidence]` | `[Who/what affected]` | `[Immediate action]` | `[Bounded corrective change]` | `[Regression or recurrence check]` |

Include meaningful failures spanning relevant data, model/algorithm, systems, human workflow, and business feedback. Label each as observed, reproduced, published, simulated, or hypothetical. Do not invent real failures to reach a row count.

For an incident, add a timestamped timeline: onset, detection, triage, containment, restore, data repair, and follow-up. Define detection/recovery metrics and their start/end points. Use a fault tree or evidence-backed causal chain; keep confirmed causes, contributors, and unproven hypotheses distinct. A five-whys sequence is an investigative aid, not proof of a single cause.

Record failed mitigations, rollback limitations, and why existing alarms/tests did not catch the issue. Make preventive actions specific, owned, and testable; avoid “be more careful” or “add monitoring” without a signal and response.

**Interview cue:** Present the symptom first. Require competing hypotheses, a discriminating diagnostic, a safe mitigation, and a prevention test.

**Acceptance:** A reader learns how to distinguish this mechanism from a similar-looking failure and how to verify the fix.

### 13. Transferable Lesson

**Required.** Draft the lesson before the analysis as a hypothesis; revise, narrow, or reject it after evaluation. Never adjust the facts to preserve the intended lesson.

| Pattern field | Required content |
|---|---|
| Canonical pattern | `[INDEX tag and short mechanism name]` |
| General principle | `[One or two precise sentences]` |
| Preconditions | `[Data, workflow, scale, resources, decision context]` |
| Mechanism | `[Why the approach helps; evidence status]` |
| Supporting evidence | `[Case findings, uncertainty, limitations]` |
| Avoid when | `[Concrete disqualifier or counterexample]` |
| Portability test | `[Question or small test to establish fit elsewhere]` |
| Adaptation | `[What changes in another domain or scale]` |

Separate a demonstrated mechanism from a plausible explanation. A single case rarely proves a universal rule. Explain which constraints are structural and which are local implementation choices.

**Interview cue:** Transfer the approach to a contrasting domain and identify the first assumption that breaks.

**Acceptance:** The lesson remains useful without the company name and includes a condition under which following it would be harmful or misleading.

### 14. Connections & Cross-References

**Required for this library.** Preserve the archived connection types and open-question register. Add the interview pack here so the original 15-section structure stays intact.

#### 14a. Related cases, theory, and cheat sheets

| Relationship | Actual target | Mechanism shared or changed | Boundary learned |
|---|---|---|---|
| `extends` | `[Existing relative link]` | `[What is added]` | `[Limit]` |
| `contrasts-with` | `[Existing relative link]` | `[Different population/scale/constraint/outcome]` | `[Why result differs]` |
| `uses-method` | `[Relevant theory/cheat sheet]` | `[Specific equation, diagnostic, or selection rule]` | `[Assumption]` |
| `superseded-by` | `[Existing successor, if any]` | `[What changed]` | `[Date/trigger]` |

Use at least one genuine method reference and a comparable/contrasting case when available. If no suitable case exists, record it as planned plain text in the backlog; do not create a broken link or imply the comparison has been validated.

#### 14b. What this case leaves open

| Open question | Why current evidence is insufficient | Data/test that would resolve it | Priority / trigger | Owner role |
|---|---|---|---|---|
| `[Question]` | `[Specific gap]` | `[Experiment, audit, replay, cohort, stress test]` | `[Condition]` | `[Role]` |

Aim for two to four substantive unresolved questions when relevant. If the bounded question is closed, explain why and retain the external-validity limits. Copy actionable follow-ups into the INDEX backlog.

#### 14c. Intense interview pack — questions with checked answers

Include **12–18 distinct questions** for a full interview-oriented case. This is a coverage target, not permission to pad. Every question must have an answer, reasoning, and a follow-up; keep candidate prompts separate from interviewer notes within this subsection.

| Coverage | Suggested allocation | What the answer must demonstrate |
|---|---:|---|
| Framing and domain | 2 | Clarify decision, objective, stakeholder, baseline, constraints |
| Data and method | 2–3 | Grain, labels, leakage, assumptions, alternative selection |
| Calculation and sizing | 2–3 | Correct setup, units, arithmetic, sensitivity, bottleneck |
| Design and production | 2–3 | Interfaces, state, failure behavior, rollout, ownership |
| Evaluation and outcome | 2–3 | Comparator, validity, uncertainty, causal boundary, guardrails |
| Debugging and transfer | 2–4 | Competing causes, diagnostic, mitigation, counterexample |

For each question, use this record:

```text
Q-ID / level / skill:
Candidate prompt and explicitly given facts:
What the candidate should clarify or assume:
Strong answer: reasoning -> calculation/design -> check -> conclusion.
Acceptable alternatives and the conditions that justify them:
Follow-up: change one constraint, failure, scale, or evidence assumption.
Red flags / common wrong answer, and why it is wrong:
Scoring: 0 incorrect; 1 partial; 2 correct with assumptions; 3 defended tradeoff.
Answer support: section, formula, artifact, or primary reference.
```

At least one question should test a plausible wrong approach, one should require a numerical estimate, one should debug an ambiguous symptom, and one should change a constraint. For conceptual topics, a quantitative example can test a defined metric or consequence rather than manufacture irrelevant infrastructure sizing.

Suggested practice sequence: a 2-minute framing explanation, a 10-minute method deep dive, and a 30–45-minute design/debug discussion. These are editorial practice formats, not claims about a named employer's interview process.

Do not script fabricated personal project experience. Distinguish what the reader implemented, reproduced, studied, or would propose.

**Acceptance:** Answers can be checked from the case and cited material, reward reasoning, and admit more than one defensible design where appropriate.

### 15. Quick Reference Card

**Required and final section of the individual case.** Preserve all six original fields. Keep values concise and bounded by the evidence.

| Field | Value |
|---|---|
| Domain | `[Industry/subdomain and decision context]` |
| Problem type | `[Task + decision]` |
| Method | `[Selected approach and meaningful baseline]` |
| Primary result | `[Scoped result + evidence label, or not measured]` |
| Key lesson | `[Mechanism + main boundary]` |
| Tags | `[One primary; up to two secondary canonical patterns]` |

Add the archived supplemental record immediately below, within the same section:

| Supplemental field | Value |
|---|---|
| Case ID / version / as-of | `[Stable identity and dates]` |
| Evidence level | `[Archived evidence vocabulary; explain mixed scope]` |
| Implementation stage | `[Actual stage; separate from evidence]` |
| Baseline / primary metric | `[Comparator, metric, unit, window]` |
| Key limitation | `[Largest uncertainty or portability boundary]` |
| Review trigger / owner | `[Concrete condition and responsible role]` |

Do not use `pilot` as an evidence label or `illustrative` as an implementation stage. Put sources, appendices, interview answers, and open questions in earlier sections so the card remains last. The 30-second reader must be able to tell what is known, what is proposed, and when the lesson applies.

## Part C — Collection INDEX: preserve all seven sections

Create the INDEX before authoring cases; finalize its links and statuses after the case content is reviewed. An INDEX entry is a structured brief and retrieval aid, not a completed case. Planning descriptions should identify the decision, mechanism, likely data, validation, failure theme, and expected learning without predicting a successful result.

### C1. Collection metadata

Keep collection owner, last reviewed, review cadence, index version, canonical case-ID format, canonical tags, and source of truth. Add audience, domain boundaries, publication-state convention, and review triggers where useful. Use one canonical `INDEX.md` or `00_INDEX.md` per collection and keep any navigation alias explicit.

### C2. Case catalog

Preserve the original 11 columns:

| Case ID | Case title / relative link | Domain | Case type | Decision / business question | Primary pattern | Secondary pattern(s) | Evidence | Stage | Last reviewed | Key result or limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| `[ID]` | `[Title; link only if file exists]` | `[Domain]` | `[Type]` | `[Concrete question]` | `[One tag]` | `[0–2 tags]` | `not-measured` | `proposed` | `[Date or not reviewed]` | `[Leave result empty; state limitation separately if useful]` |

Document editorial state in a separate column/table if needed; do not overload `Stage`. Proposed cases can have descriptive briefs and validation plans, but no invented results. Link an existing planning brief as a brief, not as a published case. Count unique case IDs, not backlinks or empty files.

### C3. Canonical reusable pattern vocabulary

Define mechanism tags with definition, diagnostic signature, preconditions, counterexample/avoid condition, related cases, theory links, and deprecated synonyms. Aim for roughly 15–20 active patterns **within a focused collection**, expanding deliberately when needed. Do not force an entire multidisciplinary library into only 20 tags or replace mechanisms with company names.

### C4. Cross-case similarity and contrast map

Retain `Pattern | Supporting case(s) | Contrasting case(s) | Theory/methodology link | Boundary learned`. Explain what changes between cases: data, population, intervention, scale, time, constraints, evidence, or outcome. A proposed comparison is a learning plan, not empirical support for the pattern.

### C5. Questions and future-case backlog

Retain `Open question | Originating case | Needed evidence | Proposed next case or experiment | Priority | Owner role | Trigger / review date`. Prefer questions that could change a decision. Mark absent source artifacts, disputed claims, missing contrasts, and untested production assumptions explicitly.

### C6. Review and retirement log

Retain `Case ID | Date | What changed | Evidence or market trigger | Action | Owner role`. Use reviewed, updated, superseded, or retired with reasons and successor links. Preserve stable IDs and historical context. Re-review after a source correction, major dependency/model change, price change, new workload, or invalidated assumption.

### C7. Pre-publish index checks

- [ ] Unique case IDs; existing files, planned briefs, and published cases distinguished.
- [ ] Case types and pattern tags describe the actual mechanism; no blanket defaults.
- [ ] Evidence, implementation stage, and document state remain separate.
- [ ] Decision descriptions are specific; no generic “decision to be framed” in a published catalog.
- [ ] Case, theory, source, contrast, and backlink targets resolve where linked.
- [ ] Measured impact claims match the corresponding case's evidence and limitations.
- [ ] Reviews have actual dates and triggers; unknown review dates are not fabricated.
- [ ] Retired/superseded cases remain discoverable with their reason and successor.
- [ ] Counts exclude placeholders, duplicate references, and unfilled outlines.

## Part D — Relevance profiles: use only those that change the case

### D1. Technical profiles and required depth

These profiles add content inside Sections 4–12, not new top-level case sections. Select multiple profiles when the decision genuinely spans them.

| Profile | Data/method/design depth | Evaluation, production, and interview pressure tests |
|---|---|---|
| Business analytics and causal inference | Metric tree, cohort construction, estimand, confounders, seasonality, identification assumptions, intervention cost | Incrementality, interference, placebo/falsification tests, heterogeneous effects, uncertainty, decision sensitivity |
| Statistics and experimentation | Sampling frame, randomization unit, missingness, estimator, multiple comparisons, stopping rules | Valid uncertainty, sample-ratio problems, practical versus statistical significance, external validity |
| Forecasting and time series | Target grain/horizon, calendar, intermittent demand, censoring, exogenous availability, rolling features | Rolling-origin backtests, seasonal/naive baselines, interval calibration, hierarchy coherence, error-to-business-cost mapping |
| Predictive ML and fraud | Label maturity, entity/time split, class imbalance, feature availability, calibration, threshold economics | Precision/recall at actual review capacity, base-rate shift, subgroup reliability, delayed feedback, retraining/recalibration |
| Search, recommendations, and ranking | Candidate generation, ranking/re-ranking, exposure logs, negative sampling, relevance labels, freshness | Retrieval versus ranking errors, position bias, cold start, diversity, feedback loops, offline-to-online mismatch |
| Optimization and operations research | Objective, variables, units, feasibility, hard/soft constraints, solver, heuristics, uncertainty | Infeasible cases, sensitivity/shadow prices where valid, runtime versus optimality, stress demand, operator override |
| NLP, speech, vision, and multimodal | Data alignment, modality-specific labels, preprocessing, augmentations, language/device/environment coverage | Slice errors, robustness, timestamp/coordinate alignment, transcription/recognition utility, data rights and access |
| LLM, RAG, and agents | Corpus permissions/provenance, chunk/index version, retrieval, context policy, model/prompt/tool versions, state | Task success, groundedness, retrieval coverage, abstention, injection, tool permissions, idempotency, approval, termination, cost per successful task |
| Training and fine-tuning | Corpus composition, deduplication, contamination, objective, tokenizer, adapters, precision, optimizer, distributed/checkpoint strategy | Baseline/ablation, convergence and instability, generalization, checkpoint recovery, throughput and memory, quality-cost tradeoff |
| Streaming and data platforms | Event/processing time, watermarks, schema evolution, keys, state, partitioning, retention, delivery semantics | Late/duplicate/out-of-order events, hot keys, replay, idempotent sinks, checkpoint restore, data correctness, backlog age |
| ML systems and MLOps | Feature consistency, registry, artifact lineage, release unit, online/offline paths, monitoring ownership | Training-serving skew, drift, stale features, feedback integrity, model/data rollback compatibility, rollout regression |
| DSA and algorithm design | Formal input/output, invariants, proof sketch, data-structure choice, assumptions, operation sequence | Adversarial inputs, edge cases, overflow, numeric error, amortized versus worst-case bounds, mutable/streaming workloads |
| Distributed systems and incidents | State ownership, ordering, consistency, transactions, concurrency, retries, failure domains | Partial failure, split ownership, duplicate effects, overload, rebalances, data repair, causal incident evidence |
| CUDA, compilers, and accelerators | Device/architecture, precision/layout, memory movement, parallelism, synchronization, compilation/runtime versions | Correctness oracle, race/numerical checks, measured profiling, realistic traffic, occupancy/bandwidth/compute bottleneck, portability |
| Edge, embedded systems, and robotics | Sensor timestamps/calibration, coordinate frames, state estimation, perception/planning/control boundaries, device budgets, connectivity | Deadline misses/jitter, sensor dropout, thermal/power limits, safe state, watchdog, network loss, simulation-to-real gap, hardware-in-loop validation |
| Graph ML and network analysis | Node/edge semantics, graph construction time, leakage, sampling, temporal/inductive setting, entity resolution | Degree skew, connected components, false-positive investigation, graph drift, explainability, graph traversal bounds |
| Security and governance | Threat model, data/model/tool trust boundaries, identities, least privilege, tenant isolation, retention | Abuse cases, policy enforcement tests, traceability, incident response, review and remediation ownership |

For robotics, distinguish cloud analytics from on-device inference and from the control/safety path. Road-network shortest-path reasoning alone is not coverage of localization, SLAM, motion planning, collision avoidance, or control stability. Require evidence for whichever of those the case actually claims.

The official [ROS 2 real-time programming tutorial source](https://github.com/ros2/ros2_documentation/blob/jazzy/source/Tutorials/Demos/Real-Time-Programming.rst) discusses deadlines, jitter, and nondeterministic operations. Use it to motivate timing analysis; it does not establish a robot's safety or the worst-case timing of an application.

For AI risk identification, the [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework) is a voluntary methodological reference. Record the actual risks, controls, evaluation, and owner in the case; citing the framework does not certify a deployment or substitute for applicable requirements.

### D2. Business-domain profiles

Pick the workflow first, then the methods. These are authoring prompts, not claims that every listed company uses a particular model or architecture.

| Domain | Decisions and data to examine | Production constraints and interview tradeoffs |
|---|---|---|
| Supply chain | Demand planning, sourcing, lead times, inventory/replenishment, allocation, warehouse picking, transport, returns, control-tower exceptions | Service level versus working capital, stockout-censored demand, perishability, supplier reliability, disruptions, network capacity |
| Offline/online stores and omnichannel | POS and online orders, item/store hierarchy, loyalty, promotions, assortment, inventory accuracy, substitutions, store fulfillment | Inventory reservation, demand transfer, store labor, shrink, pickup versus delivery, identity matching, channel cannibalization |
| E-commerce and marketplaces | Catalog/entity resolution, search, ranking, recommendations, pricing, seller quality, fraud, payments, fulfillment, returns | Multi-sided incentives, cold start, peak campaigns, ranking freshness, stock consistency, contribution margin, seller/customer fairness |
| Quick commerce and food delivery | Dark-store/restaurant availability, assortment, freshness, dispatch, batching, ETA, rider supply, substitutions | Supply-demand imbalance, weather/events, perishable stock, travel-time uncertainty, cancellation, incentives, dispatch feedback |
| Streaming media and entertainment | Content metadata, rights/windows, search/discovery, recommendations, engagement, subscription churn, ad yield, live delivery | Startup/rebuffering experience, content availability, live peaks, delayed churn labels, household profiles, rights/geography boundaries |
| Banking, payments, and financial risk | Authorization, fraud/AML triage, credit, collections, disputes, liquidity/risk forecasts | Delayed outcomes, investigator capacity, customer friction, adversarial drift, auditability, cost-sensitive thresholds |
| Insurance | Underwriting, pricing, reserving, claims severity/frequency, document intake, fraud rings, retention | Exposure, censoring/development, catastrophe shifts, adjudication delay, provider/entity linkage, human review and appeals |
| Healthcare and life sciences | Patient/provider workflow, demand/capacity, claims, medical documents, trial/biomedical signals, adherence | Temporal/clinical context, missingness, site shift, privacy/access, expert review, outcome validity; avoid unsupported clinical claims |
| Manufacturing and industrial robotics | Inspection, quality, predictive maintenance, yield, scheduling, robotic perception/picking, energy | Sensor drift, rare faults, downtime cost, line takt, hardware integration, calibration, safety envelope, maintenance response |
| Mobility, travel, and hospitality | Matching/dispatch, ETA, routing, capacity, dynamic pricing, cancellations, demand forecasting | Spatial/temporal imbalance, marketplace interference, network constraints, weather, customer and operator incentives |
| SaaS, telecom, and enterprise support | Churn, product adoption, capacity, network anomalies, support routing, enterprise knowledge workflows | Tenant boundaries, contract/SLA differences, account-level leakage, low-volume segments, knowledge freshness, escalation |
| Energy, public services, and other domains | Define assets/users, operating decision, data cadence, objective, public/customer consequence | Identify physical/operational constraints, delayed feedback, access, resilience, and the domain-specific cost of error |

Possible company research contexts include Walmart, Tesco, Target, Flipkart, Zepto, Swiggy, Disney, Warner Bros./Warner Brothers, and Hotstar. A company-specific case must name the business unit/product, relevant date/region, primary evidence, and unknown internal details. Without sufficient public evidence, write a clearly hypothetical domain exercise and avoid claiming it describes that company's implementation.

### D3. Case selection and collection breadth

Across a domain collection, plan complementary cases: an analytical decision, a predictive/optimization problem, an end-to-end production system, a failure/incident, and a contrasting approach or boundary case where relevant. This is a coverage lens, not a requirement to invent five cases per folder.

Maintain a matrix of `workflow x decision x method x scale x failure mechanism x evidence maturity`. Use it to identify missing decisions and repeated superficial examples. Case counts and company-name counts do not establish domain completeness.

## Part E — Workflow, source crosswalk, and publication gates

### E1. Recommended drafting and review sequence

1. **Discover:** Read source material, verify access, identify the decision and source basis, and record gaps.
2. **Register:** Create the INDEX brief, case ID, canonical pattern, evidence label, scope, and backlog items.
3. **Frame:** Draft Sections 1–5, the proposed Section 13 lesson, Section 9 protocol, and provisional summary card.
4. **Build the account:** Write the walkthrough and design from available evidence. Keep untested alternatives visibly untested.
5. **Report honestly:** Complete the finding, action, outcome, failures, and uncertainty. Resolve contradictions across sections.
6. **Challenge:** Apply domain profiles, independent numerical checks, adversarial interview questions, and transfer tests.
7. **Connect:** Add actual links, contrasts, method references, source ledger, and actionable open questions.
8. **Publish:** Finish Section 15 last, complete technical/editorial review, and update the INDEX and review log.

### E2. Preservation and improvement crosswalk

| Archived requirement/improvement | Preserved implementation in this template |
|---|---|
| All 15 case sections | Part B, same sequence and purposes; conditional applicability remains explicit |
| Metadata and evidence taxonomy | Case record and A3; evidence separated from stage and editorial state |
| Lead with supported surprise | Sections 1, 6, 7; decisive result allowed when no surprise exists |
| Show wrong path | Sections 5, 6, 12; distinguish tested rejection from considered alternative |
| Checkable assumptions | Section 4; verification, consequence, fallback, review trigger |
| Canonical pattern tags | Sections 1, 13, 15 and INDEX vocabulary |
| Open questions | Section 14b and INDEX backlog |
| Real versus illustrative results | A2–A4 and Sections 5–7, 9–11, 15 |
| Draft lesson before case | Section 13 and E1; revise against evidence |
| Systems architecture diagrams | Section 8 with distinct paths when applicable |
| Version/date/review trigger | Metadata, context, source ledger, summary, INDEX review log |
| INDEX before cases | Part C and E1; seven archived INDEX sections retained |
| Advanced quality controls | Sections 8–12 and relevant Part D profiles |
| End-to-end traceability | A1 and the publication gate below |
| Requested production/interview depth | Failure analysis and connections strengthened; graded interview pack inside Section 14 |

### E3. Hard publication gate

For a finished case, all applicable items must pass. A draft may retain unknowns, but unknowns must remain visible; they cannot become unsupported claims in the published document.

- [ ] All 15 headings appear in order; conditional `N/A` entries have specific reasons.
- [ ] The central decision, stakeholder, baseline, success condition, scope, and constraints are explicit.
- [ ] Frontmatter, evidence declarations, stage, document state, and as-of dates agree.
- [ ] Critical assumptions have checks; the data grain, target, cut-off, exclusions, and split are auditable.
- [ ] Method and design choices include credible alternatives, disqualifiers, and revisit triggers.
- [ ] Significant reasoning steps have intermediate outputs or clearly identified unavailable evidence.
- [ ] Quantitative conclusions have appropriate robustness/confound checks and stated uncertainty.
- [ ] Systems cases include applicable diagrams, interfaces, state, capacity, observability, security, rollout, rollback, and recovery behavior.
- [ ] Evaluation has a complete metric, comparator, threshold, guardrails, protocol, artifact, and executed/not-executed status.
- [ ] Action and deployment claims have evidence; proposed work remains proposed.
- [ ] Every important result has a label and source/assumptions; causal language matches the study design.
- [ ] Failures distinguish symptoms, causes, hypotheses, tests, mitigation, prevention, and verification.
- [ ] Lesson has preconditions, mechanism, counterexample, and a portability test.
- [ ] Interview pack covers the relevant skills with checked answers, follow-ups, and scoring cues.
- [ ] Sources, relative links, anchors, IDs, units, equations, calculations, and tables have been checked.
- [ ] Open questions identify the evidence needed; the INDEX and review log are updated.
- [ ] Section 15 contains the six original fields and separate evidence/stage metadata and is the final content section.
- [ ] No placeholder, hypothetical metric, unavailable source, or reference-only example is presented as a completed real case.

### E4. Practical review rubric

Score each applicable dimension `0 = missing/incorrect`, `1 = partial`, `2 = usable and supported`. This is an editorial aid, not a certification. Exclude genuinely inapplicable dimensions from the denominator and explain why. A high aggregate score cannot override a fabricated result, critical mathematical error, unsupported deployment claim, or missing material safety boundary.

| Dimension | Evidence for a strong review |
|---|---|
| Framing and domain fit | Decision, workflow, baseline, constraint, and cost of error are clear |
| Data and method | Provenance, availability, transformation, selection, and alternatives are traceable |
| Analysis and evaluation | Reasoning, falsification, uncertainty, and quantitative checks are defensible |
| Production design | Actual/proposed boundaries, operations, cost, recovery, and ownership are clear |
| Outcome integrity | Claims are attributed, scoped, labeled, and consistent with action evidence |
| Failure reasoning | Diagnostics discriminate causes and preventive actions are verifiable |
| Transferability | Lesson, counterexample, contrast, and open questions support reuse |
| Interview value | Answers test reasoning, calculations, changed constraints, and credible alternatives |
| Retrieval and maintenance | INDEX, links, patterns, version, dates, review triggers, and summary agree |

### E5. Archived inputs and consolidation boundary

This guide consolidates and extends both supplied case-study files:

- [Integrated Case Study Outline Template v2](archive/Case_Study_Template/Case_Study_Outline_Template_v2_Integrated.md).
- [Case Study Collection INDEX Template v2](archive/Case_Study_Template/INDEX_Case_Studies_Template_v2.md).

Their 15-section outline, seven-section INDEX, evidence vocabulary, original ten improvements, traceability controls, and publication checks are retained in operational form. Repeated historical explanations are consolidated; screenshot reconstruction notes remain in the archive. Unverified company-result examples, model comparisons, and universal-looking architecture recommendations are not promoted to facts. The technical/domain profiles and detailed interview requirements are additions for this library's intended use. External method references were consulted on 2026-11-06 and should be rechecked when used to support a new technical claim.
