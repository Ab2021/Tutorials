# Case Study Writing Guide & Outline Template — Integrated Edition (v2.0)

> **Editorial note:** This is an **enhanced version** of the 15-section case study template reconstructed from screenshots. The original prompts, examples, and 10 improvement ideas are retained below; the additional numbered tables, decision gates, and controls are **new editorial recommendations**, not claims that they appeared in the screenshots. Original v1 remains available separately.
>
> **How to use:** Keep Sections 1–15 in every case (with explicitly permitted `N/A` for inapplicable conditional paths). Treat `[REQUIRED]` as a writing requirement; apply the AI/ML, streaming, analytics, and incident controls only when relevant. Prefer accurate `unknown / not measured / not done` over invented answers. Evidence-backed production case studies and teaching examples must never be presented as equivalent.

## Case record metadata — put this at the top of each individual case

```yaml
---
case_id: CASE-YYYY-001
case_title: "[Descriptive title]"
case_version: "1.0"
created: YYYY-MM-DD
last_reviewed: YYYY-MM-DD
status: "proposed"  # proposed | pilot | production | scaled | paused | retired | superseded
case_type: "analytical"  # analytical | predictive-ml | llm-agentic | streaming-systems | incident | hybrid
domain: "[Domain]"
primary_pattern: "[canonical-index-tag]"
secondary_patterns: []
evidence_level: "not-measured"  # observed | published | estimated | simulated | illustrative | not-measured
review_trigger: "[Concrete condition under which claims/economics/design must be revisited]"
owner_role: "[Role/team rather than private personal info]"
source_index: "[Relative link to INDEX.md]"
---
```

**Metadata rules:** Use ISO dates; set `status` and `evidence_level` independently (e.g., a deployed system can still have unmeasured impact). Do not mislabel illustrative teaching metrics as actual observations. Update `last_reviewed` only after reviewing the relevant claims.

## Evidence terminology used throughout

| Label | Meaning | Can justify reported impact? |
|---|---|---|
| **Observed** | Directly measured in the described intervention or system | Yes, within stated limits |
| **Published** | Reported by a named third party with a verifiable primary link | May support a *reported claim*, not independent verification |
| **Estimated** | Modeled or extrapolated from stated assumptions | Only as an estimate |
| **Simulated** | Result of a reproducible simulator/test harness | Only for tested simulation conditions |
| **Illustrative** | Invented or composited teaching example | No |
| **Not measured** | Evidence is absent or unavailable | No |

---


> **Original guide note:** Follow the structure below for every case study you write. Each section heading is a prompt — answer it in that section. Sections marked `[REQUIRED]` must always be present. Sections marked `[OPTIONAL]` add depth; include when the information exists.
>
> The original outline was distilled from four reference collections:
>
> - `Case_Studies_AI_Agentic_Systems/` — 15-section ML/AI production case studies
> - `Pharma_Agentic_Case_Studies/` — real-world RAG/Agent implementations with sourced results
> - `DS_Domain/KDA_Solutions_Pharma_Cheat_Sheets/15_Case_Studies_KDA_Pharma.md` — 7-part analytical case studies
> - `Streams_Course_Kafka_Flink_Redpanda/Phase4_Production_CaseStudies/` — systems observability case studies

## Table of Contents

1. [The One-Line Framing (Header Block)](#1-the-one-line-framing-required)
2. [Context & Background](#2-context--background-required)
3. [The Business Question](#3-the-business-question-required)
4. [Assumptions & Constraints](#4-assumptions--constraints-required)
5. [Data, Tools & Method](#5-data-tools--method-required)
6. [Analysis Walkthrough (Step-by-Step)](#6-analysis-walkthrough-required)
7. [The Finding](#7-the-finding-required)
8. [System/Solution Design](#8-systemsolution-design-optional-for-analytical-cases-required-for-systemsml-cases)
9. [Evaluation & Validation](#9-evaluation--validation-required)
10. [Action Taken](#10-action-taken-required)
11. [Outcome & Results](#11-outcome--results-required)
12. [Failure Modes & What Went Wrong](#12-failure-modes--what-went-wrong-required-for-systemsincident-cases-optional-for-analytical-cases)
13. [Transferable Lesson](#13-transferable-lesson-required)
14. [Connections & Cross-References](#14-connections--cross-references-optional-but-strongly-recommended)
15. [Quick Reference Card](#15-quick-reference-card-required)

---

## 1. The One-Line Framing `[REQUIRED]`

**Purpose:** Force you to state the problem precisely before you write anything else. If you can't fill this in, you don't understand the case yet.

```markdown
# Case Study: <Descriptive Title>

> **Problem framing (one sentence, precisely stated):**
> - What comes in (input):
> - What comes out (output/decision):
> - Task category: (e.g. "binary classification on HCP-level data", "streaming pipeline for fraud detection",
>   "LLM fine-tuning for closed-domain Q&A", "marketing mix reallocation analysis")
> - Why this framing and NOT the simpler/more obvious one:
```

**Why this matters:** Fuzzy framing is the #1 reason case studies are hard to reuse. A precise framing tells a future reader in 10 seconds whether this case applies to their problem.

### Opening contract — integrated improvements 1, 4 and 9

A completed case should start with metadata (see **Case record metadata** above), then a **decision-ready one-line framing**:

> Given **[input + unit of analysis]**, decide/predict/build **[specific output]** for **[stakeholder]** to improve **[business or user outcome]**, subject to **[hard constraint]**, compared with **[baseline]**.

- **Pattern name / tags:** Choose one primary reusable pattern and up to two supporting tags from the controlled vocabulary in the INDEX; do not invent near-duplicate labels without updating the INDEX.
- **The surprising point (optional):** One sentence stating the most counterintuitive *supported* result. Write after validation, but position it early for readers. Do not manufacture a paradox.
- **Measurable success condition:** State the target and comparator in plain language; define the full metric in Section 9.
- **Evidence classification:** `observed / published / estimated / simulated / illustrative / not-measured`. The classification must agree with Section 11.

**Acceptance check:** Could an unfamiliar reader name the input, output, choice, baseline, intended beneficiary and boundaries in 30 seconds?


---

## 2. Context & Background `[REQUIRED]`

**Purpose:** Set the stage. Give the reader enough domain knowledge to follow the analysis.

**Fill in:**

- **Domain / Industry:** (e.g., pharma, fintech, streaming infrastructure, insurance)
- **Product / System / Brand:** What is the thing being analyzed?
- **Lifecycle stage:** (e.g., 4th year post-launch, production ML system running 18 months, early pilot)
- **Operating environment:** Key facts about scale, regulations, competitive pressures, or technical constraints that shaped the problem
- **Who is involved:** Which teams, roles, or stakeholders commissioned or were affected by this work?
- **Timing:** When did this happen? (Relative date is fine — "Q3 of year 2")

**Example format:**

> "Brand C is a once-daily oral medication in its 4th year post-launch, operating in a crowded primary-care class with 3 branded competitors and payer-driven step-edit pressure. The brand team's promotional mix has been stable for 18 months..."

### Context details that change whether this case transfers

| Context factor | What to document | Evidence or source |
|---|---|---|
| Users and workflow | Who makes/uses the decision; when; human approval path | [Interview/process doc] |
| System or market baseline | Current process/model, failure rate, cost, quality, scale | [Date and source] |
| Scale and operating conditions | Volume, traffic seasonality, geography, regions, uptime, budget | [Measurement window] |
| External and regulatory conditions | Which rules apply, in which jurisdictions, as of which date | [Authoritative source; verify] |
| Prior attempts and dependencies | Historical solutions, data permissions, upstream/downstream systems | [Link or record] |

Clarify what is **known**, **assumed**, **historical**, and **currently verified**. Make context sufficient to understand the case, not a generic overview of the industry. If privacy or proprietary restrictions apply, anonymize details and label the nature of the redaction.


---

## 3. The Business Question `[REQUIRED]`

**Purpose:** State the actual question that was asked — in the words of the person who asked it, not in technical language.

**Fill in:**

- **Verbatim question (or close to it):** Use quotes. Avoid technical paraphrase.
- **Who asked it:** Role and context (e.g., "Brand VP before the annual planning cycle")
- **Why it was asked NOW:** What changed, what event, what pressure triggered this question?
- **$ or business impact:** What is the value of a good answer? What is the cost of a wrong one?
- **Scope agreed upfront:** What is explicitly in-scope and out-of-scope for this analysis/build?

**Example format:**

> "We're being asked to find 8% of the promotional budget to fund a new indication launch elsewhere in the portfolio. Where can we cut without hurting TRx, and is there a channel we're under-investing in that we should grow instead?"

**Improvement tip:** If you can't write the business question in one or two sentences, the problem is not yet well-defined. Go back and tighten scope before continuing.

### Decision contract — go beyond stating the question

| Field | Prompt |
|---|---|
| Decision owner | Who can approve, reject or defer action? |
| Decision deadline / trigger | By when must a decision be made, and why? |
| Baseline / do-nothing scenario | What happens without the intervention? |
| Primary business metric | Exact metric, unit, population, time window and desired direction |
| Guardrail metrics | What must not worsen (cost, safety, service quality, risk, equity)? |
| Decision threshold | What numerical or qualitative evidence will trigger action? |
| Cost of error | Relative harm of false positive/false negative, bad rollout or delayed decision |
| Scope and exclusions | What is explicitly out of scope and why? |

**Traceability rule:** The main question, the primary evaluation metric (Section 9), the action (Section 10) and the observed result (Section 11) must refer to the **same decision and population**. If they do not, explain the difference.

Use direct quotes only when actually recorded; otherwise label the wording as a **close paraphrase**, not a verbatim quotation.


---

## 4. Assumptions & Constraints `[REQUIRED]`

**Purpose:** State what you took for granted and what limited your choices. This is what makes a case study honest and reusable — it tells the reader when this case would NOT apply to their situation.

**Fill in:**

- **Data assumptions:** What data was assumed to be available, clean, representative, or labeled?
- **Model/system assumptions:** What technical capabilities were assumed (e.g., "GPU budget for a 7B model fine-tune", "Kafka cluster already in production")?
- **Business/operational constraints:** Latency SLOs, cost caps, compliance requirements, data residency, org maturity
- **Explicitly stated simplifications for v1:** What was deliberately left out of the first version?
- **What would change the design if wrong:** Name 2–3 assumptions that, if violated, would lead to a different approach

**Format as a table where possible:**

| Assumption / Constraint | Type | Impact if Wrong |
|---|---|---|
| HCP data has NPI-level linkage | Data | Cannot do HCP-level segmentation |
| Kafka cluster is already provisioned | Infrastructure | Add 4–6 weeks of setup time |
| EU AI Act does not apply (US-only deployment) | Regulatory | Mandatory human oversight and logging required |

### Assumption register — use a verifiable checklist (improvement 3)

Replace assumptions buried in prose with a reusable register. The three core fields are **Assumption**, **Type**, and **Impact if Wrong**; add verification details so the table can drive an actual review.

| ID | Assumption or constraint | Type | Evidence / owner | Validation status | Impact if wrong | Fallback / review trigger |
|---|---|---|---|---|---|---|
| A-01 | [Data available at required grain] | Data | [Source / owner] | Verified / unverified | [Which step fails] | [Alternate method] |
| A-02 | [Throughput below design envelope] | Infrastructure | [Load test / SRE] | Verified / unverified | [SLO/cost effect] | [Traffic threshold] |
| C-01 | [Retention or access restriction] | Regulatory / security | [Policy + jurisdiction] | Confirmed / review needed | [Noncompliance risk] | [Approval / redesign] |

- Distinguish **assumptions** (claims requiring tests), **constraints** (requirements imposed on the solution) and **design choices** (options under your control).
- Mark the **2–3 invalidating assumptions** that would force a different method; define how each will be monitored or rechecked.
- Include data availability, selection bias, label quality, capacity, latency, cost, privacy, safety and organization dependencies **where applicable**.
- Regulation examples in the source template are **illustrative**, not a legal conclusion. Determine applicability using the actual deployment context and current guidance.

**Acceptance check:** A reader should be able to mark each assumption `holds / does not hold / not tested` for their setting and know what to do next.


---

## 5. Data, Tools & Method `[REQUIRED]`

**Purpose:** Give enough detail that someone else could reproduce or audit the analysis.

### 5a. Data Specification

| Item | Detail |
|---|---|
| Unit of analysis | (e.g., HCP, claim, Kafka partition, model checkpoint) |
| Sample size / volume | |
| Observation window | |
| Outcome / target variable | |
| Key input features / signals | |

> **Transcription note:** The bottom edge of the screenshot cuts off the next row of the data-specification table (beginning with "Exclusions and …"). Its complete label and any further entries are not visible, so they have not been invented.

### 5b. Tools & Stack

- **Languages/frameworks:** (e.g., Python + sklearn, Flink, PyTorch + LoRA, dstools)
- **Infrastructure:** (e.g., on-prem GPU cluster, AWS Bedrock, Databricks)
- **Third-party services or APIs:** (e.g., Anthropic Claude, Salesforce Agentforce)

### 5c. Method Summary

- **Algorithm or approach chosen:** (e.g., OLS + LMG decomposition, RAG with vector DB, LoRA fine-tune)
- **Why this method and not alternatives:** Name 2–3 alternatives and one-line reason each was rejected
- **Back-of-envelope sizing (for systems cases):** Latency budget, throughput, cost estimate, VRAM, token usage — whatever applies

### 5d. Data provenance, quality, exclusions and leakage controls

The original screen truncates part of the data table. The following are **editorial additions**, not claimed verbatim transcription:

| Additional data field | Capture explicitly |
|---|---|
| Dataset/source ID + owner | Database/table/file, API, version, access status, date extracted |
| Grain / primary key | Entity, event, partition, join keys, time-zone convention |
| Coverage and exclusions | Inclusion/exclusion rules, missing groups, geographies, opt-outs, known gaps |
| Sampling and split logic | Sampling frame, cohort selection, train/validation/test dates and entity isolation |
| Data quality | Completeness, freshness, duplicates, skew, label delay, outliers, label reliability |
| Privacy and permissions | Sensitive fields, consent/authorization, retention, redaction, access scope |
| Provenance | Source links, transformation lineage and reproducible snapshot location |
| Leakage / temporal correctness | Prediction cut-off, forbidden future fields, target leakage tests |

### 5e. Reproducibility and method-selection ledger

- Pin environment versions: code revision, package lockfile, model/checkpoint version, prompt version, dataset snapshot, configuration, random seeds, and evaluation artifact ID where relevant.
- Record **baseline first** (rule-based/manual/current model/no intervention), then 2–3 plausible alternatives. Contrast complexity, data requirements, cost, latency, accuracy, interpretability and maintenance.
- Describe the method as a sequence: *input → transformation → fitted rules/model → post-processing → decision/action*. Include equations/pseudocode only where they clarify reproduction.
- For modeling, document target definition, splitting, leakage prevention, weighting, hyperparameter procedure and threshold selection.
- For systems, include back-of-envelope capacity: `peak requests/s × peak resource per request`, storage growth, network egress, P50/P95 latency budget, GPU memory and unit economics. Label assumptions and measurement source.
- Mark third-party APIs/models with version **and review date**. A historical tool choice is not automatically optimal today.

| Candidate | Baseline / option | Evidence considered | Why selected / rejected | Revisit if… |
|---|---|---|---|---|
| M-0 | [Current process] | [Metric + cost] | [Comparator] | [Condition] |
| M-1 | [Chosen design] | [Test / constraints] | [Reason] | [Trigger] |
| M-2 | [Rejected design] | [Test / limitation] | [Reason] | [Trigger] |


---

## 6. Analysis Walkthrough `[REQUIRED]`

**Purpose:** Walk through the actual steps taken. Use numbered steps. Include intermediate outputs (tables, charts, intermediate results) so a reader can follow the reasoning — not just the conclusion.

**Format:**

### Step 1 — [Name of step]

- What was done
- Key intermediate result (table or numbers)

### Step 2 — [Name of step]

- What was done
- Key intermediate result

### Sensitivity / Robustness Check `[REQUIRED for quantitative cases]`

- How was the main finding stress-tested?
- What changed under different assumptions? What didn't change?
- Minimum bar: re-run with one alternative spec and confirm direction holds

**Rules for the walkthrough:**

- Show the numbers, not just the conclusion. Even if figures are illustrative composites, make them directionally realistic.
- Name which step is highest-leverage (the one that most changed the team's view)
- Flag any step where a different choice would have led somewhere different

### Reader-first analysis narrative — integrated improvements 1 and 2

Begin with **the most important supported surprise, tension or decision tradeoff** (not necessarily the first step chronologically). Then walk through the evidence chronologically. If no genuine surprise exists, lead with the decisive result rather than inventing one.

**What we tried first and why it did not work** `[REQUIRED if a rejected approach was actually tested]`

> We initially tried **[method]** because **[hypothesis]**. On **[data / traffic]**, it produced **[observed result]** compared with **[baseline]**. We stopped or changed it because **[evidence-backed reason]**. The next experiment was **[replacement]**.

For **every significant step**, capture:

| Step | Hypothesis / decision | Input snapshot | Method and parameter(s) | Intermediate output | Interpretation | Evidence link |
|---|---|---|---|---|---|---|
| [1] | [What were we testing?] | [ID/time] | [How] | [Number/table/figure] | [What changed our view?] | [Artifact] |

**Robustness matrix** `[REQUIRED for quantitative cases]`

| Test | Baseline specification | Alternative specification / stress | Result and direction | Meaning / limitation |
|---|---|---|---|---|
| [e.g., time split, population mix, parameter choice] | [Original] | [Variation] | [Change with CI if appropriate] | [Robust / not robust / inconclusive] |

- Include at least one **negative result** when available, not only successful steps.
- Separate *observation*, *interpretation* and *causal claim*; an association is not an intervention effect.
- Identify the **highest-leverage step** and one branch where a different choice would alter the result.
- Mark figures `observed`, `simulated`, `illustrative` or `not available` at the point of use. Never present fabricated figures as actual tests or customer outcomes.


---

## 7. The Finding `[REQUIRED]`

**Purpose:** State the answer in one paragraph. No jargon. A non-technical reader should understand it.

**Format:**

> "[One paragraph. Answer the business question directly. State the size/direction of the main finding. Name any qualifications or conditions on the finding. Do NOT introduce new material here — only summarize what the walkthrough showed.]"

**Improvement tip:** If your finding paragraph is longer than 5 sentences, it is too complex. Split it, or move material back to the walkthrough.

### One-paragraph decision answer (do not add new evidence)

Use this four-part sentence structure, within **3–5 sentences total**:

1. **Answer:** What should the stakeholder conclude or do?
2. **Magnitude:** How much did the metric change, versus which comparator, over what period?
3. **Confidence:** What evidence supports it and how uncertain is it?
4. **Boundary:** What must be true for it to apply, and what remains unresolved?

> **Example scaffold:** “For **[population/window]**, **[decision]** was associated with **[measured result vs baseline]**. This meets/does not meet **[predefined threshold]** under **[validated conditions]**. However, **[main limitation]**, so **[recommended next action or restraint]**.”

Do not claim a deployed benefit if only offline validation exists; use **“estimated”**, **“pilot observed”**, **“production observed”** or **“not yet measured”** precisely.


---

## 8. System/Solution Design `[OPTIONAL for analytical cases; REQUIRED for systems/ML cases]`

**Purpose:** Show the architecture. This section is mandatory for any case study involving a built system (ML pipeline, streaming architecture, agentic system, RAG system).

### 8a. Offline / Training / Batch Path

- Data pipeline steps
- Training job configuration (parallelism, optimizer, checkpointing)
- Post-training pipeline (eval gate, quantization, fine-tune → deploy handoff)
- ASCII diagram of offline path

```text
[Raw Data] → [Dedup + Quality Filter] → [Tokenizer] → [GPU Training] → [Eval Gate] → [Registry]
```

### 8b. Online / Serving / Real-Time Path

- Request lifecycle from user to response
- Latency budget breakdown (P50, P95 targets by component)
- Caching, batching, routing decisions
- Failure modes in the serving path
- ASCII diagram of online path

### 8c. Key Design Decisions

For each major decision, use this format:

| Decision | Option Chosen | Why | Alternatives Rejected |
|---|---|---|---|
| Context retrieval strategy | Full-context (1M token) | Cost-competitive post-2026 | RAG (chunking added latency, recall gaps) |
| Model parallelism | Tensor parallel, 4 GPUs | Fits VRAM budget | Pipeline parallel (bubble overhead at 7B) |

### 8d. Architecture artifacts and operational readiness (improvement 8)

**Mandatory minimum for built systems:** at least **one clearly labeled left-to-right ASCII or Mermaid diagram** from input to output with component names, major state stores and failure/approval boundaries. If training/batch and serving/online are different, include **both**. Batch-only systems can mark the online path **N/A** with reason.

```text
[Event / User / Source]
        |
        v
[Validate / Authenticate] --> [Reject / Quarantine + Audit Log]
        |
        v
[Transform / Retrieve] --> [Versioned State / Cache]
        |
        v
[Model / Rules / Agent] --> [Safety / Policy / Human Approval]
        |
        v
[Decision / Response] --> [Metrics + Traces + Feedback]
        |
        v
[Fallback / Retry / Rollback when needed]
```

Adapt this **illustrative** diagram; omit inapplicable components rather than implying all are deployed.

**Design inventory (include those that apply):**

| Area | Specific requirement |
|---|---|
| Interfaces | Inputs/outputs, schema or API contract, idempotency, error codes |
| Data and state | Consistency, durability, retention, PII handling, migration plan |
| Reliability | Availability SLO, P95/P99 targets, timeouts, retries, DLQ, graceful degradation, RTO/RPO if applicable |
| Deployment | Environments, feature flags, version pinning, rollout/canary/shadow, rollback trigger |
| Observability | SLIs, logs, traces, dashboards, alerts, drift/data-quality checks, responsible on-call |
| Security | Threat surface, least privilege, secret handling, audit trail, vendor/data boundary |
| Economics | Cost per task/request, peak capacity, resource utilization, budget guardrail |
| Human factors | Review/override workflow, escalation path, policy and user consent where required |

**AI/agentic-specific checks (conditional):** data retrieval permissions, grounding and citation quality, tool allowlists, prompt-injection resistance, agent/tool action authorization, evaluator safety, token/cost budget, context truncation, fallback to human, and external side-effect rollback.

**Streaming-specific checks (conditional):** event-time/watermarks, partition skew, late/duplicate events, backpressure, checkpoint/recovery, delivery semantics, schema evolution, retention, rebalance behavior and end-to-end correctness.

**Decision table extension:** include **measured evidence, tradeoff, failure mode and trigger to revisit** for architecture choices. The original 1M-token/full-context example is **historical/illustrative** and must not be repeated as a universal contemporary cost claim.


---

## 9. Evaluation & Validation `[REQUIRED]`

**Purpose:** Show how you knew the solution worked — or didn't.

**Fill in:**

- **Offline metrics:** List each metric, its formula, why it was chosen, and what alternatives were rejected
- **Online metrics:** A/B test, canary, shadow rollout — what was measured and over what period
- **Human evaluation design (if applicable):** How were evaluators calibrated? How many samples?
- **Validation scheme:** Held-out sets, adversarial/red-team tests, backtests, leakage checks
- **Statistical rigor:** Sample size, confidence intervals, power calculation where relevant

**Minimum bar for quantitative cases:**

- One primary metric with a threshold that defines "this worked"
- One check that rules out the most likely confound (e.g., reverse causation, data leakage)

### Evaluation protocol — make success falsifiable

Specify the following **before** reporting results:

| Gate | What must be recorded |
|---|---|
| Primary metric | Definition, numerator/denominator, unit, population, aggregation, window |
| Baseline / comparator | Control/current system, baseline period and selection logic |
| Decision threshold | Numeric pass/fail condition and who approved it |
| Guardrails | Safety, cost, latency, fairness, compliance, reliability or adoption limits as appropriate |
| Test design | Randomized holdout, quasi-experiment, temporal split, shadow or benchmark; justify choice |
| Sample and uncertainty | N, stratification, CI/interval, relevant power or practical significance |
| Failure / leakage test | Most likely confound, target leakage, duplicates, distribution shift or adversarial test |
| Evidence artifact | Run ID, dataset version, query/notebook, logs, dashboard, evaluation rubric |

**Conditional domain checks:**

- **Predictive ML:** calibration, subgroup performance, decision threshold, class prevalence, temporal drift, leakage and economic utility; avoid accuracy alone for severe imbalance.
- **LLM/RAG/agents:** task-completion accuracy, groundedness, retrieval coverage, hallucination/error taxonomy, prompt/tool injection tests, human-eval calibration, refusal/abstention accuracy, latency and cost per successful task.
- **Streaming/platform:** throughput under peak/load, delivery/duplication behavior, P95/P99 latency, correctness after recovery, backlog age, time-to-detect and time-to-recover.
- **Business/causal analytics:** seasonality, selection effects, confounders, incrementality, heterogeneity, diminishing returns and scenario sensitivity.

**Acceptance check:** Another analyst should be able to tell **what would have caused this case to fail**, not merely what score was reported.


---

## 10. Action Taken `[REQUIRED]`

**Purpose:** State what the organization actually did as a result of this finding. This is the bridge from analysis to impact.

**Fill in:**

- **Decision made:** Exactly what was changed, built, or deployed
- **Who made the decision:** Role and level (e.g., "Brand VP approved reallocation at annual planning")
- **Timeline:** How long from finding to action?
- **Rollout approach:** Phased, A/B tested, full cutover, pilot first?
- **What was NOT done (and why):** Often as informative as what was done

### Decision-to-deployment record — evidence that action occurred

| Field | Record |
|---|---|
| Approved recommendation | [What exactly changed?] |
| Decision owner + approval | [Role / dated link or record] |
| Implementer + handoff | [Team, artifact / runbook / deployment reference] |
| Release strategy | [Pilot / A/B / staged / full cutover / no deployment] |
| Effective date and scope | [Date, region, users, system version] |
| Guardrails and rollback | [When to stop or reverse? Who has authority?] |
| Excluded alternatives | [What was not pursued, and why?] |
| Status | `proposed / approved / pilot / live / paused / cancelled / not recorded` |

**Important:** If no organizational action occurred, write **“No action taken as of [date]”** and explain whether the work remains a proposal, experiment, recommendation or cancelled initiative. Do not quietly replace an unimplemented recommendation with a deployment claim.


---

## 11. Outcome & Results `[REQUIRED]`

**Purpose:** Report what actually happened after the action was taken.

**Fill in:**

- **Primary metric result:** (e.g., "TRx held flat while spend dropped 8%", "P95 latency cut from 400ms to 95ms", "SQL query accuracy: 95%")
- **Secondary metric results:** Any unexpected movements, positive or negative
- **Adoption/scale reached:** Number of users, volume of transactions, geographic scope
- **Timeline to impact:** How long until results were visible?
- **Status at time of writing:** Pilot / production / scaled / cancelled

**Format sourced results as a table (for real-world cases):** *(Original table examples are reproduced for structural fidelity; verify their sources before citing them.)*

| Metric | Before | After | Source |
|---|---|---|---|
| Manual review time | Baseline | -50% | [Roche — Bioprocess Online] |
| Stock disruption prediction | Reactive | 80% predicted pre-event | [Sanofi — Aily Labs] |

**Rule:** If results are illustrative (not real data), say so explicitly. Use the phrase: *"All figures are illustrative composites built to be directionally realistic — not any single organization's actual data."*

### Results evidence ledger — integrated improvement 6

Show **observed versus estimated versus hypothetical** outcomes separately. Every important number needs a definition, denominator or scale, comparison period, and source or experiment ID.

| Metric | Baseline (window/N) | After or treatment (window/N) | Absolute change | Relative change | Uncertainty | Evidence type + source |
|---|---|---|---|---|---|---|
| [Primary metric and unit] | [Value] | [Value] | [After − before] | [(After − before)/baseline] | [CI or N/A] | [Observed/published/estimated/illustrative + link] |
| [Guardrail] | [Value] | [Value] | [Value] | [Value] | [CI or N/A] | [Source] |

**Reporting rules:**

- Specify whether a difference is **causally attributable** to the intervention. A before/after difference alone usually cannot establish causation.
- Give **effect size and uncertainty** where measurable; also report adoption, business cost/benefit and operational tradeoffs, not only model performance.
- Distinguish **verified production results**, **pilot results**, **offline estimates**, **published vendor claims**, and **made-up teaching examples**; never pool them in a single unlabeled result.
- The original Roche and Sanofi rows demonstrate **format only**; their attribution labels in the source template are not independently verified citations. Add direct primary-source links and dates before treating such numbers as evidence.
- If measured outcomes do not yet exist, explicitly state **“Outcome not yet measured”** and list the measurement plan. Do not invent one.

**Mandatory source declaration (select one and retain near the first results table):**

> **Illustrative/synthetic:** “Figures are illustrative examples, not measurements of any named organization. They are not evidence of realized business or system impact.”
>
> **Real or published:** “Figures are drawn from the linked sources/run artifacts listed with each metric. Source claims and their scope require verification before external publication.”
>
> **Hybrid:** “Observed, third-party-reported, estimated and illustrative results are labeled separately in each table; they must not be interpreted as equivalent evidence.”

Include status and date as-of, and **what worsened or failed to move**, not only positive changes.


---

## 12. Failure Modes & What Went Wrong `[REQUIRED for systems/incident cases; OPTIONAL for analytical cases]`

**Purpose:** Capture the things that did not work as expected. This is the most reusable part of any case study.

**Format:**

For each failure mode:

- **Symptom observed:** What the system/team saw (not the root cause)
- **Root cause:** Why it happened (after investigation)
- **Detection method:** How was it found? (e.g., alert, user report, audit)
- **Resolution:** What fixed it or mitigated it?
- **Prevention:** What architectural or process change prevents recurrence?

**For post-mortem / incident cases:**

> Add a **5-Whys or fault tree** for the primary failure. Explicitly tag which 2026-era shift (or which systemic pattern) the failure belongs to — this helps readers file it correctly in their mental model.

### Failure analysis suitable for reuse and incident review

| Failure ID | Symptom + first seen | Detection | Root-cause hypothesis / confirmed evidence | Blast radius and impact | Containment / resolution | Prevention + owner + due date | Verification |
|---|---|---|---|---|---|---|---|
| F-01 | [What happened?] | [Alert/report] | [Confirmed or unconfirmed] | [Affected scope] | [Mitigation] | [Action and owner] | [How recurrence is tested] |

- Separate **confirmed causes**, **contributing factors** and **unproven hypotheses**. Do not present hindsight as certainty.
- For incidents, include a timestamped timeline, triggering change, detection latency, customer impact, error budget, recovery, and lessons about detection or containment.
- Use a **5-Whys or fault tree** for the main failure when evidence permits; favor a **blameless** account of system and process contributors.
- Include failed mitigations or abandoned rollback attempts if material; make each corrective action observable and owned.
- For non-incident analytical cases, note model or conclusion fragility, failed assumptions, sensitivity failures and data blind spots even if no production outage occurred.


---

## 13. Transferable Lesson `[REQUIRED]`

**Purpose:** Distill the one principle a practitioner should take away and apply elsewhere. This is what transforms a one-off example into a reusable pattern.

**Format:**

> **Pattern name:** (give the lesson a short name, like "Diminishing-returns segmentation before reallocation" or "Under-replicated partition as the universal canary metric")
>
> **The principle:** [One or two sentences. Must be domain-agnostic enough to apply outside this exact context.]
>
> **When to apply it:** [The conditions under which this lesson is relevant. Be specific about what must be true for it to apply.]
>
> **When NOT to apply it:** [The boundary conditions — when would this lesson lead you astray?]
>
> **Connections:** Which other cases in this collection illustrate the same pattern from a different angle?

### Draft-first, evidence-last lesson rule (improvement 7)

1. **Before Sections 6–12:** draft the proposed lesson as a **hypothesis**; use it to choose which evidence, alternatives and failure modes to document.
2. **After Section 11:** update, narrow or discard that hypothesis based on results and uncertainty. Do **not** change facts to fit the intended narrative.
3. **Before publishing:** complete the reusable pattern card below.

| Pattern field | Write this explicitly |
|---|---|
| Pattern ID / canonical tag | [Controlled vocabulary key] |
| General principle | [One or two domain-independent sentences] |
| Preconditions | [Necessary data, scale, workflow and resources] |
| Mechanism / why it works | [Cause or plausible mechanism, with evidence status] |
| Evidence and confidence | [What case findings support it? Limits?] |
| When NOT to apply | [Failure boundaries, counterexample or unacceptable risk] |
| Portability test | [One question another practitioner can answer to check fit] |
| Linked analogues | [At least one comparable or contrasting case where possible] |

**Acceptance check:** Could the lesson still make sense if every industry and product name were removed? Does it specify when it would be harmful or misleading?


---

## 14. Connections & Cross-References `[OPTIONAL but strongly recommended]`

**Purpose:** Link this case to others in the collection and to the theoretical material it draws on. This turns a collection of isolated cases into a connected knowledge graph.

**Fill in:**

- **This case extends:** `[Prior case or theory sheet]` — by adding `[what new element]`
- **Compare and contrast with:** `[Related case]` — same pattern, different domain/outcome
- **Theory/methodology applied:** `[Sheet or module reference]` — which concept from the companion theory material was used here
- **What this case leaves open:** Questions or scenarios not covered here that a follow-up case should address

### Connected-case graph and unresolved questions (improvements 5 and 10)

Use **stable, relative Markdown links** to actual local files; avoid broken links and vague references such as “see the earlier case.”

| Link type | Target | Why it matters | Evidence or contrast |
|---|---|---|---|
| `extends` | `[Case or theory](relative/path.md)` | [What is new] | [Specific comparison] |
| `contrasts-with` | `[Case](relative/path.md)` | [Different conditions/outcome] | [Why divergence matters] |
| `uses-method` | `[Theory module](relative/path.md)` | [Methodological basis] | [Applicable caveat] |
| `superseded-by` | `[Newer case](relative/path.md)` | [What changed since this case] | [Date / trigger] |

**What this case leaves open** `[REQUIRED to address; 2–4 unanswered questions where applicable]`

| Unanswered question | Why evidence is insufficient | Experiment/data that would resolve it | Priority or trigger |
|---|---|---|---|
| [Question 1] | [Missing data or causal ambiguity] | [Holdout, load test, audit, replication] | [Next milestone] |
| [Question 2] | [Boundary condition] | [Stress test, alternative cohort, monitoring] | [Revisit condition] |

If there are genuinely no outstanding known questions, say why the scope is considered closed and identify remaining **external-validity** limits. The INDEX should map each pattern tag to its cases and linked theory, rather than merely list filenames.


---

## 15. Quick Reference Card `[REQUIRED]`

**Purpose:** A six-field summary table that lets a reader scan the case in under 30 seconds. *(Editorial correction: source text calls it five rows, but displays six fields including Tags.)*

| Field | Value |
|---|---|
| Domain | |
| Problem type | |
| Method | |
| Primary result | |
| Key lesson | |
| Tags | |

### Publishable quick-reference record (improvements 4, 6 and 9)

The screenshot's **“5-row”** wording conflicts with the **six displayed rows**. The main table has been corrected to say **six fields**. Keep those six original fields and add the following concise metadata under the table, not as hidden footnotes:

| Supplemental field | Value |
|---|---|
| Case ID + version + as-of date | [Stable ID, version, last reviewed] |
| Outcome status | `observed / pilot / simulated / illustrative / proposed / not measured` |
| Decision stage | `proposed / pilot / production / scaled / paused / retired` |
| Baseline + primary metric | [One-line comparator, metric and unit] |
| Key limitation | [Biggest uncertainty or reason not to generalize] |
| Review trigger | [Condition requiring update, with owner] |

**Tagging rule:** Pick **one canonical primary tag** and at most **two secondary tags** from the shared INDEX. Prefer a reusable mechanism (`data-leakage`, `diminishing-returns`) over a vague topic label (`machine-learning`). Update INDEX definitions and backlinks when adding new tags.

**30-second test:** A reader should understand **what problem**, **what method**, **what evidence**, **what result/status**, **what transferable lesson**, and **when to revisit** without reading the whole case.


---

# Advanced Quality Controls — Choose the Relevant Case Type

These are **conditional extensions**, not mandatory bureaucracy for every story. They help answer whether the case can be **replicated, challenged and used in a real decision**.

| Case type | Additional minimum evidence | Common mistake to avoid |
|---|---|---|
| **Business analytics / optimization** | Decision threshold, time window, baseline, confound check, scenario or sensitivity test, downstream action owner | Attributing all before/after movement to the intervention |
| **Predictive ML** | Population and label definitions, split strategy, calibration/subgroup check, threshold economics, drift plan | Reporting accuracy without base rate or decision consequence |
| **LLM, RAG, agentic** | Evaluation dataset/rubric, grounding and tool safety, prompts/models versions, attack/permission tests, cost per task, human escalation | Replacing end-to-end task success with only model or retrieval scores |
| **Streaming / platform** | Data correctness, latency SLO, recovery/load tests, replay/idempotency semantics, capacity and failure boundaries | Reporting throughput but omitting recovery behavior or data loss |
| **Post-mortem / incident** | Timestamped impact timeline, confirmed causes, detection/containment/restore metrics, action owners, prevention test | Confusing a plausible cause with an established root cause |

## End-to-end traceability check

Follow one decision chain in both directions:

```text
Business question (§3)
   -> decision rule and constraints (§3–4)
   -> data + reproducible method (§5)
   -> competing explanations and robustness (§6)
   -> evidence and pass/fail test (§9)
   -> approved action or no-action status (§10)
   -> measured result or explicit not-measured (§11)
   -> bounded transferable principle (§13)
   -> linked comparable cases and open questions (§14)
   -> quick reference with tags and review trigger (§15)
```

**Publication rule:** If any arrow is unsupported, state that break explicitly rather than writing a seamless narrative that implies proof.

## Controlled pattern vocabulary — starter example for INDEX

Keep the active collection near **15–20 canonical pattern tags**; the following are **illustrative starting points**, not universal standards:

`saturation-curve`, `diminishing-returns`, `budget-reallocation`, `causal-confounding`, `data-leakage`, `label-delay`, `distribution-shift`, `evaluation-mismatch`, `rag-vs-long-context`, `agentic-blast-radius`, `human-in-the-loop`, `stream-backpressure`, `partition-skew`, `checkpoint-recovery`, `cost-latency-tradeoff`, `rollout-regression`, `model-monitoring`, `incident-postmortem`.

A good tag definition includes: **name, meaning, diagnostic signature, when it applies, when it does not, example cases, and related/contrasting tags**. Map synonyms to one canonical tag. Do not let the index expand without review.

---

# Improvement Integration Crosswalk

The original ten improvements are **preserved below** for historical/source fidelity, but their operational requirements have been embedded in the 15 sections above.

| Original improvement | Where it is operationalized | Concrete acceptance evidence |
|---|---|---|
| 1. Lead with paradox/surprise | §1, §6, §7 | Supported surprise or decisive result appears early, with evidence and qualifications |
| 2. Show the wrong path | §5, §6, §12 | Rejected/tried approach, observed result, reason for abandoning it |
| 3. Assumptions as a checklist | §4 | Checkable register, invalidating assumptions, fallback, verification status |
| 4. Controlled reusable tags | §1, §13, §15; INDEX | Canonical pattern, precise boundaries, cross-domain comparability |
| 5. What remains open | §7, §9, §14 | 2–4 unresolved questions and proposed evidence to resolve them |
| 6. Real vs illustrative data | Metadata, §5–7, §9, §11, §15 | Every key number has an evidence type and provenance or explicit disclaimer |
| 7. Lesson before the case | §13 (draft first, revise last) | Principle treated as a hypothesis, updated when evidence disagrees |
| 8. Architecture diagrams | §8, conditional systems gate | Input-to-output diagram; offline/online paths when different |
| 9. Version and date | Case metadata, §2, §8, §15 | Version, reviewed-on date, status and concrete review trigger |
| 10. Write INDEX early | §1, §13–15; INDEX pattern vocabulary | Case backlinks, pattern taxonomy, theory cross-references |

---

# How to Improve Your Case Studies Further

These are concrete, actionable improvements. Apply them in order — the earlier ones have higher leverage.

## Improvement 1: Lead With the Paradox or Surprise

**The problem:** Most case studies are written in the order things happened. This buries the most interesting part.

**How to fix it:** Start the walkthrough (Section 6) with the most counter-intuitive finding. Then explain how you got there. Example: "The analysis found that the highest-called HCPs were the ones where additional calls had the least impact — the opposite of where the team was spending."

**Why it works:** Readers remember cases that challenged their assumptions, not cases that confirmed them.

---

## Improvement 2: Show the Wrong Path Briefly

**The problem:** Most case studies only show what worked. This makes them feel like hindsight.

**How to fix it:** In Section 6, add a paragraph called **"What we tried first and why it didn't work."** One paragraph. Name the method, the result, and the reason it was abandoned.

**Why it works:** The wrong path teaches as much as the right one, and it makes the case study credible — real projects hit dead ends.

---

## Improvement 3: Make Assumptions a Checklist, Not a Paragraph

**The problem:** Assumptions listed in prose are skipped by readers. When conditions change, nobody re-checks them.

**How to fix it:** Format every assumption in Section 4 as a table with three columns: Assumption, Type (Data / Infrastructure / Business / Regulatory), Impact if Wrong. This makes the case study self-auditing — a future reader can check each row against their situation.

---

## Improvement 4: Tag Every Case with a Reusable Pattern Name

**The problem:** A collection of 20 case studies is hard to search mentally if each one is filed only by domain.

**How to fix it:** In Section 15 (Quick Reference), add a Tags field. Use a small controlled vocabulary of pattern names (e.g., `saturation-curve`, `post-mortem`, `reallocation`, `rag-to-full-context`, `data-leakage`, `diminishing-returns`, `agentic-blast-radius`). Keep the vocabulary to 15–20 terms max and define them once in the INDEX file.

**Why it works:** Two cases in different domains with the same pattern tag become immediately comparable. This is how the KDA Pharma case collection ("promotion [text obscured in screenshot] shift across lifecycle") and the AI Frontier 2026 collection (structural shift tags S1–S10) stay navigable.

---

## Improvement 5: Add a "What This Case Leaves Open" Section

**The problem:** A case study that answers everything teaches nothing — real cases always have residual uncertainty.

**How to fix it:** End Section 14 with a bulleted list of 2–4 open questions this case did NOT answer, and what evidence would resolve them. Example: "This analysis can't distinguish whether email's rising response curve reflects genuine headroom or list-quality effects in the low-exposure segment. A geo-holdout test on a randomly selected 10% of territory would resolve this."

**Why it works:** It signals intellectual honesty, and it generates the agenda for the next case study.

---

## Improvement 6: Distinguish Illustrative from Real Data — Always

**The problem:** Composite/illustrative numbers mixed with real reported results erode credibility when the case is shared externally.

**How to fix it:** Add one of two standard disclaimers, never skip them:

- **For illustrative cases:** *"All figures in this case are illustrative composites built to be directionally realistic for this category — they are not any single organization's actual data. The pattern of driver behavior is what matters."*
- **For real-world cases:** *"Figures are sourced as cited. Verify against primary sources before external use — results can change as deployments scale or projects evolve."*

---

## Improvement 7: Write the Transferable Lesson Before You Write the Case

**The problem:** If you write the case first and extract the lesson at the end, the lesson is usually too specific to the domain and too hard to reuse.

**How to fix it:** Draft Section 13 (Transferable Lesson) before you write Sections 6–12. Use it as a compass. If your walkthrough doesn't support the lesson you wanted to write, either the lesson is wrong or the case is the wrong vehicle for it.

---

## Improvement 8: Use Architecture Diagrams for Every Systems Case

**The problem:** Prose descriptions of system architecture are almost never retained by readers. ASCII diagrams are retained.

**How to fix it:** Every case study that involves a built system (ML pipeline, streaming system, agentic framework) must include at least one ASCII architecture diagram in Section 8. The minimum bar: a left-to-right flow from input to output, with component names. Add a second diagram for the online/real-time path if it differs substantially from the offline path.

---

## Improvement 9: Version and Date Every Case

**The problem:** Case studies go stale. A result that was true in 2024 may be wrong in 2026 (e.g., full-context ingestion vs. RAG economics).

**How to fix it:** Add a frontmatter block at the top of every case study file:

> **Case version:** 1.0 | **Written:** [Month Year] | **Last reviewed:** [Month Year]
>
> **Status:** [Pilot / Production / Scaled / Superseded by case X]
>
> **Review trigger:** Re-review if [specific condition, e.g., "context window pricing changes by >50%", "new biosimilar approved in this TA class", "Kafka version upgrade to 4.x"]

---

## Improvement 10: Write an INDEX First, Not Last

**The problem:** A collection of case studies without an index is unusable after more than 5–6 cases.

**How to fix it:** Before writing any case in a new domain, create an INDEX file with:

- One-line description of each planned case
- The pattern tag(s) for each
- The structural shift or driver it addresses (domain-specific equivalent of the S1–S10 shifts used in the AI Frontier collection)
- Cross-reference map to theory material

The INDEX is the instrument panel; individual cases are the data.

---

# Minimum Viable Case Study Checklist

Before publishing any case study, verify these 10 items:

- [ ] One-line framing filled in with input/output/task-category specified
- [ ] Business question quoted verbatim (or close to it), not paraphrased into technical language
- [ ] Assumptions table complete with "Impact if Wrong" column
- [ ] Data specification table present (unit of analysis, sample size, outcome variable, exclusions)
- [ ] At least one sensitivity/robustness check in the walkthrough
- [ ] Finding stated in one paragraph, no new material introduced
- [ ] Results distinguished as real (sourced) or illustrative (stated explicitly)
- [ ] Transferable lesson named and bounded (when to apply, when NOT to apply)
- [ ] Quick Reference Card (six-field table) complete
- [ ] Case cross-referenced to at least one other case or theory module it extends or contrasts

---

*Template version: 1.0 | Created: November 2026 | Based on: AI_Systems_2026, Pharma_Agentic, KDA_Solutions, Streams_Kafka_Flink_Redpanda case study collections*

---

## Enhanced Publication Gate (v2 additions)

Use the original ten checks above **plus these additional review checks**. Mark `N/A` only with a short reason.

- [ ] Individual case frontmatter includes version, ISO date, status, source index and **review trigger**
- [ ] One decision chain connects the question, metric, baseline, validation, action and reported result
- [ ] A primary **pass/fail threshold** and relevant guardrails were specified before declaring success
- [ ] Data provenance, cut-off, exclusions, sampling and leakage controls are documented
- [ ] Failed/rejected alternatives are either described with evidence or explicitly not tested
- [ ] Every key figure is labeled **observed / published / estimated / simulated / illustrative / not measured**
- [ ] Business impact is not attributed causally without a defensible identification strategy
- [ ] Important uncertainty, selection bias, generalization limits and tradeoffs are visible
- [ ] System case: architecture, SLOs, failure boundaries, deploy/rollback and security controls documented, or `N/A` justified
- [ ] Incident case: confirmed vs hypothesized causes distinguished; corrective actions have owner and verification
- [ ] Transferable lesson drafted as a hypothesis, then revised to match evidence and non-applicability conditions
- [ ] Open questions specify what evidence would resolve them, not merely that “more study is needed”
- [ ] Links and references resolve; vendor claims are distinguished from independently observed results
- [ ] Case tags are canonical and INDEX backlinks updated; the summary card contains **six original fields** plus dated status

### Quality scoring rubric — practical, not a scientific rating

Score each dimension **0 (missing), 1 (partial), or 2 (usable and substantiated)**:

| Dimension | Max | What deserves 2/2 |
|---|---:|---|
| Problem and decision framing | 2 | Clear stakeholder, baseline, decision, threshold and business scope |
| Data and method reproducibility | 2 | Provenance, exclusions, splits/configuration and method alternatives traceable |
| Evaluation rigor | 2 | Valid comparator, robustness/confound checks, uncertainty and pass/fail rule |
| Architecture/operations *if relevant* | 2 | Clear diagram, SLO, failure/rollback and ownership |
| Result honesty and provenance | 2 | Observed vs estimated vs illustrative separated; claims bounded |
| Transferability and retrieval | 2 | Named principle, when-not-to-apply, links, open questions and tags |

If architecture is not applicable, use a **10-point denominator**; otherwise use **12**. This rubric is an **editorial drafting aid**, not a formal certification. A high total does **not** override an unsourced result, fabricated deployment or critical safety omission.

---

*Integrated edition v2.0 | Prepared November 2026 | Underlying screenshot transcription v1 retained for traceability | Editorial enhancements: decision traceability, evidence taxonomy, robust evaluation, operations, retrieval and publication gates.*
