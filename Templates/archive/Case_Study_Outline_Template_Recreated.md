# Case Study Writing Guide & Outline Template

> **How to use this document:** Follow the structure below for every case study you write. Each section heading is a prompt — answer it in that section. Sections marked `[REQUIRED]` must always be present. Sections marked `[OPTIONAL]` add depth; include when the information exists.
>
> This outline was distilled from four reference collections:
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

---

## 7. The Finding `[REQUIRED]`

**Purpose:** State the answer in one paragraph. No jargon. A non-technical reader should understand it.

**Format:**

> "[One paragraph. Answer the business question directly. State the size/direction of the main finding. Name any qualifications or conditions on the finding. Do NOT introduce new material here — only summarize what the walkthrough showed.]"

**Improvement tip:** If your finding paragraph is longer than 5 sentences, it is too complex. Split it, or move material back to the walkthrough.

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

---

## 10. Action Taken `[REQUIRED]`

**Purpose:** State what the organization actually did as a result of this finding. This is the bridge from analysis to impact.

**Fill in:**

- **Decision made:** Exactly what was changed, built, or deployed
- **Who made the decision:** Role and level (e.g., "Brand VP approved reallocation at annual planning")
- **Timeline:** How long from finding to action?
- **Rollout approach:** Phased, A/B tested, full cutover, pilot first?
- **What was NOT done (and why):** Often as informative as what was done

---

## 11. Outcome & Results `[REQUIRED]`

**Purpose:** Report what actually happened after the action was taken.

**Fill in:**

- **Primary metric result:** (e.g., "TRx held flat while spend dropped 8%", "P95 latency cut from 400ms to 95ms", "SQL query accuracy: 95%")
- **Secondary metric results:** Any unexpected movements, positive or negative
- **Adoption/scale reached:** Number of users, volume of transactions, geographic scope
- **Timeline to impact:** How long until results were visible?
- **Status at time of writing:** Pilot / production / scaled / cancelled

**Format sourced results as a table (for real-world cases):**

| Metric | Before | After | Source |
|---|---|---|---|
| Manual review time | Baseline | -50% | [Roche — Bioprocess Online] |
| Stock disruption prediction | Reactive | 80% predicted pre-event | [Sanofi — Aily Labs] |

**Rule:** If results are illustrative (not real data), say so explicitly. Use the phrase: *"All figures are illustrative composites built to be directionally realistic — not any single organization's actual data."*

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

---

## 14. Connections & Cross-References `[OPTIONAL but strongly recommended]`

**Purpose:** Link this case to others in the collection and to the theoretical material it draws on. This turns a collection of isolated cases into a connected knowledge graph.

**Fill in:**

- **This case extends:** `[Prior case or theory sheet]` — by adding `[what new element]`
- **Compare and contrast with:** `[Related case]` — same pattern, different domain/outcome
- **Theory/methodology applied:** `[Sheet or module reference]` — which concept from the companion theory material was used here
- **What this case leaves open:** Questions or scenarios not covered here that a follow-up case should address

---

## 15. Quick Reference Card `[REQUIRED]`

**Purpose:** A 5-row summary table that lets a reader scan the case in under 30 seconds.

| Field | Value |
|---|---|
| Domain | |
| Problem type | |
| Method | |
| Primary result | |
| Key lesson | |
| Tags | |

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
- [ ] Quick Reference Card (5-row table) complete
- [ ] Case cross-referenced to at least one other case or theory module it extends or contrasts

---

*Template version: 1.0 | Created: November 2026 | Based on: AI_Systems_2026, Pharma_Agentic, KDA_Solutions, Streams_Kafka_Flink_Redpanda case study collections*
