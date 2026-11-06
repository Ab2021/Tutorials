# Cheat Sheet Content Master Template and Authoring Instructions

**Version:** 1.0 | **Prepared:** 2026-11-06 | **Purpose:** Write comprehensive, technically sound cheat sheets for industry decisions, production troubleshooting, and demanding interviews.

This is an authoring standard and reusable template, not a completed cheat-sheet collection. It consolidates the supplied v3 package while preserving **Parts A–E**, the **12-section individual-file outline**, all **ten improvement tips**, the **11-section INDEX**, the collection planning worksheet, and the final Common Issues convention. Add detail within this structure; do not replace it with a glossary or an unstructured list of facts.

**Companion:** [Case Study Content Master Template](CASE_STUDY_CONTENT_MASTER_TEMPLATE.md). A cheat sheet teaches a reusable method, calculation, diagnostic, or selection decision. A case study shows how such reasoning changes a decision under a particular workflow and evidence base. Cross-link the two.

**Navigation:** [Collection rules and INDEX](#part-a--collection-level-rules) | [12-section content template](#part-b--per-file-outline-preserve-sections-112) | [Ten improvements](#part-c--ten-improvement-tips-as-enforceable-instructions) | [Completion checks](#part-d--completion-and-quality-checklist) | [Planning worksheet and workflow](#part-e--collection-skeleton-and-authoring-workflow).

## How to use this document

1. Plan the topic ladder and provisional INDEX using Part A and Part E.
2. Select a focused topic, its reader task, and the relevant domain/technical profile.
3. Copy the metadata and all 12 headings from Part B into a new individual sheet.
4. Draft the Section 12 card first as a plan; build and verify Sections 1–11 around it.
5. Complete the interview answers and troubleshooting evidence before finalizing the card.
6. Put the card last in each individual file. Finalize the collection INDEX after checking actual files and links.

**Requirement labels:** `MUST` is required for this library; `CONDITIONAL` becomes required when applicable; `RECOMMENDED` adds useful depth. These are editorial requirements, not external standards. Retain all 12 numbered headings. Write `N/A — [specific reason]` only for an inapplicable conditional section. Use `unknown`, `not verified`, or `not measured` for missing evidence instead.

**Depth rule:** Comprehensive means a reader can define, derive, compute, select, diagnose, scale, and defend the material within the declared scope. It does not mean placing an entire discipline into one file. Prefer a connected collection of focused sheets to one oversized survey.

## Part A — Collection-Level Rules

### A1. Naming convention and artifact roles

Use the archived convention for a new standalone collection:

```text
00_INDEX.md
01_<Focused_Topic_Name>.md
02_<Focused_Topic_Name>.md
...
NN_<N>_Common_Issues.md
```

- Use underscores and descriptive topic names. Order by prerequisites and increasing depth, not alphabetically.
- Use consistent zero padding; choose three digits at collection creation if more than 99 files are genuinely expected. Do not casually rename published files and break incoming links.
- `NN` is the final file's sequence number; `<N>` is the actual number of completed issues, such as `09_50_Common_Issues.md`. A title promising 50 issues must contain 50 substantive issues.
- Keep one canonical index. If an existing library uses `INDEX.md`, preserve that convention and make any `00_INDEX.md` navigation alias explicit; do not maintain conflicting catalogs.
- A planned topic or an empty file is not a completed sheet. Show `planned / draft / reviewed / published / needs-review / deprecated` separately from technical evidence.
- A specialized final Common Issues file is a troubleshooting artifact; its adaptation to the 12 sections is defined in A10. It is the final numbered file, not an extra unindexed appendix.

### A2. Topic ladder and prerequisite discipline

Plan three levels before writing File 01. Give each sheet a measurable outcome and an explicit dependency chain.

| Level | Content | Expected reader capability | Boundary |
|---|---|---|---|
| 1 — Foundations | Definitions, notation, assumptions, basic mechanism | Define and perform a simple checked calculation/trace | Do not require an unexplained advanced method |
| 2 — Core Methods | Derivations, algorithms, variants, evaluation, selection | Apply and compare methods under stated conditions | Do not omit prerequisite reasoning |
| 3 — Advanced / Production | Scale, composition, diagnostics, operations, failure boundaries | Defend choices, diagnose ambiguity, adapt to changed constraints | Do not equate advanced terminology with production readiness |
| Final — Common Issues | Distinct, actionable troubleshooting entries | Diagnose, mitigate, verify, and prevent recurring failures | Placeholder rows do not count as issues |

Keep mandatory prerequisites acyclic and available before the dependent file in the chosen learning order. Optional “go deeper” links may point forward; label them as optional. If external prerequisites are assumed, state the skills and provide an available reference rather than silently creating a dependency.

Split a sheet when it has several independent outcomes, incompatible notation, different evaluation regimes, or enough new definitions to obscure the main decision. An authoring reference range is roughly 800–1,800 words of explanatory prose per focused sheet, excluding tables and derivations; it is not a completeness limit. For a deeper sheet, retain scan-friendly sections and split genuinely separate topics rather than compressing critical reasoning.

### A3. Mathematical notation and terminology standard

Choose **one canonical display-math style** per collection: LaTeX blocks or fenced plain-text equations. Choose for the actual renderer and check it there. Code blocks and inline identifiers may coexist; the rule prevents duplicate/conflicting mathematical presentations, not legitimate implementation examples.

Maintain the master glossary in the INDEX:

| Symbol/term | Meaning | Units | Shape/type | Domain/valid range | First use and other files |
|---|---|---|---|---|---|
| `[Symbol]` | `[Unambiguous meaning]` | `[Unit or dimensionless]` | `[Scalar/vector/matrix/event/etc.]` | `[Bounds/context]` | `[Actual locations]` |

Define symbols at first use even when they appear in the master glossary. Distinguish sample/population, conditional/unconditional, count/rate, event/processing time, scalar/vector, and exact/approximate values. Qualify overloaded symbols explicitly; do not let `p` silently change from probability to feature count.

For systems and DSA topics, define state, operations, keys, input sizes, invariants, and output contracts as rigorously as mathematical symbols. For robotics, define coordinate frames, units, timestamp semantics, calibration conventions, and control-cycle terms.

### A4. Formula block anatomy and reasoning requirements

Every substantive formula must be understandable independently of nearby prose. Preserve the original **five-part anatomy**, with validity and derivation introduced before use:

1. **Validity first:** State the conditions under which the formula may be used.
2. **Derivation:** Show the definition and transformations leading to it; name the identity, independence assumption, approximation, or theorem used.
3. **Five-part block:** `FORMULA -> WHERE -> CONSTRAINT -> NUMERICAL EXAMPLE -> INTERPRETATION`.
4. **Boundary check:** State a limit, counterexample, numerical hazard, or invalidating case.

```text
Validity before use: [conditions, domain, and prohibited application].
Derivation: [starting definition] -> [justified step] -> [result].

[1] FORMULA: [canonical expression; stable equation ID if referenced].
[2] WHERE: [all symbols, units, shapes, and indexing conventions].
[3] CONSTRAINT: [assumptions and validity bounds, repeated for standalone use].
[4] NUMERICAL EXAMPLE: [values -> substitution -> result with units].
[5] INTERPRETATION: [what the number changes in a decision].

Boundary/counterexample: [failure if a condition is violated].
```

Use this full treatment for each new substantive result, not a repeated five-part block for every line of algebra. Cross-reference a previous block when reusing the same equation, but ensure a copied reference card still names its critical condition.

Distinguish `=` from `approximately equal`, display rounding conventions, check conditioning and dimensions, and state estimator/parameter uncertainty where relevant. If a full proof is outside scope, give a derivation sketch and an authoritative source; do not label a sketch a proof.

For non-mathematical systems topics, retain Section 4's heading and use a precise **mechanism/invariant block**: preconditions, state transition or algorithm, justification, worked trace, and failure boundary. Include relevant sizing or correctness equations when they exist; do not manufacture formulas to decorate a process description.

### A5. Table types and selection discipline

| Table type | Required content | Quality test |
|---|---|---|
| Definition/notation | Term, formal meaning, units/shape, domain, nonexample | No ambiguous or undefined symbol |
| Comparison | Option, assumptions, metric/complexity, use when, avoid when | All options evaluated under comparable conditions |
| Decision guide | Condition, priority, choice, reason, disqualifier, fallback | Conflicting constraints have a resolution |
| Failure/issue | Symptom, cause, discriminating diagnostic, fix, verification, provenance | Reader can distinguish competing causes |
| Complexity/cost | Operation, input sizes, time, space, I/O/communication, assumptions | Bounds are not confused with benchmarks |
| Benchmark | Workload, baseline, setup, metric, result, uncertainty, source/date | No unsupported superiority claim |
| Tradeoff | Option, benefit, downside, boundary, revisit trigger | “Best” is scoped to an objective and constraints |

Use tables when they improve comparison. Explain the mechanism and decision consequence in prose; do not repeat every table cell in a paragraph. Every recommendation needs an applicable condition, a failure it prevents, and a boundary.

### A6. Collection INDEX — plan first, finalize last

Create a provisional INDEX and topic ladder before drafting. Populate the glossary, lookup, and statuses as files develop. Finalize the INDEX last after verifying real filenames, anchors, reading paths, and source records. This reconciles the archive's early planning worksheet with its “write INDEX last” instruction.

Preserve all **11 INDEX sections**, in this order:

| No. | Section | Detailed authoring instructions |
|---:|---|---|
| 1 | One-paragraph overview | Domain, reader, scope, practical decisions, assumed foundations, definition of readiness |
| 2 | Topic ladder | Level, file number, linked topic, testable outcome, prerequisites, key formula/skill, status if useful |
| 3 | Quick topic lookup | Keyword or real question, file, exact section, use condition, failure warning |
| 4 | Reading paths — three options | Beginner, Practitioner, Advanced; sequence, objective, prerequisite assumptions, optional extensions |
| 5 | Prerequisite skills | Math/theory, domain, systems/tools; concrete skills and references rather than vague expertise |
| 6 | What this collection does NOT cover | Related exclusions and actual handoffs; planned gaps labeled as planned |
| 7 | Master symbol glossary | Meaning, units, shape, domain, first appearance, file links, overload warnings |
| 8 | Quick Reference Card directory | File/card anchor, core equation/rule, use condition, critical assumption, avoid condition |
| 9 | Common Issues locator | Final troubleshooting file, actual issue count, issue ID/category/symptom/diagnostic/anchor |
| 10 | Empirical sources, versions and collection maintenance | Claim/source registry, versions, review dates, triggers, per-file evidence/status/open issues |
| 11 | Collection acceptance checklist | Scope, files, structure, math, evidence, navigation, troubleshooting, and review completion |

Retain the original Beginner path of roughly 3–4 files, Practitioner path of roughly 6–7 files, and Advanced path through all relevant files as planning defaults. Adapt honestly to collection size; a four-file collection cannot have seven distinct practitioner files. Do not skip a required prerequisite to force a short reading path. Add interview or incident-response paths as optional additions, without replacing the three original paths.

**Planning descriptions:** For an unwritten sheet, specify its outcome, derivations/mechanisms, example, selection question, failure themes, and interview skills. Do not call a topic complete because its title appears in the INDEX. Use plain text for nonexistent file targets or clearly link a real planning brief as a brief.

### A7. Metadata, evidence, sources, and maintenance

Place a small record at the top of each individual sheet:

```yaml
---
collection: "[Collection slug]"
file_id: "01"
topic: "[Focused topic]"
level: 1
status: "draft"
version: "1.0"
created: "YYYY-MM-DD"
reviewed: null # ISO date only after a substantive review.
review_trigger: "[Specific source, dependency, workload, or assumption change]"
notation_style: "plain-text" # plain-text | latex
prerequisites: [] # Actual file paths or separately stated external skills.
pattern_tags: [] # Controlled collection tags, if used.
source_status: "OPEN" # Summary only; label individual claims separately.
owner_role: "[Role/team]"
source_index: "[Actual relative INDEX path]"
---
```

Use the archived evidence labels at the point of use:

| Label | Meaning | Required support |
|---|---|---|
| `DERIVED` | Mathematical/logical consequence of stated premises | Derivation or proof sketch, assumptions, authoritative theorem source if used |
| `VERIFIED` | Author actually checked the calculation, trace, or experiment | Inputs, method, output, environment/run reference, scope |
| `SOURCED` | Attributable external statement or measurement | Direct primary reference, date/version, exact supported claim |
| `ILLUSTRATIVE` | Constructed teaching example | Explicit label, stated inputs, no real-company attribution |
| `HEURISTIC` | Experience-based or approximate rule | Rationale, applicability, limitations, validation plan |
| `OPEN` | Unresolved or unavailable support | Specific gap and evidence needed to resolve it |

These labels are not mutually exclusive for an entire file. An illustrative calculation may be arithmetically verified; that does not make its workload a measured production fact. A sourced benchmark has not automatically been reproduced by the author. Keep claim-level distinctions.

For each empirical or unstable claim, record `source ID | primary title/link | claim | publication/event date | version | workload/setup | reviewed on | limitation | review trigger`. Verify tool/API behavior, model availability, prices, hardware specifications, and applicable rules when authoring. Historical examples must carry their historical context.

Cite near the claim. Do not use a blog's mention of a paper as if you had reviewed the original experiment. If no comparable benchmark is available, write `NOT MEASURED` or `NOT VERIFIED` and provide a test protocol. Do not invent empirical evidence to satisfy a template row.

### A8. Industry relevance and technical profiles

Choose profiles by the reader's actual task. Integrate the relevant prompts into Sections 3–11. All topics still use the same 12-section sequence.

| Profile | Required theoretical/mechanistic depth | Production and interview emphasis |
|---|---|---|
| Statistics/experimentation | Estimand, sampling, estimator, assumptions, uncertainty, test design | Confounding, multiple testing, dependence, sample-ratio issues, practical impact |
| ML and fraud | Objective/loss, features, labels, split, calibration, decision threshold | Leakage, imbalance, delayed labels, review capacity, drift, business costs |
| Forecasting | Process assumptions, temporal features, horizons, backtesting, intervals | Intermittent/censored demand, seasonality, exogenous availability, hierarchy, replenishment consequence |
| Optimization | Variables, objective, constraints, feasibility, solver/heuristic, optimality concept | Infeasibility, sensitivity, robust decisions, runtime limits, operator override |
| DSA | Input/output, invariant, correctness sketch, complexity, operation model | Adversarial cases, overflow, mutation, amortized/worst-case difference, scale bottlenecks |
| Search/recommendation/graphs | Candidate/index/graph construction, ranking or propagation, time boundaries | Exposure/position bias, cold start, graph leakage, freshness, relevance versus business utility |
| LLM/RAG/agents | Context/retrieval, prompting, state, tool contracts, training/adaptation where relevant | Groundedness, task completion, contamination, injection, permissions, idempotency, approval, stopping, token/cost budgets |
| Fine-tuning/distributed training | Objective, gradients, optimizer, precision, adapters, parallelism, memory/checkpoints | Convergence, instability, communication, resumption, throughput versus quality, reproducible evaluation |
| NLP/speech/vision/multimodal | Representation, alignment, modality-specific preprocessing, labels, loss, metrics | Language/device/environment slices, timestamp/coordinate errors, robustness, domain shift |
| Data engineering/streaming | Schema, event time, ordering, partitioning, state, delivery/transaction semantics | Late/duplicate events, skew, backpressure, checkpoint/recovery, replay and sink correctness |
| Distributed systems/MLOps | Consistency, concurrency, state ownership, deployment unit, feature/artifact lifecycle | Partial failure, retry storms, skew, stale data, rollout compatibility, monitoring and recovery |
| CUDA/compilers/hardware | Layouts, memory hierarchy, scheduling, synchronization, precision, generated/executed code | Correctness, race/numeric checks, profiler evidence, bandwidth/compute bounds, hardware/version portability |
| Edge/embedded/robotics | Device constraints, timing, sensors, calibration, frames, estimation/planning/control interfaces | Jitter/deadlines, power/thermal limits, dropout, network loss, watchdog/safe state, simulation and hardware-in-loop |
| Business-domain analytics | Workflow, entity/decision grain, objective, incentives, domain terminology | Feedback loops, operational adoption, unit economics, cost of error, intervention validity |

Business context must change the explanation. For example:

| Domain | Useful grounding | Important boundary |
|---|---|---|
| Supply chain | Lead-time demand, stockout censoring, inventory/replenishment, warehouse/transport constraints | Forecast accuracy alone does not define service/cost performance |
| Omnichannel retail | POS/online demand, inventory reservation, promotions, loyalty, substitutions, pickup/delivery | Channel or store-level effects may not transfer to other fulfillment modes |
| E-commerce | Catalog, retrieval/ranking, seller/customer incentives, payments, returns, contribution margin | Click gains may conflict with availability, profitability, or long-term trust |
| Quick commerce/food delivery | Assortment, perishability, dispatch, ETA, rider/store capacity, cancellation | Decisions interact through supply, incentives, and geography |
| Streaming media | Content metadata/rights, discovery, churn, subscription/ad metrics, live delivery | Engagement, playback quality, content availability, and retention are distinct |
| Finance/insurance/healthcare | Delayed outcomes, review/adjudication, entity linkage, cost-sensitive errors | Expert review, data access, auditability, and outcome validity must be explicit |
| Manufacturing/robotics | Inspection, maintenance, picking, calibration, line cycle, physical environment | Offline model accuracy does not establish safe closed-loop operation |

Walmart, Tesco, Target, Flipkart, Zepto, Swiggy, Disney, Warner Bros., or Hotstar may be research contexts for a sourced case example. Without primary evidence, label the scenario hypothetical and avoid asserting their proprietary data, models, traffic, or results. Existing interview examples are references only; do not copy their figures into empirical benchmark tables.

### A9. Source precision and archived technical cautions

Preserve the archive's learning intent while correcting oversimplified examples before reuse:

- Do not use its unattributed HumanEval percentages as verified model/workflow comparisons. Require an original source and comparable evaluation protocol.
- Do not reuse one roofline cutoff as a universal property of a GPU. Specify device, precision, compute capability used, attainable bandwidth, and actual workload traffic.
- Do not present an illustrative OLS pass as the full fitting complexity of every ARIMA estimator. Name the estimation algorithm, iterations, and relevant dimensions.
- Distinguish prediction-error variance from variance of a conditional point forecast; define conditioning and treatment of estimated parameters.
- State full essential hypotheses for Wold decomposition, Cramér–Rao bounds, universal approximation, or any theorem used. A mnemonic is not a precise theorem statement.
- Redraw schematic geometry if it does not correctly express the claim. A labeled but mathematically wrong sketch still fails review.
- Recompute arithmetic and rounding. Do not preserve a photographed numerical mistake for stylistic fidelity in a new teaching file.

The official [scikit-learn pitfalls documentation](https://scikit-learn.org/stable/common_pitfalls.html) supports instruction on training-only preprocessing and leakage. [Google's SLO workbook](https://sre.google/workbook/implementing-slos/) supports user-centered reliability metric design. The [ROS 2 real-time tutorial source](https://github.com/ros2/ros2_documentation/blob/jazzy/source/Tutorials/Demos/Real-Time-Programming.rst) provides timing and nondeterminism examples. Use these for the specific methods they discuss, not as validation of an illustrative workload or an application's safety. References were consulted on 2026-11-06; pin relevant versions and recheck before relying on current behavior.

### A10. Final Common Issues artifact — preserve actionability and structure

Create at least **25 distinct, complete issues** for a finished collection; use **50** for an extended comprehensive collection when there are 50 meaningful issues. Do not duplicate symptoms with cosmetic wording or invent incidents to reach a number. If the target is not reached, leave the artifact visibly incomplete.

Preserve the archived categories: `VIOLATED_ASSUMPTION / NUMERICAL_EDGE / SCALE_FAILURE / CONCEPT_CONFUSION`. Record provenance separately: `OBSERVED` (documented actual failure), `REPRODUCED` (author-confirmed test), or `HYPOTHETICAL` (teaching scenario). Category is not evidence maturity. Add a subsystem tag when useful without silently redefining the four source categories.

For this library, keep the same 12 headings in the final troubleshooting file. Adapt them explicitly:

| Section | Troubleshooting adaptation |
|---|---|
| 1 | Scope, environment, reader task, issue count, provenance policy |
| 2 | Diagnostic map from symptom to subsystem/category |
| 3 | Diagnostic terms, signal definitions, severity and evidence terms |
| 4 | Core invariants or diagnostic rules and why they distinguish causes |
| 5 | One end-to-end diagnostic walkthrough with a checked result |
| 6 | Causal/diagnostic sketch when useful; otherwise justified N/A |
| 7 | Mitigation/diagnostic selection and disqualifiers |
| 8 | Complete issue index and detailed entries, with stable ISSUE IDs |
| 9 | Diagnostic overhead, scaling, and operational cost where applicable |
| 10 | Applicable guarantees/theorems or justified N/A |
| 11 | Diagnostic interview questions with checked answers |
| 12 | Final troubleshooting Quick Reference Card |

Under Section 8, retain the archived issue table:

| ID | Category | Symptom | Root cause | Diagnostic / expected signal | Fix | Provenance | Related file |
|---|---|---|---|---|---|---|---|
| `ISSUE-001` | `[One of four]` | `[Observable]` | `[Evidence/confidence]` | `[Check + expected output]` | `[Bounded action]` | `[Label + artifact]` | `[Actual link]` |

Expand each issue with `trigger/environment -> symptom -> competing causes -> discriminating diagnostic -> root-cause evidence -> immediate mitigation -> correction -> prevention -> verification -> limitations -> related sheet`. Record risks or side effects of a mitigation and an escalation path when applicable. Place the issue index, references, and checks before Section 12, keeping its card last.

## Part B — Per-File Outline: preserve Sections 1–12

**Content outline:** [1. Header](#section-1--file-header), [2. Concept map](#section-2--concept-map--mental-model), [3. Definitions](#section-3--foundational-definitions), [4. Derivations](#section-4--core-formulas-with-full-derivation), [5. Worked example](#section-5--worked-numerical-example), [6. Visual intuition](#section-6--visual--geometric-intuition), [7. Selection](#section-7--comparison--selection-table), [8. Failures](#section-8--failure-modes--edge-cases), [9. Complexity and cost](#section-9--complexity--cost-analysis), [10. Theorems](#section-10--key-theorems--results-reference), [11. Interview answers](#section-11--interview--exam-questions), [12. Final card](#section-12--quick-reference-card).

The following headings and prompts form the individual-sheet template. Replace prompts with technical content; retain numbering, sequence, and justified conditional applicability. End each substantive section with a one-line **Takeaway** that expresses a useful conclusion, not a restatement of its title. For Section 12, make the table itself the concluding content.

### Section 1 — File Header

**Required.** Establish scope, audience, prerequisites, and the task the reader can perform.

```markdown
# [Focused Topic Name]

> **Purpose:** [Specific calculation, derivation, selection, or diagnostic angle.]
> **Prerequisites:** [Actual earlier file references and concrete skills.]
> **Key outcome:** Given [inputs/conditions], you can [testable action/output].
> **This file does NOT cover:** [Two or three related exclusions and handoffs.]
> **Level:** [1 Foundations / 2 Core Methods / 3 Advanced-Production]
> **Version / reviewed:** [Version / actual review date or not reviewed]
> **Evidence:** [Summary labels; individual claims are labeled separately.]
> **Retrieval cue:** If you see [symptom/question], start at Section [number].
```

Explain the real decision the concept supports. A purpose line should add scope beyond the title: “select and calibrate a fraud threshold under review-capacity limits” is more useful than “understand fraud models.” State whether production values are measured, sourced, assumed, or illustrative.

**Must pass:** The outcome is testable, prerequisites are real, scope excludes adjacent topics explicitly, and evidence/date fields are honest.

### Section 2 — Concept Map / Mental Model

**Required and before the first substantive formula.** Give one-screen orientation using a labeled ASCII hierarchy, dataflow, state diagram, or concise comparison map. Use equation IDs rather than introducing unexplained formulas before the map.

```text
[Input / observed signal]
          |
          v
[Assumption / validity check] --fails--> [Alternative / collect evidence]
          |
        passes
          v
[Core mechanism / transformation] -> [Output / decision]
          |
          v
[Diagnostic / feedback signal]
```

Adapt the structure to the actual topic. Distinguish a dependency arrow, a dataflow arrow, a causal claim, and a control loop; they do not mean the same thing. Name the central invariant, what changes between variants, and the most important boundary.

If using a table, include `Concept | Core idea | Role/input-output | Use when | Boundary`. The map orients the reader; formal definitions and equations follow in Sections 3–4.

**Must pass:** A reader can explain the mechanism's parts and information flow without needing the later algebra.

### Section 3 — Foundational Definitions

**Required.** Define the terms needed to follow the sheet, normally about three to eight core terms. More terms may justify splitting the topic; do not omit necessary definitions to meet an arbitrary count.

For each term:

- **Definition:** Formal, unambiguous meaning.
- **Intuition:** Plain-language explanation or a bounded analogy.
- **Notation:** Symbol/type, units, dimensions, domain, indexing convention.
- **Example/nonexample:** A concrete instance and a nearby concept it should not be confused with.
- **Operational interpretation:** How it is observed, computed, or checked in a real workflow.

| Term/symbol | Formal meaning | Units / shape | Valid domain | Common confusion | INDEX glossary entry |
|---|---|---|---|---|---|
| `[Term]` | `[Definition]` | `[Unit/type]` | `[Range/context]` | `[Distinction]` | `[Actual location]` |

Distinguish necessary and sufficient conditions where applicable. Do not let an analogy replace the formal definition. Reconcile terminology with related sheets and the master glossary.

**Must pass:** Every later symbol and term is locally defined or an explicit prerequisite, and confusing near-neighbors are distinguished.

### Section 4 — Core Formulas with Full Derivation

**Required.** Use A4's five-part formula anatomy and the four preserved subsections below. For systems/process topics without a natural closed-form equation, use the mechanism/invariant adaptation rather than skipping the reasoning.

#### 4.1 Simplest / Base Case

State validity before use. Start from a definition or simple model, show a short derivation with justified transformations, then give the formula, symbol definitions, constraint, numerical substitution, and interpretation. Explain what the result lets the reader decide.

#### 4.2 General Case

Show exactly what changes: dimensions, parameters, noise, constraints, state, workload, or dependency structure. Derive or sketch the extension and demonstrate how it reduces to the base case. Identify added assumptions and computational consequences.

#### 4.3 Key Special Cases / Variants

| Variant | What changes | Assumptions | Use when | Avoid when | Link/equation ID |
|---|---|---|---|---|---|
| `[Variant]` | `[Equation/state/algorithm difference]` | `[Condition]` | `[Selection rule]` | `[Disqualifier]` | `[Reference]` |

Do not give several similar expressions without explaining their different domains or objectives. If only one case is meaningful, say why rather than creating artificial variants.

#### 4.4 Asymptotic / Limiting Behaviour and Counterexample

Check a relevant boundary: zero/large input, parameter boundary, singular matrix, empty set, heavy skew, delayed feedback, exhausted memory, unavailable dependency, or changed sampling regime. Explain what remains true and which assumption breaks.

| Result/invariant | Required assumption | Diagnostic | Violating example | Correct alternative |
|---|---|---|---|---|
| `[Equation/rule]` | `[Condition]` | `[Check]` | `[Counterexample]` | `[Bounded response]` |

For an algorithm, include input/output contract, pseudocode or operation trace, invariant, correctness argument, termination, and the operation-count model. For an API/framework topic, derive the state/retry/ordering behavior from the actual documented semantics, not a generic analogy.

**Must pass:** The reader can reproduce the main reasoning, identify its assumptions, and demonstrate a case where naive application fails.

### Section 5 — Worked Numerical Example

**Required.** Provide one end-to-end example connecting inputs to a decision. Include a small hand-checkable case and a production-scale interpretation. A constructed realistic scenario must be labeled illustrative; realistic-looking numbers are not empirical evidence.

Use this record:

```text
Example title and evidence label:
Workflow and decision:
Input data/table, units, dimensions, and source:
Valid assumptions and what is intentionally simplified:
Steps: equation/rule -> substitution -> unrounded intermediate -> result.
Final result with units and decision consequence:
Independent check: dimensions, sign, limit, alternate calculation, or oracle.
Wrong path: plausible mistake -> misleading result -> correction.
Production-scale change: what grows, fails, or needs measurement.
Remaining uncertainty and what the example does not establish:
```

| Step | Equation/rule | Inputs with units | Intermediate | Reported result | Independent check |
|---|---|---|---|---|---|
| `1` | `[ID]` | `[Values]` | `[Full precision]` | `[Rounded value + unit]` | `[Check]` |

Use the main Section 4 formulas where applicable. If important variants require different assumptions, add a separate mini-example instead of forcing incompatible formulas into one scenario. Keep input tables complete; link large artifacts instead of truncating rows that are necessary to reproduce the calculation.

**Illustrative checked example of unit discipline:**

```text
Assumptions: initially empty fluid queue, no loss/retries, constant rates,
and enough storage for the stated interval. These are teaching assumptions.

Input rate:                    1,200 events/s
Sustained service rate:         1,000 events/s
Backlog growth while overloaded: (1,200 - 1,000) = 200 events/s
Duration:                      10 min * 60 s/min = 600 s
Added backlog:                 200 events/s * 600 s = 120,000 events
Payload per event:             1 KiB = 1,024 bytes
Payload-only storage:          120,000 * 1,024 = 122,880,000 bytes
                              = 117.1875 MiB, approximately 117.19 MiB
```

This is a simplified fluid estimate, not a measured queue benchmark or a prediction of tail latency. Metadata, indexes, replication, compression, variable sizes, finite buffers, bursts, and recovery change actual capacity. Increasing storage absorbs backlog temporarily; it does not repair a sustained rate imbalance. That distinction is a useful interview follow-up.

**Must pass:** The arithmetic is independently checked, units cancel correctly, evidence is labeled, and the production interpretation does not exceed the model's assumptions.

### Section 6 — Visual / Geometric Intuition

**Conditional:** Required for mathematical proof/matrix topics in the archived template, and whenever spatial structure, timing, state, or geometry materially aids understanding. Otherwise retain the heading with a specific reason for N/A.

Use correctly labeled ASCII diagrams by default to preserve portability. If a collection deliberately permits Mermaid or linked figures, declare that extension and verify the renderer; do not make the explanation depend on an inaccessible image.

Every diagram needs axes/dimensions or state/flow labels, at least one named or valued key point, a caption, and a plain-text interpretation. State if it is schematic or not to scale. Geometry must support the claim; an attractive curve with the wrong curvature or mislabeled matrix dimension is unacceptable.

Examples of useful visuals: projection geometry, loss/constraint boundary, memory layout, timing diagram, state machine, graph traversal, coordinate-frame chain, backpressure flow, or forecast interval growth. Do not add a diagram that repeats the title without teaching a relationship.

**Must pass:** A reader can interpret the relevant relationship from the labeled diagram and caption alone.

### Section 7 — Comparison / Selection Table

**Required when two or more methods/variants are considered.** For this industry/interview library, include the simplest credible baseline whenever it is a real alternative. With one meaningful method, explain why alternatives are out of scope and compare its valid versus invalid operating conditions.

| Method/variant | Preconditions | Quality/guarantee | Time/memory/cost | Best for | Avoid when | Evidence |
|---|---|---|---|---|---|---|
| `[Baseline]` | `[Conditions]` | `[Bounded claim]` | `[Defined workload]` | `[Use]` | `[Nonblank disqualifier]` | `[Label/source]` |
| `[Alternative]` | `[Conditions]` | `[Bounded claim]` | `[Defined workload]` | `[Use]` | `[Nonblank disqualifier]` | `[Label/source]` |

Follow with a prioritized selection guide:

```text
IF a hard validity/security/safety constraint fails -> disqualify that option.
ELSE IF [highest-priority workload/objective condition] -> [choice + reason].
ELSE IF [next condition] -> [alternative + tradeoff].
ELSE -> [fallback, simpler baseline, or collect evidence].
If conditions conflict -> [explicit priority and escalation/measurement].
```

For empirical comparisons, include `workload/data | hardware/software/model versions | metric definition | sample/repetitions | result/uncertainty | primary source | reproduced? | date`. Compare like with like. Distinguish benchmark results from theoretical bounds and include failure regimes, not only favorable cases.

The archived improvement requests empirical grounding; satisfy it with verified comparable evidence where available. When absent, show a benchmark plan and `NOT MEASURED`, not invented percentages. Split more than about six options into coherent groups and explain how the groups relate.

**Must pass:** The reader can select or reject an option under conflicting constraints without relying on an unsupported “best method” claim.

### Section 8 — Failure Modes & Edge Cases

**Required for every sheet.** Include at least **four distinct substantive failures**, with additional depth determined by the topic. Cover the assumptions and production conditions most likely to mislead the reader.

| ID | Trigger/symptom | Cause or competing hypotheses | Discriminating check / expected signal | Correct action | Prevention/verification | Provenance |
|---|---|---|---|---|---|---|
| `F-01` | `[Observable]` | `[Mechanism + confidence]` | `[Test and what it establishes]` | `[Bounded fix]` | `[Check/invariant/alert response]` | `[Observed/reproduced/hypothetical + source]` |

Distinguish a symptom from its cause. The same latency spike may arise from queueing, GC, I/O, network, skew, or retries; a useful diagnostic discriminates rather than simply repeating “check logs.” State what a negative test result means and when to escalate.

| Topic family | Candidate failures to select and explain |
|---|---|
| Statistical | Violated independence, selection bias, misspecification, heteroscedasticity, changing base rate |
| Matrix/numerical | Rank deficiency, conditioning, cancellation, overflow/underflow, dtype/shape error |
| Optimization | Infeasibility, unstable steps, local minima/saddles, weak stopping condition, constraint violation |
| Probability | Degenerate cases, wrong conditioning, zero denominators, support mismatch |
| Hardware/systems | Races, invalid layouts, memory pressure, synchronization, hot keys, overload, recovery duplication |
| LLM/agents | Context truncation, retrieval misses, injection, permission leakage, repeated effects, unbounded loops |
| Forecasting | Leakage, structural breaks, lag misalignment, censoring, intermittent demand |
| Fraud/insurance | Immature labels, review bias, calibration shift, investigative-capacity mismatch |
| Edge/robotics | Timestamp/calibration/frame errors, deadline misses, sensor loss, thermal limits, unsafe fallback |

Connect reusable failures to the final Common Issues file using stable issue IDs. If the issue entry is not written, mark it planned rather than creating a false completed reference.

**Must pass:** Each row teaches a specific mechanism, a discriminating test, a bounded fix, and how to verify the change.

### Section 9 — Complexity & Cost Analysis

**Required for algorithmic, ML, systems, infrastructure, and production sizing topics.** For purely theoretical sheets without relevant computational/cost content, retain the heading and explain N/A. Do not confuse lack of measurement with lack of relevance.

#### 9a. Algorithmic Complexity

Define every size variable and operation model. State worst/average/amortized or expected-case assumptions as appropriate. Include preprocessing/build, training/fitting, query/inference, update, and recovery/rebuild where relevant.

| Operation | Input sizes | Time | Space | I/O/communication | Assumptions / dominant bottleneck |
|---|---|---|---|---|---|
| `[Build/fit]` | `[Defined variables]` | `[Bound]` | `[Bound]` | `[Relevant cost]` | `[Condition]` |
| `[Query/serve]` | `[Sizes]` | `[Bound]` | `[Bound]` | `[Relevant cost]` | `[Condition]` |
| `[Update/recover]` | `[Sizes]` | `[Bound]` | `[Bound]` | `[Relevant cost]` | `[Condition]` |

For iterative estimators/solvers, include iteration count or convergence assumptions. For distributed work, explain communication, synchronization, and stragglers. For DSA, distinguish auxiliary space from stored input/output and amortized efficiency from per-operation worst case.

#### 9b. Back-of-Envelope Sizing

Include at least one relevant worked sizing calculation when this section applies:

- **Workload:** Volume, arrival rate, burst shape, input dimensions, batch/context length, data types.
- **Environment:** Device, memory, software/compiler/runtime, topology, configuration, version.
- **Budget:** Latency/deadline, throughput, memory, storage, bandwidth, power/energy, or cost.
- **Calculation:** Formula, substitution, units, headroom assumption, result, sensitivity.
- **Bottleneck:** What limits scaling and what measurement would confirm it.
- **Production difference:** Caches, concurrency, skew, retries, replication, batching, thermal behavior, recovery, or other relevant overhead.
- **Economics:** Cost per successful/useful outcome, currency/date, included/excluded charges and labor where relevant.
- **Evidence:** Measured, sourced, estimated, simulated, or illustrative inputs clearly distinguished.

Record FLOPs, bytes, operations, tokens, time, energy, and money as different quantities. Do not infer measured latency from Big-O or equate theoretical peak throughput with attainable application performance. Define percentile metrics and load conditions; do not sum component percentiles and present the result as an established end-to-end percentile.

For hardware roofline reasoning, identify precision, appropriate compute peak, bandwidth basis, and real reads/writes including outputs and re-reads. For robotics timing, distinguish observed maximum latency from a justified worst-case bound; average speed alone does not establish deadline compliance.

**Must pass:** A reader can recompute the estimate, identify its assumptions, and explain why the theoretical model may differ from deployment measurements.

### Section 10 — Key Theorems / Results Reference

**Required for mathematical theory sheets; conditional elsewhere.** Retain the heading and explain N/A when no formal result is relevant. Do not invent a theorem to make a systems note appear rigorous.

| Theorem/result | Correct statement | Essential hypotheses | Practical implication | What it does not imply | Primary/textbook source |
|---|---|---|---|---|---|
| `[Result]` | `[Precise enough for use]` | `[Conditions]` | `[Decision consequence]` | `[Counterexample/overclaim]` | `[Exact reference]` |

Distinguish theorem, identity, bound, asymptotic result, empirical finding, implementation guarantee, and heuristic. Name conditions such as regularity, independence, stationarity, support, activation/domain, convergence mode, and model class where applicable. Link a formal statement when a compact summary omits detail.

For protocol or API guarantees, name the scope and assumptions, including failure semantics and version. A broker guarantee does not automatically establish end-to-end application behavior.

**Must pass:** The reader knows what can be concluded, under which assumptions, and a plausible situation where overgeneralization fails.

### Section 11 — Interview / Exam Questions

**Required for this library.** The archived template made this optional for general collections; it is mandatory here because intense interview preparation is a stated purpose. Preserve the three original difficulty levels and put all answers before Section 12.

Include **10–15 distinct questions per focused sheet**. Target roughly 3 recall/concept questions, 3–4 derivation/application questions, and 4–8 judgment/design/debug questions, adjusting within the total for the topic. At least half should require application, explanation, calculation, or diagnosis rather than recall alone.

#### Level 1 — Definition / Recall

Test precise meaning, key assumptions, and distinctions between related terms. Answers must include an example or nonexample where that distinction matters. Avoid trivia about an implementation version unless it is central to the sheet.

#### Level 2 — Derivation / Apply

Require at least one checked derivation or invariant argument and one numerical calculation/algorithm trace. Supply the needed inputs or specify what the candidate must estimate. Reward dimensional checks, boundary conditions, and honest assumptions.

#### Level 3 — Judgement / Design / Debug

Test method selection, diagnosis from an ambiguous symptom, changed workload, failure recovery, evaluation validity, production tradeoffs, and transfer to another domain. Include a plausible wrong answer and explain the mechanism that makes it wrong.

Use this question record:

```text
Q-ID / difficulty / skill:
Prompt and given facts:
Clarifications and acceptable assumptions:
Answer: reasoning -> derivation/calculation/trace -> result -> sanity check.
Alternative answers and when they are defensible:
Follow-up: change scale, data, latency, cost, failure, or evidence condition.
Trap/red flag and why it is wrong:
Scoring: 0 wrong; 1 partial; 2 correct with assumptions; 3 defended tradeoff.
Answer source: local section/equation or authoritative reference.
```

For a full sheet, ensure the set includes: an assumption challenge, a quantitative/trace exercise, a selection tradeoff, an ambiguous failure diagnosis, a constraint-change follow-up, and a question about verifying the claimed result. Questions can serve more than one skill, but do not count a reworded prompt as a new question.

Keep candidate-facing prompts and interviewer reasoning clearly separated within each record. Give checked short answers in the sheet; a linked expanded answer key may add detail but must not leave the core answer unavailable. Do not imply a fictional question was asked by a named employer.

**Must pass:** Every question is answerable and checked, the set exercises reasoning across the three levels, and alternate defensible approaches are recognized.

### Section 12 — Quick Reference Card

**Required and the final section/content of the individual sheet.** Draft it first as a planning compass and verify it last against the complete body. Preserve the original seven fields within a **5–10-row** card; add up to three critical rows only when they improve retrieval.

| Item | Value |
|---|---|
| Topic | `[Precise topic and decision]` |
| Core formula | `[Most important equation with critical validity cue, or invariant/rule]` |
| Key assumption | `[Condition most likely to be violated]` |
| Complexity | `[Dominant operation, defined sizes, time/space or justified N/A]` |
| Use when | `[One-line selection condition]` |
| Avoid when | `[Concrete disqualifier]` |
| Go deeper | `[Actual next file/topic link, or an explicitly stated missing handoff]` |

Optional extra rows can identify a diagnostic, operational guardrail, or evidence/version limitation. Version/review metadata belongs in the header; do not add an appendix after the card. Keep citations/source tables, answers, exercises, notes, and references within earlier sections. A card must not introduce a new claim or hide a condition that changes its recommendation.

**Must pass:** A reader scanning for 30 seconds can retrieve the decision, formula/rule, assumption, cost, warning, and next step without being misled.

## Part C — Ten Improvement Tips as enforceable instructions

| Original improvement | Required writing practice | Publication test |
|---|---|---|
| 1. Derive before you state | Section 4 begins with premises and justified steps | Every important result has a derivation/sketch or a precise source; no unexplained formula dump |
| 2. Add “Avoid when” | Sections 4/7 comparison rows have nonblank disqualifiers | Reader can reject each option under a concrete condition |
| 3. Anchor abstractions empirically | Use comparable sourced/reproduced benchmarks where available | Numeric superiority has protocol/source; absent evidence is marked and paired with a test plan |
| 4. Constraints before formula | Show validity before use and inside standalone block | A copied formula cannot lose its essential condition |
| 5. Master notation glossary | Maintain local definitions and INDEX registry | Units, shapes, domains, and symbol meanings reconcile across files |
| 6. Textbook versus production | Pair a checked small example with a labeled production-scale interpretation | Scale, environment, overhead, and uncertainty are explicit |
| 7. Label ASCII diagrams | Axes/dimensions/states, key point, caption, interpretation | Diagram is correct and understandable without nearby prose |
| 8. Build issues from concrete failure mechanisms | Use the four archived categories plus explicit provenance | At least 25 unique completed collection issues; diagnostic and bounded fix in each |
| 9. Declare excluded scope | Header and INDEX identify exclusions and real handoffs | Reader knows whether a missing topic is excluded, assumed, or planned |
| 10. Draft the card first | Use provisional card to plan, validate against body, publish last | Every card claim is supported; no section follows it in an individual file |

The archive preserves some repeated prompts and historical examples. This consolidated guide states each rule once and uses the integrated accuracy cautions. Do not preserve transcription errors or unattributed numerical claims in new teaching material.

## Part D — Completion and Quality Checklist

### D1. Collection completeness

- [ ] One canonical INDEX contains all 11 sections and clear collection identity, audience, owner, version, review date/status, and scope.
- [ ] Topic ladder spans relevant foundations, methods, advanced/production work, and final troubleshooting.
- [ ] Required prerequisite graph is acyclic and consistent with learning order; optional forward links are identified.
- [ ] Beginner, Practitioner, and Advanced reading paths are feasible for the actual collection.
- [ ] Topic lookup answers real reader questions, not only exact filename searches.
- [ ] Master glossary reconciles symbols, units, dimensions, and terms across sheets.
- [ ] Quick Reference Card directory and Common Issues locator link to actual sections.
- [ ] Planned topics, drafts, reviewed sheets, and published material are counted separately.
- [ ] Final Common Issues artifact contains at least 25 distinct complete entries, or remains explicitly incomplete.
- [ ] Company/domain examples have evidence or are visibly hypothetical reference exercises.
- [ ] Collection source registry, per-file open issues, review triggers, and deprecated/superseded paths are maintained.
- [ ] All file links and anchors resolve where presented as working links; INDEX was finalized last.

### D2. Per-file structural completeness

- [ ] Metadata and header give purpose, prerequisites, testable outcome, scope exclusions, version/review, evidence, level, and retrieval cue.
- [ ] All 12 numbered sections are present in order; conditional N/A entries explain the reason.
- [ ] Concept map appears before the first substantive formula.
- [ ] Definitions are precise and all symbols have units/types, dimensions, and valid domains as relevant.
- [ ] Section 4 retains base case, general case, variants, and boundary/counterexample reasoning.
- [ ] New substantive formulas have five-part anatomy, conditions before use, derivation, and interpretation.
- [ ] Section 5 has a checked end-to-end example, independent sanity check, wrong path, and production interpretation.
- [ ] Relevant visuals are accurate, labeled, captioned, and accessible in text.
- [ ] Selection tables include the baseline where meaningful, assumptions, avoid conditions, evidence, and fallback.
- [ ] At least four distinct failure rows have discriminating diagnostics, bounded fixes, and verification.
- [ ] Complexity and worked sizing are present where applicable; unmeasured behavior is not marked N/A.
- [ ] Theorems/guarantees are qualified and sourced when relevant.
- [ ] Section 11 has 10–15 checked questions across recall, application, and judgment, including follow-ups and answer support.
- [ ] Each substantive section has a useful one-line takeaway; the final card serves as Section 12's conclusion.
- [ ] Section 12 has all seven original fields and remains the final content, without later answers or appendices.

### D3. Technical, evidence, and production rigor

- [ ] Nontrivial claims are derived, verified, sourced, explicitly heuristic, illustrative, or open; a table alone does not constitute support.
- [ ] Source citations support the associated claims and distinguish original findings from secondary descriptions.
- [ ] Numerical values are independently recalculated, with rounding, units, signs, shapes, and boundary behavior checked.
- [ ] Conditional/unconditional quantities, point predictions/error distributions, and observed/causal effects are not conflated.
- [ ] Benchmark comparisons use compatible workload/setup/metrics, with sources, versions, dates, and reproduction status.
- [ ] Algorithmic bounds, theoretical peaks, estimates, microbenchmarks, and production observations are clearly distinguished.
- [ ] Relevant security, permissions, privacy, human review, operational ownership, and failure boundaries are explained where they change usage.
- [ ] Scale discussion includes the relevant memory/I/O/communication/concurrency/energy effects rather than Big-O alone.
- [ ] Monitoring signals have definitions and expected responses; recovery claims have a specified test or explicit evidence gap.
- [ ] Heuristic thresholds and scenario values are not presented as universal best practices.
- [ ] Examples from interview folders remain reference-only unless independently evidenced or reproduced and labeled accordingly.
- [ ] The card and interview answers introduce no unsubstantiated claims beyond the body.

### D4. Review rubric and release blockers

Score applicable dimensions `0 = missing/incorrect`, `1 = partial`, `2 = usable and supported`. Treat this as an editorial review aid, not a formal assessment standard.

| Dimension | Strong evidence |
|---|---|
| Scope and retrieval | Focused outcome, prerequisites, navigation, clear boundaries |
| Conceptual precision | Correct definitions, notation, nonexamples, assumptions |
| Derivation/mechanism | Reproducible reasoning and valid counterexample |
| Numerical integrity | Checked example, dimensions, rounding, independent sanity check |
| Selection value | Baseline, alternatives, disqualifiers, conflicting-constraint resolution |
| Diagnostic value | Observable symptom, competing causes, discriminating test, verified fix |
| Production realism | Defined workload, sizing, operations, constraints, cost, recovery limits |
| Interview depth | Checked answers and follow-ups across the three levels |
| Evidence and maintenance | Correct citations/labels, versions, triggers, honest gaps |

Do not publish with a material mathematical error, invented benchmark, unsupported company/deployment claim, broken essential reference, unanswered required question, or hidden assumption that reverses the recommendation. Missing evidence may be published as a clearly bounded teaching limitation; it must not be reported as a verified result. A high rubric total does not override a release blocker.

## Part E — Collection Skeleton and Authoring Workflow

### E1. Collection planning worksheet

Fill this before drafting the first topic:

| Planning field | Fill in |
|---|---|
| Collection name / canonical path | `[Name and actual directory]` |
| Domain and workflow | `[Specific decisions and operating context]` |
| Target audience | `[Role, experience, intended industry/interview use]` |
| Prerequisite skills | `[Concrete knowledge and reference locations]` |
| Learning outcomes | `[Derive, calculate, choose, diagnose, size, defend]` |
| Scope exclusions | `[Boundaries and handoffs]` |
| Notation/display standard | `[LaTeX or plain-text; units and naming conventions]` |
| Source/evidence policy | `[Primary sources, verification, example labels]` |
| Common Issues target | `[25 or 50; justify breadth and available evidence]` |
| Owner and reviewers | `[Roles and technical/domain review coverage]` |
| Review cadence / triggers | `[Timing and concrete invalidating changes]` |
| Pre-reading / next collection | `[Available references or explicitly planned handoff]` |

### E2. Topic ladder planning table

| Level | File ID | Focused topic | Testable outcome | Prerequisite IDs | Core mechanism/formula | Production example | Failure/interview focus | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | `01` | `[Foundations]` | `[Capability]` | `[None/external]` | `[Core idea]` | `[Labeled setting]` | `[Misconception]` | `planned` |
| 2 | `02` | `[Core method]` | `[Capability]` | `[01]` | `[Derivation]` | `[Setting]` | `[Selection/calculation]` | `planned` |
| 3 | `03` | `[Advanced/production]` | `[Capability]` | `[Earlier files]` | `[Scale/composition]` | `[Setting]` | `[Failure/design]` | `planned` |
| Final | `[NN]` | `[Actual N] Common Issues` | `[Diagnose and verify]` | `[Relevant sheets]` | `[Diagnostic rules]` | `[Failure provenance]` | `[Ambiguous debugging]` | `planned` |

Expand the rows to match the real collection; these are planning examples, not a sufficient comprehensive syllabus by themselves. Define a master glossary and plan the three reading paths before authoring. Review whether omitted topics are intentionally excluded, covered elsewhere, or missing.

### E3. Per-sheet content brief

Use one brief for every planned sheet:

```text
File ID / proposed name / level:
Reader task and business/production decision:
Scope and exclusions; prerequisite links:
Definitions and notation needed:
Base/general/variant/limit derivations or mechanism arguments:
Input table and worked example to construct or source:
Independent numerical or correctness check:
Alternatives, disqualifiers, and selection priorities:
At least four distinct failure mechanisms and diagnostic evidence:
Sizing/cost question and workload assumptions:
Applicable theorems/guarantees and their limitations:
10–15 interview questions, checked answers, and follow-ups:
Provisional seven-field card:
Sources to verify; missing evidence; related cases/sheets:
Author/reviewer roles, status, and review trigger:
```

A brief can be detailed without pretending the sheet exists. Keep planned artifact names as plain text until linked files exist. A source reading list is not a completed technical explanation.

### E4. Delivery and acceptance tracking

| File | Editorial state | Technical/source checks | Numerical/trace checks | Interview answers checked | Reviewer role | Date / unresolved gaps |
|---|---|---|---|---|---|---|
| `[Canonical INDEX]` | `planned` | `[Pending]` | `[N/A or relevant]` | `[Path coverage]` | `[Role]` | `[Status]` |
| `[Focused sheet]` | `planned` | `[Pending]` | `[Pending]` | `[Pending]` | `[Role]` | `[Status]` |
| `[Final Common Issues]` | `planned` | `[Count/provenance pending]` | `[Diagnostics pending]` | `[Pending]` | `[Role]` | `[Status]` |

Do not advance state based only on file existence or length. `Reviewed` requires substantive review; `published` requires applicable gates to pass. Preserve deprecated material with replacement links and reasons rather than silently changing the historical claim.

### E5. Recommended workflow

1. **Plan:** Fill the worksheet, ladder, glossary, scope, prerequisites, evidence policy, and provisional INDEX.
2. **Design backward:** Draft each reader outcome and Section 12 card; identify what evidence and reasoning support its rows.
3. **Build foundations:** Write the mental model and definitions, then derive the base and general cases with explicit assumptions.
4. **Compute and challenge:** Complete worked examples, sanity checks, wrong paths, boundaries, and method comparisons.
5. **Make it operational:** Write failure diagnostics, sizing, cost, deployment constraints, observability, and recovery reasoning as applicable.
6. **Make it interview-ready:** Write and check the question set, acceptable alternatives, follow-ups, and red flags.
7. **Connect troubleshooting:** Complete the final issue entries from concrete mechanisms and label their provenance.
8. **Review:** Check the ten improvements, technical accuracy, sources, notation, Markdown structure, links, and prerequisite graph.
9. **Publish:** Verify and place each card last; finalize the INDEX, issue counts, review records, and maintenance triggers.

### E6. Archived input map and consolidation boundary

All eight files in the supplied cheat-sheet folder were reviewed for this consolidation:

| Archived input | Role retained here |
|---|---|
| [Integrated outline v3](archive/Cheat_Sheet_Templates_v3_Complete/Cheat_Sheet_Outline_Template_v3_Integrated.md) | Parts A–E, 12 sections, derivation rules, ten improvements, validation cautions |
| [Source-faithful outline v3](archive/Cheat_Sheet_Templates_v3_Complete/Cheat_Sheet_Outline_Template_v3_Source_Faithful.md) | Original structure and learning intent; historical transcription remains archived |
| [Per-file blank template v3](archive/Cheat_Sheet_Templates_v3_Complete/Cheat_Sheet_Per_File_Blank_Template_v3.md) | Header, base/general/variants/limit sequence, worked example, failures, final card |
| [INDEX template v3](archive/Cheat_Sheet_Templates_v3_Complete/00_INDEX_Cheat_Sheets_Template_v3.md) | All 11 INDEX sections, reading paths, glossary, card directory, issues locator, maintenance |
| [Collection skeleton v3](archive/Cheat_Sheet_Templates_v3_Complete/Cheat_Sheet_Collection_Skeleton_v3.md) | Planning worksheet, topic ladder, scope/handoffs, delivery tracking |
| [Common Issues template v3](archive/Cheat_Sheet_Templates_v3_Complete/Cheat_Sheet_Common_Issues_Template_v3.md) | Four categories, three provenance labels, issue table/details, real completion count |
| [Package README v3](archive/Cheat_Sheet_Templates_v3_Complete/README_Cheat_Sheet_Templates_v3.md) | Workflow and source-fidelity/accuracy distinction |
| [Validation report v3](archive/Cheat_Sheet_Templates_v3_Complete/VALIDATION_REPORT_Cheat_Sheet_v3.md) | Known benchmark, roofline, ARIMA, theorem, and geometry cautions; structural checks |

The canonical per-file sequence is unchanged: **header -> concept map -> definitions -> formulas/derivation -> worked example -> visual intuition -> selection -> failures -> complexity/cost -> theorems/results -> interview questions -> final card**. Production/domain profiles, mandatory checked interview answers, and the 12-section adaptation of the final troubleshooting file are explicit additions for this library. This guide does not certify the completeness or factual accuracy of any existing course or collection.
