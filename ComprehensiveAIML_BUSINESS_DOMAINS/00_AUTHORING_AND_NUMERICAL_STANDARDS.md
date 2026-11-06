# Course authoring and numerical standards

**Applies to:** every module, domain, Case_Studies collection, and Cheat_Sheets collection in this course. **State:** authoring contract for planned content; not certification of existing lessons. **Effective:** 2026-11-06. **Owner:** curriculum maintainer.

The authoritative instructions are the [Case Study Content Master Template](../Templates/CASE_STUDY_CONTENT_MASTER_TEMPLATE.md) and [Cheat Sheet Content Master Template](../Templates/CHEAT_SHEET_CONTENT_MASTER_TEMPLATE.md). They replace the earlier short narrative/card instructions and archived template references. Local briefs describe scope; they do not waive these requirements. Keep existing `INDEX.md` names as canonical indexes; do not create competing `00_INDEX.md` catalogs.

For this course, each newly authored individual case study and cheat sheet must contain at least **1,000 lines**. This is an additional course requirement, not a substitute for the templates' depth and quality checks: use substantive explanations, distinct worked examples, production analysis and interview reasoning, not blank-line or repetitive padding. Collection indexes and existing planning documents are not individual teaching files.

Consult [directory.txt](../Templates/directory.txt) alongside the [source crosswalk](00_SOURCE_TOPIC_CROSSWALK.md) when choosing scope. A source entry remains planned until its distinct required subject is actually authored; collapsed or clipped source entries remain unresolved.

## 1. What coverage means

| State | What it establishes | What it does not establish |
|---|---|---|
| Source inventoried | A visible source entry is recorded, including uncertainty | Its contents were accessible or technically reviewed |
| Topic planned | A concrete topic has a course destination, teaching outcome, numerical/mechanism exercise, and case decision | A lesson, case, exercise answer, or cheat sheet has been authored |
| Draft | Actual content exists but publication checks remain open | Correctness, evidence, or review completion |
| Reviewed / published | Applicable master-template checks were actually completed and recorded | A proposed system became a production deployment |

The [file-level source crosswalk](00_SOURCE_TOPIC_CROSSWALK.md) and its linked focused plans extend the existing local indexes. Their planned topics are part of the syllabus even when an older local table has only five broad rows. Source PDFs, code directories, exercises, collators, archives, and overview files are tracked as supporting artifacts, not counted as independent completed lessons. Hidden contents and clipped names remain unresolved; teach a visible topic prefix without inventing its missing filename or contents.

## 2. Per-topic authoring brief

Before drafting, register a stable local sheet/case ID and fill the following fields. Source filenames are provenance identifiers, not obligatory new course filenames. Merge synonymous coverage only when the distinct source methods, assumptions, and failure boundaries remain explicit; split independent outcomes into focused sheets.

| Field | Required planning content |
|---|---|
| Teaching home | Actual module/domain INDEX link and focused-plan anchor |
| Outcome | Given specified inputs, what the learner must derive, calculate, select, diagnose, or defend |
| Topic inventory | Every visible subtopic, including variants, named protocols, implementations, and failure themes |
| Prerequisites | Concrete earlier skills and available links; mandatory graph must be acyclic |
| Cheat-sheet brief | Base/general/variant/limit reasoning; notation; numerical example or precise invariant/trace; alternatives; at least four distinct failure mechanisms |
| Case brief | One bounded consequential decision, owner, input/output grain, baseline, constraints, proposed method, evaluation and failure question |
| Quantitative task | Given values and units; derivation; substitution; independent arithmetic/oracle check; sensitivity and decision consequence |
| Evidence | Sources to obtain or verify; version/date; illustrative versus measured values; inaccessible source limitations |
| Publication work | Interview answers, local collection indexes, glossary, issue entries, backlinks and review gates still required |

Plain text names describe unwritten artifacts. Link a real planning brief as a brief, not as an authored case. A one-row topic list cannot become a published sheet by changing its status.

## 3. Case-study structure and collection gate

Every authored case preserves these 15 sections **in order**, with metadata before the title and Section 15 last. Retain conditionally inapplicable sections with a specific `N/A` reason; unknown evidence is not `N/A`.

1. The One-Line Framing
2. Context & Background
3. The Business Question
4. Assumptions & Constraints
5. Data, Tools & Method — retain 5a Data Specification, 5b Tools & Stack, 5c Method Summary, 5d provenance/quality/exclusions/leakage, and 5e reproducibility/method-selection ledger
6. Analysis Walkthrough
7. The Finding
8. System/Solution Design — applicable offline and online paths, key choices, operational readiness
9. Evaluation & Validation
10. Action Taken
11. Outcome & Results
12. Failure Modes & What Went Wrong
13. Transferable Lesson
14. Connections & Cross-References — related methods/cases, open questions, interview pack
15. Quick Reference Card — original six fields plus supplemental evidence/stage/review record

Include a checked numerical walkthrough, sensitivity for quantitative conclusions, and worked workload/capacity estimates when systems sizing is relevant. Define metric numerator, denominator, units, population, window, comparator, guardrails, uncertainty, and executed/not-executed status. System cases require a labeled input-to-output architecture and both successful and failed request/recovery reasoning. Distinguish considered alternatives from actually tested failures.

Section 14 contains **12–18 distinct interview questions**, each with a checked answer, reasoning, follow-up, acceptable alternatives, red flags, scoring, and answer support. Include numerical work, an ambiguous diagnosis, a plausible wrong method, and a changed constraint. Sources, answers and appendices belong before the final card.

Each case collection INDEX must contain the master's **seven sections**: collection metadata; 11-column case catalog; canonical reusable patterns; similarity/contrast map; questions/future-case backlog; review/retirement log; pre-publish checks. An older planning catalog with fewer sections remains a planning catalog until expanded; its presence is not template compliance. Keep the original catalog fields: Case ID, linked title if authored, domain, case type, decision, primary pattern, up to two secondary patterns, evidence, stage, last reviewed, result/limitation. Record editorial status separately.

Use the actual case type (`analytical`, `predictive-ml`, `llm-agentic`, `streaming-systems`, `incident`, or genuinely integrated `hybrid`), not a blanket default. `PLAN` IDs identify briefs; assign and crosswalk stable `CASE` IDs when authoring. Historical planning-review dates do not establish that an unwritten case was substantively reviewed.

## 4. Cheat-sheet structure and collection gate

Every authored sheet has the master's metadata and all **12 sections in order**:

1. File Header
2. Concept Map / Mental Model
3. Foundational Definitions
4. Core Formulas with Full Derivation — 4.1 base, 4.2 general, 4.3 variants, 4.4 limits/counterexample
5. Worked Numerical Example
6. Visual / Geometric Intuition
7. Comparison / Selection Table
8. Failure Modes & Edge Cases
9. Complexity & Cost Analysis — algorithmic costs and worked sizing where applicable
10. Key Theorems / Results Reference
11. Interview / Exam Questions
12. Quick Reference Card — final content, original seven fields in a 5–10-row card

Use a foundation → core methods → advanced/production ladder, followed by a final Common Issues artifact. Mandatory prerequisites must appear earlier in the chosen learning order; optional forward links must be labeled. Each substantive section ends in a useful takeaway, except Section 12 whose card is the conclusion. Include at least four diagnostic failure entries with competing causes, discriminating checks, bounded fixes and verification. Section 11 has **10–15 distinct questions with checked answers**, across recall, application, and judgment; at least half require reasoning, calculation, selection, or diagnosis.

Every cheat-sheet collection INDEX must contain the master's **11 sections**: overview; topic ladder; quick lookup; Beginner/Practitioner/Advanced reading paths; prerequisite skills; exclusions/handoffs; master symbol glossary; card directory; Common Issues locator; sources/versions/maintenance; acceptance checklist. Planned card/issue anchors are not working links until content exists.

Final troubleshooting files retain all 12 headings and contain **at least 25 distinct complete issues**, or **50 for an extended collection** where meaningful. Preserve source-specific 50-issue or 100-issue titles as authoring coverage targets; publish such a title only when that many distinct, complete issues exist. Each issue needs a stable ID, category (`VIOLATED_ASSUMPTION`, `NUMERICAL_EDGE`, `SCALE_FAILURE`, `CONCEPT_CONFUSION`), separate provenance, competing causes, expected diagnostic signal, mitigation risks, correction, prevention and verification. Empty rows do not count.

## 5. Equations, mechanisms and numerical examples

**Canonical display-math convention:** fenced `text` equations throughout this course. Inline symbol names may coexist. Define every symbol at first use and in the collection glossary: meaning, units, shape/type, domain and index convention. Do not treat counts, rates, probabilities, parameter dimensions, or event/processing time as interchangeable.

For every new substantive equation:

1. State validity **before** use and derive from a definition or named result, with essential assumptions.
2. Supply `FORMULA → WHERE → CONSTRAINT → NUMERICAL EXAMPLE → INTERPRETATION`.
3. Check dimensions, arithmetic, rounding, signs, conditioning, and a material limit/counterexample.
4. Label constructed inputs illustrative, even when the arithmetic is independently checked.
5. Cross-reference a previously derived result instead of duplicating its full block. A final card still carries its critical validity condition.

A non-mathematical process uses a precise mechanism/invariant block: preconditions, input/output, state transition, invariant and justification, worked trace, termination/recovery behavior, failure boundary. Use relevant quantitative sizing/correctness results, not decorative equations.

### Worked authoring example: backlog, storage and recovery

**Evidence:** ILLUSTRATIVE teaching workload; derived arithmetic, not a broker benchmark or observed deployment. **Validity:** initially empty fluid queue, constant arrivals and service, no loss/retries, sufficient buffer; the drain calculation assumes arrivals later fall below service capacity.

Derivation: conservation of events gives backlog change = arrivals − completed service. Integrating a positive constant imbalance over duration gives the first result; multiplying event count by payload gives payload-only storage. During recovery, spare service capacity removes queued events.

```text
FORMULA [Q-01]: B = max(0, lambda - mu) * t
WHERE: B backlog [events]; lambda input [events/s]; mu service [events/s];
       t overload duration [s].
CONSTRAINT: initially empty queue, constant rates, no losses/retries;
            this is a fluid workload estimate, not a tail-latency guarantee.
NUMERICAL EXAMPLE: lambda=1200, mu=1000, t=600;
                   B=(1200-1000)*600=120000 events.
INTERPRETATION: buffering absorbs the interval, not sustained overload.

FORMULA [Q-02]: S = B * b
WHERE: S payload storage [bytes]; B [events]; b payload [bytes/event].
CONSTRAINT: payload-only; metadata, indexes and replication excluded.
NUMERICAL EXAMPLE: b=1024; S=120000*1024=122880000 bytes
                   =117.1875 MiB (1 MiB=1048576 bytes).
INTERPRETATION: provision additional measured overhead and headroom.

FORMULA [Q-03]: T = B / (mu_r - lambda_r)
WHERE: T drain time [s]; mu_r recovery service [events/s];
       lambda_r recovery arrival [events/s]; B initial backlog [events].
CONSTRAINT: mu_r > lambda_r, constant rates, no new outages/retries.
NUMERICAL EXAMPLE: mu_r=1000, lambda_r=800;
                   T=120000/(1000-800)=600 s=10 min.
INTERPRETATION: a 200 events/s recovery surplus clears this backlog in 10 min.
```

**Independent checks:** 200 events/s × 600 s = 120,000 events in both overload and drain directions; bytes/event × events = bytes; converting MiB back yields 122,880,000 bytes. **Sensitivity/boundary:** if recovery arrivals equal 1,000 events/s, spare capacity is zero and a positive backlog never drains under this model; above that rate it grows. Increasing buffer capacity cannot fix that imbalance. These checks establish arithmetic and model consequences, not actual latency, throughput, or recovery guarantees.

Apply equally explicit reasoning to loss functions, uncertainty, calibration, pricing elasticity, causal estimands, graph/ranking objectives, training memory/communication, GPU rooflines, signal processing, quantization error, and device timing/power wherever those are the topic. Do not substitute this queue example for each topic's own mathematics.

## 6. Evidence and current facts

| Artifact | Permitted evidence vocabulary |
|---|---|
| Case | observed, published, estimated, simulated, illustrative, not-measured |
| Cheat sheet | DERIVED, VERIFIED, SOURCED, ILLUSTRATIVE, HEURISTIC, OPEN |

Evidence, implementation stage, and editorial status are independent. New source-coverage briefs remain **planned / proposed / not-measured**. No numerical target is an achieved result. Company names (including UHG/Optum, CVS, Chubb and pharmaceutical organizations) identify research contexts; they do not establish proprietary architectures, incidents or gains. Named architecture/harness assertions in the source tree are verification targets, not endorsed facts.

Obtain primary references when authoring; record the precise claim, publication/event date, version, access/review date, supported scope, limitation and recheck trigger. Verify unstable API, model, hardware, price and regulatory facts then. Historical “2025/2026 frontier” lessons must distinguish their dated setting from current behavior. Unavailable sources stay unavailable; do not manufacture logs, results or citations.

## 7. Acceptance and maintenance

A topic is not complete until an independent reader can reconstruct its reasoning, perform its numerical/trace task, reject an invalid use, diagnose a plausible failure, and defend the decision under changed constraints. The relevant master-template publication checklist is mandatory; filenames and word counts do not certify readiness.

Before publishing: verify local paths/anchors; check prerequisite order; independently recompute examples and relevant boundaries; verify source claims; check all required questions have answers; complete issue counts honestly; reconcile metadata, evidence, stage, review logs and catalog entries; keep final cards last. Record unresolved gaps beside the affected conclusion. Coverage is rechecked when source entries, course homes, templates or dependencies change.
