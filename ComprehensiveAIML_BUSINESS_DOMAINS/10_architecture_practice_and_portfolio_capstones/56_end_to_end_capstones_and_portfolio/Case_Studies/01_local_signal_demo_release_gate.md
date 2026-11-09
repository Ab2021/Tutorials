---
case_id: "CASE-CAPSTONE-001"
case_title: "Choose a release gate for a local signal-to-explanation demonstrator"
case_version: "1.0"
created: "2026-11-09"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "hybrid"
domain: "Edge signal demonstrator and portfolio evidence"
primary_pattern: "evidence-bounded release gate"
secondary_patterns: ["end-to-end contract", "bounded-work"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-09"
owner_role: "capstone technical lead"
source_index: "INDEX.md"
review_trigger: "Hardware, frame protocol, model/index/prompt bundle, scope, or evidence artifacts change."
---

# Choose a Release Gate for a Local Signal-to-Explanation Demonstrator

**Source inventory:** [`directory.txt` lines 1147–1164](../../../../Templates/directory.txt) and the [capstone source brief](../../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#edge-capstone) identify visible topic names only. Their underlying content, a device, traces, and deployment results were not available. This is an illustrative design exercise, not a medical device, clinical workflow, or report of a named system.

## 1. The One-Line Framing

Given framed signal samples and a request for a source-linked assistive explanation, decide whether to release a local demonstrator for a portfolio reviewer, subject to no diagnostic claim, explicit stale/invalid-frame behavior, and a traceable evidence bundle, compared with publishing a UI demo whose inputs and outputs cannot be reconstructed.

The primary success condition is that a reviewer can reproduce the declared trace result and see why a failed or stale path is withheld. The decision is made at each proposed release candidate, not continuously. The reusable pattern is an evidence-bounded release gate. The simpler “does the app run?” framing misses integrity, timing, retrieval provenance, and recovery.

## 2. Context & Background

The illustrative workflow is: a sample source supplies frames; firmware or a fixture producer adds version, sequence, timestamp, payload, and integrity metadata; a local serial service validates and bounds work; an optional retrieval component selects local references; a UI presents status plus a constrained explanation. A portfolio reviewer needs a claim they can inspect, while a user must not mistake a teaching display for medical interpretation.

The baseline is a screenshot plus verbal description. It is fast to create but cannot identify the input, model/configuration, corpus, queue state, or expected output. The proposed workflow saves a release manifest and trace bundle at every gate. The technical lead owns the gate; signal, service, retrieval, UI, and reviewer roles own their respective contracts. No user population, device fleet, geography, regulation, or operational outcome is asserted.

## 3. The Business Question

| Decision contract | Definition |
|---|---|
| Question | Is the evidence sufficient to present this build as a trace-checked local demonstrator, or must it remain design-only? |
| Decision owner | Capstone technical lead with an independent reviewer for evidence scope |
| Trigger/deadline | A proposed portfolio demo; decision before publishing a README, video, or benchmark statement |
| Actions | Block, publish as design-only, publish as trace-checked, or schedule HIL validation |
| Value/downside | A reproducible portfolio is useful; an overstated safety/performance claim damages trust and can mislead |
| Objective | Maximize complete evidence-linked claims, not presentation polish |
| Hard constraints | No clinical diagnosis claim; no hidden invalid/stale state; no fabricated measurement |
| Scope | One local illustrative pipeline and one declared trace contract |
| Out of scope | Care decisions, remote service operation, device approval, and real-user outcomes |

A passing parser metric is a proxy, not the objective. The release decision requires a chain from fixture to display and a visible limitation statement.

## 4. Assumptions & Constraints

| ID | Assumption/constraint | Type | Verification plan | Status | Impact if wrong | Fallback | Owner/trigger |
|---|---|---|---|---|---|---|---|
| A-01 | Frame identity is stable across retries | data | replay duplicate fixture | unverified | duplicate effects may occur | withhold release | service owner / protocol change |
| A-02 | Fixture has a correct expected result | evaluation | independent oracle review | unverified | trace pass is meaningless | label design-only | reviewer / fixture change |
| A-03 | UI renders state version and age | system | pause-input trace | unverified | stale state appears live | block UI claim | UI owner / state change |
| A-04 | Retrieval corpus has attributable sources | evidence | inspect manifest/citations | unverified | explanation is unsupported | show links only | retrieval owner / corpus change |
| C-01 | No diagnostic language | safety | adversarial prompt/review | scenario constraint | user may infer clinical action | refuse/redirect | lead / copy change |
| C-02 | Release bundle is versioned together | reproducibility | compare manifest to artifacts | proposed constraint | rollback cannot recreate behavior | block release | lead / any component change |

The assumptions most likely to reverse the decision are A-02, A-03, and A-04. A real deployment would also need safety, privacy, security, data rights, and policy review; none is inferred here.

## 5. Data, Tools & Method

### 5a. Data Specification

The unit of analysis is a frame, not a patient or diagnosis. Proposed required fields are `stream_id`, `epoch`, `sequence`, `source_time`, `schema_version`, `payload_length`, `payload`, and `integrity`. A trace record adds `received_time`, validator result, queue disposition, release-manifest ID, UI state version, and oracle result. Timestamps use UTC instants but a clock-synchronization claim is not made.

The proposed fixture set includes valid frames, duplicate valid frames, checksum-invalid frames, unsupported schema frames, sequence gaps, deliberately stale frames, and retrieval queries with both supported and unsupported claims. No source dataset, patient data, or production workload exists. A trace comparison counts every fixture in its denominator; quarantined records cannot disappear from the result.

### 5b. Tools & Stack

The architecture is proposed rather than used: a sampling/fixture producer, serial or local transport, validator, bounded queue, local storage for manifests/traces, optional embedding/retrieval service, model runtime, and UI. The visible source names mention STM32, Raspberry Pi 5, UART, RAG, llama.cpp Python, and Streamlit, but this case does not assert versions, compatibility, installation, or measurements. A real build must pin every dependency and device revision.

### 5c. Method Summary

`fixture -> frame validator -> idempotent admission -> bounded queue -> state update -> optional retrieved context -> UI status/explanation -> oracle comparison -> release gate`.

| Candidate | Evidence considered | Strength | Limitation | Status |
|---|---|---|---|---|
| Screenshot-only demo | none beyond appearance | low setup effort | non-reproducible | rejected for trace claim |
| Recorded trace + oracle | illustrative fixtures | repeatable and reviewable | no physical-interface proof | chosen proposed gate |
| HIL validation | no hardware artifact | tests declared path | unavailable now | not tested |
| Production deployment | no workflow/outcome evidence | could measure real operation | safety/governance scope absent | out of scope |

### 5d. Provenance, quality, exclusions and leakage

Every trace must name fixture source, construction date, schema, expected-output author, generator revision, and any redaction. The retrieval corpus must record document identity, access date, license/permission basis, chunking/index version, and citation mapping. Do not tune an explanation against the same hidden evaluation prompts and then call them held out. No protected, clinical, or personal data is used or claimed by this exercise.

### 5e. Reproducibility and method-selection ledger

| Artifact ID | Location | Version/date | Purpose | Accessibility | Limitation |
|---|---|---|---|---|---|
| SRC-01 | [directory inventory](../../../../Templates/directory.txt) | visible export | scope names | local | source bodies unavailable |
| SRC-02 | [capstone brief](../../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#edge-capstone) | planning record | syllabus boundary | local | no implementation evidence |
| RUN-01 | not available | n/a | proposed trace run | unavailable | no execution claimed |

## 6. Analysis Walkthrough

The decisive result is not measured: no artifact supports a trace-checked or HIL release. The walkthrough specifies the evidence that would change that answer.

| Step | Hypothesis | Input | Method | Intermediate output | Interpretation | Status |
|---|---|---|---|---|---|---|
| 1 | frame is well formed | fixture bytes | bound length/schema/integrity | accept/reject reason | transport validity only | proposed |
| 2 | retry is harmless | duplicate sequence | compare committed identity | one logical state update | requires atomic boundary | proposed |
| 3 | stale state is visible | paused input | compare source/UI timestamps | stale marker | UI is honest, not fresh | proposed |
| 4 | explanation is supported | query + corpus | retrieve and citation-check | cited/withheld response | grounding boundary | proposed |
| 5 | release claim is bounded | manifest + oracle | compare expected/actual | gate decision | claim tier selected | proposed |

**Numerical walkthrough — ILLUSTRATIVE.** With 100 samples/frame at 360 samples/s, frame span is `100/360 = 0.277777... s = 277.78 ms`. Add assumed 20 ms local inference and 30 ms serial/service work: `277.78+20+30=327.78 ms`. Units are milliseconds. Independent arithmetic: `1000/360=2.7778 ms/sample`; `100×2.7778=277.78 ms`; plus 50 ms is 327.78 ms. This is a serial-path feasibility estimate, not p99 or measured latency.

If arrival is 4 frames/s and service is 3 frames/s for 120 s, backlog is `(4-3)×120=120 frames`. If service later reaches 5 frames/s while arrival remains 4 frames/s, drain time is `120/(5-4)=120 s`. If service equals arrival, the backlog never drains. Therefore a queue limit cannot substitute for capacity or an overload policy.

**Sensitivity:** reducing the frame to 72 samples gives `72/360=200 ms`; the same assumed path becomes 250 ms. This might meet a 300 ms nominal budget but may violate model-window, overhead, or scheduling requirements. The conclusion reverses only after those omitted terms are measured.

## 7. The Finding

Do not publish this illustrative design as trace-checked, HIL-checked, deployed, clinically useful, or performance-validated. The only defensible current output is a design-only portfolio record with explicit proposed validation. The hand-derived 327.78 ms path estimate rejects a sub-300 ms claim under its stated serial assumptions, but establishes neither device timing nor user experience. A recorded fixture/oracle bundle could justify a narrower trace-checked claim; HIL and operational claims require their own evidence.

## 8. System/Solution Design

```text
[fixture or acquisition] -> [frame builder] -> [validator/admission] -> [bounded queue]
                                                              | reject log
                                                              v
[release manifest] <- [state processor] -> [optional retrieval] -> [UI: status, age, scope]
       ^                     |                       |                    |
       +---------------- [trace + oracle result] ----+--------------------+
```

The validator owns rejection reasons; the processor owns idempotent state transition; the UI owns stale/invalid rendering; retrieval owns citation mapping; the gate owner owns evidence-tier language. A failed request must preserve correlation ID and reject reason without exposing sensitive payloads. Recovery starts from a manifest, a known fixture, and committed sequence state; it does not rely on recollection or screenshots.

A successful request validates, commits one logical effect, updates state version, and records its oracle relation. A failed checksum request is quarantined and shown as invalid rather than silently coerced. A sequence gap produces an explicit degraded state until the loss/recovery policy is applied. A retrieval miss produces no asserted explanation.

## 9. Evaluation & Validation

| Metric | Numerator/denominator | Window/comparator | Guardrail | Status |
|---|---|---|---|---|
| frame acceptance | accepted valid fixtures / all fixtures by class | versioned fixture run / expected validator table | invalid must not be accepted | not executed |
| duplicate safety | duplicate fixtures with one effect / duplicate fixtures | same run / explicit oracle | distinct equal payloads retained | not executed |
| stale disclosure | stale tests with visible marker / stale tests | pause trace / oracle | live marker must not persist | not executed |
| citation support | claims linked to source / displayed factual claims | query suite / manual audit | unsupported claim withheld | not executed |
| path timing | completed correlated paths / all measured paths | defined percentile / baseline | clock and queue semantics recorded | not measured |

Validation sequence: run deterministic fixture tests; inject duplicate, corrupt, gap, and pause behavior; compare output to oracle; inspect manifest completeness; only then schedule HIL validation. A hardware test must identify board/revision, firmware hash, connection settings, measurement method, workload, repetitions, and failure count. None has been done for this case.

## 10. Action Taken

The action in this illustrative case is to block all performance, device, clinical, and deployment claims. Prepare a proposed release bundle containing architecture diagram, interface contract, fixture manifest, expected results, source/corpus ledger, known limitations, and rollback procedure. A reviewer may publish it as a design exercise only after the wording makes the evidence boundary visible.

No code, device, deployment, rollback, or validation run was performed. “Action taken” here describes the proposed decision procedure, not an executed intervention.

## 11. Outcome & Results

**Result:** not measured. No implementation artifact or run supports an outcome. The only checked values are hand-derived arithmetic on constructed inputs: 277.78 ms frame span, 327.78 ms nominal serial-path estimate, and 120-frame overload backlog. These values demonstrate why an end-to-end claim cannot be extrapolated from component estimates.

The intended outcome of a future trace run is a complete denominator and oracle result, not a polished success percentage. An HIL run would still not establish safety, clinical utility, privacy compliance, or general production reliability.

## 12. Failure Modes & What Went Wrong

| Failure | Competing causes | Discriminating signal | Containment | Prevention / verification |
|---|---|---|---|---|
| duplicate display | retry, restart, unstable identity | same epoch/sequence has multiple effects | stop duplicate side effect | replay fixture; exactly one committed effect |
| stale UI | queueing, clock mismatch, browser state bug | compare event/receipt/render times | show stale/degraded state | pause trace; marker visible |
| corrupted semantic value | unit/schema shift, calibration issue, parser bug | fixture/oracle mismatch despite valid CRC | quarantine schema/version | versioned schema test |
| unsupported explanation | retrieval miss, prompt invention, stale corpus | citation audit | withhold text or show source links | adversarial query review |
| timing overclaim | omitted batching/queue/UI, mean/p99 confusion | correlated end-to-end timestamp trace | remove claim | define percentile and denominator |
| rollback mismatch | model/index/prompt/config drift | manifest comparison | restore complete compatible bundle | recovery rehearsal |

## 13. Transferable Lesson

A capstone earns trust by making failure observable and claims falsifiable. The same pattern transfers to ML demos, agent tools, streaming dashboards, and robotics prototypes: name the unit of work, version the control bundle, define the oracle, preserve rejected inputs in the denominator, and do not let a local component measurement impersonate a system outcome.

The counterexample is a system with strong unit tests but no contract across a non-idempotent external side effect. Its components may all pass while a retry charges twice or renders stale information as current. The transfer requires changing the identity, owner, and safety boundary for each domain rather than copying a serial-frame protocol blindly.

## 14. Connections & Cross-References

Related reusable method: [Capstone Evidence Rubric](../Cheat_Sheets/01_Capstone_Evidence_Rubric.md), especially Sections 4, 5, 8, and 12. Related source scope: [edge-capstone brief](../../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#edge-capstone). Open questions: exact hardware path, supported runtime versions, data/source rights, corpus provenance, user workflow, and HIL test availability.

### Interview pack

1. **What makes this an end-to-end case?** The declared boundary starts at a fixture/acquisition contract and ends at a UI decision with oracle and recovery record. A UI alone is not end-to-end.
2. **Compute the frame span.** `100/360 s = 277.78 ms`; uniform sampling is required. This leaves 22.22 ms under a 300 ms target before all later stages.
3. **Why does a checksum not prove a correct signal?** It protects selected bytes, not semantic units, calibration, source authenticity, or clinical meaning. Use schema and oracle checks.
4. **How do you distinguish stale UI from a slow producer?** Correlate source event, receipt, processing, and render timestamps with one ID; queue depth alone is ambiguous.
5. **What is the wrong fix for duplicates?** Dropping equal payloads. Different events can have identical payloads; use stable identity/sequence and atomic state transition.
6. **When may a recorded trace be shown?** As a trace-checked simulation only after its fixture, oracle, manifest, and limitations are available. It is not HIL evidence.
7. **How would a changed requirement affect design?** A clinical decision requirement invalidates this scope; stop and obtain the necessary clinical, safety, and regulatory design process.
8. **What does a release manifest include?** Code/config/model/prompt/index versions, interface/schema, fixtures, expected outputs, evaluator revision, environment, and unresolved limitations.
9. **Why is a 50 ms component sum not end-to-end latency?** It excludes frame collection, queueing, scheduling, UI, and tail distribution. Here the frame alone is 277.78 ms.
10. **How do you evaluate a retrieval explanation?** Test attributed claims against a versioned corpus, retain query/retrieval/prompt/response, measure supported claims over a stated denominator, and withhold unsupported claims.
11. **What would prove recovery?** A replay from known input under the recorded manifest reaches the oracle without duplicate external effects; a restart alone proves little.
12. **What is the key leadership decision?** Narrow the claim rather than widen it: block promotion when evidence is missing and name the next smallest test that can change the decision.

## 15. Quick Reference Card

| Field | Case answer |
|---|---|
| Decision | publish design-only vs trace-checked demonstrator |
| Input/output | framed signal/fixture to status and constrained explanation |
| Baseline | screenshot-only demo |
| Binding constraints | no diagnosis, no hidden invalid/stale state, no fabricated result |
| Checked arithmetic | 100/360=277.78 ms; +20+30=327.78 ms; 120-frame backlog under stated assumptions |
| Evidence | illustrative design; no execution, HIL, or deployment measured |
| Gate | require versioned fixture, oracle, manifest, and complete denominator for trace-checked claim |
| Review trigger | any protocol, hardware, model/index/prompt, source, or scope change |
