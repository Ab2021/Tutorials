---
case_id: "CASE-CAPSTONE-001"
case_title: "Choose a release gate for a local signal-to-explanation demonstrator"
case_version: "1.1"
created: "2026-11-09"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "hybrid"
domain: "Local signal demonstrator and portfolio evidence"
primary_pattern: "evidence-bounded release gate"
secondary_patterns: ["end-to-end contract", "bounded-work"]
evidence_level: "illustrative"
implementation_stage: "proposed"
editorial_status: "draft"
source_basis: "illustrative design exercise; visible inventory and planning brief"
as_of_date: "2026-11-10"
owner_role: "capstone technical lead"
source_index: "INDEX.md"
review_trigger: "Any protocol, clock, hardware, runtime, corpus, prompt, UI state, or evidence artifact change."
---

# Choose a Release Gate for a Local Signal-to-Explanation Demonstrator

**Source and evidence declaration.** This is an ILLUSTRATIVE design exercise. The visible [directory inventory](../../../../Templates/directory.txt#L1147) and [edge-capstone planning brief](../../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#edge-capstone) establish only syllabus scope. Source bodies, hardware, device traffic, application runs, UI recordings, corpus documents, and performance measurements are OPEN. Nothing here is a diagnostic, clinical, device, deployment, or named-application claim.

## 1. The One-Line Framing

> Given one constructed framed signal event and a constrained request for a local, source-linked assistive explanation, the capstone technical lead decides whether a portfolio artifact is DESIGN-only, TRACE-CHECKED, or eligible to schedule HIL validation, for a reviewer, subject to visible invalid/stale states and a reconstructable evidence bundle, compared with a screenshot-only demo.

The unit is a frame and its derived UI state, never a person, diagnosis, or clinical conclusion. The output is a release-tier decision plus exact permitted wording. The primary success condition is not that a screen appears; it is that an independent reviewer can reconstruct the declared trace, account for every input, and see why withheld paths remain withheld.

The reusable pattern is an evidence-bounded release gate. Secondary patterns are an end-to-end contract and bounded work. A screenshot-only baseline cannot establish bytes accepted, sequence policy, queue age, retrieval source, configuration bundle, or oracle result.

**Evidence boundary.** `DESIGN` permits architecture and proposed contracts. `TRACE-CHECKED` requires a recorded deterministic fixture, oracle, complete manifest, and reviewer-inspectable trace; it does not establish physical transport. `HIL-VALIDATED` requires an exercised declared hardware path and setup record; it does not establish user, safety, or clinical outcomes. No tier is currently earned by this manuscript.

## 2. Context & Background

The proposed data plane is acquisition or fixture bytes to a frame builder, local serial reader, validator, bounded queue, state processor, optional retrieval, and UI. The control plane carries pinned schema, thresholds, model/prompt/index identifiers, and UI copy. The evidence plane carries fixture identity, hashes, timestamps, dispositions, and oracle decisions. Separating these planes stops a polished output from becoming evidence for an invisible input.

| Context field | Bounded statement | Evidence status |
|---|---|---|
| Stakeholders | technical lead decides; reviewer checks scope; component owners supply contracts | illustrative roles |
| Baseline | screenshot plus verbal tour | considered baseline; not tested |
| Before decision | candidate manifest and declared fixture are assembled | proposed workflow |
| After decision | only tier-appropriate portfolio wording may be released | proposed control |
| Scale | one local trace; no user population or throughput claim | scope constraint |
| Boundary | local design record; no remote service, care workflow, or device claim | scope constraint |
| Timing | gate occurs before README, video, benchmark, or portfolio statement | proposed process |

The visible inventory names UART, STM32, Raspberry Pi 5, RAG, llama.cpp Python, Streamlit, testing, profiling, and portfolio topics. Those names justify discussing interfaces and validation questions, not asserting that a particular board, application, model, library, or UI exists or interoperates.

## 3. The Business Question

| Decision contract | Definition |
|---|---|
| Question | Is the evidence complete enough for a trace-checked local-demonstrator claim, or must the artifact remain design-only? |
| Decision owner | Capstone technical lead; independent reviewer owns evidence-scope challenge. |
| Trigger | Before public portfolio wording or a claimed demonstration result. |
| Actions | Block; publish DESIGN wording; publish TRACE-CHECKED wording; schedule HIL validation. |
| Baseline | Screenshot-only presentation with no reconstructable input-to-output evidence. |
| Objective | Maximize complete, inspectable claims; not visual polish or apparent responsiveness. |
| Hard constraints | No diagnostic/clinical language; invalid and stale states visible; no fabricated run. |
| Scope | One declared trace, one version bundle, one local UI-state contract. |
| Out of scope | Safety certification, efficacy, privacy review, device approval, remote operation, and real-user outcomes. |

The proxy metric is trace completeness: eligible fixture records with a disposition, state transition, and oracle result divided by all fixture records declared before evaluation. A visually attractive UI can score well on presentation while failing the objective, so it is not a gate metric.

## 4. Assumptions & Constraints

| ID | Assumption or constraint | Type | Check | Status | If wrong | Fallback / owner |
|---|---|---|---|---|---|---|
| A-01 | Frame bytes and CRC algorithm are specified together | protocol | fixture parser and CRC oracle | OPEN | integrity decision is ambiguous | block trace claim / protocol owner |
| A-02 | Epoch plus sequence is stable across retry | state | duplicate replay trace | OPEN | side effect may repeat | withhold / service owner |
| A-03 | Source and render clock semantics are declared | time | age calculation from trace | OPEN | freshness label is misleading | show unknown / UI owner |
| A-04 | Queue capacity and overflow rule are bounded | capacity | admission trace | OPEN | work/evidence may vanish | block / service owner |
| A-05 | Retrieval documents are attributable | provenance | document and citation ledger | OPEN | explanation is unsupported | abstain / retrieval owner |
| C-01 | No diagnosis, treatment, or clinical recommendation | scope | copy and prompt audit | binding | unsafe scope expansion | refuse / lead |
| C-02 | Bundle versions move together | reproducibility | manifest equality check | proposed | replay cannot reproduce behavior | block / lead |
| C-03 | Invalid and stale UI states remain visible | honesty | render-state oracle | OPEN | viewer sees false freshness | block UI claim / UI owner |

A-01, A-03, and A-05 can reverse a release decision most directly. A CRC can be correct over semantically wrong units; a render timestamp without clock meaning cannot prove freshness; retrieved text without a source ledger cannot support an explanation.

## 5. Data, Tools & Method

### 5a. Data Specification

A frame is immutable bytes plus parsed metadata. `stream_id` is an ASCII identifier. `epoch` is a nonnegative sender restart generation. `sequence` is a nonnegative integer monotonic within `(stream_id, epoch)`. `source_time_ms` is a declared UTC millisecond instant, not proof of clock synchronization. `sample_rate_hz` is samples/s. `sample_unit` names the payload unit and is not interpreted clinically. `payload_count` is samples. `payload` is opaque bytes at this boundary. `crc32` is an unsigned 32-bit checksum over a precisely declared canonical byte range.

| Field | Type / unit | Admission rule | Evidence recorded |
|---|---|---|---|
| stream_id | string | allowlisted nonempty identifier | exact value |
| epoch | uint32 | known or explicit reset policy | exact value |
| sequence | uint64 | greater than committed sequence | exact value and predecessor |
| source_time_ms | int64 UTC ms | within declared representable range | source, receipt, render times |
| schema_version | string | equals supported version | supported-set revision |
| sample_rate_hz | positive decimal samples/s | equals declared fixture contract | value and unit |
| sample_unit | string | equals declared fixture contract | value |
| payload_count | uint16 samples | 1 through maximum 100 | count |
| payload_length | uint16 bytes | equals encoded payload length and maximum 256 | claimed/actual bytes |
| payload | bytes | opaque after structural checks | payload hash, not contents |
| crc32 | uint32 | matches canonical frame bytes | expected/actual CRC |
| correlation_id | string | unique per arrival attempt | arrival identity |

The constructed trace uses schema `sigframe/1`, `stream_id=fixture-A`, `epoch=7`, `sample_rate_hz=360 samples/s`, `sample_unit=arb-unit`, `payload_count=72`, maximum payload 256 bytes, and a FIFO with capacity 2 frames. These are illustrative contract values, not device settings.

### 5b. Tools & Stack

| Proposed component | Responsibility | What is deliberately not claimed | Owner |
|---|---|---|---|
| fixture producer | emit canonical bytes and expected dispositions | firmware behavior | fixture owner |
| frame builder | serialize declared header and CRC coverage | a specific MCU implementation | protocol owner |
| serial reader | turn byte stream into candidate frames | a physical serial run | service owner |
| validator | apply shape, version, unit, CRC, and sequence rules | semantic signal correctness | service owner |
| bounded FIFO | retain at most two admitted frames | loss-free overload handling | service owner |
| state processor | atomically commit last sequence and UI state version | a deployed database | service owner |
| retrieval adapter | return versioned local citations or abstain | corpus quality or model performance | retrieval owner |
| UI | show state, age, invalid/degraded/stale labels | a tested visual render | UI owner |
| manifest writer | bind inputs and versions to a trace | an actual repository or run | release owner |

### 5c. Method Summary

`canonical fixture -> framing -> structural validation -> CRC validation -> sequence decision -> bounded admission -> atomic state commit -> age classification -> retrieval with provenance check -> UI state -> oracle comparison -> release gate`.

| Candidate | Evidence considered | Strength | Limit | Decision |
|---|---|---|---|---|
| Screenshot-only | appearance only | fast to explain | no input/oracle/manifest | rejected for trace claim |
| Recorded fixture plus oracle | constructed deterministic trace | inspectable logic | no physical path proof | required for future trace claim |
| HIL exercise | no artifact available | tests declared physical path | not available now | not tested |
| Production operation | no workflow or outcome evidence | could measure operations | outside scope | excluded |

**Admission invariant.** A frame changes committed state only when schema, units, length, CRC, and ordering all pass and queue capacity admits it. The commit records `(stream_id, epoch, sequence, payload_hash, state_version)` atomically. A duplicate never increments `state_version`; a corrupt or gapped frame never advances committed sequence.

### 5d. Provenance, Quality, Exclusions, and Leakage

Fixture inputs are constructed and must carry author, creation date, canonical encoding revision, expected result author, and payload hash. The trace is not drawn from a device, person, patient, clinical record, or production log. Payload values are intentionally opaque because a checksum cannot validate calibration, sensor placement, biological meaning, or correctness of a signal-processing algorithm.

Retrieval is optional. A factual explanation is eligible only when each factual sentence maps to a cited chunk ID whose document ID, document hash, access date, permission basis, chunker revision, embedding/index revision, and retrieval rank are in the evidence bundle. A retrieval miss, missing mapping, or conflict produces an abstention, not a free-form substitute. Evaluation prompts must be separated from explanation drafting; otherwise hand-edited wording can leak the expected answer.

### 5e. Reproducibility and Method-Selection Ledger

| ID | Location / identity | Version or date | Purpose | Accessibility | Limitation |
|---|---|---|---|---|
| SRC-01 | directory.txt lines 1147–1164 | visible inventory | topic scope | local | bodies unavailable |
| SRC-02 | 00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#edge-capstone | planning brief | scope boundary | local | planning is not implementation evidence |
| FIX-01 | proposed `fixture-A-v1` | constructed | trace input | not stored | no run artifact |
| ORA-01 | proposed rule table below | constructed | expected dispositions | in manuscript | not executed |
| MAN-01 | proposed manifest schema | constructed | reconstruction contract | in manuscript | no emitted manifest |
| RUN-01 | not available | n/a | trace execution | unavailable | trace claim unavailable |
| HIL-01 | not available | n/a | hardware validation | unavailable | HIL claim unavailable |

A future manifest must include: manifest ID; creation time; fixture hashes; canonicalization revision; schema revision; CRC algorithm and covered bytes; queue capacity/policy; service revision; model ID/hash if used; prompt template hash if used; corpus/chunker/embedder/index IDs if used; UI revision; oracle revision; environment identifiers; all dispositions; and unresolved limitations. A manifest is complete only if a reviewer can compare every listed value to the supplied artifact; a list of product names is not a manifest.

## 6. Analysis Walkthrough

**Decisive current result:** no trace has been executed, so the release remains DESIGN-only. The following is a manually checked illustrative oracle and design trace. It demonstrates what a future trace must contain; it is not evidence of an app, serial service, UI, device, or HIL run.

### Declared illustrative trace

At run start, committed sequence for `(fixture-A,7)` is 40, queue depth is 0/2, state version is 18, and the display contains no current frame. The service receipt clock and UI render clock are assumed to share a declared monotonic reference only for this constructed calculation. The stale threshold is `A_stale=1000 ms`; `500 < age <= 1000 ms` is `DEGRADED`; `age > 1000 ms` is `STALE`. These thresholds are design choices, not safety limits.

| Arrival | Frame identity | Structural / CRC condition | Expected queue disposition | Expected committed state | Expected UI state |
|---|---|---|---|---|
| T1 | (fixture-A,7,41) | valid; 144-byte payload; CRC matches | admit then service | commit 41; state v19 | CURRENT after render |
| T2 | (fixture-A,7,41) | valid duplicate; same payload hash | duplicate-drop; no queue slot | remain 41; state v19 | CURRENT unchanged; duplicate count visible |
| T3 | (fixture-A,7,43) | valid CRC but expected 42 | gap-quarantine; no queue slot | remain 41; state v19 | DEGRADED; gap visible |
| T4 | (fixture-A,7,42) | payload says 144 bytes but 143; CRC mismatch | corrupt-quarantine; no queue slot | remain 41; state v19 | INVALID input evidence; prior state remains degraded |
| T5 | no new frame | last committed source at 1000 ms; render check 2301 ms | no admission | remain 41; state v19 | STALE; explanation withheld |

**T1 valid path.** The parser reads `payload_count=72`, `sample_rate_hz=360`, `sample_unit=arb-unit`, and `payload_length=144`. The actual byte count is also 144. The CRC recomputed over the declared header-and-payload byte range matches the transmitted value. Sequence 41 is exactly `last_committed+1`, so it is admitted. After service, the atomic transaction stores sequence 41 and increments state version from 18 to 19. A future trace must retain both arrival and commit records; an accepted parser log alone cannot establish the commit.

**T2 duplicate path.** The frame repeats epoch 7 and sequence 41 with the same payload hash. It may be a retry, a reader reconnect replay, or a producer resend; this oracle does not diagnose which. Because `sequence <= last_committed`, it is a duplicate and creates no queue record, state transition, or explanation request. Dropping frames merely because their payload values are equal would be wrong: two distinct sequences may legitimately encode equal payloads.

**T3 gap path.** Sequence 43 has a valid CRC, but `43 > 41+1`; integrity is not contiguity. The service records missing sequence 42, does not advance the committed sequence, and marks the UI `DEGRADED`. The recovery policy is bounded: request or wait for the missing sequence only if the protocol supports it; otherwise require an explicit epoch reset/resynchronization record. It must not silently call sequence 43 current.

**T4 corrupt path.** This arrival claims 144 payload bytes but supplies 143, and its CRC mismatch is expected. The validator quarantines it with both reasons in the trace. It does not consume a queue slot or repair bytes. The UI may show an `INVALID_INPUT` indicator associated with the arrival while continuing to show that the last committed state is degraded; it must not paint the corrupt candidate as a valid value.

**T5 stale path.** With no new valid commit after source time 1000 ms, a render check at 2301 ms gives age 1301 ms. Because `1301 > 1000`, the state is `STALE`, not current. The explanation panel changes to an abstention message: “No current validated frame is available; explanation withheld.” The age label is a computed display property and does not prove wall-clock synchronization in a future system.

### Sampling, transport, queue, and age budget

**Validity before use.** The following arithmetic assumes uniform 360 samples/s acquisition, nonoverlapped frame collection, a 72-sample frame, and a serial additive path. It is illustrative, excludes retries, operating-system scheduling, garbage collection, rendering variability, physical baud rate, and percentile behavior. It is therefore a design budget, never an observed latency or real-time guarantee.

```text
FORMULA [B-01]: T_collect = n / f_s
WHERE: T_collect [s]; n [samples/frame]; f_s [samples/s].
CONSTRAINT: contiguous uniform sampling; no overlap, drop, resampling, or drift.
NUMERICAL EXAMPLE: n=72 samples; f_s=360 samples/s;
                   T_collect=72/360=0.2 s=200 ms.
INTERPRETATION: collection alone consumes 200 ms of source time.
```

```text
FORMULA [B-02]: T_transport = 10 * (H + P + C) / R
WHERE: T_transport [ms]; H header bytes; P payload bytes; C CRC bytes;
       R serial bits/s; factor 10 assumes 8N1 framing.
CONSTRAINT: one direction, no escaping, retransmission, contention, or driver delay.
NUMERICAL EXAMPLE: H=24 bytes; P=144 bytes; C=4 bytes; R=115200 bits/s;
                   T_transport=10*(24+144+4)/115200*1000=14.9306 ms.
INTERPRETATION: wire-time estimate is about 14.93 ms, not service latency.
```

```text
FORMULA [B-03]: T_age = T_collect + T_transport + T_service + T_queue + T_render
WHERE: all terms [ms]; T_queue is waiting before service.
CONSTRAINT: same clock convention for source and render; serial additive path.
NUMERICAL EXAMPLE: 200 + 14.93 + 15 + 0 + 20 = 249.93 ms.
INTERPRETATION: zero-queue nominal age is below a 300 ms illustrative budget by 50.07 ms.
```

```text
FORMULA [B-04]: T_queue_max = K / mu * 1000
WHERE: K [frames] is queued frames ahead; mu [frames/s] is service rate.
CONSTRAINT: FIFO, fixed service time, no arrivals ahead, no preemption.
NUMERICAL EXAMPLE: K=2 frames; mu=4 frames/s; T_queue_max=2/4*1000=500 ms.
INTERPRETATION: a full two-frame waiting position makes age 749.93 ms under B-03.
```

The transport denominator is bits/s, so `172 bytes × 10 bits/byte = 1720 bits`; `1720 / 115200 s = 0.0149306 s = 14.9306 ms`. The frame collection denominator is samples/s, so `72 samples / (360 samples/s) = 0.2 s`. Both dimensional checks pass. The age sum uses milliseconds only.

| Sensitivity | Changed input | Recomputed result | Consequence |
|---|---|---|---|
| Lower baud | R=57600 bits/s | transport=29.86 ms; zero-queue age=264.86 ms | still below 300 ms only in this model |
| Larger frame | n=100 samples | collection=277.78 ms; zero-queue age=327.71 ms | nominal 300 ms claim fails |
| One frame ahead | K=1, mu=4/s | queue=250 ms; age=499.93 ms | current label allowed only if threshold exceeds this |
| Service slows | K=2, mu=2/s | queue=1000 ms; age=1249.93 ms | stale under 1000 ms policy |
| Faster render | render=5 ms | zero-queue age=234.93 ms | does not fix collection or backlog |

**Recovery calculation.** If arrivals are `lambda=4 frames/s` and service falls to `mu=3 frames/s` for `t=120 s` from an empty queue, backlog is `(4-3)*120=120 frames`. A two-frame queue cannot retain the remaining 118 frames. If, after stopping admissions or gaining `mu_recover=6 frames/s` with arrivals still 4/s, recovery time is `120/(6-4)=60 s`. If service remains 4/s, spare capacity is zero and recovery never completes in this fluid model. The required recovery action is therefore explicit backpressure/degraded state and evidence of drops, not a claim that bounded memory preserves all work.

### Gate reasoning

| Gate condition | Trace requirement | Current manuscript status | Decision effect |
|---|---|---|---|
| Frame contract | schema, units, CRC coverage, ordering policy | specified only | not trace-checked |
| Trace oracle | declared fixture and every disposition | manually constructed only | not executed |
| Manifest | all bundle identifiers and hashes | schema specified only | not reproducible as a run |
| Retrieval | source mapping or abstention | policy specified only | no explanation-quality claim |
| UI | recorded visual state evidence | none | no UI rendering claim |
| HIL | board/firmware/wiring/measurement record | none | no HIL claim |

The gate decision is DESIGN-only. The calculations and oracle are checked design logic, but no external run artifact binds them to implementation. It would be misleading to relabel a manually reasoned trace as TRACE-CHECKED.

## 7. The Finding

Publish only a DESIGN-only portfolio record if it retains its limitation statement and does not call the demonstrator trace-checked, HIL-validated, responsive, clinically useful, or deployed. The illustrative budget gives 249.93 ms with no queue and 749.93 ms with two frames ahead at 4 frames/s service; a 100-sample frame reverses a nominal 300 ms conclusion at 327.71 ms. These values are derived from constructed assumptions, not measured performance. TRACE-CHECKED requires an actual versioned trace, complete oracle denominator, and manifest; HIL requires a separately documented physical-path exercise.

## 8. System/Solution Design

```text
[fixture/acquisition] -> [canonical frame] -> [validator] -> [bounded FIFO] -> [atomic state] -> [age classifier] -> [UI]
                                     |                 |                  |                    |              |
                                     v                 v                  v                    v              v
                              [quarantine log]   [overflow record]   [trace ledger]       [abstain]    [stale/invalid]
                                                                              |
                         [manifest] <--- [oracle comparator] <--- [retrieval provenance checker]
```

The frame builder owns canonical byte order and CRC coverage. The validator owns reasons for rejection; it does not reinterpret signal values. The serial service owns queue admission and the atomic commit boundary. The UI owns explicit rendering of `CURRENT`, `DEGRADED`, `STALE`, and `INVALID_INPUT`; a state is not current merely because an older chart remains visible. The retrieval adapter owns cited context or abstention. The release owner owns claim wording and cannot waive missing evidence.

| UI state | Preconditions | Required display | Explanation behavior | Recovery |
|---|---|---|---|---|
| CURRENT | last valid committed frame; age <=500 ms | state version, age, manifest short ID | only cited/attributable assistive text | new valid contiguous frame |
| DEGRADED | gap, overflow, or age 501–1000 ms | reason, age, last valid sequence | withhold new factual explanation | validated resync or fresh frame |
| STALE | age >1000 ms | stale label and last update age | withhold explanation | new valid frame and recalculated age |
| INVALID_INPUT | current arrival fails validation | invalid reason and correlation ID | no explanation from invalid arrival | submit valid frame |
| ABSTAIN | retrieval provenance incomplete/miss | source unavailable statement | no factual completion | supply mapped source |

A successful path validates then commits exactly one logical state change. A failed CRC path is quarantined and cannot alter the prior state. A gap does not become contiguous through timeout alone. A full queue reports its overflow policy and affected correlation IDs; it cannot silently discard evidence. Recovery begins from a compatible manifest, known committed sequence, and fixture/oracle set. It ends only after replay reaches the declared oracle without duplicate state effects.

## 9. Evaluation & Validation

| Metric | Numerator / denominator | Window / comparator | Guardrail | Evidence status |
|---|---|---|---|---|
| validator correctness | fixtures matching declared disposition / all declared fixtures | versioned fixture set / oracle | corrupt never accepted | not executed |
| duplicate safety | duplicate arrivals with zero extra commits / duplicate arrivals | trace / atomic invariant | distinct equal payloads retained | not executed |
| gap disclosure | gap fixtures rendered degraded / gap fixtures | trace / UI oracle | sequence not advanced | not executed |
| stale disclosure | age checks rendered stale / checks above threshold | clock-defined trace / UI oracle | prior chart not marked current | not executed |
| retrieval support | displayed factual claims with citation map / factual claims displayed | versioned prompt suite / human audit | unsupported claim withheld | not executed |
| manifest completeness | required fields present and hash-verifiable / required fields | one release candidate / manifest schema | missing field blocks tier | not executed |
| transport and age | correlated paths within stated bound / all completed correlated paths | defined percentile / no baseline | clock semantics and tails captured | not measured |

Trace validation is a deterministic software/fixture check. HIL validation is separate: it must name board revision, firmware revision/hash, acquisition source, wiring/connection settings, serial parameters, host/runtime revision, timing instrumentation, repetitions, observed failures, raw trace locations, and operator. None of these artifacts is available, so HIL status is OPEN.

UI visual validation is also OPEN. The state table specifies intended text and behavior but no browser, screenshot, recording, accessibility review, or visual rendering was inspected. A state transition in a backend trace cannot prove that a human-facing indicator appeared, was legible, or remained visible.

## 10. Action Taken

The only current action is the proposed release decision: block all performance, device, clinical, HIL, UI-tested, corpus-quality, and deployment claims; permit a clearly labeled DESIGN-only artifact. No code, manifest, fixture file, device setup, UI, retrieval corpus, or validation run was created or executed by this case.

| Proposed next action | Owner | Required input | Stop condition | Claim unlocked if passed |
|---|---|---|---|---|
| freeze fixture/oracle | fixture owner | canonical bytes and expected dispositions | hash mismatch or ambiguity | none alone |
| implement manifest writer | release owner | component identifiers and hashes | missing required field | none alone |
| execute deterministic trace | service owner | frozen fixture/oracle/manifest | any undocumented disposition | TRACE-CHECKED candidate |
| inspect UI evidence | UI owner | recorded state transitions | state hidden or ambiguous | UI behavior statement only |
| schedule HIL | hardware owner | declared setup and measurement plan | setup/artifact absent | HIL candidate |
| independent scope review | reviewer | bundle plus permitted wording | claim outruns evidence | publication approval |

## 11. Outcome & Results

**Outcome status: not measured.** No implementation artifact supports an observed result. The checked outputs in this document are: a constructed five-arrival oracle; `T_collect=200 ms`; `T_transport=14.9306 ms`; zero-queue modeled age `249.9306 ms`; two-frame modeled waiting age `749.9306 ms`; and 120-frame fluid backlog under the stated overload. These are illustrative derived calculations, not a benchmark, UI observation, or device result.

| Claim ID | Exact bounded claim | Evidence type | Supports | Does not establish |
|---|---|---|---|---|
| CLM-01 | A screenshot alone is insufficient for a trace-tier claim. | derived | need for fixture/oracle/manifest | implementation behavior |
| CLM-02 | The constructed T1–T5 oracle specifies valid, duplicate, gap, corrupt, and stale outcomes. | illustrative | future expected behavior | that any software produced it |
| CLM-03 | The stated additive model is 249.93 ms without queue. | illustrative, arithmetic checked | budget reasoning | latency or percentile |
| CLM-04 | A two-frame position at 4/s adds 500 ms under B-04. | illustrative, arithmetic checked | queue sensitivity | queue implementation |
| CLM-05 | Current gate is DESIGN-only. | derived from OPEN artifacts | permitted wording | future trace/HIL result |

## 12. Failure Modes & What Went Wrong

| Failure mode | Competing causes | Discriminating check | Containment | Recovery / prevention |
|---|---|---|---|---|
| Duplicate state change | retry; reconnect replay; non-atomic commit | same epoch/sequence has >1 state version | stop side effects; flag trace | atomic identity commit; replay oracle |
| CRC-valid wrong meaning | unit mismatch; calibration error; wrong schema semantics | unit/schema ledger conflicts with fixture contract | withhold interpretation | versioned units and semantic oracle |
| Gap masked as current | loss; reader parse loss; sender restart | sequence jumps without epoch reset | show degraded | resync rule; do not advance sequence |
| Corrupt frame painted valid | length bug; CRC scope mismatch; UI mapping error | validator reason differs from UI status | quarantine candidate | shared state enum and UI oracle |
| Stale view appears live | queueing; clock mismatch; frontend cache | source/receipt/render times disagree | show stale/unknown | declared clock semantics; pause trace |
| Queue silently loses evidence | overflow; cancellation; unbounded producer | arrival count differs from dispositions | block release | bounded policy and overflow records |
| Unsupported explanation | retrieval miss; stale index; prompt invention | claim has no chunk mapping | abstain | citation mapping audit |
| Manifest cannot replay | drift in schema/model/prompt/index/config | hash mismatch or missing field | block promotion | immutable compatible bundle |
| Timing overclaim | component mean presented as end-to-end; omitted collection | missing correlation timestamps | remove performance wording | path definition and denominator |
| HIL overclaim | desktop fixture confused with hardware exercise | missing setup/raw hardware trace | remove HIL wording | named setup and measured trace |

The failure table is a design diagnostic, not an incident record. No row reports a discovered defect in an application because no application run was provided. The recovery boundary is intentionally small: preserve the rejected arrival and its reason, preserve committed state, reveal degraded/stale status, and require compatible evidence before promotion.

## 13. Transferable Lesson

A credible capstone is a falsifiable decision record: it identifies the work unit, state identity, ownership boundary, complete denominator, oracle, and claim tier. The pattern transfers to streaming dashboards, agents, robotics, and ML demos by replacing the frame identity and safety boundary while retaining the rule that a retry must not create an unexplained second effect.

A counterexample is a dashboard with fast local inference, attractive visualization, and unit tests but no source timestamp, queue policy, or render evidence. It may be technically interesting yet cannot truthfully claim freshness. Another counterexample is a sourced-looking explanation whose corpus version is unknown; citations then decorate rather than support a claim.

Transfer does not mean copying serial rules into every domain. A payment system needs transaction/idempotency semantics; a batch model needs data cut-off and cohort provenance; an agent needs tool authorization and output review. In each case, the release gate must follow the consequential state change, not the presentation layer.

## 14. Connections & Cross-References

Related method: [Capstone Evidence Rubric](../Cheat_Sheets/01_Capstone_Evidence_Rubric.md), especially its evidence tiers, frame/queue formulas, and recovery invariant. Related planning boundary: [edge-capstone brief](../../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#edge-capstone). The linked materials are planning/rubric context; neither supplies a run artifact for this case.

### Interview pack

**1. Question — What is the bounded decision?**
- **Checked answer:** Whether evidence supports DESIGN-only wording or a future trace-tier candidate; not whether a signal is clinically meaningful.
- **Reasoning:** Naming the decision prevents tool lists from substituting for an owner/action.
- **Follow-up:** What artifact or changed assumption would make your answer fail?
- **Acceptable alternatives:** A narrower release-gate phrasing is acceptable.
- **Red flags:** Answering “build the app” is too broad.
- **Scoring:** Reject clinical or deployment framing.
- **Answer support:** Section 3 decision contract.

**2. Question — Compute the collection time for 72 samples at 360 samples/s.**
- **Checked answer:** 72/360 s=0.2 s=200 ms.
- **Reasoning:** Samples cancel, leaving seconds; this assumes uniform contiguous sampling.
- **Follow-up:** What artifact or changed assumption would make your answer fail?
- **Acceptable alternatives:** Equivalent 0.2000 s is correct.
- **Red flags:** Calling it 72 ms confuses count with time.
- **Scoring:** Require units and assumption.
- **Answer support:** B-01.

**3. Question — Compute the 8N1 wire-time estimate for 172 bytes at 115200 bits/s.**
- **Checked answer:** 10×172/115200×1000=14.9306 ms.
- **Reasoning:** 8N1 carries ten bits per byte; bytes must become bits before division.
- **Follow-up:** What artifact or changed assumption would make your answer fail?
- **Acceptable alternatives:** A stated alternative framing is acceptable if recomputed.
- **Red flags:** 172/115200 ms omits framing and unit conversion.
- **Scoring:** Require denominator and framing convention.
- **Answer support:** B-02.

**4. Question — Why does a matching CRC not prove a correct signal?**
- **Checked answer:** It only checks selected transmitted bytes; units, calibration, source authenticity, and semantic meaning can still be wrong.
- **Reasoning:** Integrity and semantic validity have different oracles.
- **Follow-up:** What artifact or changed assumption would make your answer fail?
- **Acceptable alternatives:** A cryptographic-authentication distinction is acceptable.
- **Red flags:** Saying “CRC proves data is good” is unsafe.
- **Scoring:** Require at least two unprotected properties.
- **Answer support:** Sections 4 and 5d.

**5. Question — What is the correct duplicate rule?**
- **Checked answer:** For the same stream and epoch, sequence <= committed sequence is recorded as duplicate and creates no commit.
- **Reasoning:** Stable identity must be checked at the atomic state boundary.
- **Follow-up:** What artifact or changed assumption would make your answer fail?
- **Acceptable alternatives:** An idempotency key with equivalent atomic semantics is acceptable.
- **Red flags:** Dropping equal payload values can erase distinct events.
- **Scoring:** Require identity plus no second effect.
- **Answer support:** Admission invariant.

**6. Question — Diagnose a valid-CRC sequence 43 after committed sequence 41.**
- **Checked answer:** Treat it as a gap, keep committed sequence 41, mark degraded, and apply explicit resynchronization policy.
- **Reasoning:** CRC validity does not establish contiguity.
- **Follow-up:** What artifact or changed assumption would make your answer fail?
- **Acceptable alternatives:** A protocol-specific retransmit is acceptable if recorded.
- **Red flags:** Advancing straight to 43 hides loss.
- **Scoring:** Reject silent acceptance.
- **Answer support:** T3 trace.

**7. Question — What should UI show after an invalid candidate when the last valid state is already degraded?**
- **Checked answer:** Show invalid input reason for the candidate and retain degraded status for the last committed state; do not call either current.
- **Reasoning:** Arrival validity and displayed-state freshness are separate dimensions.
- **Follow-up:** What artifact or changed assumption would make your answer fail?
- **Acceptable alternatives:** A dedicated combined-state display is acceptable if both facts remain visible.
- **Red flags:** Replacing the old state with the corrupt value is wrong.
- **Scoring:** Require both identities.
- **Answer support:** T4 and Section 8.

**8. Question — Calculate modeled age with two frames ahead, service 4 frames/s, and zero-queue age 249.93 ms.**
- **Checked answer:** Waiting is 2/4×1000=500 ms, so modeled age is 749.93 ms.
- **Reasoning:** Queue time adds to the stated additive path under FIFO assumptions.
- **Follow-up:** What artifact or changed assumption would make your answer fail?
- **Acceptable alternatives:** 749.9306 ms is equivalent.
- **Red flags:** Subtracting queue time reverses causality.
- **Scoring:** Require calculation and assumptions.
- **Answer support:** B-03/B-04.

**9. Question — What changed constraint reverses the nominal 300 ms conclusion?**
- **Checked answer:** Using 100 samples at 360/s gives 277.78 ms collection; plus 14.93+15+20=327.71 ms, so it fails.
- **Reasoning:** Frame size changes source collection time before service begins.
- **Follow-up:** What artifact or changed assumption would make your answer fail?
- **Acceptable alternatives:** A different changed input is acceptable if it recomputes and changes decision.
- **Red flags:** Claiming faster UI alone fixes a larger frame is wrong.
- **Scoring:** Require a reversed gate result.
- **Answer support:** Sensitivity table.

**10. Question — When must the explanation abstain?**
- **Checked answer:** When no current validated frame exists or a factual claim lacks a provenance mapping to a versioned retrieved chunk.
- **Reasoning:** Abstention protects the evidence boundary rather than hiding an error.
- **Follow-up:** What artifact or changed assumption would make your answer fail?
- **Acceptable alternatives:** Showing a non-factual status explanation is acceptable if labeled.
- **Red flags:** Generating a plausible uncited substitute violates the contract.
- **Scoring:** Require state and provenance conditions.
- **Answer support:** Sections 5d and 8.

**11. Question — What separates TRACE-CHECKED from HIL-VALIDATED?**
- **Checked answer:** Trace tier needs executed fixture/oracle/manifest evidence; HIL additionally needs the declared physical setup and measured hardware-path trace.
- **Reasoning:** A desktop mock can validate logic but not the physical interface.
- **Follow-up:** What artifact or changed assumption would make your answer fail?
- **Acceptable alternatives:** Different hardware may be valid only if fully declared.
- **Red flags:** Calling a hand-written table HIL is false.
- **Scoring:** Require artifact distinction.
- **Answer support:** Sections 1 and 9.

**12. Question — How would you prove reproducible recovery?**
- **Checked answer:** Start from compatible manifest and committed sequence, replay frozen fixture, obtain every oracle disposition, and show no duplicate state effect.
- **Reasoning:** Recovery is evidence-preserving replay, not merely restarting a process.
- **Follow-up:** What artifact or changed assumption would make your answer fail?
- **Acceptable alternatives:** A deterministic sandbox is acceptable if artifact complete.
- **Red flags:** A screenshot after restart proves neither input nor state transition.
- **Scoring:** Require manifest, replay, and oracle.
- **Answer support:** Sections 5e and 8.

**13. Question — What is the release decision now?**
- **Checked answer:** DESIGN-only; no trace, UI visual, HIL, or deployment artifact exists.
- **Reasoning:** Claim tier follows available evidence, not quality of prose or arithmetic.
- **Follow-up:** What artifact or changed assumption would make your answer fail?
- **Acceptable alternatives:** Blocking publication entirely is acceptable if policy requires, but it is stricter than this case.
- **Red flags:** Promoting the constructed trace as executed is unsupported.
- **Scoring:** Require honest evidence label.
- **Answer support:** Section 7.

### Reviewer evidence drill-down

The following review cards are proposed inspection criteria.
They make the future release decision repeatable.
They do not report a completed review.

**R-01 — Canonical frame bytes**
- **Question:** Can the reviewer reconstruct one exact frame byte sequence?
- **Inspect:** Canonical encoding revision, byte order, header length, and payload hash.
- **Pass:** The bytes used by CRC are unambiguously delimited.
- **Fail signal:** Two implementations can serialize the same fields differently.
- **Gate effect:** Block TRACE-CHECKED promotion.
- **Owner:** Protocol owner.

**R-02 — CRC coverage**
- **Question:** Does the manifest say exactly which bytes the CRC covers?
- **Inspect:** Coverage start, coverage end, polynomial identifier, and stored expected CRC.
- **Pass:** A reviewer can recompute the declared fixture result.
- **Fail signal:** “CRC valid” appears without an algorithm or byte range.
- **Gate effect:** Keep the integrity assertion at DESIGN.
- **Owner:** Protocol owner.

**R-03 — Payload length**
- **Question:** Is payload length checked before payload interpretation?
- **Inspect:** Claimed length, actual length, maximum, and reject code.
- **Pass:** T4 remains rejected for both its length and CRC condition.
- **Fail signal:** Parser reads past the candidate or silently pads it.
- **Gate effect:** Block trace execution until bounded parsing is proven.
- **Owner:** Service owner.

**R-04 — Schema version**
- **Question:** Is an unsupported schema rejected before state mutation?
- **Inspect:** Supported-version list and unsupported-version fixture result.
- **Pass:** The unsupported candidate has a reason and no new state version.
- **Fail signal:** Unknown fields are guessed or ignored without a contract.
- **Gate effect:** Do not call the validator version-safe.
- **Owner:** Protocol owner.

**R-05 — Unit declaration**
- **Question:** Is a payload unit explicit even when values are opaque?
- **Inspect:** `sample_unit`, schema documentation, and fixture expectation.
- **Pass:** Unit mismatch causes a controlled rejection or abstention.
- **Fail signal:** Display labels invent a unit from field position.
- **Gate effect:** Prohibit semantic explanation wording.
- **Owner:** Data-contract owner.

**R-06 — Sample-rate declaration**
- **Question:** Is sampling rate a frame property rather than an unstated constant?
- **Inspect:** `sample_rate_hz`, frame-size calculation, and source contract.
- **Pass:** B-01 inputs are recoverable from manifest and fixture.
- **Fail signal:** Age calculation uses an undocumented default rate.
- **Gate effect:** Prohibit timing language.
- **Owner:** Acquisition owner.

**R-07 — Epoch reset**
- **Question:** Is sender restart distinguishable from a sequence regression?
- **Inspect:** Epoch policy, reset fixture, and state-store behavior.
- **Pass:** A new epoch follows an explicit resynchronization rule.
- **Fail signal:** Sequence zero overwrites a later sequence in an old epoch.
- **Gate effect:** Block duplicate-safety assertion.
- **Owner:** Service owner.

**R-08 — Duplicate boundary**
- **Question:** Does duplicate detection occur at the commit boundary?
- **Inspect:** Transaction record containing identity and state version.
- **Pass:** T2 cannot produce state version 20 after T1 produced 19.
- **Fail signal:** A retry is enqueued before identity comparison.
- **Gate effect:** Block idempotency claim.
- **Owner:** Service owner.

**R-09 — Equal payloads**
- **Question:** Can two distinct frames contain equal payload bytes?
- **Inspect:** Fixture pair with different sequences and same payload hash.
- **Pass:** Both are retained as separate logical events when contiguous.
- **Fail signal:** Payload deduplication substitutes for identity.
- **Gate effect:** Correct the duplicate method before evaluation.
- **Owner:** Service owner.

**R-10 — Gap policy**
- **Question:** What occurs between detecting a gap and receiving a resync?
- **Inspect:** T3 disposition, UI event, and optional retransmit request record.
- **Pass:** Last committed sequence remains 41 and state is degraded.
- **Fail signal:** Sequence 43 becomes current without sequence 42 evidence.
- **Gate effect:** Block continuity claim.
- **Owner:** Protocol and UI owners.

**R-11 — Queue bound**
- **Question:** Is capacity configured and recorded for the candidate release?
- **Inspect:** Capacity, current depth, overflow rule, and queue metric scope.
- **Pass:** Queue capacity two is visible in the proposed manifest.
- **Fail signal:** “Bounded” appears without an integer or policy.
- **Gate effect:** Block workload interpretation.
- **Owner:** Service owner.

**R-12 — Overflow evidence**
- **Question:** Does a full queue retain an accounting record for rejected work?
- **Inspect:** Correlation ID, reason, depth, capacity, and timestamp.
- **Pass:** Every arrival remains in the denominator after overflow.
- **Fail signal:** Arrival count is lower than fixture count with no dispositions.
- **Gate effect:** Block trace completeness.
- **Owner:** Service owner.

**R-13 — Receipt time**
- **Question:** Is receipt time distinct from source time?
- **Inspect:** Both fields with clock domains described.
- **Pass:** The trace can separate transport delay from source age.
- **Fail signal:** One timestamp is reused for every stage.
- **Gate effect:** Block freshness interpretation.
- **Owner:** Time-contract owner.

**R-14 — Render time**
- **Question:** Is render time evidence, not an inferred backend completion?
- **Inspect:** UI state event or visual capture linked to correlation ID.
- **Pass:** A future UI check can show the marker after the backend transition.
- **Fail signal:** Service completion is labeled UI display time.
- **Gate effect:** Keep UI validation OPEN.
- **Owner:** UI owner.

**R-15 — Stale threshold**
- **Question:** Is the stale threshold versioned and interpreted consistently?
- **Inspect:** Threshold value, comparison operator, and state table.
- **Pass:** 1301 ms is stale under `age > 1000 ms`.
- **Fail signal:** Boundary age alternates between current and stale by screen.
- **Gate effect:** Block stale-disclosure claim.
- **Owner:** UI owner.

**R-16 — Clock limitation**
- **Question:** Does the release distinguish declared time semantics from synchronization proof?
- **Inspect:** Manifest statement and UI wording.
- **Pass:** The artifact says age is computed under stated clock assumptions.
- **Fail signal:** A local time delta is advertised as externally synchronized truth.
- **Gate effect:** Remove cross-device freshness claim.
- **Owner:** Release owner.

**R-17 — Retrieval document**
- **Question:** Does every cited chunk have a stable document identity?
- **Inspect:** Document ID, document hash, access date, and permission basis.
- **Pass:** A reviewer can locate or identify the exact source snapshot.
- **Fail signal:** Citation says only “local knowledge base.”
- **Gate effect:** Require abstention for factual explanation.
- **Owner:** Retrieval owner.

**R-18 — Retrieval transform**
- **Question:** Is the chunking/index transform attributable?
- **Inspect:** Chunker revision, chunk ID, embedding/index revision, and rank.
- **Pass:** Retrieval output can be tied back to the document revision.
- **Fail signal:** A chunk excerpt exists with no source-to-index mapping.
- **Gate effect:** Do not call text grounded.
- **Owner:** Retrieval owner.

**R-19 — Prompt bundle**
- **Question:** Can an explanation be reconstructed without guessing instructions?
- **Inspect:** Prompt template hash, model identifier/hash, and response policy.
- **Pass:** The manifest binds these fields to the trace.
- **Fail signal:** The prompt changes after fixture evaluation.
- **Gate effect:** Block reproducibility claim.
- **Owner:** Retrieval owner.

**R-20 — Abstention path**
- **Question:** Is missing support visible rather than replaced with fluent text?
- **Inspect:** Retrieval-miss fixture and exact UI response class.
- **Pass:** Unsupported factual explanation is withheld.
- **Fail signal:** The system provides an uncited completion after a miss.
- **Gate effect:** Block explanation release.
- **Owner:** Retrieval and UI owners.

**R-21 — Manifest completeness**
- **Question:** Can all listed version values be checked against artifacts?
- **Inspect:** Hashes, revisions, fixture IDs, and runtime environment fields.
- **Pass:** Every required manifest field has a nonempty value or explicit `not used`.
- **Fail signal:** Generic component names replace versions or hashes.
- **Gate effect:** Block TRACE-CHECKED release.
- **Owner:** Release owner.

**R-22 — Oracle independence**
- **Question:** Was the expected result fixed before inspecting implementation output?
- **Inspect:** Oracle revision/date and fixture freeze record.
- **Pass:** The expected T1–T5 dispositions predate the candidate run.
- **Fail signal:** Oracle changed only to match observed output.
- **Gate effect:** Repeat evaluation with a frozen oracle.
- **Owner:** Independent reviewer.

**R-23 — Complete denominator**
- **Question:** Are rejected, duplicate, gap, and overflow inputs counted?
- **Inspect:** Arrival ledger compared with fixture manifest.
- **Pass:** Each declared fixture has exactly one terminal disposition.
- **Fail signal:** Reported pass rate excludes inconvenient arrivals.
- **Gate effect:** Block quality metric publication.
- **Owner:** Reviewer.

**R-24 — Claim wording**
- **Question:** Does public wording match the highest earned tier?
- **Inspect:** README, video captions, demo labels, and summary card.
- **Pass:** Current material says DESIGN-only and not measured.
- **Fail signal:** “Validated,” “real-time,” “HIL,” or clinical language appears.
- **Gate effect:** Block release until wording is corrected.
- **Owner:** Capstone technical lead.

### Contract field decision notes

Each note describes a separate design decision.
Each must become an artifact check in a future trace.

**F-01 — `stream_id`**
- It partitions independent ordered streams.
- It is part of the state key, not display decoration.
- An empty or unknown value is rejected.
- Its allowed character set is a protocol decision to pin.
- A stream change must not inherit another stream's sequence state.

**F-02 — `epoch`**
- It scopes sequence monotonicity across producer restarts.
- It must be durable enough to distinguish restart from wraparound.
- Its transition requires a documented reset policy.
- A lower epoch is not automatically an older valid frame.
- HIL must test real reset behavior if claimed.

**F-03 — `sequence`**
- It orders frames only inside one stream and epoch.
- It is not a global event timestamp.
- It supports duplicate and gap detection.
- It does not repair a missing frame.
- Its integer width and wrap rule belong in the manifest.

**F-04 — `source_time_ms`**
- It expresses source-event time in declared UTC milliseconds.
- It is separate from receipt and render time.
- It supports an age calculation only under stated clock semantics.
- It cannot prove clock synchronization by itself.
- It is retained even for rejected frames when safely permitted.

**F-05 — `schema_version`**
- It selects a parser and semantic contract.
- It protects against silent field reinterpretation.
- It must be checked before any state mutation.
- A future compatibility policy may allow explicit migrations.
- An unknown version remains evidence, not a value to guess.

**F-06 — `sample_rate_hz`**
- It gives the denominator for collection-time calculations.
- It must be positive and compatible with the frame contract.
- Its precision must be declared if fractional rates are allowed.
- It is not a performance observation.
- A rate change requires budget recomputation.

**F-07 — `sample_unit`**
- It prevents unlabeled numeric payloads from being overinterpreted.
- It identifies representation, not calibration validity.
- It must be visible to a reviewer and UI copy author.
- A mismatch blocks semantic explanation.
- It cannot authorize a diagnostic interpretation.

**F-08 — `payload_count`**
- It states the logical number of samples.
- It enables frame-span calculation with sample rate.
- It must match the payload encoding contract.
- It is distinct from byte length.
- It is bounded to contain parser work.

**F-09 — `payload_length`**
- It bounds allocation and parsing before content inspection.
- It must equal actual encoded bytes.
- It is checked even if CRC fails.
- Its maximum is a resource control, not a quality claim.
- A mismatch records a precise rejection reason.

**F-10 — `payload`**
- It is opaque at the transport boundary.
- It is hashed for trace linkage when payload disclosure is unsuitable.
- It must not be logged unnecessarily in a future sensitive context.
- It is not semantically validated by a CRC.
- Its encoding belongs to the schema revision.

**F-11 — `crc32`**
- It is a corruption-detection field under a named algorithm.
- It requires canonical byte coverage and byte order.
- It is not authentication or authorization.
- It is not a quality or safety oracle.
- A mismatch causes quarantine, not repair.

**F-12 — `correlation_id`**
- It names one arrival attempt across logs.
- It differs from stable frame identity.
- It links validator, queue, UI, and evidence events.
- It helps account for retries and rejected work.
- It must not replace epoch/sequence idempotency.

**F-13 — `received_time_ms`**
- It records service ingress under a declared clock.
- It supports transport-delay investigation.
- It cannot replace source time in an age formula.
- It must be retained with unit and time zone convention.
- It exposes queueing only when paired with later timestamps.

**F-14 — `committed_time_ms`**
- It marks the atomic state transition.
- It distinguishes admission from completed logical effect.
- It enables service-duration calculation with receipt time.
- It must be absent for rejected candidates.
- It anchors duplicate-safety inspection.

**F-15 — `rendered_time_ms`**
- It is a UI-side state observation.
- It is necessary for a display-age claim.
- It may be unavailable in headless trace execution.
- Its absence leaves UI visual validation OPEN.
- It must carry the displayed state version.

**F-16 — `state_version`**
- It monotonically identifies committed UI-relevant state.
- It changes once for a successful logical effect.
- It does not change for a duplicate or invalid frame.
- It lets a reviewer detect stale frontend caches.
- Its reset behavior is a separate application contract.

**F-17 — `manifest_id`**
- It binds the trace to a declared control bundle.
- It is displayed in reviewer evidence, not treated as a secret.
- It changes when a material bundle component changes.
- It prevents unrecorded prompt/index/config drift.
- It does not prove that artifacts are actually accessible.

**F-18 — `oracle_revision`**
- It names the expected-result logic.
- It must be frozen before a candidate trace.
- It supports independent disagreement review.
- It changes only through a documented correction.
- It does not substitute for HIL observation.

### Claim-audit probes

**P-01 — “End-to-end”**
- **Ask:** Does the claimed boundary include input, state, output, and oracle?
- **Pass evidence:** Correlated trace records span those four points.
- **Reject wording:** A UI screenshot alone.
- **Permitted current wording:** “End-to-end design contract.”
- **Decision:** DESIGN only.

**P-02 — “Real-time”**
- **Ask:** What percentile, workload, clock, and deadline support it?
- **Pass evidence:** Measured correlated paths with denominator and tail statistic.
- **Reject wording:** Additive component estimates.
- **Permitted current wording:** “Illustrative timing budget.”
- **Decision:** No real-time claim.

**P-03 — “Reliable”**
- **Ask:** Reliable across which failures and population?
- **Pass evidence:** Defined workload, failure injection, and observed denominator.
- **Reject wording:** One successful manual demo.
- **Permitted current wording:** “Proposed failure handling.”
- **Decision:** No reliability claim.

**P-04 — “Validated”**
- **Ask:** What validation mode actually ran?
- **Pass evidence:** Dated run artifact, oracle, and reviewer scope.
- **Reject wording:** Hand-derived expected output.
- **Permitted current wording:** “Validation plan specified.”
- **Decision:** No validated claim.

**P-05 — “Hardware-in-the-loop”**
- **Ask:** Was the declared physical path exercised?
- **Pass evidence:** Board, firmware, connection, raw trace, and method.
- **Reject wording:** Desktop JSON fixture.
- **Permitted current wording:** “HIL validation remains open.”
- **Decision:** No HIL claim.

**P-06 — “Local”**
- **Ask:** Is processing locality evidenced and defined?
- **Pass evidence:** Executed environment and data-boundary records.
- **Reject wording:** A diagram naming edge components.
- **Permitted current wording:** “Proposed local architecture.”
- **Decision:** Architecture statement only.

**P-07 — “Source-linked”**
- **Ask:** Can each factual sentence reach a versioned source chunk?
- **Pass evidence:** Claim-to-chunk mapping and corpus ledger.
- **Reject wording:** Unversioned citation list.
- **Permitted current wording:** “Retrieval provenance policy.”
- **Decision:** Abstain without mapping.

**P-08 — “Safe”**
- **Ask:** Which hazard, user, and safety evidence justify the word?
- **Pass evidence:** Appropriate safety case outside this scope.
- **Reject wording:** CRC, tests, or a warning label alone.
- **Permitted current wording:** “No clinical claim; scope constrained.”
- **Decision:** No safety claim.

**P-09 — “Accurate”**
- **Ask:** Accurate against which label or reference?
- **Pass evidence:** Defined label population and error metric.
- **Reject wording:** Parser acceptance rate.
- **Permitted current wording:** “Protocol oracle conformance, if executed.”
- **Decision:** No signal/clinical accuracy claim.

**P-10 — “Fresh”**
- **Ask:** What is the source-to-render age and threshold?
- **Pass evidence:** Correlated time semantics and rendered state evidence.
- **Reject wording:** Recent service log timestamp.
- **Permitted current wording:** “Proposed stale-state policy.”
- **Decision:** No observed freshness claim.

**P-11 — “Reproducible”**
- **Ask:** Can another reviewer obtain the same outcome from supplied artifacts?
- **Pass evidence:** Compatible manifest, fixtures, oracle, and accessible revisions.
- **Reject wording:** A version number without artifacts.
- **Permitted current wording:** “Reproducibility contract specified.”
- **Decision:** No reproduced run claim.

**P-12 — “Portfolio-ready”**
- **Ask:** Is the claim tier, limitation, and evidence legible to a reviewer?
- **Pass evidence:** Review record and consistently bounded public wording.
- **Reject wording:** Polish substituted for disclosure.
- **Permitted current wording:** “Design-only portfolio record.”
- **Decision:** Conditional on honest labeling.
## 15. Quick Reference Card

| Field | Case answer |
|---|---|
| Decision | Release tier for a local signal-to-explanation demonstrator. |
| Owner | Capstone technical lead, challenged by independent reviewer. |
| Input / output | One framed signal arrival -> disposition, state version, UI status, and permitted explanation/abstention. |
| Baseline | Screenshot-only demo with no reconstructable evidence chain. |
| Identity | `(stream_id, epoch, sequence)` at an atomic commit boundary. |
| Integrity | Bounded length, supported schema/units, declared CRC coverage, and sequence rules. |
| Queue | FIFO capacity 2 in constructed example; overflow must be recorded, never hidden. |
| UI states | CURRENT, DEGRADED, STALE, INVALID_INPUT, and ABSTAIN are distinct. |
| Evidence tier | DESIGN only now; TRACE-CHECKED and HIL-VALIDATED remain unearned. |
| Checked figures | 72/360=200 ms; 172-byte 8N1/115200=14.93 ms; zero queue=249.93 ms; two ahead=749.93 ms. |
| Gate | Require executed fixture/oracle, complete manifest, complete denominator, and reviewer evidence for trace tier. |
| Limits | No diagnostic, clinical, device, app-run, visual-UI, HIL, or deployment claim. |
| Review trigger | Any protocol, timing, hardware, runtime, retrieval, UI, manifest, or evidence change. |

**Final status:** Draft DESIGN-only case. The original capstone and backlog commitments remain proposed and pending; this document does not complete or substitute for them.
