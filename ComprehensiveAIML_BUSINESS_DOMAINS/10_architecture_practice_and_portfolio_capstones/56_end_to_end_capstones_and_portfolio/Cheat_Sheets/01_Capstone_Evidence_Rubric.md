---
collection: "end-to-end-capstones-and-portfolio"
file_id: "01"
topic: "Capstone evidence rubric and release gate"
level: 2
status: "draft"
version: "1.1"
created: "2026-11-08"
reviewed: null
review_trigger: "A hardware trace, primary source, deployment claim, or safety scope changes."
notation_style: "plain-text"
prerequisites: ["../INDEX.md", "../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md"]
pattern_tags: ["evidence-boundary", "end-to-end-contract", "release-gate"]
source_status: "OPEN"
owner_role: "capstone reviewer"
source_index: "INDEX.md"
---

# Capstone Evidence Rubric and Release Gate

> **Purpose:** Turn a proposed local STM32-to-Pi5 demonstration into an auditable portfolio claim without confusing a component demo, simulation, or borrowed benchmark with a deployed clinical system.
>
> **Prerequisites:** The module index; ability to distinguish measured from illustrative values; basic message, queue, evaluation, and rollback concepts.
>
> **Key outcome:** Given an architecture diagram, trace, evaluation table, and claim, you can decide which evidence tier is justified, identify the missing oracle, and define the next release gate.
>
> **This file does NOT cover:** clinical diagnosis, a device-specific wiring guide, or a claim that the visible source tree was built or benchmarked.
>
> **Level:** 2 — Core Methods.
>
> **Version / reviewed:** 1.1 / not reviewed.
>
> **Evidence:** DERIVED rubric logic; ILLUSTRATIVE calculations; OPEN hardware, software-version, and source-body evidence.
>
> **Retrieval cue:** If a demo claim has no input-to-output trace, start at Section 8.

**Source inventory.** This sheet uses the visible names in [`directory.txt`](../../../../Templates/directory.txt#L1147) and the source register’s [capstone brief](../../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#edge-capstone) and [source-boundary brief](../../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#edge-source). The bodies of those source files are unavailable. Their names are syllabus inventory, not proof of APIs, hardware compatibility, outcomes, or measured performance.

## 1. File Header

A capstone is a decision-bearing system record, not an attractive repository. The reviewer asks: what enters, what state changes, what output is shown, who may act on it, and what evidence justifies each arrow? A portfolio may be excellent while remaining a simulation; its credibility comes from saying so precisely.

Use four claim tiers. `DESIGN` means the interface is proposed. `TRACE-CHECKED` means a recorded input has been compared with an explicit oracle. `HIL-CHECKED` means the declared hardware path was exercised with its versioned setup. `OPERATED` means a defined user workflow, monitoring window, and outcome denominator were actually recorded. Do not promote a claim merely because the diagram includes real product names.

**Takeaway:** A release claim is bounded by the weakest unverified arrow in its input-to-outcome chain.

## 2. Concept Map / Mental Model

```text
[signal or recorded fixture]
          |
          v
[acquisition contract] --bad timestamp/shape--> [reject + evidence record]
          |
          v
[preprocess / local inference] --unknown model version--> [no performance claim]
          |
          v
[frame + sequence + CRC] --invalid/duplicate--> [quarantine or idempotent drop]
          |
          v
[Pi serial service + bounded queue] --over budget--> [backpressure/degraded view]
          |
          +--> [retrieval + explanation] --unsupported citation--> [withhold explanation]
          |
          v
[UI state + operator boundary] --stale state--> [visible stale marker]
          |
          v
[oracle, logs, evaluation denominator] --> [evidence tier and release decision]
```

The data plane carries a frame. The control plane changes configuration, models, prompts, or retrieval indexes. The evidence plane records immutable fixture ID, code revision, configuration, timestamps, validator result, and oracle result. These planes must not be silently merged: a UI screenshot is not an oracle, and a passing unit test is not HIL evidence.

**Takeaway:** Trace each claim across data, control, and evidence planes before accepting it.

## 3. Foundational Definitions

| Term | Meaning | Units / type | Nonexample or boundary |
|---|---|---|---|
| frame | One serialized batch with schema version, sequence, timestamp, payload, and integrity field | bytes; ordered record | A log line without a recoverable payload |
| fixture | Versioned input with expected output or invariant | immutable artifact | An ad-hoc live signal with no expected result |
| oracle | Rule or reference output used to decide pass/fail | predicate or labeled output | A reviewer’s impression of a chart |
| evidence tier | Maximum claim strength supported by linked artifacts | ordered category | A maturity score or hiring score |
| latency budget | Allocated elapsed-time allowance for a stated path and percentile | milliseconds | Sum of unrelated component averages claimed as p99 |
| freshness | Age between source event time and displayed state | seconds | Processing duration without event-time semantics |
| idempotency key | Stable identity used to make repeated delivery harmless at a named boundary | string/tuple | A random request ID regenerated on retry |
| release gate | Explicit condition which permits, blocks, or reverses promotion | boolean with owner | “Looks ready” |
| HIL | Hardware-in-the-loop exercise of the declared physical/software path | test mode | A desktop mock using compatible JSON |
| assistive explanation | Non-diagnostic text constrained to supplied sources and audience | text | Clinical advice or a factual claim without support |

**Evidence labels.** `DERIVED` marks a logical consequence of stated assumptions. `VERIFIED` marks arithmetic or a trace checked in this document. `ILLUSTRATIVE` marks constructed inputs. `OPEN` identifies an unavailable source, artifact, version, or result. `SOURCED` would require a primary source with date and scope; none is asserted here.

**Takeaway:** A term earns operational meaning only when its boundary and oracle are named.

## 4. Core Formulas with Full Derivation

### 4.1 Frame-span and serial-path budget

**Validity before use:** samples are uniformly acquired at rate `f_s`; a frame contains `n` samples; the budget is an illustrative additive estimate of one path, not a percentile or real-time guarantee.

**Derivation:** A sampling rate of `f_s` samples/s means each sample spans `1/f_s` s. A contiguous `n`-sample frame therefore spans `n/f_s` s. If processing stages are serial on this path, their modeled elapsed times add.

```text
FORMULA [CAP-01]: T_frame = n / f_s
WHERE: T_frame [s]; n [samples/frame]; f_s [samples/s].
CONSTRAINT: uniform sampling; no gap, overlap, resampling, or clock drift.
NUMERICAL EXAMPLE: n=100 samples; f_s=360 samples/s;
                   T_frame=100/360=0.277777... s=277.78 ms.
INTERPRETATION: a new complete frame cannot represent less than 277.78 ms of signal.
```

```text
FORMULA [CAP-02]: T_path = T_frame + T_infer + T_transport
WHERE: all T terms [ms]; T_infer is local computation; T_transport includes
       serialization, transmission, parsing, and service work on the chosen path.
CONSTRAINT: serial non-overlapped illustrative stages; excludes queueing, retries,
            OS scheduling, UI rendering, retrieval, and tail variation.
NUMERICAL EXAMPLE: 277.78 + 20 + 30 = 327.78 ms.
INTERPRETATION: the stated path estimate exceeds a 300 ms target by 27.78 ms.
```

**Independent check:** `100 / 360 × 1000 = 277.777…`; adding 50 ms gives 327.777… ms. Units are milliseconds throughout. **Sensitivity:** at 250 samples/s, the same 100 samples span 400 ms and the path becomes 450 ms. Reducing inference cannot satisfy a 300 ms target if framing alone is 400 ms. **Counterexample:** overlapping acquisition and processing may improve throughput, but it does not justify calling the sum a p99 latency.

### 4.2 Queue headroom

**Validity before use:** an initially empty queue, constant arrival rate `lambda`, constant service rate `mu`, and no retry amplification. This is a fluid estimate, not a finite-buffer proof.

**Derivation:** conservation gives backlog change per second as `lambda - mu`. Positive imbalance accumulated for `t` seconds is backlog.

```text
FORMULA [CAP-03]: B = max(0, lambda - mu) * t
WHERE: B [frames]; lambda [frames/s]; mu [frames/s]; t [s].
CONSTRAINT: constant rates and an initially empty queue; retries and batch variance excluded.
NUMERICAL EXAMPLE: lambda=4 frames/s, mu=3 frames/s, t=120 s;
                   B=(4-3)*120=120 frames.
INTERPRETATION: a 120-frame buffer only absorbs two minutes of this imbalance.
```

**Sensitivity:** if service rises to 4 frames/s, backlog does not drain; it remains at its initial value. If it rises to 5 frames/s after the 120-frame backlog forms while arrival remains 4 frames/s, drain time is 120/(5-4)=120 s. **Counterexample:** a queue depth graph alone cannot establish ordered, correct, or safe delivery.

### 4.3 Integrity and sequence invariant

**Precondition:** a frame has immutable bytes, an agreed checksum algorithm, a stable stream identifier, and a monotonic sequence number within a sender epoch.

**Mechanism/invariant [CAP-04]:** accept a frame only when schema is supported, payload length is bounded, checksum verifies, sender epoch is known, and `sequence > last_committed_sequence` for that stream; otherwise record the reason and do not advance the committed sequence. The sequence condition makes duplicate replay harmless at that boundary. It does not prove that separate streams are globally ordered.

**Worked trace:** committed `(epoch=7, seq=41)`. Frame `(7,42,valid)` is processed then atomically recorded as 42. A retry `(7,42,valid)` is a duplicate and does not create a second UI event. Frame `(7,44,valid)` reveals a gap; quarantine or request resynchronization according to the declared loss policy rather than silently treating it as contiguous. A CRC-valid frame can still contain semantically wrong sensor values.

**Takeaway:** Formula results are release inputs only after their assumptions and excluded tail behavior are made visible.

## 5. Worked Numerical Example

**Scenario:** ILLUSTRATIVE teaching workload, not a device measurement. A local demonstrator batches 100 samples from a 360 samples/s source. The proposal allocates 20 ms to local inference and 30 ms to serialization, serial transfer, parsing, and queued service when no queue exists.

1. `CAP-01` gives 277.78 ms per batch.
2. `CAP-02` gives 327.78 ms for the declared path.
3. A 300 ms requirement therefore fails by 27.78 ms; do not claim compliance.
4. A 250-samples/s alternative gives 400 + 20 + 30 = 450 ms, so lowering sample rate worsens this freshness budget if frame size remains 100.
5. A 72-sample frame at 360 samples/s spans 200 ms; with the same component estimates, the path is 250 ms. This is a design candidate, not proof: payload overhead, model window needs, and queueing must be checked.
6. For ingress 4 frames/s and service 3 frames/s, `CAP-03` yields 120 queued frames after 120 s. At 72 samples/frame, the buffered signal interval is 120 × 0.2 = 24 s of signal, while queue delay grows by 1 frame/s.

**Oracle inputs:** fixture ID `ILLUSTRATIVE-FRAME-360-100`; expected `T_frame=277.777...ms`; expected `T_path=327.777...ms`; expected 120-frame backlog. A spreadsheet or small pure function can recompute these values. No runtime was executed for this sheet.

**Decision:** block a “sub-300 ms end-to-end” portfolio claim. Permit a design-only statement if the source, timing convention, queue behavior, clock, and omitted components are listed.

**Takeaway:** A small calculation can invalidate a broad performance claim before expensive integration work begins.

## 6. Visual / Geometric Intuition

Imagine a horizontal timeline. The frame is a fixed-width window of source time; processing may begin only after the window is available unless a different streaming contract is designed. Queueing shifts the display rightward. A fast local inference box does not pull the left edge of an uncollected frame earlier.

```text
source time: |----100 samples / 277.78 ms----|
local path:                                 [20 ms][30 ms]
display:                                               ^ 327.78 ms estimate
queue:    arrivals ---> [bounded FIFO] ---> service
```

The contract graph is more important than the boxes: acquisition owns timestamping; firmware owns frame construction; the serial service owns validation and queue admission; retrieval owns corpus version and citations; the UI owns stale-state presentation; a named reviewer owns release evidence. If ownership is absent, recovery becomes opinion instead of a procedure.

**Takeaway:** Latency, correctness, and evidence each have different geometry; a component diagram must show all three.

## 7. Comparison / Selection Table

| Option | Use when | Benefit | Disqualifier | Evidence required | Fallback |
|---|---|---|---|---|---|
| Recorded-trace demonstrator | Hardware is unavailable | repeatable fixtures and explicit oracle | claiming device timing | trace provenance and parser results | label simulation |
| HIL demonstrator | declared hardware and safe fixtures exist | checks physical interface path | unpinned firmware/hardware setup | setup manifest and captured traces | return to recorded trace |
| Local assistive RAG view | user needs source-linked explanation | bounded contextual explanation | diagnosis, unsupported source, or unsafe instruction | corpus version, retrieval logs, citation check | show source links only |
| End-to-end performance claim | all path stages and percentile are measured | user-relevant result | averages, omitted queueing, or no clock convention | workload, timestamps, denominator, uncertainty | claim component measurement only |
| Portfolio architecture review | a decision must be defended | exposes assumptions early | no owner or acceptance criterion | decision record and threat/failure review | keep design status |

**Takeaway:** Select the narrowest claim whose evidence is complete; the narrow claim is more credible than an untestable grand claim.

## 8. Failure Modes & Edge Cases

| ID | Symptom | Competing causes | Discriminating check / expected signal | Bounded fix | Verification |
|---|---|---|---|---|---|
| F-01 | repeated UI records | transport retry; service restart; unstable ID | same epoch/sequence or idempotency key appears twice | durable dedupe at named boundary | replay duplicate fixture; one committed effect |
| F-02 | stale chart marked current | queue growth; browser state bug; source timestamp clock error | compare source event time, receipt time, and UI state version | expose age and state version; bound queue | injected pause shows stale marker |
| F-03 | valid CRC but misleading value | semantic unit mismatch; calibration issue; parser field shift | schema/unit validator and known fixture differ from reference | reject unsupported schema; retain raw frame | fixture detects changed field order |
| F-04 | component timings look fast but user wait is high | queueing; batch wait; retrieval/UI omitted | end-to-end trace spans same correlation ID | measure complete path by percentile | report denominator and percentile |
| F-05 | explanation sounds clinical | prompt wording; unsupported retrieval; user interpretation | citation audit and forbidden-claim test | constrain response and display scope | adversarial prompt is refused/redirected |
| F-06 | rollback restores code but behavior differs | model, prompt, index, policy, or config changed | release manifest differs from prior bundle | version whole compatible bundle | restore rehearsal compares manifest and oracle |

For F-01, do not “fix” duplication by removing logs; logs are evidence. For F-02, a stale indicator is not freshness recovery, but it prevents an unsafe interpretation. For F-03, a checksum protects bytes in transit, not the meaning of those bytes. For F-04, do not attach a p99 label to mean values. For F-05, an assistive display must not infer a diagnosis from a source name. For F-06, a prompt-only rollback is incomplete when retrieval state changed.

**Takeaway:** Diagnose from a discriminating signal, not from the first plausible component blamed by a dashboard.

## 9. Complexity & Cost Analysis

Frame validation is `O(b)` for `b` bytes when checksum and parsing examine the frame once; storage is `O(b)` per in-flight frame plus bounded queue overhead. Deduplication state is `O(u)` for `u` retained identities; a retention period is a correctness and storage decision, not a free configuration.

For an illustrative 120-frame backlog with 1,024-byte payloads, payload-only storage is `120 × 1024 = 122,880 bytes = 120 KiB`. Headers, allocation, logs, replication, and indexes are excluded. If a system has a 128 KiB budget, claiming it fits is unsound because those excluded costs can exceed 8 KiB.

Retrieval cost depends on index type, corpus size, embedding model, context budget, and cache state. No latency, price, power, model-quality, or hardware claim is made here because versions and workload are not available. A real portfolio should separately record one-time build work, recurring compute, storage, human review, failed requests, and cleanup.

**Takeaway:** Complexity bounds constrain a design; they do not substitute for an instrumented workload measurement.

## 10. Key Theorems / Results Reference

| Result | Use | Essential condition | Limit |
|---|---|---|---|
| conservation of flow | derive `CAP-03` | defined arrival and completion boundary | retries and drops change the boundary |
| pigeonhole principle | reason about finite sequence space | finite identifiers and more assignments than states | does not prescribe rollover policy |
| end-to-end argument | place correctness checks at application boundary | endpoint can validate desired property | lower-layer checks remain useful but insufficient |
| idempotence | make replay harmless | stable identity and atomic/consistent state transition | cannot undo already non-idempotent external effects |
| Little’s Law | relate average items, rate, and time | stable system and compatible averages | not a tail-latency or overload guarantee |

Primary references for a production use must be added with version/date and exact claim. This compact table is a method reminder, not a citation substitute.

**Takeaway:** Name the theorem, its assumptions, and the property it does not prove.

## 11. Interview / Exam Questions

1. **Why is a working UI not enough for an end-to-end claim?** It may bypass acquisition, validation, queueing, or the claimed hardware. Ask for a correlated trace and oracle. Accept a scoped UI-only claim; reject “production-ready” without the missing boundaries.
2. **Compute the 100-sample frame span at 360 samples/s.** `100/360=0.27778 s=277.78 ms`. The calculation assumes uniform sampling. A 300 ms end-to-end target leaves only 22.22 ms for every later stage.
3. **What does CRC validation establish?** Byte integrity relative to the selected algorithm and bytes; it does not establish sensor calibration, schema meaning, authenticity, or clinical validity. Add schema and semantic checks.
4. **A UI is 90 seconds behind. What do you check first?** Compare event, receipt, processing, and render timestamps for one correlation ID; then inspect queue depth and state version. Do not assume the broker is slow.
5. **How do you make serial retries safe?** Define a stable sender epoch and sequence or idempotency key, an atomic committed boundary, retention, and gap policy. A random retry ID is unacceptable.
6. **When is a recorded trace preferable to HIL?** When hardware is unavailable or unsafe, provided the limitation is explicit. It supports parser and deterministic-state claims, not electrical, timing, or power claims.
7. **What makes a retrieval explanation auditable?** Corpus/index version, query, retrieved identifiers, prompt/template version, response, citation check, and an explicit unsupported-claim policy. Fluency is not evidence.
8. **A team reports 20 ms inference and 30 ms transport. Can it claim 50 ms end-to-end?** No. It omits frame collection, queueing, scheduling, UI, and tail behavior. Under the illustrative values here the serial path estimate is 327.78 ms.
9. **What changed constraint would reverse your architecture choice?** If raw signal must remain local and no trustworthy source corpus exists, disable remote retrieval and restrict output to local status. If the user needs clinical diagnosis, this demonstrator is out of scope and requires a different governance path.
10. **What is a plausible wrong remediation for duplicate frames?** Dropping all repeated payload values. Equal payloads can be legitimate distinct events; dedupe must use a stable identity and declared retention boundary.

**Takeaway:** A defensible answer states the boundary, evidence, alternative, and what would change the decision.

## 12. Quick Reference Card

| Cue | Rule | Use when | Critical assumption | Avoid when |
|---|---|---|---|---|
| Make a claim | choose DESIGN, TRACE-CHECKED, HIL-CHECKED, or OPERATED | writing README/demo language | artifacts link each arrow | a tier is inferred from a diagram |
| Size a frame | `T_frame=n/f_s` | fixed contiguous sample batches | uniform sampling | treating it as p99 |
| Estimate path | add declared serial stages only | early feasibility check | no overlap/queueing omitted explicitly | reporting end-to-end percentile |
| Bound backlog | `B=max(0,lambda-mu)*t` | constant-rate overload estimate | no retry amplification | service is variable or unknown |
| Admit a frame | schema, length, checksum, epoch/sequence, atomic commit | serial service | identity is stable | CRC is treated as semantic validity |
| Present data | show state version and age | any delayed UI | timestamps are comparable | stale data looks live |
| Explain content | retain corpus/query/citation evidence | assistive retrieval | scope forbids diagnosis | source support is absent |
| Release | require owner, oracle, rollback, and limitation | promotion decision | compatible bundle is versioned | any required evidence is OPEN |
