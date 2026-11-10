---
case_id: "CASE-CAPSTONE-001"
case_title: "Choose a release gate for a local signal-to-explanation demonstrator"
case_version: "1.2"
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

**Source and evidence declaration.** This is an ILLUSTRATIVE design exercise. The visible [directory inventory](../../../../Templates/directory.txt#L1147) and [edge-capstone planning brief](../../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#edge-capstone) establish syllabus scope only. Source bodies, hardware, device traffic, application runs, UI recordings, corpus documents, and performance measurements are OPEN. Every number below is a constructed design value, recomputed by hand and labelled ILLUSTRATIVE; none is a benchmark. Nothing here is a diagnostic, clinical, device, deployment, or named-application claim.

## 1. The One-Line Framing

> Given one constructed framed signal event and a constrained request for a local, source-linked assistive explanation, the capstone technical lead decides whether the portfolio artifact is DESIGN-only, TRACE-CHECKED, or eligible to schedule HIL validation — for an independent reviewer, subject to visible invalid/stale states and a reconstructable evidence bundle, compared with a screenshot-only demo.

The unit of work is a frame and the UI state derived from it, never a person, a diagnosis, or a clinical conclusion. The deliverable of the gate is a release-tier decision plus the exact wording that tier permits. The primary success condition is not that a screen renders; it is that a reviewer who was not present can reconstruct the declared trace, account for every declared input, and see why withheld paths stayed withheld.

Three tiers are defined, and they are cumulative in evidence but not in claim strength:

| Tier | Minimum artifact required | Establishes | Does not establish |
|---|---|---|---|
| DESIGN | this manuscript, contracts, oracle table, manifest schema | a coherent, testable design intent | that any code, device, or UI exists |
| TRACE-CHECKED | executed deterministic fixture, frozen oracle, complete manifest, inspectable per-arrival dispositions | software logic behaved as declared on a known input | physical transport, timing on hardware, user comprehension |
| HIL-VALIDATED | the TRACE-CHECKED set plus a named hardware setup, wiring record, instrumented capture, repetition count | the declared hardware path was exercised and measured | safety, efficacy, population outcome, clinical utility |

**Current standing.** No tier above DESIGN is earned by this manuscript. The oracle table in Section 6 is hand-checked reasoning; a hand-checked table is not an executed run. The distinction matters because every downstream portfolio sentence inherits the tier: "the gate logic specifies" is supportable today, "the gate logic behaves" is not.

**Reusable pattern.** An evidence-bounded release gate: define the work unit, the state identity, the oracle, the complete denominator, and the claim tier before any presentation work begins. Secondary patterns are an end-to-end contract (every stage has an owner and a recorded disposition) and bounded work (queue capacity, frame size, and retry policy are integers, not adjectives).

**Why a screenshot fails as a baseline.** A screenshot proves that pixels existed at one instant on one machine. It cannot establish which bytes were accepted, which sequence numbers were contiguous, how old the displayed value was, which corpus revision supplied a sentence, or whether the visible explanation was permitted. The baseline is deliberately weak so that the gate has something concrete to beat.

## 2. Context & Background

The proposed system separates into three planes that are frequently conflated in student portfolios. Conflating them is the single most common reason a demo looks finished while being unevidenceable.

| Plane | Carries | Fails when |
|---|---|---|
| Data plane | acquisition or fixture bytes, frame construction, serial read, validation, bounded queue, state processing, optional retrieval, UI render | any stage drops work without a record |
| Control plane | pinned schema revision, thresholds, model and prompt identifiers, index revision, permitted UI copy | a value moves without a manifest change |
| Evidence plane | fixture identity and hashes, source/receipt/render timestamps, per-arrival disposition, oracle decision, manifest identity | evidence is written after the fact or not at all |

Separating the planes produces a specific design obligation: the evidence plane must be written by the same code path that mutates state, not by a separate reporting job that can disagree with it. A reporting job that reconstructs dispositions later will eventually reconstruct them wrongly, and the discrepancy is invisible until an auditor asks for the raw record.

| Context field | Bounded statement | Evidence status |
|---|---|---|
| Stakeholders | technical lead decides; reviewer challenges scope; component owners supply contracts | illustrative roles, no named people |
| Baseline | screenshot plus a verbal walkthrough of the pipeline | considered baseline, not tested |
| Before the decision | a candidate manifest and a declared fixture are assembled | proposed workflow |
| After the decision | only tier-appropriate portfolio wording may be released | proposed control |
| Scale | one local trace; no user population, concurrency, or throughput claim | scope constraint |
| Boundary | a local design record; no remote service, care workflow, or device claim | scope constraint |
| Timing | the gate runs before README, video, benchmark, or portfolio statement | proposed process |
| Cost | author hours and reviewer hours only; no infrastructure spend is claimed | scope constraint |

The visible inventory names UART, an STM32-class microcontroller, a Raspberry Pi 5-class host, retrieval-augmented generation, a local model runtime, a Python UI framework, hardware-in-the-loop testing, and profiling. Those names justify discussing interfaces and validation questions. They do not justify asserting that a specific board, firmware, model file, library version, or UI framework was installed, configured, or made to interoperate.

A second piece of context is organisational rather than technical: a capstone artifact is read by people who will not run it. Recruiters, reviewers, and course assessors read a repository page, maybe a short video, and a README. Their default inference from a polished screenshot is "this works". The gate exists to make the artifact's own text contradict that inference wherever the evidence does not support it — which is why the output of the gate is wording, not just a badge.

The last contextual fact is that the demonstrator has no user. There is no operator whose behaviour could be observed, no workflow whose latency could be measured end to end, and no consequence attached to a wrong explanation beyond the artifact's own credibility. This is a strength for evidence discipline, because nothing must be protected from failure, and a weakness for external validity, because nothing generalises beyond the fixture.

## 3. The Business Question

| Decision contract | Definition |
|---|---|
| Question | Is the evidence complete enough for a TRACE-CHECKED local-demonstrator claim, or must the artifact remain DESIGN-only? |
| Decision owner | Capstone technical lead, accountable for the published wording. |
| Challenge owner | Independent reviewer, who owns evidence-scope objections and cannot be overruled by the lead. |
| Decision grain | One release candidate = one manifest identity + one fixture set + one permitted-wording set. |
| Cut-off timestamp | The instant the release candidate is frozen; arrivals after it belong to the next candidate. |
| Trigger | Any proposal to publish a claim about behaviour, timing, retrieval quality, hardware, or deployment. |
| Actions available | Block entirely; publish DESIGN wording; publish TRACE-CHECKED wording; schedule HIL validation. |
| Baseline alternative | Screenshot-only presentation with no reconstructable input-to-output evidence. |
| Defer alternative | Publish nothing this cycle and keep the repository private until a fixture and oracle exist. |
| Objective | Maximise the number of complete, inspectable claims per artifact; not visual polish, not apparent responsiveness. |
| Hard constraints | No diagnostic or clinical language; invalid and stale states always visible; no fabricated run; every displayed number traceable to a manifest entry. |
| Soft constraints | Prefer the cheapest artifact that unlocks a claim; prefer one honest limitation over three hedged claims. |
| Scope | One declared trace, one version bundle, one local UI-state contract. |
| Out of scope | Safety certification, efficacy, privacy review, device approval, remote operation, real-user outcomes. |

The objective needs a counterweight, because "maximise inspectable claims" can be gamed by inventing cheap claims. The counter-metric is the number of published sentences that a reviewer can falsify from the artifact alone. A claim that cannot be falsified is not evidence of rigour; it is a sentence with no content. Under this pairing, "the validator rejects a 143-byte payload declared as 144 bytes" is a good claim because a reviewer can check it against the oracle; "the system is robust" is a bad claim because nothing in the artifact could contradict it.

| Proxy metric | Numerator | Denominator | Why it is only a proxy |
|---|---|---|---|
| Trace completeness | declared arrivals with a recorded disposition | all arrivals declared before evaluation | a complete trace of a wrong oracle is still complete |
| Duplicate safety | duplicate arrivals producing zero additional state versions | all duplicate arrivals | passes trivially if no duplicate was ever sent |
| Manifest fidelity | required fields that a reviewer can hash-match | required fields in the schema | says nothing about whether the field values are meaningful |
| Wording discipline | published sentences mapped to an earned tier | published sentences about behaviour | requires human reading, so it samples poorly |

A fourth consideration shapes the decision more than any metric: the cost of being wrong is asymmetric. Publishing DESIGN wording when the evidence happens to be good costs a little credibility and one cycle of delay. Publishing TRACE-CHECKED wording when the evidence is absent costs the entire artifact's credibility the first time a reviewer asks for the trace and finds a table. The gate is therefore biased toward under-claiming, and the bias is deliberate rather than a failure of nerve.

## 4. Assumptions & Constraints

| ID | Assumption or constraint | Type | Check | Status | If wrong | Fallback / owner |
|---|---|---|---|---|---|---|
| A-01 | Frame bytes and the CRC algorithm are specified together, including covered byte range | protocol | fixture parser plus CRC oracle | OPEN | the integrity decision is ambiguous | block trace claim / protocol owner |
| A-02 | Epoch plus sequence is stable across retry and reconnect | state | duplicate replay trace | OPEN | a side effect may repeat | withhold claim / service owner |
| A-03 | Source and render clock semantics are declared and related | time | age recomputation from the trace | OPEN | the freshness label is misleading | show unknown / UI owner |
| A-04 | Queue capacity and the overflow rule are bounded integers with a stated policy | capacity | admission trace with a full queue | OPEN | work and evidence may vanish silently | block claim / service owner |
| A-05 | Retrieved documents are attributable to a versioned chunk identity | provenance | document and citation ledger | OPEN | the explanation is unsupported | abstain / retrieval owner |
| A-06 | Fixture bytes are immutable once frozen | reproducibility | hash equality across two encodes | OPEN | replays diverge without a code change | re-freeze and re-oracle / fixture owner |
| C-01 | No diagnosis, treatment, or clinical recommendation appears anywhere in copy or prompts | scope, binding | copy and prompt audit | binding | unsafe scope expansion | refuse / lead |
| C-02 | Bundle versions move together; no partial upgrade | reproducibility | manifest equality check | proposed | behaviour cannot be reproduced | block / lead |
| C-03 | Invalid and stale states remain visible whenever they apply | honesty | render-state oracle | OPEN | a viewer infers false freshness | block UI claim / UI owner |
| C-04 | No performance figure is published without a named path and denominator | honesty | claim-audit probe | binding | a component number is read as end-to-end | retract wording / lead |

A-01, A-03, and A-05 are the assumptions that can most directly reverse the release decision, and they fail in three different ways. A correct CRC over semantically wrong units produces a frame that is provably intact and provably meaningless. A render timestamp without a stated clock relationship cannot establish freshness at all, because the difference between two timestamps from unrelated clocks is a number without a quantity. Retrieval without a chunk ledger produces text that reads like evidence while carrying no mapping from sentence to source.

C-04 deserves emphasis because it is the constraint most often broken by well-intentioned authors. A measured median handler time of 12 ms is a true statement about one component. Printed beside a 160 ms collection window, it invites the reader to conclude the system responds in 12 ms. The constraint does not forbid the number; it forbids publishing it without the path definition that makes it interpretable.

Two constraints are deliberately absent and their absence is itself a design decision. There is no accuracy target, because the artifact does not classify anything, so an accuracy number would be invented. There is no uptime target, because the demonstrator runs for minutes on a desk, so availability percentages would be arithmetic about nothing. Adding either would create the appearance of rigour at the cost of a falsifiable claim.

**Cost of each OPEN assumption.** An assumption marked OPEN is not free; each one has a price and a cheapest way to close it, and the prices differ by two orders of magnitude.

| Assumption | Cheapest closure | Rough effort | What it buys | If never closed |
|---|---|---|---|---|
| A-01 CRC coverage | write down polynomial and byte range in the manifest | minutes | an integrity claim a reviewer can recompute | the CRC is a ritual |
| A-02 retry stability | one repeated-identity fixture, executed | hours | an idempotency claim | duplicates are unaddressed |
| A-03 clock semantics | state which clock stamps source and render instants | minutes of writing, possibly days of hardware work | interpretable freshness | age numbers are meaningless |
| A-04 queue bound | record capacity and policy as integers | minutes | a workload statement | "bounded" is an adjective |
| A-05 chunk attribution | store chunk ID with every retrieved span | hours | an explanation claim | citations decorate |
| A-06 fixture immutability | two independent encodes compared by hash | under an hour | reproducible replay | the oracle may be about the wrong bytes |

The asymmetry is the design insight. A-01, A-04, and A-06 are cheap and unlock the largest claims per unit of effort, so they should be closed first even though they are the least interesting to work on. A-03 is cheap to write and expensive to prove, which means the honest intermediate state is "age computed under a declared assumption, synchronisation unproven" rather than either silence or a false claim of accuracy.

**What would force a redesign rather than a fix.** Three changes would invalidate the structure rather than one part of it. If the payload must be interpreted physiologically rather than treated as opaque, the case leaves its scope, because interpretation requires calibration and clinical justification. If the source must support more than 100 samples per frame, the age budget in B-01 cannot hold at any plausible rate and the whole demonstrator becomes a batch artifact. If the UI must run remotely, the clock relationship in A-03 acquires a network-delay term and the additive model in B-03 loses its single-clock premise.

**A constraint that looks present and is not.** There is no requirement that the demonstration be impressive, and the absence is worth stating because it is the constraint reviewers most often assume. Nothing in this document rewards a longer video, a smoother animation, or a larger model. The gate rewards exactly one thing: sentences that survive an auditor. A capstone that looks dull and quotes 216.1944 ms with its assumptions is worth more than one that looks polished and quotes a number with no denominator, and the gate is designed to make that ranking explicit rather than cultural.

## 5. Data, Tools & Method

### 5a. Data Specification

A frame is immutable bytes plus parsed metadata. The distinction matters because the CRC covers bytes, the validator reasons about fields, and the UI reasons about state; three representations of the same arrival must never be allowed to drift.

| Field | Type / unit | Admission rule | Recorded evidence |
|---|---|---|---|
| `stream_id` | ASCII identifier, 1–16 chars | allowlisted and nonempty | exact value |
| `epoch` | uint32, generation counter | known, or reset under an explicit policy | exact value and predecessor |
| `sequence` | uint64, monotonic within `(stream_id, epoch)` | strictly greater than the committed sequence | value and predecessor |
| `source_time_ms` | int64, UTC milliseconds | within the declared representable range | source, receipt, and render times |
| `schema_version` | string | equal to a supported revision | supported-set revision |
| `sample_rate_hz` | positive decimal, samples per second | equal to the declared fixture contract | value and unit |
| `sample_unit` | string | equal to the declared fixture contract | value; never interpreted clinically |
| `payload_count` | uint16, samples | 1 through the declared maximum of 96 | count |
| `payload_length` | uint16, bytes | equal to the encoded length and at most 256 | claimed and actual bytes |
| `payload` | opaque bytes | opaque after structural checks | payload hash only, never contents |
| `crc32` | uint32 | matches the canonical covered range | expected and actual CRC |
| `correlation_id` | string, unique per arrival attempt | unique | arrival identity for every disposition |

**Grain.** One row per arrival attempt. Two arrivals carrying the same `(stream_id, epoch, sequence)` are two rows and one logical event. This is the grain decision that makes duplicate accounting possible; a frame-level grain silently loses retries.

**Cut-off.** The cut-off timestamp is the candidate freeze instant `T_cut`. Any arrival with a receipt time after `T_cut` is excluded from the candidate and recorded as excluded, not deleted. The exclusion count belongs in the denominator discussion, because an artifact that quietly narrows its window can always reach 100 percent.

**Population.** The population is the declared fixture set: six arrivals, of which five are frames and one is a clock-driven state observation with no new frame. There is no sampling frame, no user population, and no natural distribution; the fixtures are chosen to exercise dispositions, not to estimate rates.

**Keys.** The logical key is `(stream_id, epoch, sequence)`. The evidence key is `correlation_id`. The bundle key is the manifest identity. Confusing the first two produces the classic audit failure where an operator can see that a frame was handled but cannot tell which arrival attempt produced the handling.

**Units.** Time in milliseconds throughout the UI and evidence tables; seconds only inside formulas that require it. Bytes and bits are never mixed without an explicit factor. Rates are frames per second for service and samples per second for acquisition; these are different quantities that both happen to be called "rate", which is a reliable source of unit errors.

**Not available at decision time.** The payload's physical provenance, sensor calibration, the meaning of `arb-unit`, any clock synchronisation proof, the operating system's scheduling behaviour, the serial driver's buffering behaviour, the actual baud rate achieved on a real link, and any human reaction or comprehension measurement. Each of these is named here so that no later section can quietly assume it.

### 5b. Tools & Stack

| Proposed component | Responsibility | Deliberately not claimed | Owner |
|---|---|---|---|
| fixture producer | emit canonical bytes plus the expected disposition for each arrival | any firmware behaviour | fixture owner |
| frame builder | serialise the declared header and CRC coverage | a specific microcontroller implementation | protocol owner |
| serial reader | turn a byte stream into candidate frames with explicit boundary errors | a physical serial run | service owner |
| validator | apply shape, version, unit, length, CRC, and ordering rules | semantic correctness of the signal | service owner |
| bounded FIFO | retain at most two admitted frames | loss-free behaviour under overload | service owner |
| state processor | commit last sequence and state version atomically | a deployed database or durable log | service owner |
| age classifier | map age to CURRENT, DEGRADED, or STALE using declared thresholds | wall-clock synchronisation | UI owner |
| retrieval adapter | return versioned local citations, or abstain | corpus quality or model quality | retrieval owner |
| UI | show state, age, and explicit invalid/degraded/stale labels | a tested or accessible visual render | UI owner |
| manifest writer | bind inputs and versions to one trace identity | an actual repository or run | release owner |
| oracle comparator | compare observed dispositions to the frozen expectation | independence from the implementer | reviewer |

The oracle comparator is listed with the reviewer as owner on purpose. If the same person writes the implementation and the expectation, the comparison measures consistency rather than correctness. For a six-arrival fixture the independence is cheap to arrange; for a larger fixture set it is the first thing dropped under schedule pressure, which is exactly why it is named as an owner boundary rather than a task.

### 5c. Method Summary

`canonical fixture -> framing -> structural validation -> CRC validation -> sequence decision -> bounded admission -> atomic state commit -> age classification -> retrieval with provenance check -> UI state -> oracle comparison -> release gate`

| Candidate approach | Evidence considered | Strength | Limit | Decision |
|---|---|---|---|---|
| Screenshot-only | appearance alone | fast to explain, zero build cost | no input, oracle, or manifest evidence | rejected for any behavioural claim |
| Recorded fixture plus frozen oracle | a constructed deterministic trace | dispositions are inspectable and cheap to freeze | no physical path or timing proof | the minimum path to TRACE-CHECKED |
| HIL exercise | no artifact available | would exercise the declared physical path | unavailable this cycle | not tested, not claimed |
| Production observation | no workflow or outcome data | could measure real operations | outside scope and unavailable | excluded |

**Admission invariant.** A frame changes committed state if and only if schema, units, length, CRC, and ordering all pass and the queue admits it. The commit writes `(stream_id, epoch, sequence, payload_hash, state_version)` in one transaction. A duplicate never increments `state_version`. A corrupt or gapped frame never advances the committed sequence. The invariant is stated as an implication rather than a sequence of steps so that it can be tested by looking at any single record: if `state_version` increased, all five predicates were true at the same instant.

**Why an invariant instead of a pipeline description.** A pipeline description ("validate, then enqueue, then commit") is satisfied by an implementation that validates, enqueues a retry, and commits twice. The invariant forbids that outcome directly, so the oracle can check it on the record rather than on a code reading.

### 5d. Provenance, Quality, Exclusions, and Leakage

Fixture inputs are constructed. Every fixture record must carry author, creation date, canonical encoding revision, expected-result author, and payload hash. The trace is not drawn from a device, a person, a patient record, or a production log. Payload values are deliberately opaque because a checksum cannot validate calibration, sensor placement, biological meaning, or the correctness of any signal-processing algorithm — the CRC knows only that certain bytes arrived as transmitted.

Three quality properties are separately tracked and are frequently merged by mistake:

| Property | Question it answers | Who can break it | Typical silent failure |
|---|---|---|---|
| Integrity | did these bytes arrive unaltered? | transport and framing | a CRC over the wrong byte range |
| Validity | do these fields satisfy the declared contract? | schema and unit drift | an unknown schema accepted by a permissive parser |
| Freshness | is this state recent enough to display as current? | queueing, clocks, caching | a stale chart left on screen with a current label |

Retrieval is optional in this design and is treated as a separate claim. A factual explanation sentence is eligible only when it maps to a cited chunk whose document identity, document hash, access date, permission basis, chunker revision, index revision, and retrieval rank are all in the evidence bundle. A retrieval miss, a missing mapping, or two conflicting sources produces an abstention. The abstention is a designed output, not an error state, and the UI copy must be written for it before the first successful retrieval is attempted.

**Leakage risk.** The evaluation prompt set must be authored separately from the explanation copy, and the expected answer must not appear in the prompt, the few-shot examples, or the fixture filenames. In a single-author project the practical leakage channel is not the data; it is the author's own memory of the intended output while writing the prompt. The cheap control is to freeze the prompt hash before writing the expected answer, then verify the hash at review. If the two were written in the same sitting with the answer visible, the retrieval evaluation measures transcription.

**Exclusions.** Excluded from the candidate: arrivals after `T_cut`; fixtures whose encoding revision differs from the frozen one; any recorded run whose manifest is incomplete; and any measurement taken while another workload shared the host, because host contention is not modelled and would be attributed to the system.

### 5e. Reproducibility and Method-Selection Ledger

| ID | Location / identity | Version or date | Purpose | Accessibility | Limitation |
|---|---|---|---|---|---|
| SRC-01 | directory inventory, lines 1147–1164 | visible inventory | topic scope | local | bodies unavailable |
| SRC-02 | `00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#edge-capstone` | planning brief | scope boundary | local | planning is not implementation evidence |
| FIX-01 | proposed `fixture-B-v1` | constructed | trace input | not stored | no run artifact exists |
| ORA-01 | the disposition table in Section 6 | constructed | expected outcomes | in this manuscript | never executed against code |
| MAN-01 | the schema below | constructed | reconstruction contract | in this manuscript | no manifest ever emitted |
| RUN-01 | not available | n/a | trace execution | unavailable | TRACE-CHECKED unavailable |
| HIL-01 | not available | n/a | hardware validation | unavailable | HIL-VALIDATED unavailable |

A manifest is complete only if a reviewer can compare every listed value against a supplied artifact. A list of product names is a shopping list, not a manifest. The required fields are: manifest identity and creation instant; fixture hashes and encoding revision; schema revision; CRC polynomial and the exact covered byte range; queue capacity and overflow policy; service revision; age thresholds; model identity and hash if any; prompt template hash if any; corpus, chunker, embedder, and index identities if any; UI revision; oracle revision; host and runtime identifiers; every observed disposition; and an explicit list of unresolved limitations.

**Method selection.** A deterministic fixture was chosen over a live capture because it is the only input that can be replayed byte for byte, and because a live capture would require a device claim that this manuscript cannot support. The cost of that choice is external validity: the fixture exercises the contract, not the world. A reviewer must not read fixture results as evidence about any physical signal source, and the manifest's limitations list is where that sentence belongs so it travels with the bundle.

## 6. Analysis Walkthrough

**Decisive current result:** no trace has been executed, therefore the release remains DESIGN-only. What follows is a hand-checked illustrative oracle and design trace. It demonstrates the shape a future trace must take; it is not evidence of an application, a serial service, a UI, a device, or a HIL run.

### Declared illustrative trace

At run start, the committed sequence for `(fixture-B, 4)` is 117, queue depth is 0 of 2, state version is 41, and the display holds no current frame. The receipt clock and the render clock are assumed to share a declared monotonic reference for the purpose of this constructed calculation — an assumption that appears as A-03 and remains OPEN in any real candidate. Freshness thresholds are `age <= 500 ms` CURRENT, `500 < age <= 1000 ms` DEGRADED, `age > 1000 ms` STALE. These thresholds are design choices, not safety limits, and they do not derive from any physiological or clinical requirement.

| Arrival | Frame identity | Structural / CRC condition | Expected queue disposition | Expected committed state | Expected UI state |
|---|---|---|---|---|---|
| T1 | (fixture-B, 4, 118) | valid; 128-byte payload; CRC matches | admit, then service | commit 118; state v42 | CURRENT after render |
| T2 | (fixture-B, 4, 118) | valid duplicate; identical payload hash | duplicate-drop, no queue slot | remain 118; state v42 | CURRENT, duplicate counter incremented |
| T3 | (fixture-B, 4, 120) | valid CRC but 119 was expected | gap-quarantine, no queue slot | remain 118; state v42 | DEGRADED with the missing index named |
| T4 | (fixture-B, 4, 119) | declares 128 payload bytes, supplies 127; CRC mismatch | corrupt-quarantine, no queue slot | remain 118; state v42 | INVALID_INPUT evidence visible, prior DEGRADED retained |
| T5 | (fixture-B, 4, 121) | valid frame arriving while the FIFO holds two unserved frames | overflow-drop with an accounting record | remain 118; state v42 | DEGRADED with overflow reason and dropped correlation ID |
| T6 | no new frame | last commit at source time 900 ms; render check at 2400 ms | no admission | remain 118; state v42 | STALE; explanation withheld |

**T1 — the valid path.** The parser reads `payload_count=64`, `sample_rate_hz=400`, `sample_unit=arb-unit`, and `payload_length=128`, and the actual byte count is also 128. The CRC recomputed over the declared header-and-payload range matches the transmitted value. Sequence 118 equals `last_committed + 1`, so the frame is admitted, served, and committed; `state_version` moves from 41 to 42 in the same transaction that stores sequence 118. A future trace must retain both the arrival record and the commit record, because an accepted-parser log alone shows intent, not effect.

**T2 — the duplicate path.** The arrival repeats epoch 4 and sequence 118 with an identical payload hash. It could be a producer resend, a reader reconnect replay, or a fixture artefact; this oracle does not diagnose which, because the cause does not change the required handling. Since `sequence <= committed`, it is a duplicate: no queue record, no state transition, no explanation request, no increment of `state_version`. The disposition is recorded with its correlation ID so the duplicate is countable rather than invisible.

Dropping frames whose payload bytes happen to equal a previous frame would be a different and wrong rule. Two distinct sequences may legitimately carry identical payload bytes — a stationary sensor produces this constantly — and payload-based deduplication would erase real events. Identity, not content, decides duplicates.

**T3 — the gap path.** Sequence 120 arrives with an intact CRC, but the committed sequence is 118, so 119 is missing. Integrity is not contiguity, and a valid checksum says nothing about whether anything was lost before this frame. The service records the missing index, leaves the committed sequence at 118, and sets the UI to DEGRADED. Recovery is bounded: request or await 119 only if the protocol supports retransmission; otherwise require an explicit epoch reset record. Sequence 120 must never be relabelled current merely because its bytes are intact.

**T4 — the corrupt path.** This arrival claims 128 payload bytes and supplies 127, and its CRC mismatches as expected. The validator quarantines it with both reasons attached, which matters because "length" and "checksum" are different evidence: a length error often indicates a framing or truncation fault, while a checksum error with a correct length often indicates corruption in transit. It consumes no queue slot and no bytes are repaired or padded. The UI may show an INVALID_INPUT indicator tied to that correlation ID while continuing to display that the last committed state is degraded. It must not paint the corrupt candidate as a value.

**T5 — the overflow path.** A structurally valid frame arrives when the FIFO already holds two unserved frames. Capacity is an integer, so the outcome is not a judgement call: the frame is dropped and an overflow record is written containing the correlation ID, the capacity, the depth at arrival, the policy revision, and the timestamp. The alternative — blocking the reader until a slot frees — converts a bounded queue into unbounded backpressure and would stall the acquisition path, so the policy is drop-with-record rather than block.

The overflow record is what keeps the denominator honest. Without it, six declared arrivals could appear as five handled arrivals and a complete-looking trace, and the missing 121 would be undetectable from the artifact. With it, the trace shows six arrivals, five dispositions, and one explicit drop — a completeness figure of five of six, which is a fact rather than a rounding.

**T6 — the stale path.** No new valid commit has occurred after source time 900 ms, and a render check at 2400 ms gives an age of 1500 ms. Since 1500 > 1000, the state is STALE rather than current, and the explanation panel switches to the abstention copy. The age is a computed display property under a declared clock assumption; it does not establish synchronisation between any real source and any real host.

### Sampling, transport, service, and age budget

Every figure in this subsection is ILLUSTRATIVE. The model assumes uniform 400 samples/s acquisition, non-overlapping frame collection, a 64-sample frame, a 128-byte payload, single-direction 8N1 serial framing, a FIFO service rate of 40 frames/s, and a serial additive path. It excludes retries, operating-system scheduling jitter, garbage collection, render variability, driver buffering, and any percentile behaviour. It is a design budget, never an observed latency and never a real-time guarantee.

```text
FORMULA [B-01]: T_collect = n / f_s
WHERE: T_collect [s] is collection time; n [samples/frame]; f_s [samples/s].
CONSTRAINT: contiguous uniform sampling; no overlap, drop, resampling, or clock drift.
NUMERICAL EXAMPLE: n = 64 samples; f_s = 400 samples/s;
                   T_collect = 64 / 400 = 0.16 s = 160 ms.
INTERPRETATION: acquisition alone consumes 160 ms of source time per frame,
                before any byte leaves the producer.
```

The dimensional check is `samples / (samples/s) = s`. If the denominator were a per-frame period instead of a rate, the same arithmetic would be inverted and the answer would be frames per sample, which is a quantity with no use here. The constraint matters more than the arithmetic: at 400 samples/s a 64-sample frame takes exactly 160 ms only if the first sample of the frame is taken at the start and no sample is skipped. Overlap in the acquisition loop breaks the formula in the optimistic direction, which is the dangerous direction.

```text
FORMULA [B-02]: T_transport = 10 * (H + P + C) / R
WHERE: T_transport [ms]; H header bytes; P payload bytes; C CRC bytes;
       R serial bits/s; the factor 10 assumes 8N1 (1 start, 8 data, 1 stop).
CONSTRAINT: one direction, no byte escaping, no retransmission, no contention,
            no driver buffering, and a line that actually sustains R.
NUMERICAL EXAMPLE: H = 20 bytes; P = 128 bytes; C = 4 bytes; R = 115200 bits/s;
                   bytes = 152; bits = 10 * 152 = 1520;
                   T_transport = 1520 / 115200 = 0.01319444 s = 13.1944 ms.
INTERPRETATION: the wire-time component is about 13.19 ms, roughly 8% of the
                collection time, so transport is not the dominant term here.
```

An independent arithmetic route confirms the same value: `10 / 115200 = 8.6806e-5 s` per byte, and `8.6806e-5 * 152 = 0.01319444 s`. The route fails if any byte needs escaping, because escaping changes the byte count after the formula's numerator is fixed. As a boundary check, a 12% escaping overhead on the same frame gives `152 * 1.12 = 170.24 bytes`, hence `1702.4 / 115200 = 14.7778 ms` — a 1.58 ms error, small here but proportional to frame size and therefore material for long payloads.

```text
FORMULA [B-03]: T_age = T_collect + T_transport + T_service + T_queue + T_render
WHERE: all terms [ms]; T_queue is the wait before service, T_service = 1000/mu.
CONSTRAINT: one clock convention relating source and render instants; a strictly
            additive path with no parallel branches and no overlap between stages.
NUMERICAL EXAMPLE: T_collect = 160; T_transport = 13.1944; T_service = 25
                   (mu = 40 frames/s); T_queue = 0; T_render = 18;
                   T_age = 160 + 13.1944 + 25 + 0 + 18 = 216.1944 ms.
INTERPRETATION: with an empty queue the modelled age is 216.19 ms, which clears
                a 250 ms illustrative budget by 33.81 ms.
```

The additive form is a modelling choice with a testable consequence: if any stage overlaps with another, the sum overstates the age, and the overstatement is bounded by the overlap. For a fully pipelined acquisition path, collection of frame `k+1` can run while frame `k` is transported, so the true age is `max(T_collect, ...)` on the pipeline rather than a sum. That variant is not modelled here and the omission is conservative only if the pipeline is not implemented; if it is implemented, this budget understates nothing and overstates age.

```text
FORMULA [B-04]: T_queue = K / mu * 1000
WHERE: K [frames] is the number of frames ahead in a FIFO; mu [frames/s].
CONSTRAINT: fixed service time, first-in-first-out discipline, no preemption,
            no priority class, and K frames that are all already admitted.
NUMERICAL EXAMPLE: K = 2 frames; mu = 40 frames/s;
                   T_queue = 2 / 40 * 1000 = 50 ms;
                   T_age = 216.1944 + 50 = 266.1944 ms.
INTERPRETATION: two frames ahead push the modelled age past the 250 ms budget
                by 16.19 ms, so the budget is a queue-depth-zero claim only.
```

This is where the formula's constraint does real work. `K/mu` is only the wait if every frame ahead takes exactly `1/mu`. With variable service times the wait is `K` times the mean service time in expectation, but any single arrival can wait much longer, and the maximum wait with `K` frames ahead is the sum of the `K` largest service times rather than `K` average ones. A queue-depth-two design therefore guarantees a bounded *count* of waiting frames and not a bounded *time*.

```text
FORMULA [B-05]: rho = lambda / mu ;  backlog = (lambda - mu) * t ;  T_drain = backlog / (mu_r - lambda)
WHERE: lambda [frames/s] arrival rate; mu [frames/s] service rate; t [s] an
       overload interval; mu_r [frames/s] the recovery service rate.
CONSTRAINT: a fluid approximation with constant rates, one server, infinite
            waiting room; real capacity is two frames, so the fluid backlog is
            an accounting quantity, not something the queue can hold.
NUMERICAL EXAMPLE: lambda = 48 frames/s; mu = 40 frames/s; t = 90 s;
                   rho = 48/40 = 1.2;  backlog = (48-40) * 90 = 720 frames;
                   T_drain = 720 / (64 - 48) = 720/16 = 45 s.
INTERPRETATION: the offered load exceeds capacity by 20%, so the queue is
                unstable and the design's obligation is explicit dropping with
                records, not a claim that two slots preserve all work.
```

The drain arithmetic is worth re-deriving because the first instinct is often wrong: with `mu_r = 64` and `lambda = 48`, spare capacity is 16 frames/s, and `720 / 16 = 45 s`, not `720/64 = 11.25 s`. The second number is the time to serve the backlog if arrivals stopped, which is a different question. As a boundary case, `mu_r = 48` gives zero spare capacity and the drain time is undefined — the backlog persists forever in the fluid model, and no amount of scheduling recovers it.

```text
FORMULA [B-06]: C_trace = dispositions_recorded / arrivals_declared
WHERE: both counts are integers over the frozen fixture set; a disposition is
       any recorded outcome, including duplicate, gap, corrupt, and overflow.
CONSTRAINT: the denominator is fixed before evaluation and every declared arrival
            is counted exactly once; an excluded arrival stays in the denominator
            with an exclusion reason.
NUMERICAL EXAMPLE: 6 declared arrivals (T1-T6); T1-T5 each carry a disposition;
                   T6 is a clock-driven state observation recorded as a
                   disposition of "no arrival";
                   C_trace = 6 / 6 = 1.000.
                   Under a five-arrival variant that omits T5's overflow record,
                   C_trace = 5 / 6 = 0.8333.
INTERPRETATION: completeness is binary in this fixture, and the 0.8333 variant is
                the exact shape a silently dropped frame takes in a trace.
```

The completeness ratio is a weak measure that becomes strong only when paired with a rule it cannot game. The complementary rule is that a disposition must name the correlation ID it belongs to, so completeness cannot be raised by recording a disposition for the wrong arrival. Two records that both claim arrival `c-05` raise the count without adding evidence, and a duplicate correlation ID is therefore a validation error rather than a completeness contribution.

| Sensitivity | Changed input | Recomputed result | Consequence |
|---|---|---|---|
| Slower link | `R = 57600 bits/s` | transport 26.3889 ms; zero-queue age 229.3889 ms | still clears 250 ms, margin falls to 20.61 ms |
| Larger frame | `n = 112 samples` at 400/s | collection 280 ms; age alone exceeds budget by 30 ms | the 250 ms budget cannot hold at this frame size |
| Slower service | `mu = 20 frames/s` | service 50 ms; zero-queue age 241.1944 ms; two ahead 341.1944 ms | the zero-queue claim survives, the queue claim collapses |
| Faster render | `T_render = 5 ms` | zero-queue age 203.1944 ms; two ahead 253.1944 ms | render savings cannot rescue a queue-depth claim |
| Wider escaping | 12% byte overhead | transport 14.7778 ms; zero-queue age 217.7778 ms | transport error is real but not decision-changing here |
| Deeper queue | `K = 4`, `mu = 40` | queue 100 ms; age 316.1944 ms | a four-slot queue turns a passing claim into a stale one |

The last row is the most useful for design review, because queue capacity is usually chosen for throughput and then published as a latency claim. Here capacity two costs 50 ms and capacity four costs 100 ms against a 250 ms budget, so any capacity increase consumes budget that the acquisition stage already spent most of.

**Boundary counterexample.** Set `n = 112` at `f_s = 400 samples/s`. `T_collect = 112/400 = 0.28 s = 280 ms`, which exceeds the entire 250 ms budget before transport, service, queue, or render are counted. This is the case where no downstream optimisation helps: render could be reduced to zero and the frame would still be 280 ms old at the moment it is complete. Any design discussion that starts from UI responsiveness rather than frame size is therefore mis-ordered, and the counterexample is the cheapest way to demonstrate that.

**A second boundary.** Formula B-04 breaks when service is priced per byte rather than per frame. If `T_service = c * (H + P + C)`, then queue wait is `K * c * 152 ms` and the fixed `1/mu` form is wrong by a factor that grows with payload size. Mixing the two conventions in one budget — a per-frame service term and a per-byte queue term — produces a number that is dimensionally consistent and practically meaningless, which is the hardest kind of error to catch in review.

### Where the modelled time actually goes

The budget's four terms are unequal by an order of magnitude, and knowing the ranking prevents the most common optimisation mistake.

| Term | Value [ms] | Share of 216.1944 ms | What changes it | What does not |
|---|---|---|---|---|
| Collection | 160.0000 | 74.0% | frame size, sample rate | baud rate, UI framework, model size |
| Transport | 13.1944 | 6.1% | byte count, baud rate, escaping | sample rate, render path |
| Service | 25.0000 | 11.6% | service rate per frame | payload contents |
| Queue | 0.0000 | 0.0% | arrivals versus capacity | anything upstream |
| Render | 18.0000 | 8.3% | UI revision, host load | protocol, framing |

Recomputing the shares: `160 / 216.1944 = 0.74007`, `13.1944 / 216.1944 = 0.06103`, `25 / 216.1944 = 0.11564`, `18 / 216.1944 = 0.08326`. The four shares sum to `1.0000` within rounding. A 10% improvement in transport buys `1.3194 ms`; a 10% reduction in frame size buys `16 ms`. The ratio of those two benefits is about 12 to 1, and the ratio of their effort is usually the reverse, which explains why optimisation effort so often lands on the term that cannot move the budget.

A corollary applies to portfolio narrative. A student who profiles the parser and reports a 40% parser speedup has made a true statement about a term that is inside service. If service is 25 ms, a 40% parser improvement moves the total by at most `25 × 0.4 = 10 ms` and in practice less, because service also includes the commit. Reporting that 40% beside a 216 ms age implies a 40% system improvement to a casual reader, which is precisely the C-04 violation.

### Why thresholds need their own derivation

The values `500 ms` and `1000 ms` were chosen, and a chosen threshold presented without a rationale invites readers to treat it as a requirement. The honest derivation is comparative: set the degraded boundary above the modelled worst-case non-queue age so that ordinary operation cannot trip it, and set the stale boundary at twice that so a single missed service interval is visibly different from a sustained outage.

```text
FORMULA [B-07]: theta_degraded >= max(T_collect + T_transport + T_service + T_render)
                theta_stale = 2 * theta_degraded
WHERE: theta_degraded [ms] is the age at which the display stops saying CURRENT;
       theta_stale [ms] is the age at which it says STALE; the max is over the
       declared operating envelope, not over observed runs.
CONSTRAINT: the envelope's maximum collection time must be declared; if the frame
            size can vary at runtime, the maximum frame size sets the max.
NUMERICAL EXAMPLE: envelope maximum at 64 samples/400 Hz:
                   max = 160 + 13.1944 + 25 + 18 = 216.1944 ms, so 500 ms leaves
                   a factor of 500/216.1944 = 2.313 of headroom;
                   theta_stale = 2 * 500 = 1000 ms.
INTERPRETATION: the thresholds are 2.31x and 4.63x the modelled nominal age, so
                they absorb jitter and one missed service interval respectively.
```

The formula exposes the failure case immediately. If the envelope is widened to 112 samples, the maximum becomes `280 + 13.1944 + 25 + 18 = 336.1944 ms`, and `500/336.1944 = 1.487` — still above one, so CURRENT remains reachable, but the headroom has more than halved. At 180 samples, collection alone is `450 ms` and the 500 ms degraded boundary is `1.11×` nominal, at which point the thresholds must be re-derived rather than reused. Reusing them would produce a display that flickers between CURRENT and DEGRADED under normal operation, and the flicker would be diagnosed as a UI bug rather than a threshold bug.

The `theta_stale = 2 * theta_degraded` rule has a limit worth naming: it assumes missed service intervals arrive one at a time. If the service loop can miss two intervals in a row during host contention, the stale boundary sits below the real worst case and the display will say CURRENT during a visible gap. A stricter derivation sets `theta_stale` above the largest *observed* gap between commits plus the largest single service time; the doubling rule is a design-time placeholder until such observations exist, which is why the thresholds carry the label "design choices" rather than "requirements".

### Transport error sources and their magnitudes

B-02 is a lower bound on wire time, and the gap between it and reality has identifiable components. Naming them converts a vague caution into a list a reviewer can check.

| Source | Effect on the B-02 result | Direction | Detectable by |
|---|---|---|---|
| Byte escaping or stuffing | adds bytes after the numerator is fixed | under-estimate | comparing declared versus observed byte counts |
| Start/stop framing error | a re-sent frame doubles the time for that frame | under-estimate | duplicate detection at the receiver |
| Achieved versus declared baud | clock tolerance of the sender | either | instrumented capture at both ends |
| Driver buffering | bytes leave later than the service writes them | under-estimate | timestamp at the pin, not in user space |
| Contention with other traffic | interleaving delays | under-estimate | gap pattern in the capture |
| Line quality | bit errors causing retransmission | under-estimate | CRC failure rate at the receiver |

Every row in the table biases the same way, which is a useful property to notice: B-02 is optimistic by construction, so a budget that passes with substantial margin is robust, and a budget that passes narrowly is fragile. The margin from Section 6 is `250 − 216.1944 = 33.8056 ms`, or about `15.6%` of the modelled total. Whether that is enough depends entirely on whether the escaping and buffering terms are bounded, and neither is bounded in this design.

**Recomputing the margin under a pessimistic transport assumption.** Suppose escaping adds 12% and a driver buffer adds a fixed 8 ms. Then transport becomes `14.7778 ms` and the total becomes `160 + 14.7778 + 8 + 25 + 18 = 225.7778 ms`, leaving `24.2222 ms` of margin, or `10.7%`. The margin has lost about a third of its value without any code change — only a change in what the transport model admits. This is the argument for publishing the assumption set beside the number rather than the number alone.

### What the fixture cannot show

The six arrivals exercise disposition logic and nothing else. They cannot show how the system behaves when two frames arrive in the same millisecond, because the arrivals are ordered by construction. They cannot show how the UI behaves when a state changes twice between two renders, because the renders are not part of the fixture. They cannot show memory growth, because six arrivals do not accumulate.

A reviewer reading the trace should therefore ask a specific question: which property of a real system does this fixture deliberately not exercise? The answer is concurrency, and the deliberate omission is defensible only if the design's concurrency story is stated elsewhere. It is stated in the queue's capacity and in the atomic commit, but neither is tested by a serial fixture, and the manifest's limitations list must say so.

The practical alternative is a two-producer fixture in which arrivals `T3` and `T4` are injected in the reverse of their logical order. That single change exercises out-of-order arrival, which is the mechanism most likely to break a naive implementation and the one least likely to appear in a hand-built trace. It costs one extra fixture row and closes the largest blind spot in the current oracle.

### Gate reasoning

| Gate condition | Trace requirement | Current manuscript status | Decision effect |
|---|---|---|---|
| Frame contract | schema, units, CRC coverage, ordering policy, all as integers or exact ranges | specified only | not trace-checked |
| Trace oracle | frozen fixture plus a disposition for every declared arrival | constructed by hand, never executed | not trace-checked |
| Manifest | every bundle identity and hash that a reviewer can compare | schema specified, nothing emitted | not reproducible as a run |
| Retrieval | a chunk-level mapping for every factual sentence, or an abstention | policy specified only | no explanation-quality claim |
| UI | recorded state transitions for each of the five states | none | no UI-rendering claim |
| HIL | board, firmware, wiring, serial parameters, instrumentation, repetitions | none | no HIL claim |

The gate decision follows mechanically from the table: six required conditions, six unsatisfied, therefore DESIGN-only. The arithmetic in Section 6 is checked and internally consistent, but no external artifact binds a single line of it to an implementation. Relabelling a hand-reasoned trace as TRACE-CHECKED would be a category error about what the tier means.

There is one subtlety worth recording. The trace above is complete — six arrivals, six dispositions — and completeness is often mistaken for the trace tier. Completeness of a designed oracle is a property of the document. The tier is a property of the artifact, and it requires an execution. A reviewer who accepts the table as evidence has accepted reasoning about a system that may never have existed.

## 7. The Finding

The release remains DESIGN-only, and the permitted artifact is a design record that keeps its limitation statement attached. Under the stated illustrative model the zero-queue age is 216.1944 ms and the two-frames-ahead age is 266.1944 ms against a 250 ms budget, so the only supportable timing sentence is one that names the queue depth it assumes. A 112-sample frame at 400 samples/s takes 280 ms to collect, which exceeds the entire budget, so frame size — not UI speed — controls whether any timing claim is available.

| Finding | Value or statement | Derivation | Claim strength |
|---|---|---|---|
| Nominal modelled age, empty queue | 216.1944 ms | B-01 + B-02 + B-03 | illustrative, arithmetic checked |
| Modelled age, two frames ahead | 266.1944 ms | B-04 added to the above | illustrative, arithmetic checked |
| Budget headroom, empty queue | 33.8056 ms | 250 − 216.1944 | illustrative |
| Frame size at which collection alone exceeds budget | 112 samples at 400/s | 250 ms × 400/s = 100 samples | illustrative boundary |
| Offer-load ratio under overload | 1.2 | 48 / 40 | illustrative |
| Backlog after 90 s of overload | 720 frames | (48 − 40) × 90 | illustrative |
| Time to drain at 64 frames/s | 45 s | 720 / (64 − 48) | illustrative |
| Trace completeness, designed oracle | 6 of 6 | B-06 | design property, not a run |

The boundary row deserves its own sentence because it is a design rule and not just a calculation: at 400 samples/s, `250 ms × 400/s = 100 samples` is the largest frame that can be collected inside the budget with zero margin for anything else. Any frame size above 100 samples makes the budget unachievable regardless of downstream engineering. This converts an abstract "keep frames small" preference into a number tied to a declared rate and a declared budget.

The finding also constrains language. Three sentences are permitted and three are not:

| Permitted wording | Basis |
|---|---|
| "The design specifies a 216 ms modelled age for an empty-queue 64-sample frame at 400 samples/s." | B-01 to B-03, labelled illustrative |
| "Queue depth is part of the latency claim; two frames ahead exceeds the illustrative budget." | B-04 |
| "The validator contract rejects a 143-byte payload declared as 128 bytes with both reasons recorded." | T4 oracle, design property |

| Prohibited wording | Why |
|---|---|
| "The demonstrator responds in 216 ms." | no execution, no measurement, no path definition |
| "Real-time performance was validated." | no hardware, no percentiles, no instrumentation |
| "The explanation was grounded in the corpus." | no retrieval run, no chunk ledger, no evaluation |

## 8. System/Solution Design

```text
[fixture or acquisition] -> [canonical frame] -> [validator] -> [bounded FIFO] -> [atomic state]
                                  |                 |                |
                                  v                 v                v
                           [quarantine log]  [rejection reasons]  [overflow record]
                                                                       |
   [UI: CURRENT | DEGRADED | STALE | INVALID_INPUT | ABSTAIN] <- [age classifier]
                    |
                    v
   [manifest] <--- [oracle comparator] <--- [retrieval provenance checker]
```

Ownership is drawn along the arrows, not along the boxes, because every defect this design anticipates occurs at a boundary between two owners rather than inside one component.

| Boundary | Left side supplies | Right side must not assume | Failure if the contract is vague |
|---|---|---|---|
| acquisition → frame builder | sample count, rate, unit, source time | that samples are contiguous | an age computed from a frame that skipped samples |
| frame builder → validator | canonical bytes, declared lengths, CRC | the algorithm or covered range | an integrity result that two implementations disagree on |
| validator → FIFO | an admitted frame plus its correlation ID | that admission implies ordering | a gapped frame committed as current |
| FIFO → state | one frame, one slot, one service instant | that admission implies capacity | silent loss under overload |
| state → UI | committed sequence, state version, source time | that the UI clock is the same clock | a stale chart labelled current |
| state → retrieval | committed frame summary plus version | that a retrieved sentence is attributable | an explanation with no source mapping |
| everything → manifest | versions, hashes, thresholds, policies | that a name is an identity | a bundle that cannot be reproduced |

**UI state contract.** The five states are mutually distinguishable by the facts they require and by what the explanation panel may do in each.

| UI state | Preconditions | Required display | Explanation behaviour | Recovery |
|---|---|---|---|---|
| CURRENT | last valid committed frame; age ≤ 500 ms | state version, age, manifest short ID | cited, attributable assistive text only | a new valid contiguous frame |
| DEGRADED | gap, overflow, or age 501–1000 ms | reason, age, last valid sequence | withhold new factual explanation | a validated resync or a fresh contiguous frame |
| STALE | age > 1000 ms | stale label and last-update age | withhold explanation | a new valid frame with a recomputed age |
| INVALID_INPUT | the current arrival failed validation | invalid reason and correlation ID | never sourced from the invalid arrival | submit a valid frame |
| ABSTAIN | retrieval mapping incomplete or a miss | source-unavailable statement | no factual completion | supply a mapped source |

The states are orthogonal in one important respect: freshness and validity apply to different objects. Freshness applies to the last committed state; validity applies to the newest arrival. A corrupt arrival does not make the committed state stale, and a stale state does not make the newest arrival invalid. The T4 row in the Section 6 trace shows both facts held simultaneously, and the UI must be able to express that conjunction without merging the two into a single ambiguous badge.

**Promotion rules.** The tier moves up only when every condition in the row below holds, checked by the reviewer rather than the implementer.

| Promotion | All of these must hold | Any single failure means |
|---|---|---|
| DESIGN → TRACE-CHECKED | fixture frozen with hashes; oracle frozen before execution; manifest emitted and complete; every declared arrival has exactly one disposition; duplicate safety demonstrated by a repeated identity; the trace is inspectable by the reviewer | stay at DESIGN and record the missing artifact |
| TRACE-CHECKED → HIL-VALIDATED | the above, plus a named board and firmware revision, a wiring and serial-parameter record, an instrumented capture of the physical path, at least three repetitions, and a stated failure count | stay at TRACE-CHECKED and publish only the software claim |
| Any tier → publication | permitted wording mapped to the earned tier, with limitations attached to the same artifact | withhold publication rather than hedge |

**Demotion rules.** Tiers also move down, and the conditions are as concrete as promotion. A published TRACE-CHECKED claim demotes to DESIGN if the fixture hash changes, if the oracle revision changes without re-execution, if a disposition is found to belong to the wrong correlation ID, if the manifest is missing a field that a component actually reads, or if the committed sequence is shown to have advanced on a gapped frame. The demotion is a wording change first and an engineering task second: retract the sentence, then fix the artifact.

**Recovery contract.** Recovery starts from a compatible manifest and a known committed sequence, replays the frozen fixture, and ends only when every disposition matches the oracle without a duplicate state effect. Restarting a process is not recovery; it is a process event. The distinction is testable: after recovery, the state version should equal the pre-failure version plus exactly the number of newly admitted frames, and the duplicate count should be unchanged.

## 9. Evaluation & Validation

| Metric | Numerator / denominator | Window / comparator | Guardrail | Status |
|---|---|---|---|---|
| Validator correctness | fixtures matching the declared disposition / all declared fixtures | one frozen fixture set against ORA-01 | a corrupt frame is never admitted | not executed |
| Duplicate safety | duplicate arrivals producing zero additional `state_version` increments / duplicate arrivals | the trace's atomic records | distinct frames with equal payloads are both retained | not executed |
| Gap disclosure | gap fixtures rendered DEGRADED / gap fixtures | trace plus UI-state oracle | committed sequence never advances across a gap | not executed |
| Stale disclosure | age checks rendered STALE / checks above the threshold | clock-defined trace plus UI oracle | a prior chart is never labelled current | not executed |
| Overflow accounting | overflow records present / frames dropped for capacity | the trace's admission log | arrival count equals disposition count plus exclusions | not executed |
| Retrieval support | displayed factual sentences with a chunk mapping / factual sentences displayed | a versioned prompt suite, human audit | an unmapped sentence is withheld | not executed |
| Manifest completeness | required fields a reviewer can hash-match / required schema fields | one release candidate | any missing field blocks promotion | not executed |
| End-to-end age | correlated paths within the stated budget / all completed correlated paths | a declared percentile over a declared path | clock semantics and tails captured | not measured |

**Uncertainty.** Three sources dominate and none is currently quantified. First, the additive age model ignores jitter, so the modelled 216.1944 ms is a central estimate with no interval; a real distribution would have a right tail that the budget must absorb. Second, the service rate of 40 frames/s is a design assumption, and a factor-of-two error moves the two-frame age from 266.19 ms to 341.19 ms, which crosses the budget decisively. Third, the threshold values of 500 ms and 1000 ms are choices, so any statement that the display is "fresh" is relative to an arbitrary boundary and must be published with that boundary attached.

**Population and comparator.** The population for every software metric is the six declared arrivals. The comparator is the frozen oracle at ORA-01, not a previous run and not a rival implementation. Using a previous run as the comparator answers "did it change" rather than "is it correct", and the two questions diverge exactly when a bug is stable.

**Guardrail discipline.** Each row has one guardrail that must pass even if the headline metric fails. If validator correctness is 6 of 6 but a corrupt frame was admitted during the run, the metric is void — the headline was achieved by a path the guardrail forbids. This ordering (guardrail first, metric second) is what stops a trace from being tuned toward the count.

**Executed versus not executed.** Nothing in the table above has been executed. No fixture file was encoded, no oracle was frozen to disk, no manifest was emitted, no code was written, no UI was rendered, no board was connected, and no timing was instrumented. The status column says "not executed" or "not measured" for every row deliberately, and that column is the single most important line of the artifact for a reviewer deciding what to trust.

**Why HIL is a separate evaluation and not an extension.** A desktop replay exercises parsing, validation, state transitions, and rendering against bytes read from a file. It cannot exercise clock recovery in a reader, a driver's buffer behaviour, a physical ground reference, cable-induced errors, or the difference between a declared baud rate and an achieved one. Those are not additional test cases for the same system; they are a different system, with the fixture producer replaced by a physical source. Adding them to the same evaluation table would imply a continuity of evidence that does not exist.

### Test design: what one fixture row can and cannot prove

Each arrival in the oracle is a single test case, and the value of a test case lies in the number of distinct hypotheses it can falsify. A row that only confirms the happy path falsifies almost nothing, which is why five of the six rows are negative cases.

| Arrival | Hypothesis it falsifies if the expected outcome is not observed | Hypotheses it leaves untouched |
|---|---|---|
| T1 | the validator rejects a conforming frame | anything involving repeats, gaps, or capacity |
| T2 | the duplicate check is missing or non-atomic | gap handling, overflow, clocks |
| T3 | sequence continuity is assumed rather than checked | duplicates, capacity, retrieval |
| T4 | length and checksum are conflated or the parser pads | ordering, capacity, clocks |
| T5 | capacity is unbounded or overflow is silent | duplicates, clocks, retrieval |
| T6 | age is computed from an arrival rather than a commit | validity, capacity, retrieval |

The design principle visible in the table is that each row targets exactly one mechanism, so a failure localises. A row that exercises two mechanisms at once — say a duplicate that is also corrupt — produces an ambiguous result: a pass confirms both, and a failure tells you one of two things went wrong without saying which. Combined rows are worth adding after the single-mechanism rows pass, as a regression net, not as the primary oracle.

**Ordering constraints inside the fixture.** The six arrivals are order-dependent: T2 is a duplicate of T1 and cannot precede it; T3's gap only exists because 119 was never committed; T4's corrupt 119 arrives after the gap that expected it. This ordering is a fixture property and must be hashed with the fixture, because reordering the rows produces a different trace with different expected dispositions. A reader who shuffles the table to "make it tidier" invalidates the oracle without changing a single expected value.

**What the T2/T4 pair establishes.** T2 shows that an identical identity with identical content is dropped. T4 shows that an identical identity with conflicting content is also dropped, but for a different reason and with a different disposition. Together they pin the rule that identity dominates content: whether the bytes agree has no bearing on whether the frame is admitted. Neither row alone establishes that, and a trace containing only T2 would be consistent with a payload-hash deduplication rule that fails on stationary signals.

### Statistical honesty for a six-row fixture

With six arrivals there is no rate to estimate and no confidence interval worth computing, and the honest statement is that the fixture is a specification test rather than a sample. A reviewer who asks "how often does the validator get this right" is asking a question the artifact cannot answer and should not pretend to.

| Question a reviewer might ask | Answerable from six rows? | Correct response |
|---|---|---|
| Does the validator implement the declared contract? | yes, for the six declared cases | report 6 of 6 with the fixture listed |
| What is the per-arrival error rate in operation? | no | state that no rate is estimable from a specification fixture |
| How does it behave on unseen malformed input? | no | name fuzzing as a separate, unexecuted activity |
| Is the disposition rule stable across runs? | partially | repeat the same fixture and compare dispositions |
| Does it scale to concurrent producers? | no | add the two-producer fixture described above |

The third row is where portfolio claims usually overreach. A trace-checked claim is a claim about the declared cases, and "the validator is robust" is a claim about unseen cases. The distance between them is exactly the fuzzing campaign that the artifact does not contain, and a single sentence in the limitations list is enough to keep the two apart.

### Guardrail interaction

Guardrails are not independent, and a run can satisfy each individually while violating their conjunction. The interaction matters because it is invisible in a per-metric report.

| Guardrail pair | Individually satisfiable | Jointly violated when | Detect by |
|---|---|---|---|
| corrupt never admitted; complete dispositions | yes | a corrupt frame is admitted and later reclassified as a gap | comparing the validator reason to the eventual state for one ID |
| sequence never advances across a gap; complete dispositions | yes | the gap is relabelled a duplicate to keep the count complete | checking epoch and sequence against the predecessor record |
| duplicate creates no state change; overflow recorded | yes | a duplicate is counted as an overflow drop | examining the reason code on the disposition |
| stale never labelled current; age recomputable | yes | the render clock is substituted for the source clock | recomputing age from the raw timestamps |

The common structure is that a disposition can be *correct in count* and *wrong in kind*. This is why the metric table pairs each count with a guardrail on the reason codes, and why the oracle comparator must compare dispositions as ordered pairs of (action, reason) rather than as bare actions. A comparator that only checks "was this arrival handled" passes every row in the interaction table.

## 10. Action Taken

The only action taken is the proposed release decision itself: block every performance, device, clinical, HIL, UI-tested, corpus-quality, and deployment claim; permit a clearly labelled DESIGN-only artifact. No code, fixture, manifest, device setup, UI, corpus, or validation run was created or executed in the course of this case.

| Proposed next action | Owner | Required input | Stop condition | Claim unlocked if passed |
|---|---|---|---|---|
| Freeze the fixture and oracle | fixture owner | canonical bytes and expected dispositions, authored before execution | hash mismatch or an ambiguous disposition | none alone |
| Implement the manifest writer | release owner | component identities and hashes | any required field missing | none alone |
| Execute the deterministic trace | service owner | frozen fixture, frozen oracle, emitted manifest | any undocumented disposition | TRACE-CHECKED candidate |
| Inspect recorded UI transitions | UI owner | captured state changes for all five states | a state hidden or two states merged | a UI-behaviour statement only |
| Define the HIL protocol | hardware owner | named setup, instrumentation plan, repetition count | a setup or instrument absent | HIL candidate |
| Independent scope review | reviewer | the bundle plus the proposed wording | any claim outrunning the evidence | publication approval |

The ordering is not arbitrary. The fixture must precede the oracle, because an oracle written after seeing the output is a transcript. The oracle must precede execution, because an expectation adjusted after the fact is a rationalisation. The manifest must precede execution, because a bundle assembled afterwards cannot prove which versions produced the result. Each of these orderings costs one cycle and each is invisible in the final artifact, which is why they are written down rather than assumed.

A note on what "block" means in practice. The block is not a refusal to publish; it is a refusal to publish certain sentences. The artifact can go up today with the design record, the oracle table, the budget derivation, and the limitation statement. What cannot go up is any sentence in the prohibited column of Section 7. Framing the gate as a wording constraint rather than a publication ban is what makes it survive contact with a deadline.

### Which action to take first, and why the order is not the obvious one

The tempting order is to build the demonstration and then gather evidence. The gate's order is the reverse, and the reason is that evidence gathered after construction is shaped by construction: an author who has seen the output writes an oracle that agrees with it, usually without intending to.

| If you do this first | You will then be tempted to | The defect that results | Cheaper alternative |
|---|---|---|---|
| Build the UI | describe the design in terms of the UI that exists | design narrowed to what is easy to render | specify states first, render second |
| Build the parser | write the oracle from parser output | a transcript passing as an oracle | freeze bytes and expectation first |
| Collect a live trace | treat the capture as the fixture | an input that cannot be replayed | freeze a fixture, then capture separately |
| Write the README | justify the tier from the prose | claims outrunning evidence | write wording last, from the manifest |

The last row is the one that fails most often in portfolio work, and it fails quietly. A README written before the evidence exists will contain hedged claims that read as decisive ("validated against representative inputs"), and once published the hedge is invisible. Writing the wording last, from the manifest, produces shorter and duller sentences that happen to be true.

### What "done" looks like for this candidate

A reviewer asking whether the candidate is finished needs a definition, and the useful one is a list of artefacts rather than a feeling of completeness.

| Artefact | Present today | Required for TRACE-CHECKED |
|---|---|---|
| This design record | yes | yes, retained as the limitations annex |
| Oracle table with six dispositions | yes, in prose | yes, frozen to a file with a hash |
| Canonical fixture bytes | no | yes, with two independent encodes compared |
| Manifest | schema only | yes, every field populated and hash-checkable |
| Execution record | no | yes, with per-arrival dispositions from the run |
| Review record from an independent reader | no | yes, one page noting what could not be verified |

The gap between the two columns is small in kind and large in effort: four artefacts, none of which requires new design thinking. That is the strongest argument that the DESIGN tier here reflects missing work rather than missing insight, and it is the argument to make when a reviewer asks why the artifact stops where it does.

## 11. Outcome & Results

**Outcome status: not measured.** No implementation artifact supports an observed result. The checked outputs of this document are a constructed six-arrival oracle, a six-formula budget, and a set of boundary values, every one of them arithmetic rather than empirical.

| Claim ID | Bounded claim | Evidence type | Supports | Does not establish |
|---|---|---|---|---|
| CLM-01 | A screenshot alone cannot support a trace-tier claim. | derived | the need for fixture, oracle, and manifest | any implementation behaviour |
| CLM-02 | The T1–T6 oracle specifies valid, duplicate, gap, corrupt, overflow, and stale outcomes. | illustrative | a future expected-behaviour contract | that any software produced these outcomes |
| CLM-03 | The stated additive model gives 216.1944 ms with an empty queue. | illustrative, arithmetic checked | budget reasoning at 64 samples and 400 samples/s | latency, percentile, or jitter |
| CLM-04 | Two frames ahead at 40 frames/s add 50 ms, giving 266.1944 ms. | illustrative, arithmetic checked | the claim that queue depth belongs in the latency sentence | any queue implementation |
| CLM-05 | At 400 samples/s, 100 samples is the largest frame that fits a 250 ms collection budget. | illustrative boundary | frame-size constraint tied to rate and budget | that any acquisition path achieves 400 samples/s |
| CLM-06 | The current gate is DESIGN-only. | derived from six OPEN conditions | the permitted wording set | any future trace or HIL result |

Two observations about this table are worth stating because they are the kind of thing a reviewer catches and an author misses. First, no claim uses the word "performance"; CLM-03 and CLM-04 are statements about a model, and the word "model" appears in each. Second, CLM-05 is a boundary statement that is true independently of the implementation — it follows from the budget definition and the rate alone, which makes it the most durable claim in the set even though it is the least impressive.

### What could be observed today

It is worth separating "no result exists" from "nothing is knowable", because they are different statements and the second would be false.

| Quantity | Status | Why |
|---|---|---|
| Modelled empty-queue age | known by construction | pure arithmetic under stated assumptions |
| Frame size that exhausts the budget | known by construction | `250 ms × 400 /s = 100` samples |
| Offered-load ratio under overload | known by construction | `48/40 = 1.2` |
| Backlog after 90 s of overload | known by construction | `8 × 90 = 720` frames |
| Real service rate of any implementation | unknown | no code was run |
| Real collection jitter | unknown | no acquisition loop exists |
| Render time on any host | unknown | no UI was rendered |
| Duplicate behaviour of any implementation | unknown | no commit path exists |

The pattern is that every "known" row is a consequence of the design's own parameters, and every "unknown" row is a property of an implementation. This is the cleanest available statement of the DESIGN tier: the design is knowable in full, and the implementation is not knowable at all because it does not exist.

### Why the honest baseline is also the useful one

A reader might reasonably ask why the prohibited wording in Section 7 matters so much when the artifact plainly states its status. The answer is that the artifact's status statement competes with everything else in it for a reader's attention, and the numbers are more memorable than the caveat.

Consider the reading order of a portfolio page: a title, a screenshot, a one-line summary, then details. If the one-line summary contains a millisecond figure with no qualifier, that figure becomes the reader's model of the artifact, and the limitation paragraph three screens later modifies a belief that has already formed. The tier discipline is not pedantry about words; it is an attempt to put the qualifier in the same sentence as the number, where it cannot be separated by scrolling.

| Placement | Reader outcome | Verdict |
|---|---|---|
| "Modelled at 216 ms (illustrative, empty queue, 64 samples)" | figure and condition arrive together | acceptable |
| "216 ms end-to-end latency" plus a limitations section | figure forms the belief, condition never lands | prohibited |
| No figure at all | reader cannot evaluate the design | under-claiming, acceptable but weak |
| "Sub-second response" | unfalsifiable, implies measurement | prohibited |

The second and fourth rows fail for the same reason from different directions: one attaches a precise number to a false frame, the other attaches a vague word to no measurement. Both leave the reader unable to reconstruct what was actually established, which is the single test the wording has to pass.

### Residual risk register

| Risk | Likelihood | Impact if realised | Mitigation in this artifact |
|---|---|---|---|
| The fixture is encoded inconsistently | medium without A-06 | every disposition is about the wrong bytes | require two independent encodes |
| Thresholds are reused after an envelope change | high | a flickering CURRENT/DEGRADED display | derive thresholds from the envelope maximum (B-07) |
| A component timing is read as end-to-end | high | an inflated performance belief | C-04 plus the share table |
| The oracle is written after the run | medium | a transcript passing as a test | freeze the oracle hash first |
| Overflow is discovered only in production-like load | high | silent loss and a dishonest denominator | integer capacity plus overflow records |
| Retrieval citations decay | medium | grounded-looking text with stale sources | index revision in the manifest |

The second and fifth rows are the ones this design is least protected against, and they are connected: if the thresholds are re-derived after the envelope widens, the overflow behaviour that produced the widening is invisible unless the overflow records exist. Closing both means one change — record capacity outcomes everywhere — and that single change addresses the two highest-impact risks in the register.

## 12. Failure Modes & What Went Wrong

| Failure mechanism | Competing causes | Discriminating check | Containment | Prevention |
|---|---|---|---|---|
| Duplicate produces a second state change | producer retry; reader reconnect replay; non-atomic commit | same `(stream, epoch, sequence)` with two distinct `state_version` values | stop the side effect, flag the trace, demote the claim | commit identity and state in one transaction, then replay the duplicate fixture |
| CRC valid, meaning wrong | unit mismatch; calibration offset; schema reinterpretation | the unit ledger disagrees with the fixture contract while the CRC matches | withhold interpretation, keep the frame quarantined | versioned units plus a semantic oracle that does not use the CRC |
| Gap masked as current | loss in transit; reader parse loss; sender restart | committed sequence jumps without an epoch reset record | show DEGRADED, do not advance the sequence | explicit resync rule and a gap fixture in every run |
| Corrupt frame displayed as a value | length bug; CRC scope mismatch; UI mapping error | the validator's reason and the UI's state disagree for one correlation ID | quarantine the candidate, keep the prior state visible | one shared state enum and a render-state oracle |
| Stale view presented as live | queueing; clock mismatch; frontend cache | source, receipt, and render times imply an age above the threshold while the UI says current | relabel STALE, withhold the explanation | declared clock semantics and a forced-stale fixture |
| Queue silently loses evidence | overflow; task cancellation; an unbounded producer | arrivals exceed disposition count with no exclusion reason | block release, recount the denominator | integer capacity, an overflow record, and an accounting assertion |
| Explanation cites nothing | retrieval miss; stale index; prompt invention under pressure | a displayed factual sentence has no chunk mapping | abstain and record the miss | withhold-by-default rendering plus a mapping audit |
| Manifest cannot reproduce the run | schema, model, prompt, index, or config drift | a required field is missing or a hash does not match | block promotion | immutable bundle identity and a schema check in the build |
| Timing overclaim | a component mean printed as an end-to-end value; omitted collection time | the published path is not the timed path | retract the sentence | name the path and denominator beside every number |
| HIL overclaim | a desktop replay described as hardware validation | no setup record and no instrumented physical capture | remove HIL wording | separate tier with its own artifact list |
| Oracle written after the run | single-author workflow; schedule pressure | oracle revision timestamp is later than the run timestamp | discard the run, re-freeze, re-execute | freeze oracle hash before execution |
| Denominator quietly narrowed | exclusions applied after seeing outcomes | excluded count is nonzero and unrecorded | restore the exclusions, restate completeness | exclude by rule before evaluation, record every exclusion |

The first four rows share a structural cause and are worth separating from the rest. Each of them is a disagreement between two representations of the same arrival — parsed fields versus bytes, committed state versus displayed state, validator reason versus UI badge. The discriminating check is always the same shape: pick one correlation ID and compare what every plane recorded about it. When the three planes agree for every ID, the class is closed; when they disagree for one, the whole run is suspect, because the mechanism that corrupted one record is not usually confined to one record.

The last four rows are procedural rather than technical, and they are more common in portfolio work than any parsing bug. An oracle written after the run, a denominator narrowed after seeing results, a component timing printed as end-to-end, and a desktop replay called hardware validation are all temptations created by a deadline rather than by a design error. Their discriminating checks are timestamps and artifact counts, which is why the manifest schema records creation instants and not just identities.

This table is a design diagnostic, not an incident record. No row reports a defect found in a running application, because no application run was supplied. The containment boundary is deliberately narrow: preserve the rejected arrival with its reasons, preserve the committed state, keep degraded and stale status visible, and require a compatible bundle before any promotion.

**A failure the table does not cover.** The table assumes the fixture is right. A fixture encoded with the wrong byte order produces a trace that is internally consistent, fully dispositioned, manifest-bound, and wrong — and every check in this document passes. The only control is an encoding revision that is compared across two independent encoders, which is why A-06 exists and why it is listed as OPEN rather than assumed.

## 13. Transferable Lesson

A credible capstone is a falsifiable decision record. It names a work unit, a state identity, an ownership boundary, a complete denominator, an oracle, and a claim tier. The pattern transfers well beyond serial framing, and it transfers by substitution rather than by analogy: replace the identity, replace the failure consequence, and the structure holds.

| Domain substitution | Work unit | Identity that makes retries safe | Failure the gate must disclose |
|---|---|---|---|
| Payment or order intake | a transaction request | idempotency key plus a monotonic ledger position | a duplicate charge or a double shipment |
| Batch model scoring | a scored cohort row | `(cohort_id, entity_id, feature_cutoff)` | a row scored on post-outcome features |
| An agent workflow | a tool invocation | `(session_id, turn, tool_call_id)` | a repeated side effect such as a sent message |
| Dashboarding | a refreshed metric | `(metric_id, window_end, revision)` | a stale tile presented as current |
| Robotics or control | a command | `(controller_epoch, command_seq)` | a command applied twice or applied out of order |

The column that matters is the second-to-last, because it is where teams substitute hope for design. A payment system without an idempotency key is the same defect as a frame stream without `(epoch, sequence)`: a retry becomes an unexplained second effect. The transfer is not that serial rules apply to payments; it is that every retried unit needs an identity that the commit boundary checks atomically.

Two counterexamples sharpen the lesson. The first is a dashboard with fast local inference, an attractive visualisation, and a green unit-test suite but no source timestamp, no queue policy, and no render evidence. It may be technically interesting and it cannot truthfully claim freshness, because freshness is a property of a timestamp relation that the artifact never records. The second is a sourced-looking explanation whose corpus revision is unknown; the citations then decorate the sentence rather than support it, and the decoration is worse than no citation because it invites trust.

A third counterexample is subtler and worth including because it fools experienced reviewers. A system with excellent telemetry — dashboards, traces, per-stage timers — but no frozen oracle will produce beautiful evidence of whatever it does, including its bugs. Observability answers "what happened" and the oracle answers "what should have happened"; only the second can detect a wrong answer, and only the first can explain it. A capstone that ships one without the other has half a validation story.

The lesson's limit is equally important. Transfer does not mean importing every rule. A payment system needs durable exactly-once semantics because money moves; a batch model needs a data cut-off and cohort provenance; an agent needs tool authorisation and output review. What travels is the shape of the argument — identity, atomic boundary, complete denominator, oracle, tier — and what does not travel is the specific invariant, which must be re-derived against the consequence that the artifact actually risks.

### Deriving the invariant for a new domain

The five-step derivation below is the transferable artefact; the serial case is one instantiation of it.

1. **Name the consequential change.** What does the system do that cannot be taken back or cheaply undone? For this case it is committing a state version and showing it as current. For a payment it is moving money. For an agent it is emitting an external side effect.
2. **Find the identity that makes a repeat safe.** The identity must be knowable before the change and stable across retries. `(stream_id, epoch, sequence)` here; idempotency key plus ledger position for payments; `(session_id, turn, tool_call_id)` for an agent.
3. **Place the check at the mutation, not before it.** Any gap between check and mutation is a race whose width grows with load, so it fails exactly when it matters.
4. **Define the denial states and make them visible.** A gap, a corrupt arrival, an overflow, and a stale view are four different disclosures and the UI must be able to show them together rather than collapsing them into one failure badge.
5. **Freeze an oracle before executing.** The expectation must predate the run and must be authored by someone who did not write the implementation, or the comparison measures consistency.

Applying the same five steps to a batch scoring pipeline produces a different invariant from the same shape: the identity is `(cohort, entity, feature_cutoff)`, the consequential change is publishing a score into a decision surface, the denial states are "cut-off too recent", "feature unavailable", and "row superseded", and the oracle is a frozen set of rows with hand-computed scores. The steps transfer; the words do not.

### Two failure patterns that only appear after transfer

**Pattern one: identity that is unique but not stable.** A UUID generated per attempt is unique and therefore looks like an identity, but it is regenerated on retry, so the second attempt carries a different UUID and the duplicate check never fires. This defect survives every review that checks for "an identity field" and fails the first retry test. The discriminating check is to send the same logical event twice with the producer's normal retry path and confirm the identity is byte-identical.

**Pattern two: visibility that is present but not persistent.** A toast notification that says "degraded" for three seconds satisfies a literal reading of "the state must be visible" while failing the intent, because the reviewer who opens the page thirty seconds later sees a clean CURRENT view. The discriminating check is a screenshot taken after the notice expires; if the defect is no longer visible, the disclosure was an event rather than a state.

Both patterns share a cause: the requirement was written as a noun ("an identity", "a visible state") rather than as a property ("identity is stable across retries", "the defect remains visible until resolved"). Writing the requirement as a property is a low-cost discipline that prevents both.

### The cost side of the gate

Evidence discipline is not free, and a fair transfer argument states the price.

| Activity | Effort in this case | Avoided cost | When it is not worth it |
|---|---|---|---|
| Freeze fixture and oracle | hours | a rerun that cannot be reproduced | when no claim about behaviour will be made |
| Emit a manifest | hours | a bundle that cannot be audited | when the artifact is never shown to anyone |
| Independent oracle author | one extra reviewer | a comparison that measures consistency | when the system is a throwaway spike |
| HIL protocol | days | a hardware claim that collapses under questioning | when the portfolio claim is software-only by design |
| Claim audit | an afternoon per release | retractions after publication | never; this is the cheapest row |

The table makes the gate's economics legible: three of the five rows cost hours and one costs an afternoon, so the expensive tier is HIL, and HIL is also the tier most often claimed without evidence. The correct conclusion for a time-constrained capstone is to close the cheap rows, skip HIL deliberately, and publish exactly the tiers the cheap rows support.

## 14. Connections & Cross-References

Related method: [Capstone Evidence Rubric](../Cheat_Sheets/01_Capstone_Evidence_Rubric.md), especially its evidence tiers and its treatment of recovery invariants. Related planning boundary: [edge-capstone brief](../../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#edge-capstone). Related unit index: [Module 56 index](../INDEX.md), which records this case as proposed. Neither the rubric nor the brief supplies a run artifact for this case; both are planning context.

Related sibling case: choosing among six cross-domain capstones (planned in `INDEX.md`; not yet authored), which applies the same tier language to a portfolio-selection decision rather than to a single release gate. The connection is one-directional: this case supplies the gate mechanics, and the selection case supplies the criteria for deciding which capstone deserves the gate's cost.

### Interview pack

**1. Question — What exactly is the decision, and who owns it?**
- **Checked answer:** Whether evidence supports TRACE-CHECKED wording or only DESIGN wording for one release candidate, owned by the capstone technical lead and challenged by an independent reviewer.
- **Reasoning:** The decision is about published language, so its owner must be the person accountable for the sentence, and the reviewer must be able to refuse without escalation.
- **Follow-up:** If the lead and the reviewer disagree, what artifact settles it?
- **Red flags:** Answering "build the demo" or "improve accuracy", which are tasks rather than decisions.
- **Scoring:** Full credit for naming both roles and the wording object.

**2. Question — Compute the collection time for 64 samples at 400 samples/s.**
- **Checked answer:** 64 / 400 = 0.16 s = 160 ms.
- **Reasoning:** Samples divided by samples per second leaves seconds; the result assumes contiguous sampling with no overlap.
- **Follow-up:** At what frame size does collection alone exceed a 250 ms budget at the same rate?
- **Red flags:** Reporting 6.25 (the reciprocal) or 64 ms, both of which drop the unit structure.
- **Scoring:** Require the unit cancellation and the contiguity assumption.

**3. Question — Compute the 8N1 wire time for a 20-byte header, 128-byte payload, and 4-byte CRC at 115200 bits/s.**
- **Checked answer:** 152 bytes, 1520 bits, 1520 / 115200 = 0.01319444 s = 13.1944 ms.
- **Reasoning:** 8N1 carries ten bits per byte, so bytes must become bits before dividing by a bit rate.
- **Follow-up:** What does 12% byte escaping do to this figure?
- **Red flags:** 152/115200 ms, which omits framing and mixes units.
- **Scoring:** Require the ten-bit factor and the final unit conversion.
- *(Escaping check: 152 × 1.12 = 170.24 bytes, giving 14.7778 ms.)*

**4. Question — Assemble the age for a 64-sample frame at 400 samples/s, 40 frames/s service, an empty queue, and 18 ms render.**
- **Checked answer:** 160 + 13.1944 + 25 + 0 + 18 = 216.1944 ms.
- **Reasoning:** Service time is the reciprocal of the service rate, 1000/40 = 25 ms, and the path is assumed strictly additive.
- **Follow-up:** Which single term would you attack first to gain budget, and why not the render term?
- **Red flags:** Using 40 ms as the service time, or summing rates instead of times.
- **Scoring:** Require the reciprocal step to be shown.

**5. Question — Two frames are ahead in a FIFO served at 40 frames/s. What age results?**
- **Checked answer:** 2/40 × 1000 = 50 ms of queueing, so 216.1944 + 50 = 266.1944 ms, which exceeds the 250 ms budget by 16.19 ms.
- **Reasoning:** `K/mu` is a wait only if every frame ahead takes exactly `1/mu`; with variable service the count is bounded but the time is not.
- **Follow-up:** What is the maximum wait rather than the mean, given three service times of 20, 60, and 40 ms?
- **Red flags:** Treating the two-slot capacity as a two-slot latency guarantee.
- **Scoring:** Credit only with the constraint restated.
- *(Max-wait check: with three frames ahead served in 20, 60, and 40 ms, the worst-case wait is 20 + 60 + 40 = 120 ms, whereas `K/mu` with `mu = 40` predicts 3 × 25 = 75 ms and the mean-based estimate understates the worst case by 45 ms.)*

**6. Question — A frame arrives with a matching CRC but sequence 120 while 118 is committed. What happens?**
- **Checked answer:** It is a gap: quarantine it, keep the committed sequence at 118, mark the UI DEGRADED, and apply the declared resync policy for the missing 119.
- **Reasoning:** Integrity is not contiguity; a correct checksum says nothing about what was lost before this frame.
- **Follow-up:** Under what condition may the committed sequence legitimately jump to 120?
- **Red flags:** Advancing to 120 because the checksum passed, or silently waiting for a timeout to legitimise the jump.
- **Scoring:** Reject any answer that advances the sequence.
- *(Legitimate jump: only with an explicit epoch reset record.)*

**7. Question — Why may a duplicate be detected only at the commit boundary?**
- **Checked answer:** Because identity comparison anywhere upstream leaves a window in which two retries can both pass and both commit; the check must be atomic with the state write.
- **Reasoning:** A check that is not atomic with the mutation is a race that widens with load, so it fails exactly when the system is busy.
- **Follow-up:** What does the T2 fixture look like if the check is placed in the queue instead?
- **Red flags:** "Compare payload hashes before enqueueing", which both races and erases distinct equal-valued events.
- **Scoring:** Require the atomicity argument, not just the rule.

**8. Question — A frame declares 128 payload bytes and supplies 127, with a CRC mismatch. How many reasons should the record hold?**
- **Checked answer:** Two, length and checksum, because a length error and a corruption error have different probable causes and merging them destroys the diagnostic.
- **Reasoning:** Truncation usually indicates framing or buffering; a checksum error at correct length usually indicates transit corruption.
- **Follow-up:** Which of the two reasons would you keep if the record had room for one, and what would you lose?
- **Red flags:** Reporting only "invalid frame", or repairing the length by padding.
- **Scoring:** Require both reasons plus a cause distinction.

**9. Question — A valid frame arrives while the FIFO holds two unserved frames. What is the correct disposition, and why not block?**
- **Checked answer:** Drop with an accounting record; blocking would convert a bounded queue into unbounded backpressure and stall the acquisition path.
- **Reasoning:** Capacity is an integer and the policy must be deterministic at the boundary, so the outcome cannot depend on a judgement made at arrival time.
- **Follow-up:** What must the record contain to keep the denominator honest?
- **Red flags:** "Grow the queue", which moves the boundary without deciding it.
- **Scoring:** Require correlation ID plus capacity plus depth.
- *(Record fields: correlation ID, capacity, depth at arrival, policy revision, timestamp.)*

**10. Question — With 48 frames/s offered and 40 served for 90 s, what is the backlog and how long to drain at 64 frames/s?**
- **Checked answer:** `rho = 1.2`; backlog `= 8 × 90 = 720` frames; drain `= 720 / 16 = 45` s.
- **Reasoning:** Drain uses spare capacity, not total capacity, so the denominator is the difference between recovery and arrival rate.
- **Follow-up:** What happens to the drain time if the recovery rate is exactly 48?
- **Red flags:** 720/64 = 11.25 s, which answers the no-new-arrivals question instead.
- **Scoring:** Require the spare-capacity denominator.
- *(Boundary: at exactly 48 the drain time is undefined under the fluid model.)*

**11. Question — The last commit was at source time 900 ms and the render check is at 2400 ms. What is the state?**
- **Checked answer:** Age 1500 ms, therefore STALE under a 1000 ms threshold, with the explanation withheld.
- **Reasoning:** The comparison is against the committed source instant, not against the arrival instant of any candidate that failed.
- **Follow-up:** What must be true about the two clocks for that 1500 ms to mean anything?
- **Red flags:** Subtracting a receipt time from a render time, or calling it 1500 ms of "processing".
- **Scoring:** Require the subtraction, the threshold comparison, and the clock caveat.

**12. Question — Distinguish the five UI states by the fact each one asserts.**
- **Checked answer:** CURRENT asserts recent committed data; DEGRADED asserts a known continuity or capacity defect; STALE asserts age over the threshold; INVALID_INPUT asserts the newest arrival failed validation; ABSTAIN asserts the explanation lacked a mapped source.
- **Reasoning:** Freshness and validity apply to different objects, so a corrupt arrival and a stale state can coexist and must be shown together.
- **Follow-up:** Which two can never both be true at the same instant, and why?
- **Red flags:** A single "error" badge covering all non-current states.
- **Scoring:** Require the freshness-versus-validity distinction.
- *(Never simultaneous: CURRENT with any of STALE, INVALID_INPUT as the current-state label.)*

**13. Question — What separates TRACE-CHECKED from HIL-VALIDATED, concretely?**
- **Checked answer:** TRACE-CHECKED needs a frozen fixture, frozen oracle, complete manifest, and inspectable dispositions; HIL additionally needs a named board and firmware, a wiring and serial-parameter record, an instrumented physical capture, and repetitions.
- **Reasoning:** A desktop replay exercises logic; only a physical exercise can reveal driver buffering, clock recovery, and achieved-baud differences.
- **Follow-up:** Name one property a desktop replay can establish that a HIL run does not.
- **Red flags:** Calling a hand-built table HIL, or calling a desktop replay a hardware test.
- **Scoring:** Require at least two HIL-only artifacts.
- *(Replay-only property: exact byte-for-byte reproducibility of the input.)*

**14. Question — The trace is 6 of 6 on dispositions. Does that earn TRACE-CHECKED?**
- **Checked answer:** No. Completeness of a designed oracle is a property of the document; the tier requires an execution against a frozen oracle with an emitted manifest.
- **Reasoning:** A perfect table of expectations is still a table of expectations until some code is run and compared to it.
- **Follow-up:** What would a 5-of-6 completeness figure usually indicate, and what record would you look for?
- **Red flags:** Equating completeness with execution, or citing the arithmetic as measurement.
- **Scoring:** Require the document-versus-artifact distinction.
- *(5 of 6: a silently dropped frame; look for a missing overflow record.)*

**15. Question — What single change most cheaply converts this DESIGN artifact into a TRACE-CHECKED one?**
- **Checked answer:** Freeze the six-arrival fixture bytes and the oracle, emit the manifest, then run one comparator pass — no hardware and no UI redesign required.
- **Reasoning:** The expensive parts of the artifact already exist as contract; what is missing is the frozen input, the frozen expectation, and the execution record.
- **Follow-up:** What must be true of the oracle's timestamp relative to the run's, and why?
- **Red flags:** "Record a video", which produces a presentation artifact rather than an evidence artifact.
- **Scoring:** Require the fixture, the oracle freeze, the manifest, and the ordering constraint.
- *(Ordering: the oracle revision must precede the run.)*

### Reviewer evidence drill-down

The following inspection cards are proposed review criteria. They make a future decision repeatable, and none of them reports a completed review.

**R-01 — Canonical frame bytes.** Can a reviewer reconstruct one exact byte sequence? Inspect the encoding revision, byte order, header length, and payload hash. Pass when the CRC-covered range is unambiguously delimited; fail when two implementations can serialise the same fields differently. Effect: block TRACE-CHECKED promotion. Owner: protocol.

**R-02 — CRC coverage.** Does the manifest state exactly which bytes the CRC covers, plus the polynomial? Pass when a reviewer can recompute the fixture's expected value unaided. Fail when "CRC valid" appears without an algorithm or a range, which makes the assertion unfalsifiable. Effect: keep the integrity claim at DESIGN. Owner: protocol.

**R-03 — Length handling.** Is `payload_length` checked against the actual byte count before any interpretation? Pass when T4 is rejected for both its declared-length and checksum conditions. Fail when the parser reads past the candidate or silently pads it — a behaviour that turns a corruption into a plausible number. Effect: block execution until bounded parsing is demonstrated. Owner: service.

**R-04 — Schema discipline.** Does an unsupported `schema_version` produce a reason and no state change? Fail when unknown fields are guessed or ignored without a contract, because that makes the validator's verdict depend on which parser version happens to run. Effect: do not describe the validator as version-safe. Owner: protocol.

**R-05 — Unit declaration.** Is `sample_unit` explicit even though payload values are opaque? Pass when a unit mismatch causes a controlled rejection or abstention. Fail when display labels infer a unit from field position, which is how an arbitrary number acquires apparent meaning. Effect: prohibit semantic explanation wording. Owner: data contract.

**R-06 — Rate provenance.** Is the sample rate a frame property rather than an undocumented default? Pass when B-01's inputs are recoverable from the manifest and fixture alone. Fail when an age calculation uses a constant that appears nowhere in the bundle. Effect: prohibit timing language. Owner: acquisition.

**R-07 — Epoch reset.** Is a sender restart distinguishable from a sequence regression? Pass when a new epoch follows an explicit resync rule and the old epoch's high-water mark is retained. Fail when sequence zero silently overwrites a later sequence in the previous epoch. Effect: block the duplicate-safety assertion. Owner: service.

**R-08 — Duplicate boundary.** Does duplicate detection occur inside the commit transaction? Pass when T2 cannot produce `state_version` 43 after T1 produced 42. Fail when a retry is enqueued before the identity comparison, because the queue then reorders two attempts at the same logical event. Effect: block the idempotency claim. Owner: service.

**R-09 — Equal payloads.** Can two distinct frames legitimately carry identical payload bytes? Pass when both are retained as separate events under contiguity. Fail when payload-hash deduplication substitutes for identity, which erases stationary signals. Effect: correct the duplicate method before any evaluation. Owner: service.

**R-10 — Gap policy.** What happens between gap detection and resync? Pass when the committed sequence remains 118 and the UI shows DEGRADED with the missing index named. Fail when 120 becomes current without evidence for 119. Effect: block the continuity claim. Owner: protocol and UI.

**R-11 — Queue bound.** Is capacity an integer recorded for this candidate, with an overflow rule? Pass when the manifest shows capacity two and the drop policy revision. Fail when the word "bounded" appears with no number. Effect: block any workload interpretation. Owner: service.

**R-12 — Overflow evidence.** Does a full queue retain a record for every rejected arrival? Pass when the arrival count equals dispositions plus explicit exclusions. Fail when arrivals are fewer than fixtures with no accounting, which is the exact signature of silent loss. Effect: block the trace-completeness claim. Owner: service.

**R-13 — Clock semantics.** Are source, receipt, and render instants separately recorded with a stated relationship? Pass when a reviewer can recompute the age from the trace and reproduce 1500 ms for T6. Fail when only a render timestamp exists, because then age is unknowable rather than small. Effect: prohibit freshness wording. Owner: UI and acquisition.

**R-14 — Retrieval mapping.** Does every displayed factual sentence carry a chunk identity with document hash, chunker revision, index revision, and rank? Pass when an unmapped sentence is withheld by construction. Fail when citations are added by a post-processing step that can disagree with the generator. Effect: prohibit explanation-quality wording. Owner: retrieval.

**R-15 — Oracle independence.** Was the oracle authored by someone who did not write the implementation, and frozen before execution? Pass when the oracle hash predates the run record. Fail when the two share an author and a timestamp. Effect: the run is evidence of consistency, not correctness. Owner: reviewer.

### Claim-audit probes

Each probe is a sentence a reader might take away from the artifact, paired with the check that decides whether the artifact supports it.

| Reader takeaway | Probe | Artifact answer today |
|---|---|---|
| "The system is real-time." | Which path was timed, over how many runs, at what percentile? | none; no path was timed |
| "The UI shows freshness accurately." | Which recorded transition shows a state going from CURRENT to STALE? | none; the states are specified, not rendered |
| "The explanation is grounded." | Which displayed sentence maps to which chunk hash? | none; no retrieval run exists |
| "The pipeline is robust to overload." | What did the overflow record contain, and how many were written? | the policy is specified; zero records exist |
| "Duplicates are handled." | How many duplicate arrivals produced a second state version? | the rule is stated; no execution exists |
| "The hardware path works." | Which board, firmware hash, wiring record, and capture? | none |
| "The numbers are measured." | Which instrument produced 216.1944 ms? | none; it is a model output |
| "The budget is met." | At what frame size and queue depth? | met at 64 samples and depth zero only |

The fifth probe is the one most often skipped and most often decisive. "Duplicates are handled" is a claim about a mechanism, and a mechanism can be described correctly and implemented incorrectly in the same document. Only the count of second state versions separates the two, and that count requires a run.

A final probe applies to the artifact rather than to any claim: is there a sentence in this document that a competent reviewer could not check? If yes, that sentence is doing decorative work and should be deleted rather than defended. Applying this test to the present draft removes nothing, because every quantitative statement names its formula or its fixture row — which is a property worth preserving in the next revision.

## 15. Quick Reference Card

| Field | Case answer |
|---|---|
| Decision | Release tier for a local signal-to-explanation demonstrator. |
| Owner | Capstone technical lead, challenged by an independent reviewer. |
| Grain and cut-off | One manifest identity plus one fixture set; freeze instant `T_cut`. |
| Input and output | One framed arrival → disposition, state version, UI state, permitted wording. |
| Baseline and defer option | Screenshot-only demo; or publish nothing this cycle. |
| Identity | `(stream_id, epoch, sequence)` checked atomically at commit. |
| Integrity | Bounded length, supported schema and units, declared CRC coverage, ordering rule. |
| Queue | FIFO of two frames; overflow must be recorded, never hidden. |
| UI states | CURRENT, DEGRADED, STALE, INVALID_INPUT, ABSTAIN — all five distinct. |
| Evidence tier | DESIGN only; TRACE-CHECKED and HIL-VALIDATED unearned. |
| Checked figures | 64/400 = 160 ms; 1520/115200 = 13.1944 ms; 216.1944 ms empty-queue; 266.1944 ms at depth two. |
| Boundaries | 100 samples at 400/s is the largest frame fitting 250 ms; `rho = 1.2` under 48/s offered. |
| Gate | Executed fixture, frozen oracle, complete manifest, complete denominator, reviewer access. |
| Limits | No diagnostic, clinical, device, app-run, visual-UI, HIL, or deployment claim. |
| Review trigger | Any change to protocol, timing, hardware, runtime, retrieval, UI, manifest, or evidence. |

**Final status:** Draft DESIGN-only case. The original capstone and backlog commitments remain proposed and pending; this document does not complete or substitute for them.
