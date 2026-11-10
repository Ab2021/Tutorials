---
case_id: "CASE-01-002"
case_title: "Evaluate an AI Vendor Claim"
case_version: "2.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "analytical"
domain: "AI procurement and evidence evaluation"
primary_pattern: "evidence-to-operation"
secondary_patterns: ["benchmark-governance", "human-approval"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "procurement lead"
source_index: "01_program_orientation_and_ai_landscape/Case_Studies/INDEX.md"
review_trigger: "Vendor version, price, retention terms, benchmark protocol, or workload mix changes"
---

# Evaluate an AI Vendor Claim

**Evidence record:** This is an ILLUSTRATIVE procurement teaching case. Values, documents, benchmark outputs, pricing, and architecture are constructed examples and are not claims about a vendor or deployment. **Implementation stage:** proposed pilot. **Editorial status:** draft. Primary-source verification, security review, procurement approval, and publication review remain OPEN.

## 1. The One-Line Framing
Given a versioned vendor capability claim and a representative but non-production evaluation pack, decide whether the procurement lead should reject, request evidence, accept a bounded pilot, or retain manual review. The decision compares a documented manual evidence-review baseline with a vendor claim under fixed reviewer capacity, total-cost, version-drift, and reversibility constraints.

The input is one claim package: claim text, vendor model/version identifier, supported task definition, evaluation protocol, price schedule, data-handling terms, and known exclusions. The output is a signed decision record, not an automated purchase order. The beneficiary is the procurement lead and accountable business sponsor; the binding constraint is that a benchmark score cannot substitute for contractual, security, or operational acceptance.

**Decision rule preview:** accept a pilot only when the claim is precisely scoped, the evaluation pack is workload-fair, the confidence interval is decision-useful, reviewer capacity and full cost fit the scenario budget, and the version/rollback contract is accepted. Otherwise defer for evidence or reject.

## 2. Context & Background
Procurement receives a claim that a hosted model will classify incoming support documents more accurately and cheaply than the current manual triage process.
The business sponsor wants faster routing, but the procurement lead owns the vendor-selection recommendation rather than the final operational policy.
The current baseline is manual review: trained reviewers inspect every document, assign one of three routing labels, and record a reason code.
The baseline has a measurable workflow even though this case does not claim observed baseline data: documents arrive, are sampled for quality, are reviewed, and are routed or escalated.
The proposed vendor path inserts a model recommendation before the same reviewer; no document is automatically rejected or actioned from model confidence alone.
The procurement question is narrower than “is the vendor good?”: it is whether one named version under one workload definition justifies a reversible pilot.
A vendor marketing benchmark can be useful evidence of a hypothesis, but it is not evidence that the buyer workload, labels, languages, attachment formats, or error costs match.
The pilot is designed as a decision instrument: it produces acceptance evidence or an explicit reason not to proceed, rather than a demonstration chosen to look impressive.
All references to reviewers, workloads, cost, and latency in this document are illustrative teaching assumptions; they are not market prices or performance claims.
No compliance determination is made here. Retention, access, residency, and contractual terms require appropriate authority and current primary documents.

| Workflow stage | Baseline owner | Proposed change | Evidence needed before promotion | Safe fallback |
|---|---|---|---|---|
| intake | operations analyst | capture claim and version | complete claim packet | hold request |
| evaluation design | procurement lead | freeze representative sample | sampling and label protocol | manual comparison |
| offline run | evaluation analyst | obtain vendor outputs | run log and configuration hash | discard run |
| review | domain reviewers | blindly adjudicate disagreements | reviewer queue and rubric | manual route |
| commercial review | finance partner | estimate full cost | price/version/volume assumptions | retain baseline |
| decision | procurement lead | accept/defer/reject pilot | signed decision ledger | no purchase |

**Takeaway:** the setting is a governed decision workflow, not a contest to select the highest reported score.

## 3. The Business Question
| Field | Decision contract |
|---|---|
| Question | For vendor version V under the stated document workload, should the organization reject, defer, or authorize a limited reversible pilot? |
| Decision owner | Procurement lead, with finance, security, domain-review, and sponsor inputs; the lead records the recommendation. |
| Trigger and deadline | A vendor proposal reaches a purchase-gate date; no urgency overrides missing evidence. |
| Available actions | Reject; request evidence; accept a time- and volume-bounded pilot; retain manual review. |
| Value | Faster routing and less reviewer effort are hypotheses, not achieved benefits. |
| Downside | Incorrect routing, hidden total cost, data mishandling, vendor lock-in, reviewer overload, and irreproducible claims. |
| Primary objective | Select a path with adequate decision quality per reviewed document under stated cost and authority limits. |
| Hard constraints | Human final authority, reversible pilot, fixed evaluation protocol, explicit version, and no unsupported security/legal claim. |
| Population | Illustrative incoming documents in the declared language, format, and task slice. |
| Out of scope | Production launch, legal/compliance approval, and a claim that the vendor is generally superior. |

The proxy “benchmark accuracy” can conflict with the objective. A system with a high aggregate score may fail on the costly label, abstain too often for available reviewers, or shift costs to retries and exceptions. The acceptance decision therefore uses a claim ledger and a capacity-aware review path.

## 4. Assumptions & Constraints
| ID | Assumption or constraint | Type | Verification | Status | Consequence if wrong | Fallback | Owner |
|---|---|---|---|---|---|---|---|
| A-01 | Claim names a fixed vendor version and endpoint | evidence | capture version and release date | unverified | benchmark cannot be repeated | defer | procurement |
| A-02 | Evaluation documents represent the intended pilot slice | sampling | compare strata to intake definition | unverified | score is not decision-relevant | resample | evaluation lead |
| A-03 | Labels are adjudicated without seeing system identity | measurement | blind review log | proposed | comparison bias | retain manual baseline | review lead |
| A-04 | Inputs were available before the recommendation cutoff | time | audit field timestamps | proposed | leakage | exclude record | data owner |
| A-05 | Review capacity is bounded and scheduled | operations | staffing plan | unverified | abstentions queue indefinitely | cap pilot | operations |
| A-06 | Vendor cost schedule covers input, output, retries and support | commercial | dated quote and assumptions | unverified | understated total cost | defer commercial gate | finance |
| A-07 | Data use and retention terms are acceptable | governance | current primary terms and review | OPEN | pilot is not authorized | do not transmit data | appropriate authority |
| A-08 | Version changes are detectable before requests | configuration | preflight version assertion | proposed | result cannot be attributed | block run | engineering |
| C-01 | No automatic business action is permitted | authority | policy statement | required | unsafe automation | manual final action | sponsor |
| C-02 | Pilot volume is capped at 1,000 documents | scope | ledger counter | proposed | unbounded exposure | stop intake | procurement |
| C-03 | Any material model or price change reopens the decision | change control | compare snapshot hashes | required | stale approval | re-evaluate | procurement |

The assumptions most likely to reverse a pilot decision are workload representativeness, reviewer capacity, and a changed vendor version. Each is a decision gate, not a footnote.

## 5. Data, Tools & Method
### 5a. Data Specification
| Field | Specification |
|---|---|
| Unit of analysis | One document-decision pair, keyed by `document_id` and `evaluation_run_id`; repeated documents retain separate attempts but share a canonical content hash. |
| Population | Illustrative English-language support documents in the agreed task taxonomy; attachments, OCR failures, and unsupported languages are excluded and counted. |
| Decision-time inputs | Document text or permitted representation, metadata available at intake, task rubric version, vendor version, prompt/configuration version, and timestamp. |
| Reference label | Two blinded reviewers assign a routing label and confidence; disagreement enters adjudication. This is a proposed protocol, not completed labels. |
| Vendor output | Label, abstention/error state, response timestamp, request/response identifiers, claimed version, token or byte counts where supplied, and raw response retention status. |
| Join semantics | One evaluation run joins one frozen input snapshot to one configuration snapshot; duplicate content is not silently deduplicated across strata. |
| Splits | Design set for rubric rehearsal; frozen acceptance set for the decision; a changed version requires a new frozen comparison set. |
| Exclusions | Missing consent/authority, malformed records, label ambiguity beyond rubric, unavailable vendor output, and post-cutoff fields are excluded with reason codes. |

### 5b. Tools & Stack
- A versioned evaluation repository stores the protocol, sample manifest, rubric, and decision ledger.
- A content-hash service detects duplicate documents while retaining the sampling stratum.
- A request broker applies a rate cap, records the asserted model version, and prevents retries from becoming duplicate accounting entries.
- A blinded-review console hides vendor identity and randomizes display order for adjudicators.
- A calculation notebook or independent script recomputes metrics from the frozen run export; its output is an evidence artifact, not a production service.
- A finance worksheet exposes each cost input, currency, period, tax/support exclusions, and sensitivity.
- A pilot kill switch prevents new submissions while preserving the immutable ledger.

### 5c. Method Summary
1. Translate the vendor statement into a testable claim with task, slice, metric, comparator, threshold, version, and validity period.
1. Freeze a workload-fair sample before seeing vendor outputs; stratify by task category, language/format, and known hard cases rather than convenience.
1. Run the baseline and vendor recommendation against identical frozen records; preserve abstentions and errors as outcomes rather than deleting them.
1. Blindly adjudicate the reference label and compare outcomes by stratum and cost-of-error category.
1. Estimate reviewer demand from disagreement, abstention, and audit rates; do not call an unstaffed escalation path a capability gain.
1. Estimate total cost from fixed fees, variable requests, input/output usage, retries, reviewer minutes, integration, monitoring, and contingency.
1. Apply reject/defer/accept-pilot gates; no aggregate metric overrides a failed hard constraint.

| Candidate | Evidence status | Strength | Limitation | Decision status | Revisit when |
|---|---|---|---|---|
| Manual review | baseline proposed for measurement | transparent, reversible | slow and variable | retain as comparator | review capacity changes |
| Vendor recommendation + human review | illustrative protocol | may reduce routine effort | requires representative evidence | candidate pilot | all gates pass |
| Vendor auto-action | not tested | low apparent latency | violates authority constraint | rejected | only if policy changes |
| Rules-only triage | not tested | auditable | may not cover language variance | considered | rules have sufficient coverage |
| Defer-all | available fallback | limits unsafe action | adds delay | fallback | evidence unavailable |

### 5d. Data provenance, quality, exclusions and leakage controls
- Freeze the sample manifest before obtaining outputs; record who selected each stratum and why.
- Use a content hash to identify near duplicates across design and acceptance sets; duplicates may inflate apparent agreement.
- Record the earliest availability time for every metadata field; remove fields created by later human resolution.
- Do not let vendor support staff relabel the acceptance set or choose items after observing failures.
- Preserve timeouts, refusals, malformed responses, and unsupported formats as explicit categories.
- Keep evaluator prompts/configuration separate from label adjudication; an explanation generated after the label cannot become an input feature.
- Record redactions and transformations because they can change the task semantics and comparability.
- Limit evaluator access to the minimum permitted material; exact handling requirements remain OPEN pending authorized review.

### 5e. Reproducibility and method-selection ledger
| ID | Artifact / claim | Version or date | Purpose | Accessibility | Reproduction limitation |
|---|---|---|---|---|---|
| SRC-01 | Vendor claim packet | to be captured | scope assertion | not reviewed | no primary source attached |
| CFG-01 | Prompt and request configuration | hash required | repeat vendor call | proposed | endpoint behavior may change |
| DATA-01 | Frozen sample manifest | run date required | workload fairness | restricted | not supplied in this case |
| LAB-01 | Blinded adjudication rubric | version required | reference label | proposed | reviewer calibration unmeasured |
| RUN-01 | Vendor output export | run date required | metric calculation | proposed | no vendor run completed |
| FIN-01 | Total-cost model | dated quote required | commercial comparison | proposed | prices and usage uncertain |
| ORC-01 | Independent metric oracle | revision required | arithmetic check | proposed | only checks declared formula |

**Takeaway:** a claim becomes evaluable only when its workload, comparator, version, labels, and costs can be traced.

## 6. Analysis Walkthrough
The following arithmetic is ILLUSTRATIVE and independently checked. It is not a vendor benchmark or a forecast. The sample is deliberately small enough to show denominator discipline.

### Claim contract and workload fairness
| Stratum | Baseline documents | Acceptance documents | Error-cost reason | Sampling rule |
|---|---:|---:|---|---|
| routine routing | 240 | 120 | low-to-medium | sample from ordinary intake |
| ambiguous requests | 120 | 60 | high review cost | oversample then report separately |
| policy-sensitive | 80 | 40 | high false-positive cost | include all available in window |
| long documents | 60 | 30 | latency/cost risk | length bucket |
| format degraded | 40 | 20 | input robustness | include failures rather than clean them |
The illustrative acceptance set has 270 documents. It is not “fair” merely because it is random: the protocol requires reporting each stratum, inclusion rule, and exclusion reason. A benchmark with only routine routing documents could show a good aggregate score while avoiding the costly work.

### Metric definition
```text
FORMULA [V-01]: agreement = matching vendor recommendation and adjudicated reference label / evaluated documents
WHERE: matching count and evaluated denominator are document-decision pairs [count].
CONSTRAINT: report abstentions, errors, exclusions, and every stratum; agreement is not business benefit or causal impact.
NUMERICAL EXAMPLE: 216 matches / 270 evaluated = 0.800 = 80.0%.
INTERPRETATION: eight of ten illustrative documents match the proposed reference process; it says nothing yet about reviewer capacity or error severity.
```

Suppose the manual baseline matches adjudication on 207 of 270 documents: 207/270 = 76.666…%, reported as 76.7%. The illustrative difference is 9/270 = 3.333… percentage points, not “3.3% better” without naming the denominator.

### Uncertainty and changed-threshold sensitivity
For a proportion p=0.800 and n=270, a normal approximation standard error is sqrt(p(1-p)/n)=sqrt(0.16/270)=sqrt(0.00059259)=0.02434. This approximation is used only as a rough planning display; small, clustered, or unrepresentative strata invalidate its simple interpretation. A rough 95% interval is 0.800 ± 1.96×0.02434 = 0.800 ± 0.0477, or about [75.2%, 84.8%].
The interval overlaps the illustrative baseline agreement of 76.7%; it does not support a confident superiority conclusion. A paired comparison and an appropriate interval would be required for a real decision. The correct procurement action in this teaching scenario is not “accept because 80% exceeds 76.7%.”

| Stress test | Original condition | Alternative | Effect on decision |
|---|---|---|---|
| Workload mix | 270 documents, 15% degraded format | 30% degraded format | defer unless degraded slice meets its own gate |
| Acceptance threshold | aggregate agreement ≥80% | high-cost stratum ≥85% | aggregate pass can become fail |
| Reviewer sampling | 10% audit | 30% audit | reviewer capacity may bind |
| Vendor version | V1 fixed | silent V2 change | invalidate comparison; rerun frozen set |
| Price | illustrative $0.04/document variable | 50% increase | total-cost recommendation may reverse |
| Abstention | 5% | 20% | manual queue may erase claimed savings |

### Reviewer-capacity model
```text
FORMULA [V-02]: R = N × (a + d + q)
WHERE: R reviewer items [items/period]; N submitted documents [items/period]; a abstention rate, d disagreement rate, q audit rate [fractions].
CONSTRAINT: rates may overlap; this simple upper-bound planning sum assumes separate review triggers and must be deduplicated in a real ledger.
NUMERICAL EXAMPLE: N=1,000, a=0.05, d=0.12, q=0.10; R=1,000×0.27=270 review items.
INTERPRETATION: a claimed automation rate is operationally irrelevant if 270 items cannot be reviewed in the period.
```
At 8 reviewer minutes per item, 270×8=2,160 reviewer minutes =36 staffed hours. With 3 reviewers each providing 10 available evaluation hours, capacity is 30 hours; the illustrative plan has a 6-hour shortfall. The safe response is to cap N at 833 items because 833×0.27×8=1,799.28 minutes, just under 30 hours, subject to rounding and overlapping-trigger measurement.

### True total-cost model
```text
FORMULA [V-03]: C_total = C_fixed + N×C_variable + R×m×C_reviewer + C_integration + C_monitoring + C_contingency
WHERE: costs are currency/period; N [documents/period], R [review items/period], m [minutes/item], C_reviewer [currency/minute].
CONSTRAINT: all items must use the same currency and period; omitted terms are zero only if explicitly justified.
NUMERICAL EXAMPLE: $2,000 + 1,000×$0.04 + 270×8×$0.75 + $500 + $300 + $400 = $4,860 per illustrative pilot period.
INTERPRETATION: reviewer time ($1,620) dominates variable API use ($40) in this scenario; optimizing token price alone can mislead procurement.
```
Arithmetic check: 2,000+40+1,620+500+300+400=4,860. If disagreement rises from 12% to 25%, R becomes 400 and reviewer cost is 400×8×0.75=$2,400; total becomes $5,640, an increase of $780. If the vendor variable price doubles, total becomes $4,900, only $40 higher. These are decision sensitivities, not prices.

### Queue boundary
If documents arrive at λ=20/hour and the staffed review service rate is μ=15/hour for a 4-hour surge, fluid backlog B=(20−15)×4=20 documents. After the surge, if arrivals are 10/hour and service is 18/hour, drain time is 20/(18−10)=2.5 hours. At equal recovery arrival and service rates the backlog never drains under this model. This calculation uses hours consistently and is not a latency guarantee.

### What the first approach gets wrong
The first tempting approach is to accept the vendor claim if its headline benchmark exceeds a baseline headline. It fails because it has no fixed workload, no version lock, no cost-of-error slice, no reviewer-capacity calculation, and no total-cost denominator. It is an alternative considered and rejected in this illustrative design, not a report of a completed procurement event.

**Takeaway:** a procurement decision can reverse when the same score is put under a fair workload, interval, reviewer-capacity, and full-cost contract.

## 7. The Finding
The illustrative evidence supports neither a vendor selection nor a claim of superiority. A versioned vendor recommendation may be eligible for a bounded pilot only if a frozen workload-fair protocol shows acceptable stratum-level outcomes, the reviewer queue fits staffed capacity, full cost fits the decision budget, and version/term changes are blocked or re-evaluated. The 80.0% illustrative agreement and 76.7% baseline comparison are uncertain and overlap under a rough planning interval; they are not a production result. The current decision is **defer pending evidence**, with manual review retained as the safe baseline.

## 8. System/Solution Design
```text
[claim packet] -> [scope/version validator] -> [frozen sample manifest] -> [request broker]
                                                              |
[manual baseline] -> [blind review + adjudication] -> [metric and cost oracle] -> [decision ledger]
                                                              |
                                                          [procurement lead]
                                                              |
                                            accept pilot / defer evidence / reject
                                                              |
                                           [kill switch -> manual review]
```

- The claim packet service owns the definition of the claim and rejects missing version, metric, or workload scope.
- The sample manifest is immutable after the acceptance run begins; amendments create a new run identifier rather than changing the denominator.
- The request broker owns rate limits, retry budgets, and request identifiers; it does not decide procurement acceptance.
- The review system owns blinded labels and adjudication history; vendor identity is separated from the reviewer display where feasible.
- The metric oracle owns deterministic aggregation from frozen exports and records the exact exclusion list.
- The finance worksheet owns cost assumptions but does not infer a legal or contractual obligation.
- The decision ledger owns the final accept/defer/reject state, actor, timestamp, rationale, and version snapshot.
- The kill switch prevents new pilot submissions and routes work to manual review; it does not erase evidence.

State transition invariant: a decision record moves from `DRAFT` to `EVIDENCE_REQUESTED`, `REJECTED`, or `PILOT_ACCEPTED`; `PILOT_ACCEPTED` requires all hard-gate assertions. A changed model version, price schedule, data term, or workload definition moves the record to `REVIEW_REQUIRED`; no cached approval can authorize the change.

## 9. Evaluation & Validation
### V01 — Claim completeness
**Setup:** remove each mandatory claim field.
**Oracle:** schema validator.
**Expected result:** missing field yields defer.
**Boundary:** schema validity does not prove capability.

### V02 — Version pinning
**Setup:** respond with a different asserted version.
**Oracle:** version comparator.
**Expected result:** run blocked and ledger records mismatch.
**Boundary:** vendor may not expose a reliable version.

### V03 — Workload balance
**Setup:** shuffle stratum labels.
**Oracle:** sample-manifest checker.
**Expected result:** counts match frozen manifest.
**Boundary:** balanced count does not guarantee representativeness.

### V04 — Label blinding
**Setup:** hide vendor identity.
**Oracle:** review audit log.
**Expected result:** reviewer cannot see system source.
**Boundary:** blinding may be imperfect.

### V05 — Metric arithmetic
**Setup:** use 216/270 matches.
**Oracle:** independent calculator.
**Expected result:** 80.0%.
**Boundary:** arithmetic does not establish external validity.

### V06 — Baseline comparison
**Setup:** use 207/270 baseline.
**Oracle:** paired-result export.
**Expected result:** denominators identical.
**Boundary:** agreement is not business utility.

### V07 — Abstention handling
**Setup:** inject 14 abstentions.
**Oracle:** outcome classifier.
**Expected result:** abstentions remain counted.
**Boundary:** policy for abstention may change.

### V08 — Cost calculation
**Setup:** use illustrative input rows.
**Oracle:** spreadsheet/oracle.
**Expected result:** total $4,860.
**Boundary:** quote and usage remain unverified.

### V09 — Capacity
**Setup:** use R=270 at 8 minutes.
**Oracle:** staffing calculation.
**Expected result:** 36 hours required.
**Boundary:** actual review time unknown.

### V10 — Queue
**Setup:** λ=20/h, μ=15/h.
**Oracle:** fluid arithmetic.
**Expected result:** 20-document surge backlog.
**Boundary:** arrival/service not constant in reality.

### V11 — Retry
**Setup:** repeat request ID.
**Oracle:** ledger query.
**Expected result:** one accounted request.
**Boundary:** provider may charge retry separately.

### V12 — Failure fallback
**Setup:** disable request broker.
**Oracle:** workflow trace.
**Expected result:** manual review receives work.
**Boundary:** manual team may also be capacity-constrained.

### V13 — Changed version
**Setup:** replace V1 with V2.
**Oracle:** decision-state machine.
**Expected result:** REVIEW_REQUIRED.
**Boundary:** does not compare quality.

### V14 — Changed price
**Setup:** raise variable cost.
**Oracle:** cost sensitivity.
**Expected result:** recompute recommendation.
**Boundary:** cost may include unpriced terms.

### V15 — Changed mix
**Setup:** double degraded slice.
**Oracle:** stratum report.
**Expected result:** separate gate re-evaluated.
**Boundary:** new sample still needed.

Acceptance for a limited pilot is proposed, not achieved: all required fields exist; no version mismatch is tolerated; every acceptance-set item has a retained outcome; high-cost and degraded slices meet owner-approved thresholds; manual review capacity is scheduled; all cost inputs are dated; and the kill switch is demonstrated. A failed test produces defer or reject, not an exception approval.

## 10. Action Taken
No procurement action, pilot, vendor run, or production deployment was taken. The proposed action sequence is:
1. Request a versioned claim packet and dated commercial assumptions.
2. Have authorized reviewers determine whether permitted evaluation data and terms are available.
3. Write and approve the sampling and label protocol before outputs are requested.
4. Run a small offline, capped evaluation with the kill switch and manual fallback rehearsed.
5. Compute stratum metrics, uncertainty display, reviewer load, and full cost from retained artifacts.
6. Ask the procurement lead to record accept-pilot, defer, or reject with each gate result.
7. Expire the decision automatically if version, terms, price, or workload changes.

Rollback is not deletion: disable submissions, preserve the run ledger, return queued documents to manual review, notify the decision owner, and classify any partial outputs as unusable until the next approved run.

## 11. Outcome & Results
**Outcome status: not measured.** No vendor was evaluated and no savings, latency, quality, security posture, or operational adoption is asserted. The only verified values in this manuscript are arithmetic derived from explicit illustrative inputs: 216/270=80.0%; 207/270=76.7%; reviewer demand 270 items×8 minutes=36 hours; and illustrative total cost $4,860.

| Claim ID | Exact scoped claim | Evidence | Supports | Does not establish | Recheck trigger |
|---|---|---|---|---|---|
| CLM-01 | A frozen sample permits a reproducible denominator | derived design | evaluation discipline | representativeness | sample protocol changes |
| CLM-02 | Illustrative agreement is 80.0% for 216/270 | verified arithmetic | metric computation | vendor quality | any input changes |
| CLM-03 | Illustrative reviewer demand is 36 hours | verified arithmetic | capacity concern | actual staffing need | time study |
| CLM-04 | Illustrative total cost is $4,860 | verified arithmetic | full-cost method | market price or savings | quote or volume changes |
| CLM-05 | V2 invalidates V1 approval | design invariant | change governance | quality decline | version policy changes |

**Takeaway:** an honest result can be a deferred decision with a precise evidence plan.

## 12. Failure Modes & What Went Wrong
### F-01 — Headline benchmark accepted
**Symptom:** Headline benchmark accepted.
**Mechanism and competing causes:** aggregate score is treated as workload evidence.
**Diagnostic check:** compare task, slice, labels, exclusions, and version.
**Immediate recovery:** freeze a representative protocol.
**Verification:** rerun every slice; reject unsupported claim.
**Mitigation risk:** stricter gates can increase manual load; disclose the queue impact to the owner.

### F-02 — Easy documents dominate
**Symptom:** Easy documents dominate.
**Mechanism and competing causes:** convenience sampling.
**Diagnostic check:** inspect stratum shares versus intake definition.
**Immediate recovery:** stratify and report separately.
**Verification:** recompute gates by stratum.
**Mitigation risk:** stricter gates can increase manual load; disclose the queue impact to the owner.

### F-03 — Vendor sees test items
**Symptom:** Vendor sees test items.
**Mechanism and competing causes:** test contamination.
**Diagnostic check:** audit access and selection timestamps.
**Immediate recovery:** invalidate exposed set.
**Verification:** create a new hidden set.
**Mitigation risk:** stricter gates can increase manual load; disclose the queue impact to the owner.

### F-04 — Reference labels leak
**Symptom:** Reference labels leak.
**Mechanism and competing causes:** reviewer sees model output.
**Diagnostic check:** inspect blind-review logs.
**Immediate recovery:** relabel blindly.
**Verification:** compare adjudication drift.
**Mitigation risk:** stricter gates can increase manual load; disclose the queue impact to the owner.

### F-05 — Abstentions disappear
**Symptom:** Abstentions disappear.
**Mechanism and competing causes:** only answered records are counted.
**Diagnostic check:** reconcile submitted, returned, and scored counts.
**Immediate recovery:** count abstentions separately.
**Verification:** verify denominator ledger.
**Mitigation risk:** stricter gates can increase manual load; disclose the queue impact to the owner.

### F-06 — Version silently changes
**Symptom:** Version silently changes.
**Mechanism and competing causes:** endpoint alias moves.
**Diagnostic check:** compare assertion and response version.
**Immediate recovery:** block run.
**Verification:** re-run frozen comparison.
**Mitigation risk:** stricter gates can increase manual load; disclose the queue impact to the owner.

### F-07 — Price excludes retries
**Symptom:** Price excludes retries.
**Mechanism and competing causes:** quote scope is narrow.
**Diagnostic check:** trace all request and support charges.
**Immediate recovery:** add cost row.
**Verification:** finance review.
**Mitigation risk:** stricter gates can increase manual load; disclose the queue impact to the owner.

### F-08 — Reviewer queue is unstaffed
**Symptom:** Reviewer queue is unstaffed.
**Mechanism and competing causes:** automation rate mistaken for workload reduction.
**Diagnostic check:** calculate R and staffed hours.
**Immediate recovery:** cap pilot.
**Verification:** observe scheduled coverage.
**Mitigation risk:** stricter gates can increase manual load; disclose the queue impact to the owner.

### F-09 — High-cost label fails
**Symptom:** High-cost label fails.
**Mechanism and competing causes:** aggregate metric masks severity.
**Diagnostic check:** report confusion by cost tier.
**Immediate recovery:** set tier gate.
**Verification:** adjudicate error impact.
**Mitigation risk:** stricter gates can increase manual load; disclose the queue impact to the owner.

### F-10 — Latency claim hides tail
**Symptom:** Latency claim hides tail.
**Mechanism and competing causes:** mean reported alone.
**Diagnostic check:** inspect p50/p95/p99 plus timeout rate.
**Immediate recovery:** keep manual fallback.
**Verification:** load test permitted workload.
**Mitigation risk:** stricter gates can increase manual load; disclose the queue impact to the owner.

### F-11 — Data term is assumed
**Symptom:** Data term is assumed.
**Mechanism and competing causes:** marketing language treated as contract.
**Diagnostic check:** obtain current primary terms.
**Immediate recovery:** do not transmit data.
**Verification:** authorized review.
**Mitigation risk:** stricter gates can increase manual load; disclose the queue impact to the owner.

### F-12 — Prompt/config changes
**Symptom:** Prompt/config changes.
**Mechanism and competing causes:** configuration is not pinned.
**Diagnostic check:** compare hash and run ledger.
**Immediate recovery:** invalidate run.
**Verification:** change control.
**Mitigation risk:** stricter gates can increase manual load; disclose the queue impact to the owner.

### F-13 — Manual baseline is weakened
**Symptom:** Manual baseline is weakened.
**Mechanism and competing causes:** comparison process differs.
**Diagnostic check:** audit same input and rubric.
**Immediate recovery:** restore symmetric baseline.
**Verification:** paired replay.
**Mitigation risk:** stricter gates can increase manual load; disclose the queue impact to the owner.

### F-14 — Pilot expands by exception
**Symptom:** Pilot expands by exception.
**Mechanism and competing causes:** cap is bypassed.
**Diagnostic check:** check ledger count.
**Immediate recovery:** stop intake.
**Verification:** owner approval for new protocol.
**Mitigation risk:** stricter gates can increase manual load; disclose the queue impact to the owner.

### F-15 — Retry double-counts
**Symptom:** Retry double-counts.
**Mechanism and competing causes:** timeout triggers new request.
**Diagnostic check:** inspect idempotency keys and charges.
**Immediate recovery:** deduplicate accounting.
**Verification:** replay same key.
**Mitigation risk:** stricter gates can increase manual load; disclose the queue impact to the owner.

### F-16 — Cost period mixed
**Symptom:** Cost period mixed.
**Mechanism and competing causes:** monthly fee plus daily volume.
**Diagnostic check:** check units and currency date.
**Immediate recovery:** normalize period.
**Verification:** independent finance calculation.
**Mitigation risk:** stricter gates can increase manual load; disclose the queue impact to the owner.

### F-17 — Language slice omitted
**Symptom:** Language slice omitted.
**Mechanism and competing causes:** scope silently narrows.
**Diagnostic check:** compare intake languages with manifest.
**Immediate recovery:** defer unsupported language.
**Verification:** add explicit handoff.
**Mitigation risk:** stricter gates can increase manual load; disclose the queue impact to the owner.

### F-18 — Format failure is cleaned away
**Symptom:** Format failure is cleaned away.
**Mechanism and competing causes:** OCR failures excluded without count.
**Diagnostic check:** inspect exclusion reason table.
**Immediate recovery:** report failure slice.
**Verification:** test raw format path.
**Mitigation risk:** stricter gates can increase manual load; disclose the queue impact to the owner.

### F-19 — Interval overinterpreted
**Symptom:** Interval overinterpreted.
**Mechanism and competing causes:** rough normal interval called proof.
**Diagnostic check:** check n, dependence, sampling.
**Immediate recovery:** use appropriate analysis.
**Verification:** state uncertainty.
**Mitigation risk:** stricter gates can increase manual load; disclose the queue impact to the owner.

### F-20 — Human approval becomes rubber stamp
**Symptom:** Human approval becomes rubber stamp.
**Mechanism and competing causes:** reviewer sees no evidence.
**Diagnostic check:** audit rationale and dissent.
**Immediate recovery:** require decision packet.
**Verification:** sample decision quality.
**Mitigation risk:** stricter gates can increase manual load; disclose the queue impact to the owner.

**Takeaway:** every failure is a decision to contain, diagnose, correct, and re-check—not a reason to hide a denominator.

## 13. Transferable Lesson
- Translate a broad vendor promise into a claim with a named version, workload, metric, comparator, threshold, owner, and expiry.
- Treat representativeness as a design choice with documented strata, not a property inherited from a benchmark name.
- Carry abstentions, errors, exclusions, and retries into the denominator and the cost model.
- Separate metric uncertainty from operational capacity: a statistically attractive result can still be impossible to review.
- Model total cost as a system: fixed, variable, retry, reviewer, integration, monitoring, and contingency costs have different sensitivity.
- Require a version-change transition that invalidates prior approval unless equivalent evidence is established.
- Keep human authority explicit; a model recommendation, benchmark score, or contract clause does not create operational permission.
- Use a reversible pilot as an evidence-generating operation only after the gates are defined.

Counterexample: if a vendor’s V1 meets an illustrative threshold on routine documents but V2 changes tokenization or response policy, the V1 result cannot be carried forward merely because the product name is unchanged. The required action is `REVIEW_REQUIRED`, a fresh workload comparison, and a re-costing of reviewer demand.

### Evidence-gate walkthroughs
The following gate cards are distinct decision checks. They deliberately separate a missing fact from a failed constraint; neither may be converted into a favorable assumption.
#### Gate 01 — claim verb
**Question:** Can the team separate “can” from “will” and “is measured”?
**Required artifact:** claim text and cited artifact.
**Failing signal:** scope claim as unverified.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** request exact evidence.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only claim verb; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 02 — task taxonomy
**Question:** Can the team map each promised task to an internal routing label?
**Required artifact:** task mapping table.
**Failing signal:** unmapped task.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** exclude from pilot.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only task taxonomy; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 03 — input modality
**Question:** Can the team state text, image, attachment, and OCR boundary?
**Required artifact:** format manifest.
**Failing signal:** unsupported modality rate.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** route to manual review.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only input modality; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 04 — language slice
**Question:** Can the team list languages before evaluating quality?
**Required artifact:** language field distribution.
**Failing signal:** unknown language.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** defer that slice.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only language slice; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 05 — document length
**Question:** Can the team bucket tokens or bytes before interpreting cost?
**Required artifact:** length histogram.
**Failing signal:** tail documents omitted.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** add tail stratum.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only document length; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 06 — timestamp cutoff
**Question:** Can the team define when each field became available?
**Required artifact:** field lineage.
**Failing signal:** post-decision field.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** remove it.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only timestamp cutoff; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 07 — deduplication
**Question:** Can the team retain duplicate occurrence count and content hash?
**Required artifact:** hash collision policy.
**Failing signal:** duplicate across sets.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** resample.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only deduplication; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 08 — baseline rubric
**Question:** Can the team freeze manual decision instructions?
**Required artifact:** rubric version.
**Failing signal:** baseline instructions drift.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** rerun paired baseline.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only baseline rubric; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 09 — reviewer blinding
**Question:** Can the team hide vendor identity and recommendation?
**Required artifact:** console audit.
**Failing signal:** identity exposed.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** invalidate affected labels.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only reviewer blinding; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 10 — adjudication
**Question:** Can the team define how reviewer disagreements close?
**Required artifact:** adjudication record.
**Failing signal:** no final label.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** mark unresolved.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only adjudication; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 11 — abstention
**Question:** Can the team define whether abstention routes to review?
**Required artifact:** outcome policy.
**Failing signal:** abstention silently dropped.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** count and schedule review.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only abstention; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 12 — timeout
**Question:** Can the team define timeout as a distinct result?
**Required artifact:** request log.
**Failing signal:** late response counted as success.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** classify timeout.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only timeout; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 13 — retry
**Question:** Can the team bound attempts and retain request key?
**Required artifact:** attempt ledger.
**Failing signal:** duplicate billable attempt.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** deduplicate and price.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only retry; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 14 — model version
**Question:** Can the team require response assertion plus request configuration?
**Required artifact:** version log.
**Failing signal:** unverifiable version.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** block acceptance.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only model version; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 15 — endpoint region
**Question:** Can the team record region only if it changes the decision?
**Required artifact:** request metadata.
**Failing signal:** region unknown where material.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** defer terms review.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only endpoint region; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 16 — prompt revision
**Question:** Can the team hash instructions and parameter settings?
**Required artifact:** configuration snapshot.
**Failing signal:** configuration changed mid-run.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** start new run.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only prompt revision; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 17 — temperature setting
**Question:** Can the team state deterministic or stochastic sampling policy?
**Required artifact:** parameter log.
**Failing signal:** mixed settings.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** segregate outputs.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only temperature setting; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 18 — tool use
**Question:** Can the team declare external tools and their failures?
**Required artifact:** tool trace.
**Failing signal:** hidden tool result.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** exclude claim.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only tool use; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 19 — retrieval corpus
**Question:** Can the team pin retrieval snapshot when used?
**Required artifact:** corpus version.
**Failing signal:** corpus changes.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** rerun comparison.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only retrieval corpus; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 20 — reference date
**Question:** Can the team date every commercial and technical source?
**Required artifact:** source register.
**Failing signal:** undated assertion.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** treat as OPEN.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only reference date; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 21 — metric numerator
**Question:** Can the team record matches as integer before rounding?
**Required artifact:** metric export.
**Failing signal:** rounded numerator.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** recompute.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only metric numerator; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 22 — metric denominator
**Question:** Can the team reconcile submitted, excluded, and evaluated counts?
**Required artifact:** run ledger.
**Failing signal:** denominator mismatch.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** stop analysis.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only metric denominator; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 23 — cost weight
**Question:** Can the team document cost of each error class?
**Required artifact:** owner-approved matrix.
**Failing signal:** implicit equal costs.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** defer threshold.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only cost weight; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 24 — threshold
**Question:** Can the team state who selected it and why?
**Required artifact:** decision record.
**Failing signal:** post-hoc threshold.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** freeze new protocol.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only threshold; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 25 — confidence display
**Question:** Can the team label approximation and assumptions?
**Required artifact:** analysis note.
**Failing signal:** interval called guarantee.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** narrow conclusion.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only confidence display; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 26 — paired analysis
**Question:** Can the team keep baseline and vendor outcomes on same items?
**Required artifact:** paired table.
**Failing signal:** different populations.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** do not compare.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only paired analysis; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 27 — slice reporting
**Question:** Can the team report routine, ambiguous, sensitive, long, degraded slices?
**Required artifact:** stratum table.
**Failing signal:** aggregate-only report.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** block acceptance.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only slice reporting; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 28 — missingness
**Question:** Can the team categorize missing fields rather than imputing silently?
**Required artifact:** missingness report.
**Failing signal:** unknown values converted to default.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** defer item.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only missingness; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 29 — outlier handling
**Question:** Can the team declare how extreme length or cost is treated?
**Required artifact:** outlier rule.
**Failing signal:** tail deleted after outcome.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** restore and report.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only outlier handling; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 30 — cost currency
**Question:** Can the team normalize every amount to one currency and period?
**Required artifact:** finance sheet.
**Failing signal:** mixed periods.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** recompute.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only cost currency; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 31 — fixed fee
**Question:** Can the team include setup and minimum commitment?
**Required artifact:** dated quote.
**Failing signal:** fee omitted.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** add to total.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only fixed fee; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 32 — variable fee
**Question:** Can the team include all input/output/request units?
**Required artifact:** usage schedule.
**Failing signal:** unit mismatch.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** reprice.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only variable fee; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 33 — reviewer rate
**Question:** Can the team state loaded cost and unit?
**Required artifact:** staffing worksheet.
**Failing signal:** unpaid time ignored.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** re-estimate.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only reviewer rate; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 34 — integration effort
**Question:** Can the team price the proposed connector and audit work?
**Required artifact:** work breakdown.
**Failing signal:** assumed zero.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** mark OPEN.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only integration effort; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 35 — monitoring effort
**Question:** Can the team include version checks and ledger retention?
**Required artifact:** operations plan.
**Failing signal:** ongoing work excluded.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** add run-rate.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only monitoring effort; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 36 — contingency
**Question:** Can the team state rationale for reserve?
**Required artifact:** sensitivity table.
**Failing signal:** contingency treated as observed.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** label illustrative.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only contingency; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 37 — privacy review
**Question:** Can the team route terms to authorized reviewer?
**Required artifact:** review ticket.
**Failing signal:** unreviewed terms treated approved.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** do not transmit.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only privacy review; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 38 — security review
**Question:** Can the team separate product claim from assurance evidence?
**Required artifact:** evidence register.
**Failing signal:** marketing assurance accepted.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** request primary evidence.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only security review; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 39 — contract exit
**Question:** Can the team identify export, deletion, and termination questions?
**Required artifact:** contract checklist.
**Failing signal:** lock-in omitted.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** defer commercial gate.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only contract exit; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 40 — manual fallback
**Question:** Can the team schedule people and instructions before pilot?
**Required artifact:** fallback drill.
**Failing signal:** fallback exists only on paper.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** rehearse drill.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only manual fallback; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 41 — kill switch
**Question:** Can the team test stopping new submissions without losing ledger?
**Required artifact:** state trace.
**Failing signal:** stop action deletes evidence.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** fix transition.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only kill switch; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 42 — review queue
**Question:** Can the team calculate hours from trigger rates and minutes?
**Required artifact:** capacity model.
**Failing signal:** overlap double counted or ignored.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** measure overlap.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only review queue; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 43 — incident escalation
**Question:** Can the team name owner for unexpected output?
**Required artifact:** on-call matrix.
**Failing signal:** no accountable responder.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** do not start pilot.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only incident escalation; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 44 — decision authority
**Question:** Can the team separate recommendation from approval?
**Required artifact:** RACI record.
**Failing signal:** vendor or model decides.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** restore authority.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only decision authority; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 45 — audit retention
**Question:** Can the team record what can be retained and why?
**Required artifact:** retention design.
**Failing signal:** evidence unavailable at review.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** resolve authorized policy.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only audit retention; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 46 — expiry
**Question:** Can the team set automatic review date?
**Required artifact:** decision ledger.
**Failing signal:** old approval reused.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** transition REVIEW_REQUIRED.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only expiry; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 47 — price change
**Question:** Can the team compare dated quote hash?
**Required artifact:** commercial snapshot.
**Failing signal:** price update ignored.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** re-cost.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only price change; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 48 — workload drift
**Question:** Can the team compare pilot intake to manifest?
**Required artifact:** drift dashboard.
**Failing signal:** new slice hidden.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** pause or resample.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only workload drift; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 49 — vendor outage
**Question:** Can the team measure failure result and fallback load?
**Required artifact:** failure drill.
**Failing signal:** outage excluded from metric.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** include reliability outcome.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only vendor outage; it does not establish vendor capability, contractual approval, or deployment readiness.

#### Gate 50 — termination decision
**Question:** Can the team state reject/defer/accept rationale with gates?
**Required artifact:** signed ledger.
**Failing signal:** ambiguous next state.
**Decision effect:** The gate is not passed; no aggregate score compensates for it.
**Recovery:** return to evidence request.
**Verification:** record the check, actor, timestamp, source version, and resulting state in the decision ledger.
**Boundary:** passing this gate establishes only termination decision; it does not establish vendor capability, contractual approval, or deployment readiness.


## 14. Connections & Cross-References
The case connects to use-case chartering (decision, baseline, owner), market mapping (deployment and control boundary), and technology due diligence (claim, evidence, cost, and expiry). It intentionally does not claim a conclusion about a named vendor.

### Q1. What exactly is being accepted?
**Answer:** A bounded pilot decision, not the vendor’s general claim. The record names the task, version, workload, gates, expiry, and owner. Follow-up: ask whether any hard gate is missing. Acceptable alternative: defer until the packet is complete. Red flag: “80% means buy.” Scoring: name scope and authority.

### Q2. Why is a benchmark score insufficient?
**Answer:** A score may use a different population, label, metric, version, and cost of error. Compare frozen workload strata and the manual baseline before inferring relevance. Follow-up: request exclusions. Acceptable alternative: reject unsupported portability. Red flag: treating a marketing chart as an acceptance test. Scoring: identifies at least three mismatches.

### Q3. How do you make the workload fair?
**Answer:** Freeze the sample before outputs, stratify the intended intake, preserve hard cases and failures, and report every inclusion and exclusion. Follow-up: compare sample shares with the declared target. Acceptable alternative: a separately governed random sample. Red flag: letting the vendor curate the set. Scoring: names a manifest and strata.

### Q4. What is the denominator for agreement?
**Answer:** The evaluated document-decision pairs: 270 in the illustrative example, including defined outcomes for abstention and error. 216/270=80.0%. Follow-up: ask how excluded records are reported. Acceptable alternative: a different metric with an explicit denominator. Red flag: divide by successful responses only. Scoring: correct units and arithmetic.

### Q5. What uncertainty changes the recommendation?
**Answer:** The rough 75.2%–84.8% planning interval overlaps the 76.7% illustrative baseline; it does not justify superiority. Follow-up: ask about pairing and sampling dependence. Acceptable alternative: defer pending a larger, appropriate analysis. Red flag: claim certainty from rounded points. Scoring: distinguishes planning interval from proof.

### Q6. How is reviewer capacity calculated?
**Answer:** Use R=N(a+d+q) as a conservative planning sum, then multiply review items by minutes/item. With 1,000, 5%, 12%, and 10%, R=270 and 36 hours at 8 minutes. Follow-up: deduplicate overlapping triggers. Acceptable alternative: simulate a logged workflow. Red flag: call all model outputs saved time. Scoring: shows units and boundary.

### Q7. What belongs in true total cost?
**Answer:** Fixed fees, variable usage, retries, reviewer time, integration, monitoring, and contingency, all normalized to one period and currency. Follow-up: identify unpriced terms. Acceptable alternative: defer commercial decision. Red flag: price only one API call. Scoring: names both human and technical costs.

### Q8. What happens if the vendor version changes?
**Answer:** The decision moves to REVIEW_REQUIRED; block or segregate requests, retain V1 evidence, and rerun the frozen comparison for V2. Follow-up: ask whether the asserted version can be audited. Acceptable alternative: reject if version cannot be pinned. Red flag: assume product name implies equivalence. Scoring: names invalidation and re-evaluation.

### Q9. When should the procurement lead reject rather than defer?
**Answer:** Reject when a hard constraint cannot be satisfied, such as no human authority, unacceptable terms after authorized review, no version trace, or no reversible fallback. Follow-up: distinguish unknown evidence from a known disqualifier. Acceptable alternative: formal no-go with conditions for resubmission. Red flag: accept “pending later fixes.” Scoring: identifies a hard gate.

### Q10. How do you avoid label leakage?
**Answer:** Blind reviewers to vendor identity and output, freeze labels independently, timestamp every field, and keep later resolutions out of decision-time inputs. Follow-up: audit access logs. Acceptable alternative: use pre-existing adjudications with provenance. Red flag: ask vendor staff to correct test labels. Scoring: names timing and blinding.

### Q11. What does a safe rollback do?
**Answer:** It stops new submissions, preserves the decision and request ledger, routes queued work to manual review, and records the trigger. It does not erase failures or pretend no evaluation occurred. Follow-up: test manual capacity. Acceptable alternative: a controlled pause. Red flag: delete evidence. Scoring: includes workflow and evidence preservation.

### Q12. Which changed constraint reverses this design?
**Answer:** If review capacity falls below the calculated demand, a pilot that looks affordable by API price becomes operationally infeasible; cap volume or defer. Another reversal is a version change that invalidates the frozen comparison. Follow-up: recompute at the new constraint. Acceptable alternative: redesign scope. Red flag: retain prior approval unchanged. Scoring: provides causal mechanism and recalculation.

### Q13. Why cannot model confidence replace approval?
**Answer:** Confidence is a model output whose calibration and task scope must be demonstrated; approval is an accountable organizational action with authority and reversible process. Follow-up: inspect the final actor field. Acceptable alternative: require a separate policy gate. Red flag: automate because a score is high. Scoring: distinguishes prediction from authority.

### Q14. How would you test a vendor claim about latency?
**Answer:** Specify request shape, concurrency, version, regions if relevant, timeout, retry policy, and percentile metric; compare the permitted workload and retain raw timing logs. Follow-up: inspect p99 and error rate. Acceptable alternative: state latency is not measured. Red flag: use an average from a different workload. Scoring: names workload and tail behavior.

## 15. Quick Reference Card
| Field | Decision card |
|---|---|
| Unit | one versioned vendor-claim evaluation on one document-decision pair |
| Baseline | manual evidence review using the same frozen inputs and rubric |
| Decision owner | procurement lead; reviewers and finance inform but do not silently approve |
| Accept-pilot gates | fixed version; workload-fair manifest; stratum metrics; scheduled review capacity; full cost; fallback; authorized terms review |
| Metric | agreement = matches / evaluated document-decision pairs; report abstentions, errors, and exclusions |
| Capacity | R=N(a+d+q); deduplicate real overlaps; convert review items to staffed hours |
| Cost | fixed + variable + retries + reviewer + integration + monitoring + contingency |
| Version rule | model, price, term, or workload change → REVIEW_REQUIRED |
| Evidence limit | all numbers are illustrative derived arithmetic, not vendor performance |
| Escalate | missing version, missing denominator, failed hard gate, unstaffed review queue, or authority ambiguity |
