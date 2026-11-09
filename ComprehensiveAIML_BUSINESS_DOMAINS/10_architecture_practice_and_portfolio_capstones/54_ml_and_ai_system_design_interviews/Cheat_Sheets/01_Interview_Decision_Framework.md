---
collection: "ml-and-ai-system-design-interviews"
file_id: "01"
topic: "Interview decision framework"
level: 1
status: "draft"
version: "1.0"
created: "2026-11-09"
reviewed: null
review_trigger: "A source, evaluation protocol, or system constraint changes."
notation_style: "plain-text"
prerequisites: ["../INDEX.md"]
pattern_tags: ["decision-contract", "evidence-boundary"]
source_status: "OPEN"
owner_role: "interview curriculum maintainer"
source_index: "INDEX.md"
---

# Interview Decision Framework

> **Purpose:** Convert an ambiguous ML or AI system-design prompt into a bounded decision contract, architecture hypothesis, metric plan, capacity calculation, failure diagnosis, and defensible answer.
>
> **Prerequisites:** Basic supervised-learning metrics, request/response systems, and dimensional arithmetic.
>
> **Key outcome:** Given a prompt and changed constraint, you can state the owner, input grain, action, baseline, objective, guardrails, evidence boundary, and next measurement.
>
> **This file does NOT cover:** vendor-specific APIs, organization architectures, or a universal design answer.
>
> **Level:** 1 — Foundations.
>
> **Evidence:** DERIVED framework and ILLUSTRATIVE calculations; OPEN source-case bodies and empirical benchmarks.
>
> **Source inventory:** the module’s [reference plan](../INDEX.md#reference-informed-additions-to-develop) names routing, counters, indexes, data structures, caches, typeahead, ranking, and deduplication as planned coverage. Its linked interview references are prompts, not evidence of a deployment.

## 1. File Header

**Source scope:** [directory.txt](../../../../Templates/directory.txt#L286) lists `09_Full_Practice_Exam/09_01_practice_exam_60q.md`; its original questions and answer key are unavailable. The 60-item illustrative ledger here is newly constructed teaching material, not an official exam reconstruction.

A strong system-design answer is a sequence of falsifiable decisions, not a component list. The interviewer’s prompt usually leaves the decision owner, population, loss function, latency scope, cost boundary, and safety requirement unstated. The candidate must ask which of those omissions changes the recommendation rather than reflexively draw a familiar architecture.

Use the framework in this order: restate the decision; expose constraints; select the simplest credible baseline; draw only the path needed to evaluate that decision; define a metric and guardrail with denominators; do one sizing check; name the failure/recovery boundary; and close with what evidence would reverse the choice. A correct answer may be “defer implementation until the label, action, or safety boundary is clarified.”

**Takeaway:** Architecture begins after the decision boundary is explicit, not when the first storage technology is named.

## 2. Concept Map / Mental Model

```text
[prompt]
   |
   v
[actor + unit + action + timing] --unknown--> [clarifying question]
   |
   v
[objective + baseline + guardrails] --proxy conflict--> [metric contract]
   |
   v
[data / state / model / service path] --assumption fails--> [fallback]
   |
   v
[capacity + cost estimate] --unstable--> [load test plan]
   |
   v
[evaluation + failure/recovery] --> [recommendation + evidence boundary]
```

The path has three separate contracts. The **business contract** says who may do what after an output. The **technical contract** says input schema, state owner, output semantics, latency, and retry behavior. The **evidence contract** says which artifacts establish a claim and which claims are still hypotheses. A design is incomplete when any contract is implicit.

**Takeaway:** Keep business, technical, and evidence contracts separate so a strong component cannot conceal a weak decision.

## 3. Foundational Definitions

| Term | Meaning | Type / units | Boundary |
|---|---|---|---|
| decision unit | smallest entity for which an action is taken | request, user, item, event, or session | not automatically the training row |
| decision owner | role authorized to accept, reject, or defer output | person/team | not the model author by default |
| baseline | current rule/manual/system process under the same population/window | comparator | “no model” is not necessarily no process |
| objective | outcome to improve at appropriate aggregation | currency, rate, time, count | proxy may conflict with outcome |
| guardrail | metric that blocks an apparently good primary result | rate/count/time | must have a threshold and owner |
| prediction time | last instant at which a feature is legitimately available | timestamp | differs from label time |
| serving boundary | named point where output becomes actionable | interface | model score alone may be upstream |
| oracle | explicit expected result or invariant for a test | predicate/value | dashboard appearance is not an oracle |
| evidence level | design, trace-checked, measured, or sourced claim scope | category | one label cannot certify every statement |

A clarification question is valuable when the answer would change action, architecture, metric, or release gate. “What database do you use?” is premature when no update/query contract exists. “What is the cost of a false positive and who absorbs it?” can reverse a threshold or even rule out automation.

**Takeaway:** Define the unit, action, comparator, and time boundary before discussing a model.

## 4. Core Formulas with Full Derivation

### 4.1 Expected decision value

**Validity:** decisions are mutually exclusive; values are defined for the same decision unit and horizon; estimated probabilities are sufficiently calibrated for the stated use. This is not valid when actions have interacting capacity or delayed externalities that are ignored.

**Derivation:** expected value is the probability-weighted sum of action outcomes. For a binary positive action with benefit `v_TP` for a true positive and cost `c_FP` for a false positive, act when expected gain is nonnegative.

```text
FORMULA [IDF-01]: EV(positive) = p*v_TP - (1-p)*c_FP
WHERE: p [probability of true positive]; v_TP [currency/action];
       c_FP [currency/action]; EV [currency/action].
CONSTRAINT: p is calibrated for the decision population; costs share a horizon.
NUMERICAL EXAMPLE: p=0.20; v_TP=$50; c_FP=$8;
                   EV=0.20*50-0.80*8=$3.60/action.
INTERPRETATION: the positive action has positive expected direct value before capacity and guardrails.
```

Independent arithmetic: `$10-$6.40=$3.60`. The break-even probability is `c_FP/(v_TP+c_FP)=8/58=0.1379`. If the false-positive cost becomes $20, break-even is `20/70=0.2857`; the original 0.20 score should not trigger action. This is why an AUC score cannot itself choose an operational threshold.

### 4.2 Stable-work estimate

**Validity:** average arrival and service rates use the same work unit; service capacity is sustainable; the estimate is not a tail-latency, burst, or retry guarantee.

```text
FORMULA [IDF-02]: B = max(0, lambda-mu)*t
WHERE: B [requests]; lambda [requests/s]; mu [requests/s]; t [s].
CONSTRAINT: initially empty fluid queue, constant rates, no losses/retries.
NUMERICAL EXAMPLE: lambda=120 requests/s; mu=100 requests/s; t=600 s;
                   B=(120-100)*600=12,000 requests.
INTERPRETATION: buffering hides a ten-minute overload; it does not solve it.
```

If recovery service is 150 requests/s while arrival stays 120 requests/s, drain time is `12,000/(150-120)=400 s`. If recovery service is 120 requests/s, the backlog never drains. The dimensions check: `(requests/s)×s=requests`.

### 4.3 Decision-contract invariant

**Precondition:** an output triggers an action or explicitly triggers no action. **Invariant [IDF-03]:** every action must be reconstructible from `(input version, feature/state version, model/rule version, threshold/policy version, timestamp, owner, outcome handle)`. Missing one field does not prove an action was wrong; it prevents a reliable audit or replay. A trace of a score without the applied threshold is not a decision trace.

**Takeaway:** A formula becomes operational only after units, decision population, and counterexample are stated.
### 4.4 Practice-exam scoring is a decision, not a vanity percentage

The source-plan exercise uses 60 questions: 45 correct, 10 wrong, and 5 unanswered.
It is ILLUSTRATIVE course arithmetic, not an official examination result or certification blueprint.
The denominator must be declared before the score is interpreted.

**Validity:** `correct`, `wrong`, and `unanswered` are mutually exclusive dispositions for one fixed question set.
An answered-only score is invalid as a claim about readiness when non-answers are possible in the real assessment.
It is useful only to isolate quality conditional on an attempt.

**Derivation:** total questions equal all three mutually exclusive dispositions.
Overall accuracy counts correct answers over the total opportunity set.
Answered-only accuracy conditions on the subset for which an answer was supplied.

```text
FORMULA [IDF-04]: N = C + W + U
WHERE: N [questions]; C [correct questions]; W [wrong questions];
       U [unanswered questions].
CONSTRAINT: one final disposition per question; no partial-credit scheme.
NUMERICAL EXAMPLE: C=45; W=10; U=5; N=45+10+5=60 questions.
INTERPRETATION: the denominator is the whole practice set, not merely attempted items.
```

```text
FORMULA [IDF-05]: A_all = C / N
WHERE: A_all [fraction of all questions]; C [questions]; N [questions].
CONSTRAINT: all questions are equally weighted and scoring is binary.
NUMERICAL EXAMPLE: A_all=45/60=0.75=75%.
INTERPRETATION: this is the completion-sensitive practice score.
```

```text
FORMULA [IDF-06]: A_answered = C / (C + W)
WHERE: A_answered [fraction of answered questions]; C,W [questions].
CONSTRAINT: U is excluded deliberately and reported separately.
NUMERICAL EXAMPLE: A_answered=45/(45+10)=45/55=0.81818...=81.82%.
INTERPRETATION: knowledge among attempts appears stronger than overall readiness.
```

The independent check is `60×0.75=45` correct questions.
For the conditional score, `55×0.81818...=45` to rounding.
The 6.82 percentage-point gap is not an error; it is the consequence of two different denominators.

**Sensitivity:** if all five unanswered questions are later answered incorrectly, overall accuracy remains 75%.
Answered-only accuracy becomes `45/60=75%`.
If they are all answered correctly, overall accuracy becomes `50/60=83.33%`.
The score range demonstrates why “81.82% accuracy” without a denominator can misallocate remediation.

**Boundary:** unequal item weights, partial credit, adaptive questioning, or a changing blueprint require a different scoring rule.
No numerical score establishes production competence, interview performance, or correctness of an answer key.

### 4.5 Capacity admission combines value with a hard operational limit

**Validity:** candidates are rankable by a comparable estimated value, each review consumes one identical capacity unit, and the horizon is fixed.
This is a selection rule, not proof that estimates are calibrated or that reviews have equal duration.

**Derivation:** with capacity `K`, a policy may admit no more than `K` candidates.
If each candidate `i` has estimated net value `EV_i`, maximizing total estimated value under equal unit cost selects the highest positive values up to `K`.

```text
FORMULA [IDF-07]: Admit = {i : rank(EV_i) <= K and EV_i > 0}
WHERE: i [candidate index]; rank [descending value rank]; K [reviews/horizon];
       EV_i [currency/candidate].
CONSTRAINT: equal review effort; no coupled decisions; estimates comparable.
NUMERICAL EXAMPLE: K=3; EV=[12, 4, -1, 9, 2] dollars;
                   sorted positives=[12,9,4,2]; admit=[12,9,4].
INTERPRETATION: a positive candidate can be deferred when capacity is exhausted.
```

The selected value is `$12+$9+$4=$25` per stated horizon.
Admitting all positive candidates would consume four review slots and violate `K=3`.
Replacing `$4` with `-$1` to fill a slot is not neutral: it lowers estimated direct value and consumes scarce review time.

**Sensitivity:** if the `$4` candidate needs two reviewer slots while others need one, equal-unit ranking fails.
Use value per constrained minute only after measuring or responsibly estimating handling times.
If a fairness guardrail requires a minimum review allocation for a protected slice, unconstrained value ranking is disallowed.

### 4.6 Reliability budget converts a verbal availability request into an error allowance

**Validity:** the stated service-level objective (SLO) has a fixed eligible request population and a declared window.
An availability SLO does not measure semantic correctness, model quality, or user value.

**Derivation:** an SLO permits the complement of its target to fail.
Multiplying eligible requests by that fraction yields the maximum count of bad eligible requests in the window.

```text
FORMULA [IDF-08]: E = R * (1 - S)
WHERE: E [eligible failed requests/window]; R [eligible requests/window];
       S [target success fraction, 0<=S<=1].
CONSTRAINT: identical eligibility and success definitions in R and E.
NUMERICAL EXAMPLE: R=2,000,000 requests/month; S=0.999;
                   E=2,000,000*(1-0.999)=2,000 requests/month.
INTERPRETATION: “99.9%” still permits 2,000 failures under this workload definition.
```

Dimension check: requests multiplied by a dimensionless fraction yields requests.
If retries are counted as separate eligible requests, the budget can be inflated while users still fail.
The request identifier and final user-visible result must therefore be available for a user-journey guardrail.

**Sensitivity:** at the same `S=0.999`, a 10× traffic increase gives `20,000` permitted failed requests.
At `R=2,000,000`, moving to `S=0.9999` gives `200` failures, not zero.
This changes the architecture discussion from “high availability” to a concrete recovery and observability obligation.

### 4.7 Latency budgets compose only when each segment is measured on the same request population

**Validity:** stages execute serially for the request path being budgeted, timestamps use a common clock or measured duration, and the budget is a planning allocation.
Do not sum independent percentile values and call the result an end-to-end percentile guarantee.

**Derivation:** elapsed serial-path time is the sum of the individual elapsed durations for one trace.
Budget allocation reserves a maximum target for each owned stage; it is not a proof of tail behavior.

```text
FORMULA [IDF-09]: T_trace = T_ingest + T_feature + T_score + T_policy + T_write
WHERE: each T_* [ms for the same request trace]; T_trace [ms/request].
CONSTRAINT: serial trace; no omitted asynchronous user-blocking work.
NUMERICAL EXAMPLE: 8+22+15+5+10=60 ms/request.
INTERPRETATION: a 75-ms target leaves 15 ms of trace-level slack.
```

The arithmetic oracle is `8+22=30`, `30+15=45`, `45+5=50`, and `50+10=60`.
A single 60-ms trace does not establish p95 or p99 latency.
For interview design, ask which segment owns the remaining 15 ms and how a timeout changes the decision.

**Sensitivity:** an added 25-ms synchronous retrieval step makes the trace 85 ms and exceeds the 75-ms target.
The candidate can cache, precompute, degrade the response, or reject the requirement; each choice changes freshness or correctness.
Calling the retrieval “asynchronous” is invalid if the action waits for it.

### 4.8 Mechanism block: decision trace and idempotent retry

**Preconditions:** a request has a stable idempotency key, a policy version, an authenticated caller, and a durable action ledger.
**Input:** `(request_id, input_version, policy_version, attempt_number)`.
**State transition:** lookup the ledger by `request_id`; return the prior committed action when present; otherwise evaluate policy, atomically record one action, then emit its side effect.
**Output:** `(action, decision_id, policy_version, replay_flag)`.
**Invariant:** at most one committed action exists per `request_id` for the contract horizon.
**Justification:** retries can repeat delivery, but a durable key lookup makes repeats return the existing decision rather than create a second action.

Worked trace:

```text
t0: request R7, policy P3 arrives; ledger[R7] absent.
t1: P3 returns REVIEW; transaction commits ledger[R7]=D91/REVIEW.
t2: network reply is lost; caller retries R7.
t3: ledger[R7] is D91/REVIEW; service returns D91 with replay_flag=true.
t4: reviewer queue receives only D91 from the transactional outbox.
```

Recovery boundary: if the action is sent before the ledger commit and the process dies, duplicate or missing actions remain possible.
The correct recovery design needs an outbox/reconciliation rule, not merely an HTTP retry count.
This trace is DERIVED from the stated transaction behavior; it is not evidence that a particular platform supplies atomicity.

**Takeaway:** A decision framework must make scoring, capacity, reliability, latency, and retry semantics auditable on their own units.

## 5. Worked Numerical Example

**ILLUSTRATIVE prompt:** “Design a system that flags possibly fraudulent purchase attempts for review.” This is not a claim about any organization or observed fraud rate.

The decision unit is one purchase attempt at authorization time. The action is route to manual review, not automatically decline. The baseline is a fixed rule which routes attempts above a simple amount/risk condition. Assume review has 100 decisions/hour capacity. A candidate score is calibrated only after a proposed temporal evaluation; that premise is unverified.

For 200 candidates/hour with `p=0.20`, `v_TP=$50`, and `c_FP=$8`, IDF-01 gives $3.60/direct action. Reviewing all 200 exceeds capacity. Rank by expected value, select at most 100/hour, and include review delay as a guardrail. The capacity calculation is not complete until the reviewer response-time distribution and opportunity cost are known.

A plausible wrong method is to maximize recall at a threshold chosen on a random split. It can leak future outcome signals, overwhelm review capacity, and increase false positives. A better proposed validation uses chronological splits, calibration checks, workload replay, reviewer-queue measurement, and a held-out business-outcome window. None is executed in this example.

**Sensitivity:** if a changed constraint says false positives impose $20 of customer harm, the 0.20 score is below break-even. The recommendation changes from review to defer/collect more evidence or require a higher threshold. This demonstrates trade-off communication: explain exactly which value changed and why the architecture did not answer it.

**Takeaway:** Compute on the action’s unit and capacity, not on an abstract model score alone.

## 6. Visual / Geometric Intuition

```text
request --> feature snapshot --> score/rule --> threshold + capacity gate --> action
                 |                  |                 |                    |
                 v                  v                 v                    v
              time audit         version audit      policy audit         outcome link
```

The score axis and the action axis are different. A threshold slices scores, but capacity may add a second boundary: high-score requests can wait, expire, or be sampled. A model can move a score upward while reducing business value if it shifts workload from valuable to low-value cases or damages a guardrail population. Visualize the queue separately from the ROC curve.

**Takeaway:** A model ranking becomes a system only after policy, capacity, and feedback paths are drawn.

## 7. Comparison / Selection Table

| Choice | Use when | Strength | Avoid when | Required check |
|---|---|---|---|---|
| manual workflow | stakes/volume uncertain | preserves judgment and reveals process | volume requires impossible response time | baseline time/error measurement |
| deterministic rule | invariant/simple policy is known | auditable and low latency | boundary is unstable or interactions complex | rule coverage and exception review |
| batch model | action can wait | simpler compute/cost control | prediction must use fresh state | cutoff-time leakage audit |
| online model | action needs current state | responsive | feature/state consistency is unavailable | point-in-time feature contract |
| retrieval/LLM assistant | output is assistive and sources are available | handles unstructured context | unsupported/unsafe action is possible | citation and permission evaluation |
| no automation | decision/objective unknown | avoids false precision | a safe bounded pilot is feasible | document missing evidence |

**Takeaway:** Choose the simplest option that satisfies the decision contract, then name the trigger for revisiting it.

## 8. Failure Modes & Edge Cases

| Failure | Competing causes | Discriminating check | Bounded fix | Verification |
|---|---|---|---|---|
| impressive offline metric, poor action value | leakage; proxy mismatch; threshold/capacity mismatch | time audit, action-value ledger, queue replay | correct split and optimize decision utility | held-out temporal evaluation |
| backlog grows despite more workers | hot key; downstream bottleneck; retry amplification | per-key/service/retry rates | partition or throttle at owned boundary | overload replay and reconciliation |
| score is unavailable at decision time | delayed feature; stale cache; schema drift | feature availability timestamp vs action timestamp | remove/lag feature or defer action | point-in-time feature test |
| LLM answer sounds grounded but is unsupported | retrieval miss; prompt injection; stale corpus | citation-to-claim audit and adversarial inputs | withhold/quote sources; approval gate | versioned evaluation set |
| rollback restores code but behavior differs | model/index/prompt/policy mismatch | compare complete release manifest | promote/recover compatible bundle | rollback rehearsal |

**Takeaway:** Diagnose by a signal that distinguishes causes; “the model is bad” is not a diagnosis.
### Diagnostic dossiers

#### FD-01 — Offline lift disappears after release

**Trigger/environment:** a temporal holdout metric is high while the first shadow replay yields low action value.
**Competing causes:** future-label leakage, a proxy objective, or a threshold applied to a different prevalence.
**Discriminating diagnostic:** rebuild one prediction using only fields timestamped before the decision; then compare score calibration and policy inputs.
**Expected signal:** a large metric drop after point-in-time reconstruction implicates leakage rather than serving latency.
**Immediate mitigation:** stop automatic action and retain the prior policy while preserving traces.
**Correction:** version feature availability and run temporal, policy-aware evaluation.
**Prevention:** require a prediction-time field ledger in design review.
**Verification:** a held-out chronological replay produces its decision trace from allowed inputs only.
**Risk/escalation:** removing leaked fields may reduce coverage; escalate a material business-rule conflict to the decision owner.

#### FD-02 — Review queue meets average capacity but violates customer deadline

**Trigger/environment:** mean arrivals are below nominal reviewer throughput yet aged requests accumulate.
**Competing causes:** bursty arrivals, heterogeneous handling times, or priority starvation.
**Discriminating diagnostic:** inspect arrival percentiles, service-time distribution, and age by policy priority.
**Expected signal:** an old low-priority cohort with stable mean throughput indicates scheduling starvation.
**Immediate mitigation:** expire or route requests according to a declared safe fallback, not silent abandonment.
**Correction:** set per-class deadlines and reserve capacity or change admission.
**Prevention:** monitor age-at-decision alongside queue length.
**Verification:** replay a burst and show every class either meets its deadline or receives its documented fallback.
**Risk/escalation:** reserved capacity can reduce aggregate value; obtain owner approval for the tradeoff.

#### FD-03 — Feature freshness silently diverges across online and offline paths

**Trigger/environment:** training data contain a same-day aggregate while serving reads a lagged cache.
**Competing causes:** delayed ETL, cache invalidation failure, or a timestamp time-zone conversion.
**Discriminating diagnostic:** compare the feature value and availability timestamp for identical entity-time keys in both paths.
**Expected signal:** systematic lag by partition suggests pipeline delay; isolated discrepancies suggest invalidation or key mismatch.
**Immediate mitigation:** remove the feature from policy or use the proven lagged definition.
**Correction:** publish a single feature contract with freshness and backfill semantics.
**Prevention:** add point-in-time parity tests to each release candidate.
**Verification:** sampled historical decisions reproduce online-visible features exactly within declared lag.
**Risk/escalation:** a freshness fix can introduce load; capacity test before broad enablement.

#### FD-04 — A retrieval assistant cites documents that do not support its answer

**Trigger/environment:** an answer contains a citation but a reviewer cannot locate the asserted proposition.
**Competing causes:** retrieval miss, misleading chunk boundary, generation hallucination, or stale index.
**Discriminating diagnostic:** inspect retrieved text, rank scores, source version, and claim-to-span alignment.
**Expected signal:** correct source absent from candidates indicates retrieval; present but unsupported wording indicates generation/grounding failure.
**Immediate mitigation:** withhold the unsupported assertion and display a source excerpt or escalation path.
**Correction:** evaluate claim-level entailment and revise chunking, prompt, or refusal policy.
**Prevention:** version corpora and keep adversarial citation tests in the release gate.
**Verification:** a fixed evaluation set records support, abstention, and unsafe-citation rates by source version.
**Risk/escalation:** aggressive abstention reduces task completion; owner chooses the acceptable balance.

#### FD-05 — Retry storm amplifies a partial outage

**Trigger/environment:** downstream latency rises and upstream request volume rises faster than client demand.
**Competing causes:** synchronized retries, missing deadlines, or a hot partition causing genuine overload.
**Discriminating diagnostic:** compare original request IDs to attempt counts and plot retry delay synchronization.
**Expected signal:** many attempts per stable request ID with aligned retry times implicates amplification.
**Immediate mitigation:** enforce deadline propagation, bounded retries, and jitter at the caller boundary.
**Correction:** make repeated operations idempotent and publish retry ownership per dependency.
**Prevention:** exercise dependency timeouts in failure injection.
**Verification:** induced downstream timeout keeps total attempts below the declared retry budget.
**Risk/escalation:** too-small retry budgets reduce recovery from transient faults; tune using a measured dependency profile.

#### FD-06 — A cache rollback restores stale policy behavior

**Trigger/environment:** code is rolled back but decisions still reflect an unintended policy or model.
**Competing causes:** stale cache, incompatible schema, wrong traffic routing, or incomplete release manifest.
**Discriminating diagnostic:** query a known decision trace and compare code, model, policy, index, and cache versions.
**Expected signal:** code match with policy mismatch identifies an incomplete bundle rather than a failed binary rollback.
**Immediate mitigation:** stop affected actions or pin traffic to a known compatible manifest.
**Correction:** release and roll back immutable bundles with explicit compatibility checks.
**Prevention:** rehearse rollback using a trace oracle before production use.
**Verification:** the known trace yields the pre-declared decision after rollback and cache invalidation.
**Risk/escalation:** cache purge can cause a load spike; sequence with capacity owner.

#### FD-07 — A fairness guardrail regresses while the aggregate objective improves

**Trigger/environment:** aggregate utility rises but one defined subgroup has a worse error or delay rate.
**Competing causes:** population mix shift, subgroup label quality, or policy threshold interaction.
**Discriminating diagnostic:** freeze cohort definitions and compare numerator, denominator, coverage, and calibration by slice.
**Expected signal:** denominator collapse is an observability problem; stable denominators with changed rates is a policy outcome.
**Immediate mitigation:** halt expansion and apply the previously approved policy for the affected slice.
**Correction:** constrain threshold/admission selection with the agreed guardrail.
**Prevention:** require slice coverage minimums and missingness alerts.
**Verification:** a held-out, time-bounded evaluation reports utility and guardrail uncertainty for each eligible slice.
**Risk/escalation:** slice constraints may reduce global metric; decision owner resolves value conflict explicitly.

#### FD-08 — Capacity calculation is numerically correct but operationally useless

**Trigger/environment:** a fluid-rate estimate says stable while p99 latency breaches.
**Competing causes:** burstiness, variable service time, synchronized dependencies, or a wrong work unit.
**Discriminating diagnostic:** compare arrival/service distributions and trace request fan-out rather than averages alone.
**Expected signal:** stable mean with clustered arrivals demonstrates the model boundary, not arithmetic failure.
**Immediate mitigation:** shed noncritical work and expose a bounded degraded response.
**Correction:** size from measured percentile workload and per-stage bottlenecks.
**Prevention:** state “fluid estimate” on every capacity result and attach a load-test protocol.
**Verification:** load replay includes burst, retry, and dependency-latency scenarios with explicit pass criteria.
**Risk/escalation:** shedding can bias outcomes; assess affected population before enabling.

#### FD-09 — Answer-key disagreement is mistaken for a learner defect

**Trigger/environment:** reviewers mark the same design answer differently.
**Competing causes:** ambiguous prompt, unsupported key, unstated constraint, or inconsistent rubric use.
**Discriminating diagnostic:** ask independent reviewers to identify the decision contract before seeing the key.
**Expected signal:** differing inferred constraints indicate prompt ambiguity rather than candidate ignorance.
**Immediate mitigation:** do not use the disputed item for remediation ranking.
**Correction:** publish an answer rationale, acceptable alternatives, and disqualifying assumptions.
**Prevention:** run answerability and inter-reviewer calibration before using the item.
**Verification:** reviewers converge on the rubric under the stated scenario, with disagreements logged.
**Risk/escalation:** tighter prompts can make practice less realistic; preserve controlled ambiguity with scoring guidance.

#### FD-10 — Interview response sounds complete but has no falsifiable claim

**Trigger/environment:** a candidate lists technologies without a baseline, metric, or reversal trigger.
**Competing causes:** missing requirements, rehearsed architecture, or interviewer failure to ask for a decision.
**Discriminating diagnostic:** request one action, one denominator, one failure boundary, and one next measurement.
**Expected signal:** inability to name any of these reveals an unbounded proposal.
**Immediate mitigation:** pause component selection and restate the decision contract.
**Correction:** use the framework’s owner-unit-action-baseline sequence.
**Prevention:** score practice answers on traceable claims, not tool count.
**Verification:** another reviewer can reconstruct the same recommendation and name what evidence would reverse it.
**Risk/escalation:** excessive questioning consumes interview time; prioritize questions that change the recommendation.

## 9. Complexity & Cost Analysis

For a request path, list feature reads, model/rule computation, retrieval calls, queue state, human review, logging, retries, and failure work. Big-O notation cannot price network calls or reviewer capacity. Under IDF-02’s constructed overload, 12,000 buffered requests require a retention and privacy decision; at 2 KiB payload-only each, the lower bound is `24,576,000 bytes = 23.4375 MiB`, excluding metadata, indexes, replication, and logs.

Measure or estimate cost at the decision boundary: request volume, peak factor, payload, storage duration, model tokens/compute if applicable, review minutes, and failed/retried work. Do not quote a vendor/model price without a dated primary source and complete workload basis.

**Takeaway:** State what a cost estimate excludes; unknown costs are safer than invented precision.

## 10. Key Theorems / Results Reference

| Result | Use | Essential condition | Limit |
|---|---|---|---|
| expected value | threshold/action comparison | comparable values and calibrated probability | capacity/externalities may invalidate independent action |
| conservation of flow | backlog calculation | same arrival/completion boundary | no tail guarantee |
| Bayes decision rule | compare conditional action losses | valid conditional probabilities/losses | does not choose the business losses |
| no-free-lunch intuition | resist universal “best model” claims | problem distribution unspecified | not a substitute for evaluation |
| end-to-end argument | place final correctness check near action | endpoint can validate property | lower-layer checks still useful |

**Takeaway:** Cite the result as a scoped tool, never as permission to skip measurement.

## 11. Interview / Exam Questions

1. **What question do you ask first?** Ask who owns the decision, what action follows output, and the cost of false actions. A tool question is secondary.
2. **Calculate the direct value at `p=.20`, `$50` true-positive value, `$8` false-positive cost.** `$3.60/action`; it is positive only under the stated calibrated-probability/cost assumptions.
3. **Why is AUC insufficient?** It does not encode threshold, action capacity, class prevalence, harm, or calibration at the operating point.
4. **What changes if review capacity is 100/hour for 200 candidates?** Add an admission/ranking policy and measure delay; do not pretend every score becomes an action.
5. **How do you find leakage?** Trace each feature’s earliest availability against prediction time; fit transformations inside training folds; use a time-respecting holdout.
6. **A queue grows after adding workers. Diagnose.** Inspect per-key load, downstream service rate, retries, and partition assignment; aggregate averages can hide a hot partition.
7. **When would a rule beat ML?** When policy is stable, exceptions are manageable, labels are weak/delayed, or auditability/latency dominates marginal predictive gain.
8. **What makes an LLM assistant safe enough to pilot?** Bounded assistive action, source/citation evaluation, permissioned tools, injection tests, human approval for consequential effects, and explicit escalation.
9. **What is a valid rollback?** Restoring a compatible code/model/prompt/index/policy/configuration bundle and proving a known trace behaves as expected.
10. **How do you communicate uncertainty?** State evidence level, population/window, untested assumption, competing explanation, and next smallest test that can change the recommendation.

**Takeaway:** Interview answers should expose decision logic and counterfactuals, not memorized architecture names.
### Answered interview pack

#### I-01 — Frame the first minute

**Checked answer:** ask for owner, decision unit, action, deadline, baseline, and irreversible harm.
**Reasoning:** these variables determine whether a classifier, queue, retrieval path, or manual process is even relevant.
**Follow-up:** if the owner is unavailable, state a provisional assumption and identify the smallest question that could reverse it.
**Acceptable alternative:** begin with the user-facing failure if it concretely exposes the decision and constraint.
**Red flag:** selecting a database or model before naming the action.
**Scoring:** full credit requires a decision-changing clarification, not a generic “requirements gathering” claim.
**Answer support:** Sections 2, 3, and IDF-01.

#### I-02 — Defend the threshold arithmetic

**Checked answer:** at `p=.20`, direct expected value is `$3.60`, but action still needs calibration, capacity, and harm checks.
**Reasoning:** positive expected direct value is necessary under the stated loss model, not sufficient for a safe policy.
**Follow-up:** raise false-positive cost to `$20`; the break-even probability becomes `0.2857` and the action no longer clears it.
**Acceptable alternative:** express the same comparison as expected loss minimization with signs reversed.
**Red flag:** treating `.20` as a threshold without defining probability calibration.
**Scoring:** full credit includes units, the arithmetic, and a changed-constraint consequence.
**Answer support:** IDF-01 and Section 5.

#### I-03 — Explain a metric disagreement

**Checked answer:** first verify metric numerator, denominator, population, window, comparator, and policy path.
**Reasoning:** two correct calculations can disagree because one reports scored requests while the other reports completed actions.
**Follow-up:** if the population denominator is missing, pause the conclusion rather than averaging percentages.
**Acceptable alternative:** reconstruct a small row-level contingency table.
**Red flag:** declaring one dashboard wrong without checking its eligibility definition.
**Scoring:** full credit identifies a concrete denominator mismatch and an audit step.
**Answer support:** IDF-04 through IDF-06.

#### I-04 — Select a baseline

**Checked answer:** retain the existing rule or manual workflow under the same action population and deadline.
**Reasoning:** a model compared with “nothing” can look good while being worse than the operational process it replaces.
**Follow-up:** if the baseline has no logs, propose a bounded observation period rather than fabricate a comparison.
**Acceptable alternative:** shadow the candidate policy while actions remain under baseline control.
**Red flag:** comparing offline score to an unrelated historical business metric.
**Scoring:** full credit distinguishes a comparator from a null hypothesis.
**Answer support:** Sections 1, 5, and 7.

#### I-05 — Diagnose an ambiguous overload

**Checked answer:** separate hot-key skew, downstream saturation, and retry amplification with per-key, dependency, and attempt telemetry.
**Reasoning:** aggregate arrivals and worker counts cannot identify which owned boundary limits service.
**Follow-up:** if attempts per request ID increase, stop retry amplification before adding capacity.
**Acceptable alternative:** use a controlled replay with retries disabled to isolate demand.
**Red flag:** adding workers as the only intervention.
**Scoring:** full credit names competing causes and a signal that separates them.
**Answer support:** IDF-02 and FD-02, FD-05, FD-08.

#### I-06 — Handle unavailable labels

**Checked answer:** use the strongest safe baseline, measure proxies as proxies, and defer automated consequential action until outcome maturity is defined.
**Reasoning:** no clever split can recover a label that does not exist at decision time.
**Follow-up:** if delayed labels arrive after 30 days, design a 30-day maturity window and avoid treating recent negatives as true negatives.
**Acceptable alternative:** use human adjudication for a bounded audit sample with stated selection bias.
**Red flag:** training on future outcomes and calling the result real-time validation.
**Scoring:** full credit states the censoring boundary and a non-fabricated validation plan.
**Answer support:** Sections 3, 5, and FD-01.

#### I-07 — Defend an LLM assistant boundary

**Checked answer:** keep the first release assistive, cite sources at claim level, enforce permissions, and require approval for consequential side effects.
**Reasoning:** fluent text is neither authorization nor evidence that a tool action is safe.
**Follow-up:** when a source is missing, the appropriate behavior is abstention or escalation, not a plausible completion.
**Acceptable alternative:** constrain output to quoted extracts plus a deterministic workflow.
**Red flag:** claiming retrieval eliminates hallucination or injection.
**Scoring:** full credit includes groundedness evaluation and an action boundary.
**Answer support:** Section 7 and FD-04.

#### I-08 — Size a recovery path

**Checked answer:** calculate overload backlog, then use recovery surplus rather than total service rate to calculate drain.
**Reasoning:** ongoing arrivals continue consuming capacity during recovery.
**Follow-up:** with 12,000 queued, 150/s recovery service, and 120/s arrivals, drain is 400 seconds.
**Acceptable alternative:** use a discrete event replay when rates or retries are nonconstant.
**Red flag:** dividing backlog by 150/s and reporting 80 seconds.
**Scoring:** full credit states the `mu_r > lambda_r` condition.
**Answer support:** IDF-02 and Section 9.

#### I-09 — Define a rollback

**Checked answer:** restore a compatible code, model, feature, index, prompt, policy, configuration, and state-manifest bundle.
**Reasoning:** behavior is composed from these artifacts, so code-only rollback can preserve the defect.
**Follow-up:** test a known trace after rollback and compare the decision ID, not merely process health.
**Acceptable alternative:** route to a separately versioned stable service if compatibility is explicit.
**Red flag:** “redeploy the old container” with no artifact ledger.
**Scoring:** full credit supplies a behavioral oracle and recovery boundary.
**Answer support:** IDF-03, IDF-08, and FD-06.

#### I-10 — Communicate uncertainty to a nontechnical owner

**Checked answer:** state the recommendation, evidence label, applicable population/window, uncertainty, and reversible next test.
**Reasoning:** uncertainty is decision-relevant only when tied to a choice that could change.
**Follow-up:** if cost estimates are unknown, separate known workload counts from unpriced components.
**Acceptable alternative:** present a decision table with “act, defer, collect evidence” rows.
**Red flag:** hiding uncertainty behind a confidence-looking decimal.
**Scoring:** full credit distinguishes illustrative arithmetic from measured outcome.
**Answer support:** Sections 1, 5, and 9.

#### I-11 — Respond to a changed latency requirement

**Checked answer:** recompose the trace budget, identify the stage causing the breach, then trade freshness, fidelity, or action scope explicitly.
**Reasoning:** adding a synchronous 25-ms retrieval step changes the 60-ms trace to 85 ms, beyond a 75-ms target.
**Follow-up:** propose cached retrieval only if its staleness contract remains valid for the decision.
**Acceptable alternative:** make the additional work asynchronous if the user action truly does not wait.
**Red flag:** calling a blocking dependency asynchronous to preserve the target on paper.
**Scoring:** full credit gives arithmetic plus a contract consequence.
**Answer support:** IDF-09.

#### I-12 — Assess a practice-exam score

**Checked answer:** report 75% overall and 81.82% among answered items, then remediate gaps by evidence rather than select the flattering denominator.
**Reasoning:** five non-answers represent missing demonstrated capability on the all-item definition.
**Follow-up:** have reviewers validate answerability and keys before ranking a learner by weak domains.
**Acceptable alternative:** score a confidence-weighted rubric only after defining how it handles non-answers.
**Red flag:** presenting answered-only accuracy as general readiness.
**Scoring:** full credit recomputes both denominators and states the limitation.
**Answer support:** IDF-04 through IDF-06 and FD-09.

### Practice-exam answerability ledger

The following compact prompts are an ILLUSTRATIVE 60-question coverage ledger.
They are not an official examination, source answer key, or evidence of certification readiness.
Each item identifies the decision skill an evaluator should be able to inspect.

| ID | Prompt | Checked answer / reasoning | Changed constraint | Red flag |
|---|---|---|---|---|
| P01 | Name the decision owner. | The accountable role can accept, defer, or reject the proposed action. | Owner lacks authority: escalate to the actual decision maker. | Calling the engineering team the owner by default. |
| P02 | Name the decision unit. | Specify request, user, item, or event and its key. | One session has many actions: define session/action linkage. | Equating a training row with an action. |
| P03 | Name the baseline. | Preserve the current rule or manual flow under the same scope. | No baseline logs: observe before replacing. | Comparing to no process. |
| P04 | Define the primary metric. | Give numerator, denominator, population, window, comparator, and unit. | Denominator changes: do not compare directly. | Saying “accuracy improved.” |
| P05 | Add a guardrail. | Block rollout on a harm, delay, or coverage boundary. | Guardrail conflicts with objective: owner resolves value tradeoff. | Treating guardrail as a dashboard decoration. |
| P06 | Identify prediction time. | Use the last legitimate feature timestamp before action. | Event arrives late: lag or exclude it. | Using label-time fields. |
| P07 | State an evidence label. | Mark constructed inputs ILLUSTRATIVE and results not measured. | Source becomes available: review precise claim. | Calling a proposal observed. |
| P08 | Choose rule versus ML. | Prefer rule when policy stable and auditability dominates. | Exceptions grow: test a bounded ML alternative. | Assuming ML is intrinsically superior. |
| P09 | Choose batch versus online. | Use batch if action tolerates freshness delay. | Deadline tightens: evaluate online state contract. | Ignoring feature freshness. |
| P10 | Define a fallback. | Return safe no-action, manual review, or cached result by risk. | Fallback itself is harmful: defer release. | Silent failure. |
| P11 | Compute expected value. | `0.2*50-0.8*8=$3.60`. | Cost rises to $20: do not act at .20. | Omitting units. |
| P12 | Find break-even probability. | `8/(50+8)=.1379`. | Benefit changes: recompute threshold. | Using AUC as threshold. |
| P13 | Explain calibration. | A .20 score should correspond to roughly .20 event frequency in scope. | Slice differs: calibrate or limit scope. | Assuming rank implies calibration. |
| P14 | Check review capacity. | Admission must not exceed `K` reviews per horizon. | Handling time varies: use constrained minutes. | Reviewing all positives by default. |
| P15 | Compute backlog. | `(120-100)*600=12,000` requests. | Retries occur: fluid model is insufficient. | Calling it p99 latency. |
| P16 | Compute drain time. | `12,000/(150-120)=400 s`. | Recovery equals arrivals: backlog cannot drain. | Dividing by gross service rate. |
| P17 | Compute payload storage. | `12,000*2 KiB=23.4375 MiB` payload-only. | Replication enabled: add measured overhead. | Calling lower bound total storage. |
| P18 | Allocate latency. | Sum same-trace stages, here 60 ms. | Add 25-ms blocking call: 85 ms breaches 75 ms. | Summing unrelated p99s as proof. |
| P19 | Compute error allowance. | `2m*(1-.999)=2,000` eligible failures. | Traffic 10x: allowance 10x. | Claiming availability equals correctness. |
| P20 | Define idempotency. | Same request key returns one committed decision. | Key expires: state horizon and duplicate risk. | Retrying non-idempotent action blindly. |
| P21 | Detect leakage. | Audit earliest feature availability against action time. | Feature delayed: exclude or lag. | Random split as proof. |
| P22 | Detect distribution shift. | Compare input, score, action, and outcome slices over time. | Labels delayed: distinguish monitoring from truth. | Declaring drift from one aggregate. |
| P23 | Evaluate retrieval. | Check source recall, claim support, abstention, and permissions. | Corpus updates: version index and rerun set. | Judging by fluency. |
| P24 | Handle injection. | Treat retrieved text as untrusted data, not authority. | Tool is high impact: approval boundary tightens. | Letting documents alter tool policy. |
| P25 | Define an oracle. | State expected value, action, or invariant for a fixed input. | Output stochastic: define allowed set or seed. | Using dashboard green status. |
| P26 | Explain rollback. | Restore the compatible release bundle and test a trace. | State migrated: use reversible migration plan. | Rolling back code only. |
| P27 | Diagnose retry storm. | Compare original IDs to attempts and synchronized retries. | Dependency recovers: drain with limits. | Adding workers first. |
| P28 | Diagnose hot key. | Inspect per-key traffic and partition ownership. | Key cannot split: isolate or redesign contract. | Trusting average utilization. |
| P29 | Diagnose stale cache. | Compare source, cache, and decision version/timestamp. | Fresh read too slow: declare stale-safe action. | Purging cache without load plan. |
| P30 | Diagnose queue aging. | Inspect age by priority and service distribution. | Low-priority deadline expires: use documented fallback. | Reporting only queue count. |
| P31 | Explain false-positive harm. | Value includes customer and operational consequence in horizon. | Harm uncertain: collect owner input before threshold. | Treating every error equally. |
| P32 | Explain proxy conflict. | Optimize business action value, with proxy guardrails. | Clicks rise but returns rise: conclusion reverses. | Calling proxy lift success. |
| P33 | Define release gate. | Require named metric, guardrail, oracle, owner, and rollback. | No oracle: release remains proposed. | “Looks good in demo.” |
| P34 | Define shadow mode. | Compute candidate outputs without changing real action. | Shadow lacks outcome link: limit inference. | Calling shadow a production result. |
| P35 | Define canary. | Expose a bounded eligible population with stop conditions. | Slice is unrepresentative: do not generalize. | Gradual rollout without guardrail. |
| P36 | Explain selection bias. | Reviewed cases differ from unreviewed cases by policy. | Policy changes: labels’ meaning changes too. | Training as if labels random. |
| P37 | Explain censoring. | Recent negatives may be immature outcomes. | Maturity doubles: redraw evaluation window. | Treating unknown as negative. |
| P38 | Explain schema evolution. | Version input contract and define default/reject behavior. | Required field absent: safe fallback. | Parsing silently to null. |
| P39 | Explain state ownership. | One component owns mutation and reconciliation authority. | Multi-writer required: define conflict semantics. | Shared mutable cache as authority. |
| P40 | Explain consistency boundary. | State what a reader may observe during refresh. | Partial outage: state degraded guarantee. | Saying “eventual consistency” without meaning. |
| P41 | Compare exact and approximate search. | Approximation trades recall/resource for stated workload. | Recall floor binding: use exact or tune. | Claiming approximate is always faster. |
| P42 | Compare synchronous/asynchronous. | Synchronous blocks decision; asynchronous requires later contract. | User waits anyway: it is synchronous in effect. | Hiding critical work in queue. |
| P43 | Compare build/buy. | Compare contract, evidence, control, and exit cost. | Vendor terms change: re-evaluate dependency. | Pricing-only selection. |
| P44 | Estimate cost honestly. | State workload, unit price source/date, exclusions, uncertainty. | Price unavailable: mark unpriced. | Inventing a precision total. |
| P45 | Prioritize observability. | Instrument decision, versions, latency, errors, outcomes, guardrails. | Privacy limits logs: minimize and aggregate. | Logging raw sensitive data by default. |
| P46 | Define data retention. | Retain only justified fields/duration with owner and deletion path. | Audit needs history: scope protected ledger. | Infinite retention for convenience. |
| P47 | Handle permission failure. | Deny by default and record decision context. | Owner grants exception: version authorization. | Model-generated authorization. |
| P48 | Handle dependency timeout. | Bound wait, retry ownership, fallback, and user-visible behavior. | Dependency critical: fail safely. | Infinite retries. |
| P49 | Handle partial write. | Use transaction/outbox or reconciliation invariant. | Sink nontransactional: detect/repair duplicates. | Assuming at-least-once is exactly-once. |
| P50 | Handle model version drift. | Bind decision trace to model and feature versions. | Model unavailable: fallback policy. | Comparing scores across models blindly. |
| P51 | Audit an answer key. | Check answerability, rationale, alternatives, and reviewer agreement. | Prompt ambiguous: revise or score alternatives. | One author’s assertion. |
| P52 | Interpret 75% score. | It is 45 of 60 items under equal binary weights. | Items unequal: report weighted scheme. | Calling it comprehensive readiness. |
| P53 | Interpret 81.82% score. | It is 45 of 55 attempted items, excluding five omissions. | Omissions consequential: use 75% overall. | Hiding the denominator. |
| P54 | Plan remediation. | Target demonstrated gap with checked rationale and re-test. | Key unverified: validate before remediation. | Teaching to a possibly wrong key. |
| P55 | Explain uncertainty. | Name evidence, scope, alternative explanation, reversal test. | Decision urgent: choose reversible safe action. | Vague “it depends.” |
| P56 | Explain architecture diagram. | Label inputs, owners, state, control path, failure/recovery path. | Shared state: show consistency contract. | Boxes and arrows without semantics. |
| P57 | Explain quality incident. | Separate observed symptom, hypothesis, test, correction, verification. | Evidence missing: label unknown. | Retrospective certainty. |
| P58 | Explain ethical boundary. | Identify affected population, harm, approval, and appeal path. | Harm high: automation may be disallowed. | Treating fairness as a metric only. |
| P59 | Close an interview answer. | Make recommendation, condition, risk, and next evidence explicit. | Evidence reverses: state alternative. | Ending with tool list. |
| P60 | Reject false precision. | Prefer bounded estimate and test protocol to invented benchmark. | Stakeholder requests number: give assumptions/range. | Unsupported decimal claim. |
### Design-review rehearsal traces

#### R-01 — Moderation queue

Prompt: flag potentially harmful posts for review during a live event.
Owner: the trust-and-safety operations lead, not the model-training team.
Unit: one post revision, keyed by post ID and revision ID.
Action: review, temporary hold, or no action.
Baseline: a documented keyword rule plus manual reports.
Objective: reduce unreviewed harmful exposure per eligible view.
Guardrail: do not hold posts without a review deadline and appeal path.
Constraint: reviewer capacity is finite and event traffic is bursty.
Evidence: this is an ILLUSTRATIVE design exercise; no moderation rate is asserted.
Baseline test: replay timestamped content using only fields available at posting.
Wrong method: random splitting revisions can place later reports beside earlier variants.
Diagnostic: compare revision and label timestamps for each joined row.
Fallback: when review is saturated, use the least harmful declared policy rather than an invisible queue.
Oracle: a duplicate revision request returns its existing review decision, not a second queue item.
Reversal trigger: if safe fallback cannot be specified, automation remains proposed.

#### R-02 — Enterprise knowledge answer

Prompt: answer an employee question from a permissioned policy corpus.
Owner: the knowledge-policy owner accepts the answer contract.
Unit: one authenticated question and corpus version.
Action: answer with support, quote sources, or abstain/escalate.
Baseline: search results and a policy-owner help channel.
Objective: increase supported task completion, not merely fluent response.
Guardrail: zero unsupported consequential policy assertions in the evaluation set.
Constraint: source permissions and corpus freshness are binding.
Evidence: ILLUSTRATIVE architecture; source availability is unknown.
Baseline test: compare user task completion with cited search results.
Wrong method: treat top-k retrieval as proof that a generated claim is supported.
Diagnostic: require a reviewer to locate each claim in the cited span.
Fallback: return excerpts and route to the policy owner.
Oracle: a denied document never appears in retrieved context or final output.
Reversal trigger: citation audit failure blocks broader action scope.

#### R-03 — Payment-risk review

Prompt: prioritize attempted purchases for manual risk review.
Owner: the payments-risk manager chooses review policy.
Unit: one authorization attempt at authorization timestamp.
Action: review or pass; automatic decline is outside this exercise.
Baseline: fixed rule based on known risk conditions.
Objective: estimated avoided loss net of false-review harm.
Guardrail: review delay and false-positive impact by eligible slice.
Constraint: 100 decisions per hour is the constructed capacity limit.
Evidence: ILLUSTRATIVE values only; no fraud prevalence is measured.
Baseline test: chronological replay with a frozen rule version.
Wrong method: maximize recall without an admission policy.
Diagnostic: inspect candidate count, queue age, and value at the selected threshold.
Fallback: defer low-confidence candidates when capacity is consumed.
Oracle: P=0.20, benefit=$50, cost=$8 gives $3.60 direct value.
Reversal trigger: false-positive cost of $20 makes that score ineligible.

#### R-04 — Search-index refresh

Prompt: publish a changed product record while preserving query behavior.
Owner: the search serving lead owns freshness semantics.
Unit: one document version and one query snapshot.
Action: serve old version, new version, or explicitly unavailable result.
Baseline: periodic batch index publication.
Objective: meet declared freshness without exposing malformed documents.
Guardrail: query result must identify index generation and document version.
Constraint: shard publication can fail independently.
Evidence: DERIVED correctness contract; no service benchmark is claimed.
Baseline test: compare small corpus results against a reference implementation.
Wrong method: publish every shard as soon as it finishes.
Diagnostic: issue the same query across generations during partial publication.
Fallback: pin queries to the last coherent generation.
Oracle: every result in a snapshot reports one generation identifier.
Reversal trigger: inability to maintain coherence requires a different publication protocol.

#### R-05 — Feature-store outage

Prompt: preserve a customer decision when an online feature dependency times out.
Owner: the product risk owner approves degraded action.
Unit: one decision request and feature snapshot timestamp.
Action: score, use a bounded fallback, or defer.
Baseline: existing deterministic policy.
Objective: preserve safe completion under dependency loss.
Guardrail: no fallback may use feature data older than its stated decision validity.
Constraint: dependency timeout is part of user-visible latency.
Evidence: ILLUSTRATIVE failure exercise; no production outage is reported.
Baseline test: inject a timeout while retaining the same request trace.
Wrong method: retry until the user-facing deadline expires.
Diagnostic: compare original request ID, attempt count, and dependency wait time.
Fallback: apply policy that is valid without the absent feature.
Oracle: retry of the request has one decision ID and bounded attempts.
Reversal trigger: if no safe fallback exists, fail closed or route to manual handling.

#### R-06 — Model bundle rollback

Prompt: undo a harmful policy rollout without changing unrelated decisions.
Owner: the release manager controls rollback authority.
Unit: one decision trace with code, model, feature, and policy versions.
Action: restore a known compatible bundle.
Baseline: prior release manifest with recorded trace oracle.
Objective: recover declared behavior quickly and audibly.
Guardrail: do not mix old code with incompatible state or policy.
Constraint: cache and asynchronous work may outlive deployment replacement.
Evidence: DERIVED release invariant; not a claim about a named platform.
Baseline test: run the frozen trace through the baseline manifest.
Wrong method: redeploy an old image and infer success from process health.
Diagnostic: compare all artifact hashes attached to the decision.
Fallback: stop consequential action while manifest compatibility is unknown.
Oracle: R7 returns D91/REVIEW after retry and rollback.
Reversal trigger: state migration incompatibility requires forward repair instead.

#### R-07 — Practice-exam remediation

Prompt: decide which topic a learner should revisit after a 60-question exercise.
Owner: the learner and curriculum reviewer jointly own remediation selection.
Unit: one item response, rationale, and answer-key version.
Action: practice a gap, review an ambiguous item, or leave a demonstrated skill.
Baseline: unstructured repetition of all questions.
Objective: improve supported reasoning, not inflate a denominator.
Guardrail: do not classify a key disagreement as learner weakness.
Constraint: item weights and scoring rules must be explicit.
Evidence: ILLUSTRATIVE arithmetic; official blueprint is unavailable.
Baseline test: compute 45/60 and 45/55 with denominators shown.
Wrong method: publish only the 81.82% answered-only score.
Diagnostic: inspect unanswered count and independent reviewer rationale.
Fallback: exclude disputed items from remediation ranking pending key review.
Oracle: converting five omissions to wrong leaves overall score at 75%.
Reversal trigger: an unanswerable prompt must be revised before re-use.

#### R-08 — Cache admission

Prompt: decide which expensive responses may enter a bounded cache.
Owner: the serving owner sets memory and freshness budget.
Unit: one cache key, version, payload size, and expiry rule.
Action: cache, bypass, or evict according to declared policy.
Baseline: no cache or a simple TTL cache.
Objective: reduce valid repeated work without serving unsafe stale state.
Guardrail: never cache a response across an authorization boundary.
Constraint: cache memory and invalidation traffic are finite.
Evidence: ILLUSTRATIVE selection reasoning; no hit rate is claimed.
Baseline test: replay fixed keys against source-of-truth responses.
Wrong method: use observed frequency without checking result volatility.
Diagnostic: compare stale-serving incidents by key class and expiry age.
Fallback: bypass cache for volatile or permission-sensitive keys.
Oracle: cache key includes the versioned authorization scope.
Reversal trigger: invalidation correctness failure removes that class from cache.

#### R-09 — Capacity promise

Prompt: decide whether a 75-ms target is plausible for a request path.
Owner: the service owner accepts the latency contract.
Unit: one end-to-end user-visible trace.
Action: accept design, simplify path, or negotiate target.
Baseline: current serial trace components.
Objective: meet the target on the declared workload and percentile plan.
Guardrail: do not label a single trace a tail-latency result.
Constraint: feature, score, policy, and write stages are user blocking.
Evidence: DERIVED 60-ms trace calculation; not a measured benchmark.
Baseline test: add identical stage durations for the same request.
Wrong method: hide a 25-ms retrieval call under “asynchronous.”
Diagnostic: inspect whether action waits for the retrieval completion.
Fallback: use an explicitly stale-safe precomputed value if valid.
Oracle: 8+22+15+5+10=60 ms; adding 25 makes 85 ms.
Reversal trigger: no valid degradation means target or scope must change.

#### R-10 — Incident narrative

Prompt: explain why an alerting system acted twice on one input.
Owner: the incident commander owns mitigation and evidence record.
Unit: one idempotency key and resulting external action.
Action: deduplicate, reconcile, and communicate affected scope.
Baseline: durable action ledger with transactional outbox.
Objective: restore at-most-one committed action under retries.
Guardrail: do not delete evidence to make the duplicate disappear.
Constraint: reply loss can cause callers to retry a committed request.
Evidence: DERIVED state trace; no actual incident is asserted.
Baseline test: replay lost reply after durable commit.
Wrong method: count HTTP 200 responses instead of committed decision IDs.
Diagnostic: group actions by original request key and ledger decision ID.
Fallback: suspend repeated side effect while reconciliation is incomplete.
Oracle: R7 maps to one D91 despite two delivery attempts.
Reversal trigger: absent durable key requires manual scope reconstruction.
### Review principles with counterexamples

**Principle 1 — A named user is not a decision owner.**
A user supplies context or receives impact.
The owner accepts trade-offs and escalation.
Counterexample: an analyst requests a ranking, but a compliance lead owns the action threshold.
Ask who may authorize a false-positive cost.
Ask who may stop the system during harm.
Record that role beside the metric contract.
This prevents an architecture answer from hiding governance uncertainty.

**Principle 2 — A score is not an action.**
A score is an intermediate value with model and calibration scope.
An action also needs policy, capacity, permission, and timing.
Counterexample: two equal scores receive different outcomes because one review queue is full.
Show the admission rule rather than claiming threshold alone decides.
Log the applied policy version with the score.
This makes changed-capacity behavior explainable.

**Principle 3 — A baseline is not necessarily primitive.**
Existing manual work can encode hard-won exceptions and safety checks.
Counterexample: a keyword rule plus human appeal may outperform an unvalidated model on severe cases.
Measure the baseline under the proposed population and deadline.
Keep it available as a fallback through early evaluation.
Do not compare a new model with an imagined empty process.
This protects against false claims of automation value.

**Principle 4 — A metric requires an eligibility rule.**
The numerator is meaningless without the candidates that could enter it.
Counterexample: supported-answer rate rises when difficult permissioned questions are silently excluded.
Define inclusion, exclusion, and treatment of unknown outcomes.
Report coverage beside quality.
Investigate denominator movement before celebrating a rate change.
This distinguishes improvement from population filtering.

**Principle 5 — A guardrail has a consequence.**
A monitored metric without a stop, escalation, or policy response is descriptive only.
Counterexample: subgroup delay is visible but no owner can pause rollout.
Set the threshold, measurement window, owner, and action before release.
Distinguish alerting uncertainty from permission to ignore harm.
Test the alert route with a controlled scenario.
This turns ethics and reliability language into operational behavior.

**Principle 6 — Time is data lineage.**
Availability time, event time, processing time, and label time answer different questions.
Counterexample: yesterday’s aggregate may be statistically useful but unavailable at authorization.
Attach the earliest legal availability timestamp to each feature.
Test joins against prediction time, not merely matching entity keys.
Preserve clock/time-zone conventions in the contract.
This blocks a common path from offline success to release failure.

**Principle 7 — A queue has a decision policy.**
Queue length alone does not say who waits, expires, or receives fallback.
Counterexample: priority work consumes all capacity while low-risk users age indefinitely.
Specify admission, ordering, deadline, cancellation, and overflow semantics.
Measure age distributions by meaningful class.
Explain whether retries re-enter at the same priority.
This connects a sizing calculation to user impact.

**Principle 8 — Degradation is a product choice.**
Fallback changes what the user can do and what evidence is retained.
Counterexample: serving a stale authorization result is faster but may violate access rules.
Name the dependency that failed and the exact alternate action.
State freshness, safety, and observability limits of the fallback.
Require owner approval for consequential degradation.
This avoids treating “graceful” as an unexamined adjective.

**Principle 9 — Recovery needs a completion oracle.**
Process restart and green health checks do not prove behavior recovered.
Counterexample: a worker is healthy while its index points to a stale generation.
Choose a fixed trace, invariant, or reconciliation count.
Run it after rollback, replay, or failover.
Record what the oracle cannot establish.
This prevents confidence from being inferred from infrastructure status.

**Principle 10 — Alternatives must be comparably described.**
“Considered” differs from “tested,” and both differ from “rejected by evidence.”
Counterexample: an LLM option is named fashionable but never evaluated against a search baseline.
For each option state workload, objective, strength, disqualifier, and evidence state.
Give the simplest credible option its strongest case.
Name the trigger that would reopen the selection.
This makes trade-off communication falsifiable.

**Principle 11 — Cost has a work boundary.**
A token, request, reviewer-minute, and storage byte are not interchangeable cost units.
Counterexample: a cheap model call drives expensive manual review and failed retries.
State volume, peak factor, retry rate, retention, and ownership of each component.
Mark unavailable prices or workload measurements as unknown.
Avoid adding incompatible components into a fictional exact total.
This yields estimates that can be audited and replaced.

**Principle 12 — A practice question needs an answer contract.**
An evaluator needs rationale, acceptable alternatives, and a disqualifying boundary.
Counterexample: “use a cache” can be correct or unsafe depending on authorization and freshness.
State the presumed facts before scoring an answer.
Let changed constraints yield a changed accepted answer.
Audit reviewer disagreement as a prompt/key problem when appropriate.
This measures reasoning rather than recall of preferred tools.

**Principle 13 — Evidence labels constrain verbs.**
Illustrative inputs support “would” and “estimate,” not “improved” or “deployed.”
Counterexample: a checked equation becomes mislabeled as an observed operational outcome.
Put the label next to the claim it qualifies.
State the artifact that would upgrade the label.
Retain unavailable-source limitations rather than reconstructing evidence.
This preserves useful teaching without invented authority.

**Principle 14 — The closing recommendation must be reversible when evidence is weak.**
Weak evidence does not force indecision, but it changes the safe action.
Counterexample: a bounded shadow replay may be appropriate while automatic enforcement is not.
Name the action, evidence boundary, failure boundary, and next measurement.
State what result changes the choice.
Avoid a conclusion that merely repeats the architecture.
This makes uncertainty a decision input rather than an apology.
### Final pre-answer trace

Write one sentence naming the decision owner.
Write one sentence naming the action and decision unit.
Write one sentence naming the baseline under the same scope.
Write one metric with its numerator and denominator.
Write one guardrail with a response owner.
State the prediction-time boundary for every influential input.
Choose the simplest credible alternative before the advanced design.
Calculate one quantity using declared units and assumptions.
Name a capacity, latency, or cost boundary that can reject the design.
Describe one failed request and its safe recovery behavior.
Attach an oracle to the release, rollback, or replay claim.
Label illustrative, derived, sourced, and not-measured claims accurately.
Name the strongest competing explanation for the proposed result.
State a test that distinguishes that explanation.
Give a changed constraint that reverses or narrows the recommendation.
End with the next evidence required, not a technology slogan.
This sequence is a review aid, not a substitute for domain expertise.
An answer missing one item may still be useful, but its unexamined boundary should be explicit.
The final card condenses the sequence without replacing the detailed reasoning above.


## 12. Quick Reference Card

| Cue | Rule | Use when | Critical assumption | Avoid when |
|---|---|---|---|---|
| Start | name owner, unit, action, deadline, baseline | every prompt | decision exists | skipping ambiguity |
| Choose metric | pair objective with guardrail and denominator | evaluating | population/window match | proxy is treated as value |
| Threshold | use expected action value and capacity | scored actions | probabilities/costs are scoped | AUC chooses threshold |
| Size | `B=max(0,lambda-mu)*t` | constant-rate feasibility | same work boundary | p99/retry claim |
| Audit | record input/state/model/policy/time/outcome | consequential action | versions are stable | score-only trace |
| Recover | restore complete compatible bundle | rollout/incident | manifest is complete | code-only rollback |
| Close | state evidence and reversal trigger | recommendation | uncertainty is visible | fictitious benchmark |
