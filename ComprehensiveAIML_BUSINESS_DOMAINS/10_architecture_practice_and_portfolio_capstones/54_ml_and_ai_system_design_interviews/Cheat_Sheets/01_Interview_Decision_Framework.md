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
