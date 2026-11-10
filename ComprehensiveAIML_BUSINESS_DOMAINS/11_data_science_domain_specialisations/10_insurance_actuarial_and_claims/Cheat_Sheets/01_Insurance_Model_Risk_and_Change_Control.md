---
collection: "insurance-actuarial-and-claims"
file_id: "01"
topic: "Insurance model risk and change control"
level: 2
status: "draft"
version: "2.0"
created: "2026-11-09"
reviewed: null
review_trigger: "Change in intended use, decision policy, source system, model artifact, validation evidence, or applicable governance requirement"
notation_style: "plain-text"
prerequisites: ["Basic probability", "Versioned artifacts", "Decision logs"]
pattern_tags: ["model-risk", "change-control", "monitoring", "rollback"]
source_status: "OPEN"
owner_role: "Model owner with independent validation and accountable business owner"
source_index: "INDEX.md"
---

# Insurance Model Risk and Change Control

## 1. File Header
> **Purpose:** Decide whether an insurance decision-support model and its proposed change have enough bounded evidence to promote, monitor, recover, or stop.
> **Prerequisites:** Basic probability, a distinction between a score and an action, and versioned data/software concepts.
> **Key outcome:** Given a model use, change set, and delayed outcome, specify the compatible release record, materiality decision, monitor, owner, and rollback/reconciliation path.
> **This file does NOT cover:** actuarial approval of a rate, jurisdiction-specific regulatory interpretation, or proof that any insurer uses these controls.
> **Level:** 2 — Core methods.
> **Version / reviewed:** 2.0 / not independently reviewed.
> **Evidence:** DERIVED control logic and ILLUSTRATIVE/VERIFIED arithmetic; visible source filename only; no original source body was accessed.
> **Retrieval cue:** If a release is described only by a model version, start at Section 4.

A model can rank claims, estimate expected loss, route documents, or flag exceptions. None of those descriptions identifies the allowed decision, affected population, action owner, evidence threshold, or recovery obligation. “Model risk” here means risk that a model-supported decision is wrong, unavailable, unauditable, applied outside its stated use, or operationally unrecoverable.

The visible source inventory names `11_Model_Risk_Governance_Explainability.md`; it does not establish the original lesson's contents. This sheet therefore teaches a general, illustrative control method rather than reconstructing a source or asserting a compliance framework.

**Takeaway:** A model name is not an intended-use statement and a deployment is not an approval record.

## 2. Concept Map / Mental Model
```text
Business decision and owner
        |
        v
Intended-use contract ----fails----> refuse, narrow, or collect evidence
        |
        v
Compatible evidence bundle
(data definition, feature code, model, threshold, policy, schema)
        |
        +----change assessment----> immaterial / material / emergency containment
        |                                  |                 |
        v                                  v                 v
Controlled release -> action logs -> monitors -> pause/rollback -> reconcile
        |                     |                 |
        +---------------------+-----------------+
                 mature outcomes and review feedback
```

The contract is the control-plane object. The data plane produces a score or recommendation. The decision plane applies a threshold or routes a case. The evidence plane records what was known, what was used, who approved it, and how recovery was verified.

A successful request has an authorized input, a versioned bundle, a deterministic or documented nondeterministic decision, a durable decision record, and an auditable downstream effect. A failed request must be distinguishable from a declined decision: it either performs no side effect, or records an idempotency key and enters a reconcilable recovery state.

**Takeaway:** Treat score production, action authorization, and recovery as separate states.

## 3. Foundational Definitions
| Term | Meaning | Units/type | Nonexample or warning |
|---|---|---|---|
| Intended use | Permitted decision, population, owner, timing, and action boundary | structured contract | “Fraud model” is not sufficient. |
| Model artifact | Versioned fitted parameters or rules | immutable identifier | A repository branch is not necessarily deployed bytes. |
| Feature contract | Fields, semantics, units, availability time, transforms, and missing-value rules | schema + lineage | A column name alone is ambiguous. |
| Policy artifact | Thresholds, overrides, eligibility, escalation, and approval rules | versioned configuration | A score threshold is not a complete policy. |
| Release manifest | Compatible set of artifacts and dependencies | immutable record | A model hash cannot recover a schema or policy. |
| Material change | Change reasonably capable of altering decision risk, evidence, control effectiveness, or scope | classification | File size is not materiality. |
| Champion | Current approved comparator | versioned bundle | Not automatically the best historical offline model. |
| Challenger | Candidate evaluated against declared comparator | versioned bundle | Not safe for production merely because offline score is higher. |
| Override | Authorized human departure from a recommended action | decision event | Silent spreadsheet correction is not an override. |
| Mature outcome | Outcome observable under a predeclared maturation rule | event/status | A same-day proxy is not necessarily truth. |
| Rollback | Restore a known compatible approved bundle or stop the action | recovery action | Reverting weights alone may preserve an incompatible policy. |
| Reconciliation | Account for every affected request and side effect after change/failure | ledger procedure | A green deployment dashboard is not reconciliation. |

Define the unit of decision before reporting a metric: a policy quote, claim, document field, investigation alert, or customer contact has different denominator and harm. Event time records when evidence existed; processing time records when a system handled it; recorded time records when a database learned it. These cannot be substituted.

**Takeaway:** Every metric and control needs a unit, time basis, and decision owner.

## 4. Core Formulas with Full Derivation
### 4.1 Base: decision coverage and release compatibility
Validity before use: a release ledger identifies every eligible decision and the bundle used at action time. Let `n` be eligible decisions and `a` be decisions carrying a complete manifest.

Derivation: coverage is the count satisfying the evidence predicate divided by all eligible decisions.
```text
FORMULA [MR-1]: c = a / n
WHERE: c is manifest coverage [dimensionless]; a is complete-manifest decisions [count]; n is eligible decisions [count].
CONSTRAINT: n > 0; “complete” requires model, feature, schema, policy, and release identifiers.
NUMERICAL EXAMPLE: a=9,950 and n=10,000; c=0.995=99.5%.
INTERPRETATION: fifty decisions cannot be reconstructed from the release ledger.
```
Boundary: 100% identifier presence does not prove the recorded identifiers were compatible or correct.

A compatible bundle is an invariant, not a model-quality formula:
```text
INVARIANT [MR-I1]: action(bundle_id) is permitted only if every required component
is approved for the same intended-use contract and interface compatibility set.
PRECONDITIONS: immutable artifact IDs; compatibility relation; authorization check.
STATE TRANSITION: proposed -> validated -> approved -> active -> retired.
FAILURE BOUNDARY: an emergency stop may move active -> paused, but may not silently
relabel a prior action as if it used the replacement bundle.
```

### 4.2 General: delayed-outcome performance
Validity before use: each evaluated action has had enough time to observe the stated outcome, and denominator exclusions are reported.

For a binary reviewed decision, define `TP`, `FP`, `FN`, and `TN` on the mature labeled population. Precision answers “of investigated alerts, how many were confirmed under this definition?”
```text
FORMULA [MR-2]: precision = TP / (TP + FP)
WHERE: TP and FP are counts of mature reviewed alerts; result is dimensionless.
CONSTRAINT: TP+FP > 0; investigated alerts may be selectively labeled.
NUMERICAL EXAMPLE: TP=18, FP=12; precision=18/30=0.60.
INTERPRETATION: 60% is an evaluated-alert proportion, not the fraud prevalence of all claims.
```

Recall needs a denominator of all actual positives, which is often unavailable promptly in selective review systems.
```text
FORMULA [MR-3]: recall = TP / (TP + FN)
WHERE: FN is mature positive cases not alerted under the stated process.
CONSTRAINT: labels for non-alerted cases must be sufficiently observed or sampled.
NUMERICAL EXAMPLE: TP=18, FN=42; recall=18/60=0.30.
INTERPRETATION: a high precision queue can still miss most defined positives.
```

### 4.3 Variants: drift and capacity
A population stability index is sometimes used as a descriptive shift signal. It is not a proof of performance change.
```text
FORMULA [MR-4]: PSI = sum_b (q_b-p_b) * ln(q_b/p_b)
WHERE: p_b is reference share and q_b current share in bin b; both are positive shares.
CONSTRAINT: identical bins, same population definition, and nonzero handled shares.
NUMERICAL EXAMPLE: p=(0.5,0.5), q=(0.6,0.4): PSI=0.1 ln(1.2)-0.1 ln(0.8)=0.0405.
INTERPRETATION: the input mix changed; it does not identify cause or decision harm.
```

For a review queue with arrival rate `lambda` and service rate `mu`, fluid backlog growth is:
```text
FORMULA [MR-5]: B = max(0, lambda-mu) * t
WHERE: B [cases], lambda and mu [cases/hour], t [hours].
CONSTRAINT: initially empty queue, constant rates, no abandonment or retry loop.
NUMERICAL EXAMPLE: lambda=55, mu=45, t=8; B=80 cases.
INTERPRETATION: a threshold change that adds 10 cases/hour requires capacity, diversion, or slower service.
```

### 4.4 Limits and counterexamples
A monitor can be stable while harm changes in a small or unmeasured slice. A PSI can rise because a benign product mix changes. Precision can rise because reviewers accept only obvious alerts. A rollback can restore scoring while leaving duplicate downstream actions. A complete manifest can record an approved but wrongly scoped intended use.

**Takeaway:** Formulas quantify a declared question; they do not validate the question's population, label, or causal story.

## 5. Worked Numerical Example
**ILLUSTRATIVE / VERIFIED:** A proposed claim-routing bundle changes a threshold and feature transformation. It is not an insurer deployment.

The bundle contains model `M17`, feature transform `F9`, schema `S4`, policy `P12`, and reviewer-routing configuration `R3`. The intended use permits routing an eligible claim to review, not denial or payment authorization. Ten thousand eligible decisions are processed in a week. Nine thousand nine hundred fifty carry all five component IDs; fifty lack `R3` after a logging integration fault.

By [MR-1], coverage is 9,950/10,000 = 0.995. The release must not claim full reconstruction. The immediate containment is to pause promotion, preserve the fifty decision IDs, and determine whether their side effects were only queue placement or included customer messages.

For a mature sampled evaluation, 30 reviewed alerts include 18 confirmed positives and 12 false positives. Precision is 60%. A separate authorized sample finds 42 mature positives that were not alerted, so recall is 30%. Neither number alone chooses the threshold: review capacity, false-positive harm, label selection, and owner policy are required.

The new threshold produces 55 review arrivals per hour while staffed service is 45 per hour for an eight-hour shift. [MR-5] gives an 80-case backlog. If the previous threshold created 40 arrivals per hour, it was stable under this fluid model. The operational decision is therefore not “better precision wins”; it is “do not promote this threshold without added capacity, a diversion policy, or a tested lower-volume alternative.”

Independent checks: `9,950 + 50 = 10,000`; `18 + 12 = 30`; `18 + 42 = 60`; and `(55-45)*8 = 80`. If service falls to 35 cases/hour, backlog becomes 160 cases. If arrivals remain 55 indefinitely, buffering delays rather than resolves the overload.

**Takeaway:** A release can be statistically promising yet operationally disallowed by evidence gaps or queue capacity.

## 6. Visual / Geometric Intuition
```text
                 approved contract
                       |
input -> F9 -> M17 -> P12 -> review / no-review
  |        |      |       |          |
  +--------+------+-------+----------+--> immutable decision ledger
                       |
                  monitor slices
                       |
       pause -> restore compatible bundle -> reconcile effects
```
A score is a coordinate along a ranking axis. Policy divides that axis into action regions, but the boundary is chosen under capacity and harm constraints. Drift moves the incoming distribution; calibration changes the relation between score and outcome; a policy edit moves the action boundary. These are distinct geometric movements and need separate logs.

**Takeaway:** When an action changes, identify whether the input population, model mapping, or policy boundary moved.

## 7. Comparison / Selection Table
| Control option | Use when | Strength | Disqualifier | Fallback |
|---|---|---|---|---|
| Manual approval record | Low-volume, high-consequence change | Explicit accountability | Cannot reconstruct actual deployed artifacts | Immutable manifest plus approval workflow |
| Shadow scoring | New score can run without side effect | Exposes input/schema failures | Shadow lacks true action feedback | Controlled pilot with declared guardrails |
| Canary release | Side effects are reversible and cohorts are observable | Limits blast radius | Cannot safely isolate population | Keep manual baseline |
| Champion/challenger | Mature comparable outcomes exist | Tests against current approved bundle | Different eligibility or label maturity | Time-aligned backtest only |
| Immediate rollback | Compatible prior bundle and side-effect ledger exist | Contains live regression | Irreversible downstream actions | Pause action and reconcile manually |

**Takeaway:** Choose a control by reversibility, observability, and decision harm—not fashion.

## 8. Failure Modes & Edge Cases
### MR-F01 — Version-complete but incompatible bundle
**Category:** CONCEPT_CONFUSION. **Provenance:** HYPOTHETICAL. Trigger: model rollback restores `M16` but leaves schema `S4` built for `M17`. Symptom: requests parse but score/action distribution changes abruptly. Competing causes: model regression, schema coercion, or product mix shift. Diagnostic: replay a retained authorized fixture through both manifests and compare feature vectors before scores. Mitigation: pause side effects; risk is delaying legitimate handling. Correction: restore the last approved compatible manifest. Prevention: enforce manifest compatibility in deployment admission. Verification: fixture, manifest, and action-log hashes match the approved record.

### MR-F02 — Apparent drift caused by delayed data feed
**Category:** VIOLATED_ASSUMPTION. **Provenance:** HYPOTHETICAL. Symptom: missingness and PSI rise together. Competing causes: real population shift, source outage, or field-semantic change. Diagnostic: compare event-time coverage with processing-time latency and upstream source counts. Mitigation: route affected records to baseline/manual handling; risk is capacity overload. Correction: repair feed and backfill only with an explicit replay policy. Prevention: freshness SLOs per field. Verification: event-time completeness returns and action decisions are reconciled.

### MR-F03 — Precision improvement from selective labels
**Category:** CONCEPT_CONFUSION. **Provenance:** HYPOTHETICAL. Symptom: reviewed-alert precision rises after threshold increase. Competing causes: actual ranking gain, narrower easy cases, or changed reviewer standard. Diagnostic: preserve a predeclared random audit of non-alerts and measure label maturation. Mitigation: do not claim global performance gain; risk is slower conclusion. Correction: report selection mechanism and audit-adjusted uncertainty. Prevention: prospective sampling plan. Verification: independent audit cohort has documented inclusion probability.

### MR-F04 — Threshold overwhelms review queue
**Category:** SCALE_FAILURE. **Provenance:** HYPOTHETICAL. Symptom: aging queue and increasing customer delay. Competing causes: threshold change, staff absence, retries, or routing-loop duplicate events. Diagnostic: compare unique idempotency keys, arrivals, completions, and requeues by hour. Mitigation: cap intake or revert threshold; risk is missed useful alerts. Correction: repair duplicate routing and choose capacity-consistent policy. Prevention: pre-release load calculation and queue guardrail. Verification: backlog drains only when service exceeds arrivals.

### MR-F05 — Override log masks policy defect
**Category:** VIOLATED_ASSUMPTION. **Provenance:** HYPOTHETICAL. Symptom: aggregate action rate is stable but overrides cluster in one product. Competing causes: legitimate expert exceptions, policy misconfiguration, or training gap. Diagnostic: compare override reasons, authority, and outcome by product and time. Mitigation: require second review for high-risk overrides; risk is slower service. Correction: amend policy or training after owner review. Prevention: structured reason codes and expiry. Verification: override concentration and correction rate are reassessed.

### MR-F06 — Rollback duplicates an external effect
**Category:** SCALE_FAILURE. **Provenance:** HYPOTHETICAL. Symptom: a replay sends two customer messages. Competing causes: missing idempotency key, retry timeout, or ledger lag. Diagnostic: join outbound provider receipt, durable command ID, and decision ID. Mitigation: stop replay and suppress duplicate sends; risk is unsent legitimate messages. Correction: use one durable idempotency key per intended effect. Prevention: failure-injection replay tests. Verification: one receipt per key across recovery.

**Takeaway:** Every symptom needs competing causes, a discriminating signal, a bounded fix, and a side-effect-aware verification.

## 9. Complexity & Cost Analysis
Let `N` be decisions, `F` features, `V` retained artifact versions, and `K` monitoring slices. Feature validation is commonly O(NF); score logging is O(N); monitoring aggregates are O(NK) if every decision enters every slice. A manifest lookup should be O(1) by immutable identifier. Replay cost is not merely compute: it includes approved input access, reviewer capacity, and potential external side effects.

ILLUSTRATIVE sizing: 10,000 decisions/week with a 2 KiB decision record retain about 20,480,000 bytes, or 19.53 MiB of payload before indexes, encryption overhead, replicas, or attachments. Retaining 52 weekly snapshots gives about 1,015.6 MiB payload. This is arithmetic, not a storage estimate or vendor price.

**Takeaway:** Operational evidence is a first-class workload; its cost must be designed rather than discovered during an incident.

## 10. Key Theorems / Results Reference
| Result | Conditions | Use | Does not establish |
|---|---|---|---|
| Conservation of queue work | Counted arrivals/completions share a unit and window | Backlog calculation | Individual waiting-time distribution |
| Bayes/selection warning | Labels may depend on review policy | Interpret precision/recall | Unbiased prevalence without sampling design |
| Immutable-record principle | Identifier refers to retained content | Reproducibility and audit | Correctness of content |
| Separation of concerns | Score, policy, and action are distinct functions | Diagnose release changes | That each function is independently safe |

**Takeaway:** Use named results as scoped tools, never as blanket assurance.

## 11. Interview / Exam Questions
1. **Why is a model version insufficient for rollback?** The action also depends on feature logic, schema, policy, routing, and dependencies; restore a compatible manifest.
2. **Calculate manifest coverage for 4 missing records out of 800.** `796/800=99.5%`; the four need reconciliation, not deletion.
3. **What is the difference between event and recorded time?** Event time is when evidence existed; recorded time is when the system learned it. Using the latter can leak future information.
4. **A precision metric rises after a threshold increase. What is the first challenge?** Confirm denominator, label maturity, and whether non-alerts are sampled; selective review can create a misleading rise.
5. **When does shadow mode help?** When scoring can run without side effects and input/schema behavior needs observation; it cannot prove action impact.
6. **Why can PSI not prove model degradation?** It describes input distribution differences, which may be benign, harmful, or caused by instrumentation.
7. **Compute backlog at 70 arrivals/hour and 50 completions/hour for three hours.** 60 cases under the stated fluid assumptions.
8. **What must be logged with an override?** Authority, reason, timestamp, relevant evidence, original recommendation, resulting action, and review path.
9. **Give a rollback failure boundary.** If a prior model requires an old feature schema no longer served, weight rollback alone is unsafe.
10. **What changes if a downstream action is irreversible?** Prefer pause/containment and reconciliation; canary and replay require stronger approval and idempotency controls.
11. **How would you compare champion and challenger with delayed outcomes?** Freeze eligibility, use aligned decision-time inputs and maturation windows, retain a policy-consistent comparator, and report missing labels.
12. **What is a plausible wrong method?** Declare a new model safe because offline AUC improved while ignoring threshold, review capacity, and action harm.

**Takeaway:** Strong answers name the unit, time boundary, comparator, owner, and recovery consequence.

## 12. Quick Reference Card
| Field | Rule |
|---|---|
| Intended use | Record decision, population, owner, permitted action, and exclusions. |
| Release unit | Promote model, features, schema, policy, routing, and dependencies as a compatible manifest. |
| Materiality | Assess effect on decision risk, scope, evidence, interfaces, and recovery—not code size. |
| Monitoring | Separate input/data health, score/action distributions, mature outcomes, slices, overrides, and queue health. |
| Delayed labels | Declare maturation and sampling; precision on reviewed alerts is not global performance. |
| Containment | Pause unsafe side effects, preserve ledger, restore compatible bundle only when valid. |
| Reconciliation | Account for every request and external effect by durable decision and idempotency key. |
| Evidence limit | All examples are illustrative; no insurer, regulatory determination, or measured deployment is claimed. |
