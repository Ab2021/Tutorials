---
case_id: CASE-01-01-003
case_title: Account takeover challenge policy with incomplete identity signals
case_version: "1.0"
created: 2026-11-09
last_reviewed: null
status: proposed
content_status: draft
case_type: predictive-ml
domain: Digital account security
primary_pattern: evaluation-mismatch
secondary_patterns: [label-delay, cost-latency-tradeoff]
evidence_level: not-measured
source_basis: illustrative
as_of_date: 2026-11-09
owner_role: Identity-risk product owner
source_index: INDEX.md
review_trigger: Authentication flow, identity-signal consent, attack pattern, response deadline, or recovery-policy change
---
# Case study: Account takeover challenge policy with incomplete identity signals

## 1. The One-Line Framing
Given a login or sensitive-account action, choose allow, step-up challenge, or approved protective hold for the account-security owner, minimizing mature account-takeover loss and unnecessary customer friction under an authentication latency and challenge-capacity constraint, compared with the existing authentication rules.

This is an **ILLUSTRATIVE** proposed design. No named service, account population, user behavior, authentication protocol, model result, or prevention outcome is represented. A score is a routing input, not proof that a person is an attacker.

**Takeaway:** the decision is an action policy under incomplete evidence, not a binary identity claim.

## 2. Context & Background
A login or sensitive event reaches an identity service. Existing authentication and abuse rules may allow it, require an available challenge, or route it for a protected recovery flow. The customer needs timely access; an attacker may use a valid credential, a compromised session, or a manipulated recovery process. A challenge can reduce unauthorized access but can also lock out a legitimate user whose device, location, or network changed.

The security product owner owns policy outcomes. The authentication platform owner owns availability and retry semantics. Customer support owns recovery pathways. Privacy and security reviewers govern permitted signals and retention. The baseline is a versioned rules policy with its documented challenge behavior. No baseline metric is asserted.

The scope is a proposed login-risk decision for one account product. Application onboarding, payment authorization, biometric verification, and downstream transaction fraud are separate decisions. Device identifiers and network data are not presumed reliable, lawful, or stable.

**Takeaway:** identify the actor who owns access, recovery, and customer impact before choosing a model.

## 3. The Business Question
| Field | Decision contract |
|---|---|
| Question | Which eligible login events should be allowed, challenged, or sent to the existing protective recovery route? |
| Owner | Identity-risk product owner with security and support approvals. |
| Trigger | Authentication request; decision must meet the actual service deadline. |
| Actions | Rules-only; allow; supported step-up; approved protective hold/recovery. |
| Objective | Mature unauthorized-access loss and account-compromise incidents per 10,000 eligible events, balanced with successful legitimate access. |
| Constraints | Approved signals only; no unreviewed automated adverse path; protocol/channel capability; latency; accessibility and support capacity. |
| Out of scope | Determining criminal identity or treating failed challenge as proof of takeover. |

The business metric needs two denominators: eligible authentication events for security outcome and legitimate eligible attempts for friction outcome. A challenge completion rate alone may rise because legitimate users abandon, not because risk improved.

**Takeaway:** separate access security, customer access, and service capacity rather than optimizing one proxy.

## 4. Assumptions & Constraints
| ID | Statement | Test | Consequence / fallback |
|---|---|---|---|
| A-01 | Event, receipt, and decision times are retained. | replay ordering audit | exclude fields without availability proof; use baseline rules |
| A-02 | A mature compromise/recovery outcome can be defined. | outcome taxonomy and delay curve | retain unknown state; do not train on silence |
| A-03 | Challenge is available and accessible for the eligible channel. | contract test by channel | do not route unsupported events to challenge |
| A-04 | Risk score path fits measured deadline. | trace p50/p95/p99 end-to-end | use approved rules fallback |
| A-05 | Account recovery has human escalation. | support workflow test | pause protective-hold policy |
| C-01 | No signal is used without approved basis and documented purpose. | field register review | remove signal before replay |
| C-02 | Same event retry returns the same action. | idempotency trace | persist decision before response |

These are unverified proposed controls. The two most material reversals are an unavailable mature outcome and a channel without a safe challenge.

**Takeaway:** incomplete identity evidence is expected; a policy must expose rather than hide its unknowns.

## 5. Data, Tools & Method
### 5a. Data Specification
The unit is one authentication or sensitive-action event keyed by an immutable event ID, account token, event version, and decision time. Candidate input fields include event type, protocol/channel, account tenure band, prior authenticated activity count, prior approved recovery state, device token presence, device-history coverage, coarse network-risk signal where approved, challenge capability, and source freshness. They are candidates only.

The feature contract records source event time and availability time. A future device-account link, support disposition, password reset, or confirmed compromise cannot enter the original decision. Missing device data is neither a new device nor an attack. Shared devices and location changes have legitimate explanations.

Outcomes include confirmed unauthorized account activity under a controlled definition, customer-reported compromise, recovery completion, false-positive appeal, unresolved state, and outcome maturity date. Reporting rates and support access vary; absence of a report is not a negative label. Split by time and account to avoid same-account history leaking into a forward test.

### 5b. Tools & Stack
Proposed components: authentication event log; consent/permission gate; point-in-time feature service; calibrated risk scorer; policy engine; durable decision store; challenge provider adapter; recovery workflow; monitoring and replay store. Versions and products are not specified because none were inspected.

### 5c. Method Summary
Input -> validate schema, eligibility, and capability -> build available features -> rank calibrated risk -> apply versioned action policy -> persist action/reason -> invoke supported challenge or existing route.

| Option | Benefit | Boundary | Status |
|---|---|---|---|
| Existing rules | known operational fallback | may be coarse | comparator |
| Rules plus conservative step-up | explainable and reversible | requires challenge coverage | proposed |
| Calibrated tabular score plus policy | can combine weak signals | needs mature, selectively observed labels | proposed |
| Automatic protective hold at score threshold | may limit damage | high customer harm and recovery dependency | considered, not selected |

A score estimates a historical label under prior actions; it does not estimate what would happen if a different challenge is imposed. Therefore score threshold, action eligibility, and recovery safeguard must be evaluated as a policy.

### 5d. Provenance, quality, exclusions and leakage controls
Maintain field owner, lawful-use approval, source version, timestamp semantics, join key, missingness, transformation, and retention rule. Use immutable snapshots in replay. Fit encoding, imputation, calibration, and feature selection only on development folds. Test event-clock skew, ID collisions, late mobile telemetry, device-token rotation, and duplicate retry records.

### 5e. Reproducibility ledger
| ID | Artifact | Purpose | Evidence boundary |
|---|---|---|---|
| SRC-01 | [NIST Digital Identity Guidelines](https://pages.nist.gov/800-63-4/) | identity terminology reference | not a claim about this scenario |
| SRC-02 | Proposed event contract | point-in-time replay | illustrative and not executed |
| RUN-01 | Proposed policy replay | temporal evaluation | no dataset or run exists |
| POL-01 | Proposed action table | action eligibility and fallback | needs security/support approval |

A real experiment records event snapshot hashes, feature/policy versions, channel eligibility, scorer hash, reason codes, challenge result, recovery path, and maturity snapshot. 

**Takeaway:** log the action contract and availability boundary, not just a score.

## 6. Analysis Walkthrough
No candidate or baseline was run. The following arithmetic is an **ILLUSTRATIVE** fixed-capacity evaluation trace.

Suppose 20,000 eligible events mature into 80 defined compromise outcomes. A challenge budget permits 400 events; 32 challenged events mature positive. Then precision@400 is 32/400=8%, and recall@400 is 32/80=40%. The remaining counts are 368 challenged events without that mature outcome and 48 positives not challenged. This does not establish prevention: a challenge changes the path and reporting.

```text
FORMULA [A-01]: precision@K = TP_K / K
WHERE: TP_K [events] are mature positives in the K challenged events; K [events] is challenge capacity.
CONSTRAINT: same eligibility, outcome definition, and maturity window; K>0.
NUMERICAL EXAMPLE: TP_400=32; K=400; 32/400=0.08=8%.
INTERPRETATION: eight constructed mature positives per 100 challenges, not a prevention rate.
```

```text
FORMULA [A-02]: recall@K = TP_K / P
WHERE: P [events] is all mature positives in the eligible cohort.
CONSTRAINT: unknown outcomes remain unknown; P>0.
NUMERICAL EXAMPLE: TP_400=32; P=80; 32/80=0.40=40%.
INTERPRETATION: this queue covers 40% of constructed mature positives under its historical definition.
```

Assume each challenge consumes 45 seconds of customer-facing interaction and 10% need 8 minutes of support. Support load is 400*0.10*8=320 minutes; this is a workload estimate, excluding retries and recovery. If the support fraction rises to 20%, it becomes 640 minutes. Capacity therefore includes support, not merely challenge-provider throughput.

```text
FORMULA [A-03]: H = K * r * m
WHERE: H [minutes] = incremental support minutes; r [support cases/challenge]; m [minutes/support case].
CONSTRAINT: constant rate and average handling time; queueing/tails excluded.
NUMERICAL EXAMPLE: 400*0.10*8=320 minutes.
INTERPRETATION: policy K must be checked against staffed support time and accessibility effects.
```

Independent arithmetic check: 0.10*400=40 support cases; 40*8=320 minutes. Sensitivity shows a doubling in support rate doubles load. A proposed policy is rejected or restricted if it cannot preserve a recovery path for the added demand.

**Takeaway:** measure security yield and the service load created by the intervention.

## 7. The Finding
There is no evidence to replace existing authentication rules. A future calibrated ranker could be evaluated only as a capability-aware action policy with point-in-time inputs, mature compromise definitions, preserved recovery paths, and explicit support/load guardrails. The illustrative 8% precision@400 and 40% recall@400 are checked arithmetic, not measured effectiveness or prevented compromise. The next decision is whether approved data and controlled evaluation can support the policy.

**Takeaway:** refuse to equate a challenged event with a stopped takeover.

## 8. System/Solution Design
```text
[Authentication event] -> [permission/schema/capability check] --fail--> [existing rules + logged fallback]
       | pass
       v
[point-in-time features] -> [score + calibrated policy] -> [durable event decision]
                                                       |             |
                                                       v             v
                                         [allow / challenge / recovery] [audit + replay]
                                                       |
                                                       v
                                        [challenge response / support escalation]
```

A policy must reject unsupported challenge actions before score ranking. Persist the decision before invoking an external challenge adapter; a retry with the same event ID and snapshot hash returns the same action. A changed event version is explicitly superseded, not treated as a duplicate. The decision log contains fallback/reason codes but must not put protected raw credential material into analyst views.

If feature service, scoring, or policy retrieval fails, use approved existing rules and record component health. If challenge provider is unavailable, do not silently allow or lock all users: execute channel-specific approved fallback. Reconcile event decisions with adapter results and recovery tickets; differences are operational incidents, not model labels.

**Takeaway:** capability and recovery paths precede model choice in an access decision.

## 9. Evaluation & Validation
| Measure | Definition | Comparator | Guardrail | Status |
|---|---|---|---|---|
| precision@K / recall@K | mature outcome counts at fixed K | rules at same eligibility | unknown-label rate | proposed |
| unauthorized-access outcome | defined mature events / eligible assignments | existing policy | reporting delay | not measured |
| successful legitimate access | completed legitimate access / eligible legitimate attempts | baseline | accessibility and abandonment | proposed |
| support load | support cases/minutes per challenged event | baseline | recovery SLA | proposed |
| technical reliability | action completion and fallback rate | baseline | deadline/availability | proposed |

Use chronological account-level splits. Replay baseline and candidate on identical snapshots. Plot maturity by cohort and channel. Shadow mode verifies data/latency but cannot estimate action effect. For a prospective evaluation, pre-specify eligibility, randomization, safety stop, recovery routing, and intent-to-treat analysis; retain all failed challenge/fallback events.

Sensitivity changes the outcome window, challenge budget, support rate, feature delay, and channel eligibility. If a policy is beneficial only after excluding failed challenges or unresolved outcomes, it is inconclusive.

**Takeaway:** test a complete access journey—including recovery—not a detached classifier metric.

## 10. Action Taken
No real action was taken. Proposed sequence: obtain security, privacy, accessibility, and support approval; measure baseline and recovery load; create point-in-time replay; test idempotency and fallback; then decide whether a controlled evaluation is justified.

**Takeaway:** no policy should be treated as live because a diagram or score exists.

## 11. Outcome & Results
Not measured. No performance, latency, compromise, customer-access, or support outcome exists for this scenario. The calculations are ILLUSTRATIVE and hand-derived under stated assumptions.

**Takeaway:** report unavailable evidence as unavailable.

## 12. Failure Modes & What Went Wrong
### F-01 — Recovery outcome leaks into login score
**Symptom:** unusually strong offline model after adding support/recovery fields. **Competing causes:** legitimate useful prior state; field arrived after action; current profile overwrote history. **Diagnostic:** replay availability timestamps at decision cutoff; expected signal is post-decision availability. **Fix:** remove/lag field and rebuild snapshots. **Risk:** rank quality falls. **Verification:** sampled feature ledger proves pre-action availability.

### F-02 — Challenge failure is labeled takeover
**Symptom:** failed challenges inflate positive labels. **Competing causes:** attacker failure; legitimate accessibility issue; provider outage; abandoned flow. **Diagnostic:** compare adapter errors, accessibility path, later recovery, and channel. **Fix:** retain separate challenge-result and outcome states. **Risk:** fewer convenient labels. **Verification:** label audit shows no automatic conversion of challenge failure to compromise.

### F-03 — Device-token missingness becomes a proxy
**Symptom:** missing-device events are challenged disproportionately. **Competing causes:** actual new device; privacy choice; app-version defect; channel mix. **Diagnostic:** stratify missingness by channel/version and mature outcome. **Fix:** explicit missing category, channel policy, and upstream repair. **Risk:** lower security coverage. **Verification:** later cohort shows corrected missingness and guardrail review.

### F-04 — Retry creates contradictory action
**Symptom:** user sees allow then challenge for same event. **Competing causes:** retry; event amendment; race between policy versions; lost acknowledgement. **Diagnostic:** compare event ID, snapshot hash, policy version, and adapter receipt. **Fix:** durable idempotent decision and explicit supersession. **Risk:** overly broad dedupe masks valid amendment. **Verification:** injected retry/amendment trace has one active action and audit chain.

### F-05 — Challenge budget overwhelms support
**Symptom:** recovery SLA worsens after increased K. **Competing causes:** high support rate; staffing change; channel outage; longer handling. **Diagnostic:** reconcile challenge, support, and recovery arrival/service rates. **Fix:** lower K, reserve support capacity, or restrict eligible channel. **Risk:** fewer attacks challenged. **Verification:** measured queue age stays within approved bound.

**Takeaway:** distinguish identity-risk evidence, authentication mechanics, and customer-support failure before changing a threshold.

## 13. Transferable Lesson
Security scoring is an intervention under partial observation. Keep decision-time availability, action capability, idempotency, recovery, and customer impact explicit. This transfers to password reset, high-risk transaction authentication, and account-recovery triage; it does not justify applying the same threshold across channels with different challenge capability or accessibility.

**Takeaway:** a safer score cannot compensate for an unsafe recovery system.

## 14. Connections & Cross-References
See [Fraud evaluation, calibration and capacity](../../../02_data_science_method_playbooks/02_fraud_modelling_playbook/Cheat_Sheets/01_Fraud_Evaluation_Calibration_and_Capacity.md) for rank-metric definitions, and the local payment-card case for a different synchronous action constraint. The original source title is visible in `directory.txt` lines 481–486; its body was not available and no source claims are inferred.

### 14a. Interview pack
#### Q01 — Why is a 0.5 threshold invalid by default?
**Answer:** score scale, action costs, capacity, and calibration determine policy; 0.5 has no inherent operational meaning. **Reasoning:** a probability estimate is not a decision rule. **Follow-up:** select K from verified capacity and guardrails. **Alternative:** optimize a pre-approved utility after calibration. **Red flag:** “probability above half means block.” **Score:** 0 unsafe; 1 identifies threshold choice; 2 connects constraints; 3 adds validation. **Support:** 5c, 9.
#### Q02 — Compute 32/400 and 32/80.
**Answer:** 8% precision@400 and 40% recall@400. **Reasoning:** different denominators. **Follow-up:** 368 challenged events are not mature positives. **Alternative:** give counts first. **Red flag:** calls 32/20,000 recall. **Score:** 0 wrong; 1 one metric; 2 both; 3 mentions maturity. **Support:** A-01/A-02.
#### Q03 — Why not call failed challenge fraud?
**Answer:** it has competing legitimate and outage explanations. **Reasoning:** intervention response is not ground truth. **Follow-up:** retain separate label state. **Alternative:** use it as a monitoring signal only. **Red flag:** automatic conversion. **Score:** 0 unsafe; 1 caveat; 2 mechanism; 3 audit plan. **Support:** F-02.
#### Q04 — How is support load computed?
**Answer:** 400*0.10*8=320 minutes. **Reasoning:** 40 cases times 8 minutes. **Follow-up:** at 20% it is 640 minutes. **Alternative:** model a distribution when available. **Red flag:** treats 400 as support cases. **Score:** 0 wrong; 1 arithmetic; 2 units; 3 sensitivity. **Support:** A-03.
#### Q05 — What creates feature leakage?
**Answer:** a value unavailable at decision time or a current-state join that overwrites history. **Reasoning:** replay needs availability time. **Follow-up:** exclude unproven fields. **Alternative:** use retained historical versions. **Red flag:** warehouse timestamp is enough. **Score:** 0 wrong; 1 leakage word; 2 timing; 3 replay test. **Support:** 5d, F-01.
#### Q06 — Why does shadow mode not prove prevention?
**Answer:** it does not change action, so alternative-action outcomes are unobserved. **Reasoning:** it tests technical compatibility, not causal effect. **Follow-up:** controlled evaluation with guardrails. **Alternative:** quasi-experiment with limits. **Red flag:** score correlation proves prevention. **Score:** 0 wrong; 1 caveat; 2 causal mechanism; 3 policy plan. **Support:** 9.
#### Q07 — A channel lacks challenge capability. What happens?
**Answer:** exclude it from challenge policy and use approved existing route or validated channel-specific policy. **Reasoning:** score cannot create an action. **Follow-up:** test capability before ranking. **Alternative:** pause expansion. **Red flag:** send unsupported challenge. **Score:** 0 unsafe; 1 fallback; 2 capability boundary; 3 verification. **Support:** 4, 8.
#### Q08 — How do retries stay consistent?
**Answer:** durable decision keyed by event ID and snapshot hash; amendments are explicit supersessions. **Reasoning:** retry is not new risk evidence. **Follow-up:** test lost response trace. **Alternative:** transactional equivalent. **Red flag:** rely on client behavior. **Score:** 0 unsafe; 1 dedupe; 2 amendment distinction; 3 trace. **Support:** 8, F-04.
#### Q09 — Why is missing device not a new device?
**Answer:** collection absence can arise from consent, channel, or outage. **Reasoning:** conflation can create proxy harm. **Follow-up:** stratify missingness. **Alternative:** route unknown to baseline. **Red flag:** default risky. **Score:** 0 unsafe; 1 notes missingness; 2 causes; 3 guardrail. **Support:** 5a, F-03.
#### Q10 — What is a plausible wrong method?
**Answer:** train a classifier on final recovery resolution then deploy at login. **Reasoning:** it leaks downstream action and label information. **Follow-up:** use cutoff snapshots. **Alternative:** exclude unavailable outcomes entirely. **Red flag:** use all historical columns. **Score:** 0 wrong; 1 says leakage; 2 explains; 3 remediation. **Support:** 5d.
#### Q11 — What changed constraint reverses the decision?
**Answer:** recovery capacity below safe load pauses the policy even if rank metrics improve. **Reasoning:** customer-access guardrail is binding. **Follow-up:** lower K and re-evaluate. **Alternative:** restrict channel. **Red flag:** ignore support SLA. **Score:** 0 unsafe; 1 pauses; 2 explains tradeoff; 3 sensitivity. **Support:** 6, F-05.
#### Q12 — What must outcome reporting say?
**Answer:** policy, eligible denominator, maturity window, comparator, uncertainty, guardrails, and evidence type. **Reasoning:** otherwise score and outcome are confounded. **Follow-up:** keep fallback traffic. **Alternative:** pre-registered dashboard. **Red flag:** only report precision. **Score:** 0 incomplete; 1 names metric; 2 full contract; 3 selective-label caveat. **Support:** 9–11.

## 15. Quick Reference Card
| Field | Value |
|---|---|
| Decision | Allow, supported challenge, or approved recovery route for an eligible account event. |
| Baseline | Existing authentication rules and recovery workflow. |
| Formula | precision@K=TP_K/K; recall@K=TP_K/P; support minutes=K*r*m. |
| Critical condition | Every model input must be available before the event decision. |
| Do not infer | Challenge failure or missing device data proves takeover. |
| Fallback | Channel-specific approved rules with logged reason and preserved recovery. |
| Evidence/stage | Illustrative, not measured; proposed and unreviewed. |
| Review trigger | Flow, consent, attack, deadline, or recovery-policy change. |
