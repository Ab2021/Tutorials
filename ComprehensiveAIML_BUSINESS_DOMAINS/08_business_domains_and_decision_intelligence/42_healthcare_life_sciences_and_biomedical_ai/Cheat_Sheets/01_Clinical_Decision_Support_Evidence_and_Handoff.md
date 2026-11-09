---
collection: "healthcare-life-sciences-biomedical-ai"
file_id: "01"
topic: "Clinical decision-support evidence and clinician handoff"
level: 1
status: "draft — substantive expansion required before publication"
version: "0.1"
created: "2026-11-09"
reviewed: null
review_trigger: "Change in intended-use population, decision authority, data source, or governing clinical policy"
notation_style: "plain-text"
prerequisites: ["Clinical workflow literacy", "confusion-matrix arithmetic", "point-in-time data handling"]
pattern_tags: ["clinical-decision-support", "human-review", "decision-time-boundary"]
source_status: "OPEN — directory inventory and source brief consulted; original scenario body unavailable"
owner_role: "Clinical AI governance lead"
source_index: "INDEX.md"
---

# Clinical Decision Support: Evidence, Decision-Time Data, and Clinician Handoff

> **Purpose:** Specify what an assistive clinical system may show, when it must abstain, and how to evaluate it without treating a model score as clinical benefit.
>
> **Prerequisites:** Confusion-matrix counts; event time versus availability time; a local clinical governance process.
>
> **Key outcome:** Given a proposed clinical-support workflow and a time-stamped cohort, you can define its unit of decision, calculate sensitivity correctly, reject leakage, and route an unsupported or out-of-authority output to a clinician.
>
> **This file does NOT cover:** Diagnosis, prescribing, device clearance, or a claim that any named provider deployed this design.
>
> **Level:** 1 — Foundations.
>
> **Evidence:** DERIVED metric arithmetic; VERIFIED arithmetic below; ILLUSTRATIVE workflow; OPEN local clinical policy and source-body evidence.
>
> **Retrieval cue:** If a high offline score is being presented as patient benefit, start at Sections 4, 5, and 8.

## 1. File Header

This is a design and review lesson, not medical advice and not a clinical protocol. The source-coverage brief requires a bounded decision, decision-time data boundary, authorization/review invariant, sensitivity calculation, and clinician escalation. The visible source inventory names clinical decision support but the original body is unavailable; no workflow, performance, or architecture is attributed to it.

The operational question is narrower than “use AI in healthcare”: for one encounter or message, what non-binding information can be displayed to a qualified clinician before that clinician makes the accountable decision? A system that drafts, ranks, or flags can be useful without diagnosing, ordering, or acting.

**Takeaway:** The intended use and authority boundary are part of the technical specification, not a disclaimer added after modelling.

## 2. Concept Map / Mental Model

```text
[Encounter/message at decision time t_d]
              |
              v
[Authorized data available no later than t_d]
       | missing/stale/unauthorized
       +---------------------------> [abstain + reason + clinician workflow]
              |
              v
[Bounded assistive rule/model] -> [evidence, uncertainty, version]
              |
              v
[clinician reviews in local workflow]
       | accepts / edits / rejects
       v
[documented accountable decision]
              |
              v
[delayed outcome + audit cohort, never fed back into t_d features]
```

The model output is an input to professional judgment. “Automation” is not a harmless synonym for assistance: it changes authority, monitoring, interface design, and required validation. Keep three clocks separate: event time (when a fact happened), availability time (when the system could access it), and decision time (when the support was shown).

**Takeaway:** A valid evaluation and a safe interface both begin with the information actually available at the decision boundary.

## 3. Foundational Definitions

| Term | Meaning | Unit/type | Nonexample or boundary |
|---|---|---|---|
| unit of decision | one defined patient-encounter, message, image study, or order-review event | identified record, not a patient lifetime | an arbitrary row after duplicating encounters |
| index time `t_d` | instant at which support is generated | timestamp with zone | label-adjudication date |
| feature availability `t_a` | earliest time a feature was available to the proposing system | timestamp | a value backfilled after the encounter |
| target | precisely adjudicated event the support attempts to help identify | binary/ordinal/time-to-event | an undocumented proxy called “risk” |
| assistance | information that a clinician may accept, edit, or reject | UI output plus provenance | an autonomous diagnosis |
| abstention | deliberately withholding a recommendation and directing a defined fallback | decision class | silently returning a low-confidence normal result |
| sensitivity | share of target-positive units flagged positive | dimensionless proportion | positive predictive value |
| clinical utility | consequence to patients/workflow under an intervention protocol | outcome with population and window | offline AUROC alone |

A feature is admissible for an index event only if `t_a <= t_d`, it is authorized for that use, and its lineage is known. This rule applies even when a warehouse presents a cleaned record with one convenient timestamp.

**Takeaway:** Define time, unit, target, authority, and fallback before naming a model family.

## 4. Core Formulas with Full Derivation

### 4.1 Base: sensitivity

**Validity before use:** Use sensitivity only after a target-positive/target-negative reference standard and a fixed evaluation population are defined. It measures detection among positives, not benefit, calibration, workload, or safety.

A target-positive unit is either flagged (`TP`) or missed (`FN`). Therefore the positive population is `TP + FN`; the fraction flagged is:

```text
FORMULA [CDS-01]: sensitivity = TP / (TP + FN)
WHERE: TP = target-positive units flagged positive [count];
       FN = target-positive units not flagged positive [count].
CONSTRAINT: one decision per unit, fixed target definition, and reference-standard status;
            do not infer benefit or prevalence from this ratio.
NUMERICAL EXAMPLE: TP=90 encounters; FN=10 encounters;
                   sensitivity=90/(90+10)=0.90=90%.
INTERPRETATION: the illustrative system flags 90 of 100 reference-positive encounters;
                it says nothing about the harm, actionability, or timing of the 10 misses.
```

**Independent check:** `TP + FN = 100`; `0.90 × 100 = 90`. The result is dimensionless. If the reference standard changes, this number is not comparable without re-adjudication.

### 4.2 General: workload and positive predictive value

Let `FP` denote target-negative units flagged. Review burden is `TP + FP`, and positive predictive value (PPV) is `TP/(TP+FP)`. These use different denominators than sensitivity.

```text
FORMULA [CDS-02]: PPV = TP / (TP + FP)
WHERE: FP = target-negative units flagged [count].
CONSTRAINT: same cohort, threshold, and reference standard; PPV depends on prevalence.
NUMERICAL EXAMPLE: TP=90, FP=210; PPV=90/(90+210)=0.30=30%.
INTERPRETATION: 300 illustrative alerts create 210 false-alert reviews; a 90% sensitivity
                claim does not show whether that workload is acceptable.
```

A review-capacity constraint can be written as `TP + FP <= C`, where `C` is clinician review capacity [alerts per shift]. This is a policy constraint, not a model metric. If `C=240` alerts/shift, the illustrative threshold is infeasible by 60 alerts.

### 4.3 Variants: abstention coverage

If the system abstains on `A` of `N` eligible units, coverage is `(N-A)/N`. Report performance both on all eligible units and on the covered subset; otherwise abstention can make a selective metric appear improved.

```text
FORMULA [CDS-03]: coverage = (N - A) / N
WHERE: N = eligible decisions [count]; A = abstentions [count].
CONSTRAINT: eligibility and abstention reason are logged before outcome inspection.
NUMERICAL EXAMPLE: N=1,000; A=100; coverage=900/1,000=90%.
INTERPRETATION: a 90% covered subset cannot establish service for the 10% routed away.
```

### 4.4 Limit and counterexample

A perfect retrospective sensitivity can be invalid at point of use if a feature arrived after `t_d`. For example, a finalized discharge field may encode information unavailable during triage. No threshold repair fixes that causal/time error; remove or time-bound the feature and rebuild the evaluation.

**Takeaway:** Sensitivity, PPV, workload, and coverage answer different questions; publish their denominators and decision-time eligibility together.

## 5. Worked Numerical Example

**Evidence:** ILLUSTRATIVE teaching cohort, with arithmetic VERIFIED; not a clinical result.

At the end of a fixed historical evaluation window, adjudication identifies 100 target-positive encounter decisions. A proposed flag is present on 90 and absent on 10. The numerator is 90 flagged positives; the denominator is all 100 positives:

```text
sensitivity = 90 / (90 + 10) = 0.90 = 90%.
```

Assume 210 target-negative decisions also receive a flag. The review queue is `90+210=300` alerts. If one clinician can review 240 alerts in the stated shift at the required evidence standard, queue excess is `300-240=60` alerts. The design choices are not “ship” or “discard”: increase verified capacity, choose a validated threshold that meets a different clinical objective, restrict the eligible population with a justified protocol, or abstain and route to an existing pathway.

**Sensitivity:** If false-positive alerts rise by 20% with positives unchanged, `FP=252`, alerts become 342, PPV is `90/342≈26.3%`, and excess over 240 becomes 102. If the target-positive denominator is discovered to include 20 unadjudicated cases, sensitivity is unknown rather than `90/120`; an unadjudicated denominator is not a valid false-negative count.

**Takeaway:** Checked arithmetic exposes whether a threshold fits the workflow; it does not authorize a clinical action.

## 6. Visual / Geometric Intuition

Think of an eligible decision as a point moving along a timeline. The vertical line at `t_d` is a data firewall: features on the left may be considered after authorization; facts on the right are outcomes, follow-up, or future documentation. The score threshold partitions left-side feature space into flag, no-flag, and abstain regions. A high score near the boundary does not cross the authority boundary: the clinician remains responsible for interpretation and action.

A confusion matrix is a partition of adjudicated units, not a map of patient benefit. Moving the threshold generally exchanges false negatives for false positives; changing prevalence changes the apparent yield of the flag queue.

**Takeaway:** The timeline firewall and the authority boundary are clearer safety controls than a generic claim that a model is “human in the loop.”

## 7. Comparison / Selection Table

| Option | Use when | Benefit | Disqualifier | Fallback |
|---|---|---|---|---|
| manual/rules baseline | stable, interpretable criteria exist | known workflow and authority | rules are unmaintained or not time-valid | repair rule ownership and audit it |
| non-binding flag with evidence | signal is auditable and clinician review exists | focuses attention without autonomous action | queue capacity or provenance absent | abstain and retain baseline |
| ranked worklist | scarce review must be prioritized | makes capacity explicit | rank alters access without fairness/utility protocol | random/manual allocation study |
| autonomous action | only after separately established authority and validation | may reduce latency | evidence, accountability, recovery, or policy incomplete | do not deploy as autonomous |

**Takeaway:** Select the least-authoritative intervention that can plausibly improve the bounded workflow and can be evaluated safely.

## 8. Failure Modes & Edge Cases

| ID | Competing causes | Discriminating check / expected signal | Bounded fix and verification |
|---|---|---|---|
| CDS-F01 future-information leakage | post-index documentation; late warehouse replication; outcome-coded field | field-level `t_a > t_d` audit; remove field and compare | exclude/time-version feature; rerun time-safe evaluation |
| CDS-F02 proxy improvement without utility | easier label; threshold change; review behavior change | compare predefined workflow outcome and queue burden, not score alone | retain baseline control; run approved intervention evaluation |
| CDS-F03 missing handoff | UI omission; unclear ownership; capacity outage | sampled abstention has no acknowledged recipient | require owner/SLA/escalation; test failed notification path |
| CDS-F04 unauthorized action | scope creep; ambiguous language; integration default | audit shows order/change without clinician confirmation | block action endpoint; require explicit confirmation and audit event |
| CDS-F05 subgroup instability | shifted prevalence; missingness; reference-standard differences | stratified denominators and uncertainty show divergence | pause affected use; investigate measurement and access boundaries |

Each containment has risk: lowering alert volume can increase misses, and adding review can exhaust staff. Verify the complete path with a trace containing a stale feature, a missing feature, an abstention, a clinician rejection, and a failed notification.

**Takeaway:** A failure entry is useful only when it distinguishes causes, limits the fix, and states what proves recovery.

## 9. Complexity & Cost Analysis

For `N` eligible decisions, a simple score pass is typically `O(Np)` for `p` features after data access; that notation does not measure clinical latency or review work. Review effort dominates when each alert requires `m` minutes: workload is `(TP+FP)m` clinician-minutes. At 300 illustrative alerts and 4 minutes/alert, workload is 1,200 minutes or 20 clinician-hours. This is a sizing calculation, not staffing advice; interruptions, escalation, documentation, and case mix are excluded.

Audit storage must retain the input lineage, feature availability, model/policy version, output, abstention reason, user acknowledgment, and outcome linkage under an approved retention policy. Do not log unnecessary sensitive content merely for convenience.

**Takeaway:** An algorithmic complexity claim never substitutes for a measured workflow-capacity and privacy review.

## 10. Key Theorems / Results Reference

- **Confusion-matrix identity (DERIVED):** for a fixed positive population, `TP+FN` is the target-positive denominator. It does not guarantee valid labels.
- **PPV prevalence dependence (DERIVED):** PPV changes when the mix of positive and negative units changes even with fixed sensitivity and specificity; compare cohorts only with aligned definitions.
- **Temporal leakage principle (mechanism):** using information unavailable at `t_d` makes retrospective evaluation answer a different decision problem than deployment.
- **No clinical-benefit implication (boundary):** predictive discrimination does not, by itself, identify intervention effect, safety, or net benefit.

Primary evidence needed for any real deployment claim: governing intended-use policy, reference-standard protocol, evaluation protocol, population/window, clinician-review process, and outcome analysis. The currently accessible source inventory does not supply these artifacts.

**Takeaway:** Mathematical identities bound what a metric means; they do not fill evidence gaps about intervention effects.

## 11. Interview / Exam Questions

1. **Why is 90/100 sensitivity not a benefit claim?** It only describes detection among adjudicated positives; benefit needs a comparison of interventions, outcomes, harms, and workflow. **Follow-up:** specify the outcome window and comparator.
2. **What timestamp prevents leakage?** The earliest availability timestamp to the proposed system, compared with `t_d`; the record’s displayed timestamp may be insufficient. **Red flag:** using label completion time.
3. **Compute sensitivity for TP=90, FN=10.** `90/(90+10)=0.90`. **Check:** denominator is 100 positive units.
4. **Why can PPV fall while sensitivity is unchanged?** More negative units or false positives can enter the queue; prevalence and specificity affect PPV. **Alternative:** inspect stratified PPV.
5. **What should an abstention do?** Emit a logged reason and route to a defined human/baseline path; it must not silently appear as a reassuring negative.
6. **A feature is documented after the encounter but describes an earlier event. Can it be used?** Only if it was actually available to the system by `t_d`; semantic event time alone is not enough.
7. **What makes a clinician handoff testable?** Owner, channel, acknowledgment, SLA, escalation, and a failure-path trace. **Red flag:** “human in the loop” without responsibility.
8. **Why is a random split often weak here?** Related encounters and time-dependent documentation can leak across splits; use a justified entity/time design.
9. **How would a changed constraint of 240 alerts/shift affect a 300-alert policy?** It is infeasible by 60 alerts; revise capacity or policy and re-evaluate error trade-offs.
10. **What evidence is still OPEN in this lesson?** The original source body, local policy, reference-standard procedure, and any measured clinical outcome; none should be inferred.

**Takeaway:** Strong answers name the denominator, the time boundary, the accountable role, and the verification artifact.

### 11.1 Source-scope and evidence ledger

| ID | Source / artifact | Exact supported scope | Date/version | Access | Does not establish | Recheck trigger |
|---|---|---|---|---|---|---|
| SRC-01 | `Train/Templates/directory.txt`, visible `Healthcare` inventory node | a visible source topic named clinical decision support | inventory consulted 2026-11-09 | local | original lesson contents, a deployment, or a result | original body becomes available |
| SRC-02 | `00_SOURCE_COVERAGE_AGENTS.md#scenario-07` | curriculum obligations: bounded decision, decision-time boundary, authorization/review invariant, sensitivity exercise | local brief, 2026-11-09 | local | clinical validity or a named organization's practice | brief revision |
| SRC-03 | `00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#deep-payer` | payer workflow distinctions and caution on named-company claims | local brief, 2026-11-09 | local | UHG/Optum/CVS architecture or performance | primary evidence acquired |
| SRC-04 | worked cohort in Section 5 | hand-derived arithmetic only | constructed 2026-11-09 | local | a measured population effect | any use beyond illustration |

The directory contains source labels rather than accessible original teaching bodies. The visible healthcare title supports teaching the topic at a bounded level; it does not support reconstructing hidden content. The payer brief explicitly contains an unresolved `part1` child, so that unavailable content remains unresolved. A learner must not convert this ledger into a claim that a company uses the workflow described here.

### 11.2 Decision-time trace exercise

Consider one illustrative encounter `E-17`. The triage timestamp is 09:00. A vital-sign observation arrives at 08:55 and is available to the proposed service at 08:56; it passes the time test. A note is entered at 08:58 but becomes available to the service at 09:03; it fails a 09:00 decision-time test even if its clinical event describes an earlier conversation. An adjudicated outcome appears at 16:00 and is strictly evaluation-only.

| Item | Event time | Availability time | Allowed at `t_d=09:00`? | Reason |
|---|---:|---:|---|---|
| vital sign | 08:55 | 08:56 | yes | authorized and available |
| clinician note | 08:58 | 09:03 | no | service could not have used it |
| final outcome | 15:40 | 16:00 | no | post-decision label |
| missing field indicator | 09:00 | 09:00 | yes | absence is known at decision time |

The invariant is evaluated per field, not once per record. A feature store snapshot requires its own availability evidence; a nightly refreshed table cannot be assumed point-in-time correct because a field has a date column.

**Worked trace check:** If an implementation scores `E-17` at 09:00 with the 09:03 note present, the trace must fail before any metric is computed. Removing the note is the correction; reusing the old metric is not verification. Re-run the extraction on a snapshot that excludes post-09:00 availability and record changed eligible count, feature missingness, and metric confidence interval.

### 11.3 Threshold-selection walkthrough

An illustrative candidate policy has three thresholds. Counts below come from one fixed adjudicated cohort and are not a deployment claim.

| Threshold | TP | FN | FP | TN | sensitivity | PPV | alerts | Decision consequence |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| low | 96 | 4 | 500 | 400 | 96% | 16.1% | 596 | exceeds 240-alert capacity |
| medium | 90 | 10 | 210 | 690 | 90% | 30.0% | 300 | exceeds capacity by 60 |
| high | 70 | 30 | 60 | 840 | 70% | 53.8% | 130 | fits capacity but increases missed positives |

At the medium threshold, the arithmetic is `90/(90+10)=0.90` and `90/(90+210)=0.30`. At the high threshold, `70/(70+30)=0.70` and `70/(70+60)=0.53846`. These hand-derived calculations require a fixed target definition and cohort. They do not calculate net benefit because relative harms, action effectiveness, timing, and patient consequences have not been supplied.

The capacity constraint eliminates the low threshold under the stated 240-alert shift, but it does not choose high. A defensible next step is to measure whether additional review capacity, a narrowly justified eligibility rule, or a different non-binding display can change the decision. Do not quietly discard the 60 excess medium-threshold alerts: the disposition of delayed alerts is part of the safety question.

### 11.4 Expanded diagnostic investigations

#### CDS-D01 — apparent performance gain after feature refresh

**Trigger:** AUROC or sensitivity rises after a warehouse refresh.

**Competing causes:** a genuine measurement improvement; post-index data leaking into a feature; changed target adjudication; an altered eligible population; duplicated encounter records.

**Discriminating diagnostics:** compare feature availability distributions against index time; join raw identifiers to find multiplicity; replay the prior extraction; inspect target-adjudication version; calculate metrics on the stable intersection population.

**Expected signals:** Leakage produces `t_a > t_d` records. Population shift changes denominator counts and missingness. Duplicate joins create repeated IDs or unexpectedly high row count.

**Immediate containment:** prevent score display for the new pipeline version while retaining the last governed baseline; do not characterize a paused display as a clinical incident without evidence.

**Correction:** amend the point-in-time query, deduplicate using the declared decision key, and version the target protocol.

**Prevention:** field-level availability contracts and release checks that fail on future timestamps.

**Verification:** execute a held-out time-safe extraction; reconcile unique decisions; compare metric and workload with documented uncertainty.

**Mitigation risk:** freezing a refresh can retain stale data; the fallback workflow must be monitored for its own failure.

#### CDS-D02 — lower alert volume with worse timely support

**Trigger:** the dashboard celebrates a drop in alerts per day.

**Competing causes:** beneficial removal of duplicates; raised threshold; missing upstream data; altered eligibility; delivery outage.

**Discriminating diagnostics:** plot created, eligible, delivered, presented, and acknowledged counts; inspect threshold and policy versions; compare missingness and time-to-review.

**Expected signals:** A delivery outage reduces presented alerts but not necessarily eligible events. A raised threshold changes score-band distribution. Missing upstream fields increase abstentions or default scores.

**Immediate containment:** disclose the count break as unexplained and route critical exceptions to existing workflow.

**Correction:** repair the specific transport/data/policy defect; do not lower a threshold until workload and error trade-offs are reviewed.

**Prevention:** state-transition counters with independent reconciliation and a change ledger.

**Verification:** inject a controlled non-production trace through eligibility, delivery, presentation, handoff, and terminal disposition.

**Mitigation risk:** compensating with repeated pages can create alert fatigue and obscure the original cause.

#### CDS-D03 — clinician rejection cluster

**Trigger:** one service line frequently rejects a recommendation.

**Competing causes:** inappropriate target; local workflow mismatch; unavailable context; UI explanation defect; selective documentation; a clinically legitimate difference in judgment.

**Discriminating diagnostics:** stratify rejection by service, input completeness, evidence display, decision time, and reason code; sample blinded expert review under approved governance.

**Expected signals:** Context absence clusters with missing fields; UI defects cluster by client version; target mismatch appears in adjudication disagreement, not merely rejection count.

**Immediate containment:** make the output explicitly advisory and strengthen escalation for that service; never coerce acceptance to improve a dashboard.

**Correction:** revise intended use, interface evidence, or data contract only after an approved review.

**Prevention:** capture structured rejection reasons while allowing free-text/unknown where policy permits.

**Verification:** repeat the workflow study with predeclared outcomes; distinguish acceptance rate from correctness and utility.

**Mitigation risk:** excessive reason capture adds workload and may lead users to select meaningless defaults.

#### CDS-D04 — missing expert handoff during abstention

**Trigger:** a missing input produces no recommendation and no visible owner.

**Competing causes:** intended silent no-op; routing configuration error; identity mismatch; queue outage; user-interface hiding the exception.

**Discriminating diagnostics:** follow a synthetic abstention event with correlation ID through route, inbox, acknowledgment, timeout, and escalation; compare to message transport logs.

**Expected signals:** a configuration error has a route decision but no recipient; a queue outage has queued but unconsumed event; UI defect has delivered event but no visible presentation.

**Immediate containment:** send the case to the defined baseline/manual process and notify the accountable workflow owner.

**Correction:** make abstention a first-class decision class with reason code, owner, SLA, and retry boundary.

**Prevention:** release test requiring a successful and failed handoff trace.

**Verification:** simulate recipient absence, duplicate delivery, timeout, and recovery; reconcile exactly one terminal disposition per eligible event.

**Mitigation risk:** unbounded retries can duplicate work or delay urgent baseline action.

#### CDS-D05 — group difference in PPV

**Trigger:** PPV differs across an allowed evaluation slice.

**Competing causes:** prevalence shift; reference-standard inconsistency; missingness; access/measurement difference; model calibration error; small sample uncertainty.

**Discriminating diagnostics:** publish slice denominator, confidence interval, target protocol, feature missingness, and decision-time availability; do not rank groups by a point estimate alone.

**Expected signals:** small denominators yield unstable intervals; missingness gap precedes score gap; changed adjudication changes labels independent of model.

**Immediate containment:** avoid expanding the affected use beyond validated scope; route uncertain cases to accountable review.

**Correction:** repair data-generation/measurement process or revise intended use; changing a threshold only for the metric can create a new fairness or workflow problem.

**Prevention:** pre-specified slice evaluation tied to the intended population and access audit.

**Verification:** repeat on a time-separated cohort with the same label and availability protocol.

**Mitigation risk:** indiscriminate collection of sensitive attributes may itself create governance and privacy obligations.

### 11.5 Interview-answer extensions

11. **A team proposes training on discharge summaries for triage. Defend or reject it.** Reject for real-time triage unless each field is proven available at triage and authorized. Discharge summaries commonly contain post-encounter information; their semantic relevance does not fix availability leakage. **Follow-up:** request a field-level availability table and a replay extraction. **Acceptable alternative:** use a time-bounded, prospectively captured subset after governance review. **Red flag:** “the summary describes the admission, so it is fine.” **Scoring:** full credit identifies decision time, availability, authorization, and re-evaluation.

12. **Why keep a manual baseline after a promising retrospective metric?** It provides the comparator and recovery path while workflow utility, harm, and reliability remain unestablished. **Follow-up:** define which baseline outcome and window are measured. **Acceptable alternative:** an approved phased evaluation with a comparable control workflow. **Red flag:** equating retrospective superiority with safe replacement. **Scoring:** full credit distinguishes prediction metric from intervention effect.

13. **An alert has 30% PPV. Is it useful?** Unknown. It could be appropriate or unusable depending on consequence, evidence, capacity, alternative workflow, and false-negative cost. Calculate workload and define a decision protocol first. **Follow-up:** with 300 alerts and 240 capacity, what fails? The policy is infeasible under the stated assumption. **Acceptable alternative:** abstain or narrow eligibility with validation. **Red flag:** declaring a universal PPV threshold. **Scoring:** full credit asks for denominators and actionability.

14. **What makes an explanation safe?** It names the evidence/provenance and uncertainty without implying diagnosis or authority beyond the intended use. It allows a clinician to detect stale or missing context. **Follow-up:** what if the explanation is unavailable? The output should abstain or use the defined baseline, not present unsupported certainty. **Acceptable alternative:** a structured evidence panel. **Red flag:** treating saliency as causal proof. **Scoring:** full credit names the fallback.

15. **How would you evaluate a changed constraint: review capacity falls from 240 to 120?** Recompute workload for every threshold, preserve false-negative accounting, and do not silently suppress alerts. The illustrative high threshold has 130 alerts and also exceeds 120 by 10. Consider capacity restoration, a governed prioritization protocol, or baseline routing. **Follow-up:** can a model select who loses review? Only with an explicit authorized objective and validation; this lesson provides neither. **Red flag:** raise threshold without documenting misses. **Scoring:** full credit shows arithmetic and authority boundary.

### 11.6 Evaluation protocol ledger

| Protocol element | Required declaration | Invalid shortcut | Audit artifact |
|---|---|---|---|
| intended use | assistive output, user, decision, population, authority | “clinical AI” | approved decision contract |
| index | one event key and `t_d` | one row after arbitrary joins | decision-key specification |
| target | adjudication rule and maturity window | billing proxy called truth | label protocol/version |
| comparator | current manual/rule workflow | no comparator | baseline trace |
| split | temporal/entity boundary | random split of correlated repeats | cohort manifest |
| threshold | fixed pre-evaluation or governed procedure | pick best test threshold | policy version |
| utility | outcomes, harm, workload, window | sensitivity only | analysis plan |
| monitoring | input, output, handoff, outcome lag | score histogram only | monitoring dashboard spec |

For an illustrative temporal split, train/tune on decisions ending 30 June, freeze the policy, and evaluate events beginning 1 July only after target maturation. This ordering avoids tuning on the final test window but does not cure feature leakage. The entity rule must additionally prevent repeated decisions for the same person from making a test event easy because its earlier related event was used in training, when that relationship violates the intended deployment setting.

An evaluator should keep an exclusion ledger. Every excluded unit needs a reason: unauthorized source, missing required input, duplicate decision key, impossible timestamp order, withdrawn authorization where applicable, or unresolved label. Report exclusions before computing metrics. A shrinking denominator can improve a percentage while reducing the reachable population.

### 11.7 Reproducibility manifest

| Artifact | Minimum fields | Why it matters |
|---|---|---|
| cohort manifest | extraction query revision, time zone, inclusion/exclusion counts | reconstruct denominator |
| feature manifest | field name, owner, availability rule, transform version, missingness meaning | detect leakage and semantic drift |
| policy manifest | model/rule version, threshold, abstention and routing configuration | reconstruct action |
| interface manifest | evidence fields, copy, client version, recipient mapping | distinguish model from presentation failure |
| review manifest | reviewer role, queue assignment, SLA, acknowledgment/disposition schema | measure handoff |
| label manifest | adjudicator/process, maturity window, corrections, timestamp | qualify target |
| run record | immutable IDs, random seed where relevant, environment, result location | reproduce calculation scope |

No manifest makes restricted patient data reproducible to an outsider. It makes the logic auditable by an authorized party. A random seed cannot reproduce a changing data warehouse, nondeterministic serving system, or a human review decision. Mark each unavailable artifact as unavailable rather than supplying a plausible substitute.

### 11.8 Operational trace and recovery oracle

Use a correlation identifier for one eligible decision. The expected state sequence is:

```text
ELIGIBLE -> INPUT_VALIDATED -> SCORED_OR_ABSTAINED -> ROUTED
         -> PRESENTED -> ACKNOWLEDGED -> DISPOSITION_RECORDED -> CLOSED
```

The terminal-state invariant is exactly one of `CLOSED_WITH_CLINICIAN_DECISION`, `CLOSED_WITH_BASELINE_ROUTE`, or `CLOSED_WITH_GOVERNED_EXCEPTION`; a decision must not be both actioned and silently abandoned. A retry uses the same correlation ID and idempotency key. If routing is retried after timeout, the audit should show attempts and one terminal owner, not duplicate clinical tasks.

**Oracle inputs:** `eligible_id=E17`, `t_d=09:00`, vital availability `08:56`, note availability `09:03`, label availability `16:00`, route attempts `[09:00,09:02]`, acknowledgment `09:04`, terminal disposition `09:07`. Expected: vital allowed; note and label excluded from feature vector; one routed task; terminal state closed. If two terminal dispositions occur, reconciliation fails. If no acknowledgment occurs by a configured deadline, expected outcome is governed exception, not a fabricated negative prediction.

### 11.9 Sensitivity boundaries

The 90/10 calculation has a binomial-count interpretation but this lesson does not calculate a confidence interval because the sampling design, independence, target adjudication, and intended estimand have not been specified. Repeated encounters, clustered care settings, and adjudication disagreement can invalidate a naive independent-binomial interval. This is not a reason to omit uncertainty; it is a reason to choose a suitable protocol with statistical expertise.

Changing the definition of positive from “adjudicated event within 24 hours” to “within seven days” changes `TP`, `FN`, prevalence, and operational timing. Reusing a previous threshold comparison after such a target change is invalid. Changing availability from contemporaneous to next-day batch also changes the system from decision support to retrospective review. The metric name may be unchanged while the decision problem is different.

### 11.10 Additional interview defense drills

16. **A manager asks for a single accuracy number. What do you answer?** First identify target prevalence, error asymmetry, decision threshold, alert capacity, and reference standard. Accuracy can be dominated by negatives and conceal unsafe misses. **Follow-up:** give an example: with rare positives, a never-flag system can have high accuracy and zero sensitivity. **Red flag:** supplying accuracy without population. **Scoring:** full credit names a task-appropriate metric and workflow guardrail.

17. **Why might a retrospective model appear calibrated and still fail operationally?** Calibration can be assessed on a dataset with different prevalence, documentation process, population, or availability boundary than live use. It also does not establish that a displayed probability produces the intended action. **Follow-up:** what changes after a policy revision? Recalibration/evaluation and policy review may both be required. **Red flag:** assuming probability semantics travel unchanged. **Scoring:** full credit separates statistical calibration, transportability, and workflow utility.

18. **A reviewer wants to remove abstention because users dislike blanks. What is the safer alternative?** Replace a blank with an explicit “insufficient or unavailable input” state, evidence/reason, and named baseline route—not a guessed reassuring output. **Follow-up:** how is abuse detected? Monitor abstention rate and reasons by input/source/version. **Red flag:** convert abstention to negative. **Scoring:** full credit specifies ownership and monitoring.

## 12. Quick Reference Card

| Field | Rule |
|---|---|
| Decision grain | One defined encounter/message/study at index time `t_d` |
| Data rule | Use authorized features only when `t_a <= t_d`; retain lineage |
| Core metric | `sensitivity=TP/(TP+FN)`; disclose target and denominator |
| Capacity rule | Review burden is `TP+FP`; compare to documented capacity |
| Authority rule | Support is non-binding unless a separately validated authority path exists |
| Abstention rule | Log reason and route to named fallback with SLA |
| Minimum audit | input/version/output/acknowledgment/outcome linkage, under approved retention |
| Avoid | Calling a score, retrospective metric, or named organization evidence of clinical benefit |
