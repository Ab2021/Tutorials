---
collection: "healthcare-life-sciences-biomedical-ai"
file_id: "01"
topic: "Clinical decision-support evidence and clinician handoff"
level: 1
status: "draft — expanded manuscript; source-body and local-policy evidence remain OPEN"
version: "1.0"
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

This is a design and review lesson, not medical advice and not a clinical protocol. The source-coverage brief requires a bounded decision, decision-time data boundary, authorization/review invariant, sensitivity calculation, and clinician escalation. The visible [source inventory](../../../../Templates/directory.txt#L248) names `07_Healthcare/07_01_clinical_decision_support.md` but the original body is unavailable; no workflow, performance, or architecture is attributed to it.

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

1. **Why is 90/100 sensitivity not a benefit claim?** **Checked answer:** it
describes detection among 100 adjudicated positive decisions, not whether a
display changed a patient or workflow outcome. **Reasoning:** benefit requires
an intervention comparator, an outcome window, harms, and a defined action.
**Follow-up:** name the baseline and outcome before claiming improvement.
**Acceptable alternative:** state that it is a screening property only. **Red
flag:** “90% means 90% better care.” **Scoring:** full credit names denominator
and the missing intervention evidence.

2. **What timestamp prevents leakage?** **Checked answer:** the earliest
availability time to the proposing system, compared with `t_d`, plus the
authorization and freshness tests. **Reasoning:** event time does not prove the
system could access a fact. **Follow-up:** ask for the ingestion receipt and
snapshot proof. **Acceptable alternative:** a versioned feature-store cutoff.
**Red flag:** use label completion time. **Scoring:** full credit distinguishes
event, availability, and decision clocks.

3. **Compute sensitivity for `TP=90`, `FN=10`.** **Checked answer:**
`90/(90+10)=0.90=90%`. **Reasoning:** the denominator is target-positive
decisions, not all alerts. **Follow-up:** what changes if 20 labels are immature?
The value is unresolved until the target protocol handles them. **Acceptable
alternative:** show numerator and denominator as counts. **Red flag:** divide by
300 alerts. **Scoring:** full credit supplies arithmetic and units.

4. **Why can PPV fall while sensitivity is unchanged?** **Checked answer:**
more false-positive alerts or a lower target prevalence enlarges `TP+FP` while
the positive-detection ratio can remain fixed. **Reasoning:** PPV’s denominator
is alerts, not positives. **Follow-up:** request threshold and cohort details.
**Acceptable alternative:** compare stratified PPV with counts. **Red flag:**
call PPV and sensitivity interchangeable. **Scoring:** full credit identifies
their different denominators.

5. **What should an abstention do?** **Checked answer:** emit a visible reason,
write an audit event, route to a named accountable fallback, track its deadline,
and close with one terminal disposition. **Reasoning:** a blank or negative
default hides an unserved decision. **Follow-up:** trace a recipient timeout.
**Acceptable alternative:** the established manual workflow. **Red flag:** “no
output means no risk.” **Scoring:** full credit includes ownership and recovery.

6. **A feature is documented after the encounter but describes an earlier
event. Can it be used?** **Checked answer:** only if it was actually available
to the proposing system by `t_d` and authorized for the use. **Reasoning:**
semantic timing cannot substitute for system availability. **Follow-up:** replay
the source snapshot. **Acceptable alternative:** exclude it and record
missingness. **Red flag:** rely on authored-note timestamp alone. **Scoring:**
full credit rejects retrospective reconstruction as deployment input.

7. **What makes a clinician handoff testable?** **Checked answer:** a recipient
selection rule, acknowledgment definition, SLA, escalation, idempotency key,
and auditable terminal disposition. **Reasoning:** message transport can succeed
while no accountable person sees the task. **Follow-up:** simulate a queue
outage. **Acceptable alternative:** governed baseline route. **Red flag:**
“human in the loop” with no responsible role. **Scoring:** full credit names a
failure-path check.

8. **Why is a random split often weak here?** **Checked answer:** repeated
encounters, related documentation, and time-dependent workflows can make test
records easier through correlated training records or future data. **Reasoning:**
the split must represent the intended decision context. **Follow-up:** specify
entity and temporal boundaries. **Acceptable alternative:** a justified
time-separated cohort. **Red flag:** shuffle rows after joining encounters.
**Scoring:** full credit links split choice to leakage risk.

9. **How does a 240-alert shift limit affect a 300-alert policy?** **Checked
answer:** it has `300-240=60` alerts without stated review capacity. **Reasoning:**
those alerts cannot all be treated as completed support. **Follow-up:** what
happens to the 60? Name a governed route, not silent suppression. **Acceptable
alternative:** increase verified capacity before evaluation. **Red flag:** raise
threshold without counting additional misses. **Scoring:** full credit gives the
arithmetic and workflow consequence.

10. **What evidence remains OPEN in this lesson?** **Checked answer:** the
original source body, local authorization/policy, reference-standard protocol,
workflow evidence, and any measured outcome. **Reasoning:** this file has
derived and checked illustrative arithmetic only. **Follow-up:** identify the
first artifact needed for a real deployment claim. **Acceptable alternative:**
a governed evaluation protocol and source access. **Red flag:** infer a named
organization’s architecture from a topic title. **Scoring:** full credit keeps
source inventory separate from evidence.

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

### 11.10 Evidence packet anatomy

The following packet is a teaching artifact for an illustrative review. It
separates an assertion from the evidence needed to decide whether the assertion
is usable. It is deliberately more demanding than a score report: a reviewer
must be able to find both the data boundary and the recovery boundary.

#### Packet item 1 — intended-use sentence

State one user, one decision moment, one eligible population, one output, and
one authority limit. “Help clinicians” fails because it supplies none of those
boundaries. “Show a non-binding, evidence-linked review cue to the assigned
clinician for an eligible encounter at `t_d`” is still incomplete until the
local workflow defines what “eligible” and “assigned” mean.

**Check:** compare the sentence to every UI action and integration endpoint.
**Reject when:** any endpoint performs an action not named in the statement.
**Recovery:** remove the endpoint or revise the intended use under governance.

#### Packet item 2 — decision-key declaration

Specify whether a decision is keyed by encounter, message, study, order review,
or another bounded event. Record the stable key, creation time, and rule for
reopened work. A patient identifier alone is not a decision key because one
person can have several independent decisions with different available facts.

**Check:** count unique keys before and after joins.
**Reject when:** joins alter decision count without an explicit multiplicity rule.
**Recovery:** aggregate or split records before scoring, then repeat metrics.

#### Packet item 3 — clock convention

Declare the time zone, clock source, precision, event time, availability time,
and decision time. Clock agreement is an assumption to test, not a formatting
choice. A feature with an apparently earlier event time can still arrive after
the decision because source clocks, replication, and documentation differ.

**Check:** inspect a sample of raw timestamps and ingestion receipts.
**Reject when:** time zones are implicit or timestamp lineage is unavailable.
**Recovery:** abstain for unprovable timing and repair source contracts.

#### Packet item 4 — authorization basis

For each field, name the owner, approved purpose, and relevant authorization
state at time of use. Do not turn a broad data-access role into an inference
that every downstream use is authorized. Cached, derived, and copied fields
need the same traceability as their originating data.

**Check:** reconcile feature manifest with authorization/purpose records.
**Reject when:** field authorization is “assumed from warehouse access.”
**Recovery:** suppress the field and document the changed feature vector.

#### Packet item 5 — target protocol

State the target event, adjudicator/process, observation window, maturity rule,
and correction policy. A billing code, later note, or service transfer may be a
proxy rather than the intended outcome. If a proxy is used, name the mismatch
and do not call it a clinical truth label.

**Check:** label a small governed sample against the written protocol.
**Reject when:** target logic is embedded only in an undocumented query.
**Recovery:** version the protocol and mark prior comparison limits.

#### Packet item 6 — exclusion ledger

Every exclusion should identify a decision key, exclusion reason, stage, and
whether it received baseline handling. Exclusions are part of the population,
not cleaning debris. A model can appear stronger simply because hard cases were
discarded after an unobserved join, source outage, or label delay.

**Check:** totals reconcile: candidate = included + excluded + unresolved.
**Reject when:** metric denominator cannot be reconciled to candidate decisions.
**Recovery:** stop comparison and reconstruct the cohort manifest.

#### Packet item 7 — feature availability proof

Record the earliest system-availability rule for each feature, not merely a
source-table update time. Backfills and corrected values often look historical
when viewed later. The proof must show what the proposing system could access,
not what an investigator can retrieve after the encounter.

**Check:** replay a stored point-in-time snapshot using a fixed `t_d`.
**Reject when:** the implementation reads a current-state table without history.
**Recovery:** use a versioned snapshot or remove the feature from the lesson.

#### Packet item 8 — transformation contract

Name input units, missingness behavior, categorical mapping, clipping, scaling,
and transform version. A valid raw measurement can become invalid after unit
conversion or a stale vocabulary lookup. Transformation should preserve an
audit trail from displayed evidence to score input.

**Check:** run edge values, missing values, and unit variants through the transform.
**Reject when:** output cannot distinguish raw absence from a transformed zero.
**Recovery:** fix the contract and invalidate incompatible scores.

#### Packet item 9 — policy contract

Document the score interpretation, threshold, abstention rule, and route chosen
for each policy version. A threshold is a workflow rule because it determines
who is asked to review work. It cannot be justified by sensitivity alone when
false positives consume finite attention.

**Check:** recompute alert counts from score bands and threshold configuration.
**Reject when:** production policy cannot be matched to an evaluated policy.
**Recovery:** route through baseline until configuration is reconciled.

#### Packet item 10 — evidence presentation contract

List what a reviewer sees: input lineage, missing/stale notices, model/policy
version, uncertainty language, and action limitation. Explain what happens when
the evidence panel is unavailable. “Explainable” is not a field; the specific
evidence and its boundary must be inspectable.

**Check:** capture a rendering snapshot for each supported client variant.
**Reject when:** a score is actionable after provenance-display failure.
**Recovery:** suppress the recommendation and emit an explicit fallback state.

#### Packet item 11 — human handoff contract

Name the owner, recipient selection rule, acknowledgment definition, deadline,
escalation, and baseline route. A queue does not by itself establish a handoff.
The contract must cover staff absence, duplicate delivery, expired work, and
the situation in which no qualified recipient is available.

**Check:** trace a synthetic abstention and a synthetic flag end to end.
**Reject when:** there is no terminal owner or only a transport receipt.
**Recovery:** make the system abstain until a governed path is restored.

#### Packet item 12 — capacity declaration

Specify the period, reviewer roles, minutes per review, interruptions excluded,
and the maximum safely handled volume. Capacity is a premise with uncertainty,
not a hard-coded integer copied between sites. It should distinguish an alert
created from an alert reviewed and an alert resolved.

**Check:** calculate queue excess using documented arrivals and capacity.
**Reject when:** policy volume exceeds capacity without an overflow procedure.
**Recovery:** change eligibility or capacity and re-evaluate the error tradeoff.

#### Packet item 13 — comparator trace

Describe the existing process at the same decision moment. The comparator might
be manual review, a stable rule, no cue, or a different queue allocation. A
retrospective score has no intervention comparator until the workflow action,
timing, and outcome collection plan are defined.

**Check:** trace one representative baseline decision from trigger to closure.
**Reject when:** baseline is described only as “current practice.”
**Recovery:** obtain the process owner’s bounded workflow specification.

#### Packet item 14 — metric dictionary

For every metric, preserve numerator, denominator, unit, population, window,
threshold, exclusions, comparator, and evidence label. Sensitivity has positive
units in its denominator; PPV has alerted units. Neither identifies benefit
without an intervention and outcome protocol.

**Check:** reconstruct each displayed rate from raw aggregate counts.
**Reject when:** a metric card lacks a denominator or target version.
**Recovery:** replace percentage-only reporting with a complete metric record.

#### Packet item 15 — uncertainty statement

State which uncertainty is quantified, which is not, and why. A binomial
interval may be inappropriate when repeated decisions are clustered or labels
are disputed. Lack of an interval should lead to a stated analysis limitation,
not a false implication of exactness.

**Check:** compare statistical method assumptions with the cohort design.
**Reject when:** independent-unit language is used for correlated repeated events.
**Recovery:** seek an appropriate analysis plan before drawing comparative claims.

#### Packet item 16 — slice and transportability plan

Define permitted evaluation slices before reading results and record denominators,
availability, missingness, target consistency, and uncertainty for each. Slice
analysis is not an invitation to gather attributes without authorization; its
scope belongs in the governance and data-minimization decision.

**Check:** verify that the slice definition is available at decision time.
**Reject when:** post hoc slice selection is represented as a preplanned finding.
**Recovery:** label the result exploratory and preserve the discovery trail.

#### Packet item 17 — safety stop rule

Write the signal, owner, response time, action, and re-entry condition for a
suspected unsafe state. Examples include unproven availability, failed routing,
version incompatibility, or queue saturation. A stop rule is usable only if it
changes system behavior and assigns an accountable response.

**Check:** simulate the signal without relying on patient data.
**Reject when:** alerting occurs but no one owns the containment action.
**Recovery:** route to baseline and escalate according to the declared policy.

#### Packet item 18 — rollback and recovery trace

Identify the last compatible policy, how it is selected, what remains logged,
and how duplicate work is avoided during rollback. Reverting a model does not
automatically revert UI text, routing, transforms, or data permissions. Each
component needs a compatibility and recovery statement.

**Check:** perform a paper trace from incompatibility detection to closure.
**Reject when:** rollback produces a mixed set of component versions.
**Recovery:** stop serving until a known-compatible bundle is active.

#### Packet item 19 — audit access boundary

Name who may inspect which artifacts and for what purpose. An audit trail can
be necessary while its raw contents remain restricted. Separate immutable event
identifiers and aggregate monitoring from unnecessary retention of full inputs.

**Check:** review role permissions against the minimum audit questions.
**Reject when:** debugging access is broader than the stated review purpose.
**Recovery:** restrict access and use approved aggregate diagnostic artifacts.

#### Packet item 20 — evidence status declaration

Distinguish DERIVED arithmetic, VERIFIED hand calculation, ILLUSTRATIVE inputs,
SOURCED external claims, and OPEN gaps. A checked calculation does not convert
constructed inputs into a measured deployment result. A source title does not
fill in unavailable logs, local policy, or a reference standard.

**Check:** inspect every conclusion for a matching evidence label and artifact.
**Reject when:** illustrative capacity or outcome is phrased as observed fact.
**Recovery:** narrow the conclusion and state the needed evidence explicitly.

### 11.11 Review-board evidence drills

The following ILLUSTRATIVE drills are executable paper-oracle inputs. They do not
represent patient records, a live clinical workflow, or a validation result.
For each drill, the reviewer records the input contract, expected disposition,
actual disposition, operator, policy version, and any exception. A “pass” means
the stated invariant held in the trace; it does not establish clinical benefit.

#### Drill A — late laboratory interface

- **Input:** `D-01`, `t_d=10:00`, authorized laboratory value event `09:40`,
  interface availability `10:06`, score request `10:00`.
- **Question:** May the feature enter the score vector?
- **Oracle:** no; `t_a=10:06 > t_d`; output is `ABSTAIN(MISSING_AT_DECISION)`.
- **Why:** a fact can predate the request clinically yet still be absent from
  the proposing service at request time.
- **Required log:** source version, event time, availability time, exclusion
  reason, fallback recipient, and terminal disposition.
- **Recovery:** use the governed baseline path; do not delay the decision merely
  to wait for the value unless the intended-use protocol permits waiting.
- **Failure signal:** a replay contains the value in a 10:00 feature snapshot.
- **Decision consequence:** reject the extraction or availability contract.

#### Drill B — authorized but semantically stale observation

- **Input:** `D-02`, `t_d=14:00`, observation available `08:00`, maximum
  freshness declared by policy `2 h`, and no newer measurement.
- **Question:** Does `t_a <= t_d` alone make the observation admissible?
- **Oracle:** no; it passes chronology but violates the stated freshness rule.
- **Why:** availability answers “could the system see it?”; freshness answers
  “may this intended use rely on it?”
- **Required log:** freshness calculation `14:00-08:00=6 h`, policy version,
  abstention reason, route, and acknowledgment.
- **Recovery:** request a current observation or route to the baseline workflow.
- **Failure signal:** score emitted with no stale-data banner or route.
- **Decision consequence:** a temporal check needs a semantic-validity companion.

#### Drill C — missing field versus measured zero

- **Input:** `D-03`, numeric field absent, schema allows `0`, model transform
  maps missing to zero, and missingness indicator is not recorded.
- **Question:** Can an auditor determine what evidence the output used?
- **Oracle:** no; absent and measured-zero states have been conflated.
- **Why:** the model may treat a collection failure as a physiological value.
- **Required log:** raw presence flag, transformed value, missingness feature,
  schema version, and source-system status.
- **Recovery:** make missingness explicit and retest the transform on both cases.
- **Failure signal:** identical feature vector for `missing` and a measured zero.
- **Decision consequence:** abstain if the required evidence cannot be separated.

#### Drill D — duplicate decision key

- **Input:** `D-04`, one encounter key, two ingestion rows, same score request,
  and a reviewer queue keyed by row identifier instead of decision identifier.
- **Question:** How many review tasks may be created?
- **Oracle:** one terminal task for the declared decision grain.
- **Why:** duplicate inputs can manufacture workload and apparent agreement.
- **Required log:** canonical decision key, source row IDs, deduplication rule,
  idempotency key, task ID, and disposition count.
- **Recovery:** reconcile rows before task creation and suppress duplicate retry.
- **Failure signal:** two acknowledgments or two contradictory dispositions.
- **Decision consequence:** metrics must be recomputed on unique decision units.

#### Drill E — delayed routing acknowledgment

- **Input:** `D-05`, valid flag at `11:00`, route accepted at `11:00:02`,
  no acknowledgment by policy deadline `11:10`.
- **Question:** Is delivery success equivalent to a completed handoff?
- **Oracle:** no; expected state is `CLOSED_WITH_GOVERNED_EXCEPTION`.
- **Why:** transport acceptance does not prove that an accountable user saw work.
- **Required log:** delivery receipt, recipient identity, deadline, escalation,
  acknowledgement status, and terminal owner.
- **Recovery:** escalate once through the approved route while preserving the
  baseline process; retries must use the same idempotency key.
- **Failure signal:** dashboard counts “delivered” as “reviewed.”
- **Decision consequence:** measure presentation and acknowledgment separately.

#### Drill F — clinician rejection with complete evidence

- **Input:** `D-06`, all inputs time-safe, support displayed with provenance,
  clinician rejects and supplies a permitted reason code.
- **Question:** Is rejection automatically a model error?
- **Oracle:** no; it is an observed workflow disposition requiring interpretation.
- **Why:** a clinician may possess authorized context outside the support inputs.
- **Required log:** output version, evidence panel version, rejection reason,
  optional free text, and later adjudication protocol version.
- **Recovery:** retain the decision and investigate only under approved review.
- **Failure signal:** product metric treats acceptance rate as correctness.
- **Decision consequence:** interface adoption and target validity are different.

#### Drill G — clinician acceptance without evidence display

- **Input:** `D-07`, recommendation displayed, provenance panel fails to load,
  clinician can still accept the proposed action.
- **Question:** May acceptance proceed under this lesson’s support contract?
- **Oracle:** no; the support is unsupported and must abstain or route baseline.
- **Why:** the evidence display is part of the non-binding review mechanism.
- **Required log:** client version, panel error, output suppression event, route,
  and recovery time.
- **Recovery:** disable unsupported acceptance; repair the interface dependency.
- **Failure signal:** accepted decision has no retrievable evidence provenance.
- **Decision consequence:** availability of explanation is a system dependency.

#### Drill H — version mismatch across services

- **Input:** `D-08`, scoring service policy `P-4`, UI label map `P-3`, and
  audit writer reports only an unversioned human-readable label.
- **Question:** Can the displayed meaning be reconstructed?
- **Oracle:** no; request is a governed exception until compatible versions align.
- **Why:** the same score can lead to different review language or routing.
- **Required log:** model, threshold, feature, UI, route, and schema versions.
- **Recovery:** enforce compatibility checks before serving and preserve rollback.
- **Failure signal:** audit replay cannot regenerate the displayed output.
- **Decision consequence:** version identity is not optional metadata.

#### Drill I — target maturation not complete

- **Input:** `D-09`, decision date 1 July, target requires 30-day maturation,
  evaluation run date 15 July.
- **Question:** May a no-event record be counted as a negative?
- **Oracle:** no; its label is unresolved under the stated target protocol.
- **Why:** early censoring converts future positives into apparent true negatives.
- **Required log:** target definition, maturity cutoff, unresolved count, and
  exclusion or survival-analysis handling.
- **Recovery:** defer outcome metric or use a predeclared censoring method.
- **Failure signal:** denominator silently includes immature outcomes.
- **Decision consequence:** report the eligible prediction count separately.

#### Drill J — policy threshold changed after test inspection

- **Input:** `D-10`, threshold candidates evaluated on a final holdout, then
  the chosen threshold is the one with the largest observed sensitivity.
- **Question:** Is the reported holdout sensitivity an unbiased final estimate?
- **Oracle:** no; selection has reused the test evidence.
- **Why:** the final sample became part of tuning.
- **Required log:** candidate thresholds, selection date, validation split,
  holdout lock status, and subsequent untouched evaluation plan.
- **Recovery:** use a governed validation procedure and reserve new evidence.
- **Failure signal:** no pre-test threshold or selection ledger exists.
- **Decision consequence:** treat the reported value as exploratory, not final.

#### Drill K — change in decision frequency

- **Input:** `D-11`, model evaluated once per encounter, proposed interface
  refreshes and scores every new observation during the encounter.
- **Question:** Does encounter-level evaluation transfer unchanged?
- **Oracle:** no; the unit and alert opportunity count have changed.
- **Why:** repeated scoring creates different workload, correlation, and exposure.
- **Required log:** trigger type, decision key, maximum alerts per encounter,
  suppression rule, and review workload.
- **Recovery:** define a new unit and evaluate the revised policy.
- **Failure signal:** one encounter produces multiple alerts without a limit.
- **Decision consequence:** recalculate denominators and capacity.

#### Drill L — location-specific source outage

- **Input:** `D-12`, one site loses an upstream feed, other sites remain normal,
  and the global aggregate missingness rate changes little.
- **Question:** Is a global monitor sufficient?
- **Oracle:** no; expected monitor slices by source/site and decision eligibility.
- **Why:** an aggregate can conceal a complete local failure.
- **Required log:** source health, site tag, missingness, abstention, route, and
  eligible-decision count by permitted slice.
- **Recovery:** abstain for the affected source and preserve baseline routing.
- **Failure signal:** site has near-zero flags and near-zero logged abstentions.
- **Decision consequence:** monitor the data-generating path, not scores alone.

#### Drill M — alert queue saturation

- **Input:** `D-13`, 300 routed alerts, capacity 240 per shift, no priority
  protocol, and 60 tasks still open at shift end.
- **Question:** May the system claim the 300 alerts were delivered support?
- **Oracle:** no; 60 have no completed review disposition under this capacity.
- **Why:** queue arrival is not clinician action.
- **Required log:** arrival time, queue position, service time, reassignment,
  expiry, and final route for every task.
- **Recovery:** reduce governed eligibility, add validated capacity, or route
  excess cases through the established baseline.
- **Failure signal:** backlog rises while acknowledgment rate is reported alone.
- **Decision consequence:** workload is a safety and evaluation variable.

#### Drill N — score scorecard without prevalence

- **Input:** `D-14`, report says “PPV 30%,” but omits TP, FP, eligible count,
  location, window, and target definition.
- **Question:** Can a reviewer compare it with another period?
- **Oracle:** no; the ratio lacks the population and operational context.
- **Why:** PPV moves with prevalence and threshold as well as model behavior.
- **Required log:** numerator, denominator, cohort, threshold, target, window,
  confidence method, and exclusions.
- **Recovery:** republish a denominator-complete metric card.
- **Failure signal:** reporting layer allows percentages without raw counts.
- **Decision consequence:** a dashboard must preserve the measurement contract.

#### Drill O — source authorization revoked

- **Input:** `D-15`, a field was authorized when a policy was trained, then its
  permission is revoked before the next score request.
- **Question:** May the service use the cached value?
- **Oracle:** no unless the governing authorization explicitly permits that use.
- **Why:** cache residence does not preserve a right to use a datum.
- **Required log:** authorization decision, policy version, cache timestamp,
  deletion or masking action, and fallback disposition.
- **Recovery:** suppress field use, invalidate dependent artifacts as required,
  and execute the approved baseline route.
- **Failure signal:** feature trace contains a source after revocation timestamp.
- **Decision consequence:** authorization must be checked at the relevant use.

#### Drill P — post-decision correction arrives

- **Input:** `D-16`, a source value available at `09:00` is corrected at `17:00`;
  score was generated at `09:05`.
- **Question:** Should the old audit record be silently overwritten?
- **Oracle:** no; preserve what was known and link a correction event.
- **Why:** auditability requires reconstructing the original decision context.
- **Required log:** original value/version, correction version, timestamps,
  whether correction triggers review, and outcome linkage.
- **Recovery:** apply the approved correction workflow; do not rewrite history.
- **Failure signal:** replay uses corrected value while claiming original output.
- **Decision consequence:** data lineage needs versioned rather than mutable facts.

#### Drill Q — unstable reference standard

- **Input:** `D-17`, two qualified reviewers disagree on target status and a
  third reconciliation process is not yet defined.
- **Question:** Is sensitivity calculable as a definitive measure?
- **Oracle:** arithmetic is calculable but the estimand is not fully specified.
- **Why:** `TP` and `FN` depend on unresolved label adjudication.
- **Required log:** reviewer labels, disagreement status, reconciliation protocol,
  blinding conditions, and target version.
- **Recovery:** mark outcome unresolved pending governed adjudication.
- **Failure signal:** one reviewer’s labels become “truth” without declaration.
- **Decision consequence:** metric precision cannot repair target ambiguity.

#### Drill R — apparent subgroup performance gap

- **Input:** `D-18`, one permitted slice has 3/5 flagged positives; another has
  90/100; the report ranks the first slice as worse.
- **Question:** Is the rank stable evidence?
- **Oracle:** no; small denominator and label process must be investigated.
- **Why:** point estimates conceal uncertainty and access/measurement differences.
- **Required log:** counts, target process, missingness, availability, time window,
  and uncertainty method suitable for the design.
- **Recovery:** avoid expanding scope based on an unstable comparison.
- **Failure signal:** slice table shows percentages but hides counts.
- **Decision consequence:** use a predeclared evaluation protocol and expertise.

#### Drill S — external model response timeout

- **Input:** `D-19`, support service calls a remote estimator, timeout at 2 s,
  and interface default renders “no concern.”
- **Question:** Is default-negative a valid failure behavior?
- **Oracle:** no; timeout is an abstention with a visible baseline route.
- **Why:** absence of a response is not evidence of absence of the target.
- **Required log:** request ID, timeout, retry policy, user-visible state, route,
  and terminal disposition.
- **Recovery:** surface unavailability and preserve accountable manual workflow.
- **Failure signal:** timeout count rises while negative rate rises in tandem.
- **Decision consequence:** technical failure states require clinical workflow paths.

#### Drill T — model score reused for a new action

- **Input:** `D-20`, a support score designed to prioritize review is proposed
  as a trigger for an autonomous downstream action.
- **Question:** Does the original evaluation authorize that reuse?
- **Oracle:** no; intended use, authority, comparator, and failures have changed.
- **Why:** the same numeric output can be harmless in one workflow and unsafe in
  another because the action and recovery boundary differ.
- **Required log:** original intended-use statement, proposed action contract,
  governance decision, and separate validation requirement.
- **Recovery:** block integration until a distinct authorization and evaluation.
- **Failure signal:** action endpoint accepts a score without clinician confirmation.
- **Decision consequence:** model portability is not decision-authority portability.

#### Drill U — display localization changes meaning

- **Input:** `D-21`, an interface translation changes “review required” into
  language interpreted as “confirmed condition.”
- **Question:** Is this merely a presentation defect?
- **Oracle:** no; it changes the decision interface and intended-use interpretation.
- **Why:** wording can alter authority, urgency, and user response.
- **Required log:** locale, copy version, rendering snapshot, user role, and
  complaint or correction route.
- **Recovery:** withdraw incorrect copy, restore baseline wording, and test role
  comprehension under approved human-factors review.
- **Failure signal:** output label differs from policy semantics by locale.
- **Decision consequence:** UI content belongs in the reproducibility manifest.

#### Drill V — audit retention conflict

- **Input:** `D-22`, operations asks to retain raw inputs indefinitely for model
  debugging; approved retention policy allows only a shorter period.
- **Question:** Does debugging value override retention?
- **Oracle:** no; retain only authorized minimum artifacts or use approved controls.
- **Why:** more logging can expand exposure without improving a bounded audit.
- **Required log:** retention rule, fields retained, purpose, access role,
  deletion schedule, and aggregate diagnostic alternative.
- **Recovery:** minimize logs and escalate policy conflict to the owner.
- **Failure signal:** debug storage contains unnecessary raw sensitive content.
- **Decision consequence:** observability has its own data-governance boundary.

#### Drill W — score drift versus workflow drift

- **Input:** `D-23`, score distribution is unchanged, but acknowledgment time
  doubles after a staffing change.
- **Question:** Is model monitoring alone sufficient?
- **Oracle:** no; the intervention path degraded despite stable scores.
- **Why:** support value depends on review and timely action, not just output.
- **Required log:** queue age, acknowledgment latency, disposition, staffing context,
  policy version, and baseline-route use.
- **Recovery:** restore workflow capacity or change governed routing; do not retune
  the model before isolating the operational cause.
- **Failure signal:** stable AUROC dashboard alongside growing queue age.
- **Decision consequence:** monitor end-to-end decision states.

#### Drill X — algorithmic confidence misread as certainty

- **Input:** `D-24`, a UI shows `0.82` with no definition, calibration context,
  or uncertainty statement.
- **Question:** What does the number authorize?
- **Oracle:** nothing by itself; the display is insufficiently specified.
- **Why:** a score may be rank, probability estimate, or arbitrary model output.
- **Required log:** score semantics, population/version, display copy, threshold,
  evidence panel, and intended reviewer action.
- **Recovery:** label or suppress the value according to governed intended use.
- **Failure signal:** users report treating the number as a diagnosis.
- **Decision consequence:** numerical displays need an action and evidence contract.

### 11.12 Minimum acceptance record for an illustrative walkthrough

Before a teaching trace is called internally consistent, record these distinct
checks rather than claiming a clinical or production result:

1. **Arithmetic check:** `90+10=100`, `90/100=0.90`, `90+210=300`, and
   `300-240=60`; units are encounters or alerts, never percentages.
2. **Time check:** every included feature has an availability proof at or before
   its individual decision time; event-time plausibility is not a substitute.
3. **Authority check:** the output cannot create an order, diagnosis, or other
   binding action within the illustrative interface contract.
4. **Route check:** each abstention has a reason, recipient, deadline, escalation,
   and one final disposition.
5. **Cohort check:** target status is mature, decision keys are unique, and
   inclusion/exclusion counts precede metric reporting.
6. **Version check:** input schema, transform, policy, display, and route versions
   can reconstruct what a reviewer saw.
7. **Capacity check:** arrivals, review minutes, queue age, and expired tasks are
   assessed against the declared workflow rather than assumed away.
8. **Evidence check:** all teaching values are marked ILLUSTRATIVE and no named
   provider, payer, product, or regulator is implied to use the design.

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
