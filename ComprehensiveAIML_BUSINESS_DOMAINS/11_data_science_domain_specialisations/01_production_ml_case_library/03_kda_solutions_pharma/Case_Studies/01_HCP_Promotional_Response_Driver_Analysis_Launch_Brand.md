---
case_id: "CASE-01-03-001"
case_title: "HCP promotional-response drivers for a fictional launch"
case_version: "1.0"
created: "2026-11-07"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "analytical"
domain: "Pharma commercial analytics"
primary_pattern: "causal-confounding"
secondary_patterns: ["evaluation-mismatch", "label-delay"]
evidence_level: "not-measured"
source_basis: "illustrative"
as_of_date: "2026-11-07"
owner_role: "Commercial decision owner"
source_index: "INDEX.md"
review_trigger: "A proposed launch, access rule, data source, consent rule, or study protocol changes"
---
# HCP promotional-response drivers for a fictional launch

## 1. The One-Line Framing
Given a weekly eligible healthcare professional (HCP) roster and pre-decision information, decide whether to propose an approved educational contact or retain usual scheduling for a fictional launch, subject to access, consent, privacy, and independent review; compare with the existing illustrative manual schedule.
The proposed decision is channel allocation, not treatment selection for a patient.
The commercial decision owner may approve, reject, or defer a reviewed experiment.
The primary success condition is a prespecified incremental eligible prescriber response over a fixed follow-up, with safety and operational guardrails.
The source directory contents are unavailable; the directory title is a topic lead, not evidence of a real project.
All entities, counts, decisions, data, and calculations below are constructed teaching assumptions.
No company, medicine, actual intervention, legal conclusion, or realized causal impact is asserted.
A descriptive driver is a correlate conditional on the model and population; it is not automatically an effect of changing that driver.
The material tradeoff is that the contacts most predictive of prescribing may have been deliberately directed to people already expected to prescribe.

## 2. Context & Background
A fictional launch team distributes approved information to eligible HCPs using a manual weekly schedule.
An HCP is a healthcare professional; an eligible prescriber is a rostered HCP satisfying all scenario-specific gates below.
The HCP independently makes clinical decisions; the team must not use this analysis to determine care.
The hypothetical commercial owner controls whether to seek a test of an allocation policy.
A field operations lead controls scheduling capacity and records delivery separately from assignment.
A medical, legal, regulatory, and privacy review group must evaluate the content, eligibility, study, and data flows before any use.
Those role names describe organizational review, not a conclusion that a proposed activity is legally compliant.
The analytics lead owns the estimand and analysis plan; data engineering owns source reconciliation.
The privacy lead owns access decisions and any approved linkage.
The HCP may decline contact; a patient is never contacted or individually targeted by this case.
The illustrative geography is a single unspecified market with one language and one weekly scheduling cycle.
No actual market, regulatory regime, product, launch date, prior results, or access arrangement is established.
The assumed baseline uses territory staff judgment and fixed capacity, with no known randomized comparison.
The baseline persists in this scenario because it is operationally understandable and supports local knowledge.
Whether it is effective, equitable, or compliant is unknown and requires separate review.
The proposed test contrasts an offer of an approved educational contact with the usual scheduling policy.
Both arms retain any independent clinical information and clinical decision making outside this study.

## 3. The Business Question
Illustrative question: should the owner allocate a limited incremental contact opportunity to eligible HCPs, or continue usual scheduling until stronger evidence exists?
The decision occurs before a hypothetical launch quarter; there is no actual deadline or approved launch.
Available choices are no change, a descriptive audit, or a reviewed randomized policy test.
The objective is the mean difference in HCP-level binary response over a prespecified 8-week outcome window.
Response is an analytical proxy, not a measure of patient health or appropriate care.
A response is at least one approved, observable qualifying prescribing event, if an approved data source can support that definition.
If the source cannot distinguish eligible events or appropriate use, the proxy must be revised before randomization.
The decision does not authorize content, clinical recommendations, reimbursement changes, or patient targeting.
High prescribing volume alone must not override consent, exclusion rules, clinical independence, or channel restrictions.
The owner may defer if review, source permissions, statistical precision, or capacity is insufficient.

## 4. Assumptions & Constraints
| ID | Scenario assumption or constraint | Verification and owner | Consequence if false |
|---|---|---|---|
| A-01 | A versioned HCP roster can be joined without patient identifiers | Data steward audits key history | Restrict to linkable population; report selection |
| A-02 | Outcome events mature within 12 weeks after follow-up | Outcome steward tests historical revisions | Extend maturity cutoff or do not analyze |
| A-03 | Eligible HCPs can be randomized without changing clinical care | Review group and operations assess protocol | Do not randomize; retain descriptive analysis |
| A-04 | Contact capacity is stable across study arms | Operations checks schedules | Restratify or revise the intervention |
| A-05 | Interference between HCPs is limited | Study lead maps practices and shared staff | Randomize by practice and use cluster inference |
| C-01 | Contact requires current permission and channel access | Privacy and operations verify before each contact | Suppress contact; retain assignment for intention-to-treat |
| C-02 | Only reviewed content may be offered | Content owner verifies version and review status | Stop delivery on missing approval |
| C-03 | Patient-level identifiers are unavailable to commercial users | Privacy lead approves architecture | Block linkage and export |
| C-04 | Small cells cannot be released without review | Privacy lead sets disclosure thresholds | Suppress or combine cells under approved policy |
| C-05 | Clinical judgment must remain independent | Medical reviewer defines prohibited uses | Stop inappropriate use and escalate |
The assumptions most likely to reverse the recommendation are source access, ability to randomize ethically and operationally, and outcome maturity.
Consent, access, and appropriateness are not inferred from the mere existence of a contact record.
Exact retention periods and jurisdiction-specific rules remain undecided until authorized reviewers determine them.

## 5. Data, Tools & Method
### 5a. Data Specification
The decision grain is one HCP at the first eligible weekly assignment in the test.
The constructed design uses 2,000 distinct HCPs, assigned 1,000 per arm; no such roster has been observed.
Eligibility requires a valid de-duplicated HCP key, active launch-relevant role, approved market, current channel permission, and no exclusion at assignment.
Exclude unavailable or declined channels, expired permission, inaccessible HCPs, unresolved identity, and HCPs already assigned during the study.
The population is eligible reachable HCPs, not all prescribers, patients, territories, or jurisdictions.
The assignment timestamp freezes pre-assignment characteristics; later changes cannot enter baseline covariates.
The primary outcome equals one if at least one qualifying event occurs from day 1 through day 56 after assignment, otherwise zero, once the source is mature.
Events on day 0 are excluded to prevent ambiguous ordering; a prespecified timestamp and timezone convention resolves boundaries.
Missing outcome coverage is not equivalent to zero; report coverage by arm and perform prespecified sensitivity analysis.
Prescriber outcomes must come from an authorized, appropriately aggregated source, with no patient identity passed to the commercial decision system.
The expected 12-week maturity lag after day 56 is hypothetical and must be empirically verified before analysis.
Assignment and response are linked one-to-one by an approved pseudonymous HCP key; outcomes may have many events per HCP.
Join assignment to aggregated outcomes only after deduplication, eligibility freeze, and event-time filtering.
A mapping vault, if authorized, is separated from analyst and scheduler roles; no reversible mapping is exported.
A roster inclusion flow records counts and reasons at every filter, with restricted low-count reporting.

### 5b. Tools & Stack
A proposed SQL-capable restricted warehouse stores immutable assignment snapshots and reconciled event aggregates.
A version-controlled statistical notebook or script computes prespecified contrasts and uncertainty; software versions are not selected yet.
A separate access-controlled scheduling system stores the permission and action records needed for delivery.
The reviewed study registry holds protocol, version, approvals, audit metadata, and analysis code references.
No named tool, platform, deployment, runtime version, or data pipeline has been verified or implemented.

### 5c. Method Summary
The descriptive audit estimates associations between past contact features and subsequent response using pre-contact covariates and explicit missingness.
The baseline is the existing manual scheduling policy, not an untreated universe.
The randomized proposal estimates an intention-to-treat effect of assignment to an incremental approved offer versus usual scheduling.
Usual scheduling may itself include contacts; the estimand is a policy contrast, not the effect of zero versus one exposure.
Assignment is by HCP only if review establishes that cross-HCP spillovers are acceptably limited.
If practice-level sharing is substantial, randomize practices and account for clustering and fewer independent units.
A delivered-contact comparison is biased if HCP availability or engagement affects receipt and response.
An observational adjustment using pre-contact confounders may be considered only if randomization is infeasible and data permissions allow it.
That alternative requires consistency, conditional exchangeability, positivity, adequate measurement, correct timing, and an explicit target population.
Unmeasured rep knowledge, launch readiness, payer access, and HCP interest can defeat exchangeability.
Inverse probability weighting cannot rescue HCPs with no realistic chance of receiving one of the policies.
A purely descriptive model remains useful for auditing workflow, but should not direct causal allocation without supporting evidence.

### 5d. Provenance, Quality, Exclusions and Leakage Controls
The only source lead is the unavailable directory listing; it supplies no project measurements, methods, or organizational facts.
Eligibility, contact, prescribing, and permissions sources are hypothetical and must each have an authorized owner and approved access basis.
Record source version, extraction time, event time, arrival time, correction policy, and transformation revision.
Do not use post-assignment opens, deliveries, visits, claims, or subsequent eligibility as baseline adjustment variables.
Changes in source coverage, identity mapping, or extract delay can mimic a treatment difference.
Complete-case analysis changes the population if outcome observation depends on assignment or engagement.
Preserve excluded counts and reasons; disclose that opt-outs and inaccessible HCPs are outside the estimand.
No patient narratives, patient identifiers, individual clinical records, or unreviewed sensitive attributes are required by the proposed analysis.
Review privacy and disclosure controls for HCP-level information as well as patient information.

### 5e. Reproducibility and Method-Selection Ledger
| ID | Artifact | Status and limitation |
|---|---|---|
| SRC-01 | Original topic directory contents | Unavailable; topic label only |
| ART-01 | Hypothetical source data and mapping | Not available or approved |
| PLAN-01 | Randomization and analysis protocol | Proposed; not registered or approved |
| CALC-01 | Arithmetic in Section 6 | Constructed figures; reproducible from shown formulas |
| RUN-01 | Actual evaluation run | None exists |
| CLM-01 | No causal outcome measured | Supported by absence of study; Sections 7, 9 and 11 |
| CLM-02 | Illustrative 5-point contrast has a wide interval | Arithmetic only; not empirical evidence |
Before an actual run, preserve protocol version, source snapshots, study configuration, randomization seed, code revision, dependency lock, reconciliation report, and authorization record.
A seed alone cannot reproduce unavailable inputs or prove identical execution across environments.

## 6. Analysis Walkthrough
The decisive finding is a limitation: the numerical example shows how a seemingly positive difference can be too imprecise for a confident decision.
First freeze the eligible roster and all gates before revealing assignment.
Second randomize equal numbers within prespecified territory and channel-availability strata, subject to the interference decision.
Third offer the reviewed contact only to the assigned intervention arm; record assignment independently of receipt.
Fourth collect outcomes through day 56 and wait for the prespecified maturity cutoff.
Fifth reconcile denominators, delivery, missingness, and sample-ratio balance without excluding nonreceivers.
Sixth calculate the difference in binary HCP response rates between assigned arms.
All arithmetic below is hypothetical; no study has been conducted.
Suppose 250 of 1,000 assigned to the offer respond and 200 of 1,000 assigned to usual scheduling respond.
The offer-arm response rate is 250/1,000 = 0.25, or 25%.
The usual-policy response rate is 200/1,000 = 0.20, or 20%.
The difference is 0.25 - 0.20 = 0.05, or 5 percentage points, not 5 percent relative.
The relative ratio is 0.25/0.20 = 1.25, corresponding to a 25% relative increase in this constructed example.
Under independent HCP randomization and a normal approximation, the standard error is sqrt(0.25×0.75/1000 + 0.20×0.80/1000) = 0.01864.
An approximate 95% interval is 0.05 ± 1.96×0.01864 = [0.0135, 0.0865], or 1.35 to 8.65 percentage points.
This interval assumes independent HCPs, complete outcomes, and the specified analysis; it is not a cluster-robust interval.
If practices share information, the effective sample size is lower and that interval can be too narrow.
For illustration, a design effect of 2 would multiply the standard error by sqrt(2), giving approximately 0.02636 and an interval of [-0.0017, 0.1017].
That sensitivity is a rough variance illustration, not a substitute for a cluster-randomized analysis.
The interval now includes zero, so the assumed independence can change whether a benefit is distinguishable from zero.
A second sensitivity changes the offer-arm response count to 220 while retaining 200 controls: the contrast becomes 2 percentage points.
Its independent standard error is sqrt(0.22×0.78/1000 + 0.20×0.80/1000) ≈ 0.01821; the approximate interval is [-1.57, 5.57] percentage points.
The plausible result would be inconclusive under the proposed 2-point minimum practical improvement rule, not a demonstrated failure or benefit.
No approach has actually been tested; there is no observed historical failed attempt to report.
A historical positive association between contact and response could instead reflect deliberate targeting of already receptive HCPs.
For example, if 80/200 contacted HCPs respond and 100/800 uncontacted HCPs respond, the descriptive difference is 40% - 12.5% = 27.5 points.
Those constructed groups differ in who was selected; this contrast is not the effect of contacting the uncontacted group.
The highest-leverage step is ensuring assignment precedes exposure and outcome while preserving valid eligibility and denominator information.

## 7. The Finding
Do not change allocation on the basis of descriptive drivers alone; seek review of the proposed policy experiment or retain usual scheduling.
No intervention outcome has been measured (CLM-01).
The constructed 5-point contrast and its interval (CLM-02) demonstrate calculation only, while the cluster sensitivity shows why precision depends on the independent assignment unit.
The next decision is whether the owner and independent reviewers can authorize a feasible design with reliable consent and mature outcomes.

## 8. System/Solution Design
### 8a. Offline / Training / Batch Path
```text
[Approved roster + permissions + scheduling records] -> [restricted validation and eligibility freeze]
           -> [reviewed randomization ledger] -> [approved scheduling handoff]
[authorized outcome aggregates] -> [maturity and identity reconciliation] -> [locked analysis]
           -> [disclosure review] -> [owner decision; no automated clinical action]
```
The study lead signs the protocol before data engineering creates an assignment snapshot.
The eligibility job fails closed on missing permission, invalid channel access, or unresolved identities.
Randomization writes an immutable assignment and version; retries use a stable study/HCP key to avoid reassignment.
The scheduler independently checks current permission before any contact, so withdrawal after assignment suppresses delivery.
The analyst retains the original assignment denominator, subject to approved retention and privacy requirements.
Corrections to source records are versioned rather than silently overwriting the original snapshot.
No predictive model is trained or served in the primary experiment.

### 8b. Online / Serving / Real-Time Path
There is no real-time model endpoint in the proposed weekly batch design.
A scheduler request authenticates the staff role, retrieves the current permission state, and checks content version and channel access.
A valid response returns the approved scheduling instruction and a pseudonymous study assignment reference.
A stale, unavailable, or ambiguous permission response returns no contact instruction and creates a restricted operational alert.
A declined contact is honored immediately; the staff member cannot bypass the gate by retrying.
A delivery callback records offer, delivery, failure, and timestamps without claiming that a contact caused an outcome.
The outcome system feeds aggregate response only after the follow-up and maturity window.

### 8c. Key Design Decisions
| ID | Proposed choice | Alternative | Boundary and trigger |
|---|---|---|---|
| DEC-01 | Randomize policy assignment | Historical regression | Use observational analysis only with defensible identification and overlap |
| DEC-02 | Preserve intention-to-treat | Compare actual recipients | Recipient comparison is selected by willingness and availability |
| DEC-03 | Batch weekly | Real-time personalization | Revisit only if a reviewed workflow requires lower latency |
| DEC-04 | Suppress on unknown permission | Assume roster permission persists | Consent source outage stops incremental delivery |
| DEC-05 | Aggregate outcome before analyst access | Direct patient records | No patient-level access is needed for the decision |

### 8d. Architecture Artifacts and Operational Readiness
The proposed workload has 2,000 assignment records and at most one initial offer per intervention HCP.
At 1 kilobyte per assignment record, raw assignment storage is 2,000 kilobytes, approximately 2 MB in decimal units, before indexes, replicas, logs, and outcomes.
This is a sizing assumption, not a measured footprint or a reason to omit security and recovery planning.
No validated infrastructure pricing, compute benchmark, throughput, staffing cost, or cost per outcome is available.
The operations lead owns scheduling availability; the data steward owns completeness and reconciliation; privacy owns access and retention approvals.
A proposed service-level indicator is the fraction of eligible scheduling requests receiving a valid permission decision before the scheduling cutoff.
The target and measurement window are not set and must be agreed before deployment; no service-level objective is claimed.
Bounded retries must not duplicate offers; permission outages suppress new delivery rather than retrying indefinitely.
Backups and restoration must preserve assignment identity, consent state history, and source lineage under the approved retention policy.
A replay must not reissue a contact; recover using immutable assignment and idempotent delivery keys.
Audit records exclude patient identifiers and must be accessible only to authorized staff.
The owner may stop a test for privacy, safety, consent, data-integrity, or content-review concerns; cessation does not erase historical randomization.
No proposed infrastructure, recovery behavior, or approval has been validated in production.

## 9. Evaluation & Validation
The primary metric is the difference in proportions of randomized eligible HCPs with at least one qualifying event in days 1–56, after data maturity.
The numerator is the number with a qualifying event; the denominator is all randomized HCPs in each arm, including nondelivered offers.
The comparator is assignment to usual scheduling in the same period and eligibility population.
Before a real test, the owner would specify a minimum practically relevant improvement of 2 percentage points and an interval procedure matched to the randomization.
A proposed success rule requires the lower bound of a two-sided 95% confidence interval to exceed 2 percentage points and all guardrails to pass.
This rule is an illustrative business threshold, not a statistical power calculation or an approved protocol.
The constructed 5-point example has a 1.35-point lower bound and would not satisfy that rule despite excluding zero under independent sampling.
The size of 2,000 is a teaching assumption, not an adequate-powered sample determination.
A statistician must perform prospective power and precision calculations using anticipated baseline rate, minimum effect, cluster structure, attrition, and feasible enrollment.
Check assignment ratios overall and by prespecified stratum; investigate any mismatch against the randomization ledger.
Check actual offered and delivered contacts as process measures, but do not substitute them for randomized assignment.
Measure permission failures, withdrawals, inappropriate contact reports, content deviations, cost, and source coverage as guardrails.
Privacy and medical reviewers set acceptable guardrail rules before testing; no legal approval is implied.
Report missing outcomes by arm and reasons, followed by prespecified plausible-range sensitivity analyses.
Use cluster-aware uncertainty for practice randomization or material dependence; avoid treating all correlated HCPs as independent.
Any subgroup contrasts are exploratory unless specifically powered and prespecified, and require disclosure review.
Do not stop early merely because the interim result looks favorable; any sequential design needs a prespecified valid stopping rule.
The proposed protocol has not been executed; no pass, failure, or causal finding exists.

## 10. Action Taken
No action taken as of 2026-11-07: this is an illustrative design exercise.
The proposed next action is to obtain review of eligibility, permissible data, content, protocol, and clinical independence.
No owner approval, release, pilot, operational handoff, or deployment record exists.
If approved later, operations would use a limited reviewed test and the owner could stop or revert to usual scheduling.
A rollback cannot undo contacts already delivered; it stops future offers and preserves approved audit evidence.
Descriptive scores, unapproved channels, and patient-specific targeting are excluded from the proposed action.

## 11. Outcome & Results
Outcome not yet measured; no real intervention or analysis run has occurred.
Figures in Section 6 are constructed teaching values, not measurements of an organization or evidence of realized impact.
| Metric | Baseline | Intervention | Change | Uncertainty | Status |
|---|---|---|---|---|---|
| 8-week eligible-HCP response | Not measured | Not measured | Not measured | Not available | CLM-01; proposed randomized comparison |
| Consent and contact guardrails | Not measured | Not measured | Not measured | Not available | Approval and operational sources absent |
| Cost and reviewer effort | Not measured | Not measured | Not measured | Not available | No actual deployment |
The analytics lead would measure the primary metric after the 8-week window and verified source maturity.
The commercial owner would use the prespecified success and guardrail rules to decide whether to continue, revise, or stop.
A positive proxy response would not establish improved clinical appropriateness, patient benefit, or compliance.

## 12. Failure Modes & What Went Wrong
No incident is claimed; all following cases are hypothetical failure diagnostics.
| ID | Symptom and competing explanation | Detection | Containment and owner |
|---|---|---|---|
| F-01 | Contacted HCPs respond more; selection or actual effect may explain it | Compare randomized assignments, not recipients | Do not scale; analytics lead |
| F-02 | Some HCPs receive duplicate offers after retry | Reconcile delivery keys and callbacks | Pause dispatch; operations lead |
| F-03 | Consent status is stale or missing | Compare permission timestamp with scheduling cutoff | Suppress and investigate; privacy lead |
| F-04 | Offer arm has fewer observable outcomes | Compare coverage and source delays by arm | Hold decision and run missingness sensitivity; data steward |
| F-05 | Same practice exchanges study material | Map shared practices and contacts | Stop inference based on independent HCPs; study lead |
| F-06 | Event arrives after outcome maturity cutoff | Review revision curves and delayed event volume | Extend cutoff under amended protocol; data steward |
| F-07 | The scheduler receives unreviewed content | Validate signed content version against registry | Stop offers; content owner |
| F-08 | A summary reveals small HCP cells | Review disclosure and aggregation | Block export; privacy lead |
| F-09 | Rep changes usual-care contact intensity in control | Compare contact logs by assignment and territory | Document policy deviation; study lead |
| F-10 | Result meets statistical but not practical threshold | Compare interval lower bound to decision rule | Defer scale; owner |
These diagnostics are proposals and have not been validated against an incident or historical system.

## 13. Transferable Lesson
The canonical pattern is causal-confounding: past allocation can make exposure predictive without making it causally effective.
A decision to intervene requires a well-defined policy contrast and a defensible comparison, not merely a feature ranking.
The constructed arithmetic demonstrates interpretation, not a real effect or a universal operating rule.
This principle transfers only when eligibility, intervention, timing, outcome and alternative can be specified.
Randomization is not appropriate if review finds the offer impermissible, disrupts care, or cannot be delivered as assigned.
In another domain, replace HCP-specific access and privacy controls with the relevant authorized population and harm constraints.
The first portability question is whether the current policy selects recipients using information related to the outcome.

## 14. Connections & Cross-References
### 14a. Related cases, theory, and cheat sheets
The [collection index](INDEX.md) identifies planned neighboring cases; their source contents are not evidence for this example.
A comparable completed local case has not been established; do not treat a planned title as a validated comparison.
Method reference: [NIST/SEMATECH Engineering Statistics Handbook, confidence limits for the difference between two proportions](https://www.itl.nist.gov/div898/handbook/prc/section3/prc33.htm) provides context for proportions; the illustrative calculation uses the explicitly stated unpooled normal approximation.
The reference is methodology only, not a source for any company or outcome claim.
### 14b. Open Questions
- Which prescriber-level outcome source, if any, is authorized, sufficiently mature, and appropriate for this decision?
- Can the organization operationally isolate an incremental offer from usual scheduling without interfering with care?
- What practice-level spillovers and baseline response rates should determine the assignment unit and sample size?
- What consent, review, disclosure and retention rules apply in the actual market and organization?
- What clinically meaningful guardrails can reviewers define without exposing patient data?
### 14c. Interview Pack

#### Interview 1: What exactly is randomized?
- The unit is the eligible HCP if interference is acceptably limited.
- Assignment is to an incremental approved offer or usual scheduling.
- The offer arm may fail to receive contact because permission changes or capacity fails.
- Those HCPs remain in their assigned arm for intention-to-treat analysis.
- Usual scheduling may include contacts in either arm.
- Therefore the contrast is not a pure one-contact-versus-zero-contact effect.
- If practices share materials, practice-level randomization may be necessary.
- That change requires cluster-aware sample sizing and inference.

#### Interview 2: Why is a promotional driver not necessarily causal?
- A driver may merely predict outcomes in a historical dataset.
- Staff can target HCPs who already have high expected response.
- The same prior propensity then influences both contact and outcome.
- Comparing contacted and uncontacted prescribers mixes targeting with any contact effect.
- Adjusting for measured baseline differences helps only under strong conditions.
- Unrecorded staff knowledge may still confound the association.
- A randomized policy contrast breaks this assignment pathway if implemented correctly.
- It still does not prove the mechanism by which any effect arose.

#### Interview 3: How do you define the population?
- Freeze the de-duplicated active HCP roster before assignment.
- Require role, market, approved channel access, and current permission.
- Exclude unresolved identity and any mandatory suppression.
- Record exclusions with counts and restricted reason codes.
- The estimand applies to eligible, reachable HCPs under these rules.
- It does not represent opt-outs or inaccessible prescribers.
- Do not condition the primary analysis on later delivery.
- Reassess generalizability if the roster or channel permissions change.

#### Interview 4: What is the correct time ordering?
- A pre-assignment timestamp closes the baseline feature window.
- Randomization precedes the incremental offer and outcome.
- The outcome window runs from day 1 through day 56.
- Data must mature before the analyst calls an absent event a zero.
- Later engagement and eligibility cannot be treated as baseline confounders.
- Source event time and arrival time must both be retained.
- Timezone and cutoff semantics require protocol specification.
- A delayed extract can otherwise masquerade as a difference in response.

#### Interview 5: Compute and interpret the example.
- The offer arm has 250 responses among 1,000 assigned HCPs.
- The usual-policy arm has 200 responses among 1,000 assigned HCPs.
- Their rates are 25% and 20%, respectively.
- The difference is 5 percentage points and the relative increase is 25%.
- The approximate independent standard error is 1.864 percentage points.
- The corresponding 95% interval is approximately 1.35 to 8.65 points.
- It does not clear the proposed two-point lower-bound threshold.
- The numbers are fabricated for instruction and establish no real impact.

#### Interview 6: What changes if practice spillovers matter?
- A colleague can see information delivered to another HCP.
- Consequently an HCP-level assignment may affect control HCPs.
- Shared exposure breaks the simple no-interference interpretation.
- Map HCPs to practices before deciding the assignment unit.
- Randomize practices if the operational and ethical review supports it.
- Use the number and size distribution of practices for power planning.
- Estimate uncertainty using a method appropriate to the number of randomized clusters.
- Do not reuse the independent-HCP interval as if the cluster issue vanished.

#### Interview 7: How should consent withdrawal be handled?
- Suppress future offers as soon as current permission is withdrawn.
- Do not override the withdrawal for the sake of protocol adherence.
- Retain assignment for intention-to-treat only where authorized.
- The appropriate retention or deletion approach requires privacy review.
- Record a restricted withdrawal event and its effective time.
- Investigate whether system latency could cause an unauthorized offer.
- Analyze delivery as a process outcome, not as the randomization denominator.
- Do not infer consent from a prior successful contact.

#### Interview 8: What does positivity mean here?
- For every covariate-defined group in the target population, both policies must be feasible under the identification strategy.
- Randomization can ensure nonzero assignment probability where both options are allowed.
- It cannot make an inaccessible or nonconsenting channel permissible.
- Exclude such HCPs from the channel-specific eligible population before assignment.
- In observational work, inspect propensity distributions and extreme weights.
- Overlap diagnostics cannot reveal every unmeasured confounder.
- Restricting to overlap changes the target population and must be reported.
- Do not extrapolate the resulting estimate to excluded HCPs.

#### Interview 9: What does missing outcome data change?
- Missing coverage is not a nonresponse.
- Compare source availability and maturity by assignment arm.
- Investigate whether the missingness could be influenced by assignment.
- Prespecify sensitivity bounds or appropriate missing-data methods.
- Report how much missingness is necessary to reverse the decision.
- Avoid silently dropping HCPs with incomplete linkage.
- Dropping them can change the target population and break randomization.
- Delay the owner decision if source coverage is not credible.

#### Interview 10: How should descriptive attribution be reported?
- State the association, population, timing, model, and conditioning variables.
- Describe feature attribution as model-dependent predictive explanation.
- Do not call a large coefficient or importance value an uplift.
- Check whether the feature was known before the decision.
- Investigate leakage from subsequent visits or outcomes.
- Compare with simple descriptive tables and a baseline model.
- Report instability and missingness rather than ranking features with false precision.
- Any proposed causal action needs additional identification evidence.

#### Interview 11: What if assignment and delivery disagree?
- Assignment is the randomized intention-to-treat exposure.
- Delivery records measure implementation and adherence.
- Staff availability, HCP access, or withdrawal can prevent delivery.
- Those causes may be associated with the outcome.
- A direct recipient-versus-nonrecipient comparison is therefore selected.
- An instrumental-variable estimate would need additional exclusion and monotonicity assumptions.
- The offer could influence behavior through channels other than recorded delivery.
- Report the operational delivery rate without claiming a delivered-contact effect.

#### Interview 12: What is the baseline and why does it matter?
- The comparator is usual scheduling, not no information or no contact.
- The study estimates the difference between two operational policies.
- Control-arm HCPs may receive contacts through normal scheduling.
- Intervention-arm HCPs may not receive the incremental offer.
- Both patterns dilute or alter the policy contrast but do not invalidate intention-to-treat by themselves.
- A policy effect cannot automatically be translated into a per-contact effect.
- Log usual scheduling without changing the treatment definition.
- If the usual policy changes mid-study, assess interpretation and protocol consequences.

#### Interview 13: How is privacy enforced?
- Use an approved pseudonymous HCP key in the analytical environment.
- Separate any identity mapping from analyst and scheduler roles.
- Aggregate authorized outcome events before commercial access.
- Do not transmit patient identities or narratives to the allocation system.
- Limit exports and apply disclosure review to small HCP groups.
- Give each role only the data needed for its task.
- Retain or delete records according to an approved policy, not an invented rule.
- Treat the architecture as proposed until privacy reviewers approve it.

#### Interview 14: How do you evaluate a nominally positive result?
- First verify randomization and source reconciliation.
- Confirm that the outcome window and maturity cutoff were honored.
- Inspect the treatment difference and confidence interval in the randomized population.
- Compare the interval lower bound with the prespecified practical threshold.
- Check consent, content, safety, operational, and privacy guardrails.
- Assess sensitivity to missingness and dependence.
- Avoid interpreting exploratory subgroups as conclusive.
- A positive point estimate alone is not an authorization to scale.

#### Interview 15: What happens when the permission service fails?
- The scheduling path returns no new contact instruction.
- The system alerts an authorized operations owner without exposing patient data.
- Retries must be bounded and idempotent.
- The scheduler must not substitute a stale permission record.
- Existing assignment remains in the protected ledger if retention is authorized.
- After recovery, current permission must be checked again before delivery.
- Reconcile missed and duplicate offers by assignment key.
- The outage and its arm-specific consequences belong in the study report.

#### Interview 16: What would change your recommendation?
- A documented inability to obtain an authorized outcome source would block the proposed test.
- Substantial spillover would require practice-level design and different precision calculations.
- A consent or content-review concern would stop the incremental offer.
- An inadequate sample size would make the proposed decision rule uninformative.
- A major change in the usual scheduling policy would change the comparator.
- An experimentally supported effect below the practical threshold would not justify scale under this rule.
- Guardrail failures would block scale even with a favorable response contrast.
- No such evidence is presently available in this fictional case.

### 14d. Proposed Data Contract and Audit Questions
These are candidate contract fields, not claims that a source contains them. Each entry identifies an independent operational check; reviewers may reject a field or access path.

#### Contract field: study identifier
- Purpose: distinguishes this protocol from another.
- Proposed steward: study registry; the accountable role must be confirmed.
- Intended use: assignment ledger; other uses need separate authorization.
- Classification: assignment; apply the access rules approved for that class.
- Change control: record corrections to study identifier instead of silently changing a frozen analysis.

#### Contract field: HCP key
- Purpose: links a person across approved HCP sources.
- Proposed steward: identity steward; the accountable role must be confirmed.
- Intended use: restricted join; other uses need separate authorization.
- Classification: identity; apply the access rules approved for that class.
- Change control: record corrections to HCP key instead of silently changing a frozen analysis.

#### Contract field: roster snapshot
- Purpose: freezes who was considered.
- Proposed steward: roster owner; the accountable role must be confirmed.
- Intended use: eligibility job; other uses need separate authorization.
- Classification: eligibility; apply the access rules approved for that class.
- Change control: record corrections to roster snapshot instead of silently changing a frozen analysis.

#### Contract field: active role
- Purpose: establishes role eligibility at assignment.
- Proposed steward: roster owner; the accountable role must be confirmed.
- Intended use: eligibility job; other uses need separate authorization.
- Classification: eligibility; apply the access rules approved for that class.
- Change control: record corrections to active role instead of silently changing a frozen analysis.

#### Contract field: market code
- Purpose: limits the authorized market.
- Proposed steward: roster owner; the accountable role must be confirmed.
- Intended use: eligibility job; other uses need separate authorization.
- Classification: eligibility; apply the access rules approved for that class.
- Change control: record corrections to market code instead of silently changing a frozen analysis.

#### Contract field: channel eligibility
- Purpose: indicates whether a channel can be used.
- Proposed steward: channel operations; the accountable role must be confirmed.
- Intended use: eligibility job; other uses need separate authorization.
- Classification: eligibility; apply the access rules approved for that class.
- Change control: record corrections to channel eligibility instead of silently changing a frozen analysis.

#### Contract field: permission status
- Purpose: represents current channel permission.
- Proposed steward: permission owner; the accountable role must be confirmed.
- Intended use: scheduling gate; other uses need separate authorization.
- Classification: permission; apply the access rules approved for that class.
- Change control: record corrections to permission status instead of silently changing a frozen analysis.

#### Contract field: permission effective time
- Purpose: locates the start of a permission state.
- Proposed steward: permission owner; the accountable role must be confirmed.
- Intended use: scheduling gate; other uses need separate authorization.
- Classification: permission; apply the access rules approved for that class.
- Change control: record corrections to permission effective time instead of silently changing a frozen analysis.

#### Contract field: withdrawal effective time
- Purpose: locates when permission ceases.
- Proposed steward: permission owner; the accountable role must be confirmed.
- Intended use: scheduling gate; other uses need separate authorization.
- Classification: permission; apply the access rules approved for that class.
- Change control: record corrections to withdrawal effective time instead of silently changing a frozen analysis.

#### Contract field: suppression reason
- Purpose: explains a blocked contact.
- Proposed steward: permission owner; the accountable role must be confirmed.
- Intended use: restricted audit; other uses need separate authorization.
- Classification: permission; apply the access rules approved for that class.
- Change control: record corrections to suppression reason instead of silently changing a frozen analysis.

#### Contract field: identity confidence
- Purpose: flags ambiguous person linkage.
- Proposed steward: identity steward; the accountable role must be confirmed.
- Intended use: restricted join; other uses need separate authorization.
- Classification: identity; apply the access rules approved for that class.
- Change control: record corrections to identity confidence instead of silently changing a frozen analysis.

#### Contract field: practice key
- Purpose: groups HCPs potentially sharing information.
- Proposed steward: identity steward; the accountable role must be confirmed.
- Intended use: design review; other uses need separate authorization.
- Classification: identity; apply the access rules approved for that class.
- Change control: record corrections to practice key instead of silently changing a frozen analysis.

#### Contract field: territory key
- Purpose: defines a possible randomization stratum.
- Proposed steward: operations; the accountable role must be confirmed.
- Intended use: randomizer; other uses need separate authorization.
- Classification: assignment; apply the access rules approved for that class.
- Change control: record corrections to territory key instead of silently changing a frozen analysis.

#### Contract field: assignment time
- Purpose: establishes treatment ordering.
- Proposed steward: study lead; the accountable role must be confirmed.
- Intended use: analysis cutoff; other uses need separate authorization.
- Classification: assignment; apply the access rules approved for that class.
- Change control: record corrections to assignment time instead of silently changing a frozen analysis.

#### Contract field: assignment arm
- Purpose: defines the randomized policy.
- Proposed steward: study lead; the accountable role must be confirmed.
- Intended use: analysis; other uses need separate authorization.
- Classification: assignment; apply the access rules approved for that class.
- Change control: record corrections to assignment arm instead of silently changing a frozen analysis.

#### Contract field: randomization stratum
- Purpose: records the prespecified balancing group.
- Proposed steward: study lead; the accountable role must be confirmed.
- Intended use: analysis; other uses need separate authorization.
- Classification: assignment; apply the access rules approved for that class.
- Change control: record corrections to randomization stratum instead of silently changing a frozen analysis.

#### Contract field: randomization seed reference
- Purpose: identifies the protected randomization configuration.
- Proposed steward: study lead; the accountable role must be confirmed.
- Intended use: reproduction; other uses need separate authorization.
- Classification: assignment; apply the access rules approved for that class.
- Change control: record corrections to randomization seed reference instead of silently changing a frozen analysis.

#### Contract field: assignment version
- Purpose: identifies the frozen randomization algorithm.
- Proposed steward: study lead; the accountable role must be confirmed.
- Intended use: audit; other uses need separate authorization.
- Classification: assignment; apply the access rules approved for that class.
- Change control: record corrections to assignment version instead of silently changing a frozen analysis.

#### Contract field: assignment idempotency key
- Purpose: prevents duplicate assignment on retry.
- Proposed steward: engineering; the accountable role must be confirmed.
- Intended use: assignment ledger; other uses need separate authorization.
- Classification: assignment; apply the access rules approved for that class.
- Change control: record corrections to assignment idempotency key instead of silently changing a frozen analysis.

#### Contract field: protocol version
- Purpose: identifies the governing study plan.
- Proposed steward: study lead; the accountable role must be confirmed.
- Intended use: audit; other uses need separate authorization.
- Classification: assignment; apply the access rules approved for that class.
- Change control: record corrections to protocol version instead of silently changing a frozen analysis.

#### Contract field: content version
- Purpose: identifies the material offered.
- Proposed steward: content owner; the accountable role must be confirmed.
- Intended use: approval gate; other uses need separate authorization.
- Classification: delivery; apply the access rules approved for that class.
- Change control: record corrections to content version instead of silently changing a frozen analysis.

#### Contract field: content approval status
- Purpose: indicates whether the exact version can be offered.
- Proposed steward: content owner; the accountable role must be confirmed.
- Intended use: approval gate; other uses need separate authorization.
- Classification: delivery; apply the access rules approved for that class.
- Change control: record corrections to content approval status instead of silently changing a frozen analysis.

#### Contract field: content approval expiry
- Purpose: bounds the approved content window.
- Proposed steward: content owner; the accountable role must be confirmed.
- Intended use: approval gate; other uses need separate authorization.
- Classification: delivery; apply the access rules approved for that class.
- Change control: record corrections to content approval expiry instead of silently changing a frozen analysis.

#### Contract field: offer timestamp
- Purpose: records the time of the incremental offer.
- Proposed steward: operations; the accountable role must be confirmed.
- Intended use: delivery audit; other uses need separate authorization.
- Classification: delivery; apply the access rules approved for that class.
- Change control: record corrections to offer timestamp instead of silently changing a frozen analysis.

#### Contract field: offer channel
- Purpose: records the contact route actually offered.
- Proposed steward: operations; the accountable role must be confirmed.
- Intended use: delivery audit; other uses need separate authorization.
- Classification: delivery; apply the access rules approved for that class.
- Change control: record corrections to offer channel instead of silently changing a frozen analysis.

#### Contract field: delivery timestamp
- Purpose: records the time of completed delivery.
- Proposed steward: operations; the accountable role must be confirmed.
- Intended use: delivery audit; other uses need separate authorization.
- Classification: delivery; apply the access rules approved for that class.
- Change control: record corrections to delivery timestamp instead of silently changing a frozen analysis.

#### Contract field: delivery status
- Purpose: distinguishes offered, delivered, and failed contacts.
- Proposed steward: operations; the accountable role must be confirmed.
- Intended use: process report; other uses need separate authorization.
- Classification: delivery; apply the access rules approved for that class.
- Change control: record corrections to delivery status instead of silently changing a frozen analysis.

#### Contract field: delivery failure reason
- Purpose: distinguishes capacity, technical, and access failures.
- Proposed steward: operations; the accountable role must be confirmed.
- Intended use: process report; other uses need separate authorization.
- Classification: delivery; apply the access rules approved for that class.
- Change control: record corrections to delivery failure reason instead of silently changing a frozen analysis.

#### Contract field: contact idempotency key
- Purpose: guards against duplicate dispatch.
- Proposed steward: engineering; the accountable role must be confirmed.
- Intended use: dispatch service; other uses need separate authorization.
- Classification: delivery; apply the access rules approved for that class.
- Change control: record corrections to contact idempotency key instead of silently changing a frozen analysis.

#### Contract field: staff role
- Purpose: identifies the authorized scheduling role.
- Proposed steward: operations; the accountable role must be confirmed.
- Intended use: access audit; other uses need separate authorization.
- Classification: delivery; apply the access rules approved for that class.
- Change control: record corrections to staff role instead of silently changing a frozen analysis.

#### Contract field: schedule cutoff
- Purpose: defines when a weekly instruction must be usable.
- Proposed steward: operations; the accountable role must be confirmed.
- Intended use: service monitoring; other uses need separate authorization.
- Classification: delivery; apply the access rules approved for that class.
- Change control: record corrections to schedule cutoff instead of silently changing a frozen analysis.

#### Contract field: permission lookup time
- Purpose: shows how current the access check was.
- Proposed steward: permission owner; the accountable role must be confirmed.
- Intended use: delivery audit; other uses need separate authorization.
- Classification: permission; apply the access rules approved for that class.
- Change control: record corrections to permission lookup time instead of silently changing a frozen analysis.

#### Contract field: baseline contact count
- Purpose: describes prior contact intensity before assignment.
- Proposed steward: contact data steward; the accountable role must be confirmed.
- Intended use: descriptive analysis; other uses need separate authorization.
- Classification: baseline; apply the access rules approved for that class.
- Change control: record corrections to baseline contact count instead of silently changing a frozen analysis.

#### Contract field: baseline contact recency
- Purpose: describes elapsed time since prior contact.
- Proposed steward: contact data steward; the accountable role must be confirmed.
- Intended use: descriptive analysis; other uses need separate authorization.
- Classification: baseline; apply the access rules approved for that class.
- Change control: record corrections to baseline contact recency instead of silently changing a frozen analysis.

#### Contract field: baseline outcome indicator
- Purpose: describes historical response before assignment.
- Proposed steward: outcome steward; the accountable role must be confirmed.
- Intended use: stratified analysis; other uses need separate authorization.
- Classification: baseline; apply the access rules approved for that class.
- Change control: record corrections to baseline outcome indicator instead of silently changing a frozen analysis.

#### Contract field: baseline data coverage
- Purpose: describes observability before assignment.
- Proposed steward: outcome steward; the accountable role must be confirmed.
- Intended use: coverage audit; other uses need separate authorization.
- Classification: baseline; apply the access rules approved for that class.
- Change control: record corrections to baseline data coverage instead of silently changing a frozen analysis.

#### Contract field: baseline access category
- Purpose: summarizes approved pre-assignment access context.
- Proposed steward: access steward; the accountable role must be confirmed.
- Intended use: descriptive analysis; other uses need separate authorization.
- Classification: baseline; apply the access rules approved for that class.
- Change control: record corrections to baseline access category instead of silently changing a frozen analysis.

#### Contract field: baseline territory capacity
- Purpose: records pre-assignment operational capacity.
- Proposed steward: operations; the accountable role must be confirmed.
- Intended use: design review; other uses need separate authorization.
- Classification: baseline; apply the access rules approved for that class.
- Change control: record corrections to baseline territory capacity instead of silently changing a frozen analysis.

#### Contract field: baseline channel preference
- Purpose: records the permitted preference known before assignment.
- Proposed steward: permission owner; the accountable role must be confirmed.
- Intended use: eligibility job; other uses need separate authorization.
- Classification: baseline; apply the access rules approved for that class.
- Change control: record corrections to baseline channel preference instead of silently changing a frozen analysis.

#### Contract field: event pseudonymous key
- Purpose: joins an approved aggregate to an HCP assignment.
- Proposed steward: outcome steward; the accountable role must be confirmed.
- Intended use: restricted join; other uses need separate authorization.
- Classification: outcome; apply the access rules approved for that class.
- Change control: record corrections to event pseudonymous key instead of silently changing a frozen analysis.

#### Contract field: outcome event time
- Purpose: places an event within or outside follow-up.
- Proposed steward: outcome steward; the accountable role must be confirmed.
- Intended use: window filter; other uses need separate authorization.
- Classification: outcome; apply the access rules approved for that class.
- Change control: record corrections to outcome event time instead of silently changing a frozen analysis.

#### Contract field: outcome arrival time
- Purpose: shows whether an event arrived after cutoff.
- Proposed steward: outcome steward; the accountable role must be confirmed.
- Intended use: maturity audit; other uses need separate authorization.
- Classification: outcome; apply the access rules approved for that class.
- Change control: record corrections to outcome arrival time instead of silently changing a frozen analysis.

#### Contract field: outcome revision time
- Purpose: records when a previously delivered event changed.
- Proposed steward: outcome steward; the accountable role must be confirmed.
- Intended use: revision audit; other uses need separate authorization.
- Classification: outcome; apply the access rules approved for that class.
- Change control: record corrections to outcome revision time instead of silently changing a frozen analysis.

#### Contract field: outcome qualification flag
- Purpose: marks events satisfying the protocol definition.
- Proposed steward: outcome steward; the accountable role must be confirmed.
- Intended use: aggregation; other uses need separate authorization.
- Classification: outcome; apply the access rules approved for that class.
- Change control: record corrections to outcome qualification flag instead of silently changing a frozen analysis.

#### Contract field: outcome coverage flag
- Purpose: distinguishes observed zero from unavailable data.
- Proposed steward: outcome steward; the accountable role must be confirmed.
- Intended use: missingness analysis; other uses need separate authorization.
- Classification: outcome; apply the access rules approved for that class.
- Change control: record corrections to outcome coverage flag instead of silently changing a frozen analysis.

#### Contract field: outcome source version
- Purpose: identifies the extract used for analysis.
- Proposed steward: outcome steward; the accountable role must be confirmed.
- Intended use: reproduction; other uses need separate authorization.
- Classification: outcome; apply the access rules approved for that class.
- Change control: record corrections to outcome source version instead of silently changing a frozen analysis.

#### Contract field: outcome aggregation version
- Purpose: identifies event-to-HCP transformation logic.
- Proposed steward: engineering; the accountable role must be confirmed.
- Intended use: reproduction; other uses need separate authorization.
- Classification: outcome; apply the access rules approved for that class.
- Change control: record corrections to outcome aggregation version instead of silently changing a frozen analysis.

#### Contract field: outcome maturity cutoff
- Purpose: records the authorized completeness deadline.
- Proposed steward: study lead; the accountable role must be confirmed.
- Intended use: analysis; other uses need separate authorization.
- Classification: outcome; apply the access rules approved for that class.
- Change control: record corrections to outcome maturity cutoff instead of silently changing a frozen analysis.

#### Contract field: follow-up start
- Purpose: defines the first included outcome time.
- Proposed steward: study lead; the accountable role must be confirmed.
- Intended use: window filter; other uses need separate authorization.
- Classification: outcome; apply the access rules approved for that class.
- Change control: record corrections to follow-up start instead of silently changing a frozen analysis.

#### Contract field: follow-up end
- Purpose: defines the last included outcome time.
- Proposed steward: study lead; the accountable role must be confirmed.
- Intended use: window filter; other uses need separate authorization.
- Classification: outcome; apply the access rules approved for that class.
- Change control: record corrections to follow-up end instead of silently changing a frozen analysis.

#### Contract field: event deduplication key
- Purpose: prevents duplicate events inflating a response.
- Proposed steward: outcome steward; the accountable role must be confirmed.
- Intended use: aggregation; other uses need separate authorization.
- Classification: outcome; apply the access rules approved for that class.
- Change control: record corrections to event deduplication key instead of silently changing a frozen analysis.

#### Contract field: source extract time
- Purpose: records when the source snapshot was retrieved.
- Proposed steward: data steward; the accountable role must be confirmed.
- Intended use: lineage audit; other uses need separate authorization.
- Classification: lineage; apply the access rules approved for that class.
- Change control: record corrections to source extract time instead of silently changing a frozen analysis.

#### Contract field: source event timezone
- Purpose: supports consistent boundary conversion.
- Proposed steward: data steward; the accountable role must be confirmed.
- Intended use: window filter; other uses need separate authorization.
- Classification: lineage; apply the access rules approved for that class.
- Change control: record corrections to source event timezone instead of silently changing a frozen analysis.

#### Contract field: source correction version
- Purpose: distinguishes corrected data from earlier snapshots.
- Proposed steward: data steward; the accountable role must be confirmed.
- Intended use: reproduction; other uses need separate authorization.
- Classification: lineage; apply the access rules approved for that class.
- Change control: record corrections to source correction version instead of silently changing a frozen analysis.

#### Contract field: access authorization reference
- Purpose: connects use to the approved access basis.
- Proposed steward: privacy lead; the accountable role must be confirmed.
- Intended use: access audit; other uses need separate authorization.
- Classification: lineage; apply the access rules approved for that class.
- Change control: record corrections to access authorization reference instead of silently changing a frozen analysis.

#### Contract field: retention policy reference
- Purpose: identifies the approved retention decision.
- Proposed steward: privacy lead; the accountable role must be confirmed.
- Intended use: lifecycle audit; other uses need separate authorization.
- Classification: lineage; apply the access rules approved for that class.
- Change control: record corrections to retention policy reference instead of silently changing a frozen analysis.

#### Contract field: export approval reference
- Purpose: records disclosure review before distribution.
- Proposed steward: privacy lead; the accountable role must be confirmed.
- Intended use: release gate; other uses need separate authorization.
- Classification: lineage; apply the access rules approved for that class.
- Change control: record corrections to export approval reference instead of silently changing a frozen analysis.

#### Contract field: small-cell suppression flag
- Purpose: blocks disallowed granular reporting.
- Proposed steward: privacy lead; the accountable role must be confirmed.
- Intended use: release gate; other uses need separate authorization.
- Classification: lineage; apply the access rules approved for that class.
- Change control: record corrections to small-cell suppression flag instead of silently changing a frozen analysis.

#### Contract field: analysis revision
- Purpose: identifies the exact analytical code.
- Proposed steward: analytics lead; the accountable role must be confirmed.
- Intended use: reproduction; other uses need separate authorization.
- Classification: lineage; apply the access rules approved for that class.
- Change control: record corrections to analysis revision instead of silently changing a frozen analysis.

#### Contract field: analysis run identifier
- Purpose: connects outputs to inputs and configuration.
- Proposed steward: analytics lead; the accountable role must be confirmed.
- Intended use: reproduction; other uses need separate authorization.
- Classification: lineage; apply the access rules approved for that class.
- Change control: record corrections to analysis run identifier instead of silently changing a frozen analysis.

#### Contract field: analysis population version
- Purpose: identifies the cohort used in a result.
- Proposed steward: analytics lead; the accountable role must be confirmed.
- Intended use: reproduction; other uses need separate authorization.
- Classification: lineage; apply the access rules approved for that class.
- Change control: record corrections to analysis population version instead of silently changing a frozen analysis.

#### Contract field: guardrail incident key
- Purpose: links a reported concern to its review.
- Proposed steward: review owner; the accountable role must be confirmed.
- Intended use: stop review; other uses need separate authorization.
- Classification: guardrail; apply the access rules approved for that class.
- Change control: record corrections to guardrail incident key instead of silently changing a frozen analysis.

#### Contract field: guardrail severity
- Purpose: supports the predefined escalation decision.
- Proposed steward: review owner; the accountable role must be confirmed.
- Intended use: stop review; other uses need separate authorization.
- Classification: guardrail; apply the access rules approved for that class.
- Change control: record corrections to guardrail severity instead of silently changing a frozen analysis.

#### Contract field: stop decision time
- Purpose: records when future offers were halted.
- Proposed steward: commercial owner; the accountable role must be confirmed.
- Intended use: audit; other uses need separate authorization.
- Classification: guardrail; apply the access rules approved for that class.
- Change control: record corrections to stop decision time instead of silently changing a frozen analysis.

#### Contract field: stop decision authority
- Purpose: records who authorized the stop.
- Proposed steward: commercial owner; the accountable role must be confirmed.
- Intended use: audit; other uses need separate authorization.
- Classification: guardrail; apply the access rules approved for that class.
- Change control: record corrections to stop decision authority instead of silently changing a frozen analysis.

#### Contract field: permission failure count
- Purpose: measures denied or unavailable permission checks.
- Proposed steward: privacy lead; the accountable role must be confirmed.
- Intended use: guardrail report; other uses need separate authorization.
- Classification: guardrail; apply the access rules approved for that class.
- Change control: record corrections to permission failure count instead of silently changing a frozen analysis.

#### Contract field: duplicate offer count
- Purpose: measures repeated dispatches for one assignment.
- Proposed steward: operations; the accountable role must be confirmed.
- Intended use: guardrail report; other uses need separate authorization.
- Classification: guardrail; apply the access rules approved for that class.
- Change control: record corrections to duplicate offer count instead of silently changing a frozen analysis.

#### Contract field: complaint count
- Purpose: captures reported inappropriate contacts.
- Proposed steward: review owner; the accountable role must be confirmed.
- Intended use: guardrail report; other uses need separate authorization.
- Classification: guardrail; apply the access rules approved for that class.
- Change control: record corrections to complaint count instead of silently changing a frozen analysis.

#### Contract field: reviewer effort
- Purpose: measures labor required for operational oversight.
- Proposed steward: operations; the accountable role must be confirmed.
- Intended use: resource review; other uses need separate authorization.
- Classification: guardrail; apply the access rules approved for that class.
- Change control: record corrections to reviewer effort instead of silently changing a frozen analysis.

#### Contract field: delivery cost
- Purpose: captures approved direct contact costs.
- Proposed steward: finance owner; the accountable role must be confirmed.
- Intended use: resource review; other uses need separate authorization.
- Classification: guardrail; apply the access rules approved for that class.
- Change control: record corrections to delivery cost instead of silently changing a frozen analysis.

### 14e. Operational Diagnostic Exercises
Each scenario is hypothetical; each response is a proposed diagnostic and containment action.

#### Diagnostic 1: Eligibility differs between arms
- Plausible mechanism: A pre-assignment filter was applied after the allocation file was generated.
- Discriminating check: Rebuild the eligibility flow from the frozen roster and compare filter timestamps.
- Immediate response: Freeze reporting and investigate whether the randomization population was altered.
- Interpretation or ownership: Data stewardship owns the reconstructed flow.

#### Diagnostic 2: A permission feed is stale
- Plausible mechanism: The last successful permission update predates the scheduling cutoff.
- Discriminating check: Compare feed watermark and individual effective times with the request time.
- Immediate response: Suppress incremental offers until a current state is available.
- Interpretation or ownership: Privacy and operations agree on recovery.

#### Diagnostic 3: An HCP appears in two territories
- Plausible mechanism: A territory reassignment produced overlapping roster records.
- Discriminating check: Inspect effective intervals and resolve against the authoritative identity mapping.
- Immediate response: Quarantine the HCP until ownership is unambiguous.
- Interpretation or ownership: The identity steward records the resolution.

#### Diagnostic 4: A contact is delivered after withdrawal
- Plausible mechanism: The scheduler may have used a cached permission decision.
- Discriminating check: Trace withdrawal, lookup, dispatch, and delivery timestamps.
- Immediate response: Stop the channel and escalate the event to the designated reviewers.
- Interpretation or ownership: Do not assume an incident occurred in this hypothetical case.

#### Diagnostic 5: A control HCP receives routine contact
- Plausible mechanism: Usual scheduling is allowed in the comparator policy.
- Discriminating check: Check whether the contact was routine or an incorrectly routed incremental offer.
- Immediate response: Keep the HCP in the assigned arm; document any actual routing deviation.
- Interpretation or ownership: The study lead evaluates the effect on policy interpretation.

#### Diagnostic 6: Offer delivery is lower than planned
- Plausible mechanism: Capacity or HCP access may have limited delivery.
- Discriminating check: Break down assigned offers by delivery failure reason and territory.
- Immediate response: Assess operational feasibility without excluding nonrecipients from the primary analysis.
- Interpretation or ownership: Operations proposes corrections only under protocol change control.

#### Diagnostic 7: Assignment ratio deviates from plan
- Plausible mechanism: An allocation bug or incomplete enrollment could explain the imbalance.
- Discriminating check: Compare immutable allocation records with expected stratified randomization counts.
- Immediate response: Pause new allocation until the cause is understood.
- Interpretation or ownership: The statistician assesses whether inference remains valid.

#### Diagnostic 8: The outcome source revises historical records
- Plausible mechanism: A correction can change a previously reported response indicator.
- Discriminating check: Compare source versions and identify HCPs whose qualifying status changed.
- Immediate response: Preserve both analysis snapshots and report the revision.
- Interpretation or ownership: The data steward determines whether a revised analysis is required.

#### Diagnostic 9: An event falls exactly on the follow-up boundary
- Plausible mechanism: Different date and timezone conversions can put the event in different windows.
- Discriminating check: Recompute the event boundary using the protocol timezone and timestamp precision.
- Immediate response: Apply the same prespecified boundary to both arms.
- Interpretation or ownership: Document any unresolvable timestamps as a data-quality issue.

#### Diagnostic 10: The same practice includes both arms
- Plausible mechanism: Shared meetings may expose controls to the offer content.
- Discriminating check: Map assignments to practices and inspect material-sharing pathways.
- Immediate response: Avoid claiming independent HCP effects until interference has been assessed.
- Interpretation or ownership: A future study may require practice-level assignment.

#### Diagnostic 11: A prescriber moves to a different practice
- Plausible mechanism: The practice identity at assignment may differ from the follow-up practice.
- Discriminating check: Compare the effective dates of both practice mappings.
- Immediate response: Use the protocol-defined assignment cluster rather than a post-outcome reclassification.
- Interpretation or ownership: Study leadership assesses spillover and cluster sensitivity.

#### Diagnostic 12: The outcome source misses a territory
- Plausible mechanism: Coverage could differ by geography and potentially by arm.
- Discriminating check: Compare source coverage by baseline territory and randomized arm.
- Immediate response: Report the missing coverage and prespecified sensitivity bounds.
- Interpretation or ownership: Do not convert absent coverage to zero response.

#### Diagnostic 13: A content version expires mid-study
- Plausible mechanism: A previously approved offer may no longer be authorized.
- Discriminating check: Compare the approved version registry with each planned dispatch timestamp.
- Immediate response: Stop the expired version before any further delivery.
- Interpretation or ownership: A reviewed replacement requires documented protocol and content approval.

#### Diagnostic 14: A study member changes eligibility rules
- Plausible mechanism: Late rule changes can alter the causal target and introduce selection.
- Discriminating check: Compare protocol versions with the roster filter revision history.
- Immediate response: Freeze enrollment and seek independent review before proceeding.
- Interpretation or ownership: Report both the original and revised target populations if authorized.

#### Diagnostic 15: The randomization list is visible to field staff early
- Plausible mechanism: Staff may change scheduling or roster inclusion based on anticipated allocation.
- Discriminating check: Audit access logs and the sequence of roster freeze and assignment disclosure.
- Immediate response: Restrict access and investigate whether selection was possible.
- Interpretation or ownership: The study lead decides whether the original allocation remains interpretable.

#### Diagnostic 16: An HCP is assigned twice
- Plausible mechanism: A retry or duplicate identity can create contradictory policies.
- Discriminating check: Reconcile assignment keys and identity mappings across all study weeks.
- Immediate response: Stop dispatch for the duplicate while the immutable ledger is inspected.
- Interpretation or ownership: The protocol must specify how to handle the original randomized record.

#### Diagnostic 17: The contact channel is inaccessible
- Plausible mechanism: Channel access can change after a valid initial eligibility check.
- Discriminating check: Compare assignment eligibility with current operational access.
- Immediate response: Do not deliver through an unapproved substitute channel.
- Interpretation or ownership: Retain the original policy assignment in the authorized analysis.

#### Diagnostic 18: A field member overrides a schedule
- Plausible mechanism: An override can reflect HCP preference or local knowledge.
- Discriminating check: Compare restricted override reasons with assignments and delivery records.
- Immediate response: Honor current permissions and preserve the reason for study review.
- Interpretation or ownership: Do not infer the causal effect of the override from a selected comparison.

#### Diagnostic 19: The response proxy rises while clinical concerns increase
- Plausible mechanism: A prescribing proxy does not directly capture appropriate clinical decisions.
- Discriminating check: Review independently defined medical guardrails and reports.
- Immediate response: Do not scale solely because the primary proxy improves.
- Interpretation or ownership: Medical reviewers and the owner determine the safe next step.

#### Diagnostic 20: Cost exceeds the scenario budget
- Plausible mechanism: Operational effort can consume more resources than expected.
- Discriminating check: Reconcile contact delivery, staffing, source access, and review costs.
- Immediate response: Reassess the policy against the agreed resource constraint.
- Interpretation or ownership: No cost threshold or actual price is asserted here.

#### Diagnostic 21: Data arrives after the maturity cutoff
- Plausible mechanism: The assumed source lag may have been too optimistic.
- Discriminating check: Examine late-arrival curves by source and assignment arm.
- Immediate response: Withhold the outcome decision until an approved maturity rule is satisfied.
- Interpretation or ownership: A protocol amendment must not be chosen based on the observed effect.

#### Diagnostic 22: The event definition changes during follow-up
- Plausible mechanism: A revised qualification rule can create incomparable outcome periods.
- Discriminating check: Compare definition versions and event adjudication dates.
- Immediate response: Apply a prespecified consistent definition or report the incompatibility.
- Interpretation or ownership: The outcome steward and statistician approve any revision.

#### Diagnostic 23: The sample has fewer practices than planned
- Plausible mechanism: HCP counts can look adequate while the number of independent clusters is small.
- Discriminating check: Count randomized practices and inspect the cluster size distribution.
- Immediate response: Use appropriate cluster inference and report limited precision.
- Interpretation or ownership: Do not substitute the number of HCPs for the number of independent practices.

#### Diagnostic 24: The point estimate is positive but the interval is wide
- Plausible mechanism: A plausible benefit may coexist with little or no effect.
- Discriminating check: Compare the full interval with zero and the practical threshold.
- Immediate response: Classify the result according to the prespecified decision rule.
- Interpretation or ownership: Do not describe uncertainty as proof that the effect is absent.

#### Diagnostic 25: The lower interval bound exceeds zero but not the business threshold
- Plausible mechanism: Statistical and practical significance answer different questions.
- Discriminating check: Compare the lower bound with the pre-agreed two-point threshold.
- Immediate response: Defer scale under the illustrative rule even if zero is excluded.
- Interpretation or ownership: A different threshold requires a prospective owner decision.

#### Diagnostic 26: A subgroup estimate looks unusually large
- Plausible mechanism: Small sample variation or multiple comparisons may explain it.
- Discriminating check: Check subgroup sample size, prespecification, and simultaneous analysis plan.
- Immediate response: Label unplanned subgroup findings exploratory.
- Interpretation or ownership: Privacy review must precede release of small-group results.

#### Diagnostic 27: A feature appears unusually predictive
- Plausible mechanism: The feature may encode information generated after the decision.
- Discriminating check: Trace feature event time, availability time, and transformation history.
- Immediate response: Remove post-assignment information from any baseline descriptive model.
- Interpretation or ownership: Rebuild the model within a point-in-time evaluation if one is later approved.

#### Diagnostic 28: An analyst requests raw patient records
- Plausible mechanism: The HCP-level decision does not require direct patient identity.
- Discriminating check: Review the requested fields against the documented analytical purpose.
- Immediate response: Deny access pending a separately approved necessity assessment.
- Interpretation or ownership: Use permitted HCP-level aggregates where they are adequate.

#### Diagnostic 29: A model recommends a prohibited channel
- Plausible mechanism: A ranking score cannot override a current permission or channel restriction.
- Discriminating check: Inspect permission state and the decision path for the recommendation.
- Immediate response: Block the action and investigate the gate failure.
- Interpretation or ownership: No model-driven recommendation is authorized in the present design.

#### Diagnostic 30: Two analysts reproduce different results
- Plausible mechanism: Different snapshots, versions, or boundary logic may explain disagreement.
- Discriminating check: Compare locked code, input versions, population filters, and timestamp conversions.
- Immediate response: Hold publication until the discrepancy is resolved.
- Interpretation or ownership: Preserve the competing outputs and the documented reconciliation.

#### Diagnostic 31: An approved contact is retried after a timeout
- Plausible mechanism: The original dispatch may have succeeded even if its acknowledgment was lost.
- Discriminating check: Check the durable contact idempotency key and delivery callback.
- Immediate response: Do not issue another offer until the prior state is reconciled.
- Interpretation or ownership: Engineering tests replay and retry behavior before any deployment.

#### Diagnostic 32: The ordinary schedule changes during the study
- Plausible mechanism: A policy shift changes the comparator and may interact with assignment.
- Discriminating check: Inspect routine contact rates and change records over time and territory.
- Immediate response: Document the deviation and assess whether the prespecified contrast is still meaningful.
- Interpretation or ownership: The owner may stop or redesign rather than reinterpret the study silently.

#### Diagnostic 33: A guardrail is not observable
- Plausible mechanism: An unmeasured harm cannot be treated as a passed safeguard.
- Discriminating check: Verify the data owner, reporting latency, and review route for that guardrail.
- Immediate response: Defer launch of the test until the agreed safeguard is measurable.
- Interpretation or ownership: The designated reviewer approves any alternative monitoring approach.

#### Diagnostic 34: A permission withdrawal affects outcome retention
- Plausible mechanism: The organization may have distinct rules for contact cessation and analysis retention.
- Discriminating check: Ask the privacy owner to determine the approved treatment of historical records.
- Immediate response: Stop contact immediately and apply the authorized data handling decision.
- Interpretation or ownership: Report any resulting changes to analysis denominators and missingness.

#### Diagnostic 35: A small cell is visible in an internal dashboard
- Plausible mechanism: HCP characteristics may be identifiable even without patient identifiers.
- Discriminating check: Apply the approved disclosure review to the displayed grouping and counts.
- Immediate response: Restrict or suppress the view before wider distribution.
- Interpretation or ownership: Do not invent a universal minimum cell-size rule.

#### Diagnostic 36: A study is stopped before the planned end
- Plausible mechanism: A privacy or safety concern may require immediate termination.
- Discriminating check: Record the stop time, reason, authority, and any contact already delivered.
- Immediate response: Prevent future dispatch and preserve only the records authorized for retention.
- Interpretation or ownership: Interpret partial-period analysis using the approved stopping and inference plan.

## 15. Quick Reference Card

| Field | Value |
|---|---|
| Domain | Fictional pharma commercial analytics for an unspecified launch |
| Problem type | Decide whether to propose a reviewed incremental HCP contact test or retain usual scheduling |
| Method | Randomized offer versus usual policy, with eligible and consenting population, maturity and review gates |
| Primary result | Not measured; illustrative 25% versus 20% yields 5 percentage points and a 95% interval of approximately 1.35–8.65 points under independence |
| Key lesson | Historical targeting can confound contact associations; practice spillovers can invalidate independent-HCP inference |
| Tags | `causal-confounding`; `evaluation-mismatch`; `label-delay` |

| Supplemental field | Value |
|---|---|
| Case ID / version / as-of | CASE-01-03-001 / 1.0 / 2026-11-07 |
| Evidence level | Illustrative; topic source contents unavailable and actual outcome not measured |
| Implementation stage | Proposed; draft unreviewed |
| Baseline / primary metric | Usual scheduling; binary eligible HCP-level response in days 1–56 after outcome maturity |
| Key limitation | No approved study, actual dataset, measured effect or legal conclusion |
| Review trigger / owner | Launch, access rule, data source, consent rule or protocol changes / commercial decision owner |

The illustrative decision rule requires a lower interval bound above two percentage points and all reviewed guardrails; the constructed example does not meet it.
Separate approved HCP linkage from analysis; missing permission, unreviewed content, privacy incidents or serious data-integrity concerns stop incremental delivery.
