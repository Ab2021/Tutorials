---
case_id: "CASE-M11-01"
case_title: "Credit-risk model design: point-in-time loss-aware underwriting"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "predictive-ml"
domain: "Consumer lending / supervised learning and feature engineering"
primary_pattern: "Point-in-time risk scoring with policy optimization"
secondary_patterns: ["Calibration before actioning", "Capacity-constrained human review"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "Retail Credit Risk Director"
source_index: "INDEX.md"
review_trigger: "Any change to product terms, label maturity, data availability, capacity, policy, or applicable requirements"
---

# Credit-risk model design: point-in-time loss-aware underwriting

## 1. The One-Line Framing

> Given one completed consumer-loan application at its decision-time snapshot, estimate the probability of a defined bad outcome and route it to approve, decline, or bounded manual review for the Retail Credit Risk Director, minimizing illustrative expected decision cost subject to review capacity and policy controls, compared with a documented rules-plus-analyst baseline.

- Task category: supervised binary classification followed by constrained decisioning.
- Unit: one application decision, not one customer lifetime or one account statement.
- Frequency: each eligible application after required verification fields arrive.
- Input: point-in-time applicant, requested-loan, bureau, affordability, and channel fields.
- Output: a calibrated probability `p_bad`, reason codes, and a proposed route.
- Beneficiary: the lending business and applicants who receive consistent, timely treatment.
- Primary success condition: lower illustrative expected decision cost without exceeding review capacity.
- Binding constraint: no score may use information that first exists after the decision timestamp.
- Baseline: deterministic hard rules, then analyst review for a fixed queue.
- Evidence declaration: this is an illustrative design exercise.
- No bank, borrower, portfolio, approval rate, loss rate, or outcome in this file is observed.
- All records, values, calculations, and workload figures are constructed teaching inputs.
- The proposed architecture is not a production claim.
- The policy is not credit, legal, compliance, or fair-lending advice.
- The obvious “maximize AUC” framing is incomplete.
- A ranking metric does not decide which applications to approve.
- A decision needs costs, capacity, a threshold, calibration, and a safe fallback.
- Primary pattern: point-in-time risk scoring with policy optimization.
- Secondary pattern one: calibration before actioning.
- Secondary pattern two: capacity-constrained human review.
- The decisive tradeoff is not accuracy alone.
- Declining a good applicant has an opportunity and customer cost.
- Approving a bad applicant has an expected loss cost.
- Reviewing an application consumes a scarce analyst slot.
- A score can rank correctly yet be badly calibrated for a cost threshold.
- A capacity rule can convert a good offline policy into an operationally impossible one.
- The proposed decision owner remains accountable for policy choices.
- The model owner is accountable for score integrity and monitoring.
- A human reviewer remains accountable for an exception decision.
- An applicant-facing explanation process remains outside model training.
- The case starts from a bounded pre-origination decision.
- It does not price loans, set limits, collect debt, or make account-management decisions.
- It does not infer protected traits or certify any fairness property.
- It does not assert that a particular jurisdictional rule applies.
- Applicability of privacy, consumer-credit, and fairness requirements is OPEN.
- Counsel and compliance must determine applicability before any deployment.
- The recommendation is conditional on the controls described below.
- The recommendation reverses if the labels or key features cannot be made point-in-time valid.
- The recommendation reverses if capacity is lower than the review band requires.
- The recommendation reverses if calibration fails on the policy population.
- The recommendation is a validation plan, not an authorization to lend.

## 2. Context & Background

- Domain: a hypothetical unsecured installment-loan intake workflow.
- Product: a fixed-term requested principal with a proposed repayment schedule.
- Channel: an illustrative digital application flow with optional assisted completion.
- Applicant starts by submitting identity, income, housing, employment, and request details.
- Required verification and fraud controls occur before this risk decision.
- This case assumes an application has passed those upstream gates.
- Passing an upstream gate is not evidence of low repayment risk.
- The risk decision occurs before offer issuance and funding.
- The next state after approval is offer generation, then applicant acceptance.
- The next state after decline is a policy-governed adverse-action and appeal workflow.
- The next state after review is analyst disposition or escalation.
- The decision owner is the Retail Credit Risk Director.
- The operations owner is the Underwriting Operations Manager.
- The model owner is the Credit Risk Analytics Lead.
- The service owner is the Decisioning Platform Lead.
- The control owner is an assigned risk-and-compliance function, not presumed by this case.
- Applicants are affected by speed, consistency, explanation, and error correction.
- Analysts are affected by queue volume and evidence quality.
- Finance is affected by loss, foregone margin, and operating cost.
- Customer support is affected by appeal and status inquiries.
- The baseline has two stages.
- Stage one applies eligibility and affordability hard stops.
- Stage two routes a fixed list of borderline applications to analysts.
- The remaining applications are approved or declined by deterministic rules.
- Illustrative baseline weakness one: coarse cutoffs discard useful ordering information.
- Illustrative baseline weakness two: a fixed queue can overflow during channel spikes.
- Illustrative baseline weakness three: analyst decisions can vary without a shared evidence view.
- Illustrative baseline weakness four: rules can become stale when applicant mix changes.
- Illustrative baseline strength one: it is inspectable and easy to stop.
- Illustrative baseline strength two: it can enforce hard policy constraints directly.
- Illustrative baseline strength three: it does not require a learned model to operate.
- Historical labels arrive only after outcome maturity.
- Therefore training data cannot contain applications whose outcome is unresolved.
- Therefore recent originations can describe current mix but cannot yet become labels.
- The case uses a 90-days-past-due-by-12-months label for illustration.
- “Bad” is a label definition, not a moral statement about a person.
- An application can be good under the label and still generate low profit.
- An application can be bad under the label and still have partial recoveries.
- A risk score is therefore a proxy for a later outcome, not a full economic model.
- Geography is intentionally unspecified.
- No jurisdiction, institution, vendor, data bureau, or policy manual is claimed.
- Time zone for all timestamps is UTC in this design.
- Application time means the persisted decision request timestamp in UTC.
- Source-system time must be preserved separately if it differs from UTC.
- The case assumes no real applicant data are accessed.
- The synthetic cohort below exists only to audit arithmetic and decisions.
- Scale assumption: 2,000 eligible applications arrive in a typical illustrative business day.
- Peak assumption: 300 eligible applications can arrive in one illustrative hour.
- Review assumption: 40 analyst decisions can be completed per business day.
- A daily capacity number does not guarantee intraday service-level compliance.
- The architecture therefore needs queue age monitoring as well as count monitoring.
- The model is not assumed to replace policy rules.
- Policy rules remain first-class controls before and after scoring.
- The project stage is proposed.
- There has been no pilot, production deployment, measured improvement, or causal result.
- Any external source body needed for requirements applicability is unavailable to this case.
- Those source bodies remain OPEN rather than reconstructed.

## 3. The Business Question

| Decision contract | Illustrative specification |
|---|---|
| Question | For each eligible application, should the policy approve, decline, or send it to manual review using only decision-time evidence? |
| Decision owner | Retail Credit Risk Director |
| Trigger and deadline | An eligible application reaches the underwriting service; return a route within 2 seconds excluding upstream verification. |
| Available actions | Keep rules baseline; approve; decline; review; or hold all automated decisions after a control breach. |
| Value and downside | Approval may create contribution but can incur loss; decline may avoid loss but can reject a good applicant; review consumes finite labor. |
| Objective | Minimize expected illustrative decision cost per application and per daily cohort. |
| Hard constraints | Point-in-time features; fixed 40/day review capacity; hard policy rules; auditable route; appeal and fallback path. |
| Scope | Eligible hypothetical consumer-loan applications at initial underwriting only. |
| Out of scope | Pricing, credit-line assignment, collections, fraud adjudication, legal applicability findings, and live deployment. |

- Illustrative stakeholder wording: “Can we use a calibrated risk estimate to route the uncertain middle safely without filling the analyst queue?”
- This is an illustrative paraphrase, not an attributed quotation.
- The decision must be made per application.
- The objective aggregates expected cost across a daily cohort.
- The candidate policy can use a score only after hard policy gates.
- The candidate policy cannot override a mandatory decline rule.
- The candidate policy cannot treat unavailable data as observed data.
- The candidate policy cannot silently change threshold values.
- The candidate policy must surface a reason for its route.
- The candidate policy must record model and policy versions.
- The candidate policy must tolerate a model-serving outage.
- The candidate policy must tolerate a review queue being full.
- The business proxy is expected decision cost.
- Expected decision cost is not the same as realized portfolio profitability.
- It omits funding cost, prepayment, recovery timing, capital, and downstream servicing costs.
- It intentionally simplifies economics to expose the decision mechanism.
- A later approved loan’s behavior is unavailable at decision time.
- A later label is available only for mature historical applications.
- The model objective may minimize weighted log loss during estimation.
- The policy objective may minimize expected business cost during threshold selection.
- These objectives must not be conflated.
- AUC is useful for discrimination but insufficient for action selection.
- Calibration is useful for expected-cost arithmetic but insufficient for fairness or operational suitability.
- Review capacity creates a constrained allocation problem.
- The policy must reserve analyst effort for applications where review can alter a route.
- The policy must not use review merely as a queue for every uncertain score.
- The no-change action remains credible if validation does not clear gates.
- The hold action is required after a severe integrity breach.
- An applicant correction or appeal is routed outside automatic retraining.
- The appeal evidence must be separately timestamped and governed.
- The requested decision horizon is one application event.
- The label horizon is twelve months after origination for the illustrative definition.
- The model’s score horizon and the policy’s decision horizon are different.
- The project’s value question is prospective and remains not-measured.
- The project’s safety question is whether unsafe automation is blocked before action.
- A correct answer can be “retain baseline and collect evidence.”

## 4. Assumptions & Constraints

| ID | Assumption or constraint | Type | Evidence or verification plan | Status | Impact if wrong | Fallback | Recheck trigger / owner |
|---|---|---|---|---|---|---|---|
| A-01 | Each feature has an availability timestamp no later than decision time. | Data | Field-level timestamp audit. | Unverified | Leakage invalidates estimates. | Remove late fields and retrain. | Schema change / Analytics Lead |
| A-02 | The 12-month label can be reconstructed from governed servicing events. | Data | Label reconciliation sample. | Unverified | Target is unreliable. | Retain baseline. | Label logic change / Risk Owner |
| A-03 | Mature cohort resembles future policy population enough for validation. | Representativeness | Temporal and channel slices. | Unverified | Threshold may misroute current applicants. | Narrow rollout or baseline. | Mix shift / Model Owner |
| A-04 | Review can change a disposition using permitted evidence. | Operations | Analyst workflow study. | Unverified | Review cost has no value. | Eliminate review band. | SOP change / Ops Manager |
| A-05 | Illustrative review capacity is 40 applications/day. | Capacity | Staffing roster and service-time measurement. | Scenario only | Queue overflows. | Narrow band or baseline. | Staffing change / Ops Manager |
| C-01 | Score service returns within 2 seconds for eligible requests. | Latency | Load and timeout test. | Proposed | Workflow stalls. | Rules fallback. | SLO breach / Platform Lead |
| C-02 | Every action has versioned reasons and an immutable audit record. | Control | Event-schema contract test. | Proposed | Decision cannot be reconstructed. | Block release. | Missing event / Platform Lead |
| C-03 | Protected-attribute use and fairness obligations are determined by authorized reviewers. | Governance | Jurisdiction and policy review. | OPEN | Inappropriate design or deployment. | Do not deploy. | Scope change / Compliance |
| C-04 | Privacy, retention, consent, and data-sharing applicability are determined by authorized reviewers. | Governance | Data inventory and legal review. | OPEN | Unlawful or unsafe processing. | Do not ingest field. | New source / Privacy Owner |
| C-05 | Appeals can correct factual input errors without reusing appeal data as unreviewed labels. | Process | Appeal SOP review. | Proposed | Feedback contamination or harm. | Manual correction queue. | Appeal launch / Ops Manager |

- Assumption A-01 is design-reversing.
- A single delayed delinquency-status field can create target leakage.
- Assumption A-02 is design-reversing.
- Inconsistent maturity logic changes the meaning of “bad.”
- Assumption A-05 is design-reversing for a review-band design.
- If capacity is zero, the policy must have no review route.
- A score is not a hard rule.
- Hard policy constraints are evaluated independently of model probability.
- Missingness must be represented as missingness, not silently imputed as safety.
- A missing bureau response may be operationally different from a thin file.
- This distinction requires approved data semantics.
- The case does not assume that protected attributes are collected.
- The case does not assume that a proxy audit is sufficient for fairness assessment.
- The case does not assume that group metrics alone establish fairness.
- The case does not assume a legal adverse-action reason mapping.
- Reason-code suitability is OPEN pending policy and compliance review.
- The case does not assume a vendor score can be reused as a label.
- The case does not assume a reviewer is error-free.
- The case does not assume human review is less biased than automation.
- The case does not assume labels are free from prior-policy selection.
- Historical approvals may create selective-label bias because declined applicants lack repayment outcomes.
- Validation must state that limitation rather than hide it.
- Exploration must not write to the production decision log.
- Training snapshots must exclude manual notes created after the decision.
- Training snapshots must exclude collections events.
- Training snapshots must exclude repayment events.
- Training snapshots must exclude offer acceptance state.
- Training snapshots must exclude funded amount if funding occurs after decision.
- Inputs must be ordered by availability, not by convenient warehouse arrival.
- Every calculated feature requires both source event time and feature materialization time.
- A materialized feature being present at scoring time does not prove it was valid at historical decision time.
- Capacity is measured in completed, quality-controlled decisions, not logins.
- Capacity is a daily operating constraint, not an invitation to defer risk indefinitely.
- The approval threshold must be versioned.
- The decline threshold must be versioned.
- The review-band tie-breaking rule must be versioned.
- A threshold change requires expected-cost recomputation and capacity recomputation.
- Any threshold recommendation is illustrative until economics are approved.
- The fallback does not learn from the outage.
- A fallback route must be explicit rather than “best effort.”
- A failed governance review means no deployment, not a relaxed check.

## 5. Data, Tools & Method

### 5a. Data Specification

| Field | Illustrative specification |
|---|---|
| Unit of analysis | One application decision keyed by `application_id`; one applicant may have repeated applications. |
| Decision timestamp | `decision_ts_utc`, the persisted request time before score computation. |
| Training window | Mature historical snapshots only; exact dates are unavailable and not invented. |
| Validation windows | Later mature time windows, held out by application decision date. |
| Outcome | `bad_12m=1` if defined 90+ days past due occurs within 12 months of origination; otherwise 0 after full maturity. |
| Censoring | Exclude or separately flag applications without the full label horizon; never label them good by default. |
| Inputs | Verified decision-time fields, point-in-time bureau attributes, request terms, affordability signals, and channel metadata where approved. |
| Exclusions | Post-decision events, label-period events, unresolved labels, duplicate decisions, and fields failing availability audit. |
| Join semantics | Left join each source snapshot on application key and latest record with `available_ts <= decision_ts_utc`. |
| Splits | Time-based train, validation, and final test; borrower-group isolation where feasible. |

- `application_id` identifies a submitted decision request.
- `borrower_key` is a governed internal entity key if available.
- Repeated applications from one borrower can leak across random splits.
- A temporal split prevents training on later business conditions than the test set.
- Borrower-group isolation prevents the same borrower’s history from appearing in train and test.
- Both protections can reduce apparent performance.
- That reduction is informative rather than a reason to remove the protections.
- Illustrative label: `bad_12m=1` when the account reaches 90 or more days past due by month 12.
- The exact servicing-event definition must be specified by the accountable data owner.
- Label date is not decision date.
- Label maturity date is decision or origination date plus the defined horizon.
- Origination timing must be clarified where an offer is accepted after decision.
- If no loan originates, repayment outcome may be structurally unobserved.
- This is a selective-label limitation.
- The training population therefore may represent originated or approved applications rather than all applicants.
- The deployment population may include all eligible applications.
- This mismatch must be evaluated before any action.
- Input `requested_principal` has unit currency and is positive.
- Input `requested_term_months` has unit months and is a permitted product value.
- Input `verified_monthly_income` has unit currency/month and requires an availability timestamp.
- Input `housing_payment_monthly` has unit currency/month and can be missing only under defined semantics.
- Input `bureau_utilization` is a ratio in `[0,1]` when available.
- Input `bureau_age_months` has unit months and cannot be negative.
- Input `recent_inquiries_6m` is a nonnegative count.
- Input `prior_internal_delinquency_count` is a nonnegative count available before decision.
- Input `channel` is a categorical acquisition route only if its use is approved.
- Input `application_hour_utc` is a timestamp-derived operational field, not applicant behavior.
- Derived debt-to-income uses only verified monthly obligations and income valid at decision time.
- Derived debt-to-income is undefined when verified income is zero or unavailable.
- Undefined is not converted to zero.
- Missingness indicators must be fitted and documented in the training pipeline.
- Imputation statistics must be learned on training data only.
- Categorical vocabulary must be learned on training data only.
- Outlier capping thresholds must be learned on training data only.
- Labels must not be imputed.
- Duplicate decisions need a policy: retain one event per actual decision or model each request explicitly.
- A duplicate record is not necessarily a duplicate application.
- A source correction arriving after decision is not valid historical input without historical versioning.
- Row count is unavailable because no dataset is supplied.
- Exact class prevalence is unavailable because no dataset is supplied.
- Exact missingness rates are unavailable because no dataset is supplied.
- These absent facts remain OPEN.
- The small synthetic cohort in Section 6 is not a training dataset.
- The small synthetic cohort is not evidence of discrimination or calibration.
- The small synthetic cohort exists to make cost arithmetic inspectable.

### 5b. Tools & Stack

- Proposed language: Python for offline feature and model experimentation.
- Proposed tabular library: a version-pinned dataframe implementation selected by the delivery team.
- Proposed estimator: regularized logistic regression as the transparent first learned candidate.
- Proposed comparator: monotonic gradient-boosted trees only after a transparent baseline is established.
- Proposed calibration: held-out isotonic or logistic calibration selected by validation evidence.
- Proposed artifact store: governed versioned object storage.
- Proposed feature snapshot store: immutable point-in-time extracts with schemas.
- Proposed registry: a model registry recording training data snapshot, code revision, thresholds, and approval state.
- Proposed serving: a stateless scoring service behind authenticated internal decisioning.
- Proposed policy engine: deterministic rule and threshold evaluator outside estimator code.
- Proposed queue: durable manual-review work queue with disposition events.
- Proposed audit sink: append-only decision event store.
- Proposed monitoring: data freshness, schema, missingness, score, route, capacity, and label-performance monitors.
- No actual package version is claimed because no executable project environment is provided.
- No actual cloud, hardware, vendor, or managed service is claimed.
- No actual model artifact was trained.
- No actual load test was executed.
- A production choice must be based on approved operational constraints rather than this example.
- Tooling must expose feature provenance for one decision.
- Tooling must expose the score calibration version.
- Tooling must expose policy thresholds without editing model weights.
- Tooling must make replay possible from retained permitted inputs.
- Tooling must not make raw sensitive values broadly visible in logs.
- Access control and retention configuration are OPEN pending applicable policy.

### 5c. Method Summary

- Pipeline: eligible application -> hard policy gates -> point-in-time feature build -> score -> calibration -> route policy -> audit event.
- Estimator target: estimate `P(bad_12m=1 | permitted decision-time features)` for the training population.
- Estimator candidate one: rules-only baseline.
- Estimator candidate two: regularized logistic regression with documented transforms.
- Estimator candidate three: monotonic gradient-boosted trees, considered but not tested.
- Estimator candidate four: analyst-only review, considered as a capacity-limited baseline.
- The selected teaching design is calibrated logistic regression plus external policy thresholds.
- This selection is proposed, not empirically established.
- Logistic regression is chosen first because its assumptions and transformations are inspectable.
- It may lose nonlinear interaction signal relative to boosted trees.
- A more complex model is not selected merely because it can increase a ranking metric.
- A model must beat the rules baseline on predeclared validation criteria to proceed.
- A model must also pass calibration checks on the action population.
- A model must also fit review capacity at the proposed band.
- A model must also support stable explanation and audit outputs.
- The model returns a raw estimate `r` in `[0,1]`.
- A calibration map returns `p_bad = g(r)` in `[0,1]`.
- Policy uses `p_bad`, not raw rank, for cost arithmetic.
- Hard decline rules execute before the model when policy requires them.
- Hard approvals are not assumed; approval still follows score and policy checks.
- The illustrative action set is approve, review, decline, or fallback.
- Illustrative default thresholds are `t_approve=0.12` and `t_decline=0.28`.
- The illustrative review band is `0.12 <= p_bad < 0.28`.
- If review capacity is exhausted, the route must not silently remain review.
- The configured overflow action in this exercise is baseline rules routing.
- A real overflow action requires owner approval and applicant-impact review.
- The review queue is ranked by closeness to the expected-cost decision boundary only if permitted.
- The illustrative review priority is smallest absolute margin to 0.20.
- A priority score does not change the underlying model probability.
- The threshold boundary derives from costs only under a simplified two-action model.

```text
FORMULA [M-01]: EC(approve | p) = p * C_bad_approve
WHERE: p = calibrated probability of bad [probability];
       C_bad_approve = cost of approving a bad application [currency/application].
CONSTRAINT: good-approval cost is set to zero in this illustrative two-action comparison;
            p is calibrated for the decision population and horizon.
NUMERICAL EXAMPLE: p=0.20; C_bad_approve=1000 currency units;
                   EC(approve)=0.20*1000=200 currency units.
INTERPRETATION: this is expected illustrative loss cost, not realized loss.
```

```text
FORMULA [M-02]: EC(decline | p) = (1 - p) * C_good_decline
WHERE: C_good_decline = opportunity/customer cost of declining a good application [currency/application].
CONSTRAINT: bad-decline cost is set to zero; costs are approved scenario inputs, not bank values.
NUMERICAL EXAMPLE: p=0.20; C_good_decline=250 currency units;
                   EC(decline)=(1-0.20)*250=200 currency units.
INTERPRETATION: at p=0.20 the simplified approve and decline costs tie.
```

```text
FORMULA [M-03]: t* = C_good_decline / (C_bad_approve + C_good_decline)
WHERE: t* = approve/decline indifference probability [probability].
CONSTRAINT: compare M-01 and M-02; fixed costs; calibrated p; no review action.
NUMERICAL EXAMPLE: 250/(1000+250)=0.20.
INTERPRETATION: approve below 0.20 and decline above 0.20 under this narrow model.
```

- Dimension check: probability times currency/application yields currency/application.
- Boundary check: if `C_bad_approve=0`, threshold is 1 and approving weakly dominates.
- Boundary check: if `C_good_decline=0`, threshold is 0 and declining weakly dominates.
- These boundaries show the formula is not a universal credit policy.
- Review is modeled as an information-gathering action with operating cost.
- Review can be useful only if it changes some routes enough to offset its cost.
- The case uses an illustrative review cost of 35 currency units/application.
- Review quality is not assumed perfect.
- Review economics must include its actual error pattern before deployment.

| Candidate | Evidence considered | Strength | Limitation / avoid when | Status | Revisit if |
|---|---|---|---|---|---|
| Rules plus analysts | Existing conceptual baseline only | Inspectable hard controls | Coarse ordering and fixed queue | Comparator; not measured | Baseline performance is documented |
| Logistic regression plus calibration | Illustrative design and arithmetic | Transparent probability path | Linear log-odds assumptions | Proposed | Validation supports it |
| Monotonic boosted trees | Considered only | Can model nonlinearities | More complex governance and calibration | Not tested | Logistic model misses material signal |
| Analyst-only | Considered only | Human contextual review | Capacity-limited and variable | Not selected | Small high-risk specialty queue |

### 5d. Data provenance, quality, exclusions and leakage controls

- `SRC-APP-01`: application submission snapshot, owner and access basis OPEN.
- `SRC-BUR-01`: bureau-like snapshot, owner and access basis OPEN.
- `SRC-VER-01`: verified income and obligations snapshot, owner and access basis OPEN.
- `SRC-SRV-01`: servicing outcome events for labels, owner and access basis OPEN.
- `SRC-POL-01`: hard policy rules, owner and approved version OPEN.
- `SRC-OPS-01`: review disposition events, owner and quality controls OPEN.
- No source body was supplied for this case.
- No source claim is inferred from a filename or common industry practice.
- Each source requires a data contract before use.
- A data contract must name owner, lawful/approved basis, retention, access, refresh, and deletion behavior where applicable.
- A data contract must provide source event time and availability time.
- Source extraction time is not a substitute for source availability time.
- A feature is eligible only if `feature_available_ts <= decision_ts_utc`.
- This predicate is evaluated per record, not by a vague table-level assertion.
- Feature backfills are high leakage risk.
- Corrected bureau data can be valid now but invalid for historical replay.
- Servicing events are label-only inputs and never model features for the same decision.
- Manual-review notes written after decision are excluded from the initial score.
- Offer acceptance status is excluded because it occurs after the route.
- Funding status is excluded because it occurs after the route.
- First payment status is excluded because it occurs after decision and may define the outcome.
- Collections contact is excluded because it occurs after delinquency.
- Charge-off status is excluded because it occurs after the risk horizon begins.
- Future application behavior is excluded from the prior application’s score.
- A borrower aggregate must use only records available before this application’s decision time.
- A cohort aggregate must use only prior completed events and freeze its version.
- Imputer fitting happens on training partition only.
- Scaling fitting happens on training partition only.
- Encoding fitting happens on training partition only.
- Calibration fitting happens after base-model fitting on a dedicated validation partition.
- Threshold selection happens after calibration and before final test evaluation.
- Final test data must not select features, transforms, calibration, or thresholds.
- Duplicate keys are quarantined until their business meaning is determined.
- One-to-many joins are prohibited unless an aggregation with a time cutoff is defined.
- Join row-count reconciliation is a release check.
- Feature null-rate comparison by time slice is a monitoring check.
- Feature range and category checks are serving-time controls.
- A delayed feature must be rejected rather than substituted by its latest value.
- In the smoke state below, `income_verified_ts > decision_ts` leads to `REJECT_DELAYED_FEATURE`.
- This rejection is intentional because a late value could create leakage or inconsistent decisioning.
- The rejected request is routed to baseline/manual fallback according to approved SOP.
- It is not scored with an imagined historical value.
- The model artifact needs a feature schema fingerprint.
- The serving request needs an input schema version.
- Schema mismatch needs an explicit deny or fallback decision.
- Raw values in audit logs require minimization and approved access controls.
- Privacy applicability is OPEN; logging design is not certification.
- Fairness applicability is OPEN; subgroup analysis is not certification.
- If approved group attributes exist, their use for monitoring must be separately governed.
- Proxy groups must not be assumed to prove or disprove disparate impact.
- Selective labels mean conventional test metrics can be conditional on historical approvals.
- A reject-inference method is not assumed valid simply because it fills missing outcomes.
- Any use of rejected-application outcomes requires an explicit methodology review.

### 5e. Reproducibility and method-selection ledger

| Artifact/source ID | Location | Version/date | Purpose | Accessibility | Reproduction limitation |
|---|---|---|---|---|---|
| SRC-APP-01 | Not supplied | OPEN | Application fields | Unavailable | No real extraction or schema check possible |
| SRC-SRV-01 | Not supplied | OPEN | Outcome construction | Unavailable | No label reconciliation possible |
| POL-01 | Not supplied | OPEN | Hard policy and fallback | Unavailable | No policy applicability determination |
| CFG-ILL-01 | This case Sections 5–6 | 2026-11-10 | Illustrative thresholds and costs | Public in repository | Not a deployed configuration |
| RUN-SMOKE-01 | Manually specified state trace in Section 6 | 2026-11-10 | Arithmetic and decision-state smoke | Reconstructable | Not a trained-model evaluation |

- Proposed code revision identifier: unavailable because no code repository revision is provided.
- Proposed random seed: must be recorded for every stochastic training run.
- A seed does not make an evolving source snapshot reproducible.
- Required configuration: label definition version.
- Required configuration: feature list and availability rule version.
- Required configuration: split boundaries and borrower grouping rule.
- Required configuration: estimator type and hyperparameters.
- Required configuration: calibration method and fit partition.
- Required configuration: cost inputs and threshold derivation.
- Required configuration: review capacity and overflow route.
- Required configuration: hard-rule version.
- Required configuration: explanation mapping version.
- Required configuration: serving schema and fallback version.
- Required run artifact: training snapshot identifiers.
- Required run artifact: feature statistics and missingness report.
- Required run artifact: discrimination and calibration results by split.
- Required run artifact: capacity simulation.
- Required run artifact: decision-cost calculation.
- Required run artifact: model approval record.
- Required run artifact: final test report.
- Required run artifact: monitoring baseline.
- Method selection gate one: baseline comparison is complete.
- Method selection gate two: leakage audit passes.
- Method selection gate three: calibration supports policy arithmetic.
- Method selection gate four: temporal and borrower-group validation do not expose unacceptable instability.
- Method selection gate five: capacity simulation fits approved staffing.
- Method selection gate six: governance review determines permitted feature and monitoring scope.
- Failing any gate results in no automated policy expansion.
- A model card is required before pilot, but no model card is claimed complete here.
- A decision-log schema is required before pilot, but no live log is claimed here.

## 6. Analysis Walkthrough

- Decisive result: under constructed costs, the no-review approval/decline indifference point is 0.20.
- This result follows from M-01 through M-03, not from a trained model.
- The proposed review band intentionally straddles that indifference point.
- The review band is then limited by a 40/day capacity scenario.
- The conclusion is conditional on calibrated probabilities and the stated costs.
- The conclusion is not a claim about a bank’s optimal threshold.

| Step | Hypothesis or decision | Input snapshot | Method/parameters | Intermediate output | Interpretation | Evidence/status |
|---|---|---|---|---|---|---|
| 1 | Is a feature decision-time valid? | Synthetic request A | Compare availability to decision timestamp | One late feature | Reject score request | Illustrative smoke |
| 2 | What is the cost boundary? | `C_bad=1000`, `C_good_decline=250` | M-03 | `t*=0.20` | Cost tie | Derived illustrative |
| 3 | Does the synthetic cohort give stated routes? | 100 applications | `0.12/0.28` thresholds | 45 approve, 40 review, 15 decline | Capacity exactly filled | Derived illustrative |
| 4 | What is cohort decision cost? | Constructed outcomes, including synthetic counterfactual labels for declines | Confusion-matrix and review costs | 10,650 units | Synthetic scenario accounting, not identified live-policy value | Derived illustrative |
| 5 | Does a wider band fit? | Same cohort | `0.10/0.30` | 55 reviews | Capacity breach | Derived illustrative |

### 6.1 Point-in-time decision trace

- Synthetic request A has `decision_ts_utc=2026-01-10T10:00:00Z`.
- Synthetic request A has `income_verified_ts=2026-01-10T10:05:00Z`.
- The income verification arrives five minutes after the proposed decision.
- The availability predicate is false.
- The score service must emit `REJECT_DELAYED_FEATURE`.
- The score service must not replace the value with a future-corrected value.
- The score service must not continue with a partial vector unless that exact partial-vector model is approved.
- The policy engine must invoke the approved fallback route.
- In this illustrative state, fallback is `BASELINE_RULES_ROUTE`.
- The audit event includes decision timestamp, offending field ID, source availability timestamp, and fallback version.
- This state does not conclude that the applicant should be declined.
- It concludes that the learned-score decision was invalid for this request.
- Recovery requires a valid snapshot or a baseline/manual disposition.
- Repeated delayed features require source-contract investigation.

```text
PRECONDITION [S-01]: request has application_id, decision_ts_utc, required feature manifest,
                     hard-rule result, and source availability timestamps.
INPUT: feature records {value, available_ts}; current model and policy versions.
STATE TRANSITION: VALIDATING -> REJECTED_DELAYED when any required available_ts > decision_ts_utc.
INVARIANT: no scored decision contains a required feature first available after decision_ts_utc.
JUSTIFICATION: future information would invalidate historical and online point-in-time consistency.
WORKED TRACE: 10:05Z > 10:00Z, so income field violates the invariant and route becomes fallback.
RECOVERY: record failure, use approved baseline/manual route, investigate source lag; do not backdate data.
FAILURE BOUNDARY: this trace cannot validate source accuracy or eligibility policy correctness.
```

### 6.2 Illustrative cohort and confusion matrix

- Constructed cohort size is 100 applications.
- It is only an arithmetic cohort.
- It is not a random sample.
- It has no inference interval.
- The policy routes 45 applications to approval.
- The policy routes 40 applications to review.
- The policy routes 15 applications to decline.
- Review count exactly equals the illustrative daily capacity of 40.
- For simple confusion accounting, assume reviewed cases receive a final approve or decline.
- The final policy approves 50 applications total.
- Of the 50 final approvals, 8 later meet the illustrative bad label.
- Of the 50 final approvals, 42 do not meet the bad label.
- The final policy declines 50 applications total.
- Of the 50 final declines, 5 would not meet the bad label.
- Of the 50 final declines, 45 would meet the bad label.
- Treat “bad” as the positive class only for this matrix.
- True positive means correctly declined bad application.
- False positive means incorrectly declined good application.
- False negative means incorrectly approved bad application.
- True negative means correctly approved good application.

The full 100-row outcome table is an omniscient synthetic oracle, not an observable lending dataset. A declined application does not originate the hypothetical loan, so its repayment label is counterfactual. The five good declines and 45 bad declines below are invented scenario values used only to demonstrate arithmetic. Do not reconstruct them from absent payments, rejected-application status or an analyst opinion. Actual policy-value evaluation needs a justified, governed identification strategy; approved-originated labels alone cannot validate this full-population matrix.

| Final action interpretation | Actual bad | Actual good | Total |
|---|---:|---:|---:|
| Decline | 45 true positives | 5 false positives | 50 |
| Approve | 8 false negatives | 42 true negatives | 50 |
| Total | 53 | 47 | 100 |

```text
FORMULA [M-04]: Accuracy = (TP + TN) / N
WHERE: TP=45; TN=42; N=100 applications.
CONSTRAINT: equal treatment of error types; unsuitable as the decision objective when costs differ.
NUMERICAL EXAMPLE: (45+42)/100=0.87=87%.
INTERPRETATION: 87% hides the different cost of eight bad approvals and five good declines.
```

```text
FORMULA [M-05]: Expected cohort decision cost = FN*C_bad_approve + FP*C_good_decline + R*C_review
WHERE: FN=8 applications; FP=5 applications; R=40 reviewed applications;
       C_bad_approve=1000; C_good_decline=250; C_review=35 currency units/application.
CONSTRAINT: costs are illustrative, additive, and fixed; review count is charged regardless of final action.
NUMERICAL EXAMPLE: 8*1000 + 5*250 + 40*35 = 8000 + 1250 + 1400 = 10650 currency units.
INTERPRETATION: this is constructed expected-cost accounting, not realized portfolio loss.
```

- The stated 10,650 units includes the review labor scenario.
- A no-review comparator is needed to isolate review value.
- Constructed rules-only comparator: 12 bad approvals and 7 good declines in 100 applications.
- Constructed rules-only comparator has zero review actions.
- Rules-only cost equals `12*1000 + 7*250 = 13,750` units.
- Illustrative difference equals `13,750 - 10,650 = 3,100` units.
- This difference is derived from constructed counts.
- It is not an observed savings claim.
- The requested acceptance arithmetic mentions approval, decline, and error costs.
- It is shown explicitly above: false negatives cost 8,000 units.
- False positives cost 1,250 units.
- Review labor costs 1,400 units.
- Total cost is 10,650 units.
- If review labor were omitted, the total would be 9,250 units.
- Omitting review labor would overstate the constructed benefit by 1,400 units.
- If a reviewer makes different errors than model routing, the matrix must be recomputed.
- The matrix does not estimate uncertainty because it is synthetic.

### 6.3 Capacity calculation and boundary

```text
FORMULA [M-06]: Review load = N * q_review
WHERE: N = applications/day; q_review = fraction routed to review; load [applications/day].
CONSTRAINT: stable daily arrival approximation; every review consumes one completed analyst slot.
NUMERICAL EXAMPLE: N=100; q_review=0.40; load=100*0.40=40 applications/day.
INTERPRETATION: the chosen synthetic review band exactly reaches, but does not exceed, capacity.
```

```text
FORMULA [M-07]: Capacity slack = K - Review load
WHERE: K=40 completed reviews/day; Review load=40 applications/day.
CONSTRAINT: K excludes unplanned absence, rework, and intraday service variation.
NUMERICAL EXAMPLE: 40-40=0 applications/day.
INTERPRETATION: zero slack means any positive volume surprise breaches this simple daily capacity model.
```

- Capacity boundary: `review_load <= K` is required for this simplified policy.
- At 40 reviews, the boundary is exactly satisfied.
- At 41 reviews, the boundary is violated by one application.
- A one-application violation is operationally meaningful when no overflow route is authorized.
- With 2,000 daily applications and the same 40 slots, review share must be at most 2%.
- The 100-application cohort is a calculation sample, not the stated typical daily volume.
- Applying its 40% review share to 2,000 applications would require 800 reviews/day.
- That would exceed 40/day capacity by 760 reviews/day.
- Therefore cohort composition and traffic scale cannot be mixed carelessly.
- At 2,000/day, thresholds must produce a much narrower review share or capacity must increase.
- A capacity simulation needs score distributions from an actual validation population.
- No such distribution is available here.
- The wide-band example below exists to demonstrate the boundary.

The following constructed score fixture reproduces the three routing counts. It tests interval boundaries and capacity only; it is not a calibrated sample, an observed score distribution or a joint outcome dataset validating the separate synthetic error matrix.

| Synthetic score p | Application count |
|---:|---:|
| 0.05 | 35 |
| 0.11 | 10 |
| 0.13 | 8 |
| 0.20 | 24 |
| 0.26 | 8 |
| 0.29 | 5 |
| 0.40 | 10 |


| Threshold scenario | Approve condition | Review condition | Decline condition | Synthetic review count | Capacity result |
|---|---|---|---|---:|---|
| Default | `p<0.12` | `0.12<=p<0.28` | `p>=0.28` | 40 | 40/40; boundary met |
| Wide | `p<0.10` | `0.10<=p<0.30` | `p>=0.30` | 55 | 55/40; exceeds by 15 |
| Narrow | `p<0.15` | `0.15<=p<0.25` | `p>=0.25` | 24 | 24/40; 16 slots slack |

- The wide scenario violates capacity by 15 applications.
- The wide scenario must invoke overflow behavior for at least 15 applications.
- Overflow behavior is not “leave pending forever.”
- In this exercise, overflow is baseline rules routing with an audit marker.
- That fallback may alter costs and applicant outcomes.
- Therefore threshold sensitivity must include fallback, not just model route counts.
- The narrow scenario leaves 16 slots unused.
- Unused slots can be prudent if review quality is focused near the boundary.
- Unused slots can also indicate that capacity is not the binding constraint.
- The right band is selected by prospective validation, not by a desire to use every analyst minute.

### 6.4 Threshold and cost sensitivity

| Scenario | `C_bad_approve` | `C_good_decline` | Derived no-review threshold | Direction of policy consequence |
|---|---:|---:|---:|---|
| Base | 1000 | 250 | 0.20 | Reference |
| Higher loss | 1500 | 250 | 0.1429 | Decline becomes favored at lower risk |
| Higher decline cost | 1000 | 500 | 0.3333 | Approval becomes favored up to higher risk |
| Equal costs | 1000 | 1000 | 0.50 | Symmetric simplified error costs |

- Higher-loss arithmetic: `250/(1500+250)=0.142857...`.
- Higher-decline-cost arithmetic: `500/(1000+500)=0.333333...`.
- Equal-cost arithmetic: `1000/(1000+1000)=0.50`.
- These scenarios show a policy threshold is economics-sensitive.
- A threshold copied from another lender is not defensible.
- A threshold optimized on an uncalibrated score is not interpretable as this cost boundary.
- A threshold may also be constrained by hard policy, fairness, capacity, and customer experience requirements.
- Those constraints can override the simple cost optimum.
- The review band width is a separate decision from the no-review threshold.
- Wider bands can improve information collection but consume more capacity.
- Narrower bands conserve capacity but may send uncertain cases directly to approval or decline.
- Calibration uncertainty should widen caution, not falsely sharpen a boundary.
- The case proposes confidence-aware monitoring but does not claim a valid uncertainty model.

### 6.5 What we tried first and why it did not work

- No model was actually trained or rejected in this illustrative case.
- Therefore no failed empirical experiment is claimed.
- The rules-only approach is a considered baseline, not a documented failed production method.
- It remains viable when training evidence is insufficient.
- The uncalibrated-score policy is rejected as a design choice, not as a measured experiment.
- It fails the decision requirement because a raw ranking score cannot support cost arithmetic.
- The random train/test split is rejected as a design choice.
- It can place later conditions and repeated borrower signals in training while evaluating earlier or related examples.
- The unlimited-review policy is rejected as a design choice.
- It violates the explicit finite-capacity constraint.
- The “use all available fields” policy is rejected as a design choice.
- It creates a high risk of delayed features and future leakage.

### 6.6 Independent numerical and decision-state smoke record

- Smoke ID: `RUN-SMOKE-01`.
- Scope: manually recomputed illustrative arithmetic and state transitions.
- This is not a model-training run.
- This is not a production test.
- Test one input: delayed income feature at 10:05Z for 10:00Z decision.
- Observed result: `REJECT_DELAYED_FEATURE`.
- Observed fallback: `BASELINE_RULES_ROUTE`.
- Test one limit: it proves only the stated comparison rule, not source integrity.
- Test two input: 100 applications with 40 review routes and capacity 40.
- Observed result: capacity boundary met exactly, `40 <= 40`.
- Test two limit: zero slack; it does not demonstrate intraday queue service.
- Test three input: wide band generating 55 review routes with capacity 40.
- Observed result: capacity boundary rejected, `55 > 40`, overflow count `15`.
- Test three limit: no empirical review-time distribution is available.
- Test four input: matrix FN=8, FP=5, R=40 with costs 1000, 250, 35.
- Observed result: `8000 + 1250 + 1400 = 10650` currency units.
- Test four limit: counts and costs are constructed, not bank results.
- Test five input: rules comparator FN=12, FP=7, R=0.
- Observed result: `12000 + 1750 = 13750` currency units.
- Test five limit: comparator is synthetic and not a benchmark claim.
- Test six input: base costs 1000 and 250.
- Observed result: decision threshold `250/1250=0.20`.
- Test six limit: omits many actual lending economics.
- The smoke establishes internal arithmetic consistency for the named inputs.
- The smoke does not establish model accuracy, calibration, fairness, or compliance.
- The smoke does not establish that any feature can legally or operationally be used.

## 7. The Finding

- Recommend a proposed, gated design rather than an immediate automated-lending deployment.
- Use a calibrated probability model only after a point-in-time feature audit and mature-label reconstruction.
- Keep hard policy rules outside the model and use versioned approve, review, decline, and fallback routes.
- Under constructed costs of 1,000 for a bad approval and 250 for a good decline, the simplified no-review boundary is 0.20.
- In the constructed 100-application arithmetic cohort, the proposed route has 10,650 units of expected cost versus 13,750 for a constructed rules-only comparator.
- That 3,100-unit difference is derived illustrative arithmetic, not a bank saving or expected deployment result.
- The proposed `0.12–0.28` review band exactly consumes the constructed 40/day capacity.
- Zero slack means the band must be reselected from real validation score distributions and staffing constraints.
- Deployment is blocked while data provenance, selective-label effects, calibration, temporal stability, borrower-group validation, privacy applicability, fairness applicability, reason mapping, and review quality remain OPEN or unverified.
- The safe answer is baseline retention if those gates fail.

## 8. System/Solution Design

### 8.1 Input-to-output architecture

```text
Applicant request
  -> eligibility and mandatory-policy gate
  -> timestamped feature retrieval
  -> feature-validity validator
  -> calibrated score service
  -> external policy engine
  -> approve | review queue | decline | baseline fallback
  -> explanation/reason mapping and appeal handoff
  -> immutable decision audit event

Control path: model registry + policy registry + release approval -> serving components.
Failure path: late/missing/schema-invalid/model-unavailable -> approved baseline/manual fallback.
State stores: immutable source snapshots; model artifacts; policy versions; review queue; audit events.
```

- The eligibility gate evaluates non-model hard requirements.
- The feature retriever obtains only permitted source values.
- The validator checks schema, ranges, timestamp availability, and required fields.
- The validator must run before model inference.
- The score service returns calibrated probability, model version, and permitted explanation inputs.
- The policy engine owns thresholds and capacity checks.
- The policy engine is separable from model code to allow policy change without weight changes.
- The review queue receives a bounded work item, not a free-form raw data dump.
- The review interface must show the evidence allowed by policy.
- The reviewer’s disposition is recorded with timestamp and reason category.
- The appeal process is a separate workflow from reviewer disposition.
- The audit sink records every route including fallback and rejection.
- The audit sink must capture input snapshot identifiers rather than unnecessarily duplicating raw data.
- The architecture does not assert that append-only storage alone meets any legal retention requirement.
- Access controls, encryption, retention, and deletion are OPEN design obligations.

### 8.2 Decision policy

- State `ELIGIBILITY_FAILED`: execute mandatory decline or designated policy route.
- State `FEATURE_INVALID`: do not score; execute fallback.
- State `MODEL_UNAVAILABLE`: do not estimate; execute fallback.
- State `SCORE_VALID`: evaluate calibrated probability and thresholds.
- State `REVIEW_CAPACITY_AVAILABLE`: enqueue if score is in approved review band.
- State `REVIEW_CAPACITY_EXHAUSTED`: execute approved overflow route.
- State `APPROVED`: create offer path only after all non-risk controls pass.
- State `DECLINED`: invoke approved notice/reason and appeal handoff process.
- State `REVIEWED`: accept analyst disposition and log evidence category.
- State `HELD`: stop automated routing after a kill-switch event.
- Invariant: every terminal action has model/policy/fallback version and reason category.
- Invariant: no terminal action claims a score when scoring was skipped.
- Invariant: review count cannot exceed configured queue admission capacity without overflow event.
- Invariant: a model version cannot be served without an approved policy version.
- Invariant: all mutable thresholds are loaded from an approved registry.

```text
INPUT: eligibility_result, feature_validation_result, model_result, policy_version, queue_capacity.
OUTPUT: route, route_reason, model_version_or_null, policy_version, audit_event.
IF eligibility_result=FAIL: return mandatory-policy route.
ELSE IF feature_validation_result=FAIL: return fallback with failure reason.
ELSE IF model_result unavailable: return fallback with service reason.
ELSE IF p_bad < t_approve: return APPROVE.
ELSE IF p_bad >= t_decline: return DECLINE.
ELSE IF queue_capacity > 0: decrement capacity and return REVIEW.
ELSE: return configured overflow fallback.
```

- This pseudocode is a decision contract, not executable production code.
- The pseudocode assumes thresholds satisfy `t_approve < t_decline`.
- A configuration validator must reject threshold inversion.
- The policy should not dynamically expand a band merely to consume capacity.
- A human reviewer should not see post-outcome information in a prospective review.
- A reviewer override must be attributable to a permitted evidence category.
- A reviewer override should not silently rewrite the original model score.
- Appeals may correct input errors or request reconsideration under approved procedure.
- Appeals should retain the original decision snapshot and record corrected-data version separately.
- Appeal outcomes should not become training labels without an explicit label-governance decision.

### 8.3 Capacity and operations

- Configured review capacity is an admission control, not a staffing forecast.
- The service must expose remaining daily capacity and queue age.
- The service must expose overflow count and fallback count.
- The service must expose score-band counts by model and policy version.
- The service must alert before capacity becomes exhausted.
- An alert at 80% utilization is an illustrative operational choice, not a prescribed standard.
- Queue age needs a separate service-level objective determined by operations.
- Daily reset behavior must respect business calendar, timezone, and carryover policy.
- A midnight reset without queue reconciliation can lose accountability.
- Capacity must account for reviewer absence, quality checks, and rework.
- If review is paused, the policy must immediately narrow to approved no-review behavior or revert to baseline.
- The model service should degrade to an explicit route, not indefinite retries.
- Retry storms can create duplicate review tasks.
- Idempotency key is `application_id` plus decision version or another approved immutable request identifier.
- A duplicate request must return or reference the original terminal route unless policy permits a new decision.
- A new materially changed application requires a new decision timestamp and record.

### 8.4 Successful request and recovery trace

- Successful request B arrives with a valid eligibility pass.
- All required feature availability timestamps are at or before the decision timestamp.
- Schema validation passes.
- The model returns raw score 0.19 and calibration returns `p_bad=0.18`.
- The active thresholds are 0.12 and 0.28.
- The score is in the review band.
- Remaining capacity is 3.
- The policy atomically admits one work item and returns `REVIEW`.
- The audit event records capacity-before 3 and capacity-after 2.
- Reviewer receives permitted decision-time evidence.
- Reviewer disposition is appended as a new event rather than overwriting initial route.
- Recovery from a reviewer-interface outage keeps task state durable and does not rescore the application.
- A completed disposition emits a final route event with reviewer reason category.
- This trace is illustrative and not an observed workflow.

### 8.5 Failed request and recovery trace

- Failed request C arrives with an unknown category outside trained vocabulary.
- Schema contract permits unknown categories only if the deployed encoder has an approved unknown bucket.
- If the encoder supports the bucket, the request proceeds with an audit marker.
- If the encoder does not support the bucket, validation fails before scoring.
- The policy routes to `BASELINE_RULES_ROUTE` or manual fallback as approved.
- The audit event records `SCHEMA_OR_VOCABULARY_INVALID`.
- The request is not dropped.
- The request is not assigned a fabricated low-risk score.
- The source change triggers incident triage and schema compatibility review.
- Recovery requires a compatible feature contract, tested artifact, and release approval.
- A second failure path is model registry unavailability.
- The serving layer must use only a locally validated last-known approved artifact if that behavior is explicitly approved.
- Otherwise it must invoke baseline fallback.
- “Use the latest model” is unsafe when registry state is ambiguous.
- Kill switch behavior is a controlled transition to baseline/manual routing.
- Kill switch activation must be auditable and owner-authorized.

## 9. Evaluation & Validation

### 9.1 Predeclared evaluation questions

- Does the learned model discriminate outcomes better than the rules baseline on a final mature temporal test set?
- Does calibrated probability match observed event frequency sufficiently for the decision bands?
- Does the proposed policy reduce estimated decision cost versus baseline under approved costs?
- Does review load remain within approved capacity by day, hour, and relevant slice?
- Does performance remain stable across later time windows?
- Does borrower-group isolation expose memorization or repeat-applicant leakage?
- Are any permitted subgroup evaluations required and valid under applicable governance?
- Does the fallback route preserve safe processing during model or feature failure?
- Can each decision be reconstructed from retained versions and permitted evidence?
- Are privacy and fairness applicability determinations complete before activation?

### 9.2 Metrics and definitions

| Metric | Numerator | Denominator | Units | Population/window | Comparator | Status |
|---|---|---|---|---|---|---|
| Bad-outcome rate among observed originations | Bad outcomes in approved-originated loans with complete follow-up | Approved-originated loans with mature observed labels | proportion | Mature observed-originations cohort | Same-population rules baseline | Not executed; conditional on origination and observation |
| Counterfactual good-decline rate | Declined applications that would repay under the defined offered loan | Final declines | proportion | Synthetic oracle only in this case | Synthetic rules comparator | Not identified from approved-originated labels |
| Cohort decision cost | Sum of defined error and review costs | All eligible applications | currency/application | Synthetic complete-label oracle | Synthetic rules comparator | Illustrative only; real full-population value requires identification |
| Calibration by bin | Observed versus predicted bad frequency | Mature labelled originations per bin | probability | Observed-originations cohort | Uncalibrated score on the same cohort | Not executed; transport to all applicants unverified |
| Review load | Review admissions | Eligible applications or day | count and proportion | Daily/hourly | Capacity K | Illustrative arithmetic only |
| Fallback rate | Fallback routes | Eligible score requests | proportion | Serving period | Release baseline | Not executed |
| Queue age | Time from admission to disposition | Review tasks | time | Operational period | Approved SLO | Not executed |

- Accuracy is reported only as a diagnostic on the synthetic matrix.

Do not interpret conditional performance on originated loans as performance for all applicants. The label horizon and selection process restrict which outcomes exist. A mature temporal holdout removes some timing errors but does not, by itself, identify declined applicants' counterfactual repayment or the value of a changed approval policy.
- Accuracy is not the policy objective.
- ROC-AUC can measure ranking but does not select thresholds.
- PR-AUC can be useful under class imbalance but does not select thresholds.
- Brier score can summarize probability error but does not identify where action thresholds fail.
- Reliability diagrams must show bin counts and uncertainty intervals where data permit.
- Calibration must be inspected around approval and decline boundaries.
- A global calibration average can hide a harmful band-local mismatch.
- Cost evaluation must include review cost and overflow behavior.
- Capacity evaluation must use actual arrival timing where available.
- An average daily count can hide hourly overload.
- Test metrics need confidence intervals or uncertainty approach suitable to the sample and dependence structure.
- No intervals are presented because no real sample is available.
- A measured metric must name exact population, period, exclusions, and label maturity.
- A metric without a comparator cannot establish improvement.

### 9.3 Split and validation design

- Partition by decision timestamp, never by label availability timestamp alone.
- Select an early mature training window.
- Select a later mature validation window for calibration and threshold selection.
- Reserve the latest mature window as untouched final test.
- Group repeated borrower applications where a governed stable borrower key is available.
- If borrower key is unavailable, document residual repeat-entity risk.
- Do not randomize individual rows across all periods as the primary estimate.
- A random split may be retained as a diagnostic only, labeled optimistic.
- Do not tune on the final temporal test window.
- Do not use a test result to silently revise threshold and call it final.
- If a revision is required, create a new holdout or wait for a later mature cohort.
- Compare training and validation feature distributions.
- Compare validation and final-test feature distributions.
- Compare score distributions by time and channel where permitted.
- Compare missingness by time and channel where permitted.
- Inspect label maturity and censoring rates by slice.
- Inspect capacity implications by score band and arrival hour.
- Inspect decision reason stability after calibration or threshold changes.

### 9.4 Borrower-group validation

- Multiple applications from one borrower can make a naive split overstate generalization.
- A group split keeps known repeated-borrower records within one partition.
- A group-time split is preferable when feasible but can reduce sample size.
- A group split must not use a future borrower identity resolution unavailable at decision time.
- Identity resolution quality is therefore a separate data-quality concern.
- Compare random, temporal, and group-aware results as sensitivity evidence.
- Do not choose the most flattering split as the official result.
- If group-aware performance collapses, investigate memorization and feature leakage.
- If group key coverage is partial, report its coverage and limitation.
- A group split cannot eliminate selection bias from historical policy.

### 9.5 Calibration validation

- Fit calibration on a partition not used to fit base-model parameters.
- Evaluate calibration on a later untouched partition.
- Report reliability by policy band, not only across all scores.
- Check mean predicted risk versus observed mature event rate per bin.
- Check bin sample size before interpreting noise.
- Consider logistic calibration when mapping is plausibly smooth and sample is limited.
- Consider isotonic calibration only with enough data to avoid overfitting.
- Never calibrate on final test labels and report the result as unbiased.
- If calibration changes materially by time, policy thresholds may be invalid.
- If calibration is poor near 0.20, do not use cost arithmetic as if it were exact.
- If calibration is unproven, retain rules or use conservative non-probability policy subject to approval.

### 9.6 Privacy, fairness, and explainability applicability

- Applicability of privacy obligations is OPEN.
- Applicability of consumer-credit, adverse-action, and fair-lending obligations is OPEN.
- This case does not certify lawful feature use.
- This case does not certify that explanation output is adequate for any required notice.
- This case does not certify a fairness threshold or protected-group method.
- Before pilot, authorized reviewers must identify relevant policies and jurisdiction-specific requirements.
- If approved attributes are available for monitoring, define population, missingness, denominator, and use restrictions.
- Do not infer an applicant’s protected status merely to complete a metric.
- Group fairness metrics can conflict with calibration and selection-rate goals.
- A favorable aggregate metric does not resolve individual explanation or data-quality issues.
- A model can be calibrated overall and still behave unevenly across relevant groups.
- A human review process can introduce disparate treatment as well as correct errors.
- Any fairness assessment requires a documented decision about comparison population and error tradeoffs.
- Any explanation mapping requires evidence that reasons are accurate for the actual route.
- Local feature attribution is not automatically a suitable customer explanation.
- Reason codes should be validated against policy and adverse-action requirements if applicable.

### 9.7 Release gates

- Gate V-01: data contracts and point-in-time availability audit pass.
- Gate V-02: label definition and maturity reconciliation pass.
- Gate V-03: baseline comparison on final temporal test is complete.
- Gate V-04: borrower-group sensitivity is complete or limitations are approved.
- Gate V-05: calibration around policy bands is acceptable under an approved criterion.
- Gate V-06: capacity simulation includes hourly arrivals, queue age, and overflow.
- Gate V-07: fallback and kill-switch exercises pass.
- Gate V-08: audit reconstruction exercise passes.
- Gate V-09: privacy and fairness applicability review is resolved by authorized owners.
- Gate V-10: explanation and appeal process is approved by authorized owners.
- Gate V-11: monitoring and rollback owners accept runbooks.
- Gate V-12: decision owner signs policy configuration.
- A gate failure blocks activation.
- A waiver must be explicit, owned, time-limited, and not silently encoded in a threshold.

## 10. Action Taken

- Action in this illustrative case: propose a gated validation package.
- No model was deployed.
- No applicant was routed by this design.
- No portfolio action was taken.
- Retain the rules-plus-analyst baseline until release gates pass.
- Create a field-level availability inventory before feature selection.
- Create a label specification with servicing owner sign-off.
- Build immutable point-in-time training snapshots.
- Implement a transparent baseline model and calibration evaluation.
- Keep rules, model, calibration, and policy artifacts versioned separately.
- Simulate threshold and capacity choices against mature validation scores.
- Build a fallback path before enabling model scoring.
- Build an appeal and data-correction path before enabling automated routes.
- Conduct authorized privacy, fairness, compliance, and explanation reviews.
- Run temporal and borrower-group validation before pilot.
- Run reviewer-workflow measurement before setting review capacity.
- Pilot only if all gates pass and authorized owners approve.
- If any gate fails, take the no-change action and document why.
- The action owner is the Retail Credit Risk Director for policy choice.
- The model owner cannot independently approve lending thresholds.
- The platform owner cannot independently waive a data-quality failure.
- The operations owner cannot silently exceed queue capacity.
- The compliance or privacy owner determines applicability within their authority.

## 11. Outcome & Results

- Implementation stage: proposed.
- Evidence label for deployment outcome: not-measured.
- No live bank outcome exists in this case.
- No model performance outcome exists in this case.
- No calibration result exists in this case.
- No fairness result exists in this case.
- No privacy or legal applicability determination exists in this case.
- No analyst-capacity measurement exists in this case.
- The only computed results are illustrative derived arithmetic.
- Derived result R-01: cost boundary is 0.20 under the stated two-action costs.
- Derived result R-02: synthetic proposed cohort cost is 10,650 units.
- Derived result R-03: synthetic rules comparator cost is 13,750 units.
- Derived result R-04: arithmetic difference is 3,100 units.
- Derived result R-05: default synthetic review load is 40 against capacity 40.
- Derived result R-06: wide-band synthetic review load is 55 against capacity 40.
- Derived result R-07: delayed feature trace rejects model scoring.
- These results support only the stated formulas and state machine.
- They do not support an expected reduction in losses at any institution.
- They do not support a claim that applicants receive better outcomes.
- They do not support a claim that human review improves decisions.
- They do not support a claim that the architecture meets regulatory requirements.
- The outcome measure proposed for a future pilot is estimated decision cost on mature holdout data.
- Its comparator is the frozen rules-plus-analyst baseline.
- Its population is the approved pilot population.
- Its window must include complete label maturity.
- Its uncertainty method must be predeclared.
- Its guardrails include queue age, fallback rate, data quality, and any applicable reviewed fairness measures.
- A future pilot result must distinguish observed outcome from counterfactual inference.
- A before-after result alone can be confounded by population, economy, policy, and operations changes.
- A randomized or carefully governed phased evaluation may be needed where appropriate and permitted.

## 12. Failure Modes & What Went Wrong

| Failure ID | Failure mechanism | Discriminating check | Consequence | Bounded recovery | Verification |
|---|---|---|---|---|---|
| F-01 | Future feature leakage | Compare `available_ts` to decision time | Inflated offline performance | Remove field; rebuild snapshots | Re-run point-in-time audit |
| F-02 | Immature labels treated as good | Check maturity horizon | Biased target | Exclude censored rows | Reconcile label counts |
| F-03 | Random split leakage | Compare temporal/group split | Optimistic metrics | Use ordered group-aware split | Holdout protocol review |
| F-04 | Miscalibration | Reliability near thresholds | Wrong cost routes | Recalibrate or retain baseline | Untouched test reliability |
| F-05 | Review overload | Admission count and queue age | Delays and ad hoc decisions | Narrow band or fallback | Capacity simulation |
| F-06 | Model outage | Synthetic dependency failure | Undefined decision | Baseline/manual fallback | Kill-switch exercise |
| F-07 | Schema drift | Contract validation | Invalid vector | Reject/fallback | Compatibility test |
| F-08 | Selective labels | Compare approval coverage | Misleading population inference | Explicit limitation; approved study | Population reconciliation |
| F-09 | Reason mismatch | Route-to-reason trace | Misleading explanation | Block release | Case review |
| F-10 | Unapproved data use | Data inventory review | Privacy/fairness harm | Remove source; hold deployment | Authorized review |

### 12.1 Leakage from delayed verification

- Failure: verified income becomes available after the historical decision but is joined as if current.
- Competing cause: timestamp timezone conversion error can mimic delay.
- Discriminating check: preserve source timezone and convert both timestamps through one tested function.
- Bad fix: backfill the income value into the earlier decision row.
- Why bad: it makes offline training use knowledge unavailable online.
- Bounded fix: exclude delayed value or defer the decision under approved process.
- Recovery reasoning: a safe fallback retains baseline operation while source lag is investigated.
- Failed recovery: score with “latest known” value and call it point-in-time.
- Verification: sampled replay shows every required `available_ts <= decision_ts`.

### 12.2 Label leakage and maturity error

- Failure: accounts without twelve months of observation are labeled good because no bad event is recorded yet.
- Competing cause: a true low-risk cohort can also have few bad events.
- Discriminating check: compute maturity date and censoring flag independently of outcome value.
- Bad fix: use current delinquency as if it were a mature twelve-month label.
- Why bad: it changes the target horizon by recency.
- Bounded fix: exclude unresolved labels or use an approved survival formulation with clear target change.
- Recovery reasoning: wait for maturity rather than manufacture negatives.
- Verification: label table includes maturity coverage by decision month.

### 12.3 Selective-label bias

- Failure: historical outcomes are seen mainly for approved borrowers.
- Competing cause: missing labels can also be caused by servicing-system gaps.
- Discriminating check: compare label availability by historical route and origination status.
- Bad fix: call approved-population performance “all applicant” performance.
- Why bad: declined applicants’ counterfactual repayment is unobserved.
- Bounded fix: narrow claim to labeled population and seek approved evaluation design.
- Recovery reasoning: explicit limitation is safer than unvalidated reject inference.
- Verification: report population funnel and exclusions.

### 12.4 Calibration failure

- Failure: score 0.20 corresponds to observed 0.35 bad rate in its action band.
- Competing cause: small-bin noise can make a stable model appear miscalibrated.
- Discriminating check: bin counts, confidence intervals, time slices, and calibration holdout.
- Bad fix: preserve the 0.20 threshold because AUC is high.
- Why bad: the cost calculation needs probability, not ordering.
- Bounded fix: refit calibration on permitted validation data or revert to baseline.
- Recovery reasoning: thresholds must move only after independent validation.
- Verification: reliability plot on untouched final period.

### 12.5 Capacity failure

- Failure: a wider review band sends 55 items to 40 slots.
- Competing cause: operational completions may be lower than nominal staffing capacity.
- Discriminating check: measure completed quality-controlled decisions and queue age by hour.
- Bad fix: admit all items and assume reviewers will catch up.
- Why bad: pending applications accumulate and treatment becomes inconsistent.
- Bounded fix: enforce admission control and explicit overflow fallback.
- Recovery reasoning: preserve a determinate route rather than hide backlog.
- Verification: overload simulation reports 15 overflow items in the stated scenario.

### 12.6 Reviewer feedback contamination

- Failure: post-decision reviewer notes are added as features for the original score.
- Competing cause: some notes may have existed before score request.
- Discriminating check: note creation timestamp and workflow state.
- Bad fix: include all notes because they are useful.
- Why bad: later human knowledge leaks into model training.
- Bounded fix: include only predecision structured fields with availability evidence.
- Recovery reasoning: retain notes for audit and workflow, not uncontrolled feature use.
- Verification: feature registry maps every note-derived field to permitted source time.

### 12.7 Drift and policy drift

- Failure: applicant mix, economic environment, terms, or verification vendor changes.
- Competing cause: score-distribution change can be caused by a logging bug.
- Discriminating check: compare raw source distributions, schema versions, and audit completeness.
- Bad fix: retrain automatically on immature labels.
- Why bad: it can codify transient noise and break governance.
- Bounded fix: hold policy, investigate, then use approved retraining process.
- Recovery reasoning: baseline fallback is available for severe integrity failure.
- Verification: monitored change has owner, threshold, and disposition.

### 12.8 Explanation and appeal failure

- Failure: applicant-facing reason is generated from a generic model feature ranking unrelated to route.
- Competing cause: policy rule and model reason can both contribute.
- Discriminating check: replay route and reason mapping from immutable versions.
- Bad fix: expose raw coefficients as a customer explanation.
- Why bad: technical attribution may be misleading or unusable.
- Bounded fix: authorized reason mapping tied to actual decision factors.
- Recovery reasoning: hold automated notices until mapping review is complete.
- Verification: sampled audit reconstructs route, factors, and policy version.

### 12.9 Privacy and fairness applicability failure

- Failure: a team assumes a metric or data source is allowed because it is common elsewhere.
- Competing cause: unclear ownership can look like deliberate evasion.
- Discriminating check: named authorized decision and source-specific inventory.
- Bad fix: declare “fair” based on one aggregate statistic.
- Why bad: applicability, measurement, and tradeoffs remain unresolved.
- Bounded fix: mark OPEN, obtain review, and restrict deployment until resolved.
- Recovery reasoning: lack of certification is not filled by model documentation.
- Verification: signed scope decision and approved monitoring plan where applicable.

### 12.10 Failed and successful recovery reasoning

- Successful recovery: late feature is rejected, fallback executes, event is auditable.
- Why successful: it preserves the point-in-time invariant and a terminal route.
- Failed recovery: late feature is silently replaced with current value.
- Why failed: it hides data lag and produces an invalid score.
- Successful recovery: queue is full, overflow follows explicit baseline policy.
- Why successful: capacity violation is visible and deterministic.
- Failed recovery: queue admits 15 excess items with no owner or SLA.
- Why failed: it converts a capacity constraint into hidden applicant delay.
- Successful recovery: calibration fails, policy reverts to baseline pending investigation.
- Why successful: cost thresholds are not used with unreliable probabilities.
- Failed recovery: threshold is hand-tuned on final test until metrics look favorable.
- Why failed: final-test leakage removes credible evaluation.

## 13. Transferable Lesson

- A classifier becomes a consequential system only when its score is tied to a controlled action.
- The action needs a population, horizon, costs, constraints, and fallback.
- Point-in-time validity is stronger than having a timestamped warehouse table.
- Every feature needs evidence of what was known when the decision was made.
- A mature label is a different object from a recent operational signal.
- Selective labels limit what historical supervised learning can claim.
- Calibration is not cosmetic when decisions use expected cost.
- AUC cannot tell an owner whether 0.20 is a defensible approval boundary.
- Capacity turns a score band into a queueing and allocation policy.
- Human review is not automatically a safety mechanism; it has error, delay, and capacity limits.
- Hard rules should remain explicit where policy requires them.
- A fallback must be designed before the primary path is trusted.
- A safe system distinguishes model failure from applicant risk.
- A delayed feature is a data-integrity event, not evidence of borrower risk.
- A review overflow is an operations event, not a reason to erase the capacity constraint.
- Fairness and privacy applicability must be determined, not assumed from an architecture diagram.
- Explanation needs to match the actual route and approved purpose.
- A synthetic calculation can teach mechanism without impersonating an institutional result.
- The strongest recommendation can be to retain baseline until validation evidence exists.
- Transfer to insurance underwriting: replace loan bad label with approved claim-risk outcome and preserve timing rules.
- Transfer to medical triage: replace cost model with clinical utility only under appropriate clinical governance; this lending design does not transfer as a clinical recommendation.
- Transfer to fraud review: replace label horizon and account for adversarial adaptation and alert latency.
- Transfer to hiring: do not copy credit costs; determine lawful purpose, fairness, and human-review meaning independently.
- Transfer to retrieval ranking: calibration may determine abstention or manual escalation, while capacity still binds.
- Transfer to any domain: first build the action contract, then select the predictive model.

## 14. Connections & Cross-References

- Related module connection: supervised learning supplies the estimator but not the policy objective.
- Related module connection: feature engineering supplies transformations but must respect availability time.
- Related module connection: model evaluation supplies splits and calibration checks but must include action capacity.
- Related module connection: optimization supplies constrained routing under review capacity.
- Related module connection: data governance supplies provenance, access, and retention obligations.
- Related module connection: responsible AI supplies questions, not a pre-certified fairness conclusion.
- Related case pattern: human-in-the-loop is useful only when review changes decisions and capacity is measured.
- Related case pattern: abstention or fallback is preferable to false confidence during input or service failure.
- Open question: which exact product economics and policy constraints should define costs?
- Open question: what population can be validly evaluated given historical selection?
- Open question: what review evidence is permitted and decision-useful?
- Open question: which privacy, consumer-credit, and fairness requirements apply?
- Open question: what explanation and appeal standards govern actual deployment?

### Interview pack

#### Q1. Why is AUC insufficient for choosing this lending route?

- Checked answer: AUC measures ranking across score pairs, while the route needs calibrated probabilities, asymmetric error costs, thresholds, and a capacity-constrained review band.
- Reasoning: two models can have similar AUC but different calibration near 0.20, producing different expected-cost decisions.
- Follow-up: ask the candidate to name a metric for probability quality; expected answer includes calibration curve, Brier score, or bin-level observed-versus-predicted rates.
- Acceptable alternatives: say ranking can be used if thresholds are empirically optimized on a valid holdout, provided calibration and policy interpretation are then treated separately.
- Red flags: “highest AUC is automatically best,” or “accuracy selects the business threshold.”
- Scoring: 0 misses action distinction; 1 names calibration; 2 explains costs and capacity; 3 also names temporal validation and band-local checks.
- Answer support: Sections 5c, 6.4, and 9.2.

#### Q2. Derive the simplified approve/decline threshold using the stated costs.

- Checked answer: equate `p*1000` to `(1-p)*250`; solving gives `1000p=250-250p`, `1250p=250`, and `p=0.20`.
- Reasoning: below 0.20, expected approval cost is lower; above it, expected decline cost is lower under the stated two-action assumptions.
- Follow-up: what happens when bad-approval cost rises to 1,500? Answer: threshold becomes `250/1750=0.1429`.
- Acceptable alternatives: derive with a decision table rather than algebra if units and conclusion are correct.
- Red flags: divide 1000 by 1250, ignore good-decline cost, or claim 0.20 is a universal lending cutoff.
- Scoring: 0 wrong setup; 1 correct result only; 2 correct derivation and units; 3 states calibration and omitted-cost limits.
- Answer support: M-01, M-02, M-03, and Section 6.4.

#### Q3. What is the most dangerous feature-engineering error in this case?

- Checked answer: using a feature that became available after the historical decision timestamp, because it creates future leakage and invalidates offline-to-online consistency.
- Reasoning: a useful late income verification can make a model look strong while being impossible to reproduce at request time.
- Follow-up: how do you detect it? Require per-record source availability timestamps and enforce `available_ts <= decision_ts`.
- Acceptable alternatives: identify post-decision servicing or reviewer notes if the candidate explains their timing.
- Red flags: “remove it only if correlation is too high,” or “warehouse load time proves availability.”
- Scoring: 0 generic leakage; 1 names future data; 2 specifies timestamp predicate; 3 gives safe fallback and audit behavior.
- Answer support: Sections 5d and 6.1.

#### Q4. A feature arrived five minutes after decision time. What route should the system take?

- Checked answer: reject learned-model scoring with `REJECT_DELAYED_FEATURE` and invoke the approved baseline/manual fallback; do not backdate or substitute the latest value.
- Reasoning: the failure concerns evidence integrity, not the applicant’s repayment risk.
- Follow-up: what must be logged? Application ID, timestamps, feature ID, failure reason, fallback route, and versions.
- Acceptable alternatives: defer decision only if an approved operational policy allows it and applicant impact is addressed.
- Red flags: impute the future verified value, default it to zero without approved semantics, or silently drop the request.
- Scoring: 0 scores anyway; 1 rejects; 2 names fallback; 3 explains invariant and audit trail.
- Answer support: Sections 5d, 6.1, and 8.2.

#### Q5. Compute the stated synthetic cohort cost.

- Checked answer: `8*1000 + 5*250 + 40*35 = 8000 + 1250 + 1400 = 10650` currency units.
- Reasoning: false-negative approvals, false-positive declines, and review work are distinct cost terms.
- Follow-up: what mistake occurs if review cost is omitted? The calculated cost falls to 9,250 and exaggerates the synthetic benefit by 1,400.
- Acceptable alternatives: use a table of error categories with the same total.
- Red flags: charge all approvals, count true positives as a loss, or call 10,650 realized loss.
- Scoring: 0 wrong arithmetic; 1 correct total; 2 correct components; 3 states constructed-input limitation.
- Answer support: M-05 and Section 6.2.

#### Q6. The review band sends 55 cases to 40 slots. What should happen?

- Checked answer: the capacity boundary fails by 15; apply the preapproved overflow route and emit an overflow audit event rather than admitting an unbounded queue.
- Reasoning: capacity is a hard operational constraint, and hidden delay changes treatment.
- Follow-up: how would you prevent repeat overflow? Recalibrate thresholds from actual distributions, adjust staffing, or change policy after validation.
- Acceptable alternatives: a priority queue is acceptable only with explicit cutoff and applicant-impact controls.
- Red flags: “review all 55 eventually,” or “raise capacity in the config without staffing.”
- Scoring: 0 ignores capacity; 1 identifies 15 excess; 2 states fallback; 3 distinguishes daily count from intraday queue age.
- Answer support: Sections 6.3 and 8.3.

#### Q7. Why use temporal validation rather than only a random split?

- Checked answer: temporal validation better approximates predicting later decisions from earlier data and reveals drift; random splitting can mix conditions and repeated patterns across periods.
- Reasoning: underwriting mix, policy, source systems, and economics can change over time.
- Follow-up: what does temporal validation not solve? It does not by itself prevent repeated borrower leakage or selective-label bias.
- Acceptable alternatives: blocked cross-validation may be useful when multiple mature periods exist.
- Red flags: “random is always unbiased,” or “time split makes calibration unnecessary.”
- Scoring: 0 says only “more realistic”; 1 names future period; 2 names drift/leakage; 3 notes remaining limitations.
- Answer support: Sections 5a and 9.3.

#### Q8. Why add borrower-group validation?

- Checked answer: repeated applications or stable borrower patterns can appear in both train and test, making a model appear to recognize a person rather than generalize to a new decision.
- Reasoning: group isolation is a sensitivity test for entity repetition.
- Follow-up: what if borrower key is unavailable? Document the limitation and explore approved stable linkage carefully; do not invent identity matches.
- Acceptable alternatives: group by household only if such linkage is lawful, accurate, and available at decision time.
- Red flags: “group split is unnecessary with many rows,” or using future identity resolution.
- Scoring: 0 no leakage concept; 1 repeats entity; 2 explains optimistic performance; 3 states coverage/timing limitation.
- Answer support: Sections 5a and 9.4.

#### Q9. Explain selective-label bias in one sentence, then give a consequence.

- Checked answer: repayment labels are often observed mainly for historically approved or originated applicants, so measured performance may not generalize to all applicants.
- Reasoning: a decline removes the counterfactual repayment outcome.
- Follow-up: is reject inference automatically the answer? No; it needs an explicit justified method and evaluation plan.
- Acceptable alternatives: describe sample-selection bias with the same causal limitation.
- Red flags: “declined applicants are always bad,” or treating missing outcomes as good labels.
- Scoring: 0 misses missing counterfactual; 1 defines selection; 2 names population consequence; 3 rejects casual imputation.
- Answer support: Sections 4, 5a, and 12.3.

#### Q10. A model has high AUC but poor calibration near 0.20. Can it use the cost threshold?

- Checked answer: not safely as a probability threshold; first validate or improve calibration on an appropriate held-out period, or retain a policy not relying on that probability interpretation.
- Reasoning: M-03 assumes `p` is calibrated.
- Follow-up: can monotonic calibration change ranking? Some calibration mappings preserve ordering, but the relevant question is validated probability quality and overfitting risk.
- Acceptable alternatives: reframe threshold empirically only if final test remains untouched and operational/cost interpretation is honest.
- Red flags: “AUC compensates for calibration,” or fit calibration on final test.
- Scoring: 0 use directly; 1 says calibrate; 2 explains threshold assumption; 3 identifies heldout data requirement.
- Answer support: Sections 5c, 9.5, and 12.4.

#### Q11. What does a fairness review need that a single overall metric lacks?

- Checked answer: authorized scope, valid group definitions and data governance, relevant comparison population, error tradeoffs, uncertainty, and interpretation of the actual policy including review and fallback.
- Reasoning: aggregate calibration or approval rate cannot settle applicability or disparate treatment concerns.
- Follow-up: should the team infer protected attributes to fill gaps? Not without explicit authorized basis and governance.
- Acceptable alternatives: emphasize legal/compliance determination as prerequisite.
- Red flags: “the model is fair because protected attributes were removed,” or “one parity metric certifies fairness.”
- Scoring: 0 generic ethics; 1 names groups; 2 names policy and governance; 3 discusses uncertainty and conflicting metrics.
- Answer support: Sections 4, 9.6, and 12.9.

#### Q12. How should the system behave during model-service outage?

- Checked answer: invoke a preapproved baseline/manual fallback or hold route, log the outage and versions, and avoid inventing a score.
- Reasoning: continuity must not bypass mandatory rules or create untraceable lending decisions.
- Follow-up: may it use a cached model? Only if a specific validated last-known-approved artifact behavior is approved and unambiguous.
- Acceptable alternatives: temporarily halt automated offers if that is the approved safe route.
- Red flags: retry indefinitely, select arbitrary default approval, or silently use unknown model state.
- Scoring: 0 no fallback; 1 names baseline; 2 names logging and rules; 3 identifies approval and artifact-version condition.
- Answer support: Sections 8.2 and 8.5.

#### Q13. What is a plausible but wrong method for setting the review band?

- Checked answer: widen the band until all analyst capacity is always used, because utilization ignores whether review changes enough decisions to justify cost and ignores peak overflow.
- Reasoning: review is an information action, not a sink for spare labor.
- Follow-up: what evidence is needed instead? Review outcomes, capacity by hour, score distribution, error/cost changes, and applicant-impact constraints.
- Acceptable alternatives: choose the band by estimated value of information subject to validated capacity.
- Red flags: “more review is always safer,” or selecting it solely from AUC.
- Scoring: 0 accepts utilization rule; 1 notes capacity; 2 explains value and peaks; 3 includes review quality.
- Answer support: Sections 5c, 6.3, and 8.3.

#### Q14. Changed constraint: capacity drops from 40 to 20 per day. What changes?

- Checked answer: retain the hard rule `review_load <= 20`, narrow or reprioritize the review band only after simulation, and define explicit overflow; do not leave prior thresholds unchanged.
- Reasoning: policy feasibility has changed even if score calibration has not.
- Follow-up: does a 20-slot constraint change the cost threshold 0.20? Not the simplified two-action boundary by itself, but it changes feasibility of the review action.
- Acceptable alternatives: increase capacity only if actual staffing and quality controls support it.
- Red flags: halve all probabilities, claim model retraining is automatically required, or admit 40 anyway.
- Scoring: 0 ignores constraint; 1 narrows band; 2 separates threshold from review feasibility; 3 includes overflow and simulation.
- Answer support: Sections 6.3 and 8.3.

#### Q15. An applicant disputes a wrong income value. Should that appeal data retrain the model immediately?

- Checked answer: no; correct the decision under the approved appeal process, preserve original and corrected snapshots, and admit data to training only through reviewed label/feature governance.
- Reasoning: appeal records can have different selection, timing, and intent than training observations.
- Follow-up: what is the immediate control? Route corrected information through the appeal or reconsideration workflow with an auditable version.
- Acceptable alternatives: use confirmed correction as a future data-quality signal after governance review.
- Red flags: overwrite original record, retrain online from one appeal, or ignore factual correction.
- Scoring: 0 immediate retraining; 1 says correction; 2 preserves versions; 3 explains feedback-selection risk.
- Answer support: Sections 4, 8.2, and 12.8.

## 15. Quick Reference Card

| Field | Quick reference |
|---|---|
| Decision | Route one eligible application to approve, decline, review, or approved fallback using decision-time evidence. |
| Owner | Retail Credit Risk Director owns policy; Analytics owns model integrity; Operations owns review workflow; authorized functions determine applicability. |
| Input/output grain | One application at `decision_ts_utc` -> calibrated `p_bad`, route, reasons, and immutable audit version set. |
| Baseline | Deterministic hard rules plus fixed analyst queue; retain it when gates fail. |
| Core formula | `t*=C_good_decline/(C_bad_approve+C_good_decline)` only with calibrated probability, fixed illustrative costs, and no review action. |
| Illustrative numbers | Costs 1000/250/35 produce `t*=0.20`; synthetic FN=8, FP=5, R=40 gives 10,650 units. |
| Capacity rule | Admit review only while `review_load <= K`; at 55 routes and K=40, 15 items overflow to approved fallback. |
| Point-in-time invariant | Required feature must satisfy `available_ts <= decision_ts_utc`; otherwise reject scoring and do not backdate data. |
| Validation | Mature temporal final test, borrower-group sensitivity, calibration near thresholds, capacity simulation, fallback exercise, audit replay. |
| Key limitation | Synthetic arithmetic is not bank data, model performance, legal advice, fairness certification, or deployment evidence. |
| Evidence/stage | Illustrative / proposed / not-measured for deployment outcome; derived arithmetic is explicitly labeled. |
| Open reviews | Label source, data contracts, privacy, fairness, consumer-credit applicability, explanation, appeal, staffing, and approved costs. |
| Stop condition | Any failed point-in-time, label, calibration, capacity, governance, or audit gate retains baseline or invokes fallback. |
| Review record | `RUN-SMOKE-01` manually observed delayed-feature rejection, 40/40 capacity boundary, 55/40 overflow, and 10,650 cost arithmetic; not a production test. |
