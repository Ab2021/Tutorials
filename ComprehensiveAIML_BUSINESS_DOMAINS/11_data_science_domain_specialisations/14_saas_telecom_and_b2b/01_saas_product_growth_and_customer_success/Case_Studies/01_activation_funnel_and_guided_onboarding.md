---
case_id: "CASE-14-01-001"
case_title: "Guided onboarding: deciding whether an activation intervention is worth shipping"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "analytical"
domain: "B2B SaaS product growth / activation and instrumentation"
primary_pattern: "evaluation-mismatch"
secondary_patterns: ["data-leakage", "human-in-the-loop"]
evidence_level: "not-measured"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "Product analytics lead with the PLG growth owner"
source_index: "../INDEX.md"
review_trigger: "Change to the event contract, workspace-account mapping, activation definition, price book, or experiment assignment service"
---

# Case study: Guided onboarding and the activation decision

**Evidence declaration.** Every count, rate, price, cost, and capacity figure in this case is
constructed for teaching. The arithmetic is checked; the population is not observed, and no
company's funnel, revenue, experiment, or conversion effect is reported here. Any statement about
what a real product achieved would require a named measurement this case does not have.

**Stage declaration.** Implementation stage: proposed. Editorial status: draft, not reviewed.
A reader may reconstruct the method and recompute every number, but nothing here is a claim that
an intervention was deployed, that an experiment ran, or that revenue changed.

## 1. The One-Line Framing

> Given **per-workspace product events and subscription records for a signup cohort**, decide
> **whether to ship a guided-onboarding intervention to all eligible new workspaces**, for the
> **PLG growth owner**, to improve **the share of workspaces that reach a defensible
> activation state within 14 days**, subject to **a support-contact guardrail and a
> fixed experiment cost**, compared with **the current self-serve checklist baseline**.

The unit of analysis is the workspace, not the user, not the signup form submission, and not the
billing account. That choice is the first substantive decision in the case: a metric computed at
the wrong grain can look better while the commercial outcome gets worse. Task category:
funnel/cohort analysis supporting a ship-or-not decision, evaluated once per cohort cycle
(monthly), with the decision recurring at each cycle rather than per request.

Success condition in plain language: more workspaces reach the activation state per 1,000
eligible signups, without breaching the support-contact guardrail, at a program cost that the
projected contribution margin covers with a stated margin of safety.

Primary pattern: `evaluation-mismatch` — the obvious dashboard metric (checklist completion) is
not the decision metric (activated workspaces that later convert). Secondary patterns:
`data-leakage` (late-arriving events and post-deadline behaviour leaking into a "14-day" metric)
and `human-in-the-loop` (support-capacity and review constraints on the intervention).

The simpler framing that misses the decision: "did checklist completion go up?" Completion can
rise because the checklist was made easier, because the completion event fires on page view, or
because ineligible trial workspaces were added to the denominator. None of those tells the owner
whether to spend the next quarter's onboarding engineering budget on this intervention.

**Interview cue.** A reader should be able to state the input grain, output decision, beneficiary,
constraint, baseline, and evidence level in thirty seconds without naming a tool.

## 2. Context & Background

| Context field | This case |
|---|---|
| Domain and product | B2B SaaS collaboration product sold per workspace with seat-based annual contracts |
| Workflow before decision | Marketing acquires a signup; the workspace is provisioned; a checklist suggests three setup steps |
| Workflow after decision | Either the intervention is shown to eligible workspaces, or the current checklist continues |
| Stakeholders | Growth owner (decision), product analytics (metric contract), platform team (event pipeline), support operations (guardrail), finance (price and margin) |
| Baseline | Self-serve checklist with no guided prompts, in place for the prior four quarters |
| Scale (illustrative) | 20,000 signups per 91-day cohort; 172,400 events across the funnel stages |
| Timing | Cohort anchored on signup date; activation window 14 days; reporting lag 3 days for late events |
| Geography and boundary | Single billing currency, single product edition for the cohort; multi-region signups share one event pipeline |
| Prior attempts | A tooltip tour shipped two quarters earlier without a pre-registered metric; its effect is unknown and it is not a valid baseline |
| Applicable requirements | Internal data-retention policy limits event retention to 25 months; no external regulation is interpreted here |

**Why the current process persists.** The checklist is cheap, requires no new instrumentation,
and gives every new workspace a consistent starting point. Its weakness is that it measures
*clicks* and not *value realisation*: a workspace can complete all three checklist items by
clicking through and still never share a document with a colleague, which is the behaviour the
commercial model depends on. Sales also cannot use checklist completion as a leading indicator,
because the correlation with later conversion was never established on a mature cohort.

**Separating known from assumed.** Known in this case: the product's activation concept (a
workspace where a document with a second collaborator exists within the window) is a chosen
internal definition, not a natural law. Assumed: that the intervention changes the probability of
reaching that state; that the price book is stable over the cohort; that support contacts are
countable per workspace. Historical condition: the previous tour had no measurement plan, so it
cannot be used as evidence in either direction.

**Interview cue.** Explain who experiences the failure and why the current process persists
despite its limitations. The answer should mention that the cheap proxy (clicks) is visible and
the expensive truth (collaborative value) arrives late.

## 3. The Business Question

The question below is an **illustrative stakeholder question**, reconstructed for teaching; it is
not a quotation from a real person.

> "We can spend the next six weeks building guided setup prompts. Before we commit the team,
> tell me whether it will move activated workspaces enough to pay for itself, and tell me what
> would have to be true for that answer to flip."

| Decision contract | This case |
|---|---|
| Question | Does a guided-setup intervention increase the 14-day activation rate enough to justify its program cost, without breaching the support guardrail? |
| Decision owner | PLG growth owner, who can approve, reject, or defer to a smaller pilot |
| Trigger and deadline | Annual planning cycle closes in three weeks; engineering capacity must be reserved or released |
| Available actions | (a) ship to all eligible new workspaces, (b) run a randomised pilot on a subset, (c) do not ship, (d) ship only to a defined segment after a pilot |
| Value and downside | Incremental gross margin from activated workspaces versus build, serving, and support cost; downside is friction, support load, and opportunity cost |
| Objective | Incremental activated workspaces per 1,000 eligible signups, aggregated over the cohort, not per session |
| Hard constraints | Support-contact rate must not exceed the baseline by more than 1.5 percentage points; no change to billing or entitlement logic; event retention 25 months |
| Scope | New self-serve workspaces in one edition, one currency, 14-day activation window, next two cohorts |
| Out of scope | Sales-assisted deals, enterprise custom onboarding, price changes, renewal behaviour beyond 12 months |

**Objective versus proxy.** The business objective is contribution margin. The decision metric is
incremental activations, because the conversion-to-paid step is too slow (median 38 days in the
illustrative plan) to observe inside the pilot horizon. That substitution is only defensible if
the assumed conversion difference between activated and non-activated workspaces is stated as an
assumption and stress-tested, which Section 6e does. Forecast error, checklist completion, and
time-in-product are all rejected as the decision metric: each can improve while margin falls.

**Expensive false decision.** Shipping a six-week build to all workspaces when the true lift is
0.2 percentage points costs the build plus a permanent serving and support increment, and it
consumes the quarter's growth engineering capacity. A cheap false decision is running a pilot so
small that its confidence interval spans both "do not ship" and "ship to everyone".

**Interview cue.** Ask which objective is optimised, what happens with no intervention, and what
an expensive false decision costs. Then ask what makes this question answerable by an action
rather than by more analysis.

## 4. Assumptions & Constraints

| ID | Assumption or constraint | Type | Evidence or verification plan | Status | Impact if wrong | Fallback | Recheck trigger / owner |
|---|---|---|---|---|---|---|---|
| A-01 | Workspace maps to exactly one billing account in the cohort | Data | Cardinality check on `workspace_id` to `account_id` over the cohort | unverified | Activation could be counted twice or attributed to a merged account | Use the account as the unit and recompute | Account-merge feature release / analytics lead |
| A-02 | `artifact_shared` requires a second distinct collaborator | Model | Inspect producer payload; sample 200 events manually | unverified | Activation would count single-user documents as collaboration | Add collaborator-count predicate to the metric | Event schema change / analytics lead |
| A-03 | Event time is trustworthy within 5 minutes | Data | Compare producer clock skew and receive-time lag distribution | unverified | Window boundary misclassification near day 14 | Use receive time with a documented window extension | SDK release / platform team |
| A-04 | The pre-deadline event set is complete after 3 days of lag | Data | Late-arrival curve by event type, 30-day history | unverified | Understated activation, biased cohort comparison | Extend the maturity lag and restate | Pipeline version / platform team |
| A-05 | Conversion difference between activated and non-activated workspaces is 0.29 | Business | Finance-approved planning assumption, not measured here | illustrative | Break-even lift changes by a factor of three | Recompute with 0.10 and 0.45 | Price book or packaging change / finance |
| A-06 | Gross margin rate 0.78 and annual contract value 5,400 USD are stable for 12 months | Business | Price book version; no discount programme changes | illustrative | Contribution per activation is overstated or understated | Use net revenue instead of list | Discount policy change / finance |
| A-07 | Support contacts are attributable to a workspace | Data | Ticket-to-workspace join coverage check | unverified | Guardrail cannot be computed | Use account-level contact rate with caveat | Ticketing schema change / support ops |
| A-08 | The intervention does not change who is eligible to sign up | Business | Eligibility rule is outside the experiment | assumed | Treatment effect mixes with selection | Freeze eligibility before assignment | Any eligibility change / growth owner |
| A-09 | The previous tooltip tour had no measurable effect either way | Business | No pre-registered metric, no clean cohort | unverified | Team may over- or under-estimate a prior learning | Treat as unmeasured, do not reuse as baseline | Retrospective study / analytics lead |
| A-10 | One assignment per workspace, no re-entry after deletion | Data | Assignment key uniqueness test | unverified | Duplicate exposure inflates treatment share | Deduplicate on assignment key | Assignment service change / platform |
| C-01 | Support-contact rate increase limited to 1.5 points | Business guardrail | Support ops sign-off before launch | confirmed for pilot | A high-friction intervention can pass the activation test and still harm customers | Cap exposure and monitor daily | Any guardrail change / support ops |
| C-02 | No billing or entitlement logic change in scope | Cost and risk | Code-ownership boundary | confirmed | Revenue and access defects | Separate change request | Billing release / finance systems |
| C-03 | Event retention 25 months | Privacy and cost | Retention policy version | confirmed | Cohorts beyond retention cannot be recomputed | Archive aggregates only | Policy revision / privacy owner |
| C-04 | Build budget 90,000 USD over 8 workstream-months | Cost | Approved capacity plan | illustrative | Program cost changes break-even | Reduce scope to partial intervention | Capacity re-plan / growth owner |

The three assumptions most likely to reverse the recommendation are A-05 (conversion difference),
A-04 (late-event completeness), and A-01 (workspace-account cardinality). A-05 moves the answer by
more than the entire measured lift; A-04 can manufacture or erase a two-point difference; A-01 can
silently change the population. Each has a named check and a fallback, and none is treated as
verified merely because it is written down.

**Interview cue.** Change one invalidating assumption and explain how the method, architecture,
or decision changes. A strong answer picks A-04 and shows that the metric's maturity lag, not the
model, is what has to change.

## 5. Data, Tools & Method

### 5a. Data Specification

| Field | Specification |
|---|---|
| Unit of analysis | Workspace (`workspace_id`), one row per workspace-signup cohort member |
| Volume and window | 20,000 signups; 172,400 funnel events; cohort window 91 days; 14-day activation window |
| Inputs | `signup_created`, `workspace_provisioned`, `data_source_connected`, `artifact_shared`, `invite_sent`, `checklist_item_completed`, `support_contact`, plan and seat fields |
| Primary key | `workspace_id`; event key `event_id`; producer sequence per workspace where available |
| Outcome/label | Activation = a shared artifact with ≥2 distinct collaborators within 14 days of `signup_created`, evaluated at maturity |
| Population and exclusions | Self-serve signups in one edition; excluded: sales-assisted, internal test tenants, workspaces deleted within 24 hours |
| Join semantics | `workspace_id` → `account_id` many-to-one in the cohort; `user_id` many-to-many across workspaces over time |
| Splits | Cohort by signup week; experiment assignment by workspace hash; no user-level reuse across arms |

Field-level contract detail that matters for the decision:

- `event_time` is producer-assigned; `receive_time` is pipeline-assigned. Both are retained. The
  activation window uses `event_time`, with a documented tolerance, because a delayed batch upload
  must not silently extend a workspace's window.
- `artifact_shared` carries `collaborator_count` at emission. A document shared with the owner
  only is not collaboration; the predicate is `collaborator_count >= 2` **and** at least one
  collaborator distinct from the owner.
- `checklist_item_completed` fires on click, not on successful completion of the underlying
  action. It is retained as a diagnostic, never as the outcome.
- `support_contact` is a ticket-created event with a channel attribute; tickets closed within one
  minute without an agent reply are labelled as noise and excluded from the guardrail, with that
  exclusion reported separately.
- Plan and seat fields are point-in-time subscription snapshots, captured at signup and at day 14,
  so a mid-window upgrade does not retroactively change eligibility.

**Exclusions and their consequence.** Excluding test tenants is uncontroversial. Excluding
workspaces deleted within 24 hours removes a real, small population that may be systematically
different; the case reports the excluded count and recomputes the rate without the exclusion as a
sensitivity. Dropping workspaces with no `data_source_connected` event would be a leakage-grade
error, because that event is part of the causal path being measured: a workspace that never
connects a source cannot reach activation, and removing it would inflate the rate.

### 5b. Tools & Stack

| Component | Role in this case | Status |
|---|---|---|
| Event pipeline (producer SDK → queue → warehouse) | Supplies behavioural facts with both timestamps | proposed for the pilot; existing in the product |
| Warehouse tables (`events`, `workspaces`, `subscriptions`, `tickets`) | Analytical joins at declared grain | proposed |
| Transformation layer with versioned models | Materialises the funnel and activation flag | proposed |
| Experiment assignment service | Deterministic hash assignment by `workspace_id` | proposed |
| Notebook or SQL analysis environment | Recomputes this walkthrough | illustrative |
| Dashboard with metric version label | Publishes one number with its contract | proposed |

No machine-learning model is required for the primary decision. That is deliberate: the case is
about a decision and its measurement, and adding a propensity model would create a second,
harder-to-validate object without changing the action. A ranking model is considered in Section
5c and rejected for this decision, with the reason recorded.

### 5c. Method Summary

`signup events -> identity and workspace resolution -> eligible cohort -> per-workspace funnel
state as of day 14 (mature) -> arm assignment -> rate comparison -> economic ledger ->
ship / pilot / do not ship`.

| Candidate | Evidence considered | Strength | Limitation / avoid when | Status | Revisit if |
|---|---|---|---|---|---|
| Checklist completion rate as decision metric | Existing dashboard | Cheap, immediate | Measures clicks, not value; gameable by UI changes | rejected | Never for this decision |
| Pre/post comparison around launch | Operational convenience | No assignment infrastructure | Confounded by seasonality, acquisition mix, price changes | rejected | Only with a stable, audited population |
| Randomised workspace-level pilot | Standard assignment service | Unbiased arm comparison; supports interval estimation | Needs volume for the target effect size; cannot measure 12-month retention | chosen | If volume falls below the power requirement |
| Propensity-matched observational comparison | Historical data available | Works without assignment | Unmeasured confounding; matching on post-treatment variables is invalid | considered, not tested | If randomisation is impossible and a defensible design exists |
| Uplift model targeting only persuadable workspaces | Requires labelled treatment/control data | Could raise efficiency | Needs the pilot to exist first; effect heterogeneity is unknown | considered, deferred | After a successful pilot produces both arms |
| Activation definition by machine-learned value proxy | Model on later conversion | Aligns metric with revenue | Circular if trained on the same cohort; unstable and unexplainable | rejected | Only with out-of-time validation and owner sign-off |

**Chosen method.** Randomised workspace-level assignment during a fixed enrolment window, with
the decision metric computed at maturity, a pre-declared guardrail, and an economic ledger whose
assumptions are finance-owned. Thresholds are fixed before launch: ship if the lower bound of the
95% interval on the activation-rate difference exceeds the break-even lift; pilot again if the
interval spans it; do not ship if the upper bound is below it.

**Baseline.** The current checklist, measured on the immediately preceding comparable cohort with
the same metric contract and the same maturity lag. Using the previous cohort rather than a
concurrent control is a weakness that the concurrent control removes.

### 5d. Data provenance, quality, exclusions and leakage controls

| Concern | Control in this case |
|---|---|
| Provenance | Each table has an owner, a refresh cadence, and a versioned transformation commit |
| Duplicates | Deduplicate on `event_id`; detect duplicate `workspace_provisioned` per workspace |
| Late events | Maturity lag 3 days; late-arrival rate measured per event type, reported |
| State order | Reject impossible sequences (`artifact_shared` before `workspace_provisioned`) from the activation flag, and count them |
| Identity | No user-level linking across workspaces for the outcome; report account-merge collisions |
| Leakage | Features and outcomes are cut at day 14 with `event_time`; no invoice, renewal, or support-resolution fields enter the decision metric |
| Selection | Report every exclusion count; recompute without the deletion exclusion |
| Retention | Cohort aggregates retained beyond event retention; no raw events after 25 months |

The most dangerous leakage path is not a feature in a model; it is the outcome window. If the
activation flag is computed from "current state" rather than from events with `event_time` within
14 days, then workspaces that activate on day 30 inflate the rate for cohorts that happen to be
older when the query runs. The control is mechanical: the flag is a function of the cohort's own
cutoff, and the query refuses to run if the cohort's maturity lag has not elapsed.

A second leakage path is assignment-time. If assignment happens at first login rather than at
workspace provisioning, workspaces that never log in are excluded from both arms, and the
comparison silently conditions on a post-treatment behaviour. Assignment is therefore at
provisioning, before any product exposure.

### 5e. Reproducibility and method-selection ledger

| Artifact/source ID | Location | Version/date | Purpose | Accessibility | Reproduction limitation |
|---|---|---|---|---|---|
| SRC-01 | Internal event schema documentation | contract v3, 2026-11-01 | Defines event names and payloads | restricted | Payload examples here are paraphrased |
| SRC-02 | Price book extract | 2026-11-01 edition | ACV and discount structure | restricted | Real discount distribution not used |
| SRC-03 | Retention policy | policy v4 | Bounds event retention | restricted | Not quoted verbatim |
| CALC-01 | This document, Section 6 | arithmetic recomputed by hand and with an independent calculator | Funnel and economics | public | Constructed inputs |
| RUN-01 | None | — | No experiment has been executed | unavailable | No observed result exists |

Random seeds are irrelevant to the primary calculation; they matter only for any bootstrap used to
estimate interval width, which the case reports as an analytic approximation with its formula.

## 6. Analysis Walkthrough

**Evidence declaration.** The cohort below is constructed. Its internal arithmetic is checked in
this section and reproduced with an independent calculator; it is not a measurement of any
product, and the words "lift", "effect", and "improvement" in this section refer to the
constructed scenario unless a hypothesis test is explicitly described.

### 6a. Funnel reconstruction and loss decomposition

The funnel is a chain of conditional probabilities, not a single ratio. Each stage has its own
denominator, and the first analytic decision is to state which denominator each rate uses.

```text
FORMULA [F-01]: q_s = N_s / N_0
WHERE: N_0 = eligible signups [workspaces]; N_s = workspaces reaching stage s by the cutoff
       [workspaces]; q_s [dimensionless, 0..1] is the unconditional stage rate.
CONSTRAINT: the stage state is evaluated at the workspace's own 14-day cutoff, using event_time,
            on a matured cohort. A stage that occurs after the cutoff does not count.
NUMERICAL EXAMPLE: N_0=20,000; N_provisioned=17,240; q=0.862.
INTERPRETATION: 862 of every 1,000 eligible signups produced a usable workspace in the window.
```

```text
FORMULA [F-02]: r_s = N_s / N_{s-1}
WHERE: r_s is the conditional conversion from the preceding stage; N_{s-1} its denominator.
CONSTRAINT: same cutoff and same population as F-01; stages must be causally ordered.
NUMERICAL EXAMPLE: r_source = 10,688/17,240 = 0.619954; r_activate = 4,640/10,688 = 0.434132.
INTERPRETATION: conditional rates localise the loss; unconditional rates size the prize.
```

```text
FORMULA [F-03]: q_activate = q_provisioned * r_source * r_activate
WHERE: each factor is as defined above; the product is the end-to-end activation rate.
CONSTRAINT: exact only when the stage definitions are nested (a workspace cannot activate
            without connecting a source). Violated if a workspace can activate through an
            import path that bypasses the source-connection step.
NUMERICAL EXAMPLE: 0.862 * 0.619954 * 0.434132 = 0.232000.
INTERPRETATION: the identity lets an analyst check that the reported end-to-end number is the
                product of the reported stage numbers, which catches inconsistent denominators.
```

The constructed cohort in table form:

| Stage | Workspaces | Unconditional rate | Conditional rate | Lost at this stage | Share of total loss |
|---|---:|---:|---:|---:|---:|
| Eligible signup (`N_0`) | 20,000 | 100.0% | — | — | — |
| Workspace provisioned ≤14d | 17,240 | 86.2% | 86.2% | 2,760 | 18.0% |
| Source connected ≤14d | 10,688 | 53.4% | 62.0% | 6,552 | 42.7% |
| Activated (shared artifact, ≥2 collaborators) ≤14d | 4,640 | 23.2% | 43.4% | 6,048 | 39.4% |

**Independent checks.** The losses sum to 2,760 + 6,552 + 6,048 = 15,360, and 20,000 − 4,640 =
15,360, so no workspace is double-counted. The loss shares sum to 0.1797 + 0.4266 + 0.3938 =
1.0001, with the 0.0001 from rounding; the exact fractions are 2,760/15,360 = 0.1796875,
6,552/15,360 = 0.4265625, and 6,048/15,360 = 0.39375, summing to exactly 1. The conditional
rates multiply back to the headline rate, which is the check that the table has consistent
denominators.

**What the decomposition does and does not say.** The largest single loss is at the
source-connection stage (42.7% of all lost workspaces), which is why the intervention targets
setup guidance. That is a prioritisation inference from a decomposition, not a causal finding: the
workspaces lost at that stage may be systematically harder (for example, no data to connect),
and moving them through a guided flow may not move them to activation. The decomposition ranks
where to experiment; it does not estimate what an experiment will find.

**A tempting wrong aggregation.** Averaging the three conditional rates gives
(0.862 + 0.620 + 0.434)/3 = 0.639, which is not the activation rate and has no decision meaning.
Averaging stage rates mixes denominators; the only defensible aggregation is the product (F-03)
or a stage-specific rate on a stated population. This is a common dashboard defect: "funnel
completion" computed as a mean of stage percentages moves when the number of stages changes,
even with identical behaviour.

### 6b. Late-arrival audit: the measurement error that can swamp the effect

The activation window is 14 days. Events do not arrive in real time: mobile clients batch, offline
clients retry, and the pipeline occasionally replays a partition. If the metric is computed before
late events settle, a cohort's activation rate is understated. The size of that understatement is
a first-class input to the decision, because it must be smaller than the effect the pilot aims to
detect.

```text
FORMULA [L-01]: N_true = N_observed / (1 - f_late)
WHERE: N_observed = workspaces with the stage event inside the window as recorded at query time;
       f_late = fraction of in-window events that had not arrived by that time; N_true is the
       estimated settled count.
CONSTRAINT: f_late must be estimated from the same event type, client mix, and pipeline version;
            the correction assumes late events are missing at random with respect to arm, which
            fails if one arm's exposure changes client behaviour.
NUMERICAL EXAMPLE: N_observed = 10,688 source connections; f_late = 0.06;
                   N_true = 10,688 / 0.94 = 11,370.2 workspaces.
INTERPRETATION: the settled source-connection rate is 11,370.2/20,000 = 56.85%, not 53.44%.
```

The understatement is 11,370.2 − 10,688 = 682.2 workspaces, or 3.41 percentage points of the
signup denominator. Compare that to the 2.0-point effect the pilot is designed to detect: an
unsettled cohort produces an error larger than the signal. Two consequences follow.

First, the maturity lag must be validated per event type, not assumed. The case uses a 3-day lag
and reports the late-arrival curve; if the 99th percentile of arrival delay for
`artifact_shared` exceeds 3 days, the lag is increased before the pilot, and every reported number
is restated. Second, the arm comparison must use the same query time and the same lag. A common
failure is to report the treatment arm from a fresh dashboard and the control arm from a
pre-launch snapshot; the difference then contains the pipeline's settling behaviour.

```text
FORMULA [L-02]: window_sensitivity = q(21d) - q(14d)
WHERE: q(w) is the activation rate with window w, same cohort, mature.
CONSTRAINT: the later window includes workspaces whose activation may be caused by the
            intervention's later prompts, so it is not a neutral extension.
NUMERICAL EXAMPLE: q(7d)=3,520/20,000=0.176; q(14d)=4,640/20,000=0.232;
                   q(21d)=5,120/20,000=0.256.
                   window_sensitivity(14->21) = 0.024.
INTERPRETATION: 2.4 points of activation arrive in days 15–21, so a 14-day window is a
                deliberate truncation, not a complete measure of eventual activation.
```

The three windows show a monotone curve (17.6%, 23.2%, 25.6%), which is the expected shape and a
useful sanity check: a non-monotone curve indicates a definitional bug or a cohort that is not
matured. It also shows why "activation rate" without a window is not a metric. The owner must
choose the window from the decision's clock, not from the data's convenience: 14 days is chosen
because the intervention's prompts stop at day 14, so a 21-day measure would credit prompts with
behaviour they did not influence.

### 6c. State-order and duplicate checks

Two data-quality defects change the number without changing behaviour.

```text
FORMULA [Q-01]: q_clean = (N_activate - N_violation) / N_0
WHERE: N_violation = workspaces whose activation event precedes provisioning, or which carry a
       single collaborator, or which are duplicated on event_id.
CONSTRAINT: violations are removed, never repaired by imputation; the removal count is reported.
NUMERICAL EXAMPLE: N_activate=4,640; violation rate 1.4% -> 64.96, rounded 65;
                   q_clean = (4,640-65)/20,000 = 4,575/20,000 = 0.22875.
INTERPRETATION: the defect inflates the headline rate by 0.325 points; small here, but it is a
                systematic, arm-correlated error if the treatment arm emits events differently.
```

```text
FORMULA [Q-02]: N_true_events = N_observed_events / (1 - f_drop)
WHERE: f_drop = fraction of activation events silently dropped by an upstream dedupe key
       collision; used to bound, not to correct, the metric.
CONSTRAINT: assumes drops are independent of the workspace's arm and behaviour; a collision
            caused by a client retry pattern violates this.
NUMERICAL EXAMPLE: f_drop = 0.08; N_true = 4,640/0.92 = 5,043.5; rate = 0.2522.
INTERPRETATION: a dedupe bug of this size would overstate the effect of any intervention that
                changed client retry behaviour, which is exactly what a guided flow can do.
```

The important lesson is arm-asymmetry. A 1.4% violation rate is tolerable in a descriptive
dashboard. In a controlled comparison it is tolerable only if it is equal across arms; the
treatment arm is more likely to emit extra events (prompts, retries, telemetry), so the check must
be run per arm and the difference reported. A defect that is 1.4% in control and 2.6% in treatment
would create a 0.3-point artifact in the opposite direction to the expected effect.

### 6d. Economic ledger per 10,000 treated workspaces

The decision is not "is the lift positive" but "does the lift pay for the program". The ledger
converts an activation-rate difference into contribution margin through three declared
assumptions: the conversion difference between activated and non-activated workspaces (0.29), the
gross margin per converted account (4,212 USD per year at 78% of a 5,400 USD contract), and the
acquisition cost already sunk at signup (1,450 USD), which matters because a converted activation
does not repay CAC twice.

```text
FORMULA [E-01]: G = L * N * delta_conv * (ACV * m - CAC)
WHERE: L = activation-rate lift [dimensionless];
       N = treated workspaces [workspaces];
       delta_conv = conversion-rate difference, activated minus non-activated [dimensionless];
       ACV = annual contract value [USD/account/year];
       m = gross margin rate [dimensionless];
       CAC = acquisition cost per account [USD/account];
       G = incremental 12-month contribution [USD].
CONSTRAINT: delta_conv is a planning assumption, not a measured causal effect; it prices
            conversion that has not yet been observed at the pilot horizon. The formula assumes
            lift converts at the same delta as the population average, which fails if guided
            workspaces are marginal (lower-propensity) converts.
NUMERICAL EXAMPLE: L=0.02, N=10,000, delta_conv=0.29, ACV=5,400, m=0.78, CAC=1,450;
                   ACV*m = 4,212; 4,212-1,450 = 2,762;
                   G = 0.02*10,000*0.29*2,762 = 58*2,762 = 160,196 USD.
INTERPRETATION: a 2-point lift on 10,000 treated workspaces is worth 160,196 USD of 12-month
                contribution under the stated conversion assumption.
```

```text
FORMULA [E-02]: P = F/K + c_s*N + c_csm*N/10,000
WHERE: F = total build and maintenance budget [USD]; K = number of cohorts the build serves;
       c_s = serving cost per treated workspace [USD/workspace]; c_csm = customer-success
       program cost per 10,000 treated workspaces [USD per 10,000].
CONSTRAINT: fixed cost is amortised over the cohorts actually served; if the intervention is
            retired after one cohort, K=1 and the per-cohort cost triples.
NUMERICAL EXAMPLE: F=90,000; K=8; c_s=0.04; c_csm=7,000; N=10,000;
                   P = 11,250 + 0.04*10,000 + 7,000 = 11,250 + 400 + 7,000 = 18,650 USD.
INTERPRETATION: the program must return more than 18,650 USD per 10,000 treated workspaces.
```

```text
FORMULA [E-03]: L_be = P / (N * delta_conv * (ACV*m - CAC))
WHERE: L_be is the break-even activation lift; other terms as in E-01 and E-02.
CONSTRAINT: same assumptions as E-01; a lift below L_be can still be worth taking if it
            compounds into retention not priced here, which the ledger does not claim.
NUMERICAL EXAMPLE: denominator = 10,000*0.29*2,762 = 8,009,800;
                   L_be = 18,650 / 8,009,800 = 0.002328 = 0.233 percentage points.
INTERPRETATION: the pilot must prove a lift comfortably above 0.23 points, not merely above zero.
```

The ledger across a range of lifts, holding the assumptions fixed:

| Lift (points) | Incremental accounts | 12-month contribution (USD) | Program cost (USD) | Net (USD) | Decision under the stated rule |
|---:|---:|---:|---:|---:|---|
| 0.2 | 5.80 | 16,020 | 18,650 | −2,630 | Do not ship |
| 0.5 | 14.50 | 40,049 | 18,650 | +21,399 | Ship only if the interval excludes zero |
| 1.0 | 29.00 | 80,098 | 18,650 | +61,448 | Ship |
| 1.5 | 43.50 | 120,147 | 18,650 | +101,497 | Ship |
| 2.0 | 58.00 | 160,196 | 18,650 | +141,546 | Ship |
| 3.0 | 87.00 | 240,294 | 18,650 | +221,644 | Ship; check guardrails before scaling |

**Independent checks.** At 2.0 points: 0.02 × 10,000 = 200 extra activations; 200 × 0.29 = 58
accounts; 58 × 2,762 = 160,196. Break-even: 18,650/8,009,800 = 0.0023283, and 0.0023283 × 10,000 =
23.28 workspaces, which at 0.29 conversion is 6.75 accounts, whose contribution is 6.75 × 2,762 =
18,644, matching the program cost within rounding. The units are consistent: points × workspaces ×
(dimensionless conversion) × (USD per account) = USD.

**What the ledger does not establish.** The conversion difference 0.29 is the assumption that does
the most work and has the least support here. If guided workspaces are marginal — they reach
activation only because they were pushed — their conversion may be closer to the non-activated
rate. At delta_conv = 0.10, contribution per point falls to 27,620 USD and break-even rises to
0.675 points; a 2-point lift still nets 36,590 USD, but a 0.5-point lift would not cover the
program. The case therefore treats delta_conv as the first sensitivity, and the decision rule
requires the interval to clear the break-even computed at the *conservative* delta, not the
planning delta.

Time also matters. The 12-month contribution ignores the year-2 renewal, which is where most of
the value sits for an annual contract. Including renewal would raise the payoff, but only under a
retention assumption the pilot cannot test; the ledger deliberately stops at 12 months so that it
claims only what one measured conversion step could support.

### 6e. Precision, power, and a cluster counterexample

Before running anything, the owner needs to know whether the pilot can resolve the effect it
cares about.

```text
FORMULA [S-01]: n_arm = (z_{1-alpha/2} + z_{1-beta})^2 * [p_1(1-p_1) + p_2(1-p_2)] / (p_2-p_1)^2
WHERE: p_1, p_2 = control and treatment activation probabilities [dimensionless];
       alpha = two-sided type-I rate; 1-beta = power.
CONSTRAINT: independent Bernoulli assignment at the workspace level, one observation per
            workspace, no interference between workspaces, no clustering in the analysis.
NUMERICAL EXAMPLE: p_1=0.232, p_2=0.252, alpha=0.05 (z=1.959964), power=0.80
                   (z=0.841621); numerator coefficient 2.8016^2=7.8490;
                   sum of variances 0.232*0.768 + 0.252*0.748 = 0.178176+0.188496=0.366672;
                   n_arm = 7.8490*0.366672/0.0004 = 7,194.9 -> 7,195 per arm.
INTERPRETATION: about 7,195 workspaces per arm are needed to detect a 2-point lift with 80% power.
```

```text
FORMULA [S-02]: se_diff = sqrt(p_1(1-p_1)/n_1 + p_2(1-p_2)/n_2)
FORMULA [S-03]: MDE = 2.8016 * se_diff  (80% power, alpha=0.05 two-sided)
WHERE: n_1, n_2 = arm sizes; se_diff [dimensionless]; MDE [dimensionless].
CONSTRAINT: independent observations; a design effect must be applied when assignment is
            clustered (see the counterexample).
NUMERICAL EXAMPLE: at 10,000 per arm, se_diff = sqrt(0.178176/10,000 + 0.188496/10,000)
                   = sqrt(0.0000366672) = 0.0060553; MDE = 2.8016*0.0060553 = 0.016966.
                   Observed difference 0.252-0.232 = 0.020;
                   95% CI = 0.020 +/- 1.959964*0.0060553 = [0.00813, 0.03187].
INTERPRETATION: at 10,000 per arm the design can detect 1.70 points, so a 2-point effect would
                be resolved; the interval excludes zero and excludes the 0.233-point break-even.
```

The enrolment arithmetic: at 20,000 signups per 91-day cohort, the arrival rate is
20,000/91 = 219.78 signups/day. Splitting 50/50, filling 10,000 per arm needs
10,000/(219.78/2) = 91.0 days. The pilot therefore consumes a full cohort, which is why the build
must be ready before the cohort starts; a mid-cohort launch splits the window and biases the
measurement unless the assignment service randomises from the first signup.

**Cluster counterexample.** Suppose the platform cannot randomise per workspace and instead
rolls out by signup week, so the analysis has 13 weekly clusters of about 1,538 workspaces each.
Treating those workspaces as independent is a design error; with an intra-cluster correlation of
0.002 the design effect is:

```text
FORMULA [S-04]: deff = 1 + (m_bar - 1) * ICC
WHERE: m_bar = average cluster size [workspaces]; ICC = intra-cluster correlation [dimensionless].
CONSTRAINT: ICC must be estimated from comparable clustered data; using a guessed ICC is a
            sensitivity, not a power guarantee.
NUMERICAL EXAMPLE: m_bar=1,538; ICC=0.002; deff = 1 + 1,537*0.002 = 4.074.
                   Inflated se_diff = 0.0059695*sqrt(4.074) = 0.012049;
                   MDE = 2.8016*0.012049 = 0.033756; CI on a 2-point effect =
                   0.020 +/- 1.96*0.012049 = [-0.00362, 0.04362].
INTERPRETATION: under week-level exposure the same 2-point effect is no longer distinguishable
                from zero, and the decision rule returns "pilot again", not "ship".
```

This is the most instructive reversal in the case. The intervention may be identical, the true
effect identical, and the cohort identical; only the assignment unit changed, and the decision
changed from ship to inconclusive. The correct response is to randomise at the workspace level
(the unit at which the outcome is measured) whenever the product allows it, and to accept a
longer enrolment if the platform forces clustering — not to pretend the clusters are independent.

A second precision trap: the analysis must not stop early and report at the first significant
look. The rule is fixed in advance: one primary analysis at maturity, with the decision rule of
Section 5c. Interleaved looks with an unchanged alpha inflate the false-positive rate; if the
owner needs interim monitoring, a sequential procedure with a declared alpha-spending function is
required, and its cost is a wider interval at the final look.

### 6f. Guardrail arithmetic and support capacity

The support guardrail is a hard constraint, not a weighted term: a lift that breaches it is
disqualified regardless of contribution. The baseline contact rate is 6.1% of signups; the cap is
a 1.5-point increase.

```text
FORMULA [G-01]: C_arm = N_arm * c_arm
WHERE: c_arm = support contacts per eligible workspace in the arm [contacts/workspace];
       C_arm [contacts].
NUMERICAL EXAMPLE: baseline c=0.061, N=10,000 -> 610 contacts; cap c=0.076 -> 760 contacts.
                   Headroom = 150 contacts over the cohort.
INTERPRETATION: the intervention may add at most 150 tickets per 10,000 treated workspaces.
```

The planned intervention is expected to add about 0.6 points of contact rate (60 tickets per
10,000), leaving 90 contacts of headroom. The headroom is thin enough that it must be monitored
daily, not at the end: a guided flow that confuses users can add tickets faster than it adds
activations, and the two effects are visible at different lags.

```text
FORMULA [G-02]: D_cap = n_agents * h_productive * r_resolve
WHERE: n_agents [agents]; h_productive [hours/agent/day]; r_resolve [tickets/agent-hour];
       D_cap [tickets/day].
CONSTRAINT: assumes the extra tickets arrive evenly and need no specialist escalation.
NUMERICAL EXAMPLE: n=1.2, h=7.5, r=11 -> D_cap = 99 tickets/day.
                   60 extra tickets over 91 days = 0.659 tickets/day.
INTERPRETATION: capacity is not the binding constraint for this volume; the guardrail rate is.
```

The asymmetry is worth stating explicitly: capacity can absorb the tickets, but the guardrail can
still be breached in a bad week. Both checks are needed because they fail differently. Capacity
failure degrades response time, which shows up as a rising backlog; guardrail failure is a
customer-harm signal, which shows up as a rate crossing a pre-declared line even when absolute
volume is small.

### 6g. Decision trace under changed constraints

The decision rule is mechanical, which is what makes it auditable:

1. Compute the activation rate per arm at maturity, with the same query time, the same 3-day
   settling lag, and the same exclusion list.
2. Compute the difference and its 95% interval under the pre-declared analysis (workspace-level,
   independent).
3. Check the guardrail. If the contact-rate difference exceeds +1.5 points, stop: do not ship,
   regardless of lift.
4. Compare the lower interval bound with the break-even lift computed at the conservative
   conversion assumption (0.675 points). If the lower bound exceeds it, ship. If the interval
   spans it, pilot again with a larger sample or a narrower intervention. If the upper bound is
   below it, do not ship.
5. Record the four inputs (rates, interval, guardrail, break-even) in the decision log with their
   versions.

Applied to the constructed numbers at 10,000 per arm: rates 0.232 and 0.252, difference 0.020,
CI [0.00813, 0.03187], guardrail difference +0.6 points (passes, 90 contacts of headroom), and
the conservative break-even 0.675 points. The lower bound 0.813 points exceeds 0.675, so the rule
returns **ship**. Under the clustered design (CI [−0.00362, 0.04362]) the same rule returns
**pilot again**, and the reason recorded is precision, not effect size.

Changing a constraint changes the method, not just the number:

| Changed constraint | Effect on the analysis | Decision consequence |
|---|---|---|
| Event settlement lag is 7 days, not 3 | Cohort maturity extended; pilot duration +4 days | Enrolment window shortens; power unchanged if arms stay balanced |
| Platform forces week-level exposure | ICC-based design effect applies; MDE ≈ 3.4 points | 2-point effect not resolvable; change assignment unit or lengthen pilot |
| Guardrail cap tightened to +0.5 points | Expected contacts already at the cap | Intervention must reduce contacts elsewhere, or be redesigned |
| Conversion difference known to be 0.10 | Break-even rises to 0.675 points | Small lifts no longer pay; decision becomes more sensitive to precision |
| Billing model moves to usage-based | ACV and margin are no longer constants | Ledger must be rebuilt per usage band; this case's E-01 does not apply |

## 7. The Finding

The constructed analysis yields five findings, each with its evidence status attached.

1. **The funnel's largest loss is source connection (42.7% of lost workspaces), not activation
   itself (39.4%).** This is a descriptive decomposition of an illustrative cohort. It justifies
   targeting setup guidance, and it does not establish that guidance recovers those workspaces.
2. **The measurement error from late events (3.41 points) exceeds the effect the pilot is
   designed to detect (2.0 points).** This is arithmetic on a constructed late-arrival rate. The
   methodological consequence is real: the maturity lag is a prerequisite of the experiment, not
   a reporting detail. A pilot run on an unsettled cohort cannot distinguish a true 2-point lift
   from settling behaviour.
3. **The break-even lift is 0.233 points at the planning conversion assumption and 0.675 points at
   the conservative one.** Under the conservative assumption the pilot must clear 0.675 points,
   which requires roughly 24,000 workspaces per arm at 80% power — more than one cohort provides
   at the assumed arrival rate, so the conservative case needs either a longer enrolment or a
   larger expected effect.
4. **Precision, not the point estimate, drives the decision.** The identical 2-point effect is
   decisive under workspace-level assignment (CI [0.008, 0.032]) and inconclusive under weekly
   clustering (CI [−0.004, 0.044]). The design choice, not the intervention, changes the answer.
5. **The support guardrail is satisfied at +0.6 points with 90 contacts of headroom, and it is a
   disqualifying constraint rather than a tradeable cost.** A larger lift obtained by a
   confusing flow would be rejected.

**What the finding does not say.** It does not say the intervention works, that any product's
activation rate is 23.2%, that 29% is the true conversion gap, or that 90,000 USD is the right
build budget. It says: given these constructed inputs, this is the arithmetic the decision
requires, and these are the four quantities that could reverse it.

**Why the obvious framing misses.** "Did the checklist completion rate improve?" would answer
yes under several interventions that add no collaborative value, including making the checklist a
single click. The decision needs a state that is costly to fake and observable at a fixed cutoff:
a document shared with a second person.

## 8. System/Solution Design

The system has two jobs: deliver the intervention consistently, and produce an auditable
measurement of it. Both are designed before launch, because a treatment that cannot be measured
is not a pilot.

### 8a. Offline / batch path

```text
raw events (event_time, receive_time, event_id, workspace_id, payload)
  -> dedupe on event_id, quarantine impossible sequences
  -> workspace state timeline (stage transitions with timestamps)
  -> cutoff evaluation at signup_date + 14 days  [mature only]
  -> activation flag + stage flags + violation flags
  -> cohort table joined to assignment records
  -> per-arm rates, difference, interval, guardrail rate
  -> decision record (metric version, query time, exclusions, result)
```

Contract details for the offline path:

| Element | Rule | Why it matters |
|---|---|---|
| Cutoff | `signup_created.event_time + 14 days`, per workspace | A single report-level cutoff mixes cohort ages |
| Maturity gate | Query refuses to run unless `now - cohort_end >= 3 days` | Prevents unsettled cohorts from being reported |
| Event ordering | Stage transitions must be causally ordered; violations counted, not repaired | A violation is evidence about the producer |
| Identity | No user-level roll-up; workspace is the unit | User-level joins cross accounts and break the denominator |
| Exclusions | Listed explicitly, counted, and re-tested without the deletion exclusion | Silent exclusions change the population |
| Versioning | Metric SQL is versioned; each published number carries its version | Restatement requires knowing what changed |
| Idempotence | Re-running the pipeline on the same snapshot yields identical output | Otherwise "the number moved" is uninterpretable |

The batch path deliberately has no model. Its correctness property is reproducibility: same
snapshot, same version, same number. That property is what makes the decision record auditable
months later, when the events have aged out of retention and only aggregates remain.

### 8b. Online / serving path

```text
signup_created (provisioning webhook)
  -> eligibility check (edition, region, internal-tenant exclusion)
  -> deterministic assignment: bucket = H(workspace_id, salt) mod 100
  -> persist assignment record {workspace_id, arm, salt_version, assigned_at}
  -> feature flag lookup on every setup screen
  -> prompt events emitted with prompt_version and assignment_id
  -> support tickets tagged with assignment_id where the join exists
```

Design rules that prevent the classic failures:

- **Assignment at provisioning, not at first login.** Assignment at login conditions on a
  post-treatment behaviour and removes workspaces that never log in from both arms.
- **Persisted assignment, not recomputed.** Recomputing the hash on each request is fine only if
  the salt never changes; a salt rotation during the pilot would move workspaces between arms.
  The record is written once and read thereafter.
- **Salt version in the record.** It makes a rotation detectable and lets the analysis exclude or
  stratify affected workspaces.
- **No mid-window reversal.** A workspace keeps its arm for the 14-day window even if it is
  deleted and recreated with the same account, which is why the assignment key includes the
  original provisioning timestamp.
- **Prompt events carry the intervention version.** Without it, a later prompt change cannot be
  separated from the original effect.

### 8c. Key design decisions

| Decision | Chosen | Rejected alternative | Reason |
|---|---|---|---|
| Unit of randomisation | Workspace | User | Outcome is measured per workspace; user assignment splits a shared workspace between arms |
| Assignment time | Provisioning | First login | Avoids post-treatment selection |
| Metric grain | Workspace, 14-day cutoff | Session or event | Sessions have no commercial meaning; events are gameable |
| Outcome | Shared artifact with ≥2 collaborators | Checklist completion | Completion is a click, not value |
| Maturity lag | 3 days, validated | Immediate reporting | Late arrivals exceed the effect size |
| Analysis | One pre-declared look | Continuous monitoring | Alpha inflation |
| Guardrail | Contact-rate cap, hard | Weighted cost term | Customer harm is not tradeable against margin |
| Serving | Feature flag with assignment record | Static release to all | Rolling back requires an identifiable cohort |
| Model | None for the decision | Propensity ranking | Adds an unvalidated object without changing the action |

### 8d. Architecture artifacts and operational readiness

| Artifact | Content | Ready when |
|---|---|---|
| Metric contract | Numerator, denominator, units, window, cutoff rule, exclusion list, version | Reviewed by analytics and the growth owner |
| Assignment service spec | Hash function, salt, record schema, key uniqueness | Tested on a staging cohort with duplicate detection |
| Event schema diff | New prompt events with version fields | Producer and warehouse agree on payload |
| Guardrail monitor | Daily contact-rate difference with alert threshold | Alerts fire on a synthetic breach in staging |
| Rollback plan | Flag off, assignment record retained, analysis still runs | Rehearsed once before launch |
| Decision log template | Four inputs plus versions | Filled at the single analysis |

Operational readiness is not "the code is deployed". It is: the flag can be turned off without
losing the cohort record; the metric can be recomputed from a snapshot; a breach of the guardrail
has an owner and a response time; and the decision log can be completed from artifacts that exist.

**Failure and recovery paths for the serving design.** If the assignment service is unavailable at
provisioning, the request must not invent an arm: it falls back to "unassigned", the workspace is
excluded from the primary analysis and reported separately, and the enrolment window is extended
by the outage duration. If a stale flag serves the treatment to a control workspace, the exposure
is recorded as a protocol deviation, the workspace stays in its assigned arm for the intention-to-
treat analysis, and the deviation rate is reported. If the salt is rotated accidentally, the
analysis uses the persisted records and excludes workspaces whose persisted arm does not match
their recomputed bucket, with the count disclosed. In each case the principle is the same:
assignment is a recorded fact, exposure is an observed fact, and the analysis must be able to
distinguish them.

## 9. Evaluation & Validation

| Metric | Numerator | Denominator | Units | Population | Window |
|---|---|---|---|---|---|
| Activation rate | Workspaces with a shared artifact with ≥2 distinct collaborators | Eligible signups in the cohort | dimensionless | Self-serve, one edition | 14 days from signup, mature |
| Stage-2 rate | Workspaces with a source connected | Workspaces provisioned | dimensionless | Same | 14 days |
| Lift | Treated rate − control rate | — | percentage points | Randomised arms | Same cutoff and query time |
| Guardrail | Support contacts | Eligible workspaces in arm | contacts per workspace | Same | 14 days |
| Break-even lift | Program cost | Contribution per point | percentage points | Treated | 12 months |

**Comparator.** The concurrent randomised control is the primary comparator. The prior cohort is
retained only as a sanity check, because it differs in season, acquisition mix, and any
unmeasured product change. A pre/post comparison would be rejected here even though it is cheaper.

**Guardrails and uncertainty.** The guardrail is a hard constraint with a pre-declared threshold.
Uncertainty is reported as a 95% interval on the rate difference under the pre-declared
independent-workspace analysis; when clustering is unavoidable, the design effect is applied and
the interval widened, and the decision rule is not relaxed to compensate.

**Executed versus not executed.** Executed for this document: the arithmetic of Sections 6a–6f,
recomputed with an independent calculator and checked for unit consistency and algebraic
identities. Not executed: any experiment, any measurement of a real cohort, any calibration of the
0.29 conversion assumption, any test of the intervention's effect on support contacts. There is no
run artifact, and no result in this case should be cited as evidence about a real product.

**Validity threats, ranked.**

1. *Outcome definition drift.* If the platform changes what counts as a collaborator mid-cohort,
   the two arms can be measured under different definitions. Control: version the definition and
   recompute both arms under one version.
2. *Arm-correlated instrumentation.* The treatment arm emits more events, so any dedupe or
   ordering defect is arm-asymmetric. Control: per-arm violation rates, reported.
3. *Interference through shared accounts.* One account with two workspaces could have both arms;
   if collaboration crosses workspaces, the control is contaminated. Control: check account-level
   arm balance and, if necessary, randomise at the account.
4. *Conversion-assumption error.* The ledger's weakest input. Control: report the decision at both
   conversion assumptions and require the interval to clear the conservative one.
5. *Seasonality interacting with enrolment.* A full-cohort enrolment spans a quarter, so any
   within-quarter trend affects both arms only if assignment is balanced over time. Control: check
   arm balance by signup week, and stratify if it drifts.

**Validation protocol for a future pilot.** Randomise at provisioning; enrol for a full cohort;
freeze the metric version before the first signup; compute one analysis at maturity; publish the
four decision inputs with the interval; run the guardrail check daily; retain the assignment
record for the retention period so the analysis can be reproduced; and record the protocol
deviations with counts. If the pilot's interval spans the conservative break-even, the next step
is a larger or narrower test, not a decision on the point estimate.

