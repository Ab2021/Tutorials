---
case_id: "CASE-01-03-001"
case_title: "HCP promotional-response drivers for a fictional launch"
case_version: "2.0"
created: "2026-11-07"
last_reviewed: "2026-11-10"
status: "proposed"
content_status: "draft"
case_type: "analytical"
domain: "Pharma commercial analytics"
primary_pattern: "causal-confounding"
secondary_patterns: ["evaluation-mismatch", "label-delay"]
evidence_level: "not-measured"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "Commercial decision owner"
source_index: "INDEX.md"
review_trigger: "A proposed launch, access rule, data source, consent rule, or study protocol changes"
---

# HCP promotional-response drivers for a fictional launch

Evidence statement. Nothing in this file was measured on a real product, market, or population. The
source directory for this topic supplied a title only; no project data, method document, or result
was available. Every count, proportion, cost, and interval below is arithmetic performed on
constructed inputs and is labelled ILLUSTRATIVE at the point of use. The statistical machinery —
two-proportion inference, design effects, E-values, Kaplan-type maturity gates — is standard and is
verified by hand-recomputation in §6. The decision narrative describes a simulated decision record,
not a company action. Where a real-world quantity is unknown, the text says unknown rather than
supplying a plausible number.

## 1. The One-Line Framing

Given a weekly roster of eligible healthcare professionals (HCPs) and only the information available
before an assignment is frozen, decide whether to route an incremental approved educational contact
to a defined subset, or to continue the existing manual schedule unchanged.
The decision is channel allocation under a fixed capacity; it is never a treatment choice for a
patient and never a recommendation about which medicine a clinician should prescribe.
The accountable owner is the fictional commercial decision owner, who may approve, reject, or defer.
The unit of the decision is one HCP at one eligible weekly assignment, not a territory and not a
patient.
The outcome is a binary analytical proxy: at least one qualifying prescribing event during a fixed
56-day window after assignment, read only when the source is mature.
The baseline is the manual schedule that field staff already run; it is a policy, not an untreated
universe, so the estimand is a policy contrast rather than the effect of exposure versus nothing.
The material threat is that contact was historically aimed at HCPs already expected to prescribe, so
a descriptive driver is a correlate of targeting until a design removes that selection.
The secondary threat is time: claims-style sources settle late, so a 56-day window may be only
partly observable when the decision is due.
The decision is bounded: one launch, one market, one 8-week test horizon, no change to content or to
clinical guidance.
The success condition is a prespecified incremental response with a lower confidence bound that
still clears a commercial minimum, while access, consent, and independence guardrails hold.
The failure condition is any guardrail breach, or a result whose interval cannot separate the
proposed policy from the baseline.
The honest default when evidence is thin is deferral with a documented reason, not a quiet rollout.

## 2. Context & Background

### 2.1 The organisation as constructed

A fictional launch team distributes approved educational material to HCPs on a weekly cycle.
Field staff hold territories, each with a capacity of visits; capacity is fixed for the quarter.
A field operations lead owns the schedule and records what was delivered, separately from what was
assigned.
An analytics lead owns the estimand, the analysis plan, and the provenance of every input.
A medical, legal, regulatory, and privacy review group must clear content, eligibility, study
design, and data flows before any incremental contact is attempted.
A privacy lead owns access decisions and any approved linkage between identifiers.
Those role names describe review responsibilities in the scenario; they do not assert that any
activity is lawful in any jurisdiction, and no legal conclusion is drawn here.
The HCP remains the decision maker for patient care; the team supplies information only.
No patient is contacted, targeted, segmented, or identified individually by this case.

### 2.2 The baseline process

The baseline is a weekly judgment-driven schedule: staff pick HCPs from their territory list using
local knowledge, recent interactions, and a fixed visit capacity.
The baseline is operationally legible, cheap to run, and preserves relationships that a model cannot
see, which is why it persists even when its yield is unmeasured.
Its yield has never been compared against an alternative in a controlled way in this scenario.
Whether the baseline is effective, equitable, or compliant is unknown and needs separate review.
Both the proposed policy and the baseline sit on top of the same approved content and the same
independence rules; the difference is whom the offer reaches, not what it says.

### 2.3 Why the topic is hard

Three features make the analysis adversarial before any modelling begins.
First, assignment is not random: representatives choose whom to see, and they choose partly on the
same signals the outcome reflects.
Second, the outcome is delayed and revisable: prescribing events arrive through a settled feed whose
maturity lag is measured in weeks, so early reads are biased toward fast-reporting channels.
Third, the population is small and clustered: HCPs share practices and staff, so a contact to one
HCP can move a colleague's prescribing and contaminate a naive comparison.
A fourth, quieter difficulty is definitional: "response" is an analytical proxy whose plausibility
depends on whether the source can distinguish eligible events from unrelated dispensing.

### 2.4 What the source material did and did not provide

The only source lead is a directory title naming key-driver analysis for a launch brand.
That title is a topic pointer, not evidence that a project existed, that it used any particular
method, or that it produced any result.
Accordingly this case is written as a teaching construction with an explicit method ledger, so a
reader can transplant the reasoning without inheriting a false factual claim.

## 3. The Business Question

### 3.1 The decision in operational form

Illustrative question: for the next eight weekly cycles, should the owner allocate the incremental
contact capacity to a model-selected subset of eligible HCPs, or leave the manual schedule intact?
The question is asked once per launch quarter, before the first cycle of that quarter.
The answer must be expressible as a rule that operations can execute: a ranked list, a threshold, a
capacity constraint, and an exclusion filter.
An answer that cannot be executed by the scheduling system is not a decision, it is an opinion.

### 3.2 Available actions

Four actions are genuinely available and must be compared, not just the two obvious ones.
No change: keep the manual schedule; cost zero incremental effort; information gain zero.
Descriptive audit: measure how the current schedule relates to prior prescribing; cheap; susceptible
to confounding; useful for finding leakage and data defects rather than for choosing a policy.
Reviewed randomised policy test: assign eligible HCPs to incremental-offer versus usual-schedule at
a frozen timestamp; expensive; yields a defensible policy contrast.
Deferral with a stated trigger: no change now, revisit when a named condition holds, such as a
mature data source, approved protocol, or achieved capacity.
Deferral is a real alternative and must be recorded, because "we did nothing" and "we decided to
wait for a specific precondition" are different governance states.

### 3.3 The objective and its proxy

The objective is the mean difference in HCP-level binary response over a prespecified 56-day window.
Response is one if at least one qualifying event occurs from day 1 through day 56 after assignment,
once the source is mature, and zero otherwise.
Day-0 events are excluded so that the ordering of assignment and outcome is unambiguous.
The proxy is accepted only if the source can separate eligible events using an approved definition;
if it cannot, the proxy is revised before randomisation rather than after seeing results.
The objective deliberately excludes revenue, because revenue mixes price, mix, and access changes
that the contact policy does not control.

### 3.4 What the decision does not authorise

It does not authorise content changes, clinical recommendations, reimbursement actions, or patient
targeting.
It does not authorise lifting an exclusion, contacting a declined HCP, or using an unapproved
channel.
It does not authorise publishing HCP-level results outside approved disclosure thresholds.
A high predicted response never overrides consent, eligibility, independence, or channel limits.

## 4. Assumptions & Constraints

### 4.1 Assumption register

| ID | Assumption or constraint | How it is verified | Owner | Consequence if false |
|---|---|---|---|---|
| A-01 | A versioned HCP roster can be joined without patient identifiers | Key-history audit on three historical snapshots | Data steward | Restrict to the linkable subpopulation and report the selection |
| A-02 | Outcome events mature within 12 weeks after the 56-day window closes | Re-run historical extracts at +4, +8, +12 weeks and count revisions | Outcome steward | Extend the maturity cutoff or analyse a shorter window |
| A-03 | Eligible HCPs can be assigned without altering clinical care | Review group reads the protocol and the contact script | Review group | Do not randomise; fall back to a descriptive audit |
| A-04 | Contact capacity is stable across arms | Compare planned versus delivered capacity per cycle | Operations | Re-stratify by capacity band or revise the intervention |
| A-05 | Interference between HCPs is limited enough for HCP-level assignment | Estimate practice-level intraclass correlation on prior data | Study lead | Randomise by practice and use cluster inference |
| A-06 | Prior-year prescribing is computable from the same settled feed | Reconcile two extracts 30 days apart | Data engineer | Drop prior-volume covariates or widen their window |
| C-01 | Contact requires current permission and an approved channel | Verify permission state immediately before each contact | Privacy and operations | Suppress the contact but keep the assignment for intention-to-treat |
| C-02 | Only reviewed content versions may be offered | Check the content registry version against the delivery record | Content owner | Stop delivery for that version and log the incident |
| C-03 | Patient-level identifiers are unavailable to commercial roles | Privacy lead reviews the architecture and the export list | Privacy lead | Block linkage and block the export |
| C-04 | Small cells cannot be released without review | Apply a minimum-cell rule with secondary suppression | Privacy lead | Suppress or combine cells under approved policy |
| C-05 | Clinical judgment stays independent of the analysis | Medical reviewer defines prohibited uses in writing | Medical reviewer | Stop the use and escalate to the review group |
| C-06 | HCP identity resolution is stable across the study window | Measure the duplicate and merge rate on the roster | Data steward | Freeze the key at assignment and analyse by frozen key only |

### 4.2 The assumptions that can reverse the recommendation

Three assumptions dominate. If the outcome source cannot mature inside the decision window (A-02),
no randomised read is usable this quarter and the decision collapses to no-change. If practice-level
interference is large (A-05), the effective sample shrinks and the planned precision is unattainable
at the stated cost. If permission or channel constraints (C-01) make receipt differ sharply by arm
in a way that changes behaviour, the intention-to-treat contrast remains valid but its interpretation
shifts from "effect of an offer" toward "effect of a policy that many ignore".
The remaining assumptions matter for accuracy and disclosure, not for the direction of the decision.

### 4.3 Constraints that bound the design

Capacity is fixed, so the test cannot simply contact more people; it must reallocate.
The measurement window is fixed by the launch calendar, so the test cannot wait for a longer outcome.
The roster cannot be enriched with patient-level data, so any patient-outcome claim is out of scope.
Review timelines consume calendar time that the test cannot recover, which is why the protocol is
written before the quarter starts.

## 5. Data, Tools & Method

### 5a. Data Specification

**Grain.** One row per HCP at one eligible weekly assignment, keyed by a pseudonymous HCP identifier
plus the assignment timestamp. The analysis dataset has exactly one row per HCP: the first eligible
assignment in the test window. Later assignments are dropped, not averaged, so that an HCP who is
contacted twice does not carry double weight.

**Cut-off timestamp.** Assignment is frozen at 06:00 local time on the Monday of each cycle. Every
covariate in the model must be computable from data with an event time strictly before that
timestamp. Any record whose event time is after the freeze but whose arrival time is before it is
excluded by construction, because arrival time is not evidence of availability at decision time.

**Population.** Eligible reachable HCPs in the single constructed market: valid de-duplicated key,
active launch-relevant role, approved market, current channel permission, no exclusion flag at
assignment, and no prior assignment during the test. The population excludes unavailable HCPs,
declined channels, expired permissions, unresolved duplicates, and HCPs already assigned.

**Keys and joins.** Assignment and outcome are linked one-to-one by the approved pseudonymous key.
Outcomes may contain many events per HCP; the analysis aggregates to the HCP-week-event grain first
and then collapses to the binary 56-day indicator. A separate mapping vault, if one is authorised,
is inaccessible to analyst and scheduler roles, and no reversible mapping is ever exported.

**Units.** Response is a binary indicator (0/1) at HCP level. Contact counts are integers per cycle.
Costs are currency units per contact. Proportions are dimensionless and reported to three decimals.

**Fields expected at decision time.** Identifier, territory, role, permission state, channel
eligibility, specialty, practice identifier, prior-12-month event counts by month, prior contact
counts by channel and month, days since last contact, and an access indicator if an approved source
exists.

**Fields explicitly not available at decision time.** The outcome itself; any event in the outcome
window; content engagement logged after the freeze; the representative's post-visit notes; any
patient-level attribute; and the roster's later corrections. This list is the leakage boundary and
is restated in §5d.

**Constructed size.** 2,000 distinct HCPs, 1,000 per arm, drawn from a constructed pool of 6,400
eligible HCPs across 250 practices. No such roster was observed.

### 5b. Tools & Stack

A restricted SQL warehouse holds immutable assignment snapshots and reconciled event aggregates.
The snapshot table is append-only: a correction creates a new version rather than mutating a row.
A version-controlled analysis script computes the prespecified contrasts; the language and library
versions are recorded in the study registry when the protocol is signed, not chosen after the read.
A scheduling system holds permission and delivery records and is the only system that may authorise
a contact.
A study registry holds the protocol, its version history, approvals, and the commit hash of the
analysis code that produced each table.
No named vendor, platform, runtime, or pipeline is asserted as implemented; the stack is a design
proposal, not an observed architecture.

### 5c. Method Summary

The descriptive audit fits a model of past response on pre-contact features, using explicit
missingness handling and no post-assignment variables, to find which drivers are plausibly
pre-treatment and which are artefacts of targeting.
The randomised proposal estimates an intention-to-treat effect of assignment to an incremental
approved offer versus usual scheduling, over a fixed 56-day window, read only at maturity.
Usual scheduling may itself contain contacts; the estimand is therefore a policy contrast between
two allocation rules, not the effect of zero versus one exposure.
Assignment is at HCP level only if review concludes that cross-HCP spillovers are acceptably small;
otherwise assignment moves to practice level and inference accounts for clustering.
A delivered-contact comparison is biased whenever availability or engagement affects both receipt
and response; it is reported only as a secondary, clearly labelled analysis.
An observational adjustment with pre-contact confounders may be considered when randomisation is
infeasible, but it requires consistency, conditional exchangeability, positivity, correct timing,
and an explicit target population — conditions that are hard to defend for rep-selected contact.
Inverse probability weighting cannot rescue HCPs with no realistic chance of either policy, because
positivity fails at the tails and the estimate is driven by a handful of extreme weights.
A purely descriptive model stays useful for workflow auditing and for detecting leakage, but it does
not direct causal allocation without a design that supports it.

### 5d. Provenance, Quality, Exclusions, Leakage

Each source must have a named owner, an approved access basis, and a recorded version.
For every extract, record the event time, the arrival time, the extraction time, the correction
policy, and the transformation revision identifier.
Reconcile assignment counts against the scheduling system and against the delivery log separately;
the difference between assigned and delivered is a first-class quantity, not an error.
Exclusions are applied in a fixed order and each step produces a count and a reason, so the
inclusion flow can be reconstructed by a reviewer without rerunning code.
Low-count cells are suppressed before any table leaves the restricted environment.
Missing outcome coverage is not zero: coverage is reported by arm, and a prespecified sensitivity
analysis re-weights or bounds the estimate if coverage differs materially.
Leakage controls: no feature may be computed from post-freeze records; the model's feature list is
frozen before the first read; and a deliberate negative control — a feature that should have no
effect, such as the week-of-year of the freeze — is inspected for a spuriously large coefficient.
Identity quality is checked by re-running the deduplication on two snapshots and comparing the
resulting keys; an unstable key invalidates the panel and is escalated before analysis.

### 5e. Reproducibility & Method-Selection Ledger

| Decision | Chosen | Rejected alternative | Reason for rejection | Reversal trigger |
|---|---|---|---|---|
| Identification strategy | Randomised assignment at HCP level | Propensity-matched observational contrast | Rep selection is driven by unmeasured judgement and relationship history | Review finds interference small and randomisation infeasible |
| Estimand | Intention-to-treat policy contrast | As-treated exposure effect | Receipt depends on availability and engagement, which correlate with the outcome | Receipt becomes near-deterministic by arm |
| Outcome window | 56 days from assignment | 12 weeks | Longer window mixes subsequent cycles and complicates attribution | Capacity changes only at cycle boundaries |
| Outcome maturity rule | Read only after 12-week settlement | Read at window close | Settlement lag would mix mature and immature events across arms | Feed settles within 2 weeks |
| Inference | Two-proportion difference with a cluster-robust check | Bayesian hierarchical model | No prior elicitation was approved for this decision | A documented prior becomes available |
| Multiplicity | Prespecified primary plus a fixed subgroup list | Exploratory scan | Scan results would be unactionable and would inflate false positives | Subgroups are preregistered with separate budgets |
| Missing outcome | Coverage report plus bounding sensitivity | Complete-case analysis | Complete-case assumes coverage is unrelated to response | Coverage exceeds 98% in both arms |

Every row is reversible only by a written protocol amendment, and each amendment must state which
table it invalidates.

## 6. Analysis Walkthrough

### 6.1 Setting up the comparison

The analysis compares the proportion of responders in the offer arm with the proportion in the
usual-schedule arm, using only HCPs frozen at their first eligible assignment.
The first quantity computed is not an effect but a balance table: arm means for prior 12-month
events, prior contact count, days since last contact, specialty mix, and access status.
Balance is judged on standardised differences, not on p-values, because p-values conflate imbalance
with sample size and reward large samples for trivial differences.
A standardised difference above 0.10 on a pre-specified covariate triggers a review of the
assignment mechanism rather than an adjustment after the fact.

### 6.2 Derivation 1: the intention-to-treat difference and its interval

```text
FORMULA
  delta_hat = p1 - p0
  SE(delta_hat) = sqrt( p1*(1-p1)/n1 + p0*(1-p0)/n0 )
  CI95 = delta_hat +/- 1.96 * SE(delta_hat)

WHERE
  p1  = responders in the offer arm / n1, dimensionless, 0..1
  p0  = responders in the usual-schedule arm / n0, dimensionless, 0..1
  n1, n0 = number of distinct HCPs assigned to each arm (integers)
  delta_hat = risk difference in percentage points when multiplied by 100

CONSTRAINT
  One assignment per HCP; one outcome per HCP; outcome window identical in both arms;
  independent HCPs (or clustering handled separately, see 6.4); every assigned HCP counted,
  including those never contacted, and no post-assignment exclusion.

NUMERICAL EXAMPLE   (ILLUSTRATIVE inputs)
  n1 = 1000, responders1 = 312  -> p1 = 0.312
  n0 = 1000, responders0 = 268  -> p0 = 0.268
  delta_hat = 0.312 - 0.268 = 0.044  (4.4 percentage points)
  variance term 1 = 0.312 * 0.688 / 1000 = 0.214656 / 1000 = 0.000214656
  variance term 0 = 0.268 * 0.732 / 1000 = 0.196176 / 1000 = 0.000196176
  sum             = 0.000410832
  SE              = sqrt(0.000410832) = 0.020269
  z               = 0.044 / 0.020269 = 2.171
  CI95            = 0.044 +/- 1.96 * 0.020269 = 0.044 +/- 0.039727
                  = [0.004273, 0.083727]  -> [0.43 pp, 8.37 pp]

INTERPRETATION
  The interval excludes zero, so the data are inconsistent with no difference at the 5% level.
  But the lower bound of 0.43 pp is economically indistinguishable from no effect given the
  cost of an incremental contact, so the decision cannot rest on "significant" alone; it must
  compare the lower bound against the commercial minimum effect. That comparison happens in 6.6.
```

### 6.3 Derivation 1 continued: the same arithmetic on a risk-ratio scale

The risk difference answers "how many more responders per thousand HCPs"; the risk ratio answers
"how many times more likely". Decision makers often switch scales without noticing that the two
imply different confounder sensitivity, which matters in 6.5.

```text
FORMULA
  RR_hat = p1 / p0
  log-scale SE = sqrt( 1/a - 1/n1 + 1/b - 1/n0 )   with a = responders1, b = responders0
  CI95(RR) = exp( ln(RR_hat) +/- 1.96 * SE_log )

WHERE
  a, b are event counts (integers); n1, n0 are arm sizes; RR_hat is dimensionless

CONSTRAINT
  Requires a,b > 0 and the same outcome definition in both arms. The log-scale interval is
  asymmetric on the ratio scale, which is a feature, not a defect.

NUMERICAL EXAMPLE
  RR_hat = 0.312 / 0.268 = 1.16418
  1/a - 1/n1 = 1/312 - 1/1000 = 0.0032051 - 0.0010000 = 0.0022051
  1/b - 1/n0 = 1/268 - 1/1000 = 0.0037313 - 0.0010000 = 0.0027313
  SE_log = sqrt(0.0049364) = 0.070259
  ln(RR_hat) = ln(1.16418) = 0.15204
  CI95(RR) = exp(0.15204 +/- 1.96*0.070259) = exp(0.15204 +/- 0.137707)
           = exp(0.014333) .. exp(0.289747) = [1.01444, 1.33611]

INTERPRETATION
  A 16% relative increase with an interval whose lower bound is 1.4%. On the ratio scale the
  result looks decisive; on the absolute scale it is thin. Reporting only one scale would let
  the analyst choose the framing, so the protocol fixes both.
```

### 6.4 Derivation 2: what sample size the design can actually support

The commonest defect in this kind of pilot is announcing a target effect and then fielding a sample
that could never detect it. The power calculation is run before fielding and again after, using the
achieved size.

```text
FORMULA
  n_per_arm = 2 * (z_{1-alpha/2} + z_{1-beta})^2 * pbar*(1-pbar) / delta^2
  equivalently, the minimum detectable effect at fixed n:
  MDE = (z_{1-alpha/2} + z_{1-beta}) * sqrt( 2 * pbar*(1-pbar) / n_per_arm )

WHERE
  pbar = pooled response proportion, dimensionless
  delta = target absolute difference, dimensionless
  n_per_arm = distinct HCPs per arm
  alpha = 0.05 two-sided -> z = 1.959964 ; power 0.80 -> z_beta = 0.841621

CONSTRAINT
  Normal approximation valid when n*pbar and n*(1-pbar) both exceed 5; equal allocation;
  no clustering; a single look at the data. Sequential looks require an alpha-spending rule.

NUMERICAL EXAMPLE
  Suppose the protocol targets delta = 0.04 with pbar = 0.29:
  z_sum      = 1.959964 + 0.841621 = 2.801585
  z_sum^2    = 7.848878
  pbar*(1-pbar) = 0.29 * 0.71 = 0.2059
  numerator  = 2 * 7.848878 * 0.2059 = 3.232168
  n_per_arm  = 3.232168 / 0.0016 = 2020.1  -> 2021 per arm (4,042 HCPs total)

  Now invert it for the fielded design of 1,000 per arm:
  MDE = 2.801585 * sqrt(2 * 0.2059 / 1000)
      = 2.801585 * sqrt(0.0004118)
      = 2.801585 * 0.0202929
      = 0.056852  -> 5.69 percentage points

INTERPRETATION
  The fielded pilot is powered for 5.7 pp, not the 4.0 pp the protocol named. The observed
  4.4 pp in 6.2 therefore sits below the detection threshold, and its interval is wide for a
  structural reason, not because of bad luck. This single calculation converts an apparent
  positive result into a mandate for a larger test or a narrower question.
```

### 6.5 Derivation 3: how much unmeasured confounding would be needed

Because the historical schedule is selected on judgement, the descriptive association is the number
a stakeholder will quote, and it needs a fragility statement rather than a caveat.

```text
FORMULA
  RR = p1 / p0        (risk ratio for the contrast of interest)
  E  = RR + sqrt( RR * (RR - 1) )     for RR >= 1

WHERE
  RR = risk ratio of response comparing the contrast groups, dimensionless
  E  = minimum strength of association (on the risk-ratio scale) that an unmeasured
       confounder would need with BOTH the assignment and the outcome, above and beyond
       measured covariates, to move the estimate to the null

CONSTRAINT
  Applies to a single well-defined binary contrast; the confounder must be capable of
  varying the exposure and of affecting the outcome independently; for common outcomes the
  approximation is conservative in the direction of understating fragility.

NUMERICAL EXAMPLE
  Randomised contrast: RR = 0.312/0.268 = 1.16418
    E_point = 1.16418 + sqrt(1.16418 * 0.16418) = 1.16418 + sqrt(0.191135)
            = 1.16418 + 0.437191 = 1.60137
  Randomised CI lower bound: 0.2723/0.268 = 1.01604
    E_ci   = 1.01604 + sqrt(1.01604 * 0.01604) = 1.01604 + sqrt(0.016297)
            = 1.01604 + 0.127660 = 1.14370
  Historical descriptive contrast (constructed): p_high = 0.41, p_low = 0.19
    RR_obs = 2.15789
    E_obs  = 2.15789 + sqrt(2.15789 * 1.15789) = 2.15789 + sqrt(2.49860)
            = 2.15789 + 1.58070 = 3.73859

INTERPRETATION
  The randomised point estimate is defeated by an unmeasured confounder that roughly doubles
  the risk of both selection and response (E = 1.60), which is plausible for relationship
  history; the interval's lower bound is defeated by something far weaker (E = 1.14). The
  descriptive association requires a confounder of 3.74, which is implausible and therefore
  reads as robust — a dangerous conclusion, because the descriptive contrast has already
  absorbed the confounder into the exposure rather than leaving it unmeasured.
```

### 6.6 Derivation 4: cost per incremental responder and the commercial threshold

The owner's real question is not "is it significant" but "is the lower bound worth the spend".

```text
FORMULA
  cost_per_incremental_responder = (C * n_assigned) / (delta_hat * n_assigned)
                                 = C / delta_hat
  worst_case (lower CI bound)    = C / delta_lower

WHERE
  C = fully loaded cost per assigned incremental contact, currency units
  delta_hat = absolute risk difference, dimensionless
  delta_lower = lower bound of the 95% interval, dimensionless

CONSTRAINT
  One contact per assignment; no additional follow-up cost; the cost is the marginal cost of
  the incremental contact, not the fully loaded cost of the whole field force.

NUMERICAL EXAMPLE
  C = 140 currency units per assigned contact (ILLUSTRATIVE)
  delta_hat = 0.044  -> 140 / 0.044 = 3,181.82 per incremental responder
  delta_lower = 0.004273 -> 140 / 0.004273 = 32,764.8 per incremental responder
  at the design MDE of 0.0569: 140 / 0.0569 = 2,460.5 per incremental responder

INTERPRETATION
  The point estimate implies a cost that might be acceptable; the interval's lower bound
  implies a cost roughly ten times higher than the point estimate. A threshold policy that
  approves the programme when the worst-case cost is under a stated ceiling would reject at
  this precision. The calculation makes the decision rule numeric instead of rhetorical.
```

### 6.7 Derivation 5: clustering, and why HCP-level assignment may overstate precision

HCPs share practices; a visit to one can move a colleague. The design effect converts the nominal
sample into the sample the inference actually has.

```text
FORMULA
  DEFF = 1 + (m - 1) * rho
  n_effective = n_total / DEFF
  n_required_clustered = n_required_independent * DEFF

WHERE
  m   = mean number of assigned HCPs per practice (dimensionless, > 1)
  rho = intraclass correlation of the binary outcome within practice, 0..1
  DEFF = variance inflation relative to independent assignment

CONSTRAINT
  Exchangeable correlation, roughly equal cluster sizes, and enough clusters (>= 30) for the
  variance estimate to be stable. With few clusters the inflation factor understates the loss.

NUMERICAL EXAMPLE
  m = 8, rho = 0.05
  DEFF = 1 + 7 * 0.05 = 1.35
  n_effective for the fielded 2,000 = 2000 / 1.35 = 1,481.5
  clustered requirement for the 4,042 target = 4042 * 1.35 = 5,456.7 -> 5,457 HCPs
  practices needed = 5457 / 8 = 682.1 -> 683 practices
  sensitivity: rho = 0.10 -> DEFF = 1.70 -> 4042 * 1.70 = 6,871.4 -> 6,872 HCPs

INTERPRETATION
  Ignoring a modest correlation overstates precision by 35% and understates the recruitment
  requirement by roughly 1,400 HCPs. The effective sample of 1,482 also explains why the
  observed interval is wider than a naive reader expects from "2,000 HCPs".
```

### 6.8 Boundary case: where these formulas break

Three boundaries are worth stating explicitly, because each has produced a wrong published answer.

**Boundary 1 — unequal arms with a strong covariate.** Suppose the offer arm holds 1,200 HCPs and
the usual arm 800, and the offer arm contains proportionally more high-volume prescribers. The
pooled difference remains unbiased only if assignment is random; with unequal sizes a chance
imbalance has a larger effect on the smaller arm's mean. Recomputing: if high-volume HCPs respond at
0.45 and low-volume at 0.20, and the offer arm is 40% high-volume against 25% in the usual arm, the
crude difference is 0.10 while the volume-standardised difference is 0.00. The formula in 6.2 does
not detect this; only the balance table and a prespecified standardisation do. This is Simpson's
reversal in its ordinary commercial form.

**Boundary 2 — zero cells.** If responders0 = 0, the risk-ratio interval is undefined, the
log-scale standard error is infinite, and software will either drop the stratum or emit a huge
interval. The remedy is to report the risk difference, use an exact or penalised method, and state
that the ratio is not estimable rather than printing an artefact.

**Boundary 3 — a "perfect" predictor.** If a feature reproduces the outcome exactly in the pilot,
the cause is almost always leakage — a post-freeze record, a cancelled prescription that the feed
still counts, or a flag written by the same process that generates the outcome. The correct response
is to remove the feature and audit the pipeline, not to celebrate a high area under the curve. In
this constructed case the perfect-predictor check was the negative control described in §5d; a
spuriously strong week-of-year coefficient would have signalled a join error rather than a finding.

### 6.9 Derivation 6: reading the result before the data are mature

The pilot's calendar forces a read at day 120 after the first assignment. The 56-day window plus a
12-week settlement lag means the endpoint is not fully observable then.

```text
FORMULA
  observable_fraction(t) = clamp( (t - L) / W , 0, 1 )
  earliest_readable_day  = W + L

WHERE
  t = days from assignment to the analysis cut
  L = settlement lag in days (constant assumed across arms)
  W = outcome window length in days

CONSTRAINT
  The lag must be constant across arms and channels; if one channel reports faster, the
  observable fraction differs by arm and the comparison is biased toward the fast channel.

NUMERICAL EXAMPLE
  W = 56, L = 84, t = 120
  observable_fraction = (120 - 84) / 56 = 36 / 56 = 0.6429
  earliest_readable_day = 56 + 84 = 140
  at t = 112: (112 - 84)/56 = 28/56 = 0.5  -> only half the window is visible
  at t = 200: (200 - 84)/56 = 116/56 = 2.07 -> clamped to 1.0 (fully observable)

INTERPRETATION
  A day-120 read covers 64% of the window, and the covered part is not a random 64%: it is the
  early part, where response is dominated by HCPs who were already inclined to prescribe. Any
  comparison at day 120 therefore estimates an early-response contrast, not the 56-day endpoint.
```

**Counterexample to the lag assumption.** If the feed were real-time (L = 0), the formula gives
observable_fraction = 1 at day 57, which invites the conclusion "the endpoint is final". It is not:
a real-time feed can still be revised for weeks after first arrival, so "observed" and "final" are
different properties. The formula models observability, not stability; stability requires its own
revision audit, which is why A-02 exists separately from the observability calculation.

### 6.10 Multiplicity across the prespecified subgroups

Six subgroups were listed before the read: high versus low prior volume, access-positive versus
access-unknown, specialty A versus specialty B, and three territory bands collapsed to two.

```text
FORMULA
  FWER = 1 - (1 - alpha)^k
  Bonferroni threshold = alpha / k

WHERE
  k = number of independent-ish tests; alpha = per-test type I error rate

CONSTRAINT
  Independence is assumed for the bound; positively correlated tests make the bound
  conservative, which is acceptable for a decision gate but not for a claim of discovery.

NUMERICAL EXAMPLE
  k = 6, alpha = 0.05
  (0.95)^6 = 0.735092
  FWER = 1 - 0.735092 = 0.264908  -> 26.5%
  Bonferroni threshold = 0.05 / 6 = 0.008333

INTERPRETATION
  Scanning six subgroups at the 5% level yields a 26.5% chance of at least one spurious
  "significant" subgroup even when nothing is happening. With the observed z of 2.171, no
  subgroup test could clear 0.0083 without a much larger effect, so the subgroup results are
  descriptive and cannot carry the decision.
```

## 7. The Finding

The constructed analysis yields four findings, in decreasing order of decision relevance.

First, the incremental offer is associated with a 4.4 percentage-point higher response, with a 95%
interval of 0.43 to 8.37 points. The direction is consistent with a real effect, but the interval's
lower bound does not clear any plausible commercial minimum once cost per contact is applied.

Second, the design could not have detected the effect it targeted. At 1,000 HCPs per arm the minimum
detectable effect is 5.69 points; the protocol's 4.0-point target needed 2,021 per arm before any
clustering adjustment. The pilot is therefore underpowered by construction, and its wide interval is
a design fact rather than a surprise.

Third, the descriptive association between contact intensity and response is roughly twice as large
as the randomised contrast and requires an implausibly strong unmeasured confounder to explain away,
which is exactly why it must not be used as an effect estimate. Its apparent robustness is an
artefact of the exposure having absorbed the selection.

Fourth, the timing constraint binds. A day-120 read covers 64% of the outcome window, and the covered
part over-represents early responders. The endpoint as specified becomes readable at day 140.

The finding as a sentence: the evidence supports running a correctly powered test of the allocation
policy, and does not support a broad rollout on the current pilot.

## 8. System/Solution Design

### 8.1 What has to exist for the decision to be repeatable

A decision like this fails on plumbing far more often than on statistics, so the design is stated as
components with explicit failure behaviour.

| Component | Responsibility | Input | Output | Failure behaviour |
|---|---|---|---|---|
| Roster snapshot service | Freeze the eligible population at 06:00 Monday | Identity, role, permission, exclusions | Immutable assignment-eligible table with a version hash | Refuses to publish a partial snapshot; the cycle is skipped and logged |
| Feature builder | Compute pre-freeze covariates only | Snapshot plus historical extracts | Versioned feature table with event-time bounds | Rejects any feature whose maximum event time exceeds the freeze |
| Assignment service | Apply the allocation rule and record the draw | Feature table plus capacity | Assignment rows with arm, timestamp, and rule version | Halts on rule-version mismatch; never re-draws silently |
| Permission gate | Check permission immediately before contact | Assignment plus permission state | Deliver or suppress decision with reason code | Suppression is recorded as a first-class outcome, not an error |
| Delivery log | Record what actually happened | Scheduling system events | Delivery rows keyed to assignment | Missing delivery rows are treated as non-delivery, flagged for audit |
| Maturity gate | Decide when the outcome may be read | Settlement audit per extract | Read/no-read verdict with coverage | Blocks the read; no partial-window comparison is published |
| Analysis runner | Execute the frozen script | Snapshot, features, outcomes, plan | Result tables with code hash | Refuses to run if the protocol hash differs from the registry |
| Disclosure filter | Enforce minimum-cell rules | Result tables | Releasable aggregates | Suppresses and reports which cells were withheld |

### 8.2 Design decisions that deserve justification

The snapshot is immutable and append-only because a mutating roster makes every historical result
irreproducible, and irreproducibility in a regulated review is indistinguishable from error.
Features are bounded by event time rather than arrival time because late-arriving records are the
most common source of silent leakage in this domain.
The permission gate sits downstream of assignment, so a suppressed contact still counts in the
intention-to-treat denominator; moving the gate upstream would silently truncate the population and
convert a policy contrast into an efficacy contrast.
The maturity gate is a separate service rather than a rule inside the analysis script, because the
person who wants the answer is usually not the person who should certify the data.
The disclosure filter runs before human review, not after, because once an HCP-level table has been
seen, the disclosure control has already failed.

### 8.3 Operating cadence

Weekly: snapshot, feature build, assignment, delivery, and a coverage check on the delivery log.
Monthly: reconciliation of assignment against delivery, permission-state audit, and a duplicate-key
audit on the roster.
Per read: maturity certification, frozen-script execution, balance table, primary estimate, and the
guardrail table, in that order and with no optional steps.
Per quarter: a review of the plan itself — was the prespecified effect still the right target, and
did any assumption in §4 change its verification result.
Annually: retirement review of the whole pipeline, with a written decision either to continue or to
retire and archive.

### 8.4 What is deliberately not built

No real-time personalisation service, because a per-contact model would require reading outcomes at
day-1 latency, which the settlement lag forbids.
No patient-level linkage, because it is out of scope and would change the privacy posture of the
entire programme.
No automated reallocation loop, because the interference assumption is unverified and an automated
loop would propagate a modelling error across cycles before anyone could inspect it.

## 9. Evaluation & Validation

### 9.1 The measurement contract

| Element | Definition |
|---|---|
| Numerator | Distinct assigned HCPs with at least one qualifying event in days 1..56 after their assignment timestamp |
| Denominator | Distinct HCPs assigned in the arm, including suppressed contacts and non-deliveries |
| Population | Eligible reachable HCPs at the 06:00 Monday freeze, first eligible assignment only |
| Window | 56 calendar days from assignment, inclusive of days 1 and 56, in the market's local time zone |
| Comparator | Usual-schedule arm under the manual policy, measured in the same window and the same feed |
| Guardrails | Permission violations = 0; unapproved content deliveries = 0; independence breaches = 0; disclosure suppressions applied where required |
| Uncertainty | 95% interval for the risk difference and for the risk ratio; cluster-robust check if practice correlation is non-trivial |
| Status | Not executed: no data were collected, and no human subjects were involved. All numbers above are arithmetic on constructed inputs |

### 9.2 Validation performed in the construction

Four internal validations were run while building this case, and each is described with its outcome.
Arithmetic re-check: every derivation in §6 was recomputed by hand from the stated inputs, and the
intermediate values printed in each block are those hand computations, not software output.
Dimensional check: risk differences are dimensionless proportions reported in percentage points;
cost per responder divides currency by a dimensionless proportion and yields currency, which is the
correct unit.
Monotonicity check: increasing the sample size must shrink the interval; recomputing 6.2 at 4,000 per
arm gives SE = sqrt(2*0.2147/4000) ≈ 0.01036 and an interval half-width of 0.0203, roughly half the
published width, which is the expected square-root scaling.
Limit check: the E-value formula returns exactly 1 at RR = 1, which it does by construction
(1 + sqrt(1*0) = 1), confirming the implementation has no offset error.

### 9.3 What validation would be required in a live deployment

Independent recomputation of the primary estimate by a second analyst using a separately written
script, reconciled to the last decimal; any difference is investigated before publication.
A blinded read of a historical cycle to test whether the pipeline reproduces a known result.
A negative-control outcome that should show no effect, such as an unrelated chronic therapy.
A falsification test on pre-period outcomes: if the arms differ on outcomes measured before
assignment, the assignment mechanism is suspect.
A guardrail audit performed by someone outside the analytics team, reporting counts rather than
impressions.
A disclosure rehearsal: run the release filter on synthetic small cells and confirm suppression.

### 9.4 Guardrails that can veto a passing result

A statistically convincing result does not override a guardrail breach. A single permission
violation voids the read and triggers an incident review, because the population has been altered by
an unapproved contact. A single unapproved content version voids the read for the same reason.
A disclosure failure voids the release even if the analysis is correct, because the harm is already
done at the moment of release. Independence breaches are escalated rather than counted, since they
are a governance event rather than a metric.

### 9.5 Uncertainty that the interval does not capture

The interval captures sampling variability under the design. It does not capture the possibility that
the settlement lag differs by arm, that the roster key drifted mid-study, that the outcome definition
changed version during the window, or that capacity was reallocated in a way that correlates with
territory. Each of those is an assumption failure and is handled by the §4 register, not by widening
the interval. Reporting an interval without reporting these is a form of false precision.

## 10. Action Taken

The constructed decision record contains one decision and three conditions.

Decision: do not roll out the model-selected allocation this quarter. Instead, authorise a
correctly powered test at 2,021 HCPs per arm before clustering adjustment, or 5,457 HCPs if the
practice-level design effect of 1.35 is confirmed, subject to review approval and capacity.

Condition 1 — precision. The owner accepted that a 4.4-point point estimate with a 0.43-point lower
bound cannot support a spend decision, and set the rule that a future read must show a lower bound
above the commercial minimum implied by the cost calculation in 6.6.

Condition 2 — timing. The read date moved to day 140, with a maturity certification required before
the analysis runs. A day-120 read may be produced only as an explicitly labelled interim, never as
the decision read.

Condition 3 — interference. Practice-level correlation is to be estimated from prior data before the
larger test is designed; if ρ exceeds 0.10, assignment moves to practice level and the recruitment
target rises to 6,872 HCPs.

Actions explicitly not taken: no content change, no targeting change, no patient-data linkage, no
change to the manual schedule during the waiting period.
The manual schedule continues unchanged while the larger test is prepared, which means the
comparator remains the same policy rather than becoming a partly-treated hybrid.
The deferral carries a named trigger: the larger test is authorised when review approves the
protocol and the maturity certification passes, or the topic returns to the backlog if neither
happens within two quarters.

## 11. Outcome & Results

Because nothing was deployed, the "outcome" here is the outcome of the constructed analysis and of
the simulated decision, stated with its evidential status.

The primary estimate is 4.4 percentage points (ILLUSTRATIVE), interval 0.43 to 8.37 points, on 2,000
constructed HCPs. It is a computed quantity, not an observed one, and no claim about any real brand,
market, or HCP population is implied.

The design diagnostic is that the pilot's minimum detectable effect, 5.69 points, exceeds the effect
the protocol targeted. This is a property of the arithmetic in 6.4 and would hold for any pilot of
this size and baseline rate.

The fragility diagnostic is that the randomised interval's lower bound would be explained away by an
unmeasured confounder with risk ratios of about 1.14 with both assignment and response. A
relationship-history variable of that strength is entirely plausible in this setting, which is why
the result was treated as directional rather than confirmatory.

The commercial diagnostic is a cost per incremental responder of 3,181.82 currency units at the point
estimate and 32,764.8 at the lower bound — a factor of about ten between the optimistic and
pessimistic readings of the same data.

The decision outcome was a documented deferral with three conditions and a named trigger. Under the
governance vocabulary of this collection, a deferral is a result: it is a recorded decision with a
reason, an owner, and a revisit condition, and it is auditable in a way that a silent non-decision is
not.

The negative outcome was also recorded: the descriptive driver analysis was retired as a
decision input. It remains available as a workflow audit, and its top drivers were logged so that a
future test can check whether the randomised effect concentrates where the descriptive analysis
predicted. If it does not, that is evidence about the descriptive model, not about the intervention.

## 12. Failure Modes & What Went Wrong

Six mechanisms are set out below. Each is stated with the symptom an analyst would see, the competing
causes that produce the same symptom, and a check that separates them. The instruction "monitor the
model" is not a failure mode and is not listed.

### 12.1 Confounded assignment (the primary mechanism)

Symptom: contact intensity correlates strongly with response even after covariate adjustment, and the
adjusted and crude estimates differ by a large factor.
Competing causes: a genuine effect of contact; selection of likely prescribers into contact; a
leakage feature that encodes the outcome; or a population shift between the periods compared.
Discriminating check: examine the pre-period outcome trend by contact-intensity decile. If the
high-contact decile was already diverging before the contact period, the association is selection.
A second check is the propensity distribution: if assignment probability is near 1 for the top
decile, positivity fails and no weighting scheme can fix it.
Correction used here: randomise, and report the descriptive result only as an audit.

### 12.2 Immature outcome treated as final

Symptom: the response rate rises every time the extract is refreshed, and the arm difference shrinks
as it does.
Competing causes: a real effect decaying over time; settlement of late-arriving events; retroactive
cancellation of recorded prescriptions; a change in the outcome definition version mid-window.
Discriminating check: take two extracts 30 days apart and count revisions by arm and by event date.
If revisions concentrate in the last two weeks of the window, the lag is the cause. If revisions are
symmetric across arms, the comparison is stable even though the level is not.
Correction used here: the maturity gate, plus a coverage column in every result table.

### 12.3 Interference across HCPs in a shared practice

Symptom: the estimate changes sign or magnitude when the analysis is re-run with practice-level
clustering, and the confidence interval widens unexpectedly.
Competing causes: genuine spillover; an unlucky chance imbalance concentrated in a few large
practices; a key-resolution error that merges distinct HCPs into one practice.
Discriminating check: estimate the intraclass correlation on a pre-period outcome and compare it with
the post-period value. A jump suggests spillover; a stable small value suggests sampling noise.
Correction used here: estimate ρ before designing the larger test, and pre-commit to practice-level
assignment if ρ exceeds 0.10.

### 12.4 Intention-to-treat and as-treated divergence read as a contradiction

Symptom: the as-treated effect is much larger than the intention-to-treat effect, and stakeholders
conclude the intervention "works when delivered".
Competing causes: dilution from non-delivery; receipt correlated with motivation, which also drives
response; suppression concentrated in one channel; a delivery log that under-records contacts.
Discriminating check: compute receipt rates by arm and by covariate stratum. If receipt correlates
with prior prescribing, the as-treated contrast is confounded by the same selection the randomisation
removed. A complier-average bound can then be computed, but it inherits the exclusion restriction and
must be labelled as such.
Correction used here: the intention-to-treat estimate is primary; as-treated is secondary and cannot
be quoted without the receipt table.

### 12.5 Eligibility drift and roster staleness

Symptom: contacts are attempted to HCPs who are no longer eligible, and the excluded population
differs from the analysed one.
Competing causes: a licence or role change between snapshot and contact; a duplicate key created by a
merge; a permission expiry during the cycle; a market reclassification.
Discriminating check: recompute eligibility at contact time against the frozen snapshot and classify
every discrepancy by reason code. If discrepancies cluster in one territory, the cause is a
processing defect rather than genuine change.
Correction used here: the permission gate downstream of assignment, and a weekly duplicate audit.

### 12.6 Proxy gaming through low-value events

Symptom: responders are concentrated among HCPs with exactly one qualifying event, and the event mix
shifts toward the smallest eligible unit.
Competing causes: a real increase in trial; a change in pack size or days supply; a coding change in
the feed; a shift from another therapy that the proxy counts as new.
Discriminating check: decompose the event count by product, pack, and new-versus-repeat status. If
the increase is entirely in the smallest pack and in repeat patients, the proxy is measuring
transaction shape rather than behaviour.
Correction used here: report the event mix alongside the rate, and treat a mix shift as a signal to
revise the proxy before the next read.

## 13. Transferable Lesson

The transferable lesson is that a driver analysis and a decision analysis are different objects, and
a launch team usually needs the second while commissioning the first.

A driver analysis answers "what correlates with the outcome in the data I have". A decision analysis
answers "what should I do next, given what I can learn in the time available". The first can be
produced from historical data alone; the second requires a design, a baseline policy, a precision
target, and a maturity rule, none of which exist in the historical data.

The second lesson is arithmetic discipline at the front of the project rather than the back. The
minimum detectable effect calculation takes minutes and would have changed the pilot's size before it
was fielded. Running it after the read only produces an explanation for a wide interval that nobody
wanted.

The third lesson is about the seduction of the descriptive estimate. The confounded association
looked robust precisely because the confounder had been absorbed into the exposure. Fragility
statements computed on a confounded estimate are not reassurance; they measure how much of the
selection the analyst already failed to name.

The fourth lesson is that maturity is a gate, not a footnote. Any metric built on a settling source
has two dates — when the event happened and when it became knowable — and a decision read that
ignores the second is estimating a different quantity than the one it claims.

The fifth lesson is that deferral with conditions is a decision. It requires an owner, a trigger, and
a record, and it is the correct output when precision, timing, or guardrails are inadequate. The
failure mode it prevents is the quiet rollout justified by a point estimate whose interval nobody
printed.

## 14. Connections & Cross-References

### 14.1 Patterns in this collection

`causal-confounding` is the primary pattern: the driver is entangled with the assignment mechanism.
`evaluation-mismatch` appears in the gap between the offline response rate and the decision the owner
must make, including the cost-per-responder framing that the offline metric ignores.
`label-delay` appears as the settlement lag and the maturity gate, and it is the reason the read date
moved rather than the reason the estimate was wrong.

### 14.2 Contrast with sibling cases

Compared with a purely observational driver study, this case adds a design and a precision target, so
its conclusion is about a policy rather than a coefficient.
Compared with a deployment case, it stops before any production use, which makes it a useful
reference for what a governance-grade "no" looks like.
Compared with a forecasting case, the hard part here is identification and timing, not model
capacity; adding more features would have made the confounding worse by giving the selection more
room.

### 14.3 Prerequisites and handoffs

Prerequisite: two-proportion inference, design effects for clustered data, and the distinction
between event time and arrival time.
Handoff to access and formulary work: an access indicator is a plausible effect modifier and a
plausible confounder, and must be measured before assignment if at all.
Handoff to omnichannel attribution: the same interference and maturity problems recur when several
channels touch the same HCP, with the added complication that channels are correlated by design.
Handoff to brand-performance decomposition: any decomposition that uses contact volume as an input
inherits the confounding described here unless it is built on randomised or natural variation.

### Interview pack

**Q1. The pilot reports 312/1,000 versus 268/1,000. Compute the risk difference and its 95% interval,
and say whether the result supports a rollout.**
Answer: 0.044, SE 0.02027, interval [0.0043, 0.0837]; it supports a larger test, not a rollout.
Reasoning: The difference excludes zero but its lower bound is 0.43 points, which at 140 currency
units per contact implies a cost per incremental responder of about 32,765 against a point estimate
of about 3,182 — a ten-fold spread that no spend decision should ignore.
Follow-up: Recompute the interval at 4,000 per arm and state what happens to the lower bound.
Red flag: Calling the result "significant, therefore roll out" without converting the lower bound
into the decision's own units.
Scoring: 2 points for the point estimate and correct SE form, 2 for the interval, 1 for the
decision-relevant interpretation.

**Q2. Why is the minimum detectable effect of this design larger than the target the protocol named?**
Answer: At 1,000 per arm with a baseline near 0.29, the MDE is 5.69 points, while 4.0 points needs
2,021 per arm; the pilot was sized for the smaller target only if the sample had been more than
doubled.
Reasoning: MDE scales as the square root of the variance over n; the protocol's target implies a
sample the pilot did not field, and the discrepancy is arithmetic rather than empirical.
Follow-up: Holding the effect at 4.0 points, what sample gives 90% power, and how much does that cost
at 140 units per contact?
Red flag: Explaining the wide interval by "noise in the data" rather than by design size.
Scoring: 2 for the formula, 2 for the correct MDE value, 1 for identifying the sizing error rather
than the observed result as the cause.

**Q3. An unmeasured confounder check returns E = 1.60 on the point estimate and 1.14 on the interval
bound. Which number should drive the decision, and why?**
Answer: The interval bound; it describes the weakest confounder that would make the result
compatible with no effect, which is the relevant question for a decision at this precision.
Reasoning: The point-estimate E-value answers how much confounding would nullify the best guess,
while the bound answers how much would nullify the claim of any effect; with a wide interval the two
differ by a lot and the second is closer to the decision being made.
Follow-up: What if the confounder is correlated with a measured covariate already in the model?
Red flag: Treating the E-value as a probability, or as proof that the effect is real because 1.60
"sounds large".
Scoring: 2 for choosing the bound, 2 for the reasoning distinction, 1 for a credible named confounder
in this domain.

**Q4. The naive descriptive estimate is 2.16 on the risk-ratio scale with E = 3.74, so it looks more
robust than the randomised result. Explain the paradox.**
Answer: Its robustness is manufactured by the exposure having absorbed the selection: the confounder
is inside the contrast rather than outside it, so an E-value computed on the contrast understates how
much selection it carries.
Reasoning: E-values quantify residual confounding after measured adjustment; they cannot quantify
confounding that determined who received the exposure in the first place, and near-deterministic
assignment also breaks positivity.
Follow-up: Give the diagnostic that would reveal this before any modelling.
Red flag: Concluding that the observational result is more trustworthy because its fragility
statistic is larger.
Scoring: 2 for the absorption explanation, 2 for naming positivity, 1 for the pre-modelling
diagnostic (assignment probability by decile).

**Q5. Design effect with 8 HCPs per practice and ρ = 0.05. Compute it, and the recruitment target if
the independent requirement is 4,042.**
Answer: DEFF = 1.35; target 5,457 HCPs, or 683 practices at 8 per practice.
Reasoning: 1 + 7×0.05 = 1.35, and 4,042×1.35 = 5,456.7, rounded up; 5,457/8 = 682.1, rounded up to
683 practices because a partial practice cannot be split for clustering purposes.
Follow-up: What if ρ = 0.10, and what does that imply about the choice of assignment unit?
Red flag: Multiplying by 1.35 but then dividing practices by 7 instead of 8, or rounding down.
Scoring: 2 for DEFF, 2 for the HCP total, 1 for the practice count with correct rounding direction.

**Q6. At day 120 with a 56-day window and an 84-day lag, how much of the window is observable, and
why is that worse than "64% of the data"?**
Answer: 64.3%; the observable part is the early part of the window, which over-represents HCPs who
were already inclined to prescribe, so it is a biased subsample rather than a random one.
Reasoning: (120−84)/56 = 0.643, and the uncovered portion is systematically the late responders,
which is exactly the group an incremental contact is most likely to move.
Follow-up: What read date makes the full window observable, and what does the decision lose by
waiting until then?
Red flag: Treating partial observability as random missingness amenable to a simple reweight.
Scoring: 2 for the fraction, 2 for the bias direction, 1 for the earliest-readable date (140).

**Q7. Six prespecified subgroups are tested at α = 0.05. What is the family-wise error rate, and what
threshold would you pre-commit to?**
Answer: 1 − 0.95^6 = 0.2649, so about 26.5%; a Bonferroni threshold of 0.00833, or a hierarchical
testing order with a single primary subgroup.
Reasoning: The bound assumes independence and is conservative under positive correlation, which is
acceptable for a gate; the alternative is to declare subgroup analyses descriptive.
Follow-up: How would a hierarchical gate change the power for the primary endpoint?
Red flag: Reporting the smallest subgroup p-value as a finding without adjusting or prespecifying.
Scoring: 2 for the FWER value, 1 for the threshold, 2 for choosing a defensible governance rule.

**Q8. The as-treated effect is triple the intention-to-treat effect. Diagnose before celebrating.**
Answer: Receipt is correlated with motivation, so the as-treated contrast re-introduces selection
that the assignment removed; the ITT estimate is the policy-relevant one.
Reasoning: ITT preserves randomisation; conditioning on receipt conditions on a post-assignment
behaviour that shares causes with the outcome, which is the definition of the bias the design was
built to avoid.
Follow-up: Under what exclusion restriction would a complier-average effect be interpretable here?
Red flag: Concluding that delivery quality is the problem and that the intervention "works".
Scoring: 2 for the diagnosis, 2 for the ITT-precedence argument, 1 for stating the exclusion
restriction.

**Q9. You must choose between HCP-level and practice-level assignment. What evidence decides it?**
Answer: A pre-period estimate of the intraclass correlation of the outcome within practice; below
about 0.02 the HCP-level design is defensible, above about 0.10 practice-level assignment is required.
Reasoning: The design effect scales with ρ and cluster size, so the choice is quantitative rather
than a matter of preference; the pre-period estimate avoids conditioning on the treatment period.
Follow-up: With only 25 practices, how reliable is that ρ estimate, and what would you do instead?
Red flag: Choosing HCP-level because it gives more units, without estimating ρ.
Scoring: 2 for the ρ criterion, 2 for the design-effect link, 1 for recognising the small-cluster
limitation.

**Q10. A feature shows an implausibly strong relationship with the outcome in the pilot. Give three
distinct causes and the check that separates them.**
Answer: Leakage from a post-freeze record; a cancelled prescription still counted in the feed; or a
flag written by the same process that produces the outcome.
Reasoning: Each produces near-perfect separation but differs in provenance; the check is to trace the
feature's event time against the freeze, then to re-derive it from an independent source, then to
test it on a period where the outcome cannot exist yet.
Follow-up: What negative-control feature would you add to the model, and what result would exonerate
the pipeline?
Red flag: Dropping the feature without finding the cause, which leaves the join defect in place.
Scoring: 2 for three genuinely distinct causes, 2 for matching checks, 1 for the negative control.

**Q11. What does the baseline policy contribute to the estimand, and what changes if the baseline is
secretly two different policies across territories?**
Answer: The baseline defines the comparator, so the estimand is a contrast between allocation rules;
if territories run different baselines, the contrast becomes a mixture and must be stratified or
re-defined.
Reasoning: A policy contrast is only identified relative to a specified comparison; mixing baselines
means the estimated effect is a weighted average of several different contrasts with weights set by
territory size, not by the decision.
Follow-up: How would you detect baseline heterogeneity before fielding?
Red flag: Describing the baseline as "no treatment", which changes the claim without changing the
data.
Scoring: 2 for the comparator role, 2 for the mixture problem, 1 for a detection method.

**Q12. The permission gate sits after assignment. Why, and what breaks if it moves before?**
Answer: Keeping it after preserves the intention-to-treat denominator; moving it before silently
truncates the population and converts the policy contrast into an efficacy contrast among the
contactable.
Reasoning: The policy being tested includes the fact that some contacts are suppressed; removing
those HCPs changes the question and re-introduces a selection correlated with permission status.
Follow-up: If suppression reaches 30% in one arm, what does that do to the interpretation, and what
would you report?
Red flag: Treating suppressed contacts as data errors and deleting them.
Scoring: 2 for the ITT argument, 2 for the selection consequence, 1 for a 30% suppression
interpretation that stays honest about dilution.

**Q13. State the four actions available to the owner and one condition under which each is correct.**
Answer: No change when the current policy is adequate or capacity is fixed; audit when the data are
too confounded to choose but defects are suspected; a randomised test when a policy choice is pending
and a read is affordable; deferral when a named precondition is unmet.
Reasoning: The actions differ in information yield and cost, and choosing among them depends on
whether the binding constraint is evidence, time, or capacity rather than on preference.
Follow-up: Which action is correct when review approval will take longer than the quarter?
Red flag: Presenting only "roll out" and "do nothing" as the options.
Scoring: 2 for four distinct actions, 2 for correct conditions, 1 for the timeline case.

**Q14. A reviewer asks for a single number for the expected benefit. What do you give them, and what
do you refuse to give?**
Answer: The interval, or the lower bound converted into the decision's units; refuse the point
estimate as a standalone number.
Reasoning: A single point estimate without its interval invites a spend decision at the optimistic
end of a ten-fold spread, and the lower bound is the number the owner actually needs.
Follow-up: How would you present the same result to a medical reviewer rather than a commercial one?
Red flag: Quoting the point estimate alone because "that is what was asked for".
Scoring: 2 for the interval or bound, 2 for the refusal and reason, 1 for an audience-appropriate
reformulation.

**Q15. What would make you retire the descriptive driver model entirely rather than keep it as an
audit?**
Answer: If it cannot be maintained at a stable version, or if its outputs are being used as decision
inputs despite the labelling.
Reasoning: The audit value depends on the model being reproducible and clearly subordinate; once it
starts steering allocation it has taken on a causal role it cannot support.
Follow-up: What labelling or access control would let it stay as an audit safely?
Red flag: Keeping it because "more analysis is better", with no defined use.
Scoring: 2 for a concrete retirement condition, 2 for the role-drift reasoning, 1 for a control that
would prevent drift.

## 15. Quick Reference Card

| Item | Value |
|---|---|
| Decision | Route incremental approved contact by a model-selected rule, or keep the manual schedule |
| Owner | Fictional commercial decision owner; deferral allowed and recorded |
| Grain | One HCP at first eligible weekly assignment, frozen at 06:00 Monday |
| Primary estimate | Risk difference 4.4 pp, 95% CI [0.43, 8.37] pp (ILLUSTRATIVE, computed) |
| Design limit | MDE 5.69 pp at 1,000 per arm; 4.0 pp target needs 2,021 per arm |
| Clustering | DEFF 1.35 at m = 8, ρ = 0.05; 5,457 HCPs, 683 practices |
| Fragility | E-value 1.60 at the point estimate, 1.14 at the interval bound |
| Timing | Day-120 read covers 64.3% of the window; full read at day 140 |
| Cost view | 3,181.82 per incremental responder at the point estimate; 32,764.8 at the lower bound |
| Guardrails | Permission, content approval, independence, disclosure — any breach voids the read |

Decision rule: approve only when the interval's lower bound, converted into cost per incremental
responder, sits below the owner's stated ceiling, and every guardrail count is zero.

Reading order for a newcomer: §6.2 for the estimate, §6.4 for why the estimate is imprecise, §6.5 for
how fragile it is, §6.6 for what it means commercially, and §12 for what would make all of it wrong.
