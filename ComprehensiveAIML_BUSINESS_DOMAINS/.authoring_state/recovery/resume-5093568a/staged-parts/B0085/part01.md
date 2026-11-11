---
case_id: "CASE-04-01-05"
case_title: "Store Experiment Governance"
case_version: "2.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "analytical"
domain: "Retail omnichannel — store operations and merchandising"
primary_pattern: "evaluation-mismatch"
secondary_patterns: ["human-in-the-loop", "distribution-shift"]
evidence_level: "illustrative"
source_basis: "ILLUSTRATIVE constructed field-test design; no retailer, vendor, or deployment is described"
as_of_date: "2026-11-10"
owner_role: "retail analytics lead (field experimentation)"
source_index: "INDEX.md"
review_trigger: "Change in the decision threshold, guardrail margins, store fleet, measurement instrument, or review capacity"
---

# Store Experiment Governance

Every quantity in this case is **ILLUSTRATIVE**: a constructed 120-store format, a constructed
fixture change, constructed variances, and constructed board decisions, assembled so that the
arithmetic can be checked line by line. Nothing here is a measured result from a real retailer, and
no vendor, system, or company policy is being described. What is *not* illustrative is the method:
the estimators, the power algebra, the sequential-monitoring penalty, the non-inferiority test, the
audit sample size and the value-of-information comparison are standard results, and each is derived
from stated inputs so a reader can substitute their own.

The case is written from the position of a governance body rather than a modelling team, because the
expensive failures in store experimentation are rarely estimation failures. They are *authorisation*
failures: a test runs on too few stores to answer the question anyone actually cares about, a
mid-flight look converts noise into a launch memo, a guardrail breach is discovered after rollout,
or a pilot's headline metric is not the metric the investment case is written on.

## 1. The One-Line Framing

**Before the first treated week, a named board must sign a decision contract that fixes the primary
metric, the threshold that triggers rollout, the smallest effect the design can detect, the
guardrails that can stop the test, and the stopping rule — and the board's answer is a signed
scope, not a p-value.**

The binding constraint is arithmetic, not politics: the effect size the business cares about is
usually smaller than the effect size a twelve-week test on a hundred stores can resolve, so the
governance question is nearly always "which design makes the decision- relevant effect detectable,
and at what operational cost", never "is the pilot significant".

The reusable pattern is `evaluation-mismatch`: the metric that is easiest to measure precisely (unit
sales, transactions, footfall) is not the metric the investment case is denominated in (gross
profit, labour cost, availability). A test can be impeccably designed for the first and structurally
incapable of resolving the second, and no amount of extra weeks on the same instrument repairs that.

The deliverable of one governance cycle in this case is a three-line answer: approve a paired
within-store design for sixteen weeks on all 120 stores; require the profit instrument to be rebuilt
before any chain-wide commitment; authorise a 30-store expansion at the end. The reasoning behind
those three lines is what the rest of the case derives.

## 2. Context & Background

A grocery-format chain of 120 stores, all in one country, all served by four distribution centres, is
proposing to change how a fixture is replenished: smaller case packs, more frequent facings, and a
reallocated replenishment shift. The merchandising owner believes the change lifts sales by roughly
4%, because two supplier case studies and one internal trial suggest a 3–5% range. The finance
controller is unconvinced, and the store operations director is worried about labour hours in stores
that are already at their task-time budget.

This is the third such proposal in eighteen months, and the previous two are the reason a governance
body exists at all. The first ran on twelve volunteer stores for six weeks with no control group and
reported a 7% sales gain; the stores had been chosen because their managers were enthusiastic, and
when the change was extended to forty stores the measured gain fell to under 1%. The second ran a
proper two-arm test but stopped it in week three when the interim number looked strong; the launch
memo quoted the week-three figure, and the twelve-month review could not reproduce it. Neither test
had a written rule for what result would trigger what action, so in both cases the action was chosen
after the number was known.

The governance body — four seats, meeting weekly, quorum of three — was created with a narrow
mandate: it does not choose merchandising strategy, it decides whether a proposed field test is
*admissible evidence* for a strategy decision, and it decides what the test is allowed to claim. It
can also stop a running test, which turns out to be the seat that matters most.

Three facts about the operating environment shape every design choice in this case.

- Stores are the natural assignment unit. Layout, fixture, and shift changes are store-level
  interventions; splitting a store between treatment and control would contaminate both, because
  staff, backroom space, and delivery slots are shared.
- Sales are strongly store-specific. Weekly sales across the fleet have a standard deviation of about
  30% of the mean, so between-store variation dwarfs the treatment effect and must be removed by
  design rather than by sample size.
- The calendar dominates the residual variation that remains. Promotional weeks, pay weeks, school
  holidays and weather move a store's sales by more than the intervention is expected to, which is
  why any design that compares treated stores in one period with control stores in another is
  fragile.

The four distribution centres matter for a reason that is easy to miss at design time: a store's
replenishment is not independent of its neighbours'. Pallets, promotional stock and delivery slots
are shared within a DC catchment, so a treatment applied to some stores in a catchment can leak into
the control stores of the same catchment. That leakage is the mechanism behind the dilution
calculation in §6 and one of the failure modes in §12.

## 3. The Business Question

**Question.** Should the fixture and replenishment change be approved for rollout, and if so, on what
evidence, at what scope, and under what stopping rule?

The question decomposes into four decisions the board must take in sequence, and each has a
different evidence standard.

| # | Decision | Evidence standard | Who decides |
|---|---|---|---|
| 1 | Is the proposal admissible for a field test at all | Written hypothesis, named primary metric, feasible instrumentation, reversal path | Board, quorum three |
| 2 | Which design may be used | Detectable effect at or below the decision threshold within the store fleet and calendar available | Board, with the analytics lead proposing |
| 3 | Did the test meet its pre-registered rule | Primary estimate against the boundary, guardrails against their non-inferiority margins, audit within its defect threshold | Board, on a fixed review date |
| 4 | What scope is authorised now | Net benefit at the shrunk estimate, payback period, instrument maturity | Board, with finance sign-off |

**Owner.** The retail analytics lead owns the design and the analysis; the merchandising owner owns
the hypothesis and the operational change; the finance controller owns the threshold and the
benefit arithmetic; the store operations director owns the guardrails and holds a unilateral stop.

**Trigger.** A scheduled decision window (the fixture calendar closes eight weeks before the season)
or an exception signal (a guardrail breach, an audit defect rate above threshold, or a competitor
move that makes the test moot).

**Actions available.** Approve as designed; approve with a reduced scope; require a design change and
resubmit; defer pending instrumentation; reject; or stop a running test.

**Objective.** Maximise the expected value of the fixture decision net of the cost of the evidence,
subject to not degrading on-shelf availability or labour productivity beyond their margins.

**Hard constraints.** No store-period data timestamped after the analysis cut-off may enter the
estimate; the treated population must be the population the decision applies to; every action must
be reversible within one replenishment cycle; the review queue must stay within the board's weekly
capacity.

**Out of scope.** Supplier contract negotiation, category-range decisions, pricing architecture, and
any claim about competitor behaviour. The board also explicitly does not adjudicate whether 4% is a
"good" number — that is the finance controller's threshold, fixed before the test.

**A note on what "answer" means here.** The board's output is a scope plus a rule, and "we cannot
answer this with a store test" is a legitimate answer. In this case that answer is reached twice:
once for the labour-cost guardrail (where the instrument is too noisy) and once for the profit
effect (where the required fleet is 6.4 times the size of the format). In both cases the correct
governance action is to change the instrument, not to lengthen the test.

## 4. Assumptions & Constraints

The design rests on six assumptions. Each carries an owner, a test that can fail it, and a
consequence if it does fail — the third column is what separates a governance document from a
wish list.

| ID | Assumption | Test that could fail it | Consequence if it fails |
|---|---|---|---|
| A-01 | Weekly sales decompose into a stable store level plus a calendar component that is common across stores in a DC | Regress each store's weekly series on DC-week dummies; a store whose residual calendar loading exceeds 2 SD is flagged | Store excluded from pairing; pairing pool shrinks and the detectable effect rises |
| A-02 | The two-week block is long enough for the replenishment change to reach steady state and short enough that a block is internally homogeneous | Compare block-1 effect against blocks 2–8; a monotone decline implies the burn-in is too short | Burn-in extended, calendar shortened, detectable effect rises |
| A-03 | Control-block outcomes are not affected by treated stores in the same DC | Compare control stores in high-exposure DCs with control stores in low-exposure DCs | Effect estimate is attenuated; the reported number is a lower bound, and the decision threshold must be re-read |
| A-04 | The on-shelf-availability measurement is taken at a consistent time of day and does not drift with audit staffing | Re-measure a fixed panel of 20 stores twice weekly; a drift above 0.5pt invalidates the guardrail series | Guardrail declared unmeasured; rollout blocked until the instrument is fixed |
| A-05 | The board can review at most 14 decisions per week | Count arrivals and completions over four weeks | Queue forms; approvals become stale and must be re-dated or expired |
| A-06 | A treated store can be reverted to the original fixture within one replenishment cycle | Dry-run the reversal in two stores outside the test | Rollback is not credible; the test's risk profile changes and the stop rule must be tightened |

| ID | Constraint | Why it binds | Workaround |
|---|---|---|---|
| C-01 | The format has 120 stores, all needed for the season | A parallel design on 60 per arm cannot resolve a 4% effect | Within-store design, or accept a larger threshold |
| C-02 | The fixture calendar allows 16 trading weeks, not 24 | Sequential monitoring inflates the required duration | Fewer interim looks, or a wider threshold |
| C-03 | Labour hours are recorded at store-week grain, not by task | The labour guardrail has no task-level attribution | Time-and-motion sub-study in 12 stores |
| C-04 | Promotional calendar is set centrally and cannot be moved for the test | Promo weeks must be balanced across arms, not excluded | Block randomisation restricted to promo-balanced calendars |

Two of these deserve emphasis because they set up the whole case.

C-01 is the crux. A parallel two-arm design on 120 stores gives 60 stores per arm. With a
between-store standard deviation of $6,600 on a $22,000 weekly mean, and a prior-year benchmark
correlation of 0.85 reducing that to $3,477, the standard error of the difference is
$3,477 × √(2/60) = $635, and the smallest effect detectable with 80% power at a two-sided 5% level is
$1,779 — 8.1% of the mean. The proposal's expected effect is 4%. The affordable parallel design is
therefore *not capable of answering the question*, and no amount of careful analysis after the fact
changes that. The board's real decision is between a within-store design, a larger threshold, and no
test.

C-02 interacts with C-01 in a way that is routinely missed. Sequential monitoring — looking at the
data before the planned end — is not free. If the board wants four interim looks, the critical value
at each look must rise, and the duration must lengthen to preserve 80% power, which collides with the
sixteen-week calendar. The arithmetic is in §6 and it is the reason the approved plan has four looks
and not a weekly dashboard.

The most dangerous assumption in the table is A-03, not A-01. Store-level leakage is invisible in the
data: nothing in a sales file marks a control store as having received a promotional pallet that was
meant for a treated neighbour three miles away. A-01 failure announces itself as a badly behaved
residual series; A-03 failure shows up as an effect that is simply too small, which is indistinguishable
from a treatment that does not work. The mitigation is structural — assign by DC-stratified pairs and
record delivery exceptions — not analytical.

## 5. Data, Tools & Method

### 5a Data Specification

**Grain.** The analysis unit is the *store-block*: one store, one two-week block. Sixteen trading
weeks give eight blocks per store, 960 store-blocks in total, of which each store contributes four
to treatment and four to control under the balanced alternating schedule (the schedule is eight
blocks of treatment and eight of control over sixteen blocks of two weeks each — see the design note
below, which is where the 1,920 store-week figure comes from).

**Population.** All 120 stores in the format that were trading for the full sixteen weeks and had a
complete prior-year comparison window. Stores opening, closing, refitting, or changing format during
the window are excluded, and the exclusion rate is reported alongside the estimate rather than
absorbed silently.

**Keys and joins.** `store_id` (stable across the window, not reused on relocation), `block_id`
(sequential within store, derived from the trading calendar and not from the delivery calendar),
`calendar_week`, `dc_id`, `promo_flag`, `fixture_version`. All joins to price, cost and labour files
are as-of joins on the effective date, so a price change mid-block is attributed to the version in
force on the day of the sale, not the version in force at extraction.

**Cut-off timestamp.** The primary analysis uses data with `event_date` up to the final Sunday of the
window and `load_timestamp` before the analysis freeze, which is 09:00 on the Tuesday after the
window closes. Rows loaded after the freeze — late-arriving corrections, restated cost files — are
excluded from the primary estimate and carried into a restatement check. The distinction between
event time and processing time is explicit in every table.

**Units.** Sales and gross profit in the local currency, per store per week. Gross profit is
sales × realised margin rate, where the realised margin rate already nets promotional funding and
shrink. On-shelf availability in percentage points of planned facings observed in stock at the audit
time. Labour in hours per $1,000 of sales. Treatment effect sizes are reported both in currency and
as a percentage of the control mean, never as a bare index.

**What is *not* available at decision time.** The following are deliberately excluded from the
decision inputs because they are only knowable after the fact: the realised block-level fixtures,
the audit defect rate for the test window, the labour hours consumed by the change, competitor
activity inside the window, and the corrected cost file. Their absence is the reason the decision
rule uses a pre-registered threshold rather than a post-hoc "is it significant" reading.

**Baseline magnitudes (ILLUSTRATIVE).** Mean weekly sales $22,000 per store; across-store standard
deviation $6,600 (30% of the mean); realised gross margin 32.0%, giving $7,040 gross profit per
store-week; on-shelf availability 96.4%; replenishment labour 6.85 hours per $1,000 of sales at $17
per hour.

### 5b Tools & Stack

The stack is described by capability, not by product, because product versions are irrelevant to the
argument and would be invented detail.

- A versioned store-week fact table with explicit `load_timestamp`, `effective_from`, and a
  correction flag; the freeze is a table snapshot, not a query filter, so the analysis can be
  replayed exactly.
- A randomisation service that takes a seed, the store list, the stratification variables (DC, prior
  sales decile, promotional exposure) and the block schedule, and writes an immutable assignment file
  with a hash. Assignment is *drawn before* the window opens and cannot be redrawn after results are
  seen; a redraw is recorded as a protocol amendment with a reason.
- An append-only decision log: every board decision, its date, the evidence version it was taken on,
  and the individual votes. This is the artefact that makes a later audit possible.
- A guardrail monitor that computes availability and labour weekly against fixed margins on a fixed
  schedule, independent of the primary metric's schedule, so that a guardrail breach does not require
  anyone to be looking at the primary estimate.
- A review queue with an explicit capacity, because approval throughput is a real constraint (see
  C-01 in §4 and the queue arithmetic in §9).

### 5c Method Summary

Three designs were costed, and the comparison is the substance of the governance decision.

**D1 — parallel cluster trial.** 60 stores treatment, 60 control, eight weeks, store-period mean as
the outcome. Simple to explain, immune to calendar confounding within the window because both arms
trade the same weeks, but it spends the entire between-store variance on the estimate. Detectable
effect $1,779 (8.1%).

**D2 — within-store switchback.** All 120 stores, alternating two-week blocks, four analysed blocks
per arm over sixteen weeks after a two-week burn-in. Every store is its own control, so the store
level — the largest variance component — cancels exactly. The cost is that treatment and control are
compared across different calendar weeks, so any common calendar shock enters the estimate unless it
is balanced. Detectable effect $559 (2.54%).

**D3 — matched-pair stepped rollout.** Ten waves of twelve stores, each wave starting a fortnight
after the previous. Operationally attractive because it phases disruption, but the treatment effect is
entangled with wave timing, and the estimate is only interpretable under an assumption of no
time-varying common shock — which the promotional calendar violates. Retained as an *operational
phasing* plan, rejected as an *evidence* design.

The board approved D2 with four amendments: a two-week burn-in excluded from the primary estimate
(so eight blocks of measurement out of nine blocks of exposure); four interim analyses at weeks 4, 8,
12 and 16 with a stated critical-value schedule; a non-inferiority margin of 2.0 points on
on-shelf availability; and an audit of 149 randomly selected store-week records against source.

The estimator for D2 is the mean of within-store treatment-minus-control block differences, averaged
across stores with equal weight. It is a weighted average of 120 store-level contrasts, so its
variance is the within-store contrast variance divided by 120 — a structure that makes the pooled
standard error small even though each individual store's contrast is noisy. That is the whole point
of pairing, and it is also where the design's fragility lives: if the within-store contrasts are
correlated across stores (a DC-wide promotional push does exactly that), the equal-weight average's
variance is larger than the independent-store formula suggests, and the reported standard error is
too small.

### 5d Provenance, Quality, Exclusions, Leakage

**Provenance.** Each input row carries a source system, an extraction timestamp, an effective date and
a version. The price file is versioned by effective date; the cost file is restated monthly and the
restatement is recorded. Sales are attributed to the store that rang the transaction, not the store
that owns the inventory, which matters for two stores that share a backroom.

**Quality gates, in the order they run.** (1) Completeness: every store must have sixteen contiguous
weeks; a gap is a hard exclusion. (2) Reconciliation: store-week sales must reconcile to the
financial ledger within 0.1%; a larger gap sends the week to manual review. (3) Duplicate detection:
at most one row per store-block-version, checked before any aggregation. (4) Range checks on margin
rate and labour hours with a documented ceiling. (5) Assignment integrity: the hash of the assignment
file must match the pre-registration record.

**Exclusions and their accounting.** Stores excluded for refit or closure: 2 of 122 candidates in the
pool, leaving 120. Store-blocks excluded for promotional contamination: identified by the delivery
exception log, not by their outcome. The exclusion rate is reported with the estimate because
excluding on outcome is the single easiest way to manufacture a result — an excluded block that
happens to be a bad week for treatment biases the mean upward, and the bias is invisible in the
final table.

**Leakage, in four forms.**

1. *Temporal leakage into the decision*: using a revised cost file, a restated promo accrual, or a
   sales correction loaded after the freeze. The defence is the snapshot, not the intent.
2. *Assignment leakage*: a store manager learns the schedule and front-loads replenishment before a
   treatment block begins. The defence is that the schedule is not published to stores in advance,
   and the check is a block-1 effect that is implausibly large.
3. *Cross-store leakage*: treatment effects crossing between stores in the same DC through shared
   pallets, shared promotional stock, or shared labour. The defence is stratification and the check
   in A-03.
4. *Metric leakage*: a guardrail measured by the same staff who implement the treatment. Availability
   audits done by the store's own team will move with the team's incentives; the defence is an
   independent audit panel.

### 5e Reproducibility & Method-Selection Ledger

| Element | Recorded value | Why it matters |
|---|---|---|
| Assignment seed | `0x5EED-2026-0417`, drawn 2026-03-02 | Redrawing the schedule after seeing block 1 is the commonest covert analysis change |
| Assignment hash | `sha256:6c1f…a93e` printed in the pre-registration | Proves the schedule in the file is the schedule that was registered |
| Snapshot | `store_week@freeze-2026-09-08T0900` | Fixes the data universe for replay |
| Estimator | Equal-weight mean of 120 store contrasts, blocks 2–8 | Named in advance; alternatives are sensitivity analyses, not substitutes |
| Primary metric | Gross profit per store-week | The decision metric, not the easiest metric |
| Decision threshold | +$847 per store-week (3.85% of sales at constant margin) | Derived from the payback constraint in §6; fixed before the window |
| Detectable effect | $559 (2.54%) at 80% power, two-sided 5% | The design's resolution, stated before running |
| Interim looks | Weeks 4, 8, 12, 16; OBF critical values 4.333, 2.963, 2.359, 2.014 | Fixed schedule; no unscheduled looks permitted |
| Guardrail margin | Availability non-inferiority −2.0pt | One-sided at 2.5% |
| Audit plan | 149 store-weeks, simple random sample, 2% defect threshold | Sample size derived in §6 |
| Reversal | Fixture revert within one replenishment cycle, dry-run in 2 stores | Makes the stop rule credible |

The ledger is the artefact the board signs. Two properties make it useful rather than ceremonial.
First, every entry is falsifiable: the seed and hash can be checked, the snapshot can be replayed,
the estimator is a sentence that can be coded. Second, the entries were fixed *before* the outcome
was known, so a later disagreement is a disagreement about interpretation, not about what was
measured.

What the ledger deliberately does not contain is a prediction of the result. Governance documents
that embed an expected effect tend to attract analyses that confirm it, so the expected effect lives
in the business case, which is a separate document with a separate sign-off.

**Reproducibility limits.** The case is illustrative, so there is no executable artefact. If this
were a live run, the minimum replayable bundle would be: the assignment file and its hash, the frozen
snapshot identifier, the estimator specification, the guardrail computation, the audit sample draw
and the decision log. Anything less and a reviewer can reproduce a number but not the decision.

