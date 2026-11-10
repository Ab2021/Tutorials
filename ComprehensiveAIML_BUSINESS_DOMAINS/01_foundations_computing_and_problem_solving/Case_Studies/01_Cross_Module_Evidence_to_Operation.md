---
case_id: "CASE-01-PART-001"
case_title: "Move an Evidence Claim into an Operational Decision"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "analytical"
domain: "Foundations and computing"
primary_pattern: "evidence-to-operation"
secondary_patterns: ["correctness-contract", "bounded-promotion"]
evidence_level: "illustrative"
source_basis: "constructed teaching scenario"
as_of_date: "2026-11-10"
owner_role: "programme operations lead"
source_index: "INDEX.md"
review_trigger: "Any change to intake volume, staffing, decision authority, or the acceptance metric"
---

# Move an Evidence Claim into an Operational Decision

**Evidence declaration.** This is an ILLUSTRATIVE teaching case. The intake volume, the pilot
result, the staffing, the cost rate, and every derived figure below are constructed so that the
decision can be audited line by line; none of them is a company deployment or a measured outcome.
The arithmetic is VERIFIED for the stated inputs and must be recomputed for any other input.
Where the manuscript says `not-measured`, it means no observation exists in this file.

**Implementation stage:** proposed. **Editorial status:** draft. Source verification and
publication review remain OPEN. Nothing here authorises an irreversible production change.

## 1. The One-Line Framing

A pilot reports that an assisted-triage workflow cut median handling time by 22%. The programme
operations lead must decide whether to promote that workflow into the live intake, and the 22%
number is not, by itself, a decision. The decision is:

```text
Given: a paired pilot of 40 cases, a headline 22% median reduction, and an
       operating cost of verification that the pilot never charged itself,
Decide: promote to full intake | promote to a bounded share | defer for
        touch-time evidence | reject and keep manual triage,
Subject to: 20 business days, 15% of intake, one named owner, and a pre-registered
        analysis plan that can return "promote" or "reject" without renegotiation.
```

The case teaches the *transfer function* between an evidence claim and an operating decision. Three
things must hold before any claim becomes an action:

1. **Same quantity.** The pilot measured a duration; the operation pays for reviewer-minutes. If the
   pilot duration includes queue wait, part of the saving is already booked as capacity, and
   promoting it would double-count.
2. **Same population.** The pilot's 40 cases were selected, and selection changes what the mean
   difference means. A 22% reduction among favourable cases is not a 22% reduction on intake.
3. **Same denominator of value.** The benefit must be net of the new work the workflow creates:
   pre-fill review, exception handling, and audit sampling all consume reviewer time.

Framing the case as "is 22% big enough" is the wrong question. The right question is "does the net,
population-corrected, touch-time gain clear the break-even threshold with enough confidence that a
bounded promotion will not lose money while it is being measured".

**Takeaway:** an evidence claim becomes decidable only after it is re-expressed in the units the
operation actually pays for.

## 2. Context & Background

### 2.1 The work being changed

The programme processes intake cases through four steps: registration, evidence assembly, decision,
and disposition. The decision step is where reviewer judgement dominates, and it is the step the
candidate workflow touches. Disposition, downstream notification, and archival are out of scope;
a workflow that shortens the decision step does not shorten them.

The current process is entirely manual with a documented but loosely enforced evidence checklist.
Reviewers are experienced; the median case takes a quarter of an hour of reviewer attention plus
whatever waiting the queue imposes. Because the queue is invisible in the handling-time ledger,
nobody in the programme can currently separate the two components, and that is the first structural
weakness this case exposes.

### 2.2 Why the baseline is retained

The manual baseline is understandable, reversible, and cheap to reason about. Any reviewer can audit
any case by reading it. That property is worth real money in a regulated back office, and it is the
reason the decision is not "automate the decision step". Replacing a readable process with an
opaque one trades an understood cost for an unmeasured risk, and that trade requires evidence the
programme does not yet have.

The baseline also anchors the counterfactual. Every claim of improvement is a claim about the
difference between two paths over the same cases, and if the baseline stops being reproducible the
difference becomes unfalsifiable. Retaining it is therefore a measurement decision as much as a risk
decision.

### 2.3 What the pilot actually did

Over eight weeks, 40 consecutive eligible cases were processed with the assisted workflow, and each
was paired with a counterfactual estimate derived from a matched historical case. The headline
statistic is a 22% reduction in median elapsed time from case open to case close: 25.0 minutes to
19.5 minutes.

Three properties of that design matter for the decision and are frequently missed:

- The pairing is *estimated*, not randomised. Matched historical cases inherit whatever the queue
  was doing when they ran, so part of the difference is queue state, not workflow.
- The elapsed measure starts at case open and ends at case close, so it contains waiting.
- The 40 cases were eligible cases only. Eligibility was judged by the reviewers who ran the pilot,
  which is a selection mechanism with an obvious optimistic direction.

### 2.4 The operating environment

The illustrative operating point is 1,200 cases per business day inside a 10-hour operating window,
60 review seats, and a loaded reviewer cost of 72 US dollars per hour. Those three numbers are the
whole economic frame: they convert minutes into money and minutes into capacity.

Because the intake is a queue and not a conveyor belt, capacity and latency are not the same thing.
Adding two minutes of verification per case reduces effective capacity in a way that grows faster
than linearly as utilisation approaches one. Section 6 derives the size of that effect rather than
asserting it.

### 2.5 The organisational constraint

The operations lead can approve, defer, or reject, and can authorise a bounded promotion. The lead
cannot authorise an unbounded one: the programme's change policy requires a stated intake share, a
stated end date, and a stated rollback trigger. That constraint is not bureaucracy; it is what keeps
the cost of being wrong smaller than the cost of learning.

**Takeaway:** context fixes which quantity, which population, and which authority the decision
actually depends on.

## 3. The Business Question

| Field | Contract |
|---|---|
| Question | Should the assisted-triage workflow be promoted into live intake now? |
| Owner | programme operations lead, named in the decision record |
| Trigger | The pilot report lands and the next intake quarter is being planned |
| Actions | promote fully; promote to a bounded share; defer for touch-time evidence; reject |
| Objective | maximise net reviewer-minutes released per business day, per unit of risk taken |
| Hard constraint | no irreversible change without a stated share, end date, and rollback trigger |
| Soft constraint | promotion must not raise the exception backlog above its current level |
| Scope | one workflow, one intake queue, one quarter of decision authority |
| Evidence available | 40 paired pilot cases, a headline 22% median reduction, staff cost, intake rate |
| Evidence unavailable | touch-time split, intake-wide effect, exception rate under the new workflow |

The proxy metric is median elapsed time. The business objective is released reviewer capacity net
of new verification work. Those two are not interchangeable, and the gap between them is the subject
of Section 6.

Two further distinctions keep the question honest. First, a *statistically detectable* effect and an
*economically sufficient* effect are different tests; passing the first while failing the second is
the common failure of pilot-to-production handoffs. Second, a *decision* differs from a *forecast*:
the lead can choose a bounded promotion whose information value exceeds its downside, and that
choice does not require believing the 22% claim.

**Takeaway:** state the objective in the units that are scarce (reviewer-minutes), not in the units
that were convenient to measure (elapsed minutes).

## 4. Assumptions & Constraints

Each assumption below carries a test and a fallback. An assumption without a fallback is a wish, and
a wish in a decision record becomes an incident later.

- **A-01 Evidence is paired on comparable cases.** Test: inspect the matching rule and the
  distribution of matched attributes. Fallback if violated: treat the estimate as unmatched and
  require randomised or time-split comparison before promotion.
- **A-02 Elapsed time and touch time are separable.** Test: check whether the ledger records
  reviewer-active intervals or only open and close timestamps. Fallback if violated: promote only a
  measurement arm that records touch time, not a full intake share.
- **A-03 Verification cost is charged to the workflow.** Test: time a sample of pre-fill reviews and
  exception closures. Fallback if violated: assume the highest observed verification cost, not zero.
- **A-04 Intake volume is stable across the promotion window.** Test: compare the trailing 20-day
  mean to the planned volume. Fallback if violated: use the trailing maximum for capacity planning.
- **A-05 Staffing is fixed for the duration of the bounded promotion.** Test: read the roster for the
  window. Fallback if violated: express results per reviewer-hour rather than per day, so a roster
  change cannot inflate the estimate.
- **A-06 The exception rate under the new workflow is bounded.** Test: measure exception codes in the
  measurement arm. Fallback if violated: cap the share and add an exception-specific review.
- **A-07 One final action per case key is enforceable.** Test: replay duplicate submissions and count
  final actions. Fallback if violated: disable automated disposition and keep manual release.
- **A-08 The manual baseline remains runnable.** Test: keep a control share of intake manual for the
  window. Fallback if violated: stop the promotion, because no comparison is then possible.

Two constraints shape the arithmetic and are not negotiable within the case:

```text
CONSTRAINT [K-1]: the bounded promotion share is 15% of intake, fixed before measurement.
CONSTRAINT [K-2]: the decision rule is pre-registered: promote if the lower bound of the
                  95% interval on net touch-time gain exceeds the break-even threshold.
```

Pre-registering K-2 is what stops the analysis from becoming a search for a favourable subgroup.
Without it, any 40-case sample contains some subgroup that looks excellent.

**Takeaway:** every assumption needs a test that a second person could run and a fallback that the
owner can execute without new authority.

## 5. Data, Tools & Method

### 5a Data Specification

| Element | Specification |
|---|---|
| Grain | one row per intake case per workflow variant |
| Keys | `case_id` stable across variants; `decision_key` unique per decision attempt |
| Event time | `opened_at`, `closed_at`, `first_touch_at`, `last_touch_at`, all UTC with offset |
| Processing time | `ingested_at` and `scored_at`, recorded separately from event time |
| Population | eligible intake cases in the operating window; ineligible cases excluded and counted |
| Units | durations in minutes; volume in cases per business day; cost in US dollars |
| Cut-off | decision-time fields must exist at or before `decision_cutoff_at` |
| Unavailable at cut-off | final disposition, downstream outcome, adjudicated quality labels |

The last row is the leakage boundary. Disposition and outcome are recorded after the decision and
must never enter the decision-time record, even though they are the fields a retrospective analysis
most wants. A field that is written later than the decision cannot justify the decision.

`first_touch_at` is the field the pilot did not have. Its absence is why the case exists: without it,
elapsed time is the only measurable duration, and elapsed time commingles queueing with work.

### 5b Tools & Stack

The proposed stack is deliberately boring: a versioned input store, a deterministic validator, a
policy evaluator that maps validated inputs to a recommendation, an append-only decision ledger, a
work queue with a bounded retry budget, and a reviewer console that shows the recommendation next to
the evidence it used.

Three properties are required of the stack and none is optional:

- The configuration hash of the evaluator travels with every decision record.
- The ledger is append-only, so a correction is a new record rather than an edit.
- The queue holds at most one in-flight attempt per `decision_key`.

No component of that stack is claimed to be operated. The stack description exists to make the
failure modes in Section 12 concrete; a diagram of components that cannot fail is a decoration.

### 5c Method Summary

```text
[case ingest] -> [schema + cutoff validation] -> [bounded evaluator]
      -> [recommendation + reason code] -> [policy gate] -> [reviewer action]
      -> [append-only ledger] -> [audit sample]
```

The baseline path is manual triage with no evaluator. The two comparison paths are: the manual
control share, and the measurement arm at 15% of intake. A third path, defer-all, is the safe state
and is always available because it requires no new component.

The analysis method is paired comparison of durations within case, with a pre-registered threshold,
plus a queueing correction that converts service-time change into capacity change. Both parts are
required: the paired comparison says whether the duration moved, the queueing model says what that
movement is worth at the operating point.

### 5d Data provenance, quality, exclusions and leakage controls

| Control | Rule | Failure response |
|---|---|---|
| Identifier | reject rows without a stable `case_id` and `decision_key` | quarantine, count, report the count |
| Ordering | reject rows where `closed_at` precedes `opened_at` | quarantine rather than repair silently |
| Timezone | require an explicit offset on every timestamp | reject the row; never infer the offset |
| Duplicates | one final action per `decision_key` | collapse to the earliest decision, log the rest |
| Availability | every decision-time field must exist at or before cut-off | drop the field, not the row |
| Exclusions | ineligible cases excluded by rule, counted in the report | report the exclusion count and reason |
| Windows | measurement window fixed before data collection | a moved window invalidates the result |

Provenance is recorded as source system, extraction timestamp, extract version, and row count. A row
count that cannot be reconciled between extract and load means the analysis population is unknown,
and an unknown population makes every rate meaningless.

### 5e Reproducibility & Method-Selection Ledger

| Artifact | Purpose | Accessibility | Limitation |
|---|---|---|---|
| ILL-01 | the 40-case paired pilot summary | constructed for this case | not a production measurement |
| ILL-02 | intake volume and rostering frame | constructed | does not reflect any real roster |
| CFG-01 | evaluator configuration and thresholds | proposed | needs change control before use |
| ORC-01 | reference oracle for the validator | proposed | checks only the stated contract |
| AN-01 | pre-registered analysis plan | proposed | binds the analysis, not the outcome |

Method selection is recorded with the reason it was chosen over the alternative:

- Paired within-case comparison chosen over independent-groups comparison because case mix differs
  between pilot weeks; rejected alternative would confound workflow with case difficulty.
- Lower confidence bound chosen over point estimate because the decision is asymmetric: promoting a
  zero-effect workflow costs verification time on every case.
- Queueing correction chosen over a pure time-saving calculation because the operation pays for
  capacity, and capacity is not linear in service time.

**Takeaway:** the method ledger exists so a later reviewer can tell a deliberate choice from an
accident.

## 6. Analysis Walkthrough

All inputs in this section are ILLUSTRATIVE; all arithmetic is VERIFIED and shown at working
precision so a reader can reproduce it with a calculator. The walkthrough has one purpose: replace
the headline "22% faster" with a quantity that the operations lead can compare against a cost.

### 6.0 The pilot ledger, before any modelling

The 40 paired cases produce the following illustrative decomposition. The left column is elapsed
time; the middle columns split it into waiting and reviewer-active minutes; the right column splits
reviewer-active minutes into the decision step and the new verification step.

| Variant | Elapsed (min) | Wait (min) | Touch (min) | Decision touch (min) | Verification (min) |
|---|---|---|---|---|---|
| Manual baseline | 25.0 | 9.6 | 15.4 | 15.4 | 0.0 |
| Assisted workflow | 19.5 | 8.4 | 11.1 | 8.6 | 2.5 |
| Difference | 5.5 | 1.2 | 4.3 | 6.8 | −2.5 |

Two readings of this table are both arithmetically correct and only one is decision-relevant:

- The elapsed difference is 5.5 minutes, which is 22.0% of 25.0 minutes. This is the headline.
- The net touch difference is 4.3 minutes, which is 27.9% of 15.4 minutes. This is what the roster
  can sell back, because reviewer-minutes are the scarce input.

The gap between 5.5 and 4.3 is not a correction of dishonesty; it is two different measurements of
two different things. The pilot measured what a case took; the operation pays for what a reviewer
did. Both must be reported, and only the second belongs in the benefit calculation.

### 6.1 Formula 1 — the transfer from elapsed to net touch time

```text
FORMULA [F-1]:  g_net = (T_dec,0 - T_dec,1) - v
WHERE:  g_net      net reviewer minutes released per case      [min/case]
        T_dec,0    decision touch time, manual baseline        [min/case]
        T_dec,1    decision touch time, assisted workflow      [min/case]
        v          verification minutes charged per case       [min/case]
CONSTRAINT: both variants must be measured on the same case population; v must include
            pre-fill review, exception closure, and any rework the workflow creates.
NUMERICAL EXAMPLE:
        T_dec,0 = 15.4, T_dec,1 = 8.6, v = 2.5
        g_net = (15.4 - 8.6) - 2.5 = 6.8 - 2.5 = 4.3 min/case
INTERPRETATION: the gross decision-step saving of 6.8 minutes is 44.2% of baseline
        decision touch, but 2.5 of those minutes are consumed by new verification, so
        the released capacity is 4.3 minutes, or 27.9% of baseline touch.
```

The distinction between `T_dec,0 - T_dec,1` and the elapsed difference is the entire case. If a
team reports only elapsed time and then multiplies it by a loaded labour rate, it books 5.5 minutes
of capacity that does not exist. The overstatement factor here is 5.5 / 4.3 = 1.279, which on an
annualised base of 300,000 cases is 1,650,000 − 1,290,000 = 360,000 reviewer-minutes, or 6,000
reviewer-hours. That is the size of the error a single unexamined denominator can create.

**Validity condition:** F-1 is valid only when `v` is measured rather than assumed. Setting `v = 0`
because nobody timed the verification step does not make the term disappear; it makes the estimate
the gross figure and relabels it net. A conservative substitute is to set `v` to the largest
verification time observed in the audit sample, which biases the decision toward deferral rather
than toward promotion.

### 6.2 Formula 2 — daily capacity release and its money value

```text
FORMULA [F-2]:  C_rel = N_d * g_net / 60          and    V = C_rel * c
WHERE:  C_rel   reviewer-hours released per business day        [h/day]
        N_d     intake cases per business day                    [cases/day]
        g_net   net release per case from F-1                    [min/case]
        c       loaded reviewer cost                             [USD/h]
        V       value released per business day                  [USD/day]
CONSTRAINT: intake and roster are constant across the window; no backlog is
            carried in; released minutes are usable, not merely saved.
NUMERICAL EXAMPLE:
        N_d = 1,200, g_net = 4.3, c = 72
        C_rel = 1,200 * 4.3 / 60 = 5,160 / 60 = 86.0 h/day
        V = 86.0 * 72 = 6,192 USD/day
INTERPRETATION: at full intake the workflow would release 86 reviewer-hours per day.
        Because the release is capacity rather than cash, it becomes money only if the
        released hours are reassigned or the roster shrinks.
```

The last clause is the part finance will ask about and the part analysts forget. Released capacity
is an option, not a receipt. If the roster is fixed and the backlog is not, 86 hours per day of
saved touch time converts into shorter queues rather than lower cost. Whether that is worth
6,192 dollars per day depends on the cost of the queue, which is a separate measurement the case
does not have.

Cross-check F-2 against the roster directly: the baseline demand is 1,200 × 15.4 = 18,480
reviewer-minutes per day against 60 seats × 600 minutes = 36,000 available, so utilisation is
18,480 / 36,000 = 0.5133. The assisted demand is 1,200 × 11.1 = 13,320, giving utilisation
13,320 / 36,000 = 0.3700. The released demand is 18,480 − 13,320 = 5,160 minutes, which is
86 hours, agreeing with F-2. Two independent routes to the same number is the minimum standard for
a figure that will appear in a decision record.

### 6.3 Formula 3 — break-even threshold for the bounded promotion

The full-intake benefit is not what is being decided. The decision on the table is a 15%
promotion for 20 business days, and that arm carries a fixed cost that the per-case gain must
amortise.

```text
FORMULA [F-3]:  g* = (F / c * 60) / (s * N_d * D) + a * t_a
WHERE:  g*      break-even net release per promoted case       [min/case]
        F       fixed cost of standing up the measurement arm  [USD]
        c       loaded reviewer cost                           [USD/h]
        s       promoted share of intake                       [fraction]
        N_d     intake cases per business day                  [cases/day]
        D       duration of the bounded promotion              [business days]
        a       audit sampling rate                            [fraction]
        t_a     audit minutes per sampled case                 [min/case]
CONSTRAINT: fixed cost is not recoverable if the promotion is stopped early, and the
            sampled cases are drawn from promoted cases only, so audit scales with s.
NUMERICAL EXAMPLE:
        F = 9,500, c = 72, s = 0.15, N_d = 1,200, D = 20, a = 0.06, t_a = 4.0
        promoted cases = 0.15 * 1,200 * 20 = 3,600
        fixed term = (9,500 / 72 * 60) / 3,600 = 7,916.67 / 3,600 = 2.1991 min/case
        audit term = 0.06 * 4.0 = 0.2400 min/case
        g* = 2.1991 + 0.2400 = 2.4391 min/case  (round 2.44)
INTERPRETATION: the bounded arm breaks even at 2.44 minutes of net release per case,
        against a point estimate of 4.3 minutes. The margin is 4.3 / 2.44 = 1.76x.
```

Check the fixed term a second way, in whole units: 9,500 / 72 = 131.9444 reviewer-hours, which is
131.9444 × 60 = 7,916.67 minutes; spread over 3,600 cases that is 2.1991 minutes each. Both routes
agree to the fourth decimal, and the rounding to 2.44 is disclosed rather than hidden.

The audit term deserves attention because it is the term teams set to zero. Sampling 6% of 3,600
promoted cases is 216 cases at 4 minutes each, which is 864 minutes or 14.4 reviewer-hours, or
1,036.80 US dollars. It is small here only because the promotion is small. At full intake the same
6% sampling over the same 20 days would be 1,200 × 20 × 0.06 = 1,440 cases, 5,760 minutes, 96
reviewer-hours, and 6,912 dollars — a figure that starts to matter against the benefit.

### 6.4 Formula 4 — the confidence interval and the pre-registered rule

The bounded promotion is authorised by a rule that refers to an interval, not a point estimate.
The paired differences give a mean and a standard deviation; the interval on the mean decides.

```text
FORMULA [F-4]:  L = d_bar - t_{0.975,n-1} * s_d / sqrt(n)
WHERE:  L        lower bound of the 95% interval on mean difference [min/case]
        d_bar    mean paired net difference                          [min/case]
        s_d      sample standard deviation of paired differences     [min/case]
        n        number of paired cases                              [cases]
        t        Student t critical value, two-sided 95%             [dimensionless]
CONSTRAINT: pairs are independent; differences are approximately normal in the mean;
            n is the number of complete pairs after exclusions, not the enrolment count.
NUMERICAL EXAMPLE (pilot, n = 40):
        d_bar = 4.3, s_d = 6.1, t = 2.0227 for 39 degrees of freedom
        SE = 6.1 / sqrt(40) = 6.1 / 6.3246 = 0.9645
        half-width = 2.0227 * 0.9645 = 1.9509
        L = 4.3 - 1.9509 = 2.3491 min/case
INTERPRETATION: the pilot's lower bound is 2.35 minutes, below the 2.44-minute
        break-even in F-3. The pre-registered rule therefore does not authorise
        full promotion, even though the point estimate clears break-even by 1.76x.
```

The gap is 2.4391 − 2.3491 = 0.0900 minutes, about five seconds per case. A decision rule that
resolves to five seconds is not a knife-edge in any practical sense; it is a rule whose sample is
too small to settle the question it was written to settle. That diagnosis, and not the sign of the
gap, is the finding.

Two checks make the interval credible rather than decorative. First, the standard error scales as
1/√n: at the bounded arm's 3,600 cases the standard error becomes 6.1 / √3,600 = 6.1 / 60 = 0.1017
minutes and the half-width becomes 1.9600 × 0.1017 = 0.1993, so the lower bound is
4.3 − 0.1993 = 4.10 minutes and the rule passes with 1.66 minutes to spare. Second, the same
arithmetic says the *minimum* sample to pass on the pilot's own numbers is n = 42 — at n = 41 the
half-width is 2.0227 × 6.1 / √41 = 2.0227 × 0.9527 = 1.9270 and the lower bound is 2.3730, still
short; at n = 42 it is 2.0227 × 6.1 / √42 = 2.0227 × 0.9413 = 1.9040 and the lower bound is
2.3960, which still falls short. The rule is only satisfied from n ≈ 45 upward, where the
half-width drops below 1.86. This is why "collect two more cases" is not a plan and a 3,600-case arm
is.

Recalculate that last step honestly: `2.0227 × 6.1 = 12.3385`, and `12.3385 / 1.86 = 6.6336`, so the
inequality requires `√n ≥ 6.6336`, which is `n ≥ 44.00`. Because the critical value shrinks as the
sample grows, one iteration is enough: at 43 degrees of freedom `t = 2.0167`, so the half-width at
n = 44 is `2.0167 × 6.1 / 6.6332 = 1.8546` and the lower bound is `4.3 − 1.8546 = 2.4454`, which
clears 2.4391 by 0.0063 minutes. Six thousandths of a minute is not a decision margin anyone should
rely on. Reporting the required-n calculation without that caveat would be a worse error than not
reporting it.

### 6.5 Formula 5 — queueing correction, and where the model breaks

Capacity is not linear in service time. As utilisation approaches one, waiting grows without bound,
so a service-time saving is worth more than its face value at high utilisation and less at low
utilisation. The standard single-class approximation makes this explicit.

```text
FORMULA [F-5]:  W_q = (rho / (1 - rho)) * ((C_a^2 + C_s^2) / 2) * T_s
WHERE:  W_q     mean queue wait                                   [min]
        rho     utilisation = lambda * T_s / (m * T_window)       [dimensionless, 0..1]
        C_a^2   squared coefficient of variation of arrivals      [dimensionless]
        C_s^2   squared coefficient of variation of service       [dimensionless]
        T_s     mean service time per case                        [min]
CONSTRAINT: single service class, stable queue (rho < 1), no abandonment, no
            priority classes, arrivals and service roughly renewal.
NUMERICAL EXAMPLE (Poisson arrivals and exponential service, so C_a^2 = C_s^2 = 1):
        baseline: rho = 0.5133, T_s = 15.4
              W_q = (0.5133 / 0.4867) * 1 * 15.4 = 1.0547 * 15.4 = 16.24 min
        assisted: rho = 0.3700, T_s = 11.1
              W_q = (0.3700 / 0.6300) * 1 * 11.1 = 0.5873 * 11.1 = 6.52 min
INTERPRETATION: the model predicts a 60% fall in waiting, from 16.24 to 6.52 minutes.
        The pilot observed a 12.5% fall, from 9.6 to 8.4 minutes. The model's
        direction is right and its magnitude is wrong by a factor of about 4.8.
```

**This is the boundary case, and the formula breaks there.** Three facts disqualify F-5 as an
absolute-minutes estimator for this queue. The observed baseline wait of 9.6 minutes is 1.69 times
smaller than the model's 16.24, so either arrivals are far less variable than Poisson
(`C_a^2 < 1`) or the pool behaves like many parallel servers rather than one aggregated server.
Second, when `C_a^2` and `C_s^2` are both set near 0.2, as they would be for a scheduled, smoothed
intake, the baseline wait becomes 1.0547 × 0.2 × 15.4 = 3.25 minutes and the assisted wait
0.5873 × 0.2 × 11.1 = 1.30 minutes, a fall of 60% again in ratio but from a base one fifth as
large. Third, at `rho → 1` the factor `rho/(1-rho)` diverges, so any conclusion drawn near
saturation is an artefact of the model's asymptote rather than a property of the queue.

What survives all three objections is the *elasticity*, not the level. Because `W_q` scales as
`rho/(1-rho)` times `T_s`, and `rho` is proportional to `T_s` at fixed arrival rate, the local
elasticity is `d ln W_q / d ln T_s = 1 + 1/(1-rho)`. At `rho = 0.5133` that is `1 + 2.0548 = 3.055`;
at `rho = 0.3700` it is `1 + 1.5873 = 2.587`. Two consequences follow and they pull in opposite
directions:

- The elasticity is *larger* near saturation, which is why the same service-time saving is worth more
  to a busy queue. At `rho = 0.85` the elasticity is `1 + 6.667 = 7.667`.
- The elasticity is a *local* statement. Applying 3.055 to a 27.9% reduction predicts an 85%
  reduction in wait, while the exact recomputation gives 59.9% (`16.24 → 6.52`). The linearisation is
  unusable at a change of this size, which is exactly how such multipliers get misquoted.

So the honest use of the elasticity is a ranking tool for small changes, not a multiplier for large
ones. For a 10% service-time reduction at `rho = 0.5133` the elasticity predicts a 30.6% wait
reduction and the exact F-5 recomputation gives 26.7%; at `rho = 0.85` the same 10% change is
predicted at 76.7% and actually gives 48.3%. The gap widens with utilisation, so the elasticity is
most misleading exactly where a planner is most tempted to use it. A reader who wants a number for a
large change should recompute F-5 rather than scale by a derivative. What the elasticity legitimately
explains is the direction and the ordering — why the same minute saved near saturation buys more than
it does at low utilisation — and why the programme should not plan on capacity savings at its current
operating point.

**Takeaway:** use F-5 to rank scenarios and to explain why utilisation matters, never to promise a
waiting-time number.

### 6.6 Sensitivity and the boundary scan

The decision hinges on four inputs. Each is perturbed one at a time from the base case, holding the
others fixed, and the effect on the pre-registered comparison is reported. The comparison quantity
is `L - g*`, so a positive value passes the rule.

| Perturbation | Changed input | Recomputed quantity | Value | L − g* | Verdict |
|---|---|---|---|---|---|
| Base case | — | L = 4.3 − 1.9509 | 2.3491 | −0.0900 | fails |
| Verification doubles | v = 5.0 | d_bar = 1.8, L = 1.8 − 1.9509 | −0.1509 | −2.5900 | fails, sign flips |
| Verification halves | v = 1.25 | d_bar = 5.55, L = 5.55 − 1.9509 | 3.5991 | +1.1600 | passes |
| Fixed cost doubles | F = 19,000 | fixed term 4.3981, g* = 4.6381 | — | −2.2890 | fails |
| Share doubles | s = 0.30 | promoted 7,200; fixed term 1.0995 | g* = 1.3395 | +1.0096 | passes |
| Spread doubles | s_d = 12.2 | SE = 1.9291, half-width 3.9018 | L = 0.3982 | −2.0409 | fails |
| Pilot n → 3,600 | n = 3,600 | SE = 0.1017, half-width 0.1993 | L = 4.1007 | +1.6616 | passes |
| Audit doubles | t_a = 8.0 | audit term 0.4800, g* = 2.6791 | — | −0.3300 | fails |

Recompute two rows independently to confirm the table is not decorative. Verification doubling:
`g_net = 6.8 − 5.0 = 1.8`, and `L = 1.8 − 1.9509 = −0.1509`. Because the interval half-width is
1.9509 and the mean is only 1.8, the estimate is consistent with a *negative* net release — the
workflow could be releasing no capacity at all. Share doubling: promoted cases become
0.30 × 1,200 × 20 = 7,200, so the fixed term is 7,916.67 / 7,200 = 1.0995 and `g* = 1.0995 + 0.24 =
1.3395`. Note that doubling the share halves the fixed term but not the audit term, which is the
only part of the threshold that does not dilute with scale.

The boundary scan produces three conclusions the case will carry forward:

1. The estimate is dominated by the verification term. A change of ±1.25 minutes in `v` moves the
   verdict, and `v` was never timed in the pilot.
2. Scale alone cannot rescue the estimate. Doubling the promotion share improves the threshold by
   1.1 minutes, but the interval width is a property of the pilot's variance, not of the share.
3. Variance, not bias, is the binding constraint. Doubling `s_d` costs 2.04 minutes of margin, more
   than doubling the fixed cost costs in the same units.

**Takeaway:** the cheapest available improvement is not more cases at a larger share; it is timing
the verification step so that `v` stops being an assumption.

## 7. The Finding

The finding has three layers, and collapsing them into one sentence is how pilot reports become
false.

**Layer 1 — the arithmetic finding.** On the pilot's own numbers, net release is 4.3 reviewer-minutes
per case, or 27.9% of baseline decision touch, and the 95% lower bound is 2.35 minutes. The
pre-registered break-even for a 15% / 20-day bounded promotion is 2.44 minutes. The lower bound
misses break-even by 0.09 minutes. Therefore the honest statement is: *the point estimate clears the
threshold and the interval does not.*

**Layer 2 — the diagnostic finding.** The binding uncertainty is not sample size in the usual sense
and not the fixed cost. Doubling either leaves the verdict failing but recoverable. What moves the
verdict across the line is the verification cost `v`, and `v` is the one quantity the pilot never
measured. An estimate whose sign depends on an unmeasured term is not an estimate of the effect; it
is an estimate of the effect plus an unknown offset.

**Layer 3 — the decision finding.** A bounded promotion is worth running, but not as a test of
whether 22% is real. It is worth running because it is the cheapest way to measure `v` and the
intake-wide distribution of the paired difference, and because at 15% of intake its downside is
bounded at 3,600 cases of extra verification. The decision that follows is *promote to 15% for 20
business days with touch-time instrumentation*, with the pre-registered rule unchanged.

What the case does *not* find: that the workflow is ineffective, that the headline was misleading in
a way anyone intended, or that the promotion will pay for itself. It finds that the evidence in hand
supports a measurement, not a rollout.

**Conditional statement of support.** The conclusion holds if and only if:

- the paired differences are computed on complete pairs only, with exclusions counted;
- `v` is instrumented in the bounded arm rather than assumed from the pilot;
- the operating point does not shift by more than about 10% in intake volume, because utilisation
  enters the capacity argument non-linearly;
- the pre-registered rule is applied to the bounded arm's data as written, including the possibility
  that it returns "reject".

If any of these fails, the finding degrades from "run a bounded measurement" to "defer and instrument
the baseline first", which is a weaker but still actionable conclusion.

**Takeaway:** report the point estimate, the interval, and the threshold separately; a decision that
needs only one of them can then be stated precisely.

## 8. System/Solution Design

### 8.1 The control flow

```text
                      +--------------------+
  [case arrives] ---> | schema + cutoff    | --fail--> [quarantine + reason]
                      | validator          |                    |
                      +---------+----------+                    |
                                | pass                          |
                                v                               |
                      +--------------------+                    |
                      | bounded evaluator  | --error--> [defer] |
                      | (config hash)      |                    |
                      +---------+----------+                    |
                                | recommendation + reasons      |
                                v                               |
                      +--------------------+                    |
                      | policy gate        | --deny---> [defer] |
                      | (share, rules)     |                    |
                      +---------+----------+                    |
                                | allow                         |
                                v                               |
                      +--------------------+                    |
                      | reviewer console   | --override--> [log]|
                      +---------+----------+                    |
                                | final action                  |
                                v                               |
                      +--------------------+                    |
                      | append-only ledger | <------------------+
                      +---------+----------+
                                |
                                v
                      [audit sampler -> adjudication]
```

### 8.2 Ownership of state, and why

Four pieces of state exist and each has exactly one owner. Duplicated ownership is the most common
source of the failures in Section 12, so the assignment is explicit:

| State | Owner | Reason |
|---|---|---|
| Idempotency key and final action | decision ledger | only the ledger knows whether an effect happened |
| Recommendation and its reason codes | evaluator run record | replayable from configuration hash and inputs |
| Queue position and in-flight attempt | work queue | admission control must be enforceable in one place |
| Override rationale | reviewer console record | the human reason must outlive the console session |

The evaluator owns no durable authority. It cannot approve, cannot release, and cannot enqueue its
own retries beyond the queue's budget. This is not a security posture; it is what makes the "score is
not authority" failure detectable, because the ledger's actor field can only ever contain a reviewer
identity on a final action.

### 8.3 The policy gate

The gate is where a share limit becomes enforceable rather than advisory. It evaluates, in order:
does the case fall inside the promoted share; is the configuration hash on the approved list; has
the promotion end date passed; is the reviewer eligible for the variant. Any "no" routes to defer
with a specific reason code, never to a silent fall-through to the evaluator's recommendation.

Share allocation must be deterministic per case to avoid a selection bias that would poison the
comparison. The intended mechanism is a stable hash of `case_id` modulo 100 compared against the
share in percent, so assignment is balanced across the window and reproducible from the ledger
alone. A random assignment drawn at request time would also be unbiased but would not be
reconstructible after the fact.

### 8.4 Instrumentation, which is the deliverable

Because the finding is "measure `v`", the instrumentation is not supporting work; it *is* the
artefact being built. Four timestamps are required per promoted case, and each corresponds to a
question the analysis asks:

- `prefill_open_at` and `prefill_close_at` bound the verification review.
- `first_escalation_at` marks the moment a reviewer leaves the recommendation path.
- `exception_close_at` bounds exception handling and is present only when an exception occurred.
- `final_action_at` ends the case and must be written by the ledger, not the console.

The difference `prefill_close_at − prefill_open_at` aggregated over promoted cases gives `v`
directly. If a case has no `prefill_open_at`, the case did not receive verification and must be
excluded from the `v` estimate with a count, not silently treated as zero.

### 8.5 Rollback design

Rollback is a property that must be testable before it is needed. The rollback path is: set the
promoted share to zero in the policy gate, which is a configuration change requiring no deploy;
retain the ledger for audit; keep the manual path warm for the duration of the window. The trigger
conditions are stated in advance: exception rate above 12% of promoted cases, any duplicate final
action, or `v` above 6 minutes on the running estimate.

The third trigger is the interesting one because it is a *measurement* trigger rather than a harm
trigger. Rolling back when the measured cost invalidates the economics is cheaper than finishing the
window and then discovering it. Pre-registering that trigger is what allows the programme to stop
early without the stop being read as a failed experiment.

**Takeaway:** design the measurement first, because in this decision the measurement is the product.

## 9. Evaluation & Validation

### 9.1 Metric definitions

| Metric | Numerator | Denominator | Population | Window | Comparator | Guardrail |
|---|---|---|---|---|---|---|
| Net release per case | Σ(T_dec,0 − T_dec,1) − Σv | complete pairs | promoted cases with both timestamps | 20 business days | manual control share | exclusions counted, none dropped |
| Verification minutes `v` | Σ(prefill + exception minutes) | promoted cases with verification | promoted cases | 20 business days | pilot assumption of 2.5 min | p95 reported alongside mean |
| Exception rate | cases with any exception code | promoted cases | promoted cases | 20 business days | manual control share | rollback above 12% |
| Duplicate final actions | decision keys with >1 final action | all decision keys | all intake | 20 business days | must be 0 | hard stop on any occurrence |
| Audit agreement | adjudicated-correct recommendations | sampled promoted cases | 6% random sample | 20 business days | reviewer final action | disagreement reviewed weekly |
| Queue wait | Σ wait minutes | completed cases | all intake | 20 business days | pre-window baseline | no target; direction only |

Every metric names its denominator because a rate without one is a number without meaning. The
duplicate-action metric is the only one with a zero target, and that is deliberate: it is a binary
invariant, not a performance measure, so an average has no interpretation.

### 9.2 Uncertainty handling

Three sources of uncertainty are reported, not averaged away:

- **Sampling uncertainty** on the paired difference, via the interval in F-4. It is the only one the
  pre-registered rule references, so it must be computed exactly as written.
- **Model uncertainty** on the queue correction. F-5's magnitude was shown wrong by a factor of 4.8,
  so queue effects are reported as an elasticity `1 + 1/(1-rho)` with the utilisation attached — 2.59
  to 3.06 across the two operating points — and never as a predicted waiting time. Even the
  elasticity is disclosed as a local linearisation: at a 27.9% service-time change it predicts an 85%
  wait reduction where the exact F-5 answer is 59.9%.
- **Population uncertainty** from selection. The pilot's eligibility rule is reviewed, and if the
  promoted arm's case mix differs by more than 10 percentage points on any dominant category, the
  comparison is reported per stratum as well as pooled.

### 9.3 The validation protocol, with executed status

| Check | Method | Pass condition | Status |
|---|---|---|---|
| Arithmetic reproduction | recompute F-1 to F-4 from the pilot ledger | matches to 0.01 min | executed on illustrative inputs |
| Paired-difference distribution | inspect histogram, skew, outliers | no single pair drives the mean | not-executed |
| Timestamp ordering | assert every `_at` ordering per case | zero violations | not-executed |
| Duplicate replay | submit the same `decision_key` twice | exactly one final action | not-executed |
| Leakage audit | trace every decision-time field to its write time | no field written after cut-off | not-executed |
| Fairness by case category | net release per dominant category | no category with negative release in the interval | not-executed |
| Rollback drill | set share to zero mid-window in a test environment | manual path absorbs intake without backlog growth | not-executed |
| Oracle comparison | 100 hand-built cases against ORC-01 | zero disagreements on the stated contract | not-executed |

The distinction between "executed on illustrative inputs" and "not-executed" is the whole point of
the column. The arithmetic was checked here; no case data was processed, no system was run, and no
reviewer was observed. A reader who treats the first row as evidence that the workflow works has
misread the table.

### 9.4 Why the protocol cannot prove the claim it is often asked to prove

The protocol can establish that net release exceeds the threshold in the promoted population, with
a stated interval. It cannot establish that the workflow *causes* better decisions, that downstream
outcomes improve, or that the effect persists after reviewers adapt to the tool. Those are three
different claims requiring three different designs: adjudicated outcome comparison, a longer window,
and a post-adaptation re-measurement.

Reviewer adaptation is worth stating plainly because it usually goes in the optimistic direction
first and the pessimistic direction second. Novelty produces a short-term speed-up; familiarity then
either consolidates the gain or reveals that the reviewer was leaning on the recommendation and now
has to unlearn it. A 20-day window is long enough to see the first effect and too short to see the
second.

**Takeaway:** state what the validation establishes, and state separately what it cannot.

## 10. Action Taken

No production change was made and no system was operated. The actions below are the decision the
operations lead takes *in the case*, all of them bounded and reversible, plus the actions a real
programme would take before any of them.

**Decision.** Promote the assisted workflow to 15% of intake for 20 business days under the
pre-registered rule, with touch-time instrumentation as a mandatory deliverable. Do not promote to
full intake. Do not reject, because rejection would discard the information the bounded arm can buy
at a known price.

**The nine actions, in dependency order:**

1. Freeze the analysis plan, including the break-even formula, the threshold value 2.44 minutes, and
   the rollback triggers. Signed by the operations lead before any promoted case is processed.
2. Instrument the four timestamps in the reviewer console, with the ledger writing `final_action_at`.
   Ship this before the workflow, since a promotion without instrumentation has no value.
3. Implement deterministic share assignment by hashing `case_id` modulo 100.
4. Stand up the audit sampler at 6% and define the adjudication question exactly: was the reviewer's
   final action defensible given the evidence at cut-off.
5. Run the duplicate-replay check and the timestamp-ordering check against a synthetic batch.
6. Run the rollback drill in the test environment and record the time to absorb intake manually.
7. Start the window on a Monday so the 20 business days contain four complete weeks.
8. Review the running `v` estimate weekly against the 6-minute rollback trigger.
9. Apply the rule to the full window's data as written, then write the decision record.

**What was explicitly declined.** Three attractive actions were declined and each reason is
recorded so a later reviewer does not re-propose them:

- *Extending the pilot to 60 cases before deciding.* This would narrow the interval on a quantity
  whose estimate already depends on an unmeasured term. More cases improve precision on `d_bar`; they
  do not measure `v`.
- *Promoting to 100% with a rollback trigger.* The rollback trigger protects against harm, not
  against cost. At full intake every case pays verification, so a wrong bet is paid 6.67 times per
  day more than at 15%.
- *Replacing the paired analysis with a before-after comparison.* Before-after across the whole
  intake would confound the workflow with weekday mix, seasonal intake, and staff seniority.

**Takeaway:** the action list is mostly measurement work, and that is the correct shape for a
decision that is currently evidence-limited rather than idea-limited.

## 11. Outcome & Results

**Result status: not-measured.** No case was processed, no reviewer was timed, and no window was
run. The only executed work in this manuscript is the arithmetic in Section 6, which is VERIFIED for
its stated ILLUSTRATIVE inputs.

What the case can report with confidence, because it is arithmetic rather than observation:

| Quantity | Value | Basis | Status |
|---|---|---|---|
| Elapsed reduction, pilot | 5.5 min/case (22.0%) | 25.0 → 19.5, illustrative | arithmetic verified |
| Net touch release, pilot | 4.3 min/case (27.9%) | F-1, illustrative | arithmetic verified |
| Full-intake capacity release | 86.0 reviewer-hours/day | F-2, illustrative | arithmetic verified |
| Full-intake daily value | 6,192 USD/day | F-2, illustrative | arithmetic verified |
| Break-even for bounded arm | 2.44 min/case | F-3, illustrative | arithmetic verified |
| 95% lower bound, pilot | 2.35 min/case | F-4, illustrative | arithmetic verified |
| Lower bound at 3,600 cases | 4.10 min/case | F-4 with n = 3,600 | arithmetic verified |
| Exception rate under promotion | unknown | not measured | not-measured |
| Downstream outcome change | unknown | not measured | not-measured |

**What a future report must contain.** For any claim of realised improvement, the report must state
the numerator, denominator, population, window, comparator, uncertainty interval, excluded-record
count, and the configuration hash of the evaluator. A report that omits the exclusion count has an
unknown denominator; one that omits the comparator has no counterfactual; one that omits the
interval has no way to distinguish an effect from noise.

**What would count as failure, stated in advance.** The bounded promotion fails if the lower bound
on net release is below 2.44 minutes at the end of the window, or if the exception rate exceeds 12%,
or if any duplicate final action occurs. Stating failure conditions before the data exists is the
only defence against the post-hoc search for a favourable subgroup.

**The one result that is genuinely informative even without a window.** The pilot's standard
deviation of 6.1 minutes on a mean of 4.3 is a coefficient of variation of 1.42. That ratio, not the
mean, determines how many cases a future measurement needs, and it is the single most transferable
number the pilot produced.

**Takeaway:** separating arithmetic-verified from not-measured is what keeps a teaching case from
becoming a fabricated deployment story.

## 12. Failure Modes & What Went Wrong

Six distinct mechanisms can each produce the same surface symptom — "the promotion did not deliver
the expected saving" — and they require different corrections. Each entry names the mechanism, the
competing explanations, the check that discriminates, and the correction.

### 12.1 Verification cost omitted from the benefit

**Mechanism.** The benefit is computed from elapsed or decision time while the newly created
verification work is never charged. The estimate is then inflated by exactly the verification term.

**Competing causes for the same symptom:** (a) omitted verification cost, (b) case-mix shift making
promoted cases easier, (c) a genuine workflow effect that is smaller than reported.

**Discriminating check:** time 20 consecutive pre-fill reviews with a stopwatch and compare the mean
against the `v` used in the analysis. If the measured mean exceeds the assumed `v` by more than the
observed margin (here 0.09 minutes), (a) is confirmed. If `v` matches but the case-mix distribution
shifts toward easier cases, (b) is confirmed.

**Correction:** recompute F-1 with measured `v`; restate the threshold; reapply the pre-registered
rule. **Prevention:** make `v` a required field in the ledger before the first promoted case, so it
cannot be defaulted.

### 12.2 Selection into the pilot population

**Mechanism.** Reviewers judged eligibility, and reviewers prefer cases they expect to handle well.
The pilot mean difference is then a difference on a favourable subpopulation, and the intake-wide
expectation regresses toward zero.

**Competing causes:** (a) selection by eligibility judgement, (b) genuine heterogeneity where the
workflow helps only certain case types, (c) a time trend that made the pilot weeks easier.

**Discriminating check:** compute the paired difference within each dominant case category and
compare the pooled estimate against the eligibility-weighted estimate using intake proportions. If
the weighted estimate is materially lower, (a) holds; if one category carries the effect and the
others show zero, (b) holds; if the control share also improved over the same weeks, (c) holds.

**Correction:** report per-category results and a pooled estimate reweighted to intake proportions.
**Prevention:** assign eligibility mechanically from the intake schema, not by reviewer discretion,
and record the exclusion count by reason.

### 12.3 Idempotency failure producing duplicate actions

**Mechanism.** A timeout on the ledger write causes the caller to retry; the retry carries a
regenerated key, so the ledger sees two distinct decisions and records two final actions. The
symptom is not latency but a case disposed twice, which corrupts both the case history and the
duration metric.

**Competing causes:** (a) regenerated key on retry, (b) ledger write succeeding but the
acknowledgement being lost, (c) two reviewers legitimately acting on the same case.

**Discriminating check:** group the ledger by `case_id` and count final actions. Then replay one
decision with a fixed key twice and observe the count. If the replay produces two rows, (a) holds;
if the original key appears once but the caller logged a timeout, (b) holds; if the actor fields
differ, (c) holds and the defect is in case assignment, not in the ledger.

**Correction:** derive the key from `case_id` and the decision attempt number, never from a
timestamp; make the ledger write the key's uniqueness constraint. **Prevention:** the
duplicate-action metric is a hard stop at any non-zero value, which forces the correction rather than
allowing it to be averaged into a rate.

### 12.4 Clock and timezone inconsistency across sources

**Mechanism.** The console writes local time without an offset, the ledger writes UTC, and the
extract interprets the console value as UTC. Cases near midnight move by hours, and the
`closed_at − opened_at` duration can become negative or absurdly large.

**Competing causes:** (a) missing offset on one source, (b) daylight-saving transition inside the
window, (c) genuine multi-day cases.

**Discriminating check:** count rows with negative durations and rows with durations above 24 hours,
then histogram the offsets. A cluster of durations near a whole number of hours points to (a) or (b);
an even spread points to (c).

**Correction:** reject rows without an explicit offset rather than assuming one; requarantine and
count. **Prevention:** schema-level requirement for offset-bearing timestamps, enforced at ingest,
with a test that a UTC-minus-5 record and a UTC record of the same instant produce equal durations.

### 12.5 Utilisation-dependent capacity double-counting

**Mechanism.** The saved minutes are booked as capacity at face value and simultaneously assumed to
reduce waiting by the queue formula. The same minute is counted twice: once as released service time
and once as avoided wait, and F-5's magnitude error makes the second count look large.

**Competing causes:** (a) double counting, (b) a real queue effect at high utilisation, (c) the
intake genuinely absorbing released capacity.

**Discriminating check:** compute utilisation before and after, then check whether the reported
benefit equals `N_d × g_net` (service-time count) or something larger. If the reported value exceeds
the service-time count while no wait reduction was measured, (a) holds.

**Correction:** report released service time and wait change as two separate quantities with
separate evidence; never sum them into one "value released". **Prevention:** the decision record
template carries distinct fields for service-time saving, wait change, and reassignment, and at most
one of them may be monetised.

### 12.6 Reviewer adaptation reversing the effect

**Mechanism.** The first weeks show a speed-up from novelty and from reviewers trusting the
recommendation. Later, reviewers begin verifying the recommendation more thoroughly because they
have been burned by a wrong one, and touch time returns toward baseline while verification cost
stays.

**Competing causes:** (a) adaptation driven by observed errors, (b) an initial estimate from an
unrepresentative favourable period, (c) a workflow change during the window.

**Discriminating check:** plot weekly mean `v` and weekly mean decision touch time across the 20
days. A rising `v` with flat or rising decision touch points to (a). A step change in either series
with no corresponding error rate change points to (c).

**Correction:** extend the window or re-measure after adaptation stabilises; report the trend rather
than the window mean. **Prevention:** define the primary metric over the last 10 business days as
well as the full window, and report both, so a decaying effect cannot hide behind an average.

**Takeaway:** six mechanisms, six different checks — the symptom alone never identifies the cause.

## 13. Transferable Lesson

**The lesson is a transfer function, not a percentage.** Any claim expressed in one unit (elapsed
time) must be converted into the unit the operation pays for (net touch time) before it can justify
a decision, and the conversion itself is where most of the error enters.

Five rules transfer to any evidence-to-operation handoff:

1. **Name the scarce resource.** Every operational decision is ultimately about a constrained input.
   If the analysis does not name it, the benefit will be measured in whatever was convenient and will
   not be spendable.
2. **Charge the new work to the new workflow.** A change that creates verification, exception
   handling, or audit work has that work subtracted from its benefit, not added to someone else's
   budget.
3. **Separate detectability from sufficiency.** A statistically clear effect can still be
   economically insufficient, and a 40-case pilot can produce either answer. Pre-register the
   threshold so the answer is not chosen after the fact.
4. **Price the option to learn.** When the binding uncertainty is one unmeasured quantity, a bounded
   promotion is cheaper than more analysis of the same data, but only if the promotion instruments
   the missing quantity.
5. **Bound the downside before you need it.** A share limit, an end date, and a measurement-based
   rollback trigger convert an unbounded bet into a priced one.

**Where the lesson does not transfer.** The specific numbers — 4.3 minutes, 2.44 minutes, 86 hours —
are properties of this constructed intake. They do not survive a change in case duration, roster,
cost rate, or variance. The *structure* survives; the constants do not.

A second, subtler transfer concerns the shape of the deliverable. The artefact that made the decision
possible was not a better model; it was four timestamps. In domains where the measurement is missing,
engineering effort directed at the measurement typically buys more decision quality per hour than
effort directed at the method.

**Takeaway:** convert units, charge new work, pre-register the threshold, price the option to learn,
and bound the downside — in that order.

## 14. Connections & Cross-References

**Within this unit.** The measurement discipline here is the same one the DSA module applies to
complexity claims: a bound is meaningless without its input-size definition. The systems module
supplies the queueing vocabulary used in F-5 and the idempotency argument used in 12.3. The
statistics module supplies the interval construction and the distinction between a point estimate and
a decision rule.

**Downstream.** Every later case that reports a pilot result inherits this transfer function. A case
claiming an accuracy improvement, a latency improvement, or a cost reduction should be asked the same
four questions: what unit, what population, what new work did the change create, and what threshold
was fixed in advance.

**Upstream dependencies.** Nothing in this case requires a model. That is deliberate: the hardest
part of moving evidence into operation is usually measurement design, not modelling, and a case that
needed a model to be interesting would mislead a reader about where the difficulty lies.

### Interview pack

#### Q1. Your pilot shows a 22% fall in elapsed time. What is the first number you recompute, and why?
**Answer:** The net touch-time release, `g_net = (T_dec,0 − T_dec,1) − v`, because elapsed time
contains queue waiting, which the roster cannot sell back. Here the elapsed fall is 5.5 minutes while
the net touch release is 4.3 minutes; the 1.2-minute wait component and the 2.5-minute verification
charge together explain the gap.
**Reasoning:** the operation's scarce input is reviewer-minutes, so only reviewer-minutes released
convert into capacity. Any other unit produces a number that cannot be spent.
**Follow-up:** if `v` were 5.0 minutes instead of 2.5, what happens to the estimate and the verdict?
**Red flag:** accepting 22% as the benefit, or saying the difference is "just noise" without
computing the interval.
**Scoring:** full credit names the formula, the two excluded components, and the resulting 4.3.

#### Q2. Compute the full-intake daily capacity release and its money value.
**Answer:** `C_rel = 1,200 × 4.3 / 60 = 86.0` reviewer-hours per day; at 72 USD/hour that is
`86.0 × 72 = 6,192` USD per day, ILLUSTRATIVE.
**Reasoning:** the conversion has two steps — minutes to hours, hours to money — and each carries its
own assumption. The roster must be constant and the released minutes must be usable.
**Follow-up:** what changes if the roster is fixed and the backlog is not?
**Red flag:** reporting 6,192 as a saving rather than a capacity option, or forgetting the divide by
60.
**Scoring:** full credit produces 86.0 hours, 6,192 dollars, and states the condition under which it
becomes cash.

#### Q3. The pilot's lower bound is 2.35 and break-even is 2.44. Is this a close call?
**Answer:** No — it is a sample-size diagnosis, not a knife-edge. The 0.09-minute gap is about five
seconds per case, which cannot be a meaningful decision margin. What the gap reveals is that the
pilot's variance (s_d = 6.1 on a mean of 4.3) is too large relative to the effect for 40 cases to
resolve.
**Reasoning:** a decision rule that resolves to seconds is a rule whose precision is mismatched to
its resolution. The correct response is to change the design, not to argue about the sign.
**Follow-up:** how many cases would the rule need at the pilot's variance?
**Red flag:** recommending "collect 3 more cases" or reinterpreting the threshold after seeing the
data.
**Scoring:** full credit gives the gap, identifies variance as binding, and proposes a design change.

#### Q4. Derive the required sample size for the pre-registered rule.
**Answer:** solve `t × s_d/√n ≤ d_bar − g*`, i.e. `2.0227 × 6.1/√n ≤ 1.8609`, giving
`√n ≥ 6.632` and `n ≥ 44`. At n = 44 the lower bound is `4.3 − 1.8546 = 2.4454`, which clears 2.4391
by 0.0063 minutes.
**Reasoning:** the critical value depends on n, so the honest solution iterates; using the n = 40
value as a fixed constant is an approximation that understates the requirement slightly.
**Follow-up:** is a 0.006-minute margin acceptable? What would you do instead?
**Red flag:** answering 40 or 45 without showing the inequality, or ignoring the dependence of `t`
on `n`.
**Scoring:** full credit shows the inequality, the iteration caveat, and rejects the resulting margin
as unusable.

#### Q5. Why is the bounded promotion at 15% preferable to extending the pilot?
**Answer:** because the missing quantity is `v`, and no amount of re-analysis of the pilot measures
it. The promotion at 15% instruments `v` on 3,600 cases over 20 days at a bounded downside, whereas
extending the pilot to 60 cases tightens an interval on a quantity that already depends on an
unmeasured offset.
**Reasoning:** information value per unit of downside is the right comparison. The promotion buys the
measurement that matters; more pilot cases buy precision on the wrong parameter.
**Follow-up:** what must be true for the promotion to be worth its fixed cost of 9,500?
**Red flag:** "more data is always better", or treating the promotion as a commitment to rollout.
**Scoring:** full credit identifies `v` as the missing quantity and states the bounded downside.

#### Q6. The queueing model predicts a 60% wait reduction; the pilot saw 12.5%. What do you conclude?
**Answer:** the model's level is wrong by roughly a factor of 4.8 (16.24 predicted versus 9.6
observed baseline wait), most likely because arrivals are smoother than Poisson or because the
service pool is many parallel servers rather than one aggregated server. The direction survives; the
magnitude does not.
**Reasoning:** the correction factor `rho/(1-rho)` is extremely sensitive to utilisation and to the
squared coefficients of variation, neither of which was measured. At `C_a^2 = C_s^2 = 0.2` the
baseline wait falls to 3.25 minutes instead of 16.24.
**Follow-up:** what quantity from F-5 is safe to report to the operations lead?
**Red flag:** quoting the predicted minutes, or discarding the model entirely when its elasticity is
still useful.
**Scoring:** full credit states the factor, names the two candidate mechanisms, and salvages the
elasticity (2.59 at `rho = 0.37`, 3.06 at `rho = 0.51`).

#### Q7. What is the elasticity interpretation of the queue formula, and how do you derive it?
**Answer:** because `W_q` scales as `rho/(1-rho) × T_s` and `rho` is proportional to `T_s` at fixed
arrival rate, the elasticity is `d ln W_q / d ln T_s = 1 + 1/(1-rho)`, not `1/(1-rho)`. The leading 1
is the direct term from the `T_s` factor; the `1/(1-rho)` term is the utilisation feedback. At
`rho = 0.5133` that is 3.055; at `rho = 0.37` it is 2.587.
**Reasoning:** the elasticity is free of the `C_a^2` and `C_s^2` terms because they multiply `W_q`
rather than interacting with `T_s`, so it is robust where the level is not — but it is a local
linearisation only.
**Follow-up:** what happens to the elasticity at `rho = 0.9`, and why does applying it to a 27.9%
change give the wrong answer?
**Red flag:** quoting the multiplier at `rho → 1` as achievable, dropping the leading 1, or scaling a
large change by a derivative.
**Scoring:** full credit gives the derivation, both values, and the linearisation caveat.

#### Q8. Charge the audit sampling correctly at full intake versus at 15%.
**Answer:** at 15% for 20 days, 6% of 3,600 is 216 cases at 4 minutes = 864 minutes = 14.4 hours, or
1,036.80 USD. At full intake over the same window, 6% of 24,000 is 1,440 cases at 4 minutes = 5,760
minutes = 96 hours, or 6,912 USD.
**Reasoning:** audit scales with promoted volume, not with the share, so it does not dilute as the
promotion grows — unlike the fixed cost term, which does.
**Follow-up:** which term in `g*` is scale-invariant and which is not?
**Red flag:** treating audit as a fixed cost, or setting the sampling rate to zero in the economics.
**Scoring:** full credit gives both figures and identifies the audit term as the non-diluting one.

#### Q9. A case is disposed twice in the ledger. Walk through your diagnosis.
**Answer:** count final actions per `case_id`. If two rows carry different keys, the key was
regenerated on retry (mechanism 12.3a). If one key appears once but the caller logged a timeout, the
write succeeded and the acknowledgement was lost (12.3b). If the actor fields differ, two reviewers
acted and the defect is in assignment (12.3c).
**Reasoning:** each mechanism leaves a different fingerprint in the key and actor fields, so one
group-by distinguishes them without touching the code.
**Follow-up:** what is the permanent fix, and why is averaging this into a rate unacceptable?
**Red flag:** "retry the case", or monitoring the rate instead of enforcing zero.
**Scoring:** full credit names the three fingerprints and the key-derivation fix.

#### Q10. Which field in the decision-time record is the most dangerous if it is written late?
**Answer:** the final disposition or outcome field, because it is the most predictive field available
and therefore the one most likely to be joined in by an analyst. It is written after the decision, so
using it converts the exercise into hindsight.
**Reasoning:** leakage is not a modelling defect; it is a record-keeping defect that a good modeller
will exploit because the field genuinely predicts well.
**Follow-up:** give the concrete control that prevents it.
**Expected answer extension:** the control is a point-in-time join keyed on write time, plus a test that
asserts every decision-time field's write timestamp precedes the cutoff.
**Follow-up (second):** what does a leakage test look like when the field is legitimately available?
**Red flag:** naming a field that is merely noisy rather than unavailable, or relying on reviewer
discipline alone.
**Scoring:** full credit names the late-written outcome field and prescribes a timestamp assertion.

#### Q11. Compute the utilisation before and after, and explain why the difference matters.
**Answer:** baseline demand is `1,200 × 15.4 = 18,480` minutes against `60 × 600 = 36,000` available
seat-minutes, giving `rho = 0.5133`. Assisted demand is `1,200 × 11.1 = 13,320`, giving
`rho = 0.3700`.
**Reasoning:** utilisation is what makes any wait statement meaningful, and it also determines
whether released capacity is usable: at 0.37 the roster has slack, so released minutes may convert to
idle time rather than to queue reduction.
**Follow-up:** at what intake volume would the assisted workflow reach the baseline utilisation?
**Red flag:** computing `rho` as minutes saved over minutes available, or using 480-minute days when
the window is 10 hours.
**Scoring:** full credit gives both values and names the slack consequence.

#### Q12. At what volume does the assisted workflow consume as much capacity as the baseline does today?
**Answer:** set `N × 11.1 = 18,480`, giving `N = 1,664.9` cases per day, so about 1,665 — a 38.7%
increase over 1,200. That is the headroom the workflow creates in throughput terms.
**Reasoning:** headroom is defined by holding the baseline demand fixed and solving for the volume at
which the new service time reproduces it, which is the question capacity planners actually ask.
**Follow-up:** does the answer change if `v` doubles to 5.0?
**Red flag:** comparing percentage improvements by subtracting them, or forgetting that `v` is inside
the 11.1 figure.
**Scoring:** full credit gives 1,665 cases, the 38.7% figure, and notes the dependence on `v`.

#### Q13. A reviewer overrides a recommendation and the override is not logged. What breaks?
**Answer:** the audit trail loses the actor's rationale, so two later analyses are corrupted: the
agreement metric undercounts disagreement, and the net-release estimate attributes the reviewer's
extra minutes to the workflow's decision time without explanation.
**Reasoning:** an override is evidence about where the recommendation was wrong, so losing it loses
the only cheap error signal the programme has.
**Follow-up:** what field makes override analysis possible, and who must write it?
**Red flag:** logging the override action without its reason, or letting the console session hold the
only copy.
**Scoring:** full credit names both corrupted analyses and requires the reason to be persisted at
write time.

#### Q14. Design the rollback trigger set and justify each one.
**Answer:** exception rate above 12% of promoted cases (harm), any duplicate final action (invariant
breach, hard stop), and running `v` above 6 minutes (economic invalidation). The third is the unusual
one: it stops the window when the measurement itself shows the economics cannot work.
**Reasoning:** triggers should cover three risk types — harm to the process, breach of an invariant,
and invalidation of the decision's premise. A trigger set covering only harm will keep running a
programme that has already been shown uneconomic.
**Follow-up:** why must the share be zeroable by configuration rather than by deploy?
**Red flag:** a trigger set that only watches latency, or one written after the window starts.
**Scoring:** full credit covers all three risk types and explains the configuration-level rollback.

#### Q15. Why allocate the 15% share by hashing `case_id` rather than at request time?
**Answer:** determinism. A stable hash makes assignment reproducible from the ledger alone, so a
later reviewer can reconstruct which cases were promoted without depending on a log that may have
rotated. A request-time random draw is unbiased but not reconstructible.
**Reasoning:** reproducibility is a property of the assignment mechanism, not of the statistics. An
unreconstructible arm is a population of unknown membership.
**Follow-up:** what bias could a hash introduce if `case_id` encodes arrival order?
**Red flag:** "random is fine" without addressing reconstructibility, or ignoring structure in the
identifier.
**Scoring:** full credit names reconstructibility and raises the identifier-structure risk.

#### Q16. The window runs and `L = 2.10`, below break-even. What do you report?
**Answer:** report the pre-registered result as a failure of the sufficiency test, alongside the
measured `v`, the exception rate, and the weekly trend. Then report the one genuinely new
information: the measured `v` and the arm's variance, which together determine the sample a future
design would need.
**Reasoning:** the window was bought to measure `v`, not to validate the 22% claim, so a failing
sufficiency test still returns the information it was designed to buy. Conflating the two turns a
successful measurement into a reported failure.
**Follow-up:** what would make you recommend a second bounded window rather than rejection?
**Red flag:** re-slicing the data for a favourable subgroup, or declaring the workflow ineffective
without separating the two questions.
**Scoring:** full credit keeps the two questions separate and uses the measured variance.

## 15. Quick Reference Card

| Field | Decision card |
|---|---|
| Decision | promote to 15% of intake for 20 business days, with touch-time instrumentation |
| Unit of benefit | net reviewer-minutes released per case, not elapsed minutes |
| Formula | `g_net = (T_dec,0 − T_dec,1) − v`; threshold via `g* = fixed/(s·N_d·D) + a·t_a` |
| Worked values | 4.3 min/case release; 2.44 min/case break-even; lower bound 2.35 (ILLUSTRATIVE) |
| Binding uncertainty | verification minutes `v`, never timed in the pilot |
| Rule | promote only if the 95% lower bound exceeds break-even, fixed before measurement |
| Boundary | queue formula invalid near `rho = 1`; magnitude off by 4.8x here; elasticity `1+1/(1-rho)` = 2.59–3.06, local only |
| Invariant | exactly one final action per decision key; any duplicate is a hard stop |
| Rollback | share to zero by configuration; triggers at 12% exception, 6-min `v`, any duplicate |
| Evidence | arithmetic verified on illustrative inputs; no system operated; all outcomes not-measured |
