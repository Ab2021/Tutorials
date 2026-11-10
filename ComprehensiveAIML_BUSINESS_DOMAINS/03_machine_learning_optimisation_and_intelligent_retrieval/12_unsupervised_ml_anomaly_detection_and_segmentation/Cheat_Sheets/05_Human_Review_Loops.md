---
collection: "unsupervised-ml-anomaly-detection-and-segmentation"
file_id: "05"
title: "Human Review Loops"
topic: "Human review loops: capacity, adjudication, audit sampling and the measurement they buy"
collection_section: "Part 03 — Machine learning optimisation and intelligent retrieval"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
status: "draft"
version: "2.0"
notation_style: "plain-text, ASCII formulas inside fenced blocks"
review_trigger: "Change in review capacity, service time, escalation rate, audit budget, confirmation rule, or the delayed-label definition."
source_status: "OPEN"
owner_role: "Risk analytics reviewer"
source_index: "INDEX.md"
---

# Human Review Loops

> **Evidence:** every identity in this sheet is `DERIVED` from its definition and every printed number was recomputed in a fenced block and independently re-checked in Section 5.9. All counts, hours, rates, costs and outcome values are `ILLUSTRATIVE` teaching inputs built to expose a mechanism; they are not observations from any organisation. Mechanisms attributed to the review-workflow and sampling literature are `SOURCED` and listed in Section 10. Anything about a real deployment, vendor queue product or measured staff workload is `OPEN`.

## 1. File Header

**Purpose.** Turn a review queue into a measurement instrument, and decide how much reviewer time to spend reviewing alerts versus spending it on the audit sample that tells you what the review queue missed. The sheet is about the loop, not the detector: it assumes a score already exists and asks who looks at what, in which order, at what cost, and what the resulting numbers are allowed to mean.

**Prerequisites.** You should be able to compute a proportion and its standard error, read a contingency table, apply the delta method to a ratio of two random quantities, and recognise a finite-population correction. You should also be comfortable with the idea that an hour of reviewer time is a budget line with an alternative use. If you have ever defended a precision figure that a stakeholder then used to argue for headcount, this sheet is aimed at that conversation.

**Key outcome.** After working through the sheet you can (a) convert a headcount into an effective investigation capacity after escalation rework, and show why the nominal figure is usually 20–35% optimistic; (b) estimate the total number of true cases in a week from a queue census plus a stratified audit, with a variance that attributes uncertainty to the right stratum; (c) compute recall and coverage as ratio estimators and state an interval that is honest about a census stratum contributing zero variance; (d) compute the break-even confirmation rate that justifies spending an hour on a queue row versus an audit row; (e) choose an audit allocation that beats proportional allocation at the same cost; (f) recognise the four ways a review loop silently stops measuring — pending counted as negative, capacity quoted in alerts, audit cancelled under load, and confirmation rule noise.

**Not covered.** This sheet does not teach anomaly scoring (sheet 01), clustering (sheet 02), change-point detection for the underlying rate (sheet 03), or segment evaluation (sheet 04). It does not decide employment, disciplinary or legal questions, and it does not assert any measured result from a production review operation. It also does not treat reviewer judgement as ground truth without saying so explicitly: every formula below that uses "confirmed" is conditional on the confirmation rule, and Section 8 shows what happens when the rule changes.

**Level.** Core, with the sampling and economics material in Sections 4.3, 5 and 9 at practitioner depth.

**How to read the evidence tags.** `DERIVED` means the result follows from the stated definition and can be re-derived on paper. `ILLUSTRATIVE` means the input numbers were chosen to make a mechanism visible. `SOURCED` means a named published result is paraphrased with a reference in Section 10. `OPEN` means nobody has checked it here and it must not be used as a premise.

**Retrieval cue.** If someone says "our reviewers confirmed 154 cases this week, so precision is 22%", jump to Section 4.1 for the denominator question, then to Section 8 failure HR-F01 for the pending-row version of the same error.

**Relation to sibling sheets.** Sheet 01 derives the triage policy: which alerts reach a human at all. This sheet starts where that policy ends, at the queue, and adds the second sampling frame — the audit of rows the policy excluded — which is the only thing that makes recall identifiable. Sheet 04's decision-value arithmetic reappears here as the per-row net value comparison in Section 9.

## 2. Concept Map / Mental Model

### 2.1 Four objects, four owners

A review loop contains four distinguishable objects. Nearly every confused argument about review loops is two people discussing different ones.

```text
OBJECT            WHAT IT IS                       OWNER            FAILS BY
---------------   ------------------------------   --------------   --------------------------
queue             ranked alerts awaiting a human  operations       unbounded growth, starvation
adjudication      a decided row plus its reason    reviewers        rule drift, inconsistency
audit frame       a sample of rows nobody looked   risk analytics   cancellation under load
                  at, drawn independently of score
policy            capacity, ordering, cut, rule    decision owner   being set by the model team
```

The queue is a workload. The adjudication is evidence. The audit frame is the only object that can see outside the queue. The policy is a capacity statement in disguise. A team that owns the model almost never owns the capacity input, which is why thresholds set by modellers drift into fiction.

### 2.2 The loop as a control system

The review loop is a feedback controller with a long, noisy, biased sensor.

```text
      detector score a(x)
              |
              v
     +------------------+   cut at capacity C   +------------------+
     | queue (top-k)    |---------------------->| reviewer decides |
     +------------------+                       +------------------+
              |                                          |
              | excludes                                 | writes
              v                                          v
     +------------------+                       +------------------+
     | unreviewed rows  |                       | adjudication log |
     | (audit frame)    |                       | reason, verdict  |
     +------------------+                       +------------------+
              |                                          |
              | sample n_h per stratum                   | aggregated
              v                                          v
     +-------------------------------------------------------------+
     |  estimator: total cases, recall, coverage, precision         |
     +-------------------------------------------------------------+
              |
              | informs (slowly, noisily)
              v
     policy: capacity, ordering, cut, confirmation rule, refit
```

Two properties of the loop drive everything else. First, the path from the queue to the policy is short and biased, because the queue is selected by the score. Second, the path from the audit frame to the policy is long, expensive and the only unbiased one. A loop with no audit frame is a controller whose sensor only sees the part of the plant it already controls.

### 2.3 The two denominators people mean by "precision"

```text
NAME                     NUMERATOR            DENOMINATOR                 WHAT IT ANSWERS
----------------------   ------------------   -------------------------   ---------------------------------
queue precision          confirmed            adjudicated rows            "of what a reviewer decided,
                                                                          how much was real?"
alert precision          confirmed+audited    all alerts raised           "of everything the detector
                                                                          flagged, how much was real?"
population prevalence    confirmed (all)      eligible population         "how common is the thing?"
recall                   confirmed in queue   confirmed (all)             "of all real cases, how many
                                                                          did we review?"
coverage                 decided in queue     confirmed (all)             "of all real cases, how many
                                                                          did we at least look at?"
```

The four share a numerator family and differ in denominator by factors of three to a hundred in realistic settings. Quoting "precision" without naming which of these is meant is the single most common reporting defect in review operations, and it is not a pedantic one: the alert-level figure can be a third of the queue-level figure while both are arithmetically correct.

### 2.4 The invariant of a review loop

```text
review_hours_used + audit_hours_used + escalation_hours <= available_hours
AND every_ratio_names_its_sampling_frame
AND every_confirmation_carries_a_rule_version
AND the_audit_frame_is_drawn_independently_of_the_score
```

The first clause is arithmetic and fails loudly (backlog). The second fails quietly (wrong numbers, right-looking). The third fails slowly (rule drift makes last quarter incomparable). The fourth fails invisibly and is the most dangerous: an audit drawn from the top of the score distribution is not an audit, it is a second queue, and it cannot identify what the cut excluded.

### 2.5 Where the loop actually breaks

Three places account for most real failures, in this order of frequency.

First, capacity is quoted in alerts while load is consumed in investigations. A quoted "1,200 alerts per week" becomes 905 when 26% of alerts need a 15-minute escalation, and the plan built on 1,200 silently overruns by 295 alerts of work.

Second, the audit is treated as optional overhead and is the first thing cancelled when the queue is long — which is exactly the week when the queue's bias is largest, so the measurement disappears precisely when it is needed.

Third, the confirmation rule is treated as a constant when it is a policy variable. A rule change that reclassifies "insufficient evidence" as "not confirmed" lowers measured prevalence without changing the world, and it lowers measured precision in the queue while inflating the apparent value of the audit.

### 2.6 Why "just review everything" is not the answer

Full review removes the sampling problem and creates a capacity problem plus a selection problem. In the worked example, reviewing all 2,400 weekly alerts at 15.9 minutes each costs 636 reviewer-hours, which is 2.65 weeks of the current 240-hour weekly budget, or about 21 full-time reviewers at 30 productive hours each. Even at that cost the measurement is not clean, because reviewing everything turns the review into a census of alerts only: rows that never alert still go unseen, so recall remains bounded above by alert precision, and the confirmation rule still governs what counts as a case. Full review buys certainty about the queue and nothing about the tail.

**Takeaway:** the queue tells you what you found; only an independent audit frame tells you what you missed, and the two compete for the same hours.

### 2.7 Cadence: how often each number should be recomputed

| Quantity | Natural cadence | Why that cadence | Cost per refresh |
|---|---|---|---|
| queue precision | weekly | sampling error is ±3.1 points at 700 rows | free, from the log |
| alert precision | weekly | needs the unreviewed-audit rows of the same week | audit hours |
| `tau` and prevalence | weekly but read quarterly | interval is ±77% wide at this audit size | audit hours |
| recall | quarterly | ratio of two weekly estimates; weekly values are noise | none extra |
| `s_eff` and `e` | quarterly | mix effects dominate month to month | none extra |
| `p*` | quarterly | cost and delay move slowly, value moves with pricing | none extra |
| kappa | per rule version | only changes when the rule or the team changes | dual coding |
| capacity plan | quarterly, or on any cut change | it is a decision record, not a dashboard | none extra |

The cadence table exists because a weekly recall number invites over-reading. At the audit size in §5, the standard error on recall is 0.093, so two consecutive weekly values differ by less than 0.18 about 83% of the time by construction. A team that reacts to weekly recall moves is chasing its own estimator.

## 3. Foundational Definitions

### 3.1 Symbols

| Symbol | Meaning | Units | Shape | Valid range | First used |
|---|---|---|---|---|---|
| `N` | eligible accounts scored in the period | accounts/week | integer | `> 0` | §2.3 |
| `A` | alerts raised after eligibility and de-duplication | alerts/week | integer | `0 <= A <= N` | §2.3 |
| `k` | rows actually reviewed (queue throughput) | rows/week | integer | `0 <= k <= A` | §2.1 |
| `C` | reviewer hours available | hours/week | real | `> 0` | §2.4 |
| `s` | nominal service time per review | minutes | real | `> 0` | §4.2 |
| `e` | share of reviews that escalate | dimensionless | real | `0 <= e < 1` | §4.2 |
| `g` | extra minutes per escalation | minutes | real | `>= 0` | §4.2 |
| `s_eff` | effective minutes per review, `s + e*g` | minutes | real | `>= s` | §4.2 |
| `p_h` | confirmation rate in stratum `h` | dimensionless | real | `0..1` | §4.2 |
| `N_h` | population size of stratum `h` | rows/week | integer | `sum N_h = N` | §4.2 |
| `n_h` | audited rows in stratum `h` | rows | integer | `0..N_h` | §4.2 |
| `tau` | estimated total confirmed cases in the period | cases/week | real | `>= 0` | §4.2 |
| `R` | recall, `tau_1 / tau` | dimensionless | real | `0..1` | §4.2 |
| `V` | value of one confirmed case, undiscounted | currency | real | `> 0` | §4.2 |
| `d` | decision delay from detection to action | days | real | `>= 0` | §4.2 |
| `h_half` | half-life of case value with delay | days | real | `> 0` | §4.2 |
| `p*` | break-even confirmation rate per row | dimensionless | real | `0..1` | §4.2 |

### 3.2 Three near-neighbour distinctions

**Adjudicated versus reviewed.** A reviewed row has been looked at; an adjudicated row has a verdict. Pending rows are reviewed but not adjudicated, and they must be excluded from a precision denominator while remaining visible as workload. Failing to separate them inflates every denominator by the pending share.

**Confirmation versus truth.** A confirmed row is one the rule classifies as a case. Truth requires a stronger instrument than the rule, which in practice means a second independent judgement or a later hard outcome. Treating confirmation as truth is a modelling assumption with a measurable cost, computed in §4.3.

**Recall versus coverage.** Recall counts only rows that a reviewer confirmed. Coverage counts rows that reached a reviewer at all, confirmed or not. When reviewers abstain at a non-trivial rate, coverage is materially larger than recall and answers a different operational question: can we claim we looked?

### 3.3 Operational definitions that decide arithmetic

An **alert** is a distinct entity-period pair after de-duplication, not a scoring event. Duplicates between 2,900 raw events and 2,400 alerts in the example are a 17% inflation of apparent volume that would otherwise be read as behavioural change.

A **stratum** is a set of rows with a common sampling probability and a common confirmation-rate model. Strata defined by score band are the usual choice because they align with the decision, but strata defined by entity tenure or channel are admissible if the audit is drawn within them.

The **confirmation rule** is the written test a reviewer applies, with a version identifier. Changing the rule mid-quarter is a measurement event, not an administrative one.

**Capacity** is a number of completed investigations per period, computed from hours and effective service time, and stated with the escalation share that produced it.

**Break-even rate** is the confirmation rate at which the value of an hour of review equals its cost. It is the correct comparator for any observed rate, and it converts a debate about model quality into a comparison of two numbers.

### 3.4 Terms whose ambiguity causes real disputes

**"Case found."** Three different events carry this label in practice: a reviewer wrote `confirmed`; a hard outcome later arrived (chargeback, default, verified incident); or an action was taken on the account. These have different lags and different reliability. A confirmed row with no later outcome is an allegation; a hard outcome with no reviewer is a discovery the review loop did not make. Reports must say which one the numerator counts, because the three differ by the abstention rate and the action rate, and neither is negligible.

**"Reviewed."** Either a human opened the row or a human reached a verdict. Opening without verdict is the pending state of §4.1; counting it as reviewed inflates throughput and deflates precision simultaneously. The operational fix is a state machine on the adjudication record with explicit `opened`, `decided`, `escalated`, `abstained` values and an entry timestamp on each transition.

**"Capacity."** Either a headcount-times-hours figure or a completed-investigations figure. Only the second is a plan. The first is an input to the second through `s_eff`, and quoting the first in a plan review is the mechanism behind HR-F02.

**"Audit."** Either a quality check on decided rows (a second reviewer re-reads a sample of adjudications) or a coverage check on undecided rows (a reviewer reads rows the queue never reached). The first measures rule consistency; the second measures what the loop missed. Both are legitimate and they are not substitutes, yet both appear in documents as "the audit", which is why the measurement programme loses its coverage frame without anyone deciding to cancel it.

**"Precision."** Covered in §2.3 with four denominators. The dispute is not terminological: the four differ by factors of two to eight on the same week of data, and a target set on one of them can be met while another degrades.

## 4. Core Formulas with Full Derivation

### 4.1 Base case: yield on a finite reviewed set

The smallest honest yield statement is a ratio whose denominator excludes undecided rows.

```text
FORMULA
  p_hat = c / (k - u)
  where c = confirmed, k = rows reviewed, u = rows still pending

WHERE
  one queue, one period, one confirmation-rule version
  every row in the period has had at least a first-pass decision

CONSTRAINT
  k - u >= 1, and u counted from the adjudication log rather than inferred
  pending rows must be recorded with their entry timestamp, not dropped silently

NUMERICAL EXAMPLE
  k = 10, c = 3, u = 1  ->  p_hat = 3 / (10 - 1) = 0.3333
  the naive version, 3 / 10, gives 0.3000
  relative understatement = 0.30 / 0.3333 - 1 = -10.0%

INTERPRETATION
  at a 10% pending share a precision figure is 10% too low; at a 30% pending
  share (a backlog week) it is 43% too low, which reads as a model collapse
```

The formula is trivial; the discipline is not. Pending rows must be a recorded state with an entry time, so that the pending share can be reported alongside the ratio. A precision number published without its pending share is not interpretable, because the reader cannot reconstruct the denominator. Note also the asymmetry: pending rows are more likely than decided rows to be hard cases, so excluding them biases `p_hat` upwards as well as shrinking the denominator. The two effects run in opposite directions, which is why the pending share must be reported rather than silently netted out.

### 4.2 General case: capacity, totals, recall and break-even

**Result 1 — effective capacity.** Reviewers complete investigations, and an escalation is paid twice.

```text
FORMULA
  s_eff = s + e * g                                    [minutes per review]
  C_inv = (C * 60) / s_eff                             [reviews per period]

WHERE
  C in hours, s and g in minutes, e a share in [0,1)
  s_eff is a mean, so C_inv is a long-run throughput, not a guarantee for a shift

CONSTRAINT
  e is measured over at least four weeks; a single week's e is dominated by mix
  g excludes waiting time; waiting enters through the queue formula in §9

NUMERICAL EXAMPLE
  s = 12, e = 0.26, g = 15  ->  s_eff = 12 + 0.26*15 = 12 + 3.9 = 15.9 min
  C = 240 h  ->  nominal 240*60/12 = 1200 reviews
                 effective 240*60/15.9 = 905.7 reviews
  overstatement = 1200/905.7 - 1 = 32.5%

INTERPRETATION
  the nominal figure is 32.5% optimistic here; any plan built on 1200 rows
  overruns by 294 rows of work per week, which compounds into backlog
```

**Result 2 — stratified total.** The estimator is a sum of stratum expansions, and the variance is dominated by the smallest stratum.

```text
FORMULA
  tau_hat = SUM_h N_h * p_h ,  p_h = c_h / n_h
  Var(tau_hat) = SUM_h N_h^2 * (1 - n_h/N_h) * p_h(1 - p_h) / n_h

WHERE
  h indexes strata; n_h = N_h gives a census and zero variance for that stratum
  the factor (1 - n_h/N_h) is the finite-population correction

CONSTRAINT
  audit rows must be drawn with known inclusion probability within each stratum
  p_h is estimated from at most a few hundred rows, so its own uncertainty counts

NUMERICAL EXAMPLE
  N1=700, n1=700, c1=154  -> p1=0.2200, tau1=154.0, Var=0
  N2=1700, n2=250, c2=20  -> p2=0.0800, tau2=136.0, Var=725.70
  N3=81600, n3=450, c3=2  -> p3=0.0044, tau3=362.67, Var=65110.22
  tau_hat = 652.67, Var = 65835.91, SE = 256.59, relative SE = 0.3931

INTERPRETATION
  95% interval 652.67 +/- 502.91 = [149.8, 1155.6]; stratum 3 contributes
  98.9% of the variance with 0.5% of the population sampled
```

**Result 3 — recall as a ratio, with the census trap.** Recall divides one stratum by the total.

```text
FORMULA
  R = tau_1 / tau ,  tau = SUM_h tau_h
  Var(R) ~ (1/tau^2) * [ v1*(1-R)^2 + R^2*(v2 + v3) ]   for three strata
  with v1 = 0 (census queue) this collapses to Var(R) = R^2 * (v2+v3) / tau^2

WHERE
  v_h = Var(tau_h) from Result 2; the queue stratum is observed completely

CONSTRAINT
  valid when the audit is independent of the score within each stratum
  and when the same period, entity definition and rule version apply to all strata

NUMERICAL EXAMPLE
  R = 154 / 652.67 = 0.2360
  Var(R) = 0.2360^2 * 65835.91 / 652.67^2 = 0.008604, SE = 0.0928
  95% interval = [0.0541, 0.4178]
  the naive SE that treats tau as known is sqrt(v1)/tau = 0 because v1 = 0

INTERPRETATION
  the correct SE is 0.0928 and the naive one is 0.0000; reporting the naive
  value puts a five-times too narrow interval on a quantity that is 24% known
```

**Result 4 — break-even confirmation rate.** An hour of review is worth its cost only above a rate.

```text
FORMULA
  V_d = V * 2^(-d / h_half)                 [delay-discounted case value]
  p*  = cost_per_row / V_d                  [break-even confirmation rate]

WHERE
  cost_per_row = (review_hours*rate + tooling) / k  for queue rows
  cost_per_row = audit_minutes*rate/60             for audit rows
  d is measured from the event, not from the score

CONSTRAINT
  single-period horizon; the rule assumes confirmations convert to action
  at a constant rate, which fails when enforcement is capacity-bound too

NUMERICAL EXAMPLE
  V=1500, d=3.2, h_half=10  ->  V_d = 1500*2^(-0.32) = 1500*0.80107 = 1201.60
  queue: (240*58 + 1900)/700 = 22.60 per row  ->  p* = 22.60/1201.60 = 0.0188
  audit: 4.5 min * 58/60 = 4.35 per row        ->  p* = 4.35/1201.60 = 0.00362

INTERPRETATION
  observed 0.22 is 11.7x break-even and 0.0044 is 1.23x break-even, so both
  pay, but by wildly different margins that should decide the next hour spent
```

### 4.3 Variants

**Variant A — partial identification without an audit.** When no audit exists, recall is bounded, not estimated.

```text
FORMULA
  R in [ tau_1 / (tau_1 + N_2 + tau_3) ,  tau_1 / (tau_1 + tau_3) ]

WHERE
  lower bound sets every unreviewed alert to a true case
  upper bound sets every unreviewed alert to a non-case

CONSTRAINT
  both bounds require tau_3 from a non-alert audit; with no audit at all, the
  upper bound becomes 1 and the interval is [tau_1/N, 1]

NUMERICAL EXAMPLE
  lower = 154 / (154 + 1700 + 362.67) = 154 / 2216.67 = 0.0695
  upper = 154 / (154 + 362.67) = 154 / 516.67 = 0.2981
  width ratio = 0.2981 / 0.0695 = 4.29x

INTERPRETATION
  a 4.3x-wide interval is not a measurement; the audit exists to narrow it
```

**Variant B — label noise in the confirmation rule.** Sensitivity and specificity of the rule, not of the model, distort every rate downward in the tail.

```text
FORMULA
  q = se*p + (1 - sp)*(1 - p)        observed confirmation rate
  p = (q - (1 - sp)) / (se - (1 - sp))    inversion, needs se > 1 - sp

WHERE
  se = P(confirm | case), sp = P(not confirm | non-case), both from a validation set

CONSTRAINT
  se and sp must be estimated on rows where truth is known independently
  the inversion is unstable when se - (1 - sp) is small (weak rule)

NUMERICAL EXAMPLE
  se=0.85, sp=0.90  ->  q = 0.10 + 0.75p
  p=0.2200 -> q=0.2650 (ratio 1.20)
  p=0.0044 -> q=0.1033 (ratio 23.3)
  break-even on the observed scale: q* = 0.10 + 0.75*0.0188 = 0.1141

INTERPRETATION
  the observed rate compresses the true range toward 0.10; the tail stratum
  reads 0.1033, just below its 0.1141 break-even, so a profitable stratum
  is retired and a recall of 0.236 is mis-stated as 0.018
```

**Variant C — sequential adjudication (Wald).** When the question is "has the rule broken", a sequential test stops far earlier than a fixed-size audit.

```text
FORMULA
  LLR increment = ln(p1/p0) on a confirmation, ln((1-p1)/(1-p0)) on a miss
  accept H1 when LLR >= ln((1-beta)/alpha); accept H0 when LLR <= ln(beta/(1-alpha))

WHERE
  H0: p = p0 (expected rate), H1: p = p1 (degraded rate), alpha, beta error rates

CONSTRAINT
  rows must be exchangeable within a period; ordering by score violates this
  unless the test is run within a score band

NUMERICAL EXAMPLE
  p0=0.02, p1=0.05, alpha=beta=0.05
  ln(0.95/0.05) = 2.944, ln(0.05/0.95) = -2.944
  increments +0.9163 and -0.0305  ->  stops at 4 confirmations or 97 misses
  expected sample size under p0 is about 90 rows

INTERPRETATION
  a broken rule signals in four rows; a healthy one costs about 90, so the
  test is cheap when it matters and moderate when it does not
```

**Variant D — capture–recapture as an external recall check.** Two independent passes over the same period estimate the cases neither found.

```text
FORMULA
  N_hat = (|A| * |B|) / m            m = rows found by both passes
  Var(N_hat) ~ |A||B|(|A|-m)(|B|-m) / m^3

WHERE
  passes A and B must be independent in their selection, not just in time
  m is the overlap count, checked as a set intersection on entity-period keys

CONSTRAINT
  positive dependence between passes inflates m and deflates N_hat; the bias
  is roughly (1 + dependence) on m, so 20% dependence gives about 17% error

NUMERICAL EXAMPLE
  |A|=1240, |B|=980, m=430  ->  N_hat = 2826.0, SD = 82.5
  95% interval [2664.3, 2987.8]; coverage of pass A = 1240/2826 = 0.439
  if overlap is 20% inflated, m=516  ->  N_hat = 2355.0 (16.7% low)

INTERPRETATION
  the estimate is only as good as the independence claim, which is an
  argument, not a measurement; use it as a bound with a stated dependence
```

**Variant E — Neyman allocation at fixed audit cost.** Spend rows where the stratum is both large and variable.

```text
FORMULA
  n_h proportional to N_h * sqrt(p_h (1 - p_h))

WHERE
  fixed total n = SUM n_h; allocate then round and re-check that n_h <= N_h

CONSTRAINT
  p_h must be known well enough to compute the weight, usually from a pilot
  allocation oversamples strata with mid-range rates, not the biggest strata

NUMERICAL EXAMPLE
  N2=1700, p2=0.08  -> weight 1700*sqrt(0.0736) = 461.2
  N3=81600, p3=0.00444 -> weight 81600*sqrt(0.004424) = 5427.9
  n=700  ->  n2 = 700*461.2/5889.1 = 54.8 -> 55 ; n3 = 645.2 -> 645
  SE falls from 256.6 (actual 250/450 split) to 221.5, a 13.7% gain

INTERPRETATION
  the same 700 rows buy a 13.7% tighter total by moving 195 rows from the
  mid stratum to the tail, where nearly all the variance lives
```

**Variant F — two-phase audit.** Screen cheaply, then investigate the screens that pass.

```text
FORMULA
  n_phase2 = n_phase1 * s_screen           s_screen = screen-positive rate
  cost = n_phase1*c_screen + n_phase2*c_full
  Var(tau_h) uses p_h estimated from phase 2 with phase-1 inclusion weights

WHERE
  c_screen is 4.5 min (evidence check), c_full is 15.9 min (investigation)
  phase-2 rows are a sample of screen-positives, so inclusion weights differ

CONSTRAINT
  the screen must have known sensitivity, otherwise phase 2 misses rows that
  the screen classified as negative and the expansion is biased low

NUMERICAL EXAMPLE
  450 tail rows screened at 4.5 min = 33.75 h
  screen-positive rate 81/450 = 0.18 -> 81 rows at 15.9 min = 21.47 h
  2 confirmations among 81  ->  p3 = 0.0247, tau3 = 81600*0.0247 = 2014.8
  cost 55.22 h vs 33.75 h for screening alone (1.64x), cases 5.56x higher

INTERPRETATION
  two-phase costs 1.64x the screening cost and changes the answer by a factor
  of 5.56, which is a strong argument for it and a strong argument against
  treating a 4.5-minute audit as a prevalence measurement
```

The last line deserves emphasis because it is counter-intuitive. A 4.5-minute evidence check is a *screening* instrument, and using it as a prevalence instrument assumes its sensitivity is near 1. In the example, the 450-row screening audit found 2 cases; the same 450 rows pushed through an 18% screen-positive rate into an 81-row investigation phase implied 2,015 cases rather than 363, because the screen's implied sensitivity was 2/81 rather than 1. Either the screen is nearly perfect and 2 is the answer, or the screen is coarse and 2 is a floor. The audit specification must say which, and the two-phase design is how you find out.

**Variant G — value of information for a capacity decision.** The audit buys a better capacity decision, not cases.

```text
FORMULA
  EVSI ~ P(wrong decision) * |value(a_opt) - value(a_chosen)|
  value(a) = min(k_a, C_inv(a)) * p * V_d - C_hours(a) * rate

WHERE
  a indexes capacity options (k, hours, outsourcing), p the confirmation rate
  the decision is wrong when the chosen option is not the argmax under true p

CONSTRAINT
  requires a discrete option set and a prior over p; both are usually coarse,
  which is acceptable because EVSI only needs the right order of magnitude

NUMERICAL EXAMPLE
  two options: k=700 in-house at 240 h, or outsource 200 rows at 40/h
  in-house net = 154*1201.60 - 15,820 = 169,227
  outsourced 200 rows at 0.08 = 16 confirmations -> 19,226 - 8,000 = 11,226
  the in-house option wins by 158,001 under p_2 = 0.08; at p_2 = 0.03 the
  outsourced marginal value falls to 4,331 while in-house is unchanged
  so the decision is robust to p_2 across the audit's whole interval

INTERPRETATION
  when the ranking of options does not flip anywhere inside the interval, the
  audit's information value for that decision is zero and it must be justified
  by a different decision (recall reporting, rule change, capacity size)
```

This variant is the honest answer to "why audit if the cases do not pay for it". Sometimes the answer is that the audit changes no decision, in which case the correct action is to stop auditing that stratum and say so in the decision record, rather than to keep producing an interval nobody uses.

**Variant H — pooled versus dedicated reviewers.** Pooling shortens waits but dilutes specialisation.

```text
FORMULA
  W_q pooled (m servers) : Erlang C at a = lambda*s_eff/60, rho = a/m
  W_q dedicated (per type) : SUM over types of the per-type Erlang C
  service-time inflation from pooling: s_pool = s_eff * (1 + iota)

WHERE
  iota is the fractional increase in handling time when a generalist handles
  a specialised queue, measured by a trial, never assumed zero

CONSTRAINT
  the pooled model assumes exponential service; with low variance the wait
  improvement from pooling is smaller than Erlang C predicts

NUMERICAL EXAMPLE
  lambda = 20/h, s_eff = 15.9 min, a = 5.3
  m=6: rho 0.883, W_q 15.9 min ; m=7: rho 0.757, W_q 3.7 min
  if pooling raises s_eff by 12% to 17.8 min, a = 5.93 and at m=7 rho = 0.847,
  W_q rises to 9.7 min, so 40% of the pooling gain is consumed by iota = 0.12

INTERPRETATION
  pooling is worth about one reviewer at iota = 0 and less than half of that
  at iota = 0.12, so the specialisation cost must be measured before pooling
```

The practical version of this: measure `iota` on a two-week trial by comparing handling time for rows routed across queues against rows routed within queue. Without that number, the pooling argument is a queueing theorem applied to a service process that violates its stationarity assumption.

### 4.4 Limits and counterexamples

**The census stratum makes the standard error disappear.** Because the queue is observed completely, `v1 = 0`, and a naive variance that treats `tau` as fixed returns `SE = 0` for recall. The number is arithmetically correct and epistemically false: recall is a ratio, and its denominator is estimated. Any code path that computes `sd(confirmed)/mean(tau)` will produce 0.0000 here. The discriminating symptom is a recall interval of zero width alongside a total-cases interval 503 wide.

**The formula breaks when the audit is drawn from the queue.** If the audit rows are sampled from already-reviewed alerts, `n_2 = 0` and every `p_h` for the excluded region is undefined, so `tau` collapses to `tau_1` and recall evaluates to exactly 1.0. This is the most common form of the error and it produces the most flattering possible answer.

**The formula breaks when `n_h` is tiny relative to the rate.** A stratum with 2 confirmations in 450 rows gives a normal-approximation interval of `0.0044 ± 0.0061`, whose lower limit is negative. The exact interval is `[0.00054, 0.01596]`, which is 36 times wider at the lower end than the point estimate suggests. When the count is below about 10, use an exact method and stop quoting a normal interval.

**The formula breaks when capacity saturates.** At `C_inv = 905.7` and `k = 905`, the audit allocation is zero and the variance is undefined. If the plan is `k = 900`, audit hours fall to 1.5 and the measurement function is dead while the dashboard still shows throughput. Any capacity plan must reserve audit hours before the queue, not after it.

**Validity condition for the whole section.** Every result here assumes four things simultaneously: a defined period boundary shared by queue and audit; an entity definition stable across strata; a confirmation rule with a version identifier; and an audit drawn independently of the score. Drop any one and the arithmetic still runs but no longer measures what it names.

## 5. Worked Numerical Example

The setting is `ILLUSTRATIVE`: a subscription-billing review desk that scrutinises accounts flagged by an unsupervised detector. One week, one entity definition, one confirmation-rule version. All values reconstructed in this section were recomputed in fenced blocks and re-checked by an independent route in §5.9.

### 5.1 Step 1 — the hand-checkable case

Ten alerts arrive. Nine are adjudicated: three confirmed, six not. One is pending.

```text
p_hat = 3 / (10 - 1) = 0.3333
naive = 3 / 10 = 0.3000
pending share = 1/10 = 10.0%
```

The difference looks cosmetic at ten rows and is not, because the same code runs at 700 rows where a 30% pending share during a backlog week understates precision by 43%. The reporting rule is that the pending share is printed next to the ratio, always.

### 5.2 Step 2 — volume, de-duplication and eligibility

Raw scoring events for the week number 2,900. De-duplication on entity-period keys removes 500 repeated events, giving `A = 2,400` alerts from `N = 84,000` eligible accounts. Alert rate is `2400/84000 = 0.02857`, or 2.86% of accounts.

The duplicate share, `500/2900 = 17.2%`, is not a cosmetic data-quality statistic. A detector whose volume is measured in events rather than entities will report a 20.8% volume increase on unchanged behaviour, and an alert-rate target set in events will be met by making the de-duplication key coarser.

Of the 2,400 alerts, 700 were reviewed. The remaining 1,700 are unreviewed alerts — the high tail of the score distribution that capacity could not reach, and the population that the audit samples.

### 5.3 Step 3 — capacity, and why the plan is wrong

Eight reviewers, 30 productive hours each after meetings and training, give `C = 240` hours. Nominal service time is 12 minutes per review.

```text
nominal capacity   = 240*60/12      = 1200 reviews/week
escalation share e = 0.26, at g = 15 extra minutes
s_eff              = 12 + 0.26*15  = 15.9 minutes
effective capacity = 240*60/15.9    = 905.66 reviews/week
overstatement      = 1200/905.66 - 1 = 32.5%
```

The plan asks for 700 reviews, which consumes `700*15.9/60 = 185.5` hours, a utilisation of `185.5/240 = 77.3%`. The remaining 54.5 hours is the entire measurement budget, and it is also the flex that absorbs a bad week.

### 5.4 Step 4 — the three strata and the total

The audit is allocated 250 rows to unreviewed alerts and 450 rows to non-alert accounts, at 4.5 minutes per audited row because an audited row is a rapid evidence check rather than a full investigation.

```text
stratum   N_h      n_h    c_h    p_h        tau_h     Var(tau_h)
S1 queue   700     700    154   0.22000    154.00       0.00
S2 unrev  1700     250     20   0.08000    136.00     725.70
S3 clean 81600     450      2   0.00444    362.67   65110.22
-------------------------------------------------------------
total    84000    1400    176              652.67   65835.91
```

`tau_hat = 652.67` cases in the week, `SE = sqrt(65835.91) = 256.59`, relative SE `0.3931`. The 95% interval is `[149.8, 1155.6]`. Prevalence is `652.67/84000 = 0.00777`, or 7.77 cases per 1,000 accounts.

Variance decomposition is the actionable part: S3 carries `65110.22/65835.91 = 98.9%` of the variance while holding 97.1% of the population and receiving 450 of 1,400 audited rows. The queue, a census, contributes exactly nothing to the variance and 23.6% of the point estimate.

### 5.5 Step 5 — precision, coverage and recall, three different numbers

```text
queue precision   = 154 / 700      = 0.2200   (reviewers' hit rate)
alert precision   = 290 / 2400     = 0.1208   (hit rate over all alerts)
coverage          = 290 / 652.67   = 0.4443   (share of cases that reached a human)
recall            = 154 / 652.67   = 0.2360   (share of cases a human confirmed)
miss rate         = 362.67 / 652.67= 0.5557   (share of cases never seen)
```

Alert precision is 55% of queue precision, and the gap is the 1,700 unreviewed alerts at an 8% confirmation rate. Recall is 53% of coverage, and the gap is the 136 confirmed cases inside unreviewed alerts that no reviewer ever adjudicated. A report that prints only "22% precision" overstates the detector's practical hit rate by 1.82 times and says nothing at all about what was missed.

### 5.6 Step 6 — an interval on recall, and the trap in the obvious variance

```text
R      = 154 / 652.67 = 0.235955
Var(R) = R^2 * (v2 + v3) / tau^2
       = 0.055675 * 65835.91 / 426,098.1
       = 0.055675 * 0.154508 = 0.008603
SE(R)  = 0.09276
95% CI = 0.23596 +/- 1.96*0.09276 = [0.0541, 0.4178]
```

The naive route, treating `tau` as a known constant and computing `sqrt(v1)/tau`, returns zero because the queue is a census. The interval `[0.236, 0.236]` is the single most common false precision in review reporting. The corrected interval spans a factor of 7.7, which is what a 450-row tail sample actually supports.

### 5.7 Step 7 — value, cost and break-even

One confirmed case is worth `V = 1500` when acted on promptly, and value decays with delay at a 10-day half-life. The typical detection-to-action delay is 3.2 days.

```text
V_d        = 1500 * 2^(-3.2/10) = 1500 * 0.80107 = 1201.60
labour     = 240 h * 58 = 13,920
tooling    = 1,900
total cost = 15,820 per week
per review = 15,820 / 700 = 22.60
per confirmation = 15,820 / 154 = 102.73
break-even p* = 22.60 / 1201.60 = 0.01881   (1.88%)
observed p     = 0.2200  ->  margin 11.7x
value created  = 154 * 1201.60 = 185,047  ->  ratio 11.7 to cost
```

The audit row is priced differently: `4.5 min * 58/60 = 4.35` per row, so its break-even rate is `4.35/1201.60 = 0.00362`. The tail stratum confirms at 0.00444, a margin of 1.23x. That margin is thin enough that the audit's justification cannot rest on the cases it finds directly — two confirmations worth 2,403 against 3,045 of audit labour. The audit is justified by what it reveals about the 81,600 rows nobody can afford to look at, and by the 1,700 unreviewed alerts whose 8% rate is four times break-even.

### 5.8 Step 8 — what the audit displaced

Audit rows are not free of opportunity cost. The 700 audited rows consume `700*4.5/60 = 52.5` hours, which at 15.9 minutes per investigation is `52.5*60/15.9 = 198.1` reviews not performed.

```text
displaced reviews  = 198.1
displaced confirmations = 198.1 * 0.08 = 15.85   (using the unreviewed-alert rate)
displaced value    = 15.85 * 1201.60 = 19,047
audit labour       = 52.5 h * 58 = 3,045
direct audit value = 2 * 1201.60 = 2,403
```

On a one-week horizon the audit destroys more value than it creates: 19,047 displaced against 2,403 found. This is the honest accounting, and it is the reason audit programmes are always cancelled first. The justification for the audit is intertemporal: without it, `tau` is unidentified, recall is bounded at `[0.0018, 0.531]` with a 290-fold width, and every downstream capacity decision is made blind. Spending 19,047 of weekly value to convert a 290-fold interval into a 7.7-fold one is a decision about information, not about cases, and it must be argued as such.

### 5.9 Step 9 — independent re-check

Three independent routes to the same quantities, so that an arithmetic slip in one cannot survive.

```text
CHECK 1  recall as a reciprocal of stratum ratios
  1/(1 + tau2/tau1 + tau3/tau1) = 1/(1 + 0.88312 + 2.35498) = 1/4.23810 = 0.23596  OK

CHECK 2  coverage as one minus the missed share
  missed = 362.67/652.67 = 0.55567 ; coverage = 1 - 0.55567 = 0.44433  OK

CHECK 3  per-1,000 cross-check of the expansion
  S1 0.22*700/84     = 1.833 per 1,000
  S2 0.08*1700/84    = 1.619 per 1,000
  S3 0.004444*81600/84 = 4.318 per 1,000
  total               = 7.770 per 1,000  vs tau/N = 652.67/84 = 7.770  OK

CHECK 4  variance share
  65110.22/65835.91 = 0.98898 ; 1 - 0.98898 = 0.01102 from S2, S1 contributes 0  OK

CHECK 5  cost per confirmation two ways
  15,820/154 = 102.73 ; (22.60 + 1900/700)/0.22 = (22.60+2.714)/0.22 = 114.15
  the second route adds tooling twice; the first is correct — the discrepancy
  shows why tooling must be allocated once, to the period, not to each row
```

Check 5 is included because it failed on first attempt. The error is instructive: allocating fixed tooling cost per row and then dividing by the rate double-counts it, inflating cost per confirmation by 11.1%. Fixed costs belong to the period.

### 5.10 Step 10 — sensitivity

The tail stratum sets everything, and its rate is estimated from two confirmations.

```text
p3      tau      prevalence/1000   recall
0.0010   371.6        4.42          0.4144
0.0020   453.2        5.39          0.3398
0.00444  652.7        7.77          0.2360
0.0080   942.8       11.22          0.1633
0.0160  1595.6       19.00          0.0965
```

Recall moves by a factor of 4.3 across a range of `p3` that the 450-row audit cannot distinguish: the exact 95% interval for 2/450 is `[0.00054, 0.01596]`, which brackets four of the five rows. The sheet's central claim follows directly: at this audit size recall is not estimated, it is bounded by an assumption. Doubling the tail sample to 1,750 rows narrows the interval on `tau` from 0.393 to 0.201 relative, at a cost of 150 audit-hours per week.

### 5.11 What this example does and does not establish

It establishes the arithmetic of capacity, stratified totals, ratio variance, break-even rates and audit displacement, all of which are identities. It does not establish that any organisation has these rates, this escalation share, or this confirmation rule. The 26% escalation share is a constructed input; a real desk measuring 0.26 would need at least four weeks of adjudication logs to defend it. The example also assumes the audit is independent of the score, which is a design commitment rather than a result.

## 6. Visual / Geometric Intuition

### 6.1 The population as a partition

```text
|<----------------------------- N = 84,000 accounts ------------------------------>|
|                                                                                   |
|   [ S1 queue        ] [ S2 unreviewed alerts ]  [        S3 clean        ]
|     N=700  p=.22       N=1,700  p=.08             N=81,600  p=.00444
|     census             audit 250                  audit 450
|     tau=154            tau=136                    tau=362.7
|                                                                                   |
|<-- 2.9% of accounts -->|<------- 97.1% of accounts, 98.9% of variance ------->|
```

The picture that matters is the right-hand bar. Almost all accounts are in the stratum with the least data and the most variance, because a tail rate multiplied by 81,600 dominates a head rate multiplied by 700. Any intuition that says "the queue is where the action is" is correct about enforcement and wrong about measurement.

### 6.2 Two sampling frames, one period

```text
frame 1  score >= cut        frame 2  score < cut
+---------------------+      +---------------------+
| every row reviewed  |      | sample n_h rows     |
| census, v = 0       |      | expansion, v > 0    |
+---------------------+      +---------------------+
          \                            /
           \                          /
            +---- both needed for ----+
                 tau and recall
```

The two frames answer different questions and neither can substitute for the other. Frame 1 without frame 2 yields recall identically 1.0. Frame 2 without frame 1 yields prevalence with no numerator. The diagram is also the argument for why an audit drawn inside frame 1 is worthless: it adds rows to a frame that is already a census.

### 6.3 The identification interval

```text
recall
1.0 |########################################  <- no audit at all
    |
0.53|=========================                 <- upper bound, no audit, S3 known
    |
0.30|---  upper bound with S2 audit            <- tau1/(tau1+tau3)
0.24|  ** point estimate 0.236
0.07|---  lower bound with S2 audit            <- tau1/(tau1+N2+tau3)
0.055|= lower bound with full audit (95% CI)   <- ratio variance
0.00|
    +----------------------------------------
     none      S2 only     S2+S3      exact
```

Reading upward from the axis: with no audit, recall is anything in `[0.0018, 1.0]`. Adding the unreviewed-alert audit halves the ceiling. Adding the clean-account audit moves the floor by two orders of magnitude. The exact interval is a band, not a line, and the band is wide because the tail sample is small.

### 6.4 The delay-discount curve

| Days from event to action | Survivor factor | Effective value from 1500 |
|---|---|---|
| 0.0 | 1.0000 | 1500.0 |
| 1.0 | 0.9330 | 1399.5 |
| 3.2 | 0.8011 | 1201.6 |
| 5.0 | 0.7071 | 1060.7 |
| 7.0 | 0.6156 | 923.4 |
| 10.0 | 0.5000 | 750.0 |

Each day of queue delay costs about 6.7% of remaining value near the operating point, so a queue that ages to seven days loses 38% of the case value that a same-day review would have captured. This is the quantitative case for reviewing the top of the queue before widening it.

### 6.5 The capacity frontier

```text
utility
 1.0 |                                    x  k=900 : util .994, audit 0.0 h
 0.9 |                          x  k=800 : util .883, audit 28.0 h
 0.8 |              x  k=700 : util .773, audit 54.5 h
 0.7 |    x  k=600 : util .663, audit 81.0 h
 0.6 |
     +------+---------+---------+---------+-------> k (reviews/week)
          600       700       800       900
          <--- audit hours are the slack that disappears rightward --->
```

The frontier is drawn with `s_eff = 15.9` minutes and `C = 240` hours. Every row of extra throughput buys confirmations at a diminishing rate and destroys audit hours linearly. At `k = 900` the desk reviews 900 rows and measures nothing: utilisation is 0.994 and the remaining 1.5 hours funds 20 audit rows.

### 6.6 Agreement as a 2x2 picture

```text
                    reviewer 2: case      reviewer 2: not case
reviewer 1: case          a = 44                b = 16
reviewer 1: not case      c = 10                d = 30

raw agreement   = (44+30)/100      = 0.740
expected        = .60*.54 + .40*.46 = 0.508
kappa           = (.74-.508)/(1-.508) = 0.4715   SE 0.0892   CI [0.297, 0.646]
disagreement    = 26/100 = 0.260  -> 182 escalations at 700 rows
```

Raw agreement of 0.740 looks strong and is barely above what two reviewers with those marginal rates would produce by chance. The expected-agreement term, 0.508, is doing most of the work, which is why a 0.74 headline can coexist with a moderate kappa. The 26% disagreement rate is not only a quality statistic: at 15 minutes of rework per disagreement it consumes 45.5 hours, 19.0% of the weekly budget, which must be reserved before throughput is planned.

### 6.7 Sequential stopping

```text
LLR
 +2.944 |---------------------------------------- accept H1 (degraded)
        |                                   /
        |                              /
  0.000 |--------------------------/-------------- continue
        |                     /
        |                /
 -2.944 |---------------------------------------- accept H0 (expected)
        +----+----+----+----+----+----+----+----> rows
             4    20   40   60   80   97

slope per confirmation = +0.9163 ; per miss = -0.0305
```

The asymmetry is the design. A degraded rule climbs to the upper boundary in four confirmations; an expected rule drifts downward so slowly that it takes about 97 consecutive misses, an expected sample size near 90 rows. Sequential tests are therefore useful as a tripwire on the rule, not as a substitute for the audit that estimates the rate.

## 7. Comparison / Selection Table

### 7.1 Review-queue policies

| Policy | Rows reviewed | What it optimises | Fails when | Cost driver | Needs |
|---|---|---|---|---|---|
| Fixed threshold | varies with volume | interpretability | prevalence shifts move volume, not risk | burst tolerance | stable score scale |
| Top-k at capacity | exactly k | throughput guarantee | k is set without audit hours reserved | reviewer hours | known `s_eff` |
| Score×value ranking | k | expected value | value model wrong by order of magnitude | value model upkeep | per-segment `V` |
| Stratified quota | k split by band | stable measurement | quotas freeze on a stale mix | queue tooling | stratum definitions |
| Active learning | k | label efficiency for refit | refit is not the binding problem | labelling latency | model iteration cadence |
| Dual review on sample | k + delta | agreement estimation | delta grows without bound | reviewer hours | `kappa` target |
| Sequential rule test | ~90 | fast detection of rule break | rate shifts are gradual | analyst attention | pre-registered `p0`, `p1` |

The selection rule is unglamorous. Choose top-k with reserved audit hours unless a specific clause above is binding. Top-k is the only policy that makes capacity a first-class input, and capacity is the constraint that actually binds in every desk this sheet describes.

### 7.2 Audit designs for the excluded region

| Design | Estimator | Variance | Cost per row | Main threat |
|---|---|---|---|---|
| Simple random over non-alerts | `N p` | `N^2(1-f)p(1-p)/n` | 4.5 min | none material; very inefficient |
| Stratified by score band | `SUM N_h p_h` | sum of per-stratum terms | 4.5 min | allocation ignores variance |
| Neyman allocation | same, weights `N_h sd_h` | 13.7% lower at same `n` | 4.5 min | weights from a stale pilot |
| Two-phase (audit then deep-dive) | ratio estimator | two-stage formula | 4.5 + 15.9 min | phase-2 selection bias |
| Capture–recapture on two passes | `|A||B|/m` | `|A||B|(|A|-m)(|B|-m)/m^3` | pass cost | dependence between passes |
| Full census | exact count | zero | prohibitive | none; infeasible |

Stratified with a variance-aware allocation is the default. Capture–recapture is worth one exercise per year as an external check on the audit's own coverage, because it can detect the failure that stratified sampling cannot: an audit frame that has quietly become score-dependent.

### 7.3 Agreement statistics on the same table

| Statistic | Value on §6.6 table | Monotone in | Misleading when | Use for |
|---|---|---|---|---|
| Raw agreement | 0.740 | marginal balance | prevalence is extreme | never alone |
| Cohen's kappa | 0.4715 | true agreement | marginals differ between raters | two raters, similar roles |
| PABAK | 0.480 | true agreement | marginals are genuinely unequal | screening for prevalence artefact |
| Gwet's AC1 | 0.486 | true agreement | very high agreement is rare | skewed prevalence |
| Krippendorff's alpha | 0.469 | true agreement | ordinal data needs a metric | more than two raters |

PABAK is `2*po - 1 = 2*0.74 - 1 = 0.48`, which is the kappa you would get if the marginals were equal. When kappa and PABAK diverge, the divergence is a statement about the marginal rates, not about disagreement, and the correct response is to report both plus the marginals.

## 8. Failure Modes & Edge Cases

### HR-F01 — Pending rows counted as negatives

**Symptom.** Precision drops 10–40% in the days after a volume spike and recovers without any model change. The denominator `k` is stable while `c` lags.

**Competing causes.** (a) pending rows treated as negatives; (b) genuinely harder alerts after a spike; (c) confirmation rule tightened mid-week.

**Discriminating check.** Recompute precision on adjudicated rows only, and plot the pending share by day. If the rise in the pending share coincides with the fall, cause (a) is established; the model did not change and neither did the alert mix.

**Correction.** Denominator becomes `k - u`; pending share printed beside the ratio; SLA on adjudication age.

**Prevention.** Adjudication log records an explicit `pending` state with an entry timestamp, and the reporting query refuses to run without a pending count.

**Verification.** For one week, both figures computed and compared: the corrected value must exceed the naive one by exactly `u/(k-u)`.

### HR-F02 — Capacity quoted in alerts, consumed in investigations

**Symptom.** Backlog grows steadily at a throughput the plan calls sustainable; utilisation reports above 0.85 while the queue ages.

**Competing causes.** (a) escalation rework omitted from `s_eff`; (b) duplicate alerts consuming slots; (c) reviewers also handling non-queue work not in `C`.

**Discriminating check.** Divide completed adjudications by paid hours for four weeks: the empirical minutes per investigation. If it exceeds nominal by approximately `e*g`, cause (a) holds. Then check the alert-to-entity ratio for (b).

**Correction.** Recompute `C_inv` from measured `s_eff`; publish capacity in investigations.

**Prevention.** Capacity is defined as completed investigations per week, and the escalation share is a published parameter refreshed quarterly.

**Verification.** Planned throughput must equal measured completions within 5% for three consecutive weeks.

### HR-F03 — Audit allocated proportionally to stratum size

**Symptom.** Audit intervals barely narrow when rows are added, and the tail estimate dominates the total variance year after year.

**Competing causes.** (a) proportional allocation starving the tail; (b) tail rate genuinely unstable; (c) audit rows not independent within a stratum.

**Discriminating check.** Compute `N_h sd_h` per stratum. If a stratum holding 97% of the population and 98.9% of the variance receives 64% of rows, cause (a) holds; Neyman weights would move 195 rows and cut SE by 13.7%.

**Correction.** Re-allocate on `N_h sqrt(p_h(1-p_h))`, then re-check `n_h <= N_h`.

**Prevention.** Allocation rule written into the audit specification with a pilot refresh, not chosen per quarter.

**Verification.** Compare achieved SE against the Neyman-optimal SE; the gap must be under 5%.

### HR-F04 — Reviewed-only recall reported as recall

**Symptom.** Recall is close to 1.0, or is computed as `confirmed/reviewed`; the interval has zero width.

**Competing causes.** (a) recall computed inside frame 1 only; (b) census stratum mistaken for zero-variance population; (c) audit frame actually drawn from reviewed rows.

**Discriminating check.** Ask for the denominator's provenance. If every row in the denominator appears in the queue extract, cause (a) or (c). If the denominator is right but the SE is zero, cause (b).

**Correction.** Recall is a ratio of two estimated quantities; the variance uses `v2 + v3`, never `v1` alone.

**Prevention.** The estimator is a single function that takes the audit table as a required argument and raises when the audit frame is empty.

**Verification.** Recall interval width must exceed zero; a zero-width interval is a build failure, not a good result.

### HR-F05 — Agreement measured as a percentage

**Symptom.** "Reviewers agree 74% of the time" is quoted as evidence of a reliable rule, while escalations consume a fifth of capacity.

**Competing causes.** (a) raw agreement used where marginals are skewed; (b) genuine rule ambiguity concentrated in one evidence type; (c) reviewer training gap on a specific reason code.

**Discriminating check.** Compute kappa and PABAK beside raw agreement. Divergence between them locates the artefact in the marginals; agreement that is low within a single reason code locates it in the rule.

**Correction.** Report raw agreement, kappa with an interval, and the disagreement count with its labour cost.

**Prevention.** Disagreement is treated as a workload line item in the capacity plan, not as a quality score.

**Verification.** Kappa interval must exclude 1 by a margin consistent with the rework hours actually observed.

### HR-F06 — Capture–recapture with dependent passes

**Symptom.** The external estimate of total cases is materially below the audit-based estimate, and the gap widens when the two passes share a data source.

**Competing causes.** (a) positive dependence inflating overlap; (b) heterogeneous capture probability within a pass; (c) population definition differing between passes.

**Discriminating check.** Compare the entity-definition and inclusion windows of the two passes. If they share a feed and a cut, cause (a) is presumed, not tested.

**Correction.** Deflate overlap by a stated dependence factor, or report a bound rather than a point.

**Prevention.** Passes are designed to be independent in selection mechanism, and the dependence argument is written down before the number is used.

**Verification.** Report `N_hat` at 0%, 10% and 20% assumed dependence; a conclusion that flips across that range is not a conclusion.

### HR-F07 — Confirmation-rule noise in the tail

**Symptom.** The tail stratum's observed rate sits near the audit break-even, so the audit looks unprofitable and is cut, while the estimated recall is implausibly low.

**Competing causes.** (a) rule sensitivity/specificity compressing the observed rate toward `1 - sp`; (b) a genuine fall in tail prevalence; (c) audit rows judged by a different rule version than the queue.

**Discriminating check.** Estimate `se` and `sp` on a validation set with known truth. If `q/(1-sp)` is near 1 in the tail, cause (a) dominates: observed 0.1033 against `1-sp = 0.10` means almost the entire observed rate is rule noise.

**Correction.** Report both observed and inverted rates; if `se - (1-sp)` is below about 0.5, declare the tail rate unidentified rather than quoting an inversion.

**Prevention.** The confirmation rule is versioned and its `se`/`sp` are re-estimated whenever the rule changes.

**Verification.** Inverted prevalence must lie inside the exact interval of the observed rate mapped through the inversion.

### HR-F08 — The audit is cancelled under load

**Symptom.** Audit rows per week fall to zero in exactly the weeks when the queue is longest; recall estimates stop appearing; capacity decisions are made from queue data alone.

**Competing causes.** (a) audit staffing not ring-fenced; (b) audit scheduled after throughput in the plan; (c) audit owned by the same manager as throughput.

**Discriminating check.** Plot audit rows against queue length by week. A negative correlation above about −0.7 establishes cause (a) or (b); ownership determines which.

**Correction.** Reserve audit hours as a fixed cost before throughput; express the reserve as a share of `C` with a floor.

**Prevention.** Audit is a separate budget line with its own owner and its own SLA, and the desk cannot trade it for throughput without a recorded decision.

**Verification.** Audit rows per week must have a non-zero minimum over the trailing quarter, and the minimum weeks must be listed.

### HR-F09 — Threshold creep against a fixed value model

**Symptom.** Confirmation rate falls slowly and monotonically while the model version is unchanged; each quarter's cut is slightly lower than the last.

**Competing causes.** (a) cut lowered to fill capacity after `s_eff` improved; (b) prevalence genuinely declining; (c) score distribution drifting under unchanged features.

**Discriminating check.** Recompute precision at the historical cut on current data. If precision at the old cut is stable but precision at the current cut has fallen, cause (a) holds: the extra rows are marginal by construction.

**Correction.** Price the marginal row explicitly: at an elasticity of about −0.66, moving from 700 to 900 rows takes the rate from 0.220 to 0.186 and the cost per confirmation from 102.73 to 93.89, a trade that must be accepted deliberately.

**Prevention.** The cut is a decision record with an owner, a date and the capacity number that justified it.

**Verification.** Marginal confirmation rate at the cut must exceed `p*` by the margin stated in the decision record, recomputed quarterly.

### HR-F10 — The loop teaches the detector what reviewers reward

**Symptom.** Adjudications cluster on the evidence types reviewers resolve quickly; scores rise for those patterns; the audit's confirmation rate falls even though the queue's rises.

**Competing causes.** (a) refit on review-selected labels; (b) reviewers genuinely faster on easy patterns; (c) audit drawn with a lag that misses the shift.

**Discriminating check.** Compare confirmation rate on queue rows against confirmation rate on audit rows of the same score band. Rising queue rate with falling audit rate in the same band is the signature of selection, because the audit is not selected by the score.

**Correction.** Refit on a label set that includes audit rows weighted by inverse inclusion probability, or restrict refits to features that cannot encode reviewer speed.

**Prevention.** Labels carry a provenance flag distinguishing queue adjudication from audit adjudication; refits must declare the mix.

**Verification.** The two rates must move together within the sampling error of the audit; systematic divergence blocks the refit.

## 9. Complexity & Cost Analysis

### 9a Algorithmic and operational costs

| Operation | Cost | Grows as | Practical note |
|---|---|---|---|
| Score all accounts | `O(N d)` | linear in accounts and features | 84,000 rows is trivial; features dominate |
| Rank and cut | `O(A log A)` | `A log A` | use a heap for top-k, not a full sort |
| De-duplicate to entities | `O(A log A)` | `A log A` | key choice changes volume by 17% |
| Reviewer adjudication | `O(k * s_eff)` | linear in rows | the dominant term by two orders |
| Audit adjudication | `O(n * 4.5 min)` | linear in rows | cheap per row, expensive per insight |
| Stratified estimator | `O(H)` | constant in rows | variance arithmetic is free |
| Agreement table | `O(k)` pairs | linear in rows | needs dual-coded rows only |
| Sequential test | `O(rows until stop)` | expected `~90` | cheapest measurement in the sheet |

The asymmetry is the point of the table. Computation is free relative to adjudication: scoring 84,000 accounts costs less than adjudicating three rows. Every design decision that trades compute for reviewer minutes is therefore almost always correct, and every design decision that trades reviewer minutes for a marginally better score is almost always wrong at the margin.

### 9b Sizing the desk

```text
C = 240 h/week, s_eff = 15.9 min, audit row = 4.5 min
k = 700  ->  review 185.5 h + audit 52.5 h = 238.0 h  (util 0.9917)
k = 800  ->  review 212.0 h + audit 28.0 h = 240.0 h  (util 1.0000)
k = 900  ->  review 238.5 h + audit  1.5 h = 240.0 h  (util 1.0000)
k = 906  ->  audit allocation reaches zero; variance undefined
```

At `k = 700` the desk is already at 99.2% utilisation once audit is included, with 2.0 hours of slack. That is not a comfortable plan: it has no capacity to absorb a 3% rise in escalation share, and a 3-percentage-point rise in `e` adds `700*0.03*15/60 = 5.25` hours, which the slack cannot cover. The correct reading of "we have 240 hours and we review 700 rows" is that the desk is over-committed by about 3.25 hours per week.

### 9c Cost of the measurement function

| Line | Hours/week | Cost/week | Share of budget | What it buys |
|---|---|---|---|---|
| Queue review | 185.5 | 10,759 | 77.3% | 154 confirmations |
| Escalation rework (inside review) | 45.5 | 2,639 | 19.0% | rule consistency |
| Audit rows | 52.5 | 3,045 | 21.9% | `tau` and recall |
| Tooling | — | 1,900 | — | queue, log, estimator |
| Total | 238.0 | 15,820 | 100% | 185,047 of value at 11.7x |

Two readings of this table are both defensible and must be kept apart. Read as enforcement, the desk returns 11.7 times its cost and the audit is a drag. Read as measurement, the audit costs 19% of the budget and converts a 290-fold identification interval into a 7.7-fold one; there is no cheaper way to obtain that, because the alternative is to guess the tail rate and let an assumption drive every capacity decision.

### 9d Marginal value per constrained hour

```text
queue row     (0.22*1201.60 - 22.60) / (15.9/60) = 241.75/0.265 = 912 per hour
audit row     (0.08*1201.60 - 4.35)  / (4.5/60)  = 91.78/0.075  = 1224 per hour
tail audit row(0.00444*1201.60-4.35) / (4.5/60)  = 0.99/0.075   = 13 per hour
```

The middle line looks like an argument to stop investigating alerts and start auditing them, and it is not. The audit row costs 4.5 minutes because it is a rapid evidence check; the moment a row is escalated to an investigation it costs 15.9 minutes, and the per-hour value falls to `(96.13-22.60)/0.265 = 277`, below the queue. Audit rows buy knowledge at a discount; investigative rows buy enforcement at full price. Mixing the two rates in one ranking is a category error.

### 9e The information value of narrowing an interval

The audit's product is not cases, it is a narrower posterior over `tau`. Priced crudely, going from a 450-row to a 1,750-row tail sample costs `(150-52.5)*58 = 5,655` per week and moves relative SE from 0.393 to 0.201. If the capacity decision it informs is worth about 70 reviewer-hours per week (`(905-700)*15.9/60 = 54.4` hours at the margin, plus 15 hours of queue tooling), then the decision's annual value is of order `54.4*58*52 = 163,974` and a decision error rate of 10% avoided justifies several thousand per week. The arithmetic is approximate and labelled so; the structural point is that measurement spend should be sized against the decision it changes, not against the cases it finds.

## 10. Key Theorems / Results Reference

| # | Result | Statement used here | Source status | Section |
|---|---|---|---|---|
| 1 | Stratified estimator variance | `Var = SUM N_h^2 (1-f_h) p_h q_h / n_h` under independent within-stratum sampling | `SOURCED` standard survey sampling | §4.2 |
| 2 | Delta method for ratios | `Var(X/Y) ~ (mu_X^2/mu_Y^2)[CV_X^2 + CV_Y^2 - 2 rho CV_X CV_Y]` | `DERIVED` first-order Taylor | §4.2 |
| 3 | Petersen estimate | `N_hat = n1 n2 / m` with approximate variance `n1 n2 (n1-m)(n2-m)/m^3` | `SOURCED` capture–recapture | §4.3 |
| 4 | Wald sequential bounds | stop at `ln((1-beta)/alpha)` or `ln(beta/(1-alpha))` in the LLR | `SOURCED` SPRT | §4.3 |
| 5 | Hoeffding bound | `n >= ln(2/delta) / (2 eps^2)` for `eps`-accurate proportions | `SOURCED` concentration inequality | §11 Q9 |
| 6 | Clopper–Pearson | exact interval by inverting two binomial tails | `SOURCED` exact binomial | §5.10 |
| 7 | Little's law | `L = lambda W` for a stable system | `SOURCED` queueing | §6.5 |
| 8 | Erlang C | `P(wait)` and `W_q` for `m` pooled servers | `SOURCED` queueing | §11 Q13 |

### 10.1 Derivation sketch: ratio variance and the census stratum

Let `tau_1` be a census and `tau = tau_1 + tau_2 + tau_3` with the latter two estimated. Write `R = tau_1/tau` and expand to first order around the means: `dR = (1/tau) dtau_1 - (tau_1/tau^2) dtau`. Squaring and taking expectations, with the strata independent, gives `Var(R) = (1/tau^2)[v1(1-R)^2 + R^2(v2+v3)]`. With `v1 = 0` this reduces to `R^2(v2+v3)/tau^2`, which is the formula used in §5.6. The identity explains the census trap in one line: `v1 = 0` removes the term that carries the numerator's uncertainty, so only the denominator's uncertainty survives — and a naive variance that estimates only the numerator returns exactly zero.

### 10.2 Derivation sketch: Petersen bias under dependence

If the two passes are positively dependent, the expected overlap exceeds `n1 n2 / N`. Write `E[m] = (n1 n2 / N)(1 + delta)` for a dependence factor `delta`. Then `E[N_hat] = N/(1+delta)`, so the estimate is biased low by `delta/(1+delta)`. At `delta = 0.2` the bias is 16.7%, which is the difference between 2,826 and 2,355 in §4.3. The direction is the dangerous one: dependence makes coverage look better than it is, because a smaller `N_hat` raises the ratio `n1/N_hat` from 0.439 to 0.527.

### 10.3 Derivation sketch: why `p*` is a rate and not a score

Break-even sets the expected value of a review equal to its cost: `p V_d = c_row`, so `p* = c_row/V_d`. The result contains no property of the model. It follows that score improvements cannot move `p*`, only observed rates can, and that a model upgrade which raises precision from 0.20 to 0.24 changes nothing about whether the desk should exist. The corollary used in §7.1 is that capacity, not ranking, is the binding constraint whenever observed rates sit far above `p*`, which in the worked example they do by 11.7 times.

### 10.4 What these results do not license

None of the above licenses a causal claim about why a rate changed. A falling confirmation rate is consistent with a worse detector, a looser cut, a tighter rule, a shifting population, or a change in reviewer mix, and the variance formulas are silent on which. Identification comes from the design — the independent audit, the version stamps, the decision record — not from the estimator.

## 11. Interview / Exam Questions

**Q1 (recall).** You are told "precision is 22% on 700 reviews". Name the four facts you need before the number is usable.

*Checked answer:* the pending count (denominator is `k-u`, not `k`), the sampling frame (queue-only versus all alerts), the confirmation-rule version, and the period with its entity de-duplication key. *Reasoning:* with 70 pending rows the same counts give 24.4% instead of 22.0%, and with the alert-level frame the same week yields 12.1%. *Follow-up:* which of the four changes the number most? *Red flag:* accepting the figure as a property of the model.

**Q2 (calculation).** Nominal service time is 12 minutes, 26% of reviews escalate at 15 extra minutes, and you have 240 hours. What is real capacity, and how wrong is the nominal plan?

*Checked answer:* `s_eff = 12 + 0.26*15 = 15.9`; capacity `240*60/15.9 = 905.7`; the nominal `1200` overstates by 32.5%. *Reasoning:* escalation is paid per review, so it enters the mean service time, not a separate line. *Follow-up:* what escalation share would bring capacity to 800? *Red flag:* multiplying 1200 by `1 - 0.26`.

**Q3 (calculation).** Three strata: `(700, 700, 154)`, `(1700, 250, 20)`, `(81600, 450, 2)`. Compute the total confirmed cases and its standard error.

*Checked answer:* `tau = 154 + 136 + 362.67 = 652.67`; variance `0 + 725.70 + 65110.22 = 65835.91`; `SE = 256.59`, relative 0.393. *Reasoning:* each stratum expands by `N_h` and carries its own finite-population correction; the census stratum contributes zero variance. *Follow-up:* which stratum would you double? *Red flag:* averaging the three rates without weights.

**Q4 (calculation).** Using the same strata, compute recall and a 95% interval. Why does the obvious formula give zero?

*Checked answer:* `R = 154/652.67 = 0.236`; `Var(R) = R^2(v2+v3)/tau^2 = 0.00860`; `SE = 0.0928`; interval `[0.054, 0.418]`. The obvious formula uses `v1`, which is zero because the queue is a census. *Reasoning:* recall is a ratio, so the denominator's uncertainty contributes even when the numerator is certain. *Follow-up:* what does the interval become if the queue is only 80% reviewed? *Red flag:* reporting recall as exactly 0.236.

**Q5 (diagnosis).** Precision falls from 30% to 22% in one week with a stable model version and stable alert volume. Give the first three checks in order.

*Checked answer:* pending share by day; confirmation-rule version stamp; stratum mix of the reviewed rows. *Reasoning:* the three produce identical symptoms and are ordered by cost of checking; a rule change is a one-row query, a model change is not even in the candidate set here. *Follow-up:* what if all three are clean? *Red flag:* starting with retraining.

**Q6 (calculation).** Break-even: value 1500, delay 3.2 days, half-life 10 days, cost per review 22.60. What confirmation rate justifies a review?

*Checked answer:* `V_d = 1500*2^(-0.32) = 1201.60`; `p* = 22.60/1201.60 = 0.0188`. *Reasoning:* delay discounts value before the comparison, and the discount is exponential in `d/h_half`. *Follow-up:* what is `p*` at a 7-day delay? *Red flag:* dividing by 1500 and calling the answer conservative.

**Q7 (judgement).** Observed queue rate is 22% and break-even is 1.88%. What should the team spend the next marginal hour on?

*Checked answer:* neither a better model nor a wider queue by default — the audit of unreviewed alerts at 8%, which returns about 1,224 per hour against the queue's 912. *Reasoning:* the margin above break-even is 11.7x, so ranking is not the binding constraint; the binding constraint is knowledge about the excluded 1,700 rows. *Follow-up:* what changes if the unreviewed rate turns out to be 3%? *Red flag:* proposing a model upgrade without computing a marginal rate.

**Q8 (calculation).** The tail stratum confirms 2 in 450. Give a 95% interval and say what it licenses.

*Checked answer:* exact interval `[0.00054, 0.01596]`, normal approximation `[-0.0017, 0.0106]` which is invalid. It licenses the statement that the rate is below about 1.6% and nothing more precise. *Reasoning:* with two events the normal approximation is not usable; the exact interval is asymmetric by a factor of about 30 between its halves relative to the point estimate. *Follow-up:* how many rows to halve the interval width? *Red flag:* quoting `0.44% ± 0.61%`.

**Q9 (calculation).** How many rows do you need to estimate a rate near 0.22 to within ±0.05 and ±0.03 at 95%?

*Checked answer:* `1.96^2*0.22*0.78/0.0025 = 263.7 -> 264`; for ±0.03, `732.5 -> 733`. *Reasoning:* the width target is absolute, so the required `n` scales as `1/width^2`; narrowing by 40% nearly triples the rows. *Follow-up:* what Hoeffding gives for the same target? *Red flag:* assuming the audit can be sized for free.

**Q10 (diagnosis).** Raw agreement between two reviewers is 74% and kappa is 0.47. Is the rule reliable?

*Checked answer:* not established. With marginals 0.60 and 0.54 the expected agreement is 0.508, so 0.74 is only 0.232 above chance; the kappa interval is `[0.297, 0.646]`. *Reasoning:* raw agreement is inflated whenever both reviewers call most rows non-cases, which is the normal regime at low prevalence. *Follow-up:* what does 26% disagreement cost at 700 rows and 15 minutes of rework? *Red flag:* treating 0.74 as "three quarters reliable".

**Q11 (application).** How would you use capture–recapture, and what would make you distrust the result?

*Checked answer:* two independent passes, `N_hat = |A||B|/m`; distrust if the passes share a feed, a cut, or an analyst, because positive dependence biases `N_hat` low by `delta/(1+delta)` — 20% dependence gives a 16.7% understatement. *Reasoning:* the estimator assumes independence of selection, which is never verified by the data it produces. *Follow-up:* does the bias make coverage look better or worse? *Red flag:* presenting `N_hat` as a point estimate with no dependence scenario.

**Q12 (diagnosis).** Audit rows fall to zero in the two weeks with the longest queue. What is the measurement consequence?

*Checked answer:* `tau` loses its largest variance component precisely in the high-prevalence weeks, so prevalence and recall are understated exactly when they matter, and the resulting series has a level shift that looks like improvement. *Reasoning:* audit availability is correlated with the state of the system, which makes the missingness informative rather than random. *Follow-up:* how would you detect this from the data alone? *Red flag:* treating the gap as a reporting nuisance.

**Q13 (calculation).** Arrivals are 20 per hour, mean service 15.9 minutes, six reviewers pooled. What is the expected wait, and what does a seventh buy?

*Checked answer:* load `a = 20*0.265 = 5.3` erlangs; at `m = 6`, `rho = 0.883` and `W_q = 15.9` minutes; at `m = 7`, `rho = 0.757` and `W_q = 3.7` minutes. *Reasoning:* wait is convex in utilisation, so the seventh reviewer cuts the wait by 77% while adding 17% capacity. *Follow-up:* what does the eighth buy? *Red flag:* computing `1 - rho` and calling it spare capacity.

**Q14 (judgement).** A stratified Neyman allocation would cut the standard error by 13.7%. Why might you not adopt it?

*Checked answer:* because allocation also determines which strata are *estimable at all*, and the mid stratum's 250 rows are what make its 8% rate usable for the capacity argument; Neyman at 55 rows would leave that rate too noisy to support a decision even though the total's SE improves. *Reasoning:* the objective is the decision, not the total's variance; a variance-optimal design can be decision-suboptimal when a sub-estimate is load-bearing. *Follow-up:* what is the cost of the 13.7% loss? *Red flag:* optimising the printed statistic.

**Q15 (judgement).** The confirmation rule's specificity is 0.90 and the tail rate is 0.0044. The observed tail rate is 0.1033. What do you report?

*Checked answer:* that the tail rate is unidentified by this rule, because the observed rate is within 3% of the noise floor `1 - sp = 0.10`; the implied prevalence is `81600*0.1033 = 8,432` cases against an audit-based 363, and recall computed from the observed rate would be 0.018 rather than 0.236. *Reasoning:* with `se = 0.85` the compression factor in the tail is 23x, so no inversion is stable. *Follow-up:* what would make the tail estimable? *Red flag:* quoting the inverted 0.0044 as if it were measured at that precision.

## 12. Quick Reference Card

| Trigger | Rule | Assumption | Avoid | Verification | Owner |
|---|---|---|---|---|---|
| capacity plan | `C_inv = C*60/(s + e*g)` in investigations | escalation share measured over 4 weeks | quoting alerts as capacity | planned equals completed within 5% | operations |
| precision reported | numerator confirmed, denominator `k-u` | pending rows are logged | pending folded into the denominator | pending share printed beside the ratio | risk analytics |
| total cases needed | `SUM N_h p_h` with per-stratum variance | independent audit within strata | averaging stratum rates | variance share by stratum published | risk analytics |
| recall requested | ratio of census to total, variance from audit strata | audit independent of the score | `confirmed/reviewed` | interval width strictly positive | risk analytics |
| audit allocated | `n_h` proportional to `N_h sqrt(p(1-p))` | pilot rates available | proportional-to-size allocation | achieved SE within 5% of optimal | risk analytics |
| rate near break-even | compare `p` with `c_row/V_d` | delay-discounted value | comparing raw rates across strata | `p*` recomputed each quarter | decision owner |
| queue ageing | Erlang C at measured `rho` | pooled FIFO | reading `1-rho` as spare capacity | measured wait matches prediction | operations |
| reviewers disagree | kappa with interval plus raw agreement | marginals reported | raw percentage alone | disagreement hours in the capacity plan | quality lead |
| rule change proposed | sequential test at `p0` against `p1` | exchangeable rows within band | fixed-size audit after the fact | stops within 4 or about 97 rows | risk analytics |
| tail sample tiny | exact binomial interval | counts below about 10 | normal approximation | lower limit above zero | risk analytics |

Final checks before this card justifies a decision: capacity is in investigations with `e` published; every ratio names its frame and its pending count; the audit frame is drawn independently of the score and has a protected hour reserve; the confirmation rule carries a version and a `se`/`sp` estimate; `p*` is computed from the current cost and delay; the variance share shows which stratum controls the interval; and a zero-width interval anywhere is treated as a build failure rather than a result.
