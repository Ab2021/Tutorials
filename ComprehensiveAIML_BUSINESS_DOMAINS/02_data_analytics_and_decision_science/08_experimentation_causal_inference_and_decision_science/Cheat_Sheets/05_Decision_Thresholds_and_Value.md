---
collection: "08-experimentation-causal-inference-and-decision-science"
file_id: "05"
topic: "Decision Thresholds and Value"
level: 2
status: "draft"
version: "1.0"
created: "2026-11-09"
reviewed: null
review_trigger: "Change in loss magnitude, false-positive penalty, review cost, review capacity, base rate, or score calibration"
notation_style: "plain-text"
prerequisites: ["01_Experiment_Design_and_Guardrails.md", "binary classification scoring", "conditional probability", "expected value"]
pattern_tags: ["experimentation", "causal-inference", "decision-theory", "thresholds"]
source_status: "OPEN"
owner_role: "applied scientist, risk or growth decisioning"
source_index: "INDEX.md"
---

# Decision Thresholds and Value

## 1. File Header

> **Purpose:** Convert a calibrated probability into an operating point by deriving the
> threshold that maximises expected monetary value, then defend that point against
> capacity limits, cost misspecification, and base-rate change.
> **Prerequisites:** a scored population with a defined unit of analysis; expected value
> arithmetic; the ability to state a false-positive penalty in money.
> **Key outcome:** Given a loss per missed positive `L`, a penalty per false alarm `G`,
> an action cost `a`, and a calibrated score, you can derive the threshold, compute the
> value of the policy, size the review capacity it implies, and show which single
> assumption would reverse the decision.
> **This file does NOT cover:** how the score is estimated (see
> `04_Uplift_and_Treatment_Heterogeneity.md` for treatment-effect scores and
> `02_Causal_Diagrams_and_Adjustment.md` for identification), experiment assignment, or
> how a threshold is deployed as a config change.
> **Level:** 2 Core Methods
> **Version / reviewed:** 1.0 / not independently reviewed; all arithmetic below is
> author-checked, no operational result is claimed for any organisation.
> **Evidence:** `DERIVED` for the threshold algebra and the optimality results;
> `VERIFIED` for the worked arithmetic in Sections 5 and 9; `ILLUSTRATIVE` for every
> population, cost, and score table; `OPEN` for the loss-value and goodwill figures,
> which are the parameters a real deployment must obtain from finance and operations.
> **Retrieval cue:** If someone says "the model is good, where do we cut?", start at
> Section 4.2 for the derivation and Section 5 for the arithmetic. If a deployed policy
> is losing money, start at Section 8.

A threshold is the cheapest and most consequential artefact in a prediction pipeline. It
costs nothing to change and it routinely moves more money than a model upgrade. It is
also the artefact most often chosen by convention: 0.5 because it is the middle of the
unit interval, the 90th percentile because a previous team used it, or the point that
maximises F1 because F1 is what the library reports.

None of those are decision rules. A decision rule is derived from the money that flows
from each of the four cells of the confusion matrix, plus the cost of performing the
action at all. Once those numbers exist, the threshold is a closed-form expression, not
a search. When the numbers do not exist, the threshold cannot be justified and any
number chosen is a guess wearing a decimal point.

This sheet teaches the derivation, the three ways the derivation breaks (capacity,
calibration, and mechanism), and the arithmetic that lets a reader show, to the nearest
hundred money units, what a proposed threshold buys and what it gives up.

The running example is transaction review: a scored queue in which a flagged
transaction may be blocked or sent to an analyst. That setting is chosen because the
action plausibly does not change the underlying outcome — the fraud either happened or
it did not — which isolates the threshold question from the separate question of
treatment effect. Section 4.3 removes that convenience and shows what changes when the
action is persuasive rather than protective.

**Takeaway:** a threshold is a derived quantity. If you cannot name `L`, `G`, and `a` in
money, you have not chosen a threshold; you have chosen a percentile and inherited its
consequences.

## 2. Concept Map / Mental Model

The decision chain has six links. Each link can fail independently, and the failure
looks identical from the dashboard: the realised value is worse than the projected
value.

```text
[scored population: n, unit, window]
        |
        v
[is the score a probability?] --no--> [rank only: use capacity/quota rule]
        | yes
        v
[four-cell payoff table]
   act & positive   : +L - a
   act & negative   : -G - a
   skip & positive  :  0 - L   (relative to acting)
   skip & negative  :  0
        |
        v
[p* = (a + G) / (L + G)]  ------> [is that p reachable? check score support]
        |
        v
[capacity check: how many decisions does p* imply?]
        |-- exceeds capacity --> [quota rule: top-K by score, K = capacity]
        v
[value at the threshold: sum of per-decision EV]
        |
        v
[monitor: base rate, calibration, cost drift] --> [recompute p*, or hold]
```

Read the diagram as a decision tree, not a pipeline diagram. The first branch is the one
that most teams skip: a gradient-boosted score whose output happens to lie in [0, 1] is
a ranking device until it has been calibrated against observed frequencies. A ranking
device supports quota rules ("send the top 60,000 to review") but not level rules
("review anything above 0.046"). Confusing the two produces a policy that is stable in
rank order and arbitrary in level.

The second branch is the payoff table. It is the only place where business knowledge
enters the mathematics, and it is where most errors originate: `L` is usually quoted as
a gross loss when the relevant quantity is a net loss, and `G` is usually omitted
entirely because nobody owns "the cost of annoying a legitimate customer".

The third branch is arithmetic. Once `L`, `G`, and `a` are fixed, the threshold is
`(a + G) / (L + G)`. There is no fitting, no grid search, and no validation split. It is
worth stating plainly because the derivation is short enough to be argued about in a
meeting.

The fourth branch is the one that changes the *shape* of the answer rather than its
value. If the derived threshold implies more actions than the organisation can perform,
the binding constraint is capacity, and the correct rule becomes a quota: rank by score,
act on the top `K`. The threshold then becomes whatever score the `K`-th unit has, which
is an output of the ranking, not an input.

The fifth and sixth branches are value accounting and monitoring. Both are covered in
Sections 5 and 8; the point of putting them in the map is that a threshold without a
value statement is untestable, and a threshold without a monitoring plan silently
becomes wrong as the base rate moves.

A second mental model is worth carrying alongside the first: the *value curve*. Plot
total expected value against the threshold on the horizontal axis. The curve is not a
hill with a rounded top; it is a piecewise-linear path with a kink at the optimum, a
gentle slope on the over-strict side, and a steep slope on the over-loose side when `G`
is small. The asymmetry matters because an error in `L` or `G` shifts you along that
curve, and the loss from being wrong depends on which side you land.

```text
total value
    |
    |            optimum
    |              *
    |            * |
    |          *   |
    |        *     |
    |      *       *  <- gentle: each wrongly-skipped unit costs (pL - a - G)
    |    *           *
    |  *               *
    | *                  *   <- steep: each wrongly-acted unit costs (a + G - pL)
    +-----------------------------> threshold t
```

The steep side is the side with many negatives, so the units being added as you loosen
the threshold are low-`p` units and each one loses roughly `a + G`. The gentle side
subtracts high-`p` units, each worth roughly `pL - a - G`. Since `p` is large there, the
per-unit loss is large too — so which side is steeper depends on the numbers, and
Section 6 shows the general shape with the crossover located.

A third model is the *iso-value line*. In the plane with `p` on the horizontal axis and
the cost ratio `G/L` on the vertical, the set of thresholds with the same value is a
curve that bends toward the vertical axis as `G` grows. Moving from one business unit to
another is a move in that plane, and a threshold that does not move with the operating
point is not a policy; it is a constant.

**Takeaway:** the threshold is the last step of a chain whose first step is "is this a
probability", and the chain forks into a level rule or a quota rule before any arithmetic
happens.

## 3. Foundational Definitions

Each definition below states the unit of analysis, the operational test that tells you
whether the definition is being honoured, and a nonexample that looks like the concept
but is not.

1. **Decision unit.** The atomic thing about which one action is taken: one transaction,
   one account, one session, one member-month. Operational test: the number of decisions
   in the window equals the number of distinct decision units, with no unit appearing
   twice. Nonexample: a transaction table joined to a refunds table where one transaction
   yields three rows because it was partially refunded twice — the row count is not the
   decision count.

2. **Score.** A function of features available at decision time that maps a unit to a
   value used for ordering or for level comparison. Operational test: the feature set is
   reproducible from an immutable snapshot with the same timestamp boundary. Nonexample:
   a "risk score" that includes the outcome of the review it was supposed to trigger —
   that is a label leak, not a score.

3. **Calibrated probability `p`.** A score for which the observed event rate among units
   assigned value `p` equals `p`, within sampling error, in the population and window
   where it will be used. Operational test: a reliability table with ten bins, each bin's
   observed rate within its binomial confidence interval of the bin's mean score.
   Nonexample: a score with excellent ranking (high AUC) but whose top decile has mean
   score 0.30 and observed rate 0.09. The ranking is fine; the level is meaningless.

4. **Loss per missed positive `L`.** The net money lost when a unit is positive and no
   action is taken, measured against the counterfactual of having acted. Operational
   test: the figure is traceable to a finance schedule, and it nets recovery, insurance,
   and salvage. Nonexample: gross transaction value on a fraudulent charge that the
   acquirer reimburses at 90% — gross value overstates `L` by a factor of ten.

5. **Penalty per false alarm `G`.** The net money lost when a unit is negative and an
   action is taken anyway. Operational test: `G` includes every consequence the
   organisation actually bears — customer contact cost, churn effect, compensation,
   regulatory exposure. Nonexample: a "cost of a false positive" set to a nominal one
   unit because nobody measured the churn effect. Understating `G` biases the threshold
   downward, toward over-action.

6. **Action cost `a`.** The money spent to perform the action on one unit, incurred
   whether the unit is positive or negative. Operational test: `a` is derived from
   workload times a loaded rate, not from a vendor quote per case. Nonexample: analyst
   salary divided by headcount, ignoring that an analyst performs 150 reviews per day,
   not one.

7. **Operating point / threshold `t`.** The score level at or above which the action is
   taken. Operational test: `t` is stored with a version, a population scope, and an
   effective date. Nonexample: a threshold that exists only as a literal inside a
   notebook cell, so the production rule and the analysis cannot be shown to agree.

8. **Policy.** The pair (rule, threshold) together with the population it applies to and
   the actions it authorises. Operational test: two analysts reading the policy produce
   the same action for the same unit. Nonexample: "review high-risk transactions" — a
   goal, not a policy.

9. **Capacity `K`.** The maximum number of actions the operation can perform in the
   window, in units of decisions per window. Operational test: `K` is derived from staffed
   hours and measured handle time, and is re-measured when either changes. Nonexample: a
   headcount number used as capacity without converting through handle time.

10. **Quota rule.** A policy that acts on the top `K` units by score, where `K` is set by
    capacity rather than by a monetary level. Operational test: the implied score cut is
    recomputed each window and moves with the score distribution. Nonexample: fixing
    `K = 60,000` while volume doubles, which halves the effective fraction treated.

11. **Expected value of the policy.** The sum over decisions of the per-decision expected
    money consequence of the action the policy chooses, relative to the no-action
    baseline. Operational test: recomputing it from the confusion matrix reproduces the
    per-unit computation to rounding. Nonexample: reporting "loss prevented" without
    subtracting the cost of the actions that prevented nothing.

12. **Incremental (uplift) effect `tau`.** The change in outcome probability caused by
    the action, as distinct from the outcome probability itself. Operational test: `tau`
    is estimated from an experiment or an identified observational design with the
    decision unit as the randomisation unit. Nonexample: using predicted conversion
    probability as if it were the effect of a coupon. See
    `04_Uplift_and_Treatment_Heterogeneity.md`.

13. **Base rate `r`.** The unconditional event probability in the scored population and
    window. Operational test: `r` is recomputed from the same snapshot as the scores.
    Nonexample: quoting a training-set prevalence as the deployed base rate after a
    controls change.

14. **Calibration drift.** Change over time in the mapping from score to observed rate,
    holding the score distribution fixed. Operational test: reliability bins tracked per
    window with a pre-declared alarm band. Nonexample: monitoring only AUC, which can
    remain high while the level shifts by a factor of two.

15. **Shadow price of capacity.** The marginal value of one additional action slot at the
    capacity-constrained optimum: the per-decision EV of the best excluded unit.
    Operational test: it equals the EV of the unit ranked at position `K + 1`.
    Nonexample: average value per action, which is not marginal and overstates the worth
    of expanding capacity.

16. **Break-even parameter.** The value of a single input at which the policy's sign
    flips, holding everything else fixed. Operational test: computed in closed form and
    reported beside the policy. Nonexample: a sensitivity plot with no marked crossing
    point, which tells a reader that inputs matter without telling them how much.

**Takeaway:** `L`, `G`, and `a` are accounting quantities with owners, not modelling
quantities with defaults; the whole derivation is only as trustworthy as those three
numbers.

## 4. Core Formulas with Full Derivation

### 4.1 Base case: one action, one threshold, symmetric-incurred cost

**Validity before use.** Two actions only (act, skip); the action does not change the
probability of the event; `L`, `G`, and `a` are constants in money for every unit in the
population; decisions are evaluated one at a time with no binding resource constraint;
the score is a calibrated probability.

**Derivation.** Write the payoff of each action relative to skipping, per unit with
event probability `p`. Acting on a positive yields `+L` from avoided loss and costs `a`,
so `L - a`. Acting on a negative yields nothing and costs `a` plus the false-alarm
penalty `G`, so `-(a + G)`. Skipping yields zero against itself. Taking expectation over
the event indicator gives a linear function of `p`:

```text
FORMULA [A-01]:  EV_act(p) = p·(L − a) + (1 − p)·(−(a + G))
                          = p·L − p·a − a − G + p·a + p·G
                          = p·(L + G) − (a + G)

FORMULA [A-02]:  EV_skip(p) = 0

Setting EV_act(p) = EV_skip(p):
        p*·(L + G) = a + G
FORMULA [A-03]:  p* = (a + G) / (L + G)
```

Note the cancellations. The `-p·a` and `+p·a` terms cancel because `a` is incurred under
both outcomes once the decision to act is made. That is why `a` appears only as a shift
of the intercept and never multiplies `p`. An analyst who writes `EV = p(L - a) - (1-p)G`
charges `a` on the positive branch only and derives `p* = G/(L + G - a)`, which for the
example inputs is `5/296 = 0.0169` — less than half the correct value, and therefore a
policy that acts on the bottom deciles at a loss.

```text
WHERE:
  p        probability the unit is positive, in [0,1], calibrated
  L        net money lost if positive and not acted on           [money/unit]
  G        net money lost if negative and acted on               [money/unit]
  a        money spent to perform the action, either outcome      [money/unit]
  p*       indifference probability, dimensionless
CONSTRAINT:
  L > 0; G >= 0; a >= 0; p is calibrated; action does not alter p.
NUMERICAL EXAMPLE:
  a = 9, G = 5, L = 300
  p* = (9 + 5) / (300 + 5) = 14 / 305 = 0.0459016...
  at p = 0.088:  EV = 0.088·305 − 14 = 26.84 − 14 = +12.84
  at p = 0.045:  EV = 0.045·305 − 14 = 13.725 − 14 = −0.275
INTERPRETATION:
  The action pays for itself above 4.59% event probability. At 8.8% it returns
  12.84 money units per decision; at 4.5% it loses 0.275.
```

**Boundary/counterexample.** Set `G = 0` and `a = 0`: then `p* = 0` and the policy acts
on everything, which is correct — a free action with no downside should always be taken.
Set `L = 0`: then `p* = (a + G)/G > 1` whenever `a > 0`, so the threshold is unreachable
and the policy never acts, which is also correct — there is nothing to gain. Neither
degenerate case is a modelling failure; both are the formula reporting that the decision
is trivial. The failure case is `L` given as a gross figure while `a` and `G` are net,
which mixes accounting bases inside one expression.

The other boundary is a unit whose `p` exceeds `p*` but whose `L` is far below average —
formula A-01 uses one `L` for everyone, and Section 4.4 shows why that is the weakest
assumption in the chain.

### 4.2 General case: expected-cost minimisation and the four-cell accounting

**Validity before use.** Same structure as 4.1, but now derived from the confusion matrix
so that the cost terms can be audited cell by cell against finance. Still requires
constant per-unit costs and a calibrated score.

**Derivation.** Let `TP(t)`, `FP(t)`, `FN(t)`, `TN(t)` be counts at threshold `t`, with
`N = TP + FP + FN + TN`. Assign operating cost to each cell, not net gain: acting on a
positive costs `a` and its loss is avoided, so the cell carries cost `a`; acting on a
negative costs `a` and triggers the penalty, so `a + G`; skipping a positive carries the
loss `L`; skipping a negative carries `0`. Total operating cost as a function of `t` is
then

```text
FORMULA [A-04]:  EC(t) = a·(TP + FP) + G·FP + L·FN

Dividing by N and using TP/N = P(p ≥ t, y = 1), FP/N = P(p ≥ t, y = 0),
FN/N = P(p < t, y = 1), and E[y|p] = p for a calibrated score:

FORMULA [A-05]:  EC(t)/N = E[ 1{p ≥ t}·(a + (1 − y)·G) + 1{p < t}·y·L ]
```

Because `EC(t)/N` equals the per-unit cost sum exactly, and the skip branch is
`E[1{p<t}·p·L]`, the objective separates into independent per-unit terms. Each unit is
therefore optimised by choosing the cheaper branch, either `a + (1 - p)·G` (act) or `p·L`
(skip), and a per-unit minimiser minimises the total.

Two subtleties in A-04 are worth stating because they are where cell accounting goes
wrong. First, `a` multiplies `TP + FP`, that is, every action taken, not every positive
found: the review cost is paid on the 124,180 legitimate alerts as well as the 15,820
fraudulent ones. Second, `L` multiplies `FN` only, because the loss on caught positives is
already excluded from the cost side by construction — it was never incurred. Writing
`(a - L)·TP + (a + G)·FP + L·FN` instead charges the loss against the positives that were
saved and produces a total that is not comparable to any P&L line.

```text
FORMULA [A-06]:  act  iff  a + (1 − p)·G  <  p·L
                        iff  a + G  <  p·(L + G)
                        iff  p  >  (a + G)/(L + G)
```

So A-03 is recovered from the four-cell accounting, and the accounting version can be
reconciled line by line with a finance schedule.

```text
WHERE:
  TP, FP, FN, TN   counts at threshold t                       [units]
  EC(t)            total expected cost at t                    [money]
  all other symbols as in 4.1
CONSTRAINT:
  Costs constant per unit; score calibrated so E[y|p] = p;
  the four cells are disjoint and exhaust the population.
NUMERICAL EXAMPLE:
  N = 200,000; at t = 0.046 the policy acts on 140,000 units:
  TP = 15,820, FP = 124,180, FN = 1,940, TN = 58,060
  EC = 9(15,820 + 124,180) + 5(124,180) + 300(1,940)
     = 1,260,000 + 620,900 + 582,000
     = 2,462,900
  Do-nothing cost = L·(TP + FN) = 300 × 17,760 = 5,328,000
  Net gain  = 5,328,000 − 2,462,900 = 2,865,100, matching Section 5.3.
INTERPRETATION:
  The four-cell form is the auditable one; the algebraic form is the fast one. Both
  must agree, and disagreement means a cell is mis-populated.
```

**Boundary/counterexample.** If the four cells do not sum to `N`, the population is not
partitioned — typically because one unit appears in two score bins after a join fan-out:
`15,820 + 124,180 + 1,940 + 58,060 = 200,000`, and any other total means a fan-out or a
dropped unit. If the `L` term is applied to `TP` as well as `FN`, the accounting
double-counts avoided loss and the total drifts toward the do-nothing figure. Both errors
are detectable by requiring that the identity `TP + FP + FN + TN = N` holds and that
`EC(t)` computed from A-04 equals `EC(t)` computed from the pointwise expression A-05
term by term on a sample of units.

### 4.3 Variants

**Variant 1 — capacity-constrained quota.** When the derived threshold implies more
actions than `K`, the problem changes from "is this unit worth acting on" to "which `K`
units". Because per-unit cost is constant and per-unit value is increasing in `p`, the
optimal selection is the `K` largest scores, and the optimum is attained by a greedy
scan after sorting (an exchange argument: any selection that excludes a higher-scored
unit while including a lower-scored one can be improved by swapping).

```text
FORMULA [A-07]:  act on units with rank_i(p) ≤ K;  t_eff = p_(K)
WHERE:
  p_(K)   the K-th largest score (an order statistic, not a fitted parameter)
  K       capacity in actions per window                             [units/window]
CONSTRAINT:
  Constant per-unit costs; value monotone increasing in p; K known and hard.
NUMERICAL EXAMPLE:
  K = 60,000 on the Section 5 table selects deciles 8–10, whose lowest score is
  0.126; total value 2,014,800 versus 2,865,100 unconstrained, a shortfall of
  850,300 for 80,000 foregone actions.
INTERPRETATION:
  Under a hard capacity the threshold is an output. Fixing it as an input silently
  under- or over-uses the operation.
```

The shadow price follows directly: the marginal action slot is worth the per-decision EV
of the best excluded unit, which in the example is the 100,000th-ranked unit at 7.96
money units, falling to 3.69 at rank 140,000. That number, not the average value per
action, is what justifies a headcount request.

**Variant 2 — three actions with nested thresholds.** Real operations usually have three
actions: allow, review manually, block automatically. Let `a_r` be the review cost, `G` the
penalty when a legitimate unit is blocked, and note that a review catches the positive
with certainty in this idealisation.

```text
FORMULA [A-08]:  cost(allow)  = p·L
                 cost(review) = a_r
                 cost(block)  = (1 − p)·G

  choose allow  iff  p·L < a_r  and  p·L < (1 − p)·G
  choose review iff  a_r < p·L  and  a_r < (1 − p)·G
  choose block  iff  (1 − p)·G < a_r  and  (1 − p)·G < p·L

  Solving the three pairwise boundaries:
    allow|review boundary:  p·L = a_r        -> p = a_r / L
    review|block boundary:  a_r = (1 − p)·G  -> p = 1 − a_r / G
    allow|block (direct):   p·L = (1 − p)·G  -> p = G / (L + G)

FORMULA [A-09]:  allow  for p < a_r / L
                 review for a_r / L ≤ p < 1 − a_r / G
                 block  for p ≥ 1 − a_r / G
                 (a non-empty block band exists iff 1 − a_r/G > a_r/L,
                  i.e. iff a_r < G·L / (L + G))
CONSTRAINT:
  Review detects the positive with probability 1; block is irreversible; a_r, G, L
  constant; the three actions are mutually exclusive.
NUMERICAL EXAMPLE:
  a_r = 9, L = 300, G = 5:
    a_r/L = 0.030; 1 − a_r/G = 1 − 1.8 = −0.8  → block band empty.
    Threshold test: G·L/(L + G) = 5·300/305 = 4.918 < a_r = 9, so no block band.
  With G = 60 instead:
    a_r/L = 0.030; 1 − a_r/G = 0.85           → block band [0.85, 1].
    Threshold test: 60·300/360 = 50 > 9, so the band exists.
INTERPRETATION:
  Automatic blocking appears only when the false-block penalty G exceeds the review
  cost by enough to justify skipping the review; with G = 5 there is no such region.
```

The condition `a_r < G·L/(L + G)` is worth reading twice, because the intuition is easy to
state backwards. Blocking does not become attractive as `G` falls — it becomes attractive as
`G` *rises*, since a large `G` means the false-block penalty is severe and yet blocking is
being compared against the alternative of doing nothing. What actually drives the band is
the ratio: the block band exists when the review is cheap enough relative to the penalty
that a review would always be worth performing before blocking. When `a_r ≥ G·L/(L + G)`,
review is so cheap relative to the consequences that there is never a score high enough to
justify skipping it, and the policy collapses to a two-action rule. Making automatic
blocking cheaper in that regime changes nothing, which is the counter-intuitive part: the
expensive component is not blocking, it is looking.

The direct allow|block boundary `G/(L + G)` is the threshold a two-action policy would use
with `a = 0`, which is the consistency check for A-09: setting `a_r = 0` collapses the
review band to a point and recovers A-03 with `a = 0`. Run that check whenever a
multi-action band is derived; if the collapse does not reproduce the two-action threshold,
one of the three branch costs is wrong.

**Variant 3 — persuasive action, threshold on `tau` not on `p`.** When the action changes
the outcome, the decision variable is the incremental effect, not the baseline risk.

```text
FORMULA [A-10]:  EV_act(x) − EV_skip(x) = tau(x)·M − a − G·(1 − p(x))
                 with a costless action and no false-alarm penalty:
FORMULA [A-11]:  act iff tau(x) > a / M
WHERE:
  tau(x)  CATE: change in P(outcome) caused by the action  [probability]
  M       gross margin of the outcome                      [money/unit]
  a       action cost                                      [money/unit]
CONSTRAINT:
  tau identified from randomisation or a defensible design; M constant;
  the action has no penalty when it fails.
NUMERICAL EXAMPLE:
  M = 90, a = 2.4  →  tau* = 2.4 / 90 = 0.026667
  segment A: p = 0.152, tau = 0.026  →  0.026·90 = 2.34 < 2.4  → skip
  segment B: p = 0.126, tau = 0.028  →  0.028·90 = 2.52 > 2.4  → act
INTERPRETATION:
  Segment A has the higher baseline risk and the lower value. Ranking on p and
  ranking on tau disagree, and only one of them answers the question.
```

The example is honest about its own limits: `p` and `tau` are not interchangeable, and a
policy built on `p` is not a rough approximation to a policy built on `tau` — it is an
answer to a different question. The reason the two rankings coincide often enough to
create complacency is that baseline risk and treatment effect are usually positively
correlated; the reason the coincidence is dangerous is that the exceptions cluster in the
highest-value segments.

**Variant 4 — base-rate correction before thresholding.** A score calibrated in one
population keeps its likelihood ratio but not its level when the base rate changes.

```text
FORMULA [A-12]:  odds_new = odds_old · [ r_new/(1 − r_new) ] / [ r_old/(1 − r_old) ]
                 p_new    = odds_new / (1 + odds_new)
WHERE:
  r_old, r_new   base rates in calibration and deployment populations
  odds_old       p_old / (1 − p_old)
CONSTRAINT:
  The score is a likelihood ratio that transfers; only the prior changes. If the
  feature-outcome relationship also changed, this correction is invalid.
NUMERICAL EXAMPLE:
  r_old = 0.0888, r_new = 0.045
  factor = (0.045/0.955) / (0.0888/0.9112) = 0.047120/0.097454 = 0.48351
  p_old = 0.088  →  odds 0.096491 → 0.046647 → p_new = 0.044569
  p_old = 0.190  →  odds 0.234568 → 0.113423 → p_new = 0.101862
INTERPRETATION:
  The mid-score unit crosses below p* = 0.0459; the top unit remains well above.
  The correction changes the membership of the acted set, not just its size.
```

A prior-shift correction is a diagnostic, not a substitute for recalibration. It tells
you the direction and rough size of the bias under a strong assumption (score transfers
as a likelihood ratio); it does not restore the score's meaning. The honest action when
`r` has moved by a factor of two is to hold the policy, remeasure, and recalibrate.

**Variant 5 — break-even in each parameter.** Every threshold decision has three
break-evens that a reader should see before approving it.

```text
FORMULA [A-13]:  C*_unit = p·(L + G) − G          (max action cost that still pays)
FORMULA [A-14]:  L*      = (a + G)/p − G          (min loss that justifies acting)
FORMULA [A-15]:  G*      = (p·L − a)/(1 − p)      (max false-alarm penalty tolerable)
CONSTRAINT:
  Other parameters held fixed; p is the score at the proposed operating point.
NUMERICAL EXAMPLE:
  p = 0.126, L = 300, G = 5, a = 9:
    C* = 0.126·305 − 5 = 38.43 − 5 = 33.43
    L* = 14/0.126 − 5 = 111.11 − 5 = 106.11
    G* = (37.8 − 9)/0.874 = 32.95
INTERPRETATION:
  The policy survives a tripling of action cost, a two-thirds fall in loss per case,
  or a six-fold rise in the false-alarm penalty. Any claim outside those bands needs
  a new derivation, not a new opinion.
```

### 4.4 Limits and counterexamples

**Limit 1 — the threshold is a level, so it inherits every calibration error.** Suppose
the score is a well-ranked but uncalibrated output whose top decile mean is 0.19 while
the observed rate is 0.06. Applying `p* = 0.0459` to the raw score treats roughly the top
60% of the population instead of the intended 70% of positives being captured with a
specific alert volume. The threshold is arithmetically correct and operationally
meaningless. The discriminating test is a reliability table, not an AUC.

**Limit 2 — heavy-tailed `L` breaks the expected-value ranking under a constraint.**
A-01 replaces a distribution of losses with its mean. That is exact for the unconstrained
sum, but under a capacity constraint the ranking by mean loss is not the ranking by
realised value when loss variance is large and the decision maker faces a solvency or
regulatory limit. The correct object in that case is a quantile or a conditional
tail expectation of the loss, and the threshold is derived from the constrained problem,
not from A-03. A counterexample: two units with identical mean loss 300, one certain and
one a 1-in-1,000 event of 300,000. With a capital constraint that a single 300,000 hit
breaches, the mean-based policy is the wrong policy.

**Limit 3 — the constant-cost assumption fails under heterogeneity.** If `a` varies by
region or channel, the threshold is not a single number; it is a schedule. Substituting
the average `a` into A-03 over-acts where `a` is high and under-acts where `a` is low.
The correct treatment is per-stratum derivation, which is the same formula applied within
each stratum, with the additional check that strata are defined by variables available at
decision time.

**Limit 4 — non-independent decisions.** When one account holds many transactions, acting
on each transaction independently double-counts the action cost and, worse, treats the
account-level consequence `G` as if it were paid per row. The threshold should be applied
at the decision unit, which means aggregating scores to the account before thresholding.
This is not a numerical correction; it is a change of unit, and it changes `L`, `G`, and
`a` simultaneously.

**Limit 5 — the score does not transfer across populations at all.** A-12 corrects the
prior and leaves the likelihood ratio alone. When the feature-outcome relationship changes
— a new payment rail, a fraud ring moving channels — no scalar correction restores
validity. The test is stratum-level calibration in the new population; if the reliability
bins are non-monotone in the new population, the score must be rebuilt.

**Limit 6 — threshold on a proxy objective.** Maximising expected value of the *measured*
outcome is not the same as maximising the value of the *intended* outcome when the label
is a proxy. A fraud label that records "confirmed by an analyst" is only produced for
reviewed cases, so unreviewed fraud is invisible and `p` is biased downward exactly where
the policy does not look. The correction is a review-capacity experiment or an
inverse-propensity reweighting, not a threshold change.

**Takeaway:** A-03 is exact under six conditions and the honest answer to "where do we
cut" is a threshold plus the list of conditions that currently fail.

## 5. Worked Numerical Example

**Evidence:** every population, score, and cost below is `ILLUSTRATIVE` teaching input.
The arithmetic is `VERIFIED` by hand and cross-checked by two independent routes
(per-decile sum and confusion-matrix totals). No organisation's data is represented, and
no claim is made that a 2.5-minute review handle time or a 300-unit average fraud loss
matches any real operation.

### 5.1 The inputs

A scored queue of `n = 200,000` transactions over a 28-day window, split into ten
score deciles of exactly 20,000 transactions each. Mean calibrated probability per
decile:

| Decile | Mean `p` | Transactions | Expected positives |
|---:|---:|---:|---:|
| 1 | 0.020 | 20,000 | 400 |
| 2 | 0.032 | 20,000 | 640 |
| 3 | 0.045 | 20,000 | 900 |
| 4 | 0.058 | 20,000 | 1,160 |
| 5 | 0.072 | 20,000 | 1,440 |
| 6 | 0.088 | 20,000 | 1,760 |
| 7 | 0.105 | 20,000 | 2,100 |
| 8 | 0.126 | 20,000 | 2,520 |
| 9 | 0.152 | 20,000 | 3,040 |
| 10 | 0.190 | 20,000 | 3,800 |
| **All** | **0.0888** | **200,000** | **17,760** |

Check the mean: `0.020 + 0.032 + 0.045 + 0.058 + 0.072 + 0.088 + 0.105 + 0.126 + 0.152 +
0.190 = 0.888`; divided by ten deciles, `0.0888`. Expected positives: `0.0888 × 200,000 =
17,760`, and the column sums to the same figure, so the table is internally consistent.

Payoffs, in money units per transaction:

- `L = 300` net loss avoided when a fraudulent transaction is blocked and not reimbursed.
- `G = 5` net cost when a legitimate transaction is declined (support contact, goodwill,
  small churn effect amortised).
- `a = 9` review cost, derived as 2.5 minutes of analyst time at a loaded rate of 216 per
  hour: `2.5/60 × 216 = 9.0`.

### 5.2 Derive the threshold

```text
p* = (a + G) / (L + G)
   = (9 + 5) / (300 + 5)
   = 14 / 305
   = 0.0459016...
```

Rounded to three decimals for communication: `p* = 0.046`. Keep the unrounded value for
comparisons; the difference matters at decile 3, whose mean is 0.045.

### 5.3 Value each decile

Per-transaction expected value from A-01, `EV = p·305 − 14`:

| Decile | `p` | `p·305` | `−14` | EV/unit | Act? | EV × 20,000 |
|---:|---:|---:|---:|---:|:---:|---:|
| 1 | 0.020 | 6.100 | −14 | −7.900 | no | 0 |
| 2 | 0.032 | 9.760 | −14 | −4.240 | no | 0 |
| 3 | 0.045 | 13.725 | −14 | −0.275 | no | 0 |
| 4 | 0.058 | 17.690 | −14 | +3.690 | yes | 73,800 |
| 5 | 0.072 | 21.960 | −14 | +7.960 | yes | 159,200 |
| 6 | 0.088 | 26.840 | −14 | +12.840 | yes | 256,800 |
| 7 | 0.105 | 32.025 | −14 | +18.025 | yes | 360,500 |
| 8 | 0.126 | 38.430 | −14 | +24.430 | yes | 488,600 |
| 9 | 0.152 | 46.360 | −14 | +32.360 | yes | 647,200 |
| 10 | 0.190 | 57.950 | −14 | +43.950 | yes | 879,000 |

The acted set is deciles 4 through 10: 140,000 transactions. Summing the right column:
`73,800 + 159,200 = 233,000`; `+256,800 = 489,800`; `+360,500 = 850,300`; `+488,600 =
1,338,900`; `+647,200 = 1,986,100`; `+879,000 = 2,865,100`.

### 5.4 Independent check through the confusion matrix

Acting on 140,000 with mean `p = 0.791/7 = 0.113`:

- Expected positives acted: `140,000 × 0.113 = 15,820`.
- Expected negatives acted: `140,000 − 15,820 = 124,180`.
- Loss avoided: `15,820 × 300 = 4,746,000`.
- Review cost: `140,000 × 9 = 1,260,000`.
- False-alarm cost: `124,180 × 5 = 620,900`.
- Net: `4,746,000 − 1,260,000 − 620,900 = 2,865,100`.

The two routes agree exactly, which is the check required before any number leaves the
document. If they differed, the error would be in either the decile mean (which must be
weighted by 20,000, not averaged unweighted) or the false-alarm count (which must be
`acted − positives`, not `acted × (1 − mean p)` computed with a rounded `p`).

### 5.5 What the alternatives cost

**Treat everything.** `EV` per transaction at `p = 0.0888` is `0.0888 × 305 − 14 = 27.084
− 14 = 13.084`, so `200,000 × 13.084 = 2,616,800`. That is `248,300` worse than the
thresholded policy. The shortfall is exactly the three excluded deciles:
`(7.900 + 4.240 + 0.275) × 20,000 = 12.415 × 20,000 = 248,300`. Consistency confirmed.

**Review nothing.** Value is zero against the baseline, and expected fraud loss is
`200,000 × 0.0888 × 300 = 5,328,000`. The policy prevents `4,746,000` of that, or 89.1%,
while declining `124,180` legitimate transactions.

**Cut at 0.16 instead of 0.046.** Only decile 10 is acted on: `879,000`. The cost of the
over-strict threshold is `2,865,100 − 879,000 = 1,986,100`, eight times the cost of the
over-loose error computed above. This asymmetry is the reason the optimum should be
approached from the loose side and tightened only on evidence.

**Random selection of the same 140,000.** Expected positives `= 140,000 × 0.0888 =
12,432`; loss avoided `12,432 × 300 = 3,729,600`; net `= 3,729,600 − 1,260,000 −
(127,568 × 5 = 637,840) = 1,831,760`. The model's value over random selection at equal
volume is `2,865,100 − 1,831,760 = 1,033,340`. That figure is the quantity a model
upgrade must beat, and it is also the correct answer to "what is the score worth".

### 5.6 Capacity binding

Review capacity is `K = 60,000` per window, from 8 analysts × 150 reviews/day × 50 days of
staffed time across the window, rounded. The quota rule acts on deciles 8–10:

- Value: `488,600 + 647,200 + 879,000 = 2,014,800`.
- Shortfall against unconstrained: `850,300`.
- Per-action value: `2,014,800 / 60,000 = 33.58`, versus `2,865,100 / 140,000 = 20.46`
  unconstrained.
- Precision: `60,000 × 0.156 = 9,360` prevented of `60,000` alerts, i.e. 15.6%.

Precision is *higher* under the tighter policy, and total value is *lower*. Anyone
optimising precision would keep tightening, which is why precision must not be the
objective when the marginal action has positive value. The stopping rule is the marginal
EV, not the precision.

### 5.7 Break-evens for this policy

Using A-13 to A-15 at the marginal decile 4 (`p = 0.058`), which is the unit that decides
whether the policy is worth having at all:

```text
C* = 0.058 × 305 − 5 = 17.69 − 5 = 12.69
L* = 14 / 0.058 − 5 = 241.38 − 5 = 236.38
G* = (0.058 × 300 − 9) / (1 − 0.058) = (17.4 − 9) / 0.942 = 8.92
```

Reading these: the review cost could rise from 9 to 12.69 before decile 4 stops paying;
the loss per case could fall from 300 to 236 before decile 4 stops paying; the false-alarm
penalty could rise from 5 to 8.92 before decile 4 stops paying. Decile 7, by contrast, is
robust: its `G*` is `(0.105 × 300 − 9)/0.895 = (31.5 − 9)/0.895 = 25.14`, a five-fold
increase. A policy stated as "act above 0.046" is therefore really a statement that `G`
is below 8.92, and the approval conversation should be about `G`, not about the threshold.

### 5.8 What changes if the base rate halves

If the deployed base rate is 0.045 rather than 0.0888 and the score transfers as a
likelihood ratio, A-12 gives a factor of 0.48351 on the odds. Recomputing deciles 6 to 10:

| Decile | `p_old` | `odds_old` | `odds_new` | `p_new` | EV at `p_new` | Act? |
|---:|---:|---:|---:|---:|---:|:---:|
| 6 | 0.088 | 0.096491 | 0.046655 | 0.044575 | −0.405 | no |
| 7 | 0.105 | 0.117318 | 0.056725 | 0.053680 | +2.373 | yes |
| 8 | 0.126 | 0.144165 | 0.069706 | 0.065164 | +5.875 | yes |
| 9 | 0.152 | 0.179245 | 0.086668 | 0.079756 | +10.325 | yes |
| 10 | 0.190 | 0.234568 | 0.113417 | 0.101864 | +17.069 | yes |

Value of the corrected policy: `(2.373 + 5.875 + 10.325 + 17.069) × 20,000 = 35.237 ×
20,000 = 704,740`. Against the `2,865,100` projected at the old base rate, that is a
75.4% overstatement, and the acted set shrinks from seven deciles to four.

This is the single most important number in the sheet: a threshold derived at one base
rate and applied at another does not degrade gracefully. The reason is structural — `EV`
is linear in `p` with a negative intercept, so halving `p` more than halves `EV` in the
region near the threshold, where the intercept dominates. The mechanical prior
correction above isolates that effect; a real deployment would recalibrate the score and
re-derive, and the corrected table is a bound on the bias rather than a forecast.

**Takeaway:** the arithmetic takes five minutes and the parameter elicitation takes five
weeks; the ratio is the reason thresholds are set by convention.

## 6. Visual / Geometric Intuition

### 6.1 The per-decision value line

`EV(p) = p(L + G) − (a + G)` is a straight line in `p` with slope `L + G` and
intercept `−(a + G)`. Its zero crossing is the threshold. Because both parameters enter
as sums, the line is only mildly sensitive to `G` when `G << L`, which is why the
threshold moves so little when the false-alarm estimate is refined — and why a large `G`
changes everything.

```text
EV
 |
 |                       /
 |                      /   slope = L + G = 305
 |                     /
 |                    /
 |                   /
 +---------*--------+------------> p
 |        0.0459   / 0.058
 |               /
 |              /
 |             /
 |            /  intercept = -(a+G) = -14
```

Read the slope as "money per unit of probability". A slope of 305 means each additional
percentage point of event probability is worth 3.05 money units per decision. That
single number lets a reader sanity-check any proposed threshold: moving the cut by five
percentage points must change value by about 15 per decision passing the cut.

### 6.2 Value against threshold

Total value as a function of the cut is a step function whose steps are the decile masses.
It rises while the marginal decile has positive EV, plateaus, then falls as high-value
deciles are excluded.

```text
value (millions)
 2.87 |                ####
      |              ##    ##
 2.61 |            ##        ##
      |          ##            ##
 2.01 |        ##                ##
      |      ##                    ###
 1.35 |    ##                         ###
      |  ##                              ####
 0.88 |##                                    ####
      +----+----+----+----+----+----+----+----+--> cut
         0.02 0.045 0.072 0.105 0.126 0.152 0.190
                  ^ optimum plateau begins at the last decile mean before the
                    first negative decile, i.e. at 0.046 in continuous terms
```

The plateau is a feature: within a range of cuts the value changes by less than the
uncertainty in `L`. Reporting a single optimum to three decimals overstates precision
when `L` is known to ±20%. Report the plateau, then use the operational costs — analyst
supply, customer impact — to pick a point inside it.

### 6.3 The confusion matrix as a money diagram

```text
                    actual positive        actual negative
                 +----------------------+----------------------+
   act           |  +L - a
                 |  cost avoided less   |  -G - a
                 |  action cost         |  penalty plus action
                 +----------------------+----------------------+
   skip          |  -L                  |   0
                 |  loss taken          |   baseline
                 +----------------------+----------------------+
```

Each row is a decision; each column is a state of the world. The threshold chooses how
much mass from the left column to move into the top row, and it cannot avoid dragging
mass from the right column with it. The ratio of the two dragged quantities is the
likelihood ratio at the threshold, which is why the optimal cut is where the ratio equals
the cost ratio `(a + G)/L` in the unnormalised form — the probability form A-03 is the
same statement after applying Bayes' rule.

### 6.4 ROC with iso-cost lines

The operating point is not "high AUC"; it is a point on the ROC where the tangent has a
specific slope determined by the costs.

```text
 TPR
 1.0 |                   ....------*
     |              ..---          |
     |          ..--               |  iso-cost lines, slope
     |       .-                    |  = (1 - r)/r · (a + G)/L
 0.5 |     .-             ,*       |
     |    .             ,'         |
     |   .            ,'           |
     |  .          ,-'             |
 0.0 +----------------------------+----> FPR
     0                          1.0
```

Two points on the same ROC curve have the same AUC and different value. The AUC is the
probability that a random positive outranks a random negative — a rank statement with no
operating point in it. A model with a higher AUC can have a lower value at the operating
point that the business can afford, which is why "we improved AUC" and "we improved the
decision" are different claims.

### 6.5 Score distribution and the two error masses

```text
density
  |      negatives            positives
  |   ####                    ####
  |  ##  ####              ####  ##
  | ##      ####        ####      ##
  |##          ####  ####          ##
  |              ######              ##
  +------------------|------------------> score
                     t
       FP mass <---- | ----> FN mass
```

Moving `t` left trades false negatives for false positives at a rate equal to the local
density ratio. The optimum is where the exchange rate equals the cost ratio. When the
distributions overlap heavily — low separability — the exchange rate is near one and both
error masses are large; no threshold can fix that, which is the correct moment to say
"the model cannot support this policy" rather than to keep tuning the cut.

### 6.6 Iso-value contours in the cost plane

```text
G/L
 0.30 |  \        \         \
 0.20 |    \ p*=0.10 \        \
 0.10 |      \        \ p*=0.05 \
 0.05 |        \        \        \
 0.02 |          \ p*=0.02\       \
      +------------------------------- > p
       0.02   0.05   0.10   0.20   0.30
```

Each curve is a set of `(p, G/L)` pairs worth the same to act on. Moving right along a
curve means accepting a lower-probability unit because its false-alarm cost is lower.
Two business lines with different `G/L` sit on different curves, and a single shared
threshold is a decision to mis-price one of them.

**Takeaway:** the threshold is where a line through the origin crosses a cost; draw the
line and the crossing is visible, and so is every assumption that moves them.

## 7. Comparison / Selection Table

| Rule family | Use when | Avoid when | Disqualifier | Fallback |
|---|---|---|---|---|
| Level rule on calibrated `p`, A-03 | Costs are known in money and the score is calibrated | Score is rank-only or base rate has moved | No reliability table | Recalibrate, then re-derive |
| Quota rule (top-`K`), A-07 | A hard capacity binds before the monetary optimum | Capacity is elastic or queued | `K` unknown or unreviewed | Compute monetary cutoff, compare |
| Cost-weighted ROC operating point | Two-class, single action, costs per class known | More than two actions or per-unit cost varies | Class costs not separable | Four-cell accounting, A-04 |
| Three-action band, A-09 | Allow/review/block with distinct costs | Actions interact or review is not reliable | `G < 2·a_r` makes block dominated | Collapse to two actions |
| Uplift threshold on `tau`, A-11 | The action changes the outcome and `tau` is identified | `tau` estimated from observational proxies | No experiment or design | Target on `p`, label it as such |
| Prior-shift correction, A-12 | Base rate moved, score transfers as a likelihood ratio | Feature-outcome relation also moved | Non-monotone reliability in new population | Rebuild or recalibrate the score |
| Risk-constrained rule (quantile / CVaR) | Loss tail breaches a capital or regulatory limit | Losses are bounded and thin-tailed | Constraint not quantified | Mean-EV rule with a stress test |
| Human-in-the-loop triage with differential capacity | Actions have different costs and different service levels | Volume exceeds total capacity | No measured handle time | Quota rule by service level |
| Bandit / adaptive allocation | Threshold must be learned while operating | Costs asymmetric and errors irreversible | Regret budget undefined | Fixed threshold with periodic review |

Three selection notes that the table cannot carry:

First, the level rule and the quota rule are not competitors; they answer different
questions. Derive the level rule first because it produces a number the business can
argue with (`p* = 0.046` means "below 4.6% we lose money"). Then compare the implied volume
against capacity. If capacity binds, the quota rule governs and the level rule becomes the
diagnostic that says how much value the capacity limit destroys — which is precisely the
number needed to justify staffing.

Second, the cost-weighted ROC formulation and the four-cell accounting are algebraically
the same rule expressed in different currencies (rates versus money). Prefer the money
form for approval and the rate form for plotting, and require them to agree at the chosen
point. When they disagree, one of the two normalisations is wrong, most often because the
positive prevalence used to convert rates to counts was taken from a different window.

Third, the uplift threshold is not a refinement of the level rule; it is a replacement
that applies when the action has a causal effect. Choosing the level rule for a persuasive
action is a category error, not a conservative approximation. The tell is a policy that
targets the highest-baseline-risk or highest-propensity segment and then reports a lift
that is smaller than the segment's baseline rate.

**Takeaway:** select the rule by asking what the action does to the outcome and what
resource binds, not by asking which rule the tooling supports.

## 8. Failure Modes & Edge Cases

### FAIL-01 — Threshold applied to a rank score as if it were a probability

**Symptom.** The policy acts on a fraction of the population that bears no relation to
`p*`. A cut at 0.046 flags 85% of units in one month and 40% in another.

**Competing causes.** (a) The score is a raw model output with no calibration step;
(b) calibration exists but was fitted on a differently filtered training set; (c) the
threshold was derived from a rate table and applied to a rank table; (d) a preprocessing
change (missing-value sentinel, feature scaling) silently rescaled the output.

**Discriminating check.** Build a ten-bin reliability table on the deployment window:
bin by score, compute observed event rate per bin, and compare with the bin's mean score.
If observed rates are systematically below bin means by a multiplicative factor, the score
overstates risk on a common scale (cause a or b). If bin means are not monotone in
observed rate, the ranking itself is broken (cause d). If the reliability is fine but the
flagged fraction moves, the threshold is being applied to a different column (cause c).

**Bounded fix.** Hold the policy, ship a calibration layer fitted on the deployment window
with a holdout, and re-derive `p*` from the same `L`, `G`, `a`. Do not rescale the
threshold by the ratio of base rates as a shortcut; that is A-12's assumption applied
without checking it.

**Verification.** Post-calibration, the reliability table must show observed rates inside
the binomial interval of bin means for at least nine of ten bins, and the flagged fraction
must match the predicted `P(p ≥ t)` within two percentage points on a held-out window.

### FAIL-02 — Threshold ported across populations with different base rates

**Symptom.** A policy tuned on one channel, region, or cohort loses money when applied to
another, while every offline metric looks unchanged.

**Competing causes.** (a) Base rate differs and the score is uncalibrated for the new
prior; (b) `L` differs because recovery rates differ by channel; (c) `a` differs because
handle time differs; (d) `G` differs because the customer relationship differs.

**Discriminating check.** Decompose the loss: recompute `EC(t)` per population with its
own `L`, `G`, `a`. If value is negative only in the new population and its base rate is
materially lower, (a) dominates. If base rates match but value still differs, compare
`G` and `a` estimates; a doubling of handle time moves `p*` from 0.046 to `(18 + 5)/305 =
0.0754`, which alone can empty the acted set at the margin.

**Bounded fix.** Derive per-population thresholds where the parameters differ, or
normalise by calibration and keep one threshold only if the calibrated score is
demonstrably transportable. Version the policy per population so a change is visible.

**Verification.** Each population has its own reliability table, its own parameter
schedule, and a documented `p*`; the sum of per-population value equals the pooled value
recomputed independently.

### FAIL-03 — Single average `L` masking a heavy tail

**Symptom.** Projected value is consistently higher than realised value, and the shortfall
concentrates in a few large cases rather than spreading evenly.

**Competing causes.** (a) `L` is a mean over a heavy-tailed distribution and the policy
optimises the mean while the organisation experiences the tail; (b) large-loss cases are
under-represented in the training window; (c) a capital or regulatory limit makes the
marginal disutility of a large loss superlinear; (d) the largest losses are only visible
after a lag longer than the evaluation window.

**Discriminating check.** Plot the loss distribution and compute the ratio of the 99th
percentile to the mean. If it exceeds roughly 20, mean-based ranking is fragile under any
constraint. Then recompute the policy under a quantile objective and compare acted sets:
if the sets differ by more than a few percent, the constraint is material.

**Bounded fix.** Either raise `L` for the high-severity stratum and derive a stratum
threshold, or move to a constrained objective. Do not simply inflate `L` globally; that
under-acts on the many small, genuinely profitable cases.

**Verification.** Realised value over two consecutive windows inside the projected
interval computed from the loss distribution, not from the mean. If the interval is not
stated, the projection is not testable.

### FAIL-04 — Targeting on baseline risk when the action is persuasive

**Symptom.** A campaign or intervention delivers lift far below the modelled effect, and
the best-performing segments by baseline probability are the worst by realised lift.

**Competing causes.** (a) The mechanism is persuasion and the decision variable should be
`tau`; (b) `tau` is real but heterogeneous and the segment definition is wrong; (c) the
action has a negative effect on some segment (crowding out an unaided conversion);
(d) measurement uses a pre/post comparison contaminated by seasonality.

**Discriminating check.** Run a holdout within each targeted segment and compute the
difference in outcome rates between treated and control. If the segment ranked first by
`p` has near-zero difference, the mechanism is (a) or (c). Compare the observed
difference-in-differences with the modelled `tau`; a consistent overestimate across all
segments points to (d).

**Bounded fix.** Re-rank on estimated `tau` with its own threshold `a/M` (A-11), and
exclude segments with a confidently negative `tau`. Keep the `p`-based ranking only for
capacity allocation within segments that have similar `tau`.

**Verification.** A holdout-based estimate of incremental outcome per treated unit,
compared against the `tau` used to select, per segment, with confidence intervals.

### FAIL-05 — Capacity treated as fixed while the queue feeds back

**Symptom.** The quota rule is respected on paper; the review queue grows without bound
and ageing alerts are eventually auto-closed without review.

**Competing causes.** (a) `K` derived from headcount rather than measured throughput;
(b) handle time rises as the queue ages because stale cases need more work; (c) arrival
rate rises when the threshold loosens, and the loosening was justified by the value
calculation alone; (d) absence, attrition, or holiday effects reduce effective capacity
below the staffing number.

**Discriminating check.** Compare `K` claimed against reviews completed per day over four
weeks, and plot queue length against time. A monotonically rising queue with flat arrivals
points to (a) or (d); rising arrivals with flat throughput points to (c); rising handle
time with stable arrivals points to (b).

**Bounded fix.** Derive `K` from measured handle time at the 75th percentile, reserve 15%
headroom, and make the threshold a function of remaining capacity in the window rather
than a constant.

**Verification.** Queue length bounded across the window; fraction of alerts auto-closed
without review below a declared ceiling; value recomputed from actual completed actions,
not planned ones.

### FAIL-06 — Cost model omits a real consequence

**Symptom.** Value is positive in the model and flat or negative in the P&L, with the gap
proportional to the number of actions taken on negatives.

**Competing causes.** (a) `G` excludes a real consequence (contact cost, compensation,
attrition); (b) `a` excludes overheads and tooling; (c) `L` is gross rather than net of
recovery; (d) actions are counted twice because two systems both act on the same unit.

**Discriminating check.** Regress realised monthly value on action count and on prevented
positives separately. A significant negative coefficient on action count with a
non-significant coefficient on prevented positives means the false-alarm consequence is
missing (a). A negative intercept means fixed cost is missing (b). Duplicate action counts
show up as action count exceeding the number of distinct units.

**Bounded fix.** Rebuild `G` from a measured experiment on declined legitimate units — the
only defensible source — and add overheads to `a`. Re-derive; expect the threshold to
rise and the acted set to shrink.

**Verification.** Model value and P&L value reconciled to within a declared tolerance for
two consecutive windows, with the residual attributed to a named component.

### FAIL-07 — Threshold frozen while the score drifts

**Symptom.** Value decays gradually and no single window looks broken.

**Competing causes.** (a) Population drift with stable model; (b) label drift because the
review process that produces labels changed; (c) feature pipeline change with no version
bump; (d) genuine change in the underlying phenomenon.

**Discriminating check.** Track three series separately: score distribution (PSI or a
quantile table), reliability by bin, and base rate. Stable score distribution with drifting
reliability means (a) or (d). Shifting score distribution with stable reliability means (c).
A simultaneous change in label rate and review volume means (b).

**Bounded fix.** Set a recalibration cadence tied to the reliability alarm band, not to the
calendar. On alarm, recompute `p*` from current parameters before changing the score, so the
two causes are separated.

**Verification.** Reliability within band on the next two windows after recalibration, and a
documented record of which of the four causes was diagnosed.

**Takeaway:** every failure above presents as "value lower than projected"; the diagnosis
is a comparison of separate series, never a single metric.

## 9. Complexity & Cost Analysis

### 9.1 Computational cost of choosing and applying a threshold

Let `n` be decisions per window, `d` features per decision, `K` the capacity, and `m` the
number of strata.

| Operation | Time | Space | Notes |
|---|---|---|---|
| Score the batch | `O(n·d)` | `O(n)` scores | Dominated by feature retrieval, not arithmetic |
| Apply a fixed threshold | `O(n)` | `O(1)` extra | One comparison per unit |
| Evaluate value at one `t` | `O(n)` | `O(1)` extra | Needs `p` and `y` or `E[y|p] = p` |
| Closed-form `p*` | `O(1)` | `O(1)` | A-03 needs only `L`, `G`, `a` |
| Quota at `K` | `O(n log K)` | `O(K)` | Bounded heap; `O(n)` with quickselect |
| Full sweep over all cuts | `O(n log n)` | `O(n)` | Sort once, prefix-sum the value curve |
| Per-stratum thresholds | `O(n log n + m·n)` | `O(n)` | Sort per stratum; or one sort with grouped prefix sums |
| Isotonic recalibration (PAVA) | `O(n log n)` | `O(n)` | Sort dominates; PAVA itself is linear |
| Reliability monitoring per window | `O(n)` | `O(bins)` | Bin and count |

The important observation is that threshold *selection* is not a search problem. Section
4.1 gives `p*` in closed form, so the cost of choosing the threshold is the cost of
eliciting three numbers. The `O(n log n)` sweep is a communication device — it produces
the value curve that lets a committee see the plateau — not a necessity.

### 9.2 Worked sizing

Take `n = 200,000` decisions per window and `d = 40` features.

**Scoring.** `200,000 × 40 = 8,000,000` feature evaluations. At an illustrative
`2 × 10^6` evaluations per second on one core, that is 4 seconds. Even at a tenth of that
rate it is 40 seconds per batch, which is noise against the decision cycle.

**Sorting for the quota rule.** `200,000 log2(200,000) ≈ 200,000 × 17.6 = 3,520,000`
comparison units; at `10^7` comparisons/second, about 0.35 seconds. A bounded heap of size
60,000 costs `200,000 × log2(60,000) ≈ 200,000 × 15.9 = 3,180,000`, marginally cheaper.

**Human review.** At 2.5 minutes per review, `60,000 × 2.5 = 150,000` minutes `= 2,500`
hours. At 7 productive hours per analyst-day that is `357` analyst-days. Across a 21
working-day month it requires `357/21 = 17.0` analyst full-time equivalents.

**The ratio.** Compute is 4 seconds; review is 9,000,000 seconds. Human review is roughly
six orders of magnitude more expensive than the computation that selects the work. Any
engineering effort spent shaving the scoring pass is misallocated by that factor; effort
belongs in handle time, queue design, and `G` measurement.

### 9.3 Money cost of the policy

| Component | Units | Rate | Cost |
|---|---|---|---|
| Reviews at `p*` | 140,000 actions | 9 per action | 1,260,000 |
| False alarms at `p*` | 124,180 units | 5 per unit | 620,900 |
| Scoring | 200,000 units | ~0 marginal | ~0 |
| Monitoring (reliability + value) | 1 analyst-day/week | 1 day | ~4 analyst-days/month |

Against `4,746,000` of avoided loss, the policy nets `2,865,100`. The monitoring line is
worth stating explicitly because it is the first thing cut and the only thing that detects
FAIL-07.

### 9.4 Cost of getting the threshold wrong

Define the *regret* of a policy as the value lost relative to the optimum. For this
population the value curve is nearly flat between cuts of 0.046 and 0.072 — deciles 4 and
5 are the marginal ones and together they contribute `233,000` of `2,865,100`, about 8%.
Beyond 0.105 the regret grows quickly: cutting at 0.126 discards decile 7 at `360,500`
and decile 6 at `256,800`, and cutting at 0.19 discards `1,986,100`.

| Cut | Acted volume | Value | Regret | Regret per 1,000 actions |
|---|---:|---:|---:|---:|
| 0.020 (all) | 200,000 | 2,616,800 | 248,300 | 1,242 |
| 0.046 (optimum) | 140,000 | 2,865,100 | 0 | 0 |
| 0.088 | 100,000 | 2,632,100 | 233,000 | 2,330 |
| 0.105 | 80,000 | 2,375,300 | 489,800 | 6,123 |
| 0.126 | 60,000 | 2,014,800 | 850,300 | 14,172 |
| 0.160 | 20,000 | 879,000 | 1,986,100 | 99,305 |

Note the non-monotonicity in the per-1,000 column between the 0.020 and 0.088 rows: acting
on everyone wastes 1,242 per thousand actions, while cutting at 0.088 wastes 2,330 — the
over-strict error is twice as expensive per action as the over-loose error per action,
because the over-strict policy forgoes units whose value is much higher than the units the
over-loose policy needlessly acts on.

The rightmost column is the number to show a stakeholder who wants to cut harder "to be
safe": each 1,000 avoided actions at a cut of 0.16 forgoes about 99,000 in value. Being
safe with someone else's money is not a safety property.

### 9.5 Sizing an expansion of capacity

If capacity rises from 60,000 to 80,000, the added 20,000 actions are decile 7, worth
`18.025` each, so the incremental value is `360,500`. At an analyst cost of 357
analyst-days per month for 60,000 reviews, 20,000 more reviews need `20,000 × 2.5/60 =
833.3` hours `= 119.0` analyst-days `= 5.67` FTE. Value per added FTE is
`360,500/5.67 ≈ 63,580` per month. That is the number to put in a staffing request; "the
model is good" is not.

The same arithmetic for every capacity increment, using the marginal decile values:

| Capacity step | Added actions | Marginal EV | Added value | Added FTE | Value per FTE |
|---|---:|---:|---:|---:|---:|
| 60,000 → 80,000 | 20,000 | 18.025 | 360,500 | 5.67 | 63,580 |
| 80,000 → 100,000 | 20,000 | 12.840 | 256,800 | 5.67 | 45,290 |
| 100,000 → 120,000 | 20,000 | 7.960 | 159,200 | 5.67 | 28,080 |
| 120,000 → 140,000 | 20,000 | 3.690 | 73,800 | 5.67 | 13,020 |

The value per FTE falls by roughly a factor of five from the first increment to the last,
and the last is only marginally positive. Capacity should expand while marginal value per
FTE exceeds the loaded cost of an FTE plus the operational risk of the queue, and no
further. Note also that this table is the correct answer to "how much review capacity
should we staff", and that it changes whenever `L`, `G`, or the score distribution changes
— which is why capacity and threshold must be reviewed together rather than annually and
independently.

**Takeaway:** threshold choice costs nothing to compute and a great deal to get wrong;
budget the analysis minutes, not the compute hours.

## 10. Key Theorems / Results Reference

**T-01 — Bayes optimality of the two-action threshold (`DERIVED`).** For a decision with
two actions, a known loss matrix, and a calibrated posterior `p`, the rule "act iff
`p > (a + G)/(L + G)`" minimises posterior expected loss pointwise and therefore in
aggregate. *Proof sketch:* the aggregate loss is `Σ_i [1{act_i}(a + (1 - p_i)G) +
1{skip_i} p_i L]`; each term involves only its own indicator, so choosing the smaller of
the two per term is simultaneously optimal. *Conditions:* the loss per unit depends only
on `p` and the action; decisions are independent; costs are constant. *Use:* this is the
justification for A-03 and the reason no search is needed.

**T-02 — Neyman–Pearson form (`DERIVED`).** For a fixed false-positive budget `β`, the
most powerful test is a likelihood-ratio threshold. In the threshold language of this
sheet, constraining `P(act | negative) ≤ β` converts the problem from a monetary level to
a quantile: act on the top `β·N_neg` units by likelihood ratio. *Conditions:* simple null
and alternative, known distributions or a monotone likelihood ratio. *Use:* it is why the
capacity-constrained rule (A-07) is not a heuristic but the constrained optimum.

**T-03 — Invariance under monotone transformation (`DERIVED`).** If `g` is strictly
increasing, then `p > t` iff `g(p) > g(t)`. A threshold transfers across monotone
rescalings only when it is rescaled too. *Consequence:* the threshold is not a property of
the model; it is a property of the (model, scale) pair. Storing a threshold without the
score version is meaningless, and this is the formal reason.

**T-04 — Additivity and greedy optimality (`DERIVED`).** Total value is a sum of
per-decision values, so maximising total value decomposes. When per-unit cost is constant
and value is increasing in `p`, the capacity-constrained optimum is the top-`K` set:
exchange argument, since swapping an included low-score unit for an excluded higher-score
unit weakly increases the sum. *Consequence:* the quota rule needs only a ranking, hence
only a monotone score, hence no calibration.

**T-05 — Greedy is not optimal when costs differ (`DERIVED`).** If per-unit cost varies,
the problem is a 0/1 knapsack; greedy by value density `EV_i/a_i` is optimal only for the
fractional relaxation. The integrality gap can be a full item. *Consequence:* with
heterogeneous `a`, either solve the knapsack, or stratify so that `a` is constant within
stratum and apply T-04 per stratum.

**T-06 — AUC is a rank statement (`DERIVED`).** `AUC = P(s⁺ > s⁻)` (Mann–Whitney), which
is invariant to any monotone transformation and therefore says nothing about any operating
point. *Consequence:* AUC cannot justify a threshold and cannot be traded against value.
The value-relevant quantity is the expected cost at a chosen point, which depends on
calibration and costs.

**T-07 — Isotonic calibration preserves ranking (`DERIVED`).** The isotonic regression of
observed rate on score under a monotone constraint is fitted by pool-adjacent-violators in
linear time after sorting. The fitted map is non-decreasing, so it changes levels without
changing order, which is exactly what a level rule needs and a quota rule does not.
*Conditions:* monotonicity is the right constraint; a non-monotone reliability curve
indicates a broken feature, not a calibration problem.

**T-08 — Expected cost equals the sum of two error costs (`DERIVED`).** From A-04,
`EC(t) = a·(TP + FP) + G·FP + L·FN`. The three terms are separable, so any two of them can
be reported as headline metrics and the third recovered. *Use:* it is why "cost of review"
and "loss prevented" cannot be quoted independently without inviting an inconsistent pair
of numbers.

**T-09 — Linear objective, mean parameter (`DERIVED`).** If the value function is linear
in the loss, substituting `E[L]` is exact. Non-linearity enters only through constraints
(capacity, capital, regulatory limits), and under a constraint the correct object is a
quantile or tail expectation. *Use:* this is the precise reason FAIL-03 is a constraint
problem rather than an estimation problem.

**T-10 — Prior shift changes levels, not order (`DERIVED`).** Multiplying all odds by a
constant preserves the ordering of `p` and therefore preserves the optimal *set* only if
the threshold is re-derived. With a fixed threshold, the acted set changes. *Consequence:*
under base-rate drift, a quota rule is more robust than a level rule, because the quota is
invariant to the shift by construction — a real and underused argument for quotas.

**T-11 — Threshold is a monotone function of each parameter (`DERIVED`).** `∂p*/∂a = 1/(L
+ G) > 0`; `∂p*/∂G = (L - a)/(L + G)^2 > 0` when `L > a`; `∂p*/∂L = -(a + G)/(L + G)^2 <
0`. *Use:* the signs are enough to audit a proposal without arithmetic — any proposal that
lowers the threshold while raising `G` is internally inconsistent.

**T-12 — Value of information on the threshold parameter (`HEURISTIC`).** The value of
better estimating `G` is approximately `|∂V/∂G| · σ_G`, and `|∂V/∂G|` equals the expected
number of false alarms at the optimum. *Use:* it tells you which parameter to measure
first — usually the one multiplying the largest count. *Limitation:* first-order
approximation; invalid near a kink in the value curve.

**Takeaway:** T-01 justifies the formula, T-04 and T-10 justify the quota rule, and T-06
explains why the metric most often reported alongside a threshold cannot justify it.

## 11. Interview / Exam Questions

**Q1. Why is `a` subtracted only once in `EV_act`?**
*Answer.* Because the action cost is incurred once per decision regardless of the outcome.
Writing `EV = p(L - a) - (1 - p)(G + a)` expands to `pL - pa - G - a + pG + pa = p(L + G)
- (a + G)`, with the `pa` terms cancelling. A common error is `EV = p(L - a) - (1 - p)G`,
which charges `a` only on positives and yields `p* = (G + a)/(L + G - a)`.
*Reasoning.* Set up the four-cell table before simplifying; the cancellation is a check
that the table is right. *Red flag.* Reciting `p* = C/(B + C)` from memory and being unable
to say what `B` and `C` contain.

**Q2. Derive the threshold when there is also a cost to acting on a positive.**
*Answer.* If acting on a positive costs `a_pos` and acting on a negative costs `a_neg`,
`EV_act = p(L - a_pos) - (1 - p)(G + a_neg)`, so `p* = (G + a_neg)/(L + G + a_pos - a_neg)`.
The asymmetric-cost version is the general one; 4.1 is the special case `a_pos = a_neg = a`.
*Reasoning.* Keep the two branch costs separate until the algebra forces them together.
*Red flag.* Assuming the formula is universal without checking whether cost varies by
outcome.

**Q3. Your model's AUC rose from 0.78 to 0.81 and value fell. Explain.**
*Answer.* AUC is `P(s⁺ > s⁻)` and is invariant to monotone transformation (T-06); it says
nothing about the operating point. Value fell because calibration at the chosen cut
worsened, or because the new model's ranking improved in a region the policy does not act
on while degrading near the cut. Diagnose with a reliability table and a value curve, not
with AUC.
*Reasoning.* Separate ranking from level; the threshold depends on level. *Red flag.*
Proposing a threshold search to recover the lost value without first checking calibration.

**Q4. Compute the threshold for `a = 12`, `G = 30`, `L = 400`, and interpret.**
*Answer.* `p* = (12 + 30)/(400 + 30) = 42/430 = 0.097674`. At 9.77% the policy is
indifferent. Each percentage point of probability is worth `4.30` per decision, and each
decision below the cut saves `42` in action and penalty cost. Compare against the observed
score distribution: if fewer than 5% of units exceed 0.098, the policy is capacity-limited
in the other direction (it will under-use the available operation), which is a business
decision, not a modelling one.
*Reasoning.* Always convert the threshold into a volume before accepting it.
*Red flag.* Reporting four decimal places for a threshold derived from a `G` estimated to
one significant figure.

**Q5. When is a quota rule better than a level rule?**
*Answer.* Three cases: capacity is hard and binding; the score is rank-only so no level
exists; and the base rate drifts, since the quota is invariant to a monotone prior shift
(T-10) while the level rule is not. The quota's weakness is that it hides the monetary
question — it cannot tell you whether the action is worth performing at all, only which
units to perform it on.
*Reasoning.* State the invariance property explicitly; it is the strongest argument for
quotas and rarely mentioned. *Red flag.* Claiming a quota is "more robust" without naming
what it is robust to.

**Q6. Estimate the value of improving `G` from ±100% to ±20% uncertainty.**
*Answer.* At the optimum the number of false alarms is about 124,180. A change `δG` moves
value by approximately `-FP·δG`, so the value of removing that uncertainty is roughly
`124,180 × (0.8 × 5) = 496,720` in expectation if the prior is centred and symmetric —
an upper bound, since the policy would also be re-optimised. The practical answer: measure
`G` before spending on model improvements, because the false-alarm count multiplies it.
*Reasoning.* Sensitivity times the quantity affected; here the count is large, so the
`G` measurement is high-leverage. *Red flag.* Treating `G` as a free parameter to be
chosen so that the answer looks reasonable.

**Q7. A stakeholder wants to set the threshold at 0.5 because "that's 50/50". What is wrong?**
*Answer.* 0.5 is the threshold that equalises the two error rates, which is optimal only
when `a + G = L + G`, i.e. `a = L` and the consequences of the two errors are symmetric.
With `a = 9`, `L = 300`, the correct threshold is 0.046, an order of magnitude lower. A 0.5
cut on the Section 5 population acts on almost nothing and forgoes nearly all the value.
*Reasoning.* A threshold encodes a cost ratio, so a "neutral" threshold encodes an
unstated and almost certainly false claim about costs. *Red flag.* Defending 0.5 by
appealing to symmetry rather than to the loss magnitudes.

**Q8. How does the answer change when the action is a coupon rather than a block?**
*Answer.* The mechanism changes, so the decision variable changes from `p` to `tau` (A-11):
act iff `tau > a/M`. The threshold on `tau` is `2.4/90 = 0.0267` in the illustrative
example. Ranking on `p` would target the highest-propensity customers, many of whom would
have bought anyway, and the realised lift would be a fraction of the modelled effect. Also
`G` largely disappears — a wasted coupon costs only `a` — which lowers the bar further.
*Reasoning.* Ask first whether the action changes the outcome; everything downstream
depends on the answer. *Red flag.* Applying a `p`-based threshold and reporting the
resulting difference as causal lift.

**Q9. Your flagged rate was 70% last month and is 35% this month with the same threshold. Diagnose.**
*Answer.* Competing causes: base-rate shift (fewer positives, same score distribution);
score distribution shift (pipeline or feature change); population change (new channel or
region); or threshold actually differing across code paths. Discriminate by tracking score
distribution, reliability, and base rate separately (FAIL-07). If the score distribution is
stable and the base rate halved, it is a prior shift and the threshold must be re-derived
(A-12, then recalibrate).
*Reasoning.* Three series, three signatures; the pattern identifies the cause before any
code is read. *Red flag.* Immediately rescaling the threshold by the ratio of flagged rates.

**Q10. What is the marginal value of the 60,001st review slot?**
*Answer.* It equals the expected value of the highest-scored excluded unit, which is the
100,000th-ranked unit in this population at 7.96 money units. Not the average value per
review (33.58 at `K = 60,000`), which overstates the marginal worth by a factor of four and
would justify capacity far beyond the point where it pays.
*Reasoning.* Marginal, not average; the average is inflated by the inframarginal units.
*Red flag.* Quoting `value per review` as if it were a derivative.

**Q11. Show that a higher false-alarm penalty always raises the threshold.**
*Answer.* `∂p*/∂G = (L - a)/(L + G)^2`. Positive when `L > a`, which holds whenever the
action is worth performing at all (if `L ≤ a`, no unit justifies the action and the
threshold is vacuous). So within the region where the policy is non-trivial, raising `G`
raises `p*`. The magnitude is small when `L >> a`: raising `G` from 5 to 50 moves `p*` from
0.0459 to `(9 + 50)/350 = 0.1686`.
*Reasoning.* Differentiate; then check the sign against the non-triviality condition.
*Red flag.* Asserting monotonicity without checking whether it holds in the relevant
region.

**Q12. Why does an `F1`-optimal threshold give the wrong operating point?**
*Answer.* `F1` is the harmonic mean of precision and recall, which implicitly weights
false positives and false negatives equally and ignores the sizes of the classes and the
money. Formally it corresponds to a specific, arbitrary cost ratio. On the Section 5
population the `F1`-optimal cut is not 0.046, and the value difference is material: a cut
near 0.19 loses 1,986,100. `F1` is a reasonable diagnostic for balanced, cost-free settings
and the wrong objective for a monetary decision.
*Reasoning.* Name the implicit cost ratio behind the metric. *Red flag.* "We chose the
threshold that maximises F1" offered as the complete justification.

**Q13. Design the monitoring for a deployed threshold in three metrics.**
*Answer.* (1) Reliability by decile with a pre-declared alarm band, catching calibration
drift; (2) base rate and score-distribution PSI per window, separating prior shift from
score shift; (3) realised value per action and per window, reconciled against the
projection with the residual attributed. Add a fourth only if the action has a causal
claim: a holdout-based uplift check. The cadence follows the alarm, not the calendar.
*Reasoning.* Each metric discriminates a different failure from Section 8; three are the
minimum spanning set. *Red flag.* Monitoring only volume or only AUC.

**Q14. Given a choice between a 3-point AUC gain and a 10% reduction in handle time, which do you take?**
*Answer.* Compute both in money. Handle time enters `a`, so a 10% reduction moves `a` from
9 to 8.1, lowering `p*` from `14/305 = 0.0459` to `13.1/305 = 0.0430`, which admits the
deciles between and adds their value — and it also frees capacity, which under a binding
quota converts directly into decile 7 at 18.025 per action. The AUC gain has no direct
monetary translation and may not move the operating point at all. Default to the handle
time unless a value curve on the new model shows a shift at the chosen cut.
*Reasoning.* Convert each option into the same unit before comparing; one option is
already in that unit. *Red flag.* Comparing a ranking metric to an operational metric
without a conversion.

**Q15. What breaks if the same threshold is used for two channels with different recovery rates?**
*Answer.* `L` differs because recovery differs, so `p*` differs: with `L = 300` and `a = 9`,
`G = 5`, `p* = 0.0459`; with recovery halving net loss to `L = 150`, `p* = 14/155 = 0.0903`,
roughly double. Applying 0.0459 to the low-recovery channel over-acts, spending review cost
and goodwill on units whose avoidable loss is smaller than the cost of avoiding it.
*Reasoning.* Re-derive per stratum rather than adjusting the threshold by intuition.
*Red flag.* "We use the same threshold everywhere for consistency" — consistency of
process bought with inconsistency of economics.

**Takeaway:** the questions that separate candidates are the ones about which parameter or
which unit is wrong, not the ones about the formula.

## 12. Quick Reference Card

| Field | Rule |
|---|---|
| Decision | Act iff `p > (a + G)/(L + G)`; act iff `tau > a/M` when the action is persuasive. |
| Population | One decision unit, one snapshot, one window; aggregate scores to the decision unit before thresholding. |
| Measure | `L` net of recovery, `G` measured not assumed, `a` = handle time × loaded rate. |
| Threshold | Closed form, no search; report the value plateau, not a spurious optimum. |
| Capacity | If implied actions exceed `K`, switch to top-`K`; the threshold becomes `p_(K)` and the shadow price is the EV at rank `K + 1`. |
| Value | `EV = p(L + G) − (a + G)` per decision; total is the sum over the acted set; reconcile with the four-cell form. |
| Guardrail | Reliability by decile plus base rate per window; a level rule dies silently under prior shift while a quota rule survives. |
| Diagnose | Three series in parallel: score distribution, calibration, base rate. Volume alone cannot separate the causes. |
| Verify | Per-decile sum equals the confusion-matrix total; acted volume equals `P(p ≥ t)·N` on a holdout. |
| Evidence | Threshold algebra `DERIVED`; Section 5 arithmetic `VERIFIED`; all populations and costs `ILLUSTRATIVE`; `L`, `G`, `a` for any real deployment `OPEN` until measured. |
