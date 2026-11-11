## 9. Evaluation & Validation

Evaluation in this case is not a model report. It is a claim that a different tier-assignment policy
would have produced different outcomes on the same population, and that claim needs seven things
stated before the result is read: numerator, denominator, population, window, comparator, guardrails,
and the uncertainty convention. Each is fixed below, in that order, with the executed/not-executed
status attached.

### 9.1 Primary metric definition

| Element | Fixed definition | Why this and not the alternative |
|---|---|---|
| Numerator | lines whose `t_notify − t_quote ≤ 120 min` | notification is the earliest moment the customer can collect |
| Denominator | the 520 lines the incumbent rule quoted at P2 | a fixed, policy-independent population |
| Population | one store, one promotion day, 14 slot instances | enough slot instances to see capacity bind, few enough to trace by hand |
| Window | quote instant to 240 minutes, truncated | beyond four hours the line is PN regardless of policy |
| Comparator | incumbent ledger rule reproduced with its own staleness | a baseline given better data is not the baseline |
| Aggregation | per-tier rate, pooled rate, fixed yardstick | three numbers because one number was the defect |
| Rounding | rates to four decimals, counts integer, no re-rounding between steps | prevents 0.86 vs 0.8596 discrepancies in a review |
| Status | **executed** on constructed data; **not executed** on production data | the arithmetic is real, the cohort is invented |

The primary metric is paired with a mandatory companion, because the case's whole argument is that
the primary metric alone can be gamed by deferral:

```text
FORMULA [E-01]: r_kept(tier) = Kept(tier) / Promised(tier)
                r_fixed      = Kept_within_120 / Eligible
                mix_gain     = r_kept(P2) - r_fixed
WHERE:
  r_kept(tier) [fraction]  promise-relative rate, moves when the tier mix moves
  r_fixed      [fraction]  yardstick rate, does not move when the tier mix moves
  mix_gain     [fraction]  the part of an apparent improvement that is definitional
CONSTRAINT:
  Eligible is fixed before any policy runs and does not shrink when a policy defers.
NUMERICAL EXAMPLE (constructed):
  gate-only policy:        r_kept(P2) = 447/487 = 0.91786
                           r_fixed    = 447/520 = 0.85962
                           mix_gain   = 0.91786 - 0.85962 = 0.05824
  gate + risk ranking:     r_kept(P2) = 447/465 = 0.96129
                           r_fixed    = 447/520 = 0.85962
                           mix_gain   = 0.96129 - 0.85962 = 0.10167
  gate + ranking + cycle counts:
                           r_kept(P2) = 469/487 = 0.96304
                           r_fixed    = 469/520 = 0.90192
                           mix_gain   = 0.96304 - 0.90192 = 0.06112
INTERPRETATION:
  mix_gain is the portion of the improvement that a reader should not attribute to operational speed.
  Reporting it explicitly converts a potential accusation into a disclosed quantity.
```

The three rows separate the contributions cleanly. The gate alone buys 5.82 points, all definitional.
Adding the risk ranker buys another 4.34 points, also definitional — 22 risky promises withheld, not
22 picks accelerated. Only the cycle-count row moves `r_fixed` at all, and in that row `mix_gain`
falls to 6.11 points because 4.23 points of genuine speed appeared. A team reporting only `r_kept(P2)`
would describe a 10.34-point improvement; the honest decomposition is 6.11 points of promise integrity
and 4.23 points of shelf accuracy, and the second one was delivered by a person with a scanner.

### 9.2 Guardrail metrics

Guardrails are the quantities that must not degrade while the primary improves. Each has a threshold
and a consequence, because a guardrail without a consequence is a dashboard tile.

| Guardrail | Numerator / denominator | Threshold | Consequence of breach |
|---|---|---|---|
| Deferral rate | lines moved off P2 / eligible | ≤ 0.10 | freeze gate; review `L_eff` and roster for the affected slots |
| Abandonment on deferred lines | orders cancelled within 2 h of deferral / deferred orders | ≤ 0.08 | re-price the tier menu before continuing |
| Manual abstention volume | manual tier decisions / store-day | ≤ 40 | cap automated abstention at observed capacity |
| Override rate | operator reversals / automated decisions | ≤ 0.05 | audit rationale templates for the reversed stratum |
| Stockout-driven lateness | lines late with first failure = stockout / late lines | ≤ 0.50 | escalate cycle-count coverage |
| P2 floor by store-week | kept P2 lines / promised P2 lines, per week | ≥ 0.90 | automated tiering suspended for that store |
| Notification delay | median `t_notify − t_stage` | ≤ 6 min | treat as an integration incident, not a model issue |

The override-rate guardrail is the most diagnostically useful and the least monitored in practice. A
rising override rate says an operator consistently disagrees with the engine on a stratum that the
engine believes is easy. In the constructed shadow log the override rate reached 0.11 in the
14:00–16:00 slot and the cause was a store-specific shelf reset that moved the high-velocity SKUs to
a second aisle, doubling `S_pick` for those lines while the roster was unchanged. No amount of
retraining would have found that; the overrides did, in three days.

### 9.3 Uncertainty

Three sources of uncertainty need different treatments, and conflating them is how a point estimate
acquires false precision.

*Sampling uncertainty on the rate.* With `n = 520` and `r̂ = 0.8596`, the standard error of a
proportion is `sqrt(r̂(1−r̂)/n) = sqrt(0.8596 × 0.1404 / 520) = sqrt(0.12069/520) = sqrt(0.0002321) =
0.01523`. A 95% interval is `0.8596 ± 1.96 × 0.01523 = [0.8298, 0.8894]`. Compare the gate-only
policy at `447/487`: `sqrt(0.9179 × 0.0821 / 487) = sqrt(0.07536/487) = sqrt(0.0001548) = 0.01244`,
interval `[0.8935, 0.9423]`. The two intervals do not overlap, so the difference survives sampling
error — but note that the difference is entirely a denominator change, which no confidence interval
can reveal.

Cluster the check by slot, because lines within a slot are not independent: they share pickers, a
queue, and a shelf state. With 14 slot-instances the effective sample size for capacity effects is
closer to 14 than to 520. The constructed within-slot intraclass correlation for the late indicator
is 0.18, which inflates the rate's variance by a design effect of `1 + (m−1)ρ` with `m ≈ 37` lines
per slot: `1 + 36 × 0.18 = 7.48`. The variance is 7.5 times the naive one, and the honest interval on
the baseline rate widens to roughly `± 0.042` rather than `± 0.030`. That interval now overlaps the
gate-only interval, which correctly says: with 14 slots, the gate's rate improvement is not
distinguishable from noise on this cohort, and the reason to adopt it is arithmetic (the gate cannot
exceed `K`) rather than the observed rate difference.

*Input uncertainty.* `L_eff = 96` minutes is an assumption with a plausible range of 88 to 104. At
88 minutes, `K` for the 10:00–12:00 slot is `floor(4 × 88 / 6.6) = floor(53.33) = 53` orders and 95
lines; at 104 minutes it is `floor(4 × 104 / 6.6) = floor(63.03) = 63` orders and 113 lines. The
feasible-line range `[95, 113]` brackets the point value of 104, and the gate's deferral count ranges
from 15 to 33. That is a wide band for a single input, and it is the strongest argument for measuring
`L_eff` from the assignment log rather than configuring it from memory.

*Structural uncertainty.* The largest unknown is whether the deployment regime would reproduce the
training regime's priority treatment. If P2 lines continue to receive pick priority, the gate's
deferral improves everyone's lot slightly by relieving the queue; if priority is removed as part of a
fairness change, the remaining P2 lines get slower and the gate's benefit shrinks. No data in this
case can settle that, and the case says so rather than assuming the favourable branch.

### 9.4 What was validated, and by what

| Claim | Validation performed | Status | Limitation |
|---|---|---|---|
| The cohort is internally consistent | reconciliation to 1,180 rows, 5 exclusion classes, all counts matched | executed | constructed data, so consistency is by construction |
| Capacity arithmetic | recomputed `K_slot` for all 14 slot instances at three `L_eff` values | executed | service-time model understates observed mean by 0.3 min |
| The inversion is real | recomputed 447/520 vs 389/468 from the same cohort | executed | the inversion magnitude is an input, not a finding |
| The gate removes only mechanical lateness | attributed 73 late lines to 27 mechanical + 46 residual | executed | spill correction of 6 lines is estimated, not traced |
| Risk ranking earns 22 lines | attributed residual by first failing event, hand-read | executed | 5 lines in two categories were hard to assign |
| Fixed yardstick rises to 0.9019 | 469 lines within 120 min, requiring 11 recovered stockouts | **not executed** | depends on a cycle-count programme not modelled here |
| Customers prefer a kept four-hour promise to a broken two-hour one | not tested | **not executed** | needs a preference study or a choice experiment |
| The design improves revenue | not tested | **not executed** | the case makes no revenue claim |

The last three rows are the ones a reader should carry into any discussion. The case's arithmetic
supports a claim about promise integrity and a claim about which lines are addressable. It does not
support a claim about customer preference, revenue, or the effect of cycle counts.

### 9.5 Validation design for a real deployment

If a store wanted to test this rather than reason about it, the design that would answer the question
is a slot-level cluster randomisation, not a line-level one. Lines within a slot share capacity, so
line-level randomisation leaks: a control line and a treated line in the same slot compete for the
same picker, and the control's outcome is contaminated by the treatment. Slot-level assignment —
alternate the gate on and off by slot-instance, or by store-day — avoids the leak.

Power, computed rather than asserted. To detect a 5-point improvement in `r_fixed` from 0.86 at 80%
power with a two-sided 5% test, the required sample per arm is approximately
`n = 2 × (1.96 + 0.84)² × p(1−p) / Δ² = 2 × 7.84 × 0.1204 / 0.0025 = 2 × 7.84 × 48.16 = 755 lines`
per arm before clustering. With a design effect of 7.48 from slot-level clustering, the requirement
becomes roughly `755 × 7.48 ≈ 5,647` lines per arm, or about 11 days at 520 lines/day in one store —
or four stores for three days. That is the honest cost of measuring the effect, and it is why the
recommendation in section 10 is a shadow log rather than an immediate A/B test.

Takeaway: state the denominator and the design effect before reading the rate, because the same 5-point
difference is decisive at `n = 5,000` and indistinguishable from noise at `n = 520`.

## 10. Action Taken

No production change was made, and this section separates the proposed actions from any executed
result. The organisation in the case is a constructed one; the sequence below is the sequence the
analysis supports, with owners and exit criteria, not a log of what happened.

**A-1 Shadow the gate, keep the incumbent promise (proposed, first two weeks).** Compute `gate_K` and
`gate_verdict` for every quote in parallel with the incumbent rule, write both to the decision log,
and show the customer the incumbent tier. Owner: fulfilment manager; instrument: decision service in
shadow mode. Exit criterion: for ten consecutive days, the gate's `slot_load` agrees with the load
recomputed from the assignment log to within 2 lines in at least 90% of slot instances.

The shadow period is not ceremony. It is the only way to discover `L_eff` empirically, and in the
constructed sequence the shadow log moved the estimate from a configured 96 minutes to an observed
92, which cut `K` for the 10:00–12:00 slot from 58 orders to 55 and reduced the deferral count from 24
to 28. A gate deployed with a configured `L_eff` would have been 5% optimistic for as long as nobody
looked.

**A-2 Fix the ready timestamp (proposed, week one, parallel with A-1).** Adopt `t_notify` as the
ready time in every report, backfill 90 days, and publish the staging-to-notification gap
distribution. Owner: data steward. Exit criterion: all promise reports cite `t_notify`; the gap
median and p90 are published per store. This has no modelling content and it changes every number
the business has been reading, which is why it must precede any claim of improvement.

In the constructed backfill the gap median was 4.5 minutes and the p90 was 19 minutes. Historical
reports built on the staging scan therefore overstated the promise-kept rate by roughly 3 to 6 points
depending on the store, and the stores with the worst notification integration looked like the best
pickers. That is a specific, checkable defect rather than a general worry.

**A-3 Adopt the gate for one tier, one store (proposed, week three, conditional).** If A-1 and A-2
pass, enable the gate for the P2 tier in one store, with the P4 route as the automatic fallback and
the abstention cap at the observed manual capacity. Owner: fulfilment manager, co-signed by
e-commerce. Exit criteria: deferral rate ≤ 0.10; abandonment on deferred lines ≤ 0.08; override rate
≤ 0.05; P2 floor ≥ 0.90 for two consecutive weeks.

**A-4 Rank within the gate, not above it (proposed, week five, conditional on A-3).** Enable the risk
ranker to order candidates inside the gate, with the label changed from promise outcome to pick
failure or re-pick. Owner: data science, with the fulfilment manager as the accepting operator. Exit
criterion: at least 15 of the 22 residual stockout lines correctly ordered below the admission cut
over two weeks.

**A-5 Cycle counts on the top 40 velocity SKUs (proposed, week one, independent of the model).**
This is the only action that moves the fixed yardstick, and it does not depend on any modelling
result. Owner: store operations. Exit criterion: on-hand accuracy on the counted SKUs improves from
the audited baseline to within 2 percentage points, measured by recount.

A-5 deserves emphasis because it is the cheapest and the most likely to be dropped. It requires no
data science, no decision service, and no policy change; it requires a person with a scanner and a
list. In the constructed attribution it recovers about half of the 22 stockout lines, which is more
fixed-yardstick improvement than the entire modelling programme delivers. A team that ships A-4 and
skips A-5 has optimised the visible artifact and left the largest addressable cause untouched.

**Deliberately not done, with reasons.** No threshold tuning to improve the 0.831 figure: tuning a
threshold on a biased score produces a better number and the same defect. No retraining on the full
1,180-line cohort: the extra 660 lines were quoted at other tiers under different treatment, so
including them without a treatment indicator would add a second regime rather than more data. No
customer-facing countdown widget: the tier menu is the commitment, and a countdown adds precision the
store cannot honour. No extension to ship-from-store in the same change: the capacity model would
need a delivery-route term and the cohort has no route data.

Takeaway: order the actions by what they can move — timestamp correction and cycle counts first,
gate second, ranking third — and refuse the tuning that improves the number without touching the
mechanism.
