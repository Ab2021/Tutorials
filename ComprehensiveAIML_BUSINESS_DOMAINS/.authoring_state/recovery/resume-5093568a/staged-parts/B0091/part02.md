## 5. Data, Tools & Method

### 5a Data Specification

**Grain.** One row per `member_id` × `decision_cycle`. A decision cycle is a Monday 06:00 local
cut-off. With 1,200,000 active members and a weekly cycle, the raw grain produces 1,200,000 rows per
cycle; after the eligibility filters described below, 250,000 rows survive as *eligible
member-weeks*. Everything downstream — scoring, selection, evaluation — uses that filtered grain.
Any join that changes the grain (for example joining transactions) must be aggregated back before
selection, or the row count becomes a transaction count and the budget constraint silently changes
meaning.

**Keys.** `member_id` is the loyalty identifier and is stable. `household_id` is present for 78% of
members; the remainder are treated as their own household for holdout assignment, which biases
measured effects *upward* slightly because true clustering is under-modelled. `decision_cycle_id`
is the ISO week plus cut-off date, and `decision_id` is a deterministic hash of member and cycle so
that replay is idempotent. `offer_id` links to the funded incentive's terms, and `campaign_version`
links to the creative and the policy envelope.

**Cut-off.** The decision timestamp is Monday 06:00:00 local. All input tables are read with
`effective_ts <= cut_off - interval '15 minutes'`. The fifteen-minute margin is a deliberate
under-estimate of observed correction lag, chosen so the quarantine counter is exercised in normal
operation rather than discovered in an incident.

**Population.** Eligible member-weeks satisfy all of:

- at least one transaction in the trailing 90 days (active),
- marketing consent in force for at least one contactable channel,
- not on any suppression list,
- no contact in the trailing 14 days,
- not part of a conflicting experiment,
- a resolvable household identifier for holdout assignment, or a deterministic fallback.

The filters are not neutral. Each removes a subpopulation with a different baseline purchase
probability, so the eligible population is not a random sample of members, and any historical
analysis that ignores the filters will not reconcile with the live queue.

**Units.** Money in US dollars, unrounded internally and rounded to cents only for display.
Probabilities in percentage points at two decimals for discussion and in proportions for
arithmetic. Time in whole days for recency and cooldown, and in UTC timestamps for event ordering.
Margins are per incremental purchase, not per order and not per line.

**The outcome.** For a member-week decided at cycle `t`, the outcome is gross margin in dollars
attributable to purchases in `[cut_off, cut_off + 28 days]`, net of returns processed up to 42 days
after cut-off. The 28-day purchase window and the 42-day return tail are separate windows because
mixing them produces a metric that appears to improve when returns are slow.

**What is not available at decision time.** This list is more important than the list of what is,
because every item on it is a leakage trap:

- whether the member purchased after the cut-off (the outcome itself),
- redemption status of the current offer,
- returns against purchases made after the cut-off,
- service interactions triggered by the current campaign,
- the next cycle's contact history,
- competitor pricing observed after the cut-off,
- any holdout assignment outcome.

**Volume.** A constructed cycle in this case has 1,200,000 active members, 250,000 eligible
member-weeks after filters, 120,000 selected for contact, 2,400 routed to human review, and 130,000
receiving no contact at all. The selected band therefore covers 48% of eligible rows, which is the
saturation level at which interference between members stops being a footnote.

### 5b Tools & Stack

The stack is described by capability, not by vendor, because vendor claims would be unsourced.

| Layer | Capability required | Why |
|---|---|---|
| Source | Event-level transaction, contact, consent, and offer tables | Point-in-time reconstruction |
| Warehouse | Versioned tables with `effective_ts` | As-of joins and replay |
| Transform | Deterministic, version-controlled feature build | Reproducible scores |
| Training | Gradient-boosted trees or a two-model difference | Handles mixed categorical and sparse signals |
| Serving | Batch scoring before cut-off, not per-request | Weekly cadence, no latency requirement |
| Decision | Constraint solver or greedy sort with gates | Feasibility before quality |
| Log | Append-only decision log with rationale | Audit and replay |
| Review | Queue with per-row policy envelope and reason codes | Human-in-the-loop |
| Measurement | Holdout assignment, outcome join, decile report | Causal claim |

Two stack choices are load-bearing and easy to get wrong.

The first is batch scoring before the cut-off rather than real-time scoring at send time. Weekly
cadence means latency is irrelevant, and batch scoring forces the feature snapshot to be a concrete
artefact that can be diffed. A real-time scorer would make the point-in-time check a property of the
code path rather than a property of the data, and the check would then be untestable in isolation.

The second is the append-only decision log. It stores, per row: the input snapshot identifier, the
feature vector hash, the model version, the estimate, the policy decision, the reason code, the
selected or abstained flag, and the reviewer identifier if reviewed. Without this log, a later
metric movement cannot be attributed to a world change or a calculation change.

Deliberately absent from the stack: a real-time feature store, a streaming pipeline, and an online
learning loop. Each would add failure surface for a decision that runs 52 times a year. The
evaluation section treats "did we need any of them?" as a live question with a specific test.

### 5c Method Summary

Four candidate methods were considered and one was carried forward. The comparison is part of the
method, not a formality, because the losing candidates each fail a specific check.

| Candidate | Target | Strength | Failure | Verdict |
|---|---|---|---|---|
| M1 rules queue | Weighted recency, tier, affinity | Auditable, zero training cost | Blind to incrementality | Baseline only |
| M2 propensity | P(purchase given contact) | Easy labels, familiar metrics | Selects sure things | Rejected |
| M3 uplift difference | P(purchase with contact) − P(without) | Directly matches objective | Needs randomised history; noisy at decile level | Carried forward |
| M4 causal forest on margin | Heterogeneous dollar effect | Handles continuous outcome | Unstable at small leaf sizes; harder to audit | Deferred |

**M1** is retained as the comparator, not as a fallback to be abandoned. A policy that cannot beat
the legacy rules queue on the frozen evaluation window does not ship, and the queue is cheap enough
to keep running while evidence accumulates.

**M2** is rejected on a structural argument rather than on measured performance. A propensity model
maximises P(purchase | contacted), and the members with the largest P(purchase | contacted) are
dominated by members with a high P(purchase | not contacted). Ranking by that target moves budget
toward the population with the least headroom. Section 6 quantifies the loss with constructed
numbers; the point is that the loss is a property of the target, not of the model class.

**M3** is carried forward as a two-model difference: one model trained on treated rows to estimate
`p1`, one on control rows to estimate `p0`, with the estimate `Δp = p1 - p0` clipped at zero for
selection and reported unclipped for diagnostics. The clipping is a policy choice and is recorded
as such; it prevents a negative-lift member from being selected by a value tie-break, but it also
means the aggregate estimate is biased upward, which the calibration report must disclose.

**M4** is deferred because the dollar outcome is heavy-tailed and the leaf-level estimates are too
noisy to pass the review gate's explainability requirement. It remains a candidate for the
following year, when two more randomised cycles will exist.

The selection rule is a greedy sort on value per contact, which is optimal here because every
contact consumes exactly one unit of the budget. The review gate is applied *after* selection, as a
partition rather than a re-ranking, so the gate cannot change the order of the selected set.

### 5d Provenance, Quality, Exclusions, Leakage

**Provenance.** Every source table carries a `source_system`, `source_version`, `ingest_ts`, and
`effective_ts`. The feature build records the snapshot identifier. The decision log records the
snapshot identifier it consumed. A score can therefore be traced to a specific read of a specific
table version, which is what "reproducible" has to mean in a batch pipeline.

**Quality rules.** Six checks run before scoring, and each has a threshold that blocks publication
of the cycle rather than merely warning:

| Check | Rule | Threshold | Action on breach |
|---|---|---|---|
| Q1 Completeness | Non-null consent state per row | 100% | Block; resolve consent source |
| Q2 Freshness | Max feature age | 26 hours | Degrade to baseline |
| Q3 Uniqueness | One row per member-cycle | 0 duplicates | Deduplicate, then block |
| Q4 Range | `p1`, `p0` in [0,1]; `Δp` in [−1,1] | 0 violations | Quarantine rows |
| Q5 Referential | `offer_id` resolves to an active offer | 100% | Block |
| Q6 Drift | Share of rows with unseen category codes | ≤2% | Flag for review |

Q2 is the interesting one, because a freshness breach should not stop the programme. Degrading to
the baseline keeps the queue running with known properties, which is better than either stopping or
scoring on stale features.

**Exclusions.** Rows are excluded for: unresolvable member identity after deterministic matching,
duplicate member-cycle keys retained from a merge, negative-tenure members (data error), offers
whose funding was withdrawn mid-cycle, and members inside a conflicting test cell. The exclusion
rate is measured every cycle and reported as a count and a share; in the constructed cycle it is
3.8% of the 260,000 pre-filter rows, or 9,880 rows.

Excluded rows are not a random draw. Members with unresolved identity skew toward multi-card
households and therefore toward higher frequency, so a policy evaluated only on resolvable members
will over-state its own coverage and under-state the frequency problem.

**Leakage.** Four leakage channels were audited, and the audit is the substantive part of this
subsection:

1. *Outcome leakage*: a feature built from the 28-day post-cutoff window. Detected by a timestamp
   audit that compares each feature's definition window against the cut-off. Not present in the
   carried-forward build; present in a rejected prototype whose offline AUC was 0.91, which is
   itself the tell.
2. *Treatment leakage*: the contact log for the current cycle appearing as a feature. Detected by
   schema review; the contact table must be joined with `event_ts < cut_off`.
3. *Selection leakage in training*: training only on rows that were previously contacted, which
   makes `p0` unidentifiable because untreated rows are absent from the training set. Detected by
   counting control rows per decile; if any decile has fewer than 500 control rows, the decile
   estimate is not reportable.
4. *Label leakage through returns*: using final net margin when the outcome window closes before
   the return tail does, which makes recent cycles look better. Detected by comparing gross and net
   outcomes by outcome age.

The first channel is the one that produced the most convincing wrong answer, and it produced it in
the most expensive way: a prototype with a beautiful offline score that failed immediately in
shadow mode because the winning feature was a post-purchase signal.

### 5e Reproducibility & Method-Selection Ledger

The ledger records, per cycle: cut-off timestamp, snapshot identifier, feature-build commit,
model artifact hash, policy version, threshold applied, budget, review capacity, and the operator
who approved publication. Replay means re-running the same snapshot with the same artifact and
obtaining the same selected set; anything else is a different decision.

A replay check is only meaningful if it is exact. The constructed cycle was replayed three times,
and the selected sets matched as sets on all three runs. They did not match in order, because the
greedy sort had no deterministic tie-break, which was fixed by sorting on `(value, member_id)`. That
defect would never have appeared in a metric; it would have appeared in an audit six months later.

Method selection is recorded with its reason, so that a later reader can see what was rejected and
why:

| Decision | Chosen | Rejected | Reason recorded |
|---|---|---|---|
| Target | Margin-weighted uplift | Propensity | Objective mismatch; see 5c |
| Learner | Two-model difference | Single model with treatment flag | Flag model entangles `p0` and `p1` |
| Calibration | Decile isotonic on holdout | Raw scores | Threshold applied to a score requires calibration |
| Selection | Greedy sort | 0-1 knapsack | Equal costs make the knapsack degenerate |
| Gate | Post-selection partition | Re-ranking | Gate must not reorder a value-optimal set |
| Frequency | Quarter cap of 2 | Cooldown only | Cooldown permits 6 contacts per quarter |
| Holdout | 10% of contacted band, household-level | Member-level 5% | Interference; see 4 |

Nothing in this case was executed against production data. The ledger exists to make the eventual
execution checkable, and every row in it is a decision that a reviewer can reverse on the record.

Takeaway: reproducibility is a property of the snapshot and the artifact together, and a replay
that matches as a set but not as a sequence is a defect worth fixing before an audit finds it.

## 6. Analysis Walkthrough

Six derivations follow. Each uses the same anatomy so that a reader can audit the arithmetic without
reconstructing the argument: the formula, where it applies, the constraint that bounds it, a
numerical example with intermediate values, and the interpretation. All inputs are ILLUSTRATIVE.

### 6.1 Net value per contact

```text
FORMULA
  V_i = m * delta_p_i - c

WHERE
  V_i        net expected value of contacting member-week i, dollars
  m          expected gross margin per incremental purchase, dollars (24.00)
  delta_p_i  estimated change in purchase probability caused by contact, proportion
  c          cost of making one contact, dollars (0.62)

CONSTRAINT
  The formula is valid only if delta_p_i is an estimate of the causal difference
  p1_i - p0_i, and only if c is the full marginal cost of the contact including
  the expected incentive payout. If delta_p_i is a propensity, the formula is
  undefined for the purpose of selection, because it omits the counterfactual.

NUMERICAL EXAMPLE
  Decile 3 member: delta_p = 0.060, m = 24.00, c = 0.62
  gross expected gain  = 24.00 * 0.060 = 1.440
  net value            = 1.440 - 0.620 = 0.820 dollars per member-week
  For 25,000 such member-weeks:
  total                = 25,000 * 0.820 = 20,500.00 dollars per cycle

INTERPRETATION
  A contact is worth making only when delta_p exceeds c / m = 0.62 / 24.00 = 0.02583,
  i.e. 2.58 percentage points. Below that threshold the funded offer transfers margin
  to members who would have bought anyway; above it, the contact creates margin.
```

The break-even threshold is the single most useful number in the case, because it converts an
abstract model output into an operational cut. It is also the number that moves when any of its
three inputs move, which is why it belongs in the monitoring dashboard rather than in a document.

### 6.2 Whole-cycle selection under a binding budget

The eligible population is 250,000 member-weeks in ten equal deciles of 25,000, ordered by estimated
uplift. The estimates below are constructed to be plausible for a grocery loyalty programme: a
strong head with high influence, a long tail with none.

```text
FORMULA
  V_total(S) = sum over i in S of ( m * delta_p_i - c )
  subject to |S| <= B

WHERE
  S     selected set of member-weeks
  B     contact budget, 120,000 contacts per cycle
  m     24.00 dollars per incremental purchase
  c     0.62 dollars per contact

CONSTRAINT
  Greedy selection by V_i is optimal only when every contact consumes exactly one
  unit of budget. If channel costs differ (SMS 0.11 versus email 0.06), the problem
  becomes a knapsack and the greedy order is no longer optimal.

NUMERICAL EXAMPLE
  decile  rows    delta_p   V_i        rows selected  value
  1       25,000  0.0910    1.5640     25,000         39,100.00
  2       25,000  0.0740    1.1560     25,000         28,900.00
  3       25,000  0.0600    0.8200     25,000         20,500.00
  4       25,000  0.0490    0.5560     25,000         13,900.00
  5       25,000  0.0400    0.3400     20,000          6,800.00
  6       25,000  0.0320    0.1480          0              0.00
  budget exhausted at 120,000 contacts, inside decile 5
  total value = 39,100 + 28,900 + 20,500 + 13,900 + 6,800 = 109,200.00 dollars

INTERPRETATION
  The threshold implied by the budget is not the break-even threshold. Break-even is
  2.583 points, which would select deciles 1-6 in full (150,000 rows). The budget
  instead cuts inside decile 5 at 4.00 points, so 30,000 rows that still carry
  positive value (5,000 from decile 5, all 25,000 of decile 6) are correctly left
  uncontacted. Two different numbers with two different causes: 2.583 is economics,
  4.000 is capacity.
```

The distinction between the economic threshold and the capacity threshold is where most programme
dashboards lose their audience. A chart showing "we contacted only 48% of eligible members" reads
as a failure; the same chart with the budget line drawn on it reads as a correct allocation.

### 6.3 Propensity ranking versus uplift ranking

This derivation isolates the `evaluation-mismatch` pattern with three constructed groups.

```text
FORMULA
  Loss from ranking on the wrong target
    L = V_total(S_uplift) - V_total(S_propensity)

WHERE
  S_propensity  top-B rows by p1 (probability of purchase given contact)
  S_uplift      top-B rows by m * (p1 - p0) - c
  B             120,000 contacts

CONSTRAINT
  The comparison is valid only when both policies face the same budget, the same
  eligible population, and the same cost. If the propensity policy also spends more
  per contact, part of the difference is budget, not ranking.

NUMERICAL EXAMPLE
  group  rows      p1     p0     delta_p   V_i
  S      60,000    0.55   0.50   0.0500    0.580
  N      90,000    0.22   0.14   0.0800    1.300
  L      100,000   0.03   0.03   0.0000   -0.620

  Rank by p1: take all 60,000 of S, then 60,000 of N
    value = 60,000 * 0.580 + 60,000 * 1.300 = 34,800 + 78,000 = 112,800
  Rank by V_i: take all 90,000 of N, then 30,000 of S
    value = 90,000 * 1.300 + 30,000 * 0.580 = 117,000 + 17,400 = 134,400
  Loss from propensity ranking = 134,400 - 112,800 = 21,600 per cycle

INTERPRETATION
  The propensity policy looks better on the metric it optimises: its response rate is
  38.5% ((60,000*0.55 + 60,000*0.22) / 120,000) against 30.3% for the uplift
  policy ((90,000*0.22 + 30,000*0.55) / 120,000). The worse policy wins the vanity
  metric and loses 21,600 dollars of margin per cycle, which is 1,123,200 per year.
```

This is the crux. A response-rate comparison would have shipped the propensity policy and celebrated
a 8.2-point improvement in response rate while destroying value. The metric did not merely fail to
detect the error; it pointed in the wrong direction, which is the defining property of an
evaluation-mismatch case rather than a mere measurement-error case.

### 6.4 Contact frequency under decaying lift

The cooldown constraint is often mistaken for a frequency policy. It is not, and the arithmetic
shows why.

```text
FORMULA
  delta_p_k = delta_p_1 * r^(k-1),  for the k-th contact within a quarter
  V_k       = m * delta_p_k - c

WHERE
  delta_p_1  lift of the first contact in the quarter, 0.060
  r          per-repeat retention of lift, 0.62 (ILLUSTRATIVE)
  k          contact index within the quarter, 1..6
  m, c       as before

CONSTRAINT
  The geometric decay is a local approximation. It assumes the member's response
  to the k-th contact does not depend on the creative used, and it assumes the
  quarter boundary does not itself reset behaviour. Both assumptions are testable
  and neither is tested here.

NUMERICAL EXAMPLE
  k=1: delta_p = 0.0600, V = 24*0.0600 - 0.62 =  0.8200
  k=2: delta_p = 0.0372, V = 24*0.0372 - 0.62 =  0.2728
  k=3: delta_p = 0.0231, V = 24*0.0231 - 0.62 = -0.0665
  k=4: delta_p = 0.0143, V = 24*0.0143 - 0.62 = -0.2768
  k=5: delta_p = 0.0089, V = 24*0.0089 - 0.62 = -0.4072
  k=6: delta_p = 0.0055, V = 24*0.0055 - 0.62 = -0.4881

  two-contact total   =  0.8200 + 0.2728 =  1.0928 per member per quarter
  six-contact total   =  1.0928 - 0.0665 - 0.2768 - 0.4072 - 0.4881 = -0.1458
  break-even decay r* satisfies 24 * 0.060 * r* = 0.62, so r* = 0.4306

INTERPRETATION
  With a 14-day cooldown a member can receive 6 contacts in a 91-day quarter. The
  six-contact schedule is net negative at the margin, and the two-contact schedule is
  optimal. The right frequency control is therefore a quarterly cap of 2, with the
  cooldown acting only as a floor on spacing. If the measured decay r falls below
  0.431, even the second contact stops paying and the cap becomes 1.
```

The boundary case is informative: at `r = 0.43` the second contact is exactly break-even, so a
programme whose measured decay is 0.40 should run one contact per quarter and no more. A programme
cannot discover this from a redemption dashboard, because redemptions keep accruing with every send
even while margin declines.

### 6.5 Minimum detectable effect for the validation holdout

An evaluation design that cannot see the effect it asserts is not an evaluation. The calculation
below establishes the resolution the holdout can deliver and, crucially, at what aggregation.

```text
FORMULA
  n_per_arm = 2 * (z_alpha/2 + z_beta)^2 * p(1-p) / delta^2

WHERE
  n_per_arm  rows required in each arm for a two-proportion comparison
  z_alpha/2  1.9600 for two-sided alpha 0.05
  z_beta     0.8416 for power 0.80
  p          baseline purchase probability in the window, 0.30
  delta      effect to detect, in proportion

CONSTRAINT
  The formula assumes independent rows. Member-weeks within a household are not
  independent, and repeated cycles for one member are not independent. The effective
  sample is smaller than the row count by the design effect.

NUMERICAL EXAMPLE
  (z_a + z_b)^2 = (1.9600 + 0.8416)^2 = 2.8016^2 = 7.8490
  p(1-p) = 0.30 * 0.70 = 0.2100
  numerator = 2 * 7.8490 * 0.2100 = 3.2966

  delta = 0.020 -> n = 3.2966 / 0.0004 =  8,241 per arm
  delta = 0.010 -> n = 3.2966 / 0.0001 = 32,966 per arm
  delta = 0.005 -> n = 3.2966 / 0.000025 = 131,863 per arm

  One decile has 25,000 rows. At a 20% holdout the control arm holds 5,000:
  delta resolvable = sqrt(3.2966 / 5,000) = sqrt(0.0006593) = 0.02568
  i.e. 2.57 percentage points, in a single cycle.

INTERPRETATION
  A single-cycle decile readout resolves 2.57 points. Decile 6 is estimated at 3.20
  points and is therefore just detectable; decile 7 at 2.50 points is not, and sits
  on the wrong side of the resolution limit even though it sits on the wrong side of
  break-even too. Pooling four cycles of the same decile gives 20,000 control rows and
  resolves 1.28 points, which is the smallest aggregation worth reporting.
```

The design effect is not hand-waved. With 1.35 member-weeks per household per cycle and an
intra-household correlation of 0.02 for the purchase outcome, the design effect is
`1 + (1.35 - 1) * 0.02 = 1.007`, which is negligible for the purchase probability and is *not*
negligible for the margin outcome, where household-level spend correlation is far higher. The
honest statement is that the purchase-probability readout is nearly unaffected and the dollar
readout should be treated as clustered.

Multiple comparisons matter here. Testing ten deciles at a nominal 0.05 gives roughly a 40% chance
of at least one false positive under the null, and the correction is not free:

```text
Bonferroni: alpha' = 0.05 / 10 = 0.005, so z = 2.8070
  (2.8070 + 0.8416)^2 = 3.6486^2 = 13.3123
  n per arm for delta = 0.020: 2 * 13.3123 * 0.21 / 0.0004 = 13,978
  versus 8,241 uncorrected, a factor of 1.70 in sample size
```

A programme that reports ten deciles with confidence intervals and no correction is reporting a
mixture of real and noise deciles with equal authority. The practical resolution used here is to
pre-register three coarsened bands (top, middle, bottom) rather than ten deciles, which restores
power without inviting a fishing expedition.

### 6.6 Margin-mix sensitivity: when the offer changes what is bought

The value formula assumed a single margin per incremental purchase. Offers change mix, so the
assumption fails exactly where the offer is most effective.

```text
FORMULA
  m_eff = basket_value * margin_rate_effective
  delta_p_required = c / m_eff

WHERE
  m_eff                effective margin per incremental purchase, dollars
  basket_value         80.00 dollars per purchase (ILLUSTRATIVE)
  margin_rate_effective blended margin of the incremental basket
  delta_p_required      break-even lift for the policy to add value

CONSTRAINT
  The formula holds only if the incremental purchase is a single basket at a single
  margin rate. When the offer shifts mix toward a low-margin promoted category, the
  blended rate falls and the required lift rises; the policy must then be re-solved
  rather than re-used with a new constant.

NUMERICAL EXAMPLE
  blended case:   m = 80.00 * 0.30 = 24.00  -> delta_p_required = 0.62/24.00 = 0.02583
  promoted case:  m = 80.00 * 0.22 = 17.60  -> delta_p_required = 0.62/17.60 = 0.03523
  ratio of thresholds = 0.03523 / 0.02583 = 1.364

  At the 24.00 margin, decile 6 (3.20 points) clears break-even and is worth contacting.
  At the 17.60 margin, decile 6 sits below the 3.52-point threshold and must be dropped,
  which removes 25,000 candidate contacts and 3,700 dollars of estimated value from the
  cycle; the freed budget moves to decile 5, not to decile 6.

INTERPRETATION
  A 26.7% fall in margin rate raises the break-even lift by 36.4%, because the
  threshold is inversely proportional to margin. The elasticity is exactly -1 in the
  ratio: a one percent fall in margin raises the threshold by slightly more than one
  percent. Programmes that fund deep discounts are therefore the most exposed to this
  error, and they are also the ones most likely to use a single blended margin rate.
```

The boundary case is a category with a margin rate of 9%, where the required lift becomes
`0.62 / (80.00 * 0.09) = 0.62 / 7.20 = 8.61` percentage points. No decile in the constructed
population reaches 8.61 points, so a 9%-margin offer should not be funded at all — a conclusion
that is invisible to any analysis that reports redemptions instead of margin.

### 6.7 Where the formulas break

Three limits are worth stating explicitly, because each is reachable in normal operation.

*Fixed cost per contact.* The break-even threshold assumes `c` is constant per contact. When SMS
costs 0.11 and email costs 0.06, the threshold differs by channel
(`0.11/24.00 = 0.458` versus `0.0258`), and the selection problem becomes a knapsack in which the
greedy order is no longer optimal. The practical fix is to solve per channel with a shared budget,
or to fix the channel by member preference before ranking.

*Non-linear value.* The formula is linear in `delta_p`, which fails when the offer has a
threshold structure ("spend $50 to earn 500 points"). Near the threshold, a small uplift can move a
member across a discontinuity, so the value is not proportional to the probability change. A
threshold offer must be valued with the distribution of basket values, not with its mean.

*Saturation and interference.* `V_i` is computed per member-week as if members were independent.
At a 48% contact rate within the eligible band, a contacted member's behaviour affects the
availability of promoted stock and the traffic in the store, which changes uncontacted members'
outcomes. The direction of the bias is toward understating the total effect and overstating the
per-contact effect, and the magnitude cannot be recovered from a member-level holdout.

Takeaway: the arithmetic is simple and each simplification has a named failure boundary, so a
reviewer can check whether the boundary is active rather than arguing about the conclusion.
