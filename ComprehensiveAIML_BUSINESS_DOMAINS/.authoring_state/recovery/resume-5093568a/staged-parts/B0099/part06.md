## 9. Complexity & Cost Analysis

Diagnosis is not free, and the wrong diagnosis is expensive in a way that compounds. This section
prices the checks, the corrections, and — most importantly — the two error directions of each
detector, so that a team can choose how much evidence to buy.

### 9.1 Cost of the checks themselves

The checks in Section 8 fall into four cost classes. An analyst under pressure should always try the
cheapest class that can separate the live hypotheses; escalating to a replay before doing so is the
single most common waste of a incident budget.

| Class | What it costs | Typical elapsed time | Settles |
|---|---|---|---|
| slice a frozen extract by one dimension | a query on data already materialised | 15–40 minutes | attribution, segment concentration, boundary enumeration |
| join against a second independent source | one join plus a reconciliation query | 1–3 hours | whether two representations agree |
| recompute under an alternate definition | one analyst plus the definition owner | 2–6 hours, mostly waiting on agreement | CONCEPT_CONFUSION entries |
| replay over a closed window | compute plus review capacity | 1–3 days | whether a correction restores the system |

Pricing the first class with an ILLUSTRATIVE loaded rate of 95 currency units per hour: a two-hour
slice check costs about 190, a fourteen-hour replay costs about 1,330. The ratio is 7:1, and the
replay is only justified when the slice has already excluded all but one mechanism. Note that the
replay is not merely 7 times more expensive in money; it consumes the review capacity that also
handles live incidents, so during a busy week the real cost is higher than the table suggests.

### 9.2 Complexity of the computational checks

Three checks dominate the compute budget, and each has a known asymptotic shape that determines when
it will stop working.

**Pairwise identity comparison.** Within a block of `m` candidates, the number of compared pairs is
`m(m-1)/2`, so the work is `O(m^2)` per block and `O(sum_j m_j^2)` overall. Because the sum of
squares is dominated by the largest block, halving the record count does not halve the run: with one
block of 120 records the job compares 7,140 pairs regardless of how many other records exist. This is
why CI-03's symptom is a run-time explosion against a flat record count, and why the correct capacity
metric is the largest block, not the total.

**Candidate-node evaluation for quoting.** If a quote evaluates `c` candidate nodes and each
evaluation is independent, the cost is `O(c)` per request and the p99 is governed by the upper tail of
the candidate-count distribution, not its mean. A service tuned on a mean of 8 candidates and
deployed against a tail of 60 will breach latency while its average latency looks healthy — the
mechanism behind CI-16.

**Queue simulation versus closed form.** A closed-form queueing approximation costs nothing and is
accurate enough for a directional decision away from saturation. For an M/M/1 approximation with
service time `ES` seconds and utilisation `rho = lambda/mu`:

```text
W_q = (rho / (1 - rho)) . ES          [seconds of waiting, exclusive of service]
NUMERICAL (ILLUSTRATIVE): ES = 3600/300 = 12 s per order, mu = 300/hour
  lambda=240 -> rho=0.80 -> W_q = (0.8/0.2) x 12 = 48.0 s
  lambda=270 -> rho=0.90 -> W_q = (0.9/0.1) x 12 = 108.0 s
  lambda=285 -> rho=0.95 -> W_q = (0.95/0.05) x 12 = 228.0 s
INTERPRETATION: a 6% arrival increase from 270 to 285 doubles the wait. Near saturation the
cost of a marginal order is not linear, and no amount of measurement precision recovers the
information lost by planning at the mean.
```

The formula is a teaching approximation, not a model of a real pick line: service times are not
exponential and the system is not a single server. Its value is the shape — the reciprocal
`1/(1-rho)` term — which is right in every queueing system and is the reason capacity plans built at
`rho = 0.95` fail in ways that look like sudden, inexplicable collapses.

### 9.3 Cost of the two error directions

Every detector in Section 8 has a false-positive and a false-negative cost, and they are almost never
symmetric. Choosing a threshold is therefore an economic decision, not a statistical one.

**Identity merging (CI-01, CI-05).** Let `q` be the probability that an unrelated pair matches on the
fields used, and `m` the block size. The expected number of false pairs per block is `m(m-1)q/2`.
At `q = 1.5e-4` and 5,000 blocks per week:

```text
m = 40  -> 780 pairs  -> 0.117 per block ->   585 false pairs per week
m = 80  -> 3,160 pairs -> 0.474 per block -> 2,370 false pairs per week
m = 120 -> 7,140 pairs -> 1.071 per block -> 5,355 false pairs per week
COST: at 12 minutes of contact-handling per false merge and 95 per hour,
      5,355 x 0.2 h = 1,071 hours = 101,745 currency units per week of avoidable handling.
```

The false-negative direction — failing to merge records that are the same person — costs a different
thing: duplicated marketing contact, split order history, and a loyalty balance that looks smaller
than the customer's own record. It is rarely visible in a contact reason, which is why it is
chronically under-invested in. The asymmetry is the point: false merges produce complaints within
hours, failed merges produce quiet erosion over quarters.

**Precision, not just recall.** With prior probability `pi` that a candidate pair is truly the same
person, sensitivity `s = 0.97`, and the pair-level false-match rate `q`:

```text
FORMULA: P(same | match) = pi.s / (pi.s + (1-pi).q)
NUMERICAL, q = 1.5e-4:
  pi = 0.050 -> 0.9971  (0.29% of accepted merges are wrong)
  pi = 0.020 -> 0.9925  (0.75% wrong)
  pi = 0.002 -> 0.9284  (7.16% wrong)
INTERPRETATION: at a fixed matcher, precision collapses as the candidate population gets sparser
in true matches. A block full of one household's records is precise; a block of unrelated
common-name records is not, and the same code produces both.
```

This is the quantitative reason a single global match threshold cannot be correct for every block.
The threshold must be conditioned on the block's prior, which is estimated from the block's own
composition — a design change, not a tuning exercise.

**Slot oversell (CI-17).** The 223-order oversell in the worked example carries a recovery cost that
is not in the slot ledger: at 8 minutes of handling per affected order and 95 per hour,
`223 x 8/60 x 95 = 2,824.7` currency units for one slot. Annualised over a comparable slot each
weekday, the figure exceeds the cost of the capacity review that would have prevented it by an order
of magnitude. The asymmetry here runs the other way from identity: the visible cost of containment is
immediate and the avoided cost is invisible, which is why capacity is chronically under-reserved.

### 9.4 Cost of the corrections

| Correction | One-time cost | Recurring cost | Hidden cost |
|---|---|---|---|
| comparator fix (CI-12, CI-21) | small code change plus a test matrix | none | the old number persists in an unrefreshed deck |
| contract version (CI-07, CI-11) | coordination across teams, one release | schema maintenance | one consumer stays on the old version silently |
| capacity increase (CI-18) | roster and equipment | ongoing operating cost | masks a release defect that will return |
| definition change (CI-04, CI-14, CI-23) | documentation plus fixtures | fixture maintenance | historical comparisons break unless restated |
| replay and restate (all) | compute plus review | none | review capacity is taken from live work |

The hidden-cost column is the one that decides whether a correction survives. A comparator fix whose
old number lives on in a monthly deck has not been corrected; it has been corrected in one place. The
rule that prevents this is that the metric name carries its definition version, so a stale deck shows
a stale name rather than a wrong value under the right name.

### 9.5 The economics of verification

Verification is the step most often skipped under time pressure, and it is the cheapest step in
proportion to what it protects. A verification costs one query or one replay; an unverified
correction that silently fails costs a repeat incident plus the credibility of the last fix.

The asymmetry is stark in the points ledger. Rounding at two levels produces an ILLUSTRATIVE drift of
three points per basket — at 6.1 million baskets per month, 18.3 million points, or 91,500 currency
units of value issued that the customer cannot see in their own arithmetic. The reconciliation query
that would have caught it runs in seconds. There is no incident in this file where the verification
is more expensive than the defect it prevents, and that is the general rule rather than a coincidence:
verification queries on a frozen extract are cheap by construction, which is why the discipline of
building them is worth the small fixed cost.

### 9.6 Where to spend the diagnostic budget

Rank the catalog's subsystems by (frequency x cost per occurrence x difficulty of detection). In
practice, for a loyalty and fulfilment stack, the ordering that survives contact with operations is:

```text
1. availability staleness        frequent, expensive, silent         (CI-07, CI-11)
2. slot capacity versus release  frequent, expensive, loud           (CI-17, CI-18)
3. identity false merges         less frequent, expensive, loud      (CI-01, CI-05)
4. rounding and comparator edges  constant, cheap each, invisible    (CI-12, CI-21)
5. definitional disagreements     periodic, expensive, invisible     (CI-04, CI-14, CI-23)
6. notification delivery gaps     periodic, moderate, invisible      (CI-26)
```

Items 1, 4, and 5 share one property: they produce no alert, because the system is behaving exactly as
specified. They are found by checks, not by monitoring, which means the only way to find them is to
schedule the check. That is the argument for a recurring diagnostic rotation — one item per week from
the top of the list, run against a frozen extract and logged — rather than a purely reactive runbook.

## 10. Key Theorems / Results Reference

The results below are the ones the entries depend on. Each is stated with its condition, because in
every case the condition is where the practical failure lives.

**R-1. Product rule for series reliability.** For independent stage successes,
`R = product s_i`. Fails when stages share a driver, in which case `R` is understated and the joint
failure rate is overstated. The practical corollary: any compound promise rate must be accompanied by
an observed joint rate computed from order-level data, and a material gap between product and
observation is evidence of a shared driver worth investigating on its own.

**R-2. Log-additive attribution.** Because `ln(product x_i) = sum ln(x_i)`, shares of a
multiplicative change are computed from log ratios, and they sum to one up to rounding. Percentage-
point differences do not sum and must never be presented as shares. Corollary: negative shares are
meaningful and must retain their sign; they represent factors that improved while the headline fell.

**R-3. Non-additivity of restoration.** Restoring two factors multiplies their individual restoration
ratios; it does not add their percentage-point gains. At modest degradations the difference is a
rounding effect (0.044 pp in the worked example); at severe ones it is large and always optimistic.
Corollary: a remediation plan budgeted by summing single-factor gains over-promises.

**R-4. Quadratic pair growth in blocking.** For block size `m`, candidate pairs are `m(m-1)/2`; the
union-bound expectation of false pairs is `m(m-1)q/2`. Corollary: capacity is governed by the largest
block, and the largest block is the correct monitoring series.

**R-5. Bayes precision collapse for entity matching.** `P(same|match) = pi.s/(pi.s + (1-pi)q)`. Holding
the matcher fixed, precision falls as the prior falls. At `pi = 0.002` and `q = 1.5e-4`, over 7% of
accepted merges are wrong. Corollary: a threshold must be conditioned on the block's prior.

**R-6. Little's law.** `L = lambda.W` in a stable system, so queue depth equals arrival rate times
mean time in system. At `lambda = 1,400` orders/hour and `W = 0.25` hour, `L = 350` orders. The
result requires stability; applying it to a system whose arrival rate is rising produces a depth that
appears to grow "for no reason".

**R-7. Queue delay near saturation.** `W_q = (rho/(1-rho))ES` for the single-server approximation;
delay is reciprocal in the utilisation gap. Corollary: capacity planned at 0.95 utilisation has a
5% margin whose loss doubles the wait, so plans built at the mean fail at the mode.

**R-8. Drain time under a fixed net rate.** `T = Q/(mu - lambda)` for `mu > lambda`; diverges
otherwise. Corollary: a queue's depth and its slope are separate observations, and containment is
verified by the slope, not the level.

**R-9. Rounding non-commutativity.** `round(sum x) != sum round(x)` whenever any `x` is non-integral;
the per-basket discrepancy is bounded by the number of rounded terms minus one under floor. Corollary:
one rounding site per flow, in the smallest integer unit.

**R-10. Minimum detectable effect.** For a two-arm comparison with equal allocation,
`MDE = (z_{1-alpha/2} + z_{power}).sqrt(2.p.(1-p)/n)` for a proportion. At `p = 0.09` and `n = 13,492`
per arm, MDE is about 0.98 pp; at `n = 2,000`, about 2.54 pp. Corollary: a result below the declared
MDE carries no decision, whatever its p-value, and repeated looks inflate the false-positive rate
independently of the effect size.

**R-11. Weighted versus unweighted pooling.** `r_pooled = sum n_j r_j / sum n_j`; the unweighted mean
answers a different question about the same strata. In the worked example the two differ by 4.62 pp.
Corollary: a rate without its population is not a measurement.

**R-12. Maturity before measurement.** A cohort is comparable only after the stated maturity; an
immature window contains in-flight units whose eventual outcome is unknown, and counting them as
failures manufactures a decline that reverses itself. Corollary: maturity is a required query
parameter and its absence is a review failure, not a stylistic preference.

**R-13. Immutability of past decisions.** A corrected fact may change a future decision and may
annotate a past one, but may never silently restate what a past decision knew. This is a policy
result rather than a mathematical one, and it is the reason the file insists on annotation over
backfill: the arithmetic of a restated decision is clean and the audit trail is not.

**R-14. Mutual exclusivity of outcome categories.** Rates that must be comparable require a
partition: `short_pick + substituted + as_ordered = lines`. Any pair of rates computed from
overlapping definitions can invert (as in CI-19, where substitution rate exceeded the short-pick rate
that generates it), and the inversion is the diagnostic rather than the curiosity.
