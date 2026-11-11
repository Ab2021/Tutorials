## 4. Core Formulas with Full Derivation

Every result below is derived from a probability statement about a pair, then converted into a decision
rule about a ledger row. The section ends each subsection with an explicit validity condition: the
conditions under which the number you just computed is meaningless.

### 4.1 Base case — evidence as a log-likelihood ratio

**FORMULA [IC-01].** For one comparison field `i`, the weight contributed when the field agrees is

```text
w_i = log2( m_i / u_i )

WHERE:
  w_i  [bits, real]        evidence that agreement on field i favours "same person"
  m_i  [fraction, (0,1]]   P(field i agrees | the pair IS a true duplicate)
  u_i  [fraction, (0,1))   P(field i agrees | the pair is NOT a duplicate)
```

**WHERE (derivation).** Bayes' rule applied to the pair-level hypothesis. Let `H` be "duplicate" and
`A_i` be "field `i` agrees". Then posterior odds equal prior odds times the likelihood ratio:

```text
P(H | A_i) / P(not H | A_i)  =  [ P(A_i | H) / P(A_i | not H) ] * P(H)/P(not H)
                             =  ( m_i / u_i ) * prior odds
```

Taking log base 2 turns the multiplication into addition, so many independent fields accumulate as a
sum. A weight of `log2(10) = 3.3219` bits means the field makes duplication ten times more likely than
it was before that field was examined.

**CONSTRAINT.** Three conditions, all necessary. (i) `u_i > 0` strictly — a field that never agrees
accidentally has no measurable non-duplicate agreement rate and its weight is unbounded. (ii) The pair
set used to estimate `u_i` must be the population the score is later applied to; estimating `u_i` among
records that already survived a tight blocking key yields a `u` that is far too high, so the weight is
far too low. (iii) Fields must be conditionally independent given the hypothesis — see §4.4 for the
quantified cost of ignoring this.

**NUMERICAL EXAMPLE.** Three fields, each estimated on the same candidate population:

| Field | `m_i` | `u_i` | ratio | `w_i` (bits) |
|---|---|---|---|---|
| Mobile phone, normalised to E.164 | 0.93 | 0.020 | 46.5000 | 5.5392 |
| Surname, casefolded, no punctuation | 0.88 | 0.045 | 19.5556 | 4.2895 |
| Date of birth, exact day | 0.95 | 0.0034 | 279.4118 | 8.1262 |

Arithmetic check for the first row: `ln(46.5) = 3.83945`; `3.83945 / 0.693147 = 5.5392`. Second row:
`0.88/0.045 = 19.55556`; `ln(19.55556) = 2.97326`; `/0.693147 = 4.2895`. Third row: `0.95/0.0034 =
279.41176`; `ln(279.41176) = 5.63269`; `/0.693147 = 8.1262`. Rounding each weight to four decimals is
safe here because the decision threshold in §4.2 is `11.8698`, which is more than `0.05` from every
reachable partial sum in the table below.

The complementary weight, when a field *disagrees*, is

```text
w_i(disagree) = log2( (1 - m_i) / (1 - u_i) )

CONSTRAINT: 1 - u_i > 0; the disagree weight is always negative when m_i > u_i.
NUMERICAL EXAMPLE (phone): log2(0.07 / 0.98) = log2(0.0714286) = -3.8074 bits.
```

Note the asymmetry that practitioners routinely get wrong: agreement on the phone adds `+5.54` but
disagreement subtracts only `3.81`. A missing phone is not evidence of difference, and the disagree
weight already carries that: `0.07` of true duplicates have a different phone recorded, which is exactly
the share where one record holds an old number.

**INTERPRETATION.** Weights are per-field *and per-population*. The same phone field is worth `5.54`
bits in a population where phones are sparse and well-maintained, and much less in a population where
phone numbers are shared household landlines. Any figure quoted without the population it was estimated
on is not reusable.

**VALIDITY CONDITION [IC-01].** The formula holds for a single field evaluated as a binary agreement
event. It does not hold for (a) continuous fields binarised by a fixed tolerance without accounting for
the tolerance's own error rate, (b) fields with more than two outcomes collapsed to agree/disagree
without re-estimating `m` on the collapsed event, or (c) fields measured on one system and matched
against a field measured on a differently-captured system, where `m` differs by capture channel.

### 4.2 General case — cost-weighted threshold on accumulated evidence

**FORMULA [IC-02].** Given prior log-odds `lambda0` inside the candidate set and total evidence `W`,
the Bayes-optimal rule under asymmetric costs links the pair when

```text
W + lambda0  >  log2( C_FM / C_FN )

equivalently   W* = log2( C_FM / C_FN ) - lambda0

WHERE:
  lambda0 [bits, real]  = log2( T / (P - T) )   prior log-odds, T = true duplicate pairs in P
  C_FM    [CU per pair] cost of treating two different people as one (a false merge)
  C_FN    [CU per pair] cost of leaving one person as two (a missed merge, a false split)
  W*      [bits, real]  the score threshold applied to W
```

**WHERE (derivation).** Two actions at a pair. Linking costs `(1 - p) * C_FM` in expectation because it
is wrong with probability `1 - p`. Not linking costs `p * C_FN`. Link when the first is smaller:

```text
(1 - p) C_FM  <  p C_FN      divide by (1-p) C_FN:
C_FM / C_FN   <  p / (1 - p) = posterior odds
log2(C_FM/C_FN) <  W + lambda0
```

The rule is a statement about *odds*, which is why the threshold is a subtraction from a log-odds
quantity and not a comparison against a probability. Converting the threshold into a probability and
comparing probabilities with `>` gives the same answer only when the conversion is applied identically
on both sides; teams that mix the two produce thresholds that look right and behave wrong near the
boundary where most of the actionable volume sits.

**CONSTRAINT.** (i) `C_FM` and `C_FN` must both be expressed in the same unit and must both be
*marginal* at the decision point, not average across all decisions. (ii) The prior `lambda0` must be
computed on the candidate set actually scored, after blocking — using the full population's duplication
rate while scoring inside a `surname + postcode` block overstates the prior by orders of magnitude,
because blocks were chosen precisely because they are enriched. (iii) The decision must be per pair, so
the cost model must be additive; a cost that is superlinear in cluster size breaks the derivation, and
§4.3(b) handles that case separately.

**NUMERICAL EXAMPLE.** Constructed ILLUSTRATIVE values: `P = 2,000,000` candidate pairs, `T = 4,000`
true duplicate pairs.

```text
prior probability  = 4,000 / 2,000,000 = 0.002
prior odds         = 0.002 / 0.998    = 0.00200401
lambda0            = log2(0.00200401) = ln(0.00200401)/0.693147
                   = (-6.21260)/0.693147 = -8.9628 bits

cost ratio C_FM/C_FN = 45 / 6 = 7.5
log2(7.5)            = ln(7.5)/0.693147 = 2.01490/0.693147 = 2.9069 bits

W* = 2.9069 - (-8.9628) = 11.8697 bits
```

Now apply to the three-field scores from §4.1:

| Evidence pattern | `W` (bits) | `W + lambda0` | posterior odds | posterior prob | action at `W* = 11.87` |
|---|---|---|---|---|---|
| phone + surname + DOB | 17.9545 | 8.9917 | 508.7 | 0.99804 | auto-link |
| phone + DOB | 13.6655 | 4.7027 | 26.05 | 0.96303 | auto-link |
| phone + surname | 9.8283 | 0.8655 | 1.8223 | 0.64568 | review |
| DOB + surname | 12.4152 | 3.4524 | 10.94 | 0.91626 | auto-link |
| phone only | 5.5393 | −3.4235 | 0.09306 | 0.08513 | no-link |
| DOB only | 8.1262 | −0.8366 | 0.5599 | 0.35895 | no-link |
| surname only | 4.2890 | −4.6738 | 0.03918 | 0.03771 | no-link |
| none | −7.6168 | −16.5796 | 1.00e−5 | 1.00e−5 | no-link |

Arithmetic check on the third row: `5.5393 + 4.2890 = 9.8283`; `9.8283 − 8.9628 = 0.8655`;
`2^0.8655 = e^(0.8655 × 0.693147) = e^0.59994 = 1.8220`; `1.8220 / 2.8220 = 0.64564`. The row is below
the threshold by `2.04` bits, which is roughly a factor of four in odds — materially uncertain, and
therefore a review candidate rather than a reject.

**INTERPRETATION.** The threshold is not a quality statement; it is a cost statement. With `C_FM = 45`
and `C_FN = 6`, a false merge is treated as 7.5× worse than a missed merge, so the rule demands evidence
that pushes odds past 7.5:1. Raise `C_FM` to `450` and the log2 term becomes `6.2296`, moving `W*` to
`15.19` bits — which disqualifies the "DOB + surname" pattern (`12.42`) that the cheaper cost model
happily accepted. That single cost change moves roughly a fifth of accepted links into review, and it is
the change most worth arguing about with the business before launch rather than after.

**VALIDITY CONDITION [IC-02].** The rule is optimal only if the score is calibrated on the same
population and window it is applied to, costs are constant per pair, and errors are independent across
pairs. Independence fails catastrophically in clusters — see §4.3(b) — and calibration drifts whenever
source systems change capture forms.

### 4.3 Variants

#### (a) Deterministic key: when exact match is enough

**Mechanism block.** Preconditions: a key derived from fields that are either present-and-correct or
absent. State transition: records sharing the normalised key are unioned without scoring. Invariant: the
key's collision rate must be low relative to the true-duplicate rate; justification is a birthday-bound
argument. Worked trace and failure boundary follow.

```text
FORMULA [IC-03]:  E[collisions] ~= N (N - 1) / (2 K)

WHERE:
  N [records]  records carrying the key
  K [keys]     size of the key space (16^h for h hex characters)
  E[collisions] [pairs] spurious pairs from random collision, assuming uniform keys
```

Worked trace at `N = 1,200,000`:

```text
h = 12 hex chars: K = 16^12 = 2.81475e14
  E[collisions] = (1.2e6 * 1,199,999) / (2 * 2.81475e14)
                = 1.439999e12 / 5.62950e14 = 0.002558 pairs
h = 8 hex chars:  K = 16^8  = 4.29497e9
  E[collisions] = 1.439999e12 / (2 * 4.29497e9) = 1.439999e12 / 8.58993e9 = 167.6 pairs
```

**Failure boundary.** Deterministic keys fail on *missing*, not on *wrong*. If 14% of records lack the
key field, those records are silently never linked, and because they never enter the audit's candidate
set, recall looks perfect. The discriminating check is a null-key census by source: `count(*) where key
is null` grouped by source must be explained source by source, not aggregated.

#### (b) Transitive closure: the amplification that pair precision hides

Link acceptance is pairwise. Entity formation is transitive. The gap between the two is where identity
systems quietly destroy their own quality.

```text
FORMULA [IC-04]:  spurious cross pairs from one bridging link  =  a * b

WHERE:
  a [entities]  distinct people in the cluster on one side of the bridge
  b [entities]  distinct people in the cluster on the other side
  the bridge is a false link between a record in one cluster and a record in the other
```

Worked trace: cluster left has 3 records belonging to 2 real people; cluster right has 4 records
belonging to 3 real people. A single false link merges them into one entity of 7 records that
misrepresents 5 people.

```text
misattributed person pairs = 2 * 3 = 6
records now exposed to the wrong person's attributes = 7 records, 5 people
if this entity is in a campaign audience, 5 people receive a message about another person's purchase
```

**Discriminating check for closure damage.** For every component of size ≥ 3, compute the minimum
pairwise score among its internal links. A component whose minimum internal score sits within
`0.5` bits of the threshold is a bridge candidate; list it with the pair that supplies the minimum. In a
healthy run, minimum internal scores are comfortably above `W*` because links made on strong evidence
cluster with links made on strong evidence.

#### (c) Repair changes rates in both directions

Identity repair is usually sold as a cleanup. It is a change of estimator, and it can move a rate either
way depending on whether the numerator or the denominator loses proportionally more.

```text
FORMULA [IC-05]:  r' = r * (1 - s_n) / (1 - s_d)

WHERE:
  r   [fraction] pre-repair rate
  s_n [fraction] share of the numerator removed or corrected by the identity change
  s_d [fraction] share of the denominator removed or corrected by the identity change
```

Two cases with the same denominator effect and opposite conclusions:

```text
case A  s_n = 0.20, s_d = 0.12
  factor = 0.80 / 0.88 = 0.90909
  r = 0.34 -> r' = 0.34 * 0.90909 = 0.30909   (rate FALLS 3.09 points)

case B  s_n = 0.12, s_d = 0.20
  factor = 0.88 / 0.80 = 1.10000
  r = 0.34 -> r' = 0.34 * 1.10000 = 0.37400   (rate RISES 3.40 points)
```

**Practical consequence.** A merge that consolidates repeat purchasers into single entities reduces both
numerator (repeat events now count once for one person) and denominator (two profiles become one), but
not equally: the numerator loses the entire duplicated event, the denominator loses one unit. That is
case A. A merge that consolidates *dormant* duplicate profiles while their purchase events were already
correctly attributed is case B. The sign of the reported movement tells you which is happening, and the
sign is a diagnostic, not an outcome.

#### (d) Experiment contamination from cross-arm units

When the same person appears in control and treatment because two profiles were resolved separately but
the person is one human being, treatment effects attenuate.

```text
FORMULA [IC-06]:  observed effect  ~=  (1 - c) * true effect

WHERE:
  c [fraction, 0-1]  share of units whose records appear in both arms, weighted by exposure
```

Worked trace: true incremental conversion lift is `4.0` percentage points; `c = 0.25`.

```text
observed lift = (1 - 0.25) * 4.0 = 3.0 pp
implied required sample to detect 3.0 pp vs 4.0 pp at 80% power, two-sided 5%:
  n per arm ~= 16 * p(1-p) / delta^2    (rule of thumb, p ~= 0.10 baseline)
  delta 0.040: 16 * 0.09 / 0.0016 = 900 per arm
  delta 0.030: 16 * 0.09 / 0.0009 = 1600 per arm
```

The attenuation costs 78% more sample for the same detectable effect. The fix is not a bigger sample; it
is resolving the identity *before* randomisation and randomising at entity grain.

#### (e) Blocking recall bounds everything downstream

```text
FORMULA [IC-07]:  achieved pair recall  =  rho * precision_at_threshold

WHERE:
  rho [fraction] blocking recall: true pairs that survive candidate generation, / all true pairs
```

If blocking has `rho = 0.86` and the scorer's precision at threshold is `0.97`, achieved recall is
`0.86 * 0.97 = 0.8342`. The scorer cannot recover what blocking discarded, and no amount of threshold
tuning moves `rho`. Measuring `rho` requires a labelled sample of true pairs drawn *independently* of
the blocking keys — typically a small hand-adjudicated sample assembled by a different method (for
example a union of three different blocking strategies), because pairs generated by the production
blocker cannot estimate the blocker's own misses.

#### (f) Consent-scoped activation denominator

```text
FORMULA [IC-08]:  contactable entities  =  sum over entities of  eligible(entity, purpose, channel, t)

WHERE eligibility is the conjunction of:
  consent(entity, purpose) valid at t
  channel verified at t        (not merely present)
  not suppressed at t          (complaint, bounce, regulatory)
  link confidence sufficient   (entity not sitting in the review band)
```

Worked trace: `1,000` entities in a loyalty segment.

```text
consent for marketing purpose, any channel       820
of those, email verified                          611
of those, not suppressed                          586
of those, entity outside the review band          574

reachable fraction tau = 574 / 1,000 = 0.574
```

A report that states "1,000 loyalty customers targeted" and a send that reaches 574 are both true and
describe different denominators. The reachable fraction `tau = 0.574` is the number the media plan
should use; the `0.574` also tells you where the next percentage point of value sits — `820 − 611 = 209`
entities had consent but no verified channel, which is a verification problem, not a consent problem.

### 4.4 Limits and counterexamples

#### (a) The independence assumption, quantified

Two fields that both change when a person moves — phone and address — are positively correlated given
"duplicate". Summing their weights overstates the evidence.

```text
field 1: m1 = 0.93, u1 = 0.020  ->  w1 = 5.5393
field 2: m2 = 0.91, u2 = 0.015  ->  w2 = log2(60.6667) = 5.9226
independent sum                  =  11.4619 bits

actual joint agreement rates (measured on a labelled sample):
  m12 = 0.90   (both agree among duplicates)
  u12 = 0.0006 (both agree among non-duplicates)
  true joint weight = log2(0.90 / 0.0006) = log2(1500) = 10.5508 bits

overstatement = 11.4619 - 10.5508 = 0.9111 bits  ->  odds overstated by 2^0.9111 = 1.879x
```

`1.88×` in odds is enough to move a pair across an `11.87`-bit threshold in the crowded region. The
counterexample that makes the direction clear: if two fields were *perfectly* redundant, the joint
weight would equal either alone (`5.54`), while the naive sum would claim `11.08` — twice the evidence
that exists.

**Discriminating check.** Split a labelled sample by the presence of one field and recompute the
other's weight in each split. If the weight moves more than roughly `0.3` bits between splits, the
fields interact and the sum must be recalibrated jointly (or replaced by a model that learns the
interaction, at the cost of explainability).

#### (b) Base-rate shift inside blocks

`u_i` estimated on the full population is wrong for every block. A `surname = "Nguyen"` block has a
much higher accidental surname agreement rate than the population average. Applying population `u` inside
that block makes surname agreement look like strong evidence when it is nearly guaranteed. The validity
boundary is sharp: field weights are only transferable between populations with comparable marginal
distributions of the field.

#### (c) Scores are not commensurable across vendors

A vendor score of `0.92` and another vendor's `0.92` denote different quantities (sometimes a
similarity, sometimes a posterior, sometimes a monotone transform with no probabilistic meaning). The
only portable object is a *calibrated probability with a stated population and window*, plus the
threshold expressed as a cost ratio. If you cannot recompute the vendor's score into a probability on
your own labelled sample, you cannot merge the two vendors' outputs — you can only choose one.

#### (d) Where the whole framework breaks

If the entity you need is not "a person" but "a buying decision unit", every probability above is
estimated for the wrong target. Two adults sharing a household genuinely share a phone, genuinely share
an address, and genuinely have different purchase histories. The fields that carry the most weight for
person resolution carry near-zero weight for household resolution, and `u_i` collapses toward `m_i`,
driving every weight toward `0` and the score toward `lambda0`. There is no threshold that fixes a
target-definition error; the decision must be re-specified before any arithmetic is applied.
