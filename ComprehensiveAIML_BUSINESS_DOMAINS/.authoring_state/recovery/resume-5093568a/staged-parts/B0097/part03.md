## 5. Worked Numerical Example

One linkage run on a constructed ILLUSTRATIVE population, small enough to re-derive by hand and complete
enough to exercise every stage: blocking with recall loss, weighted scoring, a cost threshold, a review
band, closure, and a consent gate. All counts and costs are constructed. §5.8 re-derives the headline
numbers by independent routes.

### 5.1 The constructed population

```text
N = 1,000 source records from two systems
  system A: 620 records (store loyalty signup, staff-entered)
  system B: 380 records (web account, self-entered)

E = 780 true entities behind the 1,000 records
  604 entities appear once   -> 604 records
  132 entities appear twice  -> 264 records
   44 entities appear 3x     -> 132 records
  check: 604 + 264 + 132 = 1,000 records, and 604 + 132 + 44 = 780 entities

true duplicate pairs  = 132*1 + 44*3 = 132 + 132 = 264
```

The distinction between "excess records" (`1,000 − 780 = 220`) and "true duplicate pairs" (`264`) matters
and is the first place a derivation goes wrong. Excess records count only once per redundant row; pairs
count every unordered pair inside a group, so a triple contributes three pairs for two excess records.
Using `220` as the pair count would understate the target population by `16.7%`, and every recall figure
computed against it would be overstated by the same factor.

Field availability differs by system, which is the point: `m` and `u` differ because capture differs.

| Field | Present in A | Present in B | Constructed `m` | Constructed `u` | Weight when agreeing (bits) |
|---|---|---|---|---|---|
| Mobile, E.164 | 0.91 | 0.64 | 0.93 | 0.020 | 5.5392 |
| Surname, casefolded | 0.99 | 0.98 | 0.88 | 0.045 | 4.2895 |
| DOB, exact day | 0.86 | 0.71 | 0.95 | 0.0034 | 8.1262 |
| Postcode, unit | 0.97 | 0.93 | 0.72 | 0.038 | 4.2439 |

Recompute the fourth weight, since it is the one that surprises people: `0.72/0.038 = 18.9474`;
`ln 18.9474 = 2.94154`; `2.94154/0.693147 = 4.2439`. Postcode has a strong `m` and still yields a small
weight, because `u = 0.038` says accidental postcode agreement is common — tens of households share a
unit postcode. The weight is a statement about *discrimination*, not about correctness of the field.

### 5.2 Blocking and its recall cost

Three blocking keys, unioned then deduplicated: `(surname, postcode sector)`, `(mobile last 8)`,
`(dob, postcode sector)`.

```text
key 1 (surname, postcode sector):
  distinct keys = 412; records per key = 1,000/412 = 2.4272
  pairs ~= sum C(n,2) ~= 412 * (2.4272 * 1.4272 / 2) = 412 * 1.73212 = 713.6

key 2 (mobile last 8):
  records with mobile = 620*0.91 + 380*0.64 = 564.2 + 243.2 = 807.4 -> 807
  distinct keys = 640 (shared household numbers); mean 1.2609
  pairs ~= 640 * (1.2609 * 0.2609 / 2) = 640 * 0.16448 = 105.3

key 3 (dob, postcode sector):
  records with dob = 620*0.86 + 380*0.71 = 533.2 + 269.8 = 803
  distinct keys = 505; mean 1.5901
  pairs ~= 505 * (1.5901 * 0.5901 / 2) = 505 * 0.46915 = 236.9

raw pairs 713.6 + 105.3 + 236.9 = 1,055.8 -> P = 1,055 pairs after deduplication
all pairs for N = 1,000      = 1,000 * 999 / 2 = 499,500
candidate reduction          = 1 - 1,055/499,500 = 1 - 0.0021121 = 0.99789
```

`1,055` is `0.211%` of all pairs. That is the entire economic justification for blocking.

Now the cost of it. An audit sample of `300` true duplicate pairs, drawn independently of the three keys,
finds `258` present in the candidate set.

```text
rho = 258 / 300 = 0.8600
true pairs expected in candidate set = 264 * 0.86 = 227.04 -> 227
pairs lost before scoring = 264 - 227 = 37  (14.0% of 264)
```

Of the `42` missed pairs in the 300-pair sample (a proportion that maps to `37` of `264`), the audit
records why:

| Reason | Count in sample | Share of misses |
|---|---|---|
| Mobile key split by a formatting case the normaliser missed | 11 | 26.19% |
| Postcode sector changed mid-tenancy (moved across a boundary) | 9 | 21.43% |
| Surname variant beyond edit distance 1 | 8 | 19.05% |
| Both records missing DOB and mobile | 7 | 16.67% |
| Nickname versus legal given name on the composite keys | 7 | 16.67% |

`11 + 9 = 20` of the `42` misses are fixable by normalisation or by one additional key. That is
`20/300 = 6.67%` of the sample, i.e. `0.0667 * 264 ≈ 17.6` pairs of the `264`, raising `rho` from `0.86` to
`(264*0.86 + 17.6)/264 = (227 + 17.6)/264 = 0.926`. This table is the most actionable output of the whole
exercise, and it is only producible because the audit sample was drawn independently of the blocker.

### 5.3 Scoring the candidate pairs

Scores are sums of the agreeing-field weights from §5.1. The constructed distribution over the `1,055`
candidate pairs:

| Agreement pattern | Pairs | `W` (bits) | Zone at floor 8.53 / threshold 11.87 |
|---|---|---|---|
| mobile + surname + dob + postcode | 58 | 22.1988 | auto |
| mobile + surname + dob | 41 | 17.9549 | auto |
| mobile + dob + postcode | 33 | 17.9093 | auto |
| dob + surname + postcode | 26 | 16.6597 | auto |
| mobile + surname + postcode | 19 | 14.0726 | auto |
| mobile + dob | 17 | 13.6654 | auto |
| dob + postcode | 22 | 12.3702 | auto |
| surname + dob | 9 | 12.4158 | auto |
| mobile + surname | 24 | 9.8287 | review |
| mobile + postcode | 27 | 9.7831 | review |
| surname + postcode | 31 | 8.5334 | review |
| dob only | 48 | 8.1262 | no |
| surname only | 96 | 4.2895 | no |
| postcode only | 120 | 4.2439 | no |
| none agree | 484 | −12.8974 | no |
| **total** | **1,055** | | |

Column check: `58+41+33+26+19+17+22+9 = 225`; `24+27+31 = 82`; `48+96+120+484 = 748`;
`225 + 82 + 748 = 1,055`.

Note the row ordering is by pattern strength, not by score: `surname + dob` at `12.4158` outranks
`dob + postcode` at `12.3702` even though the latter looks stronger, because surname's weight (`4.29`)
exceeds postcode's (`4.24`). Ordering such a table by human intuition about which fields "matter" gives
the wrong order roughly as often as the right one.

### 5.4 Threshold arithmetic, stated in full

**First, naively.** Reuse the wide-population prior from §4.2 (`T/P = 4,000/2,000,000`):

```text
lambda0 = -8.9629 ; log2(45/6) = 2.9069 ; W* = 11.8698 bits
```

**Then correctly.** The threshold must be computed on the population the score is applied to. This
candidate set holds `T = 227` true pairs inside `P = 1,055`:

```text
prior probability = 227 / 1,055 = 0.21517
prior odds        = 0.21517 / 0.78483 = 0.27416
lambda0           = log2(0.27416) = ln(0.27416)/0.693147 = (-1.29443)/0.693147 = -1.8675 bits

W* = 2.9069 - (-1.8675) = 4.7744 bits
```

The gap between the two thresholds is the point of the whole section: `11.8698 − 4.7744 = 7.0954` bits,
a factor of `2^7.0954 ≈ 137` in odds. A team calibrated on the wide population and applying its threshold
inside blocks rejects `130` of the `355` pairs a correct threshold would accept. A team calibrated inside
blocks and applying its threshold to a wide scan accepts nearly everything.

**Which threshold to actually deploy, and why.** Apply `4.77` to the §5.3 table:

```text
auto = 355  (every pattern at or above 8.1262)
review band (4.77 to 11.87) = 130
no-link = 570  (surname only, postcode only, none agree)
```

Two objections kill `4.77` as an operating point even though the arithmetic is right.

1. **It depends on a volatile quantity.** `lambda0` moves whenever the blocking keys move, because the
   keys determine the candidate set and therefore the prior. Add one key, the candidate set grows, the
   prior falls, `W*` rises, and the same cost model now implies a different policy. A threshold that
   changes when the blocker changes is an artifact, not a decision.
2. **It leaves no review band at the boundary that matters.** The band `4.77–11.87` contains pairs whose
   evidence is a single weak field or a lone DOB; those are precisely the pairs where the cost model is
   most likely to be wrong, and a cost model cannot review its own errors.

So the deployed policy is: auto-link at `11.87` bits (the wide-population cost-optimal value, therefore
robust to blocking changes), review floor at `8.53` bits. The band between them holds `82` pairs. That
band is a documented, budgeted inefficiency: it sends pairs worth `9.83` bits — where the wide-population
cost model says *link* — to humans, buying insurance against a `C_FM` that may be understated. The
`82`-pair band is what §5.7 prices.

### 5.5 Closure, and the pairs nobody scored

Acceptance is pairwise; entity formation is transitive. A constructed component trace:

```text
records r7, r19, r44, r61 accepted pairwise:
  (r7,  r19) W = 17.9549
  (r19, r44) W = 14.0726
  (r44, r61) W = 12.4158

closure -> one component of size 4, minimum internal evidenced edge 12.4158 bits (0.55 above threshold)
pairs asserted by equality of identity = C(4,2) = 6
pairs evidenced                        = 3
pairs implied and never scored         = 3: (r7,r44), (r7,r61), (r19,r61)
```

Score those three implied pairs anyway, as a diagnostic:

```text
(r7,  r44) =  9.8287   -> 2.04 bits below threshold; agrees mobile+surname, missing dob
(r7,  r61) = -1.1100   -> agrees surname only, contradicts mobile
(r19, r61) =  4.2895   -> agrees surname only
```

Two of three implied pairs sit far below the threshold. The distinction that decides the response is
*why*: missing fields produce low sums without contradiction, while disagreeing fields produce negative
contributions that are real evidence. Here `(r7, r61)` carries a disagreeing mobile — a contradiction —
so the component is a bridge candidate, not a sparse-record artifact.

Across the whole run, closure on the `225` accepted links produces `207` entity merges:

```text
redundant links (inside already-formed components) = 225 - 207 = 18
entities after closure = 1,000 - 207 = 793   (true entity count 780, so 13 splits remain)
```

`18/225 = 8.0%` of accepted pairs carry no new information once closure is applied. Those `18` are the
natural first candidates for reallocation: scoring implied pairs instead of re-scoring links already
inside a component buys strictly more information per comparison.

### 5.6 Consent gate, applied at entity grain

The gate must run on entities, not records, or a person with two records is counted twice in a
"reachable customers" figure.

```text
entities after closure                                    793
consent valid for the marketing purpose at send time       636   (80.2% of 793)
of those, email verified                                   483   (75.9% of 636)
of those, not suppressed                                   467   (96.7% of 483)
of those, outside the review band                          452   (96.8% of 467)

tau = 452 / 793 = 0.5700
waste factor = 793 / 452 = 1.754
```

Reading the funnel by stage rather than by total: consent removes `157` entities (`19.8%`), verification
removes `153` (`24.1%` of the consented), suppression removes `16`, and low link confidence removes `15`.
The first loss is the system working as intended. The second is a technical problem — `153` entities hold
a valid marketing consent and an address nobody has verified — and it is the largest addressable loss in
the funnel. A dashboard that prints "consented 636, mailable 483" without a bridging line invites the
reader to attribute the `153` to consent withdrawals, which would be wrong and would misdirect the fix.

### 5.7 What the run cost, and what review bought

```text
pairwise comparisons: 1,055 blocked vs 499,500 exhaustive
field comparisons:    1,055 * 4 = 4,220 vs 499,500 * 4 = 1,998,000
reduction:            1,998,000 / 4,220 = 473.5x

review volume 82 pairs; reviewer throughput 55 pairs/hour
reviewer hours = 82 / 55 = 1.4909 h
reviewer cost at 22 CU/h = 1.4909 * 22 = 32.80 CU

constructed reviewer behaviour: 71% of review-band pairs are rejected by the reviewer
false merges prevented = 82 * 0.71 = 58.22 pairs
value at C_FM = 45 CU = 58.22 * 45 = 2,619.9 CU
review ROI = 2,619.9 / 32.80 = 79.9x
```

The `71%` rejection rate is the number to monitor, not the ROI. A band whose rejection rate drifts toward
zero has moved away from the decision boundary — usually because the review floor was lowered to hit a
volume target — and is then buying no risk reduction while still costing reviewer hours. Conversely a
rejection rate near `100%` means the floor is set far too high and the band is pure waste.

### 5.8 Independent re-checks

**Check 1 — weight via exponentiation, not log identity.** For mobile, `m/u = 0.93/0.020 = 46.5`.
Independently, `2^5.5392`: split as `2^5 * 2^0.5392`; `2^5 = 32`; `2^0.5392 = e^(0.5392 × 0.693147) =
e^0.373825 = 1.45324`; product `46.5037`. Against the direct ratio `46.5`, the gap is `0.0037`, i.e.
`0.008%`, attributable to rounding the exponent. Confirmed.

**Check 2 — entity partition, by counts rather than by pairs.** `604` singletons, `132` doubles, `44`
triples: records `= 604 + 2*132 + 3*44 = 604 + 264 + 132 = 1,000` ✓; entities `= 604 + 132 + 44 = 780` ✓;
pairs `= 132*1 + 44*3 = 264` ✓. Three independent identities agree, which is why this partition is used
rather than a bare pair count.

**Check 3 — closure arithmetic from components.** With `207` merges over `1,000` records the entity count
is `1,000 − 207 = 793`. Check against accepted links: `225` links, of which `18` fall inside existing
components, so distinct merges `= 225 − 18 = 207` ✓. Check against the score table: auto-linked pairs
`= 225` ✓.

**Check 4 — achieved recall two ways.** Forward: `TP = 205` true pairs accepted, true pairs overall
`264`, so achieved recall `= 205/264 = 0.7765`. Backward: blocking recall `0.86` times scorer precision
`205/225 = 0.9111` gives `0.86 × 0.9111 = 0.7835`. The two differ by `0.0070` because `0.86` was rounded
from `258/300`; using `258/300 = 0.86` exactly and `P(true in block) = 0.86`, the backward figure is
`0.7835` while the forward figure counts the actual `227` blocked pairs. Reconciling: `205/227 = 0.9031`
is the scorer's recall *within* the candidate set, and `0.9031 × 0.86 = 0.7767` matches the forward
`0.7765` to rounding. Consistent.

**Check 5 — units.** `W`, `W*`, `lambda0` are all bits; `W + lambda0 > log2(C_FM/C_FN)` compares bits to
bits. `C_FM`, `C_FN` are CU per pair, so their ratio is dimensionless and its logarithm is bits. `tau` is
entities over entities. `rho` is pairs over pairs. No expression above divides a count by a rate.

### 5.9 Sensitivity

Perturb each constructed input by ±20% and read the effect on the two decisions that matter.

| Input | Base | −20% | +20% | Effect on auto-links | Effect on entities |
|---|---|---|---|---|---|
| `C_FM/C_FN` | 7.5 | 6.0 | 9.0 | `W*` moves −0.32/+0.26 bits; ±17 links | ∓11 entities |
| `T` (227) | 227 | 181.6 | 272.4 | `lambda0` moves +0.32/−0.26 bits; ±14 links | ∓9 entities |
| `u_dob` (0.0034) | 0.0034 | 0.00272 | 0.00408 | `w_dob` moves +0.32/−0.26; ±23 links | ∓15 entities |
| `rho` (0.86) | 0.86 | 0.688 | 1.00 | no direct effect — those pairs never exist | ∓31 entities |
| `b` (mean block size 2.43) | 2.43 | 1.94 | 2.91 | pairs scale ∓20%; runtime ∝ `b` | none |

The ordering is the actionable conclusion. **Threshold inputs move link counts in the teens; the recall
input moves entity counts by roughly double that, and moves them invisibly** — a pair the blocker never
formed appears in no dashboard, no exception report, and no audit that draws its sample from the
candidate set. A `20%` error in `u_dob` costs about `15` entities of `780` (`1.9%`); a `20%` error in
`rho` costs `31` (`4.0%`) and is undetectable without an independently drawn audit sample.

A second sensitivity is worth naming: this table assumes `m` and `u` are stable. If system B changes its
signup form to require DOB, `m_dob` rises while `u_dob` barely moves, `w_dob` grows, and the auto-link rate
jumps for reasons unrelated to the customer base. The discriminating check is a per-release recomputation
of weights on a fixed labelled sample; a weight that moves more than about `0.5` bits between releases
signals a capture change and demands recalibration, not a threshold tweak.
