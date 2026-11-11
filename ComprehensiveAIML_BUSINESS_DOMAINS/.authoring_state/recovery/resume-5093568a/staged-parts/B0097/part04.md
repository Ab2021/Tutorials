## 6. Visual / Geometric Intuition

Identity resolution has three pictures worth carrying in your head. Each one diagnoses a different class
of problem, and none of them is a chart you would show to a business audience — they are working diagrams
for the person maintaining the system.

### 6.1 The score line, and why the numbers on it are not evenly spaced

Imagine the score axis for candidate pairs as a line of bits from very negative to very positive. Real
linkage does not place pairs smoothly along that line; it piles them into a few narrow clumps, because
scores are sums of a small number of field weights and most pairs agree on the same few fields.

```text
density of candidate pairs along the score axis (constructed, matching §5.3)

  count
   130 |                             ##
   120 | ##                          ##                        ##
   110 | ##                          ##                        ##
    96 | ##        ##                ##          ##            ##
    88 | ##        ##       ##       ##          ##            ##
    72 | ##        ##       ##       ##          ##   ##       ##
    63 | ##        ##       ##       ##     ##   ##   ##       ##
    55 | ##        ##       ##       ##     ##   ##   ##   ##  ##
    44 | ##        ##       ##   ##  ##     ##   ##   ##   ##  ##
       +-+---------+--------+---+---+------+----+----+----+---+---->  W (bits)
        -7.6      4.24    8.13 9.78 9.83 12.37 12.42 14.07 17.91 22.20

  threshold markers:      review floor 8.53 ^        auto threshold 11.87 ^
```

Reading the picture: the mass to the right of the review floor (`8.53`) is the actionable set; the mass
below `4.24` is noise. The clumps are why threshold tuning feels erratic — moving the threshold by `0.1`
bit usually changes nothing at all, then a `40%` jump in volume when it crosses a rung. A team that
tunes thresholds by small increments and observes flat responses concludes, wrongly, that the threshold
does not matter.

**The geometric consequence for capacity planning.** Review volume as a function of the review floor is
a staircase, not a line. Budgeting from a linear extrapolation ("at 9.5 bits we get about half of 158")
is wrong because there is no pair mass between `9.83` and `12.37`. The only safe capacity projection is
the empirical histogram, recomputed per run.

### 6.2 The confusion matrix as a rectangle, and the base-rate shear

Take the candidate set and draw it as a rectangle whose area is `P` pairs. The true-duplicate band is a
thin horizontal strip of height `T/P`; in §5 that is `302/1,055 = 0.2863`, unusually thick because it is
a blocked set. Split vertically at the decision threshold: accept on the right. Then:

```text
              |  rejected        |   accepted
  ------------+------------------+----------------+---------
   duplicate  |  T - TP          |   TP           |  T
  ------------+------------------+----------------+---------
   not dup    |  (P-T) - FP      |   FP           |  P - T
  ------------+------------------+----------------+
              |                  |  A             |
```

Precision is `TP / A` — the share of the accepted column that falls in the thin duplicate strip. The
visual intuition that matters: **precision is a ratio of areas where the numerator is bounded by the
thin strip, so precision is dominated by how much of the accepted column lies outside it.** In a
population where duplicates are `2%`, even a scorer with a `2%` false-positive rate on the non-duplicate
mass yields precision near `0.5`, because `2%` of `98%` is `1.96%` — nearly as large as the entire `2%`
duplicate mass. That is the base-rate shear, and it is why a "99% accurate" matcher can be 50% wrong.

With the §5 numbers: `A = 616` auto-links, and a constructed audit of the auto-linked set finds
`93.00%` true, so `TP = 616 × 0.93 = 572.88` and `FP = 616 − 572.88 = 43.12`. Precision is
`572.88/616 = 0.9300`.

Now the trap. Recall needs `T`, the number of true pairs that *existed in the candidate set*. In §5 the
audit estimated `T = 302`, which would give `recall = 572.88/302 = 1.897` — greater than one, which is
impossible. The construction is inconsistent, and the inconsistency is instructive: `T = 302` came from
the 300-pair audit sample, which was drawn from the candidate set, whereas `A = 616` is a count of
accepted links over the same set. Accepting 616 links cannot yield 573 true pairs inside a set holding
302 true pairs. Either the audit's `T` is an underestimate (`302` was a floor, as §5.1 warned) or the
auto-link set contains pairs that are not duplicates of one another but are both records of the same
person as a third record — the closure case of §5.5.

**Resolving it honestly.** Set `T = 700`. Then `recall = 572.88/700 = 0.8184`, `precision = 0.9300`, and
`F1 = 2 × 0.93 × 0.8184 / (0.93 + 0.8184) = 1.5222/1.7484 = 0.8706`. The asymmetry between precision
(`0.93`) and recall (`0.82`) is the normal state of affairs: the threshold is set so that accepted pairs
are trustworthy, which necessarily leaves true pairs below the threshold.

**The reporting rule this picture enforces.** Never accept a linkage report that gives precision and
recall without also giving `A`, `T`, and the population those apply to. When the two figures come out
suspiciously symmetric — both `0.93`, or both `92%` — look for a construction that has set `T` equal to
`A`, which defines away both the missed pairs and the false merges at once.

### 6.3 The cluster graph and the bridge

Entities are connected components of a graph whose nodes are records and whose edges are accepted links.
Draw the graph and the failure geometry becomes obvious in a way that no metric conveys.

```text
healthy component                 bridged component (one bad edge)
  r1 --- r2                        r1 --- r2        r4 --- r5
          |                                 \      /
          r3                                 r3 -- rB          <- rB is the bridge record
                                                    \      \
                                                     r6 --- r7

  min internal edge = strong        min internal edge = the bridge, often just above threshold
  4 records, 3 edges                7 records, 6 edges, spanning 2 or more real people
  implied pairs asserted: 3          implied pairs asserted: 15
```

The geometry to internalise: **a component of size `k` asserts `k(k−1)/2` pairs while containing only
`k−1` edges.** The ratio of asserted pairs to evidenced pairs grows linearly in `k`:

| Component size `k` | Edges evidenced (`k−1`) | Pairs asserted (`k(k−1)/2`) | Assertion amplification |
|---|---|---|---|
| 2 | 1 | 1 | 1.00× |
| 3 | 2 | 3 | 1.50× |
| 4 | 3 | 6 | 2.00× |
| 6 | 5 | 15 | 3.00× |
| 10 | 9 | 45 | 5.00× |
| 25 | 24 | 300 | 12.50× |

At `k = 25` the graph asserts `300` equalities on the evidence of `24` comparisons. If a single edge is
wrong, the other `299` asserted pairs are unverified inferences. This is the mechanism behind the
"cluster explosion" that turns a good-looking pair-level audit into a customer-service disaster: pairs
look fine, and entities are wrong.

**The check that the picture suggests.** Colour each component's edges by whether the pair was directly
scored or implied by closure. A component where most asserted pairs are implied and whose minimum
evidenced edge is within `1` bit of the threshold is a bridge, and it will always be a bridge — clusters
do not accidentally have exactly one weak edge unless that edge is the error.

### 6.4 The consent funnel, drawn to scale

The gate from §4.3(f) and §5.6 is a funnel, and its geometry matters because each stage removes a
different kind of entity and requires a different owner.

```text
records                    1,000   ############################
entities (after closure)     396    ############
consent valid                318    ##########      -78 lost: no marketing consent
email verified               241    #######         -77 lost: consent but unverified address
not suppressed               233    #######         -8 lost: complaints and bounces
outside review band          225    ######          -8 lost: link confidence too low to act

widths to scale: each '#' represents 32 entity-slots at a fixed 30-column scale
```

Two things this drawing makes visible. First, **the largest single loss is consent, and it is the one
loss that is legitimate** — it is the system working. Second, the `77` entities lost between consent and
verified email are a *technical* problem measured on a *consent* denominator, and they are the ones that
get misreported: a dashboard that shows "consented: 318" and "mailable: 241" without a bridge line
invites the reader to assume the `77` were consent withdrawals.

### 6.5 A time picture: the bitemporal rectangle

```text
                        effective time (when it was true in the world)
                        Jan        Feb        Mar        Apr
                   +-----------+-----------+-----------+-----------+
   processing   Jan |  A        |  A(est)   |     -     |     -     |
   time         Feb |  A        |  A        |  A(arr)   |     -     |
   (when we     Mar |  A        |  A        |  A        |     -     |
    learned it) Apr |  A        |  A        |  A        |  A(arr)   |
                   +-----------+-----------+-----------+-----------+
```

`A(est)` is a value we estimated and later corrected; `A(arr)` is a value that arrived late. A merge
decided in April uses information that did not exist in January, so "January customers, computed in
April" is a different number from "January customers, computed in January". The diagonal of this
rectangle is what "as-of" means; every off-diagonal cell is a number that exists in your warehouse and
that nobody should report without saying which cell it came from.

**Why this is a picture rather than a formula.** The formula is trivial (index by two timestamps). The
failure is not arithmetic; it is that reports silently default to one axis, and different reports default
to different axes, so two correct numbers disagree. The rectangle is the artifact that makes the choice
visible in a design review.

## 7. Comparison / Selection Table

This section is the decision surface: given a situation, which mechanism, which grain, which threshold
policy. Rows are alternatives, not stages — most systems use several at once.

### 7.1 Matching mechanism selection

| Mechanism | Cost per pair | Recall ceiling | Precision at ceiling | Needs labelled data | Choose when |
|---|---|---|---|---|---|
| Exact match on normalised key | ~0 | low (misses any field error) | very high | no | key is a system-assigned ID, or a verified unique identifier |
| Deterministic rule set (multi-field exact) | very low | moderate | high | no | fields are well-maintained and you need explainability to auditors |
| Weighted Fellegi–Sunter scoring | low | high | high | yes, for `m`, `u` | the standard case: several noisy fields, cost-asymmetric decisions |
| Learned classifier on field features | low at scoring, high at build | high | high, if calibrated | yes, labels plus monitoring | field interactions matter and you can maintain a training pipeline |
| Embedding / similarity on names and addresses | moderate | high for messy text | moderate | yes | text fields are free-form, multilingual, or frequently mistyped |
| Graph community detection over transactions | moderate | high | moderate | no labels needed | no reliable contact fields, but shared devices or payment instruments exist |
| Manual adjudication only | very high | depends on sampling | very high | no | volume is tiny or the domain requires human decision by regulation |

**Selection rule of thumb, stated as a decision.** If a system-assigned identifier exists and covers
more than `95%` of records, use deterministic matching and spend the engineering effort on coverage, not
on scoring. If no such identifier exists and fields are noisy, weighted scoring is the only mechanism
whose decisions you can explain line by line to an auditor, at a cost of `O(fields)` per pair. Learned
classifiers beat weighted scoring exactly when field interactions are strong — measured as more than
about `12%` precision gain on a held-out labelled set — and otherwise add a maintenance burden without
buying accuracy.

### 7.2 Grain selection

| Grain | Definition | Use for | Breaks when |
|---|---|---|---|
| Record | one source row | operational joins, audit trails | used as "customer" in a report |
| Person | one human being | retention, lifetime value, per-person consent | household purchases are attributed to whoever signed up |
| Household | shared address or payment instrument | direct mail, catalogues, delivery | person-level frequency caps are applied at household grain |
| Device / cookie | one browser or app instance | session analytics, attribution windows | shared devices, clearing, multi-device people |
| Account | one login | fraud, entitlements, support | one person holds several accounts by design |
| Wallet / payment instrument | one card or bank token | subscription recovery, chargeback linkage | family cards, corporate cards |

The decision that follows: **choose the coarsest grain that answers the business question, then state
it.** Loyalty programme economics are usually person-level; direct-mail suppression is household-level;
attribution is device-level and then probabilistically aggregated. A metric computed at one grain and
reported at another is the most common source of "the numbers are wrong" disputes that are not actually
wrong.

### 7.3 Threshold policy selection

| Policy | How the threshold is set | Strength | Failure it invites |
|---|---|---|---|
| Precision target | fix precision at `≥ 0.995` on a labelled set, read the threshold | auditable, stable story for compliance | may accept almost nothing when duplicates are sparse |
| Cost-weighted Bayes | `W* = log2(C_FM/C_FN) − lambda0` | principled, adapts to economics | sensitive to prior drift between populations |
| F1 maximisation | maximise `2PR/(P+R)` | single number, easy to explain | treats a false merge as equal to a missed merge, which is almost never true |
| Review-capacity | fix review volume at what staff can process | operationally sustainable | quality becomes a function of headcount, silently |
| Fixed quantile | accept the top `q%` of pairs by score | stable volume, easy planning | quality degrades as the population changes, invisibly |
| Vendor default | accept whatever the platform ships | no work | no one can state the cost model or the error profile |

**Recommended combination, and why.** Use cost-weighted Bayes for the auto-link threshold, review
capacity for the review floor, and a precision floor as a hard guardrail that overrides both. Three
thresholds, three owners, and each one can be stated in a sentence. The combination is what makes a
threshold defensible in a review: the cost model explains *why this number*, the capacity explains
*why not lower*, and the precision floor explains *why not higher*.

### 7.4 Consent and suppression state model

| State | Meaning | Blocks activation | Blocks analytics | Typical source |
|---|---|---|---|---|
| unknown | no record of a choice | yes for marketing | yes if identifiable | legacy import |
| opted-in, purpose-specific | affirmative, timestamped, purpose-scoped | no | no | preference centre |
| opted-out | affirmative refusal | yes | no, if aggregated | preference centre |
| withdrawn | previously opted in, later withdrew | yes, from withdrawal time | depends on lawful basis | preference centre, support |
| suppressed — complaint | objected after contact | yes, permanently for that channel | no | complaints log |
| suppressed — hard bounce | channel technically unreachable | yes | n/a | ESP feedback loop |
| suppressed — regulatory | legal or platform requirement | yes, until cleared | n/a | compliance register |
| pseudonymous only | no direct identifier retained | yes, cannot address | yes, at aggregate grain | tokenisation pipeline |

**The rule that keeps this table honest.** Never store consent as a boolean. Store
`(purpose, channel, state, effective_from, effective_to, source)` as an append-only ledger, and derive
the current state at read time. The reason is not elegance: a boolean cannot answer "was this person
permitted at 14:03 on the day we sent", which is the only question that matters in a complaint, and a
boolean cannot represent withdrawal, because `false` conflates "never asked" with "said no" and "was
never asked" is not a lawful basis for anything.

### 7.5 Where each control belongs in the pipeline

| Control | Must be enforced | Cannot be enforced later because |
|---|---|---|
| Normalisation rules, versioned | at ingest | the raw value is the only place the original form survives |
| Blocking keys | at candidate generation | a pair that was never formed cannot be scored or audited |
| Cost model and threshold | at decision, recorded per link | a link's provenance is unrecoverable once merged into a profile |
| Closure policy (with or without bridge detection) | at entity formation | components are computed, not stored per link |
| Consent gate | at activation, every time | consent changes between build and send |
| Suppression list | at activation, every time | suppression can arrive from a channel partner minutes before send |
| Grain declaration | at metric definition | a report cannot retroactively relabel its unit |
| Clock declaration (event vs processing) | at metric definition | mixing clocks produces trends that look like behaviour |
