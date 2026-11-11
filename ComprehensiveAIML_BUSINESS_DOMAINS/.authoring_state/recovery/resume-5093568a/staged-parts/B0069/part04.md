## 6. Visual / Geometric Intuition

### 6.1 The timeline is a cut, not an arrow

```text
feature events      request t        display        outcome         label mature
     |                  |               |              |                 |
     |<--- available --->|               |              |                 |
     |                  |<-- retrieval snapshot taken here ---------------|
     |                  |               |<-- H -->|                       |
     |                  |               |              |<-- maturity lag -->|
   forbidden:            |               |              |                 |
   anything to the right of t inside the feature vector                   |
```

The vertical line at t is a hard information boundary. The outcome window H opens at display and closes H later; the label-maturity lag is the extra time the outcome pipeline needs before the label is queryable. Two distinct violations are possible and they look identical in a feature-importance chart: a feature computed from an event after t, and a label queried before maturity, which silently treats immature outcomes as negatives.

### 6.2 The eligible set is a mask over the score space

```text
score axis (cosine similarity, higher = closer) --->

  ineligible region            eligible universe E
  ####################################################................
  #  i30 rights  #  i31 stock  #   i44   i51   i60   #   i03  i07  i11 ...
  ####################################################................
                                  ^
                       exact top-3 within E = {i44, i51, i60}
                       global top-3 = {i03, i07, i11}  <- all ineligible
```

Geometrically, an unconstrained index sorts the whole axis and a constrained index sorts only the shaded region. When the mass of vectors is concentrated in the ineligible region — a common outcome when a large licensed catalog is restricted to one territory — the global head is entirely unusable and post-filtering returns nothing. This is why the fix is a partitioned index or a predicate-aware structure, not a larger K.

### 6.3 The funnel drawn as a loss ledger

```text
catalog I(t)            20 items                                ─┐
   minus rights/stock/safety  −6                               │ 6 lost to policy (not a defect)
entitled E(t)           14 items                                 │
   minus never retrieved      −8 of 14 (i10 among them)         │ 8 lost at admission
unique retrieved        6 of 7 relevant                          │
   minus truncation at K      −1 (i6)                           │ 1 lost to budget
candidate C_K           5 of 7 relevant      = 0.7143            │
   minus rank and cap         −1 (i3)                           │ 1 lost to rank/cap
slate S_5               4 of 7 relevant      = 0.5714           ─┘
```

The ledger separates a **policy loss** (items that were never allowed) from a **quality loss** (items that were allowed and not shown). Conflating them produces the classic error of "improving recall" by weakening eligibility. The two lines must be reported separately even though both reduce the display count.

### 6.4 Score-space geometry of cosine versus inner product

Cosine similarity is inner product on the unit sphere. For unnormalized vectors, `cos(u,v) = <u,v> / (‖u‖‖v‖)`. Two items with identical direction but different norms have identical cosine and different inner product, so a system that trains with inner product and serves with cosine changes the ranking of popular (large-norm) items relative to niche ones. The practical consequence: an item whose embedding norm grew because it accumulated many interactions gains under inner product and loses under cosine. If training and serving disagree on normalization, the change looks like a relevance regression but is a metric mismatch.

A worked comparison makes the size visible. Let `u = (1.0, 0.0)`, `v_a = (2.0, 0.5)`, `v_b = (0.9, 0.1)`. Inner products are `2.0` and `0.9`, so `v_a` ranks first. Cosines are `2.0/2.0616 = 0.9701` and `0.9/0.9055 = 0.9939`, so `v_b` ranks first. A 0.0238 cosine difference reverses a 1.1-magnitude inner-product difference, which is why the two metrics must not be mixed between training and serving.

### 6.5 Position bias as a multiplicative funnel

```text
slot 1   exposure 1.000   observed CTR 0.300   implied rate 0.300
slot 2   exposure 0.620   observed CTR 0.155   implied rate 0.250
slot 3   exposure 0.450   observed CTR 0.126   implied rate 0.280
slot 4   exposure 0.330   observed CTR 0.106   implied rate 0.320
```

Read the third column: the observed rate falls monotonically with position, while the implied rate is not monotonic at all. The observed decline is entirely explained by exposure propensity. Any model trained on observed clicks inherits the propensity, which is why inverse-propensity weighting with weights `1/1.000 = 1.000`, `1/0.620 = 1.613`, `1/0.450 = 2.222`, `1/0.330 = 3.030` is applied when position variation is randomized or plausibly ignorable.

**Takeaway:** the four diagrams correspond to the four seams from Section 2.3 — time, universe, budget, and exposure — and each seam has one picture that makes its failure obvious.

## 7. Comparison / Selection Table

| Approach | Preconditions | Guarantee offered | Cost driver | Use when | Avoid when | Evidence |
|---|---|---|---|---|---|---|
| Global popularity | Point-in-time counts, per-segment normalization | None beyond stability | Precompute plus filter | Cold start, degraded-mode fallback | Manipulable counts without anomaly controls | HEURISTIC |
| Collaborative filtering | Dense enough interaction matrix, stable IDs | Learns co-occurrence association | Training plus ANN serving | Repeat-behaviour domains with history | Cold users/items dominate the traffic | HEURISTIC |
| Content-based | Authorized, maintained metadata | Can retrieve unseen items | Feature maintenance | New catalog, sparse interactions | Metadata sparse, wrong, or sensitive | HEURISTIC |
| Exact dense scan | Vector availability, fixed metric | Exact top-K for the stated score | O(N·d) per query | Oracle for evaluation, small catalogs | N·d exceeds the latency budget | DERIVED |
| IVF / HNSW ANN | Tunable index, measurable oracle | Approximate; fidelity must be measured per depth | Build, memory, query nprobe/ef | Exact scan over budget and fidelity verified | Eligibility cannot be pushed into the index | SOURCED / OPEN |
| Two-tower learned retrieval | Training pairs, negative sampling | Learns a task-specific metric | Training plus embedding serving | Scale exceeds heuristic capacity | Feedback loop cannot be controlled | OPEN |
| Rank fusion (RRF) | Per-source ranks, common filter | Agreement bonus, scale-free | Trivial | Heterogeneous sources | A single source must dominate precision | DERIVED |
| Learned re-ranker | Candidate list, features at request time | Better ordering inside C_K | One forward pass per candidate | Candidate recall already acceptable | Candidate recall is the bottleneck | OPEN |

### 7.1 Selection rules that survive review

The first rule is to fail closed: an item whose eligibility cannot be established at display time is not shown, and a retrieval gain never buys an authorization exception. The second is to pick a baseline whose behaviour is explainable during an incident, because the fallback path will be running when nobody can debug an embedding. The third is to compare methods only as complete pipelines — same request population, same eligibility snapshot, same K, same filter position — and to publish the per-depth fidelity and the underfill rate alongside any average.

### 7.2 What the table does not rank

Nothing in the table orders the methods by quality, because the ranking depends on the traffic mix. A catalog where 60 percent of requests come from new users is governed by the content and popularity rows; a catalog with deep repeat behaviour is governed by collaborative and two-tower rows. The correct selection procedure is to segment the traffic by history depth, run the candidate comparison separately in each segment, and choose per segment, not globally.

### 7.3 Filter placement: pre-, post-, and partitioned

| Placement | Mechanism | Strength | Failure | Check |
|---|---|---|---|---|
| Pre-filter by ID list | Restrict search to an explicit allow-list | Exact correctness | Allow-lists can be huge; overhead grows with list size | Compare latency versus list cardinality |
| Post-filter after search | Search global, then drop ineligible | Trivial to implement | Underfill; the head may be entirely ineligible | Measure survivors per request; watch the empty tail |
| Partitioned index | Separate index per entitlement class or region | Correct and fast when classes are coarse | Class explosion; cross-class requests need unions | Count partitions per request; watch fan-out |
| Predicate-aware ANN | Index structure supporting filters natively | Balances both | Configuration-specific behaviour, version-sensitive | Re-measure fidelity after every index upgrade |
| Final check at display | Re-verify immediately before render | Catches races and revocations | Adds latency at the last hop | Audit zero unauthorized renders |

The final check is not redundant with pre-filtering, because entitlement can change inside the request lifetime. If a revocation rate of 0.4 percent per hour is plausible for the catalog, a 300-second cache implies an expected stale fraction of `1 − (1 − 0.004)^(300/3600) = 0.000334`, or roughly 1 in 3,000 cached entries, and for a 900-second cache `1 − (1 − 0.004)^0.25 = 0.0010`, or 1 in 1,000. The final check converts those from user-visible defects into rejected candidates, at the cost of a small underfill that the over-retrieval budget must absorb.

**Benchmark status: NOT MEASURED.** To turn this table into evidence, fix the catalog snapshot, the request cohort, the score definition, the index version, the hardware, the concurrency, the K distribution, and the filter placement; then report latency percentiles, per-depth fidelity, underfill rate, and cost per thousand requests for each row.
