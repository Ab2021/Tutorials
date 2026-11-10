# Graph ML for Fraud and AML Ring Detection

## 1. File Header

> **Purpose:** Build, audit and defend a time-correct graph-assisted investigation queue — deciding which accounts, payments or case bundles an analyst should examine next, and proving the score could have been computed when it was issued.
> **Prerequisites:** Adjacency-list thinking, directed versus undirected connectivity, classification precision/recall, as-of joins over a transaction ledger, and the review-capacity argument in [Fraud Evaluation, Calibration and Capacity](../../../11_data_science_domain_specialisations/02_data_science_method_playbooks/02_fraud_modelling_playbook/Cheat_Sheets/01_Fraud_Evaluation_Calibration_and_Capacity.md).
> **Key outcome:** Given an event ledger with event and availability timestamps plus a fixed review budget, you can construct an eligible graph snapshot, compute a neighbor aggregate and a normalized propagation score, rank deduplicated cases, report precision and recall at capacity with an honest denominator, and name the leakage check that would falsify the result.
> **This file does NOT cover:** Legal reporting obligations, a complete anti-money-laundering programme, Cypher or vendor-specific graph query syntax, or the separate production case study in the case library.
> **Level:** 3 Advanced / production.
> **Version / reviewed:** Draft 0.2; review OPEN, not independently reviewed.
> **Evidence:** Formula identities are DERIVED and shown with arithmetic; every scenario, ledger and outcome is ILLUSTRATIVE; no deployment measurement, throughput figure or loss saving is claimed. Unresolved items are marked OPEN.
> **Retrieval cue:** If a graph model looks far better offline than online, start at Section 4.4 (limits) and Section 8 entries F-03, F-04, F-17 and F-25.

Graph structure earns its place in a fraud workflow only when a relationship carries evidence that the account's own history does not. That is a claim about incremental value at a fixed review budget, not a claim that graphs are modern or that connectivity is suspicious. Two accounts sharing a device may be a household, a public kiosk, a corporate-issued handset pool, or a recycled identifier; two accounts exchanging funds may be a supplier and its customer. The graph tells you the relationship was observed. Everything else — intent, common control, criminal association — is an inference that needs separate evidence.

The decision this sheet supports is narrow and operational: at 18:00 today, which `k` deduplicated cases should enter tomorrow's review queue, given the relationships that had actually reached the serving path before 18:00? `k` is set by investigator labour, not by the model. The sheet therefore treats graph learning as a ranking problem under a hard capacity constraint and an information-availability constraint, and treats the graph as one feature source among several.

Two constraints govern every calculation that follows. First, the **point-in-time constraint**: a record may enter a feature only if its availability timestamp precedes the decision timestamp, regardless of its event timestamp. Second, the **capacity constraint**: the queue's size is fixed by human throughput, so improvements must show up as better use of a fixed number of reviews, not as a larger number of alerts.

A third constraint is often forgotten and matters just as much: a score is a prioritisation signal. It is not proof, not a legal finding, and not by itself a permissible reason to deny service, close an account or file a report. The investigator owns the review decision; the policy owner defines which actions are permitted and how appeals work. Any sentence in this sheet that sounds like an automated verdict should be read as shorthand for "a human considers this case earlier than the others".

**Takeaway:** A defensible graph fraud system is a decision queue with a provenance trail, not a guilt-by-association engine.

## 2. Concept Map / Mental Model

```text
   source ledger          identity evidence        adjudicated outcomes
 (payments, logins,      (account-to-person,       (confirmed fraud,
  device sightings)       device ownership)         closed-no-finding)
        |                        |                        |
        +-----------+------------+                        |
                    |                                     |
                    v                                     v
        [ typed, versioned graph snapshot ]      [ label store, with
         nodes: account/person/device/merchant   availability timestamps ]
         edges: transfer / owns / used / shares            |
                    |                                      |
                    |  (as-of decision time t ONLY)         |
                    v                                      |
        [ eligible feature computation ] <-----------------+
         degree, mean, weighted mean, PPR, motif counts
                    |
                    v
        [ scorer: rules | tabular | graph | hybrid ]
                    |
                    v
        [ case deduplication at the investigation grain ]
                    |
                    v
        [ rank + capacity policy ] --> queue of k cases
                    |
                    v
        [ human review + evidence pack + appeal ]
                    |
                    v
        [ delayed, selectively observed outcomes ] ----+
                    |                                   |
                    +--> next training window -----------+
                         (never backward into an older snapshot)
```

The map has four distinct arrow kinds and they should not be blended. The vertical arrows from sources into the snapshot are **dataflow**: bytes moving, subject to ingestion latency. The dashed-in arrow from the label store is also dataflow, but it carries the most dangerous payload in the system, because a label is downstream of the very behaviour being predicted. The arrow from scorer to queue is a **policy transformation**: many scores are reduced to `k` cases by a rule that includes deduplication and tie-breaking. The return arrow from outcomes to the next training window is a **control loop**, and it is the only legitimate direction for learning; the loop must never run backwards into an already-issued decision.

The central invariant is monotone in time: for any decision issued at `t`, every input to that decision must have been available at `t` or earlier. Call this the *eligibility invariant*. It is necessary but not sufficient, because a permitted record can itself be a derived artifact of a disallowed one — a device-ownership flag set by an investigator after a confirmed outcome is available early but is contaminated by provenance. Timestamp filtering catches the first class of leak; lineage review catches the second.

What changes across variants is only the *message function* and the *neighborhood definition*. A hand-built aggregate fixes the message as "mean of a neighbor feature over one hop". A learned message-passing network replaces the mean with a parameterised function and repeats it over layers. A propagation score replaces the hard neighborhood with a decayed probability mass over walks. A motif rule replaces the average with a count of local subgraph shapes. The snapshot, the eligibility rule, and the capacity policy are identical in all four cases, which is why the same failure catalogue in Section 8 applies to each.

The most important boundary is between *observable structure* and *adjudicated association*. Structure is a property of the snapshot: this device was seen with these accounts, this payment moved along this directed path. Association is a claim about the world: these people act together. The model operates entirely on structure. The human supplies association, with evidence, and the review record must keep the two separable so that a later audit can ask "did the analyst's association actually follow from the structure, or did it come from outside the graph?"

```text
State of one case through the pipeline (ILLUSTRATIVE):

  candidate ---> scored ---> deduplicated ---> selected ---> reviewed
     |              |             |               |            |
     |              |             |               |            +--> confirmed / not confirmed / unknown
     |              |             |               +--> not selected: outcome UNOBSERVED
     |              |             +--> merged into another case: score retained, counted once
     |              +--> snapshot id, model id, feature version recorded
     +--> eligibility determined by decision time, not by arrival in the warehouse
```

The state diagram makes the evaluation problem visible. Only the rightmost branch produces a label, and it is reached only for cases the queue selected. Everything to the left of "selected" is a region where the outcome exists in the world but not in the data. That asymmetry — not model capacity — is the binding constraint on measuring recall, and it is why Section 9 budgets audit sampling as a first-class cost rather than an afterthought.

**Takeaway:** Fix the snapshot, the eligibility rule and the capacity policy first; then any of the four scoring families can be compared on equal terms.

## 3. Foundational Definitions

### 3.1 The entity layer

**Node.** A node is a typed entity identified within an explicitly declared namespace. An account node and a person node are different types even when they share a numeric-looking identifier, and joining them because both render as eight digits is a defect, not a feature. The namespace matters because vendor identifiers, internal customer numbers and card tokens all look like integers.

**Account.** A financial relationship with the institution. One person may hold several accounts; one account may have several authorised users; a business account may have dozens of signatories. Treating the account as a proxy for the person is a modelling decision with an error rate, and the sheet will always say which of the two the target refers to.

**Device.** A shared technical resource — a handset, a browser profile, a hardware token. A device is not a legal person and possession is not ownership. Device identifiers get recycled when a handset is resold, and they collapse to a constant sentinel when instrumentation fails, which is why a single device token can acquire tens of thousands of accounts without any of them being related.

**Edge.** A typed, directed-or-undirected, time-stamped relationship between two identifiers, carrying provenance. `transfer(A -> B, amount, currency, event_time, availability_time, source)` and `used_device(A, d, first_seen, last_seen)` are different relations with different semantics, even though both are "links". Collapsing them into a single untyped adjacency loses the distinction between money moving and hardware being shared.

**Multigraph.** Repeated events between the same endpoints are distinct multiedges. Collapsing them loses amounts, counts and inter-arrival times; retaining them changes what "degree" means. A feature contract must state whether the neighbor set is a set of distinct nodes or a multiset of event occurrences, because the two produce different aggregates from the same data.

**Entity resolution.** The reversible, evidence-based mapping from source records to candidate entities. Reversible means original identifiers and the merge decision are retained so a bad merge can be undone. Evidence-based means the merge used declared attributes with measured accuracy on an adjudicated pair sample — not a fuzzy match threshold chosen because it produced a convenient number of merges.

**Component.** A maximal connected set under a stated connectivity definition on a stated snapshot. Weak connectivity ignores edge direction; strong connectivity requires a directed path in both directions for every ordered pair. A three-node chain A→B→C is one weakly connected component and three strongly connected components. A component is a mathematical object, not an organisation.

**Motif.** A small induced or non-induced subgraph pattern over typed nodes — a directed 3-cycle of transfers, a star of five accounts on one device. Motif counts become evidence only against a null model that preserves degree, because a star is unsurprising on a hub.

### 3.2 The time layer

**Event time.** When the source asserts the event happened. Recorded in the source system's own clock, which may be skewed by seconds or, in badly configured exporters, by hours.

**Availability time.** When the serving path could first read the record. For a payment this might be the authorisation commit; for a device sighting it might be the batch landing in the stream. Availability is strictly downstream of event time and the gap is the ingestion latency, which is heavy-tailed: most records land in seconds, a tail lands in hours.

**Decision time.** When the score must be returned. For a nightly queue this is a batch boundary; for an authorisation decision it is inside the payment round trip and measured in milliseconds. The decision time fixes the cutoff for every input.

**Outcome time and label availability time.** Outcome time is when the target event occurred in the world (the chargeback was filed). Label availability time is when that fact became readable by a learner or reviewer (the case was adjudicated and written back). The second is usually much later than the first, and it is the one that governs training eligibility.

**Point-in-time snapshot.** The subgraph containing exactly those records whose availability time is at or before the decision time, restricted to permitted relation types, tenants and purposes. `OPEN`: the completeness contract for a snapshot — which partitions must report a watermark before the snapshot may be promoted — is a design decision this sheet does not specify for any particular store.

**Late arrival.** A record with an early event time and a late availability time. It is invisible to the earlier decision and must stay invisible in any replay of that decision. Backdating its availability to match its event time is the single most common way a "reproducible" historical study becomes unreproducible.

### 3.3 The learning layer

**Message.** A vector computed for a neighbour and passed toward a seed. In a hand-built feature the message is the neighbour's stored attribute; in a learned layer it is a transformation of the neighbour's representation.

**Aggregation.** A permutation-invariant reduction of a message multiset — mean, sum, max, or a learned pooling. Permutation invariance is required because adjacency has no canonical order; a function whose output depends on neighbour ordering is not a graph function.

**Fanout.** The maximum number of neighbours retained per node per layer per relation under a named sampling rule. Fanout is a serving contract, not a modelling preference, because it determines latency at the tail of the degree distribution.

**Inductive versus transductive.** An inductive model computes a representation for a node it has never seen, using its features and its neighbours at inference time. A transductive model used the test nodes' structure during training and can only be evaluated under a declared protocol that says so. Fraud populations churn, so inductive behaviour is usually the requirement, and any transductive convenience must be priced in.

**Leakage.** Any use of information that the deployed system could not have had at decision time. Four distinct families recur: temporal (a later record), target (the outcome itself), provenance (a derived artifact of the outcome), and selection (a label that exists only because the model chose that case).

**Selective labels.** A labelling process in which whether you observe `Y` depends on the score. Reviewer-supplied labels are the textbook case: high scores get examined, so positives are visible; low scores are never examined, so their positives are invisible. Selective labels do not disappear with more data — they get worse, because the model keeps choosing what to label.

### 3.4 The decision layer

**Candidate unit versus case unit.** The scored object (an account, a payment) and the investigated object (a case, which may bundle twenty accounts) are different grains. Metrics computed at one grain and reported at the other are wrong by an unknown factor.

**Capacity.** The number of distinct cases the team can complete in the interval, derived from labour minutes and per-case effort — including evidence gathering, documentation and appeal handling, not just the initial review.

**Precision at capacity.** Confirmed positives among the selected `k`, divided by `k`. **Recall at capacity.** Confirmed positives among the selected `k`, divided by all positives in the defined population. The second denominator is the one that requires an outside source or a designed audit.

| Term | Formal meaning | Units / type | Valid domain | Common confusion | Glossary anchor |
|---|---|---|---|---|---|
| Node | Typed entity in a declared namespace | identifier | non-empty namespace | merged with "person" | this section, 3.1 |
| Edge | Typed, time-stamped, provenance-bearing relation | record | availability time ≤ cutoff | treated as proof of association | 3.1 |
| Degree `d_v` | Count of incident eligible edge occurrences | edges | ≥ 0 | read as unique neighbours | 4.1 |
| Snapshot | Records available at `t` under policy | graph | watermark satisfied | rebuilt from end-of-period data | 3.2 |
| Fanout `f` | Retained neighbours per layer | neighbours | ≥ 1 | treated as complete degree | 9a |
| Selective labels | `Y` observed with probability depending on score | mechanism | review is score-driven | treated as missing at random | 3.3 |
| Case | Investigated bundle | case | dedup rule fixed | equated with account | 3.4 |
| Capacity `k` | Completable cases per interval | cases / interval | ≤ labour throughput | set by model alert volume | 3.4 |

**Takeaway:** Name the entity, the relation, the clock, the grain and the label mechanism before touching a model.

## 4. Core Formulas with Full Derivation

### 4.1 Simplest / Base Case: the eligible-neighbour mean

Start from a definition rather than a formula. For seed `v` at decision time `t`, define the eligible neighbourhood as the set of distinct neighbours reachable by at least one record that passed both the event-time and availability-time filters:

```text
FORMULA [G1a] (neighbourhood definition):
  N_v(t) = { u : exists an eligible record r with endpoints (v,u), r.relation in R_allowed,
                 r.event_time <= t, r.availability_time <= t, r.tenant = tenant(v) }
WHERE: v is the seed identifier; t is decision time [timestamp]; R_allowed is the
       declared relation whitelist; N_v(t) is a set of node identifiers.
CONSTRAINT: the existence test quantifies over the ledger restricted to r.availability_time <= t;
            a record that fails the filter is absent, not present-with-a-null-flag.
NUMERICAL EXAMPLE: with the Section 5 ledger at 18:00, N_B(18:00) = {A, C, D}.
INTERPRETATION: membership is a statement about evidence availability, not about closeness.
```

Given a scalar attribute `x_u` in a stated unit, the equal-weight mean over that set is

```text
FORMULA [G1b] (mean aggregate):
  m_v(t) = ( 1 / |N_v(t)| ) * sum over u in N_v(t) of x_u
WHERE: x_u has unit U (e.g. confirmed adverse events per account);
       |N_v(t)| is a count (dimensionless);
       m_v therefore has unit U — it is an average, never a count and never a probability.
CONSTRAINT: |N_v(t)| >= 1 and every x_u is finite and computed under the same unit definition.
NUMERICAL EXAMPLE: x_A = 1, x_C = 0, x_D = 5 with |N_B| = 3, so m_B = (1+0+5)/3 = 2 U.
INTERPRETATION: the seed's three available neighbours average 2 adverse events each.
```

Derivation. Write the mean as a normalised sum of indicator-weighted contributions:

```text
  m_v = sum_u x_u * 1{u in N_v} / sum_u 1{u in N_v}
```

The numerator is the total attribute mass over eligible neighbours; the denominator counts them. Dividing by a *count* makes the result scale-free in neighbourhood size, which is the property that distinguishes a mean from a sum. The unit check is immediate: `[U] * [count] / [count] = [U]`.

The reason this specific aggregate is the right base case is that it is the maximum-entropy reduction of a multiset given only its total and its size. It also has a bounded range — for non-negative `x`, `m_v` lies in `[min x_u, max x_u]` — which makes an out-of-range value an immediate defect signal rather than a modelling curiosity.

Two consequences follow immediately and both are operational. First, dividing by `|N_v|` implements *degree dilution*: a strongly suspicious neighbour among ninety neutral ones contributes one ninetieth of its signal. That is desirable when the neighbourhood is a population and undesirable when it is a list of incidents. Second, the empty neighbourhood makes the expression undefined, not zero. `0/0` is not a measurement of calm; it is an absence of evidence, and the model needs a separate missing indicator so the two cases can be distinguished downstream.

```text
FORMULA [G1c] (seed representation, linear mix):
  h_v = a * x_v + b * m_v
WHERE: a, b are dimensionless, non-negative, declared coefficients;
       x_v and m_v share unit U, so h_v has unit U.
CONSTRAINT: m_v defined; a + b = 1 if the mix is to be interpreted as a convex combination.
NUMERICAL EXAMPLE: x_B = 2, m_B = 2, a = b = 0.5, so h_B = 0.5*2 + 0.5*2 = 2 U.
INTERPRETATION: with equal weights the seed's own history and its neighbourhood agree exactly.
```

`G1c` is a teaching rule, not a fitted fraud probability. Its purpose is to force an explicit statement of how much of the representation comes from the node itself — a quantity that matters disproportionately for new accounts, where `N_v` is empty and `h_v` collapses to `a * x_v`.

### 4.2 General Case: symmetric normalisation, relations and attention

The mean has one arbitrary feature: it weights every neighbour equally. The general layer replaces the count `|N_v|` with a normaliser derived from the graph, and replaces the scalar attribute with a transformed vector.

Derivation of the symmetric normaliser. Suppose we want a propagation operator that (i) reduces to `m_v` for a regular graph, (ii) is symmetric as an operator so its spectrum is real, and (iii) has eigenvalues bounded by one so repeated application cannot explode. Consider the scaled adjacency `S = D^{-1/2} A D^{-1/2}` with `D = diag(d)`. For a regular graph of degree `d`, `S = A/d`, whose row sums are exactly one — so `S x` computes the neighbour mean for every node simultaneously. For an irregular graph, `(S x)_v = sum_u A_vu x_u / sqrt(d_v d_u)`, which is not exactly the mean but is the symmetric choice that satisfies all three requirements. The trade-off is explicit: symmetry buys a real spectrum and stability, at the cost of no longer being exactly an average.

```text
FORMULA [G2a] (symmetric-normalised propagation, one layer):
  (S x)_v = sum over u in N_v of x_u / sqrt(d_v * d_u)
WHERE: d_v, d_u are eligible degrees (counts); x_u is a scalar in unit U;
       the result has unit U.
CONSTRAINT: d_v >= 1 and d_u >= 1 for every included u; isolated nodes are handled separately.
NUMERICAL EXAMPLE: seed B with d_B = 3, neighbours A(d=2,x=1), C(d=3,x=0), D(d=2,x=5):
  1/sqrt(6) + 0 + 5/sqrt(6) = 6/2.449489743 = 2.449489743 U.
INTERPRETATION: the low-degree neighbour D is boosted relative to the mean-of-3 value of 2 U,
because a neighbour with few alternatives is more informative about this seed.
```

The generalisation to typed relations and vector representations follows the same structure with a relation-specific parameter matrix per edge type:

```text
FORMULA [G2b] (relation-aware message passing, R-GCN form):
  h_v^(l+1) = sigma( W_self^(l) h_v^(l) + sum over r in R of sum over u in N_v,r of
                      (1 / c_v,r) * W_r^(l) h_u^(l) )
WHERE: h^(l) in R^d (dimensionless representation); W in R^(d x d) parameters per relation;
       c_v,r is a declared normaliser (|N_v,r|, or sqrt(d_v,r d_u,r));
       sigma is a pointwise nonlinearity.
CONSTRAINT: parameters are fit only on data available before the training cutoff;
            the normaliser must be computed from the eligible snapshot, never the full graph.
NUMERICAL EXAMPLE: d = 2, c_v,r = |N_v,r| = 2, h_u1 = (1, 0), h_u2 = (0, 1),
       W_r = [[1, 2], [0, 1]]; W_r h_u1 = (1, 0), W_r h_u2 = (2, 1);
       sum = (3, 1); divided by 2 gives (1.5, 0.5) as the relation message.
INTERPRETATION: separate matrices let "shares a device" and "sent money" learn different weights.
```

Attention replaces the fixed normaliser with data-dependent, non-negative weights:

```text
FORMULA [G2c] (attention weights):
  alpha_vu = softmax over u' in N_v of ( LeakyReLU( a^T [ W h_v || W h_u' ] ) )
  h_v' = sigma( sum over u in N_v of alpha_vu * W h_u )
WHERE: a in R^(2d) is a parameter vector; || denotes concatenation;
       alpha_vu is dimensionless and sum over u of alpha_vu = 1.
CONSTRAINT: softmax is over the *eligible* neighbourhood only; adding a neighbour after
            scoring changes the denominator and therefore every other weight.
NUMERICAL EXAMPLE: raw scores (2.0, 0.0) over two neighbours give e^2/(e^2+e^0) = 7.389056/8.389056
            = 0.880797 for the first and 0.119203 for the second.
INTERPRETATION: attention redistributes weight within a fixed neighbourhood; it cannot invent a neighbour.
```

The derivation of the softmax step is the standard exponential-family argument: it is the unique distribution over the neighbourhood that is differentiable, sums to one, and increases monotonically in each raw score. That is why every attention implementation uses it, and also why a large `alpha_vu` describes the model's internal allocation of weight rather than a causal contribution — Section 8 entry F-40 exists because that distinction is routinely lost in an investigator-facing UI.

### 4.3 Key Special Cases / Variants

| Variant | What changes | Assumptions | Use when | Avoid when | Equation |
|---|---|---|---|---|---|
| Distinct-neighbour mean | One vote per distinct node | Identity links are reliable | Repeated events are noise | Event frequency is the signal | G1b |
| Event-occurrence mean | One vote per eligible event | Deduplicated by source event id | Retry traffic is meaningful | Duplicates are not removed | G1b on multiset |
| Degree-normalised propagation | Weights `1/sqrt(d_v d_u)` | Degrees from eligible snapshot | Degree varies widely across the population | Degree carries the fraud signal itself | G2a |
| Relation-aware passing | One matrix per relation type | Relation provenance trustworthy | Device and money edges mean different things | Relation types are miscoded | G2b |
| Attention | Learned weights within the neighbourhood | Enough labelled edges to fit `a` | Neighbour relevance genuinely varies | Labelled data is sparse relative to parameters | G2c |
| Temporal decay weighting | `w_uv = exp(-(t - tau_uv)/tau_half)` | Decay constant chosen a priori | Recency plausibly increases relevance | `tau_uv` is itself a label-derived quantity | G5 |
| Max pooling | `max_u x_u` instead of mean | At least one neighbour | A single strong lead should surface | Noisy extremes dominate | below |
| Propagation score (PPR) | Decayed mass over walks | Transition matrix row-normalised | Reach beyond one hop matters at fixed depth | Cycles create feedback the analyst will misread | G6 |
| Motif count / z-score | Local subgraph census | Degree-preserving null model | Ring-shaped structure is the target | Hub-dominated graph generates spurious stars | G11 |
| Component rule | Full connectivity partition | Connectivity definition stated | Triage of small closed groups | One shared processor joins most of the population | 3.1 |

Max pooling deserves its own note because it is the cheapest fix for degree dilution and the cheapest way to import a new failure. With `x_u` non-negative, `max` is invariant to how many neutral neighbours were added, so a single high-value neighbour survives arbitrarily large neighbourhoods. The cost is that a single erroneous attribute — one mis-keyed dispute — propagates at full strength with nothing to average it down. Mean and max therefore fail in opposite directions, and a robust hybrid (`mean` plus a capped `max`, or a quantile) is often the honest compromise.

Temporal decay formalises "recent relationships matter more":

```text
FORMULA [G5] (decayed weighted mean):
  m_v = sum over u of w_uv x_u / sum over u of w_uv ,  w_uv = exp( -(t - tau_uv) / tau )
WHERE: t is decision time [timestamp]; tau_uv is the edge's reference time [timestamp];
       tau > 0 is a decay constant [same unit as the time difference]; w_uv is dimensionless.
CONSTRAINT: t - tau_uv >= 0 for every included edge, and the sum of weights is > 0.
NUMERICAL EXAMPLE: tau = 30 days; ages 0 and 60 days give w = 1 and e^{-2} = 0.135335;
       with x = 4 and x = 1: (1*4 + 0.135335*1)/(1 + 0.135335) = 4.135335/1.135335 = 3.642 U.
INTERPRETATION: the stale neighbour retains about 13.5% of a fresh neighbour's influence.
```

The constraint `t - tau_uv >= 0` is not pedantry: a negative age produces a weight above one and silently up-weights a record that has not happened yet. Section 8 entry F-56 covers the sign bug that arises when the difference is written the other way round.

Propagation beyond one hop needs a definition that does not explode, and personalised PageRank is the standard one:

```text
FORMULA [G6] (personalised PageRank):
  p = alpha * s + (1 - alpha) * W^T p ,  solved as p = alpha * (I - (1-alpha) W^T)^{-1} s
WHERE: s is the seed indicator (unit mass on v, zero elsewhere); W is the row-normalised
       eligible adjacency (W_ij = A_ij / sum_k A_ik); alpha in (0,1] is the restart
       probability; p is a probability distribution over nodes.
CONSTRAINT: every row of W with a non-zero out-degree sums to 1; dangling nodes need an
            explicit rule (self-loop or uniform restart), otherwise mass leaks.
NUMERICAL EXAMPLE: alpha = 0.15 restart; the Neumann expansion
            p = alpha * sum_{j>=0} ((1-alpha) W^T)^j s
            is the sum over walk lengths of 0.15 * 0.85^j times the mass at length j,
            so a path of length 2 carries 0.15 * 0.85^2 = 0.108375 of the restart budget
            before degree normalisation.
INTERPRETATION: influence decays geometrically with hop count at a fixed, declared rate.
```

The derivation is a Neumann series: since `W^T` is row-stochastic, its spectral radius is one, so `(1-alpha) W^T` has spectral radius `1-alpha < 1` and `I - (1-alpha)W^T` is invertible with the geometric-series inverse. That is the formal reason the truncation at hop `j` loses a bounded amount of mass, `(1-alpha)^(j+1)`, and therefore why a depth cap is a principled approximation rather than a hack.

The evaluation formulas are short but carry the sheet's weight:

```text
FORMULA [G7] (precision and recall at capacity):
  precision@k = |P intersect Top_k| / k
  recall@k    = |P intersect Top_k| / |P|
WHERE: Top_k is the set of k deduplicated cases selected; P is the set of positives in the
       defined evaluation population; k is the review capacity [cases/interval].
CONSTRAINT: |Top_k| = k exactly (ties resolved by a declared rule); P must be enumerated
            over the whole population, not only over reviewed cases.
NUMERICAL EXAMPLE: k = 3, Top_3 contains 2 positives, |P| = 3:
            precision@3 = 2/3 = 0.666667; recall@3 = 2/3 = 0.666667.
INTERPRETATION: the two metrics coincide here only because k equals |P|; in general they do not.
```

```text
FORMULA [G8] (lift over base rate):
  lift@k = precision@k / (|P| / N)
WHERE: N is the size of the evaluation population [cases].
CONSTRAINT: N and P defined on the same population and window as Top_k.
NUMERICAL EXAMPLE: N = 6, |P| = 3, so base rate = 0.5 and lift@3 = 0.666667/0.5 = 1.333.
INTERPRETATION: the queue is 1.33 times as dense in positives as a random draw of three cases.
```

### 4.4 Limiting Behaviour and Counterexample

**Limit 1 — empty neighbourhood.** As `|N_v| -> 0`, `G1b` is undefined. Any implementation that returns zero has decided, silently, that absent evidence equals benign evidence. For a new account with no eligible neighbours the correct output is a representation plus a `no_neighbourhood` flag, and the model must have been evaluated on that slice; Section 8 entry F-45 covers the cold-start consequence.

**Limit 2 — degree dilution.** As `d_v -> infinity` with one suspicious neighbour, `m_v -> 0` in the ordinary mean. The seed's *risk* is unchanged; only the estimator saturates. This is a counterexample to the claim that a low mean score means a benign neighbourhood: it may mean a large one. The discriminating check is to compare the mean against the max and against the degree itself.

**Limit 3 — over-smoothing.** For the linear propagation operator `S = D^{-1/2} A D^{-1/2}` with eigenvalues `1 = lambda_1 >= lambda_2 >= ... >= lambda_n >= -1`, the `L`-fold application multiplies the component along eigenvector `i` by `lambda_i^L`. For a connected non-bipartite graph with `|lambda_i| < 1` for `i >= 2`, every non-constant component decays geometrically and `S^L x` converges to a vector proportional to the degree-weighted mean. Adding layers therefore increases the receptive field and decreases the distinguishability of nodes simultaneously. The practical implication is that depth in a fraud model is bounded by information, not by compute, and that a two-layer model on a well-built snapshot often beats a five-layer model that has averaged the population together.

**Limit 4 — hub domination.** A device with `d = 40,000` contributes a message to every one of its neighbours. Under a sum aggregator with no normalisation, its total contribution scales with `d`, so a single instrumentation default — the empty device token — can dominate a batch by itself. Under the mean it contributes one vote out of forty thousand. Neither behaviour is right for all cases, which is why the relation type must carry a reliability attribute and the fanout must have a documented cap.

**Limit 5 — bipartite oscillation.** When the graph is bipartite, `S` has eigenvalue `-1` and alternating layers flip sign instead of smoothing. Two-layer propagation on a bipartite account–device graph is the classic instance. The observable symptom is a representation whose sign changes between layer counts; the fix is a relation-aware or residual formulation, not more layers.

**Counterexample to "more hops is more signal".** In the Section 5 fixture, the two-hop neighbourhood of `C` is identical to its one-hop neighbourhood, because every neighbour of `C` returns only to `C` and its other neighbours. Doubling the traversal budget produced no new accounts. Hop count is a budget, not a source of information, and a depth increase must be justified by a measured change in the feature distribution, not by intuition.

**Counterexample to "connectivity is suspicion".** A shared payment processor, a landlord's address, an employer's device pool and a marketplace's escrow account all produce high-degree hubs connecting unrelated parties. Any component rule applied to such a graph returns a component containing most of the population. The mathematical statement "these nodes are connected" is true and evidentially empty; only the relation type and its provenance can separate infrastructure from association.

| Result | Required assumption | Diagnostic | Violating example | Correct alternative |
|---|---|---|---|---|
| `m_v` in `[min x, max x]` | weights non-negative, sum > 0 | range check per batch | negative decay weight | clamp or reject the record |
| `S^L` smooths toward the dominant eigenvector | connected, `abs(lambda_i) < 1` | per-layer variance collapse | bipartite graph, `lambda = -1` | residual or relation-aware layer |
| PPR truncation loses `(1-alpha)^(j+1)` | row-stochastic `W`, `alpha` fixed | compare truncated vs converged | dangling node leaking mass | explicit dangling-node rule |
| `precision@k = \|P ∩ Top_k\| / k` | `k` completed distinct cases | count distinct case ids | alerts counted as cases | dedup before the division |
| Mean is not a probability | none | value may lie in `[0,1]` by accident | score 0.5 with 90% base rate | calibrate before interpreting |

**Takeaway:** Every aggregate has a regime where it says nothing; state the regime before quoting the number.

## 5. Worked Numerical Example

**Example title and evidence label:** "Six-account device-and-transfer fixture, decision at 18:00" — ILLUSTRATIVE throughout; the arithmetic is DERIVED and independently checked below. No step describes a real customer, institution or incident.

**Workflow and decision:** rank six candidate accounts and select `k = 3` cases for tomorrow's review queue, using an eligible one-hop neighbour mean, then repeat the selection under a leaked (post-decision) edge to show the effect.

**Valid assumptions and intentional simplifications:** all edges are treated as undirected for the aggregate, which is a teaching simplification — real money flow is directed and the direction changes the meaning. Each account is its own case. Review effort is constant at 12 minutes. Time zone is single and synchronised. Decay, currency and amount are ignored except where stated.

### Inputs

Own-history feature `x` = confirmed adverse events in the trailing 90 days (unit: events per account). All six values were computed under the same definition and were available before 18:00.

| Account | `x` (events) | Notes |
|---|---|---|
| A | 1 | established, low activity |
| B | 2 | established |
| C | 0 | newly opened, moderate activity |
| D | 5 | repeated disputes, under review |
| E | 0 | dormant |
| F | 3 | established |

Edge ledger. "Eligible at 18:00" requires both event time and availability time at or before 18:00.

| Edge | Type | Event time | Availability | Eligible at 18:00 | Why |
|---|---|---|---|---|---|
| e1 A–B | device d1 | 09:10 | 09:12 | yes | both before cutoff |
| e2 A–C | transfer A→C | 10:05 | 10:06 | yes | both before cutoff |
| e3 B–C | transfer B→C | 11:20 | 11:22 | yes | both before cutoff |
| e4 C–D | device d2 | 14:40 | 14:41 | yes | both before cutoff |
| e5 D–E | transfer D→E | 17:50 | 18:30 | **no** | arrived 30 min after the decision |
| e6 E–F | device d3 | 15:00 | 15:02 | yes | both before cutoff |
| e7 F–D | device d1 | 18:40 | 18:41 | **no** | event itself is in the future |
| e8 B–D | device d1 | 08:00 | 08:01 | yes | both before cutoff |

### Step-by-step computation

| Step | Rule | Inputs with units | Intermediate (full precision) | Reported result | Independent check |
|---|---|---|---|---|---|
| 1 | snapshot filter | 8 ledgers rows | keep e1,e2,e3,e4,e6,e8 | 6 eligible edges | 8 − 2 excluded = 6 |
| 2 | degree count | eligible adjacency | A2 B3 C3 D2 E1 F1 | degrees | sum = 12 = 2 × 6 ✓ |
| 3 | G1b, seed A | x_B=2, x_C=0 | (2+0)/2 | 1.000000 | between 0 and 2 ✓ |
| 4 | G1b, seed B | x_A=1, x_C=0, x_D=5 | (1+0+5)/3 | 2.000000 | between 0 and 5 ✓ |
| 5 | G1b, seed C | x_A=1, x_B=2, x_D=5 | 8/3 | 2.666667 | between 1 and 5 ✓ |
| 6 | G1b, seed D | x_C=0, x_B=2 | (0+2)/2 | 1.000000 | between 0 and 2 ✓ |
| 7 | G1b, seed E | x_F=3 | 3/1 | 3.000000 | equals the single neighbour ✓ |
| 8 | G1b, seed F | x_E=0 | 0/1 | 0.000000 | equals the single neighbour ✓ |
| 9 | G1c, a=b=0.5 | per seed | `h = 0.5x + 0.5m` | see table below | unit U preserved ✓ |

Representation table with the arithmetic written out:

```text
h_A = 0.5*1 + 0.5*1.000000 = 0.5 + 0.500000 = 1.000000
h_B = 0.5*2 + 0.5*2.000000 = 1.0 + 1.000000 = 2.000000
h_C = 0.5*0 + 0.5*2.666667 = 0.0 + 1.333333 = 1.333333
h_D = 0.5*5 + 0.5*1.000000 = 2.5 + 0.500000 = 3.000000
h_E = 0.5*0 + 0.5*3.000000 = 0.0 + 1.500000 = 1.500000
h_F = 0.5*3 + 0.5*0.000000 = 1.5 + 0.000000 = 1.500000
```

Ranking descending with ties broken by ascending account identifier:

```text
D 3.000000  >  B 2.000000  >  E 1.500000  =  F 1.500000  >  C 1.333333  >  A 1.000000
tie at 1.500000 resolved by identifier: E before F
selected queue (k=3): D, B, E
```

### Evaluation at capacity

Suppose a mature independent outcome source establishes that the positives in this six-account population are B, D and E. `|P| = 3`, `N = 6`.

```text
precision@3 = |{B,D,E} intersect {D,B,E}| / 3 = 3/3 = 1.000000
recall@3    = 3 / 3 = 1.000000
base rate   = 3/6 = 0.500000
lift@3      = 1.000000 / 0.500000 = 2.000
```

A self-only baseline that ranks purely on `x` yields the order D(5), F(3), B(2), A(1), C(0), E(0) and selects {D, F, B}:

```text
precision@3 = |{D,F,B} intersect {B,D,E}| / 3 = 2/3 = 0.666667
recall@3    = 2/3 = 0.666667
```

The graph aggregate selects one additional true positive in this fixture. That is a statement about six constructed rows, not evidence that graph features generalise — a single case difference across six accounts carries essentially no information.

### The leaked variant

Now recompute the ranking with the post-decision edge `e7 (F–D)` wrongly included, which is what happens when a replay uses the end-of-day graph.

```text
N_F becomes {E, D}:  m_F = (0 + 5)/2 = 2.500000
h_F = 0.5*3 + 0.5*2.500000 = 1.5 + 1.250000 = 2.750000

new rank: D 3.000000 > F 2.750000 > B 2.000000 > E 1.500000 > C 1.333333 > A 1.000000
new selection (k=3): D, F, B

precision@3 = |{D,F,B} intersect {B,D,E}| / 3 = 2/3 = 0.666667
recall@3    = 2/3 = 0.666667
```

The leaked edge displaced a true positive from the queue and replaced it with a false positive. Both queues have the same size, so the capacity constraint is satisfied in each case and nothing in the operational metrics looks anomalous. This is precisely why leakage is dangerous in a capacity-limited system: the queue length absorbs the error silently.

### Wrong path and correction

A plausible mistake is to filter on event time only. Under that rule `e5 (17:50)` enters, because 17:50 is before 18:00, and the availability timestamp of 18:30 is ignored. The result:

```text
N_E becomes {D, F}:  m_E = (5 + 3)/2 = 4.000000
h_E = 0.5*0 + 0.5*4.000000 = 2.000000
N_D becomes {C, B, E}: m_D = (0 + 2 + 0)/3 = 0.666667
h_D = 0.5*5 + 0.5*0.666667 = 2.5 + 0.333333 = 2.833333
```

`E` rises from 1.5 to 2.0 and ties `B`; `D` falls from 3.0 to 2.833. The selected set under ascending-identifier tie-breaking is still {D, B, E}, so a careless analyst comparing only queue membership would conclude that the event-time-only filter was harmless. The feature values changed by up to 0.5 U — enough to move a threshold, a downstream explanation or a banding rule — and the two accounts whose values changed are exactly the ones an appeal would scrutinise. The correction is to filter on `min(event_time, availability_time)` being at or before the decision, which in practice means filtering on availability alone.

### Independent checks

The degree sum equals twice the eligible edge count (12 = 2 × 6). Every mean lies inside its neighbourhood's attribute range. The convex-mix weights sum to one, so `h` also lies within the per-account attribute range and the global range `[0, 5]`. The `k` selected rows are distinct, so the capacity count is exact. The leaked and baseline selections differ by exactly one account, consistent with a single changed representation.

Recomputing `h_C` by the alternate route — `G2a` with `d_C = 3` — gives `1/sqrt(9) + 2/sqrt(9) + 5/sqrt(6) = 0.333333 + 0.666667 + 2.041241 = 3.041241` for the normalised sum, which is *not* comparable to `m_C`; the two normalisers answer different questions and must not be cross-checked against each other. This is itself the lesson: an alternate-calculation check is only valid when the alternate computes the same estimand.

### Weighted variant, for contrast

Declare relation weights of 2 for transfers and 1 for device sightings, with `C`'s neighbourhood `{A(transfer), B(transfer), D(device)}`:

```text
weighted numerator = 2*1 + 2*2 + 1*5 = 2 + 4 + 5 = 11
weight denominator  = 2 + 2 + 1 = 5
m_C(weighted) = 11/5 = 2.200000  (versus 2.666667 unweighted)
h_C(weighted) = 0.5*0 + 0.5*2.2 = 1.100000
```

The rank of `C` is unchanged at this capacity, but the value moved by 0.23 U. Near a threshold, that is the difference between an alert and silence. Weights are a modelling claim about provenance reliability and must be declared before evaluation — deriving them from observed outcomes would import the label into the feature.

### Production-scale interpretation

At production volume nothing about the formula changes; the surrounding engineering does. Degrees are fetched from a store with a p99 latency, not from a literal in-code list. The deduplication step is a real join against the case system. The tie-break rule has to be deterministic under concurrent insertion. And the snapshot identifier for each row has to be persisted, because a replay six months later is only meaningful if it reconstructs the same snapshot.

**Remaining uncertainty and what the example does not establish:** the labels were chosen to make the mechanism visible, so neither precision figure estimates real performance; the fixture has no true ring, no entity-resolution error and no label delay; and no latency, throughput or cost is measured.

**Takeaway:** The same arithmetic produces a valid online score and an invalid retrospective one; only the availability filter distinguishes them.

## 6. Visual / Geometric Intuition

```text
Availability timeline (schematic, not to scale). The bracket marks the decision cutoff.

09:00   10:00   11:00   12:00   13:00   14:00   15:00   16:00   17:00   18:00   19:00
  |       |       |       |       |       |       |       |       |       |       |
 e1 ----+
         e2 ----+
                 e3 ----+
                         |                                        e4 ----+
                         |                                                e6 ----+
                         |                                                        |
                         |                                                        |
 08:00 e8 ---------------+                                                        |
                                                    e5 event 17:50 ----+           |
                                                             e5 available --------+ 18:30
                                                                          |
                                                                    18:00 CUTOFF
                                                                          |
                                                              e7 event 18:40 -------> (next day)

Records counted for the 18:00 decision:  e1 e2 e3 e4 e6 e8
Records not counted:                      e5 (late availability), e7 (future event)
```

Read the timeline as a filter, not as a story. `e5` happened before the cutoff and still does not count, because the *system* could not see it. `e7` fails both tests. A replay that uses a single timestamp per row cannot express this distinction, which is why the ledger schema stores two.

```text
Eligible graph at 18:00 (undirected rendering of the Section 5 ledger):

     A(1) --------- B(2) --------- D(5)
       |              |  \           |
       |              |   \          |
       +--------------C(0) \---------+
                      |               (e5 D-E omitted: not available)
                      |               (e7 F-D omitted: future)
                     (e4)
                      |
   E(0) ----------- F(3)
        e6

degrees:  A=2  B=3  C=3  D=2  E=1  F=1     sum = 12 = 2 x 6 edges
```

The drawing is deliberately undirected, which is the simplification flagged in Section 5. In a real payment graph `e2` and `e3` would both point into `C`, and that directional convergence — two independent senders into one receiver — is a different and more interesting pattern than an undirected triangle. Drawing it undirected hides exactly the structure an investigator would find most relevant.

```text
One-hop versus two-hop from seed C:

   hop 1:  { A, B, D }                                  |  hop 2: { A, B, D }
                                                         |
        A --- C --- B                                    |  A's nbrs: {B, C}
              \                                          |  B's nbrs: {A, C, D}
               \                                         |  D's nbrs: {C, B}
                D                                        |  union minus C = {A, B, D}

   New accounts discovered by doubling traversal budget: 0
```

This is the sheet's most useful geometric fact and it is uncomfortable: additional depth did not extend the reachable set at all. The fixture is small, but small fixtures are where intuition is built, and the intuition that "two hops reach further" is simply false in dense local clusters. Depth pays off in sparse, chain-like graphs and costs latency everywhere.

```text
Neighbourhood expansion tree used for the Section 9 sizing (branching factors 10 then 5):

                     seed (1)
                    /  |  ...  \            10 first-hop nodes
                   *   *       *            = 1,000 occurrences over 100 seeds
                  /|\  ...                   each expands to 5
                 * * *  ...                 = 5,000 second-hop occurrences
                (5 children per first-hop node)

   layer-1 edge occurrences: 100 x 10          = 1,000
   layer-2 edge occurrences: 1,000 x 5         = 5,000
   total sampled occurrences                   = 6,000
   node occurrences incl. seeds: 100+1,000+5,000 = 6,100   (upper bound, pre-dedup)
```

The tree is an upper bound on work only if every branch is full; real degree distributions are heavy-tailed, so most seeds have fewer than ten neighbours and a few have thousands. The cap therefore binds on a small fraction of seeds and produces a long tail of requests that hit the cap exactly — which is where latency peaks and where the incompleteness flag matters most.

```text
Community partition used for the modularity check in Section 10:

   +----------------------+        +----------+
   |   A    B    C    D   |        |   E   F  |
   |   internal edges: 5  |        |  internal: 1
   |   degree sum: 10     |        |  degree sum: 2
   +----------------------+        +----------+
        Q contribution 0.138889         Q contribution 0.138889
        total modularity Q = 0.277778
```

Modularity near 0.28 is a weak partition by the usual reading, and in a six-node graph it is not evidence of anything. The visual is included because it shows where the number comes from — two dense-ish blocks joined by a single edge — and because it warns that a community label is a property of a chosen algorithm on a chosen snapshot.

**Takeaway:** Draw the availability cutoff and the eligible subgraph before drawing the network; the second picture is only meaningful inside the first.

## 7. Comparison / Selection Table

| Method | Preconditions | Quality / guarantee | Time / memory / cost | Best for | Avoid when | Evidence |
|---|---|---|---|---|---|---|
| Existing rules | documented, time-safe thresholds | auditable; coverage limited by rule count | negligible compute; high analyst maintenance | safety fallback and mandatory baseline | rules reference unavailable labels | DERIVED |
| Tabular gradient boosting | time-safe account and transaction features | strong on per-account history; no relational signal | one row per candidate; low serving cost | the honest comparator every graph claim must beat | the only informative evidence is relational | ILLUSTRATIVE |
| Hand-built graph aggregates | reliable snapshot and bounded fetch | interpretable; misses multi-hop and typed nuance | one fetch per candidate; latency-dominated | adding graph signal with a small team | snapshots cannot be reconstructed | DERIVED |
| Degree-normalised propagation (PPR) | row-stochastic eligible adjacency | smooth reach over many hops at fixed cost | iterative or truncated power iteration | reach matters but depth budget is fixed | cycles will be presented to investigators as rings | DERIVED |
| Sampled GNN (GraphSAGE-style) | stable neighbourhoods, serving-equivalent sampling | learns relational representation; sampling adds variance | subgraph sampling memory and tail latency | adequate infrastructure and measured incremental value | fanout instability or graph freshness unacceptable | ILLUSTRATIVE |
| Relation-aware GNN (R-GCN-style) | trustworthy relation typing | separates device from money semantics | one parameter block per relation | several heterogeneous relations with reliable provenance | relation types are miscoded or sparse | ILLUSTRATIVE |
| Component / motif rules | typed bounded graph and graph-aware review | easy to explain small closed groups | near-linear in edges | investigator leads on candidate rings | connectivity treated as proof | DERIVED |

Selection guide, applied in order:

```text
IF any option requires information unavailable at decision time
   -> disqualify it; no amount of offline lift repairs an ineligible input.
ELSE IF the graph store cannot return a reproducible snapshot id per request
   -> use tabular features; a non-reproducible graph score cannot be defended on appeal.
ELSE IF the tabular baseline is not yet measured at the real review capacity
   -> measure it first; there is nothing to beat.
ELSE IF one-hop aggregates show incremental value at equal capacity and pass the leak tests
   -> ship aggregates; they are the cheapest defensible graph signal.
ELSE IF two-hop reach adds value and cycles are rare in the candidate population
   -> try truncated PPR with a declared alpha and depth.
ELSE IF typed relational structure is the mechanism and labelled edges are plentiful
   -> evaluate a sampled or relation-aware GNN against the aggregate baseline.
ELSE -> keep the simpler model; record the measurement gap as OPEN rather than guessing.
```

The ordering is not a claim that simpler is always better. It is a claim about the burden of proof: each step rightward adds a failure mode, an operational dependency and an explanation obligation, so it must buy measured value at the same budget. A GNN that ties a one-hop mean is a loss, because it costs a sampling service, a fanout contract, a drift monitor and a story an investigator cannot verify.

Empirical comparison record — the required fields are listed with their honest status rather than invented values:

| Field | Required content | Status in this sheet |
|---|---|---|
| workload / data | candidate cohort, period, population size | OPEN — no dataset accompanies this sheet |
| hardware / software | store, model server, versions | OPEN |
| metric definition | numerator, denominator, grain, window | defined in G7/G8 |
| sample / repetitions | seeds, folds, bootstrap units | OPEN |
| result / uncertainty | point estimate with interval | OPEN — none measured |
| primary source | paper, internal report, reproduction | not applicable |
| reproduced? | yes / no / partially | no |
| date | measurement date | not applicable |

Nothing in this table should be filled with a plausible number. An unmeasured comparison plan is more useful than a fabricated benchmark, because it tells the next engineer exactly what to go and run.

### Temporal reconstruction: adversarial clock trace

The following timeline is ILLUSTRATIVE. No event describes a real investigation or a real system.

Decision for account `A` is issued at 10:00. All times share one synchronised zone.

Transfer `T1` occurs 09:40 and reaches the feature service 09:43 — eligible, because availability precedes the decision by 17 minutes.

Transfer `T2` occurs 09:50 and reaches the feature service 10:04 — ineligible, despite the earlier event time. A replay that uses `T2` produces a score the live system could not have produced.

A device sighting occurs 10:03 and arrives 10:03 — ineligible on event time alone, so availability does not rescue it.

A device record observed yesterday is corrected at 10:08. A faithful replay at 10:00 uses the pre-correction version, even though today's readers see only the corrected row.

A neighbour's confirmed outcome becomes known at 09:55. Whether it may enter a feature depends on the target and the policy: if the neighbour's outcome is a different event from the seed's target and the policy permits label features, it may; if it is the same adjudication propagated, it may not.

An investigator confirms `A` at 10:30. That confirmation is `A`'s outcome, and it is unavailable at 10:00 by construction.

A graph edge hand-created at 10:35 during that investigation is unavailable at 10:00 for two reasons: it did not exist, and its content encodes the outcome.

At 10:10 a new decision for `A` may use `T2`. That decision still may not use the 10:30 label or the 10:35 edge, because neither existed yet.

The general rule: for each historical prediction, join to the last committed record version whose availability time is at or before that prediction's cutoff. When multiple versions share a timestamp, a committed sequence number must order them; without one, exclude the ambiguous records rather than inventing an order.

Record the availability clock of the *serving path*, not the source export time. If exports run hourly but the online graph reads a stream, the export timestamp will make every feature look later than it was, and the resulting replay will understate what the model knew. The reverse error — trusting a warehouse landing time when the online service read the record earlier — overstates eligibility and creates leakage.

For distributed snapshots record a consistent watermark or per-partition offsets, not a single wall clock. A watermark is a completeness claim under a specific ingestion contract; it is not a promise that no correction will ever arrive. Preserve the request-level snapshot identifier when the store exposes one, because it is more precise than the nominal refresh schedule.

Never repair an older snapshot by backdating a late record to its event time. If a correction arrived after the decision, the honest record is "the value was X at decision time and became Y later", and both versions belong in lineage.

Three negative-control tests make the invariant checkable: insert a future-only edge into a fixture and confirm earlier features do not change; change a future label while holding permitted history fixed and confirm the old prediction is unchanged; move one edge's availability across the cutoff and confirm the feature changes exactly at the declared boundary. Passing all three establishes those invariants for that fixture, not for every production path.

### Typed-path reasoning

A path `A -> B -> C` encodes two directed payments. It does not encode control, common ownership or intent. Reversing the direction changes the economic reading completely, even though an undirected connectivity algorithm returns the same component — which is why "component" answers a different question from "who sent money to whom".

A two-hop path `account -> device -> account` records observed shared device use. It does not establish that one person controls both accounts. Merchants, employers, households, internet cafés and public terminals all produce this pattern legitimately.

A low-frequency identifier is informative but not conclusive. Rarity is a property of the population, and a token that is rare because instrumentation changed last week is not evidence of a relationship.

If an identifier is vendor-assigned, assess collision, recycling and missing-value behaviour before using its degree. A sentinel value such as the empty string or the all-zeros token must never form a normal edge: it will connect an arbitrary subset of the population and dominate every aggregate that uses it.

Component analysis must state three things: whether edges are directed, which relation types are included, and the time interval covered. Without all three the number is uninterpretable.

A chain `A -> B -> C` is one weakly connected component and three strongly connected components. A cycle `A -> B -> C -> A` may be circular movement, ordinary settlement, or unrelated transfers that happen to close; amounts and timing decide.

For an alleged funding chain, an outgoing transfer cannot be financed by a later incoming transfer unless additional funds or evidence exist. Time-ordered reachability is necessary for a money-flow claim and far from sufficient, because balances and other obligations are not in the graph.

Motif counts must state whether repeated transfers create multiple occurrences, whether reciprocal edges count separately, and whether the pattern is induced. Normalising by exposure is usually right, but do not divide by a tiny exposure without an uncertainty check.

Directionality of suspicion is not universal. High out-degree can be ordinary disbursement for a payroll processor; low out-degree can be suspicious in a peer-to-peer marketplace. The comparison group has to match account type, age, geography and opportunity to form edges.

### Production operations and investigator workflow

A decision record should contain: candidate id, decision timestamp, graph snapshot id, feature version, model artifact checksum, queue policy version, and whether graph retrieval completed, timed out, or was truncated at a cap.

`neighbours = 0 after a successful fetch` and `neighbours = 0 inserted by a fallback after a failure` must be distinguishable downstream. They look identical in a feature vector and mean opposite things.

A fallback score is not comparable to a graph score. If both enter the same queue, the ranking mixes two different scales and the calibration claim silently becomes false. Either calibrate the fallback separately or route fallback traffic to a different, smaller queue with its own capacity.

Define operational maxima for traversal depth, per-relation fanout, total edges examined and wall time. A returned-edge cap alone does not bound the work, because selecting a sample from a large adjacency list may require scanning far more edges than are returned.

If a query hits its work budget, return an incomplete flag. A silently partial neighbourhood is worse than a refused request, because the partial result is plausible and therefore survives review.

Track freshness per source. Aggregate ingestion health can look green while one relation type is fourteen hours stale, and that relation may be the one carrying the signal.

Corrections should be append-only or versioned so that old snapshots remain reconstructible. Where retention policy forbids keeping them, record the reproducibility limitation explicitly rather than synthesising a substitute.

Rollbacks must pin the graph schema and the feature transformations compatible with the older model. Rolling back a model against a newer schema produces a crash or, worse, a silent misparse.

Access control applies during traversal, including intermediate nodes, not only when formatting the explanation. A multi-tenant graph with a filter applied at output time has already read data it should not have read.

Deletion must propagate through caches, precomputed features, embeddings, training exports and investigator attachments, under the applicable retention policy. Where full model remediation is impossible, say so instead of promising instantaneous removal.

An investigation package should state the seed, the permitted relationships, the source timestamps, and why each item is relevant. It should separate raw evidence from model-generated hypotheses and later analyst annotations, and it should present benign alternative explanations — shared infrastructure, commercial relationships — alongside the risk indicators.

A sudden backlog rise can come from higher arrivals, slower handling, duplicates, or lost staffing. Measure all four before touching a threshold. As a trivial illustration: 55 arrivals and 40 completions add 15 cases in a period, and a threshold change cannot retroactively create investigator hours.

**Takeaway:** Reject unsafe methods first, then demand measured incremental value against the simplest credible baseline at the real budget.

## 8. Failure Modes & Edge Cases

Every entry is HYPOTHETICAL. Each names a competing-cause set, a check that discriminates between them, a bounded fix and a verification. A negative diagnostic does not prove the absence of other failures, and a fix counts as complete only when a replay or controlled observation confirms the intended invariant.

### F-01 Late edge ingestion

Symptom: the offline graph contains a pre-decision transaction that the online feature never saw, so replay and production disagree.

Competing causes: the record arrived after scoring (genuine late availability), or the online fetch dropped an already-available record (serving defect). The first is a data-contract issue; the second is an availability incident.

Discriminating check: compare the source event timestamp, the serving-availability timestamp and the request trace for the same account. A late availability stamp with a successful upstream write supports the first cause; a missing write or a failed fetch with an early availability stamp supports the second.

Bounded fix: enforce both event and availability cutoffs in the feature path, and repair dropped writes as a separate change. Do not fix a serving defect by loosening the cut.

Verification: replay the frozen request from an append-only availability log and confirm byte-identical feature values.

### F-02 Post-outcome relationship

Symptom: a relationship type appears almost exclusively among confirmed positives and is highly predictive.

Competing causes: the edge was created by the investigation after adjudication (downstream artifact), or the relationship genuinely existed before the decision and carries real signal.

Discriminating check: join edge creation timestamp and creator identity against adjudication time. Edges authored by the case system after the outcome are artifacts.

Bounded fix: exclude investigation-derived relation types from pre-decision features while retaining them for post-hoc review.

Verification: removing those edges should restore offline features to the values in the online log.

### F-03 Future label in a neighbourhood aggregate

Symptom: a neighbour-label rate feature produces near-perfect offline discrimination.

Competing causes: labels learned after each seed's decision entered the aggregate, or the feature captures genuinely early known alerts that were available at the time.

Discriminating check: list every label's availability timestamp used in the aggregate for a sample of seeds and compare against each seed's decision time.

Bounded fix: recompute using only labels available before each individual seed's decision — not before the training cutoff, which is a different and weaker constraint.

Verification: compare a historical feature replay to the online feature log across a stratified sample of seeds.

### F-04 Self-label contamination

Symptom: training performance far exceeds validation, and the gap survives regularisation.

Competing causes: the seed's own label reached its neighbourhood statistic through a self-loop or cycle, or the model is simply over-parameterised.

Discriminating check: remove the seed label from the store and recompute; trace every aggregation path that returns to the seed.

Bounded fix: compute label aggregates out-of-fold with explicit seed exclusion and temporal eligibility.

Verification: build a fixture with unique labels per node and assert that changing a held-out seed's label cannot change its own feature.

### F-05 Entity overmerge

Symptom: one identity suddenly acquires thousands of peers and its scores collapse toward the population mean.

Competing causes: an identifier collision merged unrelated people, or a legitimate enterprise genuinely operates many accounts.

Discriminating check: inspect match evidence per merged pair — the attributes that triggered the match and their agreement rate — and the identifier's cardinality before and after.

Bounded fix: split using reversible provenance and tighten the match rule; keep the old and new mappings.

Verification: replay prior scores, quantify the alert and customer impact, and re-check the merge on an adjudicated pair sample.

### F-06 Entity undersplit

Symptom: known repeat actors remain in separate components with no shared neighbourhood.

Competing causes: one actor fragmented across identifiers, or the apparent similarity reflects a shared service rather than a shared person.

Discriminating check: inspect independently validated linkage evidence and downstream investigation findings for the candidate pairs.

Bounded fix: add a reviewed, time-versioned linkage rather than a global fuzzy merge.

Verification: measure link precision and recall on an independently adjudicated pair sample, including hard negatives from shared infrastructure.

### F-07 Shared-device supernode

Symptom: tens of thousands of accounts share one device token and every aggregate on those accounts is dominated by it.

Competing causes: a public kiosk or corporate device pool (real sharing), a default sentinel token (instrumentation), or a genuinely coordinated ring.

Discriminating check: inspect token generation rules, session diversity, site context and the time distribution of first sightings.

Bounded fix: represent relation reliability explicitly, cap expansion per relation, and treat sentinel values as absent rather than as edges.

Verification: confirm that hub treatment does not erase confirmed small-ring signal, by measuring recall on a known small-ring audit set.

### F-08 Transfer direction reversal

Symptom: money-flow patterns look economically backwards to an investigator.

Competing causes: ingestion inverted sender and receiver, or an analyst read a computed reverse edge as a real payment.

Discriminating check: compare stored edge endpoints against source transaction records by transaction id.

Bounded fix: correct the mapping, and give computational reverse edges a distinct relation type.

Verification: replay a known sender/receiver ledger and check flow conservation per account.

### F-09 Duplicate events

Symptom: degree and event counts jump after a retry, and means shift without any new activity.

Competing causes: at-least-once delivery duplicated writes, or the source emitted legitimately repeated payments.

Discriminating check: group by immutable source event id and compare payloads; identical ids with repeated rows indicate duplication.

Bounded fix: make edge ingestion idempotent on source event id while preserving genuinely distinct events.

Verification: replay the same batch twice and assert graph state is unchanged.

### F-10 Correction overwritten

Symptom: historical feature replay disagrees with the original serving log.

Competing causes: a source correction overwrote the earlier value in place, or the original online result was itself stale.

Discriminating check: compare version history and correction arrival time against the decision timestamp.

Bounded fix: store versions and select the one available at the decision; make corrections append-only.

Verification: replay on both sides of the correction's availability boundary and confirm the feature only changes after it.

### F-11 Clock skew

Symptom: events appear to arrive before they occurred.

Competing causes: source and ingestion clocks disagree, or event-time semantics were misunderstood (for example, a scheduled rather than executed timestamp).

Discriminating check: compare clock-synchronisation telemetry with representative source payloads; a consistent offset across one source indicates skew.

Bounded fix: fix clock discipline and define conservative handling for uncertain timestamps.

Verification: assert ordering on monitored synthetic events after correction.

### F-12 Time zone and daylight saving

Symptom: the daily graph changes at an unexpected hour, or a one-hour discontinuity appears in degree counts.

Competing causes: a local timestamp parsed as UTC, or a reporting window defined in a different zone.

Discriminating check: trace one raw timestamp through zone assignment, DST rule and normalised value.

Bounded fix: store explicit zone-aware timestamps and preserve the original representation.

Verification: test timestamps on both sides of a DST transition.

### F-13 Recycled device token

Symptom: one device appears to change hands implausibly often.

Competing causes: a recycled token represents successive physical devices, or a shared device genuinely has many users.

Discriminating check: compare token lifetime, app version and platform generation policy.

Bounded fix: version token namespaces and downgrade device use to a weak relation.

Verification: validate against independently reviewed ownership evidence for a sample.

### F-14 Train/test component overlap

Symptom: validation precision looks far better than the following month's production precision.

Competing causes: near-duplicate entities span the split, or legitimate historical structure is genuinely available at deployment and the split was unrealistically harsh.

Discriminating check: measure entity and component overlap between train and test, and compare against the overlap the deployment actually has.

Bounded fix: use a temporal holdout as primary and report entity-disjoint and component-disjoint tests as stress tests, not as the headline.

Verification: state which known historical neighbours remain permissible at deployment and confirm the test split reflects it.

### F-15 Full-graph normalisation

Symptom: test-node features change when future edges are added, despite a time-based split.

Competing causes: degree and normalisation statistics were computed on the full graph, or a declared transductive protocol allowed it.

Discriminating check: record when degree statistics and scalers were computed and on which snapshot.

Bounded fix: fit every transformation on permitted training snapshots and recompute per evaluation snapshot.

Verification: assert that altering a future edge cannot change any earlier training feature.

### F-16 Unreviewed cases labelled negative

Symptom: precision looks stable while recall quietly collapses.

Competing causes: unreviewed positives were written as negatives, or the underlying fraud mix genuinely shifted.

Discriminating check: join label provenance and compare the unknown fraction by score band and by cohort.

Bounded fix: keep unknown distinct from negative, and collect randomised audit labels.

Verification: report audited estimates with inclusion probabilities and intervals, and compare against the naive figure.

### F-17 Delayed outcomes

Symptom: recent cohorts look cleaner than older cohorts at the same decision age.

Competing causes: outcomes have not matured, or controls genuinely reduced fraud.

Discriminating check: plot cumulative outcome discovery against decision age for each cohort.

Bounded fix: restrict the primary comparison to mature cohorts and report censoring explicitly.

Verification: recompute older cohorts at matched observation ages and confirm the gap narrows or persists.

### F-18 Changed review policy

Symptom: reported model precision rises immediately after a threshold or routing change.

Competing causes: reviewers now see a different case mix, or ranking genuinely improved.

Discriminating check: compare both models on the same pre-policy candidate cohort at fixed capacity.

Bounded fix: separate ranking changes from policy changes and version the policy.

Verification: use a shadow queue or an authorised randomised comparison to isolate the two effects.

### F-19 Top-k duplicates

Symptom: the queue is full but only a handful of distinct networks were investigated.

Competing causes: many account alerts belong to one case, or the incident genuinely needs several independent reviews.

Discriminating check: cluster alerts by case identity and inspect the queue before and after deduplication.

Bounded fix: rank at the case grain with traceable member alerts, and report distinct cases as the capacity unit.

Verification: measure distinct cases reviewed and positives found under the same labour budget.

### F-20 Case merge after review

Symptom: historical case metrics change retroactively with no score change.

Competing causes: multiple alerts were merged into one investigation, or outcomes were corrected.

Discriminating check: track case-merge versions and outcome attribution at each reporting cutoff.

Bounded fix: freeze the evaluation unit per report, or version case membership.

Verification: reconcile alert, case and outcome counts at each snapshot.

### F-21 High-degree traversal timeout

Symptom: tail latency spikes on a small fraction of seeds.

Competing causes: one hub causes fanout explosion, or the store has an unrelated I/O bottleneck.

Discriminating check: correlate edges examined and hub degree with per-request timing, stratified by degree bucket.

Bounded fix: bound per-relation fanout and total work, and return an explicit incomplete status.

Verification: load-test degree-stratified traffic and compare tail latency against quality loss.

### F-22 Sampling bias

Symptom: a bounded neighbourhood consistently misses a known cluster.

Competing causes: sampling favours recent or frequent edges, or the cluster genuinely was not present at decision time.

Discriminating check: log sampled and eligible neighbour ids with selection probabilities.

Bounded fix: stratify samples by relation or reserve budget for rare evidence.

Verification: measure recall and variance against full neighbourhoods on a feasible audit set.

### F-23 Sampling instability near the threshold

Symptom: the same account receives different scores on repeated requests within one snapshot.

Competing causes: random neighbour sampling changes the aggregate, or a feature dependency changed between requests.

Discriminating check: fix snapshot and model version, repeat with controlled seeds, and separate variance components.

Bounded fix: use reproducible sampling, or average several draws inside an explicit latency budget.

Verification: measure rank churn near the capacity boundary, which is where churn costs a review slot.

### F-24 Stale graph cache

Symptom: online predictions use relationships that the store has already superseded.

Competing causes: cache invalidation failed, or ingestion is legitimately delayed and the cache is correct.

Discriminating check: compare source availability, cache version and the request's snapshot timestamp.

Bounded fix: repair invalidation and expose graph freshness to the serving policy so it can degrade deliberately.

Verification: inject a known edge and trace its visibility end to end through every layer.

### F-25 Online–offline feature skew

Symptom: training and serving values differ for the same entity at the same nominal cutoff.

Competing causes: different aggregation code, a mismatched as-of snapshot, or a serialisation difference.

Discriminating check: replay the same entity, cutoff, source versions and model version through both paths and diff intermediates.

Bounded fix: unify feature semantics and keep a parity fixture in the test suite.

Verification: compare exact values on edge cases including empty neighbourhoods, single neighbours and hub neighbours.

### F-26 Missing-neighbourhood default

Symptom: new accounts score uniformly low, and a spike of identical values appears for accounts under thirty days old.

Competing causes: empty neighbourhood mapped silently to zero, or genuinely thin history.

Discriminating check: inspect the missing indicator separately from the numeric aggregate and count exact-default values by tenure.

Bounded fix: add an explicit indicator and a documented, evaluated fallback.

Verification: evaluate the cold-start slice and the outage slice as separate populations, not pooled with the main result.

### F-27 Zero-weight aggregation

Symptom: the weighted mean returns `NaN` or infinity.

Competing causes: all eligible weights are zero, or numeric precision overflowed on extreme values.

Discriminating check: log the denominator and the individual finite weights for failed requests.

Bounded fix: emit a missing-neighbourhood representation instead of dividing by zero.

Verification: test empty, all-zero, extreme and mixed-weight cases in one parametrised test.

### F-28 Graph drift

Symptom: model performance declines at stable volume.

Competing causes: relation composition changed (new instrumentation, new product), or labels matured differently across cohorts.

Discriminating check: compare degree distribution, relation mix, freshness and outcome maturity by cohort.

Bounded fix: reassess relation construction and retrain only on valid mature data.

Verification: run a stable tabular baseline alongside to separate graph drift from general population drift.

### F-29 Proxy discrimination

Symptom: one community receives a disproportionate share of alerts.

Competing causes: a weak shared-location or shared-device feature proxies a protected characteristic, or underlying exposure genuinely differs.

Discriminating check: audit score, review and confirmed-outcome rates by legally appropriate group and context, with counsel.

Bounded fix: engage fairness, legal and domain owners; restrict unsupported proxy features.

Verification: monitor appeals and subgroup error rates after any change.

### F-30 Investigator overload

Symptom: backlog rises despite high reported precision.

Competing causes: arrivals exceed completion, or case complexity increased so each case takes longer.

Discriminating check: measure incoming distinct cases, completed cases and handling-time distributions separately.

Bounded fix: reduce workload volume, add capacity, or revise prioritisation with the risk owner — not silently.

Verification: track aging, throughput and unresolved high-risk count.

### F-31 Leak through the action

Symptom: "account closed" or "card blocked" is a powerful feature.

Competing causes: closure was triggered by the very risk assessment being predicted, or a prior independent closure is legitimate evidence.

Discriminating check: trace the action's trigger, execution time and causal dependency on the label.

Bounded fix: exclude downstream actions; retain prior independent actions only when time-safe and independently sourced.

Verification: causal provenance review plus retrospective feature replay.

### F-32 Outcome leakage through a component

Symptom: a component becomes highly predictive immediately after a large investigation.

Competing causes: the investigation created links among the confirmed accounts, or the connections genuinely predated it.

Discriminating check: compare edge creation times and provenance with the first investigation timestamp.

Bounded fix: separate observed transactional relations from investigative annotations at the schema level.

Verification: demonstrate identical predictions when future annotations are removed.

### F-33 Immature negatives

Symptom: many "legitimate" negatives later become positive.

Competing causes: the fixed label horizon is too short for this fraud type, or the target definition changed.

Discriminating check: compare late discovery rates by cohort age for each fraud type.

Bounded fix: lengthen the horizon for that type, or model unresolved outcomes explicitly.

Verification: reassess performance after the chosen maturity period elapses, and report the change.

### F-34 Ring-level double counting

Symptom: reported detected loss exceeds plausible exposure.

Competing causes: losses summed across overlapping ring members, or payments genuinely carry independent losses.

Discriminating check: join losses to unique transaction and recovery ids.

Bounded fix: deduplicate at the loss-event grain and net recoveries.

Verification: reconcile the final ledger against finance totals.

### F-35 Unstable selection threshold

Symptom: alert volume oscillates around the budget and the selected set churns daily.

Competing causes: large blocks of tied scores sit at the boundary, or candidate arrivals are bursty.

Discriminating check: inspect tie mass at the cutoff, arrival profile and threshold update schedule.

Bounded fix: deterministic tie-breaks plus a documented batching or pacing policy.

Verification: measure capacity utilisation and rank churn per period.

### F-36 Graph service unavailable

Symptom: scoring failures rise during an upstream incident.

Competing causes: the graph store is down, or the model server itself is failing.

Discriminating check: compare graph-fetch error rates and latency against inference process health.

Bounded fix: an approved fallback with an explicit missing-graph status, calibrated separately.

Verification: exercise recovery and monitor the decision mix by serving path.

### F-37 Model-version mismatch in replay

Symptom: replayed scores differ from stored scores.

Competing causes: the model artifact differs, or the feature snapshot differs.

Discriminating check: pin artifact checksum and snapshot id separately, and vary one at a time.

Bounded fix: record immutable model, feature and graph versions with every score.

Verification: reproduce a stored request with its original artifacts and match exactly.

### F-38 Hidden fanout cap

Symptom: offline graph model outperforms online by a wide margin at identical candidates.

Competing causes: serving silently truncates neighbourhoods, or serving reads a fresher graph.

Discriminating check: compare fetched neighbour ids, applied caps and snapshot versions for the same seeds.

Bounded fix: train with serving-equivalent sampling and report incompleteness in the feature contract.

Verification: measure quality and latency across the actual degree distribution, not a uniform sample.

### F-39 Explanation built from the current graph

Symptom: an investigator sees a rationale containing a relationship that could not have been known at scoring.

Competing causes: the explanation was generated against today's graph, or the decision used a different snapshot than recorded.

Discriminating check: compare every explanation edge's availability against the decision timestamp.

Bounded fix: generate explanations from the recorded snapshot id only.

Verification: add a later edge and confirm explanations for old decisions are unchanged.

### F-40 Attention misread as causation

Symptom: reviewers treat a large attention weight as proof that an edge caused the score.

Competing causes: the interface presents weights as evidence, or reviewers import their own causal reading of a ranking cue.

Discriminating check: perturb the edge and observe the score change; a stable weight with no score movement demonstrates the gap.

Bounded fix: present provenance and sensitivity as leads, with explicit language about what a weight is.

Verification: usability review with investigators plus preserved appeal channels.

### F-41 Cross-tenant traversal

Symptom: a graph query returns an edge the tenant should not see.

Competing causes: an authorisation filter was omitted at traversal, or identifiers collided across tenants.

Discriminating check: trace tenant-scoped keys and the authorisation decision at each hop.

Bounded fix: enforce authorisation at data access and isolate identifier namespaces.

Verification: adversarial cross-tenant path tests plus access-log audit.

### F-42 Deletion not propagated

Symptom: a deleted identifier is still retrievable through a derived artifact.

Competing causes: deletion reached the source but not caches, feature tables, embeddings or exports.

Discriminating check: trace deletion requests through every derivative with lineage.

Bounded fix: apply the approved deletion and model-remediation policy across derivatives.

Verification: lineage-based removal checks plus an independent access audit.

### F-43 Contradictory labels on one node

Symptom: a node carries both a confirmed-fraud and a confirmed-legitimate label.

Competing causes: the labels refer to different targets or times, or the adjudications genuinely conflict.

Discriminating check: inspect label target grain, effective time and adjudication history.

Bounded fix: represent target and version explicitly and route real disputes to review.

Verification: test outcome joins with multiple events per account.

### F-44 Component explosion

Symptom: most of the population falls into one component.

Competing causes: a shared processor or device hub joins benign entities, or linkage rules overmerged accounts.

Discriminating check: decompose component growth by relation type and test removal of generic infrastructure relations.

Bounded fix: relation-aware components or bounded motifs instead of declaring one ring.

Verification: audit retained and lost links against known cases.

### F-45 Cold-start distribution shift

Symptom: performance is strong overall but collapses for accounts younger than thirty days.

Competing causes: graph methods need history that new accounts lack, or the candidate population's tenure mix changed.

Discriminating check: report degree-zero rates and performance by entity age, and compare cohorts over time.

Bounded fix: combine self features with a separate cold-start policy and a documented prior.

Verification: hold out new entities by time and monitor their outcomes separately.

### F-46 Circular review evidence

Symptom: features containing "reviewer suspicion" from a prior model keep the model self-confirming.

Competing causes: the prior annotation reflects a model score, or it reflects independent investigator evidence.

Discriminating check: trace annotation provenance and what information was on screen when it was written.

Bounded fix: separate independently observed facts from model-generated suggestions at the schema level.

Verification: blinded or randomised review where ethics and policy permit.

### F-47 Outcome attribution across a case

Symptom: every account in a confirmed ring is labelled fraudulent.

Competing causes: the case outcome was broadcast to members, or each member was separately adjudicated.

Discriminating check: compare entity-level evidence and adjudication records to the case-level status.

Bounded fix: keep case and entity labels distinct; leave unsupported member outcomes unknown.

Verification: audit a sample of case-to-member assignments.

### F-48 Calibration shift

Symptom: a score of 0.8 no longer corresponds to an 80% observed rate.

Competing causes: base rates changed, or ascertainment changed with the review policy.

Discriminating check: compare calibration on mature randomly audited cases across periods.

Bounded fix: recalibrate on a representative mature sample and document uncertainty.

Verification: monitor reliability by cohort and by action policy.

### F-49 Partial snapshot build

Symptom: a scheduled snapshot has unexpectedly low degree.

Competing causes: a source partition failed, or activity genuinely dropped.

Discriminating check: compare source offsets, partition completion markers and edge counts per source.

Bounded fix: mark the snapshot incomplete and use a documented fallback; do not promote it.

Verification: rebuild idempotently and reconcile counts before activation.

### F-50 Community instability

Symptom: small data changes substantially reassign ring membership.

Competing causes: the clustering algorithm is unstable, or a high-impact bridge edge is real.

Discriminating check: vary the seed and perturb uncertain edges at a fixed snapshot.

Bounded fix: report stability alongside membership and never present a cluster as a finding.

Verification: review stability and investigator utility before any operational use.

### F-51 Label-stream stall

Symptom: the neighbour-label feature becomes mostly missing overnight.

Competing causes: the label pipeline is delayed, or fraud genuinely decreased.

Discriminating check: compare label-stream lag and availability watermark against stable event volume.

Bounded fix: use a missingness-aware fallback and repair the stream.

Verification: confirm recovered labels do not retroactively enter old predictions.

### F-52 Coordinated graph manipulation

Symptom: many accounts intentionally share a benign high-degree identifier.

Competing causes: adversaries are manipulating a graph feature, or the identifier is a legitimate public service.

Discriminating check: inspect coordination signatures, relation provenance and independently verified activity.

Bounded fix: limit weak-edge influence and add review for sudden coordinated changes.

Verification: adversarial insertion tests measuring both ranking and false positives.

### F-53 Fragmentation evasion

Symptom: large transfers split into coordinated small transfers across new accounts.

Competing causes: the threshold rule is vulnerable to fragmentation, or behaviour changed innocently.

Discriminating check: compare aggregate flow and account-creation sequences under time-safe windows.

Bounded fix: bounded multi-entity features alongside policy review.

Verification: evaluate on later cohorts and monitor unintended impact on legitimate users.

### F-54 Wrong capacity unit

Symptom: a top-100 account ranking exceeds daily investigator capacity.

Competing causes: accounts were equated with cases, or cases vary in effort.

Discriminating check: measure distinct investigations and handling minutes for the selected accounts.

Bounded fix: rank case bundles with effort estimates and keep an escalation path.

Verification: compare actual labour minutes and outcomes against the budget.

### F-55 Censoring by intervention

Symptom: observed loss falls for the high-score group.

Competing causes: intervention prevented loss, or high scores were false positives.

Discriminating check: compare treatment and outcomes under an authorised randomised or quasi-experimental design.

Bounded fix: estimate intervention effects with a defensible design and legal oversight.

Verification: report policy, uncertainty and residual confounding.

### F-56 Decay sign error

Symptom: older relationships receive more weight than recent ones.

Competing causes: the decay sign is reversed, or ages were computed across a time-zone boundary.

Discriminating check: compute weights for two known ages at a fixed decision time.

Bounded fix: correct the age computation and assert non-negative eligible ages.

Verification: test equal ages, zero age, future events and very old events.

### F-57 Cross-relation scale imbalance

Symptom: a generic device hub overwhelms rare transfer links in a joint representation.

Competing causes: all relations share one denominator, or message magnitudes differ by orders of magnitude.

Discriminating check: inspect per-relation counts and contribution magnitudes before combination.

Bounded fix: normalise within relation and validate relation weights.

Verification: relation ablations reporting both quality and subgroup false positives.

### F-58 Cohort mismatch in comparison

Symptom: a graph model appears better than the baseline.

Competing causes: different eligible candidates or label maturity were used, or the improvement is real.

Discriminating check: join candidate ids and decision windows across both evaluations.

Bounded fix: re-evaluate both on one time-safe cohort at one capacity.

Verification: report paired uncertainty and the exclusion log.

### F-59 Unbounded two-hop expansion

Symptom: memory rises sharply after adding one layer.

Competing causes: fanout multiplies across layers, or activations are retained unnecessarily.

Discriminating check: trace distinct sampled nodes, edges and activation memory per layer.

Bounded fix: reduce fanout or layers and measure the lost relational coverage.

Verification: measure memory and quality at each workload percentile.

### F-60 Stale intervention policy

Symptom: a historically good model generates costly, low-value alerts.

Competing causes: fraud tactics changed, or the action policy no longer matches capacity.

Discriminating check: compare case mix, effort and mature outcomes against historical periods.

Bounded fix: revisit thresholds and features with policy owners, at the case grain.

Verification: monitor fraud outcomes and customer harm after the change.

### A concrete review protocol

1. Fix the decision being evaluated: rank cases for review at the start of each shift.
2. Freeze the eligibility population before looking at scores.
3. Record which accounts and transactions are legally and operationally eligible.
4. Exclude records unavailable at the cutoff and count exclusions by reason.
5. Freeze candidate generation so baseline and challenger share it.
6. Freeze case-construction rules, including how many alerts may map to one case.
7. Declare a deterministic tie-break before selecting the queue.
8. Express the budget in case-minutes, or an agreed case-count approximation.
9. Preserve the baseline's original serving features; do not add challenger-only evidence.
10. Reconstruct each candidate's graph at its own decision time.
11. Reconstruct feature versions under the same availability rule.
12. Separate graph ingestion time from source event time in the replay ledger.
13. Record the entity-resolution version and any provisional links used.
14. Do not rewrite earlier model inputs with later identity corrections.
15. If a correction changes a label, version the label independently of features.
16. Fit scaling, imputation, selection and parameters only on the training period.
17. Make early stopping and hyperparameter choices on a validation period.
18. Keep the final test period untouched until selection is complete.
19. Fit neighbour-label aggregates only from labels available at each decision.
20. Exclude the seed's own target and every path that returns it.
21. Declare whether previously seen accounts may appear in the test graph.
22. Treat that setting as distinct from an unseen-account stress test.
23. Preserve historical context an unseen account would genuinely have.
24. Test representation of a new node without a prelearned identifier embedding.
25. Score the same candidate set with both models before choosing.
26. Deduplicate both queues into the same case unit.
27. Report queue overlap so investigators can see the workload change.
28. Report distinct cases and distinct accounts in each queue.
29. Record actual or defensible estimated effort per selected case.
30. State the outcome horizon and the date through which labels are available.
31. Separate confirmed positive, confirmed negative and unresolved.
32. Report the unresolved fraction by score band and cohort.
33. Compute precision@k only with an explicit label policy and denominator.
34. Never substitute a negative for an unresolved selected case.
35. If outcomes are incomplete, publish a lower bound only under stated assumptions.
36. Recognise that even a valid lower bound does not identify true precision.
37. Compute recall only with an observed or defensibly estimated denominator.
38. Collect a randomised sample of otherwise unreviewed cases where policy allows.
39. Record each sampled case's selection probability.
40. Record non-response and completion separately from sampling.
41. Use a sampling-aware estimator only where its assumptions are justified.
42. Escalate cases with zero sampling probability; they cannot be recovered from that sample.
43. Compare on matched calendar windows so base-rate change is not read as lift.
44. Report the precision–recall trade-off across several plausible budgets.
45. Do not choose the budget after inspecting test labels without accounting for it.
46. Report absolute differences alongside ratios.
47. Attach intervals that reflect within-case dependence where feasible.
48. Group overlapping accounts by investigation or network as resampling units.
49. Explain how that grouping choice affects the interval.
50. Report time-slice variation, not a single pooled number.
51. Compare new and established entities separately.
52. Compare low-, medium- and hub-degree cases using declared bins.
53. Compare relation types without claiming an ablation proves a mechanism.
54. Inspect slices with high missing-neighbourhood rates or stale snapshots.
55. Audit whether false positives concentrate on benign infrastructure.
56. Examine appeal outcomes and erroneous actions under approved privacy protections.
57. Measure investigator time gathering evidence, not just queue size.
58. Record whether review quality changes when analysts see graph explanations.
59. Distinguish prevented from observed when interventions suppress later loss.
60. Do not estimate prevented fraud by multiplying scores by transaction value.
61. Where permissible, use a controlled comparison for incremental impact.
62. Include customer harm, case aging and operational availability as guardrails.
63. Measure graph fetch and inference latency on representative traffic.
64. Measure end-to-end latency directly rather than summing marginal percentiles.
65. Capture the share of scores served by fallback logic.
66. Evaluate fallback quality on the traffic it actually serves.
67. Predefine a stop condition for stale data, cross-tenant access or capacity overload.
68. Preserve an audit record tying each score to model and graph versions.
69. Archive the evaluation manifest without implying indefinite retention is lawful.
70. Obtain legal, security and investigation-owner approval before an action-changing rollout.

**Takeaway:** Diagnose the mechanism before applying the fix; every fix needs its own verification.

## 9. Complexity & Cost Analysis

### 9a. Algorithmic complexity

Notation: `V` nodes, `E` eligible edge records, `d` representation width, `L` layers, `B` seeds per batch, `f_l` the fanout cap at layer `l`, `S` sampled edge occurrences, `R` relation types, `k` review capacity, `h` hub degree.

| Operation | Input sizes | Time | Space | I/O / communication | Assumptions and dominant bottleneck |
|---|---|---|---|---|---|
| Snapshot materialisation | `V`, `E` | `O(V + E)` scan, dominated by sort/shuffle | `O(V + E)` | full ledger read per period | constant-time indexing; external sort may dominate |
| Degree computation | `E` | `O(E)` | `O(V)` | one pass | multiedge multiplicity must be declared |
| One-hop mean, per seed | `d_v` | `O(d_v * d)` | `O(d)` per seed | one adjacency read | latency, not arithmetic, dominates |
| Full two-hop, per seed | `f_1`, `f_2` | `O(f_1 + f_1 f_2)` edge ops | `O(f_1 f_2 * d)` | two round trips minimum | hub degree can multiply examined edges |
| Relation-aware layer | `S`, `R`, `d` | `O(S d + \|N\| d²)` | `O(\|N\| d)` | parameter tensors per relation | `d²` transform dominates at large width |
| GraphSAGE-style batch | `B`, `f_1`, `f_2`, `d` | `O(B f_1 f_2 d + B d²)` | `O(B f_1 f_2 d)` | subgraph sampling service | sampling memory and tail latency |
| PPR, truncated to `J` iterations | `E`, `J` | `O(J E)` per source, `O(J * E * seeds)` in the worst case | `O(V)` per source | sparse matrix–vector | convergence depends on `1 - alpha` |
| Motif census, 3-node | `E`, max degree `h` | `O(E h)` naive | `O(V)` | neighbourhood reads | hub degree dominates |
| Modularity of a partition | `E`, `C` communities | `O(E)` | `O(C)` | none | community labels already given |
| Rank and select top `k` | `N` candidates | `O(N log N)` sort, `O(N log k)` heap | `O(k)` | none | ties near the boundary |

Three observations matter more than the asymptotics. First, the arithmetic is trivial at fraud scale: a batch of a hundred seeds with fanout 10 then 5 performs a few million floating-point operations, which is microseconds of compute and milliseconds of network. Every interesting cost is I/O, memory for sampled subgraphs, and human review.

Second, the worst-case cost of a `k`-hop expansion is governed by the maximum degree, not the mean. A traversal bound expressed as "fanout 10" bounds retained neighbours per node, but a query that must *choose* ten neighbours from a forty-thousand-entry adjacency list still examines up to forty thousand entries unless the store maintains an index or a precomputed sample.

Third, sampling changes the statistical object. The sampled subgraph is an execution artifact; the neighbourhood the investigator would see is larger. A model trained on fanout-10 samples and evaluated on full neighbourhoods is being evaluated on inputs it will never receive.

### 9b. Back-of-envelope sizing

**Workload.** `B = 100` seeds per batch; fanout `f_1 = 10`; fanout `f_2 = 5`; width `d = 64`; float32 (4 bytes per component). Sampling without replacement inside each local neighbourhood, no global deduplication.

```text
layer-1 edge occurrences          = 100 * 10                = 1,000
layer-2 edge occurrences          = 1,000 * 5               = 5,000
total sampled edge occurrences    = 1,000 + 5,000           = 6,000
node occurrences incl. seeds      = 100 + 1,000 + 5,000     = 6,100   (upper bound)

raw representation storage:
  bytes = 6,100 nodes * 64 components * 4 bytes/component
        = 6,100 * 256
        = 1,561,600 bytes
  MiB   = 1,561,600 / 1,048,576 = 1.4892578125 MiB  (~1.49 MiB)

parameter storage (one 64x64 matrix per layer):
  bytes per matrix = 64 * 64 * 4 = 16,384 B = 16 KiB
  two layers       = 32 KiB
```

**Compute.** Per-node transform cost is `2 d² = 2 * 4,096 = 8,192` FLOPs; per-edge aggregation is `2 d = 128` FLOPs.

```text
transform FLOPs  = 100 seeds * 8,192 * 2 layers        = 1,638,400
aggregation FLOPs = 6,000 edge occurrences * 128       = 768,000
total FLOPs       = 2,406,400  (~2.41 MFLOP per batch)
per seed          = 2,406,400 / 100 = 24,064 FLOP
```

At an illustrative sustained rate of 10 GFLOP/s on a shared CPU core, that batch is 0.24 ms of arithmetic. Any statement that graph ML is "expensive" at this scale is a statement about data movement, not multiplication. `OPEN`: no measurement of actual FLOP/s or latency was performed for this sheet.

**Latency.** Assume each seed requires one round trip that returns its first-hop rows and one that returns the union of second-hop rows, each with an assumed 4 ms server-side cost, at 8-way concurrency:

```text
serial round trips per seed   = 2
serial time per seed          = 2 * 4 ms = 8 ms
total serial, 100 seeds       = 800 ms
with 8-way concurrency        = 800 / 8 = 100 ms  (idealised, no queueing)
```

The 4 ms assumption is ILLUSTRATIVE. The structural point survives any realistic value: end-to-end latency is dominated by round trips per seed times the number of seeds divided by concurrency, so the tuning levers are batching seeds per request and eliminating a round trip, not shrinking `d`.

**Memory.** The 1.49 MiB figure is raw payload only. Add edge features, indices, allocator overhead, activation buffers, batch padding and concurrent requests; a 3–10× factor over payload is common in practice, and the correct response is to measure rather than assume. Training adds gradients and optimiser state, typically several times the forward activation footprint, and is a different budget entirely.

**Review capacity.**

```text
available minutes per day      = 480 min
minutes per case (illustrative) = 12 min  (includes evidence gathering and documentation)
capacity k                      = 480 / 12 = 40 cases/day

sensitivity: at 20 min/case -> 480 / 20 = 24 cases/day
             at  8 min/case -> 480 /  8 = 60 cases/day
```

The quotient's unit is cases per day because minutes cancel. A high-complexity case can consume several slots, so a fixed `k` is an approximation whose error grows with case-complexity variance; a capacity model in case-minutes is more honest than a case count when variance is large.

**Audit sizing.** To estimate the positive rate among unreviewed cases with margin `e` and confidence `1 - alpha_s`, the normal approximation gives `n = z² p(1-p) / e²`. With `z = 1.96`, `p = 0.20`, `e = 0.05`:

```text
n = 3.8416 * 0.20 * 0.80 / 0.0025
  = 3.8416 * 0.16 / 0.0025
  = 0.614656 / 0.0025
  = 245.8624  ->  246 cases

finite population correction with N = 5,000:
n_adj = 246 / (1 + (246 - 1)/5,000) = 246 / 1.049 = 234.51  ->  235 cases
```

Two hundred and thirty-five audited cases at twelve minutes each is 2,820 investigator-minutes, or roughly six working days of the entire 480-minute daily budget. That is the real price of an identified recall estimate, and it explains why teams report precision and leave recall unidentified. The honest alternative is to say so.

**Economics.** Cost per confirmed case includes review labour, graph storage, inference, data engineering and customer remediation. With the illustrative figures above, review labour alone is `12 min * (1/60) h = 0.2 h` per case, so at a hypothetical loaded rate the labour cost per reviewed case is `0.2 h * rate`. Monetary figures are omitted because no rate, currency or date basis has been established for this sheet — quoting one would be fabrication.

**Takeaway:** Compute is free at fraud scale, data movement is cheap, and investigator minutes are the binding resource; size the system accordingly.

## 10. Key Theorems / Results Reference

| Result | Correct statement | Essential hypotheses | Practical implication | What it does not imply | Source |
|---|---|---|---|---|---|
| Handshake lemma | `sum_v d_v = 2 * \|E\|` for undirected graphs counting self-loops twice | undirected, same multiplicity convention in both counts | cheap consistency check on any snapshot | nothing about semantics of edges | DERIVED |
| Mean within range | For non-negative weights with positive total, a weighted mean lies in `[min x, max x]` | finite values, weights ≥ 0, denominator > 0 | out-of-range aggregate is a defect signal | the mean preserves rare extremes | DERIVED |
| Spectral smoothing | Applying `S = D^{-1/2} A D^{-1/2}` `L` times scales eigenvector `i` by `lambda_i^L` | symmetric `S`, eigenvalues in `[-1, 1]` | depth beyond the informative regime averages nodes together | that fewer layers always rank better | DERIVED (spectral decomposition) |
| PPR convergence | `p = alpha (I - (1-alpha) W^T)^{-1} s` with Neumann expansion; truncation error bounded by `(1-alpha)^{J+1}` | `W` row-stochastic, `alpha` in `(0,1]` | a depth cap is principled, not arbitrary | that PPR ranks fraud better than a mean | DERIVED (Neumann series) |
| Ranking optimality | Ordering by `P(Y=1 \| f)` descending maximises expected positives in the top-`k` for every `k` | the posterior is correctly specified and available | calibrate if you need an optimal queue; a monotone score suffices for ranking | that a well-ranked score is calibrated | DERIVED (exchange argument) |
| Exchange argument detail | If two cases are misordered, swapping them cannot decrease the expected positive count in the prefix | identical action cost per case | explains why AUC-like monotonicity is enough for ranking | that action costs are equal in reality | DERIVED |
| Precision at fixed `k` | `precision@k = \|P ∩ Top_k\| / k` | `k` completed distinct cases; `P` enumerated | quantifies yield per review slot | recall, or net value, without `\|P\|` | DEFINITION |
| Selective-label non-identification | If observation of `Y` depends on the score, the population positive rate is not identified from observed labels alone | selection depends on a variable the model controls | recall requires an audit or an external outcome source | that a large labelled sample fixes it | DERIVED |
| Modularity | `Q = sum_c [ L_c/m - (D_c/(2m))² ]` for `m` edges, internal edges `L_c`, degree sum `D_c` | undirected, single connected component decomposition | comparability of partitions on one snapshot | that a high `Q` partition is a fraud ring | DERIVED |
| Motif z-score | `z = (N_obs - mu) / sigma` under a degree-preserving null | `mu`, `sigma` estimated from randomised graphs | separates structure from degree artefacts | that `z` large implies coordination | SOURCED (null-model practice) |
| Fanout bound | With caps `f_l`, sampled edge occurrences are at most `B * f_1 * ... * f_L` before dedup | fixed caps, no dedup | rough compute budget | distinct-node count or wall time | DERIVED |

The ranking-optimality result deserves the derivation because it resolves a common confusion. Let `p_i` be the true probability that case `i` is positive, and consider a prefix of size `k`. The expected number of positives in the prefix is `sum over i in prefix of p_i`. If two cases `i` and `j` are included and excluded respectively with `p_i < p_j`, swapping them adds `p_j - p_i > 0` to the expectation. Repeated swapping terminates at the descending-`p` order, which is therefore optimal for every `k` simultaneously. The hypothesis is that `p` is available and correct; the result says nothing when the score is only monotone, other than that the *expected* count is optimal for the true probabilities, not for the score.

That distinction is the practical one. A monotone score gives the same ranking as its calibrated version and therefore the same selected set at any `k`; it differs only if you need to say "this case has an 80% chance" or choose per-case effort by expected value. Calibration is a separate, later problem, and it must be measured against outcomes on a representative mature sample.

Modularity's derivation is short enough to include: the null model places an edge between `i` and `j` with probability `d_i d_j / (2m)`, so the expected internal edge count for community `c` is `D_c²/(4m)`, and the observed internal count is `L_c`. The difference, normalised by `m`, is the contribution `L_c/m - (D_c/(2m))²`. In the Section 5 fixture with `m = 6` and the partition `{A,B,C,D} / {E,F}`:

```text
community 1: L = 5, D = 10 -> 5/6 - (10/12)^2 = 0.833333 - 0.694444 = 0.138889
community 2: L = 1, D =  2 -> 1/6 - ( 2/12)^2 = 0.166667 - 0.027778 = 0.138889
Q = 0.138889 + 0.138889 = 0.277778
```

A `Q` of 0.28 on six nodes is a description of where edges cluster; it is not evidence of organisation. Modularity also has a known resolution limit, so it tends to merge small communities in large graphs — which is exactly the failure an analyst looking for small rings would most want to avoid.

**Takeaway:** Identities and spectral facts justify sanity checks; none of them establishes that a graph model earns its keep.

## 11. Interview / Exam Questions

Score each answer 0 (wrong) to 3 (correct, assumption-aware, defensible at the boundary). All scenarios are constructed.

### Level 1 — Definition / Recall

#### Q1. Shared device and common control

**Prompt:** Three accounts use the same device identifier in one week. Does that establish a fraud ring?

**Answer:** No. It establishes that three accounts were observed with one device under the declared relation and time rules. That pattern arises from households, employers, kiosks, resold handsets and sentinel tokens. Common control needs independent identity evidence; criminal association needs more still.

**Alternative:** If the identifier is a strong, non-recycled, personally bound token and independent evidence links it to one person, the relation becomes strong evidence of shared access — still not of intent.

**Follow-up:** What if the token is a constant default written by a failing SDK?

**Follow-up answer:** Its evidentiary value is nil and the hub it creates will distort every aggregate; fix instrumentation and treat the sentinel as absent.

**Red flag:** Reading connectivity as a finding. It is a description of the snapshot.

#### Q2. Event time versus availability time

**Prompt:** A payment occurred at 11:30, reached the serving path at 12:05, and a score was issued at 12:00. May the payment enter the score?

**Answer:** No. Availability time governs. Event time tells you when the world changed; availability time tells you when the model could know. Only the second is a constraint on the model.

**Alternative:** If a different model received the same payment at 11:31 through an independent path, that model may use it — subject to its own cutoffs and policy.

**Follow-up:** The warehouse shows the row landed at 11:32. Does that prove the online model saw it?

**Follow-up answer:** No. Warehouse landing, graph publication, cache visibility and model fetch are four distinct transitions; the earliest possible serving time is the latest of them.

**Red flag:** Filtering on one timestamp because two are "basically the same".

#### Q3. Unknown versus negative

**Prompt:** An account was never reviewed and shows no chargeback. Is it a negative?

**Answer:** It is unknown. A negative requires a defined target, a matured horizon, and an ascertainment process that would have observed the outcome had it occurred. Absence of review and absence of chargeback are both consistent with fraud.

**Alternative:** If a complete, mature, independent outcome source covers the target population, an observed non-event can be a negative under that source's coverage statement.

**Follow-up:** What if thirty per cent of the cohort is still within the dispute window?

**Follow-up answer:** Those cases are censored, not negative; either wait for maturity or model the unresolved state explicitly.

**Red flag:** Filling the label column with `0` because the join returned null.

### Level 2 — Derivation / Apply

#### Q4. Derive the degree-normalised aggregate

**Prompt:** Seed `B` has degree 3; its neighbours are `A` (`d=2`, `x=1`), `C` (`d=3`, `x=0`), `D` (`d=2`, `x=5`). Compute both the plain mean and the symmetric-normalised sum, and explain the difference.

**Answer:** Plain mean `= (1+0+5)/3 = 2.0`. Normalised: `1/sqrt(3*2) + 0 + 5/sqrt(3*2) = 6/sqrt(6) = 6/2.449489743 = 2.449489743`. The normalised version is larger because `D` has few alternatives: a low-degree neighbour is more specific about this seed, so its message is amplified.

**Alternative:** With `c_v,r = |N|` (R-GCN style) the result is the plain mean `2.0`; both normalisers are defensible and must be declared.

**Follow-up:** What happens if `D`'s degree rises to 1,000?

**Follow-up answer:** Its contribution shrinks toward `5/sqrt(3000) ≈ 0.0913`, and the aggregate approaches the contribution of the low-degree neighbours — which is the intended behaviour for a hub.

**Red flag:** Reporting `2.449` as a probability because it lies near the `[0, 5]` range mid-point.

#### Q5. Precision and recall with an incomplete denominator

**Prompt:** A queue selects 40 cases; reviewers confirm 24 positives. What are precision and recall?

**Answer:** Precision is `24/40 = 0.60`. Recall is not determined: the denominator is all positives in the population, and the unreviewed remainder is unlabelled. Reporting `0.60` as recall would implicitly assume the population contains exactly 40 positives.

**Alternative:** If an audit of 200 unreviewed cases with known inclusion probability finds 10 positives, an estimate of the unreviewed positive count can be formed — `(unreviewed / 200) * 10` is the crude version — and recall becomes `24 / (24 + estimate)` with an interval that propagates both terms.

**Follow-up:** The audit found zero positives in 200 cases. Is recall 1.0?

**Follow-up answer:** The point estimate is near 1.0 but the interval is wide and asymmetric; zero in a sample bounds the rate, it does not establish zero.

**Red flag:** Using reviewed cases as the recall denominator.

#### Q6. Two-hop reach in a dense cluster

**Prompt:** Seed `C` has neighbours `{A, B, D}`; `A`'s neighbours are `{B, C}`, `B`'s are `{A, C, D}`, `D`'s are `{C, B}`. How many new accounts does a second hop add?

**Answer:** The union of the neighbours' neighbourhoods is `{A, B, C, D}`; removing `C` leaves `{A, B, D}` — identical to the first hop. Zero new accounts.

**Alternative:** In a chain or a hub-and-spoke graph the second hop adds many; the answer is topology-dependent, which is the point.

**Follow-up:** If depth 2 adds nothing, is depth 2 free?

**Follow-up answer:** No. It costs a second round trip and the fanout multiplication, so it should be justified by measured feature change.

**Red flag:** Assuming more hops always means more information.

#### Q7. PPR truncation error

**Prompt:** With `alpha = 0.15`, how much restart mass remains beyond a 3-hop truncation?

**Answer:** The expansion weights hop `j` by `alpha * (1-alpha)^j`, so the total retained through `j = 0,1,2,3` is `0.15 * (1 + 0.85 + 0.7225 + 0.614125) = 0.15 * 3.186625 = 0.477994`. The residual is `1 - 0.477994 = 0.522006`, which equals `(1-alpha)^4 = 0.85^4 = 0.52200625`. Both routes agree, which is the check.

**Alternative:** A different `alpha` changes the trade-off: `alpha = 0.3` leaves `0.7^4 = 0.2401` beyond hop 3.

**Follow-up:** Is a 52% residual acceptable?

**Follow-up answer:** It bounds the mass you ignored, not the ranking error; in practice you compare truncated against converged scores on a sample rather than reasoning from the bound alone.

**Red flag:** Claiming a 3-hop cap "captures most of the influence" without computing the residual.

### Level 3 — Judgement / Design / Debug

#### Q8. Offline lift that vanishes online

**Prompt:** A new neighbour-label feature doubles offline precision at `k`. Online, the selected set changes but confirmed yield does not improve. Give an ordered diagnosis.

**Answer:** First check label availability per seed and seed self-exclusion — the classic explanation is that the offline feature saw labels unavailable at decision time. Second, check online graph freshness and fanout truncation against the offline snapshot. Third, check cohort and label maturity, since the online comparison may involve younger cases. Fourth, check whether the offline evaluation used the same capacity and deduplication.

**Alternative:** If replay parity is exact and cohorts match, the remaining explanation is that the offline gain was a selection artefact from evaluating many variants; report the number of variants tried.

**Follow-up:** Replay reproduces online values exactly and the gain persists offline. Now what?

**Follow-up answer:** The offline evaluation itself is biased — most likely a split that lets near-duplicate entities or components cross the boundary. Run an entity-disjoint and a component-disjoint stress test and compare.

**Red flag:** Concluding that the feature "doesn't work in production" without establishing which of the four mechanisms applies.

#### Q9. A hub in the traversal path

**Prompt:** A public terminal creates two-hop neighbourhoods of 200,000 accounts. How do you keep latency bounded without declaring all users fraudulent?

**Answer:** Classify the device relation as weak evidence, cap per-relation fanout, cap total examined edges, return an incomplete flag, and retain account-level features for the fallback. Then measure what the cap costs in quality on a stratified audit set that includes confirmed rings on shared infrastructure.

**Alternative:** Keep the hub but down-weight it by degree and reliability, which preserves hub signal at the cost of a more complex feature contract.

**Follow-up:** Confirmed rings do use the terminal. Does that justify removing the cap?

**Follow-up answer:** No — it justifies *stratifying* rather than removing: reserve budget for high-reliability relations and for small components, and evaluate recall specifically on the shared-infrastructure strata.

**Red flag:** Either "exclude all hubs" or "treat all hub users as suspicious"; both are single-mechanism answers to a mixed-population problem.

#### Q10. Capacity arithmetic and its assumptions

**Prompt:** A team has 480 investigator-minutes per day and cases average 12 minutes. How many cases fit, and what breaks the answer?

**Answer:** `480 / 12 = 40` cases per day, assuming deterministic effort, no interruptions, and no non-review work. Effort variance, case complexity and duplicate alerts all break it; measure the effort distribution and track distinct cases rather than alerts.

**Alternative:** At 20 minutes per case the capacity is 24; the point is that the number is derived, not chosen.

**Follow-up:** The model produces 60 high-scoring accounts that map to 35 cases. Is there a capacity problem?

**Follow-up answer:** At 35 distinct cases against 40 slots there is headroom, but only if the effort per case holds; if three of those cases are complex rings, the effective load may exceed the budget.

**Red flag:** Equating account alerts with cases, or treating capacity as a property of the model.

#### Q11. Explaining a score to an investigator

**Prompt:** An investigator sees a high score and a highlighted device edge. What may they conclude, and what must the interface do?

**Answer:** They may conclude the case is prioritised and that the edge is a lead worth checking. They may not conclude intent, identity or causation. The interface must show edge provenance, first-seen and last-seen times, the snapshot used, benign alternative explanations, uncertainty, and a route to challenge a wrong link.

**Alternative:** A sensitivity display — score without the edge — is a legitimate additional aid, as long as it is labelled as model sensitivity and not as proof of that edge's importance.

**Follow-up:** The explanation includes an edge recorded after the decision. What now?

**Follow-up answer:** It is not an explanation of that decision. Regenerate from the recorded snapshot and log the defect.

**Red flag:** Presenting attention weights as evidence, or an unlabelled post-hoc graph as the rationale.

#### Q12. Leakage through provenance rather than time

**Prompt:** A feature "device flagged by previous investigation" has an availability timestamp before the decision. Is it safe?

**Answer:** Not necessarily. If the flag was set by an investigator who was looking at the outcome of the current case's adjudication, the timestamp is early but the content is downstream. Availability is necessary, not sufficient; provenance review is the second gate.

**Alternative:** If the flag came from an unrelated, earlier investigation with its own evidence and no dependence on the current label, it is legitimate prior information.

**Follow-up:** How would you test that on data?

**Follow-up answer:** Trace creator, creation time and the information on screen at creation; then compare offline features with and without investigation-authored edges and see whether the lift disappears.

**Red flag:** Accepting any feature whose timestamp passes the cutoff.

#### Q13. Choosing between an aggregate and a GNN

**Prompt:** A one-hop mean and a two-layer GNN tie on offline precision at the real capacity. Which ships, and what would change your mind?

**Answer:** The aggregate ships. It has fewer moving parts, no sampling service, no fanout contract, an explanation an investigator can verify, and lower tail latency. A tie means the GNN's extra machinery bought nothing measurable.

**Alternative:** If the GNN wins specifically on a stratum that matters — for example new accounts with sparse history — and the win survives a disjoint-entity test, the decision changes for that stratum, possibly as a routed sub-model.

**Follow-up:** What if the GNN's advantage appears only at `k` far above capacity?

**Follow-up answer:** Then it is irrelevant: the queue never reaches that region, and optimising there is an evaluation error.

**Red flag:** Choosing the more sophisticated model because it scores better on a metric the operation does not use.

#### Q14. Recall with selective labels

**Prompt:** Only high-score accounts are reviewed and reviewers supply the labels. Can unbiased population recall be computed from those labels?

**Answer:** No. Observation of the label depends on the score, so the labelled subset is a biased sample of the population and the positives outside it are systematically missing. Options: an independent mature outcome source, or a designed audit with known inclusion probabilities, reported with non-response caveats.

**Alternative:** Report precision at capacity with an explicit label policy and state that recall is unidentified. That is a legitimate, honest result.

**Follow-up:** An audit of 300 unreviewed cases finds 6 positives. What can you say?

**Follow-up answer:** You can estimate the unreviewed positive rate as `6/300 = 0.02` with an interval, and combine it with the reviewed count, provided the audit sampled with known probabilities and non-response is addressed. The interval will be wide and that should be stated.

**Red flag:** Treating the reviewed sample as representative, or dropping the interval entirely.

**Takeaway:** Strong answers separate what was knowable, what was observed, and what a capacity-limited human can act on.

## 12. Quick Reference Card

| Item | Value |
|---|---|
| Topic | Time-correct graph-assisted ranking of deduplicated fraud investigation cases under a fixed review budget |
| Core formula | `m_v(t) = sum_{u in N_v(t)} x_u / \|N_v(t)\|` with `N_v(t)` restricted to records whose availability time is at or before `t` (G1b); normalised form `sum_u x_u / sqrt(d_v d_u)` (G2a) |
| Key assumption | Every node, edge, attribute and label was readable at the seed's decision time, and no permitted record is itself derived from the outcome |
| Complexity | `O(S d + \|N\| d²)` per layer for `S` sampled edge occurrences and width `d`; latency is dominated by round trips per seed, not by FLOPs |
| Use when | A reproducible typed snapshot exists, one-hop aggregates beat the tabular baseline at equal capacity, and review capacity is expressed in case-minutes |
| Avoid when | Snapshots cannot be reconstructed per request, relation provenance is untrustworthy, or connectivity would be presented as proof |
| Diagnostic | Replay the snapshot and feature versions as of each decision; test a future-only edge, a future label change, and an availability boundary move |
| Review constraint | Rank at the case grain, never substitute unknown for negative, and compute recall only with an observed or audited denominator |
| Go deeper | [Collection index](../INDEX.md); the source-specific production case remains a planned separate handoff |
