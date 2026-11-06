# Graph ML for Fraud and AML Ring Detection

## Section 1 — File Header

> **Purpose:** Design, evaluate, and operate a time-correct graph-assisted fraud investigation queue.
> **Prerequisites:** Basic probability, classification metrics, relational joins, timestamps, and graph adjacency lists.
> **Outcome:** Given an event ledger, construct an eligible graph, compute a neighbor feature, and evaluate a capacity-limited ranking without future knowledge.
> **Scope exclusions:** Legal reporting decisions, a complete anti-money-laundering program, and vendor-specific graph APIs are outside this sheet.
> **Level:** Advanced / production.
> **Version / reviewed:** Draft 0.1; not independently reviewed.
> **Evidence:** DERIVED and ILLUSTRATIVE teaching material; no production results are claimed.
> **Retrieval cue:** For suspiciously strong offline results, begin with the point-in-time checks in Sections 3 and 8.

The graph-source entry and production case entry in the directory inventory are scope references, not accessible source content in this workspace.
The crosswalk maps those entries to separate graph-method and case-library homes; this sheet lives in the graph-method home.
No detailed claim about the unavailable files is attributed to their authors.
A risk score supports prioritization; it is not proof, a legal finding, or a reason to deny service by itself.
The actionable unit might be a transaction, account, or deduplicated investigation case; the unit must be fixed before model selection.
The investigator owns the review decision; policy owners define permissible actions and appeal processes.

**Takeaway:** Build a defensible decision queue, not an automated guilt-by-association machine.

## Section 2 — Concept Map / Mental Model

```text
Ledger + identity evidence + known outcomes
        |  observed and available by decision time
        v
Versioned typed graph snapshot -----> time-safe feature calculation
        |                                      |
        |                                      v
        +----------------------------> baseline or graph model
                                               |
                                               v
                             ranked, deduplicated case queue
                                               |
                              capacity + human review + appeal
                                               |
                                               v
                        delayed and selectively observed outcomes
                                               |
                                               +--> future training only
```

Arrows denote data flow, not causation.
The feedback arrow cannot travel backward into an earlier decision's feature snapshot.
Identity evidence is not itself a verified identity.
An edge says that the specified relationship was observed, not that it was illicit.
Typed and directed edges distinguish shared-device use from transfers of money.
A time-safe snapshot reflects the information the serving system could actually access.
The model ranks candidates while the queue policy decides which ones fit within capacity.
A case may contain many graph nodes and transactions, so candidate scores need a deduplication policy.
A confirmed fraud outcome is delayed and can be missing for people who were never reviewed.
An apparent model improvement may reflect changed review behavior rather than better detection.
A graph model should beat simpler approaches on the same eligible candidate population and review budget.

**Takeaway:** The key invariant is that no input crosses backward over its availability time.

## Section 3 — Foundational Definitions

### Entities and relationships

A node is a typed entity represented by an identifier within an explicitly defined namespace.
An account is not automatically a person: one person can own multiple accounts and an account can have several authorized users.
An identifier collision can merge unrelated people and generate spurious paths.
A split identity can hide paths between accounts that actually belong together.
A device is a shared resource, not a legal person.
A household address can legitimately connect several unrelated legal account holders.
An edge is a typed relationship between two identifiers, with provenance and time attributes.
A transfer edge is directed from sender account to receiving account.
A device-use edge connects an account to a device observation, not necessarily to the device owner.
Repeated transactions can be distinct multiedges; collapsing them loses amounts, counts, and timestamps unless retained as aggregates.
A relationship type defines which endpoint types and directions are allowed.
A reverse transfer edge used for message passing is a derived relation, not another payment.
A component is a maximal set connected under a stated connectivity definition and graph snapshot.
A connected component is not necessarily a fraud ring; common infrastructure can make it enormous.
A motif is a local pattern of typed nodes and edges, not an adjudicated criminal organization.
Entity resolution is the reversible, evidence-based mapping of records to candidate entities.
A confidence score for a possible match is not an established identity without appropriate validation.

### Time and availability

Event time is when the source asserts that an event happened.
Availability time is when the model-serving path could first use that event.
Decision time is when a prediction must be made.
Outcome time is when the target event actually occurred.
Label availability time is when the outcome became available to the learner or reviewer.
A point-in-time graph for a decision contains only permitted records available by that decision time.
A later correction can change the current ledger without changing what was knowable for the earlier decision.
An as-of join selects the version valid under both event and availability constraints.
A late-arriving transaction can have an old event time but a new availability time.
A post-decision chargeback must not contribute to the decision's neighborhood label feature.
A training cutoff controls which labels may enter model fitting; it is distinct from each example's decision time.
A mature label is one for which a specified observation horizon has elapsed and the feedback process has had time to operate.
An unlabeled case is not automatically a negative case.

### Learning and evaluation

A feature is a numeric or categorical representation computed from eligible information.
A message is a neighbor-derived value passed along an allowed relation in a graph model.
An aggregation combines messages from a defined eligible neighbor set.
A graph neural network (GNN) repeatedly combines node representations with neighbor messages.
A one-hop representation depends on direct eligible neighbors; a two-hop representation can depend on their eligible neighbors.
Fanout is the maximum number of neighbors sampled per node per layer and relation under a specified sampling algorithm.
A seed is the node or edge whose score is being computed.
Inductive evaluation assesses whether the method can handle new nodes or graphs without requiring a learned identifier embedding for each one.
Transductive evaluation permits certain unlabeled test nodes during training; its permitted information must be declared.
Leakage is use of information unavailable under the intended decision or evaluation contract.
Label leakage includes using an outcome or its downstream consequences as a feature.
Target contamination includes computing a training example's neighborhood label aggregate using that example's own label.
Precision at capacity is confirmed positives among the top eligible reviewed cases divided by the number reviewed.
Recall at capacity is confirmed positives found in that queue divided by all positives in the defined evaluation population.
Recall cannot be estimated from reviewed cases alone without assumptions or additional outcome collection.
A score calibrated in one population or policy may not remain calibrated after selection or distribution change.
Investigation capacity includes case handling, evidence gathering, documentation, and appeals, not just score generation.

**Takeaway:** Define the entity, relation, clock, candidate unit, and label before selecting a model.

## Section 4 — Core Formulas with Full Derivation

### 4.1 Simplest / Base Case

For an eligible nonempty neighbor set, each neighbor contributes equally to a mean feature.
The sum of scalar features has feature units; division by the count preserves those units.
```text
FORMULA [G1]: m_v(t) = sum(x_u for u in N_v(t)) / |N_v(t)|
WHERE: v is the seed; t is decision time; N_v(t) is the eligible neighbor set;
       x_u is a scalar feature measured in a stated unit; m_v has that unit.
CONSTRAINT: |N_v(t)| > 0; each edge and feature version was available by t.
NUMERICAL EXAMPLE: features 2 and 4 -> m_v=(2+4)/2=3 feature units.
INTERPRETATION: the two available neighbors have mean feature value 3.
```
If there are no eligible neighbors, the mean is undefined, not automatically zero.
A model can use an explicit missing-neighborhood indicator alongside a documented default.
A default should be fitted and evaluated under the same serving convention.
Duplicated edges can implicitly reweight a neighbor; define whether the set consists of unique nodes or edge occurrences.

### 4.2 General Case

For multiple relation types and vector features, aggregate separately before combining.
Let each message depend on the source embedding, permitted edge attributes, and a relation-specific transformation.
The weighted mean is obtained by summing weighted messages and normalizing by the sum of nonnegative weights.
```text
FORMULA [G2]: m_v,r = sum(w_uv * msg_r(u,v) for u in N_v,r) / sum(w_uv)
WHERE: r is relation type; w_uv >= 0 is dimensionless;
       msg_r(u,v) and m_v,r are vectors of the same length.
CONSTRAINT: sum(w_uv)>0; all weights, edges, and messages are available by t.
NUMERICAL EXAMPLE: messages 2 and 4, weights 1 and 3 -> (2+12)/4=3.5.
INTERPRETATION: the second message receives three times the relative weight.
```
Equal weights reduce G2 to G1 for scalar messages.
A weighted mean is not a probability unless the message semantics justify that interpretation.
A separate seed representation preserves individual features when neighbors are uninformative.
```text
FORMULA [G3]: h_v = a*x_v + b*m_v
WHERE: x_v and m_v are scalar feature units; a and b are dimensionless.
CONSTRAINT: both terms exist; coefficients are declared rather than inferred from labels.
NUMERICAL EXAMPLE: x_v=1, m_v=3, a=b=0.5 -> h_v=2 feature units.
INTERPRETATION: the illustrative representation mixes self and neighbor information equally.
```
This linear teaching rule is not a trained fraud probability.
A practical network may use learned matrices, nonlinearities, relation-specific parameters, residuals, and multiple layers.
Those choices require training data and deployment validation; none is guaranteed to improve ranking.

### 4.3 Key Special Cases / Variants

| Variant | Change | Use when | Avoid when |
|---|---|---|---|
| Unique-neighbor mean | One vote per distinct adjacent node | Repeated events should not multiply influence | Event frequency is the important signal and is omitted |
| Edge-event mean | One vote per eligible event | Repeated event activity is meaningful | Retries or duplicates are not deduplicated |
| Relation-specific mean | Separate messages by type and direction | Transfers and device use have different meaning | Relation provenance is unreliable |
| Weighted mean | Declared nonnegative relative weights | Confidence or age weighting is justified | Weight creation uses future information |
| Sampled mean | Aggregate a bounded neighbor sample | Full neighborhoods exceed the serving budget | Sampling exclusions erase rare informative relations |
| Self-only model | No neighbor messages | Graph freshness or identity quality is insufficient | Graph context has shown incremental value within constraints |

### 4.4 Limiting Behaviour and Counterexample

A mean over an empty set has no mathematical value because its denominator is zero.
A high-degree shared device may dominate a raw sum even if its users have no suspicious relationship.
A mean avoids linear growth with degree but can dilute a small suspicious neighborhood.
A single mistaken entity merge can create many false paths, regardless of how accurately aggregation is computed.
A later-discovered edge cannot be made time-safe merely by setting its event timestamp to the past.
A graph snapshot can be time-safe but still encode a downstream action derived from a future label if provenance is incomplete.
If two sampled neighbors differ greatly, one sample can give an unstable estimate even when the sample is unbiased.
Sampling with a fixed random seed improves reproducibility but does not eliminate sampling uncertainty.

**Takeaway:** Aggregation is meaningful only after the neighbor set, weighting, missingness, and timestamps are specified.

## Section 5 — Worked Numerical Example

### Illustrative decision and input ledger

An analyst must choose two of four account cases for a same-day review.
The score is a simple hand-designed teaching ranking, not an estimated probability.
At 12:00 the eligible account features are A=1, B=2, C=4, D=0 feature units.
All features below were available before the decision.
The edge ledger records endpoint pairs, event time, and availability time.
| Edge | Event time | Availability | Eligible at 12:00? |
|---|---|---|---|
| A—B | 10:00 | 10:01 | yes |
| A—C | 11:00 | 11:01 | yes |
| A—D | 11:30 | 12:05 | no: late arrival |
| B—C | 12:10 | 12:11 | no: future event |
| C—D | 09:00 | 09:01 | yes |

Use the eligible edges as undirected only for this toy feature aggregation.
This simplification does not erase direction for real payment flows.
The adjacency sets are A:{B,C}, B:{A}, C:{A,D}, and D:{C}.
All four sets are nonempty, so G1 is defined for every seed.
Compute A's mean: (2+4)/2 = 3 feature units.
Compute B's mean: 1/1 = 1 feature unit.
Compute C's mean: (1+0)/2 = 0.5 feature units.
Compute D's mean: 4/1 = 4 feature units.
Apply G3 with equal self and neighbor coefficients.
A has h=(1+3)/2=2.
B has h=(2+1)/2=1.5.
C has h=(4+0.5)/2=2.25.
D has h=(0+4)/2=2.
Sort descending by h, breaking equal scores by ascending account identifier.
The order is C, A, D, B; the two selected cases are C and A.
Tie breaking matters: without a stable rule, A and D can swap at the capacity boundary.
Assume eventual mature labels A=positive, B=negative, C=positive, D=negative.
These outcomes are introduced only for evaluation, not as input features.
The selected queue contains two positive cases out of two reviews.
Illustrative precision at two is 2/2 = 1.0, or 100%.
The population contains two positives, both reviewed.
Illustrative recall at two is 2/2 = 1.0, or 100%.
This tiny constructed example does not establish real-world quality or calibration.

### Independent checks and counterfactual

The undirected eligible ledger contains three edges.
The adjacency degree total is 2+1+2+1=6, equal to twice the edge count.
For A, equal weights make G2 reduce to G1: (2+4)/(1+1)=3.
For A, self plus neighbor units are compatible before multiplying by dimensionless coefficients.
The ranking selects exactly two distinct account cases, matching the capacity.
If the late A—D edge is incorrectly included, A's mean changes to (2+4+0)/3=2.
The resulting A representation falls from 2 to (1+2)/2=1.5.
That erroneous result is reproducible but unavailable to the real 12:00 decision.
If A's mature label were revealed only after 12:00, including it in a neighbor-label feature would likewise leak.
If the positive count were unknown because unreviewed cases lacked labels, recall would be unidentified from this queue alone.
If the review budget were one, C alone would be selected; precision is 1/1 and recall is 1/2 under the illustrative complete labels.
If the self coefficient changed, the ordering could change; the weights are not learned or validated here.

### Production interpretation

A real ledger requires an immutable event identifier, source namespace, event time, and availability time.
Real features must be recovered as of the original score decision, including correction history.
Case-level deduplication must occur before applying a case-level capacity constraint.
For a million candidates, hand sorting becomes a selection and data-access workload rather than a manual exercise.
Entity resolution, time zones, retries, and source outages can change the eligible graph.
The little ledger is a correctness oracle, not an estimate of operational fraud loss.

**Takeaway:** The same graph feature can be arithmetically correct and operationally invalid if the edge was unavailable at scoring time.

## Section 6 — Visual / Geometric Intuition

```text
Decision clock, illustrative and not to scale:
09:01            10:01            11:01       12:00       12:05
  |                |                |            |            |
 C—D available   A—B available    A—C available  SCORE      A—D available
                                                    ^
                                           eligibility boundary
```

The event A—D occurred at 11:30 but reached the serving system at 12:05.
The scoring boundary excludes it despite the earlier event time.
A later historical rebuild that includes A—D would not reproduce the online decision.

```text
Eligible graph at 12:00, illustrative and not to scale:
    B(x=2)
      |
    A(x=1) --- C(x=4) --- D(x=0)
```

The link from A to D is absent, not merely hidden in the picture.
A one-hop mean at A sees B and C, but not D.
A two-hop model can see D through C, subject to the same edge and feature time rules.
That two-hop reachability is not a claim that A and D are the same person or members of a ring.

**Takeaway:** A temporal graph is a snapshot of available evidence, not a retrospective diagram of everything later learned.

## Section 7 — Comparison / Selection Table

| Method | Preconditions | Cost/quality boundary | Best for | Avoid when | Evidence |
|---|---|---|---|---|---|
| Existing rules | Documented, time-safe rules | Simple and auditable; coverage is rule-dependent | Safety fallback and baseline | Rules use unavailable labels | DERIVED |
| Tabular model | Time-safe transaction and account features | No graph fetch; may miss relational signal | Strong non-graph comparison | The only useful evidence is relational | HEURISTIC |
| Graph feature model | Reliable, bounded graph aggregates | Interpretable features; misses some complex patterns | Incremental graph value with limited serving budget | Graph snapshots cannot be reconstructed | HEURISTIC |
| Sampled GNN | Stable time-safe neighborhoods and model serving | Can learn relational representations; sampling adds variance | Adequate graph infrastructure and demonstrated incremental value | Graph freshness or latency is unacceptable | HEURISTIC |
| Component/motif rules | Typed, bounded graph and graph-aware review | Easy to explain patterns; shared infrastructure causes false positives | Investigator leads | Connectivity is treated as proof | DERIVED |

First disqualify any method that violates information availability, permissions, or required latency.
Next compare the rules and tabular baselines at identical review capacity and outcome maturity.
Add graph aggregates when a point-in-time graph is reliable and adds value over those baselines.
Consider a GNN only when its incremental benefit justifies operational cost and investigation burden.
Do not claim a method is best without a comparable time-split evaluation and confidence intervals.
Benchmark plan: record candidate cohort, policy, feature cutoff, model version, hardware, load, and uncertainty.
No workload-specific speed or accuracy results have been measured for this sheet.

**Takeaway:** Reject unsafe options first, then demand incremental value against the simplest credible baseline.

## Section 8 — Failure Modes & Edge Cases

Every diagnostic below is HYPOTHETICAL, not a report of an observed incident.
The first three fields describe a symptom, a competing explanation, and a discriminating test.
A negative diagnostic does not prove that all other failure modes are absent.
A fix is complete only when a replay or a controlled production observation confirms the intended invariant.

### F-01: Late edge ingestion

- **Trigger:** Offline graph contains a pre-decision transaction missing online.
- **Competing hypotheses:** The event arrived after scoring, or the online fetch dropped an already available event.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Compare source event and serving-availability timestamps with the request trace.
- **Expected signal:** A late availability timestamp supports temporal leakage; a missing successful write supports serving loss.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Enforce both event and availability cutoffs; separately repair dropped writes.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Replay the frozen request from an append-only availability log.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-02: Post-outcome relationship

- **Trigger:** A strong edge appears only in positives.
- **Competing hypotheses:** An investigator added it after a confirmed outcome, or it truly preceded the decision.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Trace the edge creator and creation timestamp against adjudication time.
- **Expected signal:** Edges created by the case system after adjudication are downstream artifacts.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Exclude downstream artifacts from predecision features and preserve audit provenance.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Show that removing those edges changes offline features to the online values.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-03: Future label in neighborhood

- **Trigger:** Neighbor fraud rate predicts almost perfectly.
- **Competing hypotheses:** The feature uses labels learned after scoring, or it captures genuinely early known alerts.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Inspect each label availability timestamp used in the aggregate.
- **Expected signal:** Any label availability after the seed decision violates the feature contract.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Recompute from labels available before each seed decision.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Compare a historical feature replay to the online feature log.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-04: Self-label contamination

- **Trigger:** Training performance exceeds validation sharply.
- **Competing hypotheses:** The seed label contributed to its own neighborhood statistic, or the model is overfit.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Remove the seed label and trace all aggregate paths to the seed.
- **Expected signal:** A direct or cyclic path returning the seed label demonstrates contamination.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Use out-of-fold, self-excluded label aggregates with temporal eligibility.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Unit-test a tiny cycle with unique labels to verify exclusion.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-05: Entity overmerge

- **Trigger:** One identity suddenly has thousands of peers.
- **Competing hypotheses:** An identifier collision merged unrelated people, or a legitimate enterprise operates many accounts.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Inspect match evidence and identifier cardinality before and after the merge.
- **Expected signal:** Weak common attributes across dissimilar records support an overmerge.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Split the identity using reversible provenance and conservative matching.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Replay prior scores and quantify customer and alert impact.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-06: Entity undersplit

- **Trigger:** Repeated fraud actors remain disconnected.
- **Competing hypotheses:** One actor has fragmented identifiers, or the apparent similarity reflects a shared service.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Inspect validated linkage evidence and downstream investigation outcomes.
- **Expected signal:** Repeated strong independent identity evidence supports missed linkage.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Add a reviewed, time-versioned linkage rather than a global fuzzy merge.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Check link precision and recall on an independently adjudicated sample.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-07: Shared-device supernode

- **Trigger:** Thousands of accounts share one device identifier.
- **Competing hypotheses:** A public kiosk or default device token created a hub, or the device is genuinely controlled by a ring.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Inspect token generation, diversity, site context, and the time distribution.
- **Expected signal:** A default token across many unrelated sessions suggests instrumentation or shared infrastructure.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Represent device reliability and relation type; cap or sample expansion.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Check that hub treatment does not erase confirmed small-ring signals.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-08: Transfer direction reversal

- **Trigger:** Money-flow patterns appear backwards.
- **Competing hypotheses:** The ingestion schema inverted sender and receiver, or an analyst interpreted a derived reverse relation as a payment.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Compare source transaction IDs and endpoint roles to stored directed edges.
- **Expected signal:** Systematic reversal against source records identifies a mapping error.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Correct the mapping and distinguish computational reverse edges from transactions.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Replay a known sender-receiver ledger and compare flow conservation.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-09: Duplicate events

- **Trigger:** Degree and transaction counts jump after a retry.
- **Competing hypotheses:** At-least-once delivery duplicated writes, or the source emitted legitimate repeated payments.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Group by immutable source event ID and compare source payloads.
- **Expected signal:** Identical source IDs with repeated writes indicate ingestion duplication.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Make edge ingestion idempotent while preserving distinct source events.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Replay the same batch twice and compare graph state.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-10: Correction overwritten

- **Trigger:** Historical feature replay differs from old serving logs.
- **Competing hypotheses:** A source correction overwrote the earlier value, or the original online result was stale.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Compare the version history and correction arrival time to the decision timestamp.
- **Expected signal:** A correction recorded after the decision explains retrospective disagreement.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Store version history and select the version available at the decision.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Replay both before and after correction availability boundaries.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-11: Clock skew

- **Trigger:** Events appear to arrive before they happened.
- **Competing hypotheses:** Source and ingestion clocks differ, or event-time semantics are misunderstood.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Compare clock synchronization telemetry and representative source payloads.
- **Expected signal:** A common offset across a source suggests clock skew.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Fix clock discipline and define conservative uncertainty handling.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Check ordering on monitored test events after the correction.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-12: Timezone conversion

- **Trigger:** Daily graph changes at an unexpected hour.
- **Competing hypotheses:** A local timestamp was parsed as UTC, or the reporting window uses a different zone.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Trace the raw timestamp, zone, daylight-saving rule, and normalized value.
- **Expected signal:** A consistent zone offset or DST discontinuity supports a parsing defect.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Normalize explicit zone-aware timestamps and preserve the original.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Test timestamps around both daylight-saving transitions.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-13: Mutable device token

- **Trigger:** One device appears to switch users impossibly often.
- **Competing hypotheses:** A recycled token represents multiple devices, or a shared device really has many users.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Compare token lifetime, app version, and platform generation policy.
- **Expected signal:** Known recycling or default token behavior weakens identity inference.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Version token namespaces and treat device use as a weak edge.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Validate with independently reviewed device ownership evidence.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-14: Train-test component overlap

- **Trigger:** Validation precision looks unusually optimistic.
- **Competing hypotheses:** Near-duplicate entities span the split, or legitimate historical graph structure is usable in deployment.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Compare entity overlap and connected components across train and test.
- **Expected signal:** Duplicated customer records or outcome-derived edges across folds indicate contamination.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Use temporal holdout and report entity-disjoint or component-disjoint stress tests.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Explain which known historical neighbors remain permissible at deployment.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-15: Full-graph normalization

- **Trigger:** Test nodes influence training despite a time split.
- **Competing hypotheses:** Normalization statistics were computed on the full graph, or a declared transductive setup allowed them.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Trace when degree and normalization statistics were calculated.
- **Expected signal:** Future topology in training-time statistics violates an inductive historical simulation.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Fit transformations on permitted training snapshots and replay evaluation snapshots.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Check that altering future edges cannot alter earlier training features.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-16: Unreviewed cases labeled negative

- **Trigger:** Precision seems stable while recall collapses.
- **Competing hypotheses:** Unreviewed positives were assigned negative, or the underlying fraud mix truly changed.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Compare label provenance and the fraction of unreviewed candidates by score bucket.
- **Expected signal:** Large policy-dependent unknown buckets indicate selective labels.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Keep unknown outcomes distinct and collect randomized audit labels.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Report audited estimates with inclusion probabilities and uncertainty.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-17: Delayed chargebacks

- **Trigger:** Recent test cohorts appear cleaner than older ones.
- **Competing hypotheses:** Outcomes have not matured, or controls reduced fraud.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Plot outcome discovery against decision age by cohort.
- **Expected signal:** An increasing cumulative outcome curve indicates delayed feedback.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Restrict primary comparison to mature cohorts and report censoring.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Recompute older cohorts at matched observation ages.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-18: Changed review policy

- **Trigger:** Model precision rises after a threshold change.
- **Competing hypotheses:** Reviewers selected a different case mix, or model ranking actually improved.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Compare models on the same pre-policy candidate cohort and fixed capacity.
- **Expected signal:** A lift that disappears under a shared candidate pool suggests selection effects.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Separate ranking changes from review-policy changes.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Use a controlled shadow or randomized comparison where ethically permitted.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-19: Top-k duplicates

- **Trigger:** A queue is full but few distinct networks are investigated.
- **Competing hypotheses:** Many account alerts refer to one case, or the incident genuinely requires multiple independent reviews.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Cluster alerts by explicit case identity and inspect the queue before and after deduplication.
- **Expected signal:** Many repeated case identifiers demonstrate capacity being consumed by duplicates.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Rank or diversify at the investigator case grain with traceable member alerts.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Measure distinct cases reviewed and fraud found under the same labor budget.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-20: Case merge after review

- **Trigger:** Case metrics change retroactively.
- **Competing hypotheses:** Multiple alerts were merged into one investigation, or outcomes were corrected.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Track case-merge versions and outcome attribution at each reporting cutoff.
- **Expected signal:** A change in case identity without score changes points to metric grain.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Freeze the evaluation unit or version case memberships for reproducible reports.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Reconcile counts of alerts, cases, and outcomes at each snapshot.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-21: High-degree traversal timeout

- **Trigger:** Tail latency spikes on a small fraction of seeds.
- **Competing hypotheses:** One hub causes fanout explosion, or the graph store has an unrelated I/O bottleneck.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Correlate examined edges and hub degrees with per-request timing.
- **Expected signal:** Latency tracking examined-edge count supports expansion as the cause.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Bound per-relation fanout and total work; return explicit incomplete status.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Load-test degree-stratified traffic and compare tail latency with quality.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-22: Sampling bias

- **Trigger:** A bounded neighborhood misses a known fraud cluster.
- **Competing hypotheses:** Sampling systematically favors recent or frequent edges, or the cluster is absent at decision time.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Record sampled and eligible neighbor IDs and selection probabilities.
- **Expected signal:** A rare relation consistently excluded by policy indicates sampling bias.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Stratify samples by relation or reserve budget for rare evidence.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Measure recall and variability against full neighborhoods on a feasible audit set.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-23: Sampling instability

- **Trigger:** The same account gets different scores on repeated requests.
- **Competing hypotheses:** Random sampling changes neighbors, or features changed between requests.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Fix snapshot and model version, then repeat with controlled sampling seeds.
- **Expected signal:** Variance under a fixed snapshot indicates sampling-induced instability.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Use reproducible sampling or average within an explicit latency budget.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Measure ranking changes near the investigation threshold.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-24: Stale graph cache

- **Trigger:** Online predictions use old relationships.
- **Competing hypotheses:** The cache missed invalidation, or ingestion was legitimately delayed.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Compare source availability, cache version, and request snapshot timestamp.
- **Expected signal:** A newer committed version absent from cache implicates cache propagation.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Repair invalidation and expose graph freshness to the serving policy.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Inject a known edge and trace its visibility end to end.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-25: Online-offline feature skew

- **Trigger:** Training and serving values disagree for the same example.
- **Competing hypotheses:** Different aggregation code or a mismatched as-of snapshot is in use.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Replay the same entity, cutoff, source versions, and model version in both paths.
- **Expected signal:** Differences with identical inputs isolate transform or serialization divergence.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Unify feature semantics and preserve an offline-online parity fixture.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Compare exact feature values on edge cases including empty neighborhoods.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-26: Missing neighborhood default

- **Trigger:** New accounts score strangely low.
- **Competing hypotheses:** An empty neighborhood was silently mapped to zero, or genuinely low-risk evidence is present.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Inspect missing flags separately from numeric aggregate values.
- **Expected signal:** A spike in exact default values for new accounts suggests missingness confusion.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Add a missing indicator and a documented fallback policy.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Evaluate cold-start slices and outage scenarios independently.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-27: Zero-weight aggregation

- **Trigger:** A weighted mean yields NaN or infinity.
- **Competing hypotheses:** All eligible weights are zero, or numeric precision overflowed.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Log the denominator and individual finite weights for failed requests.
- **Expected signal:** A zero denominator confirms an undefined weighted mean.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Use a missing-neighborhood representation instead of dividing by zero.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Test empty, all-zero, extreme, and mixed weights.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-28: Graph drift

- **Trigger:** Model performance declines on a stable volume.
- **Competing hypotheses:** Relation composition changed, or labels matured differently across cohorts.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Compare degree, node types, edge types, freshness, and outcome maturity by cohort.
- **Expected signal:** Structural shift preceding performance decline motivates a graph-drift investigation.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Reassess relation construction and retrain only with valid mature data.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Compare a stable tabular baseline and audited cohort slices.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-29: Proxy discrimination

- **Trigger:** One community receives disproportionate alerts.
- **Competing hypotheses:** A weak shared-location feature proxies protected status, or underlying exposure differs.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Audit score, review, and confirmed-outcome rates by legally appropriate group and context.
- **Expected signal:** Large disparities require investigation but do not alone prove the cause.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Engage fairness, legal, and domain owners; restrict unsupported proxy use.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Monitor appeals and subgroup errors after any policy change.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-30: Investigator overload

- **Trigger:** Backlog rises despite high model precision.
- **Competing hypotheses:** Arrivals exceed review throughput, or case complexity increased.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Measure incoming distinct cases, completed cases, and handling time distributions.
- **Expected signal:** Sustained arrival above completion explains persistent backlog growth.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Reduce workload, increase capacity, or revise prioritization with risk ownership.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Track aging, throughput, and unresolved high-risk cases.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-31: Leak through blocking action

- **Trigger:** Downstream account closure is a powerful feature.
- **Competing hypotheses:** Closure was triggered by the same risk label, or a prior independent closure is legitimate evidence.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Trace action trigger, execution time, and causal dependency.
- **Expected signal:** An action triggered by the outcome being predicted is downstream leakage.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Exclude downstream actions and keep independent prior actions only if time-safe.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Use a causal provenance review and retrospective feature replay.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-32: Outcome leakage through component

- **Trigger:** A component suddenly becomes highly predictive.
- **Competing hypotheses:** A later network investigation created links among confirmed accounts, or earlier connections were already present.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Compare edge creation times and provenance with the first investigation timestamp.
- **Expected signal:** A wave of investigator-created edges after confirmation indicates leakage.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Separate observed transactional relations from later investigative annotations.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Demonstrate identical predictions when future annotations are removed.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-33: Incorrect negative horizon

- **Trigger:** Many negatives later become positive.
- **Competing hypotheses:** The fixed label horizon is too short, or the target definition changed.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Compare late positive discovery rates by cohort age and target policy.
- **Expected signal:** Substantial late discovery indicates immature negative labeling.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Lengthen the horizon or explicitly model unresolved outcomes.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Reassess performance after the chosen maturity period.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-34: Fraud ring double counting

- **Trigger:** Reported detected loss exceeds plausible account exposure.
- **Competing hypotheses:** Losses were summed across overlapping ring members, or payments really have independent losses.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Join losses to unique transaction and recovery IDs.
- **Expected signal:** The same loss ID attributed to multiple accounts confirms double counting.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Deduplicate at the loss-event grain and account for recoveries.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Reconcile the final loss ledger with finance totals.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-35: Unstable threshold

- **Trigger:** Alert volume oscillates around the budget.
- **Competing hypotheses:** Tied scores or bursty candidate arrivals dominate the boundary.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Inspect score ties, candidate arrival profile, and threshold update schedule.
- **Expected signal:** Large tie blocks near the cutoff explain unstable selected sets.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Use deterministic tie breaks and a documented batch or pacing policy.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Measure daily capacity utilization and rank churn.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-36: Unavailable graph service

- **Trigger:** Scoring failures increase during an upstream incident.
- **Competing hypotheses:** Graph storage is unavailable, or the model server itself is failing.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Compare graph-fetch errors and latency with model process health.
- **Expected signal:** Graph-specific errors with healthy inference identify the dependency.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Use an approved tabular or rules fallback with missing-graph status.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Exercise recovery and monitor decisions by serving path.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-37: Model-version mismatch

- **Trigger:** Replayed scores do not match historical scores.
- **Competing hypotheses:** The model artifact differs, or the feature snapshot differs.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Pin artifact checksum and feature snapshot separately in a controlled replay.
- **Expected signal:** A mismatch only after model substitution identifies artifact drift.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Record immutable model, feature, and graph versions for each score.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Reproduce a stored request with its original artifacts.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-38: Hidden fanout cap

- **Trigger:** Offline graph model outperforms online.
- **Competing hypotheses:** Serving silently truncates neighborhoods, or its graph freshness differs.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Compare fetched neighbor IDs, fanout limits, and snapshot versions.
- **Expected signal:** Systematic missing neighbors at exactly the cap indicate truncation.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Train and evaluate with serving-equivalent sampling and report incompleteness.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Measure quality and latency over the actual degree distribution.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-39: Explainer path includes future edge

- **Trigger:** An analyst sees a plausible but historically impossible rationale.
- **Competing hypotheses:** The explanation used the current graph rather than the scoring snapshot.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Compare every explanation edge availability with the decision timestamp.
- **Expected signal:** Any edge unavailable at scoring invalidates the purported historical explanation.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Generate explanation paths from the exact versioned scoring snapshot.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Replay explanations after adding a later edge and confirm they do not change.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-40: Misread attention weights

- **Trigger:** Reviewers treat a large attention weight as proof of causation.
- **Competing hypotheses:** Weights were presented as explanations, or reviewers overinterpreted a ranking cue.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Check interface language and test whether perturbing the edge changes the prediction.
- **Expected signal:** A high weight without a stable score effect shows why the weight alone is insufficient.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Present provenance and model sensitivity as leads, not legal evidence.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Validate explanations with investigators and preserve appeal channels.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-41: Privacy-boundary traversal

- **Trigger:** A graph query crosses a tenant boundary.
- **Competing hypotheses:** An authorization filter was omitted, or entity identifiers collided across tenants.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Trace tenant-scoped keys and graph-fetch authorization decisions.
- **Expected signal:** A fetched edge outside the permitted tenant confirms an access-control defect.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Enforce authorization at data access and isolate identifier namespaces.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Test adversarial cross-tenant paths and audit access logs.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-42: Deletion not propagated

- **Trigger:** A deleted identifier remains in a derived embedding.
- **Competing hypotheses:** The deletion reached the source but not caches or training datasets.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Trace deletion requests through graph, cache, features, embeddings, and lineage.
- **Expected signal:** A retained derived artifact after its deletion deadline indicates propagation failure.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Apply the approved deletion and model-remediation policy across derivatives.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Verify removal with lineage-based checks and independent access audit.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-43: Contradictory graph labels

- **Trigger:** A node has both confirmed fraud and confirmed legitimate labels.
- **Competing hypotheses:** Labels refer to different events or times, or adjudications conflict.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Inspect label target grain, effective time, and adjudication history.
- **Expected signal:** Different transaction targets can coexist without a contradiction.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Represent target and version explicitly; route real disputes to review.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Test outcome joins on multiple transactions per account.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-44: Component explosion

- **Trigger:** Most of the population joins one graph component.
- **Competing hypotheses:** A shared payment processor or device hub connects many benign entities, or linkage rules overmerged accounts.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Break down edges responsible for component growth by relation type.
- **Expected signal:** Removing a generic infrastructure relation in analysis fragments the component.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Define relation-aware components or bounded motifs rather than declaring one giant ring.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Audit retained and lost links against known cases.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-45: Cold-start distribution

- **Trigger:** New entities are underrepresented in training.
- **Competing hypotheses:** Graph methods require historical edges, or the candidate population changed.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Report degree-zero rates and performance by entity age.
- **Expected signal:** A concentrated decline for new nodes indicates cold-start weakness.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Combine self features and a separate cold-start policy.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Hold out new entities by time and monitor their outcomes.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-46: Circular review evidence

- **Trigger:** Model features include reviewer suspicion from a prior model.
- **Competing hypotheses:** The prior decision reflects a model score, or it is independent investigator evidence.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Trace reviewer annotation provenance and the information presented to reviewers.
- **Expected signal:** Annotations made after viewing the model score can encode feedback.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Separate independently observed facts from model-generated suggestions.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Use blinded or randomized review when feasible and ethically approved.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-47: Outcome attribution error

- **Trigger:** A ring investigation credits all linked accounts as confirmed fraud.
- **Competing hypotheses:** The case outcome was incorrectly broadcast to members, or each member was separately adjudicated.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Compare entity-level evidence and adjudication records to case-level status.
- **Expected signal:** Only case-level evidence with no member findings does not establish member guilt.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Keep case and entity labels distinct and retain unknown member outcomes.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Audit a sample of case-to-member label assignments.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-48: Threshold calibration shift

- **Trigger:** A score labeled 0.8 no longer corresponds to observed frequency.
- **Competing hypotheses:** Base rates changed, or label ascertainment changed with the review policy.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Compare calibration on mature randomly audited cases across periods.
- **Expected signal:** A shift on an unbiased audit set supports changed calibration.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Recalibrate on a representative mature sample and document uncertainty.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Monitor reliability by cohort and action policy.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-49: Graph build partial failure

- **Trigger:** A scheduled snapshot has unexpectedly low degree.
- **Competing hypotheses:** A source partition failed, or actual activity dropped.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Compare source offsets, partition completion markers, and edge counts by source.
- **Expected signal:** Missing completion markers identify an incomplete build.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Mark the snapshot incomplete and use a documented fallback.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Rebuild idempotently and reconcile source counts before activation.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-50: Community detection instability

- **Trigger:** Small data changes substantially alter assigned rings.
- **Competing hypotheses:** The clustering algorithm is unstable, or a high-impact bridge is real.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Vary seed and perturb uncertain edges while holding the snapshot fixed.
- **Expected signal:** Large membership changes across seeds indicate algorithmic instability.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Report uncertainty and avoid treating a cluster assignment as a finding.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Review stability and investigator utility before operational use.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-51: Loss-of-label signal

- **Trigger:** Neighbor label feature suddenly becomes mostly missing.
- **Competing hypotheses:** The label pipeline is delayed, or fraud has decreased.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Compare label stream lag, availability watermark, and case outcomes.
- **Expected signal:** A stalled watermark with stable event volume supports pipeline delay.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Use a missingness-aware fallback and repair the label stream.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Check that recovered labels do not retroactively enter old predictions.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-52: Data poisoning

- **Trigger:** Many accounts intentionally share a benign high-degree identifier.
- **Competing hypotheses:** Adversaries are manipulating a graph feature, or the identifier is a legitimate public service.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Inspect sudden coordination, relation provenance, and independently verified activity.
- **Expected signal:** A coordinated burst with weak evidence suggests manipulation but is not proof.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Limit weak-edge influence and add analyst review for coordinated changes.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Test adversarial edge insertion against both ranking and false positives.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-53: Score gaming

- **Trigger:** Fraudsters split transfers across new accounts to evade thresholds.
- **Competing hypotheses:** The rule is vulnerable to fragmentation, or behavior changed innocently.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Compare aggregate flow patterns and account creation sequences under time-safe windows.
- **Expected signal:** A shift from large transfers to coordinated small transfers is a lead, not proof.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Use bounded multi-entity features alongside policy review.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Evaluate on later cohorts and monitor unintended impact on legitimate users.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-54: Wrong capacity unit

- **Trigger:** A top-100 account ranking exceeds daily investigator capacity.
- **Competing hypotheses:** Accounts were equated with cases, or cases have variable review effort.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Measure distinct investigations and handling minutes for the selected accounts.
- **Expected signal:** More than the available handling minutes demonstrates a capacity mismatch.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Rank case bundles with effort estimates and retain an escalation path.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Compare actual labor minutes and outcomes against the budget.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-55: Score censoring by intervention

- **Trigger:** Observed fraud loss falls for the high-score group.
- **Competing hypotheses:** Intervention prevented loss, or high scores were false positives.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Compare treatment and outcomes under an authorized randomized or quasi-experimental design.
- **Expected signal:** A policy-driven gap in observation prevents naive loss comparisons.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Estimate intervention effects with a defensible design and legal oversight.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Report policy, uncertainty, and any remaining confounding.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-56: Feature decay bug

- **Trigger:** Older relationships receive more weight than recent relationships.
- **Competing hypotheses:** The decay sign is reversed, or event ages were computed with the wrong timezone.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Calculate weights for two known ages with fixed decision time.
- **Expected signal:** Older weight exceeding newer weight under a decreasing rule indicates a defect.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Correct age calculation and verify nonnegative eligible ages.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Test equal ages, future events, and very old events.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-57: Cross-relation normalization

- **Trigger:** A generic device hub overwhelms rare bank-transfer links.
- **Competing hypotheses:** All relation types share one denominator, or message scales differ.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Inspect per-relation counts and contribution magnitudes before combination.
- **Expected signal:** Dominance by one relation despite small per-edge signals supports scale imbalance.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Normalize by relation and validate relation-specific weights.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Run ablations and inspect both quality and subgroup false positives.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-58: Evaluation cohort mismatch

- **Trigger:** Graph model appears better than baseline.
- **Competing hypotheses:** Different eligible candidates or label maturity were used, or the improvement is real.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Join candidate IDs and decision windows across model evaluations.
- **Expected signal:** Nonidentical candidate sets prevent a clean paired comparison.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Re-evaluate both models on the same time-safe cohort and capacity.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Report paired uncertainty and cohort exclusions.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-59: Unbounded two-hop expansion

- **Trigger:** Memory rises sharply after adding one GNN layer.
- **Competing hypotheses:** Fanout multiplies across layers, or tensors are retained unnecessarily.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Trace distinct sampled nodes, sampled edges, and activation memory by layer.
- **Expected signal:** Growth matching fanout expansion supports graph expansion as the dominant mechanism.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Reduce fanout or layers and test the lost relational coverage.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Measure memory and quality at each workload percentile.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### F-60: Stale intervention policy

- **Trigger:** A historically good model generates costly low-value alerts.
- **Competing hypotheses:** Fraud tactics changed, or the old action policy no longer matches capacity.
- **Diagnostic input:** Preserve the original decision time, model version, graph snapshot, and affected case IDs.
- **Discriminating check:** Compare case mix, review effort, and mature outcomes against historical periods.
- **Expected signal:** Quality decline restricted to a changed action path suggests policy mismatch.
- **Interpretation:** This signal supports the stated mechanism but is not a complete causal proof.
- **If absent:** Investigate the competing explanation and check the integrity of the diagnostic logs.
- **Bounded correction:** Revisit decision thresholds and graph features with policy owners.
- **Safety boundary:** Do not convert an association or model output into an automatic finding of fraud.
- **Verification:** Monitor both fraud outcomes and customer harm after change.
- **Operational owner:** The data or model owner coordinates with the investigation policy owner.
- **Evidence status:** HYPOTHETICAL teaching scenario; no incident is asserted.

### Evaluation design: a concrete review protocol

1. Fix the intended decision: rank cases for review at the start of each investigation shift.
2. Fix the eligibility population before scores are examined.
3. Record which accounts and transactions are legally and operationally eligible.
4. Exclude records unavailable at the decision cutoff, and record the number excluded by reason.
5. Freeze the candidate-generation logic shared by baseline and challenger.
6. Freeze case construction rules, including how many alerts can map to one investigation.
7. Specify a deterministic tie-break rule before selecting the evaluation queue.
8. Define the comparison budget in case-minutes or an agreed case-count approximation.
9. Preserve the baseline's original serving features rather than adding challenger-only evidence.
10. Reconstruct each candidate's graph from records available at its own decision time.
11. Reconstruct feature versions using the same availability rule.
12. Separate graph ingestion time from source event time in the replay ledger.
13. Record entity-resolution version and any provisional links used for the score.
14. Do not use later identity corrections to rewrite earlier model inputs.
15. If such corrections change a label, version the label independently of the features.
16. Fit scaling, imputation, feature selection, and model parameters only on the training period.
17. Make early-stopping and hyperparameter decisions using a validation period.
18. Keep the final later test period untouched until selection is complete.
19. Fit neighbor-label aggregates using only allowed labels available at each decision.
20. Exclude the seed's own target and all paths that return it to the feature.
21. Declare whether previously seen accounts can appear in the test graph.
22. Treat that recurring-account setting as distinct from an unseen-account stress test.
23. Preserve historical context that would genuinely be available for an unseen account.
24. Test the ability to represent a new node without a prelearned node identifier.
25. Evaluate the same candidate set under both models before choosing top cases.
26. Deduplicate candidate alerts into the same case unit for both queues.
27. Report queue overlap so investigators can see how the proposed model changes workload.
28. Report the number of distinct cases and distinct accounts in each queue.
29. Record actual or defensible estimated review effort for each selected case.
30. State the time horizon for outcomes and the date through which labels are available.
31. Separate confirmed positives, confirmed negatives, and unresolved cases.
32. Report the fraction unresolved in each score band and each reviewed cohort.
33. Compute top-k precision only when its label policy and denominator are explicit.
34. Avoid replacing unresolved selected cases with negative outcomes.
35. If outcomes are incomplete, publish the known-positive fraction as a lower bound only under the stated label assumptions.
36. Recognize that even a valid lower bound does not identify the true precision.
37. Calculate recall only when the total-positive denominator is observed or estimated using a defensible design.
38. Collect a randomized sample of otherwise unreviewed cases if policy permits.
39. Record each sampled case's probability of selection.
40. Record nonresponse and investigation completion separately from sampling.
41. Use a sampling-aware estimate only if its selection and response assumptions are justified.
42. Escalate unsupported positivity assumptions: cases with zero sampling probability cannot be recovered from that sample alone.
43. Compare results on matched calendar windows so changing base rates do not masquerade as lift.
44. Report the precision-recall tradeoff across several plausible review budgets.
45. Do not select a budget after inspecting test labels without accounting for that selection.
46. Report absolute differences as well as ratios between challenger and baseline.
47. Attach uncertainty intervals that reflect dependence among cases where feasible.
48. Group overlapping accounts by investigation or network when choosing resampling units.
49. Explain that grouping choices affect uncertainty and can themselves be uncertain.
50. Report time-slice variation rather than relying on one aggregate number.
51. Compare new entities and established entities separately.
52. Compare low-degree, medium-degree, and hub-connected cases using declared bins.
53. Compare relation types without claiming that an ablation proves a causal mechanism.
54. Inspect slices with high missing-neighborhood rates or stale graph snapshots.
55. Audit whether false positives concentrate around common benign infrastructure.
56. Examine appeal outcomes and erroneous actions under approved privacy protections.
57. Measure investigator time spent gathering evidence, not just initial queue size.
58. Record whether review quality changes when analysts see graph explanations.
59. Distinguish action prevented from outcome observed when interventions suppress later loss.
60. Do not estimate prevented fraud by simply multiplying model scores by transaction value.
61. Where permissible, use a controlled policy comparison for incremental impact.
62. Include customer harm, case aging, and operational availability as guardrails.
63. Measure graph-store fetch latency and model inference latency on representative traffic.
64. Measure end-to-end latency directly instead of summing marginal percentiles.
65. Capture the proportion of scores served by fallback logic.
66. Compare fallback performance on the specific traffic affected by the fallback.
67. Predefine a stop condition for stale data, cross-tenant access, or capacity overload.
68. Preserve an audit record tying the score to model and graph versions.
69. Archive the evaluation dataset manifest without implying indefinite retention is allowed.
70. Obtain legal, security, and investigation-owner approval before any action-changing rollout.

### Worked outcome ascertainment example

All values in this example are ILLUSTRATIVE and describe a complete hypothetical evaluation.
A daily candidate pool contains 100 deduplicated cases.
An independent mature outcome source identifies 10 positive cases across the entire pool.
The baseline sends 20 cases for review and finds four positives among them.
The graph challenger sends a different set of 20 cases and finds six positives.
Assume that both queues use the same candidate pool, decision cutoff, and outcome definitions.
Assume that each case consumes equal review effort for this simplified comparison.
The baseline precision at 20 is four positives divided by 20 reviews, equal to 0.20.
The challenger precision at 20 is six positives divided by 20 reviews, equal to 0.30.
The absolute difference is 0.10, or ten percentage points.
The relative increase over the baseline is 0.10 divided by 0.20, equal to 0.50.
That relative quantity is 50%, conditional on a nonzero baseline precision.
The baseline recall is four detected positives divided by ten total positives, equal to 0.40.
The challenger recall is six detected positives divided by ten total positives, equal to 0.60.
The independently known denominator is essential to this recall calculation.
The two queues can overlap, so their uncertainty is not determined by these marginal totals alone.
No confidence interval can be inferred from the table without an appropriate statistical design.
Two additional confirmed positives are found in this constructed cohort, not necessarily two cases of prevented loss.
If two challenger outcomes are unresolved, the previous 0.30 precision claim is unsupported.
If those unresolved cases are distinct from six confirmed positives, the confirmed-positive share remains 6/20.
Their final outcomes could increase the number of positives among the selected cases.
If labels for unreviewed cases are unavailable, the total of ten positives is not established.
Consequently the challenger recall of 0.60 cannot be recovered from its selected queue alone.
If the graph queue takes 400 minutes and the baseline queue takes 240 minutes, equal case count is not equal labor.
The comparison must then use a shared labor budget or explicitly value the additional effort.
Those review-minute values are also illustrative and do not imply a real operational measurement.

### Point-in-time algorithm trace and invariant

Input is a candidate seed, decision timestamp, source ledger, identity version history, and allowed relation policy.
Output is a versioned feature vector and a completeness status for that single decision.
First, determine the identity mapping that was available when the score was requested.
Second, fetch relationship records whose event and availability timestamps meet the decision contract.
Third, select feature versions available by the same cutoff.
Fourth, discard disallowed relation types and inaccessible tenants.
Fifth, deduplicate using the appropriate event or neighbor identity for each feature.
Sixth, apply a declared and bounded neighbor sampling rule, if sampling is needed.
Seventh, compute relation-specific aggregates and explicit missingness indicators.
Eighth, persist snapshot identifiers, model version, and sampled-neighbor provenance.
At every stage, a retained record must satisfy the cutoff and access rules.
Each stage filters or transforms the retained records without introducing a later record.
By induction over the finite stages, the output contains no direct contribution from an excluded later record.
This invariant does not rule out a permitted record that was itself derived from an impermissible outcome.
Provenance analysis is therefore required in addition to timestamp filtering.
Termination requires finite ledger pages, explicit maximum traversal work, or both.
When a work limit is hit, the output must identify partial evidence rather than pretending to be complete.
On a transient graph failure, an approved fallback can score from permitted self features.
Recovery should reproduce the original decision snapshot rather than silently rescore with future edges.
Any later reassessment must be labeled as a new decision at a new timestamp.

**Takeaway:** Diagnose the observable mechanism before applying a fix; preserve time, provenance, and investigation context.

## Section 9 — Complexity & Cost Analysis

### 9a. Algorithmic complexity

Let V denote the number of nodes and E the number of stored edge records in a snapshot.
Let d denote representation width, L the number of message-passing layers, and B the seed batch size.
Let f_l be the maximum sampled fanout per seed-side node at layer l.
Let S be the number of sampled edge occurrences actually processed in a batch.
An adjacency-list representation requires storage proportional to V+E, excluding feature payloads and indexes.
A full scan to construct adjacency from an already materialized edge list takes O(V+E) work under ordinary constant-time indexing assumptions.
External sorting, distributed shuffles, identity joins, and storage I/O can dominate that simplified build bound.
A single layer with scalar messages must inspect O(S) sampled edge occurrences.
With d-dimensional elementwise messages, aggregation work is O(Sd), excluding transformations.
A dense d-by-d transformation per node can add O(V_sampled d²) work for the sampled nodes.
The actual implementation may reuse transformations, fuse operators, or incur extra communication.
Full-neighborhood processing makes S depend on the degree distribution and may be much larger than the number of seeds.
Under fixed fanouts, a naive upper bound on one-hop sampled occurrences is B times f_1.
The next layer can add up to B times f_1 times f_2 edge occurrences before deduplication.
Repeated neighbors may reduce the number of distinct nodes without reducing every edge operation.
Sampling with replacement can repeat neighbors and changes the effective weighting.
Sampling without replacement has a different variance and cannot exceed the degree of the node.
The sampled subgraph is an execution artifact, not necessarily the complete evidence available to an investigator.
Layer activation memory grows with sampled node count, width, dtype, and saved training intermediates.
Training requires backward-pass state and optimizer storage that an inference-only estimate excludes.
An index for historical graph versions adds storage beyond one current adjacency list.
Recovery cost includes rebuilding snapshots, verifying lineage, and warming caches.
Update cost depends on indexing, transaction semantics, and whether embeddings are recomputed or invalidated.
No asymptotic expression alone predicts network latency, p99 response time, or cloud expense.

### 9b. Illustrative sizing

Assume a batch of B=100 seed accounts, with at most 10 sampled first-hop neighbors per seed.
Assume each first-hop node contributes at most five sampled second-hop neighbors.
Assume sampling is without replacement within each local neighborhood, but do not assume global deduplication.
These are illustrative limits, not measured production fanouts.
The maximum number of first-hop edge occurrences is 100 × 10 = 1,000.
The maximum number of second-hop edge occurrences is 1,000 × 5 = 5,000.
The maximum sampled edge-occurrence total for the two expansions is 6,000.
An upper bound on node occurrences including seeds is 100+1,000+5,000 = 6,100.
Repeated nodes can reduce the number of distinct nodes, but the bound does not assume that benefit.
Suppose each node occurrence has a 64-component float32 representation.
Each component occupies four bytes under the stated representation assumption.
One raw representation per occurrence requires 6,100 × 64 × 4 = 1,561,600 bytes.
That is 1,561,600 / 1,048,576 = 1.4892578125 MiB, approximately 1.49 MiB.
An independent multiplication check is 6,100 × 256 = 1,561,600 bytes.
This is raw representation payload only, not peak process or device memory.
It omits edge features, indices, graph storage, batching buffers, activations, gradients, optimizer state, and allocator overhead.
It also omits cache behavior, RPC payloads, replication, and simultaneous requests.
If the second-hop cap doubles to ten, the second-hop bound doubles to 10,000 occurrences.
The new total node-occurrence upper bound becomes 100+1,000+10,000 = 11,100.
At the same width and dtype, raw representations require 11,100 × 256 = 2,841,600 bytes.
That equals approximately 2.71 MiB; it is still not a measured allocation.
If neighborhood sizes are smaller than the caps, actual sampled work is lower.
If the graph store pages or scans more edges than returned, actual examined-edge work can be higher.
Choose limits using measured tail latency and investigation quality, not this payload-only estimate.

### Capacity and economics

The queue's required resource is investigator effort, measured in case-minutes rather than account count alone.
A fixed top-k policy is appropriate only when case effort is sufficiently predictable or conservatively budgeted.
Suppose a team has 480 investigator-minutes per day available for a constructed queue.
Suppose each selected case consumes exactly 12 minutes, including documentation, with no interruptions.
The resulting capacity is 480 / 12 = 40 cases per day.
The numerator is labor-minutes per day and the denominator is labor-minutes per case.
The quotient therefore has units of cases per day.
At 20 minutes per case, the same team capacity becomes 480 / 20 = 24 cases per day.
Actual handling times vary; capacity planning should include the observed distribution and service-level requirements.
A high-complexity case may be valuable but prevent multiple simpler cases from being reviewed.
Use a policy-approved allocation and monitor outcome, aging, fairness, and reviewer workload.
The cost of a detected case includes review labor, graph storage, inference, data engineering, and customer remediation.
Currency, date, cost boundaries, and measured usage must be specified before quoting monetary efficiency.
No monetary cost, hardware benchmark, or production service-level objective has been measured here.

**Takeaway:** Bound graph expansion and investigator work separately; neither node count nor model throughput alone defines cost.

## Section 10 — Key Theorems / Results Reference

| Result | Statement and assumptions | Implication | Does not imply |
|---|---|---|---|
| Undirected degree sum | Every undirected edge contributes two endpoint incidences, including two for a self-loop under the usual convention | Check adjacency consistency against edge count | That edges represent distinct people or fraud |
| Mean within range | A nonnegative weighted mean with positive total weight lies between the smallest and largest scalar message | Detect impossible aggregate values | A mean preserves rare extreme neighbors |
| Historical eligibility | Removing records unavailable by the decision prevents those records from directly entering that feature | Supports point-in-time replay | All downstream or target leakage is eliminated |
| Fixed-budget precision | For a fully labeled top-k list, precision is positives in the list divided by k | Quantifies yield per review at that fixed count | Recall or net economic value without additional information |
| Neighbor expansion bound | Multiplying per-layer fanout caps bounds edge occurrences under the stated sampling algorithm | Supports rough compute budgeting | Distinct-node count or wall-clock latency equals the bound |

These are derived identities or conditional invariants, not empirical claims of model performance.
The degree-sum check assumes the graph and adjacency views use the same edge multiplicity and self-loop convention.
The weighted-mean bound fails when negative weights are permitted or the denominator is zero.
The historical eligibility invariant is necessary but not sufficient for a valid study.
For example, a predecision feature can encode a prior model action caused by a downstream label elsewhere in the workflow.
The fixed-budget precision denominator is cases reviewed, not the number of accounts in the entire graph.
A label that has not matured is unknown, not a counted false positive or true negative.
A performance difference between correlated cases requires uncertainty methods that respect the sampling and dependence structure.
Cluster-aware bootstrap or repeated temporal cohorts can be useful when their assumptions match the design.
No formal guarantee of GNN superiority over tabular models follows from these identities.

### Sources and evidence limitations

The [source inventory](../../../00_SOURCE_COVERAGE_DATA_SCIENCE.md#graph-source) lists the graph-fraud topic as planned.
The [source crosswalk](../../../00_SOURCE_TOPIC_CROSSWALK.json) maps the graph course and production case entries to their respective homes.
The directory inventory lists the two source paths, but their contents were not available to inspect in this workspace.
Accordingly, this lesson uses the filenames as topical provenance only, not as evidence of source-specific facts.
The [course standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) define the numerical and editorial conventions used here.
[GraphSAGE](https://arxiv.org/abs/1706.02216) describes learning functions that aggregate features from local neighborhoods for inductive representation learning.
That paper does not establish the fraud-detection performance, sampling policy, or operating costs of this illustrative design.
No empirical case result is claimed, and no source-specific benchmark has been reproduced.
Any deployment should separately validate its own legal basis, data permissions, actual latency, and outcome quality.

The [fraud evaluation sheet](../../../11_data_science_domain_specialisations/02_data_science_method_playbooks/02_fraud_modelling_playbook/Cheat_Sheets/01_Fraud_Evaluation_Calibration_and_Capacity.md) develops review capacity, calibration and selective-label limits beyond this graph example.
**Takeaway:** Identities support sanity checks, while production benefit requires a separately measured comparison.

## Section 11 — Interview / Exam Questions

These are constructed practice questions, not claims about a named employer's interview process.
Each answer states its assumptions; a defensible alternative must preserve the same decision-time and evaluation constraints.

### Level 1 — Definition / Recall

#### Q1 — Entity and edge semantics

**Prompt:** A device is shared by three accounts. Does that prove a fraud ring?

**Answer:** No. It proves only that the recorded relation links those accounts to a device under the specified identification and time rules. A household, public kiosk, or default identifier can create the same pattern.

**Assumption or alternative:** A bank transfer and a device-use observation should be different typed relations.

**Follow-up:** What if the device identifier is a constant default?

**Follow-up answer:** Its evidentiary value becomes weak; fix the instrumentation and avoid treating the hub as shared ownership.

**Red flag:** Assuming a shared edge establishes intent or common control.

**Scoring:** 0 incorrect; 1 partial; 2 correct with assumptions; 3 correct and defends the boundary.

**Answer support:** Sections 3–10; the example inputs are illustrative.

#### Q2 — Event time versus availability

**Prompt:** A payment occurred at 11:30, arrived at 12:05, and a score was issued at 12:00. Is it eligible?

**Answer:** No. Although its event time precedes the decision, availability time does not. A replay including it uses information unavailable to the serving system.

**Assumption or alternative:** If the same payment was successfully available at 11:31, it can be eligible subject to all other feature rules.

**Follow-up:** What if the serving cache missed an already committed edge?

**Follow-up answer:** That is a serving-reliability problem, distinct from late source availability; compare source and cache traces.

**Red flag:** Filtering only on the transaction event timestamp.

**Scoring:** 0 incorrect; 1 partial; 2 correct with assumptions; 3 correct and defends the boundary.

**Answer support:** Sections 3–10; the example inputs are illustrative.

#### Q3 — Unknown versus negative

**Prompt:** An account was not reviewed and no chargeback is recorded yet. Is its label negative?

**Answer:** Not necessarily. Its outcome may be unknown or immature, and the review policy may make labels selectively observed.

**Assumption or alternative:** A matured negative requires a defined target, observation horizon, and reliable outcome ascertainment.

**Follow-up:** What if all customers are observed through a complete, mature external outcome source?

**Follow-up answer:** Then that source may justify a negative under its stated coverage and target definition.

**Red flag:** Treating missing labels as confirmed legitimate outcomes.

**Scoring:** 0 incorrect; 1 partial; 2 correct with assumptions; 3 correct and defends the boundary.

**Answer support:** Sections 3–10; the example inputs are illustrative.

### Level 2 — Derivation / Apply

#### Q4 — Derive a mean

**Prompt:** Eligible neighbors have scalar features 2 and 4. Derive the equal-weight aggregate.

**Answer:** By definition the mean is the sum divided by count: (2+4)/2=3 feature units. The result lies between 2 and 4.

**Assumption or alternative:** This assumes two unique eligible neighbors and equal weight.

**Follow-up:** What if there are no eligible neighbors?

**Follow-up answer:** The denominator is zero, so use an explicit missing representation rather than claiming the mean is zero.

**Red flag:** Dividing by zero or silently treating absence as a measured zero.

**Scoring:** 0 incorrect; 1 partial; 2 correct with assumptions; 3 correct and defends the boundary.

**Answer support:** Sections 3–10; the example inputs are illustrative.

#### Q5 — Weighted messages

**Prompt:** Messages are 2 and 4 with nonnegative weights 1 and 3. Calculate their weighted mean.

**Answer:** The weighted sum is 1×2+3×4=14; total weight is 4; 14/4=3.5 feature units. It lies in the input range.

**Assumption or alternative:** Equal weights would give 3, not 3.5.

**Follow-up:** What if both weights are zero?

**Follow-up answer:** The weighted mean is undefined and a missingness policy is needed.

**Red flag:** Treating arbitrary weighted features as a fraud probability.

**Scoring:** 0 incorrect; 1 partial; 2 correct with assumptions; 3 correct and defends the boundary.

**Answer support:** Sections 3–10; the example inputs are illustrative.

#### Q6 — Point-in-time trace

**Prompt:** Using Section 5, which edges are eligible at 12:00 and what is A’s mean?

**Answer:** A—B, A—C, and C—D are eligible; A—D arrives at 12:05 and B—C occurs later. A sees B=2 and C=4, so its mean is 3.

**Assumption or alternative:** The degree sum is six, consistent with three undirected edges.

**Follow-up:** What if A—D is included retrospectively?

**Follow-up answer:** A’s mean becomes 2; that is a leaked historical reconstruction.

**Red flag:** Using the complete end-of-day graph as the decision graph.

**Scoring:** 0 incorrect; 1 partial; 2 correct with assumptions; 3 correct and defends the boundary.

**Answer support:** Sections 3–10; the example inputs are illustrative.

#### Q7 — Precision and recall at capacity

**Prompt:** Four mature labels are A positive, B negative, C positive, D negative; the queue selects C and A. Calculate both metrics.

**Answer:** Two of two reviewed are positive, so precision is 1.0. Both of the two positives in the defined population are selected, so recall is 1.0.

**Assumption or alternative:** These are constructed labels, not a production performance estimate.

**Follow-up:** What if labels for unreviewed cases are unavailable?

**Follow-up answer:** Precision on reviewed cases may still be observable under reliable review, but population recall is not identified without additional data or assumptions.

**Red flag:** Using only reviewed outcomes as the denominator for population recall.

**Scoring:** 0 incorrect; 1 partial; 2 correct with assumptions; 3 correct and defends the boundary.

**Answer support:** Sections 3–10; the example inputs are illustrative.

### Level 3 — Judgement / Design / Debug

#### Q8 — Graph model selection

**Prompt:** A tabular model fits the deadline; a GNN exceeds it and has no measured lift. Which should serve?

**Answer:** Choose the compliant baseline while investigating whether a graph variant adds value under the same candidate cohort and budget. The deadline is a hard constraint, not an accuracy feature.

**Assumption or alternative:** A graph aggregate model may be an intermediate candidate if its graph fetch is reliable and time-safe.

**Follow-up:** What if the GNN later improves mature-case yield and meets the deadline?

**Follow-up answer:** Evaluate a controlled rollout with guardrails, cost, and outcome monitoring before replacing the baseline.

**Red flag:** Assuming the most complex method is necessarily best.

**Scoring:** 0 incorrect; 1 partial; 2 correct with assumptions; 3 correct and defends the boundary.

**Answer support:** Sections 3–10; the example inputs are illustrative.

#### Q9 — Suspicious offline lift

**Prompt:** A new graph label feature doubles offline precision but online yield falls. Diagnose.

**Answer:** Check label availability and self-label exclusion first, then compare online graph freshness, sampling, candidate cohorts, and label maturity. Rebuild the original snapshots and compare actual features.

**Assumption or alternative:** A leaked neighbor label should disappear when labels unavailable at each decision are removed.

**Follow-up:** What if replay parity is exact?

**Follow-up answer:** Investigate cohort change, review selection, confidence intervals, and stale labels; parity does not prove no other bias.

**Red flag:** Declaring the model successful from offline precision alone.

**Scoring:** 0 incorrect; 1 partial; 2 correct with assumptions; 3 correct and defends the boundary.

**Answer support:** Sections 3–10; the example inputs are illustrative.

#### Q10 — High-degree hub

**Prompt:** A popular public device creates huge two-hop neighborhoods. How do you protect latency without treating all users as a ring?

**Answer:** Classify the device-use relation as weak evidence, limit work and relation fanout, retain incomplete-result indicators, and measure lost signal against a fuller audit set.

**Assumption or alternative:** Separate relationship types and keep account-level features for the fallback.

**Follow-up:** What if confirmed rings also use that device?

**Follow-up answer:** A hard exclusion may remove real signal; compare stratified sampling or relation-aware features and review the tradeoff.

**Red flag:** Marking every connected user fraudulent or silently truncating all hubs.

**Scoring:** 0 incorrect; 1 partial; 2 correct with assumptions; 3 correct and defends the boundary.

**Answer support:** Sections 3–10; the example inputs are illustrative.

#### Q11 — Investigation capacity

**Prompt:** A team has 480 minutes and each illustrative case takes 12 minutes. How many cases fit?

**Answer:** 480 minutes divided by 12 minutes per case gives 40 cases, assuming deterministic handling time and no other work.

**Assumption or alternative:** At 20 minutes per case, capacity falls to 24 cases.

**Follow-up:** What if case complexity varies widely?

**Follow-up answer:** Use observed handling-time distributions, prioritize cases at the correct grain, and track backlog and service constraints.

**Red flag:** Equating 40 accounts with 40 independent investigations.

**Scoring:** 0 incorrect; 1 partial; 2 correct with assumptions; 3 correct and defends the boundary.

**Answer support:** Sections 3–10; the example inputs are illustrative.

#### Q12 — Evaluation under policy feedback

**Prompt:** Only high-score accounts are reviewed, and reviewers provide most labels. Can you calculate unbiased population recall from reviewed labels alone?

**Answer:** No. Unreviewed positive outcomes are unknown, and review selection depends on the model. Obtain mature independent outcomes or a designed audit sample with known inclusion probabilities.

**Assumption or alternative:** Report the review policy and uncertainty, and keep unknown outcomes distinct from negatives.

**Follow-up:** What if some sampled low-score reviews reveal positives?

**Follow-up answer:** Use their known sampling probabilities in a justified estimator and report uncertainty and nonresponse limitations.

**Red flag:** Counting every unreviewed case as a negative.

**Scoring:** 0 incorrect; 1 partial; 2 correct with assumptions; 3 correct and defends the boundary.

**Answer support:** Sections 3–10; the example inputs are illustrative.

#### Q13 — Neighbor-label self leakage

**Prompt:** Why can a self-loop corrupt a fraud-neighbor label feature, and how would you test it?

**Answer:** If the seed’s own target label returns through a self-loop or cycle, the feature directly reveals the answer. Use time-safe out-of-fold aggregates that exclude the seed target and trace dependencies through every hop.

**Assumption or alternative:** Build a tiny graph with uniquely identifiable labels and check that changing the held-out seed label cannot change its feature.

**Follow-up:** What if another known label belongs to a genuinely distinct prior event?

**Follow-up answer:** It can be eligible if its target grain and availability precede the decision and policy allows it.

**Red flag:** Assuming that removing only explicit self-loops removes every cyclic leakage path.

**Scoring:** 0 incorrect; 1 partial; 2 correct with assumptions; 3 correct and defends the boundary.

**Answer support:** Sections 3–10; the example inputs are illustrative.

#### Q14 — Explanation and appeal

**Prompt:** An investigator sees a high-risk score and an attention weight on a shared device. What can they conclude?

**Answer:** The score is a prioritization signal and the device edge is a lead; neither establishes intent, identity, or causation. Show edge provenance, timestamps, uncertainty, and alternative benign explanations.

**Assumption or alternative:** Use authorized evidence and a human review process with challenge or appeal.

**Follow-up:** What if the explanation uses an edge recorded after the decision?

**Follow-up answer:** It is not a faithful explanation of that decision; regenerate it from the scoring snapshot.

**Red flag:** Treating an attention weight as proof of a causal or legal finding.

**Scoring:** 0 incorrect; 1 partial; 2 correct with assumptions; 3 correct and defends the boundary.

**Answer support:** Sections 3–10; the example inputs are illustrative.

#### Q15 — Fanout and reproducibility

**Prompt:** For 100 seeds, fanouts of 10 then 5, what is the maximum number of sampled edge occurrences?

**Answer:** The first layer has at most 100×10=1,000 occurrences and the second has at most 1,000×5=5,000, totaling 6,000 before deduplication.

**Assumption or alternative:** Distinct nodes may be fewer; the count is not a latency measurement.

**Follow-up:** What if the graph query scans many candidates to select each sample?

**Follow-up answer:** Examined-edge work can exceed returned-edge counts and must be bounded or measured separately.

**Red flag:** Equating a returned-edge cap with a complete resource bound.

**Scoring:** 0 incorrect; 1 partial; 2 correct with assumptions; 3 correct and defends the boundary.

**Answer support:** Sections 3–10; the example inputs are illustrative.

**Takeaway:** Strong answers distinguish what was knowable, what was observed, and what a constrained reviewer can actually act on.

## Section 12 — Quick Reference Card

| Item | Value |
|---|---|
| Topic | Time-safe graph-assisted fraud investigation ranking |
| Core formula | For a nonempty eligible neighbor set, mean = sum of eligible features / number of eligible neighbors (G1) |
| Key assumption | Each feature, relationship, and label was available by the seed decision time; absence of future edges alone does not rule out all leakage |
| Complexity | Sampled aggregation is O(Sd) for S sampled edge occurrences and width d, excluding transformation and I/O costs |
| Use when | Reliable typed relationships add measured value at the same investigation capacity and meet policy and latency constraints |
| Avoid when | Snapshot availability, entity identity, permissions, or graph freshness cannot be defended |
| Diagnostic | Replay graph and feature versions as of the original decision; inspect label availability and seed self-exclusion |
| Review constraint | Rank deduplicated cases at the actual labor budget; unknown outcomes are not negatives |
| Go deeper | [Graph ML topic plan](../INDEX.md); the source-specific production case remains a planned separate handoff |
