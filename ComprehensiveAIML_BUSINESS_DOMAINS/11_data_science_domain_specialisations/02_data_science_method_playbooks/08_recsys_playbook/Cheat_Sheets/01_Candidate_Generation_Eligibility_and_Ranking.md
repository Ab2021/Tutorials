---
collection: "recsys-playbook"
file_id: "01"
topic: "Candidate generation, eligibility and ranking"
level: 3
status: "draft"
version: "1.0"
created: "2026-11-07"
reviewed: null
review_trigger: "Policy, catalog, model, index or evaluation population changes"
notation_style: "plain-text"
prerequisites: []
pattern_tags: [retrieval, ranking, evaluation]
source_status: "OPEN"
owner_role: "Recommendation engineering and product analytics"
source_index: "INDEX.md"
---
# Candidate Generation, Eligibility and Ranking

## 1. File Header

**Purpose:** Select, evaluate and operate retrieval and ranking without treating inaccessible or unobserved items as valid recommendations.
**Audience:** Recommendation engineers, applied scientists, product analysts and technical interview candidates.
**Prerequisites:** Set operations, vectors and dot products, probability, data joins, timestamps, and controlled experiments; [Stanford IR book](https://nlp.stanford.edu/IR-book/) supplies information-retrieval foundations.
**Outcome:** Given a point-in-time authorized catalog, labels and retrieval outputs, compute matched recall and choose a safe candidate pipeline.
**Excluded:** Detailed learning-to-rank loss design, causal policy optimization, and statutory interpretation; ranking-metric and exploration sheets remain planned in [INDEX](INDEX.md).
**Status:** Draft, unreviewed; no product, deployment, performance or cost results measured.
**Evidence:** DERIVED logical claims, ILLUSTRATIVE constructed data and assumptions, SOURCED claims with primary references, and OPEN operational evidence.
**Source boundary:** The directory crosswalk names recommendation-system topics, but the original source files are unavailable and were not inspected.
**Retrieval cue:** Begin at Section 8 for operational failures, Section 5 for the checked comparison, and Section 9 for sizing.
**Takeaway:** The decision concerns an eligible candidate set and its downstream business impact, not a model score alone.

## 2. Concept Map / Mental Model

```text
Request + context + event-time snapshot
             |
             v
  Authorization / rights / safety constraints
             |
             v
Popularity + collaborative + content + ANN retrieval
             |
             v
Union -> deduplicate -> enforce current eligibility -> bounded candidate set
             |
             v
Rank -> diversity / policy constraints -> final eligibility check -> display
             |
             v
Impression + position + assignment + outcome logging -> evaluation and feedback
```

Arrows depict data flow; the final arrow describes a feedback loop, not proof of causation.
Offline evaluation reconstructs an earlier request's inputs and restrictions; a model must not read future outcomes.
The index is an acceleration structure, not an authority for permissions.
The ranker cannot recover relevant items absent from its candidate set.
**Takeaway:** Protect eligibility across stages and evaluate retrieval separately from display ranking.

## 3. Foundational Definitions

| Term | Meaning and operational test | Units/type | Confusion to avoid |
|---|---|---|---|
| Request q | User, surface, context and decision timestamp for one recommendation opportunity | event | A user may have multiple requests. |
| Catalog I(t) | Items known at timestamp t | set of item IDs | Current catalog is not necessarily historical catalog. |
| Eligible set E(q,t) | Items allowed by applicable rights, policy, availability and access at decision time | subset of I(t) | Retrieved candidates are not automatically eligible. |
| Relevant set R(q,t,H) | Eligible items satisfying a declared label rule in horizon H | subset of E | Unobserved does not imply irrelevant. |
| Candidate set C(q) | Unique items admitted by retrieval for ranking | set, bounded by budget | Candidate count is not ordered-display length. |
| Exact top-k X(q) | First k eligible items under a fixed score and deterministic tie rule | ordered list | Exact under an embedding score is not ground truth relevance. |
| ANN top-k A(q) | Approximate nearest-neighbor result under the same score and eligible universe | ordered list | Approximation recall is different from relevance recall. |
| Implicit feedback | Observed actions such as plays, clicks or purchases without explicit relevance judgment | event | Absence of action can reflect lack of exposure. |
| Exposure | An item actually rendered or otherwise surfaced to a user under a recorded policy | event | A retrieved but unseen item is not an impression. |
| Freshness | Age or lag of an index or its underlying catalog/features relative to the relevant event | seconds or duration | A fresh index build can contain stale inputs. |
| Diversity | Declared spread of the displayed slate across item or provider attributes | metric-dependent | A relevance model does not guarantee variety. |
| Guardrail | A separately monitored constraint or outcome that bounds acceptable policy changes | threshold or measure | A click increase does not establish guardrail compliance. |

Intuition: the eligible set is the legal and operational playing field; candidate generation chooses a manageable subset of it.
For streaming, territory and subscription tier can change E; for retail, fulfillment location and inventory can change E.
A relevance label may be based on a subsequent completion, but only after the selected outcome window is mature.
Offline relevance is a proxy for utility; its coverage and observation mechanism are part of the definition.
**Takeaway:** Define sets, time and observation mechanism before reporting any retrieval percentage.

## 4. Core Formulas with Full Derivation

### 4.1 Simplest / Base Case

Validity: one request; the relevant set is fully specified and nonempty, within the same eligible catalog used by the candidate system.
Derivation: count relevant items recovered by the top-k candidate list, then divide by the number of relevant items.
**[R1] FORMULA:** recall_at_k(q) = |C_k(q) intersect R(q)| / |R(q)|.
**WHERE:** C_k is the first k distinct eligible candidate IDs; R is the relevant eligible ID set; bars denote counts.
**CONSTRAINT:** |R| > 0; declare duplicate handling, ties, cutoff and label horizon before comparison.
**NUMERICAL EXAMPLE:** C_2 = {A,B}, R = {B,C}; intersection has one item, so recall is 1/2 = 0.5.
**INTERPRETATION:** One of the two eligible relevant items reached the candidate list; ranking can at best show one of them.
Boundary: if R is empty, this per-request ratio is undefined; report the count and exclusion convention rather than silently substituting zero.

### 4.2 General Case

Validity: a fixed collection Q+ of requests with nonempty relevant sets and the same as-of eligibility and labeling conventions across systems.
Derivation: averaging per-request fractions weights each request equally; pooling hits and relevant items instead weights requests by relevant-item count.
**[R2] FORMULA:** macro recall = sum over q in Q+ of recall_at_k(q) / |Q+|; micro recall = sum hits / sum relevant counts.
**WHERE:** hits are |C_k(q) intersect R(q)|; all terms are nonnegative integer counts except the resulting ratios.
**CONSTRAINT:** Q+ is nonempty; use identical Q+, k, E, label rule and horizon for the compared systems.
**NUMERICAL EXAMPLE:** request recalls 1/2 and 1/1 give macro (0.5+1)/2 = 0.75 and micro 2/3.
**INTERPRETATION:** The two summaries answer different weighting questions and need not agree.
Boundary: a policy that changes which requests have labels changes the estimand; do not silently compare different Q+ populations.

### 4.3 Key Special Cases / Variants

Validity: ANN fidelity compares an approximate list with an exact score oracle under a common eligible universe, query vector, item vectors, metric and tie rule.
Derivation: the fraction of exact top-k IDs recovered is an overlap divided by the exact-list length.
**[A1] FORMULA:** ANN recall_at_k = |A_k intersect X_k| / |X_k|.
**WHERE:** A_k is approximate output and X_k is exact top-k output; both consist of distinct eligible IDs.
**CONSTRAINT:** |X_k| > 0; when fewer than k eligible items exist use the actual exact list length and disclose it.
**NUMERICAL EXAMPLE:** X_3 = {A,B,C} and A_3 = {A,C,D}; overlap is 2, so ANN recall is 2/3.
**INTERPRETATION:** Two of three exact-score neighbors were retrieved; relevance is not established by this comparison.
Boundary: approximate output filtered after search may contain fewer than k items; evaluate the actual filtered list against the eligible exact oracle.

| Variant | Difference | Use condition | Disqualifier |
|---|---|---|---|
| Popularity | Order by permitted historical activity | Stable fallback with enough mature activity | Sensitive cohort or manipulable counts without safeguards |
| Collaborative | Infer affinity from user-item interactions | Sufficient informative histories | New users and items without a fallback |
| Content | Compare authorized item attributes and user preferences | Features exist for new items | Missing or unreliable metadata |
| Dense exact | Score every eligible item in vector space | Oracle or small catalog within budget | Full scan exceeds resource budget |
| Dense ANN | Approximate the vector-space search | Measured quality/resource tradeoff is acceptable | Eligibility, recall or freshness requirements cannot be met |

An exact dense search is exact only for the specified vector score, not for human preference or long-term value.
Maximum inner product and cosine similarity are distinct unless vectors are normalized and the intended comparison permits it.
SOURCED: Faiss documents similarity search and alternatives to exhaustive search in its [original project documentation](https://faiss.ai/); specific index properties depend on configuration and version.

### 4.4 Asymptotic / Limiting Behaviour and Counterexample

If k includes every eligible item, relevance recall equals one only when all items in R are present in that eligible set and there is no downstream truncation.
If k is zero and R is nonempty, recall is zero.
If R is empty, the per-request fraction is undefined, not automatically perfect or zero.
Counterexample: searching the unrestricted global top-k and filtering afterward can exclude all returned items even when eligible exact top-k items exist.
Over-retrieval can mitigate underfill but does not guarantee enough eligible results; a sufficiently selective filter can still remove them all.
A strict authorization check must run at the last responsible moment because rights or inventory can change between retrieval and display.

**Input contract:** request q, as-of eligibility snapshot E, source lists with scores, budget k and deterministic tie rules.
**Transition:** obtain source outputs, filter under E, deduplicate item IDs, score or merge, and take at most k items.
**Invariant:** every retained ID belongs to E; the output contains no repeated ID and has size at most k.
**Correctness sketch:** filtering removes IDs outside E, deduplication removes repeated IDs, and truncation cannot introduce either kind of violation.
**Termination:** each input list is finite and each item is processed at most once per list in this abstract procedure.
**Operation model:** with n total source entries and expected constant-time hash membership, filtering and deduplication take expected O(n); merge/sort costs depend on the selected ordering algorithm.
**Takeaway:** Correctness is relative to a defined point-in-time authorization set and deterministic score contract.

## 5. Worked Numerical Example

**ILLUSTRATIVE and arithmetically checked:** A hypothetical video request has six catalog items at decision time.
**Decision:** Compare two retrieval candidates, holding eligibility, relevance labels, cutoff and request fixed.
**Snapshot:** A, B, C, D and E are authorized and available; F is outside territory rights.
**Mature observed relevant labels:** B and D, both within the eligible set; no claim is made that unlabeled items are disliked.
**Candidate budget:** Three unique eligible IDs per system; ties already resolved consistently.

| Item | Eligible? | Relevant under declared label? | System P rank | System Q rank |
|---|---|---|---:|---:|
| A | yes | no | 1 | 2 |
| B | yes | yes | 2 | 1 |
| C | yes | no | 3 | absent |
| D | yes | yes | absent | 3 |
| E | yes | no | absent | absent |
| F | no: rights | not in target population | absent | excluded before output |

P yields {A,B,C}; Q yields {B,A,D}.
P has one relevant hit and Q has two; both are scored against the same R = {B,D}.
Using R1, P recall@3 = 1/2 = 0.5; Q recall@3 = 2/2 = 1.0.
The absolute difference is 0.5, or 50 percentage points, for this one constructed request only.
The values do not establish population superiority, precision, satisfaction, business impact or statistical significance.
**Independent check:** Enumerating hits gives P = {B} and Q = {B,D}; both denominators are two eligible labeled IDs.
**Wrong path:** Including F as relevant adds an unauthorized item to the denominator and changes the target population; the comparison is no longer eligible-item recall.
**Production change:** Reconstruct rights and features as of each request, mature the outcome window, and aggregate over a representative query cohort with uncertainty.
**Remaining uncertainty:** A user may have liked an eligible item never displayed; implicit labels are exposure-selected rather than complete relevance truth.
**Takeaway:** A fair retrieval comparison holds eligibility, labeled population and cutoff fixed, and does not equate one example with measured product impact.

## 6. Visual / Geometric Intuition

```text
Time increases left to right (schematic, not to scale)
feature event ---- request t ---- display ---- outcome ---- label maturity
                      |                         |
                 retrieval snapshot       outcome window H
```

Caption: request t is the temporal boundary; features and index contents must be available at or before it.
The outcome may occur later, but must not feed the model or catalog snapshot used to reconstruct that same decision.
```text
All catalog items: [ A B C D E F ]
Eligible E:        [ A B C D E   ]
Relevant R:        [   B   D     ]
Q candidates:      [ A B   D     ]
```
Caption: F lies outside E; both relevant IDs lie in Q's eligible candidates.
Set overlap illustrates recall, not the rank order or the user's latent preferences.
**Takeaway:** The time boundary prevents leakage; the set boundary prevents unauthorized comparisons.

## 7. Comparison / Selection Table

| Method | Preconditions | Quality or guarantee | Cost model | Use when | Avoid when | Evidence |
|---|---|---|---|---|---|---|
| Popularity | Point-in-time counts and availability | No personalization guarantee | Precompute and filter | Cold start or resilient baseline | Popularity manipulation or narrow coverage dominates | HEURISTIC |
| Collaborative | History and stable entity identity | Learns interaction association, not causality | Training and serving depend on model | Sufficient data and repeat preference | Cold users/items dominate | HEURISTIC |
| Content | Reliable authorized metadata | Can retrieve new catalog items | Feature maintenance and similarity scoring | Metadata is available before exposure | Metadata sparse, sensitive or misleading | HEURISTIC |
| Exact dense | Fixed query and vectors, score and filter | Exact top-k for defined eligible score | Full eligible scan | Oracle and manageable workload | Scan violates measured budget | DERIVED |
| ANN dense | Same vector semantics and measurable oracle | Approximate; no general fixed recall promise | Index build, memory and query tuning | Exact scan is too costly and quality measured | Authorization or recall cannot be verified | SOURCED / OPEN |

Priority: fail closed on access, rights and safety; do not trade authorization for a recall gain.
Then select a feasible baseline and compare variants on the same eligible and labeled request population.
When latency, recall and freshness conflict, quantify the tradeoff and escalate the policy decision instead of assuming the largest candidate list wins.
Cold users can receive locally eligible popularity, contextual or explicitly provided preference candidates.
Cold items can enter content-based pools with appropriate safety and exposure controls, rather than requiring interaction history first.
Diversity can be enforced during slate construction, but its constraint should not bypass availability or permissions.
**Benchmark status: NOT MEASURED.** Fix dataset snapshot, request cohort, score, index version, hardware, load, k and filter policy; report latency distributions, recall, underfill, uncertainty and cost for each method.
**Takeaway:** Compare methods as complete authorized pipelines, not as unconstrained scores from different populations.

## 8. Failure Modes & Edge Cases

All scenarios in this section are HYPOTHETICAL diagnostic exercises, not observed incidents.
The planned Common Issues artifact does not yet exist; no issue IDs are claimed.

### F-01: Rights violation after cache hit

- **Trigger / symptom:** A revoked title appears in a cached slate.
- **Competing hypotheses:** Stale cache versus missing authorization check.
- **Discriminating check:** Compare policy version and decision-time revocation with final-check audit.
- **Immediate bounded action:** Remove affected item, fail closed and revalidate every cache hit.
- **Prevention and verification:** Replay revocation across cache and serve path; audit zero unauthorized displays.
- **Escalation:** If evidence does not distinguish the alternatives, preserve the trace and involve the owning service or policy team; do not infer a root cause.

### F-02: Filtered ANN underfills

- **Trigger / symptom:** Candidates disappear after filtering.
- **Competing hypotheses:** Postfilter selectivity versus an empty catalog.
- **Discriminating check:** Compare prefilter counts, rejection reasons and exact eligible count.
- **Immediate bounded action:** Use eligible partitioning or bounded over-retrieval with safe fallback.
- **Prevention and verification:** Measure underfill per eligible-count bucket; never promise filling.
- **Escalation:** If evidence does not distinguish the alternatives, preserve the trace and involve the owning service or policy team; do not infer a root cause.

### F-03: Offline recall spikes

- **Trigger / symptom:** Sudden improvement follows a feature pipeline change.
- **Competing hypotheses:** Future features versus genuine gain.
- **Discriminating check:** Check event availability times and reconstruct a sample request.
- **Immediate bounded action:** Roll back leaking feature and rebuild point-in-time data.
- **Prevention and verification:** Add temporal availability assertions and compare on fixed requests.
- **Escalation:** If evidence does not distinguish the alternatives, preserve the trace and involve the owning service or policy team; do not infer a root cause.

### F-04: New user receives empty slate

- **Trigger / symptom:** No personalized embedding exists.
- **Competing hypotheses:** Absent history versus upstream identity outage.
- **Discriminating check:** Distinguish unknown-user state from feature-service failures.
- **Immediate bounded action:** Serve authorized context or popularity fallback.
- **Prevention and verification:** Track cold-user fill and fallback reasons by cohort.
- **Escalation:** If evidence does not distinguish the alternatives, preserve the trace and involve the owning service or policy team; do not infer a root cause.

### F-05: New items receive no exposure

- **Trigger / symptom:** Items exist but cannot be retrieved.
- **Competing hypotheses:** Missing embeddings versus policy exclusion.
- **Discriminating check:** Compare item creation, embedding, index and policy timestamps.
- **Immediate bounded action:** Enable verified metadata retrieval or controlled eligible exploration.
- **Prevention and verification:** Monitor creation-to-index lag and new-item exposure.
- **Escalation:** If evidence does not distinguish the alternatives, preserve the trace and involve the owning service or policy team; do not infer a root cause.

### F-06: Approximation recall drops

- **Trigger / symptom:** ANN differs from the exact oracle.
- **Competing hypotheses:** Changed index configuration versus vector mismatch.
- **Discriminating check:** Run both on identical vectors, eligibility and tie rule.
- **Immediate bounded action:** Restore compatible index or tune the search budget.
- **Prevention and verification:** Track matched oracle overlap by cohort and catalog snapshot.
- **Escalation:** If evidence does not distinguish the alternatives, preserve the trace and involve the owning service or policy team; do not infer a root cause.

### F-07: Recall stays high while online outcomes fall

- **Trigger / symptom:** Offline retrieved labels remain stable.
- **Competing hypotheses:** Position bias, objective mismatch or ranking regression.
- **Discriminating check:** Compare displayed positions, randomized assignment and downstream outcomes.
- **Immediate bounded action:** Hold rollout and inspect guardrails and ranking changes.
- **Prevention and verification:** Maintain independent business and relevance metrics.
- **Escalation:** If evidence does not distinguish the alternatives, preserve the trace and involve the owning service or policy team; do not infer a root cause.

### F-08: Expired catalog entries persist

- **Trigger / symptom:** Removed items remain in candidates.
- **Competing hypotheses:** Deletion propagation lag versus stale replica.
- **Discriminating check:** Trace item tombstone through source, index and serving replica.
- **Immediate bounded action:** Block item at final authorization layer and propagate removal.
- **Prevention and verification:** Test deletion propagation and monitor oldest pending tombstone.
- **Escalation:** If evidence does not distinguish the alternatives, preserve the trace and involve the owning service or policy team; do not infer a root cause.

### F-09: Popularity fallback becomes unsafe

- **Trigger / symptom:** Fallback returns ineligible or manipulated items.
- **Competing hypotheses:** Unfiltered precomputed list versus activity manipulation.
- **Discriminating check:** Recompute allowed list and inspect source-event quality.
- **Immediate bounded action:** Apply current checks and remove suspect activity.
- **Prevention and verification:** Audit fallback provenance, policy and anomaly alerts.
- **Escalation:** If evidence does not distinguish the alternatives, preserve the trace and involve the owning service or policy team; do not infer a root cause.

### F-10: Two services disagree on eligibility

- **Trigger / symptom:** Offline and online sets differ.
- **Competing hypotheses:** Clock skew, policy version mismatch or missing attributes.
- **Discriminating check:** Compare decision timestamp, input attributes and versioned policy trace.
- **Immediate bounded action:** Use consistent policy contract and quarantine unresolved discrepancies.
- **Prevention and verification:** Log version and replay sampled decisions.
- **Escalation:** If evidence does not distinguish the alternatives, preserve the trace and involve the owning service or policy team; do not infer a root cause.

### F-11: Duplicate items crowd candidates

- **Trigger / symptom:** Same item appears from several sources.
- **Competing hypotheses:** Union without canonical ID or duplicate handling.
- **Discriminating check:** Inspect raw source IDs and canonicalization mapping.
- **Immediate bounded action:** Canonicalize then deduplicate before budget truncation.
- **Prevention and verification:** Test cross-source duplicate cases and count unique candidates.
- **Escalation:** If evidence does not distinguish the alternatives, preserve the trace and involve the owning service or policy team; do not infer a root cause.

### F-12: Recall calculation divides by zero

- **Trigger / symptom:** A request has no observed relevant labels.
- **Competing hypotheses:** Unmatured outcome window versus genuinely no observed positives.
- **Discriminating check:** Check label maturity and outcome join completeness.
- **Immediate bounded action:** Mark undefined and report exclusions separately.
- **Prevention and verification:** Publish population flow and maturity lag with every report.
- **Escalation:** If evidence does not distinguish the alternatives, preserve the trace and involve the owning service or policy team; do not infer a root cause.

### F-13: A minority cohort sees poor coverage

- **Trigger / symptom:** Aggregate recall masks cohort failures.
- **Competing hypotheses:** Sparse history, rights restrictions or metadata quality.
- **Discriminating check:** Compare cohort-specific eligible counts, missingness and uncertainty.
- **Immediate bounded action:** Improve safe fallback and collect cohort-relevant evidence.
- **Prevention and verification:** Monitor disaggregated quality with privacy-preserving thresholds.
- **Escalation:** If evidence does not distinguish the alternatives, preserve the trace and involve the owning service or policy team; do not infer a root cause.

### F-14: A provider dominates the slate

- **Trigger / symptom:** Many near-identical items fill the display.
- **Competing hypotheses:** Popularity feedback versus ranker score concentration.
- **Discriminating check:** Measure provider and category mix at retrieval and display.
- **Immediate bounded action:** Apply reviewed diversity rules within the eligible pool.
- **Prevention and verification:** Monitor relevance and exposure tradeoffs together.
- **Escalation:** If evidence does not distinguish the alternatives, preserve the trace and involve the owning service or policy team; do not infer a root cause.

### F-15: Inventory vanishes before purchase

- **Trigger / symptom:** Retrieved item becomes unavailable after ranking.
- **Competing hypotheses:** Stale inventory versus concurrent reservation.
- **Discriminating check:** Compare inventory event and serve timestamps with reservation log.
- **Immediate bounded action:** Recheck availability and support substitution or safe omission.
- **Prevention and verification:** Track stale inventory age and downstream failed actions.
- **Escalation:** If evidence does not distinguish the alternatives, preserve the trace and involve the owning service or policy team; do not infer a root cause.

### F-16: Evaluation accidentally compares different cohorts

- **Trigger / symptom:** One system appears better after filtering requests.
- **Competing hypotheses:** Different label maturity, eligibility or query sampling.
- **Discriminating check:** Join by request ID and compare population flow at every stage.
- **Immediate bounded action:** Recompute on the common prespecified request population.
- **Prevention and verification:** Report request counts and reasons for exclusions.
- **Escalation:** If evidence does not distinguish the alternatives, preserve the trace and involve the owning service or policy team; do not infer a root cause.

### F-17: A vector rebuild changes rankings unexpectedly

- **Trigger / symptom:** Similarity scores drift after deployment.
- **Competing hypotheses:** Normalization mismatch versus changed model or metric.
- **Discriminating check:** Compare vector norms, model IDs and score implementations.
- **Immediate bounded action:** Restore matched embedding and index versions.
- **Prevention and verification:** Gate incompatible versions and test score consistency.
- **Escalation:** If evidence does not distinguish the alternatives, preserve the trace and involve the owning service or policy team; do not infer a root cause.

### F-18: Long-tail items disappear

- **Trigger / symptom:** Coverage falls despite stable aggregate relevance.
- **Competing hypotheses:** Index truncation, filtering or popularity bias.
- **Discriminating check:** Break down catalog inclusion and retrieval by age and activity.
- **Immediate bounded action:** Fix catalog ingestion or adjust bounded source allocation.
- **Prevention and verification:** Track eligible coverage and retrieval by age bucket.
- **Escalation:** If evidence does not distinguish the alternatives, preserve the trace and involve the owning service or policy team; do not infer a root cause.

### F-19: User deletion is incomplete

- **Trigger / symptom:** Personal data persists after deletion request.
- **Competing hypotheses:** Untracked training, feature store, cache or logs.
- **Discriminating check:** Trace identifiers through data inventory and deletion receipts.
- **Immediate bounded action:** Remove or isolate data per applicable policy and escalate exceptions.
- **Prevention and verification:** Test deletion propagation and retention boundaries.
- **Escalation:** If evidence does not distinguish the alternatives, preserve the trace and involve the owning service or policy team; do not infer a root cause.

### F-20: Latency spikes during index refresh

- **Trigger / symptom:** Serving slows while rebuilds run.
- **Competing hypotheses:** Resource contention versus traffic burst.
- **Discriminating check:** Compare refresh interval, resource saturation and request rate.
- **Immediate bounded action:** Isolate build resources or stagger and roll back refresh.
- **Prevention and verification:** Capacity-test refresh under representative traffic.
- **Escalation:** If evidence does not distinguish the alternatives, preserve the trace and involve the owning service or policy team; do not infer a root cause.

### F-21: Exposure logging is missing

- **Trigger / symptom:** Clicks exist but impression records are absent.
- **Competing hypotheses:** Instrumentation loss versus inconsistent identifiers.
- **Discriminating check:** Join clicks to request, item and render logs with timestamps.
- **Immediate bounded action:** Repair logging and exclude unsupported causal analyses.
- **Prevention and verification:** Alert on join completeness and version the event schema.
- **Escalation:** If evidence does not distinguish the alternatives, preserve the trace and involve the owning service or policy team; do not infer a root cause.

### F-22: An apparently negative item was never displayed

- **Trigger / symptom:** Training treats every unseen item as rejected.
- **Competing hypotheses:** Exposure censoring versus actual user rejection.
- **Discriminating check:** Inspect rendered impression records and position.
- **Immediate bounded action:** Distinguish unexposed examples from observed nonactions.
- **Prevention and verification:** Document sampling and test sensitivity to missing exposure.
- **Escalation:** If evidence does not distinguish the alternatives, preserve the trace and involve the owning service or policy team; do not infer a root cause.

### F-23: Stale embeddings survive rollback

- **Trigger / symptom:** Model and index vectors are incompatible.
- **Competing hypotheses:** Partial deployment versus mismatched artifact selection.
- **Discriminating check:** Check version IDs for query encoder, item encoder and index.
- **Immediate bounded action:** Rollback compatible artifact set atomically.
- **Prevention and verification:** Use compatibility manifests and staged rollout gates.
- **Escalation:** If evidence does not distinguish the alternatives, preserve the trace and involve the owning service or policy team; do not infer a root cause.

### F-24: Label horizon is not mature

- **Trigger / symptom:** Recent requests show low recall or conversion.
- **Competing hypotheses:** Delayed outcomes versus a real behavioral change.
- **Discriminating check:** Plot label arrival by request age and compare mature cohorts.
- **Immediate bounded action:** Restrict primary comparison to matured windows.
- **Prevention and verification:** Record outcome cutoff and report censoring sensitivity.
- **Escalation:** If evidence does not distinguish the alternatives, preserve the trace and involve the owning service or policy team; do not infer a root cause.

### F-25: Zero eligible items

- **Trigger / symptom:** All catalog items fail rights or availability.
- **Competing hypotheses:** True no supply versus policy service failure.
- **Discriminating check:** Inspect authoritative catalog and policy service status.
- **Immediate bounded action:** Return an honest empty or approved safe response.
- **Prevention and verification:** Monitor empty set rate and distinguish policy errors.
- **Escalation:** If evidence does not distinguish the alternatives, preserve the trace and involve the owning service or policy team; do not infer a root cause.

### F-26: Security boundary crossed in logs

- **Trigger / symptom:** Detailed recommendation logs reveal private attributes.
- **Competing hypotheses:** Overcollection versus access control misconfiguration.
- **Discriminating check:** Inspect schemas, roles, retention and access audit.
- **Immediate bounded action:** Restrict access and remove unnecessary fields under policy.
- **Prevention and verification:** Apply minimization, encryption and reviewed retention.
- **Escalation:** If evidence does not distinguish the alternatives, preserve the trace and involve the owning service or policy team; do not infer a root cause.

### F-27: Position confounds a source comparison

- **Trigger / symptom:** Top-ranked source has more clicks.
- **Competing hypotheses:** Better relevance versus higher exposure position.
- **Discriminating check:** Compare randomized assignments or position-balanced exposures.
- **Immediate bounded action:** Run a controlled test or use defensible bias adjustment.
- **Prevention and verification:** Record position, policy probability and render status.
- **Escalation:** If evidence does not distinguish the alternatives, preserve the trace and involve the owning service or policy team; do not infer a root cause.

### F-28: Item update never reaches index

- **Trigger / symptom:** Feature changes remain invisible to retrieval.
- **Competing hypotheses:** Failed event versus rebuild watermark lag.
- **Discriminating check:** Trace item version through event, embedding and index.
- **Immediate bounded action:** Replay or rebuild missing updates and enforce final filters.
- **Prevention and verification:** Monitor watermark lag and missing-version reconciliation.
- **Escalation:** If evidence does not distinguish the alternatives, preserve the trace and involve the owning service or policy team; do not infer a root cause.

### F-29: Low diversity hurts discovery

- **Trigger / symptom:** Similar items occupy every position.
- **Competing hypotheses:** Candidate pool too narrow versus reranker constraint.
- **Discriminating check:** Measure category mix before and after ranking.
- **Immediate bounded action:** Expand eligible sources or tune explicit slate constraints.
- **Prevention and verification:** Evaluate relevance and diversity jointly by cohort.
- **Escalation:** If evidence does not distinguish the alternatives, preserve the trace and involve the owning service or policy team; do not infer a root cause.

### F-30: Exact comparison is deceptively favorable

- **Trigger / symptom:** ANN is scored against an unrestricted oracle.
- **Competing hypotheses:** Oracle has different filters or ties.
- **Discriminating check:** Compare eligible IDs and score/tie rules for sampled queries.
- **Immediate bounded action:** Recompute oracle on the same authorized universe.
- **Prevention and verification:** Version oracle inputs and keep matched cohort checks.
- **Escalation:** If evidence does not distinguish the alternatives, preserve the trace and involve the owning service or policy team; do not infer a root cause.


### Operational measurement catalog

Each entry below is a proposed measurement, not an observed result or a universal threshold. Rates require nonzero denominators, explicit event-time windows and documented missing-data handling.

#### M-01 — Eligible catalog size
- **Numerator or value:** distinct eligible item IDs.
- **Denominator or observation unit:** request.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** Empty catalogs can be legitimate.
- **Action on change:** Inspect policy and supply.
- **Useful breakdown:** Territory and surface.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-02 — Rights rejection rate
- **Numerator or value:** rights-rejected candidate IDs.
- **Denominator or observation unit:** candidate IDs checked for rights.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** A high rate can reflect expected geography.
- **Action on change:** Inspect index partitioning and rights changes.
- **Useful breakdown:** Territory and policy version.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-03 — Availability rejection rate
- **Numerator or value:** availability-rejected candidate IDs.
- **Denominator or observation unit:** candidate IDs checked for availability.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** Stock volatility changes the expected rate.
- **Action on change:** Compare inventory event timestamps.
- **Useful breakdown:** Fulfillment location and item age.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-04 — Safety rejection rate
- **Numerator or value:** safety-rejected candidate IDs.
- **Denominator or observation unit:** candidate IDs checked for safety.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** Policy changes alter the population.
- **Action on change:** Inspect reason codes and policy version.
- **Useful breakdown:** Surface and policy version.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-05 — Authorization error rate
- **Numerator or value:** authorization checks returning errors.
- **Denominator or observation unit:** authorization checks attempted.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** Errors are not equivalent to denials.
- **Action on change:** Fail closed and investigate service health.
- **Useful breakdown:** Policy service version and region.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-06 — Candidate underfill rate
- **Numerator or value:** requests with fewer than k eligible candidates.
- **Denominator or observation unit:** eligible retrieval requests.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** Catalog scarcity can make underfill unavoidable.
- **Action on change:** Compare with the exact eligible catalog size.
- **Useful breakdown:** Eligible catalog size and source.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-07 — Duplicate source rate
- **Numerator or value:** duplicate canonical IDs in the source union.
- **Denominator or observation unit:** raw source-list IDs.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** Multiple sources may legitimately return the same id.
- **Action on change:** Inspect canonicalization before changing allocation.
- **Useful breakdown:** Source pair and catalog family.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-08 — Candidate uniqueness
- **Numerator or value:** distinct canonical candidate IDs.
- **Denominator or observation unit:** candidate IDs admitted before truncation.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** Deduplication after truncation can hide crowding.
- **Action on change:** Deduplicate before applying the budget.
- **Useful breakdown:** Source combination and surface.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-09 — Popularity source contribution
- **Numerator or value:** unique eligible candidates attributed to popularity.
- **Denominator or observation unit:** unique eligible candidates in the union.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** Attribution must declare how overlaps are assigned.
- **Action on change:** Review source allocation and overlap.
- **Useful breakdown:** User history bucket.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-10 — Collaborative source contribution
- **Numerator or value:** unique eligible candidates attributed to collaborative retrieval.
- **Denominator or observation unit:** unique eligible candidates in the union.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** Overlap attribution is convention dependent.
- **Action on change:** Inspect history coverage and source health.
- **Useful breakdown:** User history bucket.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-11 — Content source contribution
- **Numerator or value:** unique eligible candidates attributed to content retrieval.
- **Denominator or observation unit:** unique eligible candidates in the union.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** Metadata coverage changes the eligible pool.
- **Action on change:** Inspect feature completeness and source allocation.
- **Useful breakdown:** Item age and category.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-12 — Dense source contribution
- **Numerator or value:** unique eligible candidates attributed to dense retrieval.
- **Denominator or observation unit:** unique eligible candidates in the union.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** Embedding coverage and overlapping sources affect contribution.
- **Action on change:** Inspect artifact compatibility and index coverage.
- **Useful breakdown:** Item age and model version.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-13 — Eligible item embedding coverage
- **Numerator or value:** eligible items with compatible embeddings.
- **Denominator or observation unit:** eligible items.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** Eligibility and embedding snapshots must have the same timestamp.
- **Action on change:** Repair missing embeddings or use a fallback source.
- **Useful breakdown:** Item age and encoder version.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-14 — Eligible index inclusion
- **Numerator or value:** eligible items present in the index.
- **Denominator or observation unit:** eligible items.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** A present id may have an outdated vector.
- **Action on change:** Compare indexed and source versions.
- **Useful breakdown:** Item age and shard.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-15 — Index update lag
- **Numerator or value:** elapsed time from source change to searchable update.
- **Denominator or observation unit:** one successfully propagated update.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** Undelivered updates are missing from successful-only distributions.
- **Action on change:** Monitor backlog and failed propagation separately.
- **Useful breakdown:** Shard and update type.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-16 — Deletion propagation lag
- **Numerator or value:** elapsed time from deletion event to removal from index.
- **Denominator or observation unit:** one completed deletion.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** Incomplete deletions must be reported as outstanding.
- **Action on change:** Block at authorization and investigate the backlog.
- **Useful breakdown:** Replica and deletion type.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-17 — Stale index entries
- **Numerator or value:** indexed IDs whose item version is older than source.
- **Denominator or observation unit:** indexed IDs reconciled.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** Reconciliation itself may use a stale source snapshot.
- **Action on change:** Compare consistent watermarks and replay updates.
- **Useful breakdown:** Shard and item age.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-18 — ANN oracle overlap
- **Numerator or value:** approximate IDs also present in exact eligible top-k.
- **Denominator or observation unit:** exact eligible top-k IDs.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** This measures score fidelity rather than user relevance.
- **Action on change:** Tune the index or investigate vector mismatch.
- **Useful breakdown:** Query cohort and filter selectivity.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-19 — Candidate relevant recall
- **Numerator or value:** observed eligible relevant IDs recovered in top-k.
- **Denominator or observation unit:** observed eligible relevant IDs.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** Implicit labels are exposure-selected.
- **Action on change:** Audit labels and compare matched request cohorts.
- **Useful breakdown:** Cohort and label horizon.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-20 — Display relevant recall
- **Numerator or value:** observed eligible relevant IDs displayed.
- **Denominator or observation unit:** observed eligible relevant IDs.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** Slate length and position affect the outcome.
- **Action on change:** Compare with candidate recall to localize loss.
- **Useful breakdown:** Slate size and cohort.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-21 — Ranking retention of candidate hits
- **Numerator or value:** candidate relevant IDs retained in the displayed slate.
- **Denominator or observation unit:** candidate relevant IDs.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** Undefined when candidate hits are absent.
- **Action on change:** Inspect ranking and slate constraints.
- **Useful breakdown:** Surface and candidate-source mix.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-22 — Observed click rate
- **Numerator or value:** clicked rendered impressions.
- **Denominator or observation unit:** rendered impressions.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** Position and exposure policy confound interpretation.
- **Action on change:** Compare within randomized assignment.
- **Useful breakdown:** Position and surface.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-23 — Observed completion rate
- **Numerator or value:** completed eligible plays.
- **Denominator or observation unit:** started eligible plays.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** Completion may be unavailable or delayed.
- **Action on change:** Inspect event completeness and outcome maturity.
- **Useful breakdown:** Content duration and client.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-24 — Mature-label coverage
- **Numerator or value:** requests whose full outcome horizon has elapsed.
- **Denominator or observation unit:** requests selected for evaluation.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** Mature time alone does not guarantee complete ingestion.
- **Action on change:** Inspect outcome watermarks.
- **Useful breakdown:** Request date and outcome type.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-25 — Impression join completeness
- **Numerator or value:** rendered impressions joined to a request and item.
- **Denominator or observation unit:** rendered impressions received.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** Missing logs can occur before either side of the join.
- **Action on change:** Audit independent client and server counters.
- **Useful breakdown:** Client version and surface.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-26 — Click join completeness
- **Numerator or value:** click events joined to a recorded impression.
- **Denominator or observation unit:** click events received.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** Some interactions may have valid non-impression entry points.
- **Action on change:** Classify missing joins before exclusion.
- **Useful breakdown:** Client version and entry point.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-27 — Logged policy probability coverage
- **Numerator or value:** randomized decisions with recorded assignment probability.
- **Denominator or observation unit:** randomized decisions.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** Correctness of the probability also requires auditing.
- **Action on change:** Repair assignment logging and validate values.
- **Useful breakdown:** Experiment and policy version.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-28 — Cold-user fill rate
- **Numerator or value:** cold-user requests with a nonempty eligible slate.
- **Denominator or observation unit:** cold-user requests.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** A cold-user definition must be fixed over time.
- **Action on change:** Inspect fallback and eligibility decisions.
- **Useful breakdown:** Region and cold-start definition.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-29 — Cold-item index coverage
- **Numerator or value:** eligible new items indexed within a declared age window.
- **Denominator or observation unit:** eligible new items in that window.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** Items excluded for rights must not be counted as missing.
- **Action on change:** Trace ingestion and embedding generation.
- **Useful breakdown:** Item age and provider.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-30 — New-item exposure
- **Numerator or value:** eligible new items rendered at least once.
- **Denominator or observation unit:** eligible new items in a declared observation period.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** Opportunity time differs by item creation time.
- **Action on change:** Compare age-matched items and review exploration.
- **Useful breakdown:** Age bucket and category.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-31 — Catalog coverage
- **Numerator or value:** distinct eligible items displayed in a period.
- **Denominator or observation unit:** distinct items eligible during the same period.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** Changing catalog turnover alters the denominator.
- **Action on change:** Inspect source diversity and eligibility.
- **Useful breakdown:** Surface and category.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-32 — Provider concentration
- **Numerator or value:** displayed impressions attributable to the largest provider.
- **Denominator or observation unit:** displayed impressions.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** Provider identifiers and time window must be stable.
- **Action on change:** Inspect source and reranker concentration.
- **Useful breakdown:** Surface and territory.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-33 — Category diversity
- **Numerator or value:** distinct declared categories in a slate.
- **Denominator or observation unit:** one displayed slate.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** Category taxonomy and slate length affect interpretation.
- **Action on change:** Compare with relevance and catalog availability.
- **Useful breakdown:** Slate size and taxonomy version.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-34 — Policy version mismatch
- **Numerator or value:** requests with different eligibility versions across recorded stages.
- **Denominator or observation unit:** requests with versions recorded at both stages.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** Intentional version changes may require a final recheck.
- **Action on change:** Trace version transition and decision time.
- **Useful breakdown:** Service pair and region.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-35 — Final eligibility rejection
- **Numerator or value:** ranked candidates rejected by the final check.
- **Denominator or observation unit:** ranked candidates checked immediately before display.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** Late changes can make some rejection legitimate.
- **Action on change:** Inspect staleness and final-check reasons.
- **Useful breakdown:** Reason and time since retrieval.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-36 — Unauthorized display count
- **Numerator or value:** displayed items that fail an independently reconstructed authorization check.
- **Denominator or observation unit:** displayed items audited.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** Reconstruction itself requires accurate policy history.
- **Action on change:** Escalate immediately and remove affected access.
- **Useful breakdown:** Policy and territory.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-37 — Experiment assignment balance
- **Numerator or value:** assigned users in each experimental arm.
- **Denominator or observation unit:** all assigned users.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** Expected proportions depend on the assignment design.
- **Action on change:** Investigate assignment and logging before interpreting outcomes.
- **Useful breakdown:** Experiment and assignment stratum.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-38 — Fallback invocation rate
- **Numerator or value:** requests served by a declared fallback path.
- **Denominator or observation unit:** eligible recommendation requests.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** Fallback may be deliberate for cold starts.
- **Action on change:** Distinguish intended fallback from service failure.
- **Useful breakdown:** Fallback reason and cohort.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-39 — Candidate stage latency
- **Numerator or value:** elapsed time from retrieval start to candidate completion.
- **Denominator or observation unit:** one request.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** Component percentiles cannot simply be summed into end-to-end percentiles.
- **Action on change:** Trace slow requests and resource contention.
- **Useful breakdown:** Source and region.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.
#### M-40 — End-to-end recommendation latency
- **Numerator or value:** elapsed time from request receipt to recommendation response.
- **Denominator or observation unit:** one request.
- **Interpretation:** Track this measure over a declared observation window rather than treating it as a product outcome by itself.
- **Critical limitation:** A timeout or failed response must be included in service-level accounting.
- **Action on change:** Inspect full traces under representative traffic.
- **Useful breakdown:** Surface and traffic bucket.
- **Comparison discipline:** Keep eligibility policy, event definition and time window comparable before attributing a change to the retrieval method.


### Point-in-time and serving acceptance traces

The following are hypothetical test cases, not executed tests or observed incidents.

#### T-01: Revoked rights
- **Setup:** A title is revoked after retrieval.
- **Required condition:** Final check observes the new policy.
- **Expected behavior:** The item is omitted.
- **Disqualifying behavior:** A displayed revoked item.
- **Diagnostic trace:** Replay the policy change across cache and serving.
#### T-02: Country transition
- **Setup:** A user changes territory between requests.
- **Required condition:** Each request resolves current territory.
- **Expected behavior:** The eligible set changes only as policy permits.
- **Disqualifying behavior:** Old territory reused after the move.
- **Diagnostic trace:** Exercise a territory change with versioned attributes.
#### T-03: Subscription downgrade
- **Setup:** Entitlement changes before a new request.
- **Required condition:** The current entitlement is checked.
- **Expected behavior:** Premium-only items are excluded.
- **Disqualifying behavior:** Cached premium slate returned without recheck.
- **Diagnostic trace:** Replay downgrade and inspect final authorization.
#### T-04: Rights service unavailable
- **Setup:** Authorization dependency times out.
- **Required condition:** The system uses a fail-closed contract.
- **Expected behavior:** No unchecked item is displayed.
- **Disqualifying behavior:** Timeout treated as implicit allow.
- **Diagnostic trace:** Inject a controlled timeout in a test environment.
#### T-05: Availability changes
- **Setup:** Inventory reaches zero after retrieval.
- **Required condition:** The display path rechecks availability.
- **Expected behavior:** The unavailable item is omitted.
- **Disqualifying behavior:** A stale positive inventory response is trusted indefinitely.
- **Diagnostic trace:** Trace inventory event and response timestamps.
#### T-06: No eligible supply
- **Setup:** All items fail current constraints.
- **Required condition:** The eligible set is genuinely empty.
- **Expected behavior:** Return an honest empty or approved response.
- **Disqualifying behavior:** Fabricating an ineligible substitute to fill the slate.
- **Diagnostic trace:** Audit policy state and distinguish service failure.
#### T-07: Filter underfill
- **Setup:** Global ANN returns only excluded items.
- **Required condition:** Eligible exact neighbors exist.
- **Expected behavior:** Underfill is reported and a safe fallback may run.
- **Disqualifying behavior:** Assuming the catalog has no eligible items.
- **Diagnostic trace:** Compare prefilter output with an eligible oracle.
#### T-08: Duplicate canonical items
- **Setup:** Two source IDs map to one item.
- **Required condition:** Canonicalization precedes final budget.
- **Expected behavior:** Only one canonical ID consumes a slot.
- **Disqualifying behavior:** Deduplication occurs after truncation.
- **Diagnostic trace:** Trace raw IDs through union and output.
#### T-09: Missing user history
- **Setup:** A new user has no interaction vector.
- **Required condition:** Cold-start routing is enabled.
- **Expected behavior:** An eligible contextual or popular pool is used.
- **Disqualifying behavior:** Missing embedding is treated as an error without fallback.
- **Diagnostic trace:** Compare cold-user and warm-user routing.
#### T-10: Missing item vector
- **Setup:** A new eligible item has no embedding.
- **Required condition:** Content features are already available.
- **Expected behavior:** It can enter a permitted content source.
- **Disqualifying behavior:** An absent vector permanently suppresses discovery.
- **Diagnostic trace:** Trace item creation through all retrieval sources.
#### T-11: Stale item vector
- **Setup:** An item changes metadata after vector generation.
- **Required condition:** A compatible update is indexed.
- **Expected behavior:** New searches use the current item representation.
- **Disqualifying behavior:** Current metadata and old vector silently diverge.
- **Diagnostic trace:** Compare source, encoder and index versions.
#### T-12: Encoder incompatibility
- **Setup:** A new query encoder is deployed with an old item index.
- **Required condition:** The deployment manifest detects incompatible versions.
- **Expected behavior:** The rollout is blocked or matching artifacts are selected.
- **Disqualifying behavior:** Mismatched vectors silently reach similarity search.
- **Diagnostic trace:** Inspect manifest and compare selected artifact IDs.
#### T-13: Normalization mismatch
- **Setup:** One path normalizes vectors and another does not.
- **Required condition:** The score contract specifies both sides.
- **Expected behavior:** Exact and ANN use the intended metric consistently.
- **Disqualifying behavior:** Comparing cosine scores with raw inner products.
- **Diagnostic trace:** Recompute sample scores from recorded vectors.
#### T-14: Tied exact scores
- **Setup:** Two eligible items have equal exact score.
- **Required condition:** A stable tie rule is specified.
- **Expected behavior:** The oracle returns the same ordered list on replay.
- **Disqualifying behavior:** Arbitrary ties appear as ANN errors.
- **Diagnostic trace:** Recompute using deterministic item-ID tie breaking.
#### T-15: Fewer than k eligible
- **Setup:** The eligible universe contains two items and k is three.
- **Required condition:** Both items have defined scores.
- **Expected behavior:** The exact list has length two.
- **Disqualifying behavior:** Dividing ANN overlap by three instead of two.
- **Diagnostic trace:** Verify actual exact-list length and report it.
#### T-16: Zero relevant labels
- **Setup:** A mature request has no observed relevant items.
- **Required condition:** The per-request denominator is zero.
- **Expected behavior:** Per-request recall is undefined.
- **Disqualifying behavior:** Substituting a perfect score without disclosure.
- **Diagnostic trace:** Report zero-label count and aggregation convention.
#### T-17: Immature outcome window
- **Setup:** A recent request has not completed its observation horizon.
- **Required condition:** Label maturity is recorded.
- **Expected behavior:** Exclude it from a mature-cohort primary analysis.
- **Disqualifying behavior:** Calling delayed outcomes negative labels.
- **Diagnostic trace:** Compare request age with horizon and ingestion watermark.
#### T-18: Future user action
- **Setup:** A click occurs after the request timestamp.
- **Required condition:** Feature availability is tracked separately from outcome time.
- **Expected behavior:** The click may become a label but not an input to that decision.
- **Disqualifying behavior:** A training join loads that future click as a feature.
- **Diagnostic trace:** Inspect feature event and availability timestamps.
#### T-19: Late-arriving feature
- **Setup:** An event occurred before the request but arrived afterward.
- **Required condition:** Availability time is part of reconstruction.
- **Expected behavior:** The event is not treated as available at decision time.
- **Disqualifying behavior:** Using only event time to justify feature availability.
- **Diagnostic trace:** Compare event time and ingestion time against request time.
#### T-20: Incomparable cohorts
- **Setup:** A system excludes a portion of requests before evaluation.
- **Required condition:** A common request cohort is available.
- **Expected behavior:** Both systems use the same request IDs.
- **Disqualifying behavior:** A score gain comes from different query inclusion.
- **Diagnostic trace:** Produce an explicit request-level population flow.
#### T-21: Incomparable relevance
- **Setup:** Two methods use different outcome horizons.
- **Required condition:** Both label rules can be reconstructed.
- **Expected behavior:** Recompute both under a common definition.
- **Disqualifying behavior:** Comparing seven-day and one-day recall directly.
- **Diagnostic trace:** Join both outputs to a common matured label snapshot.
#### T-22: Different authorization sets
- **Setup:** The exact baseline scores a global catalog.
- **Required condition:** Candidate method is filtered to territory.
- **Expected behavior:** Rebuild the oracle using the same eligible universe.
- **Disqualifying behavior:** Comparing filtered ANN to unrestricted exact neighbors.
- **Diagnostic trace:** Compare eligible ID snapshots and tie rules.
#### T-23: Unlogged impressions
- **Setup:** Clicks are present but corresponding rendered impressions are missing.
- **Required condition:** Client and server event paths can be audited.
- **Expected behavior:** Separate missing logs from actual unexposed items.
- **Disqualifying behavior:** Inferring that a click proves complete impression logging.
- **Diagnostic trace:** Trace request, render and click identifiers.
#### T-24: Position confounding
- **Setup:** One source appears more often at the top position.
- **Required condition:** Position is logged and assignment can be randomized.
- **Expected behavior:** Compare policies under appropriate experimental design.
- **Disqualifying behavior:** Attributing raw click-rate differences entirely to relevance.
- **Diagnostic trace:** Inspect position distributions and randomized arms.
#### T-25: Logging probability error
- **Setup:** An exploration event logs the wrong probability.
- **Required condition:** Assignment records and randomizer configuration are available.
- **Expected behavior:** Correct the instrumentation and assess affected analyses.
- **Disqualifying behavior:** Using inverse-propensity weights with erroneous probabilities.
- **Diagnostic trace:** Audit logged probabilities against the assignment implementation.
#### T-26: No propensity support
- **Setup:** A historical policy never displays a target action.
- **Required condition:** The target policy would display that action.
- **Expected behavior:** Recognize that historical data cannot identify its outcome without additional assumptions.
- **Disqualifying behavior:** Assigning infinite or arbitrary weights to zero probability.
- **Diagnostic trace:** Document support gaps and design safe exploration.
#### T-27: Popularity feedback loop
- **Setup:** Popular items repeatedly receive most exposures.
- **Required condition:** Candidate and display exposures are both logged.
- **Expected behavior:** Separate retrieval concentration from ranking concentration.
- **Disqualifying behavior:** Equating more clicks with unbiased preference.
- **Diagnostic trace:** Compare eligible coverage and provider mix by stage.
#### T-28: Over-concentrated slate
- **Setup:** Many candidates have the same provider.
- **Required condition:** A diversity constraint is explicitly defined.
- **Expected behavior:** Apply the constraint only within eligible inventory.
- **Disqualifying behavior:** Violating rights to satisfy a diversity target.
- **Diagnostic trace:** Inspect source mix, slate composition and policy exclusions.
#### T-29: Index rebuild contention
- **Setup:** A refresh overlaps high request traffic.
- **Required condition:** Build and serve resource usage can be separated.
- **Expected behavior:** Identify whether contention explains latency.
- **Disqualifying behavior:** Assuming every latency spike is caused by the model.
- **Diagnostic trace:** Compare timelines for build, traffic and resource saturation.
#### T-30: Partial artifact rollback
- **Setup:** Model rollback leaves a newer index active.
- **Required condition:** Artifact compatibility is versioned.
- **Expected behavior:** Restore an approved compatible set.
- **Disqualifying behavior:** Rolling back only the query model.
- **Diagnostic trace:** Trace model and index IDs in served requests.

**Takeaway:** Diagnose competing causes with a discriminating trace before changing retrieval, ranking or policy.

## 9. Complexity & Cost Analysis

### 9a. Algorithmic Complexity

Let N be the number of eligible items, d the embedding dimensions, k the output size, and n the total source-list entries.
An exact brute-force dot-product scan requires O(Nd) arithmetic and can retain the top k with O(k) auxiliary storage using a heap, plus O(N log k) heap operations in a straightforward implementation.
Stored dense vectors require O(Nd) numbers; input storage is distinct from auxiliary heap memory.
A full sort instead uses O(N log N) comparison work and requires a suitable score representation.
ANN build, memory, search and update complexity depend on index family, parameters, dataset and implementation; no universal sublinear worst-case claim is made.
Index refresh requires embedding generation, transfer, build or update, validation, and replication; recovery and rebuild may overlap with live traffic.
Distributed retrieval adds fan-out, network transfer, merge work, skew and stragglers; index-wide latency cannot be inferred from a single shard's average.
Cache hits can reduce work, but policy checks and cache invalidation remain mandatory.

### 9b. Back-of-Envelope Sizing

**ILLUSTRATIVE inputs:** 1,000,000 item vectors, each with 128 float32 components, four bytes per component, and two complete replicas.
**Validity:** dense uncompressed vector payload only; no ANN graph, item IDs, metadata, allocator overhead or query features.
**Derivation:** item count times dimensions gives component count; multiplying by bytes gives payload; replication doubles it.
**[S1] FORMULA:** payload bytes = items * dimensions * bytes_per_component * replicas.
**WHERE:** each quantity is a nonnegative count and every replica stores the complete payload.
**CONSTRAINT:** homogeneous vectors with no compression, padding or sharding-dependent duplication.
**NUMERICAL EXAMPLE:** 1,000,000 * 128 * 4 * 2 = 1,024,000,000 bytes = 1.024 GB decimal, approximately 0.954 GiB.
**INTERPRETATION:** This is a lower-bound payload estimate under these assumptions, not a provisioned-memory recommendation.
**Independent arithmetic check:** one replica stores 512,000,000 bytes; two store 1,024,000,000 bytes.
**Boundary:** vector quantization, mixed dimensions, replicas with unequal contents, metadata and build-time copies change the estimate.
At 100 hypothetical requests/s, an exhaustive scan over one million items entails 100,000,000 item scores/s before filtering and batching assumptions; 128 components imply 12.8 billion component multiplications/s if every score uses all components.
These are operation counts, not measured throughput, FLOPs, power, or latency; hardware, batching, caching, concurrency and memory traffic determine real behavior.
**Cost model:** If provisioned memory costs a hypothetical $10 per GB-month, charging exactly 1.024 decimal GB for one month gives $10.24/month in payload-only memory charges.
This is not a real price quotation; storage, replicas beyond two, compute, network, logs, idle capacity and labor are excluded.
A useful-outcome unit cost requires an observed outcome denominator; no cost per successful recommendation is claimed.
**Environment and budgets:** Hardware, software, latency SLO, peak traffic and index type are OPEN and must be supplied before capacity or currency estimates can support deployment.
**Sensitivity:** Doubling dimensions to 256 doubles payload to 2.048 GB with all other assumptions fixed.
Measure build-time peak memory, index footprint, freshness lag, end-to-end tail latency and capacity under representative bursts before provisioning.
**Takeaway:** Arithmetic produces an assumption-bound capacity floor, not a benchmark or operational budget.

## 10. Key Theorems / Results Reference

| Result | Statement and hypotheses | Consequence | Does not imply | Source |
|---|---|---|---|---|
| Set containment bound | For displayed slate D contained in C and relevance set R, D intersect R is contained in C intersect R | Displayed relevant hits cannot exceed retrieved relevant hits | A larger candidate set guarantees better ranked outcomes | DERIVED from set containment |
| Exact ranking by scan | Evaluating the fixed score for every eligible item and selecting k greatest with fixed ties returns exact top-k under that score | Supplies a matched oracle for ANN | Human relevance or policy utility is exact | DERIVED from exhaustive enumeration |
| Exposure selection | Feedback is observed under a serving policy rather than for all possible items | Logged clicks are not automatically unbiased preference labels | Every bias can be repaired without overlap or logging | SOURCED: [Joachims et al., Unbiased Learning-to-Rank with Biased Feedback](https://doi.org/10.1145/3018661.3018699) |
| Randomized assignment | Under a valid randomized experiment and appropriate analysis, assignment supports causal comparison of assigned policies | Online experiments can assess end-to-end impact | Every online difference is significant or free from interference | SOURCED: [Kohavi et al., Online Controlled Experiments](https://doi.org/10.1017/9781108653985) |

**References and claim boundaries:** [Hu, Koren and Volinsky, Collaborative Filtering for Implicit Feedback Datasets](https://doi.org/10.1109/ICDM.2008.22) describes a model for implicit observations; it does not validate the illustrative workflow here.
[Covington et al., Deep Neural Networks for YouTube Recommendations](https://doi.org/10.1145/2959100.2959190) is an original research example of candidate generation and ranking; its historical architecture is not a claim about any present deployment.
[Faiss project](https://faiss.ai/) documents similarity search tools; specific deployed index behavior is version-dependent and not verified here.
[NIST Privacy Framework](https://www.nist.gov/privacy-framework) offers a privacy risk-management framework; legal applicability and organization-specific controls require review.
All links identify primary research, project or institutional sources; no external benchmark is imported as a result for this example.
Publication dates and software versions should be checked by a reviewer before external reliance; empirical workload and product results remain OPEN.
**Takeaway:** Exactness, causal validity and eligibility are separate claims with separate assumptions.

## 11. Interview / Exam Questions

### Q01 — Level 1 — Definition / Recall

**Prompt:** What distinguishes retrieval from ranking?

**Answer:** Retrieval selects a manageable candidate set; ranking orders those candidates for a display objective. A ranker cannot rank an item absent from its input.

**Scoring:** Full credit requires the stated assumptions and an explicit boundary; a confident claim without them is partial.

### Q02 — Level 1 — Definition / Recall

**Prompt:** Why is an unseen item not a negative label?

**Answer:** It may never have been exposed or visible; even an impression without a click can reflect position or context rather than dislike.

**Scoring:** Full credit requires the stated assumptions and an explicit boundary; a confident claim without them is partial.

### Q03 — Level 1 — Definition / Recall

**Prompt:** How does ANN recall differ from relevance recall?

**Answer:** ANN recall measures overlap with exact nearest neighbors under a fixed score; relevance recall measures overlap with labeled eligible items.

**Scoring:** Full credit requires the stated assumptions and an explicit boundary; a confident claim without them is partial.

### Q04 — Level 2 — Derivation / Apply

**Prompt:** For R={B,D} and C={A,B,C}, what is recall@3?

**Answer:** The intersection is {B}; its size is one and |R| is two, so recall@3 is 1/2 = 0.5, provided the eligible universe and label horizon match.

**Scoring:** Full credit requires the stated assumptions and an explicit boundary; a confident claim without them is partial.

### Q05 — Level 2 — Derivation / Apply

**Prompt:** Two requests have one hit each and relevant counts two and one. Compute macro and micro recall.

**Answer:** Macro is (1/2 + 1)/2 = 0.75; micro is (1+1)/(2+1) = 2/3. Their weighting differs.

**Scoring:** Full credit requires the stated assumptions and an explicit boundary; a confident claim without them is partial.

### Q06 — Level 2 — Derivation / Apply

**Prompt:** Prove that a ranker cannot display a relevant item absent from its candidate input.

**Answer:** For display D subset C, every element of D intersect R is also in C intersect R. Therefore its number of relevant hits cannot exceed the candidate hit count.

**Scoring:** Full credit requires the stated assumptions and an explicit boundary; a confident claim without them is partial.

### Q07 — Level 2 — Derivation / Apply

**Prompt:** What happens to recall for an empty relevant set?

**Answer:** The ratio has zero denominator and is undefined. Report exclusions and label-maturity status rather than silently setting it to zero.

**Scoring:** Full credit requires the stated assumptions and an explicit boundary; a confident claim without them is partial.

### Q08 — Level 2 — Derivation / Apply

**Prompt:** How much raw vector payload do one million 128-dimensional float32 vectors require with two replicas?

**Answer:** One million times 128 times four times two is 1,024,000,000 bytes, or 1.024 decimal GB, before index and runtime overhead.

**Scoring:** Full credit requires the stated assumptions and an explicit boundary; a confident claim without them is partial.

### Q09 — Level 3 — Judgement / Design / Debug

**Prompt:** How should a new user be served without history?

**Answer:** Use an eligible contextual or popularity baseline, optionally incorporating consented explicit preferences; monitor cohort outcomes and do not infer a personal preference from missing data.

**Scoring:** Full credit requires the stated assumptions and an explicit boundary; a confident claim without them is partial.

### Q10 — Level 3 — Judgement / Design / Debug

**Prompt:** Why can global ANN followed by filtering underfill even when eligible results exist?

**Answer:** The global top results can all be ineligible; eligible items below the global cutoff are never returned. Use an eligible oracle, filter-aware retrieval or bounded over-retrieval with a safe fallback.

**Scoring:** Full credit requires the stated assumptions and an explicit boundary; a confident claim without them is partial.

### Q11 — Level 3 — Judgement / Design / Debug

**Prompt:** Recall rises but retention falls. What next?

**Answer:** Do not assume recall caused retention. Pause or gate rollout, verify experimental assignment and data quality, compare guardrails, and investigate ranking, quality and affected cohorts.

**Scoring:** Full credit requires the stated assumptions and an explicit boundary; a confident claim without them is partial.

### Q12 — Level 3 — Judgement / Design / Debug

**Prompt:** A rights revocation arrives after the index refresh. What must protect the user?

**Answer:** A current authoritative eligibility check near display must remove the item. Refresh and cache invalidation are necessary but cannot replace the final check.

**Scoring:** Full credit requires the stated assumptions and an explicit boundary; a confident claim without them is partial.

### Q13 — Level 3 — Judgement / Design / Debug

**Prompt:** How would you compare exact and approximate methods fairly?

**Answer:** Use identical request vectors, item vectors, score, tie policy, eligibility snapshot and cutoff; evaluate filtered output against exact eligible top-k and report underfill.

**Scoring:** Full credit requires the stated assumptions and an explicit boundary; a confident claim without them is partial.

### Q14 — Level 3 — Judgement / Design / Debug

**Prompt:** How can exploration be assessed without confusing position with preference?

**Answer:** Log randomized assignment, rendered position and exposure; use a suitable controlled design with consent and guardrails. Propensity methods need correct probabilities and support.

**Scoring:** Full credit requires the stated assumptions and an explicit boundary; a confident claim without them is partial.

### Q15 — Level 3 — Judgement / Design / Debug

**Prompt:** What would invalidate an offline gain?

**Answer:** A future feature, changed eligible universe, immature labels, unequal query cohort or inconsistent event logging can invalidate the comparison; rerun only with matched point-in-time inputs and audited exclusions.

**Scoring:** Full credit requires the stated assumptions and an explicit boundary; a confident claim without them is partial.

### Q16 — Level 3 — Judgement / Design / Debug

**Prompt:** How would you balance diversity and relevance?

**Answer:** Define a slate diversity objective or constraint within the authorized candidate set, report relevance and business tradeoffs by cohort, and validate the combined policy online.

**Scoring:** Full credit requires the stated assumptions and an explicit boundary; a confident claim without them is partial.

**Takeaway:** Strong answers identify the universe, time boundary, observation mechanism and a falsifiable validation plan.

## 12. Quick Reference Card

| Item | Value |
|---|---|
| Topic | Choose and evaluate authorized retrieval candidates before ranking. |
| Core formula | Relevant recall@k = eligible relevant IDs in candidate top-k / eligible relevant IDs, for a nonempty matched relevant set. |
| Key assumption | Systems use the same as-of eligibility, query cohort, label horizon, cutoff and relevance definition. |
| Complexity | Exact scan is O(Nd + N log k) with heap selection and O(k) auxiliary space, for N eligible d-dimensional vectors. |
| Use when | A candidate set must fit a downstream ranking budget while preserving eligible relevance. |
| Avoid when | Authorization, outcome maturity, compatible vectors or a comparable evaluation population cannot be established. |
| Go deeper | [Collection index](INDEX.md); ranking metrics and exploration sheets are planned, not yet available. |
