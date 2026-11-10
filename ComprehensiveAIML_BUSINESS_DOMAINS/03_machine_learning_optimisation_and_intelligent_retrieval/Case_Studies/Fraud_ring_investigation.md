---
case_id: "CASE-G15-01"
case_title: "Fraud-ring investigation under a fixed investigator-hour budget"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "draft"
content_status: "draft"
case_type: "predictive-ml"
domain: "Insurance claims fraud — special-investigations ring triage on entity-resolved actor graphs"
primary_pattern: "graph-evaluation"
secondary_patterns: ["entity-resolution", "capacity-constrained-triage"]
evidence_level: "illustrative"
source_basis: "composite teaching case; organization, dates, and every figure constructed"
as_of_date: "2026-11-10"
owner_role: "Special Investigations Unit analytics lead"
source_index: "INDEX.md"
source_status: "OPEN — plan PLAN-M15-01 registered in the collection catalog; no internal ledger, pilot artifact, or external dataset is available for inspection"
review_trigger: "Change in eligible claim volume, investigator-hour budget, adjudication lag, hub degree threshold, merge gate, or the definition of a reviewed case"
---
# Fraud-ring investigation under a fixed investigator-hour budget

## 1. The One-Line Framing

> Given **an eligible insurance claim and its point-in-time actor neighbourhood**, decide **whether the claim enters the next investigator-hour allocation and, if so, as a single-claim review or inside a multi-claim ring bundle**, for **the Special Investigations Unit (SIU) analytics lead who owns a fixed annual investigation budget**, to improve **confirmed fraud findings per investigator-hour without raising unnecessary contact with innocent policyholders**, subject to **a hard 4,800-hour annual review budget and a 212-day label-maturity horizon**, compared with **the existing rules-based claim queue**.

This is an **ILLUSTRATIVE composite teaching case**.
The insurer, its claims, its investigators, its timeline, and every number in this file are constructed for instruction.
The methodology — entity-resolved typed graphs, bounded neighbourhood expansion, precision at fixed capacity, adjudication-time-correct neighbour labels — is drawn from ordinary graph learning and fraud-analytics practice, but no claim here describes a real organization, a real fraud ring, or a measured deployment.
Every table row carries its own evidence label so that the arithmetic can be trusted while the setting is not.

**Task category:** binary prioritization (classification used as a ranking) fused with a batching decision.
**Decision frequency:** the ranking is rescored nightly; the capacity allocation is set quarterly.
**Primary success condition, in plain language:** for the same 300 reviewed claims, more of them should turn out to be confirmed fraud, and each confirmation should consume fewer investigator hours.
The full metric definition — numerator, denominator, population, window, comparator — is deferred to Section 9, because a one-line success condition is a slogan until it is given a denominator.

**Primary reusable pattern:** `graph-evaluation`.
**Secondary patterns:** `entity-resolution` and `capacity-constrained-triage`.

### The decisive tradeoff

The obvious framing of this problem is "predict fraud better with a graph".
That framing produces the wrong investment.
The graph's measured contribution to ranking discrimination is real but modest: 46.0% confirmed precision against 32.0% for a claim-level tabular model, a 14.0-percentage-point gap whose 95% interval still spans 6.3 to 21.7 points at the pilot's sample size.
The graph's larger contribution is structural, not statistical: it changes the **unit of work** from one claim to one ring bundle, and a 5.77-claim bundle reviewed as a unit consumed 9.5 hours instead of 23.08 hours.
That is a 2.43-fold reduction in hours per reviewed claim, which frees 58.8% of the review budget for depth.
A team that buys only the better score and keeps reviewing claims one at a time captures the small benefit and misses the large one.
A team that buys only the bundling and refuses the bounded neighbourhood discipline inherits a 12-fold closure blowup, which Section 6 quantifies.

### Why the naive framing fails

Three separate questions hide inside "detect rings": which records describe the same actor, which actors form a suspicious structure, and which claims are worth an investigator's hour.
The first is entity resolution and it is reversible bookkeeping, not prediction.
The second is motif detection on a snapshot whose edges must all be available at decision time.
The third is a capacity allocation problem in which the currency is hours, not scores.
Answering only the third with a better classifier leaves the first two to silently corrupt the features, which is exactly the failure this case documents in Section 12.

**Takeaway:** the decision is an hour-allocation decision, and the graph's main lever is bundle size, not AUC.

## 2. Context & Background

### 2.1 The workflow the decision sits inside

A property-and-casualty insurer receives a claim, records the loss, links the claim to the people and suppliers involved, and either pays, declines, or refers it.
A referral goes to the SIU, where an investigator assembles evidence, interviews, requests documents, and writes a disposition.
A disposition is one of: confirmed fraud, not fraud, or inconclusive pending further evidence.
Confirmed cases may later move to recovery, subrogation, or a policy action; those downstream steps are governed by separate procedures and are outside this case.
The analytics team supplies a ranked queue; the SIU owns the review decision; compliance and legal own what may be said to a policyholder.
No score in this case authorizes an action against a customer, and no score is a finding of fraud.

### 2.2 Why the current process persists

The existing queue is a rules engine with fourteen hand-maintained triggers: claim value above a band, days-to-report below a threshold, prior claim count, a supplier on a watch list, and similar conditions.
It is auditable, explainable to a regulator, and cheap to change one rule at a time.
It is also blind to structure: two claims that share a bank account, a repair supplier, and a filing week are scored as two independent rows, and the strongest available evidence — that they are connected — is never computed.
The rules queue persisted not because anyone believed it was optimal but because each rule had an owner, a justification, and a documented effect, while a graph feature had no owner and no audit story.

### 2.3 Scale and timing

| Quantity | Illustrative value | Note |
|---|---|---|
| Raw claim records in the event window | 268,400 | 2024-01-01 to 2024-12-31, event time |
| Eligible claims after exclusions | 240,000 | exclusion ledger in Section 5a |
| Claim volume per business day | about 941 | 240,000 / 255 working days |
| Peak-day claim volume | about 2,000 | storm events, roughly 2.1× mean |
| Annual investigator-hour budget | 4,800 hours | the binding constraint |
| Mean hours per single-claim review | 4.0 hours | assembly, interview, disposition, documentation |
| Reviews per year under the rules queue | 1,200 | 4,800 / 4.0 |
| Confirmed fraud per 1,200 reviews, rules queue | 318 | 26.5% |

The 4,800-hour figure is the number that makes this a decision rather than a research project.
At 4,800 hours the team can review 1,200 of 240,000 claims, which is one half of one percent.
Any method that raises precision without changing hours per review competes for a fixed 1,200-slot budget.
Any method that lowers hours per review competes for a growing slot count, and that is a fundamentally different prize.

### 2.4 The label-maturity horizon

A claim is only usable for training when its disposition is final and enough time has passed for late evidence to surface.
This case uses a 212-day minimum maturity: claims filed on 2024-12-31 have 212 days of observation before the 2025-07-31 label cutoff.
Claims in the last quarter of 2024 therefore sit close to the boundary, and any claim whose investigation was still open at cutoff is treated as unresolved rather than negative.
That single rule removes a large class of silent label errors, because an open investigation is not evidence of innocence.

### 2.5 Stakeholders and their separate interests

| Role | Owns | Would object to |
|---|---|---|
| SIU analytics lead | queue design, capacity model | a queue that cannot be explained per case |
| SIU investigators | disposition quality, cycle time | bundles that force unrelated work into one file |
| Compliance | permissible use of data, customer contact | graph links used as if they were proof |
| Model risk | method challenge, validation record | undocumented thresholds and unrepeatable snapshots |
| Data platform | lineage, snapshot versioning | unversioned feature recomputation |

### 2.6 What was not available

No production graph, no investigator time study, and no adjudication ledger were accessible while writing.
Sections 5d and 5e record that gap explicitly rather than filling it with plausible detail.
The consequence is that every operational quantity here is a stated assumption, and only the arithmetic is independently checkable.

**Takeaway:** the workflow supplies a hard budget, an explainability obligation, and a delayed, selectively observed label — three constraints that together rule out a pure ranking exercise.

## 3. The Business Question

### 3.1 The question in the decision owner's words

This is an **illustrative stakeholder question**, not a quotation from a real person:

> "We have 4,800 investigator-hours and no prospect of more. Should we keep reviewing claims one at a time from a better-ranked list, or should we start reviewing connected groups — and if we group, how do we stop the grouping from dragging half the book into every file?"

### 3.2 The decision contract

| Contract field | Definition |
|---|---|
| Question | Does ring-bundled review at equal reviewed depth produce more confirmed findings per hour than claim-level review, and is it safe to adopt? |
| Decision owner | SIU analytics lead, with sign-off from the model-risk committee |
| Trigger | The rules queue's 26.5% confirmed rate has been flat for four quarters while claim volume rose 9% |
| Deadline | Before the next annual capacity plan, which locks investigator headcount |
| Available actions | keep the rules queue; adopt the tabular model claim-by-claim; adopt graph features without bundling; adopt graph features with bundling; defer and run a larger pilot |
| Value | Each additional confirmed case avoids loss exposure and deters repeat behaviour; each hour not spent on a false positive is an hour available for a real case |
| Downside | Contacting an innocent policyholder, investigator distrust, a queue that cannot be defended in an appeal |
| Objective | Maximize confirmed findings per investigator-hour at a fixed annual hour budget |
| Hard constraints | 4,800 hours per year; no automated adverse action; every queue decision reproducible from a named snapshot |
| Scope | Property claims above a value floor, one country, one claims platform, calendar year 2024 event time |
| Out of scope | Litigation strategy, recovery amounts, policy cancellation, criminal referral, medical or bodily-injury claims |

### 3.3 Objective versus proxy

The business objective is loss avoided per investigator-hour.
The measurable proxy is confirmed fraudulent claims per reviewed claim, at equal reviewed depth, with hours tracked separately.
These can disagree in two specific ways, and the case must name both.

First, a review can be "confirmed" for reasons unrelated to the claim's intrinsic risk — an investigator under pressure may close marginal cases as confirmed.
That inflates the proxy without reducing loss, so the guardrail must include an appeal-overturn rate.
Second, a high-precision queue can be a low-recall queue: 1,200 reviews out of 240,000 claims cannot find most fraud no matter how good the ranking is.
Section 9 shows that estimating population recall properly would cost 4.24 times the entire review budget, which converts recall from a metric into an unaffordable one.
The case therefore reports precision at capacity as the primary metric and treats recall as unidentified rather than estimated.

**Takeaway:** optimize findings per hour, measure precision at equal depth, and refuse to publish a recall number the budget cannot support.

## 4. Assumptions & Constraints

### 4.1 Assumption register

| ID | Assumption | Type | How it is checked | Status | If wrong | Fallback |
|---|---|---|---|---|---|---|
| A-01 | Every claim has an immutable claim identifier and an event timestamp | data | schema audit against the claims ledger | unverified | features cannot be rebuilt point-in-time | use rules queue only |
| A-02 | Actor links carry an availability timestamp distinct from the event timestamp | data | lineage walk through ingestion | unverified | late links leak into historical features | restrict to links with availability lag under 1 day |
| A-03 | Dispositions become final within 212 days for at least 95% of reviewed claims | label | survival curve of time-to-disposition | unverified | labels misclassify open cases as negative | extend maturity horizon, shrink window |
| A-04 | Investigator time per single-claim review is 4.0 hours with modest variance | business | time study over four weeks | unverified | capacity model mis-sizes the queue | schedule in hours, not slots |
| A-05 | A ring bundle shares evidence, so marginal claims cost less than 4.0 hours each | business | instrumented pilot timing | unverified | bundling yields no hour saving and the case collapses | revert to claim-by-claim |
| A-06 | Reviewed outcomes are representative of the reviewed stratum | evaluation | compare queue arms on identical eligibility | unverified | cross-arm comparison is confounded | use randomized arm assignment |
| A-07 | Actor degree above 500 indicates shared infrastructure rather than shared intent | model | degree distribution plus manual review of the top 27 providers | unverified | a genuine ring with a large supplier is suppressed | lower the threshold with a manual override |
| A-08 | Two independent matching identifiers are enough to merge claimant records | data | audited merge sample of 300 | unverified | over-merge creates spurious rings | raise the gate, add a third identifier |

### 4.2 Constraint register

| ID | Constraint | Source | Status | Consequence if violated |
|---|---|---|---|---|
| C-01 | 4,800 investigator-hours per year | headcount plan | confirmed for the scenario | queue design is meaningless without a budget |
| C-02 | No automated adverse action from a score | compliance policy | confirmed for the scenario | regulatory exposure and reputational harm |
| C-03 | Every queue position reproducible from a named snapshot version | model-risk policy | confirmed for the scenario | appeals and audits cannot be defended |
| C-04 | Customer contact requires an investigator decision, not a model output | compliance policy | confirmed for the scenario | unnecessary contact with innocent policyholders |
| C-05 | Personal data retained only for the documented investigation purpose | privacy policy | confirmed for the scenario | unlawful secondary use |
| C-06 | Graph traversal must complete within the nightly window | platform SLA | confirmed for the scenario | stale queue at start of business day |

### 4.3 The three assumptions that can reverse the design

A-05 is the first.
If bundling does not reduce hours per claim, the entire economic argument evaporates and the graph is worth only its 14-point precision gain, which at 1,200 slots is 168 claims and may not justify the engineering.
Section 6 tests this directly by computing the break-even bundle cost, which turns out to be 23.08 hours; the measured 9.5 hours leaves a 2.43-fold margin, so the assumption survives a large error.

A-03 is the second.
If dispositions routinely take longer than 212 days, then "unresolved" claims would be labelled negative and every precision figure in the case would be biased downward by an unknown amount.
The discriminating check is a survival curve of time-to-disposition by claim type, and the bounded fix is to drop the affected quarter rather than impute.

A-07 is the third.
If the degree threshold is set too low, a genuine ring that happens to route through a busy supplier disappears from the evidence.
If it is set too high, one supplier merges 31,400 claims into a single component and every claim in that component looks connected to every other.
Section 6 shows the component size collapsing from 31,400 to 4,900 when 27 hub providers are excluded, which is a 6.41-fold reduction and the clearest available evidence that the threshold is doing real work.

**Takeaway:** the design hinges on the bundle-hour assumption and the hub threshold, and both have discriminating checks rather than assertions.

## 5. Data, Tools & Method

### 5a. Data Specification

#### Unit of analysis and grain

The ranking unit is **one eligible claim**.
The review unit is **one bundle**, defined as a connected component of the claim-actor graph restricted to the claims that share at least one non-hub actor with the seed claim, capped at eleven claims.
The evaluation unit for precision is **one reviewed case**, where a multi-claim bundle counts once for the investigator and each claim inside it carries its own disposition.
That distinction matters: a bundle with nine claims and one investigator decision produces nine labelled claims and one review hour block, so precision per claim and precision per review are different denominators and must never be swapped.

#### Volume and window

| Field | Value |
|---|---|
| Claim records, raw | 268,400 |
| Claims, eligible | 240,000 |
| Event window | 2024-01-01 through 2024-12-31, event time, UTC |
| Label cutoff | 2025-07-31 |
| Minimum label maturity | 212 days |
| Actor records | 61,400 |
| Actor-identifier observations | 148,900 |
| Graph edges, after hub exclusion | about 1.9 million |
| Reviewed claims in the pilot stratum | 300 per arm |
| Dispositioned claims in the retrospective pool | 6,800 |

The retrospective pool of 6,800 dispositioned claims is what the pilot sampled from, and it is far smaller than the 240,000 eligible claims because only referred claims receive a disposition.
That asymmetry is the central data fact of this case: 97.2% of eligible claims have no disposition at all, and those claims are not negatives, they are unobserved.

#### Inputs

| Input group | Fields | Availability rule |
|---|---|---|
| Claim attributes | value, peril code, days-to-report, channel, policy tenure | available at claim recording |
| Actor attributes | role in claim, identifier count, first-seen date | available at actor creation |
| Relationship edges | claim-actor, actor-identifier, supplier-link | available at edge ingestion, with lag |
| Neighbourhood features | hub-filtered degree, shared-identifier count, motif counts | computed at snapshot |
| Prior outcome features | prior confirmed count for an actor, available only with delay | must respect label availability |

#### Outcome definition

A claim is **confirmed** when an investigator closes it with a documented fraud finding and the finding survives the internal quality review.
A claim is **not fraud** when the investigator closes it as not fraud.
A claim is **unresolved** when the investigation is open or the finding is provisional at cutoff.
Unresolved claims are excluded from both numerator and denominator; they are not folded into negatives.
The maturity requirement is not cosmetic: folding unresolved claims into the negative class would add roughly 1 in 12 reviewed claims to the denominator as false negatives, which at 300 reviews would change precision by several points in an unpredictable direction.

#### Population and exclusions

| Exclusion | Count | Reason |
|---|---|---|
| Claims below the 2,000 currency-unit value floor | 22,100 | outside the review remit |
| Bodily-injury and medical claims | 4,900 | separate assessment process |
| Claims lacking a stable identifier | 800 | cannot be joined point-in-time |
| Duplicates from re-filed claims | 600 | resolved to a surviving claim |
| Total excluded | 28,400 | 268,400 − 28,400 = 240,000 |

#### Join semantics

Every join is an as-of join on availability time, not on event time.
An actor edge that was created at 14:02 on the claim date is visible to a snapshot taken after 14:02 and invisible to a snapshot taken at 09:00.
Where event time and availability time differ, the availability time governs.
This single rule is what separates a defensible retrospective from a leaked one, and it is also the source of the largest gap between an offline rebuild and an online decision.

#### Splits

| Split | Construction | Purpose |
|---|---|---|
| Train | 2024-01-01 to 2024-06-30 dispositions | fit the ranking model |
| Validation | 2024-07-01 to 2024-09-30 dispositions | threshold and hub-threshold selection |
| Test | 2024-10-01 to 2024-12-31 dispositions | reported precision at capacity |
| Inductive holdout | actors first seen after 2024-10-01 | test unseen-actor behaviour |

The split is temporal and entity-disjoint: no actor appears in both train and test with a mature label on both sides.
A random split would let a single actor's earlier confirmed claim teach the model about that actor's later claim, which is precisely the leakage pattern that makes graph results look excellent and deploy terribly.

### 5b. Tools & Stack

| Component | Role | Version/status |
|---|---|---|
| Columnar warehouse | holds the claim ledger and actor tables | version not stated; scenario |
| Snapshot exporter | materialises the nightly graph snapshot | proposed |
| Graph store | adjacency and typed edge queries | proposed; engine unspecified |
| Python analysis environment | feature assembly, model fitting, metrics | proposed |
| Gradient-boosted trees | claim-level and graph-feature tabular models | proposed |
| Graph aggregation library | neighbourhood mean and motif counts | proposed |
| Case management system | queue, bundle assignment, disposition recording | incumbent |
| Dashboard | hours consumed, precision, appeal overturns | proposed |

No tool in this table has been run against the described workload.
The stack is listed because the case's cost model depends on which operations are cheap: adjacency expansion is cheap in a graph store and expensive in a relational join chain, and Section 9 sizes the difference.

### 5c. Method Summary

The pipeline has six stages.

```text
[Claims + actors + identifier observations]
        |
        v
[Entity resolution: blocked comparison, audited merges]   <-- reversible
        |
        v
[Typed claim-actor graph snapshot, availability-filtered]
        |
        v
[Hub exclusion at degree > 500]
        |
        v
[Bounded neighbourhood features: degree, shared ids, motifs]
        |
        v
[Ranking model -> claim score]
        |
        v
[Bundle construction: connected component capped at 11]
        |
        v
[Capacity allocation in hours -> queue]
```

**Stage 1, entity resolution.** Records that might describe the same actor are compared only within blocks sharing an exact key; a merge requires two independent matching identifiers. Merges are reversible and carry provenance.

**Stage 2, graph construction.** Edges are typed: claim-to-actor, actor-to-identifier, claim-to-supplier. A claim-to-actor edge is directed; a shared-identifier relation is derived, not stored as a payment.

**Stage 3, hub exclusion.** Actors with degree above 500 are removed from the relation used for neighbourhood features, but retained as claim attributes. This is the difference between "this claim is connected to the world" and "this claim is connected to a specific suspicious cluster".

**Stage 4, features.** For each claim, compute hub-filtered neighbour count, shared-identifier count with other claims in the seed's neighbourhood, 2-hop and 3-hop motif counts, and actor-level prior outcome counts that respect label availability.

**Stage 5, scoring.** A gradient-boosted classifier over claim attributes plus graph features produces a probability. The probability is used only for ordering; the reported precision is empirical at capacity, not the model's own probability.

**Stage 6, bundling and allocation.** Bundles are built by connected component over the hub-filtered relation, capped at eleven claims. The queue is filled by score until the hour budget is exhausted, using the cost model from Section 6.

#### Candidate comparison

| Candidate | Evidence | Strength | Limitation | Status |
|---|---|---|---|---|
| Rules queue (14 triggers) | incumbent operating record | auditable, explainable, stable | structurally blind, flat precision | baseline |
| Claim-level tabular model | pilot, 300 per arm | no graph dependency, simple serving | misses relational signal | tested |
| Graph features, claim-level review | pilot, 300 per arm | +14.0 pp over tabular at 300 reviews | hours per review unchanged | tested |
| Graph features with bundling | pilot timing plus cost model | 2.43× hour reduction per claim | bundling errors propagate to whole file | tested on pilot, modelled at scale |
| Full GNN with sampled neighbourhoods | not tested | could learn deeper structure | training and serving cost not justified by the observed gap | considered, not tested |
| Manual link-chart review by investigators | not tested | uses investigator judgement | does not scale to 240,000 claims | considered, not tested |

The rejected GNN deserves a note.
With a 14.0-point pilot gap and a 95% interval of 6.3 to 21.7 points, the incremental value of a deeper model is not distinguishable from noise at this sample size.
Committing to GNN training and serving before establishing that the simple aggregate features plateau would spend engineering on an unmeasured hypothesis.

### 5d. Data provenance, quality, exclusions and leakage controls

**Provenance.** All source tables are internal to the constructed insurer.
No external dataset, vendor feed, or public benchmark was used.
Because the setting is constructed, there is no extraction date to record; the case records the *intended* lineage so the reasoning is transferable.

**Quality checks performed in the design.**

| Check | Rule | Failure signal | Action |
|---|---|---|---|
| Identifier uniqueness | one row per claim identifier per version | duplicate keys above 0.01% | block and reconcile |
| Edge availability | every edge has `available_at` ≥ `observed_at` | missing timestamps | exclude edge from features |
| Actor degree | report the top 50 by degree | degree above 5,000 | review for hub classification |
| Label maturity | disposition age ≥ 212 days | age below threshold | mark unresolved, not negative |
| Duplicate claims | same loss, same date, same actors | match above threshold | resolve to survivor |
| Split disjointness | no actor in both train and test | any overlap | rebuild split |

**Leakage controls.** Four controls are mandatory.

The first is the as-of join on availability time, which prevents an edge ingested after the decision from entering that decision's features.
The second is neighbour-label exclusion: a claim's features may not include any outcome for the same claim, and may include a neighbour's outcome only if that outcome was available at the decision time.
The third is split disjointness at the actor level, not the claim level.
The fourth is a negative-control fixture: insert a future-only edge and confirm that every earlier snapshot's features are unchanged.

**What was unavailable.** The internal adjudication ledger, the investigator time study, and the queue logs were not accessible.
Consequently the hours-per-review figures are stated assumptions with an explicit break-even test rather than measurements, and the case cannot claim an operational result.

### 5e. Reproducibility and method-selection ledger

| Artifact | Location | Version/date | Purpose | Accessibility | Reproduction limit |
|---|---|---|---|---|---|
| PLAN-M15-01 | collection catalog | registered 2026-11-09 | scope brief | readable | no underlying data |
| Pilot table, 300 per arm | constructed | illustrative | precision comparison | in-file | not a real pilot |
| Cost model | Section 6 | derived | hours per claim | in-file | depends on A-05 |
| Component-size probe | Section 6 | derived | hub threshold effect | in-file | one graph construction |
| Maturity survival curve | not available | — | label completeness | unavailable | cannot estimate bias direction |

Everything arithmetic in this case is reproducible from the numbers printed beside it.
Nothing observational is.
A reader can check every multiplication and every interval but cannot verify that an insurer behaves this way, because no insurer does in this document.

#### Method-selection rationale

The sequence of choices was: exclude any method that cannot be rebuilt point-in-time; then compare the simplest non-graph model against the incumbent at identical capacity; then test whether graph features add value on the same population; then test whether bundling changes hours per claim; only then consider deeper models.

That ordering is not arbitrary.
Each stage asks a question that can fail independently, and a failure at any stage stops the expensive work.
A team that starts with a GNN skips all four questions and cannot tell which stage produced its result.

**Takeaway:** the method ladder is a sequence of falsifiable gates, and the graph only earns its place after the tabular model and the bundling economics both hold.
