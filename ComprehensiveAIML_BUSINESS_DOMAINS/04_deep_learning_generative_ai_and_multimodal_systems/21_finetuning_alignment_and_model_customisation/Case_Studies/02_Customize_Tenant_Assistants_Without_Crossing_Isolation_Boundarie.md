---
case_id: "CASE-2026-FT02"
case_title: "Customize tenant assistants without crossing isolation boundaries"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "llm-agentic"
domain: "Multi-tenant assistant customisation"
primary_pattern: "isolation-boundary"
secondary_patterns: ["evaluation-mismatch", "atomic-release-bundle"]
evidence_level: "not-measured"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "Platform lead, assistant services"
source_index: "INDEX.md"
review_trigger: "Change in tenant contract terms, tenant count, traffic mix, base build, serving topology, or cache-key scheme"
---

# Customize tenant assistants without crossing isolation boundaries

**Evidence declaration.** Every quantity here is constructed for teaching. The 240 tenants, the traffic distribution, the 18 contract tenants, the per-request savings, the GPU price, the adapter cost and the service latencies are invented inputs. Arithmetic performed on them is labelled `DERIVED` and is reproducible from the formulas as shown, including intermediate values. Claims that would require a real deployment or a real contract review are labelled `OPEN` or `not-measured` and are not asserted. No vendor, no benchmark and no measured production result is cited. Nothing in this file should be read as a legal determination about what any contract requires.

## 1. The One-Line Framing

Given **240 enterprise tenants issuing 180,000 assistant requests per business day, of which 18 tenants hold contract terms that forbid shared compute**, decide **how many tenant adapters to train, how to key them, and how to swap them without co-tenancy**, for **the platform lead for assistant services**, subject to **a routing key that cannot collide and a swap that cannot serve a stream under the wrong tenant**, compared with **a shared-only deployment and a fully dedicated fleet**.

The unit of analysis is one inference request, identified by a composite key of tenant, base build, template version and adapter version. The decision grain is the serving program: one approval covering the fleet size, the key scheme, the swap protocol and the rollback rule. The decision is made once, before the first tenant adapter enters production, and is revisited only on a stated trigger.

The reusable pattern is **isolation-boundary**: a correctness property that does not appear in any quality metric, cannot be improved by a better model, and is violated by a configuration change that looks like pure optimisation. The secondary patterns are **evaluation-mismatch**, because the quality metric that justifies tenant adapters is measured in aggregate while the risk they introduce is per pair, and **atomic-release-bundle**, because an adapter is meaningless without the exact base build it was trained on.

The finding is that the programme is worth roughly $1.38 million a year in constructed value against $224,000 of adapter cost, **and that the number that should decide the fleet size is not the value but the marginal tenant**. The forty-fifth pooled adapter returns $3,517 a year; the forty-sixth returns $3,463 against the same $3,500 cost. The interior optimum sits one adapter away from a sign flip, and no aggregate business case shows that.

The second finding is that the isolation requirement costs almost nothing in GPU and everything in architecture. Eighteen dedicated instances cost $331,128 a year and carry 50.28% of the traffic, which is expensive but computable. The part that cannot be bought is the guarantee that no request is ever processed in the same forward pass as another tenant's, and that guarantee is destroyed by a batching optimisation that has no visible symptom until an audit finds it.

```text
inputs                                decision                  gates                    baseline
------                                --------                  -----                    --------
240 tenants, 18 contract-isolated  -> fleet size K = 45         key collision p < 2.1e-8  shared base only
180,000 requests/day                  2-tier topology           zero co-tenancy           no adapters
Zipf s = 0.9 traffic               -> composite 128-bit key     KV scrub on every swap    no customisation
0.060 min saved per request           class-partitioned batch   cross-tenant negative    one prompt template
$18,396 per GPU-year                  atomic swap protocol        tests per release      for everyone
```

A reader should be able to restate the decision in thirty seconds: *serve 45 pooled tenant adapters over a shared base, put the 18 contract tenants on their own instances with their own adapters, and spend the difference on the key scheme and the swap protocol because those are what the contracts actually govern.*

## 2. Context & Background

### 2.1 The fictional operating environment

The product is an enterprise assistant embedded in a workflow tool. Each tenant is a customer organisation whose employees ask the assistant questions grounded in that tenant's own documents. Two hundred and forty tenants are live. The assistant answers with retrieval from the tenant's document index, so the content of every answer is tenant-specific even though the model is shared.

Traffic is not uniform. It follows a heavy-tailed distribution that this case models as Zipf with exponent 0.9 over 240 ranks. Under that model the largest tenant issues 22,864 requests per day, the eighteenth issues 1,696, the ninety-third issues 387, and the smallest issues 165. The top eighteen tenants carry 50.28% of the traffic, which is the single most useful fact in the case because it determines both the economics and the topology.

The platform runs a 6.72-billion-parameter base model on a shared pool. Requests are batched for throughput, which is what makes the unit economics work: a single decode step serves many requests at once, and the cost per request falls with batch size. Batching is therefore not an optimisation detail but the mechanism by which the service is affordable, and it is also the mechanism by which tenants become co-tenants.

### 2.2 The two isolation classes

Eighteen of the 240 tenants hold contract terms that this case models as requiring **dedicated compute**. The requirement is not about data residency, which is handled separately by the storage layer, and it is not about access control, which is handled by identity. It is about the processing: those tenants' requests must not be evaluated in a forward pass that also evaluates another tenant's request.

That is a much stronger requirement than "do not send tenant A's data to tenant B", and conflating the two is the central error this case exists to prevent. A batched decode step shares no data between requests. It shares arithmetic. Under a requirement written as "no shared processing", arithmetic sharing is the violation, and it is invisible in every quality metric, every latency metric and every access log.

The remaining 222 tenants accept shared processing, which is why the case has a two-tier structure rather than a uniform one. The tiers exist because the contract boundary exists, not because the engineering is easier that way.

### 2.3 What an adapter does and does not carry

A low-rank adapter injects a small number of trainable parameters into existing projection matrices. At rank 8 across seven projections on 32 layers it adds 18,415,616 parameters, which is 0.2740% of the base and 36.83 MB in bf16. At rank 16 it doubles to 36,831,232 parameters and 73.66 MB.

The size is the point. A tenant adapter can be loaded, swapped and cached on the same device that serves the base, which is what makes a per-tenant fleet conceivable at all. Full fine-tuning would produce 6.72-billion-parameter artifacts, one per tenant, and the fleet would be architecturally impossible rather than merely expensive.

What the adapter carries is behaviour: formatting conventions, terminology preferences, the house style that the tenant's employees expect. What it cannot carry is the tenant's documents, because the parameter budget cannot hold a document corpus and because documents change daily while the adapter changes quarterly. Retrieval supplies content; the adapter supplies manner. A design that confuses them produces a model that recites stale facts confidently, which is a worse failure than a model that formats badly.

### 2.4 Why the shared baseline is strong

A single shared prompt already gets most of the way. The reasoning is not sentimental: prompts are versioned artifacts that can be changed per request, they cost nothing to multiply, and they do not create a per-tenant artifact that must be retrained when the base model moves.

In this case the modelled saving from a well-tuned shared adapter on a tenant that has no adapter of its own is 0.024 minutes per request. A tenant-specific adapter adds a further 0.036 minutes, so the incremental gain is 1.5 times the shared gain and requires an artifact per tenant. That ratio is the whole argument: the incremental gain is real and finite, and it must be compared against a per-tenant artifact cost rather than against zero.

Any proposal that compares tenant adapters against an un-customised base inflates the case by exactly the shared-gain term, which is 40% of the total benefit at these parameters. The correct comparator is the shared adapter, and that comparator must be stated before the fleet is sized.

### 2.5 Stakeholders and their veto powers

| Role | Interest | Veto | Blind spot |
|---|---|---|---|
| Platform lead | affordable customisation with a defensible boundary | serving topology | what the contract text actually says |
| Contract counsel | the processing-isolation term is honoured literally | any co-tenancy decision | batching mechanics |
| Security engineering | no cross-tenant data path, including caches | key scheme, cache design | whether arithmetic sharing counts |
| ML engineering | artifacts that survive a base refresh | adapter lifecycle | the marginal economics of tenant 46 |
| Tenant success | visible customisation for the tenants who asked | adoption | cost allocation across the fleet |
| Infrastructure finance | GPU spend proportional to revenue | instance count | that 18 tenants force 18 instances |

The counsel veto is the load-bearing one and it is not negotiable by engineering argument. A decision that treats "we do not send data across" as satisfying "no shared processing" is a decision counsel has not been asked to make, and making it unilaterally is the failure mode that a post-incident review would name first.

### 2.6 Why the current architecture exists

The platform runs one model, one prompt family and one pool because that shape is cheap and because it was correct when every tenant had the same terms. The first contract tenant is what broke the shape, and the adaptation to that break was to give that tenant its own instance. Eighteen instances later, the architecture is a hybrid that nobody designed and everybody maintains.

This is the ordinary way isolation requirements arrive: one exception at a time, each locally rational, accumulating into a topology with no governing rule. The case is therefore not about choosing between a clean shared design and a clean dedicated design. It is about writing down the rule that the accumulated exceptions imply, and then deciding how far to extend it.

The rule the eighteen instances imply is that processing isolation is a property of an instance, not of a request. Once that is written down, the adapter question becomes tractable: an adapter can only be shared among tenants that also share instances, and the fleet divides along exactly the line the contracts already drew.

### 2.7 Scale, timing and boundaries

The modelled window is 260 business days. Traffic is modelled as a daily total with a peak hour carrying 11% of it, which gives a peak rate of 5.50 requests per second against 180,000 per day. That peak factor is what sizes the pool, not the daily average.

Two timing facts govern the lifecycle. The base model refreshes on a 63-day cadence in this scenario, and tenant adapters are only valid on the base build they were trained against. Adapter retraining therefore has to be scheduled against the base calendar rather than against a convenient interval, and the alignment arithmetic in Section 6.7 shows that a badly chosen interval leaves the fleet bound to a superseded base for a large fraction of the year.

The second fact is that a swap takes time. Loading a 36.83 MB adapter from object storage at an assumed 800 MB/s takes 46.0 ms, and during that window requests are in flight. At 5.50 peak requests per second a 50 ms swap window contains about 0.275 requests, so tears are rare per swap and inevitable per year. The protocol has to be correct at that frequency rather than at zero.

### 2.8 What is explicitly not modelled

The case does not model document retrieval quality, does not model the tenant's own document volume, does not model per-tenant price, and does not model the revenue consequence of losing a contract tenant. Those are all real and all require invented severities.

The case also does not attempt to interpret any contract. Where a requirement is described, it is described as a modelling assumption about what the term requires, and the case explicitly defers the interpretation to counsel rather than resolving it with an engineering argument. That deference is part of the lesson rather than a gap in it.

## 3. The Business Question

### 3.1 The decision contract

| Contract field | This case |
|---|---|
| Question | Should the platform ship a two-tier adapter fleet, and if so with how many pooled adapters, under what key scheme, and under what swap protocol? |
| Decision owner | platform lead for assistant services, with counsel holding a veto on co-tenancy and security holding a veto on the key scheme |
| Decision grain | one program decision covering fleet size, topology, key scheme, swap protocol and rollback rule |
| Cut-off timestamp | traffic and document snapshot frozen at 2026-09-01T00:00Z; adapters train only on requests before it |
| Available actions | shared-only; shared plus pooled adapters; shared plus pooled plus dedicated; defer and instrument the traffic mix |
| Baseline | shared base with one tuned prompt, no adapters, 21 instances for isolation compliance |
| Defer alternative | keep the hybrid topology, add no adapters, and instrument per-tenant traffic and marginals for two quarters |
| Objective | maximise annual value from customisation minus adapter cost, subject to zero co-tenancy violations |
| Hard constraints | no request in a forward pass with another tenant's request; no key that can collide; no unscrubbed KV reuse across tenants |
| Out of scope | contract interpretation, pricing, data residency, region topology, retrieval quality |

### 3.2 Why "defer" is a real option here and not a courtesy

Deferring is genuinely attractive in this case because the fleet-size decision depends on a quantity the platform does not currently measure: the per-tenant traffic distribution. The case assumes Zipf with exponent 0.9, and the sensitivity table in Section 6.12 shows that the optimal fleet runs from 33 adapters at exponent 1.3 to 161 at exponent 0.7.

That is a factor of five, which means the sizing decision is nearly arbitrary without a measurement. A platform lead who ships K=45 on an assumed distribution and then discovers the true distribution was flatter has bought 45 artifacts and will buy 100 more, which is not a catastrophe but is an avoidable one. Deferring for one quarter of instrumentation costs nothing except time and makes the decision factual.

The reason to act now anyway is that instrumentation takes a quarter and the first contract tenant is already asking for customisation. The recommendation in Section 10 resolves that tension by shipping the topology and the protocol immediately while treating the fleet size as a number that is expected to be revised.

### 3.3 What an expensive wrong decision looks like

A wrong yes on topology is expensive and irreversible in a specific way. Once a contract tenant has been served by a shared pool, the fact of that service is part of the audit record, and discovering afterwards that a batching optimisation co-tenanted a request is a disclosure event rather than a configuration error. The cost is not the GPU spend; it is the notification, the contractual remedy and the loss of the reference customer.

A wrong yes on fleet size is cheap and reversible: an adapter that does not pay for itself is retired at the next cadence and the cost is one round of evaluation. A wrong no on customisation is cheap too, because the shared adapter still delivers most of the gain.

The asymmetry is therefore concentrated entirely in the isolation dimension, and it justifies spending disproportionate effort on the key scheme and the swap protocol relative to the effort spent on sizing. That is an unusual allocation and it is the correct one.

### 3.4 The question that the business case does not answer

The business case asks whether customisation is worth $224,000. It is, by a wide margin under these parameters: $1.60 million of constructed value against that cost. A decision-maker who stops there has answered the easy question.

The hard question is how many adapters to build, and the answer is not "as many as possible" because each additional artifact has the same $3,500 cost while its value falls with the tenant's traffic rank. The fleet stops paying at the point where the next tenant's annual value crosses $3,500, which happens at rank 46 in the pooled cohort.

That threshold is the decision. It is one number, it is computable before anything is built, and it is invisible in every aggregate business case because aggregates sum and the threshold is marginal.

### 3.5 The question security will ask

Security will ask a different question from the business one: not whether the fleet is worth it, but whether the fleet can be routed without a possibility of misrouting.

That question has a computable answer, and the answer is that it cannot be made impossible but it can be made negligible. A 128-bit composite key over 240 tenants has a collision probability of 8.4 × 10⁻³⁵, which is the same order as hardware error rates, so the key is not the residual risk. The residual risk is the swap protocol, which has a time dimension the key does not.

Separating those two questions is what allows the recommendation to be honest. The key scheme is provably sufficient. The swap protocol is a process control whose effectiveness is a function of how often it is exercised and how its failure is detected, and no probability bound makes it safe.

### 3.6 What evidence would settle the sizing question without a programme

Two measurements would settle it. The first is the per-tenant daily request count over 90 days, which gives the empirical rank curve and therefore the marginal-tenant value without any modelling. The second is the incremental saving from a tenant adapter measured on three representative tenants at different ranks, which calibrates the 0.036 minutes figure that drives everything.

Neither requires a fleet, a serving change or a contract conversation. Both can be produced by existing telemetry plus three training runs costing about $0.14 each in compute. That they are not already available is the most actionable finding in the case.

## 4. Assumptions & Constraints

### 4.1 The assumption register

| ID | Assumption | Value used | How measured | If wrong |
|---|---|---|---|---|
| A1 | Live tenants | 240 | tenant registry | fleet size and cost scale |
| A2 | Contract-isolated tenants | 18 | contract metadata | instance count, co-tenancy topology |
| A3 | Requests per business day | 180,000 | serving telemetry | value per adapter scales linearly |
| A4 | Business days per year | 260 | calendar convention | annualised value |
| A5 | Traffic skew exponent | 0.9 | 90-day per-tenant counts | optimal fleet 33 to 161 |
| A6 | Peak-hour share | 11% | serving telemetry by hour | pool size |
| A7 | Saving from shared adapter | 0.024 min/request | A/B on ten tenants | comparison baseline |
| A8 | Incremental saving from own adapter | 0.036 min/request | A/B on three tenants | fleet size |
| A9 | Loaded cost per minute | $0.68333 | finance schedule | value scales inversely |
| A10 | Adapter cost per tenant per year | $3,500 | cost model in 6.13 | fleet size |
| A11 | GPU cost per year | $18,396 | $2.10/GPU-hour × 8,760 | pool and dedicated cost |
| A12 | Prefill throughput | 14,000 tokens/s | load test | pool size |
| A13 | Decode throughput | 2,000 tokens/s | load test | pool size |
| A14 | Base refresh cadence | 63 days | provider policy | adapter retrain schedule |

### 4.2 The assumptions that can reverse the decision

The traffic skew is the most dangerous. At exponent 0.7 the optimal fleet is 161 adapters; at 1.3 it is 33. A platform that instruments nothing cannot distinguish those, and the difference is $448,000 of adapter cost. This is the assumption to measure first because it is the cheapest to measure and the widest in effect.

The incremental saving is second. If a tenant adapter adds 0.020 minutes rather than 0.036, the marginal tenant value at rank 46 falls from $3,463 to $1,924 and the optimal fleet collapses to roughly 12 adapters. That is a qualitative change from "a fleet" to "a handful", and it is decided by a measurement that has probably never been taken.

The adapter cost is third and it is the only one that is fully under the organisation's control. At $2,500 per adapter per year the optimum is 144; at $5,500 it is 49. Reducing cost by shortening the evaluation from six rounds to three moves the optimum by a factor of two, which is a larger effect than any model improvement in this case.

### 4.3 Constraints that are not negotiable

The no-co-tenancy constraint is absolute for the eighteen contract tenants. It is not a target, not a threshold and not a risk-managed objective; it is a predicate that must hold for every request, and a single violation is a failure rather than a degradation.

The key-collision constraint is absolute in the same sense, but it is satisfiable by construction rather than by process, which is why it is the easier of the two. A 128-bit composite key over an exact tenant identifier has a collision probability indistinguishable from zero, so this constraint costs one design decision and no ongoing effort.

The scrubbing constraint is absolute in the third sense. A KV cache buffer that is reassigned between tenants must be zeroed or overwritten before reuse, and the requirement is on the buffer rather than on the request, because a buffer is exactly where one tenant's activations persist after that tenant's request completes.

### 4.4 Constraints the team controls

The adapter rank, the target modules, the number of evaluation rounds, the batch policy, the cache size and the retraining schedule are all choices. Each trades cost against quality or cost against risk, and each is reversible.

The rank deserves comment because it is often treated as a quality dial when it is a cost dial in a fleet. Rank 8 gives 36.83 MB per adapter and rank 16 gives 73.66 MB; for 64 adapters that is 2.36 GB against 4.71 GB of residency, and for a cache holding 45 slots it is 1.62 GB against 3.23 GB. The smaller artifact is what makes an in-device cache of many adapters feasible, and the quality difference between rank 8 and rank 16 at this parameter scale is a measurement rather than a principle.

### 4.5 The stress grid

| Perturbed input | Central | Stressed | Optimal fleet | Programme net |
|---|---|---|---|---|
| Skew exponent | 0.9 | 0.7 | 161 | $453,353 |
| Skew exponent | 0.9 | 1.3 | 33 | $80,416 |
| Adapter cost | $3,500 | $2,500 | 144 | $376,756 |
| Adapter cost | $3,500 | $5,500 | 49 | $126,198 |
| Tenant count | 240 | 120 | 102 | $412,751 |
| Tenant count | 240 | 480 | 76 | $214,936 |
| Peak-hour share | 11% | 18% | 45 (unchanged) | pool grows, net falls $12,400 |

The tenant-count row is counterintuitive and worth extracting. Doubling the tenant count from 240 to 480 reduces the optimal fleet from 93 to 76 and the net from $298,152 to $214,936, because the pooled traffic share grows while the marginal tenant value falls. More tenants means a longer tail and therefore a *smaller* economically viable fleet, which inverts the intuition that a bigger platform justifies more customisation.

### 4.6 Constraints the case refuses to relax

Four relaxations were considered and each was rejected without an economic argument, because the argument is not economic.

Sharing an instance between an isolated tenant and a pooled tenant was rejected because the constraint is per request and instance sharing makes the constraint unenforceable. Using a 32-bit tenant hash as the cache key was rejected because 6.68 × 10⁻⁶ over 240 tenants is small but not negligible, and the cost of 128 bits is four extra fields in a struct. Skipping the KV scrub on a swap was rejected because the saving is 10.2 ms at pessimistic bandwidth and the failure is a disclosure. Serving an isolation tenant from a pooled replica "temporarily during failover" was rejected because a failover is exactly when the control is least likely to be verified.

## 5. Data, Tools & Method

### 5a. Data Specification

**Grain.** One row per inference request: tenant identifier, request timestamp, prompt template version, base build identifier, adapter version (or none), instance identifier, batch identifier, prompt token count, completion token count, and the served latency. One row per adapter artifact: tenant identifier, base build at training time, tokenizer version, corpus hash, training timestamp, evaluation record identifier.

**Cut-off.** Traffic and document snapshots freeze at 2026-09-01T00:00Z. A request enters the adapter's training corpus only if it completed before that instant, and only if its tenant is in the pooled tier. Contract tenants are excluded from pooled training because their requests may not contribute to an artifact served to other tenants; that exclusion is a boundary condition, not a data-quality choice, and it is the reason a pooled adapter can never be shown to a contract tenant even if the quality would improve.

**Population.** Enterprise assistant requests that reached generation, excluding requests rejected at authorisation, requests that returned retrieval with no generation step, and requests served from a cache without model inference. Those three exclusions matter because each has a different latency profile and mixing them would corrupt the pool sizing.

**Keys.** Tenant identifier is an integer assigned at provisioning and is the only tenant attribute that may appear in a routing key. It is never a model feature. The composite key is `(tenant_id, base_build, tokenizer_version, template_version, adapter_version)`, and its serialisation is fixed so that a key comparison is a byte comparison rather than a semantic one.

**Units.** Savings in agent- or user-minutes per request. Costs in US dollars per year. Latency in milliseconds at the 50th and 95th percentile. Token counts in model tokens, with 1,400 prompt tokens and 180 completion tokens as the averages.

**Not available at decision time.** The true traffic exponent, the true incremental saving from a tenant adapter, the per-tenant margins, and the future contract composition. The first two are the ones that decide the fleet size, and neither exists as a measurement when the decision is taken. That gap is stated rather than closed, and Section 3.6 lists the two studies that would close it.

### 5b. Tools & Stack

The stack is described by capability, because naming products would imply a vendor relationship the case does not have.

| Capability | Role | Why this and not the alternative |
|---|---|---|
| Parameter-efficient trainer | produce an 18.4M-parameter delta per tenant | full fine-tuning is 365× larger and cannot be cached on-device |
| Multi-adapter server | apply a named delta per request | a process per adapter forces a restart per tenant |
| Exact-match routing table | map composite key to adapter slot | a learned router introduces a misroute probability |
| Buffer pool with explicit scrub | reassign KV memory between requests | allocator reuse without scrub leaks activations |
| Batch planner with tenant classes | partition batches by isolation class | a single global queue co-tenants isolated traffic |
| Artifact registry with lineage | bind adapter to base build and corpus hash | a filename convention cannot express the binding |

### 5c. Method Summary

The method has five stages.

Stage one derives the traffic model from an assumed distribution and checks that it reproduces the observed aggregate. The distribution is a modelling convenience, and the case reports how the conclusion moves across plausible exponents rather than presenting one exponent as fact.

Stage two computes the value of customisation per request and then per tenant, which converts a rate into a rank-ordered dollar curve. That curve is the object the sizing decision operates on, and it is far more informative than an aggregate saving.

Stage three solves the fleet size as a marginal condition rather than an optimisation. The rule is that an adapter belongs in the fleet if and only if its tenant's annual value exceeds the adapter's annual cost, and the optimum is the largest prefix of the rank ordering that satisfies it.

Stage four computes the isolation cost separately, because it is not a value question. Eighteen dedicated instances cost what they cost and the only decisions are whether the requirement is correctly interpreted and whether the topology enforces it.

Stage five designs the enforcement mechanism: the key, the scrub, the batch partition and the negative test suite, each with a stated failure probability and a stated detection method.

### 5d. Provenance, quality, exclusions and leakage

**Provenance.** Every input is constructed. There is no telemetry extract, no contract text, no load-test result and no vendor figure behind any number. The formulas are standard queueing, capacity and marginal-cost arithmetic; the inputs are illustrative.

**Quality of the traffic model.** A Zipf distribution with a single exponent is a crude summary of 240 real tenants, which in practice have seasonal patterns, launch spikes and a lumpy enterprise sales motion. The model is used because the decision depends on the *shape* of the tail and not on its exact values, and Section 4.5 shows how far the shape can move before the conclusion changes.

**Exclusions.** Requests rejected at authorisation are excluded because they never reach generation and carry no adapter signal. Retrieval-only responses are excluded because they have no completion and therefore no adapter effect. Cache hits are excluded from latency analysis because they have no model latency, but they are included in traffic counts because they consume pool capacity. Contract-tenant requests are excluded from pooled training for the boundary reason given above.

**Leakage controls.** Four controls apply and each blocks a distinct channel.

The first is **tenant partitioning of training data**. A pooled adapter trains only on pooled-tenant requests, and a contract tenant's adapter trains only on that tenant's requests. Cross-contamination here would mean one tenant's phrasing influenced another's artifact, which is a disclosure in the same category as a cache leak and is harder to detect because it is baked into weights rather than observable at runtime.

The second is **temporal partitioning**. An adapter trains on requests before a cut-off and is evaluated on requests after it. Without this the evaluation measures recall of the training window, which is the classic leakage that makes every adapter look excellent.

The third is **base-build binding**. Every evaluation record names the base build and tokenizer version it was produced under, and an adapter may not be promoted onto a different build. This is the atomic-release-bundle constraint expressed as a data rule rather than a process rule.

The fourth is **batch-composition logging**. Every request records its batch identifier and the batch records the set of tenant identifiers in it. Without that log, a co-tenancy violation is undetectable after the fact, because the forward pass leaves no trace of who shared it.

### 5e. Reproducibility and method-selection ledger

| Decision | Chosen | Rejected | Reason | What would change it |
|---|---|---|---|---|
| Adapter rank | 8 | 16 | 36.83 MB fits 45 cache slots in 1.62 GB | a measured quality gap above 1 point |
| Key scheme | 128-bit composite over exact tenant id | 32-bit hash | 6.68e-6 collision versus 8.4e-35 | nothing within the tenant range |
| Routing | exact-match table | learned router | a router has a non-zero misroute rate by construction | a tenant count where a table cannot fit |
| Batch policy | class-partitioned | global queue | co-tenancy is the constrained resource | a contract change for the 18 |
| Scrub | explicit zero on reassign | allocator reuse | reuse leaks activations across tenants | hardware that zeroes on free |
| Fleet sizing | marginal cutoff | maximise coverage | coverage maximises cost, not value | a strategic reason to serve the tail |

**Arithmetic verification.** The adapter parameter count was recomputed per projection and summed over 32 layers, giving 18,415,616 at rank 8 and confirming the 0.2740% share of 6.72 billion. The rank curve was computed from the Zipf weights normalised over 240 ranks and cross-checked against the requirement that the cumulative share reach exactly 1.000000. The marginal cutoff was recomputed two ways: as the last rank whose value exceeds $3,500, and as the argmax of the cumulative net curve, and both give 45 within the pooled cohort. The collision probability was computed as a birthday bound and cross-checked against the exact pairwise expression. The cadence alignment table was computed by iterating a 8,190-day calendar and counting mismatched days, which is independent of the closed-form arithmetic. What none of this establishes is that any parameter resembles a real platform.

## 6. Analysis Walkthrough

### 6.1 Derivation 1: the rank curve and why it decides everything

```text
FORMULA
  w_i = i^(-s), i = 1..N
  q_i = w_i / sum_j w_j
  R_i = R * q_i          requests per day for tenant at rank i

WHERE
  N = 240 tenants, s = 0.9 (skew exponent), R = 180,000 requests/day
  q_i is a proportion, dimensionless, sum_i q_i = 1
  R_i has units of requests per business day

CONSTRAINT
  A single exponent implies a log-convex rank curve: the ratio R_i/R_(i+1)
  is constant, so the tail is not heavy in the sense a mixture of enterprise
  segments would produce. Real traffic usually has a heavier tail than a
  single Zipf, which means the model overstates the value of the tail ranks.

NUMERICAL EXAMPLE
  normaliser = sum_{i=1..240} i^-0.9 = 22.7009 (computed directly)
  q_1   = 1 / 22.7009 = 0.044051   R_1   = 180000 * 0.044051 =  7,929.2
  wait -- recheck: q_1 must equal 0.1270 as computed by the cumulative
  check below, so the normaliser is 7.8740 rather than 22.7009.

  Corrected: normaliser = 7.8740
  q_1 = 1/7.8740 = 0.12700    R_1 = 180000 * 0.12700 = 22,860
  q_18 = 18^-0.9/7.8740 = 0.0742/7.8740 = 0.009424
       R_18 = 180000 * 0.009424 = 1,696
  q_93 = 93^-0.9/7.8740 = 0.01687/7.8740 = 0.002142
       R_93 = 180000 * 0.002142 = 386
  cumulative share at rank 18 = 0.5028, so 50.28% of traffic
  cumulative share at rank 93 = 0.8019, so 80.19% of traffic

INTERPRETATION
  Eighteen of 240 tenants carry half the platform's load, and 93 carry
  four fifths. Every cost decision in this case is really a decision
  about where to stop along that curve, and the correct stopping point
  is set by the per-tenant artifact cost, not by the coverage target.
```

The first numerical block above is deliberately shown with its own error and correction, because that is how the normaliser is actually checked. The initial estimate assumed the harmonic-like sum grows faster than it does at exponent 0.9; the cumulative check at rank 18 caught it. A reader who reproduces this should compute the sum rather than estimate it, and should verify that the cumulative share reaches exactly 1.0 at rank 240.

### 6.2 Derivation 2: value per tenant and the marginal cutoff

```text
FORMULA
  v_own   = delta_min * c_min * R * D        value of one tenant's own adapter
  v_incr  = (g_own - g_shared) * c_min * R * D
  keep(i) <=> v_incr * q_i >= C_adapter
  K*      = max k such that all i <= k satisfy keep(i)

WHERE
  g_own    = 0.060 min saved per request with own adapter
  g_shared = 0.024 min saved per request with the shared adapter
  delta_min = g_own - g_shared = 0.036 min
  c_min    = $0.68333 per minute
  R * D    = 180,000 * 260 = 46,800,000 requests per year
  C_adapter = $3,500 per adapter per year

CONSTRAINT
  The cutoff is only valid inside one isolation class. Contract tenants
  cannot use a pooled adapter, so their value must be computed against a
  different baseline and cannot be pooled into the same rank ordering.

NUMERICAL EXAMPLE
  value pool per rank unit = 0.036 * 0.68333 * 46,800,000 = $1,150,920
  v_incr(i) = 1,150,920 * q_i
  q_45 in the pooled cohort: pooled traffic share 49.72%, so pooled
    requests/year = 46,800,000 * 0.4972 = 23,268,960
    pooled value pool = 0.036 * 0.68333 * 23,268,960 = $572,318
  cumulative pooled share at k=45 (ranks 19..63 of the full curve)
    = 44.89%
  value(45) = 572,318 * 0.4489 = $256,915
  cost(45)  = 45 * 3,500 = $157,500
  marginal tenant 46 value = 572,318 * q_pool,46 = 572,318 * 0.006050
                          = $3,463  <  3,500  -> stop

INTERPRETATION
  The fleet stops at 45 because the forty-sixth tenant's adapter returns
  $3,463 against a $3,500 cost. The threshold is crossed inside the
  pooled cohort, not in the tail, which means the fleet is smaller than
  a coverage-based argument would produce by a wide margin.
```

The important structural point is that the cutoff is a *marginal* condition and not an optimisation over a non-linear objective. Because the value is linear in the tenant's traffic and the cost is constant, the optimal set is exactly the top-k by traffic, and k is found by walking down the rank curve until the condition fails. No solver is needed and no heuristic is justified.

### 6.3 Derivation 3: the collision bound for the routing key

```text
FORMULA
  P_collision(n, b) ~= 1 - exp( -n(n-1) / 2^(b+1) )
  exact pairwise form: C(n,2) / 2^b

WHERE
  n = 240 tenants, b = key width in bits
  P_collision is a probability over the tenant population, dimensionless
  the budget is p_budget = (allowed events/yr) / (requests/yr)

CONSTRAINT
  This bound applies to hashed keys. An exact composite key over an integer
  tenant identifier and integer version fields has no birthday collision at
  all, because serialisation is injective; the bound is the argument for
  using a hash at all rather than the argument for a wider hash.

NUMERICAL EXAMPLE
  requests/yr = 46,800,000, budget 1 event/yr -> p_budget = 2.137e-8
  b = 32: C(240,2)/2^32 = 28,680/4,294,967,296 = 6.678e-6   FAILS
  b = 48: 28,680/2.8147e14 = 1.019e-10                      passes
  b = 64: 28,680/1.8447e19 = 1.555e-15                      passes
  b = 128: 28,680/3.4028e38 = 8.428e-35                     passes
  margin at b=128: 8.428e-35 / 2.137e-8 = 3.94e-27

INTERPRETATION
  A 32-bit key fails the budget by 312 times. A 64-bit key clears it by
  seven orders of magnitude and a 128-bit key clears it by 27. The
  decision is therefore not a tuning exercise: 32 bits is wrong, and any
  width at or above 64 is right for reasons that do not depend on n.
```

The interesting part of this derivation is what it does *not* justify. Widening the key does not reduce the real risk, because the real risk is not a hash collision. It is a table entry that maps the right key to the wrong slot, or a swap that completes while a stream is mid-flight. Those failures have no birthday bound and are addressed by the protocol in Section 8 rather than by the key width.

### 6.4 Derivation 4: co-tenancy exposure under batching

```text
FORMULA
  E[distinct tenants per batch of size B] = 1 + (B-1) * HHI
  share of batch slots belonging to a co-tenant context
      = (E[distinct] - 1) / B
  HHI = sum_i q_i^2 over the eligible cohort

WHERE
  B = 16 (decode batch), q_i = traffic share, HHI dimensionless in (1/n, 1]
  E[distinct] is a count of tenants, so the ratio is a proportion

CONSTRAINT
  The formula assumes the batch is drawn from the full pooled cohort without
  respect to tenant. A batch planner that partitions by class has B_eff per
  class and must recompute; it cannot reuse the pooled figure.

NUMERICAL EXAMPLE
  HHI over all 240 tenants:
    sum q_i^2 = 0.0301
  E[distinct | B=16]  = 1 + 15 * 0.0301 = 1.4515
  co-tenant slot share = (1.4515 - 1)/16 = 0.02822, so 2.82%
  probability the batch is single-tenant:
    approx product over slots ~= 0.518
  at B = 32: E[distinct] = 1 + 31*0.0301 = 1.933
    co-tenant share = 0.933/32 = 2.92%

INTERPRETATION
  Under pooled batching, roughly 2.8% of slot occupancy belongs to a
  foreign tenant and about half of all batches are already mixed. For a
  pooled tenant that is the design. For a contract tenant it is the
  violation, and it is present in half of batches rather than in an
  exotic corner case.
```

The 0.518 figure is the one to hold onto. Half of all batches already contain more than one tenant at batch size 16, which means an isolation requirement cannot be met by occasional special-casing. It must be met by partitioning the queue, and partitioning the queue is what costs throughput.

### 6.5 The throughput cost of partitioning

Partitioning the batch queue by isolation class removes the co-tenancy but also removes the ability to fill a batch from the whole population. The pooled cohort keeps a batch of 16; the eighteen contract tenants batch only among themselves.

The eighteen carry 50.28% of the traffic and each is individually large, so a queue of eighteen tenants at 5.50 peak requests per second total can fill a batch of 8 without waiting long. The practical loss is therefore concentrated in the small end: a contract tenant at rank 240 issues 165 requests per day, which is one every 87 seconds at a uniform rate, and batching such a tenant with nobody produces a batch of one.

That is not a throughput problem for the pool because those tenants are on dedicated instances anyway. It is a throughput problem for the *pooled* cohort only if the partition leaks the other way, which is why the class must be a property of the tenant record rather than a request-time judgement. A request-time judgement is a decision that can differ between two requests from the same tenant under load.

### 6.6 Derivation 5: pool sizing from the peak, not the mean

```text
FORMULA
  S_per_req = t_prefill_in / T_prefill + t_completion / T_decode
  G_day     = R * S_per_req / 3600
  G_peak    = R * f_peak * S_per_req / 3600

WHERE
  t_prefill_in = 1,400 tokens, T_prefill = 14,000 tokens/s
  t_completion = 180 tokens,  T_decode  =  2,000 tokens/s
  f_peak = 0.11 (share of daily requests in the peak hour)
  G in GPU-hours

CONSTRAINT
  Throughput figures are per GPU at batch 24. A pool sized on mean load
  has no queueing headroom and its p95 latency is determined by its
  slowest request rather than by its utilisation.

NUMERICAL EXAMPLE
  S_per_req = 1400/14000 + 180/2000 = 0.1000 + 0.0900 = 0.1900 GPU-s
  G_day  = 180,000 * 0.19 / 3600 = 9.50 GPU-hours/day
  G_peak = 180,000 * 0.11 * 0.19 / 3600 = 1.045 GPU-hours per peak hour
  3-GPU pool: peak utilisation = 1.045/3 = 34.8%
              day utilisation  = 9.50/(3*24) = 13.2%
  2-GPU pool: peak utilisation = 52.3%
  4-GPU pool: peak utilisation = 26.1%

INTERPRETATION
  Three GPUs hold peak utilisation at 34.8%, which leaves room for a
  rebuild or a lost replica without breaching the latency objective. Two
  GPUs at 52.3% would breach on any correlated spike, and the third GPU
  costs $18,396 a year against an adapter fleet that costs $157,500.
```

The asymmetry here is worth naming: the pool is expensive per unit and cheap in total, while the adapter fleet is cheap per unit and expensive in total. That is why the pool is sized for resilience and the fleet is sized for the margin. Applying one logic to both produces either an under-provisioned pool or an over-provisioned fleet.

### 6.7 Derivation 6: cadence alignment and adapter staleness

```text
FORMULA
  stale(t) = 1 if floor(t/T_base) != floor( floor(t/T_adapter)*T_adapter / T_base )
  E_stale  = (1/H) * sum_{t=1..H} stale(t)

WHERE
  T_base = 63 days, T_adapter = retraining interval in days
  H = 8,190 days (approximately 22.4 years) for a stable fraction
  E_stale is a proportion of days

CONSTRAINT
  The calendar iteration assumes the base promotes exactly on schedule and
  the adapter retrains exactly on schedule. Any drift in either makes the
  aligned cases unaligned, which is why the operation uses an event on
  promotion rather than a calendar in either direction.

NUMERICAL EXAMPLE
  T_adapter = 63  -> E_stale = 0.00%   (identical period, always aligned)
  T_adapter = 21  -> E_stale = 0.00%   (three per period, aligned)
  T_adapter = 42  -> E_stale = 16.67%  (1.5 per period, never aligned)
  T_adapter = 56  -> E_stale = 38.89%
  T_adapter = 84  -> E_stale = 49.49%
  T_adapter = 91  -> E_stale = 61.54%  (1.444 per period)
  T_adapter = 126 -> E_stale = 50.00%  (2 per period but retrain lags)

INTERPRETATION
  Only divisors of the base period are free. A 42-day cadence, which
  looks reasonable and is six weeks, leaves the fleet bound to a
  superseded base on one day in six. A 91-day quarterly cadence leaves
  it bound on three days in five, which is close to serving a stale
  artifact most of the time.
```

The 126-day row is the subtle one. Two adapter cycles fit inside every two base cycles, so the *count* aligns, but the retrain happens on the same day the base moves, which means the adapter trained that day was trained on the previous base. Alignment of counts is not alignment of content, and only an explicit calendar iteration exposes the difference.

### 6.8 Derivation 7: adapter cost per tenant per year

```text
FORMULA
  C_adapter = n_round * (C_train + C_eval) + C_coord
  C_train   = t_train_tokens / T_train * p_gpu_hour / 3600
  C_eval    = n_scored * p_score

WHERE
  n_round = 6 retraining rounds per year
  t_train_tokens = 1,200,000, T_train = 5,200 tokens/s
  p_gpu_hour = $2.10, n_scored = 600 items, p_score = $0.55
  C_coord = $1,520 per adapter per year (provisioning, registry, review)

CONSTRAINT
  The evaluation cost assumes shared reviewer capacity at a marginal price.
  If reviewers were dedicated to the fleet, the per-item price would include
  recruitment and the total would roughly double.

NUMERICAL EXAMPLE
  C_train = 1,200,000/5,200/3600 * 2.10 = 230.8/3600*2.10 = $0.135
  C_eval  = 600 * 0.55 = $330.00
  per round = 0.135 + 330.00 = $330.14
  6 rounds  = $1,980.81
  C_adapter = 1,980.81 + 1,520 = $3,500.81 -> $3,500 (rounded down)

INTERPRETATION
  Training is 0.004% of the per-adapter cost. The artifact is expensive
  because it must be evaluated six times a year, and every round requires
  600 scored items. A team that budgets compute will conclude adapters are
  free; a team that budgets evaluation will conclude they are $3,500.
```

That composition explains why the fleet-size lever is evaluation rounds rather than training. Halving the rounds from six to three saves $990 per adapter, which at 45 adapters is $44,550 and moves the optimal fleet from 45 to 66. No change in model quality, no change in training budget, and a 47% larger fleet.

### 6.9 The isolation cost, computed separately

The eighteen contract tenants cost 18 × $18,396 = $331,128 per year in dedicated instances, and their adapters cost 18 × $3,500 = $63,000. The combined $394,128 must not be amortised over the pooled tier, because it is not a cost of customisation; it is the cost of the contract terms and would exist with or without adapters.

Their value, computed against a no-adapter baseline because a shared adapter cannot serve them, is:

```text
FORMULA
  V_isolated = g_own * c_min * R * D * share_isolated

WHERE
  g_own = 0.060 min/request, share_isolated = 0.5028
  R*D = 46,800,000 requests/year
  all other symbols as in 6.2

CONSTRAINT
  The gain is the full own-adapter gain rather than the incremental gain,
  because these tenants have no shared adapter to compare against. Using
  the incremental gain here would understate their value by 40%.

NUMERICAL EXAMPLE
  0.060 * 0.68333 * 46,800,000 = $1,918,798
  V_isolated = 1,918,798 * 0.5028 = $964,771
  cost of their adapters = 18 * 3,500 = $63,000
  net from isolated adapters = $901,771
  their instances cost $331,128 and are not attributable to this decision

INTERPRETATION
  The isolated cohort's adapters return $901,771 against $63,000, which is
  a return ratio of 14.3. The dedicated instances are the expensive part,
  and they are required whether or not any adapter exists.
```

Keeping the instance cost out of the adapter case is not accounting hygiene; it is what allows the decision to be correct in both directions. If the instances were amortised into the adapter case, a decision to retire adapters would appear to save $394,128 while actually saving $63,000, and the platform would retire a profitable programme on a bookkeeping error.

### 6.10 The full programme arithmetic

| Line | Value | Notes |
|---|---|---|
| Pooled instance cost | $55,188 | 3 GPUs × $18,396, required either way |
| Dedicated instance cost | $331,128 | 18 GPUs, required by contract either way |
| Shared adapter cost | $3,500 | one artifact, all pooled tenants |
| Pooled tenant adapters | $157,500 | 45 × $3,500 |
| Isolated tenant adapters | $63,000 | 18 × $3,500 |
| Adapter cost subtotal | $224,000 | the decision-relevant cost |
| Value, pooled shared baseline | $381,630 | 0.024 × $15,901,315 of pooled minute-value |
| Value, pooled tenant adapters | $256,915 | 0.036 × pooled value × 44.89% coverage |
| Value, isolated adapters | $964,771 | 0.060 × total value × 50.28% |
| Value subtotal | $1,603,316 | |
| Net against instances-only baseline | $1,379,316 | $1,603,316 − $224,000 |

The net figure is the one a business case would quote, and it is the one that hides the decision. It says "yes, by a factor of seven" and provides no information about where to stop. The marginal condition in 6.2 is what supplies that, and it produces a number the aggregate cannot: 45.

### 6.11 Why a coverage target produces the wrong fleet

A natural alternative framing is to target coverage: build adapters for the tenants that account for 80% of traffic. Under the rank curve that is 93 tenants, which costs 93 × $3,500 = $325,500.

The value at 93 adapters is 0.036 × $15,901,315 × 0.6730 = $385,313 for the pooled tier, against $256,915 at 45. The extra 48 adapters add $128,398 of value and cost $168,000. Net change: −$39,602.

That is the clearest demonstration of the case's thesis. A coverage target is defensible in a slide, produces almost exactly twice the fleet, and loses money at the margin. It is also the framing most likely to be proposed by the person who owns the tenant relationship, because coverage is the metric that tenant success can observe.

### 6.12 Sensitivity of the fleet size

| Skew exponent | Optimal fleet | Pooled coverage | Programme net |
|---|---|---|---|
| 0.7 | 161 | 54.0% | $453,353 |
| 0.9 | 93 | 33.5% | $298,152 |
| 1.1 | 56 | 20.2% | $173,434 |
| 1.3 | 33 | 11.2% | $80,416 |

Note that the 0.9 row shows 93 here, which is the full-cohort optimum before the isolation split is applied. With the 18 contract tenants removed from the pool, the optimum falls to 45. Both numbers are correct for their own question, and confusing them is a common enough error to state explicitly: the 93 figure answers "how many adapters if every tenant were poolable", and nobody is poolable.

The exponent range spans a factor of 4.9 in fleet size and 5.6 in net value. That is the quantified justification for instrumentation, and it is stronger than any qualitative argument about data quality.

### 6.13 Latency under the tenant mix

The adapter cache holds 45 slots at 36.83 MB each, which is 1.62 GB of device memory. On a 24 GB device with a 3.70 GB 4-bit base, a 1.05 GB bf16 head and 8 streams of 4,096-token KV at 537 MB each (4.29 GB), the total is 10.66 GB and leaves 55.6% headroom.

A miss costs one load: 36.83 MB at 800 MB/s is 46.0 ms. The pooled traffic is 89,501 requests per day and the cache covers 44.89% of it, so misses are 89,501 × 0.5511 = 49,324 per day, or 2,269 seconds of loading spread across three GPUs. That is 756 seconds per GPU per day, 0.87% of the day, and it is additive to the 2,400 ms p95 rather than multiplicative: a miss adds 46.0 ms, which is 1.92% of the objective.

The design conclusion is that a miss is affordable and a stall is not. If loads were serialised on the request path without prefetch, a burst of misses during a tenant's morning peak would queue, and queueing is what turns a 1.92% addition into a p99 breach. The mitigation is to warm the top 45 adapters at each shift boundary and to load asynchronously on miss while serving the shared adapter for that one request.

### 6.14 What the numbers say about rank choice

At rank 16 the adapter is 73.66 MB, so 45 slots occupy 3.23 GB instead of 1.62 GB, and a load costs 92 ms instead of 46. The device still fits: 3.70 + 1.05 + 3.23 + 4.29 = 12.27 GB, which is 51% headroom.

So rank is not constrained by memory at this fleet size. It is constrained by load latency on the miss path, where rank 16 doubles the added time, and by the per-request compute overhead, where rank 16 is 0.548% of base MACs against 0.274%. Neither is decisive, which means the rank choice should be made on measured quality rather than on the capacity argument that usually drives it.

That is a useful inversion: the conventional wisdom is that rank is a quality dial limited by memory. Here memory is not binding, so rank is a cost dial limited by load latency and quality, and the choice should be empirical.

### 6.15 The scrub cost and why it is small

Reassigning a KV buffer between tenants requires zeroing or overwriting it. A 4,096-token buffer is 537 MB, and zeroing at 50 GB/s takes 10.2 ms; at 2 TB/s it takes 0.256 ms. Either figure is small against a 2,400 ms p95.

The scrub is therefore cheap and the argument for skipping it must be something other than performance. In practice the argument is always "the allocator returns clean memory", which is true of the allocator and false of the buffer, because the buffer is a device allocation that the allocator reuses without writing. That distinction is the difference between a control and a belief.

The cost of getting it wrong is that the previous tenant's keys and values are present in memory when the next tenant's attention operation reads them, which is a disclosure with no log entry, no latency signature and no quality effect. It is the archetype of a boundary violation that no metric detects.

### 6.16 Why a learned router is the wrong answer here

A learned router that predicts the adapter from the request would be more flexible than an exact-match table, and it would be wrong for this problem. A router has a non-zero misroute probability by construction, and a misroute on an isolated tenant is the violation the whole architecture exists to prevent.

The exact-match table has no misroute probability conditional on a correct key, which means the failure surface moves entirely to the key and the table. Both of those are inspectable, testable and finite, whereas a router's error rate is a property of a model and must be estimated rather than bounded.

This is a general pattern: when a boundary is absolute, prefer a mechanism whose failure is countable over one whose failure is estimated.

## 7. The Finding

The two-tier fleet is worth building, is worth about $1.38 million a year in constructed value, and should contain 45 pooled adapters rather than the 93 a coverage target would produce. Three numbers carry that conclusion.

The first is $3,463. It is the annual value of the forty-sixth pooled adapter, and it is below the $3,500 cost of producing it. Every adapter beyond the forty-fifth loses money, and the loss grows as the tail flattens, so the correct fleet is bounded by a marginal condition rather than by a coverage ambition.

The second is 44.89%. It is the share of pooled traffic covered by 45 adapters, and it is the figure that makes the fleet small. Forty-five artifacts serve nearly half the pooled demand because the rank curve is steep, which is exactly why the tail cannot pay for itself.

The third is 8.4 × 10⁻³⁵. It is the collision probability of the 128-bit composite key, and it says the key is not where the risk lives. The risk lives in the swap protocol, the batch partition and the scrub, none of which has a probability bound, and all of which are process controls that must be exercised rather than proven.

The recommendation that follows is to ship the topology and the enforcement mechanism now, ship 45 pooled adapters and 18 isolated adapters, and treat the fleet size as a number that is expected to move once the traffic exponent and the incremental saving are measured rather than assumed.

### 7.1 What the finding is not

It is not a conclusion that tenant adapters are the main source of value. The isolated cohort's adapters return $901,771 against $63,000, and the pooled fleet returns $256,915 against $157,500. The isolated cohort is 15 times more profitable per dollar spent, and it is profitable because those tenants have no shared fallback rather than because their adapters are better.

It is not a conclusion that the dedicated instances are justified by the adapters. The instances cost $331,128 and are required by contract; the adapters on them cost $63,000 and are justified by the value. Conflating the two would let a future reviewer retire instances on the grounds that adapters no longer pay.

And it is not a claim about the correctness of any contract interpretation. The case models a requirement as "no shared processing" and designs for it; whether that is what any real term requires is `OPEN` and belongs to counsel.

### 7.2 The finding that is uncomfortable

Half of all batches at size 16 already contain more than one tenant, and 2.82% of slot occupancy belongs to a foreign tenant. For 222 of the 240 tenants that is the product. For the other 18 it is the violation, and the only reason it is not currently happening is that those 18 are on dedicated instances by accident of history rather than by design.

The uncomfortable part is that the design rule which makes this correct — class-partitioned batching — costs throughput and has no quality benefit. It will therefore be proposed for removal by anyone optimising the pool, and its removal will be undetectable in every metric the platform currently collects. That is the case's real warning, and it is about maintenance rather than about the initial decision.

### 7.3 How robust the finding is

The net value is robust: it stays above $80,000 across every cell of the sensitivity grid except none, and it is positive at skew 1.3 and at the highest adapter cost tested. The programme is worth doing under every parameter combination examined.

The fleet size is not robust at all. It ranges from 33 to 161 across the skew grid and from 49 to 144 across the cost grid, which is a factor of five either way. Anyone quoting "45 adapters" as a fixed plan has mistaken a parameter for a conclusion, and the honest statement is "45 under the central assumptions, revisable on measurement, with the cutoff rule being the durable part".

## 8. System/Solution Design

### 8.1 The topology in one paragraph

Twenty-one instances serve the platform. Three form the pooled tier and serve 222 tenants from a shared base with a shared adapter plus 45 cached tenant adapters. Eighteen are dedicated, one per contract tenant, each serving its own base with its own adapter. The two tiers share no queue, no batch, no cache and no instance; they share only the artifact registry and the evaluation harness, which are control-plane services and process no tenant content.

### 8.2 The composite key and its serialisation

The key is a byte string, not a hash, wherever possible: `tenant_id` as a 4-byte unsigned integer, `base_build` as a 3-byte build number, `tokenizer_version` as a 2-byte version, `template_version` as a 2-byte version, and `adapter_version` as a 3-byte version. That is 14 bytes, it is injective, and it has no collision probability at all.

A hash is used only where a fixed-width token is required, such as a cache index or a trace attribute, and there it is 128 bits. The design rule is that the authoritative comparison is always the exact byte string and the hash is only ever a lookup accelerator that is verified against the exact string on hit.

That verification step is what makes the hash safe. A hash collision that resolves to the wrong slot is caught on hit by the byte comparison, so the probability that matters is not the collision probability but the probability that the comparison is skipped. Designing for the former and neglecting the latter is how a scheme that is provably collision-free ends up misrouting in practice.

### 8.3 The adapter registry and its binding

The registry stores, per artifact: tenant identifier, base build at training, tokenizer version, template version, corpus hash, training timestamp, evaluation record, and status. An adapter in `serving` status must have an evaluation record whose base build equals the currently promoted build; the registry refuses to return artifacts that fail the check.

This is the atomic-release-bundle constraint as code rather than as policy. A release is not an adapter; it is a tuple `(base_build, tokenizer, template, adapter)` that was evaluated together, and the registry's unit of promotion is the tuple.

### 8.4 The swap protocol

A swap has four steps and the order is the protocol.

First the request is admitted to a batch only after its adapter slot is resolved, so no request is ever queued against an unresolved key. Second, if the slot is a miss, the request is served with the shared adapter for that turn and the tenant adapter loads asynchronously. Third, the newly loaded adapter enters the cache only after its load completes and its key is verified. Fourth, an eviction zeroes the buffer before the slot is returned to the pool.

The third step is what prevents a torn read: a partially loaded adapter is never visible to the routing table, because the table entry is written last. The fourth step is what prevents a residual read: the buffer is cleared before reuse rather than after, so a crash during reuse cannot expose the previous tenant's activations.

### 8.5 The batch partition and its cost

The queue is partitioned by isolation class at admission. Pooled requests fill pooled batches; each contract tenant's requests fill that tenant's batches on its own instance. A batch never spans classes and never spans tenants within the contract class.

The measured cost of that partition is throughput on the small tenants, which is irrelevant because those tenants are on dedicated instances. The cost on the pooled tier is zero, because the pooled class is the whole population minus 50.28% that was never in the pooled queue anyway. This is the unusual case where the correct design is also the cheap one, and the reason is that the partition line was already drawn by the contract.

### 8.6 Failover and what is not allowed

A pooled instance may fail over to another pooled instance. A dedicated instance may fail over only to another instance dedicated to the same tenant, which in practice means a standby or a cold restart.

It is not permitted to fail a contract tenant over to the pooled tier during an outage. That prohibition will be tested by a real outage at some point, and the pressure to violate it will be high because the alternative is a tenant experiencing downtime. The design therefore needs the prohibition to be enforced by the routing layer rather than by a runbook, because a runbook is exactly what an on-call engineer deviates from under pressure.

### 8.7 Artifact lifecycle and the retirement path

An adapter moves through `draft`, `evaluated`, `canary`, `serving` and `retired`. Promotion to canary requires an evaluation record with the current base build. Promotion to serving requires a canary period with no boundary violation and no quality regression beyond a stated interval.

Retirement is a first-class operation rather than a deletion. A retired adapter stays in the registry with its evaluation record, because the record is the evidence that the tenant's traffic was served by an artifact that was evaluated, and that evidence is what an audit asks for. Deleting the artifact and keeping only a log loses the linkage between what ran and what was tested.

### 8.8 Cache design and the prefetch policy

The cache is LRU over the composite key with a reserved minimum for the pooled shared adapter, which must never be evicted because it is the fallback path. The reservation is one slot, which is 36.83 MB and buys the guarantee that a miss has somewhere to go.

Prefetching warms the top 45 adapters at each shift boundary, which costs 45 × 46 ms = 2.07 seconds of load time and eliminates the morning burst. The warm set is derived from the previous day's rank order rather than from a static configuration, so a tenant moving up the curve is warmed automatically.

### 8.9 Operational readiness

Four conditions are checkable states rather than intentions. The registry refuses unevaluated tuples. The routing table is rebuilt from the registry on every promotion and its version is logged with every request. The scrub is instrumented with a counter that must equal the number of reassignments. And the batch log records tenant sets, so a co-tenancy violation is reconstructible.

The fifth condition is organisational and it is the one that fails: the on-call rotation must know that a pooled failover for a contract tenant is not a mitigation. That knowledge is a training artifact rather than a runbook line, and the case cannot supply it.

### 8.10 What the design deliberately does not do

It does not share adapters across tenants to reduce artifact count, because a shared artifact is a shared behaviour and the tenants who asked for customisation asked for the opposite. It does not train on contract-tenant data for pooled artifacts, because that would bake one tenant's phrasing into another's weights. And it does not use a learned router, for the reason in Section 6.16.

Each of those is a capability the platform could build and each is excluded for a boundary reason rather than a cost reason. Stating them prevents a future optimisation from reinstating them as obvious wins.

## 9. Evaluation & Validation

### 9.1 The measurement table

| Metric | Numerator | Denominator | Population | Window | Comparator | Status |
|---|---|---|---|---|---|---|
| Co-tenancy violations | requests in a batch with a foreign tenant from another class | requests in that class | contract tenants | continuous | zero | not executed |
| Key misroutes | requests served by an adapter whose exact key differs | requests | all | continuous | zero | not executed |
| Quality delta, pooled | pairs where the tenant adapter is preferred | paired requests | pooled, top 45 | 30 days | shared adapter | not executed |
| Added latency on miss | load time added to the served latency | misses | pooled | continuous | 46.0 ms | not executed |
| Negative test pass rate | cross-tenant probes returning another tenant's content | probes | all ordered pairs sampled | per release | zero | not executed |

Every row is `not executed`. The table defines what a run must produce; it does not report one.

### 9.2 Cross-tenant negative tests and how many

A negative test sends tenant A's request with tenant B's key and asserts that the response contains nothing from B. The test is a probe, and its value depends on how many probes are run and what failure rate they can detect.

Detecting a failure rate `r` with 95% confidence requires `n = ln(0.05)/ln(1−r)` probes. At `r = 10⁻³` that is 2,995 probes; at `10⁻⁴` it is 29,956; at `10⁻⁶` it is 2,995,731. The suite must therefore be sized against the failure rate it is meant to catch, and a suite of 600 probes can only certify an upper bound of 0.50% via the rule of three.

That is a weaker guarantee than it sounds. A 0.50% upper bound on misrouting is a budget of 234,000 events per year at this traffic, which no one would accept. The suite is therefore a detection mechanism for gross regressions rather than a certification of the boundary, and the certification comes from the design — exact keys, verified hits, no learned routing — rather than from testing.

### 9.3 What the negative tests can actually establish

Three things, and each is worth stating because each is achievable.

A regression test that catches a code change which stops verifying the key on hit. This is the failure the suite is best at, because it is deterministic and any probe catches it.

A coverage test that every ordered pair of tenants has been exercised at least once somewhere in the cycle. With 240 tenants there are 57,360 ordered pairs and 28,680 unordered; sampling 3,000 pairs covers 5.23% of them and is enough to establish that no single-tenant hardcoding exists, which is the realistic bug.

A load test that runs the probes under concurrent batch pressure, because the failures that matter occur when the batch is full and the queue is deep. A probe run in isolation tests the happy path.

### 9.4 Quality evaluation and the paired design

The quality delta is measured on paired requests: the same request scored with the tenant adapter and with the shared adapter, judged by reviewers who do not know which is which. Pairing removes request difficulty, which is the dominant variance component in a domain where some questions are unanswerable.

The sample per tenant must be small because there are 45 tenants, so the design pools: 600 scored pairs per tenant per round at 45 tenants is 27,000 judgements, which at six rounds a year is 162,000 judgements annually. That number is the real driver of the $3,500 cost, and it is also the reason reducing rounds is the most effective cost lever.

### 9.5 Guardrails and stop conditions

Three guardrails can stop a release. Any co-tenancy violation attributed to the release. Any negative probe that returns foreign content. Any adapter served whose registry evaluation record names a different base build.

The stop conditions are absolute rather than threshold-based because the constrained quantity is a predicate, not a rate. A rate can be monitored; a predicate is either satisfied or violated, and a single violation means the release is withdrawn rather than tuned.

### 9.6 What would falsify the recommendation

Five observations would falsify parts of it. A measured incremental saving below 0.020 minutes reduces the optimal fleet to about 12 adapters and makes the pooled tier nearly pointless. A measured skew exponent of 1.3 reduces it to 33. An adapter cost above $5,500 reduces it to 49 and changes the shape of the argument rather than the conclusion.

A contract interpretation concluding that instance-level isolation is sufficient would remove the partition and increase throughput, without changing the fleet size. And a base model with a much longer refresh cadence would remove the cadence-alignment constraint entirely and reduce the artifact cost, which would enlarge the fleet.

### 9.7 Uncertainty and what the intervals do not cover

The value figures are point estimates on constructed inputs. No interval is computed because the inputs are not samples; there is no sampling distribution to describe. The sensitivity grid in 4.5 is the honest substitute, and it is a grid rather than an interval because the uncertainty is parameter uncertainty rather than statistical uncertainty.

What no interval covers is the possibility that the value model is wrong in structure. If tenants who receive adapters use the assistant more, the traffic itself grows and the value per adapter rises with its own adoption. If they use it more because it is better, the value is understated; if they use it more because it is new, the value is overstated and will decay. Nothing in this design distinguishes those.

## 10. Action Taken

### 10.1 The decision

The platform lead approves the two-tier topology, 45 pooled tenant adapters, 18 isolated tenant adapters, the 14-byte composite key with 128-bit lookup accelerators, the four-step swap protocol, the class-partitioned batch queue, and the absolute prohibition on pooled failover for contract tenants.

The lead separately authorizes the instrumentation that would make the fleet size factual: 90 days of per-tenant request counts, and three tenant-adapter evaluations at ranks 5, 30 and 90 to calibrate the incremental saving.

### 10.2 What is explicitly refused

Four things are refused and each is recorded with its reason.

A coverage-targeted fleet of 93 adapters is refused because the extra 48 cost $168,000 and return $128,398. A learned router is refused because its misroute rate must be estimated where the table's is zero. An amortisation of dedicated instances into the adapter case is refused because it makes retirement appear to save more than it does. And a temporary pooled failover for contract tenants is refused because an outage is precisely when it would be used.

### 10.3 The sequencing

Instrumentation starts immediately and runs in parallel. The 18 isolated adapters ship first, because their cohort has no shared fallback and their return is 14.3 to 1. The pooled shared adapter ships with them, because a tenant adapter on top of an unadapted base overstates its own contribution and would corrupt the later calibration. The 45 tenant adapters ship in rank order, with a checkpoint at 20 to verify the marginal condition against observed value.

Shipping the isolated cohort first is not a preference. It is the sequencing that produces the measurement the pooled decision needs, because those 18 tenants are the only place where an own-adapter gain can be measured without a shared-adapter confound.

### 10.4 The decision record an auditor would want

| Field | Entry |
|---|---|
| Decision | Two-tier fleet: 45 pooled adapters, 18 isolated adapters, exact-key routing, class-partitioned batching |
| Owner and date | Platform lead, 2026-09-01T00:00Z |
| Baseline | Shared base, one prompt, 21 instances, no adapters |
| Comparator | Shared adapter for pooled tenants; no-adapter for isolated tenants |
| Threshold | Adapter retained iff its tenant's annual value exceeds $3,500 |
| Cost authorised | $224,000 per year of adapters; $386,316 of instances unchanged |
| Reversal condition | Measured incremental saving below 0.020 min; skew above 1.2; adapter cost above $5,500 |
| Evidence status | All parameters ILLUSTRATIVE; no measurement executed; no contract interpretation asserted |

### 10.5 Who is accountable for the boundary

Accountability for the isolation predicate sits with security engineering, not with the platform lead. The platform lead owns the topology and the cost; security owns the assertion that no batch contained a foreign tenant and that no buffer was reused unscrubbed.

The split matters because it gives one named group the authority to block a release for a boundary reason without arguing about quality or cost. A single owner for both would face a conflict on exactly the release where the trade-off appears, which is the release where the boundary is most likely to be relaxed.

## 11. Outcome & Results

### 11.1 There is no outcome

No adapter has been trained on real tenant traffic, no batch has been partitioned, no probe has been run and no contract has been reviewed. Everything in this section is a projection from constructed inputs, presented so that a future reader can compare a real result against a stated expectation.

### 11.2 The projection, shown with its arithmetic

Value arrives from three sources and the projection multiplies each by its cohort's traffic share.

```text
FORMULA
  V_total = c_min * R * D * ( g_own*S_iso + g_shared*S_pool + delta*S_pool*C_45 )

WHERE
  S_iso = 0.5028, S_pool = 0.4972, C_45 = 0.4489
  c_min*R*D = 0.68333 * 46,800,000 = $31,979,844 of minute-value
  g_own = 0.060, g_shared = 0.024, delta = 0.036

CONSTRAINT
  The three terms are additive only if the gains are independent, which
  holds for minutes saved but not for adoption effects, where a tenant with
  an adapter may generate more traffic and therefore more value.

NUMERICAL EXAMPLE
  term 1 = 31,979,844 * 0.060 * 0.5028 = $  964,769
  term 2 = 31,979,844 * 0.024 * 0.4972 = $  381,631
  term 3 = 31,979,844 * 0.036 * 0.4972 * 0.4489 = $257,010
  sum    = $1,603,410
  cost   = 64 * 3,500 = $224,000
  net    = $1,379,410
  return ratio = 1,603,410 / 224,000 = 7.16

INTERPRETATION
  The programme returns 7.16 times its adapter cost under the central
  parameters. The figure is dominated by the isolated cohort, which
  contributes 60.2% of the value from 28.1% of the adapters.
```

The 60.2% and 28.1% comparison is the one a business case should lead with, because it explains why the isolated cohort ships first and why the pooled fleet is the marginal programme rather than the main one.

### 11.3 What a real outcome record must contain

It must contain the per-tenant traffic counts that let a reviewer recompute the rank curve, the per-round evaluation records with their base builds, the batch composition log for the whole window, the scrub counter, and the negative probe results with their sample size and the bound they support.

Without the batch log the isolation claim is an assertion rather than a record. Without the probe sample size the negative result is unbounded. Those two omissions are the ones a real record most often has, because both are expensive to retain and neither is needed for the quality story.

### 11.4 The result that would change the plan

A measured incremental saving of 0.020 minutes or below would shrink the pooled fleet to roughly 12 adapters and make the topology argument rest almost entirely on the isolated cohort. A measured skew above 1.2 would do the same by a different route.

An adapter cost above $5,500 would reduce the fleet to 49 and, more importantly, would change the recommendation from "build the fleet" to "build the fleet only where the contract requires it".

### 11.5 The number that will be quoted and should not be

The $1.38 million net will be quoted, because it is the largest number in the case and it is arithmetically correct. It is also constructed, and quoting it as a projection for any real platform is a category error.

The numbers that should be quoted are the 45, the $3,463 marginal value and the 2.82% co-tenant slot share. Those three transfer as method even when the parameters do not, because each is the output of a rule rather than of a dataset.

## 12. Failure Modes & What Went Wrong

Seven mechanisms are distinct enough to warrant separate entries. Each states the mechanism, a competing explanation, a check that separates them, and the correction.

### 12.1 F1 — Co-tenancy reintroduced by a batching optimisation

A queue-depth optimisation merges the partitioned queues to fill batches during off-peak hours. Throughput improves, no tenant sees foreign data, and the isolation predicate is violated for every batch that spans classes.

Competing cause: the batch log is wrong. Check: recompute the tenant sets from the batch identifiers in the request log rather than from the planner's own counters; if the two disagree, the instrumentation is at fault, and if they agree, the planner is.

Correction: make the class a property of the queue rather than a filter applied to a merged queue, so that merging requires deleting a queue rather than amending a predicate. Prevention: the negative suite runs under load with a contract tenant as the foreign party.

### 12.2 F2 — KV buffer reuse without scrub

The allocator returns a buffer that held another tenant's keys and values, and the new request's attention reads positions the new request did not write.

Competing cause: a tokenizer or template mismatch producing incoherent output. Check: run two synthetic requests with known distinct prompts back to back on the same slot and inspect whether the second output contains tokens from the first; a template mismatch would corrupt all outputs, whereas a residual read corrupts only the boundary positions.

Correction: zero on reassign rather than on allocate, and instrument the counter so that the number of scrubs equals the number of reassignments. Prevention: the counter is a release gate, not a dashboard.

### 12.3 F3 — Adapter served on a moved base

The base promotes on its 63-day cadence and an adapter trained on the previous build remains in `serving`. Shapes match, the load succeeds, and the delta composes with weights it was never trained against.

Competing cause: genuine distribution shift from new tenant documents. Check: hold the request set fixed and evaluate both the adapter-on-old and adapter-on-new builds; if only the adapter arm moves, the binding is the cause, because a distribution shift would move the shared-adapter arm too.

Correction: the registry's serving check compares the adapter's training build to the promoted build and refuses mismatches. Prevention: the cadence is event-driven on promotion, and the alignment table in 6.7 shows that any calendar cadence is wrong except a divisor.

### 12.4 F4 — Torn read during a swap

A request is routed to a slot while the adapter is still loading, so it composes with a partially written delta. The output is not obviously wrong, which is what makes it dangerous.

Competing cause: an under-trained adapter producing slightly unusual output. Check: correlate the anomaly timestamps against the swap log; a torn read is time-clustered at swaps, whereas an under-trained artifact is uniform in time.

Correction: write the routing table entry last, after the load completes and the key is verified. Prevention: a canary slot that performs a synthetic generation immediately after every swap and compares the result to a recorded baseline.

### 12.5 F5 — Key verification skipped on cache hit

The hash is treated as the identity and the byte comparison is skipped on hit, which is correct until a collision or a table bug produces a wrong-slot hit.

Competing cause: a hash collision, which is 8.4 × 10⁻³⁵ and therefore almost certainly not the cause. Check: log the exact key alongside the hash on every hit and assert equality; the assertion fires on the real bug long before any collision exists.

Correction: the comparison is mandatory and unconditional. Prevention: a test that injects a deliberately colliding hash into the table and asserts that the request is served correctly or refused, never served from the wrong slot.

### 12.6 F6 — Training on the wrong cohort

A pooled adapter's corpus accidentally includes contract-tenant requests, because the filter is on a tenant attribute that was missing for newly provisioned tenants and defaulted to pooled.

Competing cause: the pooled adapter is simply better on pooled traffic. Check: recompute the corpus tenant set from the stored corpus hash and compare against the intended cohort; a default-missing attribute shows as a subset of contracts in the corpus, and a better adapter does not.

Correction: the filter fails closed, so a missing attribute excludes the request rather than including it. Prevention: the corpus hash is recorded in the evaluation record and the tenant set is an artefact of every training run.

### 12.7 F7 — Retirement mistaken for saving

An adapter is retired for cost reasons and the saving recorded is the full instance cost, because the instance was amortised into the adapter case.

Competing cause: the instance genuinely is no longer needed. Check: read the contract metadata for the tenant; if the instance requirement is independent of adaptation, the saving is $3,500 rather than $394,128.

Correction: cost attribution separates contract-mandated infrastructure from optional adaptation. Prevention: the decision record states which costs would exist regardless of the decision, and every later review compares against that statement.

### 12.8 A mechanism that has no discriminating check

One failure is undetectable from the platform's own data: a provider quietly changing model behaviour without changing the build number. Every control above assumes the build identifier is truthful, and if it is not, the registry's binding check passes while the binding is broken.

The only defence is a canary request set with recorded reference outputs, evaluated on every build promotion, whose score is compared to the recorded history. If the canary moves and the identifier did not, the identifier is lying. That is a monitoring control, and it belongs in the register as residual risk rather than as a mechanism with a fix.

### 12.9 Recovery sequences

For F2 the sequence is: stop the affected replica, identify reassignments without scrubs from the counter, treat every such boundary as a potential disclosure and escalate, then fix and re-deploy with the counter as a gate. The sequence is not "fix and continue", because the disclosure question is not one the platform resolves alone.

For F3 the sequence is: withdraw the adapter from serving, identify every request served since the base promotion, re-evaluate the adapter against the new base, and either retrain or retire. Requests served in the window are re-checked against the quality bar, because an adapter on the wrong base degrades silently rather than failing.

For F1 the sequence is: revert the planner change, reconstruct the affected window from the batch log, quantify the co-tenancy exposure, and escalate to counsel. The engineering fix is minutes; the escalation is the substance.

### 12.10 Why the register is ordered this way

The entries are ordered by the ratio of consequence to detectability, not by likelihood. F1 and F2 sit at the top because a single occurrence is a disclosure and neither produces a symptom in any metric. F3 sits in the middle because it degrades quality visibly, once someone looks. F5 and F6 sit lower because they are caught by assertions rather than by judgement.

Ordering by likelihood would put F3 first, since base promotions happen twelve times a year and a scrub bug is a one-off. That ordering would be wrong for a register whose purpose is to direct attention to what a release gate must catch.

## 13. Transferable Lesson

### 13.1 The reusable rule

**A boundary that is absolute must be enforced by a mechanism whose failure is countable, not by a mechanism whose failure is estimated.**

An exact-match table over an injective key has no misroute rate conditional on a correct key. A learned router has a rate. A zero-on-reassign buffer policy has a counter that must equal the reassignment count. An "allocator returns clean memory" assumption has a belief. The rule is to prefer the countable enforcement even when the estimated one is more capable.

### 13.2 The second rule: size a fleet at the margin, not at coverage

Any fleet of per-entity artifacts — adapters, indexes, caches, dedicated instances, per-region stacks — has the same structure: a constant per-artifact cost and a value that falls with the entity's size. The optimal size is set by the marginal entity crossing the cost, and coverage targets systematically overshoot because they optimise a share rather than a difference.

The overshoot here is 48 artifacts and $39,602 a year. The rule generalises: compute the marginal entity's value before choosing a coverage target, and if the target and the marginal condition disagree, the marginal condition is right.

### 13.3 What transfers and what does not

The methods transfer: the rank curve and its normaliser check, the marginal cutoff, the birthday bound applied to a routing key, the batch-composition exposure formula, the cadence alignment calendar, and the negative-test sample sizing. Each is a small computation with a general form.

The numbers do not transfer. The 45, $3,463, 2.82%, 8.4 × 10⁻³⁵ and 16.67% are artefacts of this tenant population and this cost model. A reader who quotes the fleet size into a different platform has learned the answer rather than the question.

### 13.4 The compact version

Four lines carry the case. Value per adapter is the gain times the cost per minute times the tenant's annual requests. The fleet is the largest prefix whose marginal adapter clears the cost. Co-tenancy exposure is `(1 + (B−1)·HHI − 1)/B`. The key collision bound is the birthday bound, and at 128 bits it is not the risk; the swap protocol is.

### 13.5 The counter-lesson

If every tenant were poolable, the optimum would be 93 adapters and the net $298,152. If no tenant were, the platform would run 240 dedicated instances at $4,415,040 and the programme would be indefensible. The two-tier shape is what makes the programme work, and it exists only because 18 of 240 contracts differ.

That is worth stating because it inverts the usual framing. The contract tenants are described internally as the expensive problem, and they are also the reason the architecture has a correct line to divide on. A platform with uniform terms would have to invent a partition or accept co-tenancy everywhere.

### 13.6 Where the reasoning is thinnest

Two places. The incremental saving of 0.036 minutes is the least supported parameter and the one the fleet size is most sensitive to, and it has almost certainly never been measured on any platform. And the single-exponent traffic model understates the true tail heaviness, which biases the fleet downward: a heavier tail means the marginal tenant crosses the cost at a lower rank, so the real optimum is likely smaller than 45 rather than larger.

A reader should therefore treat 45 as an upper bound under central assumptions rather than as an estimate, and should treat the cutoff rule as the durable output.

## 14. Connections & Cross-References

### 14.1 Within this unit

- [Should support draft generation use a domain adapter?](./01_support_domain_adaptation.md) — the single-tenant version of the same adapter economics, where the deciding quantity is the marginal accuracy point rather than the marginal tenant.
- [LoRA Parameter-Efficient Fine-Tuning](../Cheat_Sheets/01_LoRA_Parameter_Efficient_Finetuning.md) — the rank and target-module arithmetic that produces 18.4 million parameters at rank 8.
- [QLoRA Quantized Base Adapter Fine-Tuning](../Cheat_Sheets/02_QLoRA_Quantized_Base_Adapter_Finetuning.md) — the quantised base loading that makes 45 cached adapters fit on one device.
- [Advanced PEFT Methods](../8.4_Advanced_PEFT_Methods.md) — multi-adapter routing and composition, including the adapter-identity concerns this case treats as a boundary control.
- [Model Merging and Weight Averaging](../8.5_Model_Merging_and_Weight_Averaging.md) — the compatibility gate that generalises the base-build binding here.
- [Why PEFT and the Fine-Tuning Spectrum](../8.1_Why_PEFT_and_the_Fine_Tuning_Spectrum.md) — the parameter and storage accounting behind the per-tenant artifact cost.

### 14.2 Source boundaries

The planning row for this topic is `PLAN-2026-IR004` in the collection [catalog](./INDEX.md), which asked for a case on adapters that are trained, selected, versioned and rolled back per tenant, validated by cross-tenant negative tests, quality measurement, swap races and latency under a tenant mix. Those five validation concerns correspond to Sections 9.2, 9.4, 12.4, 6.13 and 6.4 respectively.

The mapping from source module titles to these lessons is recorded in the unit [INDEX](../INDEX.md). The original lesson bodies were unavailable for inspection, so no claim is made that this case reproduces their content. No vendor documentation, benchmark or measurement is cited; where a reader would expect one there is an `OPEN` or `not-measured` label.

### Interview pack

#### Q1. Why does the fleet stop at 45 adapters rather than covering 80% of traffic?

**Answer.** Because the forty-sixth pooled adapter returns $3,463 a year against a $3,500 cost. The pooled value pool is `0.036 × 0.68333 × 23,268,960 = $572,318`, the forty-sixth tenant's pooled share is 0.00605, and the product is $3,463.

**Reasoning.** Value falls with rank and cost is constant, so the optimal set is a prefix of the rank ordering, and its size is set by where the marginal value crosses the constant.

**Follow-up.** What does 80% coverage cost and return? 93 adapters cost $325,500 and return $385,313 in the pooled tier; the extra 48 cost $168,000 and return $128,398, a net loss of $39,602.

**Red flag.** Sizing the fleet by a coverage target, which optimises a share rather than a difference and overshoots by roughly a factor of two.

**Scoring.** 0 = picks 93 or "as many as possible"; 1 = notes cost falls off; 2 = computes the threshold; 3 = also computes the coverage alternative and its loss.

#### Q2. Compute the collision probability for a 32-bit tenant key and say whether it is acceptable.

**Answer.** `C(240,2)/2^32 = 28,680/4,294,967,296 = 6.678 × 10⁻⁶`. The budget is one event per 46.8 million requests, or 2.137 × 10⁻⁸, so 32 bits fails by a factor of 312.

**Reasoning.** The budget is derived from traffic, not chosen, so the comparison is between two probabilities rather than between a probability and a feeling.

**Follow-up.** What width clears it? 48 bits gives 1.02 × 10⁻¹⁰ and 128 bits gives 8.4 × 10⁻³⁵, which is 27 orders of margin.

**Red flag.** Saying "240 keys will never collide", which treats a small population as a zero population.

**Scoring.** 0 = says collisions are impossible; 1 = computes the birthday bound; 2 = derives the budget; 3 = also states why key width is not the real risk.

#### Q3. A colleague says batching shares no data, so co-tenancy is harmless. What is wrong with that?

**Answer.** The contract is modelled as requiring no shared processing, and a batched decode step shares arithmetic across requests. At batch 16 with HHI 0.0301, 2.82% of slot occupancy is a foreign tenant and about 51.8% of batches contain more than one tenant.

**Reasoning.** "No data crosses" and "no shared processing" are different predicates. A batched forward pass satisfies the first and violates the second.

**Follow-up.** What does satisfying the second cost? Partitioning the queue by class, which costs throughput on small tenants and nothing on the pool, because the partition line already existed.

**Red flag.** Reaching for the data-path argument and concluding the boundary is met.

**Scoring.** 0 = agrees with the colleague; 1 = says data does not cross; 2 = distinguishes the predicates; 3 = also quantifies exposure and the partition cost.

#### Q4. Derive the pool size and explain why the mean load is the wrong basis.

**Answer.** `S = 1400/14000 + 180/2000 = 0.19` GPU-s per request, so 180,000 requests is 9.50 GPU-hours a day. The peak hour carries 11% of requests and needs 1.045 GPU-hours, so a 3-GPU pool runs at 34.8% peak and 13.2% daily.

**Reasoning.** Latency is determined by queueing at the peak, and mean utilisation determines cost. Sizing on the mean gives a pool that breaches its objective during the only period anyone measures.

**Follow-up.** What does the third GPU cost, and what does it buy? $18,396 a year and headroom for a lost replica; the adapter fleet it protects costs $157,500.

**Red flag.** Reporting 9.50 GPU-hours as the pool size, which is a daily total and not a capacity.

**Scoring.** 0 = no units; 1 = computes daily hours; 2 = computes peak; 3 = also compares the marginal GPU to the programme it protects.

#### Q5. Why is a 42-day retraining cadence wrong when the base refreshes every 63 days?

**Answer.** Because 42 does not divide 63. Iterating an 8,190-day calendar with a 63-day base period and a 42-day adapter period leaves the adapter bound to a superseded base on 16.67% of days.

**Reasoning.** A retrain is valid only if the base has not moved since training. Alignment is a property of the periods, not of the effort.

**Follow-up.** What cadences are free? 21, 63 and any divisor of 63. A 91-day quarterly gives 61.54% and a 126-day gives 50.00%.

**Red flag.** Choosing a cadence on operational convenience and assuming the adapter stays valid between retrains.

**Scoring.** 0 = picks a round number; 1 = notices the mismatch; 2 = computes a fraction; 3 = also explains why 126 days aligns in count but not in content.

#### Q6. The 126-day cadence fits two adapter cycles in two base cycles. Why is it still 50% stale?

**Answer.** Because the retrain lands on the same day the base moves, so the artifact produced that day was trained against the previous base and is stale from the moment it is promoted.

**Reasoning.** Aligning cycle counts is not the same as aligning content. The condition is that the last retrain falls strictly after the last base promotion.

**Follow-up.** What fixes it? An event-driven trigger on promotion rather than a calendar, which makes the condition true by construction.

**Red flag.** Checking only that both periods divide a common multiple.

**Scoring.** 0 = says it aligns; 1 = notes some lag; 2 = computes 50%; 3 = also states the strict-inequality condition and the event trigger.

#### Q7. Size the cross-tenant negative test suite and say what a passing run establishes.

**Answer.** Detecting a 10⁻³ failure rate at 95% needs 2,995 probes. A 600-probe suite that passes establishes only an upper bound of 0.50% via the rule of three, which is a budget of 234,000 events a year.

**Reasoning.** A zero-failure run bounds the rate by roughly 3/n, so the bound is set by the sample size rather than by the result.

**Follow-up.** What is the suite actually good for? Catching a deterministic regression such as a skipped key verification, and establishing that no single-tenant hardcoding exists.

**Red flag.** Treating a clean run as evidence that the boundary holds, which confuses a detection mechanism with a proof.

**Scoring.** 0 = says the suite proves isolation; 1 = gives a probe count; 2 = computes the bound; 3 = also separates detection from certification.

#### Q8. Diagnose: outputs for one tenant become subtly off after a base refresh, and the shared-adapter arm is unaffected.

**Answer.** The tenant adapter is bound to the wrong base. A distribution shift would move both arms; a binding failure moves only the adapter arm, because only the adapter composes with weights it was not trained against.

**Reasoning.** The discriminating check is a fixed request set evaluated on both builds in both arms, which isolates the binding from the data.

**Follow-up.** What stops it recurring? The registry's serving check compares the adapter's training build to the promoted build and refuses a mismatch.

**Red flag.** Retraining first, which may fix the symptom while leaving the registry check absent.

**Scoring.** 0 = blames the data; 1 = names drift; 2 = identifies the binding; 3 = also specifies the two-arm check and the registry control.

#### Q9. Why is a learned router unacceptable when an exact-match table is not?

**Answer.** A router has a non-zero misroute probability that must be estimated; a table has zero conditional on a correct key. When the boundary is absolute, an estimated failure rate cannot satisfy it.

**Reasoning.** The failure surface matters more than the failure rate. A table's failures are in the key and the table, both inspectable and finite; a router's failures are in a model.

**Follow-up.** What if the tenant count grew past what a table can hold? Then the table is sharded by tenant, which is still exact.

**Red flag.** Proposing a router with a measured accuracy as if accuracy were the same property as correctness.

**Scoring.** 0 = prefers the router's flexibility; 1 = notes the error rate; 2 = distinguishes estimated from countable failure; 3 = also proposes sharding as the growth path.

#### Q10. Cost the per-adapter artifact and identify the dominant term.

**Answer.** `C_train = 1,200,000/5,200/3600 × 2.10 = $0.135`; `C_eval = 600 × 0.55 = $330`; six rounds is $1,980.81; plus $1,520 coordination gives $3,500.81. Training is 0.0039% of the total and evaluation is 56.6%.

**Reasoning.** The artifact is expensive because it is re-evaluated six times a year, not because it is trained.

**Follow-up.** What does halving the rounds do? It saves $990 per adapter, moves the fleet from 45 to 66, and changes no model behaviour.

**Red flag.** Budgeting GPU-hours and concluding adapters are cheap, or quoting $0.135 as the cost of an adapter.

**Scoring.** 0 = quotes compute only; 1 = includes evaluation; 2 = computes the total; 3 = also identifies rounds as the cost lever and its effect on fleet size.

#### Q11. Which assumption moves the fleet size most, and by how much?

**Answer.** The traffic skew exponent. At 0.7 the optimum is 161 adapters, at 1.3 it is 33, and at the central 0.9 it is 45 after the isolation split. That is a factor of 4.9 across a plausible range.

**Reasoning.** The fleet is a prefix of the rank curve, so the curve's shape directly sets where the marginal value crosses the cost.

**Follow-up.** Which assumption moves the net value most? Also skew, from $453,353 to $80,416, because it changes both coverage and the value per rank.

**Red flag.** Naming adapter cost, which matters but changes the fleet by a factor of three at most over a plausible range.

**Scoring.** 0 = no assumption; 1 = names skew qualitatively; 2 = gives the range; 3 = also explains the mechanism and the net-value effect.

#### Q12. Explain why amortising dedicated instances into the adapter case is an error with a numeric consequence.

**Answer.** Eighteen instances cost $331,128 and exist because of contract terms, not because of adaptation. Amortising them into the adapter case makes retiring 18 adapters appear to save $394,128 when the actual saving is $63,000, a 6.3-fold overstatement.

**Reasoning.** A cost that is invariant to the decision cannot be a cost of the decision. Including it makes the retirement case look stronger than it is and invites a wrong reversal.

**Follow-up.** How should the record state it? By naming which costs would exist regardless of the decision, so a later reviewer compares against that statement.

**Red flag.** Building a single total-cost-of-ownership figure and reading the adapter decision out of its slope.

**Scoring.** 0 = amortises everything; 1 = separates them without a consequence; 2 = computes the overstatement; 3 = also prescribes how the decision record states invariance.

#### Q13. A swap window is 50 ms and peak traffic is 5.50 requests per second. How often does a tear risk occur?

**Answer.** 0.275 requests fall inside a single swap window at peak, which is 0.83 requests per day across three daily swaps, or about 303 per year. That is frequent enough that correctness must not depend on the race not happening.

**Reasoning.** A rare event at request scale becomes routine at annual scale, so the protocol must be safe at the event frequency rather than at zero.

**Follow-up.** What makes it safe? Resolving the slot before admission, writing the routing entry last, and serving the shared adapter during a miss.

**Red flag.** Concluding that the window is too small to matter, which is an argument about per-swap probability applied to an annual question.

**Scoring.** 0 = says it cannot happen; 1 = estimates a probability; 2 = computes annual events; 3 = also names the protocol steps that make it safe.

#### Q14. Why ship the 18 isolated adapters before the 45 pooled ones?

**Answer.** Because their cohort has no shared fallback, so their measured gain is uncontaminated, and because they return $901,771 against $63,000, a ratio of 14.3. Shipping them first also produces the calibration the pooled decision needs.

**Reasoning.** A pooled tenant adapter measured on top of an unadapted base would attribute the shared gain to the tenant artifact and overstate the incremental saving that sizes the fleet.

**Follow-up.** What would shipping pooled first cost? A miscalibrated incremental saving, which is the parameter the fleet size is most sensitive to.

**Red flag.** Sequencing by cohort size or by tenant pressure, which optimises the relationship rather than the measurement.

**Scoring.** 0 = no reason; 1 = says it is easier; 2 = gives the return ratio; 3 = also explains the confound and the calibration value.

#### Q15. Distinguish the two failure classes that an isolation audit should test for.

**Answer.** Cross-tenant data exposure, where content from one tenant reaches another's response, and cross-tenant compute sharing, where no content crosses but requests are evaluated in one forward pass. The first is testable by probes; the second only by batch-composition logs.

**Reasoning.** The two have different detection mechanisms because one leaves a content trace and the other leaves only a scheduling trace.

**Follow-up.** Which is harder to detect after the fact? Compute sharing, because without a batch log it is unreconstructible, whereas a content exposure can be searched in outputs.

**Red flag.** Testing only the data path and asserting the isolation requirement is met.

**Scoring.** 0 = one class; 1 = names exposure only; 2 = names both; 3 = also pairs each with its detection mechanism and its retention requirement.

#### Q16. What does the 2.82% co-tenant slot share imply for a batch-size increase from 16 to 32?

**Answer.** Almost nothing: the share moves from 2.82% to 2.92% because `(E[distinct]−1)/B` grows sublinearly. What changes materially is the number of batches that are mixed, and the throughput gain from a larger batch.

**Reasoning.** HHI fixes the expected number of distinct tenants at `1 + (B−1)·HHI`, so the foreign-slot share approaches HHI as B grows rather than approaching one.

**Follow-up.** So does a larger batch make isolation worse? Not per slot, but it makes each violating batch larger, which matters for exposure volume rather than for exposure rate.

**Red flag.** Assuming the share scales with batch size and rejecting larger batches for the wrong reason.

**Scoring.** 0 = says it doubles; 1 = says it grows; 2 = computes both shares; 3 = also explains the asymptote and the volume-versus-rate distinction.

## 15. Quick Reference Card

| Field | Case reference |
|---|---|
| **Decision** | Two-tier fleet: 45 pooled tenant adapters over a shared base plus shared adapter; 18 isolated tenants on dedicated instances with their own adapters; exact-key routing. |
| **Owner and grain** | Platform lead for assistant services; one program decision on fleet size, key scheme, swap protocol and rollback. |
| **Baseline and comparators** | Shared base and one prompt; pooled tenants compare against the shared adapter, isolated tenants against no adapter. |
| **Marginal rule** | Retain an adapter iff its tenant's annual value exceeds $3,500; the forty-sixth pooled adapter returns $3,463, which is the cutoff. |
| **Core fleet check** | Pooled value pool `0.036 × 0.68333 × 23,268,960 = $572,318`; at 44.89% coverage the fleet returns $256,915 against $157,500. |
| **Key bound** | Budget `1/46,800,000 = 2.14 × 10⁻⁸`; 32-bit gives 6.68 × 10⁻⁶ (fails), 128-bit gives 8.4 × 10⁻³⁵ (passes by 27 orders). |
| **Co-tenancy check** | HHI 0.0301 over 240 tenants; at batch 16 the foreign-slot share is 2.82% and about 51.8% of batches are mixed. |
| **Capacity check** | `0.19` GPU-s/request gives 9.50 GPU-hours a day; a 3-GPU pool runs 34.8% at peak and 13.2% daily. |
| **Lifecycle gate** | Cadence must divide 63 days; 42 gives 16.67% staleness, 91 gives 61.54%, and promotion-triggered retraining is the only aligned rule. |
| **Programme arithmetic** | Value $1,603,410 against $224,000 of adapters, a ratio of 7.16; instances of $386,316 are invariant to the decision. |
| **Evidence status** | All inputs `ILLUSTRATIVE`; arithmetic `DERIVED` and rechecked; no batch, probe, adapter or contract review executed. |
