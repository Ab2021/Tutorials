---
case_id: "CASE-49-003"
case_title: "Tenant adapter isolation in a shared multi-adapter serving pool"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "llm-agentic"
domain: "Model customization / multi-tenant adapter serving and isolation"
primary_pattern: "authorization-scope"
secondary_patterns: ["atomic-release-bundle", "rollout-regression"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "Model platform owner, with security and privacy sign-off"
source_index: "INDEX.md"
review_trigger: "Any change to adapter cache keying, base checkpoint, tokenizer, slot count, tenant demand mix, or the release/rollback procedure."
---

# Tenant Adapter Isolation in a Shared Multi-Adapter Serving Pool

## 1. The One-Line Framing

Given **one customer-support request from an authenticated tenant** on a shared inference platform, decide **how tenant-specific low-rank adapters are keyed, cached, pinned and rolled back** so that the response is produced for that tenant, by that tenant's adapter version, on a compatible base checkpoint, subject to **zero cross-tenant adapter resolution and a 900 ms p95 response budget**, compared with **the current process of building a merged full checkpoint per tenant and pinning one dedicated GPU to each**.

The task category is *system operation with an allocation decision*: every request triggers a routing and caching decision, and the platform team must also make one one-off architectural choice about whether to consolidate. The decision frequency is high — the illustrative workload is 18 requests per second on average and 120 requests per second at peak — so a small per-request correctness defect compounds into a large exposure within minutes.

**Evidence declaration.** This is an **illustrative design exercise**, not a company, product, incident or measured deployment. Tenant names, traffic volumes, hardware prices, memory budgets, adapter dimensions, quality results and costs below are constructed teaching values chosen so that the arithmetic is checkable by hand. Nothing in this file reports a real system's behaviour. Where a public source is named, it is named as a *method reference* for a technique, not as evidence about this scenario.

**The decisive trade-off.** Consolidating 720 dedicated GPUs into a four-lane shared pool with a 512-slot adapter cache reduces illustrative serving cost from \$1,314,000 to \$9,125 per month, a factor of **144**. That saving is real only if two properties hold: the cache key can make cross-tenant resolution *structurally impossible* rather than statistically unlikely, and the capacity argument survives a change in the shape of tenant demand. Both properties are properties of the *design*, not of the model. A pool that is 99.99% correct is not a cheaper version of the dedicated baseline; it is a different risk product, because the baseline's failure mode is a cold start and the pool's failure mode is one customer reading another customer's customised behaviour.

**Why the obvious framing misses the point.** The intuitive framing is "serving cost per token" — put many adapters on one GPU, raise utilisation, cut spend. That framing treats the adapter as a performance artifact. In a multi-tenant customization product the adapter *is* the customer's intellectual property and the tenant boundary is a contractual control, so the adapter is closer to a credential than to a cache entry. The interesting engineering question is therefore not "how many adapters fit in HBM", but "which tuple of identifiers must be equal before a token may be generated, and what observable proves that the equality held". This case answers that question and then sizes the answer.

**Interview cue (30-second statement).** "A multi-tenant platform serves per-customer LoRA adapters from a shared GPU pool. I have to choose the cache key, the pinning scope and the rollback unit so that no request is ever answered by the wrong tenant's adapter, while keeping p95 under 900 ms and cutting dedicated-GPU spend. The hard part is that the cheap design and the safe design differ by one field in a cache key, and the difference is invisible in average latency."

---

## 2. Context & Background

### 2.1 The product and the workflow

The illustrative setting is **Meridian Support Cloud**, a hypothetical business-to-business platform that hosts a customer-support assistant for 1,800 tenant organisations. Each tenant licenses a base assistant and may additionally buy a *customization addendum*, which permits the platform to train a small set of additional weights on that tenant's own resolved support transcripts. The customization is deliberately narrow: the adapter teaches house terminology, product names, escalation phrasing and the tenant's tone. It is not a general capability upgrade, and it must never carry facts from another tenant.

The end-to-end workflow is:

```text
tenant administrator uploads corpus (or grants a scoped export)
        -> platform validates corpus licence and PII policy
        -> adapter training job (rank-16 LoRA on 4 module families)
        -> tenant-scoped evaluation suite (quality, safety, leakage canaries)
        -> promotion: adapter artifact + manifest enter the registry
        -> runtime: request arrives -> tenant identity resolved -> adapter selected
                    -> base + adapter composed -> tokens streamed -> audit record
        -> quarterly base-model refresh forces every adapter to be re-validated
```

Two properties of this workflow matter for the decision. First, **the adapter is produced from tenant-private data**, so an adapter swap is a data-disclosure event and not merely a quality regression. Second, **the base model moves underneath the adapters** on a quarterly cadence, which means adapter identity is never sufficient on its own; the pair (adapter, base) is the unit that can be validated, and therefore the unit that must be pinned.

### 2.2 Stakeholders

| Role | What they own in this decision | What they can veto |
|---|---|---|
| Model platform owner | Pool architecture, cache policy, lane sizing, release mechanics | Any design it cannot operate |
| Security architect | Tenant boundary definition, key composition, audit evidence | Any design without a structural isolation argument |
| Privacy / data protection lead | Lawful basis for customization, retention, cross-border handling | Tenant corpora leaving an approved region |
| Customer success lead | Per-tenant quality promise, incident communication | A quality regression above the contractual tolerance |
| Tenant administrator | Corpus, acceptance of the customized behaviour, notice of change | Continued use of customization |
| SRE / on-call | SLO, rollback authority, drill schedule | A rollback it cannot execute inside the window |
| Finance partner | GPU spend, unit economics per tenant | A cost trajectory that breaks the margin model |

The decision owner is the **model platform owner**, because the question is architectural. But three of the constraints below are set by people who do not report to that owner, which is exactly why the isolation argument must be written down rather than assumed: a platform owner can be persuaded by a cost table, whereas a security architect needs an invariant.

### 2.3 The baseline and its documented weaknesses

The current process, in this hypothetical, is *merge and dedicate*. For each tenant with an addendum, the platform merges the low-rank weights into a full 8-billion-parameter checkpoint, stores that checkpoint, and assigns it a dedicated GPU. Requests for that tenant are routed to that GPU only.

Three weaknesses are recorded in the scenario's own operating notes, and each is a reason the baseline is being questioned rather than a reason it is wrong:

1. **Cost scales with tenant count, not with usage.** 720 additive tenants consume 720 GPUs whether they send 40,000 requests a day or 40. Mean utilisation across the fleet is 6%, so 94% of the purchased capacity is idle by construction.
2. **Every base refresh is an N-way rebuild.** A new base model requires re-merging 720 checkpoints and re-running 720 evaluation suites before any tenant sees the improvement. The fleet therefore drifts: in the scenario, 118 tenants are two base versions behind because their rebuild never reached the front of the queue.
3. **The quality promise is hard to audit.** Because each tenant has a whole merged model, there is no cheap way to answer "which weights produced this token" — the answer is a checkpoint digest, and the checkpoint is not decomposed. Post-incident review therefore cannot separate base-model behaviour from adapter behaviour.

The weaknesses are real but the baseline has one enormous virtue that must not be discarded: **cross-tenant resolution is impossible by construction**, because tenant A's process has no access to tenant B's weights. Any consolidated design has to earn that property back explicitly. This is the recurring trap in consolidation projects: the baseline's safety is invisible because it is an accident of the deployment topology, and it is easy to remove the topology while believing the safety was a property of the code.

### 2.4 Scale, timing and boundaries

| Quantity | Illustrative value | Note |
|---|---:|---|
| Tenants on the platform | 1,800 | 720 hold a customization addendum |
| Mean request rate | 18 req/s | across all tenants, all lanes |
| Peak request rate | 120 req/s | 3× mean, business-hours peak |
| Adapter rank | 16 | applied to 4 module families |
| Adapter layers targeted | 32 | of the base model's transformer blocks |
| Base parameters | 8×10⁹ | bf16, 16 GiB resident |
| Adapter bytes (inference) | 32 MiB | bf16, derived in Section 6 |
| Adapter versions retained | 3 per tenant | current, previous, in-flight candidate |
| Base refresh cadence | quarterly | 90-day promotion window |
| GPU class HBM | 40 GiB | illustrative single-GPU budget |
| Session length (median) | 15 min | 900 s, used for pinning exposure |
| Regional boundary | two regions, tenant-pinned | no tenant crosses regions |

The timing detail that drives most of the design is the **quarterly base refresh**. On a refresh day the platform must move every adapter from base v41 to base v42. Because adapters were trained against v41's activations, a v42 base with a v41 adapter is not a validated composition; it is an untested one. The refresh therefore forces a coordinated cutover whose failure mode is not "slower" but "different model behaviour for every tenant at once".

### 2.5 Prior attempts

Two prior attempts are recorded as *considered and rejected on paper*, not as executed projects, and they are included because they explain why the chosen design looks the way it does:

- **Prompt-only customization.** House terminology was injected as a system prompt. It was cheap and required no artifact management, but it consumed 600–1,400 tokens per request, it could not change phrasing reliably, and — decisively — it gave no artifact to pin or revoke. Rejected as insufficient for the addendum's quality clause.
- **One shared adapter for all tenants.** A single adapter trained on the union of tenants was proposed to avoid the multi-adapter problem entirely. It fails the isolation requirement outright: a union-trained adapter can reproduce another tenant's product names, and there is no way to attribute which tenant's data produced a token. Rejected on privacy grounds before any quality discussion.

Neither attempt was benchmarked. They are alternatives with reasons, and Section 5c marks them as such.

### 2.6 Applicable requirements, stated honestly

The scenario assumes a contractual obligation to keep tenant corpora and tenant-specific model behaviour separate, and a data-protection regime under which a cross-tenant disclosure is a reportable event. **No specific statute, regulator or certification is claimed here.** The case deliberately avoids asserting that a particular law applies to a particular jurisdiction, because that determination is outside the case's evidence and changes over time. What the case does assert is weaker and sufficient: the platform has promised isolation in a contract, and a promise that cannot be evidenced is a liability regardless of which regime governs it.

The relevant method references are the [NIST AI Risk Management Framework](https://doi.org/10.6028/NIST.AI.100-1) for risk-vocabulary structure (voluntary, methodological, not a certification of this design) and the [OWASP Top 10 for Large Language Model Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/) for the *sensitive information disclosure* category. Neither source is evidence about Meridian; both are used in Section 9 to argue that "we tested it" is not the same as "it cannot happen".

---

## 3. The Business Question

The question is stated in the voice of the platform owner at the point of decision. It is a **reconstructed illustrative paraphrase**, not a quotation from a real person.

> "We are paying for 720 GPUs to serve traffic that would fit on four. I want to consolidate onto a shared pool with adapters cached in HBM. Before I sign, I need to know two things: can we prove that tenant A never gets tenant B's adapter, and what happens to my cost and latency if our big customers stop being big?"

### 3.1 Decision contract

| Contract field | Content |
|---|---|
| Question | Should customization move from merged dedicated checkpoints to a shared base with a bounded adapter cache, and if so, under which key, pinning scope and rollback unit? |
| Decision owner | Model platform owner; security architect holds a veto |
| Trigger | A capacity review showing 6% mean utilisation and a 720-checkpoint rebuild backlog before the next base refresh |
| Deadline | Before the v42 base-refresh window opens; after that date the migration and the refresh cannot share a change freeze |
| Available actions | (a) keep merged dedicated checkpoints; (b) shared pool with composite key, bounded cache, per-request pinning; (c) shared pool with a simple tenant-keyed cache and no pinning; (d) defer and run a measured pilot before deciding |
| Value at stake | Illustrative \$1.30M/month of dedicated capacity versus \$9.1k/month for the pool, plus the ability to complete a base refresh inside one window |
| Downside | A cross-tenant adapter resolution is a disclosure of tenant-private customization behaviour; a botched base refresh degrades all 1,800 tenants simultaneously |
| Objective | Minimise total serving cost per 1,000 requests subject to a hard isolation constraint, a p95 latency ceiling, and a rollback-time ceiling |
| Hard constraints | No request may be answered by an adapter whose tenant, version or base digest differs from the pinned bundle; p95 ≤ 900 ms; adapter rollback ≤ 15 s; base rollback ≤ 180 s |
| Scope | One platform, two regions, 1,800 tenants, 720 adapted tenants, 90-day refresh cycle |
| Out of scope | Training-method research, pricing changes, contract renegotiation, and any determination of legal applicability |

### 3.2 Objective versus proxy

The objective is *cost per 1,000 correctly served requests*. The tempting proxy is *GPU count*, and the two diverge in a specific way: GPU count falls if the cache is small and swaps are frequent, and it falls further if isolation checks are removed from the hot path. A design that minimises GPUs will therefore tend to minimise checks. The evaluation in Section 9 therefore pairs the cost metric with two guardrails that cannot be traded away — an isolation-evidence metric with a zero-tolerance decision rule, and a rollback-time metric — so that cost improvements purchased by weakening the boundary are visible as failures rather than as savings.

There is a second divergence worth naming. A *quality* proxy such as "adapter win rate on the tenant's evaluation set" can rise while the product gets worse, if the adapter memorises evaluation phrasing. Section 9 therefore uses a tenant-disjoint, temporally held-out evaluation set and a base-model control, and treats a suspiciously large win rate as a contamination signal rather than a triumph.

### 3.3 What an expensive wrong decision looks like

Three concrete wrong outcomes, in increasing order of cost:

1. **The pool is built and the isolation key is wrong.** One field — the tenant identifier — is missing from the cache key. The platform serves at 40% lower cost and, in the same week, answers several thousand requests with a sibling tenant's adapter. Cost of remediation includes notification, contract review, and the loss of the customization product's credibility.
2. **The pool is built correctly but sized on a demand assumption that fails.** The cache is sized for a Zipf-distributed tenant mix; a shift toward mid-sized tenants flattens the distribution; hit rate collapses from 84% to 28%; swap traffic triples; p95 breaches the SLO and the team buys back GPUs until the saving is gone.
3. **The pool is built and the rollback is not rehearsed.** A bad adapter version reaches 40% of traffic; the intended fix is a 5-second cache invalidation, but the release process also bumped the base digest, so the "rollback" leaves adapters pointing at a base they were not trained on and the guard refuses every request for 90 seconds. Availability incident on top of a quality incident.

### 3.4 Interview cue

Ask which quantity is being optimised, what "no intervention" costs, and what the *most expensive* false decision is. A candidate who answers "cost per token" without naming the isolation constraint has not understood the product; a candidate who names isolation but cannot size the cache has not understood the engineering.

---

## 4. Assumptions & Constraints

Assumptions are separated into those that must be tested before the design can be trusted, those imposed from outside the team, and choices the team controls. The three that can reverse the design are marked **critical**.

| ID | Assumption or constraint | Type | Verification plan | Status | Impact if wrong | Fallback | Recheck trigger / owner |
|---|---|---|---|---|---|---|---|
| A-01 | Tenant identity is resolved from the authenticated session before any adapter lookup, and never from request body or model output (**critical**) | identity | Negative test: submit a foreign tenant id in the payload and assert denial before routing | unverified | Every downstream check inherits a forgeable key; isolation argument collapses entirely | Withdraw customization; serve base only | Any change to the auth middleware / security architect |
| A-02 | Every inference path that can load an adapter passes through the one loader that enforces the composite key (**critical**) | architecture | Endpoint inventory plus a replay probe against each path | unverified | An uninstrumented path becomes a silent bypass | Disable the path until instrumented | New endpoint or SDK / platform owner |
| A-03 | Adapter artifacts are content-addressed and immutable once promoted | data | Digest of digest: re-hash registry objects, compare to manifest | unverified | A mutable "latest" tag makes pinning meaningless | Immutable bucket with object-lock | Registry migration / platform owner |
| A-04 | Tenant demand is heavy-tailed (Zipf-like), so a small cache covers most requests (**critical**) | business | Hit-rate histogram by tenant-size decile from the request log | unverified | Cache hit rate collapses; lane count rises; cost advantage narrows | Resize cache or re-introduce per-tenant replicas for the flat tail | Tenant-mix review each quarter / product analytics |
| A-05 | A 60 ms adapter swap (load 32 MiB from local NVMe into a reserved slot and rebind) is achievable at the assumed concurrency | infrastructure | Measured load test on the target GPU class | unverified | Latency model and lane count both change | Pre-warm more slots; move to a faster tier | Driver, GPU or storage change / SRE |
| A-06 | Sessions are short (median 15 min) so pinning exposure is small | business | Session-length distribution from logs | unverified | Long sessions multiply the number of users held on an old bundle during a release | Pin per turn instead of per session | Product change to long-running agents / product |
| A-07 | The quality benefit of an adapter is measurable on a tenant-scoped set with a base-model control | model | Paired evaluation with confidence interval | unverified | The addendum's value cannot be demonstrated; promotion decision becomes taste | Do not promote; refund addendum | Eval-set refresh / model owner |
| C-01 | Cross-tenant adapter resolution is prohibited, absolutely, with no risk-acceptance path | security | Contract clause and security policy | confirmed by scenario | Not a trade-off; it is a disqualifier | None — redesign | Policy change / security architect |
| C-02 | p95 end-to-end response ≤ 900 ms, measured at the edge, 5-minute windows | latency | Load test plus production SLI | confirmed by scenario | Consolidation may be infeasible at peak | More lanes or smaller cache | SLO review / SRE |
| C-03 | Adapter rollback ≤ 15 s; base rollback ≤ 180 s | operability | Timed drill | confirmed by scenario | Slow rollback extends a quality incident | Pre-staged previous bundle | Drill after each release-tool change / SRE |
| C-04 | 40 GiB HBM per serving GPU; 16 GiB base weights in bf16 | infrastructure | Vendor specification for the purchased class | assumed for sizing | The slot budget (and therefore the cache) is wrong | Re-derive K | Hardware refresh / platform owner |
| C-05 | Tenant corpora may not leave their region | privacy | Data-flow map per tenant | confirmed by scenario | Cross-region adapter would itself be a disclosure | Region-scoped pools | New region / privacy lead |

### 4.1 The two assumptions that carry the argument

**A-01 (identity provenance).** Almost every multi-tenant serving incident of this shape traces back to a key that was *derived from data the caller controls*. If the tenant identifier arrives in a JSON body, a prompt, a header the client sets, or a model-generated routing hint, then the isolation check is comparing an attacker-supplied value to an attacker-supplied value. The check will pass. The design must therefore draw the tenant identifier from the authenticated session object created by the identity middleware, and the adapter lookup function must not accept a tenant parameter at all — it must read the principal. Making the vulnerable call *unrepresentable* is stronger than making it *discouraged*.

**A-04 (demand shape).** The entire capacity argument rests on a small number of tenants generating most requests. This is an empirical claim about the customer base, and it is the kind of claim that silently stops being true after a pricing change, a self-serve tier, or a merger. Section 6 quantifies exactly how much the conclusion moves when the assumption fails, because a design that only works under one demand shape is a design with an undeclared dependency.

### 4.2 Constraints that are not negotiable

C-01 is listed as a constraint rather than an assumption because it is not testable into existence. There is no acceptable residual rate of cross-tenant resolution, no "small enough" number of affected requests, and no compensating control that makes one wrong answer acceptable. This matters for how the evaluation is written: a metric with a zero-tolerance rule cannot be demonstrated by sampling alone, so the design must carry a *structural* argument (the key cannot match) plus a *detection* argument (if the structure ever breaks, we find out fast). Section 9 builds both, and Section 9.6 explains why the detection argument alone would be insufficient.

### 4.3 Assumption register hygiene

Each row above has an owner and a recheck trigger, because assumptions in serving systems expire quietly. The two most likely to expire without anyone noticing are A-04 (demand shape, changed by a commercial decision) and A-05 (swap cost, changed by a storage-tier migration that nobody connects to latency). Both are therefore re-derived in Section 6's sensitivity table rather than tested once at design time.

---

## 5. Data, Tools & Method

### 5a. Data Specification

The unit of analysis differs between the offline and online halves of the case, and conflating them is a common source of wrong conclusions.

| Field | Offline (training and evaluation) | Online (serving and audit) |
|---|---|---|
| Unit of analysis | One (tenant, prompt, reference answer) evaluation item; one training sequence | One inference request |
| Grain and keys | `tenant_id`, `item_id`, `split`, `corpus_snapshot` | `request_id`, `session_id`, `tenant_id`, `bundle_id`, `adapter_digest`, `base_digest` |
| Volume | 720 adapted tenants × 300 evaluation items = 216,000 items; training corpus 40M tokens per tenant | 18 req/s mean, 46,656,000 requests per 30-day month |
| Window | Rolling 90 days, aligned to the base-refresh cycle | 30-day cost window; 5-minute SLO windows |
| Inputs | Prompt, tenant-scoped context, reference answer, house glossary | Prompt, session principal, routing hints, model choice |
| Outcome / label | Paired preference (adapter vs base) by a calibrated judge with human adjudication on a sample; canary-leak flag | Committed response, adapter digest actually used, latency, token counts, refusal flag |
| Population and exclusions | Excludes items whose reference answer was written after the corpus cut-off; excludes tenants with fewer than 50 usable items | Excludes synthetic health-check traffic from SLO and cost metrics, but not from isolation probes |
| Join semantics | `tenant_id` + `item_id` unique; corpus snapshot pinned per training run | `request_id` unique; `session_id` joins turns; `bundle_id` joins to the release manifest |
| Splits | Tenant-disjoint: no tenant's items appear in both a tuning and a reporting split. Temporal holdout: evaluation items dated after the corpus cut-off | Not applicable; online traffic is observed, not split |

**Why the evaluation grain is per item and not per tenant.** A per-tenant average hides the mechanism: a tenant can win on average because the adapter fixed ten escalation phrases while breaking two product names. The paired design keeps the item as the unit, so the confidence interval in Section 9 is about items, and the heterogeneity discussion can point at specific item classes.

**Why the serving grain is per request and not per session.** Pinning is a session-level property, but *evidence* is request-level. If the audit record were written once per session, a mid-session adapter switch would leave no trace and the failure in Section 12 (F-03) would be undetectable. The design therefore journals `adapter_digest` on every request even though the value is normally constant within a session.

### 5b. Tools & Stack

Everything in this table is **proposed** for the illustrative platform; no component is asserted to exist, and no version behaviour is claimed.

| Component | Role | Why it is in the design |
|---|---|---|
| Identity middleware | Creates the session principal; the only source of `tenant_id` | Removes caller-controlled tenancy (A-01) |
| Router / admission controller | Chooses a lane by adapter residency, applies queue bounds | Converts swap cost from stream-path latency into queue wait |
| Adapter loader (single-flight) | Materialises an adapter into a slot; one in-flight load per `(tenant, digest)` | Prevents duplicate loads and torn slots |
| Slot table | Authoritative map `slot -> (tenant, adapter_digest, base_digest, state)` | The object the isolation invariant is stated over |
| Inference engine | Composes base plus low-rank delta per request | Must support per-request adapter selection |
| Registry | Content-addressed adapter artifacts plus manifests | Makes pinning refer to immutable bytes (A-03) |
| Policy decision point | Evaluates bundle eligibility, region, policy version | Keeps policy out of the model path |
| Audit journal | Append-only `(request_id, principal, bundle_id, digests, decision)` | The only evidence that the invariant held |
| Canary prober | Sends tenant-unique probe strings on a schedule | Detects structural breakage within minutes |
| Evaluation harness | Paired adapter-vs-base scoring with calibrated judge | Promotion gate and drift detection |

The serving engine choice is deliberately left generic. The mechanism the case depends on is *per-request adapter selection with a slot table the platform controls*, which several engines expose; naming a specific engine and version would imply a verified capability claim that this exercise has not made. The method references in 5e cover the published techniques (low-rank adaptation, paged multi-adapter serving, batched adapter kernels) without asserting that any product behaves as this case assumes.

### 5c. Method Summary

The pipeline is:

```text
offline:  corpus -> validate -> train LoRA -> evaluate (paired, tenant-scoped)
          -> canary scan -> manifest {tenant, adapter_digest, base_digest,
          tokenizer_digest, policy_version, eval_evidence} -> registry

online:   request -> authenticate -> resolve principal (tenant) -> select bundle
          -> residency check -> [hit] compose -> generate -> journal
                             -> [miss] single-flight load -> verify digests
                                      -> compose -> generate -> journal
```

| Candidate | Evidence considered | Strength | Limitation / avoid when | Status | Revisit if |
|---|---|---|---|---|---|
| Merged dedicated checkpoints (baseline) | Operating notes: 720 GPUs, 6% utilisation, 118 tenants behind on base version | Isolation by construction; simple to reason about; no key correctness problem | Cost scales with tenant count; base refresh is N-way; attribution impossible | Incumbent; analytically rejected for the adapted fleet | If tenant count falls below ~30 or isolation must be physical |
| Shared pool, tenant-keyed cache, no pinning | Sketch only | Cheapest; simplest router | A release mid-session changes behaviour silently; a bad adapter reaches all traffic instantly with no drain | Rejected by reasoning | Never as described |
| Shared pool, composite key, bounded cache, per-session pinning | This case's derivations | Structural isolation argument; bounded blast radius; rollback is a cache operation | Depends on identity provenance and a single loader; depends on demand skew | **Proposed** | Demand mix flattens; base cadence shortens |
| Per-tenant dedicated *adapters* on a shared base, no cache | Not tested | Avoids eviction races | Memory infeasible: 56.4 GiB of adapters against a 16.4 GiB budget | Rejected by arithmetic | If adapter size drops ~4× |
| Prompt-only customization | Prior attempt, not benchmarked | No artifact management | No pinnable artifact; no reliable revocation; token cost | Rejected before quality work | If adapters prove to add no measurable value |

The chosen method is a **composition of four mechanisms**, and it is worth separating them because each can fail independently:

1. **Composite cache key** — `(tenant_id, adapter_digest, base_digest, tokenizer_digest, policy_version)`. Any mismatch is a miss, never a wrong hit.
2. **Per-session bundle pinning** — `bundle_id` is fixed at session start and cannot change mid-session, so a release never splits a conversation across two behaviours.
3. **Cache-aware routing with single-flight loading** — a request is dispatched to a lane that already holds the adapter, or it waits for exactly one load.
4. **Atomic release bundle** — base, adapters, tokenizer, policy and index move together or not at all, so a rollback cannot create an unvalidated composition.

### 5d. Data provenance, quality, exclusions and leakage controls

**Provenance.** All figures are constructed for teaching. The only inputs with an external existence are the method references in 5e; they establish that a technique exists, not that it performs as assumed here.

**Quality controls on the tenant corpus.** Each corpus is (i) licensed for customization, (ii) scanned for personal data under the platform's redaction policy, (iii) deduplicated within the tenant, and (iv) checked for **foreign-tenant canaries** — synthetic strings that belong to other tenants and must not appear in this tenant's training data. Control (iv) is unusual and is the direct lesson of this case: a training pipeline that ingests "all resolved tickets" can silently acquire another tenant's text through a shared inbox, a mislabelled export, or a support-tool integration that returns more rows than the caller expects. The canary scan is the check that the *data* boundary held, just as the runtime probe checks that the *serving* boundary held.

**Leakage controls, in order of danger.**

| Leakage path | Mechanism | Control |
|---|---|---|
| Evaluation items inside training data | The tenant's own eval prompts were used to build the corpus | Split by item timestamp; drop items dated at or before the corpus cut-off from the reporting split |
| Cross-tenant evaluation reuse | The same benchmark item is used for many tenants, so a "win" reflects benchmark familiarity | Tenant-disjoint reporting splits; report a base-model control on the same items |
| Judge contamination | The judging model was trained on similar public benchmarks | Calibrate judge against human labels on a stratified sample; report judge–human agreement |
| Adapter memorising canaries | Canary strings are in the training set, so a leak looks like normal recall | Canaries are *never* placed in training data; they are generated per tenant after training |
| Serving the wrong adapter | Cache key missing a field | Structural key plus per-request audit of the resolved digest |

**Exclusions.** Health-check traffic is excluded from cost and SLO numerators but deliberately retained in isolation-probe results, because probes are the detection mechanism and excluding them would remove the evidence. Tenants with fewer than 50 usable evaluation items are excluded from the quality gate and flagged for manual review rather than being scored on a thin sample.

**What is unavailable.** The scenario has no production logs, no measured adapter-swap latency, no real tenant demand distribution and no executed canary run. Every quantitative claim in Sections 6 and 9 is therefore *derived arithmetic under stated assumptions*, and Section 11 records outcomes as not measured.

### 5e. Reproducibility and method-selection ledger

| Artifact / source ID | Location | Version / date | Purpose | Accessibility | Reproduction limitation |
|---|---|---|---|---|---|
| SRC-01 | Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", arXiv:2106.09685 | 2021 (v2 2022) | Low-rank decomposition and the parameter-count form used in E-01 | public | Method only; the paper's benchmarks are not this scenario's evidence |
| SRC-02 | Sheng et al., "S-LoRA: Serving Thousands of Concurrent LoRA Adapters", arXiv:2311.03285 | 2023 | Paged adapter storage and batched multi-adapter kernels as prior art | public | Does not establish Meridian's memory numbers |
| SRC-03 | Chen et al., "Punica: Multi-Tenant LoRA Serving", arXiv:2310.18547 | 2023 | Multi-tenant adapter batching as prior art | public | Same limitation |
| SRC-04 | NIST AI RMF 1.0, DOI 10.6028/NIST.AI.100-1 | 2023-01 | Risk-vocabulary structure for Section 9 | public | Voluntary framework; certifies nothing |
| SRC-05 | OWASP Top 10 for LLM Applications | accessed 2026-11-10 | *Sensitive information disclosure* category used in Section 12 | public | Community guidance, not a control test |
| SRC-06 | Course standard: authoring and numerical standards | 2026-11-06 | Equation-block and evidence conventions | in-repo | Editorial, not technical authority |
| RUN-01 | Arithmetic in Sections 6, 8 and 11 | 2026-11-10 | Closed-form checks reproducible by hand | in this file | Not an executed system run |
| RUN-02 | Sensitivity grid in Section 6.9 | 2026-11-10 | Boundary behaviour of the lane/hit-rate model | in this file | Model output, not measurement |

**Reproducibility statement.** A reader can reproduce every number here with a calculator, because each derivation is written out with its inputs. A reader **cannot** reproduce the scenario's performance, because no system was run. The distinction matters for how the numbers may be cited: they are admissible as *worked examples of the method* and inadmissible as *evidence that a shared pool achieves this cost*.

**Seed and version policy.** The arithmetic has no random component, so no seed applies. The three training seeds mentioned in Section 8a exist only to make the per-tenant training cost account for run-to-run repetition; they are not a claim that seeds guarantee reproducibility across hardware or kernels.

---

## 6. Analysis Walkthrough

The decisive result is narrow and can be stated before the arithmetic: **the pool's cost advantage is large and robust, but its correctness advantage is zero unless the cache key is composite; and its capacity advantage depends on one assumption about the demand distribution that the team has not yet measured.** Everything below supports, bounds or qualifies that sentence.

### 6.1 Step 1 — Size one adapter, because the whole memory argument is a multiplication

The first step is not a system measurement but a parameter count, because the slot budget is fixed by HBM and the slot count is HBM divided by adapter size. Getting the adapter size wrong by a factor of two moves the cache from "fits comfortably" to "does not fit", which changes the recommendation.

**Validity before use.** The low-rank decomposition of SRC-01 replaces a weight update ΔW ∈ ℝ^{d_out×d_in} with a product B·A where B ∈ ℝ^{d_out×r}, A ∈ ℝ^{r×d_in}, and r ≪ min(d_in, d_out). The parameter count below assumes the update is applied to `m` module families per targeted layer, that each targeted module has square dimensions d, and that no bias or normalization parameters are trained. If any module is non-square, substitute d_in + d_out for 2d; if biases are trained, add them explicitly.

```text
FORMULA [E-01]: P = L * m * r * (d_in + d_out) ;  S = P * b / 2^20
WHERE: P adapter parameters [params]; L targeted layers [layers];
       m module families per layer [modules/layer]; r rank [dimensionless];
       d_in, d_out module dimensions [elements]; b bytes per parameter [bytes];
       S adapter size [MiB]; 2^20 bytes per MiB.
CONSTRAINT: single low-rank update per targeted module; no bias, no norms,
            no optimizer state (inference artifact only); bf16 => b=2.
NUMERICAL EXAMPLE: L=32, m=4, r=16, d_in=d_out=4096, b=2
   P = 32 * 4 * 16 * (4096 + 4096) = 32*4*16*8192 = 16,777,216 params
   S = 16,777,216 * 2 / 1,048,576 = 33,554,432 / 1,048,576 = 32.0 MiB
INTERPRETATION: one tenant's adapter is 32 MiB, i.e. 1/512 of the 16 GiB base.
               The ratio, not the absolute size, is what makes a cache viable.
```

The number 16,777,216 is exactly 2²⁴, which is a pleasant accident of the chosen dimensions (32·4·16·8192 = 2⁵·2²·2⁴·2¹³). It is worth noting explicitly, because a reader who assumes the result was reverse-engineered from the answer would be right to be suspicious; the check is that the arithmetic is dimensionally consistent regardless of whether the total is a power of two.

**Independent check.** 32 layers × 4 modules = 128 updates. Each update has r·(d_in+d_out) = 16 × 8192 = 131,072 parameters, which is 16 × 8192 = 2¹⁷. Then 128 × 131,072 = 131,072 × 128 = 16,777,216. ✓. Bytes: 16,777,216 × 2 = 33,554,432 B; dividing by 1,048,576 gives exactly 32. ✓

**Training-side footprint, for contrast.** Adam keeps two fp32 moments (8 bytes/param) plus a bf16 gradient (2 bytes/param), so a training step needs ≈ 10 bytes/param = 160 MiB of state plus the 32 MiB artifact, before activations. This is why adapter training is scheduled on a separate pool and never on serving lanes; the inference and training footprints differ by more than 6×.

### 6.2 Step 2 — Convert HBM into a slot budget

The serving GPU must hold more than adapters. The identity below is the constraint that makes K finite.

```text
FORMULA [E-02]: K_max = floor( (H - W_base - W_kv - W_ws - W_rt - R) * 1024 / (S + s_meta) )
WHERE: K_max adapter slots [slots]; H HBM per GPU [GiB];
       W_base base weights [GiB]; W_kv KV cache for the target batch [GiB];
       W_ws activation workspace [GiB]; W_rt runtime and kernels [GiB];
       R reserved headroom [GiB]; S adapter size [MiB]; s_meta slot metadata [MiB].
CONSTRAINT: one base model resident; KV sized for the configured batch and
            context; adapters occupy discrete slots with per-slot metadata.
NUMERICAL EXAMPLE: H=40, W_base=16.0, W_kv=4.0, W_ws=1.0, W_rt=2.1, R=0.5,
                   S=32.0, s_meta=0.0625
   budget = 40 - 16.0 - 4.0 - 1.0 - 2.1 - 0.5 = 16.4 GiB
   per slot = 32.0 + 0.0625 = 32.0625 MiB
   K_max = floor(16.4*1024 / 32.0625) = floor(16793.6 / 32.0625) = 523
INTERPRETATION: at most 523 of the 720 adapted tenants can be resident at once.
```

**Why the budget is small relative to HBM.** The base weights alone consume 40% of the device. That ratio is the structural reason adapter caching is interesting at all: the expensive object is shared, and the per-tenant object is small enough to page. If the base were 70B parameters, base weights alone would exceed one device and the whole calculation would become a model-parallelism problem rather than a cache-sizing problem.

**Boundary check.** 523 slots < 720 adapted tenants, so *some* misses are unavoidable even at maximum occupancy. The cache cannot be "big enough"; it can only be big enough *for the traffic distribution*. That is the hinge of the next step.

**Independent check.** 523 × 32.0625 = 16,768.7 MiB = 16.375 GiB ≤ 16.4 GiB ✓; 524 × 32.0625 = 16,800.75 MiB = 16.407 GiB > 16.4 ✗. So 523 is the true floor result.

### 6.3 Step 3 — Hit rate from the demand distribution

If requests were uniform over tenants, a 512-slot cache would hit 512/1800 = 28.4% of the time and the design would be uninteresting. The design's premise is that demand is heavy-tailed.

**Validity before use.** The Zipf form below assumes request counts over tenants follow a power law with exponent s, that the cache evicts the least-recently-used entry, and that the working set is stationary over the measurement window. Under LRU with a stationary Zipf workload, the hit rate is approximated by the share of total demand held by the K largest tenants — exact only if arrivals are ordered by popularity and the cache never needs to evict a popular entry. Real traffic is burstier and the approximation is optimistic; Section 6.9 treats the gap.

```text
FORMULA [E-03]: h(K) = H(K,s) / H(N,s),  H(n,s) = sum_{i=1..n} i^-s
WHERE: h hit rate [fraction of requests]; K resident slots [tenants];
       N tenant count [tenants]; s Zipf exponent [dimensionless];
       H generalised harmonic number [dimensionless].
CONSTRAINT: stationary demand, LRU eviction, no admission filtering,
            request-rate weights proportional to i^-s.
NUMERICAL EXAMPLE: N=1800, s=1, K=512
   H(1800,1) = 8.0730 ; H(512,1) = 6.8165
   h(512) = 6.8165 / 8.0730 = 0.8444
INTERPRETATION: 512 slots serve 84.4% of requests; the remaining 15.6% pay a swap.
```

**Independent checks.**

- h(64) = 4.7439/8.0730 = 0.5876; h(128) = 5.4331/8.0730 = 0.6730; h(256) = 6.1243/8.0730 = 0.7586. Monotone increasing in K ✓.
- h(1800) = 8.0730/8.0730 = 1.0000; h(1) = 1/8.0730 = 0.1239. Both boundaries behave ✓.
- The top-8 tenants carry 2.7179/8.0730 = 33.7% of requests; the top-32 carry 50.3%. Sanity: a heavy tail should put a third of demand on 0.4% of tenants, and it does.

**The eviction nuance.** With K = 512 and N = 720 adapted tenants, the resident set is 71% of the *adapted* population but only 28% of all tenants. Tenants without an addendum never need a slot, so the effective N for cache purposes is 720, not 1800 — but their request share is what determines h. This is a subtle and important distinction: **the cache is sized against demand mass, while the population that can contend for a slot is the adapted set.** A tenant with an adapter and no traffic is irrelevant to hit rate but relevant to eviction policy, because a rare tenant that does arrive will evict a common one if the policy is naive. Section 8c addresses this with a size-aware admission rule.

### 6.4 Step 4 — Turn hit rate into latency and lane capacity

```text
FORMULA [E-04]: E[d] = t_hit + (1 - h) * t_swap ;  C_lane = 1000 / E[d]
WHERE: E[d] mean added adapter latency per request [ms];
       t_hit adapter composition overhead on a hit [ms];
       t_swap slot load plus rebind on a miss [ms];
       h hit rate [fraction]; C_lane serial lane capacity [requests/s].
CONSTRAINT: requests are served serially per lane; the swap happens in the
            request's critical path; no queueing or batching effects included.
NUMERICAL EXAMPLE: t_hit=8, t_swap=60, h=0.8444
   E[d] = 8 + 0.1556*60 = 8 + 9.336 = 17.34 ms
   C_lane = 1000 / 17.34 = 57.7 requests/s
INTERPRETATION: one lane covers 57.7 req/s; the 120 req/s peak needs 2.08 lanes
               on mean latency, before any headroom or tail argument.
```

**Independent check.** At 18 req/s mean, 18/57.7 = 0.312 of one lane. At 120 req/s peak, 120/57.7 = 2.08. With a per-lane utilisation ceiling of 65% (a policy choice justified in 6.7), required lanes = 120/(0.65 × 57.7) = 3.20, so **4 lanes** are provisioned.

**The mean is not the SLO.** With 15.6% misses, the 95th percentile of added latency sits *inside* the miss branch: p95 = t_hit + t_swap = 68 ms, not 17.34 ms. The mean improves dramatically as K grows, while p95 stays at 68 ms until the miss rate falls below 5%. Solving 1 - h(K) < 0.05 requires H(K) > 0.95 × 8.0730 = 7.6694, which needs roughly K ≈ 900 slots — far beyond the 523-slot memory ceiling. **The cache can never make the swap path disappear from the tail.** This is the single most misread number in adapter-serving designs, and it changes the architecture: the fix is not a bigger cache but a routing rule that keeps the swap off the token path (Section 8b).

### 6.5 Step 5 — Convert p95 into an SLO statement

**Validity before use.** The additive decomposition below holds only if the adapter step and the generation step are independent contributions to end-to-end latency and neither is dominated by a shared resource (e.g. both waiting on the same NVMe device). Adding component percentiles is otherwise invalid, and the case avoids it: it uses *one* measured-or-assumed adapter percentile and *one* assumed base percentile, then adds them as a budget allocation rather than as a statistical identity.

```text
FORMULA [E-05]: L_p95 = L_base,p95 + d_p95 ;  headroom = SLO - L_p95
WHERE: L_p95 end-to-end p95 [ms]; L_base,p95 base model p95 without adapter [ms];
       d_p95 adapter contribution at the 95th percentile [ms];
       SLO the latency objective [ms]; headroom residual budget [ms].
CONSTRAINT: budget allocation, not a sum of independently estimated quantiles;
            both terms measured on the same population and window.
NUMERICAL EXAMPLE: L_base,p95=720, d_p95=68, SLO=900
   L_p95 = 788 ms ; headroom = 112 ms
INTERPRETATION: the inline-swap design passes the SLO with 112 ms to spare,
               but the spare budget is not available for cache misses beyond 5%.
```

The honest reading is uncomfortable: the design passes, but only because the base model's p95 happens to leave room. If the base p95 were 840 ms, the inline design would fail at 908 ms and the team would be forced into cache-aware routing. A design whose feasibility depends on 112 ms of headroom in someone else's component is fragile, which is exactly why the routing fix is adopted even though the naive arithmetic passes.

### 6.6 Step 6 — Quantify the failure that the cheap design creates

This is the step that changes the decision. Suppose the cache key is `(base_digest, adapter_digest)` — a natural choice, because those are the fields that determine the *computation*. Tenant identity is not part of the computation; it is part of the *authorisation*. A key built from computation fields is therefore missing exactly the field that matters.

The mechanism: two tenants can produce adapters with the same digest only if their bytes are identical, which is astronomically unlikely for independently trained artifacts. So a pure digest key would in fact separate tenants *by accident of content addressing*. The realistic defect is subtler and far more common: the key includes a **mutable logical name** rather than a digest, for example `(base_version, "support-adapter-current")`. Then every tenant's adapter shares a key, and the slot table's value is whichever tenant loaded most recently.

```text
FORMULA [E-06]: X = lambda * (1 - h) * q_wrong * 3600
WHERE: X cross-tenant serves per hour [requests/hour];
       lambda request rate [requests/s]; h hit rate [fraction];
       q_wrong probability that a miss resolves to a different tenant's adapter
       [fraction]; 3600 s/hour.
CONSTRAINT: steady state; q_wrong constant; a wrong resolve is assumed to be
            served, not rejected, i.e. no post-load digest verification.
NUMERICAL EXAMPLE: lambda=40, h=0.8444, q_wrong=23/24
   X = 40 * 0.1556 * 0.9583 * 3600 = 40*0.1556=6.224 ; 6.224*0.9583=5.965
   X = 5.965 * 3600 = 21,474 requests/hour
INTERPRETATION: a single missing key field produces ~21.5k cross-tenant
               answers per platform hour, i.e. 0.15% of all traffic.
```

**Independent check.** 5.965 wrong answers per second × 3,600 = 21,474 ✓. As a fraction of traffic: 5.965/40 = 14.9%, which is just the miss rate times q_wrong — consistent with the derivation.

**Why q_wrong = 23/24 rather than 1.** Even under a shared logical key, a miss sometimes resolves to the *correct* tenant by luck, because the most recent loader was this tenant. Modelling it as 23/24 for a 24-tenant cohort sharing a key is deliberately generous to the buggy design. The general form is q_wrong = 1 - 1/|cohort|; with no cohort (one global key) q_wrong → 1 and the rate becomes 22,406/hour at the same traffic. Either way the conclusion is the same: this is not a rare event that testing will catch by accident.

**The post-load verification that removes the mechanism.** If the loader re-reads the slot's tenant field and compares it to the principal after materialisation, a wrong resolve becomes a *rejected* request rather than a wrong answer. That check is cheap and it converts an unbounded disclosure into a latency incident. It is not, however, a substitute for the composite key, because it must be present on every path (A-02) and because it fires *after* a possible transient read of foreign weights. Structural prevention plus post-hoc detection is the pairing; either alone is insufficient, as Section 9.6 argues.

### 6.7 Step 7 — Choose K, and show that the choice is a knee rather than a maximum

The memory ceiling allows 523 slots. The chosen value is 512, and the justification is a peak-lane calculation rather than a round-number preference.

| K | h(K) | E[d] ms | C_lane req/s | Lanes at 120 req/s, ≤65% util | Adapter GiB | Fits 16.4 GiB? |
|---:|---:|---:|---:|---:|---:|---|
| 64 | 0.5876 | 32.74 | 30.5 | 6.05 → 7 | 2.00 | yes |
| 128 | 0.6730 | 27.62 | 36.2 | 5.10 → 6 | 4.01 | yes |
| 256 | 0.7586 | 22.48 | 44.5 | 4.15 → 5 | 8.02 | yes |
| **512** | **0.8444** | **17.34** | **57.7** | **3.20 → 4** | **16.03** | **yes (0.37 GiB slack)** |
| 523 | 0.8470 | 17.18 | 58.2 | 3.17 → 4 | 16.38 | yes (0.02 GiB slack) |
| 600 | 0.8640 | 16.16 | 61.9 | 2.98 → 3 | 18.79 | **no** |

Three readings of this table:

1. **The memory ceiling binds before the latency benefit flattens.** Going from 256 to 512 removes one lane; going from 512 to 600 (infeasible) would remove at most one more. The knee is at 512.
2. **The marginal lane is worth \$1,825/month** (one GPU-month at the illustrative rate), while the marginal slot costs 32 MiB of memory that is free at the margin. So the design should take slots up to the memory ceiling — which is why 512 is chosen over 256, and why 523 is rejected only on slack grounds, not on performance grounds.
3. **The choice is robust to the utilisation policy.** At a 75% ceiling rather than 65%, the lanes become 2.77 → 3 for K=512 and 3.60 → 4 for K=256; the preference for 512 is unchanged.

**Why 0.37 GiB of slack is not waste.** The KV cache grows with context length; a request at the maximum context consumes more KV than the sizing assumption. A cache with zero slack turns a legitimate long-context request into an out-of-memory failure that kills the lane. Section 8d makes this an explicit admission rule rather than an implicit hope.

### 6.8 Step 8 — Model the release boundary: pinning, exposure and rollback

Pinning is a *time* control, and its cost is measured in sessions that straddle a release.

**Validity before use.** The formula below assumes session starts form a Poisson process (rate λ_s), session durations are exponential with mean T, and releases occur as a Poisson process with mean interval τ, independent of sessions. Under those assumptions the probability a session is in flight at a release is the standard busy-period probability for an M/G/∞ system.

```text
FORMULA [E-07]: p_span = 1 - exp(-T/tau) ;  E[m] = lambda_s * T * p_span
WHERE: p_span probability a given session straddles a release [fraction];
       T mean session duration [s]; tau mean release interval [s];
       lambda_s session starts [sessions/s];
       E[m] expected sessions in flight at a release [sessions].
CONSTRAINT: Poisson starts, exponential durations, independent release process,
            one release at a time.
NUMERICAL EXAMPLE: T=900 s, tau=30 days=2,592,000 s, lambda_s=4 /s
   p_span = 1 - exp(-900/2,592,000) = 1 - exp(-3.472e-4) = 3.4716e-4
   concurrent sessions = 4*900 = 3,600
   E[m] = 3,600 * 3.4716e-4 = 1.25 sessions per release
INTERPRETATION: at 15-minute sessions, roughly one session per release is
               still running on the old bundle; that is the pinning population.
```

**Sensitivity that matters.** At T = 3,600 s (a product decision to support long troubleshooting sessions), concurrent sessions become 14,400 and p_span becomes 1 − exp(−3,600/2,592,000) = 1.388e-3, so E[m] = 20.0 sessions per release. A four-fold increase in session length produces a sixteen-fold increase in straddling sessions (4× from concurrency, 4× from duration-in-window), which is why pinning policy is a *product* dependency and not a pure infrastructure setting. If the product moves to long-running agents with hour-long sessions, per-session pinning must be replaced by per-turn pinning with an explicit "your assistant was updated" notice.

**Rollback arithmetic.** The two rollback units have different costs because they move different amounts of data.

```text
FORMULA [E-08]: R_adapter = n_reload * t_swap + t_ctrl ;
                R_base    = n_lanes * t_base_reload + t_ctrl
WHERE: R rollback wall time [s]; n_reload slots to refill [slots];
       t_swap per-slot load [s]; n_lanes serving lanes [lanes];
       t_base_reload per-lane 16 GiB reload [s]; t_ctrl control-plane
       propagation [s].
CONSTRAINT: serial reload per lane; control-plane propagation additive;
            no parallel prefetch across lanes.
NUMERICAL EXAMPLE: t_swap=0.060, t_ctrl=5, n_reload=8, n_lanes=4,
                   t_base_reload=12
   R_adapter = 8*0.060 + 5 = 5.48 s
   R_base    = 4*12 + 5 = 53 s
INTERPRETATION: an adapter rollback fits the 15 s ceiling with 9.5 s to spare;
               a base rollback fits the 180 s ceiling with 127 s to spare.
```

**The trap in R_base.** The base reload is *per lane*, so a design that scales lanes also scales rollback time: at 8 lanes, R_base = 8×12 + 5 = 101 s, still inside the ceiling but consuming 56% of it. If the fleet grows to 14 lanes, R_base = 173 s — inside the ceiling by 7 s and effectively unachievable in practice once queueing and readiness checks are added. A rollback-time SLO is therefore a *scaling* constraint, and it must be re-derived whenever lane count changes. This is a failure mode that appears only on the growth path, which is why it is easy to miss at design time.

### 6.9 Sensitivity and boundary analysis

Each row changes one specification and asks whether the recommendation survives.

| Stress / test | Original condition | Alternative | Effect on the model | Conclusion | Decision consequence |
|---|---|---|---|---|---|
| Demand shape | Zipf s=1 | Uniform over 1,800 tenants | h(512)=512/1800=0.2844; E[d]=8+0.7156*60=50.9 ms; C_lane=19.6 req/s; lanes at 120 req/s and 65% = 9.4 → 10 lanes | **Reversed** on capacity | Cache sizing must be re-derived from measured demand before migration |
| Demand shape, moderate | Zipf s=1 | Zipf s=0.7 (flatter) | Top-512 share falls; hit rate materially below 0.84 (not recomputed here without the measured exponent) | Uncertain | Treat K as a measured, not assumed, parameter |
| Swap cost | 60 ms | 250 ms (cold object store) | E[d]=8+0.1556*250=46.9 ms; C_lane=21.3; lanes at 120 req/s and 65% = 8.7 → 9 lanes | **Reversed** on capacity | Pre-warm slots locally; do not page from object storage |
| Traffic | 120 req/s peak | 240 req/s peak | Lanes at K=512: 240/(0.65*57.7)=6.4 → 7 lanes; adapter memory unchanged | Stable | Lane count scales; adapter cache does not |
| Session length | 900 s | 3,600 s | E[m] rises from 1.25 to 20.0 straddling sessions per release | Stable but costly | Per-turn pinning or maintenance windows required |
| Lane count | 4 | 14 | R_base rises from 53 s to 173 s against a 180 s ceiling | **Boundary** | Base rollback needs parallel prefetch or a faster reload path |
| Slot count | 512 | 256 | One extra lane at peak; saving falls by \$1,825/month | Stable | 512 preferred; 256 acceptable if memory is needed elsewhere |
| Adapter rank | 16 | 32 | S=64 MiB; K_max=261; h(256)=0.7586; C_lane=22.5; 8.2 → 9 lanes | **Reversed** on cost | Rank increase must be treated as a capacity change, not a quality tweak |
| Tenant count | 1,800 | 4,000 | H(4000)≈8.87; h(512)=6.8165/8.87=0.7685; more misses, more lanes | Stable but degrading | Re-tier cache or shard pools as the base grows |

**The two reversals that matter.** The uniform-demand case and the 250 ms swap case both push the design from 4 lanes to 9–10 lanes. That is not a rounding error; it is the difference between a 144× saving and roughly a 58× saving, and in the second case it also threatens the p95 budget because swap work grows faster than lane count. The lesson is not "the design is fragile" but "the design's inputs are three measured quantities — demand skew, swap latency, and adapter size — and all three are currently assumptions".

**The rank sensitivity deserves emphasis.** Doubling the adapter rank doubles adapter bytes and therefore halves the slot count. Teams routinely raise rank to chase quality without re-running the capacity model, and the result is an unexplained latency regression that looks like an engine bug. The capacity model must therefore live in the same repository as the training configuration, and a rank change should fail a test if the cache no longer fits.

### 6.10 What we tried first, and why it did not work

**Status: not tested.** No prototype was built. Two alternatives were rejected analytically:

- **Per-tenant adapters, no cache.** Rejected by E-02: 720 × 32 MiB = 22.5 GiB against a 16.4 GiB budget, and the all-tenant resident footprint for 1,800 tenants would be 56.4 GiB — 3.4× the budget. The arithmetic is decisive and requires no experiment.
- **Shared logical key without tenant identity.** Rejected by E-06: 21,474 cross-tenant answers per hour. This rejection is the case's central claim, and it is a *reasoning* result, not an empirical one. It is also the claim most worth attacking, which is why Section 9 designs a test that would falsify it.

The honest limitation: "rejected by reasoning" is weaker evidence than "rejected after a measured trial". A reader should treat these as strong priors, not as settled facts, and the pilot in Section 10 exists precisely to convert them.

### 6.11 Cross-check summary

| Check | Expression | Result | Verdict |
|---|---|---|---|
| Adapter size | 32·4·16·8192·2 / 2²⁰ | 32.0 MiB | ✓ |
| Slot ceiling | floor(16.4·1024 / 32.0625) | 523 | ✓ |
| Hit rate at 512 | H(512)/H(1800) | 0.8444 | ✓ |
| Mean added latency | 8 + 0.1556·60 | 17.34 ms | ✓ |
| Lane capacity | 1000/17.34 | 57.7 req/s | ✓ |
| Lanes at peak | 120/(0.65·57.7) | 3.20 → 4 | ✓ |
| Cross-tenant rate | 40·0.1556·(23/24)·3600 | 21,474 /h | ✓ |
| Straddling sessions | 3,600·3.4716e-4 | 1.25 | ✓ |
| Adapter rollback | 8·0.06 + 5 | 5.48 s | ✓ |
| Base rollback | 4·12 + 5 | 53 s | ✓ |

Every row is arithmetic, not measurement. The table exists so that a reader can find the first number they disagree with and re-derive from there.

---

## 7. The Finding

The platform should consolidate onto a shared base with a **composite** adapter key — tenant, adapter digest, base digest, tokenizer digest and policy version — a 512-slot cache per lane, per-session bundle pinning, and an atomic release bundle that rolls base and adapters together; the merged-dedicated baseline should be retired for the adapted fleet but retained as the isolation reference case. Under the illustrative assumptions the pool costs about \$9,125 per month against \$1,314,000 for 720 dedicated GPUs, a factor of 144, with four serving lanes at 17.34 ms mean adapter latency and a 788 ms p95 against a 900 ms ceiling. Those figures are **derived arithmetic on constructed inputs, and no outcome has been measured**; the cost conclusion survives every sensitivity in Section 6.9, but the *capacity* conclusion reverses if demand is not heavy-tailed or if adapter swaps cost 250 ms rather than 60 ms, and in the uniform-demand case the lane count rises from 4 to 10. The decisive boundary is that the isolation guarantee is structural rather than statistical: it holds because a token cannot be generated from a slot whose five key fields do not all match, and it is broken the moment tenant identity is derived from anything the caller controls, which no amount of cache tuning can repair. The next justified action is therefore not a migration but a three-lane measured pilot that establishes the demand exponent, the swap latency on the target storage tier, and the negative-test evidence that the composite key denies cross-tenant resolution — because two of those three quantities are currently assumptions, and the third is the only one that cannot be traded against cost.

---

## 8. System/Solution Design

The design is described as a **proposed** architecture. Nothing below has been implemented in the scenario, and the diagram is a design artifact rather than a description of a running system.

### 8a. Offline / training / batch path

```text
tenant corpus (region-scoped)
   -> licence and redaction validation -> PII scan -> dedup
   -> foreign-tenant canary scan (must find zero)
   -> LoRA training: rank 16, 4 module families, 32 layers, bf16, 3 seeds
   -> produce adapter artifact (32 MiB) + manifest
   -> tenant-scoped evaluation: paired adapter-vs-base on 300 items (temporal holdout)
   -> safety suite: refusal, over-refusal, canary non-reproduction
   -> quality gate: paired win rate CI lower bound > threshold AND no critical regression
   -> registry commit: content-addressed object + manifest
   -> promotion decision recorded with evidence digest and approver
```

**Training footprint and scheduling.** Each training run touches 40M tokens against an 8-billion-parameter base. Using the standard forward-plus-backward accounting of roughly 6 FLOPs per parameter per token gives 6 × 4×10⁷ × 8×10⁹ = 1.92×10¹⁸ FLOPs per run. At an illustrative sustained 2×10¹⁴ FLOP/s for the training pool, that is 9,600 s = 2.67 GPU-hours per seed, or 8.0 GPU-hours for three seeds. Across 720 tenants that is **14,400 GPU-hours ≈ 1.64 GPU-years** per full refresh cycle, which at the illustrative \$2.50/GPU-hour is about \$36,000 per cycle — a small fraction of one month of the retired dedicated fleet, and the reason training is not the constraint in this decision.

The manifest is the object that makes pinning possible, and its fields are the release contract:

| Manifest field | Why it must be in the manifest | Consequence if omitted |
|---|---|---|
| `adapter_digest` | Identifies immutable bytes | A rebuild silently replaces what a release pointed at |
| `base_digest` | Identifies the weights the adapter was trained against | A stale pairing is served as if validated |
| `tokenizer_digest` | A tokenizer change re-indexes the same text | Silent semantic shift with no test failure |
| `policy_version` | Routing and refusal behaviour are policy-dependent | A policy change is invisible to the pin |
| `eval_evidence` | Names the suite version and result digest | Promotion cannot be audited or re-derived |
| `tenant_id` | Binds the artifact to its owner | The artifact is not attributable to a principal |

### 8b. Online / serving / real-time path

```text
                      [request + timestamp]
                               |
                    [authenticate: session principal]
                               |
                     [resolve bundle_id from session]   <-- pinned, immutable
                               |
                 [policy check: region, eligibility, quota]
                               |
                   +-----------+-----------+
                   |                       |
             [slot hit?]              [slot miss]
                   |                       |
        [verify 5-field key]      [single-flight load]
                   |                       |
                   |              [verify 5-field key]
                   |                       |
                   +-----------+-----------+
                               |
                  [compose base + low-rank delta]
                               |
                     [generate tokens / stream]
                               |
        [journal: request_id, principal, bundle_id, adapter_digest,
                  base_digest, slot, decision, latency]
                               |
                    [response + safe refusal path]
```

Four properties of this diagram carry the argument, and each has a failure path drawn explicitly rather than implied:

1. **`bundle_id` is resolved once, at session start, from the principal** — not once per request, and never from the payload. A release therefore cannot split a conversation.
2. **The key verification happens after materialisation as well as at lookup.** The lookup prevents the wrong slot being chosen; the post-load check catches a race in which a slot was rebound between lookup and composition. The second check is cheap and it is what makes the guarantee robust to concurrent eviction.
3. **A miss goes through a single-flight loader.** Two concurrent requests for the same absent adapter cause one load and one wait, not two loads. Without single-flight, a popular tenant that is evicted during a burst generates N loads for one artifact, and the resulting NVMe contention converts a 60 ms swap into a multi-hundred-millisecond one.
4. **Refusal is a valid terminal state.** If the post-load key check fails, if the slot is in `draining`, or if the pinned bundle's adapter object is unavailable, the request is answered by the base model with an explicit degraded-mode flag — not by a different adapter. Degrading to the *base* model is safe because the base contains no tenant data; degrading to *any adapter* is not.

### 8c. Key design decisions

| Decision ID | Choice | Alternatives | Evidence | Trade-off | Failure boundary | Revisit trigger |
|---|---|---|---|---|---|---|
| DEC-01 | Composite 5-field cache key | tenant-only; digest-only; `(base, name)` | E-06: 21,474 wrong answers/h under a name key | Key width costs a hash of five fields ≈ microseconds | Breaks if tenant id is caller-supplied (A-01) | Any key field added or removed |
| DEC-02 | Pin per session, not per turn, not per request | per turn; no pin (always latest) | E-07: 1.25 straddling sessions per release | Straddling sessions see old behaviour for ≤15 min | Breaks if sessions become hour-long | Product change to long sessions |
| DEC-03 | 512 slots per lane, 4 lanes | 256/5 lanes; 523/4 lanes; no cache | Section 6.7 knee table | 16.03 GiB of 16.4 GiB budget | Breaks if rank rises to 32 (K→261) | Rank, base size or KV config change |
| DEC-04 | Inline swap permitted, but cache-aware routing preferred | inline-only; route-only | E-04, E-05: p95 788 ms passes with 112 ms headroom | Routing costs a stale-routing table refresh | Breaks if routing table lags slot truth | Any change to router or slot table |
| DEC-05 | Atomic release bundle (base+adapters+tokenizer+policy) | independent per-component release | F-07: independent release created unvalidated pairs | One large coordinated change instead of four small ones | Breaks if a component can be patched outside the bundle | Emergency patch process invoked |
| DEC-06 | Degrade to base model on any key failure | degrade to nearest adapter; retry other lane | Base contains no tenant data | Quality drop is user-visible | Breaks if base is not tenant-neutral | Base customization added |
| DEC-07 | Content-addressed registry with object lock | mutable `latest` tag per tenant | E-06 root cause | Registry write path is more awkward | Breaks if lock is disabled during migration | Registry migration |
| DEC-08 | Single-flight loading per `(tenant, digest)` | independent loads | Prevents load storms during eviction bursts | One slow load blocks waiters | Breaks if the loader has no timeout | Loader rewrite |

**DEC-04 deserves the most scrutiny** because it is the decision most often made by accident. If swaps stay inline and the base model's p95 creeps from 720 ms to 840 ms — through a longer context default, a larger batch, or a slower kernel — then the end-to-end p95 becomes 908 ms and the SLO fails with no change to the adapter system at all. Cache-aware routing removes that coupling: a miss becomes *queue wait on a lane that holds the adapter*, not work inside the token path, so the tail is governed by queueing rather than by storage. The design therefore adopts routing as the primary mechanism and keeps inline swap as a bounded fallback for tenants below a request-rate threshold.

### 8d. Architecture artifacts and operational readiness

| Area | Answer for this design |
|---|---|
| Interfaces | `POST /v1/generate` with session credential; response carries `bundle_id` and `degraded` flag; errors are typed: `E_KEY_MISMATCH`, `E_ADAPTER_UNAVAILABLE`, `E_QUOTA`, `E_REGION`. Idempotency key per request so retries cannot double-count quota. |
| Data / state | Slot table is the single source of truth for residency; it is rebuilt from the loader on lane start, not persisted across restarts. Registry is the source of truth for artifacts; audit journal is append-only and is the source of truth for what was served. |
| Reliability | SLI: fraction of requests served with `degraded=false` and latency ≤ 900 ms, measured at the edge, 5-minute windows, 28-day objective of 99.9%. Error budget of 0.1% ≈ 40 minutes per 28 days. |
| Resilience | Per-request timeout 2.5 s; retry budget 5% of requests, at most one retry, only on `E_ADAPTER_UNAVAILABLE`; circuit breaker per lane opens after 20 consecutive load failures; queue admission bounded at 40 waiting requests per lane. |
| Recovery | A lane restart refills 512 slots at 60 ms each = 30.7 s of warm-up; during that window the lane serves base-only. Nothing is lost on restart because slots are a cache; the audit journal is the durable record and is written before the response is flushed. |
| Observability | Per-request digest pair; histogram of swap latency; hit-rate by tenant decile; counter of `E_KEY_MISMATCH` (must be zero in normal operation and is a page-worthy alert if nonzero); canary probe success; slot-table divergence counter. |
| Deployment | Base + adapters + tokenizer + policy ship as one bundle with one version. Canary at 2% of sessions for 6 hours, then 20%, then 100%, with per-tenant eligibility. Rollback is a bundle pointer change plus cache invalidation, bounded by E-08. |
| Security | Tenant id from principal only; adapter objects access-controlled per tenant prefix; loader runs with read-only registry credentials; slot table is not writable from the request path; audit journal is write-once for the serving role. |
| Economics | 5 GPUs (4 lanes + 1 control/eval) at \$2.50/GPU-hour = \$9,125/month; \$0.196 per 1,000 requests at 46.66M requests/month; 144× below the dedicated baseline. |
| Human workflow | On-call can roll back an adapter (≤15 s) without a release manager; a base rollback (≤180 s) requires the release manager because it affects all tenants; a tenant-level opt-out is a platform action with a 1-hour target. |

**Sizing that is easy to forget.** Adapter *storage*, not adapter *memory*, is the second-order constraint. With 720 adapted tenants and three retained versions each, the registry holds 720 × 3 × 32 MiB = 69,120 MiB = 67.5 GiB per region-wide tenant set, or 33.75 GiB per region if the 720 split evenly. Local NVMe per lane must hold at least the 512 resident adapters (16.03 GiB) plus a read-ahead margin; at 6.23 swaps/s and 32 MiB per swap, the lane pulls **199.4 MiB/s** from local storage. That figure is comfortably inside a modern NVMe device but is *not* comfortably inside a network filesystem, which is the concrete reason the 250 ms swap sensitivity in Section 6.9 is a real deployment risk rather than a hypothetical one.

**Cold start and peak.** A four-lane pool at 120 req/s peak has 3.20 lanes of mean-latency capacity, so utilisation at the 65% ceiling is 120/(4 × 57.7) = 52%. That leaves room for a lane to be out of service. With three lanes healthy, capacity is 3 × 57.7 × 0.65 = 112.5 req/s, which is *below* peak — so the design's availability target and its latency target are in tension at exactly the moment a lane is lost. The mitigation is not more lanes by default but an admission policy: at three healthy lanes, the router sheds long-context requests first and returns them with a retry-after, preserving the latency SLO for short requests rather than failing all of them.

---

## 9. Evaluation & Validation

### 9.1 Primary metric and guardrails

| Field | Definition |
|---|---|
| Primary metric | **Cross-tenant adapter resolution rate**: requests whose served `adapter_digest` belongs to a tenant other than the authenticated principal's, divided by all requests served, over a 28-day window, in the population of adapted-tenant traffic, measured from the audit journal by joining `principal.tenant` against `adapter.tenant`. Unit: fraction of requests. Comparator: the dedicated baseline, where the quantity is structurally zero. |
| Secondary metric 1 | **Cost per 1,000 successfully served requests**: monthly infrastructure cost divided by (requests − degraded requests − failed requests) × 1000. Unit: USD per 1,000. Comparator: dedicated baseline at the same traffic. |
| Secondary metric 2 | **Adapter rollback time**: wall time from rollback command to the first request served on the restored bundle at full traffic. Unit: seconds. Comparator: the 15 s ceiling. |
| Secondary metric 3 | **Paired quality win rate**: fraction of evaluation items where the adapter response is preferred to the base response by a calibrated judge, with human adjudication on a stratified sample. Numerator: preferred items; denominator: judged items; population: tenant-scoped, temporally held-out items; window: current 90-day cycle. |
| Guardrail A | **Base-capability retention**: performance on a general capability suite must not fall by more than 1 percentage point absolute versus the base model. This catches adapters that fix house terminology while degrading instruction following. |
| Guardrail B | **Refusal calibration**: over-refusal rate on benign prompts must not rise by more than 2 percentage points; under-refusal on the safety suite must not rise at all. Reported separately, because a single "safety score" hides the trade. |
| Guardrail C | **p95 latency** ≤ 900 ms at the edge, 5-minute windows. |
| Guardrail D | **Degraded-response share** ≤ 0.5% of requests over 28 days. |

### 9.2 Pass/fail rule and its timing

The rule is stated with the moment at which it was chosen, because a threshold invented after seeing results is not a gate.

| Gate | Rule | Chosen when | Owner |
|---|---|---|---|
| G-1 isolation | Zero cross-tenant resolutions in the audit journal over the observation window **and** zero `E_KEY_MISMATCH` in the negative-test suite **and** a passed structural review of the key derivation | Before any pilot, at design approval | Security architect |
| G-2 capacity | p95 ≤ 900 ms at 120 req/s sustained for 30 minutes on the pilot lanes | Before pilot, derived from C-02 | SRE |
| G-3 rollback | Adapter rollback ≤ 15 s and base rollback ≤ 180 s, both executed in a drill | Before pilot | SRE |
| G-4 quality | Paired win-rate CI lower bound > 50% for at least 80% of adapted tenants, with no tenant below 45% | Before pilot | Model owner |
| G-5 cost | Modelled cost per 1,000 requests ≤ 10% of the dedicated baseline at measured traffic | Before pilot | Finance partner |

G-1's **three-part** structure is the heart of the evaluation design and deserves explanation. A zero count in the journal is evidence about *observed* traffic; it cannot establish that an unobserved path is safe. A passed negative suite is evidence that *known* attack shapes are denied. A structural review is evidence that the key's tenant component cannot be supplied by a caller. The three together are much stronger than any of them alone, and a candidate who offers only the first has offered a sampling argument for a property that must be structural. The arithmetic of why the sampling argument is weak on its own is in 9.6.

### 9.3 Test design

| Test | Design | What it can establish | What it cannot |
|---|---|---|---|
| T-01 cross-tenant negative probe | Two authenticated sessions, tenants A and B; replay identical payloads; assert `adapter_digest` differs and matches each principal; then submit a payload naming tenant C and assert denial before routing | That the routing path honours the principal | That uninstrumented paths honour it |
| T-02 canary reproduction | Per-tenant synthetic strings generated post-training, never in any corpus; probe each adapted tenant weekly and assert its own canary is not reproduced and no foreign canary appears | That a trained adapter does not memorise foreign strings for the probed prompts | Behaviour on unprobed prompts; memorisation that requires a specific trigger |
| T-03 rollback drill | Inject a deliberately degraded adapter into 2% of sessions; execute rollback; measure time to full traffic on restored bundle | That rollback is executable and bounded | That the rollback target is itself correct |
| T-04 swap storm | Sustained 120 req/s with a cache limited to 8 slots over 12 tenants, forcing a miss on nearly every request | That thrash degrades gracefully and single-flight holds | Real demand-shape behaviour at 512 slots |
| T-05 latency | 30-minute soak at 120 req/s with production-like context lengths | p95 under sustained peak | Behaviour under a correlated failure (lane loss + peak) |
| T-06 quality | Paired adapter-vs-base on 300 temporally held-out items per tenant, judge calibrated on 400 human-labelled pairs | Direction and rough size of quality effect | Small effects; cross-tenant generalisation |
| T-07 base refresh rehearsal | Move a 10-tenant subset from base v41 to v42 as one bundle; verify guard refusals, then full cutover and rollback | That the atomic bundle works end to end | 720-tenant timing at full scale |

T-04 is deliberately hostile. Restricting the cache to 8 slots makes the miss path dominant, which is the only cheap way to test the loader under contention. It is a *stress* test and its latency numbers must not be quoted as production latency; its purpose is to surface load storms, slot-table divergence and single-flight failures, not to estimate cost.

### 9.4 Sample, uncertainty and power

**For the isolation metric**, the honest statement is that a zero count does not bound the rate well. With zero observed events in `n` requests, the one-sided 95% upper bound is approximately 3/n. At 46,656,000 monthly requests, 3/n = 6.4×10⁻⁸, which sounds reassuring — but only for the paths that were exercised. The bound is a statement about *observability*, not about *possibility*: it says "if the defect occurred more often than once in 15 million requests we would probably have seen it", and it says nothing about a path that never carried traffic. This is the precise reason G-1 requires a structural review; the statistics cannot reach the property.

**For canary probes**, detection power is binomial: with per-probe leak probability q and C probes, power = 1 − (1 − q)^C. At q = 0.02, C = 50 gives 1 − 0.98⁵⁰ = 63.6%; C = 149 gives 95.0%. At q = 0.05, C = 50 gives 92.3%. So a weekly 50-probe schedule detects a 5%-per-prompt leak with 92% probability and a 2%-per-prompt leak with only 64% probability — an uncomfortable asymmetry that argues for either more probes or a lower tolerated q. Running probes per tenant rather than per platform (n = 1,800) gives a zero-event bound of 3/1,800 = 1.7×10⁻³ per tenant-week, which is a meaningful *operational* bound in a way that the platform-wide request count is not.

**For quality**, the paired design has 300 items per tenant. Treating each item as independent is optimistic because items within a tenant share phrasing and domain; a defensible approach is a cluster bootstrap over items within tenant and a separate between-tenant summary. The illustrative reporting format is a win rate with a 95% interval per tenant, plus the distribution of tenant-level win rates. With 300 items and a true win rate near 0.6, the naive standard error is √(0.6·0.4/300) = 0.028, so a 95% interval is roughly ±5.5 points; clustering will widen it. A tenant reporting 55% is therefore **indistinguishable from 50%** and must not be described as improved.

### 9.5 Validity threats

| Threat | Mechanism | Check |
|---|---|---|
| Benchmark familiarity | The same public benchmark items appear across tenants, so the adapter "wins" by having seen the style | Tenant-disjoint reporting splits; base control on identical items |
| Judge drift | The judge model is updated mid-cycle, shifting preferences | Pin judge version per evaluation manifest; re-calibrate against human labels after any change |
| Temporal leakage | Evaluation items predate the corpus cut-off and were effectively training data | Drop items dated at or before the cut-off from the reporting split |
| Workload mismatch | Lab traffic has short prompts; production has long ones | Soak test with production context-length distribution; report p95 by context bucket |
| Survivorship in the pilot | Tenants who opt into the pilot are the most engaged, with the most data | Report pilot and non-pilot separately; do not extrapolate the pilot win rate to all tenants |
| Metric gaming | Cost per 1,000 requests improves if degraded responses are excluded from the denominator | Degraded share is a guardrail with its own ceiling; cost denominator counts *successfully served* requests only, which is the stricter choice |

### 9.6 Why detection cannot substitute for structure

Suppose the team relies entirely on T-01 plus journal monitoring. The defect in E-06 produces 21,474 cross-tenant answers per hour. Detection would fire within minutes, which sounds adequate. Three things make it inadequate as the *only* control:

1. **Detection is after the disclosure.** The 21,474 answers are already delivered. A detection mechanism limits duration, not occurrence, and the contract prohibits occurrence.
2. **Detection depends on the journal being correct.** If the journal records `bundle_id` but not the resolved `adapter_digest`, the join that proves cross-tenant service is impossible. The evidence mechanism is therefore itself part of the design, not an add-on.
3. **The failure is silent in aggregate metrics.** Latency improves on a hit-heavy workload, cost falls, and quality metrics on tenant-scoped sets look unchanged — because each tenant's *evaluation* runs in isolation. Nothing in the normal metric set moves when the boundary breaks; only a boundary-specific check does.

The structural argument, by contrast, is a statement about the code: the loader's lookup is keyed on a tuple that includes the principal's tenant, and the composition step does not accept a slot whose tuple disagrees with the pinned bundle. That is checkable by inspection and by a property test, and it holds for traffic that has never occurred.

### 9.7 Execution status

| Evaluation item | Status | Reason |
|---|---|---|
| G-1 structural review | **not executed** | No implementation exists |
| G-1 negative suite | **not executed** | Fixtures designed in Section 14.2, not built |
| G-1 journal observation | **not executed** | No traffic |
| G-2 latency | **not executed** | Derived only (E-04, E-05) |
| G-3 rollback drill | **not executed** | Derived only (E-08) |
| G-4 quality | **not executed** | No evaluation run |
| G-5 cost | **not executed**; model only | Arithmetic in Section 8d |
| Sensitivity grid | **executed as arithmetic** | Section 6.9, reproducible by hand |

**Decision: inconclusive, not passed.** No gate has been executed, so the design is neither validated nor refuted. The recommendation in Section 7 is a recommendation to *test*, with the arithmetic establishing only that the test is worth running.

---

## 10. Action Taken

**No action taken as of 2026-11-10.** This is a design exercise; no migration was approved, no pilot was run, and no tenant was moved.

What exists in the scenario is a *decision record*, and it is worth writing down precisely because the distinction between "decided to pilot" and "migrated" is where case studies usually start lying.

| Action field | Record |
|---|---|
| Recommendation | Approve a three-lane pilot with the composite key, 512-slot cache, per-session pinning and atomic bundle; do not migrate the fleet until G-1 through G-5 execute |
| Approval / status | **Proposed.** Decision owner: model platform owner; security veto not yet exercised; **not approved** |
| Implementer and handoff | Model platform team builds the loader and slot table; security builds the negative suite; SRE builds the rollback drill; model team builds the paired evaluation |
| Effective scope | Pilot only: 24 tenants in one region, deliberately including 8 large, 8 mid, 8 small tenants so the demand exponent can be *estimated* rather than assumed |
| Release strategy | Canary at 2% of pilot sessions for 6 hours, then 20% for 24 hours, then 100% of pilot |
| Stop / rollback | Any `E_KEY_MISMATCH` in normal traffic stops the pilot immediately (no tolerance band). Any p95 > 900 ms for two consecutive 5-minute windows pauses expansion. Adapter rollback ≤ 15 s verified in drill before canary |
| Excluded / deferred | Fleet migration; the second region; any change to the training method; retention-policy changes for the audit journal |
| Blocker | A-04 (demand exponent) and A-05 (swap latency) are unmeasured, and both change the lane count. Migrating before measuring them would size the fleet on a guess |

### 10.1 Why the pilot is three lanes and 24 tenants

Three lanes, not four, because the pilot's purpose is to find the capacity boundary, and a four-lane pilot cannot reveal whether the fourth lane was necessary. Twenty-four tenants with three size strata, because the demand exponent is the assumption with the largest effect on the recommendation and it cannot be estimated from a pilot that contains only large tenants. Eight tenants per stratum is not a statistically generous sample, but the exponent only needs to be distinguished between "sharply skewed" (s ≈ 1) and "flat" (s ≈ 0.5), and that distinction is visible in a 24-tenant spread.

### 10.2 What the pilot must produce, in order

1. A measured swap latency distribution on the real storage tier. If p95 swap > 150 ms, the design changes shape before any capacity decision.
2. A measured demand histogram, from which s is fitted and h(512) is recomputed rather than assumed.
3. A negative-suite result set, including the "foreign tenant id in payload" case, with the denial recorded *before* routing.
4. A timed adapter rollback and a timed base rollback.
5. A paired quality result per tenant with intervals, plus the guardrail suite.

Only after (1)–(5) is the cost model re-run with measured inputs, and only then is a fleet decision legitimate. This ordering matters because running the cost model first creates an anchor — a 144× saving is a hard number to argue against — and the whole point of the pilot is to discover whether the number is real.

### 10.3 Organisational prerequisites

The handoffs that are easy to under-estimate:

- **Security must own the key specification**, not review it. If the platform team writes the key and security reviews it, the key will be written for computational convenience.
- **SRE must own the rollback drill schedule**, with a standing quarterly rehearsal that survives staff turnover. A rollback procedure that lives in one engineer's head fails the first time it is needed.
- **The tenant administrator must be told that behavior can change at a bundle boundary.** Pinning hides a release within a session but does not hide it across sessions; a customer who is promised "your assistant will not change" is being promised something the design does not deliver.

---

## 11. Outcome & Results

**Outcome not yet measured.** No pilot ran, so the table below reports *modelled* quantities with their exact status. The distinction is not pedantry: a modelled 144× saving and a measured 144× saving support completely different sentences in a board paper.

**Source declaration (mixed, stated per row):** the cost and latency rows are **DERIVED** arithmetic from Section 6 under stated assumptions; the isolation row is a **PROPOSED** property with no observation; the quality row is **NOT MEASURED**; the assumption rows are **OPEN**.

| Metric | Baseline: value / window / N | Proposed design: value / window / N | Absolute change | Relative change | Uncertainty | Evidence | Limitation |
|---|---|---|---|---|---|---|---|
| Monthly serving GPUs | 720 GPUs / month | 4 lanes + 1 control = 5 GPUs | −715 | −99.3% | Derived from the stated adapter and base sizes | DERIVED | Assumes 6% baseline utilisation and Zipf s=1; both unmeasured |
| Monthly GPU cost | \$1,314,000 | \$9,125 | −\$1,304,875 | −99.3% (144× lower) | Price assumption \$2.50/GPU-hour, 730 h/month | DERIVED | Illustrative price; no committed rate |
| Cost per 1,000 requests | \$28.16 | \$0.196 | −\$27.97 | −99.3% | Denominator 46.66M requests/month at 18 req/s mean | DERIVED | Mean-rate denominator hides peak-priced capacity |
| Mean added adapter latency | 0 ms (dedicated, always resident) | 17.34 ms | +17.34 ms | n/a (baseline has no added term) | h from unmeasured demand shape | DERIVED | Inline-swap model; routing design would differ |
| End-to-end p95 | not established for baseline | 788 ms | — | — | Base p95 assumed 720 ms | DERIVED | Two-term budget, not a statistical sum |
| Cross-tenant resolutions | 0 by construction | 0 by construction *if* key is composite | 0 | — | **No observation** | PROPOSED | Structural claim; post-load verification assumed present |
| Paired quality win rate | 50% by definition (base vs base) | not measured | — | — | — | NOT MEASURED | Requires the 300-item-per-tenant evaluation |
| Adapter rollback time | n/a (full checkpoint swap) | 5.48 s modelled | — | — | Linear in slots reloaded | DERIVED | Drill not executed |
| Base rollback time | not applicable | 53 s modelled at 4 lanes | — | — | Linear in lane count; 173 s at 14 lanes | DERIVED | Ceiling breach on the growth path |
| Straddling sessions per release | n/a (no pinning) | 1.25 sessions | — | — | Poisson/exponential assumptions | DERIVED | Session-length distribution unmeasured |

### 11.1 How to read the 144×

The 144× figure is `720/5`. It is arithmetic on the GPU count, and it embeds three assumptions that a reader should extract before quoting it:

1. **That the dedicated fleet really is 720 GPUs for 720 tenants.** If some tenants share a merged checkpoint, the baseline is smaller and the multiple shrinks.
2. **That the illustrative price is the right price.** A dedicated fleet bought on a committed-use discount has a lower effective unit price than an on-demand shared pool, so the ratio is sensitive to procurement terms in a way the GPU count is not.
3. **That the shared pool's utilisation does not erode.** The pool's cost per request rises if traffic falls, because the lanes are provisioned for peak. At 6 req/s mean instead of 18, cost per 1,000 requests triples to \$0.588 while the dedicated fleet's cost per 1,000 rises by the same factor — so the *ratio* is stable but the absolute numbers are not. This is a favourable property of the comparison and worth stating, because it means the decision is robust to a demand forecast error in the level (though not in the shape).

### 11.2 What would count as a measured result later

For the outcome section to be fillable, the pilot needs to produce these specific artifacts, and an honest later revision would cite them by ID:

- `RUN-P1`: swap latency histogram on the pilot storage tier, with p50/p95/p99 and the storage configuration.
- `RUN-P2`: fitted demand exponent with a confidence interval, from the 24-tenant histogram.
- `RUN-P3`: negative-suite transcript showing denial before routing for the foreign-tenant-id case.
- `RUN-P4`: timed rollback logs for adapter and base, with the lane count at the time.
- `RUN-P5`: per-tenant paired quality with cluster-bootstrap intervals.
- `RUN-P6`: 30-minute soak at 120 req/s with p95 by context-length bucket.

None of these exist. Their absence is the reason Section 9's decision is *inconclusive* and not *passed*.

### 11.3 A note on adverse movements

Even in a successful consolidation, some quantities get worse, and a case that reports only savings is not reporting. Three predictable regressions:

- **Cold-start behaviour.** A dedicated GPU is always warm for its tenant. A shared pool's first request after eviction pays the swap. Users of rare tenants experience this chronically, so the *distribution* of experience becomes less equal even as the mean improves. This is a fairness consequence, not only a performance one, and it should be visible to customer success before it appears in a complaint.
- **Blast radius of a platform defect.** A bug in the loader now affects all 1,800 tenants rather than one. The dedicated baseline's worst day was one tenant's model being wrong; the pool's worst day is every tenant's. This is the price of consolidation and it must be paid for with canary breadth, fast rollback, and rehearsed drills — not with optimism.
- **Attribution complexity.** With merged checkpoints, "which model produced this?" had a one-word answer. With base+adapter composition, the answer requires the journal, and the journal becomes a production dependency for audit. A journal outage is now an audit outage.

---

## 12. Failure Modes & What Went Wrong

Eight failure modes are recorded. Each is labelled with how confident the scenario is that it could occur: **hypothetical** means it is a mechanism derived from the design, not an observed incident; **simulated** means the arithmetic in Section 6 demonstrates the magnitude. None is an observed production failure, and the scenario contains no incident.

| Failure ID | Trigger and observable symptom | Detection / discriminating test | Confirmed cause or hypothesis | Impact / blast radius | Containment | Prevention / owner | Verification |
|---|---|---|---|---|---|---|---|
| F-01 | A burst of answers using another tenant's phrasing; support tickets from two customers noticing unfamiliar product names | Join audit journal `principal.tenant` vs `adapter.tenant`; if the join cannot be done, the journal schema is itself the first defect | **Hypothesis (simulated):** cache keyed on a mutable logical name, so all tenants share a key and the last loader wins. E-06 gives ~21.5k wrong answers/hour at 40 req/s | All adapted tenants in the affected cohort; a disclosure event per request | Disable adapter serving platform-wide; degrade to base model; freeze releases | Composite key with tenant from principal; post-load key verification; immutable digests. Owner: security architect | Re-run T-01 with a hostile payload naming a third tenant; assert denial before routing; assert journal join succeeds |
| F-02 | Latency p95 rises from 788 ms to 1.4 s without a cache-size change | Compare swap latency p95 against the 60 ms assumption; check storage tier and queue depth on the loader | **Hypothesis:** adapters paged from a network filesystem; 32 MiB at network latency costs hundreds of ms, so E-04's t_swap is wrong by 4× | All lanes; SLO breach; possible lane overload as queueing amplifies | Pin adapters to local NVMe; disable remote tier as a load source | Local NVMe with read-ahead; capacity model must cite the storage tier. Owner: SRE | T-05 soak with the loader forced to the remote tier; assert graceful degradation rather than SLO collapse |
| F-03 | A user reports that the assistant "changed mid-conversation" after a release | Segment the session transcript by `bundle_id`; a single session with two values proves the defect | **Hypothesis:** bundle resolved per request instead of per session, so a release splits a conversation | Sessions in flight at a release: ~1.25 per release at 15-min sessions (E-07) | Force session restart with an explicit notice; roll back the bundle | Resolve `bundle_id` once at session start; make the request path unable to choose. Owner: platform | Deliberate mid-session release in a test harness; assert constant `bundle_id` across all turns |
| F-04 | Slot-table divergence: two lanes report different residents for the same tenant; intermittent `E_KEY_MISMATCH` | Compare slot tables across lanes; if divergence is nonzero, the single-flight loader is not the only writer | **Hypothesis:** a second code path (batch endpoint, evaluation harness) loads adapters into the serving pool without the reserved-slot protocol | One or more lanes; possible torn slot during a concurrent load | Drain the lane; rebuild the slot table from the loader | Single loader with exclusive write; A-02 endpoint inventory. Owner: platform | Replay probe against every endpoint; assert exactly one writer per slot |
| F-05 | Cache thrash: hit rate falls to 0.28, lanes saturate, p95 breaches | Hit-rate histogram by tenant decile; if the histogram is flat, the demand assumption failed rather than the cache | **Hypothesis:** demand flattened after a self-serve tier launch; the Zipf premise no longer holds | All traffic; capacity shortfall of 6 lanes (Section 6.9) | Enable LRU-with-admission on large tenants only; shed long-context requests | Re-derive K from measured demand each quarter; treat s as a monitored parameter. Owner: product analytics | Recompute h from logs monthly; alarm if h(512) < 0.75 for 7 days |
| F-06 | A rank increase from 16 to 32 ships with a quality justification, and latency regresses a week later | Check adapter artifact size and K_max: rank 32 halves slots to 261, hit rate falls to 0.759 | **Hypothesis:** training config and capacity model live in different repos, so the coupling is invisible | All lanes; one to two extra lanes needed | Revert rank; or accept the extra lanes with an explicit cost decision | Put the capacity model in the training repo with a failing test when K_max < slots needed. Owner: model platform | CI check that computes S and K_max from the training config on every change |
| F-07 | After a "rollback", every request is refused by a guard for 90 s | Diff the bundle against the previous bundle: if base digest and adapter digest come from different releases, the rollback was partial | **Hypothesis:** base and adapters released independently; rolling back the adapter left it on a base it was not trained for, so the compatibility guard rejected the pair | All tenants on the affected base | Serve base-only until a consistent bundle exists; then restore | Atomic release bundle (DEC-05); rollback is a bundle pointer swap. Owner: release manager | T-07 rehearsal; assert that rollback restores a previously *validated* pair, not merely a previous version |
| F-08 | An adapter reproduces a foreign tenant's product name in a probe | Canary probe hits a foreign string; then check the training corpus for that string | **Hypothesis:** the corpus ingestion pulled a shared support inbox and acquired foreign text; the canary scan either was not run or was run after training | The affected tenant's adapter, and the tenant whose text leaked | Quarantine the adapter; retrain from a re-scoped corpus; notify | Foreign-canary scan before training, not after; corpus scoping by tenant prefix. Owner: data platform | Scan must find zero foreign canaries on the training set; add a deliberately contaminated fixture and assert the scan fails |

### 12.1 The two failures that deserve separate treatment

**F-01 is the case's reason to exist**, and its most instructive detail is that it is *not* a model failure. Every component behaves correctly: the model generates fluent text, the cache serves what it was asked for, the latency is excellent, the cost is low. The defect lives in the relationship between an authorisation concept (tenant) and a caching concept (key), and it is invisible to every metric that does not join those two concepts. A security review that asks "is the model safe?" will not find it; a review that asks "which fields determine what this cache may return?" will.

**F-07 is the failure that consolidation creates and the baseline cannot have.** With merged checkpoints, a rollback is "point the tenant at the old checkpoint" — there is no compatibility question because the checkpoint is self-contained. Splitting the model into base and adapter introduces a *composition validity* problem that did not previously exist, and a rollback that restores components independently can produce an invalid composition. The mitigation is not "be careful"; it is to make the release unit equal to the validation unit, so that anything validated can be restored and anything restored was validated. That is what "atomic release bundle" means, and it is why DEC-05 is a design decision rather than a process preference.

### 12.2 Competing causes for the most ambiguous symptom

The symptom "quality got worse for some customers after migration" has at least five competing causes, and they require different fixes. Discriminating them is the practical skill this section is trying to build.

| Hypothesis | Discriminating check | If true | If false |
|---|---|---|---|
| Wrong adapter being served (F-01) | Journal join by tenant; any nonzero count is decisive | Stop serving adapters immediately | Boundary is intact; look elsewhere |
| Right adapter, wrong base digest | Compare served `base_digest` against the manifest's | Restore the validated bundle | Composition is correct |
| Adapter quality genuinely lower (training issue) | Paired evaluation on held-out items, per tenant | Retrain; the adapter is bad | Quality is fine; look at delivery |
| Cache miss causing degraded (base-only) responses | Degraded-share metric by tenant | Investigate eviction and admission policy | Delivery is complete |
| Judge or evaluation artefact changed | Check judge version against the pinned manifest | Re-baseline the evaluation | Evaluation is comparable |

The order matters: check the boundary first, because it is the only hypothesis whose truth requires an immediate stop. A team that starts with "the adapter must be bad" will retrain for a week while a disclosure continues.

---

## 13. Transferable Lesson

| Pattern field | Content |
|---|---|
| Canonical pattern | `authorization-scope`, with `atomic-release-bundle` as the release mechanism |
| General principle | When a per-tenant artifact is cached in a shared process, the cache key must contain every field that determines *authorisation*, not only the fields that determine *computation*; and the unit of validation must equal the unit of rollback. |
| Preconditions | Tenants have private artifacts; artifacts are immutable and content-addressed; tenant identity comes from an authenticated principal; the serving engine supports per-request artifact selection; a journal can record the resolved artifact per request |
| Mechanism | A composite key makes cross-tenant resolution a key miss rather than a wrong hit, which converts an unbounded disclosure into a bounded latency event. An atomic bundle makes rollback restore a validated composition rather than a set of independently-versioned parts. Neither mechanism improves quality or speed; both remove a class of failure that no quality or speed metric would reveal. |
| Supporting evidence | E-06's arithmetic (21,474 wrong answers/hour from one missing key field) and F-07's composition argument are **derived**, not observed. The cost comparison is derived arithmetic on illustrative inputs. No part of this case is an empirical demonstration. |
| Avoid when | Tenants share a single model with no private artifact — then there is no tenant-scoped key to get wrong, and the machinery is pure overhead. Also avoid when the artifact is *not* tenant-specific but version-specific (an A/B experiment arm): pinning per session is still useful, but the isolation argument does not apply and the key can be simpler. |
| Portability test | Ask: "if I write the cache key on a whiteboard and hand it to a new engineer, can they name a request that would be served the wrong artifact?" If yes, the key is incomplete. Then ask: "if I roll back one component, is the resulting combination one that was ever validated?" If no, the release unit is wrong. |
| Adaptation | In a feature store, the same shape appears as a key of `(entity, feature_version, as_of, policy)`; in a CDN, as `(tenant, object_version, auth_scope)`; in a per-customer model router, as `(account, model_digest, prompt_version, region)`. In each case the failure is identical — a key built from the computation's inputs only — and the fix is identical. |

### 13.1 The counterexample that bounds the lesson

The lesson is *not* "always use composite keys". Consider a stateless translation service with no per-customer artifact: the cache key is the prompt and model version, there is no tenant-scoped object to confuse, and adding a tenant field would fragment the cache for no benefit — every tenant's request for the same text is legitimately the same computation. The lesson applies exactly when **the artifact carries tenant-specific information**, and it stops applying the moment it does not. Over-applying it produces fragmented caches, lower hit rates and worse unit economics, which is why the precondition column is not decoration.

### 13.2 What a reader should be able to do after this case

1. Write a cache key for a multi-tenant artifact system and defend every field, distinguishing authorisation fields from computation fields.
2. Compute adapter size from rank, module count and layer count, then convert HBM into a slot budget, and say which term in the budget is most likely to be wrong.
3. Convert a hit rate into a latency distribution, and explain why the p95 sits on the miss branch until the miss rate is under 5%.
4. Show that a cache-sizing decision can reverse under a different demand distribution, and name the measurement that would settle it.
5. Design a negative test that fails *before* routing, and explain why a zero-count journal metric is not a substitute for a structural argument.
6. Distinguish the failure that consolidation creates (composition validity on rollback) from the failure it inherits (capacity mis-sizing), and give each a different control.

---

## 14. Connections & Cross-References

### 14a. Related cases, theory and cheat sheets

| Relationship | Target | Mechanism shared or changed | Boundary learned |
|---|---|---|---|
| `uses-method` | [Threat Model and Execution Trust](../Cheat_Sheets/01_AI_Threat_Model_and_Execution_Trust.md) | Trust-boundary enumeration, principal provenance, capability scoping | That sheet covers execution surfaces; this case applies the same reasoning to a *cache* boundary, which is a boundary most threat models omit |
| `uses-method` | [Prompt Injection and Red-Team Evaluation](../Cheat_Sheets/02_Prompt_Injection_and_Red_Team_Evaluation.md) | Negative testing and canary-based detection | Injection testing targets inputs; here the input is benign and the defect is in the key, so input-side testing cannot find it |
| `extends` | [Approval-Bound Export for a Retrieval Assistant](01_Agent_Data_Exfiltration_Simulation.md) | Both bind an effect to an authenticated scope with a capability that expires | The export case bounds a *volume* of disclosure; this case bounds a *population* — the effect is not how much leaves but whose behavior answers |
| `contrasts-with` | [Approval-Bound Export](01_Agent_Data_Exfiltration_Simulation.md) | Export needs a human approval payload binding; adapter serving needs no human at all | Human-in-the-loop is not a universal control. A per-request latency budget of 900 ms and 120 req/s makes approval structurally impossible, so the control must be deterministic |
| `contrasts-with` | [Enterprise agent approval gate](../../../06_llm_agents_knowledge_systems_and_ai_augmented_development/30_agentic_systems_workflows_and_orchestration/Case_Studies/01_enterprise_agent_approval_gate.md) | That reference is a hypothetical, unreviewed gate design | Cited as a planning reference only; it establishes nothing about this case |
| `theory` | Hu et al., LoRA (arXiv:2106.09685) | Low-rank parameter count used in E-01 | The paper's efficiency claims are about training; this case uses only the parameter-count identity |
| `theory` | Sheng et al., S-LoRA (arXiv:2311.03285); Chen et al., Punica (arXiv:2310.18547) | Paged/batched multi-adapter serving as prior art | Neither establishes the memory, hit-rate or latency figures here |
| `planned` | Secure RAG design review (CASE-49-002, planned) | Same authorisation-scope pattern applied to retrieval | Not authored; no comparison is claimed |
| `planned` | Release evaluation gate (CASE-49-004, planned) | Evaluation manifest and promotion evidence | Not authored; the G-4 gate here would reuse its manifest discipline |

### 14b. What this case leaves open

| Open question | Why current evidence is insufficient | Data or test that would resolve it | Priority / trigger | Owner role |
|---|---|---|---|---|
| What is the actual demand exponent across adapted tenants? | It is assumed (s = 1); the lane count moves from 4 to 10 between s = 1 and uniform | 30-day request histogram by tenant, fitted with a truncated power law plus a goodness-of-fit check | **Highest** — before any fleet decision | Product analytics |
| What is p95 adapter swap latency on the production storage tier? | Assumed 60 ms; at 250 ms the design needs 9 lanes | `RUN-P1` histogram on the real tier, per lane, under contention | **Highest** — before capacity sign-off | SRE |
| Does a tenant adapter measurably improve quality without a base-capability regression? | Not measured; the addendum's value is asserted by the product | `RUN-P5` paired evaluation with clustered intervals and the guardrail suite | High — gates the product's premise, not just the architecture | Model owner |
| Can a foreign tenant's text enter a corpus through a shared ingestion path? | Not tested; the canary scan is designed but not run | Contaminated fixture that must fail the scan; audit of ingestion sources per tenant | High — data-side counterpart of F-01 | Data platform |
| Does per-session pinning remain adequate if sessions lengthen to an hour? | E-07 says straddling sessions rise from 1.25 to 20 per release | Session-length distribution plus a product decision on agentic sessions | Medium — triggered by any long-session product | Product |
| Does base rollback stay inside 180 s beyond 14 lanes? | E-08 is linear in lanes; at 14 lanes it reaches 173 s | Timed drill at the target lane count with parallel prefetch enabled | Medium — triggered by fleet growth | SRE |

### 14c. Intense interview pack

Fifteen questions. Each carries a candidate prompt, the facts explicitly given, what the candidate must clarify, a strong answer with reasoning, acceptable alternatives, a follow-up that changes a constraint, red flags, a scoring guide and an answer-support pointer. All scenarios are the case's illustrative ones; no question depends on a real employer's process.

```text
Scoring scale used throughout
0  = incorrect, or asserts a control without locating the boundary it protects
1  = partial: names one correct element, omits a competing cause, limit or check
2  = correct with explicit assumptions and a concrete verification
3  = in addition, defends a trade-off and shows what would reverse the choice
```

#### Q1 — Locate the authorisation boundary (framing)

**Level:** foundational. **Skill:** framing and domain.

**Prompt.** "We cache per-customer adapters in one process to cut GPU cost. Where does the security boundary actually live?"

**Given.** 1,800 tenants, 720 with adapters; adapter 32 MiB; base 16 GiB; one process serves many tenants.

**Candidate should clarify.** Whether tenant identity arrives from an authenticated session or from the request; whether the serving engine accepts a per-request adapter selector; whether any path bypasses the loader.

**Strong answer.** The boundary is not the model and not the prompt; it is the **slot table entry plus the composition step**. A token may only be generated from a slot whose `(tenant, adapter_digest, base_digest, tokenizer_digest, policy_version)` equals the pinned bundle resolved from the authenticated principal. The model has no knowledge of tenancy, so asking it to respect tenancy is not a control. The verifiable evidence is a per-request journal row recording the resolved digest, joinable to the principal's tenant.

**Reasoning.** Authorisation must be enforced where the effect is committed. In a serving stack the effect is "these weights produced these tokens", so the enforcement point is the point at which weights are bound to the request — after lookup and again after materialisation, because a slot can be rebound between the two.

**Acceptable alternatives.** Physical isolation (separate processes per tenant group) also satisfies the requirement, at higher cost; a signed routing token issued at authentication and verified by the loader is equivalent if the loader refuses unsigned selectors.

**Follow-up.** "Suppose the tenant id is in the request body because the gateway already validated it." Answer: the gateway's validation is irrelevant if the body is attacker-controlled; the loader must read the principal, and the function signature should not accept a tenant parameter at all.

**Red flags.** "The system prompt tells the model which tenant it is serving." "The API key identifies the tenant, so the body is fine." "We tested with two tenants and saw no crossover" offered as the argument rather than the evidence.

**Scoring.** 0: names the model or prompt. 1: names the cache without saying which fields matter. 2: names the slot/composition boundary and the journal. 3: also states why the second post-load verification is needed and what happens if identity is caller-supplied.

**Support:** Sections 2.1, 4 (A-01, A-02), 8b.

#### Q2 — Why the cheaper key is catastrophic (plausible wrong approach)

**Level:** advanced. **Skill:** debugging and judgement.

**Prompt.** "An engineer proposes keying the adapter cache on `(base_version, "support-adapter-current")` because digests change on every rebuild and that makes cache invalidation annoying. Convince me it is fine or not fine."

**Given.** 40 req/s, hit rate 0.8444 at 512 slots, 24 tenants share a logical key in the worst cohort.

**Candidate should clarify.** What "current" resolves to at lookup time, whether a post-load verification exists, and whether the journal records the resolved digest.

**Strong answer.** Not fine. The key omits the only field that determines *whose* artifact may answer. A miss under a shared logical key resolves to whichever tenant loaded most recently, so with q_wrong = 1 − 1/24 = 0.9583 the platform serves 40 × 0.1556 × 0.9583 × 3600 = **21,474 cross-tenant answers per hour**, about 15% of traffic. The "invalidation annoyance" is a real operational problem, but the correct fix is content-addressed immutable artifacts plus a pointer that moves, not a mutable key. If a pointer is unavoidable, the key must include the pointer's resolved digest, so the key changes when the pointer does.

**Reasoning.** Cache keys must be a superset of the fields that determine whether a cached value is *permitted*, not merely whether it is *equal*. Digest stability is a convenience property; tenant binding is a correctness property, and correctness properties do not get traded for convenience.

**Acceptance.** The candidate may accept a mutable pointer **only** if the resolved digest is part of the key, and should say so explicitly.

**Follow-up.** "What if we keep the logical key but add a post-load tenant check?" Answer: that converts disclosure into rejection — better, but it depends on the check existing on every path (A-02) and it fires after a transient read of foreign weights, so it is a mitigation, not the mechanism.

**Red flags.** "Digests are ugly, names are fine." "We will monitor for crossover." "The tenants are in different regions anyway" (regions do not partition a shared key).

**Scoring.** 0: accepts the design. 1: objects without quantifying. 2: quantifies the rate and proposes content addressing. 3: also separates the invalidation problem from the isolation problem and solves each.

**Support:** Sections 6.6, 8c (DEC-01, DEC-07).

#### Q3 — Recompute the adapter size and the slot ceiling (calculation)

**Level:** applied. **Skill:** calculation and sizing.

**Prompt.** "Rank 16, 4 module families, 32 layers, d = 4096, bf16. How big is one adapter, and how many fit on a 40 GiB GPU holding a 16 GiB base?"

**Given.** 4 GiB KV, 1 GiB workspace, 2.1 GiB runtime, 0.5 GiB reserve, 64 KiB per-slot metadata.

**Candidate should clarify.** Whether biases or norms are trained (assume not), whether d_in = d_out (assume yes), and whether the base is resident in bf16 (assume yes).

**Strong answer.** P = L·m·r·(d_in+d_out) = 32·4·16·8192 = 16,777,216 parameters. At 2 bytes each, S = 33,554,432 B / 1,048,576 = **32.0 MiB**. Budget = 40 − 16.0 − 4.0 − 1.0 − 2.1 − 0.5 = 16.4 GiB = 16,793.6 MiB. Per slot 32.0625 MiB, so K_max = ⌊16,793.6/32.0625⌋ = **523**.

**Reasoning.** The parameter count is a product of "how many low-rank updates" and "how many parameters per update"; the per-update count is r·(d_in+d_out), not r·d, which is the most common error. Bytes follow from the dtype; MiB conversion uses 1,048,576, not 10⁶.

**Checks.** 523 × 32.0625 = 16,768.7 MiB ✓; 524 × 32.0625 = 16,800.75 MiB ✗. 16,777,216 = 2²⁴ exactly, a useful sanity signal.

**Follow-up.** "Rank 32?" S = 64 MiB, per slot 64.0625, K_max = ⌊16,793.6/64.0625⌋ = 262. The capacity model must be re-run, not merely noted.

**Red flags.** Using r·d (gives 16 MiB, half the truth); forgetting the base weights; converting with 10⁶; ignoring KV cache.

**Scoring.** 0: wrong formula. 1: right formula, arithmetic slip. 2: correct with units. 3: also bounds the result and names the most fragile budget term (KV, which grows with context).

**Support:** Sections 6.1, 6.2 (E-01, E-02).

#### Q4 — Hit rate and the p95 trap (calculation + judgement)

**Level:** advanced. **Skill:** calculation, then interpretation.

**Prompt.** "With 512 slots and Zipf s=1 over 1,800 tenants, what is the hit rate, the mean added latency, and the p95 added latency at 8 ms hit / 60 ms swap?"

**Given.** N = 1,800, K = 512.

**Candidate should clarify.** Whether demand is over all tenants or only adapted tenants (the *mass* is over all traffic; the *population contending for slots* is the 720 adapted).

**Strong answer.** H(1800) = 8.0730, H(512) = 6.8165, so h = 6.8165/8.0730 = **0.8444**. Mean = 8 + 0.1556·60 = **17.34 ms**. The p95 is **68 ms**, not 17.34 — because 15.6% of requests take the miss path, the 95th percentile lies inside it. To get p95 down to the 8 ms hit path you would need a miss rate below 5%, i.e. H(K) > 0.95·8.0730 = 7.669, which needs roughly 900 slots — 1.7× the memory ceiling. **The cache can never remove the swap from the tail.**

**Reasoning.** A mixture distribution's quantile is not its mean. With miss probability m, the 1−α quantile equals the miss latency whenever m > α. Since m = 0.156 > 0.05, p95 = 68 ms.

**Follow-up.** "How do you fix the tail?" Move the swap off the token path: cache-aware routing so a miss becomes queue wait on a lane that already holds the adapter, or pre-warm on session start. A bigger cache alone cannot do it.

**Red flags.** Reporting 17.34 ms as p95. Claiming "p95 ≈ mean for large n". Concluding the cache is too small and buying HBM instead of changing the architecture.

**Scoring.** 0: confuses mean and quantile. 1: correct hit rate only. 2: correct mean and identifies the p95 problem. 3: computes the slot count needed for a 5% miss rate and shows it is infeasible.

**Support:** Sections 6.3, 6.4 (E-03, E-04).

#### Q5 — Lane count at peak (calculation + policy)

**Level:** applied. **Skill:** capacity sizing.

**Prompt.** "Mean 18 req/s, peak 120 req/s, lane capacity 57.7 req/s, and we will not run lanes above 65% utilisation. How many lanes, and what happens when one dies?"

**Given.** Four-lane candidate.

**Strong answer.** 120/(0.65·57.7) = 3.20, so **4 lanes**. At 4 lanes peak utilisation is 120/(4·57.7) = 52%. With one lane down, capacity is 3·57.7·0.65 = 112.5 req/s, **below** the 120 req/s peak — so losing a lane during peak breaks the SLO unless demand is shed. The correct response is an admission policy that sheds long-context requests first and returns a retry-after, not an automatic fifth lane, because the fifth lane costs \$1,825/month to cover a rare correlated event; that is a policy decision the SRE owner should make explicitly.

**Reasoning.** The 65% ceiling exists because queueing delay grows nonlinearly in utilisation: for an M/D/1 approximation with 17.34 ms service, mean wait is t·(1 + ρ/(2(1−ρ))), which is 26.1 ms at ρ = 0.5 and 95.7 ms at ρ = 0.9. The ceiling is not decoration; it is how the tail is protected.

**Follow-up.** "At 240 req/s peak?" 240/(0.65·57.7) = 6.4 → 7 lanes; adapter memory per lane is unchanged, so the cache is not the constraint.

**Red flags.** Dividing 120 by 57.7 and getting 2.08 → "3 lanes" (ignores the utilisation ceiling and the loss case). Provisioning for the mean. Ignoring the lane-loss interaction entirely.

**Scoring.** 0: 2 or 3 lanes with no ceiling. 1: 4 lanes without the failure analysis. 2: 4 lanes plus the loss case. 3: adds the queueing argument and an explicit shedding policy.

**Support:** Sections 6.4, 6.7, 8d.

#### Q6 — Quantify the disclosure rate (calculation)

**Level:** advanced. **Skill:** risk quantification.

**Prompt.** "If the key omits tenant, at 40 req/s and h = 0.8444 with a 24-tenant cohort, how many wrong answers per hour, and what fraction of traffic?"

**Given.** q_wrong = 1 − 1/24.

**Strong answer.** Miss rate 1 − 0.8444 = 0.1556. Wrong answers per second = 40 × 0.1556 × 0.9583 = **5.965/s**. Per hour: 5.965 × 3,600 = **21,474**. As a fraction: 5.965/40 = **14.9%** of all traffic.

**Reasoning.** Only misses can resolve wrongly (a hit is keyed correctly by construction), and among misses the probability of landing on the correct tenant's artifact is 1/|cohort|. The multiplication is exact under steady state.

**Follow-up.** "What if there is one global key instead of a 24-tenant cohort?" q_wrong → 1 and the rate becomes 40 × 0.1556 × 3,600 = 22,406/hour. "What removes it?" Making tenant a key field makes q_wrong = 0 structurally; a post-load check makes it a rejection rather than a disclosure.

**Red flags.** Multiplying by hit rate instead of miss rate. Treating the cohort as 1. Forgetting to convert seconds to hours.

**Scoring.** 0: no quantification. 1: right direction, wrong factor. 2: correct rate. 3: also states the fraction of traffic and the global-key bound.

**Support:** Section 6.6 (E-06).

#### Q7 — Diagnose a mid-conversation change (ambiguous symptom)

**Level:** advanced. **Skill:** debugging.

**Prompt.** "A customer says our assistant 'changed its mind about our product name halfway through a chat'. No alerts fired. Where do you look?"

**Given.** A release went out yesterday; sessions are pinned by design.

**Candidate should clarify.** Whether the session has one `bundle_id` throughout; whether the change involved the adapter, the base, the tokenizer or the policy; whether other tenants saw it.

**Strong answer.** First check whether the session's turns carry more than one `bundle_id`. If they do, pinning is broken (F-03) — the bundle is being resolved per request, and about 1.25 sessions per release straddle a release, so the report is consistent with the design defect rather than with a model quirk. If the session has one `bundle_id` but the *behavior* changed, the release did not cause it: look at retrieval context, tool responses, or the tokenizer digest recorded on the turn. If neither, the customer's own product data changed.

**Reasoning.** Pinning makes mid-session change impossible *by construction*, so an observed mid-session change is evidence that the construction failed, not evidence that pinning is imperfect. Working backwards from "what would have to be false" is faster than inspecting model outputs.

**Follow-up.** "Pinning is correct but two tenants report it, and one is on the old bundle." Then the common factor is upstream of the model — likely a shared retrieval index updated outside the bundle.

**Red flags.** Blaming model nondeterminism. Looking at prompts first. Concluding "pinning cannot be perfect" without checking `bundle_id`.

**Scoring.** 0: proposes retraining. 1: suspects the release without a discriminating check. 2: checks `bundle_id` per turn. 3: gives the full decision tree including retrieval and tokenizer.

**Support:** Sections 6.8, 8b, 12 (F-03).

#### Q8 — Rollback that produces an invalid pair (constraint change)

**Level:** expert. **Skill:** release design.

**Prompt.** "We roll the adapter back to the previous version, and every request is refused for 90 seconds. What happened, and how should the release have been designed?"

**Given.** Base moved from v41 to v42 in the same window; adapters were released separately.

**Strong answer.** The rollback restored an adapter trained against v41 while the base stayed at v42. The compatibility guard correctly rejected the pair — the "rollback" was partial. Design fix: the **release unit must equal the validation unit**. Ship `(base, adapter, tokenizer, policy)` as one immutable bundle with one version; rollback is a pointer swap to a previously validated bundle. Then the arithmetic is favourable too: adapter rollback = 8 × 0.060 + 5 = **5.48 s** against a 15 s ceiling; base rollback = 4 × 12 + 5 = **53 s** against 180 s.

**Reasoning.** Composition validity is a property of the pair, not of either part. Independent release of parts means any rollback can create an unvalidated pair, and a guard that rejects correctly looks like an outage.

**Follow-up.** "Fleet grows to 14 lanes?" Base rollback becomes 14 × 12 + 5 = **173 s**, inside 180 s by 7 s — effectively unachievable once readiness checks are added. The rollback ceiling is therefore a scaling constraint and needs parallel prefetch.

**Red flags.** "The guard is too strict, relax it." "Roll back the base too" without saying that this then rolls back *every* tenant. Treating rollback as a version-number operation rather than a composition operation.

**Scoring.** 0: relax the guard. 1: identifies the mismatched base. 2: proposes bundling. 3: also computes both rollback times and flags the lane-growth boundary.

**Support:** Sections 6.8 (E-08), 8c (DEC-05), 12 (F-07).

#### Q9 — When the demand assumption fails (sensitivity)

**Level:** expert. **Skill:** judgement under changed assumptions.

**Prompt.** "Marketing launches self-serve. Demand across tenants flattens. What happens to your four-lane pool?"

**Given.** K = 512, uniform demand over 1,800 tenants.

**Strong answer.** Hit rate falls to 512/1800 = **0.2844**. Mean added latency becomes 8 + 0.7156·60 = **50.9 ms**, so lane capacity is 1000/50.9 = **19.6 req/s**. At 120 req/s peak with a 65% ceiling, lanes = 120/(0.65·19.6) = 9.4 → **10 lanes**. Cost rises from 5 to 11 GPUs, so the saving falls from 144× to about 65×, and the p95 tail worsens because 72% of requests now take the miss path.

**Reasoning.** h is a statement about demand mass, not about tenant count. A flatter distribution means the same number of slots covers less mass, and the miss path dominates.

**Follow-up.** "What do you do architecturally rather than buying lanes?" Shard by tenant cohort with a per-shard cache sized to its own distribution, or admit by rate (serve adapters only above a request-rate threshold and use prompt-level personalisation below it). Both are decisions that must be measured, not guessed.

**Red flags.** "Hit rate stays the same because K is the same." Reporting the cost change without the latency change. Assuming the saving is invariant.

**Scoring.** 0: no change predicted. 1: hit rate only. 2: hit rate, latency and lanes. 3: also states the cost multiple and proposes an architectural alternative.

**Support:** Section 6.9.

#### Q10 — Build the negative test (design)

**Level:** applied. **Skill:** evaluation design.

**Prompt.** "Design a test that would catch a missing tenant field in the key. It must fail before anything is served."

**Given.** Two authenticated sessions, tenants A and B; a third tenant C exists.

**Strong answer.** (i) Authenticate as A, send a benign prompt, record `adapter_digest`, assert it equals A's manifest digest. (ii) Repeat as B, assert a different digest and B's manifest. (iii) As A, include a payload field naming tenant C or C's adapter id, and assert **denial before routing** — evidenced by the absence of a slot lookup and the presence of a typed error (`E_KEY_MISMATCH` or a policy denial). (iv) Force eviction of A's adapter, then serve B, then serve A again, and assert A receives A's digest — this is the case a plain two-tenant test misses, because it exercises the *miss* path where the wrong resolve happens.

**Reasoning.** The defect only manifests on a miss after another tenant has loaded under a shared key. A test that only exercises hits cannot find it. The eviction step is what makes the test discriminating.

**Follow-up.** "How do you know the test is running on every path?" Inventory endpoints and probe each; a path that cannot be probed is disabled (A-02).

**Red flags.** Testing with one tenant. Checking the response text instead of the resolved digest. Relying on the model to refuse.

**Scoring.** 0: prompt-level test. 1: two tenants, hits only. 2: adds the digest assertion. 3: adds the eviction/reload step and the pre-routing denial evidence.

**Support:** Sections 9.3 (T-01), 14.2.

#### Q11 — Why monitoring cannot be the control (judgement)

**Level:** advanced. **Skill:** control design.

**Prompt.** "We already log every request. If the key were wrong we would see it in minutes. Why is that not enough?"

**Strong answer.** Three reasons. **Ordering:** detection is after delivery; the 21,474 hourly answers have already been read. **Evidence dependency:** the journal only proves crossover if it records the *resolved adapter digest* and can join it to the principal's tenant — a journal with `bundle_id` alone cannot answer the question. **Silence in normal metrics:** latency improves, cost falls, and per-tenant quality evaluations look normal because each runs in isolation; nothing in the aggregate metric set moves. So monitoring bounds *duration* and cannot substitute for a *structural* prohibition, which is what the contract requires.

**Reasoning.** A metric is an observation about traffic that occurred. The requirement is a property of the program. Those are different kinds of claim, and only one of them can be established by inspection.

**Follow-up.** "What is the residual role of monitoring?" Detecting that the structure broke — a `E_KEY_MISMATCH` counter that must be zero, canary probes, and a slot-table divergence check. It is the alarm on the invariant, not the invariant.

**Red flags.** "Zero incidents so far proves isolation." "We would notice." Treating detection latency as a sufficient control.

**Scoring.** 0: monitoring suffices. 1: one reason. 2: two reasons with the journal dependency. 3: all three plus the correct residual role.

**Support:** Sections 9.2 (G-1), 9.6.

#### Q12 — Canary power (calculation)

**Level:** applied. **Skill:** statistical reasoning.

**Prompt.** "We run 50 canary probes per week per tenant. What leak probability do we actually detect?"

**Given.** Binomial detection, independent probes.

**Strong answer.** Power = 1 − (1 − q)^C. At q = 0.05 and C = 50, power = 1 − 0.95⁵⁰ = **92.3%**. At q = 0.02, power = 1 − 0.98⁵⁰ = **63.6%**. At q = 0.10, power = **99.5%**. So the schedule is decent against a broad leak and weak against a narrow one. To reach 95% at q = 0.02 needs C = ln(0.05)/ln(0.98) ≈ **149 probes**. With zero hits in n probes, the one-sided 95% upper bound is 3/n, so 1,800 tenant-weeks gives 3/1800 = 0.0017 per tenant-week — an operational bound, not a proof of impossibility.

**Reasoning.** Independence is assumed and is optimistic: probes for one tenant share a prompt template, so a leak triggered by a different phrasing is invisible regardless of C. The power calculation bounds template-level detection, not all detection.

**Follow-up.** "How do you fix the dependence?" Randomise prompt templates per probe and stratify by intent class, then report power per stratum — which lowers effective C per stratum and must be accounted for.

**Red flags.** "50 probes is statistically sound" without a number. Treating 1 − q^C as the formula. Ignoring dependence between probes.

**Scoring.** 0: no computation. 1: correct formula, wrong interpretation. 2: correct numbers. 3: adds the 3/n bound and the dependence caveat.

**Support:** Section 9.4.

#### Q13 — Pinning exposure (calculation)

**Level:** applied. **Skill:** time-based reasoning.

**Prompt.** "We pin per session. How many users are on a stale bundle right after a release, and what changes if we support hour-long sessions?"

**Given.** 4 sessions/s, 15-minute mean duration, releases every 30 days.

**Strong answer.** Concurrent sessions = 4 × 900 = 3,600. P(a session straddles a release) = 1 − exp(−900/2,592,000) = **3.47×10⁻⁴**. Expected straddling sessions per release = 3,600 × 3.47×10⁻⁴ = **1.25**. With 3,600-second sessions: concurrency 14,400, p_span = 1 − exp(−3,600/2,592,000) = 1.388×10⁻³, and expected straddlers = **20.0** — a 16× increase, because both factors quadruple.

**Reasoning.** For a Poisson arrival process with exponential durations and independent releases, the straddle probability is the busy-period fraction. The small-probability approximation p ≈ T/τ gives 3.47×10⁻⁴ directly and is accurate here to three significant figures.

**Follow-up.** "Is 20 sessions per release acceptable?" It is a business judgement: 20 conversations see old behaviour for up to an hour. If the release is urgent (a safety fix), pinning becomes the wrong control and you need a forced session restart with notice.

**Red flags.** Counting sessions started per day instead of in flight. Forgetting that duration enters the formula twice. Assuming pinning guarantees no user sees a change.

**Scoring.** 0: no model. 1: concurrency only. 2: correct straddle probability. 3: adds the 16× scaling and the forced-restart escape hatch.

**Support:** Section 6.8 (E-07).

#### Q14 — Cost per 1,000 requests and its denominator (calculation + framing)

**Level:** applied. **Skill:** unit economics.

**Prompt.** "Five GPUs at \$2.50 per GPU-hour, 18 req/s mean. What is the cost per 1,000 requests, and what is wrong with that number?"

**Strong answer.** Monthly requests = 18 × 86,400 × 30 = **46,656,000**. Monthly cost = 5 × 2.50 × 730 = **\$9,125**. Cost per 1,000 = 9,125/(46,656,000/1,000) = 9,125/46,656 = **\$0.196**. The number is misleading in two ways: capacity is provisioned for the 120 req/s peak, so the *marginal* cost of a mean request is near zero while the *average* cost is dominated by idle peak capacity; and the denominator should exclude degraded and failed responses, which raises the figure. Against the dedicated baseline's \$28.16 per 1,000, the ratio is 144×, and both numbers move together with the traffic level, so the ratio is more robust than either absolute value.

**Reasoning.** Average cost divides total cost by total output and therefore charges peak capacity to mean traffic. That is the right number for a pricing sheet and the wrong number for a keep-or-kill decision on one more lane.

**Follow-up.** "Traffic halves?" Cost per 1,000 doubles to \$0.392 while the baseline doubles to \$56.32; the ratio is unchanged, which is a favourable property worth stating explicitly.

**Red flags.** Using 720 hours instead of 730. Dividing by requests at peak. Quoting the ratio as if it were a margin.

**Scoring.** 0: no units. 1: right arithmetic, wrong hours/denominator. 2: correct figure. 3: critiques the denominator and states the ratio's robustness.

**Support:** Sections 8d, 11.1.

#### Q15 — Transfer to a feature store (transfer)

**Level:** expert. **Skill:** transfer.

**Prompt.** "Same failure, different system: an online feature store serving per-entity features to many models. What is the key, and what is the adapter analogue?"

**Strong answer.** The key is `(entity_id, feature_view_version, as_of_timestamp, policy_version)`. The adapter analogue is the **feature view** — a materialised, versioned artifact that legitimately differs per entity but must never be served for another entity or from a different `as_of`. The failure modes map one to one: a mutable view name in the key (F-01's logical-name bug), a rollback that restores a view without its schema (F-07's composition bug), and a cache sized on an assumed skew that flattens (F-05). The one thing that does **not** transfer is the privacy weight: a wrong feature is a model-quality bug, not necessarily a disclosure, so the zero-tolerance rule softens into a quality SLA.

**Reasoning.** The mechanism is "authorisation fields must be in the cache key", which is domain-independent. The *consequence* of violation is domain-dependent, and that is what determines whether the control is zero-tolerance or budgeted.

**Follow-up.** "Where does the analogy break hardest?" When features are legitimately shared across entities (a global aggregate), the entity field must still be in the key for point-in-time correctness but not for privacy — a distinction that matters for how you test it.

**Red flags.** Copying the design without re-deriving the consequence. Assuming "multi-tenant" implies "privacy-critical". Declaring the transfer invalid because the domains differ.

**Scoring.** 0: no transfer. 1: names a cache only. 2: maps the key fields. 3: also separates the mechanism from the consequence and names where the analogy breaks.

**Support:** Section 13.

### 14.2 Negative-test fixtures and control invariants

The invariants below are written as testable statements, because an invariant that cannot be tested is a slogan.

| Invariant ID | Statement | Test | Failure signal |
|---|---|---|---|
| INV-01 | The loader's adapter lookup accepts a principal, never a tenant string | Static check: the function signature has no `tenant` parameter | A `tenant` parameter exists anywhere in the lookup path |
| INV-02 | A slot's composition requires all five key fields to equal the pinned bundle | Property test: mutate each field in turn; every mutation must yield a miss or denial, never a serve | Any mutation serves tokens |
| INV-03 | `bundle_id` is constant for a session | Harness test: trigger a release mid-session; assert one value across all turns | Two `bundle_id` values in one session |
| INV-04 | Every request journals the resolved `adapter_digest` and `base_digest` | Schema check plus a sampled join against the manifest | A journal row without digests, or a row whose digests have no manifest |
| INV-05 | The serving role cannot write the slot table except through the loader | Credential review plus an attempted direct write | The direct write succeeds |
| INV-06 | No request is served on a slot in `draining` state | Force a drain; assert new requests route elsewhere or degrade to base | A request is served on a draining slot |
| INV-07 | A failed key check degrades to the base model, never to another adapter | Inject a deliberate mismatch; assert `degraded=true` and base-only output | Output attributable to any adapter |

**Fixture catalogue.** Eight fixtures, each a named scenario that the suite must run:

```text
FX-01  two-tenant hit      A then B, both resident -> distinct digests, both correct
FX-02  eviction reload     A evicted, B loaded, A returns -> A gets A's digest
FX-03  foreign-id payload  A sends tenant=C in body -> denial before slot lookup
FX-04  mid-session release release between turn 2 and 3 -> one bundle_id throughout
FX-05  drain               slot marked draining -> no new traffic, in-flight completes
FX-06  torn slot           concurrent loads of A and B -> exactly one writer per slot
FX-07  rollback pair       roll back adapter only -> guard refuses, base-only degraded
FX-08  canary              foreign canary in training set -> pre-train scan fails
```

### 14.3 Architecture success and recovery traces

**Successful request (hit).**

```text
t=0.0 ms   request arrives; session credential verified; principal = tenant 417
t=0.4 ms   bundle pinned at session start resolved: B-2026-11-07.3
t=0.6 ms   policy: region EU-W, addendum active, quota OK
t=0.9 ms   slot lookup (417, ad-9f2c, base-41, tok-7, pol-12) -> slot 88, state=ready
t=1.0 ms   post-lookup re-verify slot tuple == bundle tuple -> equal
t=1.1 ms   compose base v41 + adapter ad-9f2c; generate
t=788 ms   stream complete (p95 path)
t=789 ms   journal: request_id, principal 417, B-2026-11-07.3, ad-9f2c, base-41, slot 88
t=790 ms   response flushed
```

**Failed request, recovery traced (miss with load failure).**

```text
t=0.0 ms   request arrives; principal = tenant 902
t=0.6 ms   policy OK; slot lookup -> miss (adapter not resident)
t=0.8 ms   single-flight: no in-flight load for (902, ad-4b71) -> this request leads
t=0.9 ms   loader reserves slot 511, state=loading
t=61 ms    load fails: registry read timeout (not a key mismatch)
t=61 ms    leader marks slot free; retry budget check: 4.2% of window used, 5% cap
t=62 ms    one retry attempted with backoff 50 ms -> succeeds at 118 ms
t=118 ms   post-load verify: slot tuple == bundle tuple -> equal
t=119 ms   compose and generate; journal records load_retry=1
t=905 ms   end-to-end 905 ms -> SLO breach recorded, not hidden
alt path:  if the retry had failed, the request degrades to base with
           degraded=true and the journal records E_ADAPTER_UNAVAILABLE;
           no other tenant's adapter is ever a fallback
```

The important detail is the last line: **the fallback ladder has exactly two rungs — this tenant's adapter, or no adapter.** There is no third rung, because any third rung would be another tenant's weights. A fallback ladder with a "nearest adapter" rung is how the design in F-01 would be half-fixed and still wrong.

### 14.4 Claim and evidence ledger

| Claim ID | Exact claim and scope | Evidence type | Source / artifact | Supports | Does not establish | Recheck trigger |
|---|---|---|---|---|---|---|
| CLM-01 | One adapter is 32.0 MiB for rank 16, 4 modules, 32 layers, d=4096, bf16 | derived | E-01, RUN-01 | The slot budget | Any real model's size | Training config change |
| CLM-02 | At most 523 adapters fit in a 16.4 GiB budget | derived | E-02, RUN-01 | Cache boundedness | Achievable occupancy in practice | HBM, base size, KV or runtime change |
| CLM-03 | h(512) = 0.844 under Zipf s=1 | derived, model-dependent | E-03, RUN-02 | Lane count | Real hit rate; s is unmeasured | Demand measurement |
| CLM-04 | Mean added latency 17.34 ms; p95 added 68 ms | derived | E-04 | Capacity and SLO | Measured latency | Swap cost or hit rate change |
| CLM-05 | A name-keyed cache yields ~21.5k cross-tenant answers/hour | derived | E-06 | The central design claim | That any real system had this bug | Key specification change |
| CLM-06 | Pool cost \$9,125/month vs \$1,314,000 dedicated (144×) | derived, illustrative price | Section 8d, 11 | The economic argument | Any realised saving | Price, tenant count or demand change |
| CLM-07 | ~1.25 sessions straddle a release at 15-min sessions | derived, model-dependent | E-07 | Pinning policy | Observed straddle count | Session-length change |
| CLM-08 | Adapter rollback 5.48 s; base 53 s at 4 lanes | derived | E-08 | Operability | Drill result | Lane count or storage change |
| CLM-09 | 50 weekly canary probes detect q=0.05 at 92.3% | derived | Section 9.4 | Detection adequacy | Detection of prompt-specific leaks | Probe design change |

### 14.5 Local authoring checkpoint

| Check | State |
|---|---|
| Fifteen numbered sections in order, card last | complete in this file |
| Frontmatter declares draft, `last_reviewed: null`, evidence illustrative | complete |
| Every quantitative claim labelled derived/illustrative, none measured | complete |
| Four or more failure modes with competing causes, discriminating checks, bounded fixes, verification | eight, Section 12 |
| Interview questions with reasoning, red flags, scoring, support | fifteen, Section 14c |
| No invented company, benchmark, incident or regulatory outcome | confirmed; Meridian Support Cloud is explicitly hypothetical |
| Arithmetic independently re-derived | Section 6.11 cross-check table |
| Sensitivity and boundary analysis | Section 6.9, nine rows including two reversals |

---

## 15. Quick Reference Card

| Field | Value |
|---|---|
| Domain | Model customization / multi-tenant adapter serving and isolation |
| Problem type | System operation plus one architectural decision: how to key, cache, pin and roll back tenant adapters |
| Method | Composite 5-field cache key from the authenticated principal, 512-slot per-lane adapter cache, per-session bundle pinning, cache-aware routing with single-flight loading, atomic base+adapter+tokenizer+policy release bundle; baseline is merged dedicated checkpoints |
| Primary result | **Not measured.** Derived: 5 GPUs at \$9,125/month versus 720 GPUs at \$1,314,000 (144×), mean adapter latency 17.34 ms, p95 788 ms against a 900 ms ceiling, ~523 slot memory ceiling, 512 chosen |
| Key lesson | A cache key must contain every field that determines **authorisation**, not only those that determine computation; and the unit of rollout must equal the unit of validation, or a rollback can restore an unvalidated composition |
| Tags | authorization-scope (primary); atomic-release-bundle, rollout-regression (secondary) |

| Supplemental field | Value |
|---|---|
| Case ID / version / as-of | CASE-49-003 / 1.0 / 2026-11-10 |
| Evidence level | `illustrative` — derived arithmetic on constructed inputs; isolation property is proposed, not observed; no pilot ran |
| Implementation stage | `proposed` — design exercise; no component built, no tenant migrated |
| Baseline / primary metric | Merged dedicated checkpoints / cross-tenant adapter resolution rate (fraction of requests whose served adapter belongs to another tenant), 28-day window, with cost per 1,000 successfully served requests as the economic secondary |
| Key limitation | The demand exponent and swap latency are unmeasured; if demand flattens, lanes go from 4 to 10, and the whole isolation claim rests on tenant identity arriving from the authenticated principal — an unverified assumption |
| Review trigger / owner | Any change to cache keying, base checkpoint, tokenizer, slot count, tenant demand mix or release procedure; model platform owner with security veto |

**Thirty-second read.** Shared adapter serving is worth roughly two orders of magnitude in cost and is safe only if the key is built from the authenticated principal and the release unit is the validation unit. The two numbers to remember are **21,474 wrong answers per hour** from one missing key field, and **512 slots serving 84.4% of requests** while the p95 still lands on the swap path. Both are arithmetic; neither is measured; the pilot exists to convert them.



