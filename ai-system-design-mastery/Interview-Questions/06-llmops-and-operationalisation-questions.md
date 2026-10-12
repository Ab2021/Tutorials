# LLMOps and Operationalisation Interview Questions

**TL;DR** — Forty-two questions on running AI systems: release engineering with independently-versioned assets, capacity planning and autoscaling for GPU-bound workloads, cost and FinOps, reliability and incident response, security and compliance operations, disaster recovery, and continuous quality measurement. These are the questions that separate people who have designed AI systems from people who have *operated* them, and the answers are correspondingly concrete — real metric names, real thresholds, real rollback procedures.

**What the interviewer is listening for**

| Weak answer | Strong answer |
|---|---|
| "We monitor latency and errors" | "We alert on p95 TTFT above 2s for 5 minutes, and on judge-agreement kappa below 0.4" |
| "We can roll back" | "Rollback repoints the release manifest in about 90 seconds; it's a config change, not a deploy" |
| "We use caching to save money" | "Cache hit rate is a first-class SLI at 35%; when it collapses, the bill doubles with flat traffic" |
| "We evaluate with LLM-as-judge" | "Distilled judge inline at 100%, frontier judge on 1–5%, humans on disagreement, quarterly retrain" |
| "We have a DR plan" | "Embeddings are rebuildable, prompts are restorable, eval sets are irreplaceable — and only the third gets object-lock" |

**The operating principle that runs through all of it:** in a normal system, a regression produces an error. In an AI system, a regression produces a confident correct-looking answer with a 200 status. Everything below follows from the fact that you cannot alert on what doesn't throw.

---

## Section A — Deployment, release and CI/CD

### Q1. What's different about CI/CD for an LLM system?

**A.** The unit of release isn't the code. In a standard service, deploying the artifact deploys everything. In an LLM system there are **six independently-versioned assets** — code, prompts, the model, the index, guardrail configuration, and runtime config — and any one of them can change behaviour with no application deploy at all. A prompt edit in a database, a provider's silent model update, an index rebuild with a new embedding version, a guardrail threshold relax: each is a production change and none of them goes through your pipeline by default.

So the core mechanism is a **release manifest**: a single immutable record that pins the versions of all six, with a `release_id`. Four rules follow from it. The manifest is immutable — you never edit it in place. The pointer to the active manifest is the only thing a deploy changes. The `release_id` goes into every trace and log line, so any output can be attributed to an exact configuration. And rollback is repointing to the previous manifest, which means it takes seconds rather than a redeploy.

### Q2. How do you handle prompt releases?

**A.** Prompts are code and should be treated as such: version-controlled, reviewed, tested, and revertible. The failure mode to design against is the prompt that lives in a config table and gets edited directly in production by whoever has database access, because that change is invisible — it has no review, no test, no trace, and no rollback, and it can silently degrade quality for weeks.

Concretely: prompts in a repository with semantic versioning, a change requiring review by someone who owns the eval set, a blocking CI gate that runs the golden set against the new prompt and compares to the current one with a defined threshold (I'd use a two-point drop on a five-point scale as the failure line), and canary at 5% of traffic for at least an hour before full rollout. And the prompt version goes in the trace, because the first question in any quality investigation is "which prompt was this?"

### Q3. What does the eval gate look like in CI?

**A.** A blocking job that runs on every change to code, prompt, model pin, or guardrail config. It runs four things: a **golden set** of 200–500 hand-labelled cases scored for correctness and faithfulness; **slice metrics** broken out by query type, difficulty, and tenant, because the aggregate hides regressions; **regression checks on known-hard cases** from past production failures; and **cost and latency** — because a change that improves quality 1% and triples latency isn't a change you ship.

The pass criteria are explicit: no slice drops more than the threshold, faithfulness stays above 0.85, the hallucination rate stays under 0.03, p95 latency doesn't grow more than 15%, and cost per request doesn't grow more than 20%. And critically, the eval set is versioned separately from the system, with its own review — because a release that can edit its own exam isn't being tested, and relaxing an assertion to make a red build green is the single most common way an eval gate becomes decorative.

### Q4. How do you do a canary for a generative feature?

**A.** The tricky part is that you can't just watch error rates — a canary producing confidently wrong answers shows 100% success metrics. So the canary needs **quality signals in the comparison**, not just operational ones. My design: route 5% of traffic to the new release, and run the shadow evaluation concurrently — sample both the canary and control traffic, score them with the same judge, and compare the score distributions rather than just the point estimates.

The gate to promote: operational metrics within tolerance (error rate, p95 latency, cost per request), **judge score difference not statistically significant or in the wrong direction**, and no new failure signatures in the trace tags. Run for a minimum of an hour and a minimum sample size — with a 5% canary at low traffic you may need a day, and a canary that ran for twenty minutes at 3 requests a second hasn't seen enough to conclude anything. And I'd include a manual review of a sample of canary outputs, because that's the only thing that catches the failure where every metric looks fine and the answers are subtly wrong.

### Q5. How do you roll back an AI system?

**A.** By classifying what actually changed, because the rollback differs. **Prompt or config change** — repoint the manifest, effective in seconds, no rebuild. **Model pin change** — repoint, seconds, though if a provider has deprecated the old version you may not be able to go back, which is why provider deprecation gets its own lead time. **Index change** — this is the slow one: you keep the previous index alive under an alias and the rollback is an alias flip, which is seconds, but it means you can't tear down the old index immediately after a rollout. **Code change** — a standard deploy rollback.

The rule that makes this work: **never overwrite the thing you might need to roll back to.** Old manifests stay readable, old indexes stay warm for at least 24 hours, old prompt versions stay addressable by version. And I'd test the rollback path regularly — a rollback you've never executed is a hypothesis, and rolling back at 3am during an incident is not when you want to discover that the previous manifest referenced an index you deleted last week.

### Q6. What's your change management policy for AI releases?

**A.** Three rules that need two pairs of eyes, and everything else can be self-service. **Tightening is free, loosening needs a reviewer** — making a guardrail stricter, narrowing a permission, raising a threshold can ship without ceremony; the reverse direction needs review because relaxing a control is where risk enters. **Any change to a model pin, embedding version, or guardrail threshold needs a second reviewer**, because those have non-local effects that a single author might not see. And **any change that alters the cost profile by more than 20% needs a reviewer**, because cost changes are invisible until the bill arrives.

Everything else — prompt wording that doesn't change behaviour, adding an eval case, better tool descriptions — should be fast, because a process that makes ordinary work slow gets bypassed entirely. The measurement that tells you whether the policy is right is the audit: if you find production changes with no corresponding release manifest entry, the process is too heavy for the people using it.

---

## Section B — Capacity planning and autoscaling

### Q7. How do you capacity-plan an LLM system on a hosted API?

**A.** It's mostly a rate-limit and concurrency problem rather than a hardware problem, which changes what you plan. Start from the standard estimation chain: DAU × requests per user per day gives requests per day; divide by 86,400 for average QPS; multiply by 3 for peak QPS (the standard assumption that peak is roughly 3x average — for consumer traffic I'd consider 4–5x around daily peaks).

Then convert to the two constraints that actually bind. **Tokens per minute** — requests per minute × (average input + expected output) tokens. And **concurrency** — peak QPS × p95 latency in seconds, which for a 3-second p95 at 50 QPS means 150 concurrent requests in flight. Concurrency is the number that determines whether you need more quota, a queue, or a cheaper model tier, and it's the one people forget to compute. Then add headroom: provision for 2x peak, because provider quota is granted in tiers and upgrades take days, and you cannot buy your way out of a limit during an incident.

### Q8. What's the difference between autoscaling for inference and autoscaling for a normal service?

**A.** Four differences. **Scaling is slow** — a GPU node takes 2–10 minutes to be ready, versus seconds for a CPU pod, so reactive scaling is too late and you need predictive or scheduled scaling ahead of known traffic patterns. **GPU utilisation is the wrong metric** — it reads high even when the node is idle because of how memory allocation works, so queue depth, time-to-first-token, and KV-cache utilisation are the real signals. **Scale-down is expensive** — killing a warm node discards the KV cache and any locality you had, so aggressive scale-down creates a cold-start penalty on the next request. And **capacity is scarce and non-fungible** — you can't run a 70B model on the small instance type, so scaling by adding a different shape of node doesn't help.

So my scaling policy: scale on queue depth and TTFT, not GPU utilisation; scale up on a 30-second lookahead, scale down slowly with a 15-minute cooldown; keep a baseline minimum rather than scaling to zero, because cold start on a large model is minutes; and set the target utilisation at 60–70% rather than 80–90%, because the last 20% of capacity is where latency falls off a cliff.

### Q9. How do you handle a traffic spike?

**A.** In order of what I'd actually do under pressure. **Queue and shed** — an admission control layer with a bounded queue, serving what you can and returning a clear "at capacity, try again" for the overflow rather than letting everything degrade together. A queued request that completes in 8 seconds is often better than a shed one, but unbounded queuing means everything times out. **Cascade routing** — under load, route more traffic to the smaller model; quality drops and the system stays up, which is the right direction of degradation. **Serve from cache more aggressively** — raise the semantic cache similarity tolerance temporarily, accepting a slightly higher false-hit rate to shed load. **Reduce optional work** — drop reranking, reduce retrieved chunk count, shorten output caps — each of these is a quality-for-capacity trade you should have pre-decided. And **disable expensive guardrails last**, because safety comes out of the degradation ladder entirely; it's not a level you trade away.

The thing I would not do is let it fail open — an overloaded system that drops safety checks and serves injected content is worse than one that returns "busy."

### Q10. What capacity headroom do you provision?

**A.** For a hosted-API system: provision quota at 2x peak, and keep a secondary provider configured for at least 25% of steady-state traffic so failover is a live path rather than an untested one. For self-hosted GPU infrastructure, the arithmetic is different and the answer is expensive: reserve GPU capacity at 60–100% of the primary pool for the standby site, because GPU capacity in a second region is not something you can obtain on demand during an incident. That means DR headroom for GPU workloads costs 60–100% of primary, not the 25–40% a warm-standby CPU tier costs.

The judgement I'd articulate: **over-provisioning latency-critical capacity is cheap relative to the cost of the incident**, but it's not free, so it should be justified per tier. Availability arithmetic is the argument — 99.9% is 8.76 hours a year of downtime, and if an hour of that costs more than a year of headroom, the headroom is the rational purchase.

### Q11. What does a capacity SLO look like?

**A.** Two layers. **Resource SLOs** — GPU utilisation in the target band (60–70% for latency-sensitive), queue depth below a threshold (I'd use 10 waiting requests), KV-cache utilisation below 85%, and headroom above 20% at all times. **Service SLOs** — the ones users feel: TTFT p95 under 2s, total latency p95 under 10s, error rate under 0.5%, and throughput at the provisioned peak.

The relationship between them is what makes the pair useful: resource SLOs are leading indicators, service SLOs are lagging. By the time TTFT degrades, you're already in trouble; the queue depth crossing 10 gave you three minutes of warning. So the alerts fire on the resource metric and the page fires on the service metric, which gives you a chance to fix it before anyone notices. And the capacity review is a scheduled exercise — recompute the estimation chain monthly with actuals against the model, because the growth curve is the thing that tells you when to renegotiate quota.

---

## Section C — Cost and FinOps

### Q12. Where does the money actually go in an LLM system?

**A.** Inference dominates — roughly 80–90% of GenAI spend — and within inference, the pattern that surprises people is that **input tokens usually dominate the bill for RAG and agent workloads**, not output. There's a common belief that output is the expensive part, which is true per-token (output is typically 3–5x the price of input), but a RAG request might be 8,000 input tokens and 300 output tokens, so the input side wins on volume. System prompts are a specific offender: in a typical agentic setup the system prompt and tool definitions can be around 69% of all input tokens, and they're identical on every request.

The other two structural facts: **only about 28% of calls use caching** in the systems I've seen measured, which means most teams are paying full price for tokens they sent identically a million times before. And **reasoning models multiply output tokens 3–15x**, because the visible answer is a fraction of the tokens generated, which is a real capability gain and a real budget line.

### Q13. What are the cost levers, in order of return?

**A.** Five, and the order matters because people reach for the complex ones first. **Prompt caching** — the highest return and the easiest: it attacks the system-prompt and tool-definition problem directly, and if your static prefix is 3,000 tokens, caching it is often a 40–60% input-cost reduction for a configuration change. **Cascade routing** — classify by difficulty and send easy requests to a small model; published results land around 45–85% savings at roughly 95% of frontier quality, and those numbers are achievable because most real traffic is easy. **Batch processing** — if the workload tolerates latency, batch APIs are typically around 50% off list. **Reserved or spot capacity** — for self-hosted, committed use or spot instances cut the GPU line substantially, at the cost of interruption risk. And **distillation or fine-tuning** — expensive up front, worth it when a narrow high-volume task can be moved to a small model permanently.

The one I'd flag as under-used is caching, because it requires no quality trade-off at all. The one I'd flag as over-used is aggressive model downgrading, because it's the only lever that can silently reduce quality without any signal.

### Q14. "Our bill doubled and traffic is flat." Where do you look?

**A.** In a specific order, because there's a dominant cause. **First: cache-hit collapse.** This is the number-one cause of that exact symptom, accounting for something like 40% of the cases I've seen. Something invalidated the prompt cache or the semantic cache — a prompt version bump that changed the prefix, a cache key that now includes a field that varies, a TTL that's too short, a routing change that spreads identical prefixes across providers. Check cache hit rate first; if it fell, you've found it.

Then, in order: **output length drift** — did the model start producing longer answers, or did a prompt change raise the verbosity? **Retry rate** — are you paying for failed calls? **Agent step count** — same traffic, more steps per task. **A model pin change** — a provider silently moved you to a more expensive or more verbose version. **Retrieval volume** — did the top-k change, or did contextual enrichment start running at query time? And **guardrail sampling** — did an evaluation rate go from 1% to 100%? The disciplined approach is to instrument cost per request by component, so this is a dashboard lookup rather than an investigation.

### Q15. How do you attribute cost to teams?

**A.** In three stages, and the sequence is deliberate because the first stage is where most of the value is. **Showback** — measure and publish cost per team, per feature, per tenant, with no budget consequences. This alone changes behaviour, because visible cost is surprising cost, and it's where you find the team whose agent retries eleven times. **Soft chargeback** — budget targets with alerts, still no financial consequence, which gets teams to plan. **Hard chargeback** — actual internal billing, which is what changes architecture decisions but requires the attribution to be trustworthy first.

The attribution mechanism has to be at the request level, not estimated: every request carries a `tenant_id` and a `feature_id` in its trace, tokens are recorded with a price table version (prices change), and the rollup is by those dimensions. The failure mode is attribution that's approximately right, because a team that's charged 20% too much will spend more engineering time disputing it than the cost control was worth, and hard chargeback dies on the first billing dispute.

### Q16. How do you set a cost budget for a feature?

**A.** Top-down and bottom-up and reconcile. **Top-down**: what's the business value of this feature per interaction, and what fraction of that can go to inference? For a support deflection, the alternative is a human agent at some fully-loaded cost per contact, so if you're deflecting 30% and the contact costs $6, your budget is a meaningful fraction of $1.80. **Bottom-up**: from the estimation chain, compute tokens per request by component, apply the price table, and multiply — that gives cost per request; multiply by request volume for the monthly run rate, then add the eval and observability stack (typically 2–4% of inference).

Then set the alert at 80% of budget and a hard circuit at 100%, and — this is the part people skip — define what happens at the circuit: degrade to a cheaper path, not fail. A budget that takes the feature offline is a budget nobody will set tightly. At a million requests a day with a full eval stack, I'd expect the total operating cost to land in the tens of thousands of dollars a month, with the eval layer being a low single-digit percentage of it.

---

## Section D — Reliability and incident response

### Q17. What's the hardest thing about reliability in an AI system?

**A.** That the most important failure class doesn't produce an error. I separate failures into two classes and the separation is the whole framing: **infrastructure failures** (timeouts, 5xx, rate limits) are visible, they have status codes, and standard reliability engineering handles them. **Semantic failures** are the answer being wrong, unfaithful, or unsafe — and the request returned 200 in 340 milliseconds with a perfectly-formed response. Standard monitoring reports 100% availability while the system is 60% correct.

So the reliability stack needs a whole second instrumentation layer: sampled quality scoring, faithfulness measurement, abstention-rate monitoring, and hallucination-rate tracking — all of them SLIs with alerts, exactly like error rate. And the operational consequence is that the runbook for a semantic degradation is completely different from the runbook for an infrastructure outage, because the diagnostic question isn't "what's down" but "what changed."

### Q18. Walk me through retry policy for LLM calls.

**A.** Retry on **transient** only: 429, timeouts, 503, connection errors — with exponential backoff, base delay 1 second, doubling, max delay 60 seconds, maximum 3 attempts, and **jitter**. Jitter isn't optional: without it, a provider blip produces a synchronized retry stampede from your whole fleet that turns a transient failure into a sustained one. I'd use full or 0.5-factor jitter.

Never retry on **permanent** errors: 401, 403, 400-level validation failures, policy violations, context-length overflow, and hard quota exhaustion. Retrying those burns budget and delays the real fix.

And the rule that doesn't generalise from normal engineering: **never retry a semantically wrong answer.** If the model produced a bad answer, repeating the identical request with the identical context produces the identical bad answer with a different sample, which is a coin flip, not a fix. Retries handle transient infrastructure; the fix for a semantic failure is a change to prompt, retrieval, or model. There's also a fleet-level circuit: if the retry rate crosses 10% globally, stop retrying and start shedding, because that level means the provider is degraded and your retries are making it worse.

### Q19. Explain circuit breakers for LLM providers.

**A.** A breaker wraps calls to a specific `provider:model:operation` key and trips to OPEN after 5 consecutive failures (or a failure rate above a threshold at volume). While OPEN, calls fail fast without touching the provider. After a 30-second recovery timeout, it goes HALF_OPEN and admits 3 trial calls; if 2 succeed, it closes; if any fail, it reopens and the timeout backs off.

The reason it matters more for LLM systems than for normal services: providers fail in ways that are **expensive to discover**. A degraded provider might return 200s with truncated or nonsensical content, so you're not saving latency by failing fast — you're stopping yourself from paying for garbage. And the key granularity matters: per-provider, per-model, and per-operation, because a provider's embedding endpoint can be healthy while its completions endpoint is degraded, and a breaker at the provider level would take down the healthy path.

The interaction with retries is the subtle part: retries are inside the breaker's scope. If both are naive, you get retries amplifying a provider outage; the retry budget has to be per-request, and the breaker has to count failures per attempt.

### Q20. What are bulkheads and how do you size them?

**A.** Bulkheads partition concurrency so that one workload can't consume all capacity and starve the others. For LLM systems I'd partition into four pools with separate concurrency limits: **critical** at 10 (interactive, revenue-affecting), **realtime** at 50 (the main user-facing chat path), **agents** at 30 (long-running multi-step work), and **batch** at 200 (offline processing).

Agents getting their own bulkhead is the important design decision and it's the one people skip. An agent task can occupy a slot for two minutes across ten steps, whereas an interactive request occupies one for three seconds — so a modest number of concurrent agent tasks consumes capacity equivalent to hundreds of interactive requests. Without a separate pool, the agent workload will starve interactive traffic during exactly the period when users are complaining about slowness, and the two workloads have completely different latency requirements.

Sizing: derive from the concurrency calc (peak QPS × p95 latency), then reserve headroom. And the shed order when a pool is full should be explicit and pre-decided: batch first, then agents, then realtime, and critical never.

### Q21. How do you handle a model regression you didn't ship?

**A.** Assuming the provider changed something — a silent model update, a routing change, a capacity-driven degradation. Five defences, and they're the answer because detection is the hard part. **Daily golden set** — run a fixed set of production-representative queries against the live model every day and compare scores against a rolling baseline; this catches regressions within 24 hours and is the single highest-value defence. **Online judge sampling** on 1–5% of real traffic, alerting on a two-point drop. **Output distribution monitoring** — length, refusal rate, formatting — because a model change usually shifts these before it shifts quality scores. **Trace tagging by model version** so you can compare cohorts. And **a canary model pin** — never let a provider auto-upgrade you; pin explicitly and test the new version yourself.

The response when it happens: confirm with the golden set, quantify the affected traffic, and if quality is materially worse, fail over to the secondary provider — which is a quality decision, not an availability one, and it needs the same failover machinery. Then negotiate with the provider with evidence, which the daily golden set gives you.

### Q22. What's the difference between infrastructure and semantic incident response?

**A.** The diagnostic path and the fix. **Infrastructure** follows the standard runbook: check error rate by provider and operation, check the breaker states, check queue depth, and the fix is usually failover, scaling, or shedding. Attribution is fast because the signals are unambiguous.

**Semantic** starts with a different question: what changed? The candidate set is the six assets plus the evaluator. So the runbook is: check whether this is a model change (compare the golden set against yesterday's), a prompt change (check the release manifest diff), an index change (check whether a re-index completed and what its embedding version was), or **a judge change** — and that fourth one is the one people miss, because a judge can drift or be swapped and the quality score moves without anything in the system changing. Then: quantify the affected traffic slice, decide whether to roll back or degrade, and communicate the affected slice to whoever owns the users.

The reason to separate them operationally is that they need different on-call skills and different dashboards. Someone paged for an infrastructure incident and handed a dashboard with no error rate on it will waste ten minutes looking for a problem that isn't there.

---

## Section E — Security and compliance operations

### Q23. What security SLIs would you run?

**A.** Three zero-tolerance ones that page immediately: **PII egress** (any PII detected in an outbound response or external call — this is block-and-page, not redact-and-log), **cross-tenant retrieval** (any retrieved chunk whose tenant doesn't match the requester's — a correctness and a compliance failure simultaneously), and **ungated high-tier action** (any T3 or T4 tool executed without the required confirmation). These are configured as critical alerts because a single occurrence is a reportable event, not a metric to trend.

Then the trending ones: injection-pattern detection rate, input guardrail block rate (a sudden change means either a new attack or a false positive storm), output guardrail block rate, rate-limit trips per principal, egress destination set changes, and the count of audit records missing required fields — which sounds like a housekeeping metric and is actually the one that predicts your next compliance finding.

### Q24. How do you respond to a prompt injection that succeeded?

**A.** Treat it as a security incident with a defined timeline. **Contain** — identify the affected capability and disable the specific tool or egress path that made it possible, not the whole feature, because a full shutdown is usually disproportionate and people won't do it. **Scope** — from the audit trail, find every session that touched the injection vector, and check whether any of them performed an action influenced by it. **Assess impact** — what data left, to where, and under what authorization; this is the step that determines whether it's a legal notification event. **Remediate the architecture** — the injection succeeded because a leg of the trifecta was reachable, so remove it; a regex addition is not remediation.

**Notify** according to policy — GDPR's 72-hour window for personal data breaches is the usual binding constraint in the EU, and contractual notification windows are often shorter. And **feed back** — the injection vector becomes a permanent test case in the red-team corpus and the guardrail regression suite, so the same attack can't work twice.

### Q25. What does an audit trail need to contain for an AI system?

**A.** Enough to reconstruct any decision without the original request. Per interaction: the `release_id`, the tenant and principal, the prompt template version, the model pin, the index snapshot identifier, the retrieved chunk IDs with their scores, whether any ACL filtering occurred, the guardrail verdicts including which check fired, the tool calls with canonicalised arguments and their risk tier, the confirmation status for gated actions, token counts including `cache_read_tokens`, and the output hash.

Two details that are easy to miss and expensive to discover later. **The reason a check didn't fire is as important as the fact that it did** — you want to be able to show that a PII scan ran and found nothing, not just that nothing was flagged. And **retention has to be class-based**: high-risk system interactions are the ones you retain longest — I'd use at least six months as the floor under the EU AI Act's expectations — while routine read-only queries can be shorter. Storage in object-lock compliance mode is the right mechanism for the irreplaceable subset, because it makes tampering structurally impossible rather than merely prohibited, which is what an auditor wants to see.

### Q26. How do you keep a compliance control from becoming stale?

**A.** By binding it to something that runs. The failure pattern is a control documented in a policy that nobody executes — it passed the audit last year and nothing has re-verified it since. So every control needs a **test** and a **log signal**: the control is real when there's an automated check that fails when the control is missing, and a log field that proves it ran on each request.

That structure also makes the regulatory mapping cheap. The same controls map onto the EU AI Act's transparency obligations, the NIST AI RMF's Govern/Map/Measure/Manage functions, SOC 2's trust criteria, and ISO/IEC 42001 — so the work is done once and the evidence is cross-referenced rather than duplicated. And the OWASP LLM Top 10 works the same way: each risk maps to a control, a test, and a log signal, and if any of those three is missing for a given risk, that's the gap in your compliance posture, found before the auditor finds it.

---

## Section F — Disaster recovery and failover

### Q27. What breaks first in an AI DR scenario?

**A.** The derived data, and specifically the assumption that you can rebuild it quickly. The classification that drives the whole DR design is three asset classes. **Rebuildable** — embeddings, the HNSW index, BM25 indexes, semantic caches, reranker caches. You don't back these up; you rebuild them, and the backup that matters is the *recipe* — the chunker configuration, the pinned embedding model version, and the normalisation steps — because without the recipe you can't reproduce the index even with the source documents.

**Restorable** — prompts, configs, release manifests, adapters, agent state, memory. These are small and versioned, so they get normal backup. **Irreplaceable** — evaluation sets, human labels, red-team corpora, adjudication records, rubrics. These cannot be regenerated at any price, and they're the only class that gets object-lock and the most expensive retention.

### Q28. Why not just back up the vector index?

**A.** Because the arithmetic doesn't justify it. A 10-million-chunk index at 1536 dimensions is about 61.4 GB of raw vectors; add the HNSW graph at roughly 1.5x and metadata, and you're at approximately 97 GB per replica. Backing that up costs in the range of $150–600 a month in storage and transfer, plus the operational complexity of keeping backups consistent with a mutating index.

Rebuilding it costs roughly 10–12 hours of compute and $60–150 in embedding calls — once. So the comparison is a recurring cost against a one-time cost plus recovery time. Given an RTO of 15 minutes for the top tier, a 10-hour rebuild obviously doesn't meet the RTO for that tier — so the design is that the top tier keeps a warm replica, and the rebuild path is for the tiers whose RTO is measured in hours. The decision isn't "back up or don't," it's "which tiers need a warm path and which can wait."

### Q29. What is the difference between failover and substitution for LLM providers?

**A.** Failover implies continuity — the same thing, running elsewhere. For a third-party model, you can't have that: when you move from one provider to another, you get a **different model with different capabilities**, different tokenisation, different tool-call formatting, and different behaviour on your prompts. So provider failover is **substitution**, a capability change, and the DR plan has to treat it as one.

That means the secondary has to be exercised continuously rather than kept as a theoretical option. My design: keep the secondary at **25% of steady-state traffic** so it's warm, tested, and its failure modes are known before you need it. And gate the substitution on a **viability check**: quality within 2 points of the primary on your golden set, schema-valid output rate at or above 0.995, and tool-call parse rate at or above 0.999. If the secondary fails those, failing over makes you available and worse, which is not obviously better than being down and honest.

### Q30. What's the trap in a failover test?

**A.** The **cold cache trap**. When you fail over, the new provider has no warm prompt cache — everything that was a cached read becomes a full-price, full-latency prefill. Measured effect: cost up roughly 91% and latency up roughly 30% relative to steady state, for a period after the switch. If your DR plan was validated by a test that ran for five minutes, you validated the switch and not the aftermath.

So the drill has to run long enough to see steady state — I'd use at least an hour of real traffic — and the capacity plan for the failover state has to account for the higher cost and latency, not the steady-state numbers. The second trap is related: the failover safety gate has to verify that guardrails, the action gate, and PII scanning are all active on the new path **before** traffic opens. Failing over into a path with no guardrails converts an availability incident into a security incident, and it's happened in production more than once.

---

## Section G — LLMOps and continuous quality

### Q31. What is LLMOps and how does it differ from MLOps?

**A.** Four differences. **The unit of change** is six assets rather than one artifact, and any of them can change behaviour. **The failure mode** is semantic — a regression produces a confident wrong answer, not an error, so monitoring has to measure quality and not just availability. **The evaluation is probabilistic** — you can't assert on exact output, so you need judges, rubrics, and statistical thresholds, which means the evaluation itself is a system with its own calibration and drift. And **the iteration loop is faster** — prompt and retrieval changes ship in hours, not the weeks a retrain-and-redeploy cycle takes, which means the release machinery has to be correspondingly light.

The practical consequence: MLOps is built around the model artifact, and LLMOps is built around the **release manifest** plus a continuous evaluation layer. If you take one thing, take that.

### Q32. How do you evaluate at production scale without going broke?

**A.** With a layered judge architecture. **Distilled judges run inline on 100% of traffic** — a small model fine-tuned on your rubric, costing around 97% less and running roughly 10x faster than a frontier judge, typically sub-100ms. They agree with the frontier judge 88–92% of the time and land within 2–3 points of human gold on the taxonomy they were trained on. **A frontier judge runs on 1–5%** of traffic — enough to measure the distilled judge's agreement and to catch what it misses. **Humans review disagreements** and a weekly sample. And the distilled judge is **retrained quarterly** on the accumulated labels.

Two caveats that matter operationally. The distilled judge is trained on a fixed taxonomy — groundedness, instruction-following, toxicity, PII, off-topic, refusal — and it **regresses to a default outside that taxonomy**, so it will confidently score a novel failure shape as fine. And the economics are the reason for the architecture: running a frontier judge on everything at a million requests a day is on the order of $600,000 a month, which is unaffordable above roughly 100,000 requests a day. The layered stack lands at roughly $33,000 a month at the same volume, which is 2–4% of inference — the number that makes it defensible.

### Q33. How do you know your judge is still trustworthy?

**A.** By measuring it as a system, not treating it as ground truth. **Judge-human agreement** — Cohen's kappa against a human-labelled sample, target **κ ≥ 0.6**, alert below **0.4**. If kappa drops, the judge has drifted or the rubric no longer matches product intent, and either way the score it produces is decoration. **Judge drift** — track the score distribution over time; a judge whose average crept from 4.2 to 4.6 without any product change has moved, not the system. **Distilled-versus-frontier agreement** — the sampling layer exists precisely to measure this. And **bias checks** — position bias (swap and re-run, keep only consistent verdicts), length bias, self-preference (use a different model family as judge than the model under test), and format bias.

The one I'd emphasise is the self-preference check, because it's the one that produces a plausible number while being wrong. A judge grading its own model family's outputs will systematically prefer them, and if you're A/B testing two models with one of them acting as judge, you've built the result you wanted.

### Q34. What's the difference between Pass^1 and Pass^k and why does it matter operationally?

**A.** Pass^1 is the ordinary single-run success rate. Pass^k is the probability of succeeding on all k independent trials. The gap between them is the reliability story, and it's invisible in a single-run metric: an agent with Pass^1 of 70% and Pass^4 of 12% works on the easy path and cannot recover from any perturbation. For anything with side effects I'd target **Pass^4 ≥ 0.60**.

Why it matters operationally: Pass^1 tells you the quality, Pass^k tells you whether you can put it in front of a customer without a human watching. A system with 70% Pass^1 and 12% Pass^4 will, over a day of traffic, leave a meaningful number of tasks half-completed — and each half-completed task is a support ticket and possibly a data-integrity problem. So Pass^k is the metric that decides whether a feature can graduate from assisted to autonomous, and the evaluation harness has to support resetting world state between trials to measure it at all.

### Q35. How do you detect quality drift in production?

**A.** With a two-condition test, because either condition alone produces false alarms. You need **both an effect size and statistical significance**: a drop of at least 10% in the measured score **and** p < 0.05 on the comparison against the rolling baseline. Effect size alone fires constantly at high volume; significance alone fires on tiny meaningless differences when you have a million samples. Requiring both gives you an alert you can actually act on.

Then the response has four candidate causes and you separate them in order: the **model** (did the provider change something — check the daily golden set), the **prompt** (check the release manifest diff), the **index** (did a re-index complete, with which embedding version), and the **judge** (did its calibration move). That fourth one is the one people forget, and it produces exactly the same symptom as a real quality drop — you need to check whether the measurement moved or the thing being measured moved.

### Q36. What's a realistic 90-day plan for standing up LLMOps?

**A.** Four phases, sequenced so each one makes the next possible. **Days 1–14: visibility.** Get `release_id` into every trace, get cost and latency attributed per request, get token counts by component, and stand up a dashboard. You cannot manage what you can't see, and this phase is entirely unglamorous and entirely necessary. **Days 15–30: manifest.** Build the release manifest, pin the six assets, make rollback a repoint. Now any change is attributable and revertible. **Days 31–60: evaluation.** A golden set of 200–500 cases, a judge calibrated against human labels, a CI gate that blocks on regression, and online sampling at 1–5% with an alert. **Days 61–90: continuous.** Drift detection with the effect-size-and-significance test, the degradation runbook, cost operations, and the governance evidence falling out of the controls you already built.

The thing I'd warn about is starting at phase three. Teams that build evaluation first, before they have trace attribution, build an eval system that can't explain *why* anything changed, and they end up with a red build and no diagnostic path.

### Q37. What attributes must exist in every trace?

**A.** Nine, and they're non-negotiable because each one answers a question you will definitely ask. `release_id` — what configuration produced this. Tenant and principal — who it was for. Prompt template version — which wording. Model pin — which model. Index snapshot — which corpus state. `cache_read_tokens` — whether caching worked, which is the first thing you check when cost moves. `acl_filtered` — whether the retrieval was scoped, which is a security-relevant fact. Guardrail verdicts — which checks ran and what they decided. And the tool calls with risk tier and confirmation status, for anything agentic.

The three people routinely omit: `cache_read_tokens` (without it a cache regression is invisible until the bill), `acl_filtered` (without it you can't prove retrieval was scoped), and the prompt template version (without it a prompt change is unattributable). Each omission is cheap to fix now and expensive to discover during an incident when you can't tell whether the problem is the model, the prompt, or the index.

---

## Section H — Pushback round

### Q38. "This is a lot of infrastructure for a feature that might not work. Why not ship first?"

**A.** Because the failure is invisible, and that changes the calculus. For a normal feature you'd ship, watch error rates, and iterate. For a generative feature, a quality regression produces confident correct-looking answers with a 200 status, so you'd ship, see green dashboards, and find out from user complaints a month later — by which point you've been confidently wrong thousands of times and have no trace to tell you when it started.

So the minimum viable version is small, not zero: `release_id` in every trace, cost and latency tracking, a golden set of a few hundred hand-labelled cases in CI as a blocking gate, a sampled judge score with an alert, and a rollback that takes ninety seconds. That's a few days of work. Everything beyond it — the layered judge stack, drift detection, Pass^k, audit retention — scales with blast radius: an internal read-only tool over public docs needs far less than a system with side effects over customer data. The one thing I wouldn't skip at any scale is the sampled quality score, because it's the only thing standing between me and shipping silently worse answers.

### Q39. "Your eval stack costs $33k a month. Isn't that excessive?"

**A.** It's 2–4% of the inference cost at that volume, and the alternative is worse. The comparison I'd make: one bad release that ships a 15% quality regression for a week at a million requests a day costs far more than the eval stack costs for a year, and that's before counting the customer trust. And the architecture is already the cheap one — a frontier judge on 100% of traffic would be roughly $600,000 a month at that volume, which is why the distilled-inline, frontier-sampled, humans-on-disagreement design exists.

The honest version of the answer is that the spend should be proportional. At 10,000 requests a day the right eval stack is a few hundred dollars a month and mostly human time. At 10 million a day the distilled judge is amortised across enough volume that it's obviously worth it. What's never right is zero, because zero means you have no idea whether your last change helped or hurt, and the only way you'll find out is from a customer.

### Q40. "Why not just fail over to the secondary provider whenever anything goes wrong?"

**A.** Because provider failover isn't failover — it's substitution, and you're substituting a different model. The secondary has different capabilities, different tokenisation, different tool-call formatting, and different behaviour on your prompts. Failing over makes you *available* and possibly *worse*, and "worse but up" isn't obviously better than "down and honest," especially if worse means confidently wrong answers on a different set of inputs.

So the failover is gated on a viability check: quality within 2 points on the golden set, schema-valid rate at or above 0.995, tool-call parse rate at or above 0.999. And the secondary runs at 25% of steady-state traffic all the time, not as a theoretical option — because a failover path you've never exercised is a hypothesis, and the failure modes of an unexercised secondary (different rate limits, different context limits, no warm cache) appear at the worst possible moment. The cold-cache effect alone is cost up around 91% and latency up around 30% after the switch, which is why the drill has to run long enough to see steady state.

### Q41. "Nobody's going to read a 200-case golden set. Why not just spot-check?"

**A.** Because spot-checking has no baseline and no statistical power, and its failures are the expensive kind. Spot-checking tells you the system looks fine today. It cannot tell you the system is 6% worse than last Tuesday, which is what a regression actually looks like. And a 200-case set with defined pass criteria turns "seems fine" into a build that fails — which is the only kind of quality gate that survives contact with a deadline.

That said, the set has to be usable or it won't be used. So: stratify it across the query types that matter rather than sampling randomly (random sampling gives you mostly easy cases and a saturated score), keep it under about 500 cases so a run takes minutes rather than hours, version it with its own review, and hold out a slice you look at rarely so the main set can be iterated on without losing the ability to detect that you've overfitted it. The failure to avoid is a set made entirely of easy cases — it scores 4.8, the gate never fires, and it provides false comfort while production quality falls.

### Q42. "Give me the one LLMOps practice you'd never skip."

**A.** Putting a release identifier in every trace and every log line. It's the cheapest thing on this list and it's the one that makes everything else possible. When quality drops, the first question is always "what changed," and without the `release_id` you can't answer it — you're reduced to guessing and then bisecting by redeploying, which takes hours and is unavailable during the incident when you need it.

With it, the answer is a query: group the quality score by release, find the release where it moved, look at the manifest diff, and roll back in ninety seconds. Every other practice on this list — the eval gate, drift detection, cost attribution, the audit trail — is downstream of being able to say which configuration produced a given output. It's one field in one struct, and it's the difference between an incident you resolve in ten minutes and one you resolve in a day.

---

## Cross-links

- [Deployment, Release and CI/CD](../Operations/04-deployment-release-and-cicd.md) — Section A in depth
- [Capacity Planning and Autoscaling](../Operations/05-capacity-planning-and-autoscaling.md) — Section B in depth
- [Cost and FinOps](../Operations/06-cost-and-finops.md) — Section C in depth
- [Reliability Patterns](../Operations/07-reliability-patterns.md) — Section D in depth
- [Security, Guardrails and Compliance](../Operations/08-security-guardrails-and-compliance.md) — Section E in depth
- [DR, Failover and Backup](../Operations/09-dr-failover-and-backup.md) — Section F in depth
- [LLMOps for AI Systems](../Operations/10-llmops-for-ai-systems.md) — Section G in depth
- [RAG Evaluation and Observability](../HLD/05-ai-systems/03-rag-evaluation-and-observability.md)
- [LLM, RAG and GenAI Questions](04-llm-rag-genai-questions.md)
- [Agentic AI Questions](05-agentic-ai-questions.md)
- [Operationalisation Cheatsheet](../Cheat-Sheets/06-operationalisation-cheatsheet.md)
