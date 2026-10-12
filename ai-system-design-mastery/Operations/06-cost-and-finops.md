# Cost and FinOps for AI Systems

**TL;DR** — AI infrastructure behaves like a **cost-of-goods business**, not like zero-marginal-cost SaaS: every request carries a variable token cost, and reported AI-product gross margins run roughly **25–30 points below** traditional SaaS. That single fact makes cost engineering, not finance, an engineering responsibility. Inference is **80–90% of total GenAI spend**, and the largest single lever is sitting unused in most stacks: **system prompts are ~69% of input tokens, yet only ~28% of calls use prompt caching**. This playbook covers the cost model as eight stackable layers, the attribution and showback discipline that makes cost debuggable, the structural levers that move cost 2–50x (caching, routing/cascading, batching, RAG-vs-long-context, distillation), the cost-per-outcome unit economics that pricing depends on, and the runbook for the alarm that actually happens — *"the bill doubled and traffic is flat."* The organising insight: **you cannot optimise what you cannot attribute, and you cannot price per outcome if you don't know your cost per outcome.**

> **60-second interview answer:** "I'd start with attribution, because the honest first answer to almost every cost question is 'I can't see it yet.' Every call gets tagged through a gateway with team, feature, tenant, model, and route — without that there's no unit economics and no way to tell which use case earns its cost. Then I decompose spend into eight layers, because cost isn't one number: the system prompt, the retrieved context, conversation history, model tier, output length, reasoning tokens, retries, and agent multi-step multiplication. The biggest lever is almost always prompt caching — the system prompt is about 69% of input tokens and only about 28% of calls cache it, so restoring caching is often a 30 to 40 percent saving for a day of work. Then right-sizing and cascading, which the literature puts at 45 to 85 percent savings at around 95 percent quality retention. Then batching the entire offline tier for roughly half off. And the discipline that makes it stick is unit economics: cost per conversation, cost per resolved ticket, and AI cost as a percentage of gross margin — because at 25 to 30 margin points below SaaS, you need to know your cost per resolution before you can price per resolution."

---

## 1. Purpose

This playbook defines how cost is measured, attributed, budgeted, and reduced in an AI system. It exists because three things are true simultaneously and each breaks a different assumption:

1. **Cost is variable per request** — so a traffic increase and a cost increase are different events and must be diagnosed separately.
2. **Cost is invisible in the response** — reasoning tokens are billed at the output rate and never appear in the output. A "short" answer can cost 10x a long one.
3. **Cost multiplies in agents and reasoning models** — the unit of measurement is **cost per task**, not cost per call.

**In scope:** cost modelling, attribution, budgets and alerts, the structural and tactical reduction levers, unit economics and margin, and the cost-spike runbook.

**Out of scope:** capacity sizing (see [capacity planning](05-capacity-planning-and-autoscaling.md)) and the inference-engine internals of quantisation, batching and speculative decoding (see [LLM serving](../HLD/05-ai-systems/05-llm-serving-inference-and-gateway.md)).

---

## 2. Scope and units

### 2.1 The units that matter

| Unit | Use | Why it matters |
|---|---|---|
| **Cost per 1M tokens** (in/out) | Provider comparison | The quoted number; output is 3–5x input |
| **Cost per request** | Feature-level budgeting | Comparable across features |
| **Cost per conversation** | Product-level | The chat unit |
| **Cost per task / per run** | **Agents** | Multiplies per step — the only correct unit |
| **Cost per resolved outcome** | **The business unit** | Pricing depends on it |
| **AI cost as % of revenue** | Margin | The CFO's number |
| **Cost per 1k tokens served** | Infrastructure efficiency | Self-hosted comparisons |
| **Tokens per dollar** | The inverse, used for trend | Easier to see improvement |

**The rule:** *"I report cost per request to engineers, cost per conversation to product, cost per resolved outcome to the business, and cost as a percentage of gross margin to finance. Same data, four lenses. If I only had one, it would be cost per resolved outcome, because that is the only one that connects to pricing."*

### 2.2 The anchor facts

| Fact | Value | Implication |
|---|---|---|
| System prompt share of input tokens | **~69%** | The static prefix dominates the bill |
| Calls that use prompt caching | **~28%** | The biggest unused lever in production |
| Inference share of GenAI spend | **80–90%** | Optimise inference, not training |
| Output vs input price ratio | **3–5x** | Output is the expensive direction |
| AI gross margin vs SaaS | **−25 to −30 points** | Variable cost of goods |
| Reasoning token multiplier | **3–15x** by task | Invisible in the response |
| Batch API discount | **~50%** | For anything nothing is waiting on |
| Reported agentic task cost | **$0.10 – $3.00** | vs cents for a chat turn |

---

## 3. SLIs and SLOs

### 3.1 Cost SLIs

| SLI | Definition | Target | Alert |
|---|---|---|---|
| **Cost per request** | Total spend / requests | ≤ $0.012 | > $0.020 for 30 m |
| **Cost per resolved outcome** | Spend / successful outcomes | ≤ 25% of the human alternative | > 40% |
| **Cache-hit rate** | Cached input tokens / cacheable input | **≥ 0.70** | < 0.50 for 1 h |
| **Batch share of offline work** | Batched / total offline calls | ≥ 0.90 | < 0.70 |
| **Routing efficiency** | Calls to the cheapest sufficient tier / total | ≥ 0.70 | < 0.55 |
| **Retry multiplier** | Total calls / successful first-attempt calls | ≤ 1.15 | > 1.30 |
| **Reasoning-token share** | Reasoning tokens / total output tokens | ≤ 0.30 | > 0.50 |
| **Tag coverage** | Calls with complete attribution tags | **1.00** | < 0.99 |
| **Budget variance** | Actual / budgeted daily spend | 0.9–1.1 | > 1.15 |
| **Cost per 1k requests trend** | 30-day rolling direction | Flat or falling | Rising 3 weeks running |

### 3.2 The margin SLO — the one that matters to the business

```
   Gross margin = (revenue − COGS) / revenue

   For an AI product:
     COGS = inference + infrastructure + human-in-the-loop + error cost

   Target: AI gross margin ≥ 60%
     (traditional SaaS is 80-90%; AI products report 25-30 points below)

   Worked example — a support product priced at $2.00/resolved ticket:
     Inference            $0.11
     Infrastructure       $0.03
     Human escalation     $0.42   (30% of tickets × $1.40 human cost share)
     Error cost           $0.09
     ───────────────────────────
     COGS                 $0.65
     Gross margin         67.5%   ✓

   Now the same product at $1.20/resolved ticket:
     Gross margin         45.8%   ✗ below target

   THIS is why cost per resolution must be known before pricing
   per resolution. Outcome-based pricing fails silently on margin.
```

**The insight to state:** *"Outcome-based pricing is rising because it aligns incentives, but it's a trap if you don't know your cost per outcome. You price at $2 a resolved ticket, feel good about the 67% margin, and then discover that the escalation rate drifted from 20% to 35% and your COGS went to $0.82 and your margin is 59%. Cost per outcome is not a reporting metric, it's a pricing input."*

---

## 4. Architecture — the cost model and the attribution path

### 4.1 The eight-layer cost model

Every request decomposes into these layers. **Modelling them separately is what makes cost predictable and debuggable.**

```
   cost = Σ(layer_input_tokens × in_rate)
        + (output_tokens + reasoning_tokens) × out_rate

   cost_per_task = cost_per_request × expected_steps × (1 + retry_rate)
```

| # | Layer | Driven by | Share of bill | Primary lever |
|---|---|---|---|---|
| 1 | **System prompt** | Instructions, tool defs, few-shot | **~69% of input** | **Prompt caching** |
| 2 | **Retrieved context** | RAG chunks, injected docs | 5–40% | Chunk budgeting, k tuning |
| 3 | **Conversation / memory** | History, scratchpad | 5–30% and growing | Windowing, summarisation |
| 4 | **Model tier** | Frontier vs mid vs small | **10–100x spread** | Right-sizing, cascade, route |
| 5 | **Output length** | Verbosity, format | 10–40% | `max_tokens`, terse contracts |
| 6 | **Reasoning tokens** | Extended thinking | **3–15x multiplier** | Gate by task complexity |
| 7 | **Retry / overhead** | Transient errors, guardrail re-runs | 0–30% | Bounded retries, breakers |
| 8 | **Agent multi-step** | Plan-act-observe loops | **Multiplies 1–7 per step** | Step ceilings, per-run budgets |

**Layers 6 and 8 are the two worst, and the reason is the same: they are invisible.** Reasoning tokens are billed at the output rate and do not appear in the response. Agent steps multiply the entire stack. So the two monitoring gaps are reasoning-token share and cost per task — and both are missing from most dashboards.

### 4.2 The attribution path

```
═════════════════ REQUEST PATH WITH TAGS ═════════════════

  ┌──────────┐
  │  Client  │  tags: tenant_id, feature, user_id
  └────┬─────┘
       ▼
  ┌───────────────────────────────────────────────┐
  │  AI GATEWAY / TOKEN PROXY                     │
  │  ── the single point where cost is measured ──│
  │                                               │
  │  1. Resolve identity  → team, tenant, env     │
  │  2. Classify route    → tier, model, reason   │
  │  3. Count tokens      → in, out, cached,      │
  │                         reasoning             │
  │  4. Apply price table → per model, per version│
  │  5. Emit metric       → llm_cost_usd{...}     │
  │  6. Enforce budget    → per-key spend cap     │
  └───────┬───────────────────────────────────────┘
          ▼
  ┌───────────────────────────────────────────────┐
  │  PROVIDER(S)                                  │
  └───────┬───────────────────────────────────────┘
          ▼
  ┌───────────────────────────────────────────────┐
  │  COST STORE (time-series)                     │
  │  dimensions:                                  │
  │    team · feature · tenant · model · route    │
  │    env · release_id · step_index · cache_state│
  └───────┬───────────────────────────────────────┘
          ▼
   ┌──────────────┬──────────────┬───────────────┐
   │  SHOWBACK    │  BUDGETS     │  UNIT ECON    │
   │  dashboards  │  + alerts    │  per outcome  │
   └──────────────┴──────────────┴───────────────┘
```

**The five tag dimensions, and why each exists:**

| Tag | Answers |
|---|---|
| `team` | Who pays? (chargeback) |
| `feature` | Which product surface? (kill/keep decisions) |
| `tenant` | Which customer? (per-customer margin) |
| `model` + `route` | Which tier? (routing effectiveness) |
| `release_id` | Which deploy caused it? (attribution to change) |

**Plus two that are optional but high-value:**

| Tag | Answers |
|---|---|
| `step_index` | Which agent step is expensive? (usually step 1 and the final synthesis) |
| `cache_state` | Why didn't this hit cache? (`miss_cold`, `miss_prefix_changed`, `no_cache`) |

**The `cache_state` tag is the one nobody adds and everybody needs.** Without it, a cache-hit-rate drop is a number with no explanation. With it, you immediately see whether the misses are cold-start (fine), prefix-changed (a bug), or never-attempted (a config error).

---

## 5. Standard operating procedures

### SOP-1 — Establishing cost visibility from zero

```
STEP 1  Install the gateway / token proxy
        □ Every LLM call routes through it. No exceptions, including
          notebooks and internal tools — those are where surprise
          spend hides.
        □ Block direct provider SDK calls at the network layer if
          you can. A single un-proxied service is a 30% blind spot.

STEP 2  Build the price table
        □ Per model, per version, in/out/cached rates
        □ Include the batch rate and the provisioned rate
        □ Version it — prices change, and historical cost must be
          recomputed with the price that applied AT THE TIME
        □ Verify quarterly. Prices have fallen 3-5x/year in some tiers.

STEP 3  Emit per-call cost
        llm_cost_usd{micros} with all dimensions
        llm_tokens{micros} split: {input, cached_input, output, reasoning}
        llm_cache_state{state}

STEP 4  Build three dashboards
        □ Spend by team/feature (the showback view)
        □ Spend by model/route (the efficiency view)
        □ Spend per request and per outcome (the unit-economics view)

STEP 5  Set budgets and alerts BEFORE the first spike
        □ Per-tenant daily cap
        □ Per-feature daily cap
        □ Global daily cap with a hard enforcement action
        □ Spike alert at 2x the 30-day median hourly

STEP 6  Establish unit economics
        □ Cost per request, conversation, task, resolved outcome
        □ AI cost as % of revenue and of gross margin
        □ Publish monthly, to the same audience that sees revenue
```

**The step everyone skips is step 5's enforcement action.** *"An alert that says 'spend is high' at 3 a.m. does nothing. A budget needs a declared action: degrade to a smaller model, shed non-critical traffic, or hard-stop. Decide it before the spike, because during the spike you will be arguing about it while the meter runs."*

### SOP-2 — The cost-spike runbook ("the bill doubled, traffic is flat")

This is the alarm that actually fires.

```
T+0    CONFIRM AND SEGMENT (first 10 minutes)
       □ Is it real or a pricing/reporting artefact?
         → check for a provider price change or a price-table update
       □ Break by dimension. Which one moved?
         team · feature · tenant · model · route · release_id
       □ Traffic vs cost-per-request: which one actually changed?
         cost = traffic × cost/request. If traffic is flat,
         cost-per-request doubled. That is the whole diagnosis.

T+10m  IF COST-PER-REQUEST DOUBLED, CHECK IN THIS ORDER
       (empirically the most common causes, highest first)

       1. CACHE-HIT RATE COLLAPSE                    ~40% of cases
          □ cache_hit_rate by cache_state
          □ "miss_prefix_changed" → something moved into or within
            the cached prefix (a timestamp, a request ID, reordered
            tool definitions, a per-user value at the front)
          □ Check git on the prompt bundle and the release_id marker
          □ FIX: move the dynamic value to the end of the prompt

       2. REASONING TOKENS ENABLED                   ~15% of cases
          □ reasoning_tokens / output_tokens ratio
          □ Did a config default change? A model upgrade that
            enables thinking by default?
          □ FIX: gate thinking by task complexity; add an explicit
            "no thinking" path for classification and extraction

       3. RETRY MULTIPLIER                           ~12% of cases
          □ total_calls / successful_calls
          □ A provider degradation causing silent retries
          □ FIX: circuit breaker; check the provider status page

       4. OUTPUT LENGTH GROWTH                       ~10% of cases
          □ mean_output_tokens trend
          □ A model version change that got more verbose, or a
            prompt change that removed a length constraint
          □ FIX: restore max_tokens; add a terse-output contract

       5. CONTEXT / RETRIEVAL GROWTH                 ~8% of cases
          □ mean_input_tokens minus cached_input
          □ k changed? chunk size changed? a new doc type with
            much larger chunks got ingested?
          □ FIX: re-tune k; re-check chunk budgets

       6. AGENT STEP CREEP                           ~7% of cases
          □ mean_steps_per_task trend
          □ The agent is looping more — usually a tool that started
            failing, or a retrieval recall drop
          □ FIX: fix the tool; check the no-progress detector

       7. ROUTING REGRESSION                         ~5% of cases
          □ share of traffic by tier
          □ The classifier degraded, or a fallback is firing constantly
            and routing everything to the frontier tier
          □ FIX: check fallback rate; re-eval the classifier

       8. TENANT / FEATURE MIX SHIFT                 ~3% of cases
          □ Not a bug. A new large tenant, or a feature launch
          □ FIX: budget allocation, not engineering

T+30m  MITIGATE
       □ Apply the fix for the identified cause
       □ If the cause is unknown: enable the cost degradation ladder
         (SOP-6) to bound the bleeding while you investigate

T+1d   PREVENT
       □ Add an alert for the specific signal that would have caught
         it earlier
       □ Add a CI check if it was a release-introduced regression
         (e.g. assert cache hit rate in the canary gate)
```

**The single most useful table in this playbook** is the ordered list in T+10m, because it encodes the empirical base rate. **Presenting that ordering in an interview is a strong signal** — it shows you have debugged this for real rather than read about it.

### SOP-3 — Capturing prompt caching (the top lever)

```
STEP 1  RESTRUCTURE THE PROMPT: STATIC FIRST, DYNAMIC LAST

   ✗ WRONG (nothing caches):
     [user query] [system prompt] [tools] [retrieved docs]

   ✗ WRONG (caches, but breaks constantly):
     [system prompt with today's date and request_id]
     [tools] [user query]

   ✓ RIGHT (stable prefix, dynamic suffix):
     ┌─────────────────────────────────────┐
     │ CACHED PREFIX                        │
     │  system instructions                 │  ← never changes
     │  tool definitions (pinned order)     │  ← never changes
     │  few-shot examples                   │  ← rarely changes
     ├─────────────────────────────────────┤
     │ SEMI-STATIC (cache per doc set)      │
     │  retrieved chunks                    │  ← changes per query,
     │                                        but repeats across users
     ├─────────────────────────────────────┤
     │ DYNAMIC (never cached)               │
     │  conversation turn                   │
     │  user query                          │
     └─────────────────────────────────────┘

STEP 2  PURGE THE PREFIX OF VOLATILE CONTENT
        □ No timestamps · no request IDs · no UUIDs · no nonces
        □ No per-user values (name, tenant) — move them to the end
        □ Pin tool-definition ordering — a dict iteration order
          change silently invalidates everything
        □ No "current date" unless it is a coarsened value like
          "2026-Q3" that changes quarterly
        □ No random few-shot selection — use a fixed set

STEP 3  SIZE THE PREFIX FOR THE CACHE MINIMUM
        □ Provider minimums are commonly 1,024-2,048 tokens
        □ Below the minimum, nothing caches — measure and confirm
        □ A cache hit rate of 0 on a 900-token prefix is expected,
          not a bug

STEP 4  CONSIDER THE WRITE PREMIUM
        □ Some providers charge ~25% extra to WRITE a cache entry
        □ Break-even is roughly 2 reads
        □ Bursty low-reuse traffic may not benefit — check before
          enabling everywhere

STEP 5  INSTRUMENT AND DEFEND
        □ cache_hit_rate as a first-class SLI with an alert at < 0.50
        □ cache_state breakdown (cold / prefix_changed / not_attempted)
        □ A CI assertion in the shadow stage: hit rate must not drop
          more than 10 points vs the current release
```

**The discount structure to state precisely:** *"Provider prefix caching gives a steep discount on the repeated prefix — roughly 50% from OpenAI where it's automatic above a threshold with no write fee, up to about 90% from Anthropic with explicit cache breakpoints and a small write premium, and around 75% from Google. The critical caveat, which is where people misquote this, is that the 50-to-95 percent number applies to the cached prefix only, not to the whole bill. If your prefix is 69% of input tokens, a 90% discount on that prefix is a 62% reduction in input cost — which is enormous, but it is not 90% off the bill."*

### SOP-4 — Model routing and cascading

```
   CASCADE TIERS

   Tier 0  Classifier (0.5-1B)              ~$0.00 per call
           complexity, intent, language
           ↓ every request
   Tier 1  Small (3-8B)                     ~$0.0002 per call
           70-90% of traffic: greetings, entity extraction,
           simple RAG, classification, formatting
           ↓
   Tier 2  Mid (frontier-class mid tier)    ~$0.004 per call
           8-25%: multi-hop reasoning, synthesis, long context
           ↓
   Tier 3  Frontier / reasoning             ~$0.03-0.15 per call
           1-5%: expert-level, agentic planning, hard code
           ↓
   Escalation rule: escalate on LOW CONFIDENCE, not on failure.
   A wrong Tier-1 answer that is confidently wrong never escalates.

   Reported result: 45-85% cost reduction at ~95% quality retention
   (FrugalGPT is the canonical reference). The live variable is the
   ESCALATION RATE — track it as a first-class metric.
```

| Routing signal | Reliability | Cost |
|---|---|---|
| Query length | Weak | Free |
| Keyword / regex | Weak but predictable | Free |
| Classifier model | Good | ~$0.00001 |
| Confidence of the first answer | **Best** | One extra call |
| Self-consistency disagreement | Very good | 3–5x the small model |
| Learned router (trained on outcomes) | Best at scale | Training cost |

**The failure mode to warn about:** *"Cascading fails when the escalation rule is wrong. If you escalate on failure, you've already served the bad answer. If you escalate on confidence, you need a calibrated confidence signal — and small models are frequently overconfident. So the strongest cheap version is self-consistency: run the small model twice at temperature 0.7, and escalate if the answers disagree. It costs 2x on the small tier, which is nothing, and it catches the case where the small model is confidently wrong."*

**The break-even arithmetic for self-hosting:**

```
   Self-host break-even (reported): high tens to hundreds of
   millions of tokens per month, or ~500M tokens/month for a
   70B-tier model.

   THE WARNING: raw GPU rental is only 30-40% of true cost.
     GPU rental         $1.00  (per unit)
     Networking         $0.15
     Storage            $0.10
     Observability      $0.08
     Engineering (1-2 FTE)$0.90
     ─────────────────────────
     True cost          ~$2.50-3.00   → apply a 2.5-3x multiplier

   So the raw break-even number is off by ~3x. Most teams that
   "save money" self-hosting are not counting the engineers.
```

### SOP-5 — Batch and reserved capacity

```
DECISION TREE

   Is a human or a system waiting on this token?
   │
   ├── YES → synchronous API
   │         └── latency-critical? → provisioned throughput
   │
   └── NO  → BATCH API (~50% off, 24 h completion ceiling)
             ├── evals and regression suites
             ├── bulk classification and labelling
             ├── corpus-scale summarisation
             ├── backfills after a prompt or model change
             ├── embedding generation for index builds
             └── A/B testing prompt variants

   Is utilisation high AND steady?
   │
   ├── YES → provisioned / reserved throughput
   │         reported 15-70% savings; pay hourly regardless of use
   │         □ Reserve only the FLOOR, never the peak
   │         □ Break-even is high and predictable utilization
   │
   └── NO  → on-demand (including batch)
             pay-for-silence beats a reservation at low duty cycle

   Spot instances (self-hosted only):
     □ 30-second reclamation notice
     □ Mitigate with live KV-cache migration to another node
     □ Only for non-real-time work — never for interactive traffic
```

**The one-line rule:** *"If nothing is waiting on the token, batch it. That's half off, and for the entire offline tier of a product, not batching leaves about half the money on the table."*

### SOP-6 — The cost degradation ladder

Shedding for cost is different from shedding for capacity, and it must never touch quality-critical paths without sign-off.

```
   COST LADDER (in order)

   1. Pause all bulk P3 work                    invisible
   2. Move remaining offline work to batch      invisible (slower)
   3. Disable reasoning tokens on Tier-0/1 tasks invisible for simple tasks
   4. Cap max_tokens 1024 → 512                 shorter answers
   5. Reduce retrieval k 50 → 20                minor quality
   6. Shorten the system prompt (drop few-shot) slight quality dip
   7. Route Tier-1 traffic down a model tier    quality dip
   8. Disable speculative/proactive features    visible
   9. Hard-stop new non-critical sessions       visible, last resort

   NEVER in the cost ladder:
     ✗ Reduce guardrail depth
     ✗ Skip the PII pass
     ✗ Disable the action gate
     ✗ Drop the eval sampling that feeds quality monitoring
       (this is the one people forget — turning off eval sampling
        to save money blinds you exactly when you need to see)
```

### SOP-7 — Charging back to teams

```
STAGE 1  SHOWBACK (months 1-3)
   □ Dashboards only. No invoices. No blame.
   □ Goal: get tags to 100% coverage and make spend legible.
   □ Publish a weekly top-10 spender list by feature.
   □ Expect this to change behaviour on its own — it usually does.

STAGE 2  SOFT CHARGEBACK (months 4-9)
   □ Monthly report to each team showing their spend
   □ Budgets agreed, not imposed
   □ Over-budget triggers a conversation, not a bill

STAGE 3  HARD CHARGEBACK (month 10+)
   □ Only after tag coverage is 100% and stable for two quarters
   □ Only after teams agree the numbers are correct
   □ Requires a dispute process — there will be disputes
   □ Requires forecasts per team

   THE RULE: never jump to stage 3. Charging back on untrustworthy
   numbers destroys the credibility of the whole programme, and you
   will not get it back.
```

**Why showback first:** *"The moment you bill a team, the conversation becomes about the accuracy of the meter instead of the efficiency of the system. Get the meter right and let visibility do the work first — in my experience a weekly top-spender list changes behaviour without anyone being charged a cent."*

---

## 6. Failure modes and mitigations

| # | Anti-pattern | Mechanism | Fix |
|---|---|---|---|
| 1 | **No caching** | Re-paying for the ~69% static prefix every call | Stable prefix + provider caching |
| 2 | **Prefix pollution** | A timestamp or request ID in the cached region | Static-first ordering; purge volatiles |
| 3 | **Oversized model** | Frontier tier on tasks a small model handles | Right-size, cascade, route |
| 4 | **Unbounded output** | No `max_tokens`; verbose formats | Caps + terse output contracts |
| 5 | **Reasoning on by default** | Extended thinking for trivial tasks | Gate thinking by complexity |
| 6 | **Retry storms** | Transient errors trigger unbounded retries | Bounded retries + circuit breakers |
| 7 | **Runaway agent loops** | A plan-act loop that never terminates | Hard step/token/cost ceilings **inside** the loop |
| 8 | **Unbounded memory** | History accrues without summarisation | Windowing + summarisation |
| 9 | **Long-context stuffing** | A giant context as the default | RAG for large or changing corpora |
| 10 | **No attribution** | Untagged shared spend | Gateway proxy + tags |
| 11 | **Real-time for offline work** | Sync API for evals and backfills | Batch API |
| 12 | **Cache key without `release_id`** | Post-deploy cold-cache storm attributed to a bug | Include `release_id`; pre-warm top queries |
| 13 | **Semantic-cache false hits** | Similar-but-different query served a wrong answer | Threshold ≥ 0.93; validate on a labelled set |
| 14 | **Price table staleness** | Cost reports drift from reality | Version and verify quarterly |
| 15 | **Unproxied services** | A notebook or internal tool bypasses the gateway | Block direct SDK calls at the network layer |
| 16 | **Evaluation blind-spot saving** | Disabling eval sampling to cut cost | Treat eval spend as fixed, protected budget |
| 17 | **Tenant cost invisibility** | One tenant is 60% of spend and nobody knows | Per-tenant tagging and a margin report |
| 18 | **Cost per call for agents** | Measuring the wrong unit | **Cost per task**, always |

### 6.1 Runaway agent costs — expanded

*"Reported real incidents have runaway agents burning tens of thousands of dollars over a single weekend before anyone noticed."* The mechanism is specific:

```
   HOW A $40,000 WEEKEND HAPPENS

   1. A tool starts returning an error whose message reads like a
      retryable condition ("rate limited", "try again").
   2. The agent's error contract treats it as retryable.
   3. The loop retries, gets the same error, retries again.
   4. Each retry re-sends the FULL context — 20k tokens and growing,
      because the scratchpad appends the error each time.
   5. Steps 1-4 repeat for every task in the queue.
   6. Cost per task goes from $0.30 to $40.

   THE DEFENCE IS INSIDE THE LOOP, NOT AFTER IT:
     □ hard step ceiling (10)
     □ hard token ceiling (100k)
     □ hard cost ceiling ($0.50) — enforced per task, in the loop
     □ hard wall-clock ceiling (120 s)
     □ duplicate-call detector: same (tool, args) twice → stop
     □ no-progress detector: same state hash 3x → stop
     □ per-tool retry cap: 2, then move on

   AN ALERT IS NOT A CEILING. By the time the alert fires and a
   human reads it, the money is spent. The ceiling is what stops it.
```

**Say this in an interview and it lands:** *"An alert is a notification, not a control. For anything that can spend money in a loop, the control has to be a counter inside the loop that refuses to continue. Everything else is forensics."*

---

## 7. Capacity and cost

### 7.1 A worked cost model — a RAG product at 800k requests/day

| Layer | Tokens/req | Rate | Cost/req | Share |
|---|---|---|---|---|
| System prompt (cached 90%) | 3,000 | $3.00/M in | $0.00360 | 30% |
| System prompt (uncached 10%) | 300 | $3.00/M in | $0.00090 | 7% |
| Retrieved context (5 chunks) | 2,200 | $3.00/M in | $0.00660 | 55% |
| Conversation history | 400 | $3.00/M in | $0.00120 | 10% |
| Output (180 tokens) | 180 | $15.00/M out | $0.00270 | — |
| **Subtotal** | | | **$0.01500** | |
| Retry multiplier 1.12 | | | $0.00180 | |
| **Cost per request** | | | **$0.0168** | |
| Rerank | | | $0.00120 | |
| Guardrails (input + output) | | | $0.00040 | |
| **Total per request** | | | **$0.0184** | |

**Daily and monthly:**

| Item | Daily | Monthly |
|---|---|---|
| Generation | $11,760 | $352,800 |
| Rerank | $960 | $28,800 |
| Guardrails | $320 | $9,600 |
| Embedding (query) | $80 | $2,400 |
| Ingestion (amortised) | $750 | $22,500 |
| Infrastructure (compute, storage) | $420 | $12,600 |
| **Total** | **$14,290** | **$428,700** |

**Now the optimisation sequence, in the order I would do it:**

| Step | Lever | Monthly saving | Effort | New total |
|---|---|---|---|---|
| 0 | Baseline | — | — | $428,700 |
| 1 | **Prompt caching to 90%** | −$95,000 | 1 day | $333,700 |
| 2 | **Cap `max_tokens` at 256** | −$14,000 | 2 hours | $319,700 |
| 3 | **Trim few-shot from 5 to 2 examples** | −$22,000 | 4 hours | $297,700 |
| 4 | **Batch the offline eval tier** | −$8,000 | 1 day | $289,700 |
| 5 | **Cascade: route 55% of lookups to a small model** | −$64,000 | 2 weeks | $225,700 |
| 6 | **Semantic cache at 25% hit rate** | −$44,000 | 1 week | $181,700 |
| 7 | **Reduce k from 8 to 5 with rerank** | −$19,000 | 3 days | $162,700 |
| 8 | **Distil the router's narrow task** | −$12,000 | 4 weeks | $150,700 |

**Cumulative: a 65% reduction, $278,000 a month, with step 1 alone delivering 34% of the total for one day of work.** That ordering — cheapest effort first, biggest lever first — is the answer to "how would you reduce this bill."

### 7.2 Structural levers ranked by impact

| Lever | Reported effect | Effort | Risk |
|---|---|---|---|
| **Prompt caching** | 30–62% of input cost | Hours | None |
| **Model routing / cascade** | **45–85%** at ~95% quality | Weeks | Quality regression |
| **Batch API (offline tier)** | ~50% on that tier | Days | None (latency only) |
| **Distillation** | **5–40x** per-token | Months | Narrow tasks only |
| **RAG vs long context** | 10–50x per query | Weeks | Recall risk |
| **Output/prompt engineering** | 20–40% of tokens | Days | Slight quality |
| **Quantisation (self-host)** | 2–4x throughput | Days | Small quality loss |
| **Provisioned throughput** | 15–70% at steady use | Weeks | Commitment risk |
| **Self-hosting** | Variable; needs 3x cost honesty | Months | Ops burden |
| **Speculative decoding** | 2–3x tok/s | Days | Complexity |

---

## 8. Metrics and alerts

### 8.1 Alerts

```yaml
cost_alerts:
  # ── Spend ────────────────────────────────────────────────
  - name: daily_spend_vs_budget
    condition: daily_spend > 1.15 * daily_budget
    severity: warning
    action: notify_finops_owner

  - name: daily_spend_hard_cap
    condition: daily_spend > 1.50 * daily_budget
    severity: critical
    action: enable_cost_ladder_step_1_2, page

  - name: hourly_spend_spike
    condition: hourly_spend > 2 * 30d_median_hourly
    window: 1h
    severity: warning
    action: notify_owner
    note: run SOP-2

  - name: tenant_spend_spike
    condition: tenant_daily_spend > 2.5 * tenant_30d_median
    severity: warning
    note: a new tenant or a runaway integration

  # ── Efficiency (the leading indicators) ──────────────────
  - name: cache_hit_rate_drop
    condition: cache_hit_rate < 0.50
    window: 1h
    severity: warning
    action: check cache_state breakdown
    note: EXPECTED for ~30 min after a release (cold cache);
          alert only if it persists past the warm-up window

  - name: cache_prefix_invalidation
    condition: cache_state{state="prefix_changed"} > 0.20 of misses
    window: 30m
    severity: critical
    note: something volatile moved into the cached prefix

  - name: reasoning_token_share
    condition: reasoning_tokens / output_tokens > 0.50
    window: 1h
    severity: warning
    note: thinking is on where it probably should not be

  - name: retry_multiplier
    condition: total_calls / successful_first_attempt > 1.30
    window: 15m
    severity: warning
    note: often a provider degradation — check their status page

  - name: output_token_growth
    condition: mean_output_tokens > 1.25 * 30d_median
    window: 24h
    severity: warning
    note: the verbosity regression — often a model change

  - name: routing_tier_drift
    condition: share_of_traffic_on_frontier_tier > 1.5 * 30d_median
    window: 1h
    severity: warning
    note: the classifier degraded or a fallback is stuck open

  - name: agent_step_creep
    condition: mean_steps_per_task > 1.30 * 30d_median
    window: 24h
    severity: warning
    note: usually a tool failing or a retrieval recall drop

  - name: cost_per_task_p99
    condition: p99_cost_per_task > 5.00
    severity: warning
    action: check for loops

  # ── Integrity ────────────────────────────────────────────
  - name: untagged_calls
    condition: calls_without_full_tags > 0
    severity: critical
    note: an attribution blind spot; find the unproxied client

  - name: price_table_staleness
    condition: price_table_age_days > 90
    severity: warning

  - name: spend_without_release_id
    condition: cost_by_release{release_id=""} > 0
    severity: warning

  # ── Unit economics ───────────────────────────────────────
  - name: cost_per_outcome_breach
    condition: cost_per_resolved_outcome > 0.25 * human_alternative_cost
    window: 24h
    severity: critical
    note: the margin SLO is at risk

  - name: gross_margin_erosion
    condition: ai_gross_margin < 0.60
    window: 7d
    severity: critical
    action: review pricing and COGS together
```

### 8.2 Dashboards

**Dashboard 1 — Spend (daily)**

| Panel | Content |
|---|---|
| Total spend | Today vs budget vs same-day-last-week |
| Spend by feature | Stacked, top 10 |
| Spend by model/tier | Shows routing effectiveness |
| Cost per request | Trend with release markers |
| **Cache hit rate** | With the 0.70 target line |
| Anomaly | Spend vs a 30-day seasonal forecast |

**Dashboard 2 — Efficiency (weekly)**

| Panel | Content |
|---|---|
| Layer decomposition | The eight layers as a stacked bar, per week |
| Tokens per request | in / cached / out / reasoning, split |
| Routing mix | Share by tier over time |
| Retry multiplier | Trend |
| Batch share | Offline work batched vs sync |
| Top 10 costly requests | The p99 tail, with `trace_id`s to inspect |

**Dashboard 3 — Unit economics (monthly, for the business)**

| Panel | Content |
|---|---|
| Cost per request / conversation / task / outcome | Trend |
| AI cost as % of revenue | Trend |
| AI cost as % of gross margin | Trend vs the 60% target |
| Per-tenant margin | Waterfall, sorted |
| Human-escalation share | The hidden COGS driver |
| Error cost | Refunds, credits, corrections |

**The `Top 10 costly requests` panel is the one that finds bugs.** *"Aggregate metrics hide the tail. When I want to know why the bill moved, I sort by request cost descending and read the traces. Nine times out of ten the answer is one feature, one loop, or one tenant. I have never once found it by staring at a monthly total."*

---

## 9. Change management

| Change | Owner | Review | Notice |
|---|---|---|---|
| Price table update | FinOps | Platform | Immediate, versioned |
| Per-feature budget | Feature owner + FinOps | FinOps | 1 week |
| Global budget cap | FinOps + eng lead | — | 1 week |
| Cost ladder enablement | Platform (automatic) | Pre-approved | — |
| Cost ladder step reorder | Platform + product | Product | 1 week |
| Routing thresholds | ML owner | Eval owner | 1 week |
| Reservation / commitment | FinOps + eng lead | Finance | **1 quarter** |
| Cache configuration | Platform | — | 1 day |
| Chargeback stage change | FinOps + finance | All teams | 1 quarter |

**The commitment change is the one with real risk.** *"A provisioned-throughput commitment is a bet on your own forecast. Sign it for the floor, never the peak, and always model the downside — if traffic halves, the commitment is a fixed cost against halved revenue."*

---

## 10. Audit and compliance

| Record | Contents | Retention |
|---|---|---|
| Per-call cost record | Tags, tokens, model, route, computed cost | 2 years |
| Price table versions | Every rate change with an effective date | 7 years |
| Budget definitions | Per feature/tenant, with owners | 3 years |
| Cost ladder activations | When, why, which steps, who was notified | 3 years |
| Chargeback statements | Per-team monthly, with the method | 7 years |
| Spend cap breaches | Provider, amount, cause, action | 7 years |
| Commitments | Reserved capacity, term, utilisation | 7 years |
| Cost per outcome | Monthly, with the outcome definition | 3 years |

**The compliance-relevant point:** *"Cost records are also audit records. If you bill a customer per resolved ticket, the per-call cost record plus the outcome definition is your evidence for the unit economics you quoted. And if a regulator asks how you priced something, 'we estimated' is not an answer."*

---

## 11. Interview Q&A

**Q1. Your LLM bill doubled month over month with flat traffic. How do you find and fix it?**
First, attribution — if every call isn't tagged by feature, team, model, and route through a gateway, that's the actual first fix, because you can't debug what you can't see. With attribution, cost is traffic times cost-per-request, so flat traffic means cost-per-request doubled, and that narrows it enormously. Then I'd check in order of empirical frequency. Cache-hit rate collapse is the biggest one, around 40% of cases — the system prompt is about 69% of input tokens, so a cache regression is huge, and it's usually something volatile that moved into the cached prefix, like a timestamp or a reordered tool definition. Then reasoning tokens silently enabled, which is billed at output rate and invisible in the response. Then retry multiplier, output length growth, context growth, agent step creep, and routing regression. The highest-ROI fix is almost always restoring prompt caching, then right-sizing and capping output. And I'd move any offline work to the batch API for roughly half off and set per-feature budgets so the next doubling pages someone on day one.

**Q2. Why is prompt caching such a big deal?**
Because of the shape of the traffic. Datadog's 2026 data puts system prompts at about 69% of input tokens, and only about 28% of calls using caching. So the largest, most static, most cacheable chunk of the bill is going uncached in most production stacks. The discounts are large — roughly 50% from OpenAI where it's automatic above a threshold, up to about 90% from Anthropic with explicit breakpoints, around 75% from Google. The caveat I always state is that those percentages apply to the cached prefix only, not the whole bill. If the prefix is 69% of input and you get a 90% discount on it, that's a 62% reduction in input cost. Enormous, but not 90% off the bill.

**Q3. How do you actually make caching work?**
Four things. Order the prompt static-first, dynamic-last, because prefix caching matches on the exact prefix and anything that changes near the front invalidates everything downstream. Purge volatiles from the prefix — no timestamps, request IDs, UUIDs, per-user values, and pin the tool-definition ordering, because a dict iteration order change silently kills the whole cache. Size the prefix above the provider minimum, which is commonly 1,024 to 2,048 tokens, or nothing caches at all. And watch the write premium — some providers charge about 25% extra to write a cache entry, so break-even is around two reads and bursty low-reuse traffic may not benefit. Then instrument cache-hit rate as a first-class SLI, and tag the *reason* for a miss — cold, prefix-changed, or not-attempted — because without that a hit-rate drop is just a number.

**Q4. How do you model cost for an agent, as opposed to a chat system?**
The unit is different. For chat, cost per request or per conversation works. For an agent, the right unit is cost per task, because each task runs multiple steps and each step re-sends the whole stack — the system prompt, the context, the scratchpad. So cost per task is cost per request times expected steps times one plus the retry rate. For a support agent that's 5 to 8 steps, so a $0.012 request becomes $0.10. For a coding agent that's 20 to 24 turns with huge contexts, so it becomes $1.40. And the failure mode is that cost per call looks fine while cost per task explodes, which is why I always measure the task.

**Q5. What's the biggest cost anti-pattern?**
Runaway agent loops, in terms of damage. Reported incidents have agents burning tens of thousands of dollars over a weekend. The mechanism is specific: a tool returns an error that reads as retryable, the loop retries, each retry re-sends a growing context because the scratchpad appends the error, and it repeats for every task in the queue. Cost per task goes from $0.30 to $40. And the defence has to be inside the loop — hard step, token, cost, and wall-clock ceilings, a duplicate-call detector that stops on the same tool and arguments twice, and a no-progress detector. An alert is a notification, not a control. By the time the alert fires and someone reads it, the money is spent. The counter inside the loop is what stops it.

**Q6. When does self-hosting beat an API?**
The reported break-even is high tens to hundreds of millions of tokens a month, or around 500 million for a 70B-tier model, and it depends on constant high throughput and high utilisation. But the load-bearing warning is hidden cost: raw GPU rental is only 30 to 40% of the true cost, so you should apply a 2.5 to 3x multiplier for networking, storage, observability, and engineering — and engineering labour often exceeds infrastructure. So the naive break-even is off by about 3x. For most teams in 2026 managed APIs are cheaper once you count the whole stack. Self-hosting wins at high, predictable, well-utilised volume, or for data-residency reasons where cost isn't the deciding factor.

**Q7. How do you reduce cost by 50% on an existing RAG product?**
I'd sequence by effort-to-impact, because the biggest lever is also the cheapest. First, prompt caching to a 90% hit rate on the static prefix — one day of work, and on my worked example that's a 34% reduction on its own. Second, a `max_tokens` cap and a terse output contract, which is two hours and cuts output cost meaningfully. Third, trim few-shot examples from five to two, which is another big chunk of the prefix. Fourth, batch the entire offline eval tier for half off. Only then do I go to the structural work — cascading, which is two weeks but delivers the largest single reduction at 45 to 85%, and semantic caching. The point is that if you start with the two-week cascade project you'll get a big win eventually, but you'd have captured a third of the total in the first day.

**Q8. What is cost per outcome and why does it matter?**
It's total spend divided by successful outcomes — cost per resolved ticket, per completed task, per accepted PR. It matters because it's the only unit that connects to pricing. AI products run at roughly 25 to 30 margin points below traditional SaaS, so outcome-based pricing is rising, and it's a trap if you don't know your cost per outcome. You price at $2 a resolved ticket, feel good about a 67% margin, then discover the escalation rate drifted from 20% to 35%, COGS went to $0.82, and your margin is 59%. So cost per outcome is a pricing input, not a reporting metric.

**Q9. How do you attribute cost across teams?**
A gateway or token proxy that every call routes through, tagging team, feature, tenant, model, route, and release ID. Then the discipline is showback before chargeback — dashboards first, with no invoices, until tag coverage is 100% and stable. The reason is that the moment you bill a team, the conversation becomes about whether the meter is right instead of whether the system is efficient, and you lose the credibility you need. In my experience a weekly top-ten spender list changes behaviour on its own. And hard chargeback needs a dispute process and per-team forecasts, otherwise it becomes a political problem, not a cost problem.

**Q10. How do you handle a tenant whose cost is disproportionate to their revenue?**
First, make it visible — per-tenant tagging and a margin waterfall sorted by tenant. Then look at why. Usually it's one of four things: they use long contexts heavily, they have a low cache-hit rate because their queries are highly diverse, they're heavy on the frontier tier because their queries classify as complex, or they're hitting retries from an integration bug on their side. Each has a different fix: context budgets, a per-tenant cache strategy, routing, or an engineering conversation with them. If it's genuinely intrinsic, that's a pricing conversation, not an engineering one — and I'd rather have that conversation with data than with an average.

**Q11 (pushback). Isn't optimising cost just going to hurt quality?**
Some levers are free and some are not, and the ordering matters. Prompt caching costs nothing in quality — it's the same prompt, just billed differently. Batching costs nothing but latency. Output caps cost a little if you set them too low, which is why I set them above the p95. Cascading is the one with real quality risk, and it's why I escalate on confidence rather than failure and measure against a locked eval — the literature puts cascades at 45 to 85% savings at around 95% quality retention, which means you're accepting a small quality cost for a large cost saving. The line I hold is that I never take cost out of safety. The cost ladder never disables guardrails, PII filtering, or the action gate. Capability and cost are negotiable; safety is not.

**Q12. What would you monitor to catch cost problems before the invoice?**
Six leading indicators. Cache-hit rate, because it's the biggest lever and it degrades silently. The breakdown of cache misses by reason, because that tells you whether it's cold-start or a bug. Reasoning-token share, because those are invisible. The retry multiplier, because that's a provider problem showing up as your cost. Mean output tokens, which catches the verbosity regression from a model change. And routing tier mix, because a stuck fallback routes everything to the frontier tier. All six move before the bill does. The bill is a lagging indicator by up to a month, which is why the invoice is never the first thing that should tell you.

**Q13. How do you decide between RAG and long-context stuffing on cost grounds?**
Retrieval is dramatically cheaper per query because you pay for a few relevant chunks instead of a giant prompt — I've seen 10 to 50x on the same corpus. Long context wins when the document set is small and static, because then the whole thing caches and you pay the cached rate, and there's no retrieval pipeline to build or maintain. RAG wins for large or frequently changing corpora and high query volume, because the retrieval cost is a fraction of the context cost. The decision rule I use: if the corpus fits comfortably in a cached prefix and changes rarely, long context is simpler and can be cheaper. If it's large, changing, or per-user, RAG.

**Q14. What's the relationship between cost and the other system goals?**
Cost fights performance and reliability directly. Headroom for latency is idle capacity, and idle capacity is money — my numbers put it at about 21% of infrastructure spend. N+1 for availability is another GPU you're not fully using. And quality costs money in the obvious direction: a bigger model, more retrieval, deeper guardrails, more validation passes. So cost is never optimised in isolation. The way I frame it is that I have a quality floor and a latency SLO, and cost is what I minimise subject to those constraints. That's a different objective from minimising cost, and stating it that way is what stops cost work from becoming a quality problem nobody authorised.

**Q15. What one change would you make first in a system you've never seen?**
Measure first, then cache. I'd install the gateway and get tag coverage to 100%, because without it every subsequent decision is a guess. That takes a week or two. Then I'd look at the cache-hit rate, and I would bet — with a 69% static-prefix share and only 28% of calls caching industry-wide — that there's a 30 to 40 percent saving waiting there for about a day of work. That ordering matters because it's the cheapest effort with the biggest lever, and it also builds the trust in the measurement that I need before I propose anything structural.

---

## 12. Cheat-sheet summary

| Dimension | Value |
|---|---|
| **Business model** | Cost-of-goods, not zero-marginal-cost SaaS |
| **AI gross margin** | ~25–30 points below traditional SaaS |
| **Margin target** | ≥ 60% |
| **Inference share of GenAI spend** | **80–90%** |
| **System prompt share of input tokens** | **~69%** |
| **Calls using prompt caching** | **~28%** — the biggest unused lever |
| **Output vs input price** | 3–5x |
| **Prefix cache discount** | 50% (OpenAI) · ~75% (Google) · **~90% (Anthropic)** |
| **Cache discount scope** | **Cached prefix only, not the whole bill** |
| **Cache minimum prefix** | 1,024–2,048 tokens |
| **Cache write premium break-even** | ~2 reads |
| **Batch discount** | ~50%, 24 h ceiling |
| **Provisioned throughput saving** | 15–70% at steady high utilisation |
| **Cascade saving** | **45–85%** at ~95% quality |
| **Reasoning token multiplier** | 3–15x, invisible in output |
| **Distillation saving** | 5–40x per token |
| **RAG vs long context** | 10–50x per query |
| **Output/prompt engineering** | 20–40% of tokens |
| **Self-host break-even** | ~500M tokens/month (70B tier) |
| **Self-host true-cost multiplier** | **2.5–3x** (GPU rental is only 30–40%) |
| **Agent cost per task** | $0.10–$3.00 (vs cents per chat turn) |
| **Cost unit for agents** | **Cost per task, not per call** |
| **Cost unit for business** | **Cost per resolved outcome** |
| **Cost formula** | `Σ(in × in_rate) + (out + reasoning) × out_rate` |
| **Task formula** | `cost_per_request × steps × (1 + retry_rate)` |
| **Cache-hit-rate SLO** | ≥ 0.70 · alert < 0.50 |
| **Retry multiplier SLO** | ≤ 1.15 |
| **Reasoning-token share SLO** | ≤ 0.30 |
| **Batch share of offline SLO** | ≥ 0.90 |
| **Tag coverage SLO** | **1.00** |
| **Top spike cause** | Cache-hit collapse (~40%) |
| **2nd cause** | Reasoning tokens enabled (~15%) |
| **3rd cause** | Retry multiplier (~12%) |
| **Spike diagnosis** | flat traffic → cost/request doubled |
| **Budget requires** | A declared **enforcement action**, not just an alert |
| **Chargeback sequence** | Showback → soft → hard (never skip) |
| **Runaway agent defence** | **Ceilings inside the loop**, not alerts after it |
| **Never in the cost ladder** | Guardrails · PII · action gate · **eval sampling** |
| **The rule that never breaks** | **An alert is a notification; only a counter is a control.** |

---

## 13. Cross-links

- [Capacity Planning and Autoscaling](05-capacity-planning-and-autoscaling.md) — the idle-capacity side of the cost trade-off
- [Deployment, Release and CI/CD](04-deployment-release-and-cicd.md) — cost regression gates and per-release attribution
- [Reliability Patterns](07-reliability-patterns.md) — bounded retries and circuit breakers that prevent retry storms
- [LLMOps for AI Systems](10-llmops-for-ai-systems.md) — cost as part of the operational lifecycle
- [LLM Serving, Inference and Gateway](../HLD/05-ai-systems/05-llm-serving-inference-and-gateway.md) — prefix caching and KV reuse mechanics
- [Fine-Tuning vs RAG vs Prompting](../HLD/05-ai-systems/09-fine-tuning-vs-rag-vs-prompting.md) — distillation break-even arithmetic
- [Agentic System Design](../HLD/05-ai-systems/06-agentic-system-design.md) — step, cost, and token budgets
- [Production RAG System](../HLD/05-ai-systems/01-production-rag-system.md) — chunk budgeting and retrieval cost
- [AI Customer Support Agent Case Study](../Case-Studies/HLD/10-ai-customer-support-agent.md) — cost per resolved ticket in practice
- [Agentic Coding Assistant Case Study](../Case-Studies/HLD/11-agentic-coding-assistant.md) — cost per task at 800k tokens
- [Operationalisation Cheatsheet](../Cheat-Sheets/06-operationalisation-cheatsheet.md)
