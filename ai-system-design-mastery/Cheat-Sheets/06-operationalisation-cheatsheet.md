# Operationalisation Cheatsheet

**How to use this** — The ops reference: release engineering, capacity, cost, reliability, security operations, disaster recovery, and LLMOps. Every number is a working default with a rationale. The three things to carry into any operational interview: **a semantic regression returns 200 with a confident wrong answer**, **rollback is a pointer change, not a deploy**, and **you cannot alert on what doesn't throw — so quality must be an SLI**.

---

## 1. The release manifest

```
  release_id:  rel_2026_09_27_4a91
  ┌──────────────────────────────────────────────────────┐
  │  code            v2.14.0        git sha 4a91c2e      │
  │  prompt          v19            hash 8f3a...         │
  │  model           provider=gpt-x pin=2026-08-14       │
  │  index           snap_8812      embedder v3, 1536d   │
  │  guardrails      v7  (thresholds 0.72 / 0.65)        │
  │  config          v31  (top_k 5, rerank on, timeout)  │
  └──────────────────────────────────────────────────────┘
       │
       └─▶ the ONLY thing a deploy changes is the pointer to the active manifest
```

**The four rules:**

1. The manifest is **immutable** — never edit in place
2. The **pointer is the only deploy change**
3. `release_id` appears in **every trace and log line**
4. **Rollback = repoint** — seconds, not a redeploy

**The six assets and how each fails:**

| Asset | Changes | Fails as | Rollback |
|---|---|---|---|
| Code | deploy | exceptions, wrong behaviour | redeploy |
| Prompt | config / DB | silent quality shift | repoint |
| Model | provider or pin | capability/format change, cost change | repoint (if not deprecated) |
| Index | ingestion | stale or missing content | alias flip |
| Guardrails | config | over/under-blocking | repoint |
| Config | config | latency, cost, retrieval shifts | repoint |

**Rule:** never overwrite the thing you might roll back to. Old manifests stay readable; old indexes stay warm ≥ 24 h.

---

## 2. CI/CD pipeline

```
  change (code | prompt | model pin | guardrail | config)
      │
      ▼
  ┌────────────────────┐
  │ BUILD / LINT       │
  └─────────┬──────────┘
            ▼
  ┌────────────────────┐   golden set 200–500 cases
  │ EVAL GATE          │   + slice metrics (query type, difficulty, tenant)
  │ (BLOCKING)         │   + known-hard regression cases
  │                    │   + cost & latency
  └─────────┬──────────┘
            │ FAIL → no ship. Relaxing an assertion is not the fix.
            ▼
  ┌────────────────────┐   5% traffic, min 1 hour, min sample size
  │ CANARY             │   operational metrics WITHIN tolerance
  │                    │   + judge score diff NOT significant / wrong direction
  └─────────┬──────────┘   + no new failure signatures in trace tags
            ▼
  ┌────────────────────┐
  │ FULL ROLLOUT       │   manifest pointer flip
  └────────────────────┘
```

**Gate pass criteria:**

| Check | Threshold |
|---|---|
| Any slice drop | > defined threshold → fail |
| Faithfulness | ≥ 0.85 |
| Hallucination rate | ≤ 0.03 |
| p95 latency growth | ≤ 15% |
| Cost per request growth | ≤ 20% |

**Change management — three four-eyes rules:**

1. **Tightening is free; loosening needs a reviewer**
2. Model pin, embedding version, or guardrail threshold change → second reviewer
3. Change altering the cost profile by > 20% → second reviewer

Everything else self-service. **Audit test:** if you find production changes with no manifest entry, the process is too heavy.

---

## 3. Capacity estimation

```
  DAU × requests/user/day           = requests/day
  requests/day ÷ 86,400             = avg QPS
  avg QPS × 3                       = peak QPS   (4–5x for consumer daily peaks)

  1 day ≈ 10^5 seconds
  1M requests/day ≈ 12 QPS

  CONCURRENCY = peak QPS × p95 latency (seconds)
             = 50 QPS × 3 s = 150 concurrent requests
```

**Two constraints that bind:**

| Constraint | Formula | Why it matters |
|---|---|---|
| Tokens / minute | req/min × (avg input + expected output) | provider quota is granted in tiers, upgrades take days |
| Concurrency | peak QPS × p95 latency | determines quota, queue, or model tier — the one people forget |

**Availability arithmetic:**

| Target | Downtime / year |
|---|---|
| 99% | 3.65 days |
| 99.9% | 8.76 hours |
| 99.99% | 52.6 minutes |

Two providers ≈ **99.99%, not 99.9999%** — correlated failure (same cloud, same region, same dependency) caps the gain.

**Headroom:** hosted API → provision quota at **2x peak**, secondary provider at **25% steady state**. Self-hosted GPU → reserved standby at **60–100% of the primary pool**, because GPU capacity in a second region isn't obtainable on demand.

**Autoscaling — four differences from CPU services:**

| Difference | Consequence |
|---|---|
| GPU node ready in 2–10 min | scale on prediction/schedule, not reaction |
| GPU utilisation is a bad metric | scale on queue depth, TTFT, KV-cache utilisation |
| Scale-down discards KV cache | slow down-scale, 15-min cooldown |
| Capacity is non-fungible | can't run 70B on the small instance type |

**Target utilisation 60–70%**, not 80–90% — the last 20% is where latency falls off a cliff.

---

## 4. Cost and FinOps

**Where the money goes:** inference = **80–90%** of GenAI spend.

| Structural fact | Value |
|---|---|
| Output price vs input | 3–5x per token |
| But input usually dominates the bill | e.g. 8,000 in / 300 out |
| System prompt + tool defs share of input | **~69%**, identical every request |
| Calls using caching | **~28%** — the biggest untapped lever |
| Reasoning token multiplier | 3–15x |
| Eval/observability stack | 2–4% of inference cost |

**Levers in value order:**

| Lever | Return | Trade-off |
|---|---|---|
| **1. Prompt caching** | 40–60% input cost cut | needs stable static-first prefix |
| **2. Cascade routing** | 45–85% saving at ~95% quality | quality risk on misroute |
| **3. Batch API** | ~50% off | latency — offline only |
| **4. Reserved / spot** | large GPU line reduction | interruption risk |
| **5. Distillation / fine-tune** | moves a task to a small model permanently | high up-front cost, narrow fit |

The under-used lever is caching (no quality trade). The over-used one is aggressive downgrading (the only lever that silently reduces quality).

**"Bill doubled, traffic flat" — check in this order:**

| # | Cause | Signal |
|---|---|---|
| 1 | **Cache-hit collapse (~40% of cases)** | cache hit rate vs baseline |
| 2 | Output length drift | avg output tokens |
| 3 | Retry rate | retries per request |
| 4 | Agent step count | steps per task |
| 5 | Model pin change | provider changelog; output distribution |
| 6 | Retrieval volume | top-k, query-time enrichment |
| 7 | Guardrail sampling rate | eval sample % |

**Attribution stages:** **showback** (measure + publish, no consequence — where most of the value is) → **soft chargeback** (budgets + alerts) → **hard chargeback** (internal billing; requires trustworthy attribution or it dies on the first dispute).

**Budget mechanism:** alert at **80%**, circuit at **100%**, and the circuit **degrades to a cheaper path — it does not fail the feature.**

**Self-host economics:** below **4,000–8,000 calls/day** it never wins. Apply **2.5–3x** to the GPU line (cluster, people, idle). 70B-class crossover ≈ **500M tokens/month**.

---

## 5. Reliability

**Two failure classes — the framing that matters:**

| | Infrastructure | **Semantic** |
|---|---|---|
| Signal | 5xx, timeout, 429 | **200 OK, 340 ms, wrong** |
| Standard monitoring | catches it | **reports 100% availability** |
| Fix | failover, scale, shed | prompt/model/index/judge change |
| Runbook | standard | **"what changed?"** |

**Retry policy:**

| Retry | Do NOT retry |
|---|---|
| 429, timeout, 503, connection error | 401, 403, 400, policy violation |
| | context-length overflow, hard quota |
| | **a semantically wrong answer** |

```
max_retries       = 3
base_delay        = 1.0 s
exponential_base  = 2.0
max_delay         = 60.0 s
jitter            = 0.5   ← MANDATORY (prevents fleet retry stampede)
fleet retry cap   = 10%   ← above this, stop retrying and shed
```

**Circuit breaker:**

```
failure_threshold    = 5 consecutive (or rate-based at volume)
recovery_timeout     = 30 s
half_open_max_calls  = 3
success_threshold    = 2

key  = provider:model:operation   ← granular, so a healthy embed
                                    endpoint doesn't get taken down
states: CLOSED → OPEN → HALF_OPEN → CLOSED
```

Retries sit **inside** the breaker's scope; both count per attempt, or retries amplify an outage.

**Bulkheads:**

| Pool | Concurrency | Rationale |
|---|---|---|
| critical | 10 | interactive, revenue-affecting — never shed |
| realtime | 50 | main user-facing path |
| **agents** | 30 | **own pool** — one agent task holds a slot for ~2 min across 10 steps |
| batch | 200 | offline |

**Shed order:** batch → agents → realtime. Critical never.

**Timeouts (layered):**

| Layer | Value |
|---|---|
| connect | 5 s |
| read | 30 s |
| **total** | 60 s |
| AdaptiveTimeout | p99 × 1.2, clamped 10–120 s, over 1,000 observations |

**Degradation ladder:**

```
FULL → REDUCED → MINIMAL → CACHED → OFFLINE

Rule 1: SAFETY IS NEVER A LEVEL — guardrails don't come out of the ladder
Rule 2: NEVER SILENTLY SERVE A WORSE ANSWER — say it's degraded
```

**Semantic validation stack (6 stages):** schema → grounding → PII → citation → freshness → consistency. Sample the expensive stages; keep the cheap ones at 100%.

**Detecting model regression — five layered defences:**

1. **Daily production golden set** (highest value — catches within 24 h)
2. Online judge sampling 1–5%
3. Output distribution monitoring (length, refusal rate, formatting)
4. Trace tagging by model version for cohort comparison
5. **Canary model pin — never let a provider auto-upgrade you**

---

## 6. SLIs and alerts

**Semantic SLIs (these are the ones people forget):**

| SLI | Target | Alert |
|---|---|---|
| Sampled quality score | ≥ 4.2 / 5.0 | < 3.8 |
| Faithfulness | ≥ 0.85 | < 0.80 |
| Hallucination rate | ≤ 0.03 | > 0.05 |
| Abstention correctness | in band | outside 8–15% band either way |
| Schema-valid rate | ≥ 0.995 | < 0.99 |
| Semantic-cache false-hit rate | ≤ 0.005 | > 0.01 |
| Judge-human kappa | ≥ 0.6 | < 0.4 |
| Retrieval recall@k | ≥ 0.80 | drop > 10% |

**Operational SLIs:**

| SLI | Target |
|---|---|
| Availability | tiered: critical 99.99%, standard 99.9%, best-effort 99% |
| TTFT p95 | < 2 s |
| Total latency p95 | < 10 s (critical < 3 s) |
| Error rate | < 0.5% |
| Queue depth | < 10 waiting |
| KV-cache utilisation | < 85% |
| Cache hit rate | baseline ± 15% |
| Cost per request | baseline + 20% |
| Freshness lag | within SLO |

**Zero-tolerance alerts (page immediately, single occurrence is a reportable event):**

- PII in an outbound response or external call
- Cross-tenant retrieval
- High-tier (T3/T4) action executed without confirmation
- Unrecoverable / undocumented DR asset discovered

---

## 7. Security and compliance operations

**The lethal trifecta:**

```
   private data  +  untrusted content  +  outbound channel
   Remove ANY ONE → exfiltration is structurally impossible
   Guardrails remove NONE of them
```

**Four architectural controls (all outside the model):** least privilege · no credentials in context · egress allowlist · out-of-band confirmation for irreversible actions.

**Guardrail performance budget:**

| Path | Cost | Notes |
|---|---|---|
| Input stack | ~160 ms | normalise FIRST (zero-width, homoglyphs, HTML comments, base64) |
| Output stack | ~330 ms | |
| **Total** | **~490 ms** | **~$0.0007/request (~$2.70 per 1M)** |

**14 injection regex patterns + a classifier.** Instruction hierarchy: **system > developer > user > tool content.**

**PII on output = BLOCK, never silently redact.** Silently redacting produces an answer the user can't verify and hides that a control fired.

**Risk tiers:**

| Tier | Action class | Gate |
|---|---|---|
| T0 | read | auto |
| T1 | low write | auto + audit |
| T2 | medium | auto + notify |
| T3 | high (customer-facing, financial, access) | **confirmation** |
| T4 | destructive | confirmation + typed scope |

**The gate is in code:** permission → verification → tier → flags → velocity → budget. Confirmation bound to **(tool, canonicalised args)**, post-proposal, out-of-band, **single-use**, **60 s TTL**.

**Governance — live dates to know:**

| Regime | Date | Detail |
|---|---|---|
| **EU AI Act Art. 50** (transparency) | **2 Aug 2026** | penalties **EUR 15M or 3%** of worldwide turnover |
| EU AI Act prohibited practices | in force | **35M or 7%** |
| Art. 101 GPAI fining power | **2 Aug 2026** | |
| High-risk Annex III | **2 Dec 2027** | via Reg. (EU) 2026/1744 |
| High-risk Annex I | **2 Aug 2028** | |
| **California SB 942 (as amended by AB 853)** | **2 Aug 2026** | > 1M monthly users; **$5,000 per violation per day** |

**California latent-disclosure four fields:** provider name · system name and version · timestamp · unique identifier. AB 853 **excludes text**. **Build the union** of regimes — one control set satisfying the strictest applicable.

**Frameworks:** NIST AI RMF (Govern / Map / Measure / Manage) + GenAI Profile (12 categories) · SOC 2 · ISO/IEC 42001.

**Audit retention:** high-risk interactions **≥ 6 months**; object-lock compliance mode for the irreplaceable subset.

**OWASP crosswalk rule:** every risk maps to **a control, a test, AND a log signal**. If any of the three is missing, that's the gap — found before the auditor finds it.

**Security SLIs beyond the zero-tolerance three:** injection detection rate, guardrail block rate, rate-limit trips per principal, egress destination set changes, audit records missing required fields.

---

## 8. Disaster recovery

**Three asset classes — the load-bearing classification:**

| Class | Assets | Treatment | Cost |
|---|---|---|---|
| **REBUILDABLE** | embeddings, HNSW, BM25, semantic cache, reranker cache | **don't back up — rebuild** | recipe is the backup |
| **RESTORABLE** | prompts, configs, manifests, adapters, agent state, memory | normal backup, versioned | low |
| **IRREPLACEABLE** | eval sets, human labels, red-team corpora, adjudications, rubrics | object-lock, most expensive retention | cannot be regenerated at any price |

**The recipe is the real backup:** chunker config + pinned embedding version + normalisation steps. Without it you can't reproduce the index even with the source documents.

**RPO/RTO tiers:**

| Tier | RPO | RTO | Note |
|---|---|---|---|
| T0 | 0 | 15 min | warm path required |
| T1 | 5 min – 1 h | 1 h | |
| T2 | 1 h | 4 h | rebuild acceptable |
| T3 | 24 h | 24 h | |

**Size and criticality are uncorrelated** — an irreplaceable 2 GB eval set is T0; a rebuildable 97 GB index is T2.

**Index economics (10M chunks × 1536d):**

```
raw vectors            61.4 GB
× 1.5 HNSW             92.2 GB
+ metadata/chunk text   5.0 GB  (~500 B/chunk)
                  ≈     97 GB per replica

Rebuild:  ~10–12 hours, ~$60–150
Backup:   $150–600 / month, recurring
→ for T2 and below: DON'T BACK UP, REBUILD
→ for T0: keep a warm replica; rebuild can't meet 15 min
```

RAPTOR tree rebuild for the same corpus: **~$2,000** — "rebuildable" does not mean "cheap to rebuild." Snapshot it.

**Provider failover is SUBSTITUTION, not replication** — a capability change, not continuity.

| Failover viability gate | Threshold |
|---|---|
| Quality vs primary on golden set | within **2 points** |
| Schema-valid rate | ≥ **0.995** |
| Tool-call parse rate | ≥ **0.999** |
| Steady-state traffic share | **25%** (keeps the path warm and exercised) |

**The cold-cache trap:** after failover, no warm prompt cache → **cost ~+91%, latency ~+30%**. Drill for **≥ 1 hour of real traffic**, or you validated the switch and not the aftermath.

**DR postures (% of primary cost):**

| Posture | Cost | RTO |
|---|---|---|
| Backup & restore | 2–5% | hours |
| Pilot light | 10–15% | ~1 h |
| **Warm standby** | 25–40% | minutes |
| Active-active | 90–130% | ~0 |

GPU standby: **60–100% of the primary pool** — the exception to every other row.

**Four restore runbooks:** region loss · index corruption · provider deprecation · corpus loss. **Five drills** on a schedule.

**Failover safety gate:** verify **guardrails + action gate + PII scan are ACTIVE on the new path BEFORE opening traffic.** Failing over into an unguarded path converts an availability incident into a security incident.

**Restore rules:** never rebuild in place; store chunk text in the payload; snapshot expensive derived artefacts.

---

## 9. LLMOps

**The metric taxonomy:**

| Family | Examples |
|---|---|
| Operational | latency, error rate, throughput, availability |
| **Quality** | faithfulness, hallucination, judge score, kappa, Pass^k |
| Cost | cost/request, cache hit rate, tokens by component |
| Agent | steps/task, tool errors, trajectory success, delegation depth |

**Layered judge architecture — the core of affordable evaluation:**

```
  100% of traffic   →  DISTILLED judge, inline, sub-100 ms
   1–5% of traffic  →  FRONTIER judge (calibrates the distilled one)
   disagreements    →  HUMANS
   quarterly        →  RETRAIN the distilled judge
```

| Distilled judge | Value |
|---|---|
| Cost vs frontier | ~**97% cheaper** |
| Latency | ~**10x** lower, sub-100 ms |
| Agreement with frontier | 88–92% |
| vs human gold | within 2–3 points |
| Taxonomy | groundedness, instruction-following, toxicity, PII, off-topic, refusal |
| **Known limitation** | **regresses to a default outside its taxonomy** |

**Economics at 1M requests/day:** frontier-on-everything ≈ **$600K/month** (unaffordable above ~100K req/day). Full layered stack ≈ **$33K/month** = **2–4% of inference**.

**Judge biases and fixes:**

| Bias | Fix |
|---|---|
| Position | swap and re-run; inconsistent → tie |
| Length | instruct to ignore length |
| Self-preference | **different model family** as judge |
| Format | diverse calibration examples |

**Judge SLIs:** human agreement **κ ≥ 0.6** (alert < 0.4) · distilled-vs-frontier agreement · score distribution drift.

**Drift detection — needs BOTH:**

```
  effect size  ≥ 10%   AND   p < 0.05
  (effect alone fires constantly at volume;
   significance alone fires on meaningless differences)
```

**Four causes of a quality drop — separate in this order:**

| # | Cause | Check |
|---|---|---|
| 1 | Model | daily golden set vs yesterday |
| 2 | Prompt | release manifest diff |
| 3 | Index | re-index completion + embedding version |
| 4 | **Judge** | **calibration — the one people miss** |

**Pass^k:**

```
Pass^1 70% + Pass^4 12%  →  works on the easy path only
Target Pass^4 ≥ 0.60 for anything with side effects
Verification = post-conditions in code, NOT LLM transcript scoring
```

**Trace attributes — nine required:**

`release_id` · tenant · principal · prompt template version · model pin · index snapshot · `cache_read_tokens` · `acl_filtered` · tool calls with tier + confirmation status

**The three people omit:** `cache_read_tokens` · `acl_filtered` · prompt template version.

**Trajectory failure modes (five):** reasoning-action mismatch · over-retrieval · tool flailing · premature commitment · self-jailbreaking.

**Memory evals are per-operation:** extraction / update / QA. A **5% extraction error compounds.**

**Never disable eval sampling** — it's the only signal that catches silent quality loss.

**90-day plan:**

| Days | Phase | Deliverable |
|---|---|---|
| 1–14 | **Visibility** | `release_id` in traces, cost + latency attributed, dashboard |
| 15–30 | **Manifest** | six assets pinned, rollback = repoint |
| 31–60 | **Evaluation** | golden set, calibrated judge, CI gate, 1–5% online sampling |
| 61–90 | **Continuous** | drift detection, degradation runbook, cost ops, governance byproduct |

**Do not start at phase three.** Evaluation without trace attribution can't explain *why* anything changed.

---

## 10. Incident runbook card

```
┌─ IS THERE AN ERROR RATE? ─────────────────────────────────────┐
│  YES → INFRASTRUCTURE PATH                                    │
│    1. error rate by provider + operation                      │
│    2. breaker states (CLOSED/OPEN/HALF_OPEN)                  │
│    3. queue depth per bulkhead                                │
│    4. failover viability gate → switch secondary              │
│    5. shed: batch → agents → realtime (critical never)        │
│    6. degradation ladder: FULL→REDUCED→MINIMAL→CACHED         │
│       (safety is never a level)                               │
└───────────────────────────────────────────────────────────────┘

┌─ QUALITY DOWN, NO ERRORS? ────────────────────────────────────┐
│  YES → SEMANTIC PATH.  Question 1: WHAT CHANGED?               │
│    1. model?  daily golden set vs baseline                    │
│    2. prompt? manifest diff                                   │
│    3. index?  re-index complete? embedding version?           │
│    4. JUDGE?  recalibrate vs human labels (kappa)             │
│    5. quantify the affected slice, not the aggregate          │
│    6. roll back the pointer (90 s) OR degrade explicitly      │
│    7. tell the slice owner — the users, not just the graph    │
└───────────────────────────────────────────────────────────────┘

┌─ SECURITY EVENT? ─────────────────────────────────────────────┐
│  1. CONTAIN: disable the specific tool/egress path            │
│  2. SCOPE: audit trail → every session touching the vector    │
│  3. ASSESS: what left, where, under what authorisation        │
│  4. REMEDIATE THE ARCHITECTURE (a regex is not remediation)   │
│  5. NOTIFY per policy (GDPR 72 h; contracts often shorter)    │
│  6. FEED BACK: attack → permanent red-team + regression case  │
└───────────────────────────────────────────────────────────────┘
```

---

## 11. Numbers to remember

| Number | Value |
|---|---|
| Guardrail budget | ~490 ms, ~$0.0007/request |
| Retry | 3 attempts, base 1 s, max 60 s, jitter 0.5 |
| Fleet retry cap | 10% |
| Circuit breaker | 5 failures, 30 s recovery, 3 half-open, 2 to close |
| Bulkheads | critical 10 / realtime 50 / agents 30 / batch 200 |
| Timeouts | connect 5 s / read 30 s / total 60 s |
| Quality SLO | ≥ 4.2 / 5.0, alert < 3.8 |
| Faithfulness | ≥ 0.85 |
| Hallucination | ≤ 0.03 |
| Schema-valid | ≥ 0.995 |
| Judge kappa | ≥ 0.6, alert < 0.4 |
| Drift test | ≥ 10% effect AND p < 0.05 |
| Distilled judge | 97% cheaper, 10x faster, 88–92% agreement |
| Eval stack at 1M req/day | ~$33K/month = 2–4% of inference |
| Frontier-on-everything | ~$600K/month (unviable > 100K req/day) |
| Abstention band | 8–15% |
| Cache-hit collapse share of "bill doubled" | ~40% |
| Cascade routing saving | 45–85% at ~95% quality |
| Self-host break-even | 4,000–8,000 calls/day |
| Autoscale target utilisation | 60–70% |
| Provisioned quota | 2x peak |
| Secondary provider share | 25% |
| GPU DR standby | 60–100% of primary |
| Failover cold-cache penalty | +91% cost, +30% latency |
| T0 RPO / RTO | 0 / 15 min |
| Index size 10M × 1536d | ~97 GB |
| Index rebuild | 10–12 h, $60–150 |
| Confirmation TTL | 60 s |
| EU AI Act Art. 50 | live 2 Aug 2026, EUR 15M / 3% |
| California SB 942 | live 2 Aug 2026, $5,000/violation/day |
| Audit retention (high-risk) | ≥ 6 months |
| Pass^4 target | ≥ 0.60 |

---

## 12. Interview one-liners

- "In a normal system a regression produces an error. In an AI system it produces a confident wrong answer with a 200 status."
- "Rollback is a pointer change, not a deploy."
- "Retry handles transient, breaker handles systemic, bulkhead bounds the blast radius — and none of them addresses that the answer is wrong."
- "Never retry a semantically wrong answer."
- "Two providers give you 99.99%, not 99.9999% — correlated failure caps the gain."
- "You cannot fail over a third-party model. Provider failover is a capability change, not continuity."
- "You don't back up embeddings. You rebuild them — and the recipe is the real backup."
- "LLM-as-judge is the worst-case fallback, not the default."
- "Drift needs an effect size AND significance, or it's noise in one direction or the other."
- "Four causes of a quality drop: model, prompt, index, and judge. Check the judge — that's the one people miss."
- "The gate is in code, not in the prompt."
- "PII on output blocks. Silent redaction hides that a control fired."
- "Safety is never a degradation level."
- "Attach a release_id to every trace. It's one field and it's the difference between a ten-minute incident and a one-day one."

---

## Cross-links

- [Deployment, Release and CI/CD](../Operations/04-deployment-release-and-cicd.md)
- [Capacity Planning and Autoscaling](../Operations/05-capacity-planning-and-autoscaling.md)
- [Cost and FinOps](../Operations/06-cost-and-finops.md)
- [Reliability Patterns](../Operations/07-reliability-patterns.md)
- [Security, Guardrails and Compliance](../Operations/08-security-guardrails-and-compliance.md)
- [DR, Failover and Backup](../Operations/09-dr-failover-and-backup.md)
- [LLMOps for AI Systems](../Operations/10-llmops-for-ai-systems.md)
- [LLMOps and Operationalisation Questions](../Interview-Questions/06-llmops-and-operationalisation-questions.md)
- [RAG and LLM Cheatsheet](04-rag-and-llm-cheatsheet.md)
- [Agentic AI Cheatsheet](05-agentic-ai-cheatsheet.md)
- [Foundations and Estimation Cheatsheet](01-foundations-and-estimation-cheatsheet.md)
