# DR, Failover and Backup for AI Systems

**TL;DR** — Disaster recovery for an AI system is *not* disaster recovery for a web app, because an AI system has assets a web app does not: embeddings, vector indexes, fine-tuned adapters, prompt libraries, eval sets, human-labelled golden data, agent event histories, and memory stores. The single most useful classification is by **recoverability**, not by size: some assets are **rebuildable** (embeddings, indexes, semantic caches — recomputable from source at a cost in time and money), some are **restorable** (configs, prompts, weights — mechanical to bring back from a backup), and some are **irreplaceable** (an eval set with six months of human labels, a red-team corpus, a reviewer-judgement dataset — not recomputable at any price). The second insight is that **you cannot fail over a third-party LLM the way you fail over a database.** There is no replica of `gpt-5` in another region. Provider failover means a *different model* with different tokenisation, different tool-call format, and different behaviour — which is a capability change, not a continuity event. And the third: **GPU capacity is region-scarce**, so a DR region you did not reserve capacity in may have no accelerators available when you need them.

> **60-second interview answer:** "I'd start by inventorying the assets, because the failure mode in AI DR is that people back up the database and forget the eval set. There are three classes. Rebuildable: embeddings, vector index, semantic cache — I can recompute those from source documents for about $60 and six hours on a 10-million-chunk corpus, so I do not back them up, I keep the pipeline that rebuilds them tested. Restorable: prompts, configs, adapters, model pins — mechanical, in git and object storage with versioning. And irreplaceable: eval sets with human labels, red-team corpora, reviewer judgements. Those have an RPO of zero and they get the strongest protection in the system, because no amount of money recomputes them. Then the two AI-specific constraints: you cannot fail over a third-party model — provider B is a different model, not a replica, so 'failover' is a capability downgrade you have to have evals for; and GPU capacity is region-scarce, so a DR region without reserved capacity is not a DR region, it's a hope."

---

## 1. Purpose

Define how an AI system survives the loss of a region, a provider, an index, a corpus, or a critical artifact — and how it recovers within a stated RPO and RTO.

**What makes AI DR different from ordinary DR:**

| Ordinary system | AI system |
|---|---|
| Stateless tier is the bulk | Stateless serving is a *thin* tier over an API |
| Database is the state | **Six stateful asset classes**, one of which is irreplaceable |
| Back up the DB, restore the DB | **Backups are the wrong answer for most assets** — rebuild is |
| Failover = same system elsewhere | **Provider failover = a different model with different behaviour** |
| Capacity is fungible | **GPU capacity is region-scarce and quota-limited** |

**The framing to lead with:** *"Ordinary DR asks 'can I restore the data?'. AI DR asks a prior question: 'can I rebuild this, or is it gone forever?' A 10-million-vector index is not precious — it's a derived artifact I can regenerate for the price of an embedding job. A 2,000-example eval set annotated by three domain experts over six months is precious and cannot be regenerated at any price. If I protect those two things with equal effort, I've misallocated my DR budget."*

**In scope:** asset inventory, RPO/RTO by class, backup procedures, restore runbooks, region failover, provider failover, durability of agent state, DR drills.

**Out of scope:** steady-state provider failover mechanics (see [reliability patterns](07-reliability-patterns.md)) and capacity saturation (see [capacity planning](05-capacity-planning-and-autoscaling.md)).

---

## 2. Scope — the asset inventory

### 2.1 The six asset classes

| # | Asset | Size (10M-chunk reference) | Recoverability | Where it lives |
|---|---|---|---|---|
| 1 | **Source corpus** | 500 GB raw | **Irreplaceable if deleted externally** | Object storage, versioned |
| 2 | **Embeddings / vector index** | ~97 GB | **Rebuildable** (~6 h, ~$60) | Vector DB |
| 3 | **Model artifacts** | 0 (API) or 15–280 GB (self-hosted) | **Restorable / re-downloadable** | Provider or registry |
| 4 | **Fine-tuned adapters** | 10–200 MB (LoRA) | **Irreplaceable** if the data is gone | Registry + object storage |
| 5 | **Prompts, configs, code, release manifests** | < 1 GB | **Restorable** | Git |
| 6 | **Eval sets, golden data, red-team corpora, human labels, agent traces** | 1–50 GB | **IRREPLACEABLE** | Object storage + git-lfs |
| 7 | **Semantic cache** | 20–200 GB | **Rebuildable** (warm-up ~hours) | Redis / vector store |
| 8 | **Agent event history / durable state** | 10–500 GB | **Restorable (journaled)** | Temporal / Postgres |

### 2.2 Recoverability classification — the load-bearing table

| Class | Definition | Examples | DR strategy | RPO | RTO |
|---|---|---|---|---|---|
| **REBUILDABLE** | Recomputable from a cheaper source | Embeddings, HNSW index, semantic cache, BM25 index, reranker cache | **Do not back up.** Keep the rebuild pipeline tested | 24 h (source) | 6–12 h |
| **RESTORABLE** | Backed up; restore is mechanical | Configs, prompts, manifests, adapters, model pins, agent state | Snapshot + version + tested restore | 5 min – 1 h | 15 min – 4 h |
| **IRREPLACEABLE** | Not recomputable at any price | Human-labelled eval sets, red-team corpora, reviewer judgements, adjudicated traces, domain expert rubrics | **Multi-region replication + immutability + versioning** | **0** | 15 min |
| **THIRD-PARTY** | Not yours; you can only pick another | Frontier model APIs, provider embeddings, managed vector DB | Pin, document, and have a *different* fallback with evals | n/a | 5 min – 4 h |

**The rule that follows:** *"Backup effort is inversely proportional to size and directly proportional to irreplaceability. The 97-gigabyte index gets no backup — it gets a tested rebuild script. The 8-megabyte human-labelled eval set gets cross-region replication, object-lock immutability, and versioning, because it's the single most valuable thing in the system per byte."*

### 2.3 The RPO/RTO tiers

| Tier | RPO | RTO | Assets | Availability target |
|---|---|---|---|---|
| **T0 — Critical** | 0 | 15 min | Eval sets, red-team corpora, human labels, source corpus | 99.99% |
| **T1 — High** | 5 min | 1 h | Prompts, configs, release manifests, adapters, agent state | 99.9% |
| **T2 — Medium** | 1 h | 4 h | Vector index (rebuildable), memory stores | 99.9% |
| **T3 — Low** | 24 h | 24 h | Semantic cache, reranker cache, benchmark artifacts | 99% |

**Why eval sets sit at T0 while the index sits at T2** — the sentence to say out loud: *"The index is 97 gigabytes and I'd lose it happily, because I can rebuild it in six hours for sixty dollars. The eval set is eight gigabytes and losing it would cost me six months and possibly be impossible, because the three domain experts who annotated it have moved on. Size and criticality are uncorrelated in AI systems, and that catches people who apply ordinary DR intuition."*

---

## 3. SLIs and SLOs

| SLI | Definition | Target | Alert |
|---|---|---|---|
| **Backup success rate** | Successful backups / scheduled | 100% | **Any miss = page** |
| **Backup freshness (T0)** | Age of the newest replica | ≤ 5 min | > 15 min |
| **Backup freshness (T1)** | Age of the newest backup | ≤ 1 h | > 3 h |
| **Restore test success (T0/T1)** | Successful restore drills / attempted | 100% | Any failure = page |
| **Restore duration (T0)** | Drill time to serveable | ≤ 15 min | > 30 min |
| **DR drill cadence** | Drills per quarter | ≥ 1 | Missed quarter |
| **Failover time (region)** | Detect → serve in region B | ≤ 15 min | > 30 min |
| **Failover time (provider)** | Detect → serve on provider B | ≤ 5 min | > 15 min |
| **Index rebuild time** | Source → queryable | ≤ 8 h | > 12 h |
| **Index rebuild cost** | Per full rebuild | ≤ $150 | > $400 |
| **Cross-region replication lag** | T0 asset lag | ≤ 60 s | > 5 min |
| **Unrecoverable asset count** | Assets with no tested restore | **0** | **Any = critical** |
| **Undocumented asset count** | Assets not in the inventory | **0** | **Any = critical** |

**The two zero-tolerance rows matter most.** *"Every asset must have a tested restore path and be in the inventory. An asset nobody remembered is exactly the asset that turns a four-hour incident into a four-week one, and the way I find those is a quarterly drill where we deliberately try to rebuild the whole system from backups and see what's missing."*

---

## 4. Architecture

```
════════════ DR TOPOLOGY — ACTIVE-PASSIVE WITH WARM STANDBY ════════════

   REGION A (primary)                    REGION B (standby)
   ┌──────────────────────┐              ┌──────────────────────┐
   │  EDGE / GATEWAY      │              │  EDGE / GATEWAY      │
   │  · auth · rate limit │              │  (deployed, drained) │
   └──────────┬───────────┘              └──────────┬───────────┘
              ▼                                     ▼
   ┌──────────────────────┐              ┌──────────────────────┐
   │  SERVING TIER        │              │  SERVING TIER        │
   │  stateless           │              │  stateless           │
   │  · orchestrator      │              │  · IDENTICAL IMAGE   │
   │  · guardrails        │              │  · same manifest     │
   │  · prompt templates  │              │  (scaled to 0 or     │
   │  · agent runtime     │              │   minimum)           │
   └──────────┬───────────┘              └──────────┬───────────┘
              │                                     │
     ┌────────┴────────┐                   ┌────────┴────────┐
     ▼                 ▼                   ▼                 ▼
┌──────────┐   ┌────────────┐        ┌──────────┐   ┌────────────┐
│ VECTOR   │   │ POSTGRES   │        │ VECTOR   │   │ POSTGRES   │
│ index    │   │ + agent    │  ────► │ replica  │   │ read       │
│ (primary)│   │ state      │  async │ (warm)   │   │ replica    │
└──────────┘   └────────────┘  repl. └──────────┘   └────────────┘
     │                │                     ▲             ▲
     └────────────────┴─────────────────────┘             │
              │                                          │
     ┌────────▼──────────────────────────────────────────┴────┐
     │  OBJECT STORAGE — versioned + object-lock (WORM)        │
     │  ┌──────────────────────────────────────────────────┐   │
     │  │ T0 IRREPLACEABLE (replicated, RPO 0)             │   │
     │  │  eval sets · red-team corpora · human labels     │   │
     │  │  reviewer judgements · adjudicated traces        │   │
     │  ├──────────────────────────────────────────────────┤   │
     │  │ T1 (versioned, RPO 1 h)                          │   │
     │  │  adapters · corpus · manifests · memory exports  │   │
     │  ├──────────────────────────────────────────────────┤   │
     │  │ REBUILD SOURCES (the index's real backup)        │   │
     │  │  chunker config · embedding model pin · recipes  │   │
     │  └──────────────────────────────────────────────────┘   │
     └─────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────┐
   │  LLM PROVIDERS — NOT FAILOVER, SUBSTITUTION              │
   │  primary ───► secondary ───► tertiary                    │
   │  each is a DIFFERENT MODEL with different behaviour      │
   │  each needs: pinned snapshot · eval pass · prompt compat │
   └──────────────────────────────────────────────────────────┘
```

### 4.1 Why the serving tier is easy and the data tier is not

```
   SERVING TIER  = stateless + an API call
     Loss → redeploy the image in region B. Minutes. Solved problem.

   DATA TIER     = the AI-specific hard part
     · Vector index:  97 GB, rebuildable in ~6 h — acceptable RTO,
                      if and ONLY if the rebuild pipeline is tested
     · Eval sets:     irreplaceable — replicate continuously
     · Adapters:      small, irreplaceable — replicate continuously
     · Agent state:   journaled — restore from the event history
     · Corpus:        irreplaceable if the source is external and gone
     · Cache:         rebuildable, but warm-up is hours of degraded cost

   ⇒ DR effort goes to the DATA TIER. The serving tier is a Terraform
     apply that you should be doing quarterly anyway.
```

### 4.2 The third-party constraint

```
   ⚠  YOU CANNOT REPLICATE A THIRD-PARTY MODEL.

   There is no standby replica of a frontier API. What you have is:
     · a pinned snapshot on the same provider (a deprecation risk)
     · a DIFFERENT model on a different provider (a behaviour change)

   ⇒ "Provider failover" is a CAPABILITY CHANGE, not continuity.
   ⇒ It must be:
        ① exercised continuously (25% steady-state traffic)
        ② evaluated with a cross-provider eval suite
        ③ normalised at the boundary (tool-call format, tokeniser)
        ④ disclosed to users when answer style shifts materially
```

**The eval requirement that makes provider DR real:** *"A second provider is only a DR plan if a prompt that works on provider A also works on provider B. So I run the full eval suite against every provider monthly, and the fallback is only considered viable if it stays within two points of the primary on the quality score. A fallback that fails the eval suite is not a fallback — it's a second outage with a different error message."*

### 4.3 GPU capacity is region-scarce

| Self-hosted scenario | Reality |
|---|---|
| "Fail over to region B" | Region B may have no accelerators available on demand |
| On-demand GPU at failover | Quota may be 0; capacity may be sold out (weeks of lead time) |
| Reserved capacity | The only thing that works — and it costs whether used or not |
| Provider API instead | The realistic DR path for most teams |

**Provisioning lead times to plan around:**

| Resource | Lead time |
|---|---|
| Cloud quota increase (GPU) | 2 days – 3 weeks |
| New region enablement | 4 – 12 weeks |
| Reserved GPU capacity | 4 – 16 weeks |
| Managed vector DB cross-region replica | Hours – days |
| Provider account + limits raise | Days |

*"If your DR plan is 'spin up GPUs in region B', and the quota there is zero, your RTO is however long a quota increase takes — two days to three weeks. So the honest DR plan for a self-hosted model is either reserved standby capacity in region B, which is expensive, or degrading to a provider API, which is a capability change but available in minutes. I'd state the choice explicitly rather than pretend a region failover is a config flag."*

---

## 5. Standard operating procedures

### SOP-1 — Build and maintain the asset inventory

```
   EVERY ASSET GETS A ROW. No exceptions.

   | asset | owner | class | RPO | RTO | store | backup | restore |
   |       |       |       |     |     |       | method | command |

   The quarterly question that finds the gaps:
     "If region A disappeared right now, could we rebuild this from
      what's in region B and our source control? Show me the command."
     → If nobody can show the command, the asset is not protected.
```

**The four questions per asset:**

| Question | If "no" |
|---|---|
| Is it in the inventory? | **Critical gap** — add it |
| Is it classed (rebuildable/restorable/irreplaceable)? | Classify before you spend |
| Is there a restore command someone has run? | **Untested = unprotected** |
| Does the restore fit the RTO? | Redesign the strategy |

**The assets people forget, in the order they forget them:** eval sets → human labels → red-team corpora → the corpus itself → embedding model *pin and version* → the chunker config → the reranker model → the guardrail classifier weights → the prompt template *history* → the annotation rubric → the semantic cache *schema* → the tenant ACL snapshot.

**The chunker-config point deserves its own line:** *"A vector index is not rebuildable if you don't know how it was chunked. Same documents, different chunking strategy, different index — and different retrieval quality. So the chunker configuration, the embedding model version, and the normalisation rules are part of the backup, because they are the recipe. Backing up the corpus without the recipe is backing up flour and calling it bread."*

### SOP-2 — Backup procedures by class

**Class A — REBUILDABLE: do not back up. Rebuild.**

```python
class IndexRebuildPlan:
    source_bucket    = "s3://corpus/"          # versioned, replicated
    chunker_config   = "git://config/chunking.yaml@sha"
    embedding_model  = "text-embedding-3-large@2024-02"   # PINNED
    vector_store     = "qdrant://prod/knowledge_v7"
    batch_size       = 512
    concurrency      = 32
    target_throughput = 500    # chunks/second

    # 10,000,000 chunks ÷ 500/s ≈ 20,000 s ≈ 5.6 hours
    # 10M chunks × 300 tokens = 3.0B tokens × $0.02/1M = ~$60
    # Index build: HNSW M=32 ef_construction=256 on 10M × 1536
    #   ≈ 4–6 hours additional
    # ⇒ Realistic end-to-end RTO: 10–12 hours for a cold rebuild
    #   ⇒ Warm standby replica brings that to minutes
```

| Rebuildable asset | Rebuild cost | Rebuild time | Backup? |
|---|---|---|---|
| Vector index (10M chunks) | ~$60 embedding + compute | 10–12 h cold | **No** — replica or rebuild |
| BM25 / lexical index | ~$0 | 1–2 h | **No** |
| Semantic cache | ~$0 | Warm-up over 4–24 h | **No** |
| Reranker score cache | ~$0 | 1 h | **No** |
| RAPTOR summary tree | ~$2,000 LLM cost | 6–10 h | **No** — but know the cost |

**The rebuild-cost trap:** *"Rebuildable does not mean free. A RAPTOR-style summary tree over 10 million chunks is an LLM pass over every chunk — roughly two thousand dollars and most of a day. So it's technically rebuildable but it's expensive enough that I'd keep the artifact in object storage rather than recompute it. The rule is: rebuild if the rebuild is cheap relative to a backup's storage and operational cost; snapshot if it isn't."*

**Class B — RESTORABLE: backup with versioning.**

| Asset | Method | Frequency | Retention | Notes |
|---|---|---|---|---|
| Prompts / templates | Git, every commit | Continuous | Forever | History is the artifact |
| Release manifests | Git | Per release | Forever | Includes model pins |
| Configs / IaC | Git + state backup | Continuous | Forever | Terraform state is critical |
| LoRA adapters | Object storage, versioned | Per training run | Forever | 10–200 MB; **treat as irreplaceable** |
| Agent event history | DB snapshot + WAL archiving | Continuous | 90 days hot, 1 y cold | The journal *is* the state |
| Memory store (long-term) | Object storage export | Hourly | 1 year | Validate writes on restore |
| Tenant ACL snapshot | DB snapshot | Hourly | 90 days | Needed for a consistent re-index |

**The release manifest is a recovery artifact.** *"Because the manifest pins code version, prompt version, model snapshot, index version, guardrail version and config, restoring it restores the system definition. So it's the first thing I restore in a DR event — it tells me exactly what to rebuild and in what combination, and it removes the 'what version were we on?' question during the most stressful hour of the year."*

**Class C — IRREPLACEABLE: protect at all costs.**

```python
class T0Protection:
    # The highest protection in the system, for the smallest data
    storage        = "object storage, multi-region replicated"
    immutability   = "object-lock, compliance mode, 7-year retention"
    versioning     = "every write is a new version, never overwritten"
    access         = "write via service account, delete requires 2-person"
    replication    = "synchronous cross-region, RPO 0"
    inventory      = "hash manifest written weekly, verified monthly"
    also_in        = "git-lfs for the text portions (eval Q&A, rubrics)"
```

| T0 asset | Why irreplaceable |
|---|---|
| **Human-labelled eval set** | Months of expert annotation; annotators may be gone |
| **Adjudication records** | The disagreements and resolutions are the ground truth |
| **Red-team corpus** | Real attack strings from real adversaries; not synthesizable |
| **Reviewer-judgement dataset** | The data behind judge calibration |
| **Adjudicated production traces** | Sampled failures with human verdicts |
| **Domain rubrics** | Expert-authored grading criteria |
| **Consent / licence records** | Legal necessity; not reconstructible |

**The line that carries the section:** *"I would spend more on protecting eight gigabytes of human labels than on protecting a hundred gigabytes of index. The index costs sixty dollars and an afternoon. The labels cost six months of three experts' time, and if those experts have left the company, no amount of money brings them back. Nobody's ordinary DR instinct tells them that, which is exactly why it's worth saying in an interview."*

### SOP-3 — Restore runbooks

**RESTORE-1: full region loss → serve in region B**

```
T+0    DETECT
       □ Synthetic check fails in ≥ 3 regions
       □ Provider status page confirms
       □ Declare: incident commander, comms channel, status page

T+2m   DECIDE — the failover authority question
       □ Who can declare region failover? (pre-named, one person)
       □ Do NOT wait for a root cause. Failover is reversible;
         a long outage is not.
       □ Cost of an unnecessary failover: minutes of DNS + a
         reconciliation job. Cost of waiting: the outage duration.

T+5m   SERVE
       □ 1. Promote region B's data replicas to primary
       □ 2. Shift DNS / traffic manager to region B
       □ 3. Scale region B's serving tier to production size
           (this is where reserved capacity or provider APIs matter)
       □ 4. Restore the release manifest → deploy the SAME release_id
       □ 5. Verify: synthetic checks, quality sample, guardrail trips
       □ 6. Confirm the semantic validator and guardrails are LIVE
           before opening traffic — a failover that skips safety
           is not a recovery

T+15m  VERIFY
       □ Health endpoints green in region B
       □ Sampled quality score within 5% of the pre-incident value
       □ Retrieval recall sanity check on a known set
       □ Rate-limit and quota headroom in region B confirmed
       □ No degraded-level responses after T+20m

T+1h   STABILISE
       □ Freeze deploys until the incident is closed
       □ Watch for the two predictable failover failures:
           · region B quota exhaustion (it was never sized for full load)
           · cold cache → cost spike (see SOP-5)
       □ Post-incident: what was missing from the inventory?

T+POST REPAIR
       □ Rebuild the vector index in region A from the pinned recipe
       □ Reconcile writes that happened in B during the failover
       □ Decide whether to fail back (and it is a decision —
         failing back is another outage window)
```

**RESTORE-2: index corruption or accidental deletion**

```
□ 1. STOP ingestion to the affected collection immediately
□ 2. Determine the scope: one shard, one tenant, or all?
□ 3. Prefer the NARROWEST recovery:
      one tenant → re-index that tenant only
      one shard  → restore/rebuild that shard
      all        → full rebuild (10–12 h) or fail to a replica
□ 4. Rebuild using the PINNED recipe — chunker config, embedding
     model version, normalisation rules. Not the current config.
□ 5. Build into a NEW collection; never rebuild in place
□ 6. Dual-index eval: query both old (if available) and new,
     compare nDCG. Block the alias flip if nDCG drops > 3 points.
□ 7. Flip the alias atomically. The alias is the deploy.
□ 8. Re-enable ingestion, then re-verify freshness lag
```

**Why "never rebuild in place":** *"If you rebuild into the same collection name, you have a window where the index is partially rebuilt and answers are drawn from a mixture of old and new — which is worse than an outage, because it's silent. Build alongside, verify with dual-index eval, flip the alias. The alias flip is atomic and reversible, and it's the only step that touches production."*

**RESTORE-3: provider deprecation or outage**

```
□ 1. Confirm: deprecation notice, or hard outage?
□ 2. Check the pinned snapshot is still served
     → mismatch with the manifest = critical alert (see reliability file)
□ 3. If deprecated or unavailable: switch the manifest to the
     secondary provider
□ 4. Run the CROSS-PROVIDER eval before opening traffic:
      · schema-valid output rate ≥ 0.995
      · quality score within 2 points of the primary's baseline
      · tool-call parse rate ≥ 0.999
      · guardrail behaviour unchanged
□ 5. If the eval fails: DO NOT SHIP. Degrade to a lower tier
     (smaller model on the primary provider) or abstain.
□ 6. Re-run cost model — provider B may be 2–4x the price
□ 7. Notify users if answer style shifts materially
```

**RESTORE-4: corpus loss**

```
   ⚠  THE INDEX IS A COPY, NOT THE SOURCE.
      If the source documents are gone, the index is not rebuildable —
      it becomes the only remaining copy, which means it must be
      snapshotted, which means it is now T0.

   □ 1. Inventory what remains: index, caches, exports, backups
   □ 2. Extract text + metadata FROM THE INDEX as a recovery corpus
        (vectors are not text — this only works if you stored the
         chunk text in the payload, which is why you store it)
   □ 3. Snapshot immediately, then treat as irreplaceable
   □ 4. Re-source from upstream systems where possible
   □ 5. This is the scenario where storing chunk text alongside
        the vector pays for itself — a vectors-only index is
        unrecoverable
```

**The design decision hidden in RESTORE-4:** *"Store the chunk text in the vector payload, not just the ID and the vector. It costs maybe 500 bytes per chunk — five gigabytes on a ten-million-chunk index — and it's the difference between an index that is a cache and an index that is a second copy of your corpus. If someone argues for ID-only payloads to save space, this is the scenario that settles it."*

### SOP-4 — DR drills

```
   CADENCE: quarterly, minimum. Monthly for T0 assets.

   DRILL 1 — RESTORE (monthly, 30 min)
     □ Restore one T0 asset from object storage into a scratch location
     □ Verify the hash manifest matches
     □ Time it. If it exceeds the RTO, the RTO is fiction.

   DRILL 2 — REBUILD (quarterly, half day)
     □ Rebuild a 1% slice of the vector index from source
     □ Use the PINNED recipe, not the current config
     □ Extrapolate the time and cost to a full rebuild
     □ Compare against the documented RTO — does it still hold?

   DRILL 3 — REGION FAILOVER (quarterly, game day)
     □ Announce it. Unannounced drills destroy trust.
     □ Fail traffic to region B at a quiet hour
     □ Serve production traffic from B for 30–60 minutes
     □ Measure: detect time, decide time, serve time, first success
     □ Verify safety: guardrails live, gate live, PII scan live
     □ Verify quality: sampled score, retrieval recall
     □ Fail back. Measure that too — failing back is a second event.

   DRILL 4 — PROVIDER SUBSTITUTION (quarterly)
     □ Route 100% of traffic to the secondary provider for 1 hour
     □ Run the cross-provider eval
     □ Measure: parse failures, quality delta, cost delta, latency delta

   DRILL 5 — INVENTORY AUDIT (quarterly, 1 hour)
     □ For each asset row, ask "show me the restore command"
     □ Any asset without a demonstrated command → new finding
     □ Track findings to closure; findings-per-drill is a real metric
```

**Why drills find what monitoring cannot:** *"The things that break in a real DR event are almost never the things on the dashboard. They're the missing asset, the undocumented dependency, the engineer who knew the password, and the quota nobody raised. All four are discoverable only by trying to rebuild, which is why the drill is the control and the backup is just a prerequisite."*

### SOP-5 — The cache-cold cost trap

A subtle AI-specific DR failure: failover works, latency and availability are fine, and the bill explodes.

```
   NORMAL STATE
     semantic cache hit rate        ~35%
     prompt cache hit rate          ~72%
     cost per request               $0.011

   AFTER FAILOVER (cold caches)
     semantic cache hit rate        0%   (different store or cold)
     prompt cache hit rate          0%   (cache is per-provider)
     cost per request               $0.021  (+91%)
     AND p95 latency up ~30% (no cache short-circuit)

   ⇒ A successful failover can nearly double unit cost for hours.
   ⇒ Budget for it: pre-warm the cache in region B, or accept
     a temporary cost spike and alert on it so nobody panics.

   MITIGATION RANKING
     ① Pre-warm region B's semantic cache with the top-N queries
        (top 5,000 queries ≈ 40–60% of cache-hit volume)
     ② Keep region B's prompt cache warm with a synthetic canary
        traffic stream at ~1% of production
     ③ Accept the spike, alert on it, and document the expected
        magnitude so the on-call doesn't chase it as a bug
```

**And the prompt cache is provider-scoped, which people miss:** *"Prompt caching is per-provider and often per-account. Failing over to provider B means every cached prefix is cold, and if your system prompt is 3,000 tokens on every request — which it usually is — that's a real cost and latency difference for the first hours. The way to keep it warm is a synthetic canary: continuous low-volume traffic to provider B so its cache stays populated. That's part of the redundancy cost, and it's cheap relative to the spike."*

### SOP-6 — Agent state durability and recovery

Long-running agents need DR at the *task* level, not just the system level.

```
   THE PROBLEM
     An agent task runs for 45 minutes across 20 steps and 6 tools.
     The node dies at step 14. Three of those tools had side effects.

   THE FAILURE MODES, and why a state snapshot is not enough
     ✗ In-memory state         → everything lost, restart from zero
     ✗ Checkpoint between steps → unsafe window: the side effect
                                  committed but wasn't recorded
     ✗ Naive retry             → duplicates the side effects

   THE ANSWER: durable execution
     · The agent loop is a WORKFLOW
     · Each model call and tool call is a DURABLE ACTIVITY
     · Results are journaled; replay returns the recorded result
     · Side-effecting activities carry IDEMPOTENCY KEYS derived
       from workflow_id + step_id
     · Timers and human-approval pauses are DURABLE — a workflow
       can wait hours with no compute held

   THE RECOVERY PROPERTY
     Crash at step 14 → replay steps 1–13 from the journal
     (no re-paid tokens, no re-fired side effects) → resume at 14
```

**The determinism constraint, stated precisely:** *"Recovery works by replay, so during a replay the steps taken must match the steps in the log. That's why workflow code itself must be deterministic — no reading the clock, no random numbers, no direct network calls, no unseeded UUIDs inside the workflow. All nondeterminism and all side effects get pushed into activities, whose results are recorded once and replayed thereafter."*

**The versioning hazard, which is the operational cost of the whole approach:** *"Because long-running workflows replay old histories, changing the workflow code can break replay and cause incidents. That's the single most-cited operational hazard of durable execution, and it's why I'd only take on a full engine — Temporal-class — when the agent has irreversible side effects, must pause for human approval across restarts, or is long enough that re-running on a crash is unacceptable. For a mostly-read-only research agent with idempotent tools, framework checkpointing plus idempotency keys on the few non-idempotent calls is enough and much cheaper to operate."*

| Approach | Recovers | Gives exactly-once side effects | Ops cost |
|---|---|---|---|
| In-memory | Nothing | No | None |
| Framework checkpointer (LangGraph) | State at each super-step | No | Low |
| DBOS / Restate (library) | State + journal | **Yes** | **Low–medium** |
| Temporal (full engine) | State + journal + timers + signals | **Yes** | High |
| Step Functions (declarative) | State machine | Yes, for supported integrations | Managed |

**The decision order:** *"First, does any tool call have an irreversible external side effect — a payment, an email, a deploy, a write to another system? If no, checkpointing plus a retry queue with a dead-letter handler is enough. If yes, can a run outlast a deploy, or must it pause for approval across restarts? If yes, I need durable timers and durable pauses. Then, would re-running the whole agent on a crash be unacceptable in cost, duplicate effects, or lost multi-hour progress? If yes, durable execution is justified — and I'd pick the lightest weight class that satisfies it, which usually means the library-on-Postgres option rather than a cluster."*

---

## 6. Failure modes and mitigations

| # | Failure | Symptom | Root cause | Mitigation |
|---|---|---|---|---|
| 1 | **Eval set lost** | Cannot measure quality at all | Never classed as irreplaceable | T0 replication + object lock + git-lfs |
| 2 | **Chunker recipe lost** | Index rebuilds with different quality | Only the corpus was backed up | Pin chunker + embedding version in git |
| 3 | **Embedding model deprecated** | Cannot rebuild the index identically | Unpinned or unpinned-and-gone model | Pin by snapshot; archive the model where self-hosted |
| 4 | **Adapter lost** | Fine-tuned behaviour gone | LoRA treated as a build artifact | Versioned object storage, per-run retention |
| 5 | **Region B has no GPU quota** | Failover stalls for days | Quota never raised in the DR region | Reserve capacity or plan for API degradation |
| 6 | **Provider B fails the eval** | Fallback produces worse answers | Never run cross-provider eval | Monthly cross-provider eval; 25% steady traffic |
| 7 | **Cold cache cost spike** | Bill doubles after failover | Semantic + prompt caches are per-region/provider | Pre-warm; alert on expected magnitude |
| 8 | **Index rebuilt in place** | Silent mix of old and new answers | No alias flip | Always build alongside, dual-index eval, alias flip |
| 9 | **Corpus deleted upstream** | Index is the only copy, unsnapshotted | Assumed rebuildability | Store chunk text in payload; treat as T0 on loss |
| 10 | **Agent task restarts from zero** | 45 min of work lost, tokens re-paid | In-memory state | Durable execution or checkpointing |
| 11 | **Duplicate side effects on resume** | Two refunds after a crash | No idempotency keys | Deterministic keys from workflow+step IDs |
| 12 | **Replay breaks after a deploy** | Workflows fail to resume | Non-deterministic workflow code | Determinism rules; version workflows carefully |
| 13 | **Human approval pause lost on deploy** | Approval wait dies | In-process pause | Durable timers + signals |
| 14 | **Memory store restored inconsistently** | Agent recalls contradictory facts | Restored without validation | Validate on restore; replay to a consistent point |
| 15 | **Tenant ACL snapshot missing** | Cannot rebuild with correct isolation | ACLs resolved live, never captured | Hourly ACL snapshot; store ACL per chunk |
| 16 | **Failover skips safety** | Guardrails/gate not live in region B | Recovery runbook omitted them | **Explicit verify step before opening traffic** |
| 17 | **Fail-back is a second outage** | Second incident an hour later | Fail-back treated as automatic | Fail-back is a planned change with its own window |
| 18 | **Backup exists but never restored** | Restore fails during the incident | No drill | **Untested = unprotected**; monthly restore drill |
| 19 | **Undocumented dependency** | Restore fails on a missing secret/vendor | Not in the inventory | Quarterly inventory audit |
| 20 | **Provider account suspended** | Total loss of the primary model | Billing failure, ToS event | Second funded account; alert on provider account health |

**Failure 16 in full, because it is the one that turns a DR event into a liability:**

```
   THE SCENARIO
     Region A is lost. The team follows the failover runbook.
     Region B's serving tier comes up. Traffic is shifted.
     The system answers.

     But region B was scaled to minimum two months ago, its
     guardrail classifier service was never deployed there,
     and the action gate in B is pointing at a stale config
     where the tier ceiling is permissive.

   The system is UP. It is also UNGATED and UNGUARDED.

   ⇒ The failover runbook MUST have a safety verification step
     that blocks traffic until it passes:
       □ input guardrails responding
       □ output guardrails responding
       □ action gate responding AND at the correct tier ceiling
       □ PII egress scan live
       □ the degradation level is FULL (not a leftover MINIMAL)

   ⇒ And it must be in the DRILL, because a check that has never
     been run during a drill is a check that will be skipped
     during an incident.
```

---

## 7. Capacity and cost of DR

### 7.1 Cost of the DR posture, by strategy

| Strategy | Standby cost | RTO | RPO | When |
|---|---|---|---|---|
| **Backup & restore** | ~2–5% of primary | 10–24 h | 1–24 h | T3 assets only |
| **Pilot light** | 10–15% | 1–4 h | 5 min – 1 h | T2, cost-sensitive |
| **Warm standby** | 25–40% | 15–60 min | ≤ 5 min | T1, most production AI |
| **Active-active** | 90–130% | ~0 | ~0 | T0, regulated, four nines |
| **Reserved GPU standby** | 60–100% of the pool | 5–15 min | 0 | Self-hosted only |

**The AI-specific cost lines:**

| Item | Cost |
|---|---|
| Vector DB cross-region replica (97 GB) | ~$150–600/month |
| T0 object storage, multi-region, object-lock (50 GB) | ~$15–40/month |
| Reserved GPU standby (8× H100 class, region B) | **~$18,000–35,000/month** |
| Provider B steady-state traffic (25%) | 25% of inference spend |
| Cross-region egress for replication | ~$0.02/GB → ~$10/month at 500 GB |
| DR drills (engineer time + traffic) | ~40 engineer-hours/quarter |
| **Typical total** | **~5% of primary run cost + 25% of inference** |

**The reserved-GPU line is the one that decides the architecture:** *"Reserved standby accelerators in a second region cost roughly sixty to a hundred percent of the primary pool, because you pay for them whether or not a disaster happens. That's usually what pushes a self-hosting team to accept provider-API degradation as their DR path — the standby GPU bill is larger than their entire annual inference spend would be on APIs for the same work. I'd rather state that trade explicitly than pretend a region failover is free."*

### 7.2 Rebuild economics — the numbers to have ready

```
   10,000,000 chunks

   Storage
     vectors:   10M × 1536 dims × 4 bytes        = 61.4 GB
     HNSW overhead ×1.5                          = 92.2 GB
     metadata:  10M × 500 bytes                  =  5.0 GB
     ─────────────────────────────────────────────────────
     total                                       ≈ 97 GB

   Rebuild
     embedding: 10M × ~300 tokens = 3.0B tokens
                × $0.02/1M                      = ~$60
     index build (HNSW M=32, ef_c=256)          = 4–6 h compute
     chunking + orchestration                    = 1–2 h
     embedding throughput at 500 chunks/s        = 5.6 h
     ─────────────────────────────────────────────────────
     cold rebuild end-to-end                     ≈ 10–12 h, ~$60–150

   ⇒ Backing this up costs ~$150–600/month forever.
   ⇒ Rebuilding it costs ~$100 and half a day, ONCE, when needed.
   ⇒ The backup is the wrong choice. The TESTED REBUILD SCRIPT is
     the right one — and "tested" is the whole word.
```

---

## 8. Metrics and alerts

```yaml
dr_alerts:
  # ── Critical ──────────────────────────────────────────────
  - name: t0_replication_lag
    condition: cross_region_replication_lag_seconds > 300
    severity: critical
    action: page_data_oncall
    note: T0 assets must have RPO 0

  - name: t0_restore_test_failed
    condition: restore_drill_success == false
    severity: critical
    action: page_platform_lead
    note: untested = unprotected

  - name: backup_missed
    condition: backups_completed < backups_scheduled
    severity: critical

  - name: backup_stale
    condition: newest_backup_age_hours > rpo_hours × 3
    severity: critical

  - name: unrecoverable_asset_found
    condition: asset_without_tested_restore > 0
    severity: critical
    note: from the quarterly inventory audit

  - name: region_health_degraded
    condition: synthetic_checks_failing_regions >= 2
    severity: critical
    action: incident_commander, evaluate_failover

  - name: safety_check_missing_after_failover
    condition: failover_complete AND (guardrails_live == false OR gate_ceiling != expected)
    severity: critical
    action: STOP_TRAFFIC
    note: an unguarded failover is not a recovery

  # ── Warning ───────────────────────────────────────────────
  - name: dr_drill_overdue
    condition: days_since_last_drill > 90
    severity: warning

  - name: index_rebuild_time_drift
    condition: rebuild_drill_extrapolated_hours > documented_rto × 1.2
    severity: warning
    note: the RTO is stale; re-document or fix the pipeline

  - name: cross_provider_eval_regression
    condition: provider_b_quality_delta > 2.0 points
    severity: warning
    note: the fallback is no longer viable

  - name: standby_capacity_insufficient
    condition: region_b_max_capacity < peak_qps × 0.8
    window: daily
    severity: warning

  - name: post_failover_cost_spike
    condition: cost_per_request > baseline × 1.6
    window: 2h after failover
    severity: warning
    note: expected — cold semantic and prompt caches

  - name: provider_account_health
    condition: provider_billing_status != healthy OR provider_quota_used > 0.9
    severity: warning
    note: account suspension is a real outage cause

  - name: ingest_freshness_lag
    condition: index_freshness_lag > sla
    window: 1h
    severity: warning
```

**The DR dashboard:**

| Panel | Content |
|---|---|
| Asset coverage | Every asset row with class, RPO, RTO, last restore test, status |
| Backup freshness | Heatmap by asset, coloured against its RPO |
| Replication lag | Cross-region, per asset |
| Drill calendar | Last completed, next due, findings open |
| Standby capacity | Region B max vs required peak |
| Provider viability | Cross-provider eval delta, per provider |
| Failover readiness | Checklist state — is every step currently satisfiable? |

**The last panel is the unusual one and worth explaining:** *"A failover checklist is only useful if every item on it is currently satisfiable. So I make it a live dashboard: is region B deployed at the right version, does it have the quota, are its secrets current, is its gate config matching. Any red item means the DR plan is partially fictional, and I'd rather know that on a Tuesday morning than during an incident."*

---

## 9. Change management

| Change | Owner | Reviewer | Notice | Gate |
|---|---|---|---|---|
| Add an asset to the inventory | Asset owner | Platform | — | Documented restore command |
| Change an asset's class | Platform | Data owner | 1 week | Re-derive the DR strategy |
| Change RPO/RTO | Platform + product | Product | 2 weeks | Cost re-derivation |
| **Change backup frequency** | Platform | — | 1 week | Cost + storage check |
| Change the embedding model | ML owner | Platform + eval | 1 quarter | **Full index rebuild + dual-index eval** |
| Change the chunker config | ML owner | Eval owner | 1 month | Full rebuild + quality comparison |
| Change retention | Privacy | Legal | 2 weeks | Regulatory check |
| **Change a model pin** | Platform | Eval owner | Per release | Cross-provider eval if changing provider |
| Add a provider | Platform | Security + legal | 1 quarter | DPA + full eval pass |
| Change the failover runbook | Platform | Incident commander | 1 week | **Run it in the next drill** |
| Change the DR region | Platform | Eng lead | 1 quarter | Capacity + quota + data residency |

**The runbook rule:** *"A runbook change that hasn't been exercised is a hypothesis. So the review gate for any failover-runbook change is that it gets executed in the next drill before it's considered live. That's the only way changes to incident procedures get validated — you can't code-review them into correctness."*

**The data-residency addition:** *"A DR region is a data-residency decision as well as a capacity decision. If my primary is in the EU and my DR is in the US, I've moved personal data across a border as a side effect of a resiliency plan, and that needs legal sign-off and probably a different architecture — replication of derived artifacts is fine, replication of raw personal data is a transfer."*

---

## 10. Audit and compliance

| Record | Contents | Retention |
|---|---|---|
| Asset inventory | Class, RPO, RTO, owner, restore command | Live; versioned quarterly |
| Backup logs | Timestamp, scope, size, hash, result | 1 year |
| Restore records | Drill date, asset, duration, outcome, findings | 3 years |
| Failover events | Trigger, timestamps per step, who authorised | 3 years |
| Data-residency map | Which assets live in which region, legal basis | Live |
| T0 access log | Who read/wrote the irreplaceable assets | 3 years |
| Immutability config | Object-lock settings, retention mode | Life of the system |
| DR findings register | Issue, owner, due date, closure evidence | 3 years |

**Why failover events are auditable artifacts:** *"A failover moves personal data between regions and changes who can process it. So the record needs the trigger, the authorising person, and the timestamps of each step — because the first regulatory question about a cross-border event is 'when exactly did processing move, and under what authority'."*

**Object-lock in compliance mode is a legal commitment:** *"Object-lock compliance mode is not a setting you can undo — not by an administrator, not by the account owner, not by the vendor, until the retention period expires. That's exactly what makes it right for T0 assets and exactly why it needs a legal review before you turn it on, because you may be committing to retain data you later wish you could delete. The GDPR tension is real: a deletion request cannot reach into a compliance-locked copy, which is a defensible position only if you can show the lock is a legal-hold or an audit obligation rather than a convenience."*

---

## 11. Interview Q&A

**Q1. How is DR for an AI system different from DR for a normal web application?**
Three ways. First, the asset inventory is different — an AI system has embeddings, vector indexes, adapters, prompt libraries, eval sets, human-labelled data, and agent event histories, and none of those live in the database you'd normally back up. Second, and more important, the right DR strategy depends on recoverability rather than size. A 97-gigabyte vector index is *rebuildable* from source for about sixty dollars and half a day, so I don't back it up — I test the rebuild script. An eight-gigabyte human-labelled eval set is *irreplaceable*, because it represents months of expert time, and it gets the strongest protection in the system. Third, you can't fail over a third-party model. There's no standby replica of a frontier API in another region — provider failover means switching to a *different model* with different tokenisation and tool-call formats, which is a capability change you need evals for, not a continuity event. And GPU capacity is region-scarce, so a DR region without reserved capacity is a hope rather than a plan.

**Q2. What's your RPO and RTO for an AI system?**
It's per asset, not per system, because a single number would be wrong for almost everything. I use four tiers. T0 is the irreplaceable stuff — eval sets, human labels, red-team corpora, adjudication records — with an RPO of zero and an RTO of fifteen minutes, because losing them isn't a delay, it's a permanent capability loss. T1 is prompts, configs, manifests, adapters, and agent state: RPO five minutes to an hour, RTO one hour. T2 is the vector index, which is rebuildable: RPO of a day for the source, RTO of four to twelve hours. T3 is caches, which I don't protect at all. The pairing that matters is that the tier is set by irreplaceability, so the smallest assets often get the strongest protection, which is the opposite of ordinary backup intuition.

**Q3. What is the most irreplaceable asset in an AI system?**
The human-labelled eval set, and it's not close. The index costs sixty dollars and an afternoon. The model is an API call. But an eval set of a few thousand examples annotated by domain experts is months of work, the rubric behind it is expert-authored, and the adjudication records — where annotators disagreed and how it was resolved — are the actual ground truth. And if those annotators have left the company, no amount of money reconstructs it. So it gets cross-region synchronous replication, object-lock immutability, versioning where every write is a new version, two-person authorisation for deletion, and a weekly hash manifest that's verified monthly. I'd spend more protecting eight gigabytes of labels than a hundred gigabytes of index, and I'd say that explicitly because it's counterintuitive.

**Q4. Why wouldn't you back up the vector index?**
Because it's a derived artifact, and backing it up is the expensive way to solve a problem that rebuilding solves cheaply. Ten million chunks at 1536 dimensions is about 97 gigabytes with HNSW overhead; a cross-region replica runs $150 to $600 a month forever. Rebuilding costs roughly $60 of embedding plus four to six hours of index build and a couple of hours of orchestration — call it ten to twelve hours and under $150, once, on the rare occasion you need it. So the protection isn't a backup, it's a *tested rebuild pipeline* — and tested is the load-bearing word, because a rebuild script nobody has run in eight months is not a recovery plan. The catch is that "rebuildable" is only true if you have the recipe: the chunker configuration, the embedding model version pinned exactly, and the normalisation rules. Backing up the corpus without the recipe is backing up flour and calling it bread.

**Q5. What about agent state and long-running tasks?**
That's where ordinary DR genuinely doesn't apply, because an agent task is a long-running, stateful, side-effecting process. If it crashes at step 14 of 20 and three of those steps had side effects, an in-memory restart loses everything, a checkpoint between steps still leaves an unsafe window between the side effect committing and it being recorded, and a naive retry duplicates the side effects. The answer is durable execution: the agent loop becomes a workflow, each model call and tool call becomes a durable activity whose result is journaled, and side-effecting activities carry idempotency keys derived from the workflow and step ID. On a crash you replay the journal to reconstruct state and resume at step 14 — no re-paid tokens, no re-fired side effects. The cost is determinism: workflow code can't read the clock, can't use unseeded randomness, can't make direct network calls, because recovery works by replay and the steps must match the log.

**Q6. When is durable execution overkill?**
When the agent has no irreversible side effects, runs short enough to fit inside a process and deploy cycle, and would be fine to simply re-run on failure. A read-only research or summarisation agent with idempotent tools doesn't need a Temporal-class engine. There I'd use framework-native checkpointing to recover state, plus idempotency keys on the few non-idempotent calls, plus a retry queue with a dead-letter handler. The determinism constraints and especially the *versioning* hazard — long-running workflows replay old histories, so changing workflow code can break replay and cause incidents — are a real operational cost, and I'd only take them on once the agent has irreversible effects, needs to pause for human approval across restarts, or is long enough that re-running on a crash is unacceptable. If operational overhead is the blocker, the library-on-existing-Postgres approach gets most of the value without running a cluster.

**Q7. How do you fail over a third-party LLM provider?**
I'd reframe it: you don't fail over, you substitute, and that's a capability change rather than continuity. There's no replica of a frontier API — what you have is a pinned snapshot on the same provider, which is a deprecation risk, or a different model on a different provider, which behaves differently. So the requirements are: run 25% of steady-state traffic through the secondary so the path is exercised and the differences are known before the incident; normalise the boundary so tokeniser and tool-call format differences don't surface as parse failures; and run the full eval suite against every provider monthly, because a fallback that fails the eval isn't a fallback, it's a second outage with a different error message. My gate is that provider B has to stay within two points of the primary's quality score or it isn't considered viable, and if it doesn't pass I degrade to a smaller model on the primary provider instead of shipping worse answers.

**Q8. What's the biggest DR risk that people miss?**
The cold-cache cost and latency trap after a failover. The failover works, availability is fine, and then the bill nearly doubles and p95 latency jumps thirty percent, because the semantic cache is cold and the prompt cache is per-provider and per-account. If your system prompt is three thousand tokens on every request — which it usually is — provider B's cache is empty for the first hours, so you pay full input price on every call and lose the cache short-circuit on latency. On a system where caching is worth about a third of the cost, that's a real event. The mitigations are pre-warming region B's semantic cache with the top few thousand queries, keeping the prompt cache warm with a synthetic canary at around one percent of traffic, and documenting the expected spike magnitude so on-call doesn't chase it as a bug.

**Q9. How do you test a DR plan?**
Drills, quarterly at minimum, and I'd break them into five distinct exercises rather than one big one. A monthly restore drill that takes one T0 asset out of object storage into a scratch location and verifies the hash manifest and times it. A quarterly index-rebuild drill that rebuilds a one percent slice from source using the *pinned* recipe and extrapolates the full cost and time, which is how you find out your documented RTO has gone stale. A quarterly region-failover game day — announced, because unannounced drills destroy trust — serving production traffic from region B for thirty to sixty minutes, measuring detect time, decide time, and serve time separately, verifying that the guardrails and the action gate are live before traffic opens, then failing back and measuring that too. A quarterly provider substitution where a hundred percent of traffic goes to the secondary for an hour with the cross-provider eval run. And a quarterly inventory audit that asks, for every asset, "show me the restore command" — because the things that break a real DR event are the missing asset, the undocumented dependency, and the quota nobody raised, and all three are discoverable only by trying to rebuild.

**Q10. A region fails mid-task for a user's agent. What happens?**
Three things need to be true. First, the task must not restart from zero — that's durable execution: the journal replays and it resumes from the last completed activity. Second, the side effects already taken must not be duplicated — that's the idempotency keys. And third, the user has to be told accurately what happened, which means the task ends in an explicit terminal state — completed, partial, or failed — never an ambiguous "done". A partial result enumerates exactly what was done and what wasn't, so the caller can see where it stopped. The worst outcome isn't a failure, it's a task reporting success having completed fourteen of twenty steps, because the user then trusts an incomplete outcome. And if the failover moves processing to a different region, that's a data-residency event that needs to be logged with its authorising person and timestamps.

**Q11. How would you protect against a provider deprecating the model you pinned?**
Layers. First, pin the snapshot rather than a floating alias, and record the served snapshot on every request so a silent substitution is a critical alert rather than a discovery. Second, monitor deprecation notices as a first-class input — most providers announce months ahead, so this is a planning problem, not an incident. Third, keep a funded, evaluated secondary provider so substitution is available in minutes. Fourth, run the cross-provider eval monthly so I know whether the fallback is currently viable rather than assuming it. And fifth, if the model is self-hosted or the provider allows it, archive the weights for the pinned revision — that's the only case where you truly control the artifact, and it's one argument in favour of self-hosting for a system where a specific model's behaviour is load-bearing.

**Q12. What about the corpus itself?**
That's the case where the index stops being a cache and becomes the only copy. If the source documents are deleted upstream — a vendor pulls a knowledge base, a wiki is wiped, a contract ends — the index is no longer rebuildable, it's the last remaining copy, and it becomes a T0 asset retroactively. Two design decisions protect against that. Store the chunk text in the vector payload rather than just the ID and the vector, which costs about five hundred bytes per chunk — five gigabytes on ten million chunks — and is the difference between an index that's a cache and one that's a second copy of the corpus. And version and retain the source corpus in object storage from the moment it enters the pipeline, because you cannot know at ingest time which source will disappear in two years.

**Q13. How much should a DR plan cost?**
It scales with what a failure costs, and I'd set the posture by tier rather than applying one strategy everywhere. Backup-and-restore is roughly two to five percent of primary run cost and gives a ten-to-twenty-four-hour RTO, which is fine for a benchmark artifact. Warm standby is twenty-five to forty percent and gets you to fifteen to sixty minutes, which is where most production AI systems should sit. Active-active is ninety to a hundred and thirty percent and buys near-zero RTO, and I'd only pay it for a regulated tier-zero system. The AI-specific line item that usually decides the architecture is reserved GPU standby in the second region, which for a self-hosted model runs sixty to a hundred percent of the primary pool because you pay whether or not anything happens — that bill is often larger than the entire API cost of the same workload, which is why most teams accept provider-API degradation as their DR path. I'd state that trade explicitly rather than implying a region failover is free.

**Q14 (pushback). This all sounds like a lot of ceremony. We're on a managed vector DB and a managed LLM API — isn't DR just their problem?**
Partly, and I'd be precise about which part. The managed services cover the availability of *their* infrastructure — the vector DB has its own replication and the provider has its own regions. What they don't cover is everything that's yours: whether your eval set survives, whether you know how your index was chunked so it can be rebuilt, whether your adapters are versioned, whether your agent tasks can resume, whether your prompts and manifests are in version control, and whether your quota in the second region was ever raised. Those are all your assets and your problem. And the failure modes I'd actually expect aren't dramatic region losses — they're the accidental index deletion, the corpus that disappeared upstream, the provider that deprecated a model, and the engineer who deleted a collection while testing. So I'd scale the ceremony: for a small system on managed services, that might be T0 replication, prompts in git, a documented rebuild script, and one drill a year. What I wouldn't do is assume the managed service covers an asset it never had.

**Q15. What's the one question you'd ask a team about their DR plan?**
"If region A disappeared right now, show me the command that rebuilds each thing you need." That single question finds the missing asset, the undocumented dependency, the lost recipe, the untested script, and the quota nobody raised — all five of the things that actually make DR fail. A backup that has never been restored isn't a backup, it's a hope, and the only way to convert it into a plan is to have run the command.

---

## 12. Cheat-sheet summary

| Dimension | Value |
|---|---|
| **Three asset classes** | **Rebuildable · Restorable · Irreplaceable** |
| **Protection rule** | Effort is inversely proportional to size, proportional to **irreplaceability** |
| **REBUILDABLE** | Embeddings, HNSW index, BM25, semantic cache, reranker cache |
| **RESTORABLE** | Prompts, configs, manifests, adapters, agent state, memory |
| **IRREPLACEABLE** | **Eval sets, human labels, red-team corpora, adjudications, rubrics** |
| **THIRD-PARTY** | Frontier APIs — **you can only substitute, never replicate** |
| **T0** | RPO **0** · RTO **15 min** · eval sets, labels, corpora, source |
| **T1** | RPO **5 min–1 h** · RTO **1 h** · prompts, configs, adapters, agent state |
| **T2** | RPO **1 h** · RTO **4 h** · vector index, memory stores |
| **T3** | RPO **24 h** · RTO **24 h** · caches, benchmarks |
| **Index size (10M chunks)** | **~97 GB** (61.4 GB vectors + 1.5× HNSW + 5 GB metadata) |
| **Index rebuild** | **~10–12 h · ~$60–150** |
| **Index backup cost** | $150–600/month forever — **so don't** |
| **The recipe is the backup** | Chunker config + **pinned embedding version** + normalisation |
| **The recipe loss failure** | Backing up the corpus without the chunker = flour, not bread |
| **RAPTOR tree rebuild** | ~$2,000 LLM cost — rebuildable but snapshot it instead |
| **Chunk text in payload** | ~500 B/chunk = 5 GB on 10M — **the index must be able to become the source** |
| **Provider failover is** | **A capability change, not continuity** |
| **Fallback viability gate** | Within **2 quality points** of primary; schema ≥ 0.995; tool parse ≥ 0.999 |
| **Secondary traffic** | **25% steady state** — never 0% |
| **Prompt cache** | **Per-provider and per-account** — cold after failover |
| **Cold-cache effect** | Cost **+~91%**, latency **+~30%** until warm |
| **Cache pre-warm** | Top 5,000 queries ≈ 40–60% of hit volume |
| **GPU DR reality** | Region B quota may be 0; **reserved capacity or API degradation** |
| **Quota lead time** | 2 days – 3 weeks · new region 4–12 weeks · reserved GPU 4–16 weeks |
| **Reserved GPU standby** | **60–100% of primary pool cost** — the architecture decider |
| **DR postures** | Backup 2–5% · pilot light 10–15% · **warm standby 25–40%** · active-active 90–130% |
| **Index rebuild rule** | **Build alongside, dual-index eval, atomic alias flip** |
| **nDCG block** | > 3-point drop blocks the flip |
| **Durable execution** | Agent loop = workflow · model/tool call = **durable activity** |
| **Exactly-once** | Idempotency key from **workflow_id + step_id** |
| **Determinism rule** | No clock, no random, no direct network in workflow code |
| **Durable execution cost** | Determinism + **versioning hazard** (replay breaks on code change) |
| **Decision order** | Irreversible side effect? → Outlasts a deploy? → Re-run unacceptable? |
| **Agent terminal states** | completed · **partial** · failed — never ambiguous "done" |
| **Failover safety gate** | **Verify guardrails + gate + PII scan BEFORE opening traffic** |
| **Failover without safety** | System UP and **UNGATED** — worse than an outage |
| **Fail-back** | A **second planned event**, not an automatic reversal |
| **Drills** | Restore monthly · rebuild/region/provider/inventory quarterly |
| **The DR finding metric** | Findings per drill — should trend to zero, never spike to zero |
| **Untested backup** | **A hope, not a backup** |
| **The one question** | **"Show me the command that rebuilds this."** |

---

## 13. Cross-links

- [Reliability Patterns](07-reliability-patterns.md) — steady-state failover, breakers, and the degradation ladder
- [Capacity Planning and Autoscaling](05-capacity-planning-and-autoscaling.md) — standby capacity sizing and quota planning
- [Cost and FinOps](06-cost-and-finops.md) — the cold-cache spike and the cost of redundancy
- [Security, Guardrails and Compliance](08-security-guardrails-and-compliance.md) — credential rotation, data residency, retention
- [Deployment, Release and CI/CD](04-deployment-release-and-cicd.md) — the release manifest as a recovery artifact
- [LLMOps for AI Systems](10-llmops-for-ai-systems.md) — DR inside the operational lifecycle
- [Advanced RAG Patterns](../HLD/05-ai-systems/02-advanced-rag-patterns.md) — index versioning and alias flips
- [RAG Evaluation and Observability](../HLD/05-ai-systems/03-rag-evaluation-and-observability.md) — the eval sets worth protecting
- [Agentic System Design](../HLD/05-ai-systems/06-agentic-system-design.md) — durable execution and checkpointing
- [Memory and Context Engineering](../HLD/05-ai-systems/08-memory-and-context-engineering.md) — memory store durability
- [Operationalisation Cheatsheet](../Cheat-Sheets/06-operationalisation-cheatsheet.md)
