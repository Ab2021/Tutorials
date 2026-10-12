# Autocomplete / Typeahead — Serving Top-K for Every Prefix in Under 100 ms

> **Source note.** Autocomplete is named in the roadmap video (T1, phase 2, as part of the HLD design list). The transcripts name it without designing it, so the design below is **supplementary** — standard industry practice, not quoted from the transcripts. Figures are labelled **[A]** (stated assumption for the exercise) or **[D]** (derived arithmetic).

> **TL;DR.** Autocomplete looks like a search problem and is actually a **precomputation and freshness problem**. You cannot rank suggestions at query time — the latency budget is ~100 ms and the request rate is enormous because every keystroke fires a request. So you **precompute the top-k suggestions for every prefix worth serving**, store them, and serve a lookup. The three things that make it hard are: (1) **the request amplification** — autocomplete QPS is roughly an order of magnitude above search QPS, and it is extremely Zipfian, so caching is the dominant lever; (2) **freshness versus stability** — you cannot rebuild the whole prefix index every minute, so you layer a stable base index with a streaming delta and a trending layer and merge at serve time; (3) **curation** — the index is built from user queries, so without aggressive filtering the system will confidently suggest whatever the internet is currently spamming, which is a reputational event, not a bug.

> **60-second interview answer.** "First I'd pin the numbers, because they set the design. The latency budget is about 100 ms p99 from keystroke to render, and every keystroke can fire a request, so autocomplete QPS is roughly 10× search QPS — on the order of 100,000 QPS average and 300,000 peak in my assumptions. At that rate I cannot rank candidates online. So the architecture is a **two-phase system**: an offline pipeline that aggregates query logs, filters and ranks, and builds an index of top-k suggestions per prefix; and an online path that does a lookup, optionally reranks for personalisation, and returns. The key structural decisions: shard by **prefix range** so a lookup touches one shard, but replicate the hot prefixes everywhere because the distribution is Zipfian; layer the index into a stable base plus a streaming delta plus a trending layer so trending news appears within seconds without rebuilding everything; and filter hard at build time, because the corpus is literally what users typed. Then I'd close on the two things that actually bite: the out-of-order response problem on the client, and the fact that offline recall metrics do not predict whether users pick a suggestion — that needs an A/B test."

---

## 1. Definition & mental model

**Definition.** Autocomplete (typeahead) returns, for a partial input string, an ordered list of completions the user is likely to select — fast enough that it appears while they are still typing.

It is a **ranking system with a hard latency floor and an adversarial corpus**. The latency floor forbids online ranking; the adversarial corpus makes curation a correctness requirement rather than a nicety.

| Property | Value | Consequence |
|---|---|---|
| Latency budget | ~100 ms p99 | Rules out online ranking over a large candidate set |
| Trigger | Every keystroke (debounced) | Request amplification ~10× search |
| Input distribution | Extremely Zipfian | Caching is the dominant optimisation |
| Corpus | User-generated queries | Needs filtering: spam, offensive, PII, illegal |
| Truth signal | Which suggestion was selected | Click-through, not recall, is the real metric |
| Freshness need | Seconds for trends, days for the tail | Forces a layered index |

**Mental model — a printed phone book, not a librarian.** A librarian would look up your request on the spot. A phone book is computed in advance and merely *read* at request time, which is why it can be consulted instantly. Autocomplete is a phone book that is reprinted constantly, in layers — a big stable edition daily, a thin supplement every minute, and a post-it note for whatever is happening right now.

**The second mental model — the corpus is a mirror.** Whatever you index is what people typed. If you index naively, your product will, at some point, prominently suggest something vile during a news cycle. So filtering is not an add-on stage; it is a first-class stage with its own ownership, its own audit trail, and its own incident process.

**What it is *not*.** It is not a search engine (no relevance ranking over a document corpus — you are ranking *queries that other people typed*). It is not a spell-corrector, though they are often combined. And it is not a per-user personalised model in the first instance (personalisation is a second-stage rerank over a global candidate set, because you cannot afford per-user precomputation).

---

## 2. Why it exists — the problem it solves

**For the user.** Typing is slow and error-prone. Offering the exact query in one tap converts a 20-keystroke interaction into a 3-keystroke one, which on mobile is the difference between using the product and abandoning it.

**For the product.** Autocomplete is a **query-shaping instrument**. Because it suggests the queries that previously worked, it nudges users toward high-quality, high-volume, well-indexed queries — which improves downstream search results, reduces zero-result searches, and increases engagement. The suggestion list is one of the highest-leverage surfaces in a search product.

**For the business.** Suggestions are also the natural place for commercial insertion (promoted suggestions, brand terms), which means the ranking function has both a relevance objective and a commercial one — and those must be tuned against each other deliberately.

**What breaks without it.** Typing frustration, especially on mobile; more zero-result searches; slower interaction; and a measurable drop in searches per session.

---

## 3. Internal mechanics

### 3.1 The two-phase split

The single most important architectural decision: **precompute offline, look up online.**

```
════════ OFFLINE / STREAMING PIPELINE (seconds → hours) ════════

  Query logs (every search, with timestamp, locale, user, selection)
        │
        ▼
  ┌────────────────┐   ┌────────────────┐   ┌────────────────┐
  │ AGGREGATE      │──►│ FILTER         │──►│ RANK           │
  │ count per query│   │ spam · adult   │   │ score = f(count,│
  │ per time window│   │ offensive·PII  │   │  recency, CTR,  │
  │ (1h/1d/30d)    │   │ illegal·legal  │   │  quality, rev)  │
  └────────────────┘   └────────────────┘   └────────┬───────┘
                                                     │
        ┌────────────────────────────────────────────┘
        ▼
  ┌──────────────────────────────────────────────────────────┐
  │  INDEX BUILD — insert each query along its prefix chain   │
  │  materialise the top-k list at every SERVABLE node        │
  └────────┬────────────────────┬─────────────────────┬───────┘
           ▼                    ▼                     ▼
    ┌─────────────┐     ┌─────────────┐       ┌─────────────┐
    │  BASE INDEX │     │ DELTA INDEX │       │  TRENDING   │
    │  rebuilt    │     │ streaming   │       │  spike      │
    │  daily      │     │ counts,     │       │  detection, │
    │  (stable,   │     │ ~1 min lag  │       │  ~seconds   │
    │  big)       │     │ (small)     │       │  (tiny)     │
    └──────┬──────┘     └──────┬──────┘       └──────┬──────┘
           └───────────────────┴────────────────────┘
                               │  merged at serve time
═══════════════════════════════│═══════════════════════════════
                               ▼
════════════ ONLINE SERVING (≈100 ms budget) ════════

   keystroke ──► debounce (50–100 ms, client)
                     │
                     ▼
              ┌──────────────┐
              │ EDGE CACHE   │  hot prefixes served here
              │ (CDN / local)│  Zipf ⇒ very high hit rate
              └──────┬───────┘  miss ↓
                     ▼
              ┌──────────────┐
              │ QUERY ROUTER │  prefix range → ONE shard
              └──────┬───────┘
                     ▼
              ┌──────────────────────────────┐
              │ SHARD: prefix-range partition │
              │  trie node → top-N (N≈100)    │
              └──────┬───────────────────────┘
                     ▼
              ┌──────────────┐
              │ RERANK       │  personalisation, locale,
              │ N → k (k≈10) │  commercial insertion
              └──────┬───────┘
                     ▼
              ┌──────────────┐
              │ SAFETY GUARD │  final blocklist check
              └──────┬───────┘
                     ▼
                  top-10
```

### 3.2 The index: a trie with materialised top-k

The natural structure is a **prefix tree (trie)**. Each node represents a prefix; each node on the path of an indexed query stores that query's score.

Naively, to answer a prefix you would traverse the subtree and find the highest-scoring leaves — expensive, and unbounded in the worst case (the prefix `"a"` has an enormous subtree). The fix is to **materialise the top-k list at the node itself**. Then a lookup is O(prefix length) — follow the characters, read the list, done. This is the central trick: you have traded storage and build time for a lookup that is effectively O(1) in the size of the corpus.

**The storage problem that follows.** You cannot materialise top-k at *every* node — the number of distinct prefixes is enormous, and most have negligible traffic.

> Let *D* = distinct queries clearing the volume threshold, and *L* ≈ 20 the average prefix-path length. Nodes ≈ *D × L*. Materialising a top-10 list of ~20-byte strings (~200 B with counts and overhead) at every node gives:

```
Assume D = 100M qualifying distinct queries   [A]
Nodes ≈ 100M × 20                        ≈ 2B nodes      [D]
Full materialisation: 2B × 200 B         ≈ 400 GB        [D]
```

400 GB of hot-ish data is large but shardable — *however*, most of it is useless, because only high-traffic nodes are ever served directly.

**The fix — significant-node materialisation.** Materialise top-k only at nodes whose traffic clears a threshold; below the threshold, fall back to a **bounded subtree walk** (explore the subtree to a depth or node budget and take the best). With the threshold set so that ~10% of nodes qualify:

```
Materialised: 200M × 200 B ≈ 40 GB     [D]  ← 10× reduction
Below-threshold nodes: bounded walk, cheaper and rarely hit
```

This is the storage/latency trade made explicit, and it is exactly the kind of number a strong candidate produces unprompted.

### 3.3 Sharding — and why prefix-range wins

Two candidate schemes:

| Scheme | Lookup cost | Load distribution | Verdict |
|---|---|---|---|
| **Hash by query text** | Must fan out to **every** shard and merge top-k | Uniform per shard, but every request hits all shards | Terrible tail latency — one slow shard slows everything |
| **Partition by prefix range** | **One** shard (the one owning that prefix) | **Skewed** — `"a"`, `"s"`, `"t"` are enormously hotter than `"xyz"` | Correct choice, plus skew mitigation |

Prefix-range partitioning is the right default because a lookup is a single-shard operation. Its cost is skew, handled with three layered techniques:

1. **Consistent hashing over prefix *ranges* with many virtual ranges**, assigning several ranges per physical shard so hot ranges are spread.
2. **Explicit hot-prefix replication** — the top few thousand prefixes (which by Zipf cover a huge share of traffic) are replicated to *every* shard, so any shard can answer them locally.
3. **Edge caching in front of everything** — the Zipfian distribution means a small cache absorbs most of the traffic.

### 3.4 Zipf is the whole performance story

Query traffic follows a Zipfian distribution: rank-*r* query frequency ∝ 1/*r*^s with *s* ≈ 1. That has a decisive consequence.

```
Top 1% of distinct prefixes ≈ 50%+ of all autocomplete requests   [A]
```

So a relatively small edge cache — or a small replicated hot layer — serves the majority of traffic without touching the trie at all. **Caching is not an optimisation layer here; it is the primary serving path**, and the trie is the fallback for the tail. State that explicitly in an interview; it reframes the whole capacity discussion.

### 3.5 Freshness — the layered index

The hardest requirement. You cannot rebuild a 40 GB index every minute, but a breaking-news query must appear within seconds.

| Layer | Rebuilt | Holds | Serves |
|---|---|---|---|
| **Base** | Daily (full rebuild) | All qualifying queries, stable scores | The bulk of traffic |
| **Delta** | Per-minute (incremental) | Recent count changes — new queries, rising ones | Freshness for the tail |
| **Trending** | Per-second (streaming) | Detected spikes, tiny | Breaking news, virality |

**Serving merges the three:** look up the prefix in all layers, union the candidates, take the max score per query, re-sort, return top-k.

**Trending detection is its own subsystem.** The signal is a query's *current* rate compared to its *expected* rate — a spike, not a level. A query with zero history that suddenly appears 10,000 times in 60 seconds is a trend; a query with a steady 10,000/hour is just popular. So:

- Maintain a baseline expectation per query (or per query cohort).
- Compute the current short-window rate.
- Flag when the ratio exceeds a threshold *and* the absolute volume is material.
- Trending queries bypass the normal volume threshold — they have not had time to accumulate one, and this is precisely the case the base index structurally cannot serve.

**The stability counterweight.** Trending injection must be bounded and reversible: cap the share of the suggestion list that trending can occupy, and expire trending entries automatically. Otherwise a coordinated spam campaign can occupy the suggestion list for a term of its choosing — which is a real, well-documented attack.

### 3.6 Personalisation as a second-stage rerank

You cannot precompute per-user suggestions at internet scale — the corpus would be *users × prefixes*. So personalisation is a **rerank** over the global candidate set:

```
Global top-N  (N ≈ 100, from the index)      ~10 ms
        ▼
Rerank for this user                          ~20 ms
  • user's own query history (this prefix, previously typed)
  • locale / language
  • recent session context
  • commercial insertion (promoted, capped)
        ▼
top-k  (k ≈ 10)
```

Fetching N=100 and reranking to 10 is the standard shape: it keeps the expensive per-user computation bounded, and it means the personalisation model can be swapped or A/B tested without touching the index.

### 3.7 Filtering and curation — the stage that prevents incidents

The corpus is user input, which means every abuse category in the product shows up in it. Filtering happens at **build time** (so it can never reach the index) and again as a **serve-time guard** (defence in depth):

| Category | Mechanism |
|---|---|
| Offensive / hateful | Blocklist + classifier, applied to the query before indexing |
| Adult | Classifier + locale-specific policy |
| PII (names, addresses, phone numbers, card numbers) | Pattern detection + suppression of suggestions that appear to be personal data |
| Illegal / regulated | Policy blocklist with legal ownership |
| **Spam / SEO poisoning** | Velocity anomaly detection — a query spiking from many accounts with no organic pattern |
| Defamation / legal takedown | Removal pipeline with an auditable record and an SLA |

**Why build-time matters most:** if a slur never enters the index, no ranking bug, cache-staleness bug, or serve-time failure can ever surface it. Serving-time filtering is a second line of defence, not the first.

**The subtle failure.** Blocklists operate on exact strings, so they are trivially evaded by small mutations — spacing, character substitution, homoglyphs. This is why a classifier must backstop the blocklist, and why the mutation space needs periodic adversarial review.

---

## 4. Variants & taxonomy

| Variant | Mechanism | When it applies | Cost |
|---|---|---|---|
| **Top-k per prefix (trie, materialised)** | Precomputed lists at nodes | The default; the only thing that meets the latency budget at scale | Build pipeline; storage |
| **Bounded subtree walk** | Traverse and score at query time | Low-traffic prefixes, small corpora | Latency grows with corpus size; fine at small scale |
| **Finite-state transducer (FST)** | Compressed mapping from prefix to ranked completions | Very large corpora where memory dominates; supports weighted traversal | Complex to build; harder to update incrementally |
| **n-gram / language model** | Predict next tokens from a language model | Sentence completion, chat input, code completion | Much heavier; adds latency; needs a fallback |
| **Semantic / embedding-based** | Retrieve by vector similarity to the partial input | "Intent" completion rather than literal prefix match | Does not satisfy the literal-prefix contract users expect; usually a complement |
| **Client-side / local** | Index shipped to the device, or browser history | Privacy-sensitive, offline-capable, very low latency | No global trends; index staleness; device storage |
| **Two-stage (global → personalised rerank)** | Global top-N, then per-user reorder | The standard personalisation shape | Extra latency; needs user features at serve time |
| **Federated / multi-source** | Union of several providers (history, trending, catalogue, commerce) | Rich product surfaces | Merge logic; conflicting objectives; latency of the slowest source |

---

## 5. Pros / Cons

### Precomputed top-k over a trie

| Pros | Cons |
|---|---|
| Lookup is O(prefix length) — meets a 100 ms budget easily | Storage grows with corpus × prefix depth |
| Ranking complexity lives offline, where it is cheap | Freshness is limited by rebuild cadence |
| Deterministic, cacheable, easy to reason about | Incremental updates to a trie are fiddly |
| Easy to A/B test by swapping the build | A ranking change requires a rebuild |

### Prefix-range sharding

| Pros | Cons |
|---|---|
| One shard per lookup — no fan-out, predictable tail latency | Severe load skew from Zipfian prefixes |
| Simple routing: prefix → shard is a pure function | Hot ranges need replication, which costs memory on every shard |
| Scales by adding shards for hot ranges | Rebalancing ranges is operationally delicate |

### Layered freshness (base + delta + trending)

| Pros | Cons |
|---|---|
| Breaking news appears in seconds without rebuilding the world | Three indexes to build, merge, monitor and reason about |
| The stable base keeps quality high for the bulk of traffic | Merge logic is a correctness surface — dedupe, score reconciliation |
| Trending can be capped and expired, limiting blast radius | Trending is directly attackable by coordinated spam |

### Personalised rerank

| Pros | Cons |
|---|---|
| Personalisation without per-user precomputation | Adds latency in the tightest part of the budget |
| Model can be A/B tested independently of the index | Needs user features at serve time — a privacy and compliance surface |
| Improves selection rate, the metric that actually matters | Can narrow users into a filter bubble of their own past queries |

---

## 6. Exceptions, edge cases & failure modes

| # | Failure | Mechanism | Mitigation |
|---|---|---|---|
| 6.1 | **Out-of-order responses** | Keystrokes for `"ca"` and `"car"` race; `"ca"`'s response arrives last and overwrites the newer list | Client sequences by request id and discards stale responses; render only if the request matches the current input |
| 6.2 | **Request storm** | Every keystroke fires a request; fast typists multiply QPS | Client debounce (50–100 ms); cancel in-flight requests on new input; minimum prefix length |
| 6.3 | **Empty result for a valid prefix** | The prefix has no qualifying queries above the volume threshold | Fall back to the bounded subtree walk; then to a spelling correction; then return nothing rather than something wrong |
| 6.4 | **Trending spam hijack** | Coordinated accounts push a chosen term into the trending layer | Velocity anomaly detection on *account diversity*, not just volume; cap trending share of the list; automatic expiry; human review for high-risk terms |
| 6.5 | **Offensive suggestion surfaces** | A slur enters the index via a real query spike before filtering catches it | Build-time classifier, not just a blocklist; serve-time guard; a fast removal path with an audit trail |
| 6.6 | **Hot-shard overload** | A single trending prefix concentrates traffic on one prefix-range shard | Hot-prefix replication across shards; edge cache absorbs the burst; trending served from the small replicated trending layer |
| 6.7 | **Stale cache after removal** | A suggestion is removed from the index but remains in edge caches | Short TTLs on suggestive content; an explicit purge API on the cache; version the cache key by index generation |
| 6.8 | **Score staleness in the base layer** | Yesterday's index serves today's reality for non-trending queries | Delta layer captures drift; daily full rebuild bounds the worst case |
| 6.9 | **Merge duplication** | The same query appears in base, delta and trending with different scores | Canonical key per query; take max (or a defined combination) at merge; dedupe before returning |
| 6.10 | **Locale / language mismatch** | A user typing in one script gets suggestions in another | Locale as a routing and reranking input; per-locale indexes; explicit language detection |
| 6.11 | **CJK / no-whitespace languages** | "Prefix" is not well-defined; IME composition complicates the keystroke model | Segment by characters, transliterate (pinyin/romaji) into a parallel index, and trigger on composition events rather than raw keys |
| 6.12 | **Typos in the input** | The user mistypes and the prefix matches nothing | Fuzzy prefix (edit distance 1) as a fallback — expensive, so cap the search and serve it only on a miss |
| 6.13 | **PII leakage** | A query containing a person's name or number becomes a suggestion | Pattern detection, suppression of low-diversity high-specificity queries, and a removal pipeline |
| 6.14 | **Feedback loop / popularity bias** | Suggested queries get selected, gain volume, get ranked higher — starving novel queries | Reserve list slots for non-popularity signals; measure the discovery impact of the ranker |
| 6.15 | **Cold start on a new index** | A fresh build has no CTR data, so ranking degrades | Blend count-based and CTR-based scores; carry CTR forward across builds where the query is unchanged |
| 6.16 | **Metrics blind spot** | Offline recall@k is excellent while selection rate falls | Treat selection rate as the primary metric; A/B test every ranker change on live traffic |

---

## 7. When to use / When NOT to use

**Use precomputed top-k autocomplete when:**

- Latency budget is tight (< 200 ms) and request volume is high.
- The corpus is large but bounded — queries people actually type, not all possible strings.
- You have query logs to aggregate (without logs there is nothing to precompute).
- Literal-prefix matching is the expected interaction.

**Use bounded subtree walk / query-time ranking when:**

- Corpus is small (thousands, not hundreds of millions) — a catalogue, a contact list.
- Traffic is low enough that caching does not save you and precomputation is not worth the pipeline.
- The ranking must depend on inputs unavailable at build time.

**Use a language model rather than a prefix index when:**

- The task is sentence or code completion, where the "prefix" is a context window, not a word.
- The user expectation is generative rather than a fixed list.

**Do NOT build a global autocomplete index when:**

- The user's own history is the dominant signal (a browser URL bar, a personal contacts search) — ship the index client-side.
- Privacy constraints forbid aggregating user queries centrally.
- The candidate set is a small fixed catalogue — a plain filtered list is simpler and correct.

**Decision rule:** *large user-generated corpus + 100 ms budget + query logs exist ⇒ precomputed top-k, layered for freshness, with caching as the primary serving path. Small or private corpus ⇒ local or query-time.*

---

## 8. Numbers, defaults & tuning

> Assumptions marked **[A]**; arithmetic **[D]**.

| Quantity | Value | Basis |
|---|---|---|
| Search queries / day | 2.5B | [A] (500M DAU × 5 searches) |
| Search QPS (average) | 2.5B / 86,400 ≈ **29,000 QPS** | [D] |
| Keystrokes per search | ~8 | [A] |
| Autocomplete requests per search after debounce | ~0.3–0.5 | [A] debounce collapses most keystrokes |
| **Autocomplete QPS (average)** | ~**100,000 QPS** | [D] |
| **Autocomplete QPS (peak, 3×)** | ~**300,000 QPS** | [D] |
| End-to-end latency p99 | **< 100 ms** | [A] |
| — network / RTT | ~20 ms | [A] |
| — index lookup | ~10 ms | [A] |
| — personalised rerank | ~20 ms | [A] |
| — render + margin | ~50 ms | [A] |
| Client debounce | 50–100 ms | [A] |
| Minimum prefix length | 2–3 characters | [A] |
| Suggestions returned (k) | **10** | [A] |
| Candidates fetched from index (N) | **100** | [A] |
| Qualifying distinct queries (D) | 100M | [A] |
| Full trie materialisation | ~400 GB | [D] see §3.2 |
| With significant-node threshold | ~40 GB | [D] |
| Base index rebuild | daily | [A] |
| Delta update lag | ~60 s | [A] |
| Trending detection lag | seconds | [A] |
| Edge cache TTL (stable suggestions) | minutes–hours | [A] |
| Edge cache TTL (trending-eligible) | seconds | [A] |

### The arithmetic that shapes the design

**1. Request amplification — the number that rules out online ranking.**

```
Search QPS          ≈ 29,000
Autocomplete QPS    ≈ 100,000     → ~3.4× search, ~10× before debounce
```

If you debounce 8 keystrokes down to ~0.3 requests per search, you have already removed most of the load — client debounce is the single cheapest capacity win in the system, and it belongs in the design discussion, not the appendix.

**2. Zipf is the caching story.**

```
Top 1% of prefixes ≈ 50%+ of requests   [A]
⇒ a cache holding ~1M prefixes absorbs the majority of 100k QPS
⇒ the trie fleet is sized for the tail, not the average
```

**3. Trending must bypass the volume threshold — by construction.**

```
Normal qualification: query needs sustained volume over a window  → hours to days
Breaking news:        needs to appear in                 → seconds
⇒ structurally impossible in the base layer; requires a separate streaming path
```

This is the argument for the layered index, stated as an arithmetic impossibility rather than a preference. It is the strongest form of the point.

**4. Storage forces selective materialisation.** As computed in §3.2: naive materialisation ≈ 400 GB, thresholded ≈ 40 GB. Both are defensible at scale; the thresholded version is what a candidate who has actually thought about it will propose.

**Metric set to quote:**

| Metric | Meaning | Why it matters |
|---|---|---|
| **Selection rate (CTR on suggestions)** | Share of sessions where a suggestion was chosen | The primary metric — the only one that reflects real utility |
| Keystrokes saved | Input length before vs after selection | The product-facing value |
| p99 latency, keystroke → render | The budget | Above ~100 ms the list arrives after the user has typed past it |
| Cache hit rate | Edge-cache effectiveness | Directly determines fleet size |
| Zero-suggestion rate | Prefixes returning nothing | Quality signal; drives the fallback chain |
| Trending precision | Share of trending entries that were genuine | Guards against spam hijack |
| Removal latency | Time from takedown request to purge everywhere | Compliance and incident metric |
| **Recall@k (offline)** | Overlap with the eventual full query | Useful for regression testing — **never** as the product metric |

---

## 9. Interview Q&A

**Q1. How would you frame this before designing?**
I would establish four things. What is the latency budget from keystroke to render — I will assume 100 ms p99, because above that the list arrives after the user has typed past it. What is the request rate, and specifically how much amplification is there over search, because every keystroke can fire a request. Is the corpus user-generated queries or a fixed catalogue, because that decides whether curation is a correctness requirement. And what is the freshness requirement — do breaking-news terms need to appear in seconds? Those four answers determine the entire architecture, and the third and fourth are the ones candidates forget to ask.

**Q2. What is the single most important design decision?**
Precompute offline and look up online. The latency budget and the request rate together make online ranking impossible at scale — you cannot afford to score candidates per keystroke at 100,000 QPS. So the ranking work moves into a pipeline and the serving path becomes a lookup with a small rerank on top. Everything else — the trie, the sharding, the layers — follows from that split.

**Q3. How do you answer a prefix lookup efficiently?**
A trie, with the top-k list **materialised at the node** rather than computed by traversing the subtree. That makes a lookup O(prefix length) — follow the characters, read the prepared list. The naive alternative, walking the subtree and scoring leaves, is unbounded: the prefix `"a"` has an enormous subtree, so worst-case latency is uncontrollable. Materialisation trades storage and build time for a lookup that is effectively constant.

**Q4. What does that cost you in storage?**
It depends on how much you materialise, and that is a real design lever. With 100M qualifying distinct queries and an average prefix path of 20 characters, that is about 2B nodes; at roughly 200 bytes per materialised top-10 list, full materialisation is on the order of 400 GB. But most of those nodes are never served directly, because traffic is Zipfian. So I would materialise only at nodes above a traffic threshold — around 10% of them — which brings it to roughly 40 GB, and answer the rest with a bounded subtree walk. That is the storage-versus-latency trade, made explicitly.

**Q5. How do you shard it?**
By prefix range, not by hashing the query text. Hashing means a single prefix lookup must fan out to every shard and merge top-k, which makes your tail latency the slowest shard's latency — unacceptable when one slow shard degrades every request. Prefix-range partitioning routes each lookup to exactly one shard, because "which shard owns this prefix" is a pure function. The price is load skew: `"a"`, `"s"` and `"t"` are vastly hotter than `"xyz"`. I would handle that with consistent hashing over many virtual ranges, explicit replication of the hottest few thousand prefixes to every shard, and an edge cache in front, which brings me to caching.

**Q6. Why is caching so central here?**
Because query traffic is Zipfian — the top 1% of prefixes account for over half of all requests. That means a relatively small edge cache absorbs the majority of the 100,000 QPS without touching the trie at all. So caching is not an optimisation bolted on at the end; it is the primary serving path, and the trie fleet is sized for the tail. Framing it that way changes the capacity answer completely: I am not sizing for 300,000 peak QPS against the index, I am sizing for the miss rate.

**Q7. How do you handle a breaking-news query?**
It cannot be handled in the base index, and the reason is arithmetic rather than preference: the base index only includes queries that have accumulated sustained volume over a time window, which takes hours to days, whereas a breaking-news term needs to appear within seconds. So the index is layered. A stable base, rebuilt daily, holds the bulk. A delta layer updated per minute captures drift. And a small trending layer, updated per second from a stream, captures spikes — detected as the ratio of a query's current rate to its expected rate, not its absolute volume, because a steady 10,000-per-hour query is popular, not trending. Serving merges all three, takes the max score per query, dedupes and returns.

**Q8. Isn't the trending layer attackable?**
Yes, directly, and that is a real incident class — coordinated accounts pushing a chosen term onto the suggestion list. So the trending layer needs to be bounded and reversible by design. I would detect velocity anomalies on *account diversity* rather than raw volume, so a thousand accounts searching one term looks different from one account searching it a thousand times. I would cap the share of the suggestion list trending entries can occupy, so they can never own the surface. And every trending entry expires automatically, with high-risk terms routed to human review.

**Q9. How do you prevent offensive suggestions?**
Filter at build time, and treat this as a correctness stage rather than a nicety. If a term never enters the index, no ranking bug, cache staleness bug or serve-time failure can ever surface it — that is why build-time filtering is the primary defence and serve-time checking is the second line. Concretely: a blocklist for known terms, plus a classifier, because blocklists are trivially evaded by spacing, character substitution and homoglyphs. Plus a removal pipeline with an audit trail and a latency SLA, because removals must propagate to edge caches too — which means either short TTLs on suggestive content or an explicit purge API.

**Q10. How does personalisation fit?**
As a second-stage rerank, not as per-user precomputation. Precomputing per user would mean the index size is users × prefixes, which is not viable. So instead: fetch the global top-N — N around 100 — from the index, then rerank those for this user down to k=10 using their own query history, locale, and session context. That keeps the expensive per-user work bounded and, importantly, means the personalisation model can be A/B tested and swapped without rebuilding the index.

**Q11. What is the metric? Is it recall@k?**
Recall@k is useful for offline regression testing — it tells you whether a ranking change broke something obvious — but it is not the product metric, and treating it as one is a classic mistake. The metric that matters is **selection rate**: what share of sessions end with the user picking a suggestion. A ranker can improve recall@k while making suggestions less useful, because recall measures overlap with the final query rather than whether the list helped. I would A/B every ranker change on live selection rate, with keystrokes-saved as the secondary product metric.

**Q12. What is the trickiest client-side problem?**
Out-of-order responses. The user types `"ca"` then `"car"`; two requests are in flight; the response to `"ca"` arrives second and overwrites the newer list, so the user sees suggestions for a prefix they have already moved past. The fix is to sequence requests with an id and discard any response that does not match the current input — plus cancel in-flight requests on new input, and debounce so most keystrokes never produce a request at all. It is a small amount of client code guarding a bug that is otherwise very hard to reproduce from server logs.

**Q13. How do you handle languages without whitespace, like Chinese or Japanese?**
The prefix model assumes word boundaries that do not exist there, and IME composition means raw keystrokes are intermediate states rather than meaningful characters. So the design changes in two ways: trigger on composition events rather than raw key events, so I am only querying on meaningful input; and maintain a parallel index keyed by transliteration — pinyin or romaji — so a Latin-script prefix can retrieve the intended characters. That is a real structural addition to the index, not a locale flag.

**Q14. How do you remove a suggestion?**
An explicit removal pipeline, distinct from the volume-based ageing that naturally drops queries. A removal request creates an entry in a suppression list keyed by query and locale, which is applied at build time to the base index and to the delta, and is also enforced at the serve-time guard for immediate effect. Then I need to purge edge caches — either via a purge API or by versioning the cache key with the index generation, so a new generation invalidates everything at once. And the whole path needs an audit record, because removals are often legal requests with a response deadline.

**Q15. ⚡ Pushback: "Isn't this just a cache with extra steps?"**
That is closer to the truth than it sounds, and I would say so — the serving path really is a lookup, and the cache really does carry the majority of traffic. But calling it a cache misses the two hard parts. First, what is *in* the cache has to be produced by a ranking pipeline over user-generated data, which is where the real engineering is: aggregation, filtering, scoring, and the freshness layers. Second, the corpus is adversarial and the distribution is Zipfian, so both the curation story and the capacity story are unusual. The lookup is easy; deciding what to look up is the system.

**Q16. ⚡ Pushback: "Why not just use an LLM to generate completions?"**
Because the contract is different. Users expect a literal prefix match — if I type `"new y"` I expect things starting with `"new y"`, not a plausible sentence. An LLM gives me semantic completion, not prefix completion, and it costs orders of magnitude more latency and money per request at 100,000 QPS. That is not a marginal difference; it is three orders of magnitude on the serving budget. Where generative models do fit is a *different surface* — sentence completion in a chat box or code completion in an editor, where the context is a window rather than a word, and the latency budget and request rate are far more forgiving. So the right answer is that the two solve different problems rather than that one supersedes the other.

**Q17. ⚡ Pushback: "Your volume threshold means some legitimate queries get no suggestions at all."**
Correct, and that is a deliberate trade rather than an oversight. Materialising top-k at every node costs roughly 10× the storage for traffic that essentially never arrives, so the threshold buys a large storage reduction for a rare miss. What matters is the fallback chain, so a miss still produces a useful answer: first a bounded subtree walk over the below-threshold region, then a spelling-correction path if the prefix looks like a typo, and only then an empty list. And zero-suggestion rate is a monitored metric — if it rises, the threshold is wrong and I have a number to tune rather than a guess.

---

## 10. Common mistakes candidates make

1. **Ranking online.** Proposing to score candidates at query time, without confronting the 100 ms budget and the ~100k QPS.
2. **Missing the request amplification.** Not recognising that autocomplete QPS exceeds search QPS, and not proposing client debounce.
3. **Forgetting the corpus is adversarial.** Designing the pipeline as aggregate → rank, with no filter stage, and no plan for the inevitable offensive-suggestion incident.
4. **Sharding by query hash.** Fan-out on every lookup, and no discussion of tail latency.
5. **Ignoring Zipf.** Sizing the trie fleet for peak QPS instead of for the cache-miss rate.
6. **One monolithic index.** No answer for breaking news, because the volume-threshold structure cannot represent a query that is one minute old.
7. **Per-user precomputation.** Attempting personalisation in the index instead of as a serve-time rerank over a global candidate set.
8. **Recall@k as the product metric.** Optimising an offline proxy that does not measure whether the list helped.
9. **Missing out-of-order responses.** A client bug that server-side logs cannot reveal.
10. **No fallback chain.** Nothing defined for a prefix with no suggestions, so the design degrades to an empty box.
11. **Unbounded trending.** Trending entries that can occupy the whole list and never expire — a spam vector and a reputational risk.
12. **Treating CJK as a locale flag.** Whitespace-free languages need a structural change — composition events and a transliteration index.

---

## 11. Cheat-sheet summary

| | |
|---|---|
| **It is** | A precomputation + freshness system disguised as a search problem |
| **Core decision** | **Rank offline, look up online** — online ranking is impossible at this latency and rate |
| **Latency budget** | **< 100 ms p99**: RTT 20 · lookup 10 · rerank 20 · render 50 |
| **Amplification** | Autocomplete QPS ≈ **10×** search before debounce, ~3.4× after |
| **Load** | ~100k QPS avg · ~300k peak [A] |
| **Index** | Trie with **top-k materialised at the node** ⇒ O(prefix length) lookup |
| **Storage** | Full materialisation ~400 GB · thresholded (~10% of nodes) ~**40 GB** |
| **Sharding** | **By prefix range** (one shard per lookup) — never by query hash (fan-out) |
| **Skew fix** | Virtual ranges + hot-prefix replication + edge cache |
| **The real perf story** | **Zipf** — top 1% of prefixes ≈ 50%+ of traffic ⇒ **cache is the primary serving path** |
| **Freshness** | 3 layers: **base** (daily) + **delta** (minute) + **trending** (second); merge at serve |
| **Trending signal** | Rate **vs expected rate** (a spike), **not** absolute volume |
| **Personalisation** | Global top-N (≈100) → per-user rerank → top-k (≈10) |
| **Curation** | Filter at **build time** first; serve-time guard second; blocklist **+ classifier** |
| **Client must-haves** | Debounce 50–100 ms · request cancellation · **discard stale responses by request id** |
| **Primary metric** | **Selection rate (CTR)** — *not* recall@k |
| **Fallback chain** | Bounded walk → spelling correction → empty (monitor the empty rate) |
| **The close** | "If the corpus were a fixed catalogue instead of user queries, I'd drop the whole build pipeline and serve a filtered list — simpler, instant, and no curation problem at all." |

---

## 12. Cross-links

- [Caching strategies & layers](../../HLD/03-scaling/02-caching-strategies-and-layers.md) — the primary serving path; Zipf, TTL and the cache-hit-rate argument
- [Cache invalidation & consistency](../../HLD/03-scaling/04-cache-invalidation-and-consistency.md) — purging a removed suggestion everywhere
- [CDN deep dive](../../HLD/02-networking/02-cdn-deep-dive.md) — edge caching of hot prefixes
- [Sharding, replication & consistency](../../HLD/03-scaling/05-sharding-replication-and-consistency.md) — prefix-range partitioning, virtual ranges and skew
- [API gateway & rate limiting](../../HLD/02-networking/04-api-gateway-and-rate-limiting.md) — protecting the fleet from request storms
- [NoSQL deep dive](../../HLD/04-data/02-nosql-deep-dive.md) — the KV and streaming stores behind the delta and trending layers
- [Netflix — streaming and recommendation](12-netflix-recommendation-and-streaming.md) — ranking pipelines, and the CTR-versus-recall distinction
- [Observability & monitoring](../../Operations/02-observability-and-monitoring.md) — selection rate, cache hit rate and trending precision as SLIs
- [Security, guardrails & compliance](../../Operations/08-security-guardrails-and-compliance.md) — blocklists, takedown SLAs and audit trails
- [LLD: rate limiter](../../LLD/problems/11-rate-limiter.md) — the class-level treatment of the limiter protecting this service
- [URL shortener](01-url-shortener.md) — the other high-QPS read-path service; compare its lookup design
