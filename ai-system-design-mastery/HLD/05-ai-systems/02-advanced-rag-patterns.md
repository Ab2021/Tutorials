# Advanced RAG Patterns

**TL;DR** — Basic RAG (embed → top-k → stuff → generate) tops out at a recall@10 around 0.65–0.75 on real enterprise corpora. Every advanced pattern in this file exists to fix one of three specific defects: *the query does not look like the document* (query transformation), *the chunk does not contain enough context to be retrieved or understood* (contextual and hierarchical retrieval), or *the evidence needed spans more than one hop* (iterative and graph retrieval). Each pattern buys recall at a known price in latency, cost, and operational complexity. The senior-engineer skill is not knowing the patterns — it is knowing the **trade-off table** well enough to say "that pattern costs 1.8 seconds and $0.004 per query for a 6-point recall gain, so we adopt it only for the 12% of queries the router sends to that path."

> **60-second interview answer:** "I treat advanced RAG patterns as a menu with prices. Query transformation — HyDE, multi-query, decomposition, step-back — fixes vocabulary mismatch for maybe 5–10 points of recall at 300–800 ms and one extra LLM call, so I apply it selectively via a router, not universally. Contextual retrieval and parent-document retrieval are the two highest-value structural fixes: contextual chunk headers took Anthropic's retrieval failure from 5.7% to 1.9% combined with contextual BM25 and reranking, and parent-document gives small-chunk precision with large-chunk context at the cost of 2x storage. Graph RAG only clears its bar when at least 30% of questions are genuinely relationship-shaped, because the extraction and community-summarization maintenance tail is brutal. Agentic RAG is for multi-hop, and I cap it at 3–4 iterations because that is already 8–12 seconds. The one rule I hold to: adopt a pattern only after the eval harness shows the specific failure it fixes, and measure the delta on the golden set before and after."

> **Source grounding:** The **retrieval-strategy catalogue** (§3.5), the transcript's **four-way advanced-RAG variant taxonomy** (§4.3), the **adaptive-RAG effort ladder** (§3.4), the **corrective-RAG grade-then-research loop** (§3.4), the **graph-RAG entity/edge mechanics** (§3.3) and the **query-transformation enumeration** (§3.2) are folded in from the admitted source transcript *nitmonk — "24 Hours of AI Agent Engineering, Part 1"*, from the advanced-RAG segment at **~04:38–04:42** (retrieval strategies, HyDE, multi-query), **~04:43–04:46** (corrective / agentic / graph / self / modular RAG, adaptive routing) and **~04:45–04:47** (graph RAG, Neo4j, entities and edges). Everything else — the Anthropic contextual-retrieval ladder (5.7% → 1.9%), the ColBERT/PLAID storage arithmetic, the per-pattern dollar and latency figures in the trade-off tables, the blended-cost model in §8.2 and the maturity ladder in §8.3 — is supplementary `[A]` general knowledge or `[D]` derivation; **the transcript does not state those numbers**. Numeric labels follow the repo convention: `[S]` transcript-stated, `[A]` general knowledge, `[D]` derived. Where the transcript's auto-transcription garbles a figure, that is noted inline rather than silently corrected.

---

## 1. Definition & mental model

**Plain-English definition.** "Advanced RAG" is the set of techniques layered onto the naive retrieve-then-generate loop in three places: **before** retrieval (transform the query so it matches how documents are written), **during** retrieval (change what gets indexed and what gets returned so chunks are self-contained and correctly scoped), and **after** retrieval (iterate — check whether the retrieved evidence is sufficient and retrieve again if it is not).

**The analogy.** Naive RAG is a single librarian sprint. Advanced RAG is the same librarian with three upgrades:

- **Query transformation** is a *reference interview* — the librarian asks "so what you actually mean is X?" before walking into the stacks.
- **Contextual and hierarchical retrieval** is a *better card catalogue* — each card says which book and chapter it came from, and you can pull the whole chapter once you know which page you want.
- **Agentic and graph retrieval** is *going back for a second trip* — the librarian returns, says "I found the refund window but not the EU exception, let me look again," and then follows the cross-reference to a related document.

**The organising insight:** retrieval failure is not one failure. It has three distinct signatures, and each has a different fix. Diagnosing which one you have is half the interview answer.

| Retrieval failure signature | Root cause | Pattern that fixes it |
|---|---|---|
| Query and document use different vocabulary | Embedding mismatch | HyDE, multi-query, query expansion |
| **A** chunk is retrieved but lacks the context to be *understood* | Chunk isolation | Contextual chunk headers, parent-document, sentence-window |
| The answer requires combining evidence from chunks that never co-occur | Multi-hop | Query decomposition, iterative retrieval, graph RAG |

---

## 2. Why it exists — the problem it solves

Naive RAG fails in measurable, reproducible ways. Published and commonly observed numbers:

| Failure | Observed rate / symptom | Cost of not fixing |
|---|---|---|
| Vocabulary mismatch on short queries | Query < 8 words, recall@10 drops 15–25 points vs. long queries | Users get "I don't have information about that" for answerable questions |
| Chunk-context starvation | Chunk says "the limit is 30 days" with no subject | Retrieved correctly, but the generator cannot use it, or uses it wrongly |
| Answer split across chunks | ~10–20% of multi-clause answers | Model sees half the answer and completes it from priors → hallucination |
| Multi-hop questions | Any question needing 2+ documents | Recall near zero — no single chunk contains the answer |
| Relationship questions | "Which teams depend on the service that owns this table?" | Dense retrieval returns lexically similar chunks, never the connection |
| Follow-up turns in conversation | Turn 2+ recall collapses | Conversation feels amnesiac |

**What breaks without contextual retrieval specifically.** Anthropic's controlled measurement on a real corpus: baseline retrieval failure rate **5.7%**. Adding contextual embeddings (each chunk embedded with a generated context preamble) → **3.7%**. Adding contextual BM25 as well → **2.9%**. Adding reranking on top → **1.9%**. That is a **67% reduction in retrieval failures** and it is the single best-documented quality ladder in the RAG literature. The cost: one cheap-model call per chunk at ingestion, ~$0.0003/chunk with a Haiku-class model under prompt caching.

**What breaks without parent-document retrieval.** Small chunks retrieve precisely but answer poorly; large chunks answer well but retrieve imprecisely. Teams oscillate between the two and never fix it. The pattern breaks the trade-off instead of trading it.

---

## 3. Internal mechanics — how it actually works

### 3.1 Where each pattern plugs in

```
                        ┌──────────────────────────────────────────┐
                        │  INGESTION-TIME PATTERNS                 │
                        │                                          │
                        │  • Contextual chunk headers (CCH)        │
                        │  • Parent/child hierarchy                │
                        │  • Sentence windows                      │
                        │  • RAPTOR recursive summary tree         │
                        │  • Graph extraction + communities        │
                        │  • Multimodal page embeddings            │
                        │  • Late interaction (ColBERT) index      │
                        └───────────────────┬──────────────────────┘
                                            │
  user query                                ▼
      │                          ┌────────────────────────┐
      ├──▶ QUERY-TIME TRANSFORMS │   RETRIEVAL LAYER      │
      │    • Rewrite (context)   │  dense + sparse + RRF  │
      │    • HyDE                │  + reranker            │
      │    • Multi-query         └───────────┬────────────┘
      │    • Decomposition                   │
      │    • Step-back                       │
      │    • Router (pick a path)            ▼
      │                          ┌────────────────────────┐
      └─────────────────────────▶│  POST-RETRIEVAL LOOP   │
                                 │  • Self-RAG critique   │
                                 │  • CRAG grade + route  │
                                 │  • Iterative / multi-  │
                                 │    hop follow-up       │
                                 └───────────┬────────────┘
                                             ▼
                                   Context assembly → Generate
```

### 3.2 Query transformation patterns

**Query rewriting (decontextualization).** Mandatory for any conversational RAG. Given history + the new turn, produce a standalone query.

```
History:  user: "What's the refund window?"  assistant: "30 days."
New turn: "and for EU customers?"

Rewritten: "What is the refund window for EU customers?"
```

Cost: one small-model call, ~100–200 ms. Without it, turn-2 retrieval is effectively random.

**HyDE (Hypothetical Document Embeddings).** Generate a *hypothetical answer* to the query, embed that answer, and retrieve with the answer embedding instead of the query embedding. The insight: a fabricated answer looks much more like a real document than a question does, so it lands closer in embedding space. Effective for short, vague, or jargon-mismatched queries.

Risk: **HyDE hallucinates**. If the hypothetical answer invents the wrong vocabulary, you retrieve confidently wrong documents. Mitigations: generate 2–3 hypotheses and average their embeddings; run the original query in parallel and fuse with RRF; instruct the generator to write "an encyclopaedia-style passage that might answer this" rather than a final answer.

**Multi-query.** Generate 3–5 paraphrases, retrieve for each, fuse with RRF. Fixes phrasing sensitivity at the cost of N retrieval round-trips (run them in parallel — the latency cost is ~one round trip, not N).

**Decomposition.** Break a compound question into atomic sub-questions and retrieve for each.

```
"What is the refund window for EU customers buying through a reseller?"
  ├─ "What is the refund window for EU customers?"
  ├─ "What are the reseller purchase terms?"
  └─ "Do reseller purchases have different refund rules?"
```

Essential for multi-hop. This is the pattern that most directly fixes near-zero recall on compound questions.

**Step-back prompting.** Ask a more general question first, retrieve background, then answer the specific question with both the general context and the specific evidence. Useful when the specific question presupposes concepts the corpus explains elsewhere.

**Routing.** Classify the query and send it down one of several paths (no-retrieval / single-shot RAG / agentic RAG / graph RAG / code RAG). This is what makes the other patterns affordable: you pay for HyDE only on the ~10% of queries that need it.

```
                    ┌── simple factoid ──▶ dense+sparse+rerank (fast path)
   classifier ──────┼── identifier/code ──▶ BM25-heavy + AST index
   (SLM, ~20ms)     ├── multi-hop ────────▶ decompose → agentic loop
                    ├── relational ───────▶ graph index
                    └── no-answer-needed ─▶ direct generation
```

### 3.3 Ingestion-time structural patterns

**Contextual Retrieval (contextual chunk headers).** For each chunk, prompt a cheap model with the full parent document plus the chunk and ask for 50–100 tokens of situating context. Anthropic's published prompt shape:

```
<document>
{{WHOLE_DOCUMENT}}
</document>
Here is the chunk we want to situate within the whole document:
<chunk>
{{CHUNK_CONTENT}}
</chunk>
Please give a short succinct context to situate this chunk within the
whole document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
```

Then store `context + "\n\n" + chunk` as the embedded and BM25-indexed text. Prompt-cache the `<document>` block so a document with 40 chunks pays full input price once and ~10% thereafter. Result ladder: **5.7% → 3.7% → 2.9% → 1.9%** as you add contextual embeddings, contextual BM25, then reranking.

**CCH vs. full LLM contextualization.** Cheaper alternatives: use heading hierarchy as the header (`H1 > H2 > H3` path), use the first 200 characters of the document, or use the document title + summary. These get a meaningful fraction of the benefit for zero LLM cost. The rule: **start with heading-path headers, measure, and only pay for LLM-generated context if the eval says you need it.**

**Late chunking.** Embed the whole document with a long-context embedding model, then mean-pool the token embeddings belonging to each chunk. Each chunk's vector thus carries the document's context without an extra generation call. Requires a long-context embedding model and more careful engineering; attractive when you cannot afford per-chunk LLM calls.

**Parent-document retrieval.** Index small children (300 tokens) but return the parent (1,500 tokens) at query time. Implementation: children store `parent_id`; a parent store (key-value, not vector) holds the full parent text. Match on child, fetch parent, dedupe parents in the top-k, then budget.

**Sentence-window retrieval.** Index individual sentences (or small windows), return ±3–5 sentences around each hit. Highest precision for factoid lookups; poor for narrative questions where the answer is the shape of a section, not a sentence.

**RAPTOR (recursive summarization tree).** Build a tree: leaves are chunks; each parent is an LLM summary of its children; recurse until one root. At query time you can retrieve at any level, so a "what are the themes" question retrieves a summary node while a factoid question retrieves a leaf. Cost: one summarization pass over the corpus (comparable to contextualization). Value: it is the cleanest answer to "summarize the corpus" questions, which flat RAG cannot do at all.

**Graph RAG.** Extract entities and relationships with an LLM, build a knowledge graph, run community detection (Leiden), and summarize each community. Query modes: *local* (entity neighbourhood) and *global* (community summaries). **The bar: at least 30% of your questions must be genuinely graph-shaped** (relationship, dependency, "who owns what", "what breaks if X changes") before the extraction cost and the quarterly-refresh maintenance tail pay off. The **graph-as-reranker** variant gets 70–80% of the quality at ~20% of the cost: keep flat retrieval, use the graph only to expand and reorder the candidate set.

The transcript's account of graph RAG is more concrete about *what the graph is made of* `[S]` (~04:45–04:47), and it is the version to say out loud: the graph is built over **entities that are real business nouns** — *"it's about company, it's about person, people, department"* — connected by **edges that are the verbs of the domain** — *"signed a contract"*, and *"contract has a clause"*. Neo4j is called out as the leading engine because it is a graph database rather than a vector store. The worked shape, therefore, is not an abstract "knowledge graph"; it is:

```
(PERSON)──SIGNED──▶(CONTRACT)──HAS_CLAUSE──▶(CLAUSE)──OBLIGATES──▶(DEPARTMENT)
    │                                                                    ▲
    └─────────────────── OWNS ───────────────────────────────────────────┘
```

Two details from that segment are worth keeping. First, the reason the graph helps at all is that it is **faster and easier for the LLM to grab**: the traversal has already done the join, so the model receives the relationship rather than having to infer it from lexically-similar prose. Second, and more useful for the interview, the transcript is explicit that **graph RAG still contains ordinary vector search** — *"everything inside RAG is HNSW and other ANN algorithms being applied to partition, to build a centroid, to get the answer"* — the graph is an additional index and an additional retrieval mode, not a replacement for the vector one. Say that and you avoid the common candidate error of describing graph RAG as a separate architecture.

**Multimodal page embeddings (ColPali-style).** Treat each PDF page as an image. A vision encoder (SigLIP So400m) produces ~1024 patches per page; a small language model (Gemma 2B) projects each patch to 128 dims; retrieval uses late interaction (MaxSim) over patch embeddings. No OCR, no layout parsing, no table extraction — the layout *is* the representation. Cost: 1M pages ≈ 1B vectors, 2–5 s/page to embed, $0.01–0.05/page if using a hosted VLM. Worth it when 40–60% of the corpus is non-textual.

**Late interaction / ColBERT.** Store one 128-dim vector per *token* rather than one per chunk, and score with MaxSim:

```
Score(q, d) = Σ_{i ∈ q} max_{j ∈ d} ( E_q[i] · E_d[j] )
```

Each query token finds its best match in the document, so no single embedding has to summarize the whole chunk. Quality is substantially above single-vector dense (typically +5–10 NDCG points). Storage is the problem: naive ColBERT is ~100x the storage of single-vector. ColBERTv2 residual compression brings 16–32x reduction; PLAID adds a 4-stage pruning pipeline that runs exact MaxSim in 50–100 ms over 10M+ documents. Order-of-magnitude storage for 10M documents: uncompressed ~1 TB → ColBERTv2 ~64 GB → PLAID-indexed ~30 GB.

### 3.4 Post-retrieval iterative patterns

**Self-RAG.** The generator emits critique tokens during generation — is retrieval needed? is this passage relevant? is my output supported? — and the loop re-retrieves when the relevance critique fails. Adds critique tokens to every generation, so it costs both latency and output tokens.

**CRAG (Corrective RAG).** A lightweight evaluator grades each retrieved document as **correct / ambiguous / incorrect**:

```
retrieve → grade each doc
  ├── correct   → refine and generate
  ├── ambiguous → generate from retrieved + one web/alternate-source lookup
  └── incorrect → discard retrieved, fall back to alternate source (web, larger index)
```

This is the pattern that most cheaply prevents "confidently answering from bad context," because the *incorrect* branch refuses to use the retrieved documents at all.

**Iterative / multi-hop retrieval.** The agent retrieves, reads, forms a follow-up question, retrieves again. Terminate on: sufficient-evidence judge, step budget, or token budget. **Budget 3–4 turns maximum** — 3–4 iterations at 2–4 s per turn is already **8–12 s** end-to-end, which is at the edge of user patience for interactive use.

**Adaptive retrieval.** Retrieve only when needed. A cheap classifier decides whether the query is answerable from the model's parametric knowledge (small talk, definitions, arithmetic) or requires retrieval. Saves both latency and cost on the 20–30% of traffic that never needed retrieval.

The transcript's formulation is **effort by difficulty**, and it is worth quoting because it is the cleanest one-sentence version of the idea `[S]`: *"adaptive RAG — effort by difficulty — router how hard: simple, medium, complex. Simple → answer directly. Medium → retrievable, one or two iterations. Complex → go ahead with the agent work."* That is a **three-rung** ladder, not a binary retrieve/don't-retrieve switch:

| Rung | Query class | Path | Effort |
|---|---|---|---|
| 1 | Simple | Answer directly, no retrieval (or one dense lookup) | ~0 extra calls |
| 2 | Medium | Retrieval + one or two iterations; rewrite and rerank | 1–2 extra calls |
| 3 | Complex | Agentic multi-step — decompose, retrieve, grade, retrieve again | 3–4 calls, 8–12 s |

The important design point is that the ladder is **graded by the router before any retrieval happens**, so the expensive rungs are entered deliberately rather than by escalation drift. The transcript also places corrective RAG on rung 2 and graph RAG off to the side — graph is not "more effort", it is a *different index*, which is why it is selected by query *shape* rather than query *difficulty*.

### 3.5 The retrieval-strategy catalogue `[S]`

Asked "what retrieval strategies do you know?", the transcript's answer at **~04:40–04:42** is a flat enumeration of ten. It is a good answer precisely because it is flat — then you rank them by where they sit in the pipeline:

| # | Strategy | What it actually does | Where it sits |
|---|---|---|---|
| 1 | **Top-K** | Return the K nearest vectors; K is a parameter you tune | The baseline |
| 2 | **Similarity threshold** | Return everything above a distance/angle cutoff instead of a fixed count — "how close the vectors are together, that theta" | Replaces or guards top-K |
| 3 | **Hybrid** | Lexical (BM25/TF, keywords) **plus** semantic (vector similarity — cosine, dot product, Euclidean) | Query time |
| 4 | **Multi-query** | "Updating the query into multiple ones" — paraphrase, retrieve for each, union, dedupe, rerank, merge. "Different wordings surface different [chunks]"; boosts recall for ambiguous questions | Before retrieval |
| 5 | **Metadata filtering** | Attach metadata to every chunk at ingestion, then apply `and`/`or` predicates at query time | Before retrieval |
| 6 | **Parent retrieval** | Chunk completely, search the **child**, retrieve the **parent** | After retrieval |
| 7 | **Contextual retrieval** | Add document/section context to each chunk so it is retrievable and understandable standalone | Ingestion |
| 8 | **Multi-vector** | More than one vector per chunk (summary vector + raw vector, or per-question vectors) | Ingestion |
| 9 | **Hierarchical** | Index at multiple levels of granularity (chunk, section, document) and retrieve at the level the question needs | Ingestion |
| 10 | **Query routing** | Classify the query and send it down one of several retrieval paths | Before everything |

**How to use this list in the interview.** Do not recite it. Sort it: strategies **1–5 change what you retrieve** (they operate on the candidate set), strategies **6–9 change what you index** (they cost money at ingestion and nothing at query time), and strategy **10 decides which of the other nine run at all**. Then say the sentence the transcript insists on — *"everything starts from R-A-G: the first retrieval happens, then augmentation happens, then generation happens"* — because it re-anchors the whole list to the pipeline the interviewer is actually scoring you against.

---

## 4. Variants & taxonomy

### 4.1 The master trade-off table

| Pattern | Extra latency | Extra cost/query | Typical recall gain | Operational complexity | Adopt when |
|---|---|---|---|---|---|
| Query rewrite (decontextualize) | +100–200 ms | ~$0.0002 | Turn-2 recall: large (>20 pts) | Low | **Always, for conversational RAG** |
| Multi-query (3–5 paraphrases) | +200–400 ms (parallel) | ~$0.0005 + N retrievals | +3–8 pts recall@10 | Low | Short/vague queries; phrasing sensitivity |
| HyDE | +400–800 ms | ~$0.001 | +5–12 pts on short queries | Medium | Vocabulary mismatch is the diagnosed failure |
| Step-back | +300–500 ms | ~$0.0008 | +3–6 pts on "why/how" questions | Medium | Questions presuppose unexplained concepts |
| Query decomposition | +400–900 ms | ~$0.001 + N retrievals | Multi-hop recall from ~0.2 → 0.7+ | Medium | Compound questions are common |
| Router / classifier | +20–50 ms | ~$0.00005 | Enables all others affordably | Medium | **Adopt before any other expensive pattern** |
| Contextual chunk headers (LLM) | 0 (ingestion only) | ~$0.0003/chunk once | Failure 5.7% → 3.7% | Medium | Corpus is stable; highest ROI structural fix |
| Contextual BM25 | 0 | negligible | 3.7% → 2.9% | Low | Always pair with context and hybrid |
| Parent-document retrieval | +10–30 ms | +2x storage | +5–10 pts answer quality | Low–Medium | Answers split across chunks |
| Sentence-window | +10 ms | +storage | High precision on factoids | Low | Factoid-heavy workload |
| RAPTOR summary tree | 0 (ingestion) | Summarization pass | Enables whole-corpus questions | High | "Summarize/themes" questions exist |
| Graph RAG | +300–800 ms | High ingestion + quarterly refresh | Large for relationship questions | **High** | ≥30% of questions are graph-shaped |
| Graph-as-reranker | +50–150 ms | ~20% of full GraphRAG | 70–80% of GraphRAG quality | Medium | Relationship signal helps but does not dominate |
| Late interaction (ColBERT/PLAID) | +20–60 ms | 30–100x storage | +5–10 NDCG pts | High | Quality bar is high and storage is available |
| Multimodal page embeddings | +100–300 ms | $0.01–0.05/page ingestion | Enables non-text retrieval | High | 40–60% of content non-textual |
| Self-RAG | +1–3 s | +critique tokens | +3–8 pts faithfulness | High | Faithfulness is the binding constraint |
| CRAG | +200–600 ms | ~$0.0005 | Prevents bad-context answers | Medium | Retrieval quality is variable across queries |
| Iterative / multi-hop | +2–4 s **per turn** | +$0.002 per turn | Multi-hop recall up sharply | Medium–High | Multi-hop questions; research flows |
| Adaptive retrieval | −0 to +50 ms | **saves** cost | Neutral | Low | 20–30% of traffic is retrieval-free |

**Reading the table:** latency and cost are per-query *when the pattern fires*. Because they fire selectively under a router, the blended cost is the pattern cost × the fraction of traffic routed to it. HyDE at +600 ms on 10% of traffic is +60 ms blended — cheap. HyDE on 100% of traffic is +600 ms, which blows a 2-second p95 budget by itself.

### 4.2 Pattern families at a glance

| Family | Question it answers | Patterns | Ingestion or query time? |
|---|---|---|---|
| **Vocabulary bridging** | "The user and the document don't speak the same language" | Rewrite, HyDE, multi-query, expansion, step-back | Query |
| **Chunk self-sufficiency** | "The chunk doesn't say enough to be used" | CCH, late chunking, parent-document, sentence-window | Ingestion (+ assembly) |
| **Structural coverage** | "The answer isn't in any single chunk" | RAPTOR, graph RAG, decomposition | Ingestion + query |
| **Sufficiency control** | "Is what I retrieved actually enough?" | Self-RAG, CRAG, iterative retrieval, adaptive retrieval | Query |
| **Representation shift** | "Text isn't the right unit" | ColBERT, multimodal page embeddings | Ingestion |
| **Cost control** | "Which of these do I actually need?" | Router, adaptive retrieval, caching | Query |

### 4.3 The variant taxonomy — each variant is named after its trigger `[S]`

The transcript at **~04:43–04:45** lists the advanced-RAG variants as a set of buzzwords and then, crucially, attaches each one to the *condition that justifies it*. That is the version worth memorising, because "when X fails, you use Y" is a stronger answer than "there are five kinds of RAG":

| Variant | Trigger — the condition that calls for it `[S]` | Loop shape |
|---|---|---|
| **Corrective RAG** | *"Weak and wrong retrieval"* — the retrieved documents may not answer the question | Retrieve → **grade each doc relevant or not** → good: generate. Weak: go research it, correct it, until the answer is better. *"Grade before you answer."* |
| **Agentic RAG** | *"A complex multi-step question"* | The agent decides whether the retrieved evidence is enough; if not, it decides which source to retrieve next. Loops until sufficient or the budget runs out |
| **Graph RAG** | A **relationship** question — company, person, department, contract | Not a loop: a different index. Traverse entities and edges instead of comparing vectors |
| **Self-RAG** | Faithfulness needs checking, *"sometimes unnecessarily"* | The model reflects on its own retrieval and output. Adopt it deliberately, not by default — note the transcript's own warning that it is *sometimes unnecessary* |
| **Modular RAG** | Composable, reusable components across several pipelines | Architecture-level: retrieval, augmentation and generation as swappable modules rather than a fixed chain |

Three things to take from this framing.

1. **Every variant is a response to a named symptom, not a maturity level.** Corrective RAG exists because retrieval quality *varies*; agentic RAG exists because the question *spans hops*; graph RAG exists because the question is *relational*. Say the symptom first and the variant second, and you sound like someone who has operated one.
2. **Corrective RAG and agentic RAG are the same loop at different strictness.** Both retrieve, both judge sufficiency, both re-retrieve. Corrective RAG's contribution is the explicit *grade* step and the willingness to **discard** the retrieved set; agentic RAG's contribution is that the *agent chooses the next source*. They are often deployed as one loop, and it is fine to say so.
3. **The transcript calls the whole discipline "RAG engineering"** — the same way prompt engineering is engineering the prompt, RAG engineering is engineering the retrieve–augment–generate loop. That framing is a good closing sentence: the patterns are not features you enable, they are the discipline of tuning a three-stage pipeline against measured failures.

**What the transcript's taxonomy does *not* cover** `[A]`: it gives no adoption criteria, no cost or latency figures, and no ordering rule. Everything in §4.1, §7 and §8 is the supplement that turns this taxonomy into a decision procedure — which is the gap an interviewer will probe.

---

## 5. Pros / Cons

| Pros | Cons |
|---|---|
| Each pattern fixes a *named, measurable* failure — you can prove the delta on a golden set | Every pattern adds a knob, and knobs multiply into unmaintainable configurations |
| Query transforms require no re-indexing, so they are cheap to trial | Query transforms add one or more LLM calls *before* retrieval, directly hitting TTFT |
| Contextual retrieval delivers the best documented quality-per-dollar in RAG (67% failure reduction) | Contextualization is a one-time full-corpus LLM pass — real money and real GPU-hours |
| Parent-document breaks the small-chunk/large-chunk trade-off rather than picking a side | 2x storage and a parent store you must keep consistent with the index |
| Graph RAG answers questions flat RAG *cannot* answer at all | Extraction quality is the bottleneck; communities need quarterly refresh; cost is high |
| Agentic RAG is genuinely necessary for multi-hop and research tasks | 3–4 iterations is 8–12 s; cost scales linearly with iterations; loops can thrash |
| ColBERT and multimodal extend retrieval to token-level and image-level granularity | 30–100x storage for late interaction; 1B vectors for 1M pages of multimodal |
| Routing makes the expensive patterns affordable by paying for them selectively | The router itself is a classifier with its own accuracy, latency, and failure modes |
| Late chunking gets contextual benefit without per-chunk generation cost | Requires a long-context embedding model and custom pooling code |

---

## 6. Exceptions, edge cases & failure modes

### 6.1 HyDE hallucinating the retrieval target

**Symptom:** HyDE improves some queries and *degrades* others — you retrieve documents matching the fabricated answer's wrong vocabulary.
**Mitigation:** never deploy HyDE alone. Run original-query retrieval in parallel and fuse with RRF; generate 2–3 hypotheses and average; restrict HyDE to queries below a length/confidence threshold; A/B it on the golden set and keep it only if the *net* recall improves, not just the average.

### 6.2 Decomposition multiplying cost

**Symptom:** one user question becomes 5 sub-questions × 50 retrieved chunks × 1 rerank pass. Cost per query rises 5x, latency 4x.
**Mitigation:** cap sub-questions at 3; cap total retrieved chunks across the whole decomposition at ~20; share one reranker pass over the union rather than reranking per sub-question; only decompose when the router classifies the query as compound.

### 6.3 Agentic retrieval thrash

**Symptom:** the agent reformulates the same query 6 times, each returning the same chunks, burning tokens and time. **This is the most common agentic RAG failure in production.**
**Mitigations:** track retrieved chunk IDs across iterations and refuse to re-query with an identical sub-query; a step budget of 3–4; a sufficient-evidence judge that can terminate early; a hard token budget for the retrieval loop; log the trajectory so you can see the loop.

### 6.4 Tool storms and context bloat in agentic RAG

**Symptom:** iterations accumulate retrieved chunks until the context is 50k tokens of redundant evidence and quality *falls*.
**Mitigating rule:** re-assemble the context from scratch each iteration using the *current* best-ranked evidence, rather than appending. Cap context growth at ~6k tokens of retrieved evidence regardless of iteration count.

### 6.5 Graph RAG's maintenance tail

**Symptom:** the graph is excellent at launch and wrong within a quarter. Entities are stale, communities no longer reflect the corpus, and nobody owns refresh.
**Mitigation:** only adopt if you can commit to a quarterly (or event-driven) rebuild; scope the graph to the subdomain where relationships actually matter rather than the whole corpus; prefer graph-as-reranker to keep the blast radius small; instrument graph staleness as an SLI.

### 6.6 Parent-document inconsistency

**Symptom:** children are re-embedded after a document edit but the parent store is not updated, so the generator returns the old parent text.
**Mitigation:** write parent and children in one transaction; key parents by `content_hash`; a reconciliation job that detects orphaned or version-mismatched parents; index version in every cache key.

### 6.7 ColBERT storage blow-up

**Symptom:** the index that fit at 10M documents does not fit at 100M; storage becomes the dominant cost line.
**Mitigation:** ColBERTv2 residual compression (16–32x); PLAID centroid pruning; deploy ColBERT as a **reranker** over a cheap first-stage rather than as the primary index — you get most of the MaxSim quality while the first stage controls candidate count.

### 6.8 Multimodal ingestion cost surprise

**Symptom:** a "just index the PDFs as images" decision turns into a $50k ingestion bill and 1B vectors.
**Mitigation:** compute the math up front (pages × $0.01–0.05, pages × 1024 patches × 128 dims); use a text-first path with image escalation only for pages that actually fail text extraction; deduplicate near-identical page images; consider image-only retrieval for *figures* and text retrieval for body text (dual representation).

### 6.9 Router misclassification

**Symptom:** a multi-hop question gets the fast path and returns a shallow answer, silently. Or a simple question gets the agentic path and takes 11 seconds for nothing.
**Mitigation:** the router is a model — evaluate it like one. Log routing decisions with the final quality signal and retrain/threshold-tune on that feedback. Always keep a fallback: if the fast path's evidence score is below a floor, escalate to the deeper path rather than answering.

### 6.10 Prompt injection amplified by iterative retrieval

**Symptom:** an agentic loop retrieves a poisoned document, follows its instructions, and issues a *follow-up query* designed by the attacker.
**Mitigation:** retrieved content can never influence the *authorization* or the *tool set*; every iteration re-applies the ACL filter; injection scanning runs on every retrieved chunk, not just on the first turn; log every sub-query the agent generates so an injected query pattern is visible.

### 6.11 Silent quality regression from a pattern "improvement"

**Symptom:** adding step-back improved the specific class of question the team was looking at, but degraded the dominant factoid class by 4 points. Nothing alerted.
**Mitigation:** any pattern addition ships behind a flag, is measured on the **full** golden set (not a targeted slice), and is canaried on 5% of traffic with per-class metric breakdowns before full rollout.

### 6.12 Cost explosion across a pattern stack

A stack of rewrite + HyDE + multi-query + decompose + self-RAG + rerank can easily be 6 LLM calls and $0.02 per query before generation. At 800k queries/day that is $16,000/day just in retrieval-side overhead.
**Mitigation:** instrument cost per pattern per query as a first-class metric; set a retrieval-stage cost cap (e.g. $0.003/query); enforce it with the router; the router is not an optimization, it is the control plane that makes advanced RAG economically viable.

---

## 7. When to use / When NOT to use

### When to use each pattern

| Pattern | Use when | Do NOT use when |
|---|---|---|
| Query rewrite | Any multi-turn conversation | Single-turn stateless API |
| Multi-query | Queries are short and phrasing-sensitive | Queries are long and precise |
| HyDE | Diagnosed vocabulary mismatch on short queries | Domain has precise jargon the model would invent wrongly |
| Decomposition | Compound/multi-hop questions are common | Questions are atomic factoids |
| Step-back | "Why/how" questions where background matters | Purely look-up workloads |
| Router | **Always, before any other expensive pattern** | Trivial single-path prototypes |
| Contextual retrieval | Corpus is stable and quality is the binding constraint | Corpus changes every hour and re-contextualization is unaffordable |
| Parent-document | Answers are split across chunks | Chunks are already self-contained |
| Sentence-window | Factoid-heavy, precise lookups | Summary/narrative questions |
| RAPTOR | Whole-corpus summary/themes questions exist | All questions are lookups |
| Graph RAG | ≥30% of questions are relationship-shaped, and you can maintain it | Relationships are incidental |
| Graph-as-reranker | Relationship signal helps but does not dominate | You need full global-community reasoning |
| ColBERT / PLAID | Quality bar is high and storage budget exists | Storage-constrained or cost-constrained |
| Multimodal page embeddings | 40–60% of content is non-textual | Corpus is clean text |
| Self-RAG | Faithfulness is the binding constraint | Latency budget is under 2 s |
| CRAG | Retrieval quality varies a lot across query types | Retrieval is uniformly strong |
| Iterative / multi-hop | Research and multi-hop tasks | Interactive latency expectations are tight |
| Adaptive retrieval | 20–30% of traffic needs no retrieval | Every query is genuinely knowledge-seeking |

### Decision table: pick a path from the symptom

| Observed symptom | First fix | Second fix | Last resort |
|---|---|---|---|
| Wrong answer, gold chunk missing from top-50 | Hybrid retrieval + RRF | Raise `ef_search`; check ACL filters | Re-embed with a better model |
| Gold chunk in top-50 but not top-5 | **Reranking** | Raise rerank depth K from 50 to 100 | Late-interaction retrieval |
| Gold chunk in top-5 but answer still wrong | Fix chunk context (CCH) | Parent-document retrieval | Increase context budget |
| Answerable question returns "I don't know" | Query rewrite / expansion | HyDE | Multi-query |
| Follow-up turns degrade | Query rewrite from history | Conversation-scoped memory | — |
| Multi-hop questions fail | Decomposition | Iterative retrieval (cap 3) | Graph RAG |
| "Summarize the corpus" fails | RAPTOR | Graph community summaries | Map-reduce over chunks |
| Relationship questions fail | Graph-as-reranker | Full Graph RAG with community summaries | — |
| Tables/charts unanswerable | Atomic table chunks | VLM layout extraction | ColPali page embeddings |
| Cost per query too high | Router + adaptive retrieval | Semantic cache | Trim the pattern stack |
| Latency p95 too high | Move patterns behind the router | Parallelize multi-query | Drop self-RAG |

---

## 8. Numbers, defaults & tuning

### 8.1 Per-pattern parameter defaults

| Pattern | Parameter | Default | Range | Why |
|---|---|---|---|---|
| Query rewrite | Model | Small (Haiku-class) | — | The task is trivial; do not pay frontier prices |
| Query rewrite | Timeout | 400 ms | 200–800 ms | Fall back to raw query rather than fail |
| Multi-query | N paraphrases | 3 | 2–5 | Beyond 5 the retrievals are redundant |
| Multi-query | Fusion | RRF k=60 | 10–100 | Same as hybrid fusion |
| HyDE | Hypotheses | 2–3, averaged | 1–3 | Averaging reduces hallucination-driven drift |
| HyDE | Trigger | Query < 8 tokens | 5–15 tokens | Long queries already carry vocabulary |
| Decomposition | Max sub-questions | 3 | 2–5 | Cost and latency scale linearly |
| Decomposition | Shared rerank pool | 20 chunks total | 15–30 | Prevents k × sub-questions blow-up |
| Step-back | Extra retrieval | 1 general + 1 specific | — | Fuse both with RRF |
| Router | Classifier latency | 20–50 ms | < 100 ms | Must be negligible vs. the paths it selects |
| Router | Classes | 4 | 3–6 | More classes = more misclassification surface |
| Contextual retrieval | Header length | 50–100 tokens | 30–150 | Longer headers dilute the chunk embedding |
| Contextual retrieval | Generator model | Cheap (Haiku-class) | — | ~$0.0003/chunk vs $0.002–0.01 for larger |
| Contextual retrieval | Prompt cache | Enabled on `<document>` | — | ~90% input discount on repeated parent docs |
| Parent-document | Parent size | 1,500 tokens | 1,000–2,500 | Enough surrounding context |
| Parent-document | Children per parent | 5 × 300 tokens | 3–8 | Precision/context balance |
| Sentence-window | Window | ±3 sentences | ±2–5 | More is just a worse parent-document |
| RAPTOR | Tree depth | 3–4 | 2–5 | Beyond 4 the summaries are too abstract |
| Graph RAG | Extraction model | Mid-tier | — | Cheap models produce incoherent graphs |
| Graph RAG | Community algorithm | Leiden | — | Hierarchical, standard |
| Graph RAG | Refresh cadence | Quarterly or event-driven | — | The maintenance tail is the real cost |
| ColBERT | Vector dim | 128 per token | 128 | ColBERTv2 standard |
| ColBERT | Compression | Residual (16–32x) | — | Uncompressed is unaffordable |
| ColBERT | First-stage candidates | 1,000 → PLAID → exact MaxSim | 500–2,000 | PLAID's 4-stage pruning |
| ColPali | Page resolution | ~1024 patches/page | — | SigLIP So400m patch grid |
| ColPali | Patch dim | 128 | — | After Gemma 2B projection |
| CRAG | Grade threshold | correct / ambiguous / incorrect | — | Tune the boundary on the golden set |
| Self-RAG | Critique model | Same as generator | — | Critique tokens come from the generator |
| Iterative | **Max iterations** | **3–4** | 2–5 | 3–4 iterations = 8–12 s already |
| Iterative | Per-iteration budget | $0.002 | — | Cap the loop, not just the steps |
| Adaptive retrieval | No-retrieval fraction | 20–30% of traffic | 10–40% | Small talk, definitions, arithmetic |
| Any pattern | Rollout | Flag + 5% canary | — | Never ship a pattern change unmeasured |

### 8.2 Blended-cost arithmetic (the number interviewers want)

Assume 100k queries/day and a router that sends 70% down the fast path, 20% to query-transform, 8% to decomposition, 2% to agentic.

```
Fast path (70k):     rewrite only             $0.0002  → $14
Transformed (20k):   multi-query + HyDE       $0.0015  → $30
Decomposed (8k):     3 sub-Qs, shared rerank  $0.0010  → $8
Agentic (2k):        3 iterations × $0.002    $0.0060  → $12
                                              ---------------
Retrieval-side total:                                   $64/day
Generation (all 100k, 3k in + 300 out):   $0.0105/query → $1,050/day
```

Retrieval-side overhead is **~6% of the bill**. Now run every query through the full pattern stack:

```
Full stack (100k): rewrite + HyDE + multi-query + decompose + self-RAG ≈ $0.006/query → $600/day
Generation:                                                                            $1,050/day
```

Retrieval-side overhead becomes **36% of the bill** — a 9x increase in non-generation spend for a quality gain that must be proven, per class of question, on the golden set. **This arithmetic is the argument for routing.**

### 8.3 Quality expectations by maturity level

| Maturity | Recall@10 | Faithfulness | p95 latency | Cost/1k queries |
|---|---|---|---|---|
| Naive | 0.65–0.75 | 0.75–0.85 | ~1.2 s | ~$8 |
| + hybrid + rerank | 0.80–0.88 | 0.82–0.90 | ~1.5 s | ~$9 |
| + contextual retrieval | 0.85–0.92 | 0.88–0.93 | ~1.6 s | ~$11 (amortized) |
| + router + transforms | 0.88–0.94 | 0.90–0.95 | ~1.7 s | ~$12 |
| + agentic on the routed slice | 0.90–0.96 (multi-hop) | 0.90–0.96 | ~2.5 s p95, 9–12 s on agentic slice | ~$15 |

### 8.4 Latency budget with advanced patterns (p95)

| Configuration | Retrieve | Transform | Rerank | Generate | Total |
|---|---|---|---|---|---|
| Fast path | 100 ms | 120 ms (rewrite) | 150 ms | 1,200 ms | **~1.6 s** |
| + HyDE + multi-query | 250 ms | 700 ms | 180 ms | 1,200 ms | **~2.3 s** |
| + decomposition | 400 ms | 900 ms | 220 ms | 1,400 ms | **~2.9 s** |
| + agentic (3 iters) | 3 × 350 ms | 3 × 400 ms | 3 × 150 ms | 1,500 ms | **~9–12 s** |

**Design rule:** anything above ~2.5 s should be an explicit, user-visible mode ("deep research") or an async job with a progress indicator — never a silent fallback inside a synchronous chat turn.

---

## 9. Interview Q&A

**Q1. What is HyDE and when would you not use it?**
HyDE generates a hypothetical answer to the query and embeds *that* instead of the query, because an answer-shaped text sits closer to real documents in embedding space. I would not use it when the domain has precise jargon the model is likely to invent incorrectly — a fabricated answer with wrong terminology retrieves confidently wrong documents — or when queries are long and already carry the document's vocabulary. If I do use it, I run the original query in parallel and fuse with RRF, and I gate it behind a router so only short queries pay the ~600 ms.

**Q2. Explain contextual retrieval and quantify the benefit.**
Each chunk is stored with a 50–100 token generated preamble situating it in its parent document, and both the embedding and the BM25 index use that augmented text. The published ladder: baseline retrieval failure 5.7% → 3.7% with contextual embeddings → 2.9% adding contextual BM25 → 1.9% adding reranking. That is a 67% reduction. The cost is one cheap-model call per chunk at ingestion — roughly $0.0003/chunk — and prompt caching on the parent document block cuts the input cost by about 90%. I would start with heading-path headers for free and only pay for LLM contextualization if the eval shows I need it.

**Q3. Parent-document retrieval — why does it work?**
It decouples the unit you *match* from the unit you *return*. Small chunks give sharp embeddings and precise matching; large parents give the generator enough context to answer correctly. Instead of choosing a chunk size that compromises both, you index 300-token children with a `parent_id` and fetch the 1,500-token parent at query time. Cost is roughly 2x storage plus a parent store, and the failure mode is version skew between children and parents — so I write them transactionally and reconcile.

**Q4. When is Graph RAG worth it?**
When at least 30% of the questions are genuinely relationship-shaped — "which services depend on this table", "who owns this policy", "what breaks if X changes". Flat RAG answers those by lexical similarity and misses the connection entirely, so the quality gain is not incremental, it is categorical. Below that bar it is not worth it: entity/relationship extraction is the bottleneck, community summarization needs periodic refresh, and the quarterly maintenance tail is real. When I want the benefit without the cost I use graph-as-reranker, which gets 70–80% of the quality at about 20% of the cost.

**Q5. What is CRAG and how does it differ from Self-RAG?**
Both are corrective loops. CRAG puts a lightweight evaluator between retrieval and generation and grades each retrieved document correct, ambiguous, or incorrect. Correct → refine and generate. Ambiguous → generate plus one alternate-source lookup. Incorrect → discard the retrieved set entirely and fall back to a different source. The key property is that the incorrect branch refuses to use bad context. Self-RAG puts the critique inside the generator as special tokens — should I retrieve, is this relevant, is my output supported — so the model self-corrects during generation. CRAG is cheaper and more controllable; Self-RAG is more integrated but adds critique tokens to every generation.

**Q6. How many iterations should an agentic RAG loop run, and why?**
Three to four, maximum. Each iteration is a retrieval plus a reasoning call, which is 2–4 seconds; 3–4 iterations is already 8–12 seconds end-to-end. Beyond that, returns diminish sharply and the loop is more likely to be thrashing than making progress. I enforce it with a hard step budget, a token budget for the retrieval loop, and — critically — a check that refuses to re-issue a sub-query that is identical or near-identical to one already issued, because retrieval thrash is the most common agentic RAG failure I have seen in production.

**Q7. How do you decide which advanced patterns to adopt?**
I never adopt from a list. I diagnose from the eval harness: run the golden set with per-class breakdowns, find which class of question fails and *where* it fails — retrieval miss, rank miss, context miss, or generation miss. Then I adopt the pattern that addresses that specific stage, measure the delta on the full golden set, and keep it only if the net is positive. Almost every pattern helps the class it was designed for; the risk is the collateral damage to the dominant class.

**Q8. What does late-chunking buy you over contextual chunk headers?**
Both give a chunk document-level context. Contextual chunk headers do it with an extra LLM generation per chunk and store the augmented text. Late chunking embeds the whole document with a long-context embedding model and then mean-pools the token embeddings belonging to each chunk, so the context is carried implicitly in the pooling rather than explicitly in text. Late chunking is cheaper at scale and preserves the original chunk text, but it requires a long-context embedding model and custom pooling, and it does not boost BM25 the way contextual text does. I pick CCH when I need hybrid gains, late chunking when I need scale economics.

**Q9. Explain ColBERT's MaxSim and the storage problem.**
ColBERT stores a 128-dim vector per *token* instead of one vector per chunk, and scores with MaxSim: for each query token, find its maximum dot product against any document token, then sum over query tokens. That means no single vector has to summarize the whole chunk, which is why it beats single-vector dense by 5–10 NDCG points. The cost is storage: naive ColBERT is roughly 100x single-vector. ColBERTv2's residual compression brings that to 16–32x, and PLAID adds a four-stage pruning pipeline that still computes exact MaxSim but runs in 50–100 ms over 10M+ documents. For 10M documents, the arc is ~1 TB uncompressed → ~64 GB with ColBERTv2 → ~30 GB PLAID-indexed.

**Q10. How do you keep an advanced RAG stack affordable?**
Routing is the control plane. I classify each query in ~20–50 ms and send it to one of a few paths: a fast path for factoids, a transform path for short vague queries, a decomposition path for compound questions, an agentic path for genuine multi-hop, and a no-retrieval path for the 20–30% of traffic that never needed retrieval. Then I measure cost per pattern per query as a first-class metric, because the difference is dramatic: with routing, retrieval-side overhead is about 6% of the bill; running the full pattern stack on every query makes it 36%. That 9x difference buys the same quality on the queries that need it.

**Q11 (pushback). Isn't routing just a hack that loses quality on misclassified queries?**
It is a model, so it has errors, and I treat it that way — I log routing decisions alongside the downstream quality signal and tune thresholds and retrain from that feedback. But the comparison is not "router vs. perfect routing," it is "router vs. paying 6x for patterns that mostly harm the cases they were not designed for." And I always keep an escalation path: if the fast path's top evidence score is below a floor, or a sufficiency check fails, the request escalates to a deeper path rather than answering shallowly. Routing plus escalation is strictly better than either uniform fast or uniform deep.

**Q12 (pushback). Does RAPTOR or Graph RAG make contextual retrieval unnecessary?**
No, they solve different problems and they compose. Contextual retrieval fixes chunk-level self-sufficiency — the chunk now says what it is about, so it gets retrieved and understood. RAPTOR fixes coverage at the corpus level — you can retrieve a summary node for "what are the themes" questions that no leaf chunk answers. Graph RAG fixes relational structure — you can traverse from an entity to its dependencies. A real system uses CCH on the leaves, a summary tree above them for global questions, and a graph alongside for relational ones, each behind a router. They are additive, not substitutable.

**Q13. What's the operational cost of multimodal RAG that people underestimate?**
The vector count. A page-image approach produces roughly 1024 patch embeddings per page, so 1M pages becomes ~1B vectors, and at 128 dims that is still ~500 GB before index overhead. Add 2–5 seconds per page to embed and $0.01–0.05 per page if you use a hosted VLM, and the ingestion bill for a mid-size archive is five figures and hundreds of GPU-hours. I do the multiplication before committing, and I prefer a dual representation — text chunks for body text, image embeddings only for genuinely visual content like charts and diagrams.

**Q14. How would you debug a case where decomposition made things worse?**
First I check whether the router is over-triggering — decomposition firing on atomic factoids, where splitting the query destroys the retrieval signal because each fragment is too underspecified to match anything. That is the classic decomposition regression. Then I check the shared rerank pool: if each sub-question reranks independently and I concatenate, I can end up with 5 chunks per sub-question of which 3 are irrelevant to the composed answer. The fix is one rerank pass over the union, scored against the *original* question, not the sub-questions. And I'd cap sub-questions at 3.

**Q15. Give me the one-line rule for choosing among these patterns.**
Fix the cheapest, highest-yield structural problems first — contextual retrieval and parent-document — because they are ingestion-time, cost nothing at query time, and fix the most common failure modes. Then add a router so expensive query-time patterns only fire where they help. Then, and only then, add agentic behavior for the multi-hop slice you can actually measure. Never invert that order: an agentic loop over a badly chunked, single-vector, unreranked index just burns tokens retrieving the wrong things.

**Q16. Name the retrieval strategies you'd consider, and then tell me how you'd organise them.**
Top-K, similarity threshold, hybrid, multi-query, metadata filtering, parent retrieval, contextual retrieval, multi-vector, hierarchical, and query routing. Recited flat that is a list, not an answer — so I sort them. Strategies one through five change **what you retrieve** at query time: top-K and threshold control how much, hybrid controls which signals (lexical plus semantic), multi-query and metadata filtering control which candidate set. Strategies six through nine change **what you index**: parent-document, contextual, multi-vector and hierarchical all cost money at ingestion and nothing at query time, which makes them the cheapest quality you can buy. Query routing sits above all of them and decides which of the other nine run at all. And I'd close by re-anchoring: all of this is still retrieve, augment, generate — the strategies are ways of making the *retrieve* step find the right evidence, not alternatives to RAG.

**Q17. What's the difference between corrective RAG, agentic RAG, and adaptive RAG?**
They are three answers to three different questions. **Corrective RAG** answers "what if the retrieved documents are wrong?" — it grades each retrieved document before generating and, on the incorrect branch, discards the retrieved set entirely and falls back to another source. Its defining property is permission to throw the retrieval away. **Agentic RAG** answers "what if one retrieval isn't enough?" — the agent judges whether the evidence is sufficient and, if not, decides which source to query next, looping until it is or the budget runs out. **Adaptive RAG** answers "when should I even bother?" — a router grades the question as simple, medium or complex before anything is retrieved: simple answers directly, medium does retrieval with one or two iterations, complex goes agentic. Adaptive RAG is the control plane; the other two are the paths it routes to. In production I would ship adaptive RAG first, because it is what makes the other two affordable.

### Pushback round

**P1. Isn't corrective RAG just a retry with extra steps?**
No, and the difference matters. A retry re-runs the same query against the same index and gets the same bad documents — that is exactly the retrieval thrash failure. Corrective RAG changes something structural: on the incorrect branch it *abandons the retrieved set* and goes to a different source — a larger index, the web, an alternate corpus. That is a different retrieval, not the same retrieval again. The grade step is also what produces the signal — you learn *which* queries retrieve badly, which is how you find out your corpus has a gap or your chunking is wrong for that document class. A blind retry teaches you nothing and, in an agentic loop, actively wastes tokens.

**P2. Graph RAG sounds like an over-engineered join. Why not just put the relationships in the metadata and filter?**
Sometimes that is exactly the right answer, and you should say so — for one-hop relationships, metadata filtering is cheaper and you should use it. The graph earns its place when the relationship is **multi-hop and variable-length**: "which departments are affected if this supplier contract is terminated" requires traversing contract → clause → obligation → department, and the number of hops depends on the document, so there is no metadata field that expresses it in advance. The second thing the graph buys is that the traversal happens *at retrieval time and is auditable* — the path is the citation. That said, the bar is real: if fewer than about 30% of your questions are genuinely relational, you are paying full extraction and quarterly-refresh cost for a minority slice, and graph-as-reranker gets most of the benefit for a fraction of it.

---

## 10. Common mistakes candidates make

| Mistake | Why it is wrong | Do this instead |
|---|---|---|
| Listing patterns without prices | Every pattern has a latency/cost/dollar cost; a list is not a design | Quote the trade-off table: "+600 ms, +$0.001, +5–12 recall pts" |
| Applying every pattern to every query | Makes retrieval-side overhead 36% of the bill and blows the latency budget | Route; pay only on the slice that benefits |
| Adopting HyDE without a fallback | HyDE fabricates vocabulary and can degrade recall | Parallel original query + RRF fusion, or don't ship it |
| Decomposition without a shared rerank pool | k × sub-questions chunks flood the context | Rerank the union against the original question, cap at 3 sub-Qs |
| Agentic loops with no step budget | 8–12 s and unbounded cost; thrash is common | Cap at 3–4 iterations plus a token and dollar budget |
| Claiming Graph RAG is generally better | It is worse unless ≥30% of questions are relational | State the bar and the maintenance tail |
| Forgetting contextual retrieval's ingestion cost | A full-corpus LLM pass is real money | Use prompt caching; start with heading-path headers for free |
| Treating parent-document as free | 2x storage and a consistency hazard | Write parent+child transactionally; reconcile orphans |
| Ignoring ColBERT's storage | 100x uncompressed; 30–100x even compressed | Use ColBERTv2 + PLAID, or deploy as a reranker over a cheap first stage |
| No eval for the router itself | The router is a model with its own error rate | Log routing decisions against downstream quality; keep an escalation path |
| Shipping a pattern without a canary | Collateral damage to the dominant query class is invisible | Flag + 5% canary + per-class metric breakdown |
| Confusing "more context" with "better context" | Iterative loops accumulate redundant context and quality falls | Re-assemble context from current best evidence each iteration |
| Quoting RAPTOR/GraphRAG without the ingestion pass | It is a full-corpus summarization/extraction, not a config flag | Say the pass, the model, and the refresh cadence |
| Reciting the RAG variants as a list of buzzwords | "Corrective, agentic, graph, self, modular" with no triggers is a vocabulary test, not a design answer | Lead with the **symptom** each variant fixes: weak retrieval, multi-hop, relational, faithfulness, composability |

---

## 11. Cheat-sheet summary

| Dimension | Answer |
|---|---|
| **Three defect classes** | Vocabulary mismatch → query transforms. Chunk isolation → CCH/parent-doc. Multi-hop → decomposition/agentic/graph |
| **Best ROI pattern** | Contextual retrieval: 5.7% → 1.9% retrieval failure (67% reduction), $0.0003/chunk one-off |
| **Second best** | Parent-document: small-chunk precision + large-chunk context, 2x storage |
| **Mandatory for chat** | Query rewrite / decontextualization, ~100–200 ms, ~$0.0002 |
| **Query transforms** | Rewrite, multi-query (3, RRF k=60), HyDE (2–3 averaged), decomposition (≤3 sub-Qs), step-back |
| **Control plane** | Router — 20–50 ms classifier, 4 classes; makes everything else affordable |
| **Cost with routing** | Retrieval-side ≈ 6% of bill. Full stack on every query ≈ 36%. 9x difference |
| **Graph RAG bar** | ≥30% relational questions; graph-as-reranker = 70–80% quality at ~20% cost |
| **Graph RAG tail** | Quarterly/event-driven community refresh; extraction quality is the bottleneck |
| **RAPTOR** | Recursive summary tree, depth 3–4; the only clean answer to "summarize the corpus" |
| **ColBERT** | 128-dim per token, MaxSim; ~1 TB → 64 GB (v2) → 30 GB (PLAID) at 10M docs |
| **Multimodal** | ~1024 patches/page, 128 dims, 1M pages ≈ 1B vectors, 2–5 s/page, $0.01–0.05/page |
| **CRAG branches** | correct → refine+generate; ambiguous → +alternate source; incorrect → discard retrieved set |
| **Self-RAG** | Critique tokens inside generation; +1–3 s; use when faithfulness is the binding constraint |
| **Iteration cap** | **3–4 max**; 3–4 iterations = 8–12 s; refuse duplicate sub-queries |
| **Adaptive retrieval** | 20–30% of traffic needs no retrieval — save the cost |
| **Latency ladder** | Fast 1.6 s → transforms 2.3 s → decomposition 2.9 s → agentic 9–12 s |
| **Above 2.5 s** | Must be an explicit "deep research" mode or async — never a silent fallback |
| **Adoption rule** | Diagnose the failing stage → adopt the matching pattern → measure the full golden set → canary 5% |
| **Biggest hidden risk** | Collateral damage to the dominant query class from a pattern that helps a minority class |
| **Debug order** | Retrieval miss → rank miss → context miss → generation miss. Fix in that order |
| **Retrieval strategies (10)** `[S]` | top-K · similarity threshold · hybrid (lexical + semantic) · multi-query · metadata filtering · parent retrieval · contextual · multi-vector · hierarchical · query routing |
| **Sort the strategies** `[S]` | 1–5 change **what you retrieve** (query time) · 6–9 change **what you index** (ingestion cost, free at query time) · 10 decides which of the others run |
| **Adaptive ladder** `[S]` | Effort by difficulty: simple → answer directly · medium → retrieve, 1–2 iterations · complex → agentic multi-step |
| **Variant = trigger** `[S]` | weak/wrong retrieval → corrective · multi-step question → agentic · relationship → graph · sometimes unnecessary → self-RAG · reusable composite → modular |
| **Corrective RAG's defining property** `[S]` | "Grade before you answer" — and on the incorrect branch, **discard** the retrieved set rather than retrying it |

---

## 12. Cross-links

- [Production RAG System](01-production-rag-system.md) — the baseline pipeline these patterns extend
- [RAG Evaluation and Observability](03-rag-evaluation-and-observability.md) — the golden set that decides which pattern to adopt
- [Guardrails and Safety for AI Systems](04-guardrails-and-safety-for-ai-systems.md) — injection risk amplified by iterative retrieval
- [Agentic System Design](06-agentic-system-design.md) — step budgets, trajectory eval, loop termination
- [MCP, Tool Use and Multi-Agent](07-mcp-tool-use-and-multi-agent.md) — retrieval as a tool; tool-schema design
- [Memory and Context Engineering](08-memory-and-context-engineering.md) — context budgeting when iterations accumulate evidence
- [LLM Serving, Inference and Gateway](05-llm-serving-inference-and-gateway.md) — the router as a gateway concern
- [Production RAG Case Study](../../Case-Studies/HLD/09-production-rag-case-study.md) — patterns in a full worked design
- [Cost and FinOps](../../Operations/06-cost-and-finops.md) — per-pattern cost attribution
- [RAG and LLM Cheatsheet](../../Cheat-Sheets/04-rag-and-llm-cheatsheet.md) — revision sheet
