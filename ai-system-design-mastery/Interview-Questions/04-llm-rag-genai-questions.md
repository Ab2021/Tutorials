# LLM, RAG and GenAI Interview Questions

**TL;DR** — Fifty questions spanning RAG fundamentals, chunking, retrieval and ranking, advanced patterns, evaluation, LLM serving and inference optimisation, and the fine-tuning-versus-RAG-versus-prompting decision. Ordered easy → hard → pushback within each section. Every answer is written the way you'd say it out loud: a direct claim first, then the mechanism, then the number or trade-off that proves you've operated a system rather than read about one.

**How to use this bank**

| If you have | Do this |
|---|---|
| 30 minutes | Read the bolded first sentences only — that's the answer skeleton |
| 2 hours | Read every answer once, out loud, against a timer |
| 2 days | Answer each one from memory, then diff against the written answer |
| Before an onsite | Re-read the pushback section — that's where offers are won and lost |

**The three habits that separate strong answers from weak ones:**

1. **Lead with the claim, not the context.** "I'd use hybrid retrieval with RRF" beats "Well, there are several retrieval approaches and it depends..."
2. **Give one number.** A chunk size, a recall figure, a latency budget, a cost delta. Numbers are the fastest credibility signal in an AI interview.
3. **Name the failure mode.** Anyone can describe how a system works. Naming how it breaks — and what you'd monitor to catch it — is what operators do.

---

## Section A — RAG fundamentals

### Q1. What is RAG and why does it exist?

**A.** Retrieval-Augmented Generation is a pattern where you retrieve relevant documents at query time and put them in the model's context, so the model answers from evidence rather than from parametric memory. It exists to solve three problems that are unsolvable by prompting alone. First, **knowledge cutoffs and freshness** — a model trained to a date cannot know last week's policy change, and retrieval lets you update the corpus in minutes rather than retraining in months. Second, **hallucination on specific facts** — a model asked about your internal refund policy has no basis to answer, and giving it the actual policy text converts a guess into a grounded answer. Third, **attribution** — a retrieval system can cite the source chunk, which is what makes the answer auditable, and auditability is frequently the actual product requirement in enterprise settings.

The one-line framing I'd use: **prompting gives instructions, RAG gives knowledge, fine-tuning gives behaviour.** They answer different questions and they're not substitutes.

### Q2. Walk me through the architecture of a production RAG system.

**A.** Two pipelines. The **ingestion pipeline** runs offline: source documents are parsed, cleaned, chunked (typically 500–800 tokens with about 12% overlap), embedded with a pinned model, and written to a vector store alongside their metadata and ACL. Most serious systems also build a lexical index (BM25) at the same time, because hybrid retrieval beats either alone. There's usually an enrichment pass — contextual retrieval, where each chunk gets a short LLM-generated summary of its place in the document, which reduced retrieval failure from 5.7% to 1.9% in the published Anthropic work.

The **query pipeline** runs online, and the latency budget usually looks like: query rewrite or expansion (~50 ms), embedding (~30 ms), hybrid retrieval of 50–100 candidates (~100 ms), fusion with RRF (~10 ms), cross-encoder reranking down to the top 5 (~250 ms), generation (~2,000 ms), plus guardrails (~400 ms). The rerank step is where most of the quality comes from, and it's usually the last thing teams add.

### Q3. Why not just put all the documents in the context window?

**A.** Three reasons, and the third is the one people don't expect. **Cost** — context is priced per token on every request, so a 400,000-token corpus costs the same on request ten thousand as on request one, whereas retrieval costs a fixed small amount. **Latency** — attention is superlinear in sequence length, so a long context is slower even with FlashAttention, and TTFT degrades noticeably past a few tens of thousands of tokens. And **quality** — the "lost in the middle" effect is real: models attend most reliably to the beginning and end of a long context, so stuffing 100 documents in and hoping the relevant one gets noticed is measurably worse than retrieving the best 5. There's also a practical ceiling: even a 1M-token window doesn't hold a 50-million-document corpus.

The honest counterpoint is that long context has eaten some RAG use cases. For a 200-page contract you're querying repeatedly, loading the whole thing once with prompt caching is simpler and often better than chunking it. So the decision is per-corpus: retrieval for corpora too large to fit, long context for corpora that fit and are queried often.

### Q4. What is hybrid retrieval and why does it beat dense-only?

**A.** Hybrid retrieval runs a dense (embedding) search and a sparse (lexical, usually BM25) search in parallel and merges the results. It wins because the two methods fail differently. Dense retrieval captures semantic similarity — it finds "how do I cancel my subscription" when the document says "terminating your plan" — but it's weak on rare tokens: product codes, error identifiers, proper nouns, and exact strings, because those get smeared into an averaged embedding. BM25 is the opposite: it nails exact term matches and fails completely on paraphrase.

Merging is usually done with **Reciprocal Rank Fusion**, `score = Σ 1/(k + rank)` with `k = 60`. RRF only uses ranks, not raw scores, which is exactly why it works — a cosine similarity of 0.82 and a BM25 score of 14.3 are not comparable numbers, and any attempt to normalise and add them is fragile. Typical gain: 8–15 points of recall@10 over dense-only on mixed query workloads.

### Q5. What is reranking and why is it a two-stage architecture?

**A.** Reranking is a second, more expensive ranking pass over a small candidate set. The reason it's two stages is a compute trade-off: the retriever has to score millions of documents, so it must be cheap — which means it's approximate, usually a bi-encoder where query and document are embedded independently and compared by cosine similarity. A cross-encoder feeds the query and document through the model *together*, so it can model their interaction, which makes it far more accurate and far too slow to run over the whole corpus.

So the shape is: retrieve 50–100 candidates cheaply, rerank them expensively, keep the top 5. Typical numbers: a cross-encoder reranker takes nDCG@10 from about 0.65 to 0.78, adds around 250 ms, and costs a fraction of a cent. The rule I'd state is that if your RAG system has a quality problem and you haven't added a reranker yet, that's the highest-return change you can make — before touching chunk sizes or swapping embedding models.

### Q6. How do you handle documents that are tables, images, or scanned PDFs?

**A.** With a document-processing stage that runs before chunking, because chunking a scanned invoice as if it were prose produces garbage. The pipeline is: layout detection to identify regions (text block, table, figure, header), OCR for scanned content, and table extraction to structured form — because a table flattened into a text stream loses the row-column relationships that carry the meaning. For tables I'd extract to markdown or CSV and embed a natural-language summary of the table alongside the structured data, so the retriever can find it and the generator can reason over it.

For figures, generate a caption with a vision model at ingest time and index the caption; that makes the image retrievable by text without needing a full multimodal retrieval stack. Quality signal: measure the parse error rate at ingest, per source type. If tables are 3% of your corpus and parsing fails on 20% of them, that's your retrieval quality ceiling, and no amount of reranking fixes a corpus that was ingested wrong.

---

## Section B — Chunking and indexing

### Q7. How do you choose a chunk size?

**A.** Start at 500–800 tokens with 10–15% overlap, then measure. The trade-off is real in both directions: small chunks give precise retrieval but fragment context, so the generator gets a sentence without the paragraph that explains it; large chunks preserve context but dilute the embedding, so the vector represents several topics and matches none of them well. Overlap exists because a fact that straddles a boundary is otherwise lost by both chunks.

The refinement that matters more than the base size is **structure-aware chunking**: split on document structure — headings, sections, list boundaries, code blocks — rather than on a fixed token count, and only fall back to fixed windows when the structure is absent. And I'd tune empirically rather than by intuition: build a retrieval eval set of 100–200 query-document pairs, then sweep chunk sizes at 256, 512, 768, and 1,024 and compare recall@10 and nDCG. The answer is usually corpus-specific and it's usually not the default.

### Q8. What is contextual retrieval and when is it worth the cost?

**A.** Contextual retrieval prepends a short, LLM-generated context sentence to each chunk before embedding — something like "This chunk is from the section on international refund rules in the 2026 policy handbook." It fixes the orphan-chunk problem: a chunk that says "the deadline is 30 days" is nearly unretrievable on its own, because the embedding has no idea what deadline or which 30 days.

The published numbers are strong: retrieval failure rate dropped from 5.7% to 1.9% when combined with hybrid search and reranking. The cost is one LLM call per chunk at ingest — for 10 million chunks at roughly 300 tokens each, that's a few thousand dollars and several hours. So the decision is: worth it for a high-value, relatively static corpus where retrieval quality is the product; not worth it for a large, rapidly-changing, low-stakes corpus where the ingest cost and latency would dominate. And note it's a one-time cost per corpus version, not per query.

### Q9. What's the difference between HNSW and IVF indices?

**A.** Both are approximate nearest-neighbour indexes, and they trade recall for speed differently. **HNSW** builds a multi-layer navigable small-world graph — you enter at the top layer, which is sparse and long-range, and descend to denser layers for refinement. It gives excellent recall and low latency, but it has a large memory footprint (roughly 1.5x the raw vectors) and slow, expensive inserts. Typical parameters: `M = 32` connections per node, `ef_construction = 256` for build quality, `ef_search = 64–128` for the query-time breadth. Higher `ef_search` means better recall and more latency.

**IVF** clusters vectors into `nlist` cells and searches only the nearest `nprobe` cells. It's much more memory-efficient and scales to very large corpora, but recall depends heavily on `nprobe` tuning and it degrades on data where clusters are uneven. The practical guidance: HNSW for up to roughly 10–50 million vectors where latency matters, IVF-PQ for hundreds of millions or billions where memory dominates, and a quantised variant (product quantisation, or binary quantisation which gives about 32x memory reduction at some recall cost) whenever the memory bill is the binding constraint.

### Q10. How do you keep an index fresh, and what does staleness cost?

**A.** Freshness is a pipeline problem, not a database feature. The typical design: change detection at the source (CDC, webhooks, or a scheduled diff), incremental upsert rather than full rebuild, and a freshness SLO — "a document edited at time T is retrievable by T + 15 minutes." For deletes, you need an explicit delete path, and orphaned vectors are a common bug: the document is gone from the source but the vector remains, so the system confidently cites a deleted policy.

Staleness costs are asymmetric and mostly invisible. The system doesn't error; it answers from an outdated version of the truth, which is worse, because the answer looks authoritative. So you need both a freshness SLI and a **reconciliation job** — a daily scan comparing the index against the source, reporting orphaned vectors and missing documents. And for embedding-model upgrades, the rule is absolute: changing the embedding model requires a full re-index, and you must never mix vectors from two models in one collection, because their spaces aren't comparable and the similarity scores become meaningless.

### Q11. How do you handle multi-tenancy in a vector store?

**A.** Three models, and the choice is a cost-isolation trade-off. **Silo** — one index per tenant — gives the strongest isolation and the highest cost; it's right for enterprise and regulated tenants. **Pool** — one shared index with a mandatory `tenant_id` filter on every query — is cheapest and right for SMB SaaS at scale. **Bridge** — silo the top N tenants by size or contract, pool the rest — is what most real systems end up doing; the reference pattern I'd cite is the top 20 tenants siloed and the remaining 480 pooled.

The non-negotiable rule is that the tenant filter is enforced **in the retrieval query**, not in the prompt. "Only answer from documents this user can see" in a system prompt is not an access control — it's a request, and it fails the moment an injection asks the model to ignore it. So the query carries `WHERE tenant_id = $1 AND acl_group = ANY($2)` at the vector-store level, the ACL is stored with the chunk at ingest time, and there's a CI test that seeds tenant B's documents and asserts tenant A never retrieves them. Any cross-tenant retrieval event is a critical alert, not a warning.

---

## Section C — Retrieval and ranking

### Q12. What retrieval metrics would you track?

**A.** Two families, and separating them is the whole point. **Retrieval quality**: `recall@k` (did we get the relevant documents at all — the ceiling on everything downstream), `precision@k` (how much of what we retrieved is relevant — drives context pollution), `MRR` (is the best document ranked near the top), and `nDCG@k` (a graded relevance measure that accounts for position, which is the best single number because it reflects what the generator actually sees).

**Operational**: retrieval latency p50/p95, index size, freshness lag, filter latency, and the score distribution of returned results. That last one is underrated — if the top score collapses from 0.82 to 0.55 across the board, the corpus or the embedder changed, and you'll see it there before you see it in quality scores. And I'd always measure retrieval quality separately from generation quality, because a bad answer has two possible causes with completely different fixes, and conflating them is how teams spend a week tuning prompts when the problem was recall.

### Q13. When do you use a query rewrite, and what are the failure modes?

**A.** Query rewriting runs a cheap LLM pass to turn the user's utterance into a better retrieval query — resolving pronouns from conversation history ("what about the second one?" → "what is the cancellation policy for the Pro plan?"), expanding acronyms, and generating multiple paraphrases for multi-query retrieval.

Failure modes, in order of how often I've seen them: **topic drift**, where the rewrite invents a specific interpretation the user didn't mean and narrows retrieval to the wrong thing — this is why the rewrite should preserve ambiguity rather than resolve it. **Latency**, an extra 50–150 ms on the critical path. **Cost**, one more small-model call per query. And **silent degradation on short queries** — a two-word query has little to rewrite, so the rewriter hallucinates context. My mitigation is to log the original and rewritten query side by side and sample-review them weekly, because a rewriter that's 5% wrong is invisible in aggregate metrics and very visible to the users it affects.

### Q14. What is multi-query retrieval and when is it worth 3x the retrieval cost?

**A.** Generate 3–5 paraphrases of the query, retrieve for each, and merge the results — usually with RRF. It helps when the user's phrasing is unlikely to match the document's phrasing, which is exactly the paraphrase gap hybrid search only partially covers. Worth it for short, ambiguous, or vocabulary-mismatched queries.

It is not worth it universally, because it triples retrieval cost and adds latency and, more subtly, it can *reduce* precision by pulling in documents that match a bad paraphrase. So I'd apply it selectively — route to multi-query only when the query is short, when the initial retrieval's top score is low, or when the corpus is known to have vocabulary mismatch. That conditional routing gets most of the recall benefit at a fraction of the cost. The signal to watch is precision@k: if multi-query raises recall and drops precision, you're now doing more reranking work to recover.

### Q15. How does Reciprocal Rank Fusion work and why k=60?

**A.** RRF combines ranked lists without needing comparable scores: for each document, sum `1/(k + rank)` across all the lists it appears in, and sort by the total. The genius of it is that it only uses rank position, which sidesteps the fact that a cosine similarity of 0.82 and a BM25 score of 14.3 live on incomparable scales — any normalisation scheme is fragile and corpus-dependent.

`k = 60` comes from the original TREC work and it functions as a smoothing constant that controls how much the top ranks dominate. Small `k` makes rank 1 vastly more valuable than rank 5, which is aggressive and brittle; large `k` flattens the curve toward uniform, which loses the ranking signal. 60 is the empirically-derived default that works well across corpora, and it's worth knowing that it's a tunable — if your retrieval is dominated by one retriever, raising `k` gives the other one more influence. And the document that appears in *both* lists gets the sum of both contributions, which is exactly the consensus signal you want.

### Q16. How do you evaluate whether a retrieval miss or a generation failure caused a bad answer?

**A.** By measuring each stage independently and having a labelled set that separates them. The diagnostic sequence: check `context_recall` first — if the relevant document wasn't retrieved, the generator never had a chance and no prompt change will fix it. If recall is fine, check `context_precision` — too much irrelevant context is a real cause of bad answers because it dilutes attention. If both are fine and the answer is still wrong, then it's the generator: either the instruction was unclear or the model ignored the evidence.

In production, the cheap version of this is instrumenting the trace so every request records retrieved chunk IDs, their scores, and the reranked order, then sampling failures and manually categorising them as retrieval-miss, retrieval-precision, or generation. It's unglamorous and it takes a few hours a week, but a misattributed failure is expensive — teams that assume generation problems when they have retrieval problems spend months tuning prompts against a ceiling that prompts cannot move.

### Q17. What is the role of score thresholds in retrieval?

**A.** A threshold turns "retrieve the top k" into "retrieve the top k but only if they're actually relevant," which is what enables abstention. Without a threshold the system always has context and therefore always answers, even when the corpus doesn't contain the answer — which is the single most common cause of confident hallucination in RAG.

Setting it: don't guess a cosine number. Instead, take a labelled set with positive and negative pairs, plot the score distributions, and pick the threshold that maximises the separation — usually somewhere around 0.7 for cosine similarity on a well-calibrated embedder, but it's embedder-specific. The critical calibration point is that **both directions of error are failures**: a threshold too low produces over-confident wrong answers, and one too high produces excessive abstention, and users route around a system that says "I don't know" too often. So I set a target abstention band — typically 8–15% when the corpus genuinely lacks coverage — and alert when the rate leaves it in either direction.

### Q18. What is late-interaction retrieval (ColBERT) and when would you use it?

**A.** ColBERT keeps a vector per *token* rather than per document, and scores with MaxSim — for each query token, take the maximum similarity over all document tokens, and sum. It sits between bi-encoders (fast, one vector per document) and cross-encoders (accurate, but must see query and document together). It gets most of the cross-encoder's accuracy at a fraction of the latency, typically 50–100 ms even at 10M+ documents, using a multi-stage PLAID pipeline that prunes 99% of candidates by centroid and only decompresses residuals for promising ones before exact scoring on fewer than 100 documents.

The cost is storage: one vector per token means 10–100x the index size of a single-vector-per-document approach. So I'd use it for medium-scale corpora — roughly 1M to 50M documents — where accuracy is paramount and the storage overhead is affordable. Below about a million documents, hybrid search with a cross-encoder reranker is simpler and good enough; above fifty million, the storage bill usually forces you back to a quantised single-vector index.

---

## Section D — Advanced RAG patterns

### Q19. What is GraphRAG and when does it beat vector RAG?

**A.** GraphRAG builds a knowledge graph — entities and their relationships — from the corpus, usually with an LLM extraction pass, and answers queries by traversing the graph rather than (or in addition to) similarity search. It wins on a specific query class: **multi-hop and global questions**. "Which suppliers are affected by the semiconductor shortage across all our product lines?" requires joining facts that live in separate documents, and vector search retrieves documents *similar to the query*, not documents *connected through a relationship*. A graph traversal follows the connections.

It loses on cost and freshness: the extraction pass is expensive (an LLM call per chunk plus entity resolution), and the graph must be rebuilt or incrementally updated as the corpus changes, which is harder than upserting a vector. So the decision is query-shape-driven. If your users ask "what does document X say about Y," vector RAG with a reranker is better, cheaper, and simpler. If they ask relational and aggregative questions across the whole corpus, GraphRAG is the only thing that works, and I'd scope it to the entity types that matter rather than trying to graph everything.

### Q20. What is agentic RAG and how does it differ from a fixed pipeline?

**A.** Agentic RAG gives the model control over retrieval: it decides whether to retrieve, what query to use, which index or tool to search, whether the results are sufficient, and whether to retrieve again. A fixed pipeline does one retrieval and one generation; an agentic system might decompose a question into sub-questions, retrieve for each, discover one sub-answer is missing, reformulate, and then synthesise.

It's better on complex questions and much harder to operate. The costs: latency goes from one round trip to 3–10, cost multiplies accordingly, and you inherit all the agent failure modes — loops, step creep, premature commitment. So I'd gate it: use a fixed pipeline when a single retrieval suffices (classify the query first), and route to the agentic path only for multi-hop or exploratory questions. And the loop needs ceilings inside it — a step budget, a retrieval-call budget, and a no-progress detector — because an agent that retrieves eleven times without improving is common and expensive.

### Q21. What is RAPTOR and what problem does it solve?

**A.** RAPTOR builds a tree of recursively summarised clusters: chunks are clustered by embedding similarity, each cluster is summarised by an LLM, and the summaries are clustered and summarised again, up to a root. Retrieval then searches across all levels, so the system can answer a high-level question from a summary node and a specific question from a leaf chunk. It solves the summarisation-versus-specificity tension that a flat chunk index can't: a question like "what are the main themes across this 500-page report?" has no matching leaf chunk, because the answer is distributed across all of them.

The costs are honest: it's an LLM call per cluster plus a hierarchy build, so for a large corpus it's thousands of dollars and a multi-hour build (I'd budget roughly $2,000 and six to ten hours for a 10-million-chunk corpus), and it needs rebuilding when the corpus changes materially. So it's right for a relatively static, high-value document collection where global questions are asked. And because the rebuild is expensive, I'd snapshot the tree rather than treat it as a purely derived artifact — "rebuildable" doesn't mean "cheap to rebuild."

### Q22. How do you handle a query that needs information from multiple documents?

**A.** Three escalating strategies. **Wider retrieval** — increase k and rely on the reranker to keep the best across documents; cheap, works when the documents are topically close. **Query decomposition** — an LLM splits the question into sub-questions, each retrieved separately, then the results are merged and synthesised; this is the right answer for genuinely multi-part questions and it's what I'd reach for first. **Agentic retrieval** — the model iterates, retrieving, assessing, and reformulating until it has enough; the most capable and the most expensive and least predictable.

The failure mode to guard against is that multi-document synthesis is where contradiction shows up: two retrieved documents disagree, and the generator either picks one silently or blends them into something that's true of neither. So for multi-document answers I want the prompt to explicitly handle conflict — "if sources disagree, state both and note the disagreement" — and I want the eval set to include contradiction cases, because they're rare in production and catastrophic when they occur.

### Q23. What is semantic caching and what's the risk?

**A.** A semantic cache embeds the incoming query and looks for a previously-answered query whose embedding is above a similarity threshold, then returns the cached answer without calling the model. It's one of the highest-return cost optimisations available — typical hit rates of 20–40% on support-style traffic, and it cuts both cost and latency proportionally.

The risk is the **false hit**, and it's a quality risk, not a cost risk. "What's the refund window for the Pro plan?" and "What's the refund window for the Enterprise plan?" are semantically near-identical and have different answers. So the threshold has to be tuned conservatively — I'd rather have a 20% hit rate with a 0.2% false-hit rate than a 40% hit rate with a 3% false-hit rate — and the cache key must include every dimension that changes the answer: tenant, locale, user permissions, plan tier, and the release version. That last one is a common bug: a cache that isn't invalidated on a release serves answers from the previous prompt for days. And I'd measure the false-hit rate explicitly by sampling cache hits and re-running them fresh, because it's otherwise completely invisible.

### Q24. How would you build RAG over a corpus that changes hourly?

**A.** Three changes from the standard design. **Incremental ingestion** rather than batch rebuild — change detection at the source, upsert of only the changed documents, and a delete path for removals, all within a freshness SLO of minutes rather than hours. **A tiered index** — a small, fast-updating "hot" collection for recent documents and a larger "cold" collection for the stable corpus, queried together, so the expensive full re-index only runs for structural changes rather than every update. **And drop the expensive enrichment for the fast tier** — contextual retrieval and RAPTOR-style summarisation are one-time-per-version costs, so they run on the cold corpus and are skipped for documents that will be replaced within a day.

The operational pieces that matter most here are the reconciliation job (comparing index against source daily for orphans and gaps), the freshness SLI, and cache invalidation wired to ingestion — because a semantic cache with a 24-hour TTL over an hourly-changing corpus is serving stale answers to a third of your traffic and nothing in the metrics will say so.

---

## Section E — Evaluation

### Q25. How do you evaluate a RAG system?

**A.** At three levels, because each has a different failure and a different fix. **Retrieval**: recall@k, precision@k, MRR, nDCG — measured against a labelled set of query-document pairs. **Generation**: faithfulness (is every claim supported by the retrieved context), answer relevance (does it address the question), and completeness. **End-to-end**: answer correctness against ground truth where it exists, plus user signals — thumbs up/down, regeneration rate, task completion.

The process: build an eval set of 200–500 examples with hand-labelled relevance, run it on every change as a CI gate, sample 1–5% of production traffic for judge-based quality monitoring, and route disagreements to humans to keep the gold set honest. And critically, break results down by **slice** — query type, difficulty, tenant — because a 4.3 aggregate can hide a 2.8 on multi-hop questions, and the aggregate is the number that lies.

### Q26. What are the limitations of LLM-as-judge?

**A.** Four known biases and one structural limitation. **Position bias** — the judge prefers whichever option came first; mitigated by swapping positions and keeping only consistent verdicts. **Length bias** — longer looks better; mitigated by instructing the judge to ignore length and by normalising. **Self-preference** — a judge favours its own model family's style; mitigated by using a different model as judge than the one under test. **Format bias** — markup and structure influence the score; mitigated with diverse calibration examples.

The structural limitation is that a judge is only reliable inside the distribution it was calibrated on. Highly domain-specific content, subtle factual errors, and safety edge cases are where judges fail, and they fail silently — the score looks fine while the verdict is wrong. So the operational answer is: calibrate the judge against human labels and track Cohen's kappa (target ≥ 0.6), never rely on a judge alone for high-stakes decisions, and use judges for rapid iteration rather than as the final word.

### Q27. What is a golden set and how do you build one?

**A.** A golden set is a fixed, hand-labelled evaluation dataset that you never tune against. Building it: sample real production queries stratified across the query types you care about (factual, multi-hop, no-answer, ambiguous, adversarial), include the hard cases rather than the easy ones, get 2 annotators on a 10–20% overlap so you can compute agreement, and write the rubric down with 3–5 examples at each score level — because without examples, annotators invent their own scale and kappa collapses.

Two rules make it worth having. It's **versioned**, with every change in its own pull request and a second reviewer. And it **never ships in the same release as the change it measures** — because a release that can edit its own exam isn't being tested, and the temptation to relax an assertion to make a red build green is universal. Keep a small held-out slice that you look at rarely, so your main set can be iterated on without losing the ability to detect that you've overfitted it.

### Q28. How do you know your eval set hasn't gone stale?

**A.** Three signals. **Score inflation** — if the aggregate climbs steadily without a corresponding product change, the set is being overfitted, usually by the team tuning against it. **Distribution drift** — if the production query mix no longer resembles the eval set's categories, the set is measuring the wrong thing; I'd compare the category distribution monthly and refresh a fraction of the cases from recent production traffic. **Kappa decay** — if annotators increasingly disagree on the older cases, the rubric no longer reflects the current product intent.

The refresh discipline is: replace 10–20% of the set each quarter with recent production failures, but keep a frozen core that never changes, so you retain a consistent baseline across time. And the failure to watch for is a set that's all easy cases — it saturates at 4.8, the gate never catches anything, and its greenness becomes false comfort. The signal for that is the score distribution: a healthy eval set has a spread and a meaningful tail, not a spike at 5.

### Q29. How do you evaluate an agentic system?

**A.** Grade the trajectory, not just the final answer, and use deterministic post-conditions where possible. The final answer hides three failure shapes: the right answer reached by wrong reasoning, the right answer reached after four destructive tool calls, and the right answer reached with forty-seven retrieval calls instead of two. So I score the path — process reward models score each step, and an auditor agent replays the trajectory asking whether each step was justified at each node. The failure modes I'd track explicitly are reasoning-action mismatch, over-retrieval, tool flailing, premature commitment, and self-jailbreaking.

For reliability I use **Pass^k**, the probability of succeeding on all k trials. Pass^1 of 70% with Pass^4 of 12% says the agent works on the easy path but can't recover from any perturbation — and that gap is invisible in a single-run success rate. Grading should be verifier-based wherever possible: check deterministic post-conditions (the order is cancelled, the refund exists, the seat is changed) rather than having a judge read the transcript, because post-conditions are cheap, unambiguous, and impossible to talk your way past.

### Q30. What's the difference between offline and online evaluation?

**A.** They're the same system from two angles and treating them as separate is the mistake. **Offline** runs before shipping, on a golden set, as a blocking CI gate, using deterministic evaluators plus judges — it's how you catch a regression before it reaches a user. **Online** runs after shipping, on sampled production traffic, as an SLI with a pager, plus user signals — it's how you catch what your eval set didn't anticipate, which is most novel failures.

They share the judge, the rubric, the calibration against human labels, and ideally the trace schema, so the loop closes in both directions: production failures become new eval cases, and a CI failure can be traced back to the production traffic that motivated the case. The gap to watch for is offline-online divergence — if CI is green and production quality is falling, the eval set isn't representative, and that's a finding about the eval set, not about the model.

---

## Section F — Serving and inference

### Q31. What is the KV cache and why does it dominate serving economics?

**A.** During autoregressive generation, the model attends to all previous tokens. Naively it would recompute the key and value projections for every previous token at every step, which is quadratic work. The KV cache stores those projections so each new token only computes its own, making generation linear per step. It's the single most important optimisation in LLM serving, and it's also the binding memory constraint.

The size formula: `2 × layers × kv_heads × head_dim × seq_len × batch × bytes_per_param`. For a 70B model with grouped-query attention, that's roughly 0.31 MB per token; for an 8B model, about 0.131 MB per token. So on an A100 80GB running an 8B model in FP16 (16 GB of weights), you have about 62 GB left for KV, which at a 4k average sequence length is roughly 118 concurrent sequences. That number — not FLOPs — is usually what determines how many users a GPU can serve, which is why paged attention, GQA, and KV quantisation matter so much in practice.

### Q32. Explain the difference between prefill and decode, and why it matters for capacity.

**A.** Prefill processes the entire input prompt in parallel — it's a big matrix multiplication, compute-bound, and it determines **time to first token**. Decode generates output tokens one at a time — each step is a small amount of compute but a full read of the model weights and the KV cache, so it's **memory-bandwidth-bound**, and it determines inter-token latency.

Why it matters: they're different bottlenecks, so a GPU can be compute-saturated on prefill while bandwidth-idle on decode, or vice versa, and capacity planning has to account for both. This is the trap in the naive capacity estimate — a system that looks like it needs one replica by token throughput may need four because prefill, not decode, is saturating. It's also the reason for **continuous batching**: rather than batching a group of requests and waiting for all of them, the server admits new requests as slots free up, keeping the GPU busy through the decode phase. And it's why **chunked prefill** exists — splitting a long prefill into chunks so a huge prompt doesn't stall all the in-flight decodes.

### Q33. What is speculative decoding?

**A.** A small, fast draft model generates k candidate tokens ahead; the large target model then verifies all k in a single forward pass, accepting the ones that match its own distribution and rejecting the rest. Because verification is parallel and cheap relative to generation, you get a 2–3x speedup when the draft model's acceptance rate is high — typically above 70%.

The trade-offs: you need a draft model that's both fast and well-aligned with the target (a poorly-aligned draft wastes compute on rejected tokens and can be *slower* than no speculation), and the speedup varies enormously by task — it's high on predictable text like code and summaries, and low on high-entropy creative generation. There are also self-speculative variants that avoid needing a separate model. It's a latency optimisation, not a cost optimisation: you use more total compute to get a faster response.

### Q34. How do you decide between a hosted API and self-hosting?

**A.** The break-even is a function of volume, latency requirements, data constraints, and utilisation — and the naive comparison is wrong by a factor of two to three because it only counts GPU rental. The true cost of self-hosting includes the GPU, plus the cluster (CPU, memory, networking, storage), plus the people to run it, plus the idle capacity you're paying for. I've seen the honest multiplier land around 2.5–3x the GPU line, with GPU rental being only 30–40% of the real total.

So the shape: **below roughly 4,000–8,000 calls a day, self-hosting never wins** — the fixed cost dominates. Above that it depends on utilisation: self-hosting wins when you can keep utilisation high (60–70% as a target for latency-sensitive workloads) and when you have a specific reason APIs can't serve you. The three reasons that actually justify self-hosting are data residency or air-gapping where you cannot send data to a third party, a latency requirement an API can't meet (single-replica, sub-50 ms TTFT needs specialised hardware like Cerebras or Groq, not GPUs), and very high sustained volume where the per-token price crosses the fixed cost. For a 70B-class tier, the crossover I'd use as a planning anchor is roughly 500M tokens per month.

### Q35. How do you reduce latency in a RAG system?

**A.** Attack it stage by stage, because p95 is the sum of the stages and the biggest one isn't always the one people think. **Retrieval**: quantise the index, tune `ef_search` down, cache embeddings of frequent queries, and parallelise the dense and sparse searches. **Reranking**: this is often the second-largest cost — cap the candidate set at 50 rather than 200, use a smaller cross-encoder, and consider skipping it for queries where the retrieval top score is already very high. **Generation**: this usually dominates — reduce the output length cap (output is 3–5x the price of input and directly proportional to latency), use streaming so TTFT is what the user perceives, apply prompt caching so a 3,000-token system prompt isn't re-processed, and route simple queries to a smaller model. **Guardrails**: keep the in-path stack under 500 ms, move the expensive checks to a sample.

And the framing point: **perceived latency is TTFT, not total**. A 6-second answer that starts streaming in 600 ms feels faster than a 3-second answer that appears all at once. So streaming plus a fast prefill path is often a bigger UX win than shaving 200 ms off the total.

### Q36. What's the difference between a gateway and a load balancer for LLM traffic?

**A.** A load balancer distributes connections; an AI gateway makes **semantic routing decisions**. The gateway sits between your application and the providers and handles: routing by cost or capability ("classify this with the small model, escalate only if confidence is low"), failover across providers with per-route circuit breakers, prompt-cache-aware routing (send requests with the same cached prefix to the same provider, or the cache is wasted), rate-limit and quota management across accounts, token accounting and cost attribution, guardrails, and a normalisation layer that hides provider differences in tokenisation and tool-call format.

The reason it's a distinct component: all of these decisions are per-request and semantic, and putting them in application code means every team reimplements them slightly differently. The gateway is where "which model, which provider, which cache, which guardrails, what budget" gets decided once. And the biggest practical win is usually cost — routing by difficulty through a cascade gives 45–85% savings at about 95% of quality, and that only works if there's one place making the routing decision.

---

## Section G — Fine-tuning vs RAG vs prompting

### Q37. When do you fine-tune versus use RAG?

**A.** The distinction is **knowledge versus behaviour**. If the problem is that the model doesn't *know* something — your policies, your product catalogue, this week's pricing — that's RAG, because knowledge changes and RAG lets you update it in minutes without retraining. If the problem is that the model doesn't *behave* the way you need — it doesn't follow your output format, it's too verbose, it doesn't adopt your tone, it doesn't reliably emit valid JSON for your schema — that's fine-tuning, because you're changing the model's disposition rather than its facts.

The test I'd apply: if I gave the model the right document in the context, would it answer correctly? If yes, it's a retrieval problem. If no — it has the information and still does the wrong thing — fine-tuning is the lever. And they compose: the strongest systems often do both, with fine-tuning for format and behaviour and RAG for facts.

### Q38. Explain LoRA. Why does it work?

**A.** LoRA freezes the base weights and learns two small low-rank matrices whose product approximates the weight update: `h = Wx + (BA)x · (alpha/r)`, where `B` is `d×r`, `A` is `r×k`, and `r` is typically 8–64. So instead of updating a `d×k` matrix with millions of parameters, you train `r(d+k)` — often less than 1% of the original. The hypothesis behind it is that task adaptation has low intrinsic rank: the update needed to make a model follow your format doesn't need the full parameter space.

The practical consequences: the adapter is 10–200 MB rather than tens of gigabytes, so you can keep many adapters and swap them per request; training fits on a single GPU for a 7B model; and the base model is untouched, so you can serve it plus several adapters. The common hyperparameter convention is `alpha = 2r`, and QLoRA adds 4-bit quantisation of the base weights so you can fine-tune a larger model on smaller hardware.

### Q39. How much data do you need to fine-tune?

**A.** Far less than pretraining and far more than people hope. For a narrow behavioural task — output format, tone, classification into your taxonomy — a few hundred to a few thousand high-quality examples can work, because you're nudging an existing capability rather than teaching a new one. For a genuinely new capability or a specialised domain vocabulary, you're looking at tens of thousands at minimum, and you should ask hard whether RAG solves it instead.

The quality point dominates the quantity point. A thousand carefully-constructed, diverse, correctly-labelled examples beat ten thousand noisy ones, and the failure mode of "we generated 50,000 examples with a big model" is that you've distilled the big model's errors into a smaller one along with its competence. The practical recipe: start with 500–1,000 hand-curated examples, hold out 10–20% as a test set that you never train on, and measure whether the fine-tune beats a well-prompted base model with few-shot examples — because that's the baseline you have to beat, and it's a higher bar than people expect.

### Q40. What are the failure modes of fine-tuning?

**A.** Five I'd name. **Catastrophic forgetting** — the model loses general capability while learning your task; mitigated by LoRA (which touches fewer weights), by mixing in general instruction data, and by evaluating on a general benchmark before and after, not just your task metric. **Overfitting to the format** — the model learns to imitate your training examples' surface patterns rather than the underlying task; visible as strong eval performance and poor production performance, which is a distribution-shift signal. **Data leakage** — your test set leaked into training, so the eval is meaningless; the fix is a strict holdout established before any training data is collected. **Staleness** — the fine-tuned knowledge goes out of date and now you need to retrain rather than re-index, which is the cost you took on by choosing fine-tuning for knowledge in the first place. And **silent regression on slices you didn't test** — the fine-tune improves your target metric and quietly degrades multi-turn handling or non-English inputs, which is why the eval gate has to cover slices, not just the aggregate.

### Q41. When would you fine-tune an embedding model rather than a generator?

**A.** When retrieval quality is the bottleneck and the vocabulary is domain-specific. If your corpus uses terminology the base embedder wasn't trained on — internal product names, medical or legal jargon, chemical identifiers — then relevant chunks won't be near the query in the embedding space no matter how good your reranker is, because the reranker only reorders what the retriever found.

Fine-tuning a bi-encoder on (query, relevant-document) pairs from your own traffic is often the highest-return change available in that situation, and it typically needs only a few thousand pairs. The operational catch is that it triggers a full re-index, which for a 10-million-chunk corpus is a multi-hour, non-trivial job — so it's a scheduled change, not a hotfix, and the new index has to be built alongside and evaluated with a dual-index comparison before the alias flips. And I'd check the cheap things first: if you haven't added hybrid search and a reranker, do those before training an embedder, because they usually get most of the gain for a fraction of the effort.

---

## Section H — Context, memory and prompting

### Q42. What is context engineering and how is it different from prompt engineering?

**A.** Prompt engineering is about the instruction; context engineering is about **everything in the window** — the instruction, the retrieved evidence, the tool definitions, the conversation history, the memory, and the ordering of all of it. It's a broader discipline because in a real system the instruction is often a small fraction of the tokens: in a typical agentic setup, the system prompt and tool definitions are the majority of the input, and the user's actual question is a few dozen tokens.

The principles: **static first, volatile last** so prompt caching can hit on the stable prefix; **relevance over volume**, because irrelevant context measurably degrades answers rather than just costing money; **explicit structure** with delimiters and provenance tags so the model can tell instructions from data; and **a budget** — decide the token allocation per component and enforce it, rather than letting history grow until it crowds out retrieval.

### Q43. How do you manage conversation history in a long session?

**A.** Four techniques, usually combined. **Sliding window** — keep the last N turns verbatim; simple, and it loses early context. **Summarisation** — periodically compress older turns into a summary; keeps the gist, loses detail, and the summarisation itself can introduce errors. **Retrieval over history** — embed past turns and retrieve the relevant ones for the current query; keeps detail but needs an index and adds latency. **Structured state extraction** — pull out the facts that matter (the user's plan, the order ID, the constraints they stated) into a structured record and inject that, rather than the raw turns.

The design I'd defend: structured state for the facts that must never be lost, a sliding window for recent conversational flow, and summarisation for the middle. The key insight is that most of what a long conversation contains isn't needed for the next turn, so the goal isn't to preserve everything — it's to preserve the small set of things that will change the answer, and to be explicit about which those are.

### Q44. What is the difference between short-term and long-term memory in an agent?

**A.** Short-term memory is the context window plus any working state within a session — the current task, the tool results, the plan. It's bounded by the context limit and it disappears when the session ends. Long-term memory persists across sessions: user preferences, facts learned about the user, and successful strategies. It's usually implemented as a store — often a vector store plus a structured records store — with a write path (when does something get remembered?) and a read path (when is it recalled?).

The hard parts are both of those paths, not the storage. On the write side, an agent that remembers everything drowns in noise, so you need extraction with a quality bar and a deduplication step; and memory poisoning is a real attack surface — if an injected instruction can write to memory, the hijack persists across sessions, which is why memory writes need validation. On the read side, you need relevance-based recall with a budget, or the recalled memories crowd out the actual task. And because memory is written by one session and read by another, an extraction error compounds — which is why memory gets per-operation evals (extraction, update, and QA separately) rather than one aggregate accuracy number.

---

## Section I — Pushback round

### Q45. "Why not just use a bigger context window and skip retrieval entirely?"

**A.** For some corpora, you should — and I'd say so rather than defending RAG reflexively. If the corpus is a 200-page contract that's queried repeatedly, loading it once with prompt caching is simpler, avoids chunking decisions entirely, and removes the retrieval-miss failure mode. That's a genuine win and long context has eaten a real share of RAG use cases.

But three things don't go away. **Scale**: a 1M-token window doesn't hold 50 million documents, so anything corpus-wide still needs retrieval. **Cost**: context is priced per token on every request, so a full-corpus load costs the same on request ten thousand as on request one, while retrieval is a small fixed cost — at 800,000 requests a day that difference is hundreds of thousands of dollars a month. **And attention quality**: the lost-in-the-middle effect means models attend most reliably to the start and end of a long context, so a relevant document buried in the middle is measurably less likely to be used than the same document retrieved and placed deliberately. So my answer is that the choice is corpus-size- and query-frequency-dependent, and the honest version is "retrieval for corpora too large to fit, or queried often enough that per-token cost matters; long context for corpora that fit and are read once."

### Q46. "Your RAG system answers 'I don't know' a lot. Isn't that a failure?"

**A.** It's a signal, and the question is whether the rate is calibrated. Some abstention is correct behaviour — if the corpus genuinely doesn't contain the answer, answering anyway is a hallucination, and a system that never abstains is over-confident by construction. The published target I'd use is 8–15% abstention when the corpus has real coverage gaps.

But a high abstention rate can also mean one of three real bugs: the retrieval threshold is set too aggressively, so relevant documents are being filtered out before generation; recall is genuinely low, so the documents aren't being found at all; or the corpus is missing content users expect. Those have different fixes and the diagnostic is to sample the abstentions and check whether a relevant document existed — if it did, it's the threshold or recall, and if it didn't, the corpus has a gap that's a content problem, not a model problem. The one thing I'd never do is lower the threshold to make the number look better, because that converts visible abstentions into invisible wrong answers, which is strictly worse.

### Q47. "Your chunking is 512 tokens. Why not 256, or 1,024?"

**A.** Because I tuned it, and I'd want to show the tuning rather than defend a number. The trade-off is real in both directions: smaller chunks give more precise retrieval but fragment context, so the generator gets a sentence without the paragraph that makes it meaningful; larger chunks preserve context but dilute the embedding across multiple topics, so it matches none of them strongly. Overlap of 10–15% exists because facts that straddle a boundary are otherwise lost by both chunks.

The way I'd actually choose: build a retrieval eval set of 100–200 query-document pairs, sweep 256, 512, 768, and 1,024 with the real corpus, and compare recall@10 and nDCG with everything else held constant. The answer is corpus-specific — dense technical documentation usually wants smaller chunks with heavy overlap, narrative documents want larger ones — and the default is wrong about as often as it's right. The other thing I'd say is that structure-aware chunking (splitting on headings and sections rather than a fixed token count) usually matters more than the exact size, and it's cheaper to get right.

### Q48. "Isn't fine-tuning strictly better than RAG since the knowledge is baked in?"

**A.** No, and the reason is operational rather than theoretical. Baked-in knowledge is **stale knowledge**. If a policy changes, RAG is a re-index measured in minutes; fine-tuning is a retrain-and-redeploy measured in days or weeks, and it costs money every time. So fine-tuning for knowledge trades a cheap, frequent operation for an expensive, rare one — which is exactly backwards for any corpus that changes.

There's a second problem: fine-tuning for facts is unreliable. The model learns the *distribution* of your training data, not a retrievable record, so it can confabulate plausible facts it never saw, and it has no citation mechanism. RAG gives you attribution for free because you know which chunk the answer came from, and in enterprise settings auditability is frequently the actual requirement. So the division of labour stands: RAG for knowledge, fine-tuning for behaviour, and fine-tuning for knowledge only when the knowledge is genuinely static and the model needs it as a skill rather than as a reference.

### Q49. "Your evaluation is an LLM judging an LLM. Why should I trust that number?"

**A.** You shouldn't trust it raw — you should trust it calibrated. The number only means something once I've shown it agrees with human judgement on a sample, which is what Cohen's kappa measures. Target is 0.6 or better, meaning moderate-to-substantial agreement, and I alert below 0.4. If kappa is 0.7, the judge is a cheap proxy for human labels and I can run it on every trace; if it's 0.3, the judge is measuring something else and the score is decoration.

Beyond calibration I'd add three things. Swap-test the judge for position bias — run each comparison in both orders and keep only consistent verdicts, because position bias alone can manufacture a two-to-five-point difference that doesn't exist. Use a different model family as judge than the model under test, to avoid self-preference. And keep humans in the loop on disagreements and on the high-stakes path, because that's where ground truth actually comes from. The honest summary is that a judge is a measurement instrument, and like any instrument it needs calibration before you believe its readings — an uncalibrated judge is worse than no judge, because it produces confident numbers that are wrong.

### Q50. "This all seems like a lot for a chatbot. Why not ship and iterate?"

**A.** Because the failure mode is invisible, and that's the whole difference from shipping ordinary software. If you ship a web app with a bug, you get errors, alerts, and stack traces. If you ship a RAG system whose retrieval recall quietly dropped after a chunker change, you get perfectly-formed confident wrong answers with a 200 status code, and nothing in a standard monitoring stack will tell you — you'll find out from a user complaint weeks later, by which point the system has been confidently wrong thousands of times.

So the minimum I'd ship with is small and cheap: `release_id` in every trace so a regression is attributable, cost and latency tracking, a golden set of a few hundred hand-labelled examples wired into CI as a blocking gate, a sampled judge score with an alert, and a rollback that takes ninety seconds. That's a few days of work, not a platform. What scales up is everything beyond it — trajectory grading, per-operation memory evals, the frontier-judge calibration layer — and those should match the blast radius: a read-only internal tool over public docs needs far less than a system with side effects over customer data. The one thing I wouldn't do is skip the sampled quality score, because it's the only thing standing between me and shipping silently worse answers for a month.

---

## Cross-links

- [Production RAG System](../HLD/05-ai-systems/01-production-rag-system.md) — the full architecture behind Section A
- [Advanced RAG Patterns](../HLD/05-ai-systems/02-advanced-rag-patterns.md) — Section D in depth
- [RAG Evaluation and Observability](../HLD/05-ai-systems/03-rag-evaluation-and-observability.md) — Section E in depth
- [LLM Serving, Inference and Gateway](../HLD/05-ai-systems/05-llm-serving-inference-and-gateway.md) — Section F in depth
- [Fine-Tuning vs RAG vs Prompting](../HLD/05-ai-systems/09-fine-tuning-vs-rag-vs-prompting.md) — Section G in depth
- [Memory and Context Engineering](../HLD/05-ai-systems/08-memory-and-context-engineering.md) — Section H in depth
- [Agentic AI Questions](05-agentic-ai-questions.md) — the agent-specific bank
- [LLMOps and Operationalisation Questions](06-llmops-and-operationalisation-questions.md)
- [Behavioral and Project Questions](07-behavioral-and-project-questions.md)
- [RAG and LLM Cheatsheet](../Cheat-Sheets/04-rag-and-llm-cheatsheet.md)
- [Foundations and Capacity Questions](01-foundations-and-capacity-questions.md)
