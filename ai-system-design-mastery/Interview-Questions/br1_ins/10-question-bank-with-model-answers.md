# 10 — QUESTION BANK WITH MODEL ANSWERS

> **TL;DR.** ~130 questions with spoken-style model answers, grouped by theme. Every answer is written the way you should *say* it — headline first, then Why, then How, then the trade-off. Use this file for the **F4 delivery drill** (one question, under 60 seconds, out loud). The questions marked ⭐ were asked in the previous Brit round; ⚑ are the ones most likely to be graded at Lead level. **§L** covers LLM internals (tokenisation, sampling, attention) — the depth questions interviewers use to find the floor of your knowledge.

**How to use:** cover the answer, say yours out loud, record it, then compare. Fix one thing per question, not five.

---

## A. LLM fundamentals, tokens & context

**1. What is a token, and why does it matter commercially?**
Headline: *"A sub-word unit — roughly 1.3 words per token in English — and it's the unit of every cost, latency and capacity calculation."* Then: cost is per token, context limits are in tokens, and word-based estimates are systematically wrong.

**2. Why can't you estimate cost from word counts?** Because tokenisation varies by model and by content — code, numbers and non-English text tokenise far worse. Always measure tokens, never words.

**3. Why does tokenisation affect reasoning?** ⚑ *"'12345' is one token; '1 2 3 4 5' is five. The model sees different units, so digit-level arithmetic behaves differently. It's a reminder that the model operates on tokens, not characters or numbers."*

**4. What happens when a prompt exceeds the context window?** You truncate — and truncation is silent data loss. Mitigations: hierarchical summarisation, parent-child retrieval so you send only the relevant parent, and a hard token budget per component with an alert when you approach it.

**5. Why is conversation history expensive?** It grows **O(N²)** in tokens: turn N re-sends all previous turns. Fixes: sliding window plus a running summary; summarise rather than replay; cache the stable prefix at the provider.

**6. Embedding model vs generative model?** An embedding model maps text to a fixed vector for similarity; a generative model predicts the next token. One is for retrieval and clustering, the other for producing text. They are not interchangeable and you should not use a generative model for search.

**7. Encoder-only vs decoder-only — why decoder-only for chat?** Encoder-only (BERT-family) is bidirectional and good for classification/embedding, but it cannot generate autoregressively. Decoder-only generates left-to-right, which is exactly the chat use case.

**8. Why does a bigger context window not solve RAG?** ⑀ Cost scales with tokens, attention is superlinear, and the model attends poorly to the middle of a long context — "lost in the middle". More context is often worse than better-selected context.

---

## B. Inference & serving

**9. Prefill vs decode — why does it matter?** ⚑ Prefill processes the whole prompt in one parallel pass (fast, cheap, ~$0.50/M) and builds the KV cache. Decode is sequential — token *n* needs tokens 1…*n*−1 — and is **memory-bandwidth bound** (~$1.50/M, ~3× the price). So **output length is the expensive dimension**.

**10. Why can't decode be parallelised?** Each token depends on all previous tokens. There's no way to compute token 20 without token 19.

**11. What is the KV cache and what runs out first?** Cached key/value attention tensors for every input token: `2 × layers × heads × head_dim × seq_len × bytes`. A 7B fp16 ≈ **0.5 MB per 1,000 tokens**; a 200K context ≈ **100 GB**. **VRAM, not FLOPS, is what runs out.**

**12. TTFT vs ITL/TPOT — which matters more?** ⭐ *"TTFT is what the user feels — it's retrieval + rerank + prefill, 0.5–1.2 s for RAG. ITL/TPOT is per-token time and sets total completion. Optimise TTFT for perceived speed and stream; optimise ITL for completion time."*

**13. What is speculative decoding?** A small draft model proposes N tokens; the target model verifies them in one parallel pass. **2–3× decode speedup with an identical output distribution** — a latency win, not a quality change. Best when the draft model is much cheaper and acceptance rates are high.

**14. Continuous vs static batching?** Static waits for the whole batch, so every request is as slow as the longest. Continuous (vLLM, TGI) inserts new requests at each step — the single biggest throughput win.

**15. What is PagedAttention?** Non-contiguous KV cache pages, which removes the memory fragmentation that wastes 60–80% of VRAM in naive serving.

**16. Quantisation trade-offs?** 7B: fp16 14 GB → int8 7 GB → int4 3.5 GB. fp16 lowest loss; int8 usually safe; int4 GPTQ/AWQ visibly lossy on complex reasoning — **A/B it on the golden set, don't assume**.

**17. Why does GPU utilisation matter for cost?** A GPU at 30% costs the same as at 90% but serves 3× fewer requests. Continuous batching, PagedAttention and speculative decoding raise *effective* utilisation — that's how a real p95 cut halved a GPU bill without new hardware.

**18. What is prompt caching and its limit?** Reusing the computed prefix at the provider — cache reads at ~10% of input price. The limit: **one changed token invalidates the cache**, so it only helps with a genuinely stable prefix (system prompt, shared instructions).

**19. Why can't you just use the biggest model?** Cost 10–30×, latency worse, and on many tasks it isn't more accurate. Tier it: small for routing/extraction/classification, mid for reasoning, frontier for synthesis only.

**20. Which LLM serving stack would you pick and why?** vLLM for throughput and continuous batching as the default; TensorRT-LLM when you're on NVIDIA and need maximum single-node performance; SGLang when the workload is prefix-heavy (shared system prompts, multi-turn); managed endpoints when you lack GPU ops capability or the volume doesn't justify it. Trade-off: self-hosting wins on residency, control and steady high volume; managed wins below that.

---

## C. RAG

**21. Explain RAG architecture in detail — with code.** ⭐ See [`06`](06-coding-exploration-playbook.md) §3 for the full spoken answer. Headline: *"Ingestion — clean, chunk, attach metadata, embed, index, plus a parallel BM25 index. Query — understand/expand, hybrid retrieve, metadata pre-filter, rerank top-50→5, assemble most-relevant-first, generate with citations, validate. Two paths, one index."*

**22. Why hybrid search and not just embeddings?** Dense embeddings miss exact tokens — error codes, product IDs, version numbers, proper nouns. BM25 catches those but misses synonyms and intent. **Hybrid is not optional in production.** Fuse with RRF: `Σ 1/(k + rank)`, k=60.

**23. What is a reranker and why does it matter?** A cross-encoder scores the (query, chunk) pair *jointly*, unlike a bi-encoder that embeds them separately. Run it over the top-50 to produce the top-5. **Usually the single highest-leverage quality improvement** — at the cost of 100–200 ms and some compute.

**24. Chunking — which strategy and what numbers?** See [`05`](05-system-design-round-playbook.md) §5.1. Fixed 300–500 tokens for general, 500–1,000 for legal, 100–300 for FAQ, overlap 10–20%; structure-aware when the document has structure; parent-child (index 100–200, send 500–1,000) for long documents; semantic (split below ~0.4 adjacent-similarity) for topic-shifting prose; propositions for high-stakes legal/financial facts.

**25. What is context dilution?** Mechanical splitting puts multiple topics in one chunk, so its embedding is an average of several meanings and matches none well.

**26. Parent-child retrieval — why?** Small chunks embed precisely but starve the model of context; large chunks give context but noisy embeddings. Index the small child, retrieve it, send its parent.

**27. The correct chunk sits at rank #12 but you only pass top-5 — and raising K blows the context window. What do you do?** ⑀ *"Retrieve wide, rank hard. Fetch top-50 cheaply, rerank to the best 5–8, and only put the winners in the context window. The context window is for the winners."* If ranking can't fix it, then revisit chunking — a document split across a boundary is a chunking problem, not a ranking one.

**28. How do you keep the index fresh?** Hash chunk content and re-embed only what changed; subscribe to webhooks/CDC not polling; upsert by deterministic chunk ID and delete vectors for removed sections; keep the write path off the read path via a queue and an atomic index swap; track index lag per document with an alert. *"Freshness is an incremental-sync problem, not a re-index schedule."*

**29. Semantic caching — how, and what's the risk?** Embed the query, match above ~0.95 cosine. Cache only stable content (FAQs, policy wordings — never live prices, balances or availability). **Key on context, not just text** (tenant, date, locale, version). Invalidate on the write path, driven by CDC/webhooks rather than a timer. And **verify before serving** — cheap re-check that cited facts still match the source; mismatch = cache miss. *"A high hit rate with rising complaints means the threshold is too loose."*

**30. A RAG system answers "what is my account balance?" by retrieving statement documents. The pipeline is healthy. Why is the design broken?** ⑀ Because it's a **live value**, not a document. The index is a cache of documents; state belongs in the system of record. A fluent, confident, stale answer is the worst possible failure.

**31. The RAG worked for three months, then started retrieving irrelevant results with no code change. What happened?** Corpus drift (documents changed), query drift (users ask new things), embedding-model drift, or index staleness. Diagnose by re-running the golden set with frozen documents — if it passes on the old corpus and fails on the current one, it's corpus drift, not a model regression.

**32. How do you debug a RAG answer that's wrong?** The isolation tree: is the right chunk in the top-K (retrieval)? Present but ranked low (reranking)? Present and ranked first but the answer still wrong (generation)? **Never blame the model first.**

**33. Handling conflicting documents?** Prefer the most recent timestamp or the higher authority level via metadata rules; if there's no clear winner, instruct the model to state both positions and the disagreement. *"Never let the model silently pick one version — in a regulated domain that's liability."*

**34. Multi-lingual RAG?** A multilingual embedding model (e.g. BGE-M3), language metadata for filtering, and either translate-then-retrieve or retrieve-then-generate-in-language. Test retrieval quality per language — it is usually much worse in the minority language and nobody checks.

**35. What is CRAG?** A lightweight evaluator grades each retrieved chunk Correct / Ambiguous / Incorrect *before* generation; Incorrect chunks are discarded, Ambiguous ones treated as unreliable, and weak results trigger an external fallback. Generation never sees ungraded raw top-k.

---

## D. Agents & orchestration

**36. Agent vs chain — what's the difference?** A chain is a fixed sequence; an agent decides its own steps at run time. Agents handle unanticipated inputs but fail unpredictably. **A workflow handles the 95% of ordinary cases reproducibly; the agent earns its place only on the genuinely variable remainder.**

**37. When do you reach for a real agent?** ⚑ When the task needs **actions** (API calls, writes), when the number of steps depends on intermediate results, and when the branches change frequently. Otherwise a router, a fixed workflow, or RAG alone is cheaper and debuggable.

**38. How do you stop an agent looping forever?** ⭐-adjacent, asked at Swiggy. Fingerprint tool calls (name + normalised args) and flag repeats; track **progress, not steps** — no new information or state change in three turns means stuck; intervene with a *different* strategy, not just a cut-off; and hold a hard budget (max steps, tokens, cost) as a circuit breaker outside the model.

**39. Your agent calls the same search tool with the same query repeatedly. Detect and break it.** Detect via the fingerprint; break by feeding back "this query already returned nothing — try a different formulation or a narrower scope", forcing a different tool, or escalating. A budget alone stops the bill, not the loop.

**40. Two agents on the same model and toolset perform differently. Why?** The gap is everything *around* the prompt: tool naming/description/schema quality; what history and memory enters each step and how it's summarised; control flow (planning step, retries, verification, loop limits); how tool errors are reported back and whether the agent re-plans; and whether the team iterates on real failures. *"Agent quality is system design, not prompt wording."*

**41. How do agents share state?** A shared store (Redis or a queue) with a **single writer per key** to avoid lost updates, plus an explicit typed handoff object rather than free-form text passing between agents.

**42. What is the lost-update problem in shared memory?** Two agents read the same value, both write, the second silently overwrites the first. Fix: optimistic concurrency (version/etag) or locking, and one writer per key.

**43. How do you design tools well?** One clear responsibility per tool with no overlap; enums for constrained parameters (`action: "refund" | "exchange" | "cancel"`); required arguments marked; 2–3 few-shot call examples; **server-side validation of every argument**; structured error messages, because the model reads them and re-plans from them.

**44. Tool-calling fails — how do you fix it?** In order: tool boundary clarity → schema strictness → few-shot examples → server-side validation → human-in-the-loop for high-risk tools. *"Usually a design failure, not a model failure."*

**45. Memory architecture for an agent?** Four layers — short-term (context window), long-term (external store), shared (cross-agent), episodic (append-only log). Every record carries a timestamp and confidence; validate before use; archive stale entries. **Stale memory is worse than none.**

**46. LangGraph vs a custom state machine?** Framework when the topology is uncertain and will change; state machine when it's known and hot. *"It's a judgement about how often the graph changes, not about which is more modern."*

**47. Multi-agent — when?** Only when sub-tasks need different tools, different prompts or different models. Otherwise it's one agent with more tools — cheaper and easier to trace. More agents means more failure nodes.

**48. A 20-step workflow runs at 97% per step. Is that good?** ⚑ `0.97²⁰ ≈ 54%`. Pushing to 98% only reaches 67%; cutting to 7 steps at 97% reaches **81%**. Step count dominates per-step accuracy.

**49. Where do you put a verifier?** Per step, not at the end. A per-step verifier catching 80% of errors lifts a 12-step chain from ~54% to ~89% for ~19% more cost. An end-of-task check loses because *"a failure at step three poisons every step after it."*

**50. Routing economics — is a router worth it?** 70% easy traffic; cheap model 95% at $0.001, strong model 96% at $0.012; a 5%-misroute router lands at ~$0.0046/request — **62% cheaper for ~1.2 accuracy points**. The ongoing obligation: a confidence-based fallback escalation path, and the router itself needs an eval.

**51. When does reflection earn its cost?** On checkable tasks, gains run **+8, +4.8, +2.9, +1.7** points per pass — so one or two passes usually, then diminishing. On tasks with no ground truth to check against, reflection mostly adds cost.

**52. Self-consistency — does voting always help?** Only if errors are independent. When the model misreads one recurring phrasing the same way, the majority **confirms it fifteen times**. Voting fixes sampling noise, not systematic bias.

---

## E. Evaluation & metrics

**53. How did you evaluate your agents?** ⭐⭐ **The most important answer in this pack.** Full script: [`07`](07-agent-evaluation-and-guardrails.md) §9. Headline: *"Three layers — component, trajectory, online — plus a frozen golden set gating CI."*

**54. RAG eval metrics?** Retrieval: Recall@K, MRR, context precision. Generation: faithfulness, answer relevance, citation correctness. Build a test set of 50–120 real questions with known answers and known source chunks.

**55. Precision or recall for an auto-approval system?** ⚑ *"Both, at different thresholds. Precision at the auto-adjudication threshold, because writing bad risk is the expensive, hard-to-unwind error. Recall in the human triage queue, so nothing valid is silently dropped. The metric follows the cost of error at each decision point."*

**56. How do you build an LLM judge you can trust?** Explicit rubric; calibrate against a few hundred human labels; measure the confusion matrix; use it for *relative* comparisons only, because a stable bias cancels in a comparison but not in an absolute score; never judge what a deterministic test can check.

**57. How accurate is your LLM judge?** ⚑ *"With a 10% false-positive and 8% false-negative rate, a true 85% is reported as about 79.7%. The mapping is invertible so I can correct it, but I'd never quote the raw absolute number."*

**58. Your prompt change improved scores by 2 points on 50 examples. Trust it?** No. On a 90% baseline, 50 examples give roughly an **8-point interval** — undetectable at 2 points. You'd need ~**865 examples** for that effect.

**59. Your release gate uses 114 examples and catches 5-point regressions. What's the catch?** A 1-point gate needs ~**2,828 examples** — 25× more. Size the gate to the regression you actually care about.

**60. p=0.04 with +4 points on 200 samples, vs p=0.003 with +1.5 points on 5,000. Which do you ship?** Significance is not effect size. The second is stronger *evidence* of a smaller, more certain gain. Ship what matters, and report both.

**61. What SLOs would you set for an LLM service?** ⑀ Beyond uptime and p95: **grounded-answer rate**, tool-success rate, escalation latency, regression-pass rate, cost per resolved task. *"Availability SLOs tell you whether the service answered, not whether the answers were any good."*

**62. How do you detect a silent quality regression?** Canary a slice, compare output distributions old vs new, sample production continuously with an automated judge plus a rotating human sample, and track behavioural metrics (length, refusal rate, tone, format adherence).

**63. Your provider silently updates the model and users complain — how do you catch it first?** Pin versions; never auto-float to latest; run the golden set on every provider release *before* it becomes default; treat a version bump like a dependency upgrade.

**64. What's the eval loop for a new use case?** 50 labelled seed cases from experts → baseline → CI gate → sample production → cluster failures → add them to the gate → repeat. The gate grows from real failures, not from imagination.

---

## F. Reliability

**65. Why is a 90 ms average latency misleading?** The mean hides the tail. Users experience the slowest call in their request, and fan-out compounds it: at 1% slow per call and 100 calls, 63% of page loads feel slow. **P99, not P50 or the mean.**

**66. Fan-out math?** `P(at least one slow) = 1 − (1−p)^N`. n=100, p=1% → **63.4%**; drop p to 0.1% → 9.5% (7× better). Reduce fan-out, partially load, or hedge.

**67. What is request hedging?** Send the same request to two replicas and take whichever answers first — only when there are no side effects, the call is expensive to miss, and the backend has capacity. *"Never hedge a write."*

**68. Timeout architecture?** Downstream timeouts must be **shorter** than upstream, and cancellation must propagate inward. Otherwise you get zombie work nobody is waiting for. Example budget: browser 10 s > gateway 8 s > service 6 s > provider 5 s.

**69. Retry amplification?** Three layers each retrying 3× is up to **27 requests per click** (3³). Fix: **one layer retries**, exponential backoff with **full jitter** (`random(0, min(cap, base·2^attempt))`), and a **retry budget** — a token bucket capping retries at ~10% of request rate.

**70. Why full jitter and not fixed backoff?** Fixed intervals synchronise retries into a thundering herd. Full jitter spreads them across the window.

**71. Idempotency for a refund?** Key on the **intent**, not the HTTP request — a UUID generated at the user action, stored in Redis with in-flight status; a retry with the same key waits or returns the cached result. Reads retry freely; **writes always gate**.

**72. What if Redis (the idempotency store) is down?** Fail closed for writes — do not execute without a durable key store — or fall back to a database unique constraint. Reads can proceed. *"The store is a dependency; I'd run it replicated with automatic failover, and I'd rather reject a write than double-charge."*

**73. Idempotency vs exactly-once delivery?** Idempotency makes a *duplicate* safe; exactly-once is a delivery guarantee you rarely get. In practice: at-least-once delivery plus idempotent handling.

**74. Streaming at the provider — is that enough?** No. Disable buffering in the service, the gateway, the reverse proxy (`proxy_buffering off`), and **disable gzip/brotli on streaming endpoints** — compression waits to accumulate bytes and re-introduces the latency you just removed.

**75. What happens if you gzip a streaming endpoint?** The compressor buffers to build a block, so tokens arrive in bursts or all at once — the user sees nothing until the buffer flushes. Streaming is nullified.

**76. How does resume work on a dropped stream?** SSE's `Last-Event-ID`: the server keeps a short event ledger and replays only events after that ID. It also prevents duplicate tool calls on reconnect.

**77. Cancellation propagation?** Detect the client disconnect, cancel the provider stream. Otherwise the GPU keeps generating tokens for a dead connection — wasting compute and money.

**78. SSE vs WebSockets for AI streaming?** SSE: one-way, standard HTTP, native reconnect, no sticky sessions. WebSockets: bidirectional, persistent TCP, needs sticky sessions or a pub/sub bus. **For token streaming, SSE wins** — one-way is all you need.

**79. Stateless servers — how?** Move session state to Redis (with replication/failover), preferences to a DB, uploads to object storage. Then any server can serve any request. Cost: a network hop and a new critical dependency.

**80. How would you design a system to detect tail-latency spikes?** Histogram-based percentiles (`histogram_quantile(0.99, rate(latency_bucket[5m]))`), alert on 2+ consecutive minutes over a 5-minute window, and decompose the latency by stage so you know *which* stage spiked.

---

## G. Cost & capacity

**81. Design to a $0.02 per-conversation budget.** Model tiering (75% small / 20% mid / 5% frontier), semantic + prompt caching, output-length control, and RAG that sends the top-5 rather than everything. All-frontier costs 10–30× that budget.

**82. Why do costs grow super-linearly with conversation length?** History is re-sent each turn → O(N²) tokens. Summarise and window.

**83. How do you attribute cost in a multi-tenant system?** Tag every request with tenant, feature and model; track tokens in and out separately; a gateway that logs per-tenant token spend and rate-limits per tenant on tokens, not requests.

**84. Which lever moves the bill most?** ⚑ Usually **output tokens**, because output is ~3× the input price and output length is uncontrolled by default. After that: model tiering, then caching, then prompt length.

**85. Capacity sizing for a vector DB?** `chunks × dims × 4 bytes × (1.5–2× index overhead)`. Watch the stride: a 1,200-token page chunked at 600 with 80 overlap produces **3** chunks, not 2 — a 50% sizing error people make.

**86. How would you handle a 100× traffic spike?** Queue + backpressure (never a synchronous API), autoscale on queue depth rather than CPU, a degraded mode (smaller model or cached/template answers), and load shedding with a clear message. *"Protect the core path; shed the optional."*

---

## H. Security, governance & regulated AI

**87. Prompt injection — how do you defend?** Defence in depth: treat retrieved content as **data, never instructions**; delimit instructions from retrieved text in the prompt; scan retrieved content for injection patterns; **never let retrieved content trigger a tool call** — all tool calls are validated server-side; and require human approval for irreversible actions. *"A better system prompt is not a security control."*

**88. A poisoned KB article instructs the agent to skip verification on refunds. Why is 'improve the system prompt' structurally insufficient?** Because the injected text arrived through a legitimate channel and the model can't reliably distinguish it from policy. The fix must be structural: the refund authorisation lives in code and is evaluated against the tool call, not derived from retrieved text.

**89. Multi-tenancy — where do you enforce isolation?** **At retrieval, not at the output.** Filter by the caller's real entitlements *before* the vector search. Output-scanning isn't equivalent: *"the forbidden chunks already entered the context and shaped the answer"* — the model can paraphrase private data no redactor reliably catches.

**90. Three isolation layers?** Data (separate namespaces/partitions/filters), cache (per-tenant prefixes; a cache hit for A must never serve B), compute (per-tenant rate limits on RPS **and** tokens so one noisy tenant can't monopolise the GPU).

**91. How do you make a past LLM answer reproducible for an auditor?** ⑀ *"You can't reproduce it — sampling means a re-run differs. You can only prove what produced it."* Log the full trace: exact model ID and parameters, the final prompt *text*, the system-prompt version, the retrieved chunks *themselves*, index and embedding-model versions, then replay from the record.

**92. Data poisoning defence?** Control provenance (source, author, hash per example; train only on signed, versioned datasets); scan before training (dedupe, outlier and embedding-cluster checks, rare tokens co-occurring with one label); human-review risky slices; test for backdoors after training by probing candidate triggers against a clean baseline; keep dataset snapshots so you can retrain without a suspect source. *"Treat training data like code."*

**93. A regulation forbids sending financial data to any third-party provider. Architect it.** Open-weight model (Llama/Mistral) quantised on owned hardware, served via vLLM behind an internal-only API; a local vector DB over the bank's own knowledge base; nothing leaves the network; encryption in transit and at rest; RBAC so the model surfaces only what the requester is entitled to; full audit logging; and a local golden set for evaluation since no external judge is reachable. **Monitoring and patching must be designed on day one, because there's no external fallback.**

**94. Design agent action safety so destructive actions need approval.** Classify tools by **blast radius** — read-only, reversible write, irreversible/external. Gate the irreversible class in code: return a pending request, never execute. Human approves with the exact recipients and content visible. Add hard limits (max recipients per call, rate limits, daily spend cap, second approver above a threshold), environment separation (dev agents on a sandbox provider), idempotency keys, and a kill switch. *"Trust the agent with the work, not with the consequences."*

**95. Why is 99.9% accuracy not enough for an agent?** Because 0.1% of a million actions is a thousand irreversible mistakes. **The control must not depend on the model being right** — it depends on the action being gated.

**96. Model risk in insurance — what does it mean practically?** A model that informs underwriting or claims is a controlled artefact: documented purpose and limitations, validation before use, monitoring after, a named accountable owner, change control with evidence, and human oversight for the decision. A model card plus a decision log plus the trace is the evidence pack.

**97. What is your four-controls answer if asked to cut two from a refund agent?** Keep the controls that sit **outside the model**: server-side authorisation, and the human gate on irreversible actions. Cutting prompt-level guards saves nothing real, because they were never the control.

---

## I. Classical ML, data & coding

**98. RAG vs fine-tuning?** ⚑ RAG for **knowledge** that changes and must be citable; fine-tuning for **behaviour** — persona, tone, rigid format. *"Fine-tuning cannot reliably inject new facts; RAG cannot reliably enforce a rigid format."*

**99. Fine-tuning vs training from scratch?** Fine-tuning adapts an existing model (LoRA/PEFT for cost); training from scratch is for a frontier lab with a novel architecture. For 99% of commercial problems, neither — it's prompting, RAG and orchestration.

**100. Supervised vs unsupervised here?** Supervised when you have labels and a decision to make (classification, extraction, scoring); unsupervised for structure discovery (segmentation, outlier detection, clustering) — often as a *precursor* to a supervised model or to find the long tail your labels miss.

**101. Feature engineering still matters?** Yes — the best signal is usually a well-built feature, not a bigger model. And in a regulated setting, a small number of explainable features beats a black box you can't defend.

**102. How do you handle class imbalance?** Choose the metric first (precision/recall or PR-AUC, never accuracy), then resampling/class weights, threshold tuning against the cost of error, and per-class reporting. In claims/damage, rare classes are the important ones.

**103. Model evaluation beyond accuracy?** Calibration (are the probabilities meaningful?), stability across time and segments, per-segment performance for fairness, and drift monitoring. *"A model that's accurate on average but systematically worse for one segment is a regulatory problem, not just a modelling one."*

**104. Git — linear history with multiple people?** ⭐ Full answer: [`06`](06-coding-exploration-playbook.md) §4. One line: *"Protected main · short branches · pull --rebase · squash-merge · --force-with-lease on my own branch only · never a merge commit on main."*

**105. Rebase or merge?** Rebase my own un-pushed branch to stay current and clean; merge only when a branch is genuinely shared and others depend on its SHAs; squash to land on main.

**106. Run-length encoding / find a substring in a million strings?** RLE: single pass, O(n) time, O(1) extra space beyond output. Substring search across 1M strings: O(M×N) naive; use a set/trie for exact matches, or an inverted index / suffix automaton if you need many queries. *"State the complexity before you code."*

**107. How would you test an LLM app?** Unit (chunking, validators), contract (mocked model responses including malformed ones), eval regression (golden set in CI, gating merges), trajectory eval for agents, canary/shadow in production, and adversarial tests (injection strings, empty retrieval, 10× documents).

---

## J. Business, stakeholders & leadership

(The full set is in [`12`](12-people-leadership-and-stakeholder-round.md) §8. Highest-yield few:)

**108. How do you say no to a stakeholder?** *"Give them a choice of when, not whether: 'A now, or B now and A next quarter — which is worth more?' Saying no makes you a gatekeeper; making them choose makes them a partner. If it's truly non-negotiable, I say clearly what it displaces."*

**109. What does the model contribute to a successful AI service?** ⑀ *"Maybe 10%. The rest is who acts on the output, which system of record it writes to, what happens when it's wrong, and how the human's day changes."*

**110. How do you measure whether an AI project succeeded?** Adoption first (are people using it after the novelty fades?), then the business number (time saved, cost avoided, revenue), then the quality metric, then the cost per resolved task. *"A high eval score with low adoption is a failed project."*

**111. How do you build trust in a model the business doesn't believe?** Shadow mode first — run it alongside the human, show agreement, let them see where it disagrees. Then a human gate they control. Trust comes from being able to overrule it cheaply, not from a benchmark.

---

## K. Azure & low-code (see [`11`](11-insurance-lloyds-domain-primer.md) for the full stack)

**112. Which Azure AI components have you used?** ⭐ Answer with **components and their purpose**, not a list: Azure OpenAI / AI Foundry for model access and deployment; **Azure AI Search** for hybrid retrieval with the semantic ranker; **Document Intelligence** for layout-aware PDF extraction; **Azure ML** for classical model lifecycle; **Content Safety** for guardrails; plus VNet/private endpoints and managed identity for the security boundary.

**113. Azure AI Search vs a purpose-built vector DB?** AI Search wins when you're already Azure-native and need hybrid (BM25 + vector) plus a semantic ranker and metadata filters out of the box, with enterprise identity integration. A dedicated vector DB wins at very large scale, for specialised index tuning, or if you need features AI Search doesn't expose. *"Trade-off: integration and governance versus index control and ceiling."*

**114. When would you choose low-code over pro-code?** ⚑ *"When the capability isn't a differentiator, the process is simple, and the business needs it fast — Copilot Studio or Power Platform gets you there in weeks with a supported maintenance path. I move to pro-code when I need control over retrieval, evaluation, cost, or integration that the low-code tool doesn't expose. And I'd document that decision either way, because 'why we didn't build it' is part of the architecture."*

**115. Where does low-code bite you?** Ceilings on retrieval quality and evaluation depth, cost visibility, version control and testing, and vendor lock-in for anything you'd want to move later.

**116. How do you keep data inside the tenant?** Private endpoints and VNet integration, no public network access on the AI resource, managed identity rather than keys, customer-managed keys where required, and no egress of document content to a provider outside the boundary.

---

## L. LLM internals & sampling — the depth questions

The applied-AI rounds rarely ask you to derive backprop, but they *do* probe whether you understand the machine you're budgeting around. These eight come up whenever the interviewer wants to find the floor of your knowledge.

**117. What is BPE, and why does vocabulary size matter?**
*"Byte-Pair Encoding builds the vocabulary by starting from raw bytes and repeatedly merging the most frequent adjacent pair until it hits the target size — about 50k for GPT-2, over 100k for modern models. The vocabulary size is a trade-off: too small and common words fragment into many tokens (slower, more expensive, worse at long-range structure); too large and rare tokens appear too few times to be learned well, and the embedding matrix balloons."* The practical point: **tokenisation is why word counts lie about cost.** Code, digits and non-Latin scripts tokenise far worse — 1 Chinese character can be 1–2 tokens, and `12345` is one token while `1 2 3 4 5` is five.

**118. Temperature, top-p and top-k — what do they do, and what would you set?**
Headline: *"They're three different ways of shaping the sampling distribution after the model produces its logits."*
- **Temperature** divides the logits: `T<1` sharpens toward the top token (more deterministic), `T>1` flattens (more random). `T=0` is greedy.
- **Top-k** keeps only the k highest-probability tokens before sampling — a hard cap on the candidate set.
- **Top-p (nucleus)** keeps the smallest set whose cumulative probability reaches p — an *adaptive* cap, so the candidate set widens when the model is uncertain and narrows when it's confident.
*"For anything I need reproducible — extraction, classification, routing — temperature 0 and structured outputs. For drafting or summarisation where variety is fine, a low temperature around 0.2–0.4. I'd rarely touch top-k; top-p is the better default because it adapts."*

**119. Temperature 0 — is that deterministic?**
*"No, and this is a favourite trap. Greedy decoding is deterministic *given identical logits*, but logits aren't bit-identical across runs — floating-point non-associativity, batched kernels that depend on batch composition, and MoE routing all move them. So for genuine reproducibility I pin the seed where offered, pin the model version, and log the exact request — which is what a regulated audit trail needs anyway."*

**120. Why is attention O(n²), and what follows from it?**
Every token attends to every other token, so doubling the sequence length roughly quadruples the attention computation — and the KV cache grows linearly while memory pressure compounds. *"That's why long context is expensive on both compute and VRAM, why 'just use a 200K window' is not a retrieval strategy, and why prefix caching and PagedAttention matter so much."*

**121. What do positional encodings do — and what is RoPE?**
Transformers have no inherent sense of order, so position has to be injected. **RoPE** (rotary position embedding) encodes position by rotating the query/key vectors by an angle proportional to position, which makes attention depend on *relative* offset — that's why RoPE-based models extrapolate to longer contexts better than learned absolute embeddings, and it's the basis for the context-extension techniques (NTK scaling, YaRN) you'll hear named.

**122. Where does a transformer actually store knowledge?**
The prevailing view is the **feed-forward layers** hold most factual associations, while attention routes and mixes information between positions. *"Practically it means fine-tuning for style and format lands well, but fine-tuning to add volatile facts is a losing game — retrieval is the right tool for knowledge, fine-tuning for behaviour."*

**123. What are the multiple attention heads for?**
Each head learns a different relationship pattern — one may track syntax, another coreference, another positional locality. Their outputs are concatenated and projected. *"And it's the reason GQA works: you can share key/value heads across query heads and cut the KV cache 8× for almost no quality loss."*

**124. Why does the model 'lose the middle' of a long context?**
Attention weights and training data both favour the beginning and the end of a sequence, so evidence buried in the middle is recalled less reliably. *"So ordering matters: put the most relevant retrieved chunk first, not last, and re-rank rather than pad. More context is often worse than better-chosen context."*

**125. Would you fine-tune to add knowledge?**
No — *"fine-tuning teaches behaviour, not facts. It's right for house style, output format, domain vocabulary, and shrinking a prompt; it's wrong for volatile or citable knowledge, where I want RAG so the source is traceable and updatable without a retrain. And in a regulated shop, a fact baked into weights can't be cited to an auditor."*

**126. How do you budget the context window?**
*"I treat it as a fixed budget I allocate: system and instructions, then retrieved evidence, then conversation history, then room for the answer — with a hard token cap per component and an alert when any one approaches it. When history starts crowding out evidence, that's the signal to summarise history rather than to buy a bigger window."*

---

## The four questions you should ask *them*

1. "Where does AI sit relative to the underwriters — shared roadmap or requirements-taking?"
2. "Which Smarter Underwriting use case is furthest into production, and what's been hardest — the model or the adoption?"
3. "How is model risk signed off for AI-assisted underwriting decisions?"
4. "For the India GCC team — is the mandate delivery capacity for the UK, or capability-building too?"

---

## Cheat-sheet summary

- **⭐ = asked last time:** team size · Azure components · RAG with code · agent evaluation · linear Git history · image-anomaly design · RAG code critique · project drill-down.
- **⚑ = Lead-graded:** evaluation layers · cost-of-error metrics · compounding agent reliability · low-code vs pro-code · fine-tuning vs RAG · reproducibility for audit.
- **Every answer:** headline → why → how → trade-off. Under 60 seconds.
- **Numbers that recur:** 3× output cost · 0.5 MB/1k token KV · 0.97²⁰ = 54% · 63% fan-out at n=100 · 865 examples for a 2-pt effect · 0.95 cache threshold · 20–40% cache hit · 75/20/5 tiering.

## Cross-links

- Depth on eval: [`07`](07-agent-evaluation-and-guardrails.md) · design: [`05`](05-system-design-round-playbook.md) · code: [`06`](06-coding-exploration-playbook.md)
- Azure/low-code/domain: [`11`](11-insurance-lloyds-domain-primer.md)
- **A second, verified bank of real questions asked at OpenAI/Meta/xAI and Lead-level loops in 2024–2026:** [`14-industry-intel-lead-ai-interview-research.md`](14-industry-intel-lead-ai-interview-research.md) §2 — use it as extra day-6 drill material. It is complementary, not overlapping: this file gives spoken model answers, `14` §2 gives the *source-verified phrasings* and the answer patterns interviewers reward.
- Leadership answers: [`12`](12-people-leadership-and-stakeholder-round.md)
- Drill format: [`09`](09-mock-interview-strategies-and-drills.md)
