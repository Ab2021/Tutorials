# RAG Evaluation and Observability

**TL;DR** — You cannot improve a RAG system you cannot measure, and you cannot measure it with a single number. Evaluation splits into **retrieval metrics** (recall@k, precision@k, MRR, nDCG, hit rate) which answer "did we fetch the right evidence?", **generation metrics** (faithfulness/groundedness, answer relevance, context precision/recall, hallucination rate) which answer "did we use it correctly?", and **operational metrics** (cost, latency, cache hit rate, error rate) which answer "is it sustainable?". Offline you run a maintained **golden set** (~200–500 curated items) through an eval harness with LLM-as-judge scoring and **CI regression gates**; online you sample live traffic, capture explicit and implicit feedback, and trace every request end-to-end. The gate that matters most in practice: **any change that drops recall@10 by more than 5 points or faithfulness by more than 3 points does not ship.**

> **60-second interview answer:** "I evaluate RAG on three layers. Retrieval: recall@10, MRR, nDCG@10 and precision@5 against a golden set of roughly 200–500 items composed about 40% simple factoid, 25% multi-hop, 15% unanswerable, 10% edge cases and 10% adversarial — I gate CI at recall@10 ≥ 0.85, MRR ≥ 0.70, nDCG@10 ≥ 0.70. Generation: faithfulness, answer relevance, and context precision/recall using RAGAS-style LLM-as-judge, gated at faithfulness ≥ 0.80 with a 3-point drop budget. Operations: p95 latency, cost per query, cache hit rate, error rate, plus a nightly sampled judge run on live traffic. Because LLM judges have known biases — position bias, verbosity bias, self-preference for same-family models — I calibrate with human labels, randomize or swap positions, and use different model families for the judge and the generator. And everything is traced per request so a user complaint becomes a specific query_id I can replay through each stage."

> **Source grounding:** The **RAG failure/hallucination taxonomy** (§3.11), the **four-part evaluation framework** — generation → retrieval → agent → production — with the three evaluation *modes* (deterministic, LLM-judge, human-in-the-loop) and the **three jobs** (observe / evaluate / regress → Langfuse / RAGAS / DeepEval) in §3.12, the **per-sentence claim/evidence citation model** (§3.13), the **context-precision and context-recall formulas** quoted under §3.2, and the **Pinpool golden-dataset case study** in §3.5 are folded in from the admitted source transcript *nitmonk — "24 Hours of AI Agent Engineering, Part 1"*, from the RAG-failure-modes segment at **~04:35–04:37**, the citation/grounding and rank-evaluation segment at **~04:51–04:53**, the RAGAS-metrics and observability segment at **~04:53–04:56**, and the evaluation-framework segment at **~11:04–11:55**. Everything else — the 200–500-item golden set, the 40/25/15/10/10 composition, the CI gate values, the LLM-judge bias magnitudes, the eval cost figures, the maturity ladder and the drift detectors — is supplementary `[A]` general knowledge or `[D]` derivation; **the transcript does not state those numbers**. Numeric labels follow the repo convention: `[S]` transcript-stated, `[A]` general knowledge, `[D]` derived. Where the auto-transcript garbles a term or a figure, that is noted inline rather than silently corrected.

---

## 1. Definition & mental model

**Plain-English definition.** RAG evaluation is the discipline of measuring, separately and continuously, whether the right documents were retrieved, whether the answer was faithful to those documents, and whether the system is healthy enough to run in production. RAG observability is the instrumentation layer that makes those measurements possible on live traffic — traces, spans, token counts, costs, and feedback signals attached to a `query_id`.

**The analogy.** Think of a restaurant kitchen. The **retriever** is the shopper: did they bring back the right ingredients? (Recall, precision, MRR.) The **generator** is the chef: given those ingredients, did they cook the right dish without substituting something from the pantry that wasn't there? (Faithfulness, relevance.) The **observability layer** is the health inspector's clipboard plus the CCTV: temperature logs, ticket times, and a recording of every service so you can replay the night a customer complained.

The critical insight from that analogy: **a bad meal has two very different causes, and blaming the chef when the shopper failed is the single most common diagnostic error in RAG engineering.** Roughly 73% of wrong answers are retrieval failures.

**The RAG Triad**, the mental model interviewers expect you to name:

```
                    ┌─────────────────────────┐
                    │   Context Relevance     │  Is the retrieved context
                    │   (retriever quality)   │  relevant to the question?
                    └───────────┬─────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐      ┌────────────────┐      ┌──────────────────┐
│   Question    │─────▶│  Groundedness  │◀─────│     Answer       │
│               │      │ (is the answer │      │                  │
│               │      │  supported by  │      │                  │
└───────────────┘      │  the context?) │      └────────┬─────────┘
                       └────────────────┘               │
                                                        ▼
                                             ┌────────────────────┐
                                             │ Answer Relevance   │
                                             │ (does it address   │
                                             │  the question?)    │
                                             └────────────────────┘
```

Every RAG failure is a break in one of these three edges, and you must measure all three because they fail independently. A perfect retriever with an ungrounded generator produces hallucination. A faithful generator with a bad retriever produces "I don't have information about that" for answerable questions — which users rate as *worse* than a wrong answer, because it looks broken.

**The one-sentence framing:** *RAG evaluation is a funnel diagnostic — you measure each stage, find the first stage that lost the signal, and fix that stage instead of guessing.*

---

## 2. Why it exists — the problem it solves

Without evaluation, RAG development is vibes-driven and the following things happen, reliably, in this order:

1. **The demo works.** Ten hand-picked queries all return good answers. The team ships.
2. **Real traffic exposes the tail.** 20% of queries fail, in ways nobody can characterize because nobody categorized them.
3. **Fixes become guesses.** Someone raises top-k from 5 to 10. It seems better. Nobody knows.
4. **Silent degradation.** A parser upgrade, a chunking tweak, or a provider-side model update drops faithfulness from 0.88 to 0.71. No error rate moves, no alert fires, and it goes unnoticed for six weeks until a customer escalates.
5. **Cost drifts.** Context grew, top-k crept, rerank depth was raised "just to be safe". The bill is 4x the model.
6. **Argument replaces engineering.** Stakeholders disagree about whether the system is good and there is no artefact to settle it.

**What each metric class buys you:**

| Metric class | Question it answers | Decides |
|---|---|---|
| Retrieval metrics | Did we fetch the gold evidence? | Chunking, embedding model, hybrid weights, rerank depth, `ef_search` |
| Generation metrics | Is the answer supported and on-topic? | Prompt, model choice, temperature, context budget, abstention policy |
| Operational metrics | Is it fast/cheap/reliable? | Caching, model tiering, autoscaling, degradation policy |
| Safety metrics | Is it safe and compliant? | Guardrail thresholds, injection defences, PII policy |
| Business metrics | Is it useful? | Whether to keep investing |

**The specific thing evaluation prevents that nothing else catches — silent quality degradation.** It is the only failure mode in RAG with no natural alarm: latency alerts do not fire, error rates stay at zero, and the system is quietly answering with 0.71 faithfulness instead of 0.88. The only detector is a scheduled eval that runs the same golden set every night and compares against a stored baseline.

---

## 3. Internal mechanics — how it actually works

### 3.1 The full evaluation architecture

```
┌────────────────────── OFFLINE EVALUATION ──────────────────────────┐
│                                                                    │
│  Golden set (200-500 items)                                        │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │ { question, expected_answer, gold_chunk_ids[],            │     │
│  │   category, difficulty, must_abstain: bool }              │     │
│  └───────────────────────┬──────────────────────────────────┘     │
│                          │                                         │
│                          ▼                                         │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  Eval harness                                              │   │
│  │   for each item:                                           │   │
│  │     run pipeline → capture retrieved_chunks + answer       │   │
│  │     compute retrieval metrics  (deterministic, cheap)      │   │
│  │     compute generation metrics (LLM-as-judge, costly)      │   │
│  │     attach cost + latency per stage                        │   │
│  └───────────────────────┬────────────────────────────────────┘   │
│                          ▼                                         │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  CI regression gates                                       │   │
│  │   recall@10 < 0.85            → BLOCK                      │   │
│  │   recall@10 drop > 5 pts      → BLOCK                      │   │
│  │   faithfulness < 0.80         → BLOCK                      │   │
│  │   faithfulness drop > 3 pts   → BLOCK                      │   │
│  │   p95 latency > 2.5 s         → WARN                       │   │
│  │   cost/query > $0.02          → WARN                       │   │
│  └────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────── ONLINE OBSERVABILITY ────────────────────────┐
│                                                                    │
│  Every request emits a trace keyed by query_id:                    │
│   ├─ span: cache_lookup      (hit/miss, similarity)                │
│   ├─ span: query_rewrite     (rewritten text, model, tokens)       │
│   ├─ span: retrieve_dense    (top-k ids+scores, ef_search)         │
│   ├─ span: retrieve_sparse   (top-k ids+scores)                    │
│   ├─ span: rrf_fusion        (fused ranking)                       │
│   ├─ span: rerank            (pre/post ranking, model)             │
│   ├─ span: context_assembly  (token count, chunks kept, deduped)   │
│   ├─ span: generate          (model, in/out tokens, TTFT, cost)    │
│   └─ span: post_check        (citation validity, PII, safety)      │
│                                                                    │
│  + feedback: thumbs, copy, rephrase-within-30s, escalation         │
│  + nightly sampled judge on 1-5% of live traffic                   │
└────────────────────────────────────────────────────────────────────┘
```

### 3.2 Retrieval metrics — the definitions and the formulas

| Metric | Definition | Formula / notes |
|---|---|---|
| **Recall@k** | Fraction of golden runs where the gold chunk appears in the top k | `hits_with_gold_in_top_k / total_queries`. The single most important retrieval metric |
| **Precision@k** | Fraction of the top k that are relevant | `relevant_in_top_k / k`. Falls naturally as k grows |
| **MRR** | Mean reciprocal rank of the first relevant result | `mean(1 / rank_of_first_relevant)`. Measures "how high is the first good hit" |
| **nDCG@k** | Normalized discounted cumulative gain — rewards relevant results appearing high, with graded relevance | `DCG@k / IDCG@k` where `DCG = Σ (2^rel_i − 1) / log2(i+1)` |
| **Hit rate** | Fraction of queries with at least one relevant result | Binary version of recall@k. Good for quick sanity checks |
| **Context recall** | Fraction of the golden answer's claims that are supported by retrieved context | RAGAS metric; needs claim decomposition |
| **Context precision** | Fraction of retrieved chunks that are actually useful for the answer | RAGAS metric; precision of the *context*, not the ranking |

**Worked example.** 10 golden items. Gold chunk found at ranks: 1, 3, 2, 47, 1, 5, 2, 1, 12, 4.

- **Recall@5** = items with gold in top 5 = ranks 1,3,2,___,1,5,2,1,___,4 = **8/10 = 0.80**
- **Recall@10** = ranks ≤ 10 = 1,3,2,__,1,5,2,1,__,4 = **8/10 = 0.80** (rank 12 and 47 miss)
- **MRR** = mean(1/1, 1/3, 1/2, 1/47, 1/1, 1/5, 1/2, 1/1, 1/12, 1/4) = mean(1.0, 0.333, 0.5, 0.021, 1.0, 0.2, 0.5, 1.0, 0.083, 0.25) = **0.489**

That MRR of 0.489 tells you the retriever *usually* works but rank-4-through-10 quality is poor — which points at reranking, not at the embedding model. **That is the diagnostic value of the metric set: each metric points at a different stage.**

**The rank-evaluation framing, as the transcript states it** `[S]` (~04:51–04:53). The transcript introduces this metric group with a causal claim and then four one-line definitions — and the causal claim is the part that scores points:

> *"R comes first. If retrieval is bad then A and G will be definitely bad."*

| Metric (transcript wording) `[S]` | Definition as stated | Equivalent here |
|---|---|---|
| **Recall@k** | "Are the needed chunks in the top k?" — the first metric to take care of | Recall@k |
| **Precision@k** | "How much of the top k is relevant?" | Precision@k |
| **MRR** | "How high is the first correct hit?" | MRR |
| **nDCG** | "Relevant docs ranked near the top" | nDCG@k |

The two **RAGAS context metrics** are also given as explicit arithmetic `[S]` (~11:20–11:30), and it is worth quoting the transcript's exact framing because the two are constantly confused:

```
context precision = relevant retrieved chunks ÷ all retrieved chunks      → "ranking"
context recall    = reference claims supported by context ÷ total claims  → "coverage"
```

The transcript's own summary of the distinction: **context precision is about ranking — did the useful chunks come back *high*? Context recall is about coverage — did enough of the needed information come back *at all*?** Its worked illustration is a five-chunk return in which the genuinely useful chunk sits at rank 3–5 rather than rank 1: precision suffers because the useful chunk is ranked low, not because it is absent. That is exactly the signature that points at the reranker rather than at the embedding model, and it is why both metrics must be read together.

**One caveat on the transcript's numbers.** The worked failure example it walks through — the question *"what is the refund policy for an annual subscription?"*, an answer asserting a 90-day refund window, and a reported score of *0.31* — is presented as an illustration, and the transcript does not say which metric the 0.31 belongs to. Treat it as a teaching example, not a benchmark figure, and do not quote it as `[S]` data. The separately quoted *42% faithfulness* in the trace-debugging example (~11:42) is likewise illustrative.

### 3.3 Generation metrics

| Metric | Definition | How measured | Target |
|---|---|---|---|
| **Faithfulness / groundedness** | Every factual claim in the answer is supported by the retrieved context | Decompose the answer into atomic claims; for each, check entailment against the context (NLI model or LLM judge) | ≥ 0.80 gated; 0.90 good |
| **Answer relevance** | The answer actually addresses the question asked | LLM judge: generate reverse-questions from the answer and measure similarity to the original; or direct 1–5 scoring | ≥ 0.75 |
| **Hallucination rate** | Fraction of answers containing ≥1 unsupported claim | Derived from faithfulness; count, not average | < 5% gated; < 2% good |
| **Answer correctness** | The answer matches the gold answer semantically | LLM judge or exact/embedding match against the golden answer | ≥ 0.80 |
| **Abstention correctness** | For unanswerable questions, does it correctly abstain? | Count of correct refusals / total unanswerable items | ≥ 0.85 |
| **Citation validity** | Every citation label resolves to a real retrieved chunk | Deterministic check against the citation map | ≥ 0.995 |
| **Citation support** | The cited chunk actually supports the claim it is attached to | LLM judge per citation | ≥ 0.90 |
| **Toxicity / safety** | Output contains no policy-violating content | Classifier | ~0 |

**Faithfulness worked example.** Question: "What is the EU refund window?"
Retrieved context: *"EU customers may request a refund within 30 days of delivery. Refunds are processed within 5 business days."*
Answer: *"EU customers have 30 days to request a refund, and refunds are processed within 5 business days."*
Claims: (1) EU refund window is 30 days → entailed. (2) Refunds processed within 5 business days → entailed.
**Faithfulness = 2/2 = 1.0.**

Now change the answer to: *"EU customers have 30 days to request a refund, and refunds are processed within 3 business days."*
Claim 2 is contradicted. **Faithfulness = 1/2 = 0.5.** The NLI label is `contradiction`, which is treated as a hard failure (not merely "unsupported") because a contradiction is worse than a gap — it is actively wrong.

### 3.4 LLM-as-judge — and its biases

LLM-as-judge is the only economically viable way to score faithfulness and relevance at scale. It is also a measurement instrument with known systematic errors, and quoting them is what separates a senior answer from a junior one.

| Bias | What it looks like | Magnitude | Mitigation |
|---|---|---|---|
| **Position bias** | The judge prefers whichever candidate is presented first | 5–15% preference swing | Randomize or swap order and average; or score pairwise both ways |
| **Verbosity bias** | Longer answers score higher regardless of quality | 10–20% | Normalize by length; instruct "ignore length"; include length in analysis |
| **Self-preference bias** | A judge favors outputs from its own model family | 5–15% | Use a **different model family** for the judge than the generator |
| **Sycophancy / agreement bias** | Judge agrees with a confident, assertively phrased claim | Variable | Blind the judge to which system produced the output |
| **Score clustering** | Judge gives 4/5 to almost everything | Common with 1–5 scales | Use binary/ternary labels or require a rationale before the score |
| **Reference leakage** | Judge sees the gold answer and anchors on it | Variable | Separate "with reference" and "without reference" eval runs |

**Practical judge design rules:**
1. Ask for the **rationale before the score** — it makes the score traceable and reduces clustering.
2. Use **binary or 3-level labels** for claim-level checks (entailed / contradicted / unsupported); reserve 1–5 scales for holistic quality.
3. **Calibrate against human labels** on 50–100 items once per quarter; report judge-human agreement (Cohen's kappa > 0.6 is acceptable).
4. Use the **cheapest model that passes calibration** — Haiku-class judging runs ~$0.002 per triad vs ~$0.018 for GPT-4o-class, a 9x difference.
5. **Version the judge prompt.** A judge prompt change is a metric definition change and invalidates historical comparisons.

### 3.5 Golden set construction

The golden set is the most valuable artefact in a RAG project and the one teams most often underinvest in.

**Composition target:**

| Category | Share | Why |
|---|---|---|
| Simple factoid (single chunk answers) | **40%** | The dominant production traffic shape |
| Multi-hop / multi-document | **25%** | Where naive RAG fails hardest |
| Unanswerable (must abstain) | **15%** | Measures the abstention policy — otherwise untested |
| Edge cases (tables, code, acronyms, non-English) | **10%** | Where chunking and hybrid retrieval earn their keep |
| Adversarial (injection attempts, PII probes, jailbreaks) | **10%** | Measures guardrails |

**Size:** 200 items is the minimum to see a 5-point change; 500 gives you per-category signal. Below ~150, metric noise exceeds the effect size of most changes you will make. Grow it by 20–50 items every time production reveals a failure the set did not contain — **production failures are the best golden-set source there is.**

**Sourcing:**
- 30–40% hand-curated by domain experts (highest quality, highest cost).
- 30–40% mined from real production queries with human-verified gold answers.
- 20–30% **synthetically generated** — take a chunk, ask a strong model to write a question that chunk answers, and record the chunk as gold. Fast and cheap, but biased toward questions that look like their answers, so it must be blended with real queries.

**Versioning:** the golden set lives in git, changes are reviewed, and every eval run records the golden-set version so metric history is comparable.

**The transcript's own worked case: the Pinpool golden set** `[S]` (~11:43–11:52). This is the most concrete golden-set story in the source and it is worth being able to retell, because it is a *code-generation* golden set rather than a question-answering one — which is exactly the case candidates fail to generalise to.

- **The product.** An agentic system that turns a natural-language request — *"build me a 10-million-customer e-commerce order system using AWS serverless components"* — into working infrastructure: Lambda, DynamoDB, EventBridge, generated as Terraform.
- **The business question.** Not "is the answer grounded?" but *"is the prompt 90%, 95%, even 100% accurate?"* — an output-correctness question, asked before shipping.
- **The golden set.** Roughly **2,000–3,000** `{question, answer}` pairs where the answer is the **hand-written Terraform** for that architecture. The same question is then passed to the built agentic system (a supervisor agent with several child agents) and the **generated Terraform is compared against the reference**.
- **The grading.** The reference is `A`; a similar generation is `A'` and grades well; an unrelated generation is `B` and grades badly. *"A prime is better than B; A prime gets a good grading."* Note what this implies: the grade is a **similarity judgement between two artefacts**, which for code can be deterministic (structural/AST comparison, plan diff, does it apply) rather than an LLM judgement — the transcript is explicit that both a *"deterministic way — equal-to-equal check"* and *"an LLM as a judge"* exist, and that you choose.
- **The loop.** Grade candidate prompts against the set, deploy the winner to production, and repeat — driving toward *"the best set of prompts, skills, tool calls, tool descriptions, input schema."*
- **Why it never ends.** When the user population changes, the question distribution changes. The transcript names this as **data drift and concept drift**: your prompt and system that passed yesterday can fail on the new questions, so the golden set is rebuilt for the new distribution and the comparison run again — prompt v1 versus prompt v2, deploy the better.

**The transferable lesson, which is stronger than the anecdote:** a golden set is not a Q&A list, it is a **`{input, reference output}` pair list where the reference is the artefact you actually care about** — a Terraform plan, a JSON object, a SQL query, a generated ticket. Grade by comparing the produced artefact to the reference, using the cheapest method that is honest about similarity. The 2,000–3,000-item scale is also instructive: far larger than the 200–500 that suffices for RAG faithfulness measurement, because correctness-of-artefact needs per-shape coverage rather than per-category coverage. `[D]`

### 3.6 Offline vs online evaluation

| Dimension | Offline | Online |
|---|---|---|
| Data | Golden set | Live traffic |
| Judge | Full metric suite (LLM-as-judge) | Sampled (1–5%) + deterministic checks |
| Cost | ~$0.002–0.02 per triad | Sampled, so ~$0.0001/query blended |
| Latency to signal | Immediate (minutes) | Hours to days |
| Detects | Regressions, capability changes | Real distribution, real failure modes, drift |
| Gates | CI/CD — blocks deploys | Alerts, rollbacks |
| Weakness | Distribution shift from real traffic | No ground truth; slow; expensive |

**The pattern: offline gates the deploy, online validates the deploy.** Run the golden set in CI (block on regression), then canary 5% of production traffic with per-class metric comparison for 24–48 hours before full rollout.

### 3.7 Tracing

Every request gets a **trace** with **spans** for each pipeline stage. The OpenTelemetry GenAI semantic conventions define the standard attribute names; LangSmith, Langfuse, Arize Phoenix, and Braintrust implement them natively.

Minimum span attributes:

| Span | Attributes to capture |
|---|---|
| `retrieve` | query text (post-rewrite), index version, `ef_search`, top-k ids, scores, latency |
| `rerank` | pre-rank → post-rank mapping for the top 10, model, latency |
| `context_assembly` | chunks kept, chunks dropped (budget/dedupe), final token count |
| `generate` | model + version, input tokens, output tokens, temperature, TTFT, total latency, cost |
| `post_check` | citation validity, groundedness score, safety flags |

**The single most valuable property of a trace: replayability.** When a user complains, you retrieve their `query_id`, replay the trace, and see exactly which stage lost the signal — without guessing and without reproducing the query.

### 3.8 Feedback loops

| Signal | Type | Interpretation |
|---|---|---|
| Thumbs up / down | Explicit | Direct quality label; ~2–5% response rate |
| Comment / free text | Explicit | High signal, low volume; route to golden-set candidates |
| **Rephrase within 30 s** | Implicit | **Strong negative.** The first answer failed. Best implicit signal available |
| Copy / export / cite click | Implicit | Positive engagement |
| Conversation continues deeper | Implicit | Mild positive |
| Escalation to human / ticket opened | Implicit | Strong negative, and measurable in cost |
| Session abandonment | Implicit | Negative |

**The rephrase-within-30-seconds heuristic is worth instrumenting explicitly** — it correlates better with human-judged quality than thumbs do, because it requires no user effort.

### 3.9 Regression gates

```yaml
# CI quality gates — evaluated on every pipeline change
gates:
  # Retrieval — deterministic, cheap, run on every commit
  recall_at_10:
    min: 0.85
    max_drop_from_baseline: 0.05
    severity: block
  mrr:
    min: 0.70
    max_drop_from_baseline: 0.05
    severity: block
  ndcg_at_10:
    min: 0.70
    max_drop_from_baseline: 0.05
    severity: block
  precision_at_5:
    min: 0.60
    max_drop_from_baseline: 0.05
    severity: warn

  # Generation — LLM-as-judge, run nightly or on-demand
  faithfulness:
    min: 0.80
    max_drop_from_baseline: 0.03
    severity: block
  answer_relevance:
    min: 0.75
    max_drop_from_baseline: 0.03
    severity: block
  context_precision:
    min: 0.60
    max_drop_from_baseline: 0.05
    severity: warn
  abstention_correctness:
    min: 0.85
    max_drop_from_baseline: 0.05
    severity: block
  citation_validity:
    min: 0.995
    severity: block

  # Operations — always gate these
  p95_latency_ms:
    max: 2500
    severity: warn
  cost_per_query_usd:
    max: 0.02
    severity: warn
  error_rate:
    max: 0.01
    severity: block
```

**Why the drop-based gates matter more than the absolute floors:** an absolute floor of 0.85 recall lets a 0.86 → 0.86 → 0.85 slow slide through. The `max_drop_from_baseline: 0.05` clause catches the first commit that causes the slide, when the cause is obvious.

### 3.10 Drift detection

| Drift type | Signal | Detection |
|---|---|---|
| **Query drift** | New topics/intents appear that the golden set does not cover | Embed live queries, cluster weekly, alert on new clusters > 5% of traffic |
| **Corpus drift** | Document distribution shifts (new sources, new formats) | Track chunk length distribution, parse-failure rate, OCR rate, new source counts |
| **Embedding drift** | Provider silently updates a hosted embedding model | Nightly retrieval benchmark; **pin model versions explicitly** |
| **Model drift** | Provider updates the generator | Nightly faithfulness run; version every model in the trace |
| **Index staleness** | Ingestion lag grows | `p95(source_change → searchable)` as an SLI |
| **Answer drift** | Outputs change shape (longer, more caveated, more refusals) | Track output token distribution and abstention rate |

### 3.11 The hallucination taxonomy — hallucinations have causes, and the causes are upstream `[S]`

The single most useful reframing in the transcript's failure-modes segment (~04:35–04:37) is that "hallucination" is not one failure — it is a **label users apply to a family of failures that originate at different stages**, most of them before generation. Asking "how do I stop hallucination?" is the wrong question; asking "which of these is it?" is the right one. Here is the transcript's list, regrouped by the stage that actually caused it:

| Origin stage | Named failure `[S]` | What it is | The metric that catches it |
|---|---|---|---|
| **Ingestion** | Bad chunking | The split destroyed the meaning — the unit of retrieval is not the unit of meaning | Recall@k, chunk-quality review |
| **Ingestion** | Wrong chunk | A chunk that exists but asserts something the source does not | Citation support |
| **Retrieval** | **Missing chunk** | The gold evidence was never returned | Recall@k / context recall |
| **Retrieval** | **Low recall** | Only part of the needed evidence came back — the answer is assembled from an incomplete set | Context recall (coverage) |
| **Retrieval** | Semantic mismatch | The query and the document mean the same thing in different words, and dense retrieval missed it | Recall@k on a paraphrase slice of the golden set |
| **Retrieval** | Keyword mismatch | An exact identifier, code or rare term was never matched | Recall@k on the identifier slice; hybrid-vs-dense comparison |
| **Retrieval** | **Ambiguous query** | The query itself does not say what it wants — no retriever can be blamed | Rewrite quality; human review of the raw query |
| **Generation** | **Context ignore** | The evidence is *in the window* and the model answers from its priors instead | Faithfulness against the assembled context |
| **Generation** | **Context confusion** | Contradictory or overlapping chunks in the window and the model blends them into a claim no chunk makes | Faithfulness (contradiction label); context precision |
| **Attribution** | **Citation mismatch** | The cited source does not contain the claim it is attached to | Citation *support*, not just citation *validity* |

**Say the grouping out loud and it becomes a diagnostic.** Seven of the ten are retrieval-or-ingestion failures; only two are generation failures. That is the same 73%-retrieval thesis this file opens with, but expressed as a taxonomy an interviewer can probe: if you cannot name more retrieval-side causes of hallucination than generation-side ones, you will reach for the prompt when you should be reaching for the index.

**The transcript's defence stack, in the order it gives it** `[S]` (~04:47–04:51). It frames this as *"production system stack defence across retrieval, prompting and verification"* and then walks the layers:

1. **Better retrieval.** Not a single technique — *"retrieval is not a single part, it's hybrid search"*: semantic similarity, BM25 lexical search, metadata filtering, and query writing (HyDE, multi-query, decomposition, enhancement).
2. **A strict prompt.** *"Answer only from the context"* — plus an explicit abstention path so the model has a sanctioned way to say it does not know.
3. **Require citations, and refuse if unsupported.** The answer must carry the provenance of each claim.
4. **A grounding check after generation.** Compare the generated answer against what was retrieved, claim by claim — *"if generation was done but it does not contain what was retrieved, then there is a problem."*
5. **A faithfulness/groundedness score on every answer**, plus confidence and abstention — the transcript's framing is that this score runs *on every answer*, not on a sample, which is the aspiration the sampled-live-judging design in §3.6 approximates with a 1–5% sample.

**The worked mismatch to have ready** `[S]`: the chunk says *15 days* of leave and the model generates *30 days*. Nothing about that answer is ungrammatical, unreferenced or obviously wrong — it is a **contradiction of the evidence**, which is why the check must be claim-level entailment against the retrieved chunks and not a fluency or citation-label check. A contradicted claim is a hard failure, distinct from an unsupported one.

### 3.12 The four-part evaluation framework `[S]`

The transcript's evaluation segment (~11:04–11:15) gives a mental model that is broader than RAG evaluation and is the right answer when an interviewer asks "how do you evaluate an *AI system*" rather than "how do you evaluate a retriever". It has **four parts, evaluated in this order**, because that is the order in which the system produces them:

| # | Part `[S]` | What is measured | Why it sits here |
|---|---|---|---|
| 1 | **Generation** | Faithfulness, answer correctness, answer relevance | "Whenever things are getting generated" — the output itself |
| 2 | **Retrieval** | Context precision, context recall, retrieval relevance | The evidence the generation depended on |
| 3 | **Agent** | Tool calls, task completion, trajectory | Only meaningful once tools and MCP are in the loop |
| 4 | **Production** | Latency, cost, tokens, error rate | Only measurable once real traffic exists |

**Say it as a funnel, not a list:** the output is scored first because it is what the user sees, then you descend to what produced it, and the production layer is the one that only exists after deployment. The transcript's own phrasing — *"start with output, then goes to retrieval/RAG, then the complete set of agent, then production"* — is the sentence to reproduce.

**Three evaluation *modes*, which are orthogonal to the four parts** `[S]` (~11:08–11:10):

| Mode | What it is | Use it for |
|---|---|---|
| **Deterministic** | Equality/structural check — no model involved | Retrieval metrics, citation validity, schema conformance, code similarity (the Pinpool `A` vs `A'` grade) |
| **LLM-as-judge** | A model receives the question, the answer and the context, and returns a score *plus a reason* | Faithfulness, relevance, context precision/recall at scale |
| **Human in the loop** | A person approves, rejects, or attaches a reason | Calibration, escalations, high-stakes approvals, the cases the judge disputes |

**Three jobs — and this is the cleanest distinction in the whole segment** `[S]` (~11:14–11:16). Observability, evaluation and regression testing are **three different jobs, and each tool does exactly one of them**:

| Job `[S]` | The question it answers | Tool named in the transcript | What it cannot do |
|---|---|---|---|
| **Observe** | *"What exactly happened?"* — traces of tool calls, pipeline calls, timings, scores, tokens, cost | **Langfuse** | Does not tell you whether the output was good |
| **Evaluate** | *"Is my RAG system retrieving and answering correctly?"* — faithfulness, context precision, context recall, answer relevance | **RAGAS** | *"It has no visibility into what happened during execution — only the final output"* |
| **Regress** | *"Did my changes break something?"* — the complete pipeline, in CI | **DeepEval** *(the transcript auto-renders the name as "deep")* | Does not diagnose; it gates |

That last line is worth quoting in an interview because it is a real limitation rather than a feature list: **RAGAS scores the output, it does not see the run.** When a RAGAS score is bad you still have to open the trace to find *which stage* lost the signal — the two tools are complements, and a design that has RAGAS without tracing will tell you that quality fell without telling you why.

**The four things observability must capture** `[S]` (~04:54–04:55), stated as a list to memorise: **timings, scores, tokens, cost** — *"every stage with timings, scores, tokens and cost"*. Timings per stage (retrieve / augment / generate); scores per stage (the similarity and judge values — "0.98, 0.83"); tokens both at ingestion (embedding) and at query time (retrieval into the prompt, then generation); and cost wherever tokens are involved. The transcript's implementation note is that Langfuse callbacks are how you harvest these, and RAGAS is how you get the faithfulness/recall values — which is exactly the split in the table above.

### 3.13 Citation and grounding — the per-sentence claim/evidence map `[S]`

The transcript's citation segment (~04:51–04:52) is short but it names the artefact precisely, and it is the artefact most candidates omit from an evaluation design:

> *"Claim–evidence mapping per sentence — where exactly the sentence is coming from, from which paragraph, which section, which table."*

Four properties, in the transcript's own vocabulary `[S]`:

| Property `[S]` | Meaning | Maps to here |
|---|---|---|
| **Claim–evidence mapping per sentence** | Every sentence of the answer is tied to the specific chunk *and* location — paragraph, section, table | Citation map, at sentence granularity rather than answer granularity |
| **Citation correctness** | The mapping resolves and points at the right source | Citation validity + citation support |
| **Source completeness** | *"All claims covered"* — no sentence of the answer is left uncited | The inverse of hallucination rate: the fraction of claims with an evidence link |
| **Flag answered claims** | Which claims were answered (supported) versus not — surfaced, not hidden | Per-claim labels: entailed / contradicted / unsupported |

**The interview test the transcript gives you.** When someone claims your RAG is hallucinating, the defence is not an argument — it is the artefact: *"you write that this is section 4.5, paragraph [n] — and you go to that paragraph and the same answer matches, then there is no hallucination happening."* That is why sentence-level mapping beats answer-level citation: a single citation on a five-sentence answer can be correct while three of the sentences are unsupported, and answer-level validity will never reveal it. Pair it with the transcript's *rank result* idea: the retrieved chunk's position is part of the record, so you can see whether the evidence that justified the claim was rank 1 or rank 5 — which is what tells you whether the answer is hanging off a marginal retrieval.

---

## 4. Variants & taxonomy

### 4.1 Evaluation tooling

| Tool | Strength | Weakness | Best for |
|---|---|---|---|
| **RAGAS** | Purpose-built RAG metric suite; open source | LLM-judge cost; needs care on judging prompts | The default for RAG-specific offline eval |
| **DeepEval** | Pytest-style; CI-native | Smaller metric library | Teams that want eval as unit tests |
| **TruLens** | Feedback-function framework; strong tracing | More assembly required | Custom metric workflows |
| **UpTrain** | Broad metric coverage; open source | Less RAG-specific depth | General LLM eval |
| **Braintrust** | Strong experiment tracking + eval UI | Commercial | Teams running many experiments |
| **LangSmith** | Deep LangChain integration; tracing + eval in one | Best in the LangChain ecosystem | LangChain-based stacks |
| **Langfuse** | Open-source tracing, self-hostable, cost tracking | Less opinionated on metrics | Teams needing self-hosted observability |
| **Arize Phoenix** | OTel-native, strong on drift and embedding analysis | — | Production monitoring |
| **OpenTelemetry GenAI** | Vendor-neutral standard attribute names | A convention, not a product | Avoiding lock-in |

### 4.2 Evaluation cadences

| Cadence | What runs | Cost | Purpose |
|---|---|---|---|
| **Per commit** | Retrieval metrics only (deterministic, no LLM) | ~$0 | Catch chunking/index regressions before merge |
| **Per PR to prompt/model** | Full suite on 200-item golden set | ~$1–5 | Gate the deploy |
| **Nightly** | Full suite + drift detectors | ~$1–5/night | Catch silent degradation, provider-side changes |
| **Weekly** | Full suite on the full 500-item set + per-category breakdown + judge calibration spot-check | ~$5–15 | Trend analysis, identify the weakest category |
| **Continuous** | Sampled live judging (1–5%), operational metrics, feedback capture | ~$0.0001/query | Real-world quality, alerts |
| **Quarterly** | Human calibration of the judge, golden-set refresh, adversarial red-team | Time | Keep the measurement instrument honest |

### 4.3 Cost of evaluation

| Judge model class | Cost per RAG triad | Cost to run 500 items |
|---|---|---|
| Small (Haiku-class) | ~$0.002 | ~$1.00 |
| Mid (GPT-4o-class) | ~$0.018 | ~$9.00 |
| Frontier reasoning | ~$0.05+ | ~$25+ |

**Rule: use the cheapest judge that passes human calibration.** A Haiku-class judge at kappa > 0.6 against human labels is a better instrument than a frontier judge at 9x the price, because you can then afford to run it nightly and on live samples. Reserve the expensive judge for adjudicating disagreements and for the quarterly calibration run.

### 4.4 Maturity levels

| Level | What exists | Typical numbers |
|---|---|---|
| **L0 — None** | Vibes | No idea whether it works |
| **L1 — Manual** | A spreadsheet of 20 queries a human runs | Recall unknown; regressions invisible |
| **L2 — Golden set + retrieval metrics** | 100–200 items, deterministic metrics in CI | Recall@10 known; generation quality still guessed |
| **L3 — Full offline + tracing** | RAGAS metrics, CI gates, per-request traces | Regressions blocked; complaints debuggable |
| **L4 — Continuous + drift + feedback loop** | Nightly eval, live sampling, feedback-driven golden set growth, drift alerts | Silent degradation caught within 24 h |
| **L5 — Closed loop** | Feedback → golden set → prompt/model iteration → A/B → auto-promote | Quality compounds; the system improves without a human in the loop for routine changes |

---

## 5. Pros / Cons

| Pros | Cons |
|---|---|
| Converts "does it work?" from opinion into a number you can trend | The golden set is expensive to build and nobody wants to fund it |
| CI gates make regressions impossible to ship silently | LLM-as-judge costs money and has biases that must be calibrated away |
| Tracing turns a vague complaint into a replayable `query_id` | Full-fidelity tracing is a storage and PII liability of its own |
| Per-stage metrics localize failures to the right stage (retrieval vs. generation) | Metric gaming: optimizing nDCG can hurt the metric that actually matters to users |
| Online feedback gives ground truth the golden set cannot contain | Feedback is sparse (2–5%), biased toward extreme experiences |
| Drift detection catches provider-side changes you did not make | Detection requires a baseline that must itself be revalidated |
| Eval cost is small relative to generation cost (~$1–9 per full run) | Eval *engineering* cost is large — 0.5–1 FTE for a serious harness |
| Synthetic test generation scales coverage cheaply | Synthetic questions are biased toward their own answers and miss real query shapes |
| Enables safe A/B testing of prompts, models, and retrieval changes | A/B tests on quality require enough volume; low-traffic systems cannot reach significance |

---

## 6. Exceptions, edge cases & failure modes

### 6.1 Optimizing the metric instead of the outcome

**Symptom:** nDCG@10 climbs steadily while user thumbs-down also climbs.
**Cause:** the golden set drifted from real traffic, or the metric rewards something users do not.
**Mitigation:** always pair offline metrics with an online signal; review the top 20 thumbs-down queries weekly and check whether they look like the golden set; rebalance the golden set toward real query shapes.

### 6.2 Golden set contamination

**Symptom:** metrics improve after a model update in a way that does not reproduce in production.
**Cause:** the golden set's questions were generated from chunks, so they embed very close to their gold chunks — the set is easier than reality. Or the model was trained on the corpus.
**Mitigation:** hold out a **human-authored** subset (30–40%) that was never generated from the corpus; track metrics on that subset separately; add a `source: human | synthetic | production` field to every item and report metrics by source.

### 6.3 LLM judge drift

**Symptom:** faithfulness jumps from 0.84 to 0.91 overnight with no system change.
**Cause:** the judge model was updated by its provider. Your measurement instrument changed, not your system.
**Mitigation:** pin the judge model version; store judge model + prompt version with every eval run; include a small set of **frozen reference items** with known human labels in every run — if the judge's score on the frozen items moves, the instrument moved.

### 6.4 Eval cost surprise

**Symptom:** the nightly full-suite eval on 5,000 items with a frontier judge costs $300/night.
**Mitigation:** eval on 200–500 items, not the whole set, nightly; full set weekly; use the cheapest calibrated judge; cache judge results for unchanged (question, context, answer) triples — most nightly runs re-score identical triples, and caching takes eval cost to nearly zero on stable days.

### 6.5 Tracing PII leakage

**Symptom:** the observability platform stores full query text and retrieved documents, including customer PII, in a system outside your compliance boundary.
**Mitigation:** redact PII in spans before export; store full text in a short-retention store (7–30 days) and only redacted aggregates long-term; treat the trace store as in-scope for your data-residency and deletion obligations; never log secrets or auth tokens in spans.

### 6.6 Feedback bias

**Symptom:** feedback-driven improvements make the system worse for the silent majority.
**Cause:** the 2–5% who leave feedback skew toward extreme experiences; the satisfied majority never speaks.
**Mitigation:** pair explicit feedback with implicit signals (rephrase rate, escalation rate, session length) which cover 100% of traffic; sample non-responders for review; never tune solely on explicit feedback.

### 6.7 The unanswerable question nobody tested

**Symptom:** the system confidently answers questions the corpus does not cover, because the golden set contained no unanswerable items.
**Mitigation:** **15% of the golden set must be unanswerable**, with a required abstention. This is the only way the abstention policy is ever exercised, and it is the single most commonly missing category.

### 6.8 Metric noise mistaken for signal

**Symptom:** recall@10 "dropped" from 0.86 to 0.84 on a 150-item set and the team panics.
**Cause:** with n=150 and binary outcomes, the standard error on a 0.85 proportion is roughly ±0.03. A 0.02 move is noise.
**Mitigation:** size the golden set to the effect you need to detect (200 minimum, 500 for per-category); always report confidence intervals; use paired comparisons on the same items rather than independent runs — paired tests are far more sensitive.

### 6.9 Eval-set staleness

**Symptom:** the golden set has been frozen for 18 months; the product has changed completely.
**Mitigation:** add 20–50 items per quarter from production; retire items whose source documents no longer exist; version the set and record the version with every run.

### 6.10 Silent quality degradation (the headline failure)

**Symptom:** nothing alerts; users slowly stop trusting the system.
**Cause:** a provider-side model update, an embedding model update, a parser change, a new document source with different formatting, or a chunking tweak.
**Mitigation:** nightly automated eval with drop-based gates; frozen reference items to distinguish system drift from judge drift; canary deploys with per-class metric comparison; alert on any metric moving beyond its noise band for two consecutive runs.

### 6.11 Citation validity passing while citation support fails

**Symptom:** every citation label resolves, so `citation_validity = 1.0`, but the citations point at chunks that do not support the claims.
**Cause:** validating labels is a string check; supporting a claim requires entailment.
**Mitigation:** measure both. `citation_validity` is a deterministic string check; `citation_support` requires an LLM judge per citation and is the metric that actually correlates with user trust.

### 6.12 Evaluation of the evaluation

**Symptom:** the harness itself has bugs — it retrieves with different parameters than production, uses a different prompt, or scores against a stale index.
**Mitigation:** run the harness against production configuration through a shared config object, not a duplicated one; assert that the harness's retrieval parameters equal production's; include a smoke test with a known-answer item whose score must be 1.0.

---

## 7. When to use / When NOT to use

### When to invest in each layer

| Layer | Invest when | Skip when |
|---|---|---|
| Retrieval metrics (deterministic) | **Always** — near-zero cost, immediate value | Never skip |
| Golden set | Always, even at 50 items to start | Never skip; grow it continuously |
| LLM-as-judge generation metrics | You are changing prompts/models, or quality is the binding constraint | Retrieval is clearly the bottleneck and unfixed |
| Full per-request tracing | You have production users and complaints to debug | Pre-production prototype |
| Sampled live judging | Traffic > ~10k queries/day | Low traffic where the golden set is sufficient |
| Drift detection | Any hosted model/embedding dependency | Fully self-hosted with pinned versions (reduced need, not zero) |
| A/B testing | Volume is high enough for significance | Low traffic — use offline paired comparisons instead |
| User feedback loops | Always, but do not over-trust them | Never skip; just weight them correctly |
| Synthetic test generation | You need coverage fast and cheaply | You need realism — blend, never substitute |

### Decision table for "how do I know if my RAG system is good?"

| Question | Metric | Gate |
|---|---|---|
| Is the evidence being found? | Recall@10 | ≥ 0.85 |
| Is the evidence ranked well? | MRR, nDCG@10 | ≥ 0.70 |
| Is the context relevant, not just present? | Context precision | ≥ 0.60 |
| Is the answer grounded? | Faithfulness | ≥ 0.80 |
| Is the answer on-topic? | Answer relevance | ≥ 0.75 |
| Does it know when it doesn't know? | Abstention correctness | ≥ 0.85 |
| Are the citations real? | Citation validity | ≥ 0.995 |
| Do the citations mean anything? | Citation support | ≥ 0.90 |
| Is it fast enough? | p95 end-to-end | < 2.5 s |
| Is it affordable? | Cost per query | < $0.02 |
| Is it getting worse? | 7-day trend on all of the above | No drop > noise band |

### When NOT to gate on a metric

- Do not gate on a metric whose confidence interval is wider than the effect you are trying to detect (small golden set).
- Do not gate on a metric the judge cannot measure reliably — validate the judge first.
- Do not gate a warning-level metric as a blocker; teams route around blockers and disable them.
- Do not gate on end-to-end latency p50; p95/p99 is where the user experience lives.

---

## 8. Numbers, defaults & tuning

### 8.1 Golden set defaults

| Parameter | Default | Range | Why |
|---|---|---|---|
| Total size | 200–500 | 150 min, 2,000 max | 150 = noise floor; > 2,000 rarely changes decisions |
| Simple factoid share | 40% | 35–50% | Matches production traffic |
| Multi-hop share | 25% | 20–30% | Where failure concentrates |
| Unanswerable share | 15% | 10–20% | Only way to test abstention |
| Edge cases (tables/code/acronyms/non-English) | 10% | 5–15% | Where chunking earns its keep |
| Adversarial share | 10% | 5–15% | Guardrail coverage |
| Human-authored share | 30–40% | ≥ 30% | Avoids synthetic bias |
| Growth rate | +20–50 items/quarter | — | Keeps the set aligned with the product |

### 8.2 Metric gates (copy into your CI)

| Metric | Floor | Max drop from baseline | Severity |
|---|---|---|---|
| Recall@10 | 0.85 | 0.05 | block |
| MRR | 0.70 | 0.05 | block |
| nDCG@10 | 0.70 | 0.05 | block |
| Precision@5 | 0.60 | 0.05 | warn |
| Faithfulness | 0.80 | 0.03 | block |
| Answer relevance | 0.75 | 0.03 | block |
| Context precision | 0.60 | 0.05 | warn |
| Context recall | 0.70 | 0.05 | warn |
| Abstention correctness | 0.85 | 0.05 | block |
| Citation validity | 0.995 | — | block |
| Hallucination rate | ≤ 0.05 | +0.02 | block |
| p95 latency | ≤ 2,500 ms | +20% | warn |
| Cost per query | ≤ $0.02 | +25% | warn |
| Error rate | ≤ 0.01 | — | block |

### 8.3 Metric interpretation table — what a bad number implicates

| Symptom | Likely cause | Fix |
|---|---|---|
| Low recall@10 | Chunking, embedding model, index recall, ACL filter | Re-chunk, better embedding model, raise `ef_search`, audit filters |
| Good recall@50, poor recall@10 | Ranking/fusion | Add or tune reranking; tune RRF k |
| Good recall, poor MRR | The right chunk is found but ranked low | Reranker — this is the textbook signal |
| Good retrieval, poor faithfulness | Prompt, temperature, context budget | Groundedness prompt, lower temperature, dedupe context |
| Good faithfulness, poor answer relevance | Answer drifts off-topic | Prompt, model, context over-loading |
| High abstention on answerable questions | Similarity floor too high, retrieval miss | Lower the floor, fix retrieval |
| Low abstention on unanswerable | No abstention instruction, no floor | Explicit abstention prompt + floor |
| High latency, good quality | Rerank depth, context size, sequential stages | Reduce K, trim context, parallelize |
| Good quality, high cost | Context bloat, no caching, big model | Context budget, semantic cache, model tiering |
| Steady quality decline | Drift — provider model, corpus, or query mix | Nightly eval, pinned versions, drift alerts |

### 8.4 Operational metric thresholds and alerts

```yaml
alerts:
  - name: retrieval_recall_degraded
    condition: nightly_recall_at_10 < 0.85 or drop_from_baseline > 0.05
    for: 1 run
    severity: critical
    action: page_oncall, block_deploys

  - name: faithfulness_degraded
    condition: nightly_faithfulness < 0.80 or drop_from_baseline > 0.03
    for: 1 run
    severity: critical
    action: page_oncall

  - name: retrieval_latency_p99
    condition: retrieve_p99_ms > 500
    for: 10m
    severity: warning
  - name: retrieval_latency_p99_critical
    condition: retrieve_p99_ms > 2000
    for: 5m
    severity: critical

  - name: ann_recall_low
    condition: bench_recall < 0.90
    for: 30m
    severity: warning
    note: HNSW ef_search likely too low or index degraded

  - name: cache_hit_rate_low
    condition: semantic_cache_hit_rate < 0.15
    for: 1h
    severity: warning
    note: cache is not earning its complexity

  - name: cost_spike
    condition: hourly_cost > 2 * trailing_7d_mean_hourly_cost
    for: 1h
    severity: warning

  - name: daily_budget_80
    condition: daily_cost > 0.8 * daily_budget
    for: 15m
    severity: warning
  - name: daily_budget_100
    condition: daily_cost > daily_budget
    for: 5m
    severity: critical
    action: enable_hard_caps

  - name: hallucination_rate_high
    condition: sampled_hallucination_rate > 0.05
    for: 24h
    severity: warning

  - name: abstention_rate_anomaly
    condition: abs(abstention_rate - trailing_7d_mean) > 0.10
    for: 6h
    severity: warning
    note: either retrieval broke or the prompt changed

  - name: ingestion_lag
    condition: p95_source_change_to_searchable > 15m
    for: 30m
    severity: warning

  - name: ingestion_failures
    condition: ingestion_failed_docs_24h > 100
    for: 1h
    severity: warning

  - name: judge_frozen_items_drift
    condition: frozen_item_score_delta > 0.03
    for: 1 run
    severity: critical
    note: the JUDGE moved, not the system — invalidates comparisons
```

### 8.5 Cost and cadence defaults

| Activity | Frequency | Items | Judge | Cost/run |
|---|---|---|---|---|
| Retrieval-only CI check | Every commit | 200 | none | ~$0 |
| Full golden-set CI gate | Every prompt/model PR | 200 | small | ~$1 |
| Nightly regression | Nightly | 200 (cached triples) | small | ~$0.20–1.00 |
| Weekly deep dive | Weekly | 500 + per-category | mid | ~$5–15 |
| Live sampling | Continuous | 1–5% of traffic | small | ~$0.0001/query |
| Judge calibration | Quarterly | 50–100 human-labelled | frontier | Time + ~$5 |
| Adversarial red team | Quarterly | 50–100 crafted | frontier | Time |

---

## 9. Interview Q&A

**Q1. How do you evaluate a RAG system?**
Three layers. Retrieval: recall@k, MRR, nDCG@k, precision@k against a golden set, all deterministic and cheap. Generation: faithfulness, answer relevance, context precision and recall, abstention correctness, citation validity and support — RAGAS-style with LLM-as-judge. Operations: p95 latency, cost per query, cache hit rate, error rate, ingestion lag. Offline gates the deploy through CI; online validates it through sampled judging, implicit feedback, and drift detection. The single most important design choice is measuring retrieval and generation *separately*, because they fail independently and the fix is completely different.

**Q2. What's in your golden set?**
About 200 to 500 items, composed roughly 40% simple factoid, 25% multi-hop, 15% unanswerable so the abstention policy is actually exercised, 10% edge cases like tables and code and acronyms, and 10% adversarial. At least 30–40% must be human-authored, because synthetically generated questions — chunk in, question out — embed too close to their own gold chunks and make the set artificially easy. Sourcing is roughly a third hand-curated, a third mined from production with verified gold answers, a third synthetic but blended. It lives in git, it's versioned, and I grow it by 20–50 items per quarter, with production failures being the best source of new items.

**Q3. Why is recall@10 your headline retrieval metric?**
Because it is the gate on everything downstream. If the gold chunk is not in the top 10, no generator, prompt, or reranker can fix it — the information simply is not in the prompt. Recall@10 also has the useful property of being diagnostic by comparison: if recall@50 is 0.95 and recall@10 is 0.80, your retriever is fine and your *ranking* is the problem, which points at the reranker. I gate CI at recall@10 ≥ 0.85 with a 5-point max drop from baseline.

**Q4. What is faithfulness and how do you measure it?**
Faithfulness is the fraction of atomic factual claims in the answer that are supported by the retrieved context. You decompose the answer into claims and check each one for entailment against the context — either with an NLI model or an LLM judge. Three labels: entailed, contradicted, unsupported, and a contradiction is treated as a hard failure because it's worse than a gap — it's actively wrong. I gate at 0.80 with a 3-point drop budget, and I derive the hallucination rate from it — the fraction of answers containing at least one unsupported claim — because a rate reads better to stakeholders than a mean.

**Q5. What are the biases in LLM-as-judge and how do you handle them?**
Position bias — the judge prefers whichever candidate is first, a 5–15% swing — mitigated by randomizing or swapping order and averaging, or scoring pairwise both ways. Verbosity bias — longer scores higher, 10–20% — mitigated by instructing the judge to ignore length and by analysing length as a covariate. Self-preference — a judge favours its own model family by 5–15% — mitigated by using a different family for the judge than the generator. Score clustering — everything gets 4/5 — mitigated by requiring a rationale *before* the score and preferring binary or ternary labels for claim-level checks. And I calibrate against human labels on 50–100 items quarterly, reporting Cohen's kappa, targeting above 0.6.

**Q6. How do you catch silent quality degradation?**
Nightly automated eval on the golden set with **drop-based** gates, not just absolute floors — an absolute floor of 0.85 lets a slow slide 0.86 → 0.86 → 0.85 through, whereas a max-drop-from-baseline of 0.03 catches the first commit that caused it. Plus frozen reference items with known human labels in every run, so I can distinguish *system* drift from *judge* drift: if the judge's score on frozen items moves, my measurement instrument moved, not my system. Plus pinned model and embedding versions on everything, drift detectors for query clusters and corpus shape, and canary deploys at 5% with per-class metric comparison before full rollout.

**Q7. Offline vs online evaluation — which matters more?**
Neither alone; they answer different questions. Offline gives you ground truth, immediate signal, and the ability to gate a deploy — but the golden set is not the real query distribution. Online gives you the real distribution and real failure modes — but no ground truth, it is slow, and it costs. The pattern I use is that offline gates the deploy and online validates it: CI blocks on golden-set regression, then the change canaries at 5% of production for 24–48 hours with per-class metrics compared against the control before full rollout.

**Q8. How much does evaluation cost?**
Less than people fear, if you design it right. A small judge model runs about $0.002 per RAG triad, so the full 200-item golden set costs around a dollar, and a mid-tier judge costs around nine. Nightly runs are nearly free because most triples are unchanged — cache judge results by (question, context, answer) hash. Live sampling at 1–5% adds roughly $0.0001 per query blended. The real cost is engineering: a serious harness is roughly half an FTE. But compare that to generation cost — eval is a rounding error against the model bill, and it is the only thing that prevents a 4x cost regression from shipping unnoticed.

**Q9. How do you build a golden set from scratch?**
Start with 50 items in a week rather than 500 in a quarter. Mine the top 200 production queries by frequency, have a domain expert verify gold answers for 30 of them. Hand-write 10 unanswerable questions — this category is always missing and always matters. Take 10 documents, have a strong model generate a question per document, and have a human validate them. That gives you a usable 50-item set with all five categories represented. Then grow it: every production failure becomes a golden-set item, and you add 20–50 per quarter. Version it from day one, because metric history is worthless if the set changed underneath it.

**Q10. A user complains that an answer was wrong. How do you debug it?**
Retrieve the `query_id` and replay the trace. Stage 1: was the gold chunk in the top-50 candidates at all? If not, it's a retrieval miss — check chunking, the embedding model, ACL filters, and whether the document was ever ingested. Stage 2: if it was in the top-50 but not in the top-5, it's a ranking problem — the reranker or the fusion. Stage 3: if it was in the top-5, check the context assembly span — was it dropped by the token budget or deduped? Stage 4: if it reached the prompt and the answer is still wrong, it's a generation problem — check the prompt, temperature, and whether the model followed an instruction embedded in a retrieved document. Only then do I look at changing the model.

**Q11 (pushback). Aren't you just optimizing numbers that users don't care about?**
That is the real risk, and the defence is to keep the metrics tethered to user outcomes. I instrument implicit signals on 100% of traffic — the strongest being a rephrase within 30 seconds, which means the first answer failed — plus escalation rate and session abandonment. I review the top 20 thumbs-down queries weekly and check whether they resemble the golden set; if they don't, the golden set is drifting and the metrics are optimizing the wrong distribution. And any change that improves an offline metric but worsens an online signal gets reverted, not celebrated.

**Q12. How do you test the abstention behaviour?**
15% of the golden set is unanswerable questions with a required abstention, and I measure `abstention_correctness` — the fraction of those where the system correctly refused — gated at 0.85. That is the only way the abstention policy gets exercised at all, and it's the most commonly missing category in golden sets. I pair it with a retrieval-side floor: if no retrieved chunk clears a similarity threshold — typically 0.5 cosine — the system short-circuits to "I don't have information about that" without even calling the generator. And I alert on abstention-rate anomalies in both directions: a sudden rise means retrieval broke, a sudden fall means the prompt or the floor changed.

**Q13. What would you monitor on a RAG dashboard?**
Four layers. Quality: nightly recall@10, faithfulness, hallucination rate, abstention correctness, all trended against a baseline band. Latency: per-stage p50/p95/p99 — cache, rewrite, retrieve, rerank, TTFT, generate — because stage-level latency localizes the problem. Cost: cost per query, daily spend against budget, cache hit rate, tokens in/out, cost broken down by stage. Health: error rate, rate-limit hits, ingestion lag, ingestion failures, index size, ANN benchmark recall. Plus a safety row: guardrail trigger rate, injection detections, PII redactions. And a one-line "judge frozen items" panel, because if that moves, every quality number above it is invalid.

**Q14. How do you A/B test a prompt change?**
Only when traffic supports significance — a few thousand queries per arm minimum for a 3-point faithfulness difference. I split by `user_id` hash, not by request, so a user gets a consistent experience. I hold retrieval constant so the experiment isolates the prompt. I measure both offline (the golden set, pre-deploy) and online (thumbs, rephrase rate, escalation rate, and sampled judging). I run for at least a week to cover weekly seasonality, and I pre-register the primary metric so I don't cherry-pick afterward. If the offline metric improves but the online rephrase rate worsens, the offline metric is wrong and I investigate that before shipping.

**Q15. What's the biggest mistake you see in RAG evaluation?**
Tuning the generator when the retriever is broken. Teams change prompts, swap models, adjust temperature — and never measure recall@10. If recall@10 is 0.72, the generator literally does not have the answer in its prompt for 28% of queries, and no prompt engineering fixes that. The second biggest is not having an unanswerable class, so the abstention behaviour is completely untested and the system confidently hallucinates on questions outside the corpus. The third is trusting LLM-judge numbers without calibration — a judge with unmeasured bias produces a metric that looks precise and is not.

**Q16. Distinguish context precision from context recall.**
They answer two different questions about the same retrieved set, and the transcript's one-line versions are the ones to use: **context precision is about ranking — did the useful chunks come back high? Context recall is about coverage — did enough of the needed information come back at all?** The formulas are `relevant retrieved chunks ÷ all retrieved chunks` for precision, and `reference claims supported by context ÷ total claims in the reference` for recall. The worked signature to have ready: retrieval returns five chunks and the genuinely useful one is sitting at rank 3 or 4. Recall may be fine — the evidence *was* retrieved — but precision suffers, and that points squarely at the reranker rather than at the embedding model. Read them together, because a high recall with low precision means you are carrying dead weight into the context window, and a low recall means no amount of reranking will help.

**Q17. How would you evaluate a system whose output is code rather than prose?**
You build a `{input, reference artifact}` golden set rather than a Q&A list, and you grade by comparing the produced artefact to the reference. The transcript's worked case is an agentic system that turns "build me a 10-million-customer order system on AWS serverless" into Terraform: roughly 2,000–3,000 question-and-reference pairs where the reference is the hand-written Terraform, the same question is passed to the agent, and the generated code is graded against it — a close generation is `A'` and grades well, an unrelated one is `B` and grades badly. The important design choice is that the grade should be **deterministic where it can be** — structural comparison, does the plan apply, does it match the intended resource graph — and LLM-judged only for the parts that are genuinely semantic. Faithfulness-style metrics still apply to any prose explanation attached to the code, and the regression gate runs the same comparison in CI. The scale is also different: a few thousand items, not a few hundred, because you need coverage of artefact shapes rather than of question categories.

**Q18. Observability, evaluation, regression testing — are those not the same thing?**
No, and conflating them is a design error. They are three jobs and each tool does exactly one. **Observability** answers "what exactly happened?" — per-request traces of every pipeline and tool call with timings, scores, tokens and cost; Langfuse is the named example. **Evaluation** answers "is my RAG system retrieving and answering correctly?" — faithfulness, context precision, context recall, answer relevance; RAGAS is the named example, and the crucial limitation is that it has **no visibility into what happened during execution, only the final output**. **Regression testing** answers "did my change break something?" — the same golden set run in CI to gate a deploy; that is DeepEval's job. The practical consequence: a bad RAGAS score tells you quality fell and tells you nothing about where, so you open the trace to find the stage. Ship only one of the three and you have a system that is either undiagnosable, unmeasured, or free to regress silently.

### Pushback round

**P1. Isn't "hallucination rate" too blunt a metric to be worth reporting?**
It is blunt if you report it alone, and it is exactly the right headline if you pair it with the taxonomy underneath it. The reason is that hallucination is a *user-facing label* for a family of failures that mostly originate before generation — missing chunk, low recall, semantic mismatch, keyword mismatch, ambiguous query, bad chunking, context ignore, context confusion, citation mismatch. Seven of those ten are retrieval-side. So I report hallucination rate as the outcome metric for stakeholders — a rate reads better than a mean — and I break the same failures down by origin stage for engineering: context recall for the missing-evidence class, faithfulness-with-contradiction-label for the context-confusion class, and citation support for the attribution class. The blunt number is what tells you whether to care; the taxonomy is what tells you what to fix.

**P2. Your golden set is 200–500 items — isn't that too small to trust any of these numbers?**
For a 5-point change it is enough, and it is deliberately not bigger, because the alternative is a set nobody maintains. The mathematics is that below roughly 150 items metric noise exceeds the effect size of most changes you will make; at 200 you can see a 5-point shift, and at 500 you get per-category signal. What matters more than size is *composition and freshness*: 15% unanswerable so abstention is exercised at all, at least 30–40% human-authored so the set is not biased toward its own synthetic questions, and growth driven by production failures — every real failure becomes an item. And I would not read a 1–2 point offline move as signal at all; that is what the drop-based gates and the online canary are for. If you genuinely need finer resolution than 200 items gives you, you do not need a bigger golden set, you need a paired comparison on the same items, which cancels most of the noise.

---

## 10. Common mistakes candidates make

| Mistake | Why it is wrong | Do this instead |
|---|---|---|
| Reporting one number for "RAG quality" | Retrieval and generation fail independently | Report the RAG Triad: context relevance, groundedness, answer relevance |
| No golden set | Every fix becomes a guess; regressions are invisible | Build 50 items in a week, grow to 200–500 |
| Golden set with no unanswerable items | Abstention is never tested; the system hallucinates off-corpus | **15% unanswerable with required abstention** |
| All-synthetic golden set | Questions embed too close to their gold chunks; the set is too easy | ≥ 30–40% human-authored, blended sourcing |
| Absolute floors only | A slow slide passes silently | Drop-based gates (`max_drop_from_baseline`) |
| Using the same model family as judge and generator | Self-preference bias inflates scores | Different family for the judge |
| Quoting LLM-judge numbers without calibration | An uncalibrated judge is a biased instrument | Human calibration quarterly, report kappa |
| Forgetting judge-model version pinning | A provider update moves your metric definition | Pin, version, and include frozen reference items |
| No tracing | Complaints cannot be debugged, only guessed at | Per-request trace with spans per stage |
| Tracing full text into a third-party store with PII | Creates a compliance problem in your observability layer | Redact before export; short retention for raw text |
| Trusting thumbs as ground truth | 2–5% response rate, biased to extremes | Pair with implicit signals (rephrase-within-30s, escalation) |
| Not measuring citation *support*, only validity | Labels resolve but point at irrelevant chunks | Both: string check + entailment check |
| A/B testing on low traffic | Cannot reach significance; results are noise | Offline paired comparisons on the golden set |
| Ignoring eval cost | A frontier judge on 5,000 items nightly is $300/night | 200–500 nightly with cached triples, cheapest calibrated judge |
| Optimizing nDCG while user satisfaction falls | Metric drift from real distribution | Review top thumbs-down weekly; rebalance the golden set |
| Calling recall, precision, MRR and nDCG one thing | "Rank evaluation" is four metrics pointing at four different stages | Separate them: recall = did we fetch it, precision = how much of what we fetched was useful, MRR = how high was the first correct hit, nDCG = were relevant results near the top |
| Treating observability, evaluation and regression as one purchase | Each tool does exactly one job; RAGAS cannot see inside the run | Langfuse to observe, RAGAS to evaluate the output, DeepEval to gate the deploy |
| Citing the answer instead of the sentence | One correct citation can hide three unsupported sentences | Per-sentence claim–evidence map with source completeness (no claim left uncovered) |
| Assuming a golden set must be Q&A | For code or structured output the reference is the artefact | `{input, reference artifact}` pairs, graded by similarity — deterministic where possible |

---

## 11. Cheat-sheet summary

| Dimension | Answer |
|---|---|
| **RAG Triad** | Context relevance, groundedness, answer relevance — measure all three |
| **Retrieval metrics** | Recall@k, Precision@k, MRR, nDCG@k, hit rate |
| **Generation metrics** | Faithfulness, answer relevance, context precision/recall, hallucination rate, abstention correctness, citation validity + support |
| **Headline gates** | Recall@10 ≥ 0.85 (−5 pts), Faithfulness ≥ 0.80 (−3 pts), Citation validity ≥ 0.995 |
| **Golden set size** | 200–500 items (150 = noise floor) |
| **Golden set mix** | 40% factoid / 25% multi-hop / 15% unanswerable / 10% edge / 10% adversarial |
| **Human-authored share** | ≥ 30–40% — synthetic-only sets are biased toward their own answers |
| **Growth** | +20–50 items/quarter; every production failure becomes an item |
| **Judge cost** | ~$0.002/triad (small) vs ~$0.018 (mid) — use the cheapest that calibrates |
| **Judge biases** | Position 5–15%, verbosity 10–20%, self-preference 5–15%, clustering |
| **Judge mitigations** | Randomize order, ignore length, different model family, rationale before score, quarterly human calibration (kappa > 0.6) |
| **Gate design** | Drop-based beats absolute floors — catches slow slides at the causal commit |
| **Frozen items** | Include human-labelled frozen items every run to separate system drift from judge drift |
| **Tracing** | Spans per stage keyed by `query_id` — replayable, the core debugging asset |
| **Strongest implicit signal** | Rephrase within 30 s = the first answer failed |
| **Offline vs online** | Offline gates the deploy; online validates it (5% canary, per-class metrics, 24–48 h) |
| **Alert thresholds** | retrieve p99 > 500 ms warn / > 2000 ms critical; ANN bench recall < 0.90 warn; cache hit < 0.15 warn; hourly cost > 2 × trailing mean |
| **Latency targets** | p95 end-to-end < 2 s (gate ≤ 2.5 s) |
| **Cost target** | < $0.02/query |
| **Cadences** | Per commit: retrieval only. Nightly: full + drift. Weekly: 500 + per-category. Quarterly: judge calibration + golden-set refresh + red team |
| **Diagnosis table** | Low recall → retrieval. Good recall@50/poor @10 → ranking. Good retrieval/poor faithfulness → prompt. |
| **Biggest mistake** | Tuning the generator when recall@10 is 0.72 |
| **Silent killer** | Quality degradation with zero alert volume → only a scheduled eval detects it |
| **Hallucination taxonomy** `[S]` | Ten named causes in three groups: ingestion (bad/wrong chunking), retrieval (missing chunk, low recall, semantic mismatch, keyword mismatch, ambiguous query), generation (context ignore, context confusion), attribution (citation mismatch). **Seven of ten are retrieval-side** |
| **Defence stack order** `[S]` | Better retrieval → strict prompt ("answer only from the context", abstain) → require citations, refuse if unsupported → grounding check per claim → faithfulness score on every answer |
| **Context precision vs recall** `[S]` | Precision = ranking (useful chunks high): `relevant retrieved ÷ all retrieved`. Recall = coverage (enough returned): `claims supported ÷ total claims` |
| **Four-part framework** `[S]` | Generation (faithfulness, correctness, relevance) → Retrieval (context precision/recall) → Agent (tool calls, trajectory, task completion) → Production (latency, cost, tokens, error rate) |
| **Three eval modes** `[S]` | Deterministic (equality/structural) · LLM-as-judge (question + answer + context → score + reason) · Human in the loop (approve / reject / reason) |
| **Three jobs** `[S]` | Observe = *what happened* → **Langfuse** · Evaluate = *is it right* → **RAGAS** (output only, no visibility into the run) · Regress = *did I break it* → **DeepEval** |
| **Observability captures** `[S]` | Timings, scores, tokens, cost — per stage |
| **Citation artefact** `[S]` | Per-sentence claim–evidence map (paragraph, section, table) · citation correctness · source completeness (all claims covered) · flag answered claims |
| **Golden set as artefact pairs** `[S]` | For code/structured output: `{input, reference artifact}` — Pinpool used ~2,000–3,000 `{question, reference Terraform}` pairs, graded `A'` (close) vs `B` (unrelated), then deployed the winning prompt |

---

## 12. Cross-links

- [Production RAG System](01-production-rag-system.md) — the pipeline these metrics instrument
- [Advanced RAG Patterns](02-advanced-rag-patterns.md) — the golden set decides which patterns to adopt
- [Guardrails and Safety for AI Systems](04-guardrails-and-safety-for-ai-systems.md) — safety metrics and adversarial items
- [Agentic System Design](06-agentic-system-design.md) — trajectory evaluation, task success rate, cost per task
- [Memory and Context Engineering](08-memory-and-context-engineering.md) — context budget as a measured variable
- [Fine-Tuning vs RAG vs Prompting](09-fine-tuning-vs-rag-vs-prompting.md) — evaluation requirements differ per approach
- [Deployment, Release and CI/CD](../../Operations/04-deployment-release-and-cicd.md) — eval-gated pipelines and canary rollout
- [LLMOps for AI Systems](../../Operations/10-llmops-for-ai-systems.md) — the continuous monitoring loop
- [LLMOps and Operationalisation Questions](../../Interview-Questions/06-llmops-and-operationalisation-questions.md)
- [RAG and LLM Cheatsheet](../../Cheat-Sheets/04-rag-and-llm-cheatsheet.md)
