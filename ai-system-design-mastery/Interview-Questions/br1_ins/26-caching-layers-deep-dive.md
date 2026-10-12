# Caching Layers Deep Dive — KV, Prefix and Semantic Cache

> **Provenance.** Harvested in full from `extracted/questions_rough/_extracted_txt/caching` on **2026-10-12**. The source is two documents in one: (i) a three-layer comparison of **KV cache / semantic cache / prefix (prompt) cache** with its citation list, and (ii) a long article from the series *Understanding AI: Beyond the Buzzwords* (Part 6, *KV vs Semantic Cache*) that builds the **semantic-cache risk case** on a librarian analogy. Every distinct idea from both is placed below.
>
> **What this adds that the rest of the pack doesn't.** `05` §5.5 names caching as one of five cost levers in a single table row; `10` §18 gives prompt caching one line; `22` has it in the cheat sheet. This file is the **only place** the pack separates the three mechanisms properly, states *why* each one fails, and gives a decision procedure for when a reuse is safe. It is the file you reach for when the panel says *"tell me more about the caching layer"* and you have four minutes, not four seconds.
>
> **Where it sits.** The long-form companion to [`05`](05-system-design-round-playbook.md) §5.5 (Cost control), [`05`](05-system-design-round-playbook.md) §5.4 (Serving & inference) and [`15`](15-rag-architecture-deep-dive.md) §9 (Production RAG challenges).
>
> **Vocabulary note.** This file uses the pack's four dials as **Quality · Cost · Latency · Safety**. The harvested deep-dive set names the same four as **Accuracy · Latency · Cost · Reliability** — the fourth is the same dial seen from the reliability/safety/auditability angle. Reconciliation table in [`24`](24-stage1-strategy-deep-dive.md) § banner.

---

## Table of Contents

1. [The one-sentence distinction](#1-the-one-sentence-distinction)
2. [Layer 1 — the KV cache](#2-layer-1--the-kv-cache)
3. [Layer 2 — the prefix (prompt) cache](#3-layer-2--the-prefix-prompt-cache)
4. [Layer 3 — the semantic cache](#4-layer-3--the-semantic-cache)
5. [Semantic caching's seven failure modes](#5-semantic-cachings-seven-failure-modes)
6. [Where it works and where it must not be used](#6-where-it-works-and-where-it-must-not-be-used)
7. [Running both layers together](#7-running-both-layers-together)
8. [Using it in an interview](#8-using-it-in-an-interview)
9. [Measuring it properly](#9-measuring-it-properly)
10. [Brit translation](#10-brit-translation)
11. [Interview Q&A](#11-interview-qa)
12. [Cheat-sheet](#12-cheat-sheet)
13. [Cross-links](#13-cross-links)

---

## 1. The one-sentence distinction

The source hammers one sentence, and it is the sentence to say in the room:

> **KV caching makes inference more efficient. Semantic caching attempts to avoid inference.**

Everything else in this file is a consequence of that split. One mechanism makes a model call cheaper; the other tries not to make it at all. They operate at different layers, they are owned by different teams, they fail in different ways, and they belong in different parts of the architecture.

> **The line to say:** *"There are three caches with the word 'cache' in them, and only one of them is a decision. The KV cache and the prefix cache are inference optimisations — they make the model call cheaper. The semantic cache is an application policy — it decides not to make the model call at all. One of those is a performance topic. The other is a correctness topic, and that is the one I'd want a named owner for."*

### 1.1 The three-layer overview

| Feature | **KV cache** | **Semantic cache** | **Prefix cache** (prompt cache) |
|---|---|---|---|
| **What it stores** | Key and Value attention tensors for a sequence | Finished text response strings (plus the metadata that makes them reusable) | Reusable KV cache tensors from a shared prompt prefix |
| **Lookup method** | Exact active sequence token progression | Cosine similarity over prompt embeddings | Hash chain / exact token-ID match from the start of a prompt |
| **Where it operates** | During the token generation (**decode**) phase of a **single request** | **Outside / before** the LLM; returns the stored response directly | During the initial prompt processing (**prefill**) phase, **across requests** |
| **Bypasses the model?** | **No** — helps generate tokens faster | **Yes** — skips the model entirely on a hit | **No** — runs the model only on the remaining new tokens |
| **Lifetime** | One request (or one served sequence) | Days to months (an application data store) | Provider-side, minutes to hours, policy-defined |
| **Owned by** | The inference engine (vLLM, TGI, TensorRT-LLM) | The application / gateway team | The provider (or self-hosted engine) |
| **Failure mode** | Memory exhaustion; VRAM is what runs out | **Confidently wrong, stale, private or irrelevant answers — fast** | Silent: prefix mismatch means you simply pay full price |
| **Main dial it buys** | Cost + Latency | Cost + Latency (at the expense of **Quality** and **Safety**) | Cost + Latency, with no correctness risk |
| **Main dial it costs** | Cost (memory) | Quality, Safety, Auditability | Prompt-engineering discipline |

### 1.2 The two questions that decide everything

The source's own closing framing is worth keeping as a diagnostic. Before choosing any of the three, ask:

| Question | Why it changes the answer |
|---|---|
| **What kind of workload is this?** (chat, RAG pipeline, code generation, classification, extraction) | Repetition-heavy chat and classification workloads have high reuse; open-ended generation has almost none |
| **Is the primary goal low latency (TTFT) or lower GPU memory / lower cost?** | TTFT is a prefix-cache and streaming question; GPU memory is a KV-cache and quantisation question; cost is a semantic-cache and tiering question |

> **The line to say:** *"Caching is not one decision. It's three, and they're made by three different people. So before I say 'we'll add a cache', I'd want to know whether we're buying throughput, buying TTFT, or buying a lower bill — because those point at different mechanisms."*

### 1.3 Where each lives in the architecture

```
Client
  │
  ▼
API Gateway / AI Gateway            ← semantic cache usually lives HERE (application layer)
  │   ├─ exact response cache (normalised key)
  │   └─ semantic cache (embedding + cosine ≥ threshold)  ← SKIPS the model
  │
  ▼
RAG Module (vector DB + BM25 + reranker)
  │   └─ embedding cache (hash content → skip re-embed)   ← ingestion-side sibling
  │
  ▼
LLM Service / inference engine       ← prefix cache and KV cache live HERE (model layer)
  │   ├─ prefix cache: reuse KV blocks for a shared token prefix
  │   └─ KV cache: reuse K/V tensors for the current sequence
  │
  ▼
Response
```

> **One cache lives close to model inference. The other usually lives in the application, gateway or platform layer.** The KV cache contains intermediate model state; the semantic cache contains reusable *application results*. That single sentence explains why the two have such different risk profiles.

---

## 2. Layer 1 — the KV cache

### 2.1 What it is

A **runtime memory store inside GPU VRAM** holding the attention **keys and values** for tokens that have already been processed by the current sequence.

Every transformer attention layer, for every token it processes, produces a Key vector and a Value vector. Once computed, those vectors are needed again the next time the sequence is attended to. The KV cache keeps them instead of throwing them away.

### 2.2 Why it exists

During **autoregressive generation** — the decode phase — the model produces **one token at a time**. Token *n* depends on tokens 1…*n*−1, so the attention computation for the whole prefix has to be available at every step.

Without a cache, the engine would recompute the attention matrices for **all previous tokens** at every single step: an O(n²) recomputation over a sequence that is itself growing. With the cache, it reads the saved K/V pairs from memory and only computes the maths for the **new** token.

> *"The KV cache is not an optimisation you bolt on. It is what makes autoregressive decoding tractable at all. Remove it and the cost of generating token 500 is roughly what it would be for the whole sequence, every time."*

### 2.3 The librarian analogy (from the source)

The source's librarian analogy, which is worth keeping because it makes the *scope* obvious:

> A customer asks the librarian to read a long company policy and answer a question. While reading, the librarian **creates notes** about the material. When writing the answer, those notes prevent the librarian from re-reading the entire document before producing every new word.
>
> **Those temporary notes represent the KV cache.** They exist *inside* the active reading and writing process. They are tied to the exact material the librarian processed. Their purpose is to make **continuation** faster.

The two things to notice: the notes are **temporary** (they die with the task) and they are **reusable only for this exact material**. That is precisely the KV cache's scope — the lifetime of a single request.

### 2.4 Scope

> **Exists temporarily in memory for the lifetime of a single request.**

It is not a database. It is not shared between users. It is not even necessarily shared between two requests from the same user, unless the serving engine's prefix caching (Layer 2) explicitly arranges for that. When the request finishes, the memory is reclaimable.

### 2.5 The prefill / decode split

The KV cache is built during **prefill** and consumed during **decode**. That split is where the whole cost model comes from:

| Phase | What happens | Cost character | Where the KV cache is |
|---|---|---|---|
| **Prefill** | The whole prompt is processed in one parallel pass; the KV tensors for every prompt token are computed and stored | Compute-heavy but parallel — fast and cheap per token (~$0.50/M in [`05`](05-system-design-round-playbook.md) §5.4's working numbers) | **Being written** |
| **Decode** | One token at a time, sequentially; each step attends over the cached prefix | **Memory-bandwidth bound** — the engine is reading the whole KV cache per step, not doing much maths (~$1.50/M, ~3× the input price) | **Being read, and appended to** |

This is why **output length is the expensive dimension**, why decode dominates end-to-end latency, and why the KV cache is a *memory* problem rather than a *compute* problem.

> **The line to say:** *"Prefill builds the KV cache; decode reads it. Decode is memory-bandwidth bound, so at long context the thing that runs out is not FLOPS — it is VRAM. That's why 'just use a 200K context window' is not a strategy, and why the serving-engine optimisations are all about memory management."*

### 2.6 The cost: memory, especially long context

- The KV cache grows **linearly** with sequence length, in addition to the attention compute growing quadratically.
- **Hugging Face's documentation** (cited by the source) notes that KV caches can consume **a significant amount of memory, particularly for long-context generation**.
- The pack's working figure, from [`05`](05-system-design-round-playbook.md) §8.5: a 7B fp16 model is roughly **0.5 MB per 1,000 tokens**; a 200K context is on the order of **100 GB**. A 70B-class model with GQA in BF16 is ~0.320 MB/token; with FP8 KV, ~0.160 MB/token.
- The formula: `KV_bytes/token = 2 × layers × KV_heads × head_dim × dtype_bytes`.

### 2.7 What manages that cost

These belong to the inference engine, not the application. Full depth in [`15`](15-rag-architecture-deep-dive.md) §9 and [`05`](05-system-design-round-playbook.md) §5.4 — cross-link rather than repeat:

| Mechanism | What it does for the KV cache |
|---|---|
| **PagedAttention** (vLLM) | Stores the KV cache in **non-contiguous pages**, removing the fragmentation that wastes 60–80% of VRAM in naive contiguous pre-allocation |
| **Continuous batching** | Inserts new requests into the running batch at each decode step instead of waiting for the slowest — the biggest throughput win, and the reason KV memory is sized by *in-flight* requests, not by QPS |
| **GQA / MQA / MLA** | Reduces the number of distinct KV heads, cutting KV bytes per token by 8× (GQA) to 57× (MLA) for near-zero quality loss |
| **KV quantisation** | FP8 KV at <0.3% quality loss; halves KV footprint again |
| **Prefix caching / RadixAttention** | Turns the KV cache from a per-request structure into a cross-request one — **this is Layer 2** |

> **The sizing rule from [`05`](05-system-design-round-playbook.md) §3.1:** `KV cache = in_flight × avg_context_tokens × KV_bytes_per_token`. In the worked support-assistant example that is 21 GB — and *"this is the sizing constraint, not FLOPS."*

### 2.8 When the KV cache is the right answer

**Always, whenever autoregressive generation occurs.** There is no semantic-correctness question attached to it, because the KV cache never decides that one user's completed answer should satisfy another user. It reuses internal computation associated with tokens that have already been processed, within the request that produced them.

The trade-offs it carries are **memory, cache management and workload efficiency** — an engineering problem with engineering answers. That is the whole point of contrast with Layer 3.

> **The line to say:** *"The KV cache doesn't have a correctness problem. It has a memory problem. Every other cache in this conversation has the opposite shape."*

---

## 3. Layer 2 — the prefix (prompt) cache

### 3.1 What it is

A **server-side persistence optimisation built directly on top of the KV cache**.

When multiple requests share the **exact same starting text** — long system instructions, a shared few-shot prompt block, a common RAG document, a stable conversation history — the server keeps the computed KV tensors of that prefix in memory. A new request that matches the prefix **skips the expensive prefill computation** and loads the state directly, then computes only the remaining new tokens.

vLLM's documentation frames prefix caching exactly this way: **reusing KV-cache blocks when a new request shares the same prefix with an earlier request.** In vLLM's implementation that sharing is by **content-hash of the block**, which is why the matching is on tokens, not on meaning.

| Property | Prefix cache |
|---|---|
| Built on | The KV cache — it *is* stored KV tensors, just retained across requests |
| Matching | **Hash chain / exact token-ID match from the start of the prompt** |
| Scope | Shared across multiple requests that start with **identical token sequences** |
| Saves | The prefill cost of the shared prefix (compute and latency), plus provider billing on cache reads |
| Bypasses the model? | **No** — the model still runs, just on fewer tokens |
| Correctness risk | **Essentially none.** You cannot get a wrong answer from reusing the *state* of exactly the same tokens |

### 3.2 Pricing

> **Providers often bill cache hits at a fraction — e.g. 0.1× — of standard input token rates.**

The pack's `05` §5.5 row says the same thing in its own words: *"cache reads at ~10% of input price — 80–90% off the repeated prefix; one changed token invalidates it."* For a RAG system whose system prompt is 800 tokens and whose retrieved context is 2,000 tokens, the arithmetic on the *stable* part of the prefix is where the money is.

### 3.3 The key point: **token identity, not meaning**

This is the single most important sentence in the source's prefix-cache section, and the one most candidates miss:

> **KV caching does not ask whether two prompts mean the same thing. It operates on the model state created from specific tokens.**

The source's pair of examples makes it concrete. Take this system instruction:

```
You are a customer-support assistant. Answer only from the approved
support documentation.
```

Many requests begin with exactly these tokens, so a platform with prefix caching may reuse the KV state created while processing that shared beginning.

Now reword it — same instruction, different words:

```
Act as a support assistant and rely exclusively on authorized documentation.
```

> **For a human, the instructions are nearly equivalent. For KV-prefix reuse, they are generally different prefixes.**

Different tokens, different token IDs, different hash chain — **cache miss**. The second prompt pays full prefill on every request from then on. Nothing errors. Nothing warns. The bill just quietly goes up.

### 3.4 The consequence for prompt engineering discipline

This gives prompt engineering a new, harder constraint. It is no longer only *"is this instruction clear?"*; it is also *"is this instruction stable?"*

| Prompt-engineering practice | Why it matters for prefix reuse |
|---|---|
| **Version-control system prompts as artefacts, not strings in code** | A copy-edit by a well-meaning person is a cache invalidator and a silent cost regression |
| **Put the stable text first, the variable text last** | Prefix matching runs from the start of the prompt; any variable token near the beginning kills reuse for everything after it |
| **Do not inject a timestamp, request ID, session ID or user name into the system prompt** | One variable token at position 3 invalidates the whole prefix — the classic, self-inflicted cache miss |
| **Keep the few-shot block byte-identical across routes** | Two near-identical few-shot blocks are two prefixes, doubling the resident KV footprint for no benefit |
| **Re-measure prefill cost per prompt version** | The regression is invisible in quality metrics; it shows up only in the token bill |
| **Do not reword prompts for cosmetics in production** | Rewording is a cost change with no evaluation evidence behind it — a model-risk finding waiting to happen |

> **The line to say:** *"Prompt caching is a token-identity optimisation, not a meaning optimisation. Reword the system prompt and the reusable prefix collapses — even though any human would call the two instructions equivalent. So a prompt change in production isn't just a quality event any more; it's a cost event, and I'd want both measured."*

### 3.5 The RAG-specific version of this

The same mechanism extends to **long shared documents**. If a RAG system puts a retrieved policy document at the *front* of the prompt, and many questions about that same document follow, the prefix cache will reuse the document's KV state across requests.

But note the discipline it imposes:

- If the retrieved document is placed **before** the system prompt, or if the system prompt varies per request, the prefix breaks.
- If chunk ordering or the retrieved set varies slightly between otherwise identical questions, the prefix breaks.
- **A retrieval change is therefore also a prefix-cache change.** Tuning `top_k` or enabling a reranker can invalidate the prefix you were relying on.

> *"I'd order the prompt deliberately for reuse: immutable system instructions first, then the stable document or few-shot block, then the conversation history, then the user's variable question last. That ordering is the difference between a 10× and a 1× input bill on the stable part — and it costs nothing but discipline."*

---

## 4. Layer 3 — the semantic cache

### 4.1 The opening the source uses

Two customers contact the same company within a few minutes of each other.

The first asks:

> **"How can I reset my password?"**

The second asks:

> **"What should I do if I've forgotten my login password?"**

To a person, these are **effectively the same question**. A traditional application cache may not agree.

Traditional caching depends on **exact or carefully constructed keys**. If the incoming request matches a stored key, the application returns the cached result. If even a small part of the key changes, the request is a **cache miss**. The two password questions contain different words and produce different token sequences, so an exact cache sends both to the model — even though the same answer would probably satisfy both users.

A semantic cache attempts to recognise that similarity.

### 4.2 How it works

1. The incoming prompt is turned into a **vector embedding**.
2. The system runs a **nearest-neighbour search** (cosine similarity) over the embeddings of previously stored prompts.
3. If the best match's similarity **exceeds a threshold** — the source quotes **0.85–0.95** — the system returns the **stored response string immediately**, **without calling the LLM**.
4. If not, the request proceeds normally and (usually) the new response is written back into the cache with its embedding.

### 4.3 Scope and economics

| Property | Semantic cache |
|---|---|
| Stored | Previous requests and their **completed responses** |
| Matching | Cosine similarity over prompt embeddings; threshold e.g. 0.85–0.95 |
| Scope | **Cross-request and model-independent** |
| Tokens saved | **Both input and output tokens** — because no model call happens at all |
| Bypasses the model? | **Yes** |
| Latency saved | Most of the model latency, not just part of it |
| Secondary benefits | Reduces pressure on **provider rate limits**; improves **resilience** when the model backend is temporarily unavailable |
| Trade-off | Risk of returning a **"confidently wrong" or slightly misaligned** answer if two differently worded questions have different factual needs |

That last row is the whole of §5 below.

### 4.4 Who else calls it what

The source cites two vendor framings, and both are worth having correctly in your head because they are the sentences a vendor will use at you:

| Source | Framing | The layer it names |
|---|---|---|
| **Microsoft's AI gateway documentation** | Semantic caching as **reusing previous completions by comparing the vector proximity of incoming prompts with earlier requests** | The **application / gateway** layer |
| **vLLM** | Prefix caching as **reusing KV-cache blocks when a new request shares the same prefix with an earlier request** | The **inference engine** layer |

These are not the same feature described twice. Microsoft's is a *response* cache; vLLM's is a *computation* cache. When a vendor says "we do caching", ask which of these two sentences they mean.

### 4.5 The librarian analogy for the semantic cache

Continuing the source's analogy:

> Now imagine the librarian finishes answering the question and **files both the question and the final answer in a cabinet**.
>
> Later, another customer asks something similar. Before reading the policy again, the librarian **searches the cabinet** and discovers that an equivalent question may already have been answered. If the match is considered safe, the librarian **returns the stored answer**.
>
> **That cabinet represents the semantic cache.**

The two analogies side by side are the cleanest way to hold the distinction in your head:

| | The librarian's notes | The librarian's filing cabinet |
|---|---|---|
| Mechanism | KV cache | Semantic cache |
| Lives | Inside the active reading and writing process | Outside it, between customers |
| Reused by | The *same* task | A *different* customer |
| Contains | Intermediate working state | Completed answers |
| Safe because | The tokens are identical | **Only if a judgement call is correct** |

**The KV cache contains intermediate model state. The semantic cache contains reusable application results.** A KV-cache hit does not normally return a finished answer. A semantic-cache hit *does*.

### 4.6 The one-sentence risk

> **A semantic cache asks: "Is this new request close enough in meaning to a previous request that the old response is still suitable?"**
>
> That **"close enough"** makes semantic caching powerful. It also makes it risky. An exact cache has a clear hit condition — the key either matches or it does not. A semantic cache makes a **judgment based on similarity**.
>
> **And similarity is not equivalence.**

---

## 5. Semantic caching's seven failure modes

This is the heart of the file. Each subsection gives the worked example the source uses, the mechanism, and the mitigation.

> **Framing sentence for the whole section:** *"Every one of these is a case where the cache did exactly what it was built to do — return a fast, cheap, confident answer — and the answer was wrong, stale, private or unauditable. That's what makes semantic caching different from every other optimisation in the stack. It fails by succeeding."*

---

### 5.1 "Close enough" is not equivalence

#### The mechanism

An embedding model places semantically related questions near each other in vector space. A similarity threshold then decides that "near" means "reusable". But **near in embedding space is a statement about topic, not about the correct answer**. Two questions can share a subject and require answers that differ in facts, permissions, audit requirements and security controls.

#### Worked example A — the password pair

| Query | Correct answer | Why it differs |
|---|---|---|
| **"How can I reset my password?"** | The **self-service** flow: a link, an email, a reset token | Any user may do this to their own account |
| **"How can an administrator reset another user's password?"** | An **administrative** flow: elevated permission, an approval or dual-control step, an **audit entry** naming both parties | Different actor, different entitlement, different audit requirement |

These are semantically close. Both discuss resetting passwords. An embedding model may place them near each other. **But they may require completely different answers, permissions, audit requirements and security controls.**

If the second question is answered with the first answer, the user is told to click a self-service link that does not exist for another user's account — at best a wasted interaction, at worst an attempt to do something the system should have refused.

#### Worked example B — the subscription pair

| Query | Correct answer | Why it differs |
|---|---|---|
| **"Can I cancel my subscription?"** | "Yes — here is the cancellation flow" | A billing-process question |
| **"Can I cancel my annual subscription and receive a full refund?"** | Depends on the refund policy, the elapsed term, the terms accepted at purchase, sometimes the jurisdiction | A **contractual and financial** question, not a process question |

Again, the questions are related — **but the correct answer to the second depends on details that do not appear in the first.**

#### The principle

> **Similarity is not equivalence.**

Write it on the whiteboard if there is one. It is the single sentence that carries the whole section.

#### Mitigation

| Mitigation | How it works | Cost of it |
|---|---|---|
| **Intent classification before the cache check** | Route by intent, not by embedding proximity — a password-reset intent never matches an admin-reset intent | An extra classifier call and a taxonomy to maintain |
| **Namespace the cache by intent class** | A cache for `billing_process` questions and a cache for `billing_contract` questions are separate stores | Fewer hits; more moving parts |
| **Require an exact-match fallback path for high-impact intents** | Any question in a "privileged", "financial" or "access control" class bypasses semantic reuse entirely | Loses the saving exactly where it matters least — which is the point |
| **Never reuse across a permission boundary** | The cached entry carries the entitlement level it was produced under, and it is only reusable at or below that level | Fewer hits |
| **Log the matched cached query alongside the incoming query** | If reviewers can see the pair, they can find the near-miss classes empirically | Storage and a review process |

> **The line to say:** *"An exact cache has a binary hit condition; a semantic cache has an opinion. That opinion — 'these two questions are close enough' — is doing the work of deciding that two users deserve the same answer. I'd want that decision made by intent and entitlement, not by a cosine score."*

---

### 5.2 The threshold is a product-risk decision, not a tuning parameter

#### The mechanism

The threshold controls a **trade-off between two different kinds of failure**:

| Threshold | Effect on hits | Effect on correctness | Business consequence |
|---|---|---|---|
| **Too strict** (e.g. 0.97) | Many valid reuse opportunities become **misses** | Very little correctness risk | The application keeps calling the model and **saves little cost** |
| **Balanced** | Moderate | Moderate | Depends entirely on the domain |
| **Too loose** (e.g. 0.80) | Many hits | **Risk of an answer relevant to the general subject but wrong for the actual question** | Confidently wrong, at scale, fast |

#### The source's contrast

| Domain | Tolerable threshold behaviour | Why |
|---|---|---|
| **Music-recommendation assistant** | A **loose** threshold is fine | Returning a *similar playlist* is unlikely to cause serious harm. A slightly imperfect match is a slightly imperfect recommendation |
| **Banking assistant** | A **tight** threshold, plus routing on intent | Two questions that both mention **international transfers** may concern **fees**, **fraud**, **sanctions** or **cancellation**. Reusing one for another is a regulatory event |

> **The acceptable threshold depends on what happens when the system is wrong.**

#### Why "just tune it" is the wrong answer

This is not merely a technical tuning problem. A tuning parameter is something you set by measuring a quality metric and moving it until the metric improves. **A product-risk decision is something you set by asking what the failure looks like to a customer, an auditor and a regulator** — and then choosing the point on the curve you can defend.

The evidence for that framing:

- There is **no universal correct value**. A threshold that is conservative for a banking assistant is wasteful for a FAQ bot.
- The threshold interacts with the **embedding model**. Change the embedding model and the same numeric threshold means something different, because the geometry of the space changed. A threshold is therefore **coupled to a model version** and must be re-derived when the model changes.
- Static thresholds are known to be unreliable. [`05`](05-system-design-round-playbook.md) §8.5 records the state of the art: **vCache (Berkeley)** uses **per-embedding learned thresholds** — up to **12.5× higher hit rate and 26× lower error** than static GPTCache. The learning here is exactly that *one threshold for all queries is the wrong shape*.
- The pack's own working figure, from [`05`](05-system-design-round-playbook.md) §5.5, is **cosine ≥ 0.95** with a **20–40% hit rate** on repetition-heavy traffic — deliberately conservative, and paired with the caveat that it **must be keyed on context and invalidated on the write path**.

#### Mitigation

| Mitigation | Effect |
|---|---|
| **Set the threshold from the cost of a wrong answer, not from a hit-rate target** | The dial is chosen by the failure you can tolerate, then measured |
| **Tier the threshold by intent class** | High-impact classes get a near-exact threshold (or no semantic reuse); low-impact classes get a looser one |
| **Use a learned/predictive threshold per embedding rather than one global value** | Recovers hits in the safe regions of the space without loosening the unsafe ones |
| **Re-derive the threshold whenever the embedding model changes** | Prevents a silent quality regression from an infrastructure upgrade nobody linked to caching |
| **Verify before serving on a hit, where the domain warrants it** | An extra check costs less than a wrong answer; the source's own "grounded in approved sources" gate (§5.7) is the same idea |
| **Write the chosen threshold down with its justification** | In a regulated context this is a model-risk artefact, not a code comment |

> **The line to say:** *"I'd never present the threshold as 0.92 and move on. I'd present it as: for this class of question, what does a wrong answer cost? A music playlist app can run loose; a banking assistant can't run loose on anything mentioning international transfers. The number falls out of that, not out of a hit-rate target."*

---

### 5.3 The query is not the whole context

#### The mechanism

Semantic-cache demonstrations compare **only the text of the user's question**. Production applications need to consider much more.

#### Worked example — identical text, different correct answer

Two users ask, verbatim:

> **"What is my current plan?"**

**The text is identical. The correct answer is not.** One user may have a **free account**; the other may have an **enterprise contract**. A shared semantic cache that considers only the question could return one customer's answer to another. This is not a near-miss; it is a **wrong answer to a correct question** — the worst shape of failure, because the question matched perfectly.

#### The same problem, in other dimensions

| Dimension that varies | Example question whose correct answer depends on it |
|---|---|
| **User / account tier** | "What is my current plan?" |
| **Location / jurisdiction** | "Is this covered?" / "How long is the cooling-off period?" |
| **Language** | "What are the payment terms?" — same meaning, different answer, different legal text |
| **Permissions / entitlement** | "How do I reset another user's password?" |
| **Product version** | "How do I export the report?" — the menu changed two releases ago |
| **Policy date** | "What is the excess on my policy?" — the renewal changed it |
| **Conversation history** | "Can I return it?" — *it* has no referent without the thread |
| **Model configuration** | Same prompt, different temperature / system prompt / tool set |
| **Tool results** | The question is identical; the live data behind it is not |

The source's own examples of questions with **no stable meaning on their own**:

> **"Can I return it?"** has no stable meaning without knowing what **"it"** refers to.
>
> **"Is this allowed?"** depends on the current conversation.
>
> **"What is the price?"** may depend on **country, currency, contract and time**.

#### The architectural principle

> **Two requests are reusable only when the factors that determine their correct answers are also equivalent.**

**Semantic similarity is one of those factors. It is rarely the only one.** A safe semantic-cache key therefore **cannot always be based only on the latest user message** — the system may need to separate cache entries by tenant, user class, region, language, model, system prompt, knowledge-base version, or other relevant attributes. In some applications, the surrounding context is so important that **broad semantic reuse is inappropriate altogether**.

#### The cache-key attribute checklist

Treat this as a specification. Every attribute in the key is a claim that "two requests differing on this attribute do **not** have the same correct answer".

| Attribute | In the key when… | Failure it prevents |
|---|---|---|
| **Tenant / organisation** | Always, unless the deployment is single-tenant | Cross-tenant data leak |
| **User id or user class** | Answers are account-specific or entitlement-scoped | One customer's answer served to another |
| **Region / jurisdiction** | Legal, regulatory or product terms differ by geography | Wrong-law answer |
| **Locale / language** | The answer is rendered or legally valid in a specific language | Wrong-language or wrong-terminology answer |
| **Model / model version** | Output format, tone or capability differs | Schema mismatch; stylistic regression |
| **System-prompt version** | The system prompt has ever changed (it has) | Answers that no longer satisfy the current instruction |
| **Knowledge-base / index version** | Retrieval is in the path | Answer grounded in a superseded document |
| **Policy / document version** | The source document is versioned | Serving v4 after v5 shipped |
| **Intent class** | High-impact intents must not cross-match | The password pair; the banking case |
| **Entitlement / access level** | Answers describe permitted actions | Access-control answer served to an unauthorised user |
| **Tool set / tool result version** | Tools are in the path | Answer computed from a different live state |
| **Conversation-context hash** | The question is deictic ("it", "this", "that one") | Referent-less reuse |
| **Time bucket / TTL epoch** | The answer is time-sensitive | The stale-answer failure (§5.4) |

> **The line to say:** *"The cache key is the answer to the question 'what determines the correct answer here?' If the key only contains the user's text, the cache is asserting that nothing else does — and almost no production system can honestly make that claim."*

#### Mitigation

| Mitigation | How |
|---|---|
| **Composite key, not text-only key** | Build the key from the attribute checklist above; embed the text *within* a namespaced space |
| **Partition the vector index by the hard attributes** | Tenant and entitlement are **filters**, not similarity dimensions — filter first, then search |
| **Refuse to cache deictic or context-dependent queries** | Detect pronouns, demonstratives and short follow-ups and route them to fetch-then-generate |
| **Require a minimum query-information content** | A three-word question with no noun phrase probably is not self-contained |
| **Log the full key with every served answer** | If you cannot reconstruct why one answer applied to two requests, you cannot audit it |

---

### 5.4 The cache can preserve outdated answers

#### The mechanism

> **A language-model response may be correct when generated and wrong a week later.**

Product prices change. Policies are revised. Software versions are released. Employees leave companies. Exchange rates move. Support procedures are updated.

If a semantic cache stores an answer and later reuses it, the system **bypasses not only model inference but potentially any retrieval or tool call that would have supplied newer information**. That second clause is the one people miss: the cache does not merely skip the *model*, it skips the whole pipeline that was going to make the answer *current*.

#### The refund-policy v4 → v5 story

> Imagine that a support assistant generated an answer from **version 4** of the company's refund policy. The policy is later replaced by **version 5**, but the cached response remains available.
>
> A user asks a **semantically similar** question, and the system **returns the old answer instantly**.
>
> **From a performance perspective, the cache worked perfectly. From a business perspective, the application failed.**

That sentence — *"the cache worked perfectly and the application failed"* — is the one to keep. It is the clearest statement in the source of why caching metrics are the wrong success metrics (§9).

#### Why detection is harder than it looks

The obvious heuristic is language. Questions containing **"current", "today", "latest", or "my balance"** should immediately raise suspicion.

> **But detecting time sensitivity from language alone is not enough. An apparently timeless question may still depend on changing business data.**

Consider **"What is the excess on my policy?"** — no time word anywhere, and yet the correct answer depends on which policy version is in force. Or **"Who is our contact at the broker?"** — timeless phrasing, and the person left last month.

And the invalidation problem is generically hard:

> **Classic cache invalidation is already difficult. Semantic caching adds another dimension because the reused request does not exactly match the original one.**

With an exact cache you can delete the key. With a semantic cache, "the key" is a region of embedding space, and there is no `DELETE` for a region.

#### Mitigation

| Mitigation | How it works | Caveat |
|---|---|---|
| **Short TTLs** | Entries are assigned short lifetimes and expire | Trades hit rate for freshness; needs per-intent TTLs, not one global value |
| **Version-tag entries to the source** | Associate every entry with a **document or policy version** it was derived from | Requires the retrieval layer to expose versions — which [`15`](15-rag-architecture-deep-dive.md) §11 already argues insurance RAG must |
| **Invalidate on source change** | Delete entries when the underlying document changes | Needs a change feed from the source, and a mapping from document to affected entries |
| **Bypass for time-sensitive intents** | Time-sensitive requests **may need to bypass the cache entirely** | Language detection is a useful but insufficient trigger |
| **Prefer retrieval-time grounding over cached text** | Store the *answer shape*, re-fetch the facts | Weakens the saving; strongest safety posture |
| **Alert on age distribution of served entries** | A served-from-cache answer that is 90 days old is visible before a user reports it | Observability, not prevention |

> **The line to say:** *"A semantic cache bypasses retrieval as well as inference. So a stale-entry bug isn't only a stale model answer — it's an answer that never got the chance to be refreshed. That's why every entry needs a source version and a TTL, and why 'current', 'today' and 'my balance' are a starting heuristic, not the control."*

---

### 5.5 What exactly should be cached

#### What a cache entry contains

A semantic cache might store the user's prompt, its embedding, the generated response, **and metadata describing the environment in which the answer was created**. The source's point is blunt:

> **That metadata can become as important as the response itself.**

The metadata questions to answer for every entry:

| Metadata field | The question it answers |
|---|---|
| **Which model produced the answer?** | Is this answer still representative of what the system would produce now? |
| **Which system prompt was active?** | Does this answer still satisfy the current instruction set? |
| **Which documents were retrieved?** | What was this grounded in, and are those documents still authoritative? |
| **Which tools were called?** | Does this answer depend on live data that has since changed? |
| **Which customer or tenant owned the data?** | Who is allowed to see this again? |
| **Which policy version was used?** | Is this superseded? |
| **When was the response generated?** | Is it within its freshness window? |
| **What was the retrieval/rerank configuration?** | Is this answer reproducible from the current pipeline? |

Without that information, **the application may not know whether a cached answer is still valid.** Note the phrasing: not "may serve an invalid answer" — *may not know*. The cache has removed the system's own ability to reason about its output.

#### The two concrete stories

| Scenario | What breaks | Why metadata saves it |
|---|---|---|
| **A company updates the system instructions to enforce a new compliance requirement** | Previously generated answers **may no longer satisfy the new policy**, even where their factual content remains similar | A `system_prompt_version` field lets you invalidate every entry produced under the old prompt in one operation |
| **A switch to a model that produces a new structured-output schema** | Old cached responses **may not match the expected format** — the downstream parser fails, or worse silently mis-maps fields | A `model_version` / `schema_version` field lets you scope the invalidation and re-generate |

Both of these are **silent**. Nothing errors at the cache. The failure surfaces downstream, as a compliance finding or a malformed record, days later and far from the cache code.

#### The architectural conclusion

> **A semantic cache should be treated as part of the application's state, not as an invisible speed feature that can be added without affecting correctness.**
>
> **The cache stores decisions that the application may later repeat.**

That second sentence is the deepest idea in the source. A cached response is not a *string*; it is a **decision the application has committed to repeat** for a class of future inputs. It belongs in the architecture diagram, in the data model, in the change-control process and in the audit trail — right next to the system of record, not in a performance appendix.

#### Mitigation

| Mitigation | Detail |
|---|---|
| **Treat the cache as a versioned data store** | Schema, migrations, and a documented invalidation procedure — the same discipline as any other stateful component |
| **Store full provenance with every entry** | Model, prompt version, retrieved doc IDs and versions, tool calls, tenant, timestamp, pipeline version |
| **Make invalidation an operation, not a hope** | "Invalidate all entries whose `system_prompt_version` < N" must be one command, tested |
| **Gate writes on quality conditions** (see §5.7) | Not every successful generation deserves to become a reusable answer |
| **Review the cache in threat/risk models** | A stale or leaked cached answer is a risk scenario that belongs in the risk register |
| **Include the cache in the model card / system description** | "This system reuses previously generated answers" is a material fact about the system, not an implementation detail |

> **The line to say:** *"I'd put the cache in the architecture diagram with the same status as the system of record, because that's what it is — a store of decisions the application will repeat. The moment it's invisible, both the compliance change and the model upgrade become silent regressions."*

---

### 5.6 Privacy changes the architecture

#### The question

> **Semantic caching can reuse answers across requests, which raises an immediate question: across whose requests?**

#### The tension

- A cache **shared by all users** offers the **largest opportunity for reuse**.
- It also introduces the **greatest privacy and isolation risk**.

A response may contain **personal information, internal company data, retrieved document content, or details inferred from a previous conversation**. If that answer becomes eligible for reuse by another user, **the cache can become a data-leak mechanism**.

The source points at a research direction rather than a product: research on **privacy-aware semantic caching** has highlighted that conventional shared semantic caches can expose information through **cross-user reuse**, and has proposed **user-centric or privacy-aware designs** to reduce that risk. The design question is not novel or unsolved in principle — but it is routinely unsolved in practice, because the easy default is a global cache.

#### The three options

| Option | How it works | Trade-off |
|---|---|---|
| **Isolate by user or tenant** | Cache entries are namespaced; no cross-user reuse at all | The **safest** design; reduces reuse sharply. *Security boundaries should not be weakened merely to increase the cache-hit rate* |
| **Cache only approved, generic responses** | A support platform might allow **cross-user reuse for public documentation** but **disable it for account-specific answers** | Recovers much of the benefit in the safe part of the space; requires a reliable "is this generic?" classifier |
| **Classify before the cache check** | Public, stable, low-risk questions **may be eligible**; **personalised, confidential or high-impact questions always proceed to retrieval and generation** | Best balance; adds a classifier in the hot path and a taxonomy to govern |

The source's own preference is visible in how it closes the section:

> **This approach usually produces fewer cache hits than caching everything.**
>
> **That is often the correct result.**
>
> **A cache should optimise the workload without changing the application's security model.**

#### Why that last sentence matters

It is the cleanest statement of a principle that applies to every optimisation in the stack. An optimisation is supposed to change **how costly** the application is, not **what it is allowed to do**. A cache that makes cross-tenant reuse possible has changed the security model — and no amount of cost saving justifies that being a side effect of a performance ticket.

> **The line to say:** *"A cache should optimise the workload without changing the security model. If the only way to hit the cost target is to share a cache across tenants, then the cost target is wrong — not the isolation boundary. And I'd rather report a lower hit rate and explain why than quietly widen the blast radius."*

#### Mitigation summary

| Mitigation | Where it belongs |
|---|---|
| Namespace by tenant (and by user for personal data) | Cache key / index partition |
| Eligibility classifier before the cache check | Gateway, hot path |
| Approved-generic-response allow-list with human ownership | Content policy |
| PII detection on the cached payload before it is written | Write path |
| Data-retention limit on cached entries, aligned with the retention policy | Lifecycle job |
| Test the cache as a data-exfiltration path in security review | Assurance |

---

### 5.7 Caching amplifies a bad answer

#### The mechanism

> Suppose the model **hallucinates an incorrect support procedure** and the application stores the response in its semantic cache.
>
> **Without caching, the error affects one request. With semantic caching, the same answer may be reused for many related questions.**
>
> **Caching turns a single generation into a reusable decision.**

This is the failure mode that most cleanly distinguishes semantic caching from an inference optimisation. The KV cache cannot amplify an error, because it never produced a reusable *answer* in the first place. The semantic cache turns one bad output into a **de facto policy**.

#### The amplification arithmetic — say this out loud

| | Without semantic cache | With semantic cache |
|---|---|---|
| Requests affected by one bad generation | **1** | **Every future request whose embedding lands within the threshold** |
| Visibility | The user reports it | Each affected user may report it, or none may |
| Blast radius growth | Linear with request rate | **Super-linear in the similarity region's traffic** |
| Remediation | Apologise, fix the prompt | **Find and purge the region, then explain why a wrong answer was served consistently** |
| Auditability | One generation trace | *"Why did the system decide this class of question had the same answer?"* — and the answer is a cosine score |

> *"Caching magnifies both quality and error."*

#### The gate: do not store every successful response

> **This is why the cache should not automatically store every successful model response.** The system may require **additional conditions before an entry becomes reusable.**

Candidate entry conditions, from the source:

| Condition | What it admits | What it excludes |
|---|---|---|
| **The response must be grounded in approved sources** | Answers traceable to the authoritative corpus | Free-form generation, model prior knowledge |
| **The citations must be verified** | Answers whose claims were checked against the retrieved text | Answers that cite a document they were not actually derived from |
| **Only deterministic templates are eligible** | Fully constrained, parameterised outputs (a defined process, a defined field) | Anything with free-text reasoning |
| **The answer must be reviewed or receive positive feedback before it enters the shared cache** | Human-approved or user-validated answers | Everything unvetted |

Note the shape of these. They are not tuning knobs; they are **admission criteria**. The cache becomes an allow-list of answer *classes* rather than a store of answer *instances*.

> **The principle:**
>
> **Do not make an answer easier to repeat before deciding whether it deserves to be repeated.**

#### Mitigation

| Mitigation | Detail |
|---|---|
| **Gate on grounding** | Store only answers where every claim has a cited, retrieved, in-version source |
| **Gate on verification** | Run the faithfulness check ([`15`](15-rag-architecture-deep-dive.md) §9, [`07`](07-agent-evaluation-and-guardrails.md)) *before* the write, not only before the response |
| **Gate on determinism** | For a narrow set of intents, cache only template-rendered answers — then the cache can only reproduce something that was already correct by construction |
| **Gate on human approval for the high-impact classes** | A reviewed answer can be reused widely; an unreviewed one should not be reused at all |
| **Include a confidence floor** | A low-confidence generation should never become a high-confidence cached answer; the cache flattens confidence |
| **Make purge a rehearsed operation** | You will need to purge. Practise it before you need it, and measure how long it takes |
| **Report the amplification factor as a risk metric** | Cached-answer reuse count per entry is the number a risk committee wants to see |

> **The line to say:** *"The cache doesn't just make answers faster — it makes them permanent. So the question before storing anything isn't 'was this a good response?', it's 'do I want this response to be the system's answer to a whole class of questions?' If the answer isn't an unambiguous yes, the entry shouldn't be written."*

---

## 6. Where it works and where it must not be used

### 6.1 Where semantic caching works well

> **Semantic caching is most attractive when requests are repetitive, answers are stable, and moderate similarity can safely imply the same response.**

| Good fit | Why it fits |
|---|---|
| **Public FAQs** | *"Users ask the same questions using many different phrasings, while the approved answer changes infrequently."* The canonical case |
| **General explanations** | Conceptual answers that do not depend on the requester |
| **Product instructions** | Stable, non-account-specific, versioned — and the version tag handles the change |
| **Educational content** | Stable by nature; a slightly different phrasing deserves the same explanation |
| **Classification tasks** | The *label* is the reusable artefact and it is narrow by construction |
| **Any workload with high request repetition** | Hit rate is a function of traffic shape, not of cleverness |
| **Any workload whose answers are stable over the cache's TTL** | Freshness is the gating property |
| **Where moderate similarity safely implies the same response** | The safe-similarity condition, stated as an admission test |

**The benefits can be significant.** A cache hit may:

- remove **most of the model latency**,
- avoid **token charges** (both input and output),
- reduce pressure on **provider rate limits**,
- improve **resilience when a model backend is temporarily unavailable** — a real, underrated benefit: a semantic cache is arguably the cheapest degraded-mode path you can build.

> **The important qualifier:** *"Semantic caching is not a universal layer that should intercept every prompt. It is a **selective optimisation for requests whose answers are genuinely reusable**."*

### 6.2 Where it must not be used

> **It is usually unsafe to reuse responses for live account data, financial values, medical decisions, legal determinations, access-control questions, or actions whose validity depends on current state.**

| Unsafe | Why |
|---|---|
| **Live account data** | The correct answer changes with the account's state, which the query text does not carry |
| **Financial values** | Prices, balances, rates and limits move; a cached value is a wrong value with a confident tone |
| **Medical decisions** | The cost of error is catastrophic and the answer is person-specific |
| **Legal determinations** | Jurisdiction, contract and date all matter; similarity is not a legal test |
| **Access-control answers** | Answering "can I do X?" from a cache is a security control failing open |
| **Actions whose validity depends on current state** | The action may have been valid when cached and invalid now |
| **Anything personalised** | The personalisation factors are exactly the ones the query text omits |
| **Anything rapidly changing** | TTL becomes so short the cache returns nothing of value |
| **Anything highly contextual** | Deictic and thread-dependent questions have no standalone meaning |
| **Anything high-risk** | Where the correct threshold is effectively "exact match", semantic reuse is the wrong tool |

**The pattern behind both lists.** It is not about the domain — it is about whether the **factors that determine the correct answer** are (a) fully captured in the cache key and (b) stable over the entry's lifetime. If either is false, semantic caching is unsafe regardless of how mature the tooling is.

> **The line to say:** *"I'd draw the line at: is the correct answer a property of the question, or a property of the world? If it's a property of the world — the account, the balance, the policy version, the entitlement — then semantic reuse is asserting that the world hasn't changed. That's not a caching decision, that's a risk decision, and I'd say no."*

### 6.3 Where the KV cache remains useful

The source is careful to keep this contrast explicit, because it explains why the two mechanisms are not competitors:

> **The KV cache does not face the same semantic correctness question, because it does not decide that one user's completed answer should satisfy another user.** It reuses internal computation associated with tokens that have already been processed. During an active generation, this is what prevents the model from recalculating the entire sequence for each new token.
>
> **Prefix caching can extend that reuse across requests that begin with the same tokens** — repeated system instructions, or multiple questions about the same long document.
>
> The main trade-offs are **memory, cache management and workload efficiency**, rather than whether two differently phrased questions deserve the same answer.

> **The summary pair, memorise it:**
>
> **"One is a fundamental inference optimisation. The other is an application-level policy."**

---

## 7. Running both layers together

The two caches are **not competitors**. A production architecture may — and usually should — use both.

### 7.1 The source's ordering

> 1. **First, decide whether a previous *answer* is safe to reuse.**
> 2. **If not, decide whether previous *model computation* can be reused.**
> 3. **If neither applies, perform the full inference.**

Restated in the source's own words:

> When a request arrives, the application or AI gateway can first determine whether the request is **eligible for semantic caching**. If a sufficiently similar **and valid** response exists, the cached answer is returned and **inference is avoided**.
>
> If there is **no safe semantic hit**, the request proceeds to the model.
>
> The model-serving layer may then **reuse a shared KV prefix** for stable system instructions, previously processed conversation history, or a repeated document. During generation, the ordinary **KV cache** prevents redundant computation for earlier tokens.

The two layers save different kinds of work:

| Layer | Optimises | Level of abstraction | Where it lives |
|---|---|---|---|
| Semantic cache | Across **requests**, at the level of **meaning and completed responses** | Application / gateway | Bypasses the model |
| KV + prefix cache | Model **computation**, at the level of **tokens and internal representations** | Inference engine | Accelerates the model |

> **The source's key operational insight:** *"A semantic miss can still benefit from KV-prefix reuse. A request that cannot safely use semantic caching may still be accelerated by the inference engine."*

That is the reassuring half of the story. The safety posture (semantic cache refuses the high-risk classes) does not condemn those requests to full-cost inference — the inference engine still gives them the prefix and KV wins.

### 7.2 The decision flow

```text
                    ┌─────────────────────────────┐
  Request  ────────►│  1. CLASSIFY & BUILD KEY     │
                    │  intent · tenant · entitlement
                    │  · locale · versions · time  │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │ 2. ELIGIBLE FOR SEMANTIC?   │
                    │  personalised? high-impact? │
                    │  time-sensitive? confidential?
                    └───────┬─────────────┬───────┘
                       yes  │             │  no
                            ▼             │
              ┌───────────────────────────┐ │
              │ 3. SEMANTIC LOOKUP        │ │
              │  embed → ANN over the     │ │
              │  namespaced entry space   │ │
              │  similarity ≥ threshold?  │ │
              └────┬────────────────┬─────┘ │
             HIT   │                │ MISS  │
                   ▼                │       │
      ┌────────────────────────┐    │       │
      │ 4. VALIDATE THE ENTRY  │    │       │
      │  model / prompt version│    │       │
      │  doc version, TTL,     │    │       │
      │  source unchanged?     │    │       │
      └────┬──────────────┬────┘    │       │
      VALID│              │STALE    │       │
           │              ▼         │       │
           │     ┌──────────────────┴───────┴────┐
           │     │ 5. PREFIX-CACHE ELIGIBLE?     │
           │     │  does the request share an    │
           │     │  EXACT token prefix with a    │
           │     │  resident prefix block?       │
           │     └───┬────────────────────┬──────┘
           │   HIT   │                    │ MISS
           │         ▼                    ▼
           │  ┌──────────────┐   ┌──────────────────┐
           │  │ 6a. Load KV  │   │ 6b. Full prefill │
           │  │ blocks; run  │   │ of the prompt,   │
           │  │ decode only  │   │ then decode      │
           │  └──────┬───────┘   └────────┬─────────┘
           │         │                    │
           │         └────────┬───────────┘
           │                  ▼
           │        ┌──────────────────────────┐
           │        │ 7. KV CACHE DURING       │
           │        │    DECODE (always)       │
           │        └────────────┬─────────────┘
           │                     ▼
           │        ┌──────────────────────────┐
           │        │ 8. ADMISSION GATE        │
           │        │  grounded? verified?     │
           │        │  deterministic? approved?│
           │        └────┬──────────────┬──────┘
           │      admit │              │ reject
           │            ▼              └──► no entry written
           │   ┌──────────────────┐
           │   │ 9. WRITE ENTRY   │
           │   │ + full provenance│
           │   │ + TTL + version  │
           │   └────────┬─────────┘
           │            │
           └────────────┴──────────► 10. RESPOND
                                        (log: which layer served it)
```

### 7.3 The five design rules that fall out of the flow

| # | Rule | Consequence if broken |
|---|---|---|
| 1 | **The semantic check comes first because it saves the most** | You pay for inference you did not need |
| 2 | **Eligibility is decided before similarity, not after** | A high-impact request gets a fast wrong answer |
| 3 | **Validation is a separate step from matching** | Stale entries are served with high confidence |
| 4 | **A semantic miss still gets a prefix-cache attempt** | You lose the free win on the requests that most need it |
| 5 | **Every response records which layer served it** | You cannot compute any of the metrics in §9 |

> **The line to say:** *"The two layers answer different questions. The semantic cache asks 'have we answered this before, and is that answer still valid?' The prefix cache asks 'have we processed these exact tokens before?' If the first says no, the second can still say yes — and that's the reason you want both, not one instead of the other."*

---

## 8. Using it in an interview

### 8.1 The source's "what leaders should ask"

> **When a vendor says that its AI platform includes caching, the word alone tells you very little.**

The four questions that disambiguate:

| Question | What it distinguishes |
|---|---|
| **Is it caching exact API responses?** | A response cache — safe, limited to exact repeats |
| **Is it performing embedding-based semantic matching?** | A semantic cache — the correctness risk lives here |
| **Is the model provider reusing prompt prefixes?** | Prefix caching — a pure cost/latency win, no correctness risk |
| **Is the inference engine preserving KV state during generation?** | The KV cache — table stakes, not a feature |

> *"These mechanisms have different implications for cost, performance, privacy and correctness."*

### 8.2 The follow-up questions, split by mechanism

**For semantic caching** — which requests are eligible? How are similarity thresholds selected and justified? How are users and tenants isolated? How do entries expire? **What happens when source information changes?** Can personalised data enter the cache? **Can an incorrect model response be reused automatically?**

Note the last two. They are the questions that separate a demo from a production system, and they are the two most vendors will not have a good answer to.

**For KV / prefix caching** — how repetitive are the prompts? What is the context length? What is the time to first token? How much GPU memory is in play? **Does the provider expose any performance or pricing benefit from cached prefixes?** That last one is commercial, not technical, and it is the one finance will thank you for.

### 8.3 The demand for *separate* measurements

> **Most importantly, teams should demand separate measurements.**

Five numbers, and they must be reported **separately**:

| # | The measurement | Why separately |
|---|---|---|
| 1 | **What percentage of requests receive a semantic-cache hit?** | The headline number — and the one that lies on its own |
| 2 | **How often are those hits correct?** | Without this, #1 is not a performance metric, it is an exposure |
| 3 | **How much latency is saved?** | The Latency dial — the reason the cache was proposed |
| 4 | **What cost is avoided?** | The Cost dial — the amount finance will ask about |
| 5 | **How many incorrect or stale responses were returned?** | The **Quality and Safety** dials — the number nobody volunteers |

> ### **A high hit rate is not a success metric if the cache is returning the wrong answer.**

That is the sentence the source lands, and it is the sentence to say in the room.

> **The closing frame from the source, worth quoting nearly verbatim:** *"Caching should improve the system's economics without silently weakening its behaviour."*
>
> And the deeper one: *"As with every optimisation, the important question is not merely whether it improves performance. **The important question is what assumption makes the optimisation safe.** For KV caching, the assumption is that previously processed token state remains reusable. For semantic caching, the assumption is much stronger: that a previous answer remains valid for a different request. **That assumption should never be left to a similarity score alone.**"*

### 8.4 The four distinct mechanisms — do not conflate them

| # | Mechanism | Stores | Risk to name |
|---|---|---|---|
| 1 | **Exact response cache** | Normalised request → response | Almost none; misses on any variation |
| 2 | **Semantic response cache** | Prompt embedding → response (+ metadata) | Correctness, staleness, privacy, amplification |
| 3 | **Prefix / prompt cache** | KV tensors of a shared token prefix | None; cost regression if the prefix is rewritten |
| 4 | **KV cache** | K/V tensors for the active sequence | None; it is a memory-sizing constraint |

> **Why the list matters:** three of the four are *"cheaper"*; one of them is *"different behaviour"*. In a design round, naming which one you are proposing — and therefore which risk you are accepting — is the whole answer.

### 8.5 Land the trade-off statement

> **The line to say:** *"I'd add the semantic cache for the FAQ and product-instruction traffic, because there the answer is a property of the question and the cost of a near-miss is small. I would not let it touch anything account-specific or time-sensitive — those always go to retrieval and generation, and they still get the prefix-cache win on the shared system prompt, so nothing is wasted. And I'd measure hit rate and hit *correctness* as two separate numbers, because a high hit rate on wrong answers is not a saving, it's an exposure."*

### 8.6 If they ask "isn't this just a gateway concern?"

Yes — and the source's own series says so. The next article in that series is on **AI gateways**, where *"semantic caching, routing, rate limits, fallbacks, governance, and model observability increasingly come together."* The framing is that direct connections from every application to every model provider become difficult to manage, and the gateway is where the shared controls concentrate.

For the interview, the useful version is: caching is one of the controls a gateway exists to provide, alongside routing, token accounting, failover and audit. Depth on that layer lives in [`16`](16-agentic-ai-systems-deep-dive.md) and [`17`](17-azure-ai-services-deep-dive.md).

> *"Semantic caching is a gateway-layer control, which is why I'd want it in one place rather than implemented per application. If three apps each build their own cache with their own threshold, I have three different risk postures and no shared evidence."*

---

## 9. Measuring it properly

### 9.1 The metrics table

| Metric | How to compute | Target | The failure it catches |
|---|---|---|---|
| **Semantic hit rate** | `semantic_hits / eligible_requests` | Domain-dependent; the source's framing points at the FAQ-heavy end | Not a success metric on its own — but a hit rate of ~0 on eligible traffic means the threshold or the embedding model is wrong |
| **Hit correctness rate** | Sample audited hits (or run the faithfulness judge on them): `correct_hits / audited_hits` | **> 0.99** for anything user-facing; 1.0 for high-impact classes | **The core semantic-cache failure** — a high hit rate on wrong answers |
| **Stale-served rate** | `hits where entry_source_version < current_source_version` / hits | **0** | The refund-policy v4 → v5 failure |
| **Mean entry age at serve time** | Mean TTL remaining or age of the served entry | Per-intent freshness SLO | Slow drift into staleness that no single event flags |
| **Latency saved (P50 / P95)** | `latency(full_inference) − latency(cache_hit)` on a matched workload | The saving you claimed in the business case | A cache that hits but is not actually faster (e.g. the lookup is slow) |
| **Cost avoided** | `Σ (input_tokens + output_tokens) × price` over avoided calls | The saving you claimed | The commercial justification going unverified |
| **Prefix-cache hit rate** | `prefix_cache_hits / requests` on the provider console | High for a stable system prompt | **A prompt edit having silently destroyed the reusable prefix** |
| **Prefix-cache cost delta** | Month-over-month input cost on the stable prefix, normalised for volume | Flat or falling | The invisible prompt-cache regression (§3) |
| **KV cache occupancy** | `in_flight × avg_context × KB_per_token` vs VRAM headroom | Headroom > 20% | VRAM exhaustion as the real scaling wall |
| **Amplification factor** | Max / mean reuse count per cache entry | Bounded; investigate outliers | One hallucinated answer becoming a policy (§5.7) |
| **Cross-tenant leakage** | Count of hits where `entry.tenant != request.tenant` | **Must be 0**, asserted in a test | The privacy failure of §5.6 |
| **Eligibility misclassification rate** | `high-impact requests that reached the semantic cache` / high-impact requests | **0** | The classifier letting a banking-assistant question into a music-app cache |
| **Purge time** | Wall-clock to invalidate all entries from a prompt or document version | Minutes, rehearsed | Discovering during an incident that you cannot un-say a cached answer |
| **Cache-served share of answers** | `answered_from_cache / total_answers` | Monitored, not maximised | Losing sight of how much of the system's output is not freshly generated |

### 9.2 The cost-of-error framing (the pack's rule 5)

Never present a caching metric without its cost-of-error counterpart. The pack's fifth rule (`00` §3) is explicit: *"Connect every metric to cost-of-error."*

| Dial | Caching metric in that dial's language |
|---|---|
| **Quality** | Hit correctness rate × the cost of one wrong answer. *"A 70% hit rate at 99.5% correctness on 100,000 monthly questions is 350 wrong answers a month. I need to know what one is worth."* |
| **Cost** | Cost avoided per 1,000 eligible requests, versus the cost of building and running the cache (embedding calls, vector store, the classifier, the review process). The cache is only a saving if it nets out |
| **Latency** | P95 latency saved on hits, against the P95 of a cache **miss**, which is now *slower* than a plain request because it also paid for an embedding and an ANN lookup |
| **Safety** | Stale-served rate, cross-tenant leakage count, amplification factor, misclassification rate — this is the dial where caching earns or loses its licence to exist |

> **The line to say:** *"Four numbers, reported separately, and never one without the other three: how often we hit, how often the hit was right, what we saved, and how many wrong or stale answers we served. And I'd add a fifth — the miss penalty, because a semantic cache makes every miss more expensive than no cache at all."*

### 9.3 The miss penalty — the cost nobody budgets

The source does not name this explicitly, but it follows directly from its own architecture and is worth stating because it is a common design-review catch:

| Path | Work performed |
|---|---|
| No cache at all | Retrieve → rerank → prefill → decode |
| Semantic cache **hit** | Embed → ANN lookup → validate → return |
| Semantic cache **miss** | Embed → **ANN lookup** → validate-fail → Retrieve → rerank → **prefix check** → prefill → decode → **write-back** (embed + index) |

So a semantic cache adds cost and latency to the requests it does **not** serve. The break-even is not "hit rate > 0" — it is:

```text
saving_on_hits  >  penalty_on_misses  +  running_cost_of_the_cache
```

For a workload with a 10% hit rate, the miss penalty can easily exceed the saving. **Hit rate alone does not tell you whether the cache pays.** That is a genuinely strong point to raise in a design round, because it reframes the discussion from "how do we raise the hit rate" to "how do we make sure the cache earns its keep".

### 9.4 The evaluation harness — what to run offline

| Test | What it asserts |
|---|---|
| **Golden adversarial pairs** | For each of the source's example pairs — password/self-service vs password/admin, cancel vs cancel-and-refund — assert the cache does **not** cross-match. Run in CI |
| **Tenant isolation test** | Ingest an entry as tenant A; assert tenant B can never receive it, at any threshold |
| **Staleness test** | Write an entry, bump the source document version, assert the entry is invalidated or refused |
| **Prompt-version test** | Change the system prompt; assert all pre-change entries are unreachable |
| **Eligibility test** | Feed a high-impact intent; assert it bypasses the semantic layer |
| **Amplification test** | Inject a known-wrong answer; assert the admission gate refuses it |
| **Miss-penalty benchmark** | Measure P95 on the miss path and confirm the cache is net-positive at your real hit rate |

> **Cross-link:** the evaluation and guardrail depth for these is [`07`](07-agent-evaluation-and-guardrails.md); the RAG-side equivalents (faithfulness, conflict detection, staleness audit) are [`15`](15-rag-architecture-deep-dive.md) §8–§9.

---

## 10. Brit translation

**Correctness rule for this section.** The Brit-flavoured illustrations below are marked **(illustrative)** — they are generic Lloyd's specialty-insurer patterns, not claims about Brit's systems, vendors, org or numbers. Brit's verified facts live in [`11`](11-insurance-lloyds-domain-primer.md) §3–§7 and [`14`](14-industry-intel-lead-ai-interview-research.md); assert nothing beyond those.

### 10.1 Why a Lloyd's specialty insurer is a hard case for semantic caching

| Property of the domain | Consequence for caching |
|---|---|
| **Answers depend on the policy, the slip, the endorsement and the effective date** | The cache key must carry **policy version**, and invalidation must be tied to the document version — exactly the version tags [`15`](15-rag-architecture-deep-dive.md) §11 already requires for RAG |
| **Brokers, coverholders and syndicates are distinct data owners** | The cache key must carry **tenant / broker / syndicate**. A shared cache across them is a confidentiality breach, not a performance win |
| **A submission is confidential, and often contains a named insured** | **Never cache an answer derived from a submission or naming an insured.** These are the §5.6 "confidential" class, and they belong on the always-retrieve path |
| **Underwriting and claims outputs inform regulated decisions** | Consumer Duty (**FCA PS22/9**) applies to automated outcomes; **PRA SS1/23** expects model risk management; **SM&CR** means a named person is accountable. A served-from-cache answer is still an **answer the firm is accountable for** |
| **Sanctions and AML screening are hard controls** | A cached "yes, that's fine" to a screening-adjacent question is a control failing open. Access-control and screening questions must `never` be served semantically |
| **Terms can change at renewal or by endorsement** | Endorsements supersede the schedule ([`15`](15-rag-architecture-deep-dive.md) §11). A cached answer from the pre-endorsement version is stale *and* wrong in a way a reviewer would treat as a controls failure |

### 10.2 The cache-key for this context (illustrative)

```text
cache_key = {
    tenant_id,                 # broker / coverholder / syndicate — never a global cache
    user_class,                # underwriter / claims handler / ops / broker-facing
    entitlement_level,         # what this principal is allowed to be told
    jurisdiction,              # England & Wales vs other
    locale,                    # language of the answer
    intent_class,              # FAQ | product-guidance | submission-derived | claims |
                               # screening | access-control | live-value
    model_id + model_version,  # answers are model-dependent
    system_prompt_version,     # compliance instructions change
    knowledge_base_version,    # index snapshot
    policy_document_version,   # v4 vs v5 — the whole point
    endorsement_set_version,   # endorsements supersede the schedule
    tool_set_version,          # if tools were in the path
    time_bucket,               # TTL epoch
}
```

**And the classes that never enter the semantic path at all** (illustrative):

| Class | Why it is excluded |
|---|---|
| Anything derived from a **submission or slip** | Confidential by nature; the answer is about a specific risk |
| Anything naming an **insured, claimant or broker contact** | Personal and contractual data |
| **Live values** — limits, balances, exposures, premium | [`05`](05-system-design-round-playbook.md) §5.1's live-value trap: the index is a cache of documents, not a database of state |
| **Access-control** answers | Security control must fail closed |
| **Screening / sanctions** questions | Hard control; human owns the decision |
| **Coverage determinations** | A coverage answer is a regulated communication |
| **Anything whose validity depends on current state** | The action may have been valid when cached and invalid now |

### 10.3 The four dials, stated in this context

> **The line to say:** *"For a Lloyd's specialty insurer I'd frame the caching layer as: we are buying **Cost** and **Latency** on the FAQ and product-guidance traffic, and we are explicitly refusing to buy them on anything submission-derived, account-specific or screening-related. The **Quality** dial is what we pay with on the traffic we do cache, which is why I'd want the threshold set from the cost of a wrong answer and the entry gate requiring grounding in an approved source. And the **Safety** dial is what makes this different from a consumer app: a served-from-cache answer is still an answer the firm is accountable for, so the cache-hit path has to be auditable with the same rigour as the generation path — which layer served it, against which prompt version and which policy version, and who approved that class of answer for reuse. The fourth dial here isn't uptime — it's auditability."*

That is the reconciliation line from [`00`](00-README-and-7-day-plan.md) §4 applied to caching: *"quality, latency, cost — and the fourth dial is reliability; in a Lloyd's-regulated context that means safety and auditability, not just uptime."*

### 10.4 The audit requirement, concretely

A cached answer must produce an audit record as complete as a generated one (pattern from [`15`](15-rag-architecture-deep-dive.md) §11 and [`07`](07-agent-evaluation-and-guardrails.md)):

| Field | Why the auditor wants it |
|---|---|
| **Which layer served the answer** (semantic cache / prefix-accelerated generation / full generation) | Determines whether the answer was freshly generated or reused |
| **The matched cached query and its similarity score** | The decision that two questions deserved the same answer, made explicit |
| **The source document and version the entry was derived from** | Provenance — the same requirement as a generated answer |
| **The prompt version and model version in force at write time** | Shows the answer satisfied the controls current when it was produced |
| **The admission gate decision** (which condition admitted the entry, and by whom if human-approved) | Shows the reuse was deliberate and governed |
| **Entry age and TTL at serve time** | Shows the freshness control operated |
| **The tenant / entitlement the answer was produced under** | Shows the isolation boundary held |

> *"The trace is the audit evidence — and it has to exist for a served-from-cache answer too. If the cache path is a blind spot in the trace, then a whole class of answers has no evidence behind it."*

### 10.5 What I would and would not put behind a semantic cache (illustrative)

| Traffic | Semantic cache? | Reasoning |
|---|---|---|
| **"How do I submit a slip?" / process FAQs** | **Yes** | Public, stable, repeated, low cost of a near-miss |
| **"What cover does product X provide?" general guidance** | **Yes, version-tagged** | Stable and repetitive; the version tag handles the change |
| **Internal "how do I use the platform" support** | **Yes** | Classic FAQ shape |
| **Document classification / routing** | **Yes** | The label is the reusable artefact, and it is narrow |
| **"What is the excess on policy POL-…?"** | **No** | Live value + policy version; always retrieve |
| **"Is this risk acceptable?"** | **No** | Underwriting determination; human-owned |
| **"Is this party sanctioned?"** | **No** | Hard control; human owns the screening decision |
| **"What's the current status of claim …?"** | **No** | Live state |
| **"What does this submission say about X?"** | **No** | Submission-derived; confidential |
| **Anything naming an insured or broker** | **No** | Confidential / personal |
| **Access-control answers** | **No** | Must fail closed |

---

## 11. Interview Q&A

### Q1. What's the difference between the KV cache and a semantic cache?

**Headline:** *"KV caching makes inference more efficient; semantic caching attempts to avoid inference."*

The **KV cache** stores the keys and values for tokens already processed, inside GPU VRAM, for the lifetime of a single request. During autoregressive decoding the model would otherwise recompute attention over the entire sequence for every new token; the cache means it only computes maths for the new one. It never returns a finished answer and it never decides that two different questions are the same.

The **semantic cache** stores completed responses keyed by the embedding of the prompt. On a hit — cosine similarity above a threshold, commonly 0.85–0.95 — it returns the stored response **without calling the model at all**, saving both input and output tokens.

> *"One lives near model inference; the other lives in the application or gateway layer. One contains intermediate model state; the other contains reusable application results. That difference is what makes one an engineering problem and the other a correctness problem."*

---

### Q2. And where does the prefix cache fit?

**Headline:** *"The prefix cache is the KV cache made shareable, across requests instead of within one."*

It is a **server-side persistence optimisation built directly on top of the KV cache**. When many requests share the same starting tokens — a long system instruction, a few-shot block, a shared RAG document — the server retains the computed KV tensors for that prefix. A new request matching it skips the expensive prefill and loads the state directly, running the model only on the remaining new tokens.

Providers commonly bill cache hits at a fraction — around **0.1×** — of standard input rates, so the saving is real and shows up on the invoice.

The critical property: **matching is by hash chain on exact token IDs from the start of the prompt — token identity, not meaning.** Reword the system prompt and the reusable prefix collapses, even though a human would call the two instructions equivalent.

---

### Q3. Why is the prefix cache safe but the semantic cache risky?

Because they make different claims.

The prefix cache claims that **the same tokens produce the same internal state** — which is true, and is a statement about computation. It cannot return a wrong answer, because it never produces an answer; it produces a state that the model then continues from.

The semantic cache claims that **a previous answer remains valid for a different request** — a much stronger claim, and one that depends on facts about the world rather than facts about the tokens.

> *"For KV caching the assumption is that previously processed token state remains reusable. For semantic caching the assumption is that a previous answer remains valid for a different request. That assumption should never be left to a similarity score alone."*

---

### Q4. Give me an example where a semantic cache produces the wrong answer.

The cleanest one is the password pair. **"How can I reset my password?"** and **"How can an administrator reset another user's password?"** are semantically close — both about resetting passwords, and an embedding model places them near each other. But they require completely different answers, permissions, audit requirements and security controls. The first is a self-service flow any user may perform; the second is a privileged action with dual control and an audit entry.

Same shape with **"Can I cancel my subscription?"** versus **"Can I cancel my annual subscription and receive a full refund?"** — the second depends on details the first does not contain.

> *"Similarity is not equivalence. That's the whole lesson."*

---

### Q5. How would you choose the similarity threshold?

**Headline:** *"It's a product-risk decision, not a tuning parameter."*

Too strict and valid reuse becomes misses — you keep calling the model and save nothing. Too loose and the cache returns an answer relevant to the general subject but wrong for the actual question.

A music-recommendation assistant can tolerate a loose threshold; a slightly imperfect playlist is harmless. A banking assistant should not reuse an answer merely because two questions mention international transfers — one may concern fees, another fraud, another sanctions, another cancellation.

> *"The acceptable threshold depends on what happens when the system is wrong. So I set it per intent class from the cost of a wrong answer, I re-derive it whenever the embedding model changes, and I'd look at a learned per-embedding threshold rather than one global number — static thresholds are known to be unreliable."*

---

### Q6. Two users ask exactly the same question. Is that a safe cache hit?

Not necessarily — **the query is not the whole context.**

**"What is my current plan?"** is identical text with a different correct answer for a free user and an enterprise customer. The same problem arises with location, language, permissions, product version, policy date, conversation history, model configuration and tool results. **"Can I return it?"** has no meaning without knowing what *it* is.

> *"The architectural principle is: two requests are reusable only when the factors that determine their correct answers are also equivalent. Semantic similarity is one of those factors; it is rarely the only one. So the cache key has to be composite — tenant, user class, region, locale, model, system-prompt version, knowledge-base version — and if the key can't carry those, the correct answer is not to cache."*

---

### Q7. How do you handle the cache going stale?

Three controls, layered.

**Time:** per-intent TTLs, shortest for anything that touches a value that can move.
**Provenance:** every entry carries the document or policy version it was derived from, so invalidation is a query against a version field rather than a guess.
**Change feed:** when a source document changes, invalidate the affected entries — the mirror image of [`15`](15-rag-architecture-deep-dive.md)'s incremental-sync approach for the index.

And a bypass rule: **time-sensitive requests may need to skip the cache entirely.** Words like *current*, *today*, *latest*, *my balance* should raise suspicion — but detection from language alone is not enough, because an apparently timeless question can still depend on changing business data.

> *"The point people miss is that a semantic hit bypasses not just inference but any retrieval or tool call that would have supplied fresher data. So a stale-entry bug is worse than a stale model answer — the answer never got the chance to be refreshed."*

---

### Q8. What exactly would you store in a cache entry?

The response **and its provenance**: which model produced it, which system prompt was active, which documents were retrieved, which tools ran, which customer or tenant owned the data, which policy version was used, and when it was generated.

> *"That metadata becomes as important as the response, because without it the application can't know whether the cached answer is still valid. Consider a company that updates system instructions to enforce a new compliance requirement — previously generated answers may no longer satisfy the new policy even where the facts are similar. Or a switch to a model with a new structured-output schema, where old responses don't match the expected format. So I'd treat the cache as part of the application's state, not as an invisible speed feature."*

**The line:** **the cache stores decisions the application may later repeat.**

---

### Q9. Is a global shared cache fine if the answers are generic?

It is the largest reuse opportunity and the largest leak surface. A response may contain personal information, internal company data, retrieved document content, or details inferred from a previous conversation. If it becomes eligible for reuse by another user, **the cache has become a data-leak mechanism**.

The options are to isolate by user or tenant, or to cache only approved generic responses — public documentation yes, account-specific answers no — or to classify requests before the cache check and send personalised, confidential or high-impact ones straight to retrieval and generation.

> *"That produces fewer hits than caching everything. That is often the correct result. A cache should optimise the workload without changing the application's security model."*

---

### Q10. Hostile: "You added a semantic cache, cost dropped 40%, but users say they're getting answers to questions they didn't ask. What went wrong?"

**Answer — go straight to the mechanism, not the defence.**

The symptoms identify the failure precisely: the cache was **serving hits for questions whose correct answers differ**. Cost dropping 40% is consistent with that, not evidence against it. Four things in that sentence point at the likely root causes, and I'd check them in this order:

1. **The threshold was set from a hit-rate target rather than from the cost of a wrong answer.** A loose threshold produces exactly this symptom — answers relevant to the general subject but wrong for the actual question. First fix: audit the matched-pair log and see what the near-misses look like.
2. **The key is text-only.** If the cache key is the user's embedding and nothing else, then every account-specific, tenant-specific, version-specific and entitlement-specific question is being cross-matched. **"What is my current plan?"** produces exactly this complaint. Second fix: composite key, filter first then search.
3. **Nothing is validating the entry before serving it.** A match is not an admission. The entry's model version, prompt version, document version and TTL all need checking before the response goes out.
4. **The eligibility classifier is missing or permissive.** High-impact intents — access control, financial, personalised — should never reach the semantic layer at all.

**Then the honest framing:**

> *"I'd roll the threshold up to effectively exact-match within the hour — that costs the majority of the 40% — because the correct sequence is to restore correctness first and re-earn the saving with evidence. Then I'd want the two numbers that answer the question: the audit of matched pairs, and the hit-correctness rate on those hits. And I'd say the underlying mistake plainly: I optimised a proxy — hit rate — instead of the outcome. A 40% cost reduction on a cache returning wrong answers isn't a saving, it's an exposure with a discount attached."*

**And the systemic point:** this is why the pack's rule is *measure hit rate and hit **correctness** separately*. The measurement would have caught this before users did.

---

### Q11. Can caching make quality worse in a second way — beyond serving the wrong answer?

Yes, and it's the one people don't think about: **it amplifies a single bad generation into a reusable decision.**

If the model hallucinates an incorrect support procedure and the application stores it, then without caching the error affects one request; with semantic caching the same answer may be reused for many related questions. **Caching turns a single generation into a reusable decision.**

> *"So the cache shouldn't automatically store every successful model response. I'd gate admission: the response must be grounded in approved sources, or its citations verified, or it must be a deterministic template, or it must have been reviewed or received positive feedback. The principle is: do not make an answer easier to repeat before deciding whether it deserves to be repeated. Caching magnifies both quality and error."*

---

### Q12. If a vendor says "our platform has caching", what do you ask?

**Headline:** *"The word alone tells me very little."*

I'd separate four mechanisms: exact API response caching, embedding-based semantic matching, provider-side prefix reuse, and inference-engine KV preservation. They have different implications for cost, performance, privacy and correctness, and most platforms have one or two and describe them all as "caching".

Then, for semantic caching specifically: which requests are eligible? How were the similarity thresholds selected? How are users and tenants isolated? How do entries expire? What happens when source information changes? Can personalised data enter the cache? **Can an incorrect model response be reused automatically?**

For KV and prefix caching: how repetitive are the prompts, what's the context length, what's the time to first token, how much GPU memory is in play, and does the provider pass through any pricing benefit for cached prefixes?

> *"And then I'd ask for the numbers to be reported separately — hit rate, hit correctness, latency saved, cost avoided, and how many wrong or stale answers were served. A high hit rate is not a success metric if the cache is returning the wrong answer."*

---

## 12. Cheat-sheet

```text
━━━ THE ONE SENTENCE ━━━
KV caching makes inference MORE EFFICIENT. Semantic caching tries to AVOID IT.
Similarity is not equivalence.
A high hit rate is not a success metric if the cache is returning the wrong answer.

━━━ THE THREE LAYERS ━━━
KV cache      stores K/V attention tensors for the active sequence
              lookup: exact token progression · scope: ONE request
              prefill writes it, decode reads it · NEVER bypasses the model
              cost = VRAM (~0.5 MB/1k tokens, 7B fp16 · ~100 GB at 200K)
              risk = none, it's a memory problem
              managed by: PagedAttention · continuous batching · GQA/MLA · FP8 KV

Prefix cache  server-side persistence of the KV tensors of a SHARED PREFIX
              (the KV cache made shareable across requests)
              lookup: hash chain / EXACT token IDs from the START of the prompt
              runs the model only on the NEW tokens · NEVER bypasses the model
              providers bill cache hits at ~0.1x input rate
              risk = none — but a REWORDED SYSTEM PROMPT silently kills it
              ★ token IDENTITY, not MEANING. Humans say "equivalent"; tokens say "different".

Semantic cache  external store: prompt embedding → completed response
              lookup: cosine similarity ≥ threshold (source: 0.85–0.95)
              cross-request, model-independent, saves INPUT + OUTPUT tokens
              ★ BYPASSES THE MODEL ENTIRELY on a hit
              secondary wins: rate-limit relief, degraded-mode resilience
              risk = confidently wrong, stale, private, unauditable, amplified

━━━ WHO CALLS IT WHAT ━━━
Microsoft AI gateway → "reusing previous completions by comparing the vector
                        proximity of incoming prompts with earlier requests"
vLLM                 → "reusing KV-cache blocks when a new request shares the
                        same prefix with an earlier request"

━━━ THE SEVEN FAILURE MODES ━━━
1. "Close enough" ≠ equivalence        password/self vs password/admin; cancel vs cancel+refund
2. Threshold = product-risk decision   music app loose · banking tight · set from cost of error
3. Query ≠ context                     "What is my current plan?" — free vs enterprise
                                       → reusable only when the DETERMINANTS are equivalent
4. Cache preserves outdated answers    refund policy v4 → v5. Bypasses retrieval too.
                                       TTL + version tag + invalidation + bypass
5. What exactly is cached              response + PROVENANCE (model, prompt, docs, tools,
                                       tenant, version, timestamp)
                                       → the cache stores decisions the app will REPEAT
6. Privacy changes the architecture    across WHOSE requests? isolate · generic-only · classify
                                       → a cache should not change the security model
7. Caching amplifies a bad answer      one hallucination → a reusable policy
                                       → don't make it easier to repeat before deciding it deserves it

━━━ GO / NO-GO ━━━
GO      public FAQs · general explanations · product instructions · educational content
        classification · repetitive requests · stable answers · moderate similarity safe
NO-GO   live account data · financial values · medical decisions · legal determinations
        access-control answers · validity-depends-on-current-state · personalised
        rapidly changing · highly contextual · high risk
GOLD    The KV cache is useful whenever autoregressive generation occurs.
        Semantic caching is useful only when the app can safely reuse a previous result.

━━━ RUNNING BOTH ━━━
1. Is a previous ANSWER safe to reuse?   → semantic cache (application layer)
2. If not, can previous COMPUTATION be reused? → prefix cache (engine layer)
3. If neither → full inference (KV cache still helps during decode)
★ a semantic MISS can still get a KV-PREFIX HIT
★ the KV cache cannot amplify an error — it never produced a reusable answer
★ ONE is a fundamental inference optimisation. THE OTHER is an application-level policy.

━━━ WHAT LEADERS ASK A VENDOR ━━━
Exact response cache? · Embedding semantic match? · Provider prefix reuse? · Engine KV state?
Then: eligibility · threshold selection · tenant isolation · expiry · source-change behaviour
      · can personalised data enter? · can a WRONG response be reused automatically?
Then: separate numbers — hit % · hit CORRECTNESS · latency saved · cost avoided · wrong/stale served

━━━ MEASURE ━━━
hit rate · hit correctness (>0.99) · stale-served (0) · mean entry age · latency saved (P50/P95)
cost avoided · prefix hit rate · prefix cost delta · KV occupancy (headroom >20%)
amplification factor · cross-tenant leakage (0) · eligibility misclassification (0) · purge time
★ MISS PENALTY: saving_on_hits > penalty_on_misses + running_cost
  (a semantic cache makes every miss MORE expensive than no cache at all)

━━━ FOUR DIALS ━━━
Semantic cache buys COST + LATENCY, pays with QUALITY + SAFETY → a named owner, not a perf ticket.
Prefix + KV cache buy COST + LATENCY, pay only MEMORY → an engineering problem.
In a Lloyd's context the fourth dial is AUDITABILITY, not uptime.

━━━ THE CLOSING FRAME ━━━
"The important question is not merely whether it improves performance.
 The important question is what assumption makes the optimisation safe."
KV:        assumption = previously processed token state remains reusable.
Semantic:  assumption = a previous answer remains valid for a different request.
           → never leave that to a similarity score alone.
```

**Footnote on the source's citation list.** The comparison half of the source carries seven references: a LinkedIn post framing *"the four caches in an LLM serving stack"*; a Redis blog on **prefill vs decode**; a second LinkedIn post on the misunderstood prefix-cache concept; a YouTube talk; the **Modular handbook** page on prefix caching; a DailyDoseOfDS post on *KV vs Prefix vs Prompt vs Semantic*; and a Medium piece on *Semantic Cache vs KV Cache — two very different optimizations*. Note the framing discrepancy: reference [1] names **four** caches in the serving stack, while this harvest enumerates the **three** the source compares (KV, semantic, prefix/prompt). The fourth in that framing is typically the **embedding cache** on the ingestion path — worth naming if you are asked, but the source itself does not develop it.

---

## 13. Cross-links

**Within this pack**

- The system-design round this feeds, and the five cost levers in order: [`05`](05-system-design-round-playbook.md) §5.5
- Serving depth — prefill/decode, PagedAttention, continuous batching, quantisation, KV sizing numbers: [`05`](05-system-design-round-playbook.md) §5.4 and §8.5
- The RAG pipeline this sits in front of — chunking, hybrid search, RRF, reranking, freshness, version handling, audit logging: [`15`](15-rag-architecture-deep-dive.md) §6, §9, §11
- Evaluation and guardrails — faithfulness, LLM-as-judge, the gate you would reuse as a cache admission test: [`07`](07-agent-evaluation-and-guardrails.md)
- Question bank — prompt caching in one line ([§18](10-question-bank-with-model-answers.md)) and the cost-budget question ([§81](10-question-bank-with-model-answers.md)): [`10`](10-question-bank-with-model-answers.md)
- Domain constraints — FCA/PRA, Consumer Duty (FCA PS22/9), PRA SS1/23 model risk, SM&CR, data residency, sanctions screening: [`11`](11-insurance-lloyds-domain-primer.md) §3–§7
- Verified Brit facts — the authoritative sources for anything you assert about Brit: [`11`](11-insurance-lloyds-domain-primer.md) §3–§7 and [`14`](14-industry-intel-lead-ai-interview-research.md)
- The gateway layer where caching, routing, rate limits, fallbacks and observability converge: [`16`](16-agentic-ai-systems-deep-dive.md), [`17`](17-azure-ai-services-deep-dive.md)
- Day-of density (frameworks, one-pagers, the four dials): [`22`](22-quick-reference-cheatsheet.md)
- Coding-round tie-in — the ordered-dict LRU sketch and the cache-key normalisation point: [`06`](06-coding-exploration-playbook.md) §2
- The four-dial reconciliation (Accuracy · Latency · Cost · Reliability ↔ Quality · Cost · Latency · Safety): [`24`](24-stage1-strategy-deep-dive.md) § banner

**Repo-wide**

- AI-systems theory (serving, guardrails, LLMOps): [`../HLD/05-ai-systems/`](../../HLD/05-ai-systems/)
- Operations playbooks (LLMOps, observability, FinOps): [`../Operations/`](../../Operations/)

---

*Harvested from `caching` on 2026-10-12 · Target: Brit Insurance Lead AI Engineer*
