# Memory and Context Engineering

**TL;DR** — Context engineering is the discipline of deciding what occupies the model's window at each step; memory is the discipline of deciding what persists *outside* it. The production model is a **three-tier hierarchy**: **L1 working memory** (the context window and its KV cache, <50 ms), **L2 episodic memory** (past sessions and trajectories in a vector store, 100–300 ms), and **L3 semantic memory** (durable facts and relationships in a graph or SQL store, >500 ms). L1 is managed with prefix caching, sliding windows, summarization, and compaction; L2/L3 are managed with a consolidation loop that **extracts** facts, **deduplicates** them, **resolves conflicts by recency**, and **decays** what is not reinforced. The engineering insight that matters most: **a 2M-token window does not replace memory.** Attention dilutes, TTFT scales with context length, and cost scales linearly — a lean 10k-token context with strategic retrieval beats a 500k-token dump on quality, latency, and cost simultaneously. The characteristic failures are context poisoning, memory contamination across users, staleness, and memory fatigue.

> **Source grounding.** Two sources feed this file and they use *different* mental models, which is why both are here. The **nitmonk transcript** (*23+ Context Engineering & AI Agent Memory Techniques*) supplies the **context / state / memory** tri-partite split, the RAM-and-disk analogy, the six-verb decomposition of context engineering, the compaction-vs-compression distinction, the short-lived/long-lived memory taxonomy (including *environment* and *project* memory), and the named failure modes of a rotting window. The **L1/L2/L3 tier model** and the operational numbers (relevance gates, TTLs, cache economics, SLOs) are this repo's synthesis `[A]`. The two models are not in conflict — the transcript's *state* is the L1 plan, and its *semantic / user / project / environment* memories are all L3. Where the transcript states a number, it is marked `[S]`; the auto-transcript garbles digits, so any figure here has been sanity-checked before labelling.

> **60-second interview answer:** "I separate memory into three tiers. L1 is the context window — working memory — and I manage it with prefix caching, a sliding window of the last N turns, and periodic compaction, keeping static content like the system prompt and tool schemas at the front so the prefix cache hits. L2 is episodic: past sessions and successful trajectories in a vector store. L3 is semantic: extracted durable facts, often in a graph or a managed layer like Mem0 or Zep, with conflict resolution by recency and decay for unreinforced entries. The reason I do not just use a huge context window is that attention dilutes — the lost-in-the-middle effect is real — and TTFT and cost both scale with context length, so a lean context with retrieved memory beats a giant dump on all three axes. My two hard rules: context lives in the untrusted zone, so anything recalled is sanitized before it enters the prompt, and memory is namespaced per user with row-level security, because a cross-user memory leak is the worst failure this layer can have."

---

## 1. Definition & mental model

**Plain-English definition.** *Context engineering* is the set of decisions about what tokens are in the window for a given model call: system prompt, tool schemas, retrieved documents, conversation history, memory, and scratchpad. *Memory* is the durable storage that survives across turns, sessions, and processes, together with the logic that decides what to write, what to retrieve, and what to forget.

**The analogy: an office desk and a filing room.**

```
┌──────────────────────────────────────────────────────────────┐
│  L1 — THE DESK (working memory / context window)              │
│  What you are looking at right now. Small, fast, expensive    │
│  per square inch. Anything not on the desk is invisible to    │
│  the person working.                                          │
│                                                               │
│  ├── The standing lamp note: system prompt (never moves)      │
│  ├── The tool tray: tool schemas (rarely moves)               │
│  ├── Today's papers: recent turns (slides off the edge)       │
│  └── Today's sticky notes: scratchpad, current plan           │
└──────────────────────────────────────────────────────────────┘
                            │  write when done
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  L2 — THE DRAWER (episodic memory)                            │
│  "What happened in past meetings." Searchable by topic and    │
│  time. You pull a folder, not the whole drawer.               │
└──────────────────────────────────────────────────────────────┘
                            │  distilled into
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  L3 — THE FILING ROOM (semantic memory)                       │
│  Durable facts and relationships: "this client prefers        │
│  email", "this account is on the enterprise tier". Slow to    │
│  reach, authoritative, shared across desks.                   │
└──────────────────────────────────────────────────────────────┘
```

**The three ideas that make this click:**

1. **The desk is not storage.** Its job is focus. A desk covered in every document you have ever owned is not a better desk — it is an unusable one.
2. **Moving something from the drawer to the desk costs tokens, and tokens cost latency and money.** Retrieval is the act of paying that cost deliberately.
3. **Facts on a sticky note can be wrong.** Memory is not ground truth; it is a prior. Every recalled fact needs provenance and a freshness check.

**The framing sentence:** *Context engineering is the art of the small desk; memory is the art of the well-organised filing room — and the value is in knowing which three folders to pull.*

### 1.1 The source's three-way split — context, state, memory

The transcript drills a split that is *finer* than the desk analogy and is worth memorising separately, because interviewers use the three words loosely and expect you to separate them cleanly.

| | Question it answers | Where it lives | One-line definition from the source |
|---|---|---|---|
| **Context** | What can the agent see *right now*? | The model's window | *"Context is what the agent can think with."* |
| **State** | What is the agent doing *right now*? | The session/task object | *"State is what the agent is currently doing"* — the current task. |
| **Memory** | What can the agent *recall*? | Outside the window, in a store | *"Memory is what the agent can remember."* |

The source's own analogy is a **desk, a drawer, and a library** — and it is slightly different from this file's desk/drawer/filing-room diagram above in one instructive way: the transcript puts **state on the desk**, not in a tier. You are sitting at the desk looking at papers (context); what you are *trying to solve* using those papers is the state; what is in the drawer is memory. Pulling one document out of the drawer and onto the desk is exactly what **just-in-time memory** means — the document becomes context the moment it lands.

Worked example, because this is where the three get confused: a user says *"book me a ticket."* The **origin, destination, date, and time collected so far are state** — they are the current task. The **system prompt, tool schemas, and retrieved fare rules are context**. The **fact that this user always books aisle seats is memory**. If you cannot say which of the three a given piece of data is, you will mismanage its lifecycle — state must never be compacted away, context is discarded every call, and memory needs decay.

### 1.2 The hardware analogy — one diagram that makes all of it stick

This is the source's most transferable image, and it maps cleanly onto real infrastructure:

```
   ┌───────────────────────────────────────────────────────────────┐
   │  AGENT HARNESS  ≈  OPERATING SYSTEM                           │
   │  Decides what runs, when to page memory in, what the window   │
   │  shows. Schedules everything below it.                        │
   └───────────────────────────────────────────────────────────────┘
        │                    │                      │
        ▼                    ▼                      ▼
   ┌──────────┐      ┌──────────────┐      ┌──────────────────┐
   │ CONTEXT  │      │ MEMORY STORE │      │  CPU CACHE       │
   │  ≈ RAM   │      │ ≈ DISK / SSD │      │  = what was just │
   │          │      │              │      │    retrieved     │
   │ Fast,    │      │ Durable,     │      │                  │
   │ volatile,│      │ cheap, slow, │      │ Avoids a second  │
   │ small,   │      │ effectively  │      │ round trip for   │
   │ expensive│      │ unbounded    │      │ the same fact    │
   └──────────┘      └──────────────┘      └──────────────────┘
        ▲                    ▲
        │                    │
   the LLM reasons      file system / SQL / vector DB /
   here (≈ the CPU)     graph — the physical memory store
```

**Why the analogy earns its place:** a movie does not sit in RAM until you press play; code does not sit in RAM until it executes. Both rest on disk and are paged in *on demand*. That is precisely the retrieval discipline — **memory lives on disk, and only what the current step needs is paged into context.** If you find yourself arguing for "just keep everything in the window," the analogy is the counter-argument: nobody runs an operating system by loading the entire disk into RAM.

The **CPU cache** row is the part people forget: after a memory retrieval, the fetched fact should be pinned for the rest of the task so you do not pay the retrieval round-trip twice in one run.

---

## 2. Why it exists — the problem it solves

### What breaks without it

| Failure | What it looks like |
|---|---|
| No memory | Every session starts from zero; the user re-states context every time |
| Memory = full history | Context grows unboundedly; cost and TTFT climb linearly per turn |
| Naive summarization only | Key facts and formatting details are lost; the summary drifts from truth |
| Naive vector recall only | Topically similar but irrelevant facts flood the prompt — "memory fatigue" |
| No consolidation | Duplicate and contradictory records accumulate; the agent contradicts itself |
| No decay | Stale facts outlive their truth ("the user is on the free tier" — three upgrades ago) |
| No namespacing | Session A's context leaks into session B; a genuine cross-user data breach |
| No prefix discipline | Every call recomputes a 5,000-token static prefix; the cache never hits |

### The economics of getting it wrong

Consider a support agent with an 8,000-token static prefix (system prompt + tool schemas + policy docs) called 20,000 times a day.

| Design | Cost per call on the input side | Daily input cost at $3/1M in |
|---|---|---|
| No prefix caching | 8,000 input tokens billed | $480 |
| Prefix caching at 90% discount on the cached portion | 800 effective | ~$48–90 |

**A 5–10x input-cost difference from ordering the prompt correctly.** This is why context engineering is not cosmetic — it is the largest single lever most teams leave on the table.

And the cost of *too much* context is not just money:

| Context length | TTFT | Attention quality |
|---|---|---|
| 4k tokens | ~80 ms prefill | Sharp |
| 32k tokens | ~400 ms | Slight dilution |
| 128k tokens | ~1.6 s | Measurable lost-in-the-middle |
| 500k tokens | ~6 s+ | The middle is often ignored |

**Lost in the middle is the empirical result that kills "just use the big window."** Recall of information placed in the middle of a long context is materially worse than recall of information at the start or end. Padding a window with plausible-but-irrelevant history actively degrades reasoning about the current task.

### Why not just use a 2M-token window for everything?

1. **Cost** — you pay per token, every turn. A 1M-token call per turn is not a viable unit economics.
2. **Attention dilution** — more context is not more signal; irrelevant context is noise that competes with the signal.
3. **Latency** — TTFT scales with prefill length; a large window makes interactive use impossible.
4. **Determinism** — a fixed, curated context produces repeatable output; a growing one drifts.
5. **Persistence** — the window is per-request; memory must survive process restarts, deployments, and sessions.

**The architectural conclusion:** long context is a *substitute for retrieval within a session*, not a substitute for memory across sessions. Use it deliberately and briefly; do not use it as storage.

---

## 3. Internal mechanics — how it actually works

### 3.1 The three tiers, precisely

| Tier | Name | Storage | Latency | Scope | Volatility |
|---|---|---|---|---|---|
| **L1** | Working memory | Context window / KV cache | < 50 ms | Current call | Discarded after the call |
| **L2** | Episodic memory | Vector DB, local graph | 100–300 ms | This user, across sessions | Persists, decays |
| **L3** | Semantic memory | Graph DB, SQL, managed layer (Mem0/Zep) | > 500 ms | Global within a namespace | Persists, authoritative |

### 3.2 The context assembly pipeline

This is the diagram to draw when asked "how do you build the prompt?"

```
        ┌────────────────────────────────────────────────────┐
        │  STATIC PREFIX  (never changes → prefix cache hit)  │
        │  1. System prompt / role / policy                   │
        │  2. Tool schemas                                    │
        │  3. Stable few-shot examples                        │
        └────────────────────────┬───────────────────────────┘
                                 ▼
        ┌────────────────────────────────────────────────────┐
        │  SEMI-STATIC  (changes per session, not per turn)   │
        │  4. Long-term profile from L3 (top-N facts)         │
        │  5. Compressed session summary (L2/L1 compaction)   │
        └────────────────────────┬───────────────────────────┘
                                 ▼
        ┌────────────────────────────────────────────────────┐
        │  DYNAMIC  (changes every turn → cache miss region)  │
        │  6. Retrieved documents / tool results              │
        │  7. Recent turns (sliding window, last N)           │
        │  8. Current user message                            │
        │  9. Scratchpad / current plan                       │
        └────────────────────────┬───────────────────────────┘
                                 ▼
                       ┌──────────────────┐
                       │   MODEL CALL     │
                       └────────┬─────────┘
                                ▼
                  ┌──────────────────────────┐
                  │ POST: extract memory,    │
                  │ update summary, evict    │
                  └──────────────────────────┘
```

**The ordering rule:** *static first, dynamic last.* Anything that changes every turn placed early destroys the prefix cache for everything after it. This is a free performance win and the most commonly missed detail.

### 3.3 L1 management techniques

**(a) KV cache and PagedAttention.** The KV cache is the precomputed key/value tensors for tokens already processed. PagedAttention breaks the cache into fixed-size **blocks** rather than contiguous allocations, reducing memory fragmentation by **60–80%** — which directly enables larger batches and longer contexts on the same hardware.

**(b) Prefix caching.** The server keeps the KV cache for the *static prefix* resident. On a cache hit you pay and wait only for the new suffix. This matters enormously for agents: an agent making 10 tool calls sends the same 2,000-token system prompt plus tool schemas ten times.

| Scenario | Input tokens billed (cached) | Input tokens billed (uncached) |
|---|---|---|
| 10-step agent run | 2,000 + 10 × ~500 new ≈ 7,000 | 10 × 2,500 = 25,000 |
| Ratio | **1x** | **3.6x** |

**(c) Sliding window vs. summarization vs. hybrid.**

| Method | Mechanics | Pro | Con | Use when |
|---|---|---|---|---|
| **Sliding window** | Keep the last N tokens exactly | High fidelity for recent turns | "Dory" effect — the beginning is gone | Short sessions, recent context dominates |
| **Summarization** | Compress old turns into text | Preserves key facts | Loses nuance, formatting, exact numbers | Long sessions |
| **Hybrid** | Last N turns verbatim + one running summary | Best fidelity/cost balance | Slightly more complexity | **Default for production** |
| **Hierarchical** | Summary of summaries | Bounded context at any length | Each level loses detail | Very long-running agents |

The **hybrid** is the production default: keep the last ~10 turns verbatim plus a rolling summary of everything before.

**(d) Contextual compression.** Two techniques: *selective dropping* removes irrelevant reasoning blocks from prior turns (a stale chain of thought is noise), and *token pruning* uses a small model to rewrite a long user message into a roughly 50%-shorter equivalent before it reaches the reasoning model.

**(e) Compaction.** When the context crosses a threshold (typically 70–80% of the application budget), run a compaction pass: summarize the oldest span, extract any durable facts to L2/L3, drop tool outputs already summarized, and preserve the plan and current task verbatim. **Never compact the plan or the system prompt.** Compact in the background where possible so the user does not wait.

### 3.4 The application context window

The distinction interviewers test:

| | Model context window | Application context window |
|---|---|---|
| Definition | Hard architectural limit (e.g. 200k) | An engineering budget you choose (e.g. 32k) |
| Set by | Model | You |
| Purpose | Feasibility | Latency, cost, and quality control |
| Typical production value | — | 16k–64k regardless of model ceiling |

**Always leave a buffer zone** for the model's response and for the next tool result. A 32k budget with a 4k response reserve means 28k of usable input.

### 3.5 Consolidation — how memories move between tiers

```
   L1 (session)
      │
      │  1. EXTRACT — a background "reviewer" model reads the session
      │     and emits candidate facts: (subject, predicate, object)
      ▼
   Candidate facts
      │
      │  2. DEDUPLICATE — embed and compare against existing records
      │     similarity > 0.95 → treat as the same fact
      ▼
   New / duplicate / conflict
      │
      ├── 3a. NEW → write to L2 (vector) and L3 (graph/SQL)
      ├── 3b. DUPLICATE → reinforce (bump confidence, update last_seen)
      └── 3c. CONFLICT → supersede by recency, keep the old as history
      │
      │  4. DECAY — unreinforced memories lose confidence; below a
      │     floor they are archived or deleted
      ▼
   L2 / L3 durable store
```

**The fact-shaped representation.** Store `(User, Preferred_Mug_Color, Blue)`, not "the user said they like blue coffee mugs." Facts are deduplicable, conflict-resolvable, graph-queryable, and cheap to inject. Raw transcripts are none of those things.

**The temporal dimension.** Facts need `valid_from`, `valid_to`, `last_reinforced`, `confidence`, and `source_turn`. Without temporality you cannot answer "what did we discuss last Tuesday" and you cannot resolve a contradiction — you just have two records and no rule.

**Decay policy:**

| Memory class | TTL | Rationale |
|---|---|---|
| Ephemeral chatter | 24 hours | "It's raining" is not a fact about the user |
| Session summaries | 90 days | Useful for continuity, then stale |
| Stable preferences | Indefinite, decaying confidence | Re-reinforced on every confirmation |
| Task/goal nodes | Until completed + 30 days | Drives proactive reminders |
| Explicit user instructions | Indefinite, pinned | Never decay — the user said so |

### 3.6 Memory backends

| System | Model | Best for | Trade-off |
|---|---|---|---|
| **Raw vector store** | Embed and search | Simple recall, you own the logic | You implement dedup, conflict, decay |
| **Knowledge graph** | Entities + relations | Multi-hop factual queries, explainability | Schema design and extraction cost |
| **Mem0** | Extract → dedup → merge loop | Broadest standalone memory layer | Managed dependency; per-fact cost |
| **Zep** | Temporal knowledge graph | Production pipelines needing time-awareness | More setup; graph modelling required |
| **Letta** | OS-style paging of memory blocks | Long-running agents with a persistent identity | Heavier runtime |
| **Cognee** | Knowledge-graph-first | Graph RAG plus memory in one layer | Graph-first is not always the right shape |
| **Hybrid** | Vector for recall + graph for facts | Most production systems | Two systems to operate |

**Why not just write to Postgres yourself?** You can, and for a narrow use case you should. The managed layers earn their keep on the parts that are genuinely hard: **entity linking** across identity surfaces (the same person is `om` in Slack and `om.bharatiya` in email), **temporal weighting** that prioritises new facts without losing history, and **deduplication** that does not silently create six records for one preference. If you build it yourself, budget for those three specifically — they are the ones that fail in production, not the storage.

### 3.7 Retrieval from memory — relevance gating

Recalled memory is *injected content*, which means it competes with the task for attention. Memory fatigue — the agent bringing up irrelevant past details — is a real and common quality regression.

**The gating policy:**

| Gate | Threshold | Effect |
|---|---|---|
| Minimum relevance score | **0.85** for preference facts | Below this, do not inject |
| Maximum injected facts | 5–10 | Caps attention cost |
| Recency weighting | Newer facts rank higher at equal similarity | Prevents stale dominance |
| Negative retrieval | Inject memory only when it answers a current unknown or contradicts a likely hallucination | Avoids gratuitous recall |
| Contradiction check | If recalled fact conflicts with retrieved documents, **documents win** and the memory is flagged | L3 is a prior, not ground truth |

**The precedence rule to memorise:** retrieved documents > L3 semantic memory > L2 episodic memory > L1 conversational assumption. A stale preference must never override a fresh authoritative document.

### 3.8 Agent state management

**The state object is the single source of truth for a session.**

```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]   # append-only
    plan: list[str]                                        # never compacted
    current_task: str                                      # never compacted
    tool_results: dict[str, Any]                           # pruned after use
    user_context: dict[str, Any]                           # namespaced
    iteration_count: int
    budget: dict[str, Any]                                # steps/$, consumed
```

**Two rules:** state should be **strictly typed** (an untyped dict is where bugs hide) and **append-only where possible** so a long run cannot lose data through an overwrite.

**State machine vs. while loop.** Production has converged on cyclic graphs (LangGraph-style) rather than a bare loop:

| | While loop | State graph |
|---|---|---|
| Visualisable | No | Yes — it renders as a diagram |
| Unit-testable | Poorly | Per node |
| Backtracking | A mess of nested conditions | A new edge |
| Parallel execution | Manual | Fork/join primitives |
| Persistence | You write it | Handled between nodes |
| Debuggability | Low | High — the path is data |

**Checkpointing and resume.** Every state transition is persisted (Postgres for durability, Redis for speed). If the worker crashes, the orchestrator reloads the last checkpoint and resumes. This is what makes async agents possible: the user gets "I'm working on it" and a notification ten minutes later when the state is `complete`. It is the same primitive as durable execution — a checkpoint is a durable activity boundary.

**Fork/join.** Fan out the state to N sub-workers, then a join node merges their outputs back into the parent state. The join must be explicit about conflict resolution: last-write-wins is a bug, not a policy.

**Time travel.** Because every transition is checkpointed, you can load a past state, edit it, and re-run the graph from that point. This is the mechanism behind human-in-the-loop correction and behind replaying a production failure with a fix.

**State bloat prevention:**

| Technique | Detail |
|---|---|
| Prune `tool_results` | Drop a sub-task's raw results once it is summarized into the plan |
| Summarizer node | Every ~10 turns, compress history into one block |
| Cap the plan | A plan over ~20 steps means the task should be decomposed differently |
| Bounded scratchpad | Keep only the current step's reasoning, not the whole chain |
| Artefact references | Store large outputs externally and keep a pointer |

### 3.9 The budget of a real window — what must be in it, and what has headroom

The source walks the window line by line and asks of each item: *does this have to be here every single call?* That question is the whole discipline. The answer produces a budget with very different eviction rules per line.

| Line in the window | Must it be present? | Evictable? | Notes |
|---|---|---|---|
| **System / agent instruction** | Yes, every call | **Never** | The source is emphatic: it is passed every time and is never evicted. It is also the anchor of your prefix cache. |
| **Tool definitions** (name, description, input schema) | Only the ones in scope | Yes — **scope them down** | This is the line that breaks at scale. See the tool-scoping discussion below and file 7's *progressive discovery*. |
| **Conversation** (user turns + model responses) | Recent ones | Yes — window it | Grows without bound if you let it. |
| **Retrieved RAG content** | For this step | Yes — per step | Retrieved for the turn, discarded after. |
| **Tool results** | Until consumed | Yes — aggressively | The source names tool-result explosion as a top cause of window rot. |
| **Headroom for reasoning + output** | Must be *reserved* | — | Not content: a reserved region so the model has room to think and answer. Never fill the window to the brim. |

The two lessons to carry: **the system prompt and tool schemas are your stable prefix and belong at the front**, and **tool results are the fastest-growing, least-valuable line — evict them first.** The source calls the practice of trimming tool output *pruning*: you keep the conclusion, discard the raw dump.

### 3.10 Context engineering is six verbs, not one

The most useful definitional framing in the transcript: context engineering is not a single act but a pipeline of six, and every technique you know belongs to exactly one of them. It is a good structure to answer with, because it shows you have a map rather than a bag of tricks.

| Verb | Question | Techniques that live here |
|---|---|---|
| **Design** | What should the window's *shape* be? | The static/semi-static/dynamic layout; where the system prompt sits; the budget split |
| **Select** | Which of the candidate items get in? | Tool scoping, relevance gating, negative retrieval, top-k on retrieval |
| **Retrieve** | How do I *find* the right memory or document? | Hybrid search (BM25 + semantic), lexical search, re-ranking, metadata filters — see file 2 |
| **Compress** | How do I shrink what's already in? | Summarisation, token pruning, contextual compression |
| **Budget** | How many tokens does each line get? | The reservation table above; compaction trigger; per-line caps |
| **Order** | Where in the window does each item go? | Static-first ordering; position matters — models attend unevenly across a long window |

**Order is the verb candidates forget.** The source states it plainly: *position matters — the model attends unevenly across a long window.* That is the same empirical fact as lost-in-the-middle, and it is the reason the ordering rule in §3.2 is not merely a cache optimisation.

### 3.11 Compaction is not compression — and the difference is state

The transcript flags this as *"a major difference"* and it is the distinction most candidates blur. Both shrink the window; only one is safe to do on a long-running agent.

| | **Compression** | **Compaction** |
|---|---|---|
| **What it does** | Paraphrases / rewrites the text — the source's word is *paraphrasing the transcript* | **Extracts the important *state* and rebuilds a smaller context around it** |
| **What survives** | The gist | The **state**, deliberately and structurally |
| **Failure** | Details and exact values drift or vanish | Can still lose unextracted detail at the boundary |
| **Shape of the output** | Prose, shorter | Prose *plus structured nodes* |

The mechanism the source describes for compaction — and which explains why it is state-preserving — is **keep the tail verbatim, replace the middle with a structured node.** Recent turns stay exact because they carry the live task; the middle of a long session becomes a compact, machine-updatable *record* rather than a paragraph. Prose summaries of the middle are what drift.

**The structured note.** A related idea from the transcript: instead of keeping prose, keep a *compact machine-updatable record*. A prose summary of "we discussed the database migration, the user prefers Postgres, there were concerns about the index" is lossy and un-editable. `{migration: done, db: postgres, open_concern: index_size}` is updatable, de-duplicable, and survives a re-summarisation intact. This is the same argument as the fact-shaped representation in §3.5, arrived at from a different direction.

**Compaction is what real coding agents do.** The source points at the compaction behaviour visible in Claude Code and Cursor as the production instance of this technique — a first-party agent hitting its budget, extracting state, and rebuilding a smaller window mid-task.

### 3.12 Refresh, isolation, prune, order — the four maintenance operations

Beyond compaction, the source names four ongoing operations. They are cheap to remember and each has a distinct trigger:

| Operation | What it is | Triggered by |
|---|---|---|
| **Refresh** | Periodically **reconstruct the window from scratch** after appending, rather than only trimming it | Drift — a window that has been appended to for hours is not the window you would have designed |
| **Isolation** | Give each agent its **own scope window** so context does not bleed between agents | Any multi-agent system; more urgently with A2A protocol, where two agents each carry their own context, state, and memory |
| **Pruning** | Discard context you do not need, keep what you do | Tool results especially — the source is blunt that without pruning, a tool returning thousands of lines floods the window |
| **Ordering** | Place content where the model attends best | Any long window |

**Isolation deserves emphasis** because it is the multi-agent analogue of the cross-user leak in §6.2, and the source raises it in exactly those terms: *how will you make sure the context of this agent and that agent does not get collared?* When agent A's working context — including anything sensitive it has seen — flows into agent B's window, you have created an implicit data-sharing channel between two components that were never designed to share. With A2A, each agent has its own context, state, and memory; the isolation boundary is what keeps them separate, and the privacy/RBAC arguments are the same arguments as namespacing.

### 3.13 Just-in-time memory, and why the CPU cache row matters

**Just-in-time memory** is the source's name for the pattern this file calls retrieval: rather than pre-loading everything that *might* be relevant, you hold it in the store and page it in at the moment a step needs it — *retrieve, inject, or reason*. The document in the drawer becomes context only when it lands on the desk.

The reason to name it explicitly is that it makes the *cost* transparent: every page-in is tokens, latency, and a retrieval round-trip. That is why the source pairs it with a cache — once you have paid to page a fact in, pin it for the remainder of the task, so step 7 of a ten-step run does not re-retrieve what step 3 already fetched. In the hardware analogy of §1.2 this is the **CPU cache**, and it is the cheapest context-engineering win available: the fact is already known-good and already in the window, so re-fetching it is pure waste.

---

## 4. Variants & taxonomy

### 4.1 Memory types

| Type | Content | Storage | Example |
|---|---|---|---|
| **Working (L1)** | Current focus | Context / KV cache | This turn's messages |
| **Episodic (L2)** | Events and trajectories | Vector DB | "Last Tuesday we debugged the deploy" |
| **Semantic (L3)** | Facts and rules | Graph / SQL | "Account tier: enterprise" |
| **Procedural** | How to do things | Prompt / fine-tune / tool set | Learned workflows and heuristics |
| **Prospective** | Future intentions | Goal nodes with dates | "Remind me about the budget Friday" |
| **Entity/identity** | Who the user is | Profile store | Name, role, org, locale |

#### The source's taxonomy — short-lived vs long-lived

The transcript splits memory by **lifetime** first, then by kind, and the split is worth keeping because it gives you a place to put classes the table above misses. The source's own examples are in the right-hand column and they are the reason this is useful: *environment memory* and *project memory* are real, load-bearing categories that most candidates never name.

| Lifetime | Class | What it holds | The source's own example |
|---|---|---|---|
| **Short-lived** | **Working memory** | The active working set for the task in hand — *"whatever is on my desk"* | *"Build proper rate limiting for the tools"* → it pulls the coding files, PDFs and docs it needs and works on that |
| **Short-lived** | **Session memory** | What happened earlier in *this* session | *"What happened to that OAuth problem I asked about 30 minutes ago?"* |
| **Long-lived** | **Episodic** | Dated events and experiences — what happened, when | *"Last week the Stripe integration failed"*; *"on 15 June this broke"* |
| **Long-lived** | **Semantic** | Durable facts, independent of when you learned them | *"We migrated from MySQL to Postgres"* → stored as the fact `db = postgres` |
| **Long-lived** | **Procedural** | How to carry out a recurring task — the ordered steps | *"Build me a Next.js app with Firebase"* → understand Firebase → enable Google login → implement the admin SDK → integrate the client SDK |
| **Long-lived** | **Task / project** | What this project *is* | Which database, which cloud, which language, what the `CLAUDE.md` says, whether it uses GCP or Firestore |
| **Long-lived** | **User** | The user's own habits and standing preferences | *"They code in Next.js most of the time"*; *"for agent orchestration, always reach for LangGraph"* |
| **Long-lived** | **Environment** | What exists in the working environment the agent operates in | GitHub, Jira, AWS tooling, the repo itself, and the package managers — `uv`, `pip`, `npm` |

**Two insights from this table that the tier model alone does not give you.**

**1. Environment memory is a first-class category for *coding* agents.** A Cursor or Claude Code session that does not know it has `uv` rather than `pip`, or that GitHub and Jira are reachable, will propose work it cannot do. This is not user preference and it is not project fact — it is the agent's model of its own tooling, and it is discovered, cached, and refreshed rather than authored.

**2. Episodic memory sediments into procedural memory.** The source's phrasing: episodic memory *"later gets collected as a procedure."* A one-off episode — *"last time, the Stripe webhook failed because the signature was checked before the body was parsed"* — is an event. When the same shape recurs, it stops being an episode and becomes a *procedure*: check the signature against the raw body first. That consolidation path (episode → pattern → procedure) is a genuinely different write policy from the extract-dedupe-decay loop in §3.5, and it is how an agent gets better at a task rather than merely more informed about a user.

### 4.2 Context management strategies

| Strategy | Mechanism | Bounded? | Fidelity | Cost |
|---|---|---|---|---|
| **Full history** | Send everything | No | Perfect | Grows linearly |
| **Sliding window** | Last N tokens | Yes | Recent only | Constant |
| **Summarization** | Compress old turns | Yes | Lossy | Constant + summarizer cost |
| **Hybrid** | Window + rolling summary | Yes | High | Constant + low |
| **Hierarchical** | Summary of summaries | Yes | Lossy per level | Constant |
| **Retrieval-augmented context** | Retrieve relevant past turns per query | Yes | Query-dependent | Retrieval + tokens |
| **Compaction on threshold** | Compress at 70–80% | Yes | Lossy at boundaries | Bursty |

### 4.3 Memory write policies

| Policy | When written | Pros | Cons |
|---|---|---|---|
| **Write-through** | Every turn | Nothing is missed | Cost, noise, duplicates |
| **Write-on-idle** | At session end | Cheaper, cleaner | Risk of losing a crashed session |
| **Write-on-signal** | When an extractor detects a fact | Cheap, precise | Depends on extractor quality |
| **Background consolidation** | Periodic batch job | Best quality of the three | Latency to availability |
| **Explicit user instruction** | On demand ("remember this") | Highest precision, user-controlled | Requires UX |

**Production default:** write-on-signal for immediate facts, plus background consolidation for deduplication, decay, and cross-session entity linking.

### 4.4 Namespacing and isolation models

| Model | Isolation | Use |
|---|---|---|
| **Per-user namespace** | Strongest | Consumer apps — the default |
| **Per-tenant namespace** | Strong | B2B SaaS with shared org knowledge |
| **Per-tenant + per-user subspace** | Strong, with sharing | Enterprise: org knowledge plus personal prefs |
| **Global with ACL filters** | Weakest — one filter bug is a breach | Only where the corpus is genuinely public |
| **Row-level security at the DB** | Defence in depth beneath the app | **Always, in addition to namespacing** |

---

## 5. Pros / Cons

| Pros | Cons |
|---|---|
| A lean, curated context improves quality, latency, and cost simultaneously | Curation is engineering work, not a config flag |
| Prefix caching cuts input cost 5–10x on static-heavy agent prompts | Requires discipline in prompt ordering; one dynamic token early kills it |
| Memory enables personalisation and continuity across sessions and surfaces | Every stored fact is a privacy liability and a poisoning surface |
| Fact-shaped memory is deduplicable, conflict-resolvable and cheap to inject | Extraction quality determines everything; a bad extractor stores bad facts |
| Compaction bounds context for arbitrarily long sessions | Compaction is lossy at boundaries and can drop the one detail that mattered |
| Checkpointed state graphs make long-running agents resumable and auditable | Persisting state adds a store, a schema, and a migration story |
| Explicit relevance gating prevents memory fatigue | Threshold tuning is empirical; 0.85 is a starting point, not a law |
| Namespacing plus row-level security makes multi-tenant memory defensible | Two layers of isolation to maintain; both must be tested |
| Time travel makes human correction and failure replay practical | State rewriting means the audit trail must record the rewrite |
| Decay keeps the store from accumulating stale truth | Aggressive decay loses facts users expect you to remember |

---

## 6. Exceptions, edge cases & failure modes

### 6.0 The organising idea: context grows, and then it rots

The single most quotable framing in the source is that **a context window does not fail by overflowing — it fails by rotting.** Size and quality move in opposite directions, and the failure is gradual, which is what makes it dangerous: nothing errors, quality just declines. The transcript names the decay modes, and each one has a distinct cause and a distinct remedy.

| Decay mode | The source's description | First remedy |
|---|---|---|
| **Overload** | Too much information in the window at once — the model cannot attend to all of it | Budget; never fill the window |
| **Dilution** | *"The one crucial line is buried amongst thousands of irrelevant ones"* | Selection; relevance gating; ordering |
| **Degradation / rot** | *"Effective performance drops as the window gets noisy"* — long-context degradation | Compaction; refresh |
| **Redundancy** | The same fact repeated dozens of times across accumulated turns | Deduplication; structured notes |
| **Tool-result explosion** | An unpruned tool returns thousands of lines and floods the window | Prune tool results to the conclusion |
| **Staleness** | Information that *was* true and no longer is — *"the file has since changed"* | Freshness checks; documents beat memory |

Notice that **only one of these is a security problem** and the rest are engineering hygiene — which is why "the window is too long" is the wrong diagnosis. The right question is *which of the six is happening*.

**The remedies the source names.** The transcript introduces these as the steps to keep a window healthy. The clearly-named ones are: the **sliding window** — and the source is explicit that this is the answer to give first in an interview, and that you window **by token budget, not by turn count**; **summarisation**; **compaction** (§3.11); **structured notes** (keep a machine-updatable record, not prose); and **retrieval** — *"don't keep information hot at all; store it and fetch it back only when a step needs it."* Together with the maintenance operations in §3.12 (refresh, isolation, prune, order), that is the full remedy set. The auto-transcript garbles the enumeration, so treat this as the consolidated list rather than a verbatim seven.

### 6.1 Context poisoning

**Symptom:** a retrieved document, a tool result, or a prior turn contains injected instructions, and the model follows them.
**Root cause:** context is the **untrusted zone** — anything in it can carry instructions.
**Mitigation:** sanitize all retrieved content; delimit and label it as data; never let context influence authorisation or the available tool set; apply the guardrail layers from file 4; keep a provenance record for every injected item.

**The source's three-way poisoning taxonomy — know the difference, because the defences differ.**

| Poisoning | Where the payload hides | Why it is dangerous | Primary defence |
|---|---|---|---|
| **Memory poisoning** | Written into *memory* as a stored fact | Outlives the session and is re-injected confidently in every future one | Extract from user turns only; sanitize before extraction; anomalous-write-rate alerting (§6.13) |
| **RAG poisoning** | In the *corpus* — a document a retriever will surface | Affects every user who retrieves it, not just one | Corpus access control; provenance; injection screening on retrieved chunks (file 4) |
| **Tool poisoning** | In a tool's *name or description* | The agent's selection logic is the attack surface — a description crafted to make the model pick and misuse a tool | Tool-catalogue review; least-privilege tool scoping; see file 7 |

The reason to keep them separate: a corpus is curated by you and shared by everyone, memory is written by the agent and scoped to one user, and a tool description is *code you installed*. Three different trust boundaries, three different review processes. Collapsing them into "prompt injection" means you defend the wrong boundary.

### 6.2 Memory contamination across users

**Symptom:** user A's recalled fact appears in user B's context — a genuine data breach.
**Root causes:** a missing namespace filter, a shared cache key that omits the user ID, a similarity search over a global index, a copy-paste bug in the memory retrieval node.
**Mitigation:** namespace at write **and** read; row-level security at the database; the namespace filter applied *before* the similarity search, never after; a cache key that includes the namespace; and an automated test that runs a cross-user recall probe in CI. **This is the single most important test in the memory layer.**

### 6.3 Staleness

**Symptom:** the agent confidently states a preference or fact the user changed months ago.
**Mitigation:** `last_reinforced` timestamps; recency weighting in ranking; explicit supersession on conflict; decay for unreinforced facts; and — critically — **documents beat memory** when they disagree. Show the user where the fact came from so they can correct it.

### 6.4 Contradictory memories

**Symptom:** two records say different things and the agent picks arbitrarily (usually whichever ranks higher by an accident of embedding distance).
**Mitigation:** a conflict-resolution policy that is explicit rather than implicit: highest confidence wins; ties broken by recency; the loser is kept as history with a `superseded_by` pointer. Never silently delete the old fact — you lose the ability to answer "you told me X before."

### 6.5 Memory fatigue

**Symptom:** the agent volunteers irrelevant past details, cluttering answers and degrading trust.
**Mitigation:** relevance gate at **0.85**; cap injected facts at 5–10; negative retrieval — inject only when the memory answers a current unknown or contradicts a likely hallucination; delete low-value memories after 24 hours.

### 6.6 Context bloat and mid-session degradation

**Symptom:** quality is good for the first ten turns and quietly degrades after; the model starts ignoring the system prompt.
**Root cause:** context growth past the point where attention dilutes, plus lost-in-the-middle effects.
**Mitigation:** compaction at 70–80% of budget; keep the system prompt and plan at the front and re-anchor them in the summary; monitor context length as a *quality* signal, not just a cost signal; alert when p95 context length crosses 70% of budget.

### 6.7 Compaction data loss

**Symptom:** after compaction, the agent forgets a constraint it was told ten turns ago and re-asks or violates it.
**Mitigation:** never compact the plan, the current task, or explicit user instructions; extract durable facts to L2/L3 *before* discarding the source; verify post-compaction by asking an automated probe whether key constraints survive; keep the pre-compaction transcript retrievable.

### 6.8 Summary drift

**Symptom:** over many summarization rounds the summary drifts from the truth — details mutate, numbers change, hedges become assertions.
**Mitigation:** summarize from the **source turns**, not from the previous summary, wherever the source is still available; otherwise cap the summary-of-summary depth at one level; include exact numbers verbatim rather than paraphrased.

### 6.9 Prefix cache misses

**Symptom:** input costs stay high and TTFT is worse than expected despite prefix caching being enabled.
**Root causes:** a timestamp, session ID, or user name at the top of the prompt; tool schemas serialized in nondeterministic key order; a retrieved document placed before the static block.
**Mitigation:** byte-identical static prefix; deterministic JSON serialization with sorted keys; cache-key versioning so a prompt change is a deliberate cache flush; monitor the cache hit rate as a first-class metric.

### 6.10 Memory write amplification and cost

**Symptom:** the memory layer costs more than the agent. Every turn triggers an extraction model call, a dedup embedding, a vector write, and a graph update.
**Mitigation:** write-on-signal rather than write-through; batch consolidation in a background job; use a small model for extraction; cap extraction frequency per session; measure the memory layer's cost as a share of the total and keep it under ~10% for chat workloads.

### 6.11 Cross-surface identity fragmentation

**Symptom:** the same person is three different entities; preferences set in the web app do not apply in Slack.
**Mitigation:** an explicit identity graph with a canonical `user_id` and linked external identifiers; entity linking at consolidation time; a reconciliation job for merges; and a rule for what happens when two entities merge (union of facts, newest wins on conflicts).

### 6.12 State bloat in long-running agents

**Symptom:** the state object grows to megabytes, checkpoints get slow, resume takes seconds.
**Mitigation:** prune `tool_results` after each sub-task into the plan; store large artefacts externally with a pointer in state; cap the messages list via a summarizer node every ~10 turns.

### 6.13 Poisoned memory persistence

**Symptom:** an injected instruction from a retrieved document becomes a *stored fact*, and now it influences every future session. This is the worst version of prompt injection because it outlives the session.
**Mitigation:** the extraction step must consume only **user turns**, never model output and never retrieved content; sanitize before extraction; a review queue for high-impact facts; the ability to quarantine and purge by source; alert on anomalous fact write rates.

### 6.14 PII leaking into durable memory

**Symptom:** a credit card number, password, or health detail is extracted and stored indefinitely.
**Mitigation:** a PII scrubbing layer **between L1 and L3** — not just at the input boundary; classify and either drop or tokenize sensitive fields; never store credentials in any tier; enforce retention limits per data class; support deletion requests end-to-end (the store, the graph, the cache, and backups).

### 6.15 Memory layer as a single point of failure

**Symptom:** the memory service is down and the agent fails entirely, when it could have degraded to a stateless turn.
**Mitigation:** fail-soft — if memory retrieval times out (budget ~200 ms), proceed without it rather than failing the request; cache the last known profile; log degraded turns so you can measure the quality impact.

### 6.16 Silent quality degradation

**Symptom:** everything is green and answers get quietly worse. Memory is the usual culprit: facts went stale, decay removed something important, or the extractor started producing lower-quality facts after a model upgrade.
**Mitigation:** a memory-quality eval on a frozen set (does recall return the right fact; is contradiction rate rising; is the injected-fact relevance distribution stable); alert on the *distribution* of relevance scores, not just errors; run the same probe across model versions.

### 6.17 False memory — a fact that was never true

**Symptom:** the agent asserts something the user never said and no document supports. Distinct from staleness (§6.3, which was true once) and from contradiction (§6.4, where two records disagree). Here there is a single record and it is simply wrong.
**Root causes:** an extractor that inferred rather than transcribed — the user mused about *maybe* migrating databases and the extractor stored `db = postgres`; a summariser that turned a hedge into an assertion (§6.8 is the same disease at a different layer); entity-linking that attached a fact to the wrong person.
**Why it is worse than stale memory:** a stale fact has a `last_reinforced` date you can argue about. A false fact has provenance pointing at a turn that does not contain it, so nobody can tell it is wrong by inspection.
**Mitigation:** extraction must record the **source turn ID verbatim** so any fact can be traced back and audited; distinguish *asserted* from *inferred* facts and store the inference confidence separately; never store a fact from a hypothetical or a question; periodic sampled audit of stored facts against their source turns.

### 6.18 Retrieval failure — the right memory exists and is not found

**Symptom:** the fact is in the store, the agent does not recall it, and the user says *"I already told you this."*
**Root causes:** the query is embedded in the wrong register (a terse question retrieving nothing from a verbose stored fact); the namespace filter is too tight; top-k is too small and the right fact ranks just outside it; the embedding model was upgraded and the old vectors are in a different space.
**Why it is easy to miss:** every metric you are likely to have — write success, retrieval latency, store size — looks healthy. Only end-to-end recall quality catches it, which is why §9's testing answer probes recall rather than retrieval.
**Mitigation:** hybrid retrieval (lexical + semantic) rather than pure vector, because exact identifiers and names fail vector search; a generous top-k with aggressive re-ranking rather than a tight top-k; a re-embedding migration plan whenever the embedding model changes; and recall@k as a tracked eval metric.

### 6.19 Retrieval pollution — the wrong memory is found and it crowds out the right one

**Symptom:** the agent recalls something topically adjacent and irrelevant, and the genuinely relevant fact never makes it in. This is the mechanism underneath memory fatigue (§6.5).
**Root causes:** similarity is not relevance — "the user mentioned Postgres in a planning meeting" and "the user's database is Postgres" are near-identical vectors with completely different standing; no recency weighting; millions of low-value memories diluting the index.
**Mitigation:** the relevance gate and recency weighting from §3.7; **negative retrieval** — inject memory only when it answers a current unknown, rather than on every turn; a hard cap of 5–10 injected facts; and eviction of low-value memories (`millions of low-value memories accumulate` is the source's own warning), because index dilution is the cause and the gate is only the symptom treatment.

---

## 7. When to use / When NOT to use

### Memory tier decision table

| Need | L1 only | L1 + L2 | L1 + L2 + L3 |
|---|---|---|---|
| Single-turn Q&A | ✅ | ❌ | ❌ |
| Multi-turn chat in one session | ✅ | ❌ | ❌ |
| Returning user, personalised | ❌ | ✅ | ✅ |
| Cross-surface continuity | ❌ | ✅ | ✅ |
| Factual queries over user/organisational knowledge | ❌ | ⚠️ | ✅ |
| Multi-hop reasoning over relationships | ❌ | ❌ | ✅ |
| Long-running agent with goals over days | ❌ | ✅ | ✅ |

### When NOT to add memory

- **The session is genuinely stateless** — a one-shot classifier or extractor. Memory adds a store, a privacy surface, and a failure mode for zero benefit.
- **The "memory" is really a document** — if the user's facts live in a CRM, query the CRM. Do not duplicate it into a memory store where it can go stale.
- **You cannot enforce namespacing yet** — do not ship multi-tenant memory without isolation and a cross-user probe test. This is a breach waiting to happen.
- **The task is latency-critical** (p95 budget under 500 ms) — an L3 graph lookup at >500 ms does not fit. Use L1 with a cached profile.
- **You have no extraction quality story** — a memory layer that stores bad facts is worse than no memory, because bad facts persist and get injected confidently.
- **Regulatory constraints forbid retention** — if you must not retain, do not build retention. Ephemeral context is a valid design.

### Context strategy decision table

| Situation | Strategy |
|---|---|
| Short sessions, < 10 turns | Full history |
| Long sessions, recent context dominates | Sliding window |
| Long sessions, early details matter | Hybrid (window + rolling summary) |
| Very long-running agent | Compaction on threshold + hierarchical summary |
| Large static prefix, high call volume | Prefix caching + strict static-first ordering |
| Long user inputs, cheap reasoning | Token pruning with a small model |

---

## 8. Numbers, defaults & tuning

### 8.1 Context budgets

| Parameter | Default | Range | Rationale |
|---|---|---|---|
| Application context budget | 32k tokens | 16k–64k | Independent of the model ceiling |
| Reserved for response | 4k tokens | 2k–8k | Never fill the window |
| Reserved for next tool result | 2k tokens | 1k–4k | Tool output is the fastest way to overflow |
| Sliding window (verbatim turns) | 10 turns | 6–20 | Fidelity for recent context |
| Compaction trigger | 70–80% of budget | 60–85% | Compact before degradation |
| Summary-of-summary depth | 1 level | 0–2 | Each level loses detail |
| Summarizer cadence | Every 10 turns | 5–20 | Balance cost vs. freshness |
| Max injected retrieved docs | 5–10 | 3–15 | Attention cost |
| Max injected memory facts | 5–10 | 3–15 | Memory fatigue guard |

### 8.2 Memory thresholds

| Parameter | Default | Rationale |
|---|---|---|
| Memory relevance gate | **0.85** | Below this, fatigue outweighs value |
| Deduplication similarity | **0.95** | Above this, treat as the same fact |
| Conflict resolution | Higher confidence wins; ties → recency | Explicit, not emergent |
| Decay: ephemeral chatter TTL | 24 hours | Not a fact about the user |
| Decay: session summary TTL | 90 days | Continuity horizon |
| Decay: stable preference | Indefinite, decaying confidence | Reinforced on confirmation |
| Decay: goal nodes | Completion + 30 days | Proactive reminder window |
| Decay: explicit user instruction | Never | The user said so |
| Memory retrieval timeout | 200 ms | Fail-soft threshold |
| Memory layer cost share | < 10% of total | Write amplification guard |
| Cross-user probe test | Every CI run | The one non-negotiable test |

### 8.3 Memory layer performance targets

| SLI | Target | Alert |
|---|---|---|
| Memory write p95 | < 300 ms | > 800 ms warn |
| Memory retrieval p95 | < 200 ms | > 500 ms warn |
| Cross-namespace recall | **0 occurrences** | Any occurrence = critical, page |
| Duplicate rate | < 5% of facts | > 15% warn |
| Contradiction rate | < 3% | > 10% warn |
| Injected-fact relevance p50 | > 0.90 | < 0.85 warn |
| Memory-layer cost share | < 10% | > 20% warn |

### 8.4 Prefix cache efficiency

| Metric | Target | Why |
|---|---|---|
| Static prefix size | 2k–8k tokens | Large enough to matter, small enough to be stable |
| Cache hit rate | > 0.80 | Below 0.5 means the prefix is not actually static |
| Cache discount on hit | ~0.1x input price | The economics of the whole technique |
| Prefix change frequency | Weekly or less | Every change is a full cache flush |
| Prompt version key | In the cache key | So a prompt change deliberately invalidates |

### 8.5 Alerts

```yaml
memory_alerts:
  - name: cross_namespace_recall
    condition: recall_result_namespace != request_namespace
    severity: critical
    action: page_oncall_and_security, disable_memory_reads
    note: this is a data breach, not a bug

  - name: context_length_p95_high
    condition: p95_context_tokens > 0.7 * budget
    severity: warning
    note: quality signal, not just cost

  - name: compaction_failure_rate
    condition: compaction_failures / compaction_attempts > 0.05
    severity: warning

  - name: memory_retrieval_timeout_rate
    condition: memory_timeouts / requests > 0.02
    severity: warning
    note: turns are degrading to stateless

  - name: memory_contradiction_rate
    condition: contradictions / facts_recalled > 0.10
    severity: warning

  - name: memory_write_rate_anomaly
    condition: facts_written_per_session > 3 * trailing_mean
    severity: warning
    note: possible injection-driven memory poisoning

  - name: injected_fact_relevance_drop
    condition: p50_relevance < 0.85 for 24h
    severity: warning

  - name: prefix_cache_hit_low
    condition: prefix_cache_hit_rate < 0.5 for 30m
    severity: warning
    note: usually a nondeterministic prefix

  - name: pii_in_memory_write
    condition: pii_classifier_positive > 0
    severity: critical
    action: quarantine_fact, page_security

  - name: state_object_size_high
    condition: p95_state_bytes > 5_000_000
    severity: warning
```

---

## 9. Interview Q&A

**Q1. What is context engineering, and how is it different from prompt engineering?**
Prompt engineering is about *how you phrase* an instruction. Context engineering is about *what is in the window* when the model runs — the system prompt, tool schemas, retrieved documents, memory, recent turns, and scratchpad — plus the mechanics of managing that set over time: prefix caching, sliding windows, summarization, compaction, and eviction. Prompt engineering optimises a string; context engineering optimises a pipeline and a lifecycle. In production, context engineering is where the leverage is, because a perfectly-worded instruction surrounded by 200k tokens of noise will lose to a mediocre instruction with a clean 8k context.

**Q2. Why not just use the largest context window available?**
Four reasons. Cost scales linearly with tokens on every turn — a 500k-token call per turn is not viable unit economics. TTFT scales with prefill, so latency becomes unacceptable for interactive use. Attention dilutes — the lost-in-the-middle effect is empirical, and recall of information in the middle of a long context is materially worse than at the start or end. And the window is per-request, so it does not give you persistence across sessions. Long context is a substitute for retrieval *within* a session, used deliberately; it is not a substitute for memory.

**Q3. Describe your memory architecture.**
Three tiers. L1 is working memory — the context window and KV cache, sub-50ms, discarded after the call. L2 is episodic: past sessions and successful trajectories in a vector store, 100–300ms, retrieved by similarity and recency. L3 is semantic: extracted durable facts and relationships, in a graph or SQL or a managed layer like Mem0 or Zep, over 500ms, and authoritative within a namespace. Memories move between tiers through consolidation — extraction from L1, deduplication and conflict resolution, then decay for anything unreinforced. And there is a strict precedence rule: retrieved documents beat L3, L3 beats L2, and L2 beats an assumption from the conversation.

**Q4. How do you manage the context window inside a long session?**
A hybrid. The last ~10 turns stay verbatim, and everything older is compressed into a rolling summary. At 70–80% of my application budget I trigger compaction: summarize the oldest span, extract durable facts to L2/L3 before discarding the source, drop tool outputs that are already summarized into the plan, and preserve the plan, current task, and explicit user instructions verbatim. In parallel I keep the prompt ordered static-first so the prefix cache hits — system prompt, tool schemas, and stable examples at the front, retrieved content and the current message at the back. That ordering is worth a 5–10x input-cost difference on an agent making ten tool calls.

**Q5. What is prefix caching and how does it change prompt design?**
The inference server keeps the KV cache for the static prefix of the prompt resident, so on a cache hit you only compute and pay for the new suffix. It forces a design rule: **static content first, dynamic content last.** The anti-pattern is putting the user's name, a timestamp, or a session ID at the top — that breaks the cache for every token after it. I also keep tool schemas serialized deterministically with sorted keys, because nondeterministic JSON key order produces a different byte sequence and misses the cache. And I version the cache key by prompt version, so changing the prompt is a deliberate flush rather than an accident.

**Q6. How does memory differ from RAG?**
RAG retrieves from a *corpus* — documents that exist independently of the conversation. Memory stores *derived* state about the user and the interaction: preferences, past sessions, goals, entities. The operational differences matter. RAG sources are authoritative and versioned; memory is a prior that can be stale or wrong, so documents win on conflict. RAG is typically read-heavy and write-rare; memory is written on almost every session, which makes write amplification and privacy the dominant concerns. And RAG indexes are rebuilt; memory is continuously mutated, so it needs deduplication, conflict resolution, and decay — none of which RAG needs.

**Q7. How do you prevent cross-user memory leakage?**
Four layers. Namespace at **write** and at **read** — the namespace filter is applied before the similarity search, never as a post-filter, because a post-filter means you already loaded the other user's data. Row-level security at the database so an application bug cannot bypass it. A cache key that includes the namespace, because a shared cache is a classic leak path. And an automated cross-user recall probe that runs in CI — I construct a fact for user A and assert that user B's retrieval cannot see it. That probe is the one test I would not ship without, because this failure is a breach, not a bug.

**Q8. What is memory fatigue and how do you handle it?**
The agent volunteering irrelevant past details, which clutters answers and erodes trust. I gate on relevance — a fact is only injected at a score above 0.85 — and cap injected facts at 5–10. I use negative retrieval: memory is injected only when it answers a current unknown or contradicts a likely hallucination, not on every turn. And I delete low-value memories after 24 hours, because "the user mentioned it's raining" should never be recalled a week later.

**Q9. How do you handle contradictory memories?**
With an explicit policy rather than letting embedding rank decide. Highest confidence wins; ties break by recency. The loser is not deleted — it gets a `superseded_by` pointer and remains as history, because you need to be able to answer "you told me X before." And there is a higher rule: if a recalled memory conflicts with a retrieved document, the document wins and the memory is flagged for review. Semantic memory is a prior, not ground truth.

**Q10. What is the biggest security risk in a memory layer?**
Persistent memory poisoning. An injected instruction arrives through a retrieved document, the extraction step stores it as a fact, and now it influences every future session — it outlives the session that was attacked. That is strictly worse than ordinary prompt injection. The defence is that extraction consumes only user turns, never model output and never retrieved content; everything is sanitized before extraction; high-impact facts go through a review queue; and I alert on anomalous fact-write rates, because a spike in writes per session is the signature of an attacker trying to plant something.

**Q11 (pushback). Isn't a dedicated memory service just a fancy way to write to Postgres?**
The storage is not the hard part — you could absolutely write facts to Postgres. The hard parts are entity linking across identity surfaces, temporal weighting so new facts supersede old ones without losing history, and deduplication that does not silently create six records for one preference. Those are the three things that fail in production, and they are what the managed layers actually sell. So the honest answer: build it yourself if your use case is narrow and you have budgeted for those three specifically; use a managed layer if you want them solved and are willing to take the dependency and the per-fact cost.

**Q12. How do you test a memory system?**
On a frozen set of scenarios with known-correct recall. Does retrieval return the right fact for a given query. Is the contradiction rate stable. Is the distribution of injected-fact relevance scores stable — I alert on the distribution, not just on errors, because a slow drift toward irrelevant recall is the silent failure. Does a cross-user probe fail to leak. Does compaction preserve a set of constraints I planted at turn three and query at turn thirty. And I run the same probe across model versions, because a model upgrade can change extraction quality without any other signal moving.

**Q13. How would you keep the memory layer from becoming a single point of failure?**
Fail-soft. Memory retrieval gets a hard 200ms timeout; if it times out, the turn proceeds without memory rather than failing, and the turn is tagged as degraded so I can measure the quality impact. I cache the last known profile per user in Redis with a short TTL so a memory-service outage degrades to a slightly stale profile rather than no profile. And I separate the read path from the write path, because a write outage should not block serving — writes can queue and drain.

**Q14. When would you decide NOT to build memory?**
When the session is genuinely stateless, like a one-shot classifier. When the facts already live in a system of record — if the user's data is in a CRM, query the CRM rather than duplicating it into memory where it can go stale. When I cannot yet enforce namespacing, because shipping multi-tenant memory without isolation is shipping a breach. When the latency budget is under 500ms and L3 lookups do not fit. And when regulation forbids retention — if you must not keep it, do not build a system that keeps it.

**Q15. Separate context, state, and memory for me.**
Context is what the agent can *think with* — everything in the window right now: system prompt, tool schemas, retrieved documents, recent turns. State is what the agent is *doing* — the current task and its accumulated parameters. Memory is what the agent can *remember* — what persists outside the window. The test for which is which is the lifecycle: context is discarded every single call, state must be checkpointed and never compacted away because it is the live task, and memory needs deduplication and decay. The hardware analogy I use: the context window is RAM, the memory store is disk, the agent harness is the operating system deciding what gets paged in, and anything already retrieved and still needed should be cached like a CPU cache so you do not pay for it twice.

**Q16. What belongs in the window on every call, and what should you scope down?**
The system and agent instruction is never evicted — it goes in every call, which is exactly why it is also the anchor of your prefix cache. Tool definitions are the line that breaks at scale: you cannot put a thousand tools with their names, descriptions, and input schemas into the window, so you scope them down and retrieve the relevant subset. Conversation is windowed. Retrieved RAG content is per-step. And tool results are the fastest-growing, least-valuable line — I prune them to their conclusion as soon as they are consumed, because an unpruned tool returning thousands of lines is one of the fastest ways to rot a window. I also always reserve headroom for the model's reasoning and output, because a window filled to the brim has no room to answer.

**Q17 (pushback). Isn't compaction just summarisation with a nicer name?**
No, and the difference is the one that matters on a long-running agent. Summarisation *paraphrases* — you get the gist, and details and exact values drift over repeated rounds. Compaction **extracts the state and rebuilds a smaller context around it**, keeping the recent tail verbatim and replacing the middle with a structured, machine-updatable record rather than a paragraph. That is why compaction is safe to run repeatedly on an agent working for hours and a summary-of-summaries is not: a structured record `{db: postgres, migration: done}` survives re-compaction intact, whereas a prose paragraph describing the same thing mutates a little each time until it is wrong. The practical consequence is the rule I never break — never compact the plan, the current task, or an explicit user instruction.

**Q18 (pushback). A 2M-token window is cheaper than building a memory pipeline. Why not skip it?**
Because the window does not fail by overflowing, it fails by rotting — and none of the decay modes are fixed by more room. Overload, dilution, degradation, redundancy, tool-result explosion, staleness: a larger window makes *overload and dilution worse*, because the crucial line is now buried among even more irrelevant ones. Beyond that, the window is per-request, so it gives you no cross-session persistence at all — a returning user still starts from zero. And cost scales with tokens on every turn, so a large window is a recurring charge, whereas retrieval is a one-off lookup. Long context is a substitute for retrieval *within* a session, used deliberately. It is not a substitute for memory, and it is not a substitute for keeping the window clean.

**Q19. What is environment memory, and why does a coding agent need it?**
It is the agent's model of the working environment it operates in — that GitHub and Jira are reachable, that the repo is checked out, and critically *which* package manager is present: `uv`, `pip`, or `npm`. It is distinct from user memory (their preferences) and project memory (what the project is built with), because it is discovered from the environment rather than authored by anyone. A coding agent that assumes `pip` when the project uses `uv` will propose commands that fail, and it will do so confidently because nothing in its context told it otherwise. So environment memory is discovered once, cached, and refreshed when the environment changes — and it is a large part of why tool-equipped coding agents work at all.

**Q20. How do you stop two agents' contexts from contaminating each other?**
Isolation — each agent gets its own scoped window, and I treat the boundary as a security boundary rather than a tidiness one. It matters most with A2A-style protocols, where two agents each carry their own context, state, and memory: if agent A's working context flows into agent B's window, I have created an implicit data channel between two components that were never designed to share anything, and anything sensitive A happened to see is now in B. So the same machinery as per-user namespacing applies — scope at the source, filter before the boundary, and apply RBAC/ACL to what each agent may retrieve — and I give each agent its own memory namespace rather than a shared pool.

**Q21. Give me the one-line summary of this whole topic.**
Context engineering is deciding what is on the desk right now; memory is deciding what goes in the filing room, what gets pulled back out, and what gets shredded — and the failure that will end your incident review is a sticky note from someone else's desk.

---

## 10. Common mistakes candidates make

| Mistake | Why it is wrong | Do this instead |
|---|---|---|
| "I'd just use the 1M-token context window for memory" | Cost, TTFT, attention dilution, and no cross-session persistence | Three tiers with strategic retrieval |
| Putting dynamic content at the top of the prompt | Breaks prefix caching for everything after it | Static first, dynamic last |
| Nondeterministic tool schema serialization | Cache misses on every call | Sorted keys, byte-identical prefix |
| One memory namespace for everyone | Cross-user leak — a breach | Namespace at write and read + row-level security |
| Post-filtering by namespace after similarity search | You already loaded the other user's data | Filter before the search |
| No relevance gate on recalled memory | Memory fatigue; irrelevant facts crowd the prompt | Gate at 0.85; cap at 5–10 facts |
| No decay policy | Stale facts persist forever and get injected confidently | TTL by memory class |
| Treating memory as ground truth | It silently overrides correct documents | Documents beat memory; flag the conflict |
| Extracting facts from model output and retrieved content | Persistent memory poisoning | Extract from user turns only; sanitize first |
| No PII scrubbing between L1 and L3 | Sensitive data retained indefinitely | Scrub at consolidation, not just at input |
| Summarizing from the previous summary | Compounding drift over rounds | Summarize from source turns; cap depth at 1 |
| Compacting the plan or system prompt | The agent forgets its constraints mid-task | Preserve plan, task, and explicit instructions verbatim |
| State as an untyped dict | Bugs hide; nothing is testable | Strictly typed, append-only where possible |
| Last-write-wins on fork/join merge | Silent data loss | Explicit conflict resolution at the join |
| Memory retrieval with no timeout | The memory service becomes an availability dependency | Fail-soft at 200ms; proceed degraded |
| Only measuring errors, not relevance distribution | Drift is invisible until users complain | Alert on the relevance-score distribution |
| No cross-user recall test in CI | The one failure you cannot recover from is untested | Probe every build |

---

## 11. Cheat-sheet summary

| Dimension | Answer |
|---|---|
| **Three tiers** | L1 working (context/KV, <50ms) · L2 episodic (vector, 100–300ms) · L3 semantic (graph/SQL, >500ms) |
| **Context vs state vs memory** | Context = what it can *think with* (the window) · state = what it is *doing* (the current task) · memory = what it can *remember* (outside the window) |
| **Hardware analogy** | context = RAM · memory store = disk/SSD · agent harness = OS · retrieved fact = CPU cache |
| **Six verbs of context engineering** | design · select · retrieve · compress · budget · order |
| **Never evicted** | system/agent instruction (also the prefix-cache anchor) |
| **Evict first** | tool results — prune to the conclusion |
| **Compaction ≠ compression** | Compaction **extracts state** and rebuilds a smaller context; compression **paraphrases** and drifts |
| **Compaction shape** | Keep the recent tail verbatim; replace the middle with a structured, machine-updatable record |
| **Window by** | **Token budget**, not turn count — and name the sliding window first in an interview |
| **Maintenance ops** | refresh · isolation · prune · order |
| **Multi-agent** | isolate scope windows — A2A contexts bleed like a cross-user leak |
| **Memory lifetimes** | short-lived: working, session · long-lived: episodic, semantic, procedural, task/project, user, environment |
| **Environment memory** | the agent's model of its own tooling — which package manager, which integrations exist |
| **Consolidation path** | episode → recurring pattern → procedure |
| **Window rot modes** | overload · dilution · degradation · redundancy · tool-result explosion · staleness |
| **Poisoning taxonomy** | memory (stored fact) · RAG (corpus) · tool (name/description) — three trust boundaries |
| **Precedence rule** | Retrieved documents > L3 > L2 > conversational assumption |
| **Context ordering** | Static first (system, tools, examples), dynamic last (retrieved, recent, current) |
| **Prefix caching** | 5–10x input-cost difference; requires a byte-identical prefix |
| **Application window** | 16k–64k chosen by you, not the model ceiling |
| **Budget split** | 32k context − 4k response reserve − 2k tool reserve |
| **Window strategy** | Hybrid: last 10 turns verbatim + rolling summary |
| **Compaction trigger** | 70–80% of budget |
| **Never compact** | Plan, current task, explicit user instructions |
| **Summarize from** | Source turns, never the previous summary |
| **Consolidation loop** | Extract → dedupe (0.95) → merge/conflict → decay |
| **Fact shape** | `(subject, predicate, object)` + validity + confidence + source |
| **Conflict policy** | Higher confidence wins; ties → recency; keep the loser as history |
| **Relevance gate** | 0.85 minimum; 5–10 facts maximum injected |
| **Decay TTLs** | Chatter 24h · session summaries 90d · preferences indefinite · explicit instructions never |
| **Isolation** | Namespace at write + read, filter **before** search, plus row-level security |
| **The one test** | Cross-user recall probe in CI — any hit is critical |
| **Backends** | Vector (recall) · graph (facts) · Mem0 (broadest) · Zep (temporal) · Letta (paging) · Cognee (graph-first) |
| **State object** | Strictly typed, append-only, checkpointed |
| **Fork/join** | Explicit merge policy — never last-write-wins |
| **Time travel** | Checkpointed transitions enable correction and replay |
| **Memory retrieval timeout** | 200 ms, fail-soft |
| **Memory cost share** | < 10% of total |
| **Worst failure** | Persistent memory poisoning (outlives the session) |
| **Second-worst** | Cross-user memory leak (a breach) |
| **Silent failure** | Relevance-score drift — alert on the distribution |

---

## 12. Cross-links

- [The Agent Engineering Hands-On Stack](11-agent-engineering-hands-on-stack.md) — the checkpointers, thread IDs and summarisation middleware that implement this file
- [Agentic System Design](06-agentic-system-design.md) — the loop that consumes memory and state
- [MCP, Tool Use and Multi-Agent](07-mcp-tool-use-and-multi-agent.md) — supervisor context explosion; inter-agent message handling
- [Production RAG System](01-production-rag-system.md) — corpus retrieval vs. memory
- [Advanced RAG Patterns](02-advanced-rag-patterns.md) — Graph RAG and the semantic tier
- [Guardrails and Safety for AI Systems](04-guardrails-and-safety-for-ai-systems.md) — context as the untrusted zone; PII controls
- [LLM Serving, Inference and Gateway](05-llm-serving-inference-and-gateway.md) — prefix caching, PagedAttention, KV cache mechanics
- [RAG Evaluation and Observability](03-rag-evaluation-and-observability.md) — measuring memory and context quality
- [Fine-Tuning vs RAG vs Prompting](09-fine-tuning-vs-rag-vs-prompting.md) — procedural memory and when weights are the right store
- [Reliability Patterns](../../Operations/07-reliability-patterns.md) — fail-soft and degradation ladders
- [Security, Guardrails and Compliance](../../Operations/08-security-guardrails-and-compliance.md) — data retention, deletion, and audit
- [RAG and LLM Cheatsheet](../../Cheat-Sheets/04-rag-and-llm-cheatsheet.md)
