# 06 — CODING EXPLORATION PLAYBOOK

> **TL;DR.** "Coding Exploration" at Lead level is rarely a LeetCode grind. From the candidate feedback it was: *"there was a RAG code, and they asked me to explain it and what improvement I can offer"* and *"how can you have linear commit history in Git when working with multiple people?"* — i.e. **reading**, **critiquing**, and **team engineering practice**. This file gives you a three-pass method for reading unfamiliar code aloud, a 20-point critique checklist for any RAG/agent code, the definitive Git answer, and the Python patterns most likely to be asked for live.

**60-second interview answer (if asked how you approach a coding exploration):** *"I'd read it in three passes and narrate as I go — first the data flow, then the parameters that matter, then the failure handling. Then I'd offer improvements in priority order: correctness first, then reliability, then cost, then style."*

---

## 1. The method — three-pass code reading (out loud)

Never go silent. Narrate. The interviewer scores *how you think*, and this structure is itself a signal.

| Pass | What you look at | What you say |
|---|---|---|
| **1 — Data flow** | Entry point → what goes in → transformations → what comes out | "So this takes a query, embeds it, searches the index, takes the top-k, builds a prompt, calls the model, returns the text. The data flow is clear." |
| **2 — The parameters that matter** | chunk size, top-k, model, thresholds, retries, timeouts | "The numbers that decide quality and cost: chunk size 500, top-k 5, no reranker, no metadata filter. Those three are where I'd look first." |
| **3 — Failure handling** | try/except, retries, timeouts, validation, what happens on empty retrieval | "There's no timeout on the model call, no validation of the output, and if retrieval returns nothing it still calls the model — that's a hallucination path." |

**Then** give improvements in this priority order — say the order out loud, it sounds senior:

> **Correctness → Reliability → Cost → Latency → Style.** Style last, always.

---

## 2. The 20-point critique checklist (for any RAG or agent code)

Run this checklist against any code they show you. You do not need all 20 — **three to five, well-argued, with the trade-off named**, is a strong answer.

### Correctness & grounding
| # | Check | Improvement to offer |
|---|---|---|
| 1 | **Grounding** — does the prompt say "answer only from the context"? | Add an explicit instruction + an "I don't know" path |
| 2 | **Context ordering** — is the most relevant chunk first? | Reorder; the model attends poorly to the middle ("lost in the middle") |
| 3 | **Empty retrieval** | Return "no evidence found" and route to human, **do not** call the model blind |
| 4 | **Citation** — can you trace the answer to a source? | Attach chunk IDs; in a regulated domain this is mandatory |
| 5 | **Schema** — is the output validated? | Pydantic/JSON-schema validation before the value is used |

### Reliability
| # | Check | Improvement to offer |
|---|---|---|
| 6 | **Timeouts** — on the model call and the DB? | Downstream timeout < upstream timeout; propagate cancellation |
| 7 | **Retries** | Only **one** layer retries; exponential backoff **with full jitter**; bounded by a retry budget (≈10% of request rate) |
| 8 | **Idempotency** | Key per *intent* (not per HTTP request), stored in Redis with in-flight status |
| 9 | **Error classification** | Temporary (429, timeout) → retry; permanent (schema violation, bad input) → fail fast to human |
| 10 | **Streaming/buffering** | Streaming at the provider is not enough — disable buffering in the gateway and proxy; no gzip on streaming endpoints |

### Retrieval quality
| # | Check | Improvement to offer |
|---|---|---|
| 11 | **Dense-only search** | Add BM25 — exact tokens (IDs, error codes, version numbers) are missed by embeddings |
| 12 | **No reranker** | Add a cross-encoder over top-50 → top-5. *Usually the single highest-leverage quality win.* |
| 13 | **No metadata filter** | Pre-filter by tenant/LOB/date **before** the ANN search, not after |
| 14 | **Fixed-size chunking on structured docs** | Use structure-aware or parent-child chunking |
| 15 | **Chunk overlap missing** | Add 10–20% overlap to stop boundary context loss |

### Cost & latency
| # | Check | Improvement to offer |
|---|---|---|
| 16 | **One model for everything** | Tier: small model for routing/extraction, mid for reasoning, frontier for synthesis only |
| 17 | **No cache** | Semantic/exact cache for repeated queries; cache embeddings — re-embedding the same corpus is pure waste |
| 18 | **Re-embedding on every call** | Persist the index; separate ingestion from serving |
| 19 | **Prompt bloat** | Compress context; cap top-k; measure tokens per request |

### Security
| # | Check | Improvement to offer |
|---|---|---|
| 20 | **Retrieved text treated as instructions** | Delimit instructions from retrieved content; scan for injection patterns; never let retrieved text trigger a tool call |

**Template sentence for the critique:**
> *"The highest-priority change is X, because it fixes a correctness problem — currently [failure]. Then I'd add Y for reliability. Z is a cost win I'd do after those, because it's an optimisation, not a fix. The trade-off on X is [cost]."*

---

## 3. Worked example — the RAG code answer (rehearse this shape)

If they paste a naive RAG function, answer like this (adapt to the actual code):

> **Data flow:** *"So this is a classic retrieve-then-generate: embed the query, cosine search the index, take the top five, concatenate them into a prompt with the question, call the model, return the completion. Data flow is clean."*
> **What matters:** *"The parameters that decide everything here: chunk size, top-k = 5, a single embedding model for both document and query, no rerank, no filter, no threshold on similarity. And I notice the temperature isn't set."*
> **Failure paths:** *"Three problems. First, if retrieval returns nothing above the threshold — actually there's no threshold — it will still call the model, which is a hallucination path. Second, there's no timeout on the model call and no retry logic, so a 429 will surface to the user. Third, the retrieved text is concatenated straight into the same message as the instructions, so a document containing 'ignore previous instructions' is an injection vector."*
> **Improvements, in priority order:** *"Correctness first: add a similarity threshold and an explicit 'answer only from context, otherwise say you don't know' instruction, and put a guard for empty retrieval. Then reliability: timeout, one retry with jitter, and a Pydantic-validated structured output if the consumer needs fields. Then quality: add BM25 alongside the dense search and a cross-encoder rerank over the top 50 — that's usually the biggest single quality win. Then cost: cache the query embeddings and tier the model. Style and naming last — the variable names are fine."*
> **Trade-off:** *"The reranker adds maybe 80–150 ms and a bit of cost per query; I'd take that in a triage workflow where the underwriter waits anyway, but not in a sub-second chat."*

---

## 4. The Git question — the definitive answer

**Question asked verbatim:** *"How can you have linear commit history in Git when you're working with multiple people in a team?"*

This is a **policy** question. Answer it as policy, state the trade-off, and do not hedge.

### 4.1 The headline

> *"A linear history comes from one rule: **only one thing ever lands on main, and it lands as a squash or a rebase — never a merge commit.** The mechanism is a protected main branch plus pull requests. Locally, each developer rebases onto the latest main before pushing; on merge, the PR is squash-merged so a whole feature becomes exactly one commit on main."*

### 4.2 The mechanics, in the order you'd say them

| Step | What happens | Command/practice |
|---|---|---|
| **1. Protect main** | Nobody pushes to main directly. Branch protection: PR required, CI must pass. | server-side setting |
| **2. Short-lived feature branches** | Branch off main, live for hours-to-days, not weeks. Long branches are what cause painful conflicts. | `git switch -c feature/x` |
| **3. Keep current by rebasing, not merging** | `git fetch && git rebase origin/main` — rebase replays your commits on top of the latest main, so there's no merge bubble. | `git pull --rebase` |
| **4. Squash-merge the PR** | The whole branch becomes one commit on main. This is the step that *guarantees* linearity. | squash merge in the PR UI |
| **5. Never merge main into a feature branch** | The moment you do, you've created a merge commit and the history is no longer linear. | rebase instead |
| **6. Force-push only your own branch** | After a rebase you must force-push your feature branch — use `--force-with-lease` so you don't clobber someone's work. **Never force-push main.** | `git push --force-with-lease` |
| **7. Merge queue (at scale)** | When many PRs are landing, a merge queue serialises them and re-runs CI on the rebased tip — so main is never red and stays linear without humans racing each other. | GitHub merge queue / Bors |
| **8. Provenance** | Each commit carries an author identity. In a regulated shop, sign commits (GPG/SSH) and enforce it server-side — *who* changed the model code matters as much as the linearity. | `git commit -S` + required signatures |

### 4.3 The trade-off — say this to show judgement

> *"The cost of linear history is that you lose the individual commit boundaries of a branch and the record of when a branch diverged, and rebasing rewrites SHAs, so anyone else on that branch must re-sync. I take that trade because the log becomes readable — one commit per feature, bisectable, easy to revert as a unit. The rule I'd write on the wall is: **rebase your own branch, never rebase a shared one, never merge into a feature branch, and never force-push main.**"*

### 4.4 The related questions they may follow up with

| Follow-up | Answer |
|---|---|
| "Rebase vs merge?" | "Rebase for my own un-pushed/local branch, to keep it current and clean. Merge only when the branch is genuinely shared and other people depend on its SHAs. For landing on main: squash." |
| "What if two people are on the same branch?" | "Then don't rebase — merge main in, accept the merge commit, or better, split the work onto separate branches. Rewriting history under someone else's feet is the thing that breaks teams." |
| "How do you handle a conflict?" | "`git rebase origin/main`, resolve in the file, `git add`, `git rebase --continue`. If it goes wrong: `git rebase --abort` and start again. Small branches make this a two-minute job." |
| "What about hotfixes?" | "Branch from main, fix, squash-merge, and cherry-pick to release branches if you keep them — cherry-pick, not merge, to preserve linearity." |
| "How do you enforce it?" | "Branch protection + CI required + a PR template + squash as the only enabled merge strategy. Policy that isn't enforced by the platform isn't policy." |
| "Trunk-based vs GitFlow?" | "Trunk-based with short-lived branches and feature flags, for a team that deploys continuously. GitFlow's long-lived develop/release branches are what make histories tangled — I'd use it only if there's a hard release-train constraint." |
| "How do you stop main going red with a busy team?" | "A merge queue: PRs enter it, it rebases each onto the current tip and re-runs CI serially, so only one lands at a time and main is never broken. That's the scale answer to 'everyone rebasing at once'." |
| "Does authorship matter, or just the graph?" | "Both. In a regulated shop I'd require signed commits and review approval, because the audit question isn't only *what changed* but *who made the change and who approved it* — the same evidence discipline as the model card." |

### 4.5 The one-line version (if you only get 20 seconds)

> *"Protected main, short branches, `pull --rebase` to stay current, squash-merge to land, `--force-with-lease` on my own branch only, and never a merge commit on main."*

---

## 5. Live-coding patterns most likely to be asked

Practise these until you can type them without thinking. All are AI-engineering-flavoured rather than algorithmic.

### 5.1 Async concurrency for many model calls (the classic)

```python
import asyncio, httpx

async def extract(client, doc, sem):
    async with sem:                      # bound concurrency — protects the provider and your budget
        for attempt in range(3):
            try:
                r = await client.post(URL, json=doc, timeout=20.0)
                r.raise_for_status()
                return r.json()
            except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
                if attempt == 2 or (isinstance(e, httpx.HTTPStatusError)
                                    and e.response.status_code < 500
                                    and e.response.status_code != 429):
                    raise                     # permanent → fail fast, don't retry
                await asyncio.sleep(min(8, 2 ** attempt) * random.random())  # backoff + full jitter
                # then the loop retries

async def main(docs):
    sem = asyncio.Semaphore(16)
    async with httpx.AsyncClient() as c:
        return await asyncio.gather(*(extract(c, d, sem) for d in docs), return_exceptions=True)
```

**Say while typing:** *"I/O-bound, so async and one client with connection pooling. The semaphore bounds concurrency — without it I'd get rate-limited. And I classify the error: 429 and 5xx retry, 4xx fails fast, because retrying a bad request just wastes budget."*

### 5.2 Idempotency for a write operation

```python
def process(request_id: str, payload: dict):
    key = f"idem:{request_id}"
    # SET NX: only one caller wins
    if not redis.set(key, "in-flight", nx=True, ex=3600):
        return {"status": "already_processing", "key": key}   # or wait for the first to finish
    try:
        result = do_the_write(payload)
        redis.set(key, json.dumps(result), ex=86400)
        return result
    except Exception:
        redis.delete(key)          # allow a genuine retry after a real failure
        raise
```

**Say:** *"Keyed on the intent, not the HTTP request — that's the whole point. Reads don't need this; writes always do."*

### 5.3 Token-aware chunking with overlap (and why)

```python
def chunk(text, size=400, overlap=60):
    toks = enc.encode(text)                       # token-aware, not character-aware
    step = size - overlap
    for i in range(0, len(toks), step):
        yield enc.decode(toks[i:i + size])
```

**Say:** *"Token-based because the model's limit is in tokens, and ~10–15% overlap stops context being lost at the boundary. For legal/wording documents I'd go larger and use structure-aware splits; for FAQs, small self-contained chunks."*

### 5.4 Validated structured output

```python
from pydantic import BaseModel, Field, ValidationError
from typing import Literal

class Extraction(BaseModel):
    insured_name: str
    line_of_business: Literal["property", "casualty", "marine", "specialty"]
    sum_insured: float = Field(gt=0)
    confidence: float = Field(ge=0, le=1)
    source_page: int

def parse(raw: str):
    try:
        return Extraction.model_validate_json(raw), None
    except ValidationError as e:
        return None, e            # caller routes to human review — never guess
```

**Say:** *"Enums for constrained fields, explicit bounds, and an explicit failure path. The model proposes; the validator disposes. If it doesn't validate, a human sees it — I never repair it with a regex and hope."*

### 5.5 A tiny LRU cache (they may ask for the classic)

```python
from collections import OrderedDict
class LRU:
    def __init__(self, cap): self.cap, self.d = cap, OrderedDict()
    def get(self, k):
        if k not in self.d: return None
        self.d.move_to_end(k); return self.d[k]
    def put(self, k, v):
        if k in self.d: self.d.move_to_end(k)
        self.d[k] = v
        if len(self.d) > self.cap: self.d.popitem(last=False)   # O(1) amortised
```

**Say:** *"O(1) get and put via an ordered dict — that's the same structure I'd use for a semantic cache keyed on a normalised query."*

### 5.6 Retry with exponential backoff + full jitter (as a function)

```python
def backoff_delay(attempt, base=1, cap=30):
    return random.uniform(0, min(cap, base * 2 ** attempt))   # FULL jitter
```

**Say:** *"Full jitter — `random(0, window)` — not fixed backoff. Fixed backoff and even equal jitter leave synchronised spikes; full jitter is what actually de-synchronises a thundering herd."*

---

### 5.7 Concurrency: I/O-bound vs CPU-bound, and the GIL (a source topic, and Brit's exact problem)

This is the classic follow-up to §5.1, and it maps *directly* onto Brit: a chat/agent service is I/O-bound, while their post-catastrophe **image inference is CPU/GPU-bound**. If they hand you "a chatbot that also accepts image uploads" (a scenario in your source material), this is the answer.

| | **I/O-bound** | **CPU-bound** |
|---|---|---|
| Waiting on | network, disk, another service | arithmetic — image decode, tiling, local inference |
| The bottleneck | the wait, not the CPU | the CPU (or GPU) |
| Python tool | **`asyncio`** (one event loop, thousands of concurrent awaits) | **`multiprocessing`** (process pool — separate interpreters) |
| Why not threads | threads work but each costs ~8 MB | **the GIL serialises bytecode**, so threads give you no parallelism |
| Brit example | broker submission over an API, LLM streaming, DB reads | tiling + running the damage model over aerial imagery |

**What the GIL actually is:** CPython allows only one thread to execute Python bytecode at a time. So threads do **not** speed up CPU-bound Python — they interleave, and you pay context-switching for nothing. Threads *do* help I/O-bound work (a blocked thread releases the GIL), and they *do* help CPU work in C libraries that release the GIL (NumPy, many image libraries).

**The pattern to state:**

> *"I match the concurrency model to the workload. I/O-bound — the model calls, the DB, the broker API — is `asyncio` on one event loop, because the cost is waiting, not computing. CPU-bound — image tiling and local inference — goes to a **process pool**, because the GIL means threads wouldn't parallelise it. And I never do heavy CPU work *inside* the event loop, because a blocking call freezes every other coroutine on it."*

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor

pool = ProcessPoolExecutor(max_workers=4)          # CPU-bound: separate interpreters, past the GIL

def tile_and_score(image_bytes):                    # pure CPU: decode, tile, run the model
    ...

async def handle_upload(image_bytes):
    loop = asyncio.get_running_loop()
    # offload CPU work so the event loop keeps serving other requests
    result = await loop.run_in_executor(pool, tile_and_score, image_bytes)
    return result

async def main(uploads):
    # I/O + CPU together: async fans the requests in, the pool does the heavy lifting
    return await asyncio.gather(*(handle_upload(u) for u in uploads))
```

**The `run_in_executor` line is the whole point:** it keeps the blocking CPU work *off* the event loop, so a slow image doesn't stall every chat connection sharing that loop.

**Likely follow-ups, and the answers:**

| Follow-up | Answer |
|---|---|
| "Why not just spawn a thread per upload?" | "Threads don't parallelise CPU-bound Python — the GIL. They'd interleave on one core and add overhead. A process pool gives real parallelism." |
| "When *would* you use threads?" | "I/O-bound work where the library is blocking and not async-aware — a thread pool offloads the blocking call without rewriting the client. And CPU work in a C library that releases the GIL, like NumPy." |
| "Async vs parallel — same thing?" | "No. Concurrency is *interleaving* (one worker making progress on many tasks while waiting); parallelism is *simultaneous execution* on multiple cores. Async is concurrency; a process pool is parallelism. I usually need both, together — which is exactly the snippet above." |
| "At scale, would you do this in-process at all?" | "No — long before that I'd move image scoring to its own **stateless worker service** behind a queue, so it scales independently of the chat tier and a burst of uploads can't starve the API. The process pool is for the single-node case." |
| "How would you benchmark it?" | "Measure the bottleneck first: if it's waiting, the fix is concurrency; if the CPU is pinned, it's parallelism. Amdahl's law caps the win — the serial fraction sets the ceiling no matter how many workers I add." |

**Say while typing (one sentence):** *"I/O-bound gets async, CPU-bound gets processes — the GIL is why — and I'd split them into separate services once either one starts starving the other."*

---

## 6. Testing and evaluation in code (they will ask "how would you test this?")

| Layer | What it is | Example |
|---|---|---|
| **Unit** | Pure functions, no LLM | chunking boundaries, the validator, the parser |
| **Contract** | The interface to the LLM/DB | mocked model responses in every documented shape, including malformed |
| **Eval regression** | The golden set | 50–120 curated cases with expected sources/answers, run in CI, **gates the merge** |
| **Trajectory eval** | Agent step sequence | did it call the right tools in the right order; how many steps vs the expected path |
| **Canary / shadow** | Real traffic | run the new version alongside, compare, no user impact |
| **Property/robustness** | Adversarial | prompt injection strings, empty retrieval, 10×-length documents |

**Line to say:** *"The rule I set is that no prompt, retrieval or model change merges without passing the golden set. Without that gate, every change is a coin flip."*

---

## 7. Practice set — do these out loud this week

| # | Task | Time |
|---|---|---|
| 1 | Read a naive RAG function and produce 5 improvements in priority order | 10 min |
| 2 | Write an async fan-out of 100 model calls with bounded concurrency and retries | 15 min |
| 3 | Add idempotency to a payment-like write path | 10 min |
| 4 | Write a Pydantic model for a 10-field insurance extraction with two enums | 10 min |
| 5 | Answer the Git question in 20 seconds, then 90 seconds | 5 min |
| 6 | Token-aware chunker with overlap, plus the parameters you'd choose for legal vs FAQ | 10 min |
| 7 | Explain your own project's code aloud, three passes | 10 min |
| 8 | Sketch §5.7 from memory: async for I/O, process pool for CPU, `run_in_executor` off the event loop — and say why the GIL forces it | 10 min |

---

## 8. Common mistakes in this part

| Mistake | Fix |
|---|---|
| Going silent while reading | Narrate in three passes |
| Jumping to style/naming first | Correctness → reliability → cost → latency → style |
| Rewriting the code wholesale | Offer **prioritised** improvements with trade-offs |
| Not knowing the Git answer | §4 — memorise the one-liner |
| Ignoring failure paths in code review | Look for timeouts, retries, validation, empty retrieval — first |
| Writing code without stating complexity/interfaces | Restate the requirement and the signature first |

---

## 9. Cheat-sheet summary

- **Three-pass read, out loud:** data flow → parameters → failure handling.
- **Improvement order:** correctness → reliability → cost → latency → style.
- **Top three RAG fixes:** similarity threshold + empty-retrieval guard; hybrid + rerank; structured validation.
- **Git:** protected main · short branches · `pull --rebase` · squash-merge · `--force-with-lease` on your branch only · never merge into a feature branch · never force-push main.
- **Test layers:** unit → contract → eval regression (CI gate) → trajectory → canary → adversarial.
- **Patterns to have in your fingers:** async bounded fan-out, idempotency, token chunking, Pydantic validation, full-jitter backoff, and the **I/O-bound → `asyncio` / CPU-bound → process pool** split (the GIL is why).

## 10. Cross-links

- The evaluation narrative: [`07-agent-evaluation-and-guardrails.md`](07-agent-evaluation-and-guardrails.md)
- RAG depth for the critique: [`05-system-design-round-playbook.md`](05-system-design-round-playbook.md) · [`10-question-bank-with-model-answers.md`](10-question-bank-with-model-answers.md)
- Reliability mechanics: [`../Operations/`](../../Operations/)
