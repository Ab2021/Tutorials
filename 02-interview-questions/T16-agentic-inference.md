# Interview Bank: Agentic Inference

> `T16` · **Transcript coverage:** primary · [Cheat sheet](../00-cheat-sheets/T16-agentic-inference.md) · [Case study](../01-case-studies/T16-agentic-inference.md) · [Design blueprint](../03-design-blueprints/T16-agentic-inference/HLD.md)
> **Questions:** 28 (8 × L3, 13 × L4, 7 × L5) · **Format:** why agents break chat serving, then the loop, the stack, the economics, and the open design

## How to use this bank

Agentic inference is the topic where a candidate's chat-serving instincts actively mislead them, so the
bank opens by making those instincts fail visibly (Q1–Q7) before building the correct picture. Ask in
order for a full loop; for a senior screen, start at Q10.

The discriminator throughout is whether the candidate reasons from the **98%-prefill workload shape**
or recites agent-framework vocabulary. A candidate who talks about LangGraph but cannot say why
prefix caching is the highest-leverage optimisation has learned the wrong layer of the stack.

---

### Why agents break chat serving assumptions

#### T16-Q1 · How is an agentic request shaped differently from a chat request?
**Difficulty:** L3 · **Depth expected:** 2–3 min

**Question:** Your serving stack was tuned for chat and works well. Now you are asked to serve agents
on it. What actually changes about the traffic?

**Model answer:** The shape inverts, and everything downstream follows from that.

A chat request is one short prompt and one long output. An agent is the **inverse: a huge, growing
prefix and a tiny output, repeated many times per task**, with **idle gaps between tool calls** `[T]`
llm-d. The corpus gives the numbers: **~98% of agentic tokens are prefill and ~2% are decode** `[T]`
llm-d.

The consequence is that the workload is **TTFT-bound, not ITL-bound** `[T]`/`[D]`. Every optimisation
built for chat is aimed at the 2% — speculative decoding, inter-token-latency tuning,
output-length-based batching. They are marginal here. What matters instead is how fast you can ingest a
growing context, and whether you can avoid ingesting it again.

That is why **prefix caching and KV offload are the two highest-leverage optimisations** in an agentic
stack. The prefix grows monotonically with turn count — each turn appends the tool result and the next
model output — so a task at turn 10 is carrying a prefix several times longer than at turn 1, and it
re-reads that prefix on every turn unless the cache holds.

The second structural change is the idle gap. Between a tool call and its result, the agent is
waiting; the corpus's platform framing puts this at agents being **"idle 99.999% of the time"** `[T]`
Steinder. During that gap the KV may be evicted `[T]` llm-d — and the next turn then pays to rebuild
context that was already computed.

**Signal:** States the inversion as *growing prefix, tiny output, repeated, with gaps* — and derives
TTFT/ITL consequences from it rather than listing them as separate facts.

**Follow-ups:**
- *Which chat optimisations stop mattering?* — anything output-side; speculative decoding and ITL tuning.
- *What does the idle gap cost you?* — KV eviction; see T16-Q14.
- *How does 98/2 change your batching strategy?* — you batch prefill, not decode.

**Red flags:** Says agents are "just chat with more turns," or talks only about orchestration
frameworks without touching the workload shape.

---

#### T16-Q2 · Why is "agents are just multi-turn chat" wrong?
**Difficulty:** L3 · **Depth expected:** 2 min

**Question:** Push back on this: an agent is a chat conversation with tool calls in it, so existing
serving should handle it.

**Model answer:** The sentence is true at the API level and false at the workload level, and the
difference is where all the engineering goes.

At the API level, yes — each turn is a completion request with a message list. But three properties
diverge sharply from multi-turn chat `[T]` llm-d:

**The prefix grows without bound.** A chat conversation is typically bounded and human-paced; an agent
at turn 40 may carry a prefix an order of magnitude larger than a chat session, and it re-reads all of
it every turn. With **98% of tokens in prefill** `[T]`, the cost is dominated by that re-read.

**The turn rate is machine-driven.** An agent can issue turns as fast as tools return, so a single
task can present as a burst of concurrent requests rather than a human-paced trickle.

**There are idle gaps with no analogue in chat.** A human pausing keeps their session warm in a way an
agent's 30-second tool call does not, because the gap is long enough for the KV to be evicted `[T]`.

So "multi-turn chat" describes the interface, not the traffic. The practical test: if your chat-tuned
stack handles agents well, check your prefix cache hit rate and your TTFT at turn 10 — the failures
show up as per-turn cost growth, not as errors.

**Signal:** Separates the API-level truth from the workload-level falsehood, and names a measurable
way the difference would show up.

**Follow-ups:**
- *What is the single best metric to detect the difference?* — per-turn cost, or prefix cache hit rate.
- *Would you run agents and chat on the same pool?* — the tradeoff; see T16-Q16.
- *What breaks first?* — cost growth with turn count; see T16-Q19.

**Red flags:** Accepts the framing and moves to framework comparison, or claims agents need a wholly
separate stack with no shared primitives.

---

#### T16-Q3 · Walk me through the ReAct loop and where the inference cost goes
**Difficulty:** L3 · **Depth expected:** 2–3 min

**Question:** Describe the standard agent loop and identify where the inference cost accumulates.

**Model answer:** ReAct is **reason → act → observe**, repeated until the agent decides it is done
`[T]`. In serving terms each iteration is one completion call, and the message history is the state.

The cost location is the thing to get right. Each turn appends the previous output and the tool result
to the history, so **the prefix grows monotonically with turn count** `[T]`. The model then reads the
entire prefix to produce one small output. Summed over a task:

```
tokens_per_task = Σ_turns (prefix_tokens + output_tokens)
```

and because `prefix_tokens` grows with turn count, the sum is **quadratic in the number of turns unless
the prefix is cached** `[D]`. That is the single most important arithmetic fact about agent economics.
At turn N, the naive stack has read roughly the first N prefixes in full.

Two mitigations follow directly, and they are the two the corpus names `[T]` llm-d:

**Prefix caching** — if the history is append-only and the prefix is stable, each turn only computes the
new suffix. This turns the quadratic back into linear.

**KV offload** — during the idle gap between the tool call and its result, the KV can be parked rather
than discarded, so the next turn resumes instead of rebuilding. The corpus reports roughly **5× TTFT
improvement on session return** from KV offload `[T]` llm-d.

Note also that the loop itself is not the expensive part — the *number of iterations* is. An agent that
retries or self-critiques multiplies the whole sum by its turn count (T16-Q22).

**Signal:** Writes or states the summation, identifies the quadratic-without-caching behaviour, and
connects it to the two named mitigations.

**Follow-ups:**
- *What makes the prefix cache miss?* — rewriting history, or routing to a different replica.
- *Where does the idle gap enter?* — between act and observe; see T16-Q14.
- *How does Reflexion change the arithmetic?* — it multiplies turn count; see T16-Q22.

**Red flags:** Describes the loop without locating cost, or believes output length drives the bill.

---

#### T16-Q4 · What are the three classes of agent you will be asked to serve?
**Difficulty:** L3 · **Depth expected:** 2 min

**Question:** You are told "we run agents." What follow-up questions determine what you are actually
serving?

**Model answer:** The corpus's platform framing separates agents into three classes, and each one
changes what you can control `[T]` Steinder.

**Class one: in-house agents.** You wrote the harness, so you own the loop, the prompts, the tool list
and the retry policy. Everything in this bank is available to you — append-only history, parallel tool
calls, budgets, prefix stability. This is the class where optimisation is possible.

**Class two: harness-with-hooks.** A third-party harness that exposes extension points — you can
intercept calls, inject policy, observe. You have less control than class one, but an interception
layer gives you a place to enforce budgets, log trajectories, and apply guardrails without owning the
loop.

**Class three: blackbox.** A harness you cannot modify. You see requests and responses and nothing
else. Here your levers collapse to what the serving layer can do on its own: prefix-aware routing,
KV offload, admission control, fairness scheduling, and accounting. You cannot fix a client that
rewrites its history every turn.

The reason this is a first question rather than a taxonomy exercise is that it decides the entire
conversation. A candidate who proposes "make the history append-only" for a blackbox harness has not
asked which class they are in.

Two further properties cut across all three, and both are quoted from the corpus: the instruction set
is **open-ended** — the model may choose from a growing set of tools — and there is **no
instruction/data separation** `[T]`. Those are why the guardrail layer (T18) has to exist regardless
of which class you are serving.

**Signal:** Asks which class before proposing solutions, and connects the class to the available
control surface.

**Follow-ups:**
- *What can you still do for a blackbox harness?* — serving-layer levers only.
- *Where does an interception layer fit?* — class two; policy and telemetry without loop ownership.
- *Why does the open-ended instruction set matter?* — it is why tool-call rails exist; T18.

**Red flags:** Assumes you own the agent loop, or treats all agents as equivalent.

---

#### T16-Q5 · Why is "no instruction/data separation" a serving problem, not just a security problem?
**Difficulty:** L3 · **Depth expected:** 2 min

**Question:** The corpus notes there is no instruction/data separation in LLM agents `[T]`. Why does
that matter to an inference engineer?

**Model answer:** Because it removes an assumption that serving stacks have relied on since long before
LLMs.

In a conventional system, code and data occupy different places and are handled by different machinery.
A query parameter cannot become a command. In an LLM agent everything arrives in the **same token
stream**: the system prompt, the user request, retrieved documents, tool outputs and web pages are all
text in one context window. Tool output — a web page, a file, an email body — can contain text that
reads as an instruction, and there is no structural marker distinguishing it from one.

The serving consequences are concrete:

**Inputs become adversarial by default.** Any content the agent retrieves must be treated as untrusted
input `[T]`. That is not a hypothetical — it is why the tool boundary needs a rail (T18).

**Trust must be carried as data.** Since the token stream cannot express provenance structurally, the
system has to tag it explicitly somewhere — and the corpus's formulation is that **trust level is data,
not metadata** `[T]`. Anything that rides along outside the request can be stripped or spoofed.

**You cannot validate an output the way you validate a program.** There is no type system for
instructions, so output validation is a scoring problem rather than a correctness proof.

And there is a second, related property the corpus names: **"we essentially do not have any reliable
error codes"** `[T]`. A failed tool call, a hallucinated API name and a refused instruction all arrive
as ordinary text. So the serving layer cannot reliably distinguish "the agent is failing" from "the
agent is thinking" without evaluating content.

**Signal:** Frames it as a lost structural guarantee with serving consequences, and connects it to both
trust tagging and the absent error codes.

**Follow-ups:**
- *Where do you enforce trust then?* — trust tagging as data; T18's four rails.
- *Why do the missing error codes matter operationally?* — you cannot alert on failure without content
  evaluation; T17.
- *Does this change retry policy?* — yes; a "failure" may be indistinguishable from success.

**Red flags:** Treats it purely as a prompt-injection topic and hands it to security, missing the
observability and retry consequences.

---

#### T16-Q6 · How many idle agents can one node hold?
**Difficulty:** L3 · **Depth expected:** 2 min

**Question:** An agent is idle between tool calls. Does an idle agent consume serving capacity?

**Model answer:** It should consume almost none, and that gap between "should" and "does" is where the
platform design lives.

An idle agent is not issuing inference requests, so it holds no GPU compute. The corpus's framing is
provocative and useful: agents are **"idle 99.999% of the time"** `[T]` Steinder. What an idle agent
does hold is **state** — the conversation history, the session context, the KV if you chose to retain
it.

That reframes the capacity question. You are not sizing for concurrent *compute*; you are sizing for
concurrent *sessions* multiplied by whatever you retain per session. If you retain KV for every idle
session, memory becomes the constraint, and the arithmetic is brutal: a 50k-token prefix at typical
KV-per-token costs is a large amount of HBM per session, so the number of simultaneously retained
sessions is far smaller than the number of live agents.

The design answer is tiering. Keep KV in HBM for the hot sessions, **offload to a CPU/DRAM tier for
idle ones**, and rely on the **retention API** to express how long a session's KV should be kept `[T]`
llm-d. The corpus reports roughly **5× TTFT improvement on session return** when the offload tier has
the session `[T]`, which is the payoff for retaining it at all.

The platform-side corollary, from the Agent Substrate work: because compute is idle almost always, the
right model is **pause/suspend/resume** with a golden snapshot, rather than keeping a process warm per
agent `[T]` Steinder. Cold start is **10–15 s** and the wake target is **low hundreds of
milliseconds** — which is only achievable because resuming a snapshot is not the same as starting a
container.

**Signal:** Separates compute capacity from state capacity, and reaches for tiering plus retention
rather than assuming everything is resident.

**Follow-ups:**
- *What is the wake target and why is it achievable?* — low hundreds of ms; snapshot resume, not cold
  boot.
- *What does the retention API express?* — how long a session's KV is kept; T12.
- *What breaks if you retain everything?* — HBM exhaustion competing with live traffic.

**Red flags:** Treats idle agents as free with no memory consequence, or proposes keeping a warm
process per agent.

---

#### T16-Q7 · What is the agentic compute multiplier?
**Difficulty:** L3 · **Depth expected:** 2 min

**Question:** The corpus cites a 10–100× agentic compute multiplier `[T]`. What is being multiplied,
and why should an infrastructure team care?

**Model answer:** The multiplier is against the equivalent single-shot interaction, and it comes from
three compounding sources.

**Turn count.** One user request becomes many model calls — one per loop iteration — each reading the
whole prefix `[T]`. Ten turns is ten inference calls.

**Prefix growth.** Because each turn's input includes all previous turns, the *total* tokens processed
across a task is quadratic in turn count unless caching intervenes `[D]`. So turn count understates the
multiplier; the token count overstates the naive one.

**Tool and retry overhead.** Parallel tool calls, retries after failures (which you cannot reliably
detect — see T16-Q5), and self-critique loops each add calls `[T]`.

Why infrastructure should care: the multiplier is not a constant you can budget once. The corpus's
scale figures make the point — **3.2 quadrillion tokens per month** `[T]` Tiwary — which is a number
that only makes sense if per-user token consumption has grown by an order of magnitude over the chat
era.

Operationally, three consequences follow. First, **capacity planning must be per task, not per
request**; a request-based model under-provisions by the multiplier. Second, **cost attribution must be
per task** for the same reason (T16-Q21). Third, **the multiplier is a design variable** — it is
reduced by caching the prefix, by parallelising tool calls, and by bounding retries. A team that treats
it as a fixed property of "using agents" has given up the lever.

**Signal:** Decomposes the multiplier into turn count, prefix growth and overhead — and identifies it
as something design can reduce.

**Follow-ups:**
- *Which component is largest?* — prefix growth, because it compounds; it is why caching matters most.
- *How do you reduce it?* — stable prefix, parallel tools, retry budgets.
- *How does it change capacity planning?* — size per task; see T16-Q20.

**Red flags:** Treats the multiplier as a fixed constant, or equates it with turn count alone.

---

### The loop, the harness and the sandbox

#### T16-Q8 · Design the client side of an agent loop for cacheability
**Difficulty:** L4 · **Depth expected:** 4 min

**Question:** You control the agent's code. What rules do you follow to make it efficient against a
prefix-caching serving stack?

**Model answer:** Two rules, and both are performance requirements rather than style preferences.

**Rule one: append to the history, never rewrite it.** A prefix cache works by matching a stable token
prefix. If the client rebuilds the message list each turn — reordering, editing an earlier message,
injecting a system reminder at the front — the cached blocks no longer match and every turn pays full
prefill. The corpus's configuration expresses it directly: keep the prefix stable, **append, don't
rewrite** `[T]`/`[D]`.

**Rule two: put volatile content last.** Anything that changes every turn — a timestamp, a live counter,
freshly fetched data — placed early in the prompt invalidates everything after it. Move it to the end
of the message list so the stable prefix stays stable `[D]`.

Beyond those, three more that consistently pay `[D]`:

**Parallelise independent tool calls.** It is usually the cheapest available speedup, but note that it
**multiplies concurrency on the serving tier** — which is precisely what forces the 20,000-concurrency
class of deployment `[T]`. You are trading serving-side concurrency for wall-clock latency.

**Budget tokens, steps and wall-clock per task.** An agent with no budget is a denial-of-service
against your own fleet. The corpus lists "agent loops forever" with the fix as *budget + termination
check* `[T]`.

**Pin the tool list.** A tool schema that changes between turns changes the prompt. Version it.

The reason to state these as rules rather than tips is that they are cheap to follow at design time and
expensive to retrofit: every one of them changes the client, and a client you do not control (a
blackbox harness) cannot be fixed at all.

**Signal:** Leads with append-only and volatile-last as hard rules, and volunteers the parallel-tool
concurrency tradeoff rather than presenting it as a free win.

**Follow-ups:**
- *Why is append-only a performance rule?* — it is what makes the prefix cache valid.
- *What does parallel tool calling cost you?* — serving-tier concurrency; see T16-Q20.
- *What can you do if you do not own the client?* — serving-layer levers only; T16-Q4.

**Red flags:** Treats prompt construction as a style matter, or presents parallel tool calls as free.

---

#### T16-Q9 · Where does the sandbox sit, and why is it a serving concern?
**Difficulty:** L4 · **Depth expected:** 3–4 min

**Question:** Agent code executes in a sandbox. Is that a security topic or an infrastructure topic,
and what does it change about your serving design?

**Model answer:** It is both, and treating it as purely security is how teams end up with an agent tier
that cannot scale.

The security reason is straightforward: code-execution agents (CodeAct-style) run model-generated code,
and tool outputs are untrusted input. The sandbox is the boundary that stops a generated instruction
from becoming a host action `[T]` — a security boundary, not a convenience.

The infrastructure reason is that the sandbox is a **separate tier with its own lifecycle**, and the
serving design has to account for it:

**It has different scaling characteristics from the GPU tier.** Tool execution is CPU/IO bound and
scales differently than inference. Co-locating them means neither scales cleanly.

**It introduces the idle gap.** The tool call is where the agent waits, and the wait is what evicts the
KV `[T]`. So sandbox latency is not just latency — it is a KV-retention problem (T16-Q14).

**It is where the corpus's serverless pattern applies.** The platform framing is a **stateless loop
plus a durable session log plus a sandbox tier** `[T]` Steinder — the same pattern as serverless
compute: the loop holds no durable state, the session log is the source of truth, and the sandbox is
ephemeral. That decomposition is what makes **pause/suspend/resume** possible, because resuming means
restoring from the log rather than reconstructing state.

**It constrains parallel tool execution.** Parallel calls only help if the sandbox can run them
concurrently; a serialised sandbox makes the "cheapest speedup" unavailable `[D]`.

So the design answer: keep the sandbox a distinct tier with its own scaling policy, make the agent loop
stateless with a durable session log, and treat sandbox latency as an input to your KV retention
strategy rather than just a latency number.

**Signal:** Names the sandbox as a separate tier with its own scaling, and connects it to the
stateless-loop / durable-log / sandbox decomposition.

**Follow-ups:**
- *Why does a stateless loop matter?* — it is what makes pause/resume and preemption survivable.
- *What fails if the sandbox serialises?* — parallel tool calling; see T16-Q8.
- *How does the sandbox affect KV retention?* — it creates the idle gap; T16-Q14.

**Red flags:** Treats the sandbox as purely a security concern, or puts it inside the serving process.

---

#### T16-Q10 · Stateless loop, durable log, sandbox — why this decomposition?
**Difficulty:** L4 · **Depth expected:** 4 min

**Question:** The corpus describes a serverless pattern for agents: stateless loop, durable session log,
sandbox tier `[T]`. Why is that the right decomposition, and what does it buy?

**Model answer:** It buys the same thing serverless bought for web services: **the ability to move,
preempt and resume work freely**, because no durable state lives in the compute.

**The stateless loop** holds only in-flight reasoning. It can be killed and restarted at any point
without losing the task, because nothing in it is the record of truth. That is what makes preemption
survivable — and in a fleet where agents are "idle 99.999% of the time" `[T]` Steinder, preemption is
exactly what you want to do: an agent waiting on a 30-second tool call should not be occupying a
reserved slot.

**The durable session log** is the record of truth: the message history, tool results, decisions. It is
what the loop rebuilds from on resume. This is also what makes long-horizon agents possible — a task
running for hours cannot survive on in-process state, and the corpus's failure signature "session lost
mid-task" has *checkpoint/restore* as its fix `[T]`.

**The sandbox tier** isolates execution and scales independently, as in T16-Q9.

What the decomposition enables concretely `[T]` Steinder:

- **Pause/suspend/resume** with a **golden snapshot**, against a **low-hundreds-of-milliseconds** wake
  target rather than the **10–15 s cold start**.
- **Scale to very large fleets** — the design targets up to **200,000 nodes** — because sessions are
  cheap to hold when they are logs rather than processes.
- **Independent scaling of the three tiers**, since the loop, the store and the sandbox have different
  profiles.

The cost is a state-management problem you did not have before: the log must be durable, consistent,
and fast enough that resume is genuinely hundreds of milliseconds rather than seconds. That is the real
engineering, and it is why the pattern is worth stating as an architecture rather than a slogan.

**Signal:** Explains the decomposition as *what makes preemption and resume possible*, and connects it
to the idle-99.999% observation rather than reciting the three boxes.

**Follow-ups:**
- *Why is a stateless loop required for preemption?* — no in-process record of truth to lose.
- *What makes resume fast?* — snapshot restore versus cold start.
- *What is the new failure mode?* — log durability and consistency; T16-Q27.

**Red flags:** Describes the pattern without saying what it enables, or keeps state in the loop process.

---

#### T16-Q11 · What is an interception layer and when do you need one?
**Difficulty:** L3 · **Depth expected:** 3 min

**Question:** You must apply policy to agents you did not write. Where does an interception layer go,
and what can it actually enforce?

**Model answer:** An interception layer sits between the agent harness and the model endpoint, and it is
what makes a harness you cannot modify still governable. It is the class-two answer from T16-Q4 `[T]`.

What it can enforce `[D]`:

**Budgets and rate limits.** Because it sees every call, it can count tokens and steps per session and
terminate a runaway agent. This is the only place that works for a blackbox client — you cannot put a
budget in code you do not own.

**Identity and attribution.** The corpus makes a point that matters here: **agent identity is separate
from the user identity** `[T]`. An agent acting on a user's behalf is not the user, and a delegated
action should be attributable to the agent. An interception layer is the natural place to attach and
propagate that identity, and doing it at the serving boundary means every downstream system sees it
consistently.

**Guardrails.** Input and output rails can be applied centrally rather than in each harness (T18).

**Telemetry.** Trajectory spans must be emitted somewhere; if the harness does not emit them, the
interception layer can (T17).

Its limits are equally important. It **cannot fix a client that rewrites its history** — that destroys
prefix cacheability before the interception layer sees the request, and nothing downstream can recover
it. It **cannot reduce turn count**, which is where the multiplier lives. And it **adds a hop**, so it
must be cheap and must not become the bottleneck.

So the honest framing: an interception layer converts an uncontrollable client into a *governable* one,
but not into an *efficient* one. Efficiency requires owning the loop.

**Signal:** Names budgets, identity and telemetry as the enforceable set, and states clearly what
interception cannot fix.

**Follow-ups:**
- *Why is agent identity separate from user identity?* — delegation and attribution `[T]`.
- *What can it not fix?* — client-side prefix instability and turn count.
- *Where does it sit relative to the gateway?* — in the request path before routing; T14.

**Red flags:** Claims an interception layer can retrofit efficiency onto any client, or conflates agent
and user identity.

---

#### T16-Q12 · Context condensation — when is it worth it?
**Difficulty:** L4 · **Depth expected:** 3–4 min

**Question:** A team adds summarisation to keep long agent histories inside the context window. What
does that do to your serving economics?

**Model answer:** It trades cache for space, and the trade is often correct — but it must be made
knowingly, because the cache cost is invisible in most dashboards.

**The mechanism.** Summarising history replaces a block of cached tokens with a new, shorter block. The
prefix that the cache held no longer matches the token stream, so the next turn's prefill is a cache
miss over the changed region. The corpus states the consequence plainly: context condensation
**reduces prefix-cache hit rate** `[T]`.

**What you gain.** You stay inside the context window, and you reduce the per-turn token count — which
reduces both the time to read the prefix and, if the cache is missing anyway, the fresh-token bill.

**What you pay.** The corpus's cost arithmetic makes the magnitude concrete: **cache reads are ~10×
cheaper** than fresh tokens `[T]`, and with a long prefix the cache hit rate is the dominant term in
per-turn cost `[D]`. So a condensation step that drops the hit rate from high to low can cost more than
the summarised tokens saved — particularly if condensation runs often.

**When it is worth it** `[D]`:

- **The context window is genuinely the binding constraint.** Then you have no choice, and the question
  is only how often you condense.
- **The summarised region is unlikely to be re-read.** Condense old material the agent will not revisit.
- **Condensation is infrequent and the stable prefix is preserved.** Summarise at the *front* and keep
  the recent window stable, so the volatile region is small.

**When it is not:** condensing every turn, or condensing a region the agent is still actively working
in. Both destroy the cache for a saving that is smaller than the loss.

The right instrumentation is the same as everywhere in this topic: measure the **prefix cache hit rate
before and after** `[T]`, and treat a drop as a cost you chose rather than a mystery.

**Signal:** Names the cache-hit-rate cost as the price, quantifies it with the ~10× cache-read figure,
and gives conditions under which the trade is still correct.

**Follow-ups:**
- *How would you detect the cost?* — hit rate before/after, per pod `[T]`.
- *What is the ideal condensation policy?* — infrequent, front-loaded, stable window.
- *Does it interact with KV offload?* — yes; both touch what is retained.

**Red flags:** Says summarisation always saves money, or cannot name the cache consequence at all.

---

#### T16-Q13 · How do you evaluate an agent — outcome or trajectory?
**Difficulty:** L4 · **Depth expected:** 3–4 min

**Question:** Your agent's success rate looks fine but users are unhappy. What is your evaluation
missing?

**Model answer:** You are evaluating the **outcome** and not the **trajectory**, and for agents that is
the difference between a metric and a liability.

Outcome evaluation asks: did the task complete correctly? Trajectory evaluation asks: did the agent
take a defensible path — right tools, right order, no unsafe steps, no absurd retries `[D]`.

Three failure classes are invisible to outcome-only evaluation `[T]`:

**Right answer, unacceptable path.** The agent reached the goal by a route you would not sanction — for
instance by acting on an injected instruction from retrieved content. The outcome looked fine; the
behaviour is a production incident waiting to happen.

**Wrong path that happened to work.** Over-retrieval, tool flailing, or reasoning-action mismatch (the
model says one thing and does another) can still land on a correct answer and will not next time.

**Silent degradation.** Because there are **no reliable error codes** `[T]`, a failing agent and a
thinking agent look alike in the logs. Only the step sequence distinguishes them.

The corpus's failure-mode vocabulary is worth knowing precisely because these are the patterns to write
assertions for: **reasoning-action mismatch, over-retrieval, tool flailing, premature commitment,
self-jailbreaking** `[T]`.

Practically, trajectory evaluation requires the spans to exist — one span per LLM call, per tool call,
per retrieval — which is why T17 and T16 are the same conversation from two angles. And it needs
assertions rather than scores: "no tool called with an argument not present in the user's request" is
checkable, where "was this a good trajectory" is a judge call.

**Signal:** Names trajectory failure modes specifically, and connects the missing error codes to why
outcome-only evaluation cannot detect them.

**Follow-ups:**
- *Which failure mode is most dangerous?* — right answer by an unsafe path, or self-jailbreaking.
- *What do you need to evaluate trajectories?* — spans, not just inputs and outputs; T17.
- *How do you grade a trajectory?* — assertions first, judges for the rest.

**Red flags:** Treats agent evaluation as a success-rate metric, or has no vocabulary for trajectory
failure.

---

### The serving stack for agents

#### T16-Q14 · Why does KV offload matter so much for agents?
**Difficulty:** L4 · **Depth expected:** 3–4 min

**Question:** Explain the KV offload tier and why the corpus reports roughly 5× TTFT improvement from
it `[T]`.

**Model answer:** Because the agent's defining behaviour — waiting — is exactly what destroys the cache.

**The mechanism of loss.** An agent issues a turn, gets a tool call, and then waits for the tool. That
gap can be seconds to minutes. During it, the session's KV is a candidate for eviction: it is holding
HBM that live traffic wants, and there is no active request using it. When the tool returns and the
next turn arrives, a stack without an offload tier rebuilds the entire prefix from scratch. That is the
**28k-input recomputation cliff** `[T]` ROCm/WideEP appearing in agentic form — the same failure, driven
by the agent's own idle gap.

**What the offload tier changes.** Instead of discarding the KV, you move it down a tier — CPU/DRAM —
and bring it back when the session resumes `[T]` llm-d. Moving it back is far cheaper than recomputing
it, because a transfer is a copy while a recompute is a full prefill over the whole prefix. The corpus
reports roughly **5× TTFT improvement on session return** `[T]`.

**Why retention is explicit.** A **retention API** exists to express how long a session's KV should be
kept `[T]`. You need this because keeping everything is not an option (T16-Q6) — retention is a policy
decision with a memory cost, and it must be expressible per session or per class of session.

**The router's role.** The corpus is precise here, and it is a detail candidates miss: the router
consumes **per-request KV create AND KV evict events**, not just creates `[T]`. Eviction is a normal,
evented part of the lifecycle, so routing decisions can account for a session whose KV has been
offloaded versus one whose KV is resident. Note also that eviction here is routine cache management,
not a pathology — the offload tier and retention API are what make it orderly.

**Signal:** Connects the idle gap to eviction to recompute, explains transfer-versus-recompute as the
core trade, and knows the router sees evict events as well as create events.

**Follow-ups:**
- *Why does the router care about evict events?* — routing decisions depend on where KV actually lives.
- *What does the retention API express?* — how long a session's KV is kept `[T]`.
- *What is the memory cost of offloading everything?* — the CPU tier becomes the constraint.

**Red flags:** Describes offload as a pure win with no retention policy, or believes the router only
needs create events.

---

#### T16-Q15 · Prefix-cache-aware routing — why is round-robin wrong?
**Difficulty:** L4 · **Depth expected:** 3–4 min

**Question:** Your agent service round-robins across replicas. What does that cost, and what should it
do instead?

**Model answer:** Round-robin is the single most expensive default in an agentic deployment, because it
guarantees the prefix cache misses.

**The mechanism.** A prefix cache is local to a replica. Turn 1 goes to replica A and builds a KV entry
for the prefix. Turn 2 round-robins to replica B, which has never seen the prefix, so it pays full
prefill. Every turn after that misses again, on a prefix that grows each time. You have paid the
quadratic cost from T16-Q3 in full, while believing you built a cached system.

**What it costs, in the corpus's own ladder.** The published cost ladder is **100 → 42 → 26 → 11** `[T]`.
The steps correspond to successively better serving behaviour, and the corpus's guidance is explicit
that **prefix-cache-aware routing is the difference between the 26 and the 11** `[T]`. That is a factor
of ~2.4 within the serving layer alone, from routing policy.

**What to do instead.** Route on the prefix: hash or match the stable prefix and send the request to a
replica that holds it — or, better, one whose KV the router knows about via its create/evict events
(T16-Q14). The corpus summarises the rule as **never round-robin an agent** `[T]`.

**Two caveats worth volunteering.** First, there is a real tradeoff with load balance: prefix affinity
concentrates traffic, so you need a policy that breaks affinity when a replica is saturated. Second,
this only works if the **client keeps the prefix stable** — a client that rewrites its history defeats
prefix routing entirely, which is why T16-Q8 and this question are paired.

**Signal:** Identifies the cache-locality mechanism, cites the 26→11 step of the ladder, and notes the
load-balance tension rather than presenting affinity as free.

**Follow-ups:**
- *What breaks if the prefix is unstable?* — the whole routing strategy; fix the client.
- *How do you avoid hot-spotting a replica?* — affinity with a saturation escape hatch.
- *Where does this sit in the routing layers?* — L2 performance/prefix; T14.

**Red flags:** Defends round-robin as fair, or attributes the cost ladder entirely to model choice.

---

#### T16-Q16 · Would you run agents and chat on the same pool?
**Difficulty:** L4 · **Depth expected:** 4 min

**Question:** You already serve interactive chat. Should agent traffic share those replicas or get its
own pool?

**Model answer:** I would start shared for small agent volumes and plan to separate, and the reasoning
is about interference rather than efficiency.

**The case for sharing.** Both are LLM inference on the same models and the same engine. Sharing
amortises your warm floor (T15) across two workloads, and at low agent volume a dedicated pool would be
mostly idle.

**The case for separating — three interference mechanisms.**

**Latency-class mismatch.** Chat is interactive with a human waiting; agent turns are machine-paced and
often batch-tolerant. Mixing them means either the agent traffic suffers chat's latency SLO or chat
suffers the agent traffic's queueing. The corpus's agentic numbers are stark: **98% prefill** `[T]`
means a burst of agent turns is a burst of long prefills, which is precisely the traffic that hurts an
interactive TTFT SLO.

**Concurrency-class mismatch.** Parallel tool calling multiplies serving-tier concurrency, and the
tuned agentic deployment runs at a **~20,000-concurrency** class with a **prefill-heavy replica ratio**
(`prefill: 2, decode: 4` in the corpus's example) `[T]`. That is a different shape from a chat
deployment, and one pool cannot be two shapes at once.

**Fairness interference.** The documented agentic hazard is **one large session monopolising dispatch
while short sessions starve** `[T]`. Session-level fairness fixes it *among agents*; a chat request
competing with agent sessions is a different fairness problem, because a chat request is not a session
with accumulated service.

**The deciding question** is not "can one pool serve both" — it can — but "what is my blast radius when
an agent workload goes pathological?" A runaway agent loop is a self-inflicted denial-of-service `[D]`,
and separation is what stops it from taking chat down with it.

So: share while agent volume is small and budgets are enforced; separate once agent traffic is a
material share, and separate prefill from decode within the agent pool.

**Signal:** Frames it as interference across latency class, concurrency shape and fairness — and names
blast radius as the deciding factor.

**Follow-ups:**
- *Which interference hits first?* — prefill bursts against the interactive TTFT SLO.
- *What makes separation affordable?* — the same warm-floor logic as T15, applied twice.
- *What must be true to share safely?* — enforced budgets and session-level fairness.

**Red flags:** Answers on efficiency alone with no discussion of interference or blast radius.

---

#### T16-Q17 · Session-level fairness — state the hazard correctly
**Difficulty:** L4 · **Depth expected:** 3–4 min

**Question:** The corpus reports **2–3× lower latency** after changing scheduling for agentic
workloads `[T]`. What changed, and what was the failure it fixed?

**Model answer:** The failure is **one large session taking all the dispatch cycles while short sessions
starve** `[T]` llm-d. The direction matters and candidates frequently invert it.

**Why the large session wins under FCFS.** An agent program is a sequence of turns, and a long-running
program keeps presenting work. Under first-come-first-served at the *request* level, that continuous
demand competes against a short session that has only a few requests to make. The long session gets
served disproportionately because it is always there, and the short session's few requests queue behind
it. Short sessions are not slow because they are hard; they are slow because they are outnumbered by a
neighbour that never stops.

**The fix.** Schedule by **least-attained service** measured over the **session** (the agentic program),
not the request: `attained(session) = service_received / demand` `[T]`. A session that has received
little relative to what it needs gets priority, whatever its request count.

**Why it improves everything, not just fairness.** This is the counter-intuitive part worth pressing on.
By favouring the short sessions, they finish quickly — **"leaving the room for larger sessions to also
finish much faster"** `[T]`. The fleet stops holding many partially-complete programs at once. The
corpus reports **2–3× lower request latency** as the outcome `[T]`.

**The companion policy: turn priority.** Finish the agents nearest completion, so their KV is freed and
the working set shrinks `[T]`. It looks unfair — it helps whoever is nearly done — but it is right for
the same reason shortest-job-first is right: it reduces the number of live sessions competing for
memory.

Together these are the fairness layer, and they only work if the queue is per *session*, which means the
scheduler needs session identity — an interception-layer concern (T16-Q11).

**Signal:** States the starvation direction correctly (big session starves short ones), explains the
FCFS mechanism, and knows why helping short sessions helps long ones too.

**Follow-ups:**
- *Write the fairness metric.* — `attained = service_received / demand`, per session.
- *Why is turn priority right despite looking unfair?* — it frees KV and shrinks the working set.
- *What does the scheduler need to do this?* — session identity, carried from the interception layer.

**Red flags:** Inverts the direction — says short sessions starve long ones — or treats fairness as a
mean-latency optimisation with no mechanism.

---

#### T16-Q18 · Admission control and the P99 problem
**Difficulty:** L4 · **Depth expected:** 3–4 min

**Question:** Your mean latency is good but P99 is terrible during agent bursts. What is missing?

**Model answer:** Admission control — a saturation gate. Mean latency can be good while the tail is
catastrophic, because the mean is dominated by the many requests that arrive before saturation and the
tail is every request that arrives after.

**Why agentic traffic makes this acute.** Parallel tool calls multiply concurrency `[D]`, and agent
bursts are machine-paced, so the fleet can be driven past saturation faster than a human-paced workload
ever would. Once past saturation, queue depth grows without bound and every new request's queue wait
compounds — the classic hockey-stick.

**The fix is to refuse work rather than queue it** `[T]`. A saturation gate admits requests up to a
capacity limit and rejects or sheds beyond it, so admitted requests keep meeting their SLO. The corpus
lists "P99 terrible despite good mean" with the first check being *no admission control* `[T]`.

**What to gate on.** Not CPU. The right signal is KV-cache utilisation and active-request count. The
corpus's own saturation examples are the concrete form: alert on **KV at 80%** and on **active requests
above 8** `[T]` llm-d. Those are the thresholds at which the system is about to lose the property that
makes it fast — resident KV and a short queue.

**What to do with the shed work.** For agents, shedding is unusually acceptable: an agent loop can
retry, and a short delay is often harmless. That is a real advantage over interactive chat, where a
shed request is a human staring at a spinner. But shed work must be *retried intelligently* — with
backoff and a budget — or the retry storm recreates the saturation you just relieved.

**Why goodput is the right SLO here.** A latency percentile can be gamed by shedding; goodput — the
fraction of requests meeting both TTFT and ITL targets — cannot (T15).

**Signal:** Diagnoses saturation rather than a latency bug, names KV utilisation and active requests as
the gate signals, and connects shedding to goodput as the honest SLO.

**Follow-ups:**
- *Why not gate on CPU?* — it does not reflect LLM saturation; see T15.
- *Why is shedding more acceptable for agents than chat?* — the loop can retry with backoff.
- *What happens if retries are unbudgeted?* — the retry storm recreates saturation.

**Red flags:** Adds replicas without an admission gate, or proposes gating on CPU utilisation.

---

#### T16-Q19 · Cost grows with turn count. Diagnose it.
**Difficulty:** L4 · **Depth expected:** 3–4 min

**Question:** A team reports that agent cost per task grows faster than linearly with the number of
turns. Walk through the diagnosis and the fix.

**Model answer:** Super-linear growth is the signature of a **prefix cache that is not being hit**, and
the arithmetic says so directly.

**The mechanism.** Each turn's input includes all previous turns. If the cache holds, turn N pays only
for its new tokens and total cost is linear in turns. If the cache misses, turn N pays full prefill over
an N-sized prefix, and the total is **quadratic** `[D]`:

```
extra_cost(turn) ≈ prefix_tokens(turn) × (fresh_token_cost − cached_token_cost)
```

With **cache reads ~10× cheaper** than fresh tokens `[T]`, that difference is large, and it is the single
largest avoidable cost in an agentic system.

**The diagnosis, in order** `[D]`:

1. **Measure the prefix cache hit rate**, per pod and over time. A low or falling rate confirms the
   hypothesis immediately. This is the corpus's named signal for exposing bad routing `[T]` llm-d.
2. **Check the routing policy.** Round-robin guarantees misses (T16-Q15). Fixing this is the **26 → 11**
   step of the cost ladder `[T]`.
3. **Check the client's history construction.** If the harness rewrites the message list, reorders it,
   or injects a timestamp early in the prompt, no router can help. This is a client bug that presents as
   a serving cost (T16-Q8).
4. **Check KV retention across idle gaps.** If the KV is discarded during tool waits, every turn is a
   cold turn. This is the offload-tier question (T16-Q14).
5. **Check for condensation.** Summarisation *reduces* hit rate by design `[T]`; if it was added
   recently, it may be the cause (T16-Q12).

**The fix follows from which of those fired.** In my experience the first two account for most of it,
and both are cheap: prefix-aware routing and an append-only client.

**Signal:** Recognises quadratic growth as a cache-hit-rate signature, and orders the diagnosis from the
cheapest check to the most expensive.

**Follow-ups:**
- *What is the single number to look at first?* — prefix cache hit rate, per pod.
- *How much is at stake?* — a factor of ~2.4 in the corpus's own ladder (26 → 11).
- *What if the client is a blackbox?* — routing and offload only; T16-Q4.

**Red flags:** Attributes cost growth to model pricing or output length, or proposes quantisation as the
first fix.

---

#### T16-Q20 · Capacity-plan an agentic deployment
**Difficulty:** L5 · **Depth expected:** 6–8 min

**Question:** You must size a deployment for an agent product. You have a target: N tasks per hour at a
p95 completion time. Walk me through the capacity model.

**Model answer:** I would build it from tokens per task, because that is the unit the multiplier operates
on, and I would refuse to size from requests.

**Step 1 — tokens per task.** With **98% of tokens in prefill** `[T]`, the model is `[D]`:

```
tokens_per_task ≈ Σ_turns (prefix_tokens(t) + output_tokens(t))
output_tokens ≈ 2% of total, so ≈ negligible for capacity
prefix_tokens(t) grows with t  →  Σ is quadratic UNLESS cached
```

The cached case is `≈ prefix(L) + Σ new_tokens(t)` — linear, and dramatically smaller. **So the first
question in any capacity model is whether the prefix is cached**, because the answer changes the
requirement by a large factor. I would model both and treat closing the gap as an engineering task
(T16-Q15, T16-Q19).

**Step 2 — convert tokens to GPU-seconds.** Prefill and decode throughput per GPU come from a benchmark
of your own deployment, not from a spec sheet. Because the workload is prefill-dominated, prefill
throughput is the number that matters, and **prefill and decode should be sized separately** — the
corpus's agentic shape is a prefill-heavy replica ratio, e.g. `prefill: 2, decode: 4` `[T]`.

**Step 3 — concurrency, not just throughput.** Because agents **idle 99.999% of the time** `[T]` and
parallel tool calls multiply in-flight requests, the binding constraint is usually KV capacity — how
many sessions you can hold resident — not raw compute. The tuned agentic deployment's **~20,000
concurrency on a 1P1D pair** `[T]` is the class of answer, and the **28k-input cliff** `[T]` is the
failure to size against.

**Step 4 — the warm floor.** Add the T15 logic: the floor is sized by the **ramp**, not the average, and
cold boot is a **15–20 s** floor you cannot beat reactively.

**Step 5 — state the falsifier.** The model is wrong if prefix cache hit rate comes in below assumption,
so I would instrument it from day one and re-derive.

**Signal:** Leads with tokens-per-task and cached-versus-uncached as the pivotal variable, sizes prefill
and decode separately, and identifies KV concurrency rather than FLOPs as the usual binding constraint.

**Follow-ups:**
- *What single assumption most changes the answer?* — cache hit rate; it moves the total by a large
  factor.
- *Why size prefill and decode separately?* — different bottleneck shapes; the corpus's ratio is
  prefill-heavy.
- *What is the usual binding constraint?* — KV capacity for resident sessions; see T16-Q6.

**Red flags:** Sizes from requests per second, ignores caching, or treats prefill and decode as one pool.

---

### Cost, budgets, multi-agent and long horizon

#### T16-Q21 · Why is cost per task, not per call, the right unit?
**Difficulty:** L4 · **Depth expected:** 3 min

**Question:** Your dashboard reports cost per API call and it looks fine. Why is that the wrong unit for
an agent product?

**Model answer:** Because a task is the unit the customer experiences and the unit the multiplier acts
on, and per-call cost can look flat while per-task cost explodes.

**The mechanism.** An agent task is many calls, and the number of calls is a design variable (T16-Q7).
A change that makes each individual call cheaper — a shorter system prompt, a cheaper model, aggressive
condensation — can *increase* total cost if it causes more turns or more retries. Per-call metrics
cannot see that, because each call still looks fine.

**The concrete example the corpus gives.** Its cost accounting is expressed as `cost_task` with three
terms — fresh prompt tokens, cached prompt tokens at ~10× cheaper, and completion tokens `[T]` — and it
is attributed **per tenant and per agent** `[T]`, precisely so the cost ladder is visible in production
rather than in a blog post.

**The four things per-task accounting buys you** `[D]`:

- **Visibility of the multiplier.** You can see the difference between a task that took 6 turns and one
  that took 40.
- **Attribution.** Per tenant and per agent, which is what makes chargeback and abuse detection possible.
- **Regression detection.** A prompt or model change that improves per-call cost but worsens per-task
  cost is caught.
- **A target to optimise.** Cost per completed task is the number the business cares about, and stating
  it forces the turn-count and caching conversations that actually move it.

The instrumentation requirement is that the trace must carry token counts per call *and* a task
identifier that ties calls together — which is exactly the trace structure T17 describes.

**Signal:** Explains how a per-call improvement can worsen per-task cost, and names the task identifier
as the instrumentation requirement.

**Follow-ups:**
- *What ties calls into a task?* — a task/session ID propagated through the trace.
- *Which change most often improves per-call but worsens per-task?* — condensation, or a weaker model
  that needs more turns.
- *How does this connect to the cost ladder?* — the ladder is expressed per task `[T]`.

**Red flags:** Reports cost per call, or cannot name a change that improves the one and worsens the
other.

---

#### T16-Q22 · Reflexion and self-critique — what do they cost?
**Difficulty:** L4 · **Depth expected:** 3 min

**Question:** A team wants to add a self-critique step so the agent reviews and improves its own answer.
What does that do to your serving economics?

**Model answer:** It **multiplies turns**, and turns are the thing that compounds.

**The mechanism.** Reflexion adds a critique-and-retry cycle to the loop `[T]`. Each critique is a full
inference call with the whole prefix attached, and each retry is another one. So a two-attempt policy is
not "10% more expensive" — it is close to doubling the call count *and* extending the prefix, since the
critique and the revised answer both append to the history.

Put against the T16-Q3 arithmetic, where total tokens are quadratic in turns without caching, adding
critique steps increases both the constant and the exponent's practical effect.

**When it is worth it** `[D]`: when the task has a verifiable quality ceiling that a single pass does not
reach — code that fails tests, maths with a checkable answer, structured output with a schema. The
corpus's own position on verification is that verifier-based grading and outcome checks are how you know
a step earned its cost `[T]`.

**When it is not:** when the critique is unverifiable, because then you are paying twice for a second
opinion of unknown value, and the corpus's warning about judging without measurement applies.

**Three mitigations if you adopt it** `[D]`:
- **Cap the retries.** Two attempts, then return the best. Unbounded critique is a runaway loop (T16-Q8).
- **Make the critique prefix cacheable** by appending rather than rewriting.
- **Measure per-task cost and success rate together**, so you can see whether the second attempt is
  actually buying quality.

The honest framing for a candidate: self-critique is a quality technique with a serving bill, and the
serving bill is paid in the currency this whole topic is about — turns. Adopt it deliberately, with a
cap and a measurement.

**Signal:** Quantifies it as a multiplier on turns and prefix, and requires a verifiable quality signal
to justify it.

**Follow-ups:**
- *When is critique justified?* — verifiable outcomes; the check must exist.
- *What cap would you set?* — a small fixed attempt count, then return best.
- *How do you know it paid?* — per-task cost and success rate measured together.

**Red flags:** Treats self-critique as a free quality win, or adopts it with no retry cap.

---

#### T16-Q23 · Justify every hop in a multi-agent topology
**Difficulty:** L5 · **Depth expected:** 5 min

**Question:** A team proposes a supervisor agent coordinating four specialist agents. What is the serving
cost of that topology?

**Model answer:** Every hop is a **full inference call with its own prefix**, so the topology multiplies
cost by roughly the number of agents involved — and that must be justified, not assumed.

**The arithmetic.** With a supervisor and four specialists, a single task involves the supervisor's turns
plus each specialist's turns, and the supervisor reads the accumulated results. The corpus's guidance is
direct: **multi-agent multiplies cost by the number of agents; every hop is a full inference call with
its own prefix — budget it explicitly** `[T]`.

Two amplifiers make it worse than the naive count `[D]`:

- **Prefixes are not shared.** Each agent has its own history, so each pays its own prefill. There is no
  cross-agent cache reuse unless the agents genuinely share a prefix.
- **Supervisor context grows.** The supervisor accumulates specialist outputs, so its prefix grows
  faster than a single agent's would — meaning the quadratic behaviour applies to the most expensive
  participant.

**The other protocol shapes matter too.** The corpus places **A2A** in this space: agent B becomes a
*tool* of agent A `[T]`. Framed that way, a multi-agent hop is structurally identical to a tool call —
which is clarifying, because it means the same questions apply: is this hop necessary, what does it cost,
and can it be parallelised?

**How I would approach it** `[D]`:
- **Justify each hop** against a single-agent alternative. Specialisation is worth it when the specialist
  has a materially different prompt, toolset or model.
- **Prefer parallelism where the graph allows it**, since independent specialists can run concurrently —
  at the cost of serving-tier concurrency (T16-Q8).
- **Route by difficulty**, sending hops that do not need the frontier model to a smaller one (T14).
- **Budget per task across the whole graph**, not per agent, or one agent's overrun is invisible.

**Signal:** States the multiply-by-agent-count rule, notices that supervisor context growth makes it
superlinear, and requires justification per hop.

**Follow-ups:**
- *Which agent's context grows fastest?* — the supervisor's, since it accumulates outputs.
- *When is specialisation justified?* — materially different prompt, tools or model.
- *How does A2A framing help?* — a hop becomes a tool call, so the same cost questions apply `[T]`.

**Red flags:** Assumes multi-agent improves quality at no cost, or cannot say what a hop costs.

---

#### T16-Q24 · Long-horizon agents — what changes at hours-to-days scale?
**Difficulty:** L5 · **Depth expected:** 5 min

**Question:** Your agents currently run for minutes. The product wants tasks that run for hours or days.
What breaks first, and what do you design for?

**Model answer:** Durable execution breaks first, and it is not a serving-layer fix.

**What breaks, in order** `[D]`:

**Session survival.** A task running for hours will outlive a process, a node, or a deployment. Without
**durable execution and checkpointing**, a preempted agent loses its work — the corpus's failure
signature is "session lost mid-task" with *checkpoint/restore* as the fix `[T]`. At minute scale you can
tolerate this; at hour scale losing an hour of work is unacceptable.

**Retention economics.** A session idle for hours cannot hold KV in HBM — the retention question from
T16-Q6 becomes severe, and the answer is the offload tier plus an explicit retention policy rather than
an assumption of residency.

**Memory beyond the window.** Hours of work exceed any context window, so the agent needs an **external
memory** store: the corpus frames it as an external store for facts and history that **trades inference
cost for retrieval cost** `[T]`. That is a real trade, and it is the right one here, because re-reading
hours of history is unaffordable.

**Cost governance.** A task that runs for days can spend without bound. Per-task budgets and step limits
stop being hygiene and become the only thing standing between you and an unbounded bill (T16-Q8).

**What I would design for** `[T]`: the corpus's platform decomposition — **stateless loop, durable
session log, sandbox tier** — because it is precisely what makes long-horizon work survivable: the log
is the record of truth, so any component can be lost and the task resumes. Plus the Agent Substrate
lifecycle of **pause/suspend/resume with a golden snapshot**, targeting a resume in the **low hundreds
of milliseconds** rather than a **10–15 s cold start**, because a long-horizon fleet is mostly suspended
sessions and cannot afford cold starts per wake.

The reframing: at minute scale you are optimising inference; at hour scale you are designing a durable
distributed system whose compute happens to be inference.

**Signal:** Names durable execution first and explains why the serving optimisations stop being
sufficient — then reframes the problem as distributed-systems design.

**Follow-ups:**
- *What is the record of truth?* — the durable session log, not the loop process.
- *How do you handle history exceeding the window?* — external memory; it trades inference for retrieval
  cost `[T]`.
- *Why does the resume target matter more at this scale?* — mostly-suspended fleets cannot pay cold
  starts per wake.

**Red flags:** Treats long-horizon as "the same agent, run longer," with no durability or memory design.

---

#### T16-Q25 · Harness internalisation — how does it change your architecture?
**Difficulty:** L5 · **Depth expected:** 4–5 min

**Question:** The corpus describes harness internalisation — scaffolding moving *into* the model `[T]`.
What does that mean for someone building an agent-serving platform?

**Model answer:** It means the call volume per task falls over time, and the platform should be built so
that this changes your economics without invalidating your architecture.

**What the trend is.** Today much of an agent's competence comes from scaffolding outside the model: a
ReAct loop, a retry policy, tool-selection heuristics, prompt templates. The corpus's observation is that
this scaffolding is being **internalised** — the model absorbs it, so the same task needs **fewer calls**
`[T]`. The corpus's own framing of the research direction is "agentic modeling via internalizing agent
harnesses."

**What it changes** `[D]`:

- **Fewer turns per task**, which reduces the multiplier from T16-Q7 directly. The quadratic prefix
  behaviour softens.
- **Longer, more autonomous single calls** — so the workload shifts back toward the decode side and away
  from the 98/2 prefill/decode split that defines today's agentic serving.
- **Different failure modes** — an agent that plans internally gives you fewer observable steps to
  evaluate, which makes trajectory evaluation harder even as it makes serving cheaper (T16-Q13).

**What it does not change** `[D]`: the serving primitives. Prefix caching, KV offload, session fairness,
admission control and budgets are all still required; the traffic shape changes, not the requirements.
That is the architectural point: a platform built on those primitives **survives** internalisation, while
one built on assumptions about turn counts does not.

**The design implication I would draw.** Avoid hard-coding turn-count assumptions into capacity models
and SLOs. Express them as measured workload-shape parameters that can be re-derived, exactly as T11's
"no universal winner" principle demands for parallelism. And keep the eval suite trajectory-aware, since
internalisation reduces the trajectory you can observe without reducing your need to evaluate it.

**Signal:** Correctly predicts fewer calls and a shift away from prefill-dominance, while holding that
the serving primitives survive unchanged.

**Follow-ups:**
- *What happens to the 98/2 split?* — it shifts toward decode as calls get longer and fewer.
- *Which primitives survive?* — all of caching, offload, fairness, admission, budgets.
- *What gets harder?* — trajectory evaluation, with fewer observable steps.

**Red flags:** Treats internalisation as making the serving stack irrelevant, or as having no effect.

---

#### T16-Q26 · What would you refuse to build?
**Difficulty:** L5 · **Depth expected:** 4–5 min

**Question:** You are asked to ship an agent platform. Name the things you would refuse to do, and why.

**Model answer:** Five, and each is a case where the short-term convenience is a long-term structural
problem.

**Unbounded agents.** No token, step or wall-clock budget `[D]`. An agent without a budget is a
denial-of-service against your own fleet, and the corpus lists "agent loops forever" as a known failure
with *budget + termination check* as the fix `[T]`. Refusing this is refusing to be woken at 3am by your
own product.

**Round-robin routing for agents** `[T]`. It guarantees prefix cache misses and, per the corpus's own
cost ladder, it is the difference between the 26 and the 11 `[T]`. Refusing it is cheap; retrofitting
prefix affinity later means revisiting every deployment.

**Untrusted tool output without a rail.** With no instruction/data separation `[T]`, tool output is
adversarial input by default. A tool boundary without a rail is a documented incident path (T18).

**A shared pool with no isolation once agent volume is material.** Not because sharing is wrong, but
because a runaway agent loop should not be able to take down interactive chat (T16-Q16).

**Uninstrumented trajectories.** Because there are **no reliable error codes** `[T]`, an agent platform
without spans cannot distinguish failure from thinking, and cannot evaluate the paths that produce the
incidents. Shipping agents you cannot observe is shipping a system you cannot operate (T17).

The unifying principle: every one of these is a case where the correct design decision is available at
build time and expensive afterwards. The refusal list is not conservatism — it is a statement about which
decisions are one-way doors.

Worth adding: I would *not* refuse to serve blackbox harnesses, or to run agents and chat together at low
volume. Those are tractable with the serving-layer levers. Refusing everything is as unhelpful as
refusing nothing.

**Signal:** Names structural one-way doors rather than general caution, and includes at least one
observability item alongside the safety and cost ones.

**Follow-ups:**
- *Which of these is hardest to retrofit?* — routing policy; it touches every deployment.
- *Why is observability on a refusal list?* — the missing error codes make it load-bearing, not optional.
- *What would you happily accept?* — blackbox harnesses and shared pools at low volume.

**Red flags:** Gives vague caution rather than specific refusals, or refuses so much the product cannot
ship.

---

#### T16-Q27 · Design the state model for a resumable agent
**Difficulty:** L5 · **Depth expected:** 6–8 min

**Question:** Whiteboard the state model for an agent that can be preempted at any point and resume in
under a second. What is durable, what is ephemeral, and where does KV fit?

**Model answer:** I would separate three kinds of state by their recovery cost, which is the only
distinction that matters.

**Durable — the record of truth.** The session log: message history, tool results, decisions, and the
task's budget counters. This must survive process, node and deployment loss. It is what resume rebuilds
from, and the corpus's decomposition names it explicitly — **stateless loop plus durable session log
plus sandbox tier** `[T]` Steinder. It should be append-only, which also makes it a natural prefix-cache
key (T16-Q8).

**Disposable — the loop.** In-flight reasoning, scratch state, the current tool invocation. Losing it
costs at most one step, provided the log has the last committed step. This is why the loop is stateless:
it makes preemption free.

**Cached — the KV.** This is the interesting one, because it is neither truth nor garbage — it is a pure
performance artefact that can be recomputed but at high cost. Three tiers `[T]` llm-d:
- **HBM** for hot, active sessions;
- **CPU/DRAM offload** for sessions idle between turns, with the corpus reporting roughly **5× TTFT
  improvement on session return** `[T]`;
- **dropped**, for sessions past their **retention** window, expressed via the retention API `[T]`.

The KV is never the record of truth. Losing it costs latency, not correctness. That property is what
makes the whole design work: you can evict freely under memory pressure without endangering a task, and
the router can treat create and evict events as ordinary lifecycle signals `[T]`.

**The resume path.** On wake: load the session log, find the last committed step, restore or recompute
KV, resume the loop. The target is **low hundreds of milliseconds** against a **10–15 s cold start** `[T]`
Steinder — achievable precisely because the first two steps are cheap and the third is the only expensive
one, and because the **golden snapshot** gives a known-good starting state rather than a reconstruction.

**The failure I would design against:** a log write that succeeds but whose side effect did not, or vice
versa. That is the classic exactly-once problem, and it is why steps should be idempotent and committed
before their side effects are considered durable. The corpus's long-horizon material lands on the same
point — checkpointing is what stops a preemption costing hours of work `[T]`.

**Signal:** Separates state by recovery cost rather than by component, places KV firmly in the
recoverable-performance tier, and raises the commit-ordering problem unprompted.

**Follow-ups:**
- *Why can KV be evicted freely?* — it is a performance artefact, not truth; recomputable.
- *What makes resume fast?* — cheap log load plus snapshot restore; only KV restoration is expensive.
- *What is the hard correctness problem?* — commit ordering versus side effects; use idempotent steps.

**Red flags:** Puts session state in the loop process, treats KV as durable truth, or has no answer for
partial-failure during resume.

---

#### T16-Q28 · What is the biggest open problem in agentic inference?
**Difficulty:** L5 · **Depth expected:** 5 min

**Question:** Give me your view: what is the most important unsolved problem in serving agents at scale?

**Model answer:** I would argue it is **observing and controlling failure in a system with no error
codes**, and I would defend that over the more obvious candidates.

**Why not the obvious ones.** Turn latency, caching and fairness all have known solutions in this
corpus — prefix-aware routing, KV offload, least-attained service, admission gates. They are engineering
work, not open problems. Capacity is arithmetic.

**The actual gap.** The corpus states it plainly: **"we essentially do not have any reliable error
codes"** `[T]`. Combined with **no instruction/data separation** `[T]`, this means:

- **Failure is indistinguishable from work.** A tool call that failed, one that hallucinated an API, and
  one that is still reasoning all arrive as ordinary text. You cannot alert on it without evaluating
  content, which means every failure detector is a judge call with its own accuracy.
- **Success is indistinguishable from lucky failure.** An agent can reach a correct answer by an unsafe
  path (T16-Q13), and outcome metrics will call it a success.
- **The instruction set is open-ended** `[T]`, so you cannot enumerate the failures in advance and write
  a test for each.

**What would count as progress** `[D]`:

- **Structured failure signalling** — agents and tools emitting typed outcomes rather than prose, so the
  serving layer can retry, alert and account reliably.
- **Cheap trajectory evaluation at scale**, since trajectory is the only place these failures are
  visible, and full-fidelity evaluation is currently too expensive to run on everything (T17's layered
  judge architecture is the current best answer).
- **Budgets that are enforced structurally** rather than by inspecting content — a bound on steps and
  tokens works regardless of whether you can classify the failure.

**Why it is the biggest.** Every other problem in this topic is a performance or cost problem with a
measurable target. This one is an *operability* problem: without it you cannot tell whether your agent
platform is working, and you cannot improve what you cannot measure.

**Signal:** Picks a specific, defensible problem rather than a generic "scale," justifies it against the
stronger-sounding alternatives, and proposes what progress would look like.

**Follow-ups:**
- *Why not latency or cost?* — those have known solutions in this corpus; they are work, not unknowns.
- *What is the current best answer?* — layered judges plus assertions on trajectories; T17.
- *What would you build first?* — structured failure signalling, since everything else depends on it.

**Red flags:** Names a solved problem, or gives an answer with no argument for why it outranks the
alternatives.

---

## Whiteboard exercises

### Exercise 1 — Diagnose runaway agent cost
**Prompt.** "Our agent product's cost per task has grown 5× over two months while usage grew 1.5×. The
dashboard shows per-call cost is flat. Turn count per task is up from an average of 7 to 11. Find it."

**What the candidate must produce:** a hypothesis with a mechanism that explains flat per-call cost and
rising per-task cost simultaneously, a ranked diagnostic list, and a fix with an expected magnitude.

**Expected answer sketch:**

```
Given: per-call cost FLAT, turn count 7 -> 11, per-task cost x5
Key identity:  cost_task = turns x avg_cost_per_turn
               turn count rose 1.57x, but cost rose 5x
            => avg_cost_per_turn rose ~3.2x  = THE REAL SIGNAL

Why would cost per turn rise with turn count?
  [D] per-turn cost ≈ prefix_tokens(t) x fresh_price
      prefix grows with t  ->  quadratic IF CACHE MISSES
      cached: turn t costs only NEW tokens  -> linear

Ranked diagnosis                                  Check
1. Prefix cache hit rate collapsed            -> hit rate per pod over time      [T]
2. Routing changed (round-robin?)             -> is traffic prefix-affine?       [T]
3. Client history rewritten / reordered       -> diff the message list  [D]
4. KV discarded during longer tool waits      -> offload tier + retention        [T]
5. Condensation added                         -> hit rate drop after deploy      [T]

Magnitude at stake: corpus cost ladder 100 -> 42 -> 26 -> 11;
  routing alone is the 26 -> 11 step  = ~2.4x                       [T]
  cache reads ~10x cheaper than fresh tokens                        [T]

Fix: restore prefix-aware routing + append-only history
     then RE-MEASURE hit rate to confirm the diagnosis was right
```

**Grading rubric (full marks requires all four):**
- Derives that **cost per turn**, not cost per call, is the quantity that moved — and says so before
  listing causes.
- Explains the quadratic-prefix mechanism and ties it explicitly to cache misses.
- Orders the diagnosis cheaply-first, and includes the client-side history check (not just serving).
- Quantifies the prize (the 26→11 ladder step, ~2.4×) and insists on re-measuring the hit rate to
  confirm.

---

### Exercise 2 — Fix a fairness complaint
**Prompt.** "Customers running long agent programs complain their tasks take far longer than expected.
Simultaneously, customers running short tasks complain about slow responses and timeouts. Both groups
are right. Diagnose and design the fix."

**What the candidate must produce:** the correct direction of starvation, the mechanism, the scheduling
change, and the reason the change helps both groups.

**Expected answer sketch:**

```
The hazard (get the DIRECTION right)                    [T] llm-d
  ONE LARGE SESSION takes all the dispatch cycles
  -> SHORT sessions starve
  (not the reverse)

Why, under request-level FCFS                    [D]
  a long program always has another turn queued -> it is
  always present in the queue; a short session has only a
  few requests and waits behind the continuous demand

The fix: schedule per SESSION, by least attained service
  attained(session) = service_received / demand           [T]

Why it helps BOTH groups (the counter-intuitive part)
  short sessions finish fast
  -> "leaving the room for larger sessions to also finish
      much faster"                                       [T]
  -> fewer partially-complete programs held at once
  -> reported outcome: 2-3x lower request latency         [T]

Companion policy: TURN PRIORITY -- finish agents nearest
  completion so their KV is freed and the working set shrinks [T]

Requirement: the scheduler needs SESSION IDENTITY
  -> carried from the interception layer / gateway       [T]

Also add: a saturation gate (KV 80%, active reqs > 8)    [T]
  because a good mean with a bad P99 is an admission problem,
  not a fairness problem
```

**Grading rubric:**
- States the starvation direction correctly — the big session starves short ones — and does not invert it.
- Explains the FCFS mechanism (continuous demand beats intermittent demand).
- Explains why the fairness fix *improves the long sessions too*, rather than presenting it as a
  tradeoff between groups.
- Identifies that the P99 complaint may need an admission gate in addition to fairness, and that the
  scheduler needs session identity to work at all.

---

### Exercise 3 — Design the serving tier for an agent product
**Prompt.** "Design the serving layer for an agent product: ~500 concurrent sessions, hour-long tasks,
tool calls taking 5–60 s, a mix of our own harness and one third-party harness we cannot modify. State
your architecture and your three biggest risks."

**What the candidate must produce:** a tiered architecture, the state model, the client contract split
between the two harness classes, and named risks with mitigations.

**Expected answer sketch:**

```
                    ┌──────────────────────────┐
   our harness ────▶│   INTERCEPTION LAYER     │  budgets, identity,
   (class 1)        │  budgets | identity |    │  rails, telemetry
                    │  rails   | telemetry     │  (the only control we have
                    └────────────┬─────────────┘   over the 3rd-party harness)
   3rd-party  ──────────────────▶│
   harness (class 2)             ▼
                    ┌──────────────────────────┐
                    │  PREFIX-AWARE ROUTER     │ consumes KV create AND
                    │  never round-robin       │ evict events          [T]
                    └────────────┬─────────────┘
                                 ▼
              ┌──────────────────┴──────────────────┐
              │  PREFILL pool      DECODE pool      │  sized separately,
              │  (prefill-heavy)   (smaller)        │  98% prefill  [T]
              └──────────────────┬──────────────────┘
                                 ▼
                    ┌──────────────────────────┐
                    │  KV TIERS                │  HBM -> CPU/DRAM (offload)
                    │  retention API           │  -> drop past retention [T]
                    └──────────────────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  SANDBOX TIER            │  separate scaling,
                    │  (tool execution)        │  parallel-capable
                    └──────────────────────────┘
                    ┌──────────────────────────┐
                    │  DURABLE SESSION LOG     │  append-only; the record
                    │  (stateless loop)        │  of truth; resume source
                    └──────────────────────────┘

  STATE BY RECOVERY COST:  durable log (truth) | loop (disposable)
                           KV (recomputable performance artefact)
  TARGETS: resume low-hundreds-of-ms vs 10-15 s cold start     [T]
           ~20,000-concurrency class is the tuned agentic shape [T]
           watch the 28k-input recomputation cliff             [T]

  RISKS
  1. 3rd-party harness rewrites its history -> cache useless.
     Mitigation: prefix-aware routing + offload still help;
     cannot fix the client. Accept the cost, cap it with budgets.
  2. Runaway session -> DoS against the fleet.
     Mitigation: enforced token/step/wall-clock budgets at the
     interception layer + saturation gate on admission.
  3. KV working set exceeds HBM during hour-long tasks.
     Mitigation: tiered KV + retention policy + session-level
     fairness (least-attained) + turn priority to free KV early.

  SIZING NOTE: model both cached and uncached prefix paths;
  the cache hit rate assumption moves the requirement by a large
  factor, so instrument it and re-derive capacity from it.
```

**Grading rubric:**
- Separates prefill and decode pools and justifies it from the 98/2 split rather than by assertion.
- Splits the client contract by harness class, and is explicit that the third-party harness's efficiency
  problems are **not fixable** — only governable via the interception layer.
- Places KV in a recomputable tier distinct from the durable log, and names the resume target.
- Names at least three risks with concrete mitigations, including the runaway-agent budget risk and the
  cache-defeating client behaviour.

---

## Sources

- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt`
  — the 98% prefill / 2% decode split and the TTFT-bound conclusion; KV create *and* evict events
  consumed per request, the offload tier, the retention API; the ~5× TTFT improvement on session return;
  the 2–3× latency improvement from session-level fairness with least-attained service and turn
  priority; the fairness-starvation direction (one large session monopolising dispatch while short
  sessions starve); the saturation examples of KV at 80% and active requests above 8; and "never
  round-robin an agent."
- `refs/Agentic_AI_Infra_transcripts_3/Gosia_Steinder_-_Beyond_Harnesses_Platform_Solutions_for_Agent_Reliability_Secur.txt`
  — the three agent classes (in-house, harness-with-hooks, blackbox); the stateless loop plus durable
  session log plus sandbox tier serverless pattern; the actor, actor template, worker, per-node manager,
  enlightened proxy and golden snapshot; pause/suspend/resume; agents "idle 99.999% of the time"; the
  10–15 s cold start against a low-hundreds-of-milliseconds wake target; and the 200,000-node design
  target.
- `refs/Agentic_AI_Infra_transcripts_2/Saurabh_Tiwary_-_From_Models_to_Agents_to_Discovery_Building_the_Full_Stack_of_A.txt`
  — the 3.2-quadrillion-tokens-per-month scale figure and the full-stack framing.
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Distributed_Inference_on_ROCm_with_WideEP_on_vLLM_llm-d.txt`
  — the ~20,000 max concurrency on a 1P1D pair, and the 28k-input KV-recomputation cliff at concurrency
  256.
- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/How_AI_Agents_Actually_Work_ReAct_Tools_Reflexion.txt`
  — the ReAct reason-act-observe loop, Reflexion's critique-and-retry cycle, and the 100→42→26→11 cost
  ladder.
- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Multi-Agent_Systems_with_LangGraph_The_Klarna_Uber_Lessons.txt`
  — multi-agent topologies and the cost of each hop.
- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Cut_LLM_Cost_Latency_KV_Cache_Batching_Quantization_vLLM.txt`
  — cache reads at roughly 10× cheaper than fresh tokens, and per-request token accounting.
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_11_Agents_and_Multi-Agent_Communication.txt`
  — agent and multi-agent communication, and the A2A framing where one agent becomes another's tool.
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_12_Reward_Models_and_Best-of-N.txt`
  — verifier-based grading and reward models, the basis for when self-critique is checkable.

**Derived content in this bank (`[D]`):** the quadratic-in-turns prefix-cost arithmetic and the
`extra_cost(turn)` expression; the latency decomposition applied to agents; the four-item lists of what
per-task accounting buys and what long-horizon agents break; the ordered diagnostic ladders in T16-Q19
and T16-Q20; the five items on the refusal list; the three-way state separation by recovery cost in
T16-Q27; the risk ranking in Exercise 3; and the "compute is idle, state is not" framing used
throughout. Every derived claim is labelled inline where it appears.
