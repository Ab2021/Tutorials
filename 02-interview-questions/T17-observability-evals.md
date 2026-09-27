# Interview Bank: Observability, Tracing & Evaluation

> `T17` · **Transcript coverage:** primary · [Cheat sheet](../00-cheat-sheets/T17-observability-evals.md) · [Case study](../01-case-studies/T17-observability-evals.md) · [Design blueprint](../03-design-blueprints/T17-observability-evals/HLD.md)
> **Questions:** 28 (8 × L3, 13 × L4, 7 × L5) · **Format:** the three planes, then traces, metrics, evaluation, and the agent-specific problems

## How to use this bank

The failure this bank is built to detect is a candidate who can describe OpenTelemetry but has never
decided what to sample, what to redact, or how they would know their judge is any good. Those decisions
are Q10, Q12 and Q20.

Ask in order for a full loop. The L5 block (Q24–Q28) is where staff-level candidates separate: it asks
what you do when the correct answer is expensive, ambiguous, or both.

---

### Foundations — three planes, and what each actually answers

#### T17-Q1 · What are the three planes of LLM observability?
**Difficulty:** L3 · **Depth expected:** 2–3 min

**Question:** A team says "we have observability — we have Grafana." What are they missing?

**Model answer:** Two of the three planes, and the two they are missing are the ones that answer "why."

The three planes `[D]`:

**Plane 1 — system observability.** Metrics: TTFT, ITL/TPOT, goodput, queue depth, KV-cache
utilisation, error rates, and per-pod prefix cache hit rate `[T]`. This is the ops dashboard. It
answers *is it healthy, fast and affordable?*

**Plane 2 — trace observability.** A span per LLM call, per tool call, per retrieval step, carrying
model, token counts, cost, status and the OpenTelemetry GenAI semantic attributes `[T]`. It answers
*where did this request spend its time, and what did it actually do?*

**Plane 3 — quality evaluation.** Offline and online scoring of outputs *and trajectories* — golden
sets, judges, regression suites, sampled production scoring `[T]`. It answers *was the output good?*

The specific gap in "we have Grafana" is the difference between plane 1 and planes 2–3, and it has a
signature failure: **quality complaints with green dashboards** `[D]`. Everything is up, latency is
fine, the GPUs are busy — and users are unhappy. No metric can tell you that, because the metric you
are missing is not a number about the system; it is a judgement about the output.

A useful way to hold the distinction: **metrics say *something is wrong*; traces say *what*; evals say
*whether it was any good*** `[D]`. A team with only plane 1 can tell you the GPU is busy and nothing
else.

**Signal:** Names all three, and — the discriminator — gives the *signature failure* of having only
plane 1 rather than just listing what is absent.

**Follow-ups:**
- *Which plane do teams most often skip?* — plane 3; it is the one with no off-the-shelf dashboard.
- *Do traces and evals share a pipeline?* — no; different consumers, different retention `[D]`.
- *What ties them together?* — the trace/task ID that lets you pull a real failure into the eval set.

**Red flags:** Equates observability with metrics, or lists tools rather than planes.

---

#### T17-Q2 · Trace versus eval — what is the difference?
**Difficulty:** L3 · **Depth expected:** 90 s

**Question:** A trace and an eval both look at a request. Why do you need both?

**Model answer:** Because they answer different questions about the same object, and neither substitutes
for the other.

**A trace tells you what happened.** It is a record: this call took 1.2 s, of which 90 ms was queue
wait and over 1 s was prefill; it used model X; it made three tool calls; it cost this many tokens. It
is descriptive and it is about *this* request.

**An eval tells you whether it was good.** It is a judgement: the answer was correct, the trajectory was
sound, the retrieval was sufficient. It is evaluative and it is about a *class* of requests.

The failure modes are mirror images `[D]`. With traces but no evals you can debug a slow request
perfectly and have no idea your quality is degrading week over week. With evals but no traces you know
quality dropped and cannot find the request that shows why — you have a score with no evidence.

They also consume different pipelines, which is why conflating them causes engineering problems `[D]`:
traces are high-volume, sampled, retained briefly and mostly discarded; evals are low-volume,
deliberately curated, and retained as a regression suite. Different retention, different cost model,
different owners.

The connection is the handoff: a trace that reveals a failure is how a real production case enters the
eval set. That loop — trace identifies, eval enshrines — is what stops an eval suite from becoming a
set of cases someone imagined in 2024.

**Signal:** States the descriptive-versus-evaluative distinction crisply, and names the
trace-feeds-eval handoff.

**Follow-ups:**
- *Which retains longer?* — the eval set; it is curated and small.
- *How does a real failure become a test?* — pull the trace, label it, add it to the gold set.
- *What does each cost?* — traces dominate volume cost; see T17-Q10.

**Red flags:** Treats one as a special case of the other, or believes a good eval suite removes the need
for traces.

---

#### T17-Q3 · What do you put on a span?
**Difficulty:** L3 · **Depth expected:** 2–3 min

**Question:** You are instrumenting an LLM call. Beyond the obvious, what attributes make a trace
genuinely debuggable?

**Model answer:** The standard attributes get you a record; the application-specific ones get you an
explanation.

**The standard set** `[T]`: the OpenTelemetry GenAI semantic conventions — `gen_ai.system` (`vllm`, say),
`gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`,
`gen_ai.response.finish_reasons`. Plus the structural fields: trace ID, parent span ID, start and
duration, status.

**The attributes I would add** `[T]`/`[D]`, because these are the questions you actually ask at 3am:

- **`app.prefix_cache_hit`** — how many tokens came from cache. Without it you cannot explain a latency
  change after a routing update (T16-Q15).
- **`app.route_replica`** — which replica served this. Routing is the first suspect for both latency and
  cost regressions.
- **`app.tool_calls`** — count, and ideally names and durations.
- **Cacheable/uncached token split, and cost** — the currency the business cares about (T16-Q21).
- **A task or session ID** — so per-call spans aggregate into per-task cost and trajectory.

The nesting matters as much as the attributes. The trace should be a tree: the agent task at the root,
each turn beneath it, each LLM call, tool call and retrieval beneath that. Then a span's duration is
attributable to its parent, and the latency decomposition (T17-Q13) becomes readable off the trace
rather than reconstructed from logs.

The test I would apply: **can I explain a specific slow or wrong request from its trace alone?** If not,
the instrumentation is missing something — and the missing thing is usually an application attribute
rather than a protocol one.

**Signal:** Distinguishes the standard conventions from the application attributes that make traces
useful, and includes at least one cache/routing attribute.

**Follow-ups:**
- *Why is `app.route_replica` worth carrying?* — routing is the first suspect for latency and cost
  regressions.
- *What does nesting buy you?* — attribution of duration to parent spans.
- *What must NOT go on a span?* — verbatim PII; see T17-Q12.

**Red flags:** Lists only the protocol attributes, or puts raw prompt text on the span without a
redaction policy.

---

#### T17-Q4 · Which metrics matter for LLM serving?
**Difficulty:** L3 · **Depth expected:** 2 min

**Question:** Give me the metrics you would put on the LLM serving dashboard.

**Model answer:** Five families, and I would name the ones that are LLM-specific because the generic ones
will already be there.

**Latency, split by phase.** TTFT (time to first token) and ITL/TPOT (inter-token latency / time per
output token). These must be separate metrics, because they measure different phases with different
bottlenecks: TTFT is prefill and queueing, ITL is decode. A single "latency" number hides which one
degraded `[T]`/`[D]`.

**Goodput.** The fraction of requests meeting *both* their TTFT and ITL targets — the honest headline
SLO. A pure latency percentile is gameable by shedding load; goodput is not (T17-Q14).

**Queue depth.** The leading indicator of saturation. Latency degrades only after the queue builds `[D]`.

**KV-cache utilisation.** The resource that actually binds LLM serving. The corpus's own saturation
thresholds are concrete: alert at **KV 80%** and at **active requests above 8** `[T]` llm-d.

**Cache efficiency.** Prefix cache hit rate, **per pod**. The corpus names this as the signal that
exposes bad routing `[T]` llm-d, and it is the first check for both a cost regression and a latency
regression on agentic traffic.

I would add **error rate** and **cost per task** to complete the picture — the latter because for an
agent product the business metric is per task, not per call (T16-Q21).

The organising principle: the dashboard exists to answer *is it healthy, fast and affordable*, and each
metric should map to one of those three words. Metrics that map to none of them are decoration.

**Signal:** Names TTFT and ITL as separate metrics with a reason, and includes KV utilisation and cache
hit rate — the two LLM-specific ones candidates most often omit.

**Follow-ups:**
- *Why not one latency number?* — it conflates queue, prefill and decode.
- *What is the leading indicator of saturation?* — queue depth, then KV utilisation.
- *Which metric detects bad routing?* — per-pod prefix cache hit rate `[T]`.

**Red flags:** Lists generic web-service metrics, or gives one aggregate latency figure.

---

#### T17-Q5 · Why is non-determinism at temperature 0 a real problem?
**Difficulty:** L3 · **Depth expected:** 2 min

**Question:** A test asserts an exact string match on an LLM response and fails intermittently. The
engineer says "temperature is 0, it should be deterministic." Who is right?

**Model answer:** The engineer is wrong, and the corpus states it plainly: **at temperature 0, outputs
still differ** `[T]` CMU lecture 2.

**Why.** Temperature 0 makes the sampling greedy, but it does not make the computation deterministic.
Floating-point addition is not associative, so the order in which a reduction is performed changes the
result. That order depends on batch composition, kernel selection, and how many sequences are in flight
— none of which are fixed in a serving system. Two identical prompts batched differently can produce
different logits at the margin, and near a tie a tiny difference flips the token.

The corpus adds two further facts that matter operationally: **quantization worsens the
non-determinism**, and it **fingerprints the model** `[T]` — that is, the quantized artefact has its own
behavioural signature, distinct from the full-precision one.

**The engineering consequences** `[D]`:

- **Never assert exact string equality** on model output. Assert properties: does the JSON parse, does
  it contain the required field, does the tool call have the right shape, does the SQL run.
- **Do not treat a single differing output as a bug.** It is the expected behaviour of the system. The
  bug is the test.
- **Use statistical assertions for quality** — score a set, compare distributions — rather than
  comparing single outputs.

The reason this belongs in an observability bank rather than a testing bank is that it also constrains
what your *eval* suite can promise. An eval that reports a pass/fail on one run of one case is reporting
noise. That is part of why the corpus's more rigorous agent benchmarks use repeated trials (T17-Q23).

**Signal:** Names the floating-point non-associativity mechanism rather than saying "GPUs are random,"
and derives the testing rule from it.

**Follow-ups:**
- *What does quantization do to it?* — worsens it, and fingerprints the model `[T]`.
- *What kind of assertion works?* — property and schema assertions, not string equality.
- *How does this affect eval design?* — you need repeated trials, not single runs.

**Red flags:** Believes temperature 0 guarantees determinism, or classifies this as a model bug.

---

#### T17-Q6 · What is trajectory evaluation and why does it matter?
**Difficulty:** L3 · **Depth expected:** 2 min

**Question:** Your agent's task success rate is high. Why would you still evaluate the trajectory?

**Model answer:** Because a correct answer reached by an unacceptable path is a production incident that
your success metric is actively hiding.

Outcome evaluation asks whether the task completed. Trajectory evaluation asks whether the path was
sound — right tools, right order, no unsafe steps, no absurd retries `[D]`. For a single-turn LLM call
the two nearly coincide; for an agent they diverge sharply.

The corpus's failure-mode vocabulary names what lives in the gap: **reasoning-action mismatch** (the
model says one thing and does another), **over-retrieval**, **tool flailing**, **premature commitment**,
and **self-jailbreaking** `[T]`. None of these necessarily produces a wrong answer on this run. All of
them are paths you would not sanction.

There is a second reason specific to this topic: because there are **no reliable error codes** `[T]`, a
failing agent and a working agent look identical in the logs. The only place the difference is visible
is the sequence of steps. So trajectory evaluation is not a refinement of outcome evaluation — for
agents it is the primary detection mechanism for a whole class of failure.

It also requires the spans to exist. You cannot evaluate a trajectory you did not record, which is why
T16 and T17 are the same conversation from two angles: the trace structure is the prerequisite for the
evaluation.

Practically, trajectory evaluation is best done with **assertions before scores** `[D]` — "no tool was
called with an argument absent from the user's request" is checkable and cheap; "was this a good
trajectory" is a judge call that needs validation (T17-Q20).

**Signal:** Names at least two trajectory failure modes and connects trajectory evaluation to the absent
error codes as a *detection* mechanism, not just a quality nicety.

**Follow-ups:**
- *Which failure mode is most dangerous?* — self-jailbreaking, or a right answer by an unsafe path.
- *What does it require?* — spans; you cannot evaluate what you did not record.
- *Assertions or judges?* — assertions first, judges for the residue.

**Red flags:** Treats task success rate as sufficient for an agent, or has no vocabulary for trajectory
failure.

---

#### T17-Q7 · Cost as an observability signal
**Difficulty:** L3 · **Depth expected:** 2 min

**Question:** Why does cost belong on the observability dashboard rather than in a monthly finance
report?

**Model answer:** Because by the time it reaches finance it is unactionable. Cost is a real-time
behavioural signal about the system, and it is the only signal that catches several classes of
regression.

**What cost telemetry consists of** `[T]`: tokens in and out per request, the cached/fresh split, and
cost — attributed per request, per tenant, per model, per agent. The corpus's cost model is expressed
this way, with `cost_task` built from fresh prompt tokens, cached prompt tokens at roughly **10×
cheaper**, and completion tokens `[T]`.

**What it detects that latency does not** `[D]`:

- **Prefix cache collapse.** A routing change or a client change can leave latency acceptable while
  cost doubles, because the prefill work is still being done — just unnecessarily. Per-task cost sees
  it; TTFT may not (T16-Q19).
- **The turn-count multiplier.** Cost per task rising while cost per call stays flat is the signature of
  a turn-count regression (T16-Q21).
- **Tenant abuse or runaway agents.** Per-tenant attribution is what makes an anomalous consumer
  visible before it becomes a budget overrun.
- **Quantisation and model-change value.** Whether a cheaper model actually saved money depends on
  whether it needed more turns — a cost-per-task question.

**Why real time matters.** All of these are regressions you want to catch in the deploy that caused
them, not at month end. The corpus's framing is that attributing cost per tenant and per agent is what
makes the cost ladder **visible in production rather than in a blog post** `[T]`.

The instrumentation requirement is the same as T16-Q21: token counts on every span, plus a task
identifier to aggregate them.

**Signal:** Names a regression class that cost detects and latency does not — ideally the cache-collapse
case — rather than saying "cost is important."

**Follow-ups:**
- *What does per-tenant attribution buy?* — abuse detection and chargeback.
- *Which is the right unit?* — per task; see T16-Q21.
- *What must the trace carry?* — token counts and a task ID.

**Red flags:** Treats cost as a finance concern, or reports only aggregate spend.

---

### Traces and spans

#### T17-Q8 · Draw the latency decomposition
**Difficulty:** L4 · **Depth expected:** 3–4 min

**Question:** A user says a request took 4 seconds. Decompose that into the terms you would instrument.

**Model answer:** Six terms, and every span in a correct trace should map to exactly one of them.

```
end_to_end = queue_wait + prefill(TTFT) + N × ITL + Σ tool_time + retries
```

`[D]`, with each term carrying a different owner and a different fix:

**`queue_wait`** — time before the request was admitted. Grows with saturation; the fix is capacity or
admission control (T16-Q18).

**`prefill`** — this *is* TTFT for a non-queued request. Dominated by prompt length and cache hit rate.
For agentic traffic this is the big term: with **98% of tokens in prefill** `[T]`, prefill is where the
time goes.

**`N × ITL`** — the decode phase. N output tokens at the inter-token latency. Fixed by model, batching
and hardware; usually the *small* term for agents.

**`Σ tool_time`** — the tool calls. For an agent this is frequently the largest wall-clock term and has
nothing to do with the GPU. Critically, it is also where the KV is evicted (T16-Q14).

**`retries`** — anything retried, multiplied by its own cost. Invisible unless instrumented, and the
usual explanation for a tail that will not reconcile.

**The discipline I would enforce** `[D]`: **if a span does not map to a term, your instrumentation is
missing something.** A 4-second request should decompose into spans that sum to 4 seconds. If they sum
to 1.2, you have an uninstrumented hop — most often the gap between the gateway and the engine, which is
exactly the "cannot explain a slow request" failure signature `[D]`.

**A worked example.** Suppose the trace shows 1.2 s of prefill, 90 ms of decode, and over 1 s of tool
time. The conclusion is immediate and non-obvious: the GPU is not the problem, the tool tier is. Without
the decomposition you would have scaled the GPU tier and changed nothing.

**Signal:** Gives all six terms, includes retries (the most-forgotten term), and states the reconciliation
discipline rather than just listing the formula.

**Follow-ups:**
- *Which term dominates for agents?* — prefill and tool time; decode is small.
- *What if the spans do not sum?* — an uninstrumented hop, usually gateway to engine.
- *Which term has no GPU involvement at all?* — tool time, and often retries.

**Red flags:** Gives only "queue + inference," or omits tool time and retries for an agent workload.

---

#### T17-Q9 · Trace propagation breaks. Debug it.
**Difficulty:** L4 · **Depth expected:** 3 min

**Question:** Your traces show a gateway span and an engine span that are unrelated — no parent-child
link. What is broken and why does it matter?

**Model answer:** Context propagation, and it matters more than it sounds because an unlinked trace is
close to useless.

**The mechanism.** Distributed tracing works by carrying trace context — trace ID and parent span ID —
in the request across every hop. If the LLM gateway, the router or the engine does not forward that
context, the downstream service starts a *new* trace. You then have two short traces instead of one
tree, and the causal link that made them useful is gone.

**Why it matters concretely** `[D]`:

- **You cannot compute end-to-end latency.** Each fragment has its own duration and the gap between
  them is invisible — which is precisely the "cannot explain a slow request" signature.
- **You cannot attribute cost.** Tokens recorded at the engine cannot be tied to the tenant at the
  gateway.
- **You cannot reconstruct a trajectory.** For agents, the tool calls and the LLM calls that caused them
  end up in different traces, so trajectory evaluation is impossible (T17-Q6).

**Where it usually breaks** `[D]`: at boundaries that are not HTTP-aware — an async queue, a batched
inference call, a custom protocol between router and engine, or a proxy that reconstructs the request.
The corpus's instrumented path runs through a gateway and a router to the engine, which is two hops that
each have to forward context.

**The fix.** Ensure every hop propagates the W3C trace context headers, and — because the engine may be
reached over a non-HTTP path — explicitly inject the context into whatever the transport is. Then verify
it, because silent propagation failure is the default: assert in a staging test that one request
produces one trace with the expected span count.

The check I would add to the runbook: **a canary request whose trace must contain N spans.** If it
contains fewer, propagation broke in a deploy.

**Signal:** Names context propagation specifically, and describes the failure as *two traces instead of
one tree* rather than "traces are missing."

**Follow-ups:**
- *Where does it most often break?* — non-HTTP boundaries: queues, batched calls, custom transports.
- *How would you detect it in CI?* — assert span count on a canary trace.
- *What does it cost you for agents?* — trajectory reconstruction becomes impossible.

**Red flags:** Says "add more logging," or treats a broken trace as a cosmetic problem.

---

#### T17-Q10 · Design your sampling policy
**Difficulty:** L4 · **Depth expected:** 4 min

**Question:** You cannot afford to trace every request. Design a sampling policy and defend it.

**Model answer:** Tail sampling with a biased policy, and the bias is the design decision.

**Why tail sampling.** Head sampling decides at the start of a request, when you do not yet know whether
it will be interesting. Tail sampling decides at the end, when you know its status and duration — so you
can keep the traces that matter and drop the ones that do not `[D]`. For LLM serving this is decisive,
because the traces worth keeping are exactly the ones that went wrong or went slow.

**The policy I would ship** `[T]`: **100% of errors, 100% of slow requests, and a low percentage
baseline of successes.**

```yaml
tail_sampling:
  policies:
    - errors     # status_code: ERROR      -> keep 100%
    - slow       # latency > 10000 ms      -> keep 100%
    - baseline   # probabilistic ~5%       -> keep for aggregate stats
```

**Why keep any successes at all.** Because if you only keep failures, you have no denominator and no
baseline. You cannot compute a failure *rate*, and you cannot compare a slow success against a normal
one. The success sample is what makes the error sample interpretable.

**Why 100% of errors is non-negotiable.** Errors are rare, they are the expensive ones to debug, and the
cost of keeping them is bounded by their rarity. Sampling errors is false economy.

**The tuning question** `[D]`: the baseline percentage is a budget decision. Set it by working backwards
from a monthly trace-storage budget, not by picking a round number — and lower it as traffic grows,
because the error and slow policies scale with traffic while your storage budget does not. The corpus's
own figures for evaluation sampling are in the 1–5% range for judge-scored traffic `[T]`, which is a
reasonable starting point for the baseline policy too.

**What I would not sample:** the *metrics*. Counters and histograms are cheap; keep them at 100% and
sample only the traces.

**Signal:** Chooses tail over head sampling with a reason, keeps 100% of errors, and justifies the
success baseline as the denominator rather than as an afterthought.

**Follow-ups:**
- *Why not head sampling?* — you cannot know at request start whether it will be interesting.
- *Why keep successes?* — you need a denominator and a baseline.
- *What scales with traffic and what does not?* — error retention scales; the budget does not.

**Red flags:** Proposes uniform sampling, samples errors, or samples metrics as well as traces.

---

#### T17-Q11 · What is the cost of tracing, and where does it bite?
**Difficulty:** L4 · **Depth expected:** 3 min

**Question:** Traces are cheap per request and expensive at scale. Where does the cost actually
accumulate, and what do you do about it?

**Model answer:** Three places, and the biggest one is not storage.

**Volume.** An agent task generates a tree, not a span: a root per task, a turn per iteration, and per
turn an LLM call, several tool calls and retrievals. At N tasks per hour that is a large multiple of the
request count `[D]`. This is the volume that tail sampling (T17-Q10) exists to control.

**Span payload size.** A span with token counts and routing metadata is small. A span with prompt and
completion text is large — and prompt text is often the largest single field, because agent prefixes are
huge. This is the term people underestimate: **you are paying to store the prompt, repeatedly, in every
turn's span.** The mitigation is to store a hash or a pointer rather than the text `[D]`.

**Collector and pipeline cost.** Every span crosses a collector, which has CPU and network cost. At
sustained high volume the collector tier is a real fleet, and it is the first thing to fall over under a
traffic spike — usually at exactly the moment you most need traces.

**What I would do** `[D]`:

1. **Tail-sample**, so volume is bounded by the policy rather than by traffic.
2. **Do not put raw prompt text in spans by default.** Carry token counts, hashes and metadata; sample
   the text deliberately, and only where a redaction policy allows it (T17-Q12).
3. **Tier the retention.** Hot storage for a few days for debugging, cold or aggregate beyond that; the
   eval set is the curated permanent artefact.
4. **Budget from the collector**, not from storage — the pipeline is what breaks first.

The framing to offer: tracing is a **sampled** system by design, and the sampling policy *is* the cost
control. A team that traces everything has not built observability; it has built a bill.

**Signal:** Identifies prompt-text-in-spans as an underestimated cost, and names the collector tier
rather than only storage.

**Follow-ups:**
- *What is the largest per-span cost?* — payload, dominated by prompt text.
- *What breaks first under a spike?* — the collector tier.
- *What is the permanent artefact?* — the curated eval set, not the trace store.

**Red flags:** Discusses only storage cost, or proposes dropping traces rather than sampling them.

---

#### T17-Q12 · Redaction and PII in traces
**Difficulty:** L4 · **Depth expected:** 4 min

**Question:** Your traces contain prompts, which contain customer data. Design the controls.

**Model answer:** Redact at the boundary, before storage — not at query time, and not as a policy
document.

**Why at the boundary.** The corpus's framing is that redaction happens **at the collector, before
storage** `[T]`. The reason is that anything stored may be read by someone who should not see it — an
engineer debugging an unrelated issue, a support tool, a backup, a breach. Query-time redaction protects
none of those, because the data is already there.

**What to redact** `[D]`:

- **Direct identifiers** in prompts and completions — names, emails, account numbers, card fragments.
- **Secrets** that agents handle: API keys, tokens, connection strings. Agents are unusually likely to
  have these in context because they use tools.
- **Retrieved document text**, which may be more sensitive than the prompt — a retrieval step can pull a
  document from a system with different access controls than the caller.

**The design that makes this tractable** `[D]`: **do not store raw text by default.** Carry token counts,
hashes and metadata on every span; store text only when a deliberate, policy-governed sampling decision
says to, and apply the redaction processor in the collector pipeline for those. This inverts the default
from "store everything, redact what we can" to "store nothing identifiable unless we chose to."

**The tension to name honestly.** Redaction degrades debuggability — a redacted prompt is harder to
reason about — and eval quality, since judges need real text. The resolution is tiered access: a
redacted default path for general engineering, and a governed path with access control and audit for
cases that genuinely require raw text.

**Two failure modes I would test for** `[D]`: **redaction that misses a field** (a new attribute added
by a library, carrying something sensitive), and **re-identification through combination** — a redacted
prompt plus a trace ID plus timing may still identify a person. Both argue for the store-nothing default
rather than a blocklist.

**Signal:** Insists on boundary redaction before storage with a reason, and proposes an inversion of the
default (store metadata, not text) rather than a redaction ruleset.

**Follow-ups:**
- *Why not redact at query time?* — the data is already stored; every reader is exposed.
- *What is the debuggability cost?* — real; the resolution is tiered governed access.
- *Which is riskier to store, prompts or retrieved documents?* — retrieved documents; different access
  controls may apply.

**Red flags:** Proposes a regex blocklist as the control, or accepts storing raw prompts "for debugging"
with no governed path.

---

### Metrics, SLOs and alerting

#### T17-Q13 · Why is goodput a better SLO than a latency percentile?
**Difficulty:** L3 · **Depth expected:** 3–4 min

**Question:** Your SLO is "p99 TTFT under 2 seconds." What is wrong with it?

**Model answer:** It is gameable by doing less work, and it measures the wrong thing at the boundary.

**The gaming mechanism.** Latency percentiles are computed over *served* requests. If the system sheds
or rejects load when it is under pressure, the requests that would have been slow never enter the
percentile — they become errors, which are a different metric. So a system can improve its p99 by
refusing to serve the hardest requests, and the SLO will look better while users are worse off `[D]`.

This is not hypothetical for LLM serving, because shedding is a legitimate and recommended technique
(T16-Q18). The moment you adopt admission control, a latency-percentile SLO starts rewarding you for
using it more aggressively.

**The alternative.** Goodput:

```
goodput = |{r : TTFT(r) ≤ T_t AND ITL(r) ≤ T_i}| / |requests|
```

the fraction of *all* requests — including rejected ones, which count as failures — that met both the
time-to-first-token and inter-token-latency targets `[D]`. It cannot be improved by shedding, because a
shed request is a miss. And it captures both phases in one number, which matters because a request that
starts fast and then crawls is not a good request.

**The second problem with p99** is that it reports after the fact. Users have already suffered by the
time the alert fires. So goodput should be paired with leading indicators on the dashboard — **queue
depth**, and the corpus's saturation thresholds of **KV at 80%** and **active requests above 8** `[T]`
— which fire *before* goodput degrades.

**What I would still keep.** Percentiles for diagnosis, not for SLOs. When goodput drops, TTFT and ITL
percentiles tell you which phase and how badly. The SLO is goodput; the percentiles are the
investigation.

**Signal:** Names the shedding-gaming mechanism precisely, and pairs goodput with leading indicators
rather than replacing percentiles entirely.

**Follow-ups:**
- *How exactly does shedding improve p99?* — shed requests leave the denominator.
- *What are the leading indicators?* — queue depth, KV utilisation, active requests.
- *Do you drop percentiles?* — no; they are the diagnostic, not the SLO.

**Red flags:** Accepts the p99 SLO, or proposes goodput without noticing that admission control makes
the difference.

---

#### T17-Q14 · Alert design — what would you page on?
**Difficulty:** L4 · **Depth expected:** 3–4 min

**Question:** You have the dashboard. Which alerts page a human, and which just file a ticket?

**Model answer:** I would page on **user-visible SLO breaches and imminent saturation**, and file
everything else. The test is whether a human action in the next fifteen minutes changes the outcome.

**Page** `[D]`:

- **Goodput below target** — users are already suffering; this is the canonical page.
- **Saturation thresholds crossed** — the corpus's concrete examples are **KV at 80%** and **active
  requests above 8** `[T]`. These page because they are *leading*: capacity added now prevents an
  incident in ten minutes.
- **Error-rate spike** — a step change, not a slow drift.
- **Cost anomaly** — per-tenant or per-task cost spiking. This pages because an unbounded agent loop is
  a self-inflicted denial-of-service, and it compounds while you sleep (T16-Q8).

**Ticket, not page** `[D]`:

- **Slow drift in any quality metric.** Quality moves on the timescale of deployments; it is investigated
  in working hours. Exception: a step change right after a deploy is a page, because it is a rollback
  decision.
- **Cache hit rate below expectation.** Real, but rarely urgent.
- **Individual slow requests.** That is what traces are for.

**The two design rules I would insist on:**

**Alert on the leading indicator, and page on the lagging one.** KV utilisation and queue depth page
*before* goodput drops; goodput pages when it has. Both belong, but a system with only the lagging alert
is always reacting.

**Thresholds must come from measurement, not convention.** The corpus's KV-80% and active-requests-8
figures are measurements from a specific deployment `[T]`; the useful act is to find *your* equivalent
by load-testing to the knee, not to copy the number.

**Signal:** Frames the page/ticket split by *whether human action changes the outcome in 15 minutes*,
and distinguishes leading from lagging alerts.

**Follow-ups:**
- *Why page on cost?* — a runaway agent compounds while you sleep.
- *Where do thresholds come from?* — your own load test to the knee, not a copied number.
- *Which alert fires first?* — saturation, before goodput degrades.

**Red flags:** Pages on CPU, pages on every metric, or has no leading indicators at all.

---

#### T17-Q15 · Detect quality drift in production
**Difficulty:** L4 · **Depth expected:** 4 min

**Question:** Your dashboards are green but a customer says quality dropped last month. How do you detect
that earlier next time?

**Model answer:** You need online evaluation sampled from live traffic, because offline eval sets cannot
see distribution drift.

**Why offline evals miss it.** A golden set tests the cases you thought of, on the distribution you had
when you wrote it. Production drift — new user behaviour, a new document source, a prompt change that
interacts with a model update — is by definition outside it. So the eval suite stays green while quality
falls. The corpus's failure signature is exactly this: **eval passes, users unhappy**, with the cause
being an eval set that is **too easy or off-distribution**, and the fix being to add hard cases and
**real traffic samples** `[T]`.

**The design** `[T]`/`[D]`:

1. **Sample live traffic** at a low rate — the corpus's evaluation sampling sits in the **1–5%** range —
   and score it asynchronously. This is the online eval plane.
2. **Score with a validated judge.** Not a judge you trust; a judge whose agreement with humans you have
   measured (T17-Q20).
3. **Track the score over time** and alert on a drift threshold. The corpus uses a **10% drift
   threshold** as the trigger `[T]`.
4. **Feed failures back into the golden set.** A sampled production failure is the most valuable eval
   case you can have, because it proves the set was missing something.

**The complementary signals** `[D]`: user feedback (thumbs, escalations, regeneration rate) is a cheap
and honest label, and it usually arrives before your sampled judge notices. Track regeneration and
retry rates — a user who immediately re-asks is telling you the answer was bad.

**What I would be careful about:** a drift alert is only as good as the judge, and judges drift too
(T17-Q21). So the judge's own agreement must be re-measured on a schedule, and the gold set kept fresh,
or you have built a smoke detector that has been disconnected.

**Signal:** Prescribes sampled online scoring from live traffic with a drift threshold, and names the
judge-validation dependency rather than treating the judge as ground truth.

**Follow-ups:**
- *Why can offline evals not catch this?* — they test the distribution you already knew.
- *What is the threshold?* — the corpus uses 10% drift `[T]`.
- *What is the cheapest honest signal?* — user feedback and regeneration rate.

**Red flags:** Proposes expanding the offline eval set only, or trusts the judge without measuring its
agreement.

---

#### T17-Q16 · Attribute a latency regression to a component
**Difficulty:** L4 · **Depth expected:** 3–4 min

**Question:** TTFT doubled overnight. No deploy happened. How do you find the cause?

**Model answer:** I would work from the decomposition outward, using the trace structure to bisect rather
than guessing.

**Step 1: is it prefill or queue?** These are the two terms in TTFT and they have different fixes. The
trace decomposition (`queue_wait + prefill`) separates them immediately. If `queue_wait` grew, you have
a capacity or traffic problem; if `prefill` grew, you have a caching or workload problem. Getting this
wrong wastes the whole investigation.

**Step 2: if it is queue, look at the leading indicators.** Queue depth and KV utilisation, and whether
traffic changed. Check whether a new tenant or a new workload shape arrived — an agent workload with
**98% prefill** `[T]` landing on a chat-tuned pool is a classic, and it looks like a regression with no
deploy.

**Step 3: if it is prefill, check the cache first.** Per-pod prefix cache hit rate is the corpus's named
signal for exposing bad routing `[T]`, and a hit-rate drop is the single most common cause of a prefill
regression. Causes, in order: a routing change (was anything rescheduled?), a client change (did the
harness start rewriting its history?), a KV retention change (did sessions start getting evicted during
idle gaps?), or condensation added to the client.

**Step 4: check what changed that was not a deploy.** The list is longer than people expect `[D]`:
traffic mix, a new tenant, a client release, a tool latency change (which lengthens the idle gap and
therefore the eviction rate), a model or quantisation change made by another team, and autoscaling
behaviour (a warm floor that did not hold).

**Step 5: confirm with a counter-metric.** Whatever the hypothesis, verify it against something
independent — if you think cache hit rate fell, check that cache-related token counts and cost moved too.

The general principle: **decompose first, then bisect; never start from the component you suspect.**

**Signal:** Uses the decomposition to separate queue from prefill before hypothesising, and reaches for
cache hit rate as the first prefill suspect.

**Follow-ups:**
- *Why separate queue from prefill first?* — different fixes; conflating them wastes the investigation.
- *What is the top prefill suspect?* — cache hit rate, per pod.
- *What changed without a deploy?* — traffic mix, client behaviour, tool latency, another team's model
  change.

**Red flags:** Starts by scaling GPUs, or has no way to separate queue from prefill.

---

#### T17-Q17 · Eval-gated CI
**Difficulty:** L4 · **Depth expected:** 3–4 min

**Question:** How do you make quality regressions hard to ship?

**Model answer:** By making the eval suite a merge gate, which is the only mechanism that makes quality
checks durable.

**The mechanism** `[D]`: every change that can affect output — prompt, model, quantisation,
configuration, retrieval — runs the eval suite in CI, and the merge is blocked on regression. The corpus
states the underlying rule in its most-repeated form: **"always rerun your evals after quantizing"** `[T]`.
Eval-gated CI is what turns that rule from a good intention into an enforced one.

**Why the gate rather than a dashboard.** Ungated evals decay into unused scripts. Someone runs them
during a quiet week, they drift out of date, a few cases break for unrelated reasons, and eventually
nobody trusts them. A gate has an owner (whoever's merge is blocked) and a forcing function.

**What I would gate on** `[D]`:

- **Absolute thresholds** for core capabilities — the things that must never break.
- **Regression versus the previous version**, not versus an absolute number, for everything else.
- **Both** quality and cost, since a prompt change that improves quality and triples per-task cost is
  also a regression (T16-Q21).

**The practical problems, named honestly** `[D]`:

- **Flakiness**, because outputs are non-deterministic even at temperature 0 (T17-Q5). The fix is
  property and statistical assertions, not exact matching, and repeated trials for the noise-sensitive
  cases.
- **Runtime.** A full suite on every commit is slow; I would split into a fast smoke suite that gates
  every merge and a full suite that gates release.
- **Judge cost.** Scoring on every CI run is expensive, which is why the layered judge architecture
  (T17-Q19) matters — cheap inline judges for CI, frontier judges for the periodic deep run.

**Signal:** Frames the gate as what makes evals durable, and names flakiness and runtime as the real
obstacles with concrete mitigations.

**Follow-ups:**
- *Why a gate rather than a dashboard?* — ungated evals decay; a gate has an owner.
- *How do you handle flakiness?* — property assertions plus repeated trials; see T17-Q5.
- *What should the fast suite contain?* — the core capabilities that must never break.

**Red flags:** Proposes an eval dashboard with no enforcement, or asserts exact output equality in CI.

---

### Evaluation mechanics — judges and gold sets

#### T17-Q18 · What are LLM-as-judge's biases and how do you mitigate them?
**Difficulty:** L4 · **Depth expected:** 4 min

**Question:** You want to use a model to grade outputs. What are the known failure modes?

**Model answer:** Three named biases, plus a validation requirement that matters more than all of them.

**The biases** `[T]`:

**Position bias** — the judge prefers whichever answer appears first. Mitigation: randomise the order of
candidates across trials, so the bias averages out rather than systematically favouring one arm.

**Verbosity bias** — longer answers score higher, regardless of quality. This is the most dangerous one
in practice, because it is also a *cost* failure: a judge with verbosity bias will reward the more
expensive answer. Mitigation: pin length where the task allows, or score length-controlled.

**Self-preference** — a judge favours outputs from its own model family. Mitigation: use a different
family for the judge than for the generator. This also decorrelates failures.

**The validation requirement, which is the whole game** `[D]`:

```
agreement = |judge agrees with human on gold set| / |gold set|
```

**Do not trust a judge you have not measured.** The judge is an instrument, and an uncalibrated
instrument produces confident numbers of unknown meaning. Build a human-labelled gold set, measure
agreement, and — critically — **re-measure when the judge model or judge prompt changes**, because a
judge upgrade is a silent redefinition of your metric.

**The scale reality.** The corpus's 2026 architecture addresses the cost of doing this properly: a
distilled model inline for the bulk of scoring, a frontier model on a **1–5% sample**, and human
labelling for the gold set — reported at roughly **97% lower cost** and ~10× lower P50 than running the
frontier judge on everything, with **88–92% agreement** `[T]`. Treat those as vendor-reported figures
for the specific setup described, not as universal constants.

And the honest limit: a judge measures agreement with a rubric on a distribution. It is a proxy for
quality, not quality itself, and it inherits every blind spot of the human labels it was calibrated
against.

**Signal:** Names all three biases with a specific mitigation each, and treats judge validation as
mandatory rather than optional.

**Follow-ups:**
- *Which bias is also a cost failure?* — verbosity bias; it rewards the more expensive answer.
- *When must you re-measure agreement?* — whenever the judge model or prompt changes.
- *What is the reference architecture?* — distilled inline, frontier on a sample, human gold `[T]`.

**Red flags:** Uses a judge without measuring agreement, or uses the same model family as judge and
generator.

---

#### T17-Q19 · Design the judge architecture for cost and coverage
**Difficulty:** L4 · **Depth expected:** 4 min

**Question:** You need to score a high volume of production traffic and cannot afford a frontier model
on everything. Design it.

**Model answer:** A layered architecture — cheap and universal at the bottom, expensive and rare at the
top, humans at the anchor.

**Layer 1 — inline, distilled judge, run on everything** `[T]`. A small model, or a distilled scorer,
applied to all or most traffic. It is cheap enough to be universal, and its job is to catch the obvious
failures and produce a continuous signal. It does not need to be right about nuance; it needs to be
consistent and fast.

**Layer 2 — frontier judge on a sample, 1–5%** `[T]`. A strong model, with a carefully written rubric,
applied to a random sample plus everything layer 1 flagged. This is where accuracy comes from. 1–5% is
enough to estimate a rate and to audit layer 1.

**Layer 3 — human gold labels, a small curated set** `[T]`. This is the anchor: the only ground truth in
the system. Its job is to calibrate layers 1 and 2 — you measure each judge's agreement against it.

**Why the layering is the right shape rather than a compromise** `[D]`: the three layers do different
jobs. Layer 1 gives *coverage* (every request scored), layer 2 gives *accuracy* (trustworthy scoring on
a sample), layer 3 gives *validity* (a reference to check the other two against). A single-layer system
has to trade one of the three away: use only the frontier model and you cannot afford coverage; use only
the small model and you have accuracy you cannot verify.

**The corpus's reported economics** `[T]`: roughly **97% lower cost** and about **10× lower P50 latency**
than running the frontier judge on everything, at **88–92% agreement** with the human labels. I would
treat those as vendor-reported figures for that configuration and re-measure in my own setting.

**The operational discipline** `[D]`: measure layer 1 against layer 2 periodically (does the cheap judge
still agree with the expensive one?), and both against layer 3. A layer that drifts out of agreement is
a broken instrument, and the drift is otherwise invisible because the scores keep coming.

**Signal:** Assigns a distinct job to each layer — coverage, accuracy, validity — rather than describing
it as a cost-saving compromise, and marks the vendor figures as vendor figures.

**Follow-ups:**
- *What job does each layer do?* — coverage, accuracy, validity respectively.
- *Why keep a human layer at all?* — it is the only ground truth; you cannot validate judges without it.
- *What drifts?* — the cheap judge against the expensive one; measure it on a schedule.

**Red flags:** Proposes a single judge tier, or cites the 97%/88–92% figures as universal constants.

---

#### T17-Q20 · Build a gold set
**Difficulty:** L4 · **Depth expected:** 3–4 min

**Question:** You need a human-labelled evaluation set. How do you build one that stays useful?

**Model answer:** Four rules, and the fourth is what keeps it alive.

**One: draw from real traffic, not from imagination.** The corpus's failure signature is **eval passes,
users unhappy** with the cause being an eval set that is too easy or off-distribution, and the fix being
to add hard cases and **real traffic samples** `[T]`. A set someone wrote by hand in a quiet week tests
the cases they thought of, which are precisely the cases the system already handles.

**Two: bias it toward hard and adversarial cases.** A gold set of easy cases produces a high score that
means nothing. Include the ambiguous, the edge-of-policy, the multilingual, the malformed input, and —
for agents — the trajectories that went wrong (T17-Q6).

**Three: label defensibly.** Every case needs a clear rubric, and ideally two independent labellers with
disagreements resolved explicitly. Disagreements are informative: a case where two experts disagree is a
case where your rubric is underspecified, and fixing the rubric improves the whole system.

**Four: keep it fresh, and keep it separate.** Add cases as failures appear — a production failure pulled
from a trace is the highest-value addition possible, because it proves the set had a gap. And hold a
portion back as a true test set: if you tune against the whole gold set, it stops being an evaluation and
becomes a training target `[D]`.

**The tension to name:** freshness versus comparability. A gold set that changes every week cannot be
used to compare this quarter to last. The resolution is versioning: freeze a set for a period, record
which version a result was measured against, and change it deliberately rather than continuously.

**Signal:** Insists on real traffic and hard cases, and raises the held-back set and versioning — the two
things that keep a gold set honest over time.

**Follow-ups:**
- *Where do the best cases come from?* — production failures pulled from traces.
- *Why hold some back?* — otherwise you tune against your test set.
- *How do you compare across quarters?* — version the set and record it with each result.

**Red flags:** Proposes synthesising cases only, or treats the gold set as a fixed artefact that never
changes.

---

#### T17-Q21 · Detect judge drift
**Difficulty:** L5 · **Depth expected:** 4–5 min

**Question:** Your quality scores have been stable for six months while user complaints have risen.
What could be wrong, and how would you have caught it?

**Model answer:** The likely answer is that the judge has drifted out of agreement with reality, and the
stability of the score is the *symptom*, not the reassurance.

**The three mechanisms** `[D]`:

**Judge model drift.** If the judge is a hosted model, it changes under you — version updates, silent
improvements, sometimes a deprecation. Your metric is now measuring a different instrument against the
same rubric. The corpus's guidance to re-measure agreement **when the judge model or prompt changes**
`[T]` is precisely about this, and the trap is that a hosted model can change without you changing
anything.

**Gold-set staleness.** The judge still agrees with the human labels — but the labels are from a
distribution you no longer serve. Agreement is maintained against a historical reference, so it reads
healthy while the traffic moves.

**Rubric ossification.** The judge scores an *old* definition of good. If the product's notion of a good
answer evolved, the judge is now confidently wrong.

**How I would have caught it** `[T]`/`[D]`:

1. **Re-measure judge agreement on a schedule**, not only when you change something. A periodic human
   spot-check of judge decisions is the only mechanism that detects judge drift.
2. **Track judge agreement as a monitored metric in its own right.** Its stability is a claim that must
   be continuously re-earned, like everything else.
3. **Triangulate with an independent signal.** User feedback, regeneration rate and escalation rate do
   not depend on the judge. If the judge says stable and users say worsening, believe the users and
   investigate the judge. **Cross-signal divergence is the alarm.**
4. **Re-baseline the gold set** on a schedule, drawing from recent traffic (T17-Q20), and version it so
   you can tell whether a score change is a quality change or a set change.

The general principle: **an evaluator is a component of the system and needs its own monitoring.** A
team that monitors the model but not the judge has an unmonitored dependency at the centre of its
quality story.

**Signal:** Names judge drift as the hypothesis rather than quality drift, and proposes cross-signal
divergence as the detection mechanism — not just "monitor more."

**Follow-ups:**
- *What is the alarm?* — the judge's score diverging from an independent signal like user feedback.
- *How often do you re-measure agreement?* — on a schedule, and whenever the judge or prompt changes.
- *What makes a score change interpretable?* — versioning the gold set alongside it.

**Red flags:** Concludes quality must actually be stable, or has no independent signal to compare
against.

---

#### T17-Q22 · What is verifier-based grading, and when can you use it?
**Difficulty:** L4 · **Depth expected:** 4 min

**Question:** Model-as-judge is expensive and biased. What is the alternative, and what does it require?

**Model answer:** Verifier-based grading — check the output against something that can decide correctness
mechanically — and it is strictly better *when it applies*.

**The idea** `[T]`: rather than asking a model whether an answer is good, execute a check. The corpus
covers this territory in the reward-model and best-of-N material, where a verifier scores candidate
outputs, and it is the same mechanism that makes best-of-N work: you can only select the best of N if
you have something that can tell them apart.

**What counts as a verifier** `[D]`:

- **Unit tests** for generated code — the strongest form.
- **A schema or parser** for structured output.
- **A reference answer** for maths, or a proof checker.
- **A tool's own response** — did the API call succeed, did the query return rows.
- **A retrieval check** — is the answer's claim present in the retrieved context (the faithfulness
  question).

**Why it is better when available.** It is deterministic, it is cheap, it does not have position or
verbosity bias, and it produces a signal you can act on — a failing test is a bug report, not a score.
And it scales linearly with compute rather than requiring a frontier model per sample.

**What it requires, and where it fails** `[D]`: it needs a **checkable outcome**. That is why it works
beautifully on code, structured extraction and maths, and not at all on "write a better marketing email."
The corpus's own position on this is the honest one: **when the outcome is verifiable, verify; when it
is not, you need a judge — and then you need to measure the judge** `[T]`.

**The design implication** `[D]`: maximise the verifiable fraction of your workload *by construction*.
If you ask the model for structured output, you can verify the structure. If you ask it to call a tool,
you can verify the call. Each of those removes a case from the judge's remit, which is where the cost and
the bias live.

**Signal:** Distinguishes verifiable from unverifiable outcomes as the deciding criterion, and proposes
restructuring the task to increase the verifiable fraction.

**Follow-ups:**
- *Which workloads verify well?* — code, structured output, maths, tool calls.
- *What do you do when nothing is checkable?* — a judge, with measured agreement (T17-Q18).
- *How do you reduce judge load?* — restructure output to be checkable.

**Red flags:** Treats judges as the only option, or claims verification works for subjective quality.

---

#### T17-Q23 · What is Pass^k and why does it matter more than pass@k?
**Difficulty:** L5 · **Depth expected:** 4–5 min

**Question:** Agent benchmarks report Pass^k. What does it measure, and why is it the more honest number
for production?

**Model answer:** Because reliability across repeated attempts is what production asks for, and Pass^k
measures exactly that.

**The definitions.** `pass@k` asks: did the agent succeed *at least once* in k attempts? At k=1 they
coincide. `Pass^k` asks: did the agent succeed **every time**, across k independent attempts `[T]`. The
corpus's agent-benchmark material uses the Pass^k formulation — tau2-bench is the reference — precisely
because it measures consistency rather than peak capability.

**Why it is the honest number** `[D]`. Production does not get to pick the best of k runs. A user makes
one request and gets one answer. So the relevant question is not "can this agent ever do this task" but
"does this agent do this task, reliably" — and those two numbers diverge sharply as k grows:

```
pass@k  →  rises toward 1 as k grows (best-of-k selection)
Pass^k  →  falls toward 0 as k grows (all k must succeed)
```

An agent with 70% per-attempt success reports a flattering pass@5 and a sobering Pass^5 of about 0.17.
The gap between them *is* the reliability problem, and only one of the two numbers shows it.

**Why this connects to everything else in this topic.** Pass^k requires **repeated independent trials**,
which is only meaningful if you have handled non-determinism correctly (T17-Q5) — you need real
variation between attempts, not the same run replayed. It also needs a **verifier or a judge** to score
each attempt (T17-Q22). And it reframes an agent's quality as a distribution rather than a point, which
is what makes regression detection possible at all (T17-Q17).

**The practical rule I would apply** `[D]`: report both, and make decisions on Pass^k. A pass@k number
in a vendor deck is telling you about capability; a Pass^k number is telling you about whether you can
put it in front of a customer.

**Signal:** States the divergence direction for both metrics, and connects Pass^k to the reliability
question production actually asks.

**Follow-ups:**
- *How do the two move as k grows?* — pass@k rises, Pass^k falls.
- *What does Pass^k require methodologically?* — real independent trials, plus a scorer per attempt.
- *Which would you put in a customer-facing claim?* — Pass^k; pass@k overstates reliability.

**Red flags:** Treats pass@k as the standard reliability metric, or reports a single run as a rate.

---

### Agent-specific problems and open questions

#### T17-Q24 · Instrument a multi-agent system
**Difficulty:** L5 · **Depth expected:** 5 min

**Question:** Your agent delegates to sub-agents. How does that change your observability design?

**Model answer:** The trace becomes a graph rather than a tree, and the accounting has to follow it.

**What changes structurally** `[D]`:

**Span relationships stop being purely hierarchical.** A supervisor delegating to a specialist that
itself calls tools produces a deep tree, which is fine. But when agents communicate peer-to-peer — the
corpus's A2A framing, where agent B is a *tool* of agent A `[T]` — you can get a span with multiple
logical parents, or a cycle if agents can call back. The trace model needs an explicit decision: model
the peer call as a tool call from A to B (simple, loses B's internal structure at A's level) or as a
linked trace with its own root (complete, but you must join them).

**Attribution becomes the hard part.** Cost and latency must roll up per *task*, across all agents, not
per agent. A supervisor whose own spans look cheap can be driving four expensive specialists. The
per-task cost discipline from T16-Q21 is what makes multi-agent cost legible at all.

**Budgets need to be hierarchical.** Each agent needs its own budget *and* a share of the parent's,
because a runaway sub-agent is invisible if only the supervisor is bounded.

**What to carry on the span** `[D]`: in addition to the T17-Q3 attributes, a **task ID** that is stable
across all agents, an **agent identity** (which is separate from the user identity `[T]` — a delegated
action should be attributable to the agent that took it), and the **delegation edge** (who asked whom).

**What I would refuse to give up** `[D]`: the ability to reconstruct one task's full trajectory from
one query. If diagnosing a failure requires joining traces by timestamp across three systems, you have
built a system you cannot debug — and for multi-agent, trajectory reconstruction is the *only* way to
see a reasoning-action mismatch or a delegation loop (T17-Q6).

**Signal:** Identifies the tree-becomes-graph problem and the hierarchical budget requirement, and names
agent identity as a span attribute.

**Follow-ups:**
- *What is the hardest part?* — attributing cost and latency across agents to one task.
- *Why hierarchical budgets?* — a runaway sub-agent is invisible if only the parent is bounded.
- *Why agent identity?* — delegated actions are the agent's, not the user's `[T]`.

**Red flags:** Assumes the trace tree scales unchanged, or attributes cost per agent without rollup.

---

#### T17-Q25 · Evaluate memory systems
**Difficulty:** L5 · **Depth expected:** 4–5 min

**Question:** An agent has a long-term memory store. How do you evaluate whether the memory is working?

**Model answer:** Memory has three distinct jobs, and evaluating it as one thing is why memory systems
look fine and behave badly.

**The three operations** `[T]`: the corpus's memory-evaluation framing separates **extraction** (what
gets written to memory), **update** (how existing memories change as new information arrives), and
**QA** (whether reading memory improves answers). HaluMem is the referenced evaluation approach `[T]`.

**Why the separation matters** `[D]`. Each operation fails differently and needs a different test:

- **Extraction failure**: the memory store is empty of the thing that mattered, or full of trivia. Test:
  given a conversation, is the fact that should have been stored actually present?
- **Update failure**: a stored fact is contradicted later and both versions persist, so the agent
  retrieves a stale truth confidently. Test: present a contradiction and check the store's state
  afterwards. This is the failure that produces the most damaging behaviour — confidently wrong, with a
  citation.
- **QA failure**: retrieval does not surface the right memory, or surfaces it in a form the model cannot
  use. Test: end-to-end task accuracy with memory enabled versus disabled.

**The measurement I would insist on** `[D]`: **the ablation**. Run the same task set with memory enabled
and disabled. If accuracy does not improve, the memory system is a cost centre with a retrieval bill —
and the corpus's framing of memory as trading **inference cost for retrieval cost** `[T]` makes the
comparison the whole point. A memory system that does not beat no-memory on your tasks is not earning
its complexity.

**The observability hook**: memory needs its own spans — a span per write, per update, per retrieval —
carrying what was retrieved and what it cost. Without them, a memory-related failure is indistinguishable
from a reasoning failure.

**Signal:** Separates extraction, update and QA with a distinct test for each, and proposes the
enabled-versus-disabled ablation.

**Follow-ups:**
- *Which failure is most damaging?* — stale memories after a failed update; confidently wrong.
- *What is the core comparison?* — memory on versus off, same tasks.
- *What must be instrumented?* — write, update and retrieval as distinct spans.

**Red flags:** Evaluates memory as a single accuracy number, or never runs the ablation.

---

#### T17-Q26 · Alerting on an unsupervised system
**Difficulty:** L5 · **Depth expected:** 5 min

**Question:** Your agent runs autonomously overnight with no human in the loop. How does that change your
observability requirements?

**Model answer:** It raises the cost of every gap, because there is no human to notice the thing you
failed to measure — and it changes what an alert is *for*.

**What intensifies** `[D]`:

**Blast radius before detection.** During the day, a human notices a weird output within minutes. Overnight
the agent has hours. So the detection budget shrinks from hours to seconds, which means you cannot rely
on sampling for safety-relevant signals — a 5% trace sample may contain no instance of the failure at all.
Safety-critical checks must run inline at 100% (T18), even though evaluation sampling stays sampled.

**The need for automatic circuit-breaking.** Because no one is there to stop it, the system needs a
mechanism to stop itself: hard budgets on tokens, steps and wall-clock, enforced structurally rather than
by inspecting content (T16-Q8); and automatic halting on a safety-rail trip or an anomaly detector.

**The need for a clean audit trail.** If something went wrong at 3am, the only way to reconstruct it is
the trace + the durable session log. That makes retention of the *right* things a requirement rather than
a cost decision: the session log is durable by design (T16-Q27), and the traces for autonomous runs may
need longer retention than interactive ones `[D]`.

**What I would not change** `[D]`: the same three planes, the same goodput SLO, the same sampled
evaluation. Autonomy changes the *thresholds and the enforcement*, not the architecture.

**The reframing worth stating.** In an interactive system, observability supports humans making decisions.
In an autonomous system, observability *is* the control loop — the alert is what a human would have been.
So the alert design principle from T17-Q14 (does a human action in 15 minutes change the outcome?) has to
be relaxed: for autonomous systems, the correct answer is often that the *system* must act on the signal,
and the human is informed rather than paged.

**Signal:** Names circuit-breaking and 100% inline safety checks as the additions, and reframes
observability as the control loop rather than a decision aid.

**Follow-ups:**
- *Why can you not sample safety checks?* — a 5% sample may contain no instance of the failure.
- *What stops the agent?* — structural budgets and automatic halting, not content inspection.
- *What changes about retention?* — autonomous runs may need longer trace retention for audit.

**Red flags:** Applies interactive assumptions unchanged, or relies on a human to notice the failure.

---

#### T17-Q27 · What would you instrument first, given one week?
**Difficulty:** L5 · **Depth expected:** 4–5 min

**Question:** You inherit a deployed LLM system with no observability. You have one engineer-week. What
do you build?

**Model answer:** I would build the smallest thing that answers "what is broken and where," and I would
deliberately defer the quality plane.

**Day 1–2: the four metrics that bound everything.** TTFT, ITL, queue depth, and KV-cache utilisation —
plus per-pod prefix cache hit rate and error rate `[T]`. These are cheap, they come from the engine's
existing metrics endpoint, and they immediately make the system *legible*: you can tell the difference
between a capacity problem, a caching problem and a model problem. I would also set the saturation
alerts — the corpus's **KV 80%** and **active requests > 8** `[T]` — because they are leading indicators
that fire before users complain.

**Day 3–4: traces at the two boundaries that break.** A span at the gateway and a span at the engine,
with trace context propagated between them, carrying model, token counts, cost and the route replica
(T17-Q3). This is the minimum that lets you decompose a slow request (T17-Q8), and the propagation fix
is the highest-value hour in the whole week because it is the thing most often broken.

**Day 5: tail sampling and a redaction decision.** Sample before volume makes it impossible — 100% of
errors and slow requests, a low baseline of successes (T17-Q10) — and decide what text may be stored,
defaulting to not storing it (T17-Q12). Retrofitting both of these later is painful.

**What I would deliberately defer:** the eval plane. It is the most valuable plane long-term and the
least useful in week one, because you have no failure cases to curate it from yet. Instead I would start
the *collection*: begin capturing traces that will later become the gold set (T17-Q20), which costs
almost nothing now and is the input to the eval work later.

**What I would refuse to defer:** the saturation alerts and the trace-boundary fix. Both are
high-value-per-hour, and both are much harder once traffic grows.

**Signal:** Sequences by *what makes the system legible fastest* rather than by completeness, and
justifies deferring the quality plane while still starting its data collection.

**Follow-ups:**
- *Why defer evals?* — no failure cases yet; start collecting them instead.
- *What is the highest-value single hour?* — trace context propagation across the gateway/engine boundary.
- *Why sample on day 5?* — retrofitting sampling after volume grows is painful.

**Red flags:** Proposes building the full three-plane system, or starts with the eval suite.

---

#### T17-Q28 · The hardest problem in LLM observability
**Difficulty:** L5 · **Depth expected:** 5 min

**Question:** Give me your view: what is genuinely unsolved here?

**Model answer:** I would argue it is **evaluating whether a trajectory was acceptable, at a cost that
allows you to do it on most traffic**, and I would defend that over the more obvious candidates.

**Why not the obvious ones.** Tracing is a solved engineering problem — OpenTelemetry, spans, sampling,
collectors. Metrics are solved. Judge bias has known mitigations (T17-Q18). Judge cost has a working
architecture in the layered design (T17-Q19). These are all *implementation* problems with known shapes.

**The actual gap.** Consider what you would need to be confident an agent was well-behaved in production.
You would need to judge the *path*, not the outcome (T17-Q6), because outcome metrics miss the failures
that matter. But a path judgement is:

- **Expensive** — a rich judgement per task, and tasks are numerous.
- **Subjective** — "was this a reasonable plan" has no mechanical check for open-ended tasks (T17-Q22).
- **Distribution-shifting** — the notion of an acceptable path evolves with the product.
- **Adversarially fragile** — a judge that scores trajectories can be optimised against, and the corpus's
  own vocabulary includes **self-jailbreaking** `[T]`, where an agent evades the constraint rather than
  satisfying it.

**What would count as progress** `[D]`:

- **Structured agent self-reporting** — agents emitting typed claims about what they did and why, which
  are cheap to check against the trace. Partly verifiable, so partly mechanisable.
- **Cheap trajectory scorers** — the same layered pattern as T17-Q19, but for paths rather than outputs.
  This is an open engineering problem, not an open research one.
- **Assertion libraries for common trajectory invariants** — the field currently rewrites the same checks
  in every codebase.

**Why it outranks the rest.** The other problems in this topic limit how well you can *optimise*. This
one limits whether you can *trust* — and an autonomous system you cannot trust cannot be deployed, which
makes it the binding constraint on everything else in this knowledge base.

**Signal:** Picks a specific gap rather than "quality is hard," justifies it against the stronger-sounding
alternatives, and proposes what progress would look like.

**Follow-ups:**
- *Why not judge cost?* — there is a working architecture for it (T17-Q19).
- *Why is adversarial fragility specific to agents?* — a scoreable trajectory can be optimised against.
- *What is the nearest-term improvement?* — cheap trajectory scorers using the layered pattern.

**Red flags:** Names a solved problem, or gives an answer with no argument for why it outranks the others.

---

## Whiteboard exercises

### Exercise 1 — Decompose a 4-second request
**Prompt.** "A user reports a request taking 4 seconds. Your trace shows: queue wait 40 ms, prefill
1.2 s, decode 90 ms across 45 output tokens, three tool calls totalling 1.1 s, and one retry. The
remaining time has no span. Explain what you know, what you do not, and what you would do about each
part."

**What the candidate must produce:** the terms that reconcile, the size of the unexplained gap, a ranked
list of what could occupy it, and the instrumentation fix.

**Expected answer sketch:**

```
Reconciliation                     [D]
  queue_wait       0.04 s
  prefill (TTFT)   1.20 s
  decode  45 tok   ~0.09 s   (45 x ~2 ms ITL)
  tool_time        1.10 s
  retry            ~?        (the retry's own cost is IN the terms above,
                              unless it is separate -- ASK)
  ------------------------------------
  accounted        ~2.43 s
  reported          4.00 s
  UNEXPLAINED      ~1.57 s   <-- 39% of the request

What the gap could be (ranked)                   [D]
  1. Client-side time NOT instrumented -- agent framework
     between turns, prompt assembly, serialisation
  2. A hop whose context propagation is broken
     -> its span started a NEW trace (T17-Q9)
  3. Tool time not fully captured -- a tool that
     internally makes several calls, only outer timed
  4. Retry backoff sleeping (no span for the sleep)
  5. Gateway -> engine gap (the classic)

What I know for certain
  the GPU is not the problem: 1.29 s of 4 s is inference
  scaling the GPU tier would change almost nothing

Fix
  - canary trace assertion: request must produce N spans
  - instrument client-side turn overhead explicitly
  - time retry backoff as its own span
  - propagate context across every non-HTTP boundary
```

**Grading rubric (full marks requires all four):**
- Actually reconciles the numbers and quantifies the unexplained gap (~1.5–1.6 s) instead of hand-waving.
- Concludes explicitly that GPU scaling will not help — the inference terms are a minority of the request.
- Identifies broken context propagation as a candidate cause of the gap, not just "missing logging."
- Proposes a **canary span-count assertion** as the durable detection mechanism.

---

### Exercise 2 — Design the eval pipeline
**Prompt.** "Design an evaluation pipeline for an agent product: ~50,000 tasks/day, a mix of verifiable
(tool calls, structured output) and unverifiable (written summaries) work. You have a budget for judge
calls, not a blank cheque. State the architecture, the sampling, and how you know the pipeline is
working."

**What the candidate must produce:** a layered architecture, a sampling policy with a rationale, the
verifiable/unverifiable split, and a validation mechanism for each layer.

**Expected answer sketch:**

```
LAYER 0 -- VERIFIABLE FIRST (cheapest, best)               [T]
  tool-call validity, schema/JSON parse, unit tests on
  generated code, retrieval faithfulness check
  -> runs on 100% of what is verifiable
  -> deterministic, no bias, no judge cost
  -> GOAL: maximise this fraction by construction (T17-Q22)

LAYER 1 -- INLINE DISTILLED JUDGE
  everything not verifiable, plus everything layer 0 failed
  cheap, fast, consistent-not-perfect                  [T]

LAYER 2 -- FRONTIER JUDGE on 1-5% sample + all layer-1 flags
  this is where accuracy comes from                    [T]

LAYER 3 -- HUMAN GOLD SET (small, curated, versioned)
  the only ground truth; calibrates layers 1 and 2     [T]

REPORTED ECONOMICS (vendor figures, re-measure locally)  [T]
  ~97% lower cost, ~10x lower P50, 88-92% agreement

SAMPLING                                  [T]/[D]
  traces: 100% errors + 100% slow + ~5% baseline
  evals : 1-5% of live traffic, plus 100% of flagged

HOW I KNOW IT WORKS
  - agreement(layer1 vs layer2) measured on a schedule
  - agreement(layer2 vs human) measured on a schedule
  - cross-signal divergence: if judge says stable and
    regeneration/escalation rates rise -> suspect the JUDGE
    (T17-Q21)
  - Pass^k on the agent tasks, not pass@k             [T]
  - gold set versioned with every reported number

COST CONTROL
  verifiable fraction first; sample the rest; cache judge
  results for identical (task, output) pairs
```

**Grading rubric:**
- Puts verification ahead of judging and aims to *increase* the verifiable fraction — the highest-value
  structural move.
- Assigns each layer a distinct job (coverage / accuracy / validity) rather than a cost tier.
- Names a concrete validation mechanism for the judge layers, and includes cross-signal divergence as the
  drift detector.
- Uses Pass^k rather than pass@k for the agent tasks, and versions the gold set alongside any reported
  number.

---

### Exercise 3 — The green dashboards problem
**Prompt.** "Every dashboard is green. Goodput is 99.4%, TTFT p95 is 800 ms, error rate is 0.2%. But three
enterprise customers have escalated in a month, all saying the answers got worse. You have two weeks.
What do you do?"

**What the candidate must produce:** a hypothesis space, the fastest discriminating measurement for each,
and a concrete two-week plan.

**Expected answer sketch:**

```
Hypothesis space                       Fastest discriminator
1. No quality plane at all        ->  can you score a sample of live
   (metrics only = the corpus's        traffic AT ALL? if not, that is
   named failure mode)                 the finding.             [D]
2. Judge exists but has drifted   ->  compare judge score vs an
                                       independent signal: regeneration
                                       rate, escalation rate, thumbs [T]
3. Eval set off-distribution      ->  sample the escalations' actual
   ("eval passes, users unhappy")      tasks; run them through the
                                       current suite. Do they pass? [T]
4. Real regression, invisible     ->  slice metrics by tenant/model/route
   in aggregates                       -- all three customers escalating
                                       points at a shared path      [D]

The tell: three escalations, one month, green dashboards
  -> almost certainly (1) or (2). You cannot see quality.

TWO-WEEK PLAN
Week 1
  - stand up sampled online scoring on live traffic (1-5%)   [T]
  - pull the escalating customers' actual failing tasks from
    traces and label them -> instant gold set                [T]
  - find the independent signal (regeneration/escalation) and
    graph it against the judge score
Week 2
  - measure judge agreement on the new gold set
  - if agreement is poor -> fix the judge before trusting any
    score it produces
  - add the failures to a versioned gold set and GATE the
    next deploy on it                                       [D]

What I would NOT do
  - not tune latency; it is already fine and unrelated
  - not trust any judge number before measuring its agreement
  - not accept "the model got worse" without a distribution
```

**Grading rubric:**
- Recognises immediately that the system has **no quality plane**, and says so rather than starting with
  latency tuning.
- Proposes the fastest path to real labelled data — pulling the escalating customers' own failing traces —
  rather than building an eval set from scratch.
- Includes a step that **validates the judge before trusting it**, rather than reporting its scores.
- Ends with an enforced gate (eval-gated CI), so the same class of regression cannot ship again, and
  names at least one thing they would explicitly *not* do.

---

## Sources

- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/LLM_Observability_Traces_Spans_OpenTelemetry_for_AI_Apps.txt`
  — spans and traces for LLM applications, the OpenTelemetry GenAI semantic attributes, span nesting and
  the instrumentation shape used in T17-Q3 and the configuration sketch.
- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/How_to_Evaluate_LLM_Apps_LLM-as-a-Judge_RAGAS_Without_the_Bias.txt`
  — judge biases (position, verbosity, self-preference) and their mitigations, RAGAS, and the requirement
  to measure judge agreement.
- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Cut_LLM_Cost_Latency_KV_Cache_Batching_Quantization_vLLM.txt`
  — the cost model with cached tokens at roughly 10× cheaper, token accounting per request and per tenant,
  and the mandatory rule to rerun evals after quantising.
- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Prompt_Management_as_Code_Versioning_Injection_DSPy.txt`
  — prompt versioning as code, and gating prompt changes the way code changes are gated.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/14-evaluation-and-observability/02-observability.md`
  — the three pillars, the metric and alert tables, the 1–5% evaluation sampling figure, the 10% drift
  threshold, cost attribution, and the severity tiers.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/14-evaluation-and-observability/01-llm-evaluation.md`
  — the layered 2026 judge architecture (distilled inline plus frontier on a 1–5% sample plus human gold,
  reported at roughly 97% lower cost, ~10× lower P50 and 88–92% agreement), tau2-bench and Pass^k,
  agent-as-judge, and HaluMem's extraction/update/QA separation.
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt`
  — per-pod prefix cache hit rate as the signal that exposes bad routing, and the saturation thresholds of
  KV at 80% and active requests above 8.
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts_2/CMU_LLM_Inference_2_Probability_Review_and_Code_Examples.txt`
  — non-determinism at temperature 0, its worsening under quantisation, and model fingerprinting.
- `refs/Agentic_AI_Infra_transcripts_2/Weizhu_Chen_-_Continuous_Model_Improvement.txt` and
  `refs/Agentic_AI_Infra_transcripts_2/Ankit_Sobti_-_From_Agent_Demos_to_Production_How_Postman_Is_Building_Reliable_AI.txt`
  — continuous evaluation of trajectories, production reliability practice, and the trajectory failure
  vocabulary.
- `refs/Agentic_AI_Infra_transcripts_3/Gosia_Steinder_-_Beyond_Harnesses_Platform_Solutions_for_Agent_Reliability_Secur.txt`
  — agent identity as distinct from user identity, and the durable session log as the reconstruction
  substrate for autonomous runs.

**Derived content in this bank (`[D]`):** the three-plane framing and the "metrics say something is
wrong, traces say what, evals say whether it was good" formulation; the latency decomposition and the
reconciliation discipline in T17-Q8 and Exercise 1; the tail-sampling policy and its rationale; the
boundary-redaction and store-metadata-not-text defaults; the goodput-versus-percentile gaming analysis;
the alert page/ticket split; the five-step latency-regression bisection; the layer/job mapping in the
judge architecture; the drift-detection triad in T17-Q21; the one-week sequencing and its deferral
rationale; the multi-agent graph-versus-tree analysis; and the risk ranking in Exercise 3. Vendor and
speaker figures are attributed inline; every derived claim is labelled where it appears.
