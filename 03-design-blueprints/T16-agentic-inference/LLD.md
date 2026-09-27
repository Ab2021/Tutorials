# T16 — Agentic inference: low-level design

> `T16` · **Transcript coverage:** primary · [HLD](HLD.md) · [Sequences](docs/SEQUENCES.md) · [Runnable core](run.py)

Every interface below is given as a **data structure with named fields**, because the
serving decisions in this blueprint are all decided by what is in the structure and what
is not. The runnable core (`sim/agent_cost.py`) implements the arithmetic of §3, §4, §6
and §7 in about two hundred lines of stdlib Python; it is a mechanism model, not a
simulator of any real deployment.

---

## 1. Component inventory

| # | Component | Lives where | Owned by | Section |
|---|---|---|---|---|
| 1 | Task/session state | control plane | this blueprint | §2 |
| 2 | Context assembler | harness, client side | harness | §3 |
| 3 | Tool executor + sandbox | data plane | this blueprint | §4 |
| 4 | Action parser | harness | harness | §5 |
| 5 | Error recovery policy | harness | harness | §5 |
| 6 | Memory store + scope API | control plane | this blueprint | §6 |
| 7 | Sub-agent scheduler | control plane | this blueprint | §7 |
| 8 | Trajectory stream | data plane | this blueprint | §8 |
| 9 | Budget controller | control plane | this blueprint | §9 |
| 10 | Prefix-affinity router | shared with T14 | T14 | §10 |

The two components not owned here (#2, #4, #5) are listed anyway because the platform's
performance is determined by their output. The platform cannot fix a prefix-unstable
context assembler by adding cache — it can only detect it (§10) and report it.

---

## 2. The task and session model

```python
@dataclass(frozen=True)
class Turn:
    index: int
    prompt_tokens_est: int      # what we THINK the prompt will be, before building it
    prompt_tokens_actual: int   # what was actually sent -- the accuracy signal
    output_tokens: int
    tool_calls: list[str]       # tool ids invoked this turn
    tool_wall_s: float
    model_wall_s: float
    ttft_ms: int
    cache_hit_prefix_tokens: int  # how much of the prompt the replica found resident
    stop_reason: str            # "tool" | "final" | "budget" | "error" | "compacted"

@dataclass
class TaskState:
    task_id: str
    tenant: str
    parent_task_id: str | None       # set for sub-agents; the fan-out tree edge
    scope_id: str                    # which memory scope this task reads/writes (§6)
    turns: list[Turn]
    budget: Budget
    started_at: float
    status: str                      # "running" | "done" | "failed" | "stopped"
```

Three fields do the heavy lifting.

**`prompt_tokens_est` alongside `prompt_tokens_actual`.** The gap between them is the only
honest measure of whether the context assembler is predictable. A task whose estimate is
consistently 40% under the actual is a task whose admission decision was made on the wrong
number — and admission decisions are what flow control (T14 §5) and autoscaling (T15 §3)
are driven by. The estimate is produced by the harness, not by the platform; the platform's
job is to *record the error and expose it* rather than to fix it.

**`parent_task_id` and `scope_id` as separate fields.** They answer different questions and
conflating them is the mistake that makes fan-out expensive. `parent_task_id` is lineage —
who spawned whom, used for budget propagation and for the tree-shaped diagnostics in §7.
`scope_id` is **cache identity** — which prefix is shared, and therefore which tasks can be
placed on the same replica without re-prefilling. A sub-agent can be a child of a parent
and be in a *different* scope (deliberately: it was given a fresh prompt), and that is
exactly the case that `run.py` §5 shows costing 2.9 GPU-s instead of 7.8 at two turns.

**`stop_reason` includes `"budget"`.** A task that stopped because it ran out of budget is
not a failure and must not be reported as one — but it must be *distinguishable* from a
task that finished, because the two have completely different operational meanings. This is
the single field that makes the difference between an agent platform and an agent
demo.

### 2.1 Context growth, as the platform sees it

`sim/agent_cost.py:new_tokens_at_turn` is the reference implementation of the per-turn
prompt cost, and it is worth restating because the two terms behave differently:

```
new_tokens(i) = (1 - reuse) * context(i-1)     # cacheable: the prefix that existed
              + (context(i) - context(i-1))    # NOT cacheable: this turn's delta
```

The first term is what the cache stack (T12, T13) and the router (T14) attack. The second
term is unreachable by any cache by construction — the tokens did not exist when the cache
was written. In the 40-turn reference task the second term alone is 54,800 tokens, so even
a perfect cache leaves a floor. Sizing a fleet on the assumption that caching makes agents
cheap is sizing on half the arithmetic.

---

## 3. The context assembler and prefix stability

The assembler is harness-side, but it determines whether the cache works, so the platform
publishes a **contract** it must satisfy. This is the highest-leverage interface in the
blueprint.

**Contract: the prompt prefix must be byte-stable across turns.** Formally, for turns
`i < j`, `prompt(i)` must be a prefix of `prompt(j)` except for the appended history.

Violations seen in practice, all of which silently destroy the hit rate:

| Violation | Effect |
|---|---|
| a timestamp or request id in the system prompt | the first block differs every turn — **zero** reuse |
| tool schemas serialised from a `dict` | key order varies with insertion order; the block is not stable |
| "current time: ..." injected at the top | same as above |
| retrieved documents placed **before** the system prompt | every retrieval invalidates the system prompt too |
| tool results truncated with a different policy per turn | the prefix diverges retroactively |

The last one is the subtle one: truncating a *later* part of the history changes bytes the
cache already holds, so the cache is invalidated from the truncation point onward — not
from the point of truncation forward. A compaction policy (§5) is therefore a
cache-destroying event and must be treated as such.

**Detection.** `Turn.cache_hit_prefix_tokens` is compared against the expected reuse:

```
prefix_stability_ratio = cache_hit_prefix_tokens / (prompt_tokens_actual - delta_tokens)
```

A ratio that sits at 0 while the assembler believes it is emitting a stable prefix is a
*harness bug* being observed from the serving side, and it is worth an alert on its own
(§10) because it is invisible from the harness's own logs — the harness sees a correct
prompt; only the replica knows the cache did not match.

**Tool-schema handling.** Per HLD §4.1 the schema block is 19.1% of prompt tokens across a
40-turn task. It must be (a) placed in the stable prefix, (b) serialised deterministically,
and (c) **retrieved per-task rather than shipped whole** where the catalogue is large.
Note (c) changes the prefix between tasks with different tool needs, which *reduces*
cross-task cache sharing — so tool selection is a trade of per-task tokens against
cross-task hit rate, and the correct side of that trade depends on the workload mix. There
is no universal answer and this blueprint does not assert one.

---

## 4. The tool execution layer

Tools are where the wall clock goes (73% of the user's wait at warm cache, `run.py` §1) and
where the fleet's cost does *not*. The design goal is therefore not throughput — it is
**not leaking resources** and **not lying about time**.

```python
@dataclass
class ToolLease:
    lease_id: str
    task_id: str               # the task that owns this execution
    tool_id: str
    scope_id: str              # sandbox identity, shares §6's scoping
    started_at: float
    deadline_s: float          # hard wall-clock limit
    sandbox_handle: str
    state: str                 # "pending" | "running" | "done" | "timeout" | "reaped"
```

Three rules, each with a specific failure it prevents.

**A lease outlives the turn, not the task.** A tool call that takes 40 minutes is legal; a
tool call whose *task* died must be reaped. The reaper is keyed on `task_id`, not on
`lease_id`, because the failure mode is an agent that crashes while its subprocess
continues to hold a GPU, a database connection, or worse. Without this, a fleet accumulates
orphans at the rate of its crash rate, and the orphans are invisible — they are not in any
task's trace.

**Deadlines are hard and are not the model's decision.** `deadline_s` is set by the tool
policy, not chosen by the agent. An agent that can extend its own tool deadline is an agent
that can hold a sandbox open indefinitely, and the panel's overnight anecdote `[T]` is
precisely the case where no human is watching to intervene.

**Tool latency is measured and re-exported per-tool.** `Turn.tool_wall_s` aggregated over
`tool_id` produces the distribution that says where the 100 seconds actually go. This is
the one metric in this blueprint that directly contradicts a common assumption: teams
routinely believe the model is the slow part of an agent, and at warm cache it is 16% of
the wait (`run.py` §1).

**Concurrency.** Tools are I/O-bound and the fan-out pattern means many tools run at once
per task. The executor's concurrency limit is therefore a *per-task* limit, not a global
one, or a single fan-out task starves every other task on the node. The value is
workload-dependent; what is not workload-dependent is that a global semaphore is wrong.

---

## 5. The harness execution layer, and where the platform's boundary is

Gao's execution layer is "a prompt builder, action parser, task decomposition modules, and
some error recovering mechanisms" `[T]`. Three of the four matter to the platform because
they change the *shape* of the request stream:

**Task decomposition** decides `turns` before the platform sees the task. Since cost is
superlinear in turns (`run.py` §2), this is the highest-leverage number in the system and
it is chosen by client-side code. The platform's lever is not to override it but to
*price* it (§9) — a task budget is the mechanism by which the decomposition decision is
made accountable.

**Error recovery** is where the turn count explodes. A retry loop that re-runs a turn
costs a full prompt prefill each time, and the retried turn's prompt is *larger* than the
original because it now contains the failure. A retry storm is therefore
superlinear × retries, and the platform sees only a task that is taking a long time.
Counter it with a per-task retry budget and a `retry_count` on the turn, surfaced in §9's
metrics.

**Compaction** is the one recovery mechanism with a *serving* contract: when the context
approaches the window, the harness must summarise or truncate history. Per §3 this
invalidates the cache from the compaction point onward, so the platform should be *told* —
`stop_reason: "compacted"` — so that the resulting cache-miss spike is attributable to a
design decision rather than to a routing or cache bug. A compaction that is not announced
looks exactly like a prefix-instability bug in the metrics, and diagnosing the wrong one
costs days.

---

## 6. Memory and the scope API

The corpus's framing: agents need "some kind of shared memory", and "there are new
standards to be developed to describe these kinds of scoped shared storage systems and
access and message passing between agents", because prior systems "were either services or
kind of human scale" `[T]`.

This blueprint's contribution is to point out that **scope is a cache key**. The interface
is deliberately small:

```python
class MemoryScope:
    def open(self, scope_id: str, *, read: bool, write: bool) -> ScopeHandle
    def read(self, h: ScopeHandle, key: str) -> bytes
    def write(self, h: ScopeHandle, key: str, value: bytes) -> None
    def children(self, scope_id: str) -> list[str]   # for lineage diagnostics
```

The three design rules:

**Scope is declared, not inferred.** A sub-agent's `scope_id` is set by the spawner, and it
determines whether the sub-agent's prompt inherits the parent's prefix. Because of that,
the choice has a *service cost* — the crossover at ~5.6 sub-agent turns derived in `run.py`
§5 — and the API should therefore make it explicit and loggable rather than letting a
framework default decide it silently.

**Scopes are not a tree, and modelling them as one is the common error.** `children()` is
for lineage diagnostics only. Two sibling sub-agents may share a scope the parent is not in
(the panel's "team that works together" `[T]`), and a sub-agent may deliberately be given a
fresh scope to keep its prompt small. A tree-shaped memory model cannot express either, and
will force the expensive choice.

**Reads and writes are separately gated.** The scope that a task *reads* determines its
cache identity; the scope it *writes* determines who is invalidated later. Conflating them
means a sub-agent that only reads the parent's workspace cannot be placed on a replica that
holds it, because the system cannot tell reading from writing.

---

## 7. Sub-agent scheduling and the fan-out budget

Fan-out is a scheduling problem before it is a billing problem, because the decision to
spawn N is made *before* any cost is incurred.

```python
@dataclass
class FanoutRequest:
    parent_task_id: str
    n_requested: int
    scope_plan: list[str]        # one entry per sub-agent: shared or fresh (§6)
    expected_turns: int          # the number that decides shared-vs-fresh (§5.1)
    per_subagent_budget: Budget
```

`expected_turns` is the field that changes the answer. Per `run.py` §5, a two-turn
sub-agent should get a fresh scope and a twenty-turn one should inherit; a framework that
applies one policy to both is wrong on one of them, and being wrong on the short one is the
expensive direction (7.8 vs 2.9 GPU-s per agent at N=100,000 is the difference between
~780k and ~290k GPU-seconds).

**Budget as a scheduling input, not an alert.** The `Budget` is checked at three points and
each one has different semantics:

| Point | Action on exhaustion |
|---|---|
| admission (before turn 1) | refuse, with `stop_reason: "budget"` — the cheapest possible outcome |
| turn boundary | stop cleanly, **flush the trajectory** (§8) — the task is not lost |
| spawn (fan-out) | clamp `n_requested` to what the budget affords, and report the clamp |

The third is the one that prevents the panel's overnight scenario `[T]`: an agent spawning
100,000 sub-agents has to be clamped by arithmetic that runs *before* the spawn, not by an
alert that fires after the invoice arrives. Clamping and reporting is preferred to refusing,
because a partially-completed fan-out is usually more useful than none, and because the
clamp is itself the signal that the task's decomposition was wrong.

---

## 8. The feedback layer: trajectory streaming

Gao calls trajectory collection "the key of the agentic modeling" `[T]`. Combined with
Tworek's latency argument — a 12-hour trajectory yields two gradient steps a day `[T]` —
the design requirement is unambiguous and it is a *serving* requirement:

> A trajectory that is lost costs, at the corpus's own numbers, half a day of training
> throughput. A trajectory that is streamed and truncated costs only its missing turns.

So the trajectory is written **incrementally, per turn**, not assembled at the end:

```python
def emit_turn(task: TaskState, turn: Turn, payload: bytes) -> None:
    """Append one turn to the task's trajectory stream. Must be durable before the
    next turn starts."""
```

Three properties, each with the failure it prevents:

- **Durable before the next turn begins.** Otherwise a crash between turns loses the turn,
  and a crash is exactly when the task is most likely to be abandoned.
- **Includes the failed turns.** A trajectory containing only successful turns is a biased
  training sample — and, per Tworek, the *whole* trajectory is what gets one reward `[T]`,
  so a partial trajectory is not merely biased, it is unscorable.
- **Carries `cache_hit_prefix_tokens` and `stop_reason`.** These are the fields that let a
  training-side reader tell a task that finished from a task that was stopped, which is
  the difference between a positive and a negative example.

Retention is the open question and it is flagged, not resolved: trajectory streams are
large, they are the training signal, and this blueprint found no corpus guidance on how
long to keep them.

---

## 9. The budget controller

```python
@dataclass
class Budget:
    gpu_seconds_limit: float       # in GPU-seconds, never in currency (see HLD §9)
    tokens_limit: int
    wall_clock_limit_s: float
    retry_limit: int
    spawn_limit: int
    consumed: BudgetUsage
```

**Denominated in GPU-seconds and tokens, never in currency.** No vendor price is asserted
anywhere in the corpus this blueprint draws on, so a currency-denominated budget would
require a fabricated price. A reader with a rate multiplies at the edge; the controller
does not need to know it.

**Enforced at admission, not observed after the fact.** This is the same placement as T14's
flow control and for the same reason: by the time a cost metric has been scraped, the
tokens are already spent. The controller must sit on the request path.

**Per-task and per-tenant, and the second one is not optional.** A per-task budget stops a
runaway agent. It does not stop a thousand well-behaved agents, which is the normal case
and the one that produces the panel's overnight surprise. The tenant budget is the one that
bounds the fleet, and it is the one T19 (FinOps) develops further.

---

## 10. Metrics, counters and the two dashboards

The metric set is deliberately split into two groups that are never graphed on the same
axis, because HLD §3's finding is that a single dashboard produces a wrong decision.

**Latency group (what the user feels):**

| Metric | Notes |
|---|---|
| `agent_task_wall_seconds` | end-to-end, p50/p95/p99 |
| `agent_tool_wall_seconds{tool_id}` | expect this to dominate at warm cache |
| `agent_model_wall_seconds` | expect this to be small; if it is not, the cache is cold |
| `agent_turns_per_task` | the shape of the workload; drives everything in §2 |

**Cost group (what the fleet spends):**

| Metric | Notes |
|---|---|
| `agent_prefill_tokens_total` | **the number the whole blueprint is about** |
| `agent_prompt_tokens_per_turn` | should be linear in turn index at working reuse |
| `agent_cache_hit_prefix_tokens` | per turn; §3's stability signal |
| `agent_gpu_seconds_per_task` | the capacity unit |
| `agent_fanout_ratio` | `total GPU-s / parent GPU-s`; §7's clamp trigger |

**The two derived ratios that must exist:**

```
cost_per_completed_task = agent_gpu_seconds_per_task / completed_tasks
prefill_per_turn_growth  = slope of agent_prompt_tokens_per_turn vs turn index
```

The second is the diagnostic that separates the two causes of an expensive agent fleet. A
*linear* slope with a large intercept is the tool-schema tax (HLD §4.1) — fixable by moving
schemas into the stable prefix and retrieving them per task. A *superlinear* slope is
missing prefix reuse (HLD §4) — fixable by cache work and placement. The two have the same
symptom on the bill and completely different fixes, and one graph separates them.

**Prefix-affinity routing.** The router is T14's; this blueprint consumes it. The
requirement it adds is that `scope_id` from §6 becomes an input to placement alongside the
prefix identity T14 already uses. A sub-agent placed on a replica that does not hold its
parent's scope pays full cold prefill, and per `run.py` §5 that is the difference between
1,924 and 2,418 GPU-seconds per 512 sub-agents at 90% reuse — a 20% swing decided entirely
by placement.

---

## 11. Failure handling

| Failure | Detection | Response |
|---|---|---|
| Prefix instability (§3) | `cache_hit_prefix_tokens` ≈ 0 with a stable-looking prompt | alert; the harness must be fixed, not the cache |
| Context overflow | prompt approaches the window mid-task | forced compaction, `stop_reason: "compacted"`, cache-miss spike expected and attributed |
| Orphaned tool | lease alive, owning task terminal | reaper kills by `task_id` (§4) |
| Retry storm | `retry_count` rising per task | per-task retry limit; charge retries against the budget |
| Runaway fan-out | `spawn_limit` reached | clamp and report (§7) |
| Budget exhaustion | `consumed` ≥ limit | stop at the turn boundary, flush trajectory first (§8) |
| Task crash mid-turn | turn emitted without a terminal `stop_reason` | task is `"failed"`; the partial trajectory is still emitted |
| Trajectory loss | stream gap | the only unrecoverable failure in this table — hence §8's durability rule |

The last row is the one to design against hardest. Everything else in this table costs
money or time; a lost trajectory costs the training signal that the whole harness exists to
produce `[T]`, and it is unrecoverable because the agent's decisions cannot be replayed
without re-running the task.

---

## 12. Configuration surface

The knobs, with their defaults justified rather than asserted:

| Knob | Default | Why |
|---|---|---|
| `context.stable_prefix_required` | `true` | §3; off means the cache is decorative |
| `context.tool_schema_placement` | `"stable_prefix"` | HLD §4.1: 19.1% of prompt tokens |
| `tools.per_task_concurrency` | workload-derived | a global limit starves tasks (§4) |
| `tools.default_deadline_s` | policy-set, not agent-set | §4 |
| `fanout.shared_scope_max_turns` | `5` | the ~5.6-turn crossover from `run.py` §5 — below this, a fresh scope is cheaper |
| `fanout.spawn_limit` | budget-derived | §7 |
| `budget.denomination` | `gpu_seconds` | HLD §9: no fabricated price |
| `trajectory.emit` | `per_turn_durable` | §8; the only setting whose violation is unrecoverable |
| `retry.limit_per_task` | small, and charged to the budget | §5 |

`fanout.shared_scope_max_turns` is the one derived constant in this table and it is
derived from *this* parameterisation. It moves if the parent context, the tool-result size
or the per-turn output size moves. It is written as a derivation in `run.py` §5 rather than
as a fact, and should be recomputed for a real workload rather than copied.

---

## 13. Testing

Three levels, and the middle one is the one teams skip.

**Unit — the arithmetic.** `sim/agent_cost.py` is the reference: `new_tokens_at_turn` is
the specification of the cache contract, and any harness context assembler should be
testable against it. A test that asserts `prefill_tokens(shape, 1.0) ==
prefill_tokens_ideal(shape)` is checking that the two definitions of "perfect reuse" have
not drifted apart.

**Integration — prefix stability under a real assembler.** Build the same prompt twice,
one turn apart, and assert that the second is a byte-extension of the first. This test
catches every row of §3's violation table and it is cheap. Almost nobody writes it, and its
absence is why "we enabled prefix caching and saw no improvement" is such a common report.

**Load — the two-dashboard invariant.** Under a synthetic agent workload, assert that
`agent_tool_wall_seconds / agent_task_wall_seconds` is *rising* as cache residency
improves. If it is not, either the cache is not working or the tool latency is not being
measured, and this one assertion distinguishes HLD §3's two regimes without needing a
baseline.

## Sources

- `refs/Agentic_AI_Infra_transcripts_2/Jianfeng_Gao_-_Agentic_Modeling_via_Internalizing_Agent_Harnesses.txt`
  — the harness as "a bunch of code" and as "hard-coded modules"; the information layer
  (memory, context management, tools and skills), the execution layer (prompt builder,
  action parser, task decomposition, error recovering mechanisms) and the feedback layer
  ("collect all the trajectories generated by the agents", called "the key of the agentic
  modeling").
- `refs/Agentic_AI_Infra_transcripts_2/Panel_Agentic_AI_Infrastructure_Platform.txt`
  — overnight agents with "massive tool volume and token prefill" and the auto-refilled
  token provider; the hierarchy of agents and teams; scoped shared storage as an open
  standards problem; "an agent can in an ad hoc way spawn 100,000 sub agents"; the prior
  systems having been "either services or kind of human scale".
- `refs/Agentic_AI_Infra_transcripts_2/Jerry_Tworek_-_Opportunities_and_Challenges_for_Long_Horizon_Agents.txt`
  — cost proportional to trajectory length and "information per token... roughly one
  divided by n", hence a quadratic decay in the learning signal; the 12-hour trajectory at
  two gradient steps a day, 14 a week, 60 a month; `/goal`, plan mode and sub-agents as
  harness mechanisms that "are not back propagated through"; "whatever we can back
  propagate through wins".
- `refs/Agentic_AI_Infra_transcripts_2/Ankit_Sobti_-_From_Agent_Demos_to_Production_How_Postman_Is_Building_Reliable_AI.txt`
  — compounding cost and compounding confusion; the handoff of context as the biggest
  challenge; the monolith agent as the answer, cited here as the counter-argument to
  decomposition-by-default.
- `refs/Agentic_AI_Infra_transcripts_3/Gosia_Steinder_-_Beyond_Harnesses_Platform_Solutions_for_Agent_Reliability_Secur.txt`
  — why harness-local solutions do not compose into a platform.
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt`
  — saturation-based autoscaling and the workload variant autoscaler, consumed here through
  T15 and applied to the agent workload shape of §2.
