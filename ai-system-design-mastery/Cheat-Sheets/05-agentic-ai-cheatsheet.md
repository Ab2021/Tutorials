# Agentic AI Cheatsheet

**How to use this** — The agent-specific reference: the loop, the budgets, tool design, MCP, multi-agent topologies, agent evaluation, and the security surface that only exists once a model can act. The three things to carry into any agent interview: **every loop needs a budget enforced in code**, **tool results are attacker-controlled input**, and **grade the trajectory, not just the answer**.

---

## 1. The agent loop

```
        ┌──────────────────────────────────────────────┐
        │                                              │
        ▼                                              │
  ┌───────────────┐                                    │
  │ BUDGET CHECK  │  steps < 10?  cost < $0.50?        │
  │ (top of loop) │  wall < 120 s?  tool calls < 3?    │
  └───────┬───────┘  ← BEFORE the action, never after  │
          │ exceeded → graceful stop, return partial   │
          ▼                                            │
  ┌───────────────┐                                    │
  │ ASSEMBLE      │  system prompt (static, first)     │
  │ CONTEXT       │  + tool definitions                │
  │               │  + conversation history            │
  │               │  + tool results (TRUNCATE these)   │
  └───────┬───────┘  + memory                         │
          ▼                                            │
  ┌───────────────┐                                    │
  │ MODEL CALL    │  → reasoning + tool_calls OR answer│
  └───────┬───────┘                                    │
          ▼                                            │
  ┌───────────────┐   ┌────────────────────────────┐   │
  │ PARSE         │──▶│ final answer → EXIT        │   │
  └───────┬───────┘   └────────────────────────────┘   │
          │ tool calls                                 │
          ▼                                            │
  ┌───────────────┐                                    │
  │ SAFETY GATE   │  permission → schema → tier →      │
  │ (in code)     │  flags → velocity → budget         │
  └───────┬───────┘  T3/T4 → out-of-band confirmation  │
          ▼                                            │
  ┌───────────────┐                                    │
  │ EXECUTE TOOL  │  timeout, retry transient, sandbox │
  │ → structured  │  errors as data, not exceptions    │
  │   result      │                                    │
  └───────┬───────┘                                    │
          │                                            │
          └────────────────────────────────────────────┘

  EVERY step emits a trace span:
  reasoning · tool name · canonical args · raw result · state change
```

**The invariant:** the budget check is at the **top** of the loop, before the action. A budget checked after the side effect isn't a budget.

---

## 2. Agent or workflow?

| Question | If yes |
|---|---|
| Is the step sequence known at design time? | **workflow** — enumerate, test, ship |
| Would a competent human sometimes check, then decide, then check again? | agent candidate |
| Does the task need context isolation across sub-tasks? | multi-agent candidate |
| Is the path fixed with one or two model-filled slots? | **workflow** |
| Can one agent with the same total token budget do it? | **single agent** |

**Decision table:**

| Pattern | Cost | Testability | Use when |
|---|---|---|---|
| Deterministic workflow | lowest | full | steps known |
| Workflow + agentic sub-steps | low | high | **the default recommendation** |
| Single agent, fixed tools | moderate | medium | path genuinely unknown |
| Agent with planning | higher | medium | dependent multi-step task |
| Multi-agent supervisor | 4–10x single | low | context isolation required |

**The rule:** every path you can enumerate is a path you can test. Ship the workflow skeleton and put agentic steps only where the path can't be enumerated.

---

## 3. Budgets

| Budget | Default | Why that number |
|---|---|---|
| **Steps** | 10 | legitimate step counts are front-loaded; >10 usually means badly-scoped |
| **Cost** | $0.50 / task | steps don't bound cost — one step can be 50k tokens |
| **Wall clock** | 120 s | a user won't wait longer; catches hung tools |
| **Same tool + same args** | 3 calls | catches the most common loop pathology |
| **Delegation depth** | 3 | bounds recursive delegation |
| **Supervisor fan-out** | 3–5 workers | above 5, synthesis quality degrades and cost grows linearly |

**Loop-breaking, three layers:**

1. Step budget — bounds the total
2. **No-progress detector** — same call twice, or unchanged state across 3 steps → stop at step 4 instead of step 10 (~60% cost saving)
3. Repetition penalty — show the model its own call history so it can see the repeat

Watch for the **oscillating loop**: alternating states that each undo the previous. No-progress detection catches it because *state* repeats even when steps differ.

---

## 4. Tool design

**The eight rules:**

1. **Name says what it does** — `search_orders` vs `lookup_order` is a wrong-selection generator
2. **Description says when to use AND when not to** — "use `track_package` for a specific package"
3. **Typed, constrained parameters** — enums validate before execution; free text doesn't
4. **State the return shape and size**
5. **Mark read-only vs mutating** — the safety gate needs it
6. **Validate arguments before execution** — the model produces schema-violating args
7. **Truncate large results** — a 200 KB JSON blob blows the context on step two
8. **Return errors as structured data** — code, message, retryable flag

**Tool count:**

| Count | Effect |
|---|---|
| 10–15 | reliable selection — **target** |
| 20–30 | degradation begins, non-gracefully |
| 40+ | confident wrong selection — worse than refusing |

Scaling beyond 15: tool namespaces/hierarchy, retrieval over tool definitions, or sub-agents with focused toolsets. All trade latency for accuracy; all beat a flat list.

**Error taxonomy for tool results:**

| Class | Examples | Behaviour |
|---|---|---|
| Transient | timeout, 503, 429 | tool layer retries internally; model sees it only after exhaustion, with "retrying may help" |
| Permanent | 400, validation error, not found | return the specific failure so the model changes approach |
| Permission | denied, not authorised | never silently worked around — an alternate route is an audit finding |

---

## 5. Function calling mechanics

```
request:  messages + tool definitions (name, description, JSON Schema)
          │
model:    returns tool_call { id, name, arguments: JSON }
          │
you:      validate arguments → execute → append result message with matching id
          │
model:    sees the result, decides next
```

**Parallel tool calls** — one model round, several independent executions. Real latency win when calls don't depend on each other.

**The tool description is a prompt.** It is the highest-leverage text in an agent system — vague descriptions cause wrong selection far more often than model capability limits do.

---

## 6. Planning architectures

| Architecture | Shape | Strength | Failure |
|---|---|---|---|
| **ReAct** | thought → action → observation, repeat | auditable trace | reasoning-action mismatch |
| **Plan-and-execute** | plan up front, then execute | reviewable artefact, cost estimate, resumption | plan persistence after invalidation |
| **Replan-on-failure** | plan, execute, replan when a step fails | adaptive | replanning every step = slow reactive |
| **Reflexion** | retry with the error fed back | recovers many failures | cost multiplies |

**ReAct's real value:** the reasoning is a **trace you can audit**. It distinguishes "the tool lied" from "it misread the result" from "it had the wrong goal." Without it you have the action and no idea what the agent believed.

**Plan-and-execute needs a replanning trigger:** step failure, a result contradicting an assumption, or an explicit signal. Unbounded replanning defeats the point.

---

## 7. Model Context Protocol

**Topology:** host (the AI app) contains clients → each client holds a 1:1 connection to a server → servers expose tools, resources, prompts.

**Why it exists:** N applications × M data sources = N×M integrations. MCP makes it N clients + M servers.

| Transport | Where the server runs | Auth | Use when |
|---|---|---|---|
| **stdio** | local subprocess of the host | none needed | **default** — no network surface at all |
| **SSE** | remote | HTTP | legacy; avoid for new work |
| **Streamable HTTP** | remote | OAuth 2.1 | remote resources, current standard |

**The security reality:**

| Finding | Value |
|---|---|
| Publicly exposed MCP server instances | ~21,000 |
| Without OAuth | **~92%** |

**Three problems, layered:**

- **Exposure** — bound to a public interface when it should be local
- **No auth** — the transport carries no identity
- **Confused deputy** — the server holds credentials the agent doesn't, so the agent can borrow its privileges

**Six controls:**

1. stdio by default; TLS + OAuth 2.1 with token audience binding if remote
2. Credential scoping — read-only DB user for a read-only tool
3. Egress allowlisting — makes injection exfiltration structurally fail
4. Per-tool authorisation, checked per call, not just at connect
5. Argument validation + result scanning (the server is a content channel)
6. Pin versions; treat a server update as a reviewed change

**When an MCP server changes:** schema drift (fails validation, looks like a bad call), capability removal (no fallback), **description changes** (silent behaviour change, hardest to catch), expanded scope (a new tool your agent starts using — permission escalation with no human decision). Mitigation: **allowlist the tool surface, never discover blindly**, plus contract tests and a canary.

---

## 8. Multi-agent topologies

| Topology | Accountability | Termination | Use when |
|---|---|---|---|
| **Supervisor / hierarchical** | single owner | **provable** (depth counter) | default choice |
| **Peer-to-peer** | none | needs a global budget + a done-rule | fluid, specialist-handoff tasks |
| **Pipeline** | stage owner | natural | sequential stages |

**Handoff levels:**

| Level | Keeps | Loses |
|---|---|---|
| Full history | everything | blows context, leaks noise |
| Summary | gist | the detail that mattered |
| **Structured handoff** | goal, constraints, evidence, success criteria, state | — **use this** |

Use **references, not copies**, for large artefacts. The field people omit is **success criteria** — without it the worker optimises for something plausible.

**Failure modes:** delegation loops (cap depth, don't give workers delegation tools), context loss at handoff, synthesis failure (contradictory outputs blended into something true of neither), redundant work, cost multiplication.

**Cost profile:** superlinear. Single-agent $0.05–0.50 → supervised multi-agent $0.30–2.00 for the same task, roughly **4–10x**. Justify it by a sub-task that genuinely needs its own context window.

---

## 9. Agent memory

| Type | Lifetime | Contents | Failure |
|---|---|---|---|
| Working | one task | plan, current step, accumulated results | context overflow |
| Episodic | across sessions | what happened before with this user/task | noise accumulation |
| Semantic | durable | facts, preferences, environment state | **poisoning** |

**Write policy — three tests:**

1. **Durable?** Will it be true and relevant next week?
2. **Reusable?** Will it change a future decision?
3. **Safe?** Could writing it cause harm if wrong or attacker-influenced?

Then: extraction → dedupe (an agent that remembers a preference four times has polluted itself) → **validation** (reject writes containing instructions rather than facts).

**Memory poisoning controls:** provenance tracking, validate before write, tenant/user scope binding, periodic audit. The danger is that a poisoned agent **doesn't fail** — it succeeds with a subtly wrong belief for as long as the memory persists.

**Per-operation evals:** extraction / update / QA measured separately. A **5% extraction error compounds**, because a wrong memory generates future wrong behaviour.

---

## 10. Agent failure modes

| Failure | What it looks like | Detection signal | Fix |
|---|---|---|---|
| **Reasoning-action mismatch** | thought says X, call does Y | validate call against stated intent | add intent-consistency check |
| **Over-retrieval** | retrieves far more than needed | retrieval calls per task vs baseline | cap retrieval calls |
| **Tool flailing** | repeated calls with variations | per-tool counter + argument similarity | no-progress detector |
| **Premature commitment** | ignores contradicting evidence | agent ignores a later contradictory observation | replanning trigger on contradiction |
| **Self-jailbreaking** | finds a route around a restriction | egress monitoring; audit actual actions | architectural controls, not prompts |
| Runaway loop | same call repeatedly | step counter + state comparison | budgets (all four) |
| Oscillation | A→B→A→B | state repeat detection | no-progress detector |
| Context blowout | step 2 fails on a huge tool result | context token count per step | truncate tool results |
| Cost explosion | one task consumes the budget | cost per task p95 | cost budget + cheap routing |
| Partial completion | some side effects done | post-condition checks | durable execution + compensation |
| Silent wrong answer | right format, wrong content | **post-condition verification** | verifiers in code |

**Partial failure contract:** when steps 1–2 succeed and step 3 fails, say so explicitly ("I cancelled the subscription but could not process the refund; reference: X"). Never report a generic failure that hides the partial side effect, and never report success. Where partial completion is unacceptable, **compensate** — undo steps 1–2.

---

## 11. Agent evaluation

**Four layers:**

| Layer | Grades | Method |
|---|---|---|
| Task success | did the goal get achieved | **post-condition check in code** |
| Trajectory | were the steps justified | PRM / auditor agent / rules |
| Efficiency | steps vs minimum, cost per task | counters |
| Safety | stayed within permissions and tiers | audit + boundary tests |

**Why trajectory grading is mandatory:** the right answer reached by wrong reasoning, and the right answer after four destructive calls, both score 100% on outcome. The outcome metric is necessary, not sufficient.

**Prefer deterministic verification.** A post-condition ("the order is cancelled", "the file exists") is cheap, unambiguous, and **impossible to talk your way past**. An LLM reading a transcript is not.

**Pass^k:**

```
Pass^k = P(succeed on all k independent trials)

Pass^1 70%  +  Pass^4 12%  →  works on the easy path only
Target: Pass^4 ≥ 0.60 for anything with side effects
```

Requires an eval harness that can **reset world state** between trials. Pass^1 is quality; Pass^k is whether you can ship it without a human watching.

**Debugging sequence** — from the wrong action back to the wrong belief:

1. Was the information **available**? → tool/retrieval problem
2. Available but **misread**? → context problem (noise, truncation)
3. Understood but **wrong goal**? → task specification problem
4. Goal right, **plan wrong**? → planning problem

Four causes, four fixes. Assuming planning when it's a tool-result problem wastes weeks.

**Reliability levers, in order of return:** constrain the action space (fewer, better tools) → add deterministic verification → reduce degrees of freedom (workflow skeleton) → retries with reflection → route easy cases away from the agent entirely.

**Warn against:** prompt engineering as the primary lever. If the agent has a structural problem — a missing post-condition, a tool it shouldn't have — no wording fixes it.

---

## 12. Agent security

**The lethal trifecta:**

```
   private data  +  untrusted content  +  outbound channel
                     ↓
        exfiltration becomes possible

  Remove ANY ONE → structurally impossible, regardless of the injection
```

Guardrails act on **none** of the three. The four architectural controls do:

| Control | Removes |
|---|---|
| Least privilege | which data is reachable |
| No credentials in context | the key that could be emitted |
| Egress allowlisting | the outbound channel |
| Out-of-band confirmation | self-approval by injected instruction |

**Why agents differ from chatbots:** a jailbroken chatbot produces a transcript; a jailbroken agent with an email tool and a customer database **sends the database**. The blast radius is categorically different.

**Untrusted input channels:** user message, retrieved documents, tool results, memory, anything the agent fetched. Every one is attacker-controllable; the model can't distinguish operator instruction from content-embedded instruction without architectural help.

**Risk tiers and gates:**

| Tier | Gate | Confirmation |
|---|---|---|
| T0 read | auto | no |
| T1 low write | auto + audit | no |
| T2 medium | auto + notify | no |
| T3 high | blocked | **yes** |
| T4 destructive | blocked | yes + typed scope acknowledgement |

**Gate order (in code):** permission → schema verification → risk tier → suspicious flags → velocity → budget.

**Confirmation rules:** bound to the **canonicalised arguments** (exact recipient, amount, resource), post-proposal, out-of-band, **single-use**, ~60 s TTL.

**Sandbox config for code execution:**

```
egress:        loopback only
metadata IP:   169.254.169.254 BLOCKED
mounts:        none
secrets:       none in env
wall clock:    300 s
lifecycle:     ephemeral, destroyed after use
```

**Bounding blast radius:** per-agent credentials (not a shared service account), per-resource scoping, **server-side volume caps** on bulk operations (not prompt-level), per-tool and per-agent rate limits, human gates on destructive ops, ephemeral sandboxes. Then **red-team it** — deliberately inject an exfiltration instruction and confirm it fails.

**Indirect injection is the main agent attack surface** because tool results arrive through a channel the model treats as trustworthy. Detection helps; **removing a leg of the trifecta is the control.**

---

## 13. Durable execution

**Mapping:**

| Concept | Agent equivalent |
|---|---|
| Workflow | the agent loop |
| Durable activity | each model call and each tool call |
| Idempotency key | `workflow_id + step_id` |
| Durable timer | budget/wait periods |

**Decision order — use durable execution if any is yes:**

1. Does the step have an **irreversible side effect**?
2. Does the workflow **outlast a deploy**?
3. Is **re-running it unacceptable** (cost, side effects, user-visible duplication)?

**The determinism constraint:** replay must reproduce the same decisions, so non-deterministic values must come from recorded results, not recomputation. **Versioning hazard:** changing the workflow definition while instances are in flight can break replay — version the definition, don't edit it in place.

| Approach | Resumption | Complexity | Use when |
|---|---|---|---|
| Stateless retry | restart from scratch | low | short, cheap, idempotent |
| Checkpoint to DB | resume from last checkpoint | medium | moderate length, idempotent steps |
| Durable execution engine | resume mid-step | high | long, side-effecting, must-not-duplicate |

---

## 14. Numbers to remember

| Number | Value |
|---|---|
| Step budget | 10 |
| Cost budget | $0.50 / task |
| Wall clock budget | 120 s |
| Same tool + args cap | 3 |
| Delegation depth cap | 3 |
| Supervisor fan-out | 3–5 workers |
| Multi-agent cost multiplier | 4–10x single agent |
| Tools in active surface | 10–15 |
| Tool degradation onset | 20–30 |
| Sandbox wall clock | 300 s |
| Confirmation TTL | 60 s |
| Pass^4 target | ≥ 0.60 |
| MCP exposed instances | ~21,000 |
| MCP without OAuth | ~92% |
| Memory extraction error compounding | 5% per operation |
| Trajectory failure modes | 5 named |
| Agentic RAG round trips | 3–10 |
| Multi-query paraphrase count | 3–5 |

---

## 15. Interview one-liners

- "The loop needs a budget enforced in code, not in the prompt."
- "Every tool call is an untrusted-input boundary, because tool *results* are attacker-controlled."
- "I grade the trajectory, not just the answer."
- "Every path you can enumerate is a path you can test."
- "The question isn't whether the agent can be trusted — it's what happens if it's fully compromised, and whether that's acceptable."
- "Fewer, well-scoped tools beat a large flat surface. Past 20–30 options you get confident wrong selection."
- "A cost cap is a poor proxy for damage. Steps, time, and per-tool limits are closer to the action."
- "Multi-agent is a context-isolation mechanism before it's anything else."
- "Removing any one leg of the lethal trifecta makes exfiltration structurally impossible. Guardrails remove none of them."
- "Pass^1 is quality. Pass^k is whether you can ship it."
- "Prefer a post-condition check to a judge reading the transcript — post-conditions can't be argued with."
- "MCP standardises the interface, not the permissions."

---

## Cross-links

- [Agentic System Design](../HLD/05-ai-systems/06-agentic-system-design.md)
- [MCP, Tool Use and Multi-Agent](../HLD/05-ai-systems/07-mcp-tool-use-and-multi-agent.md)
- [Memory and Context Engineering](../HLD/05-ai-systems/08-memory-and-context-engineering.md)
- [Guardrails and Safety for AI Systems](../HLD/05-ai-systems/04-guardrails-and-safety-for-ai-systems.md)
- [Security, Guardrails and Compliance](../Operations/08-security-guardrails-and-compliance.md)
- [Reliability Patterns](../Operations/07-reliability-patterns.md)
- [LLMOps for AI Systems](../Operations/10-llmops-for-ai-systems.md)
- [Agentic AI Questions](../Interview-Questions/05-agentic-ai-questions.md)
- [Agentic Coding Assistant Case Study](../Case-Studies/HLD/11-agentic-coding-assistant.md)
- [RAG and LLM Cheatsheet](04-rag-and-llm-cheatsheet.md)
- [Operationalisation Cheatsheet](06-operationalisation-cheatsheet.md)
