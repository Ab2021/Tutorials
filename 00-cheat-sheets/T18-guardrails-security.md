# Cheat Sheet: Guardrails, Security & Agent Governance

> `T18` · **Transcript coverage:** primary · [Case study](../01-case-studies/T18-guardrails-security.md) · [Blueprint](../03-design-blueprints/T18-guardrails-security/HLD.md) · [Interview bank](../02-interview-questions/T18-guardrails-security.md)

---

## Numbers to know

| Quantity | Value | Source |
|---|---|---|
| Rail positions | **input · retrieval · tool-call · output** | `[T]` guardrails |
| The dominant attack for agents | **indirect prompt injection** — payload arrives via tool/retrieved content | `[T]` |
| Why it is different from classic injection | the content is *data the agent was told to trust* | `[T]` |
| Non-determinism | temperature 0 is **not** deterministic — security controls cannot rely on replay | `[T]` CMU lecture 2 |
| Model fingerprinting | quantization scheme is **inferable from outputs** | `[T]` CMU lecture 2 |
| Blast radius control | **sandbox + scoped credentials**, not prompt instructions | `[D]` |
| Agent identity | the agent acts **as** someone — that identity needs scoping, not a shared key | `[T]` |
| Governance requirement | audit trail of every tool call and decision | `[T]` |

**The reframing that matters** `[D]`: a chatbot's worst case is a bad answer. An **agent's worst
case is a bad action** — it has tools, credentials and reach. Security therefore moves from
*filtering text* to **bounding capability**: what can this agent actually do, as whom, and what is
the damage if it is fully compromised?

---

## The one-table summary

| Rail | Sits | Catches | Blind to |
|---|---|---|---|
| **Input rail** | before the model | jailbreaks, PII, topic violations, oversized input | anything introduced later |
| **Retrieval rail** | on retrieved documents | poisoned/irrelevant content | — this is where **indirect injection** lands `[T]` |
| **Tool-call rail** | before execution | dangerous arguments, unauthorised tools, exfiltration | a *valid* call with valid-but-harmful intent |
| **Output rail** | after generation | leakage, unsafe content, broken schema | a correct-looking answer acted on downstream |
| **Sandbox** | around execution | arbitrary code effects | exfiltration through an allowed channel `[T]` |
| **Identity / delegation** | at the credential layer | privilege escalation, confused-deputy | over-broad scopes you granted `[D]` |
| **Budget / rate limits** | at the gateway | runaway loops, cost attacks | slow, cheap abuse `[D]` |
| **Audit log** | everywhere | nothing — but it is how you investigate | — `[T]` |

**Defence in depth is mandatory because no single rail is sufficient** `[T]`. The corpus's own
framing treats guardrails as a *system* of checks at each boundary, not a filter bolted onto the
prompt.

---

## Formulas / models

**The indirect-injection path** `[T]`:
```
untrusted source (web page, PDF, email, tool result, issue body)
   → enters context as "data"
   → model treats it as instruction
   → model emits a tool call
   → tool executes with the agent's credentials
```
Every arrow is a place to break the chain. The corpus's guidance lands on the last two:
**the tool-call rail and the credential scope** `[T]`.

**Blast radius** `[D]`:
```
blast_radius = capability(credentials) × reach(tools) × reversibility(actions)
```
Reduce any factor. Read-only credentials, a network-egress allowlist, and a human confirmation for
irreversible actions each cut it multiplicatively. **Prompt instructions do not reduce blast
radius** — a compromised model ignores them.

**Least-privilege for agents** `[D]`:
```
scope(agent) = tools it needs for THIS task
             ∩ resources it needs THIS run
             − anything irreversible without approval
```
Per-task, short-lived, narrow. The corpus's delegation discussion points the same way: the agent
acts **as** a principal, so scope that principal `[T]`.

**The confused-deputy check** `[D]`: does the agent have a permission the *user* does not? If yes,
a user can borrow it via injection. That is the whole vulnerability class.

---

## Configuration

```python
# Four rails, in the order content actually flows [T]
def handle(user_input, session):
    check_input_rail(user_input)                       # 1. input
    docs  = retrieve(user_input)
    docs  = check_retrieval_rail(docs)                 # 2. retrieved content is UNTRUSTED
    reply = llm.chat(build(user_input, docs))
    if reply.tool_calls:
        for call in reply.tool_calls:
            check_tool_rail(call, session.scopes)      # 3. tool-call: args + authority
            if irreversible(call):
                require_human_approval(call)           # [T] governance
            result = sandbox.run(call, session.scopes)
        reply = llm.chat(append(reply, results))
    check_output_rail(reply)                           # 4. output
    return reply
```

```yaml
# Sandbox + credential scoping — the controls that actually bound damage [D]
sandbox:
  network: {egress: allowlist, allow: ["api.internal.example"]}   # no open internet
  filesystem: {read: [/work], write: [/work/out]}
  limits: {cpu: 2, mem: 2Gi, wall: 60s}
credentials:
  type: short-lived-token          # minutes, not days
  scope: per-task                  # narrow to this run
  read_only: true                  # unless the task must write
audit:
  log: [tool_name, args_hash, principal, decision, result_status]
```

**Log the decision, not just the call** `[T]`: for each tool invocation record the principal, the
scope it used, whether a human approved it, and the outcome. This is what governance and incident
response actually need.

---

## Failure signatures

| Symptom | Likely cause | First check |
|---|---|---|
| Agent exfiltrates data via a tool | indirect injection + over-broad egress | network allowlist; credential scope `[T]` |
| Injection via a retrieved document | no retrieval rail | rail 2 is where it lands `[T]` |
| Agent does something irreversible | no approval gate | require confirmation for irreversible actions `[T]` |
| Guardrail bypassed by rephrasing | pattern-matching rail | rails are not a parser; add capability limits `[D]` |
| Agent has more access than the user | confused deputy | compare scopes; this is the bug `[D]` |
| Runaway loop burns budget | no step/token budget | budget at the gateway `[D]` |
| Cannot investigate an incident | no audit trail of tool calls | add structured audit logging `[T]` |
| Secrets in prompts or logs | no redaction | redact at the boundary `[T]` |
| Provider can infer the model | quantization fingerprint | a disclosure concern, not a perf one `[T]` |
| Blocked legitimate traffic | over-tight rail | tune with a measured false-positive rate `[D]` |

---

## Gotchas

- **Indirect injection is the agent-era attack, and it does not look like an attack.** The payload
  is a web page, a PDF, a tool response, a code comment `[T]`. Treat **all** retrieved and
  tool-returned content as untrusted input.
- **Prompt-level defences are not security controls.** They reduce frequency, not blast radius. The
  controls that bound damage are **sandboxing, scoped credentials, and approvals** `[D]`.
- **Guardrails belong at four positions, not one.** Input, retrieval, tool-call, output. Skipping
  the tool-call rail leaves the step that actually has consequences `[T]`.
- **Assume the output rail will be bypassed eventually** — then ask what the agent can still do.
  That question is your real security posture `[D]`.
- **Non-determinism defeats replay-based security.** Temperature 0 is not deterministic `[T]`, so
  you cannot reason "it said the same thing last time."
- **Quantization fingerprints your model** `[T]` — relevant if you consider the inference stack
  itself confidential.
- **The confused-deputy check is a one-liner with real teeth**: does the agent hold a permission the
  user does not? Fix those first `[D]`.
- **Human approval is a rail, not a bottleneck** — apply it to *irreversible* actions only `[T]`.
- **Budget limits are a security control.** An agent without step/token limits is a self-inflicted
  denial of service `[D]`.
- **Audit is non-optional for governance.** Regulated deployments need to reconstruct exactly what
  the agent did, as whom, and why `[T]`.
- **Isolation must be at the execution layer.** A sandbox that shares a filesystem or the network
  namespace is not a sandbox `[D]`.

---

## When to use what

| Situation | Apply |
|---|---|
| Any agent with tools | four rails + sandbox + scoped credentials |
| Reads untrusted content (web, email, docs) | retrieval rail; never trust retrieved text `[T]` |
| Executes code | hard sandbox: no network, ephemeral FS, resource caps `[D]` |
| Destructive capability exists | human approval on irreversible actions `[T]` |
| Multi-tenant | per-tenant scopes; never a shared service account `[D]` |
| Regulated / audited | full audit trail + retention policy `[T]` |
| Cost-sensitive | budgets and rate limits as controls `[D]` |
| Confidential inference requirement | confidential computing / no fingerprint leakage — see [T19](T19-finops-sovereignty.md) `[T]` |

---

## Sources

- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/LLM_Guardrails_Stop_Injection_Leaks_Hallucination_NeMo_Guardrails.txt`
- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/How_AI_Agents_Actually_Work_ReAct_Tools_Reflexion.txt`
- `refs/Agentic_AI_Infra_transcripts_2/Ankit_Sobti_-_From_Agent_Demos_to_Production_How_Postman_Is_Building_Reliable_AI.txt`
- `refs/Agentic_AI_Infra_transcripts_2/Panel_Agentic_AI_Infrastructure_Platform.txt`
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_10_Incorporating_Tools.txt`
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_11_Agents_and_Multi-Agent_Communication.txt`
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts_2/CMU_LLM_Inference_2_Probability_Review_and_Code_Examples.txt`
