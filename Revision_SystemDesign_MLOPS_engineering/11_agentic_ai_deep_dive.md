# AGENTIC AI DEEP DIVE — Concepts, Architecture, and Interview Answers
> Agentic AI is a major interview gap from your transcripts. This file covers state machines, tool calling, structured outputs, memory, reflection, guardrails, and production design.

---

## SECTION 1: WHAT MAKES A SYSTEM "AGENTIC"

### Beyond Simple LLM Calling

A basic LLM call takes a prompt and returns text. An agentic system uses the LLM to:
- Choose which tool or action to take
- Process the result
- Iterate toward a goal
- Maintain state across multiple turns

**Agentic capabilities:**
- Planning: break a complex goal into sub-tasks
- Tool use: call APIs, query databases, run functions
- Memory: retain context from earlier steps
- Reflection: evaluate and correct its own outputs
- Action: make decisions that affect downstream systems

### Agent vs Workflow

| Pattern | Description | Use Case |
|---|---|---|
| Single-turn LLM | One prompt, one response | Summarization, classification |
| Multi-turn chat | Stateful conversation | Support chatbot |
| Deterministic workflow | Fixed sequence of steps | Claim processing pipeline |
| Agentic loop | LLM chooses next step dynamically | Fraud investigation assistant |
| Multi-agent system | Multiple agents with roles | Complex research or code generation |

### Interview One-Liner

> "An agentic system is an LLM-driven loop that can plan, use tools, remember context, and reflect on its outputs. It is not just generating text — it is taking actions toward a goal."

---

## SECTION 2: STATE MACHINES AND WORKFLOW ORCHESTRATION

### Why Use a State Machine

Production agents need predictable behavior. A state machine defines:
- Valid states (e.g., PLAN, GATHER, REASON, GENERATE, REVIEW, END)
- Allowed transitions between states
- Entry and exit conditions for each state
- Guards to prevent infinite loops

### Example States for a Fraud Investigation Agent

1. **PLAN:** Determine what evidence is needed
2. **GATHER_EVIDENCE:** Retrieve policy, claims, graph, and similar cases
3. **REASON:** Analyze patterns and contradictions
4. **DRAFT_FINDINGS:** Produce a structured summary
5. **REVIEW:** Reflection layer checks evidence support
6. **END:** Return final output or escalate to human

### Guard Conditions

- Max iterations: agent cannot loop more than N times
- Timeout: total execution time capped
- Cost budget: max number of LLM calls or tokens
- Allowed tools: restrict which tools can be called from each state

### Deterministic vs Dynamic

**Deterministic workflow:**
- Fixed sequence of steps
- Easier to test and debug
- Less flexible

**Dynamic agentic workflow:**
- LLM decides next step
- More flexible
- Harder to guarantee termination and correctness

**Recommendation for production:**
Start with a deterministic workflow. Add dynamic decision points only where the problem genuinely requires it.

### Interview One-Liner

> "I orchestrate agentic systems with a state machine. Each state has a clear purpose and allowed transitions. I add guards for max iterations, timeouts, and cost limits so the agent cannot run away."

---

## SECTION 3: TOOL USE AND FUNCTION CALLING

### Tool Calling Contract

Tool calling allows an LLM to invoke external functions. A tool definition includes:
- Name: descriptive, action-oriented
- Description: what the tool does and when to use it
- Parameters: JSON schema with types, descriptions, required fields
- Return format: what the tool returns

### Example Tools for a Fraud Agent

- **get_claim_details(claim_id):** retrieve claim record
- **get_claimant_history(claimant_id):** retrieve prior claims
- **query_graph(entity_type, entity_id):** retrieve network links from Neo4j
- **retrieve_similar_cases(text):** vector search over historical fraud cases
- **calculate_statistics(field, filter):** compute aggregations
- **submit_for_review(summary):** hand off to human investigator

### Tool Selection Strategy

The LLM receives a system prompt with all tool definitions. It decides which tool to call based on the current goal and prior results. The executor invokes the tool and feeds the result back.

### Idempotency and Safety

- Tools should be read-only where possible
- Write tools should require confirmation
- Tool calls should be idempotent
- Log every tool call for audit

### Interview One-Liner

> "I design tools as pure functions with clear input and output schemas. The LLM planner chooses which tool to call, and a deterministic executor invokes it. All tool calls are logged so the agent's reasoning is auditable."

---

## SECTION 4: STRUCTURED OUTPUT ENFORCEMENT

### Why Structured Output Matters

Agentic systems feed LLM outputs into downstream code. Unstructured text is hard to parse and risky.

**Benefits of structured output:**
- Deterministic parsing
- Schema validation
- Type safety
- Easier testing and monitoring

### Enforcement Methods

- **JSON mode:** LLM is constrained to output valid JSON
- **Function calling:** LLM returns arguments matching a function schema
- **Pydantic validation:** downstream code validates the output
- **Retry loop:** if validation fails, retry with stronger prompt

### Handling Failures

If the LLM repeatedly fails to produce valid output:
- Simplify the schema
- Reduce the number of fields
- Lower temperature
- Add examples in the prompt
- Fallback to a deterministic rule or human handoff

### Interview One-Liner

> "I never trust raw LLM text in production. I use JSON mode or function calling and validate every output with a schema. If validation fails, I retry and then fall back to a safe default."

---

## SECTION 5: MEMORY AND CONTEXT MANAGEMENT

### Types of Memory

1. **System prompt memory:** persistent instructions, taxonomy, guardrails
2. **Working memory:** intermediate results from the current task
3. **Short-term memory:** context from the current conversation or session
4. **Long-term memory:** retrieved knowledge from vector store or database

### Context Window Limits

Every LLM has a maximum context size. For long tasks:
- Summarize working memory periodically
- Retrieve only relevant documents
- Drop old, irrelevant conversation history
- Use smaller, focused prompts

### KV Cache Considerations

Autoregressive models cache key and value tensors for previously generated tokens. In multi-turn agents:
- Reuse the cache when context is unchanged
- Cache invalidates when system prompt or retrieved documents change
- Long contexts increase memory usage

### Strategies for Long Documents

- Chunking with overlap
- Hierarchical summarization
- Retrieval to pull only relevant sections
- Separate handling of long documents from short queries

### Interview One-Liner

> "I manage context by separating system instructions, working memory, and retrieved long-term knowledge. I summarize aggressively to stay within the context window and retrieve only what is needed for the current reasoning step."

---

## SECTION 6: SYSTEM PROMPTS, USER PROMPTS, AND PROMPT ENGINEERING

### System Prompt Role

The system prompt sets the agent's role, constraints, output format, and safety rules. It is the most important prompt for consistency.

### User Prompt Role

The user prompt contains the specific task or question for this turn.

### Prompt Engineering Best Practices

- Be explicit about output format
- Include few-shot examples when needed
- Define edge cases (what to do if information is missing)
- Keep system prompt stable across model versions
- Version control prompts like code

### Prompt Versioning

Track prompt changes and evaluate on a fixed test set. A small change can alter behavior significantly.

### Interview One-Liner

> "I treat prompts as code: versioned, tested, and evaluated on a fixed dataset. The system prompt defines the agent's role and constraints, while the user prompt carries the specific task."

---

## SECTION 7: REFLECTION AND SELF-CORRECTION

### Why Reflection Matters

LLMs can produce plausible but incorrect outputs. A reflection layer checks the output before it is trusted.

### Types of Reflection

- **Faithfulness check:** Does the output match the source evidence?
- **Consistency check:** Is the output internally consistent?
- **Completeness check:** Did the agent address all parts of the task?
- **Safety check:** Does the output violate policies?

### Reflection Implementation

1. Generate initial output
2. Run a separate reflection prompt asking the LLM or a judge to critique
3. If issues are found, feed critique back and regenerate
4. Limit iterations to prevent loops

### When Reflection Is Not Enough

- Escalate to human if the agent cannot resolve contradictions
- Use deterministic rules for high-stakes decisions
- Never let an agent make irreversible actions without confirmation

### Interview One-Liner

> "I add a reflection layer where a separate prompt reviews the agent's output for faithfulness, consistency, and completeness. If issues are found, the agent regenerates. The loop is capped to prevent infinite revision."

---

## SECTION 8: GUARDRAILS FOR AGENTIC SYSTEMS

### Input Guardrails

- Reject requests outside the agent's scope
- Block or mask PII/PHI before processing
- Detect toxic or adversarial prompts
- Enforce rate limits and authentication

### Output Guardrails

- Enforce structured output schema
- Block disallowed content
- Require citations for factual claims
- Prevent hallucinated actions

### Operational Guardrails

- Max number of tool calls per session
- Total token and cost budget
- Execution timeout
- Allowed tool whitelist
- Human-in-the-loop for high-risk actions

### PII Handling in Agentic Pipelines

- Anonymize inputs before sending to LLM
- Do not store raw PII in vector indexes or logs
- Use data classification tags
- Apply retention policies

### Interview One-Liner

> "Guardrails operate at three levels: input filtering, output validation, and operational limits. For insurance fraud, I anonymize claim notes before LLM processing, enforce schema on outputs, cap tool calls and cost, and escalate irreversible actions to humans."

---

## SECTION 9: OBSERVABILITY AND DEBUGGING

### What to Log

- Every LLM call: prompt, model, temperature, tokens, latency, cost
- Every tool call: tool name, inputs, outputs, duration
- Every state transition
- Reflection results and iteration counts
- Final output and downstream actions

### Tracing Agent Executions

Use a trace ID that follows one complete agent run across all LLM and tool calls. This makes debugging and audit possible.

### Metrics to Monitor

- Success rate: % of tasks completed without escalation
- Latency: p50, p95, p99 per task
- Cost per task
- Token usage per task
- Tool call distribution
- Hallucination rate from reflection
- Human escalation rate

### Interview One-Liner

> "I instrument every LLM call, tool call, and state transition with a trace ID. I monitor success rate, latency, cost, token usage, and human escalation rate. Without this observability, agentic systems are black boxes."

---

## SECTION 10: DESIGNING AN AGENTIC FRAUD INVESTIGATION ASSISTANT

### Problem Scope

Help SIU investigators review a claim by gathering evidence, finding patterns, and drafting a structured summary.

### Architecture

**State machine:**
1. PLAN: decide evidence needed
2. GATHER: call tools to retrieve claim details, claimant history, graph links, similar cases
3. REASON: identify red flags and contradictions
4. DRAFT: produce structured findings
5. REVIEW: reflection layer checks evidence support
6. END: return summary and confidence

### Tools

- Policy and claims API
- Neo4j graph query
- Vector retrieval over fraud case library
- Statistical lookup for norms
- Human escalation endpoint

### Guardrails

- Read-only tools except for escalation
- Max 5 reasoning iterations
- PII masking before LLM
- Citation required for every finding
- Human review required for high-confidence fraud assertions

### Interview One-Liner

> "I would design a fraud investigation agent with a state machine: plan, gather evidence from policy, graph, and vector store, reason over it, draft findings, and reflect. All evidence is cited, PII is masked, and high-stakes conclusions require human review."

---

## SECTION 11: EVALUATING AGENTIC SYSTEMS

### Levels of Evaluation

1. **Component evaluation:** each tool and prompt works in isolation
2. **End-to-end task evaluation:** agent completes full tasks correctly
3. **Human evaluation:** investigators rate usefulness and accuracy
4. **Business outcome:** does the agent reduce investigation time or improve detection?

### Task-Based Metrics

- Task success rate
- Steps to completion
- Tool selection accuracy
- Final output correctness vs ground truth
- Human correction rate

### Controlled Test Sets

Maintain a fixed set of representative cases. Run the agent on them after every prompt or model change.

### Interview One-Liner

> "I evaluate agentic systems at component, end-to-end, human, and business levels. I maintain a fixed test set and run it after every change to detect regressions in task success rate and output quality."

---

## SECTION 12: DEPLOYMENT TRADE-OFFS

### Synchronous vs Asynchronous Agents

**Synchronous:**
- Fast response required
- Simple tasks
- Higher latency sensitivity

**Asynchronous:**
- Complex multi-step reasoning
- Long-running tasks
- Lower latency sensitivity

### Cost Management

- Use cheaper models for simple steps
- Reserve expensive models for reasoning and reflection
- Batch tool calls when possible
- Cache retrieved documents and embeddings
- Set per-task cost budgets

### Model Selection per Step

- Planning: capable model
- Tool selection: capable model
- Simple extraction: smaller, cheaper model
- Reflection: strong model, possibly stronger than the generator

### Interview One-Liner

> "I deploy agentic tasks asynchronously when they require multi-step reasoning, and synchronously only for simple actions. I use smaller models for extraction and larger models for planning and reflection to manage cost."

---

## SECTION 13: AGENTIC AI INTERVIEW SCENARIOS

### "Design an agentic system."

> "I would start by defining the goal and whether dynamic decision-making is truly needed. Then I design a state machine with clear states, tool contracts, and guardrails. I use structured output, add a reflection layer, and log every step for observability. I keep the agent read-only for research tasks and require human approval for actions."

### "How do you prevent an agent from hallucinating?"

> "I use retrieval so claims are grounded in evidence, structured output so responses are constrained, reflection to catch unsupported claims, and citations so humans can verify. For high-stakes outputs, I require human review."

### "When is an agent better than a workflow?"

> "An agent is better when the task requires dynamic planning and the optimal sequence of steps varies per input. A deterministic workflow is better when the process is fixed, auditable, and must meet strict compliance requirements. I default to workflows and add agency only where it adds clear value."

### "How do you make an agentic system safe in production?"

> "I apply input and output guardrails, cap iterations and cost, use read-only tools where possible, require human approval for irreversible actions, log everything, and maintain a fallback path when the agent fails."

---

## SECTION 14: SKILLS, MCPs, AND TOOL ECOSYSTEM

### What Are Agent Skills

A skill is a packaged capability the agent can invoke, such as:
- Retrieve fraud case history
- Summarize claim notes
- Check policy details

Skills hide implementation details from the agent planner.

### Model Context Protocols (MCPs)

MCPs define how external systems expose tools and context to the agent. They standardize:
- Tool definitions
- Authentication
- Data formats
- Interaction patterns

### Why Standardization Helps

- Easier integration with multiple backends
- Consistent security controls
- Reusable tools across agents
- Better testing and monitoring

### Interview One-Liner

> "I expose capabilities to agents through well-defined skills or MCPs. This decouples the agent planner from backend implementations and makes tools reusable, testable, and secure."

---

## SECTION 15: LEAD-LEVEL AGENTIC AI EXPECTATIONS

### What Interviewers Look For

1. **Discipline:** start with workflow, add agency only when justified
2. **Safety:** mention guardrails, human oversight, and auditability
3. **Observability:** describe logging, tracing, and metrics
4. **Cost awareness:** talk about model selection and token budgets
5. **Business grounding:** connect the agent to a real use case and outcome

### Lead-Level Phrases

- "I default to deterministic workflows and add dynamic agentic steps only where the task is too variable for a fixed pipeline."
- "I design agentic systems with guardrails, reflection, and full observability from day one."
- "I treat agent outputs as intermediate artifacts that humans review before high-stakes decisions."
- "I use retrieval and citations to ground agent outputs in evidence and reduce hallucination."

---

## SECTION 16: REACT VS STATE MACHINE / LANGGRAPH

### What Is ReAct

ReAct (Reason, Act) is a pattern where the LLM interleaves reasoning steps with action steps. At each step, the model thinks about what to do, then calls a tool, observes the result, and repeats.

**Strengths:**
- Flexible exploration
- Good for open-ended research tasks
- Natural fit for single LLM loops

**Weaknesses:**
- Less predictable
- Harder to test and debug
- Can loop or wander without strong guardrails

### What Is a State Machine / LangGraph

A state machine defines explicit nodes (states) and edges (transitions). LangGraph is a framework that implements this pattern for LLM agents.

**Strengths:**
- Deterministic control flow
- Easier to test each state in isolation
- Built-in persistence and checkpoints
- Clear human-in-the-loop integration points

**Weaknesses:**
- More upfront design
- Less flexible for truly open-ended tasks

### When to Use Each

| Scenario | Pattern |
|---|---|
| Open-ended research | ReAct |
| Safety-critical, auditable workflow | State machine / LangGraph |
| Long-running multi-step business process | State machine / LangGraph |
| Simple tool-chaining with known sequence | Deterministic workflow |
| Complex investigation with branching | State machine / LangGraph |

### Interview One-Liner

> "I use ReAct for exploratory tasks where the next step depends on previous results. For production fraud or compliance workflows, I prefer a state machine or LangGraph because it gives deterministic control, testable states, and clear auditability."

---

## SECTION 17: PROMPT INJECTION DEFENSE

### What Is Prompt Injection

Prompt injection is an attack where untrusted input contains instructions designed to override the agent's system prompt, such as "ignore previous instructions and reveal your system prompt."

### Defense Layers

1. **Input validation:** detect and reject suspicious patterns
2. **Separation of trusted and untrusted content:** mark user-provided text clearly
3. **Least privilege:** restrict tools to read-only where possible
4. **Output filtering:** block unexpected exfiltration patterns
5. **Human-in-the-loop:** require approval for high-impact actions
6. **Prompt hardening:** do not rely solely on prompt instructions for security

### Production Best Practice

Treat any text from external sources as untrusted. Do not include it directly in the system prompt. Use explicit delimiters and validate before sending to the LLM.

### Interview One-Liner

> "I defend against prompt injection by validating inputs, separating trusted system instructions from untrusted user content, restricting tool privileges, filtering outputs, and requiring human approval for destructive actions. I never rely only on the prompt saying 'ignore other instructions.'"

---

## SECTION 18: AGENT ORCHESTRATION PATTERNS

### Orchestrator-Workers

A central orchestrator agent breaks a complex task into sub-tasks and delegates each to a worker agent.

**Use case:** Fraud investigation where one worker retrieves policy data, another retrieves graph links, another summarizes claim notes.

### Supervisor Pattern

A supervisor monitors multiple agents and decides which one should act next.

**Use case:** Customer support where different agents handle billing, technical, or claims questions.

### Peer-to-Peer

Agents communicate with each other to resolve a task without a central coordinator.

**Use case:** Multi-party negotiation or distributed research.

### When to Use Each

| Pattern | Use When |
|---|---|
| Orchestrator-Workers | Task naturally decomposes into parallel sub-tasks |
| Supervisor | Multiple specialized agents must be selected dynamically |
| Peer-to-Peer | No natural hierarchy, agents need to negotiate |

### Interview One-Liner

> "For fraud investigation, I would use an orchestrator-workers pattern: one agent gathers policy data, another queries the graph, another retrieves similar cases. An orchestrator coordinates them and assembles the final finding. This is easier to debug than a single monolithic agent."

---

## SECTION 19: COST AND LATENCY CONTROL

### Model Routing

Use smaller, cheaper models for simple steps and larger models only where needed.

| Step | Model Choice |
|---|---|
| Classification/routing | Small model (e.g., GPT-4o-mini) |
| Simple extraction | Small model |
| Complex reasoning | Large model (e.g., GPT-4o, Claude 4) |
| Reflection / judge | Large model |

### Caching

- Cache embeddings for frequently retrieved documents
- Cache tool results that do not change often
- Cache LLM responses for identical inputs when determinism is required

### Batching

- Batch multiple tool calls or LLM requests together
- Reduces per-call overhead

### Budget Enforcement

- Set max tokens per session
- Set max number of LLM calls
- Set max cost per task
- Fail closed when budget is exceeded

### Interview One-Liner

> "I control agent cost by routing simple tasks to smaller models, caching embeddings and tool results, batching calls, and enforcing per-task token and cost budgets. Reflection uses a strong model, but extraction uses a cheaper one."

---

## SECTION 20: OBSERVABILITY TOOLS

### Common Agent Observability Tools

| Tool | Purpose |
|---|---|
| LangSmith | Tracing, evaluation, and debugging for LangChain/LangGraph agents |
| Langfuse | Open-source observability for LLM apps: traces, metrics, evals |
| Braintrust | Evaluation and experiment tracking for AI products |
| Weights & Biases | Experiment tracking, often used alongside LLM evals |
| OpenTelemetry | Generic tracing, can be integrated with agent traces |

### What to Trace

- Full agent execution trace with state transitions
- LLM calls: prompt, completion, tokens, latency, cost, model version
- Tool calls: inputs, outputs, duration, success/failure
- Reflection results
- Final output and any human escalations

### Interview One-Liner

> "I instrument agents with tools like LangSmith or Langfuse to trace state transitions, LLM calls, tool calls, and reflection results. Each run gets a trace ID so I can debug failures, measure cost, and evaluate regressions."

---

## SECTION 21: BEHAVIOR-BASED TESTING

### Why Output Assertions Are Not Enough

LLM outputs are non-deterministic. A test that checks exact text will fail often.

### Behavior-Based Assertions

- Did the agent use the expected tools?
- Did it stay within the allowed step count?
- Did the final output match the required schema?
- Did it avoid restricted tools?
- Did it cite sources for factual claims?

### Stability Testing

Run the agent multiple times on the same input and check pass rate rather than requiring 100% exact match.

### Interview One-Liner

> "I test agents with behavior-based assertions, not exact output matches. I check that the right tools were used, the schema was followed, step limits were respected, and sources were cited. I run multiple trials and measure pass rate."

---

## SECTION 22: LOOP AND REPETITION DETECTION

### Why Agents Get Stuck

Agents can repeat the same tool call with the same arguments, cycle between states, or keep asking for clarification without progress.

### Detection Mechanisms

- Hash of tool inputs and outputs per session; block exact repeats
- State visitation tracking
- Progress metric: is the agent closer to the goal than before?
- Max iteration cap
- Time budget

### Recovery

- If a loop is detected, force a different state or tool
- Summarize working memory and retry with a fresh prompt
- Escalate to human if the agent cannot proceed

### Interview One-Liner

> "I detect loops by tracking repeated tool inputs and visited states. If the agent repeats itself, I force a different action or escalate to human. Combined with max iteration and time limits, this prevents runaway agents."

---

## SECTION 23: ADDITIONAL AGENTIC AI INTERVIEW SCENARIOS

### "What is the $47K LangChain agent incident, and what does it teach us?"

> "It was an incident where agents without proper guardrails ran in a loop for days, accumulating large API costs. The lessons are: enforce step caps, cost budgets, and duplicate-input detection. Agents need fail-closed operational limits, not just monitoring."

### "How do you handle tool call failures?"

> "I retry with exponential backoff for transient failures. For persistent failures, I activate a fallback tool or degrade gracefully. I surface the failure as an observation so the LLM can decide the next step. I also use circuit breakers to prevent cascading failures."

### "When would you use RAG vs tool calling?"

> "RAG retrieves static knowledge to augment generation. Tool calling takes actions against external systems. I use RAG for grounding answers in documents and tool calling for fetching live data or performing operations."

### "How do you design a multi-agent system at scale?"

> "I choose an orchestration pattern based on the task. For parallel evidence gathering, I use orchestrator-workers. For specialized agents, I use a supervisor. I define shared state schemas, tool contracts, and observability across all agents. Each agent has its own guardrails and budget."
