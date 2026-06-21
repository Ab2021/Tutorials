# 🤖 Agentic AI — Deep Dive Interview Prep
### Abhishek Bhardwaj | Solutions Architect ML/AI @ Huge

---

> [!IMPORTANT]
> This is the **#1 priority topic** for your interview at Huge. The JD explicitly calls out: *"agents, agent tools, MCP, A2A, agentic workflows, RAG, semantic routing and caching"* and *"agentic development tools such as Gemini Code Assist, Claude Code, v0, Codex, and Cursor."* Your LangGraph project at Chubb is your anchor — drill everything from it.

---

## SECTION 1: Core Agentic AI Concepts

---

### Q1.1: "What is an AI agent? How does it differ from a chain or a pipeline?"

**Why they're asking**: This is the foundational question. They want to know if you understand the conceptual distinction — many candidates confuse these terms.

**Model Answer**:

The distinction lies in **autonomy and decision-making authority**. A pipeline is a deterministic sequence of steps — data flows through fixed transformations. A chain (like LangChain's LCEL) is similar but parameterized; it composes LLM calls and tools but the flow is pre-determined by the developer. An agent, by contrast, has **the LLM itself deciding the next action** based on observation of the current state.

The canonical definition: an agent is a system where an LLM acts as the **reasoning engine** that (1) perceives its environment through observations, (2) decides which action or tool to invoke based on those observations, and (3) receives feedback from tool execution to iteratively progress toward a goal. The key property is that the LLM's own output determines the control flow — not a hard-coded sequence.

A system becomes truly "agentic" when it satisfies three criteria: **goal persistence** (it pursues a goal across multiple steps), **environmental feedback** (it reads tool/API responses and updates its plan), and **dynamic decision making** (it can choose from multiple actions and even abandon a path if it's not working). My Agentic Data Scientist at Chubb satisfied all three — the LLM read a natural language query, decided whether it needed to query SQL, run Python, or search the knowledge base, executed those tools, and synthesized a narrative answer.

💡 **Key insight to convey**: "I think of the spectrum as: LLM call → chain → reactive agent → autonomous agent → multi-agent system. Where you sit on that spectrum depends on how much decision authority you hand to the model, and that has direct implications for reliability, observability, and cost."

**Cross-questions**:

**Q: "When would you NOT use an agent and stick to a simple chain?"**
Answer: When the task is well-defined, the steps are known upfront, and the cost of non-determinism is too high. For example, a claims classification system where the input is a form and the output is a category — no need for an agent. Agents introduce latency (multiple LLM calls), cost (more tokens), and unpredictability. I apply the rule: if you can write the algorithm yourself, don't use an agent. Use agents when the problem requires dynamic tool selection, multi-step reasoning with unknown intermediate steps, or adaptation based on incomplete information.

**Q: "What failure modes are unique to agents vs chains?"**
Answer: Agents can get stuck in reasoning loops, call tools infinitely, hallucinate tool schemas, or run up massive token costs with no stopping condition. Chains fail predictably — a step either works or doesn't. This is why production agents need max_iterations limits, tool call budgets, and circuit breakers. I implemented a max_steps=15 guard in my LangGraph agent and a fallback node that routes to a graceful error response if iterations exceeded the budget.

---

### Q1.2: "Explain the ReAct pattern. Walk me through a trace."

**Model Answer**:

ReAct (Reason + Act) was introduced in the 2022 Yao et al. paper as a way to interleave **reasoning traces** (the model "thinking out loud") with **action steps** (the model calling a tool or API). Before ReAct, models either reasoned without acting (pure chain-of-thought) or acted without reasoning (just output a tool call). ReAct showed that combining both significantly improves performance on multi-step tasks.

A ReAct trace for "What is the 30-day moving average of claims for policy type Auto in Q1 2025?" looks like:

```
Thought: I need to get claims data filtered by policy_type='Auto' for Q1 2025 (Jan 1 - Mar 31).
Action: sql_executor
Action Input: SELECT date, COUNT(*) as claim_count FROM claims WHERE policy_type='Auto' AND claim_date BETWEEN '2025-01-01' AND '2025-03-31' GROUP BY date ORDER BY date
Observation: [returned 90 rows of date, claim_count]

Thought: I now have the daily counts. I need to compute the 30-day moving average. I'll use the Python REPL for this.
Action: python_repl
Action Input: import pandas as pd; df = pd.DataFrame([...observation...]); df['ma30'] = df['claim_count'].rolling(30).mean(); print(df[['date','ma30']].tail(10).to_string())
Observation: [10 rows of date, ma30]

Thought: I have the moving average. Now I should generate a chart and write a narrative.
Action: chart_generator
...
Final Answer: "The 30-day moving average for Auto claims in Q1 2025 peaked at [X] in late February, with..."
```

The critical insight in ReAct is that the **Thought** step forces the model to articulate its reasoning before acting, which dramatically reduces hallucinated tool calls and schema errors. In my LangGraph implementation, I exposed this via LangSmith traces — every node execution was a traceable span, so I could see exactly where reasoning went wrong and fix prompt issues surgically.

⚠️ **Trap question**: "Isn't ReAct just chain-of-thought with tool calls?"  
Answer: Not exactly. CoT is purely generative — the model reasons but doesn't act. ReAct creates a feedback loop: the action result becomes an observation that updates subsequent reasoning. This is the key difference — it's a closed loop, not an open one. The **Observation** step is what makes it agentic.

---

### Q1.3: "What are planning agents? How do PLAN-and-EXECUTE patterns work?"

**Model Answer**:

Planning agents separate **high-level planning** from **low-level execution**. In a pure ReAct agent, planning and execution are interleaved — the model decides the next step one at a time, which can lead to local optima and inconsistent strategy. A PLAN-and-EXECUTE architecture uses a dedicated planner LLM that first generates a multi-step plan, then an executor agent that implements each step.

The architecture: (1) Planner receives the goal and generates an ordered list of sub-tasks (e.g., "1. Retrieve Q1 revenue data. 2. Compute MoM growth. 3. Compare to industry benchmarks. 4. Generate executive summary"). (2) Each sub-task is dispatched to a specialized executor agent with the relevant tools. (3) Results from each step are passed to the next. (4) A final synthesizer combines outputs.

This is powerful for complex analytical tasks. For my Chubb Agentic Data Scientist, I could redesign it as: a **Query Planner** that decomposes "What drove the spike in fraudulent auto claims in March?" into sub-queries, and specialized **SQL Agent**, **Statistical Analysis Agent**, and **Narrative Agent** subagents. The planner handles the "what to do" problem while executors handle the "how to do it."

💡 **Key insight**: "Planning agents shine when tasks have natural decomposition into independent subtasks. The risk is that the plan becomes stale — if Step 3 reveals something unexpected, a rigid plan can't adapt. That's why in practice I use a hybrid: loose planning with re-planning checkpoints if an execution step returns unexpected results."

---

## SECTION 2: LangGraph Deep Dive

---

### Q2.1: "Walk me through your LangGraph Agentic Data Scientist at Chubb — the StateGraph, nodes, and edges."

**Why they're asking**: This is your flagship project. They want architectural depth, not just a high-level description. This tests whether you truly built it or just described it.

**Model Answer**:

The Agentic Data Scientist was built on LangGraph's `StateGraph`, which models the agent as a directed graph of nodes (functions/LLM calls) connected by edges (deterministic or conditional transitions). Let me walk through the architecture layer by layer.

**State Definition**: First, I defined the typed state schema using Python's `TypedDict`:

```python
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]  # append-only
    query: str
    query_type: str  # "sql", "statistical", "visualization", "narrative"
    sql_result: dict | None
    python_result: str | None
    chart_path: str | None
    iteration_count: int
    final_answer: str | None
    error: str | None
```

The `Annotated[List[BaseMessage], operator.add]` is critical — it tells LangGraph to **append** new messages rather than overwrite the list, which maintains conversation history correctly across node transitions.

**Nodes**: I built four primary nodes:
- `intent_classifier_node`: Takes the user query, classifies it into query_type (sql, statistical, visualization, complex). Uses GPT-4 with a structured output schema.
- `sql_executor_node`: Generates and executes SQL against our data warehouse. Includes schema injection in the prompt and parameterized query validation to prevent injection.
- `python_repl_node`: Executes Python code in a sandboxed environment (RestrictedPython) for statistical analysis and chart generation.
- `synthesizer_node`: Takes all collected results and generates a natural language narrative with citations.

**Edges**: The routing logic used conditional edges:

```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(AgentState)
workflow.add_node("classify_intent", intent_classifier_node)
workflow.add_node("execute_sql", sql_executor_node)
workflow.add_node("run_python", python_repl_node)
workflow.add_node("synthesize", synthesizer_node)
workflow.add_node("error_handler", error_handler_node)

workflow.set_entry_point("classify_intent")

def route_after_classification(state: AgentState) -> str:
    if state["error"]:
        return "error_handler"
    if state["query_type"] == "sql":
        return "execute_sql"
    elif state["query_type"] == "statistical":
        return "execute_sql"  # SQL first, then Python
    elif state["query_type"] == "visualization":
        return "execute_sql"
    return "synthesize"

workflow.add_conditional_edges("classify_intent", route_after_classification, 
    {"execute_sql": "execute_sql", "synthesize": "synthesize", "error_handler": "error_handler"})

# After SQL, route to Python for statistical/visualization types
def route_after_sql(state: AgentState) -> str:
    if state["error"]:
        return "error_handler"
    if state["query_type"] in ["statistical", "visualization"]:
        return "run_python"
    return "synthesize"

workflow.add_conditional_edges("execute_sql", route_after_sql,
    {"run_python": "run_python", "synthesize": "synthesize", "error_handler": "error_handler"})

workflow.add_edge("run_python", "synthesize")
workflow.add_edge("synthesize", END)
workflow.add_edge("error_handler", END)

app = workflow.compile()
```

**Production additions**: I added checkpointing using LangGraph's `MemorySaver` for short-term session persistence, and a `max_iterations` guard at the graph level. The Streamlit UI called `app.stream()` to stream partial state updates to the frontend, giving users live feedback as the agent worked.

**Cross-questions**:

**Q: "How did you handle errors mid-graph? What if SQL execution failed?"**
Answer: Each node returned an updated state with an `error` field populated if something failed. The conditional edge after every substantive node checked for errors and routed to a dedicated `error_handler_node` that would either (a) attempt a retry with a reformulated query (up to 2 retries), or (b) return a graceful degradation response explaining what it could and could not determine. I tracked retry counts in state to prevent infinite loops.

**Q: "How does LangGraph differ from LangChain LCEL? When do you choose which?"**
Answer: LCEL (LangChain Expression Language) is excellent for linear or mildly branching chains — it's a functional composition framework optimized for streaming and async. LangGraph is for stateful, cyclic, multi-step workflows where the flow itself is dynamic. LCEL doesn't handle cycles natively; you can't "loop back" easily. LangGraph gives you a first-class state object that persists across steps and explicit graph topology. My rule: use LCEL for "do these steps in sequence with some branching," use LangGraph for "build a system that decides its own next step."

💡 **Key insight to convey**: "LangGraph essentially gives you the primitives to build state machines with LLMs as the transition logic. This maps cleanly to how I think about complex analytical workflows — you have states (what do I know?), transitions (what should I do next?), and terminals (when am I done?)."

---

### Q2.2: "How did you manage context in your Agentic Data Scientist? This is a critical topic."

**Why they're asking**: Context management is one of the hardest production problems in agentic AI. They want to see if you've grappled with it — not just read about it.

**Model Answer**:

Context management in a multi-step analytical agent is genuinely hard because the context window accumulates: the initial query, the SQL schema injected into the prompt, the SQL result (potentially large), the Python output, and the conversation history for multi-turn interactions. In my system, a complex analytical workflow could easily push 20,000+ tokens before reaching the synthesizer node.

I addressed this at three layers:

**Layer 1 — Selective context injection**: Rather than passing full SQL results to the LLM, I implemented a **result summarizer** that would: (a) for large tabular results (>50 rows), compute summary statistics (mean, std, min, max, percentiles) and pass those + a sample of 5 rows rather than the full table; (b) for multi-query workflows, only carry forward "key findings" from each step as structured summaries, not raw outputs. This kept the synthesizer node's context tight.

**Layer 2 — Message pruning**: In the `messages` state field, I only retained the last N=6 messages for multi-turn conversations, plus always kept the original query pinned at position 0. This is a sliding window approach. For production agents handling long conversations, I later explored summarization-based pruning — when messages exceeded a token threshold, I'd call a lightweight `gpt-3.5-turbo` summarizer to compress older messages into a digest.

**Layer 3 — Separate long-term memory via RAG**: For institutional knowledge — historical reports, business context, previous analyses — I stored these in a FAISS vector store and retrieved only the top-3 relevant chunks via semantic search. This kept company knowledge out of the main context and only surfaced what was relevant to the current query.

```python
# Context budget management in synthesizer node
def synthesizer_node(state: AgentState) -> AgentState:
    # Compute approximate token budget
    sql_result_text = json.dumps(state.get("sql_result", {}))
    if len(sql_result_text) > 8000:  # ~2000 tokens
        # Summarize to key statistics only
        sql_result_text = summarize_sql_result(state["sql_result"])
    
    python_result = state.get("python_result", "")
    if len(python_result) > 4000:
        python_result = python_result[:4000] + "... [truncated]"
    
    # Build context-bounded prompt
    prompt = synthesizer_prompt_template.format(
        query=state["query"],
        sql_findings=sql_result_text,
        python_findings=python_result,
        chart_reference=state.get("chart_path", "None")
    )
    response = llm.invoke(prompt)
    return {**state, "final_answer": response.content}
```

**What I'd do differently today**: I'd use a proper memory layer like **Zep** or **Letta** (formerly MemGPT). Zep provides graph-based memory with automatic fact extraction, entity tracking, and temporal awareness — it solves the "what does the user care about across sessions" problem. Letta's MemGPT architecture is fascinating — it models the LLM like a CPU with a main context (RAM) and external storage (disk), and the model itself decides what to load/unload from memory. For an analytics agent serving returning business analysts who build on previous analyses, this would be transformative.

**Cross-questions**:

**Q: "What is the difference between context stuffing, sliding window, and retrieval-based context management?"**
Answer: Context stuffing: put everything in the context window and rely on the model's long context capability (Claude 200K, Gemini 1M). Works for simple cases but expensive and slow. Sliding window: keep only the last N turns. Loses early context — problematic for long analytical sessions where the initial query's constraints need to be remembered. Retrieval-based: compress history into a vector store and retrieve relevant past context on demand. Most scalable but adds latency and complexity. In practice I used a hybrid: sliding window for recent messages + RAG for long-term knowledge base.

**Q: "How do you handle a business analyst's 10-turn conversation where each turn builds on prior analysis?"**
Answer: This is where session memory becomes critical. I maintained a session-level state in Redis (via LangGraph's `MemorySaver`) keyed by user ID and session ID. Each turn's key findings were extracted and stored as structured summaries (not raw messages), and the agent was prompted with: "Context from this session: [summaries]." This gave continuity without ballooning context. For cross-session memory ("We discussed the Q1 fraud spike last week"), a dedicated memory store with Zep-style entity extraction would be ideal.

---

## SECTION 3: Multi-Agent Systems — MCP, A2A, Supervisor Patterns

---

### Q3.1: "What is the Model Context Protocol (MCP)? How does it work?"

**Why they're asking**: MCP is explicitly in the JD. Many candidates have heard the term but can't explain it architecturally.

**Model Answer**:

MCP (Model Context Protocol) is an **open standard** introduced by Anthropic in November 2024 that defines how LLM applications communicate with external tools, data sources, and services. Think of it as a **USB standard for AI** — before MCP, every AI application had to write bespoke integrations for every tool. MCP standardizes the protocol.

The architecture has three components:
- **MCP Host**: The application running the LLM (e.g., Claude Desktop, your custom agentic app).
- **MCP Client**: A protocol client embedded in the host that manages connections to servers.
- **MCP Server**: A lightweight server that exposes capabilities — **Tools** (functions the LLM can call), **Resources** (data sources like databases or files), and **Prompts** (reusable prompt templates).

The communication protocol uses JSON-RPC 2.0 over stdio or HTTP. An MCP server exposes a manifest of its capabilities, the LLM host discovers them, and during inference the LLM can invoke tools via the protocol without any bespoke integration code.

**A concrete example**: Instead of me writing custom code to connect my LangGraph agent to our FAISS vector store, a BigQuery MCP server, and a Confluence wiki, I would write three MCP servers (or use community-provided ones) and the host application automatically discovers all available tools. The LLM sees a unified tool catalog.

**Why this matters for Huge**: As a Solutions Architect serving multiple Fortune 500 clients, MCP lets you build a **standardized tool ecosystem** once and reuse it across client projects. A Brand Guidelines MCP server could serve both Nike and McDonald's clients. A Analytics MCP server could expose BigQuery and internal dashboards. This dramatically reduces integration overhead per client.

```python
# Example MCP server definition (Python SDK)
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("analytics-tools")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="query_claims_data",
            description="Execute SQL queries against the insurance claims database",
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "The SQL query to execute"},
                    "limit": {"type": "integer", "default": 100}
                },
                "required": ["sql"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "query_claims_data":
        result = execute_bigquery(arguments["sql"], arguments.get("limit", 100))
        return [TextContent(type="text", text=json.dumps(result))]
```

**Cross-questions**:

**Q: "What's the difference between MCP and function calling / tool use?"**
Answer: Function calling (OpenAI) and tool use (Anthropic) are **model-level** protocols — the model is trained to output structured JSON matching a tool schema, and the application parses that and calls the function. It's tightly coupled to the application. MCP is an **infrastructure-level** protocol — it standardizes how the *servers* that provide tools communicate with the *hosts* that use them. MCP sits one layer above function calling: the LLM still uses tool-use internally, but the tool implementations are served via MCP servers rather than inline application code. You can think of it as: function calling is the API, MCP is the service mesh.

**Q: "What is A2A (Agent-to-Agent protocol)?"**
Answer: A2A is Google's open protocol (announced April 2025) for **inter-agent communication**. While MCP handles agent-to-tool communication, A2A handles agent-to-agent communication — how one agent can discover, call, and collaborate with another agent. Each agent exposes an **Agent Card** (a JSON manifest describing its capabilities, authentication, and supported modalities). Agents communicate via task messages, supporting both synchronous and async (SSE) patterns. The key use case is **multi-agent orchestration at enterprise scale**: an orchestrator agent at Huge could discover specialized sub-agents (a Content Agent, an Analytics Agent, a Brand Compliance Agent) via their Agent Cards and delegate work to them over A2A, regardless of what framework each was built with. MCP + A2A together form a complete interoperability layer for the agentic ecosystem.

---

### Q3.2: "Design a multi-agent system for your Agentic Data Scientist at Chubb."

**Model Answer**:

If I were redesigning the Agentic Data Scientist as a multi-agent system today, I'd decompose it into four specialized agents coordinated by a supervisor:

**Architecture**:

```
User Query
    ↓
[Supervisor/Orchestrator Agent] (GPT-4o)
    ├── [Query Planner Agent] — decomposes complex queries into sub-tasks
    ├── [SQL Data Agent] — specialized in data retrieval, schema understanding
    ├── [Statistical Analysis Agent] — runs pandas/scipy/statsmodels code
    ├── [Visualization Agent] — generates charts, picks appropriate chart types
    └── [Narrative Synthesizer Agent] — produces final English explanation
```

**Supervisor pattern**: The supervisor receives the user query, routes to the Query Planner for decomposition, then dispatches sub-tasks to specialized agents via their Agent Cards (if using A2A) or via direct function calls (if using LangGraph's supervisor node). Critically, the supervisor also handles result aggregation and quality checking.

**Why specialized agents vs one monolithic agent**:
1. **Better prompts**: Each agent has a tightly focused system prompt. The SQL agent's system prompt includes full schema context; the statistics agent has mathematical frameworks; the narrative agent has writing guidelines and business context.
2. **Independent optimization**: I can swap the SQL agent's backing LLM for a cheaper model (GPT-3.5 or even text-to-SQL models) without affecting the narrative quality.
3. **Parallel execution**: The visualization agent and statistical analysis agent can run concurrently after SQL retrieval, halving latency.
4. **Fault isolation**: If the visualization agent fails, the narrative agent can still respond without a chart, rather than the whole workflow failing.

💡 **Key insight**: "In practice, the hardest part of multi-agent systems isn't building the agents — it's the **communication protocol between them**. State passing, error propagation, and ensuring agents have exactly the context they need (not more) is where most production issues arise."

---

## SECTION 4: AgentOps — Production Agentic AI

---

### Q4.1: "How do you evaluate a production agentic system? What metrics matter?"

**Why they're asking**: Most people can build an agent demo. Fewer can answer: how do you know if it's working in production?

**Model Answer**:

Evaluating agentic systems is fundamentally different from evaluating a single LLM call, because you have a sequence of decisions, each of which can compound errors. I think about evaluation at four levels:

**Level 1 — Task Success Rate**: Did the agent complete the user's goal? For my analytics agent, this is measured by: (a) task completion rate (% of queries that resulted in a final answer without timeout/error), (b) user satisfaction signals (thumbs up/down, re-query rate — if a user immediately asks the same question differently, the first answer was probably wrong).

**Level 2 — Trajectory Quality**: Even if the final answer is correct, was the path efficient? Metrics: average number of LLM calls per task, average tool calls per task, token consumption per task. An agent that correctly answers in 3 steps is better than one that wanders for 12 steps. I track **trajectory efficiency** as `correct_answer / steps_taken`.

**Level 3 — Tool Use Accuracy**: Are tool calls correct? Did the SQL agent generate valid SQL? Did the Python agent produce executable code? This can be automated: run the generated code/queries and check for errors. I instrumented this in LangSmith — every tool call was a traced span with success/failure status.

**Level 4 — Answer Quality**: Is the final answer faithful to the data? This requires LLM-as-judge evaluation. I used a separate GPT-4 evaluator that received `{question, agent_trajectory, final_answer}` and scored on faithfulness (0-1), completeness (0-1), and clarity (0-1). For production, I ran this on a 5% sample of traffic to keep costs manageable.

**Production dashboards I maintained**:
- Task completion rate (target: >92%)
- P95 latency per task type
- Average cost per query (USD)
- Tool call error rate per tool
- LLM-as-judge quality scores (sampled)
- Context window utilization rate (% of max context used)

⚠️ **Trap question**: "Can you use accuracy metrics from classic ML for agents?"  
Answer: Not directly. Classic ML metrics assume a fixed input-output mapping. Agents have variable trajectories — two different trajectories can produce the same correct final answer. You need trajectory-aware metrics. Also, "correct" is often subjective for open-ended analytical questions. Evaluation frameworks like HELM, AgentBench, and ToolBench define agent-specific benchmarks.

---

### Q4.2: "What are the failure modes in production agents? How did you guard against them?"

**Model Answer**:

Production agent failures cluster into five categories:

**1. Reasoning loops / infinite tool calls**: The agent gets stuck calling the same tool repeatedly with slightly different inputs, making no progress. Guard: `max_iterations` limit in LangGraph (I used 15), tool call deduplication (if the same tool is called with the same inputs twice, break the loop), and a "loop detector" node that checks if the last 3 tool calls are identical.

**2. Context window overflow**: Long analytical sessions exceed the context limit, causing the API to return an error or silently truncate. Guard: proactive token counting before each LLM call (using `tiktoken`), context compression when approaching 80% of the limit, and hard caps on SQL result sizes.

**3. Hallucinated tool schemas**: The LLM confidently calls a tool with parameters that don't exist. Guard: strict schema validation on all tool inputs using Pydantic before execution, and error messages that include the correct schema when validation fails (this helps the agent self-correct).

**4. SQL injection / code execution risks**: The Python REPL and SQL executor are powerful and dangerous. Guard: SQL — use parameterized queries, whitelist allowed SQL keywords (SELECT only, no DROP/DELETE/UPDATE), validate against a schema allowlist. Python — use RestrictedPython (removes dangerous builtins like `exec`, `eval`, `__import__`), run in a Docker container with no network access and read-only filesystem mounts.

**5. LLM output format violations**: The agent's output doesn't match the expected JSON schema, causing downstream parsing failures. Guard: use structured output APIs (OpenAI's `response_format={"type": "json_object"}`, Anthropic's tool_use schema enforcement), and retry with an error-correcting prompt on format failures.

```python
# Example guard: token budget check before LLM call
import tiktoken

def check_token_budget(messages: list, max_tokens: int = 120000) -> bool:
    enc = tiktoken.encoding_for_model("gpt-4")
    total = sum(len(enc.encode(m.get("content", ""))) for m in messages)
    if total > max_tokens * 0.8:
        # Trigger context compression
        return False
    return True
```

💡 **Key insight to convey**: "In my experience, the most common failure in production isn't the big obvious ones — it's subtle reasoning errors that produce plausible-but-wrong answers. This is why LLM-as-judge evaluation on sampled production traffic is non-negotiable. The agent confidently told a business analyst that claims were up 23% when they were actually down 12% — the SQL was correct but the LLM misread the sign convention. Only human review caught it."

---

### Q4.3: "What observability tools do you use for agents? How do you debug a failing agent trace?"

**Model Answer**:

Agent observability requires **distributed tracing** at the LLM level, not just application-level logging. Standard APM tools (Datadog, New Relic) don't understand LLM spans or token counts. Specialized tools I've used:

**LangSmith** (LangChain's platform): Automatically traces every LangGraph node execution as a span. I can see the full input/output at each node, token counts, latency, and the complete message history. For debugging, I can replay a failing trace, edit the agent state at any point, and re-run from that state — this is invaluable. The LangSmith feedback API lets users rate responses inline, which feeds into evaluation datasets.

**Phoenix/Arize**: More ML-observability focused. Good for embedding drift detection (are the embeddings your agent generates for retrieval drifting over time?), LLM output distribution monitoring, and integrations with evaluation frameworks.

**W&B Traces**: Weights & Biases has an agent tracing module. If you're already using W&B for experiment tracking (which I was via MLflow's similar API at Chubb), W&B traces integrate naturally.

**My debug process for a failing agent**:
1. Pull the LangSmith trace for the failing session ID
2. Identify which node the failure originated in (look for the first node with an error or unexpected output)
3. Examine the full input state to that node — was the context correctly populated?
4. Check the LLM's reasoning trace (the Thought steps) — did it misunderstand the task?
5. If it's a tool execution failure, check the tool's raw input and output
6. Fix at the appropriate layer: prompt engineering, schema validation, state initialization, or tool implementation

---

## SECTION 5: Advanced Agentic Patterns (2024–2025)

---

### Q5.1: "What is LLM-as-Judge? How do you use it for agent evaluation?"

**Model Answer**:

LLM-as-judge is a technique where a **separate, often more capable LLM** evaluates the output of another LLM or agent, replacing or supplementing human evaluation. The judge receives the input, the output, and an evaluation rubric, and returns a score and reasoning.

For agent evaluation, I use LLM-as-judge at two levels:
- **Step-level**: After each tool call, a judge evaluates whether the tool's usage was appropriate and the input was correct. This catches reasoning errors early.
- **Final output**: The judge receives `{user_query, full_trajectory, final_answer}` and evaluates on faithfulness (does the answer match the data?), relevance (does it address the question?), and completeness.

The key is the rubric. A vague rubric produces unreliable judgments. I use rubrics with concrete examples:

```python
judge_prompt = """
You are evaluating an AI analytics assistant's response.

User Query: {query}
Agent's SQL query: {sql}
Data retrieved (sample): {data_sample}
Agent's Final Answer: {answer}

Rate on these dimensions (0.0-1.0):
1. Faithfulness: Does every claim in the answer have direct support in the retrieved data? 
   - 1.0: All claims are directly supported
   - 0.5: Most claims supported, minor extrapolation
   - 0.0: Claims contradict the data

2. Completeness: Does the answer fully address the user's question?
   - 1.0: Addresses all aspects of the question
   - 0.5: Addresses main question, misses nuances
   - 0.0: Largely misses the question

Respond as JSON: {{"faithfulness": X.X, "completeness": X.X, "reasoning": "..."}}
"""
```

⚠️ **Trap**: "Isn't LLM-as-judge circular? You're using an LLM to evaluate an LLM."  
Answer: The circularity concern is valid but manageable. The key is using a **different model** (ideally more capable) as the judge, providing detailed rubrics with concrete examples, and validating the judge itself on a golden dataset with human labels. Studies show GPT-4-as-judge has ~80-85% agreement with human evaluators on well-defined rubrics — not perfect, but scalable for production monitoring at scale.

---

### Q5.2: "What is Semantic Routing? How do you implement it?"

**Why they're asking**: The JD explicitly mentions "semantic routing and caching." This is a must-answer concept.

**Model Answer**:

Semantic routing is the technique of using **embedding similarity** to classify an incoming query and route it to the most appropriate handler (agent, RAG pipeline, model, or response template) — rather than keyword-based routing or rule-based if/else logic.

The architecture:
1. Define **route categories** with example queries for each (e.g., "SQL Analytics", "Document Search", "General Conversation", "Chart Generation")
2. Embed the example queries using an embedding model, storing them as prototype vectors
3. For each incoming query, embed it and compute cosine similarity to each prototype
4. Route to the handler with the highest similarity (above a threshold)

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import OpenAI

client = OpenAI()

ROUTES = {
    "sql_analytics": [
        "What were total claims last quarter?",
        "Show me fraud rates by state",
        "Compare Q1 vs Q2 premium volume"
    ],
    "document_search": [
        "What does our fraud policy say about auto claims?",
        "Find all claims with water damage",
        "What are the underwriting guidelines for coastal properties?"
    ],
    "visualization": [
        "Plot the trend of claims over the last 12 months",
        "Show me a heatmap of fraud by region",
        "Create a bar chart comparing claim types"
    ]
}

def embed(texts):
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([r.embedding for r in response.data])

# Pre-compute route prototype embeddings (do this once at startup)
route_prototypes = {route: embed(examples).mean(axis=0) for route, examples in ROUTES.items()}

def semantic_route(query: str, threshold: float = 0.75) -> str:
    query_embedding = embed([query])[0]
    scores = {route: cosine_similarity([query_embedding], [proto])[0][0] 
              for route, proto in route_prototypes.items()}
    best_route = max(scores, key=scores.get)
    if scores[best_route] < threshold:
        return "general_conversation"  # fallback
    return best_route
```

**Semantic caching** is closely related: before hitting the LLM, you embed the incoming query and check if a semantically similar query (cosine similarity > 0.95) has been answered recently. If yes, return the cached response. This dramatically reduces latency and cost for repeated or very similar queries.

For Huge's multi-client architecture: semantic routing lets you route queries to client-specific handlers without maintaining complex routing rules. A query about "brand voice" gets routed to the Brand Knowledge RAG; "campaign performance" routes to the Analytics Agent; "design asset" routes to the Creative Retrieval system. The routing learns from the semantics, not brittle keyword rules.

---

## SECTION 6: Agentic AI for Huge's Business

---

### Q6.1: "How would you architect an agentic system for a client like McDonald's?"

**Model Answer**:

McDonald's represents a fascinating agentic AI opportunity because of their **scale** (40,000+ locations globally), **data richness** (transaction data, loyalty data, marketing touchpoint data), and **marketing complexity** (menu personalization, promotional planning, regional adaptation).

I'd architect a **McDonald's Marketing Intelligence Agent** as follows:

**Core Agents**:
1. **Menu Personalization Agent**: Takes customer profile (loyalty data, order history, location, time of day) and recommends next best item. Uses a fine-tuned recommendation model + LLM for natural language explanation. Runs in real-time at POS and mobile app.
2. **Campaign Optimization Agent**: Reads marketing mix model outputs (my MMM background is directly applicable here), current campaign performance, budget remaining, and proactively recommends reallocation. This is where my Axtria MMM experience maps directly.
3. **Market Intelligence Agent**: Monitors competitor pricing, local events, weather (all factors affecting restaurant traffic), and surfaces insights to regional marketing managers. Uses RAG over news sources, competitor data, and internal reporting.
4. **Content Generation Agent**: Given a promotional brief and brand guidelines (retrieved via RAG from the brand guide vector store), generates localized ad copy, social media content, and email campaigns in the appropriate tone for each market.

**Infrastructure**: Vertex AI Agent Builder for orchestration, BigQuery for the data warehouse, Vertex AI Vector Search for knowledge retrieval, GCP Dataflow for real-time event processing (loyalty events, transaction signals), and a Pub/Sub event bus connecting the agents.

💡 **Key insight**: "The agentic layer is most valuable at the **last mile** of decision-making — connecting insights to actions. McDonald's already has dashboards and reports. The agent's value is in translating 'regional sales are down 8%' into 'here are the specific actions I recommend, with projected impact, that you can approve with one click.'"

---

*End of Agentic AI Deep Dive Document*

---

> [!TIP]
> Study order recommendation: Start with Section 2 (LangGraph deep dive on your actual project), then Section 3 (MCP/A2A since it's explicitly in the JD), then Section 4 (AgentOps for production credibility), then Section 1 (conceptual foundations to anchor everything).
