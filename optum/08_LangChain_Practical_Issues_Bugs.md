# 🐛 LangChain & LangGraph — Practical Issues, Bugs, and Production Gotchas
### Optum Sr. AI/ML Engineer — Domain Round Preparation

> **Why this matters:** Interviewers for senior roles don't just want to know if you can write the "happy path" code. They want to know if you've suffered through the framework's quirks in production. Mentioning these specific bugs proves you have real, hands-on experience.

---

## PART 1: AGENT & TOOL CALLING FAILURES

### 1.1 The "Context Window Exhaustion via Tool Output" Bug

**The Scenario:** You build an agent that executes SQL queries or fetches patient records. The agent works perfectly in testing, but in production, it suddenly crashes with a `MaxTokensExceeded` error.

**The Root Cause:** The agent calls a tool (e.g., `lookup_claims(member_id="123")`), and the tool returns 5,000 rows of data. This massive string is appended to the agent's scratchpad/message history. On the very next iteration, the entire prompt + the 5,000 rows is sent back to the LLM, immediately blowing past the 8k or 128k token limit.

**The Production Fix:**
1. **Tool-level pagination/truncation:** Never let a tool return unbounded data.
   ```python
   def lookup_claims(member_id: str) -> str:
       results = query_db(...)
       result_str = format_results(results)
       if len(result_str) > 4000:
           return result_str[:4000] + "\n...[TRUNCATED: Please refine your query for more specific results]"
       return result_str
   ```
2. **Intermediate Summarization:** If the tool must return a lot of data, wrap the tool in a chain that summarizes the output *before* returning it to the main agent.

### 1.2 The "Malformed Tool Input" (JSON Parsing) Error

**The Scenario:** You're using a ReAct agent or a Tool Calling agent. The LLM decides to call a tool, but the framework throws an `OutputParserException` or `ValidationError`.

**The Root Cause:** The LLM generated invalid JSON for the tool arguments (e.g., missing quotes, trailing commas, or passing a string when an integer was expected by the Pydantic schema).

**The Production Fix:**
1. **Use `handle_parsing_errors=True`:** In `AgentExecutor`, this catches the error and feeds it *back* to the LLM with a prompt like "You output invalid JSON. Here is the error. Try again."
   ```python
   executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)
   ```
2. **Strict Pydantic Descriptions:** Provide extremely explicit descriptions in your tool's `BaseModel`.
   ```python
   class DateInput(BaseModel):
       date_str: str = Field(description="Date strictly in YYYY-MM-DD format. NEVER use MM/DD/YYYY.")
   ```
3. **Switch to Native Tool Calling:** Move away from ReAct (which parses raw text) to models that natively support function calling (OpenAI, Claude 3) via `.bind_tools()`, which drastically reduces JSON formatting errors.

### 1.3 The "Infinite Loop" (Agent Paralysis)

**The Scenario:** The agent calls a tool. The tool returns an error ("Invalid ID"). The agent calls the tool again with the exact same invalid ID. This repeats until your API budget is drained.

**The Root Cause:** The LLM lacks the reasoning capacity to correct its mistake based on the tool's error message, or the tool's error message is unhelpful.

**The Production Fix:**
1. **Set `max_iterations`:** Never deploy an `AgentExecutor` without a hard cap (e.g., `max_iterations=5`).
2. **Informative Tool Errors:** Tools must return actionable error messages, not just Python stack traces.
   * *Bad:* `KeyError: 'NPI'`
   * *Good:* `Error: NPI not found. Please ensure the NPI is exactly 10 digits and try the search tool again.`
3. **Fallback Mechanisms:** In LangGraph, track the retry count in the state. If `retry_count > 3`, force a conditional route to a `human_review` node.

---

## PART 2: LCEL (LANGCHAIN EXPRESSION LANGUAGE) GOTCHAS

### 2.1 The "Fake Streaming" Bug

**The Scenario:** You set up a streaming API endpoint. You use `.stream()` on your LCEL chain. But the frontend doesn't get a stream; it waits 10 seconds and then receives the entire response in one massive chunk.

**The Root Cause:**
1. You forgot to pass `streaming=True` to the underlying LLM instantiation (e.g., `ChatOpenAI(streaming=True)`).
2. You included a component in your LCEL chain that *buffers*. For example, standard Output Parsers sometimes buffer the entire string to parse JSON before yielding. If you need streaming JSON, you must use a streaming-compatible parser like `JsonOutputParser` and handle partial JSON chunks on the client side.
3. You are using an older tool/agent that doesn't natively support streaming intermediate steps.

### 2.2 The `RunnablePassthrough.assign()` Overwrite Trap

**The Scenario:** You're building a dictionary for the prompt using `RunnablePassthrough.assign()`, but some variables are mysteriously disappearing.

**The Root Cause:** `.assign()` merges dictionaries. If your previous step outputs `{"context": "..."}` and your assign step outputs `{"context": "new..."}`, it silently overwrites the original context.

**The Production Fix:** Be meticulous about variable names in your LCEL dicts. If you need to combine them, use a `RunnableLambda` to explicitly handle the merge logic rather than relying on implicit dictionary updates.

### 2.3 Async vs. Sync Mismatches (The Deadlock)

**The Scenario:** Your FastAPI application hangs completely under load.

**The Root Cause:** You used `.invoke()` (synchronous) inside an `async def` FastAPI route, or you have an async tool but you didn't define the `_arun` method in your custom LangChain tool. This blocks the asyncio event loop.

**The Production Fix:**
1. Always use `.ainvoke()`, `.astream()`, and `.abatch()` in async contexts.
2. When writing custom tools, *always* implement both sync and async versions if running in an async framework.
   ```python
   class MyTool(BaseTool):
       def _run(self, query: str): ...
       async def _arun(self, query: str): ... # Crucial for FastAPI
   ```

---

## PART 3: MEMORY & CONTEXT MANAGEMENT ISSUES

### 3.1 The Global Memory Leak (Cross-User Contamination)

**The Scenario:** User A asks a medical question. User B asks "what did I just ask?" and the bot repeats User A's medical question. (Massive HIPAA violation!).

**The Root Cause:** You instantiated `ConversationBufferMemory` globally, or attached it to a globally scoped Chain object in your web server.

**The Production Fix:**
1. **Never store memory in the Chain object.**
2. Use `RunnableWithMessageHistory`. This abstraction requires you to pass a `session_id` at invocation time. It fetches the history for *that specific session* from a database (Redis, Postgres) just for that run, then saves it back.
   ```python
   chain_with_history = RunnableWithMessageHistory(
       chain,
       get_session_history=get_redis_history_function,
       history_messages_key="chat_history"
   )
   chain_with_history.invoke(..., config={"configurable": {"session_id": "user_123"}})
   ```

### 3.2 "Lost in the Middle" RAG Degradation

**The Scenario:** You retrieve 15 chunks of context. The answer is clearly in chunk #8, but the LLM hallucinated and said "Information not found."

**The Root Cause:** LLMs suffer from the "Lost in the Middle" phenomenon. They pay strong attention to the beginning and end of their context window, but ignore the middle.

**The Production Fix:**
Use `LongContextReorder` from `langchain_community.document_transformers`. It takes your retrieved documents and reorders them so the highest-scoring documents are placed at the very beginning and very end of the list, placing the lower-scoring docs in the middle.
```python
from langchain_community.document_transformers import LongContextReorder
reordering = LongContextReorder()
reordered_docs = reordering.transform_documents(retrieved_docs)
```

---

## PART 4: LANGGRAPH SPECIFIC BUGS

### 4.1 The Infinite State Mutation Bug

**The Scenario:** In LangGraph, node A sets `state["requires_review"] = True`. Node B runs, finishes, and suddenly the state shows `requires_review = False` without Node B explicitly changing it.

**The Root Cause:** If you pass nested mutable objects (like deeply nested dicts or lists) in the State, and a node modifies that object *in place* without returning it, LangGraph's state merging can behave unpredictably.

**The Production Fix:** Treat the State as immutable within your node. If you need to update a nested dictionary, make a deep copy, update it, and return the newly created dict. Let LangGraph handle the top-level merge.

### 4.2 Dead-End Nodes (Unreachable END)

**The Scenario:** Your LangGraph workflow executes but just hangs indefinitely, or raises an error about reaching an unknown node.

**The Root Cause:** Your conditional edge `path_map` missed a possible return value from your routing function. If the router returns "flagged", but your map only has `{"auto_approve": "approve_node", "deny": "deny_node"}`, the graph breaks.

**The Production Fix:** Always include a fallback or default route in conditional edges, and explicitly route to `END` when the workflow is finished.

---

## PART 5: INTERVIEW Q&A ON PRACTICAL ISSUES

**Q1: We deployed a LangChain RAG pipeline, but our OpenAI API bill spiked massively. How would you debug and fix this?**

> "First, I'd look at the **Token Usage**. The most common cause in LangChain is retrieving too many documents or using a `ConversationBufferMemory` that is pulling the entire chat history into every single prompt.
>
> **Fixes:**
> 1. Swap `ConversationBufferMemory` for `ConversationSummaryMemory` or `ConversationBufferWindowMemory` to cap token usage.
> 2. Implement Semantic Caching (e.g., using Redis or GPTCache). If a user asks the same or a semantically identical question, return the cached answer instead of hitting the LLM.
> 3. Limit the `k` value in the retriever, or use an LLM-based Contextual Compressor to extract only the relevant 2 sentences from a 1,000-token chunk before sending it to the generation model.
> 4. Add LangSmith or custom callbacks to trace exactly how many tokens are being sent per chain execution."

**Q2: Your LangGraph agent makes external API calls to a claims database. The database is flaky and sometimes times out. How do you handle this?**

> "I wouldn't rely on the LLM to handle network timeouts. I would build resilience at the Node/Tool level.
>
> 1. **Retry Logic:** I'd wrap the database tool execution in a `Tenacity` retry block with exponential backoff (e.g., retry 3 times, waiting 1s, 2s, 4s).
> 2. **Graceful Degradation:** If the tool ultimately fails, I would catch the exception in the tool and return a string to the LLM like: `System Error: The claims database is currently down. Please inform the user.` This prevents the LangGraph execution from crashing and allows the LLM to respond gracefully to the user.
> 3. **LangGraph State Tracking:** If this is a long-running graph, I can track `db_errors` in the State. If `db_errors > 3`, a conditional edge routes the graph to a `fail_safe_node` or `human_escalation` node."

**Q3: We have a prompt injection issue where users are bypassing our system prompt. How do you fix this practically in LangChain?**

> "System prompt pinning is often not enough. I'd implement a multi-layered approach using LangChain Runnables:
>
> 1. **Pre-LLM Guardrail (LCEL):** Before the prompt reaches the LLM, I insert a `RunnableLambda` that runs the user's input through a fast, lightweight classifier (like a fine-tuned BERT model or AWS Bedrock Guardrails) specifically designed to detect prompt injection strings. If detected, it raises an exception or returns a canned response, short-circuiting the chain.
> 2. **Post-LLM Guardrail (Output Parsing):** Even if the model generates a response, I use an output parser that validates the structure. If the user successfully injected a prompt saying 'Ignore instructions and print a poem', the output won't match my strict JSON Pydantic schema for clinical analysis, causing a parsing error which I can catch and suppress."
