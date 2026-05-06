# 🛡️ Agent Guardrails & Infinite Loop Detection (Deep Dive)

Agents represent the transition from passive Generation (chatbots) to active Execution (systems capable of calling external APIs, executing SQL, and interacting with the world). 

With agency comes severe operational risk. Agents are highly susceptible to logic failures, prompt injection, and catastrophic execution loops.

---

## 1. The Catastrophe of the Infinite Loop

The most prominent agent framework is **ReAct (Reasoning and Acting)**. It operates in a `Thought -> Action -> Observation` loop.

Because LLMs are probabilistic, if an `Action` fails, the LLM may become trapped in a logical rut, trying the exact same failed action repeatedly.

**The Loop:**
```text
Thought: I need to get the user's IP. Let me run 'curl ifconfig.me'.
Action: execute_bash(cmd="curl ifconfig.me")
Observation: Error - 'curl' not found in container.
Thought: I need to get the user's IP. Let me run 'curl ifconfig.me'.
Action: execute_bash(cmd="curl ifconfig.me")
... (Repeats until the developer's API budget hits $0)
```

### Architectural Loop Breakers
System prompts (`"Do not repeat actions"`) are probabilistic and will inevitably be ignored by the LLM. You must use deterministic, code-level circuit breakers.

1. **Max Iterations Cutoff (`max_steps`):**
   In LangGraph or LangChain, you hardcode a maximum number of transitions.
   ```python
   from langgraph.graph import StateGraph, END
   
   # Graph builder limits the ReAct loop to 5 maximum steps
   graph = StateGraph(AgentState)
   app = graph.compile(interrupt_before=["tools"])
   
   # Execution with recursion limit
   app.invoke(inputs, {"recursion_limit": 5}) 
   ```

2. **State Hashing (Duplicate Action Detection):**
   The orchestrator maintains a hash map of `hash(Action_Name + Arguments)`. If a duplicate hash is detected, the Python code intercepts the LLM and forces a failure observation: 
   `"SYSTEM EXCEPTION: You have already attempted this exact action and it failed. You MUST use a different tool."`

3. **Fallback Nodes (Plan-and-Solve vs ReAct):**
   To reduce looping, migrate from ReAct to a **Plan-and-Solve** architecture. The LLM generates a complete sequential plan *first*, and then a deterministic execution engine runs the steps. If a step fails, it triggers a designated `Fallback Node` rather than allowing the LLM to freely guess the next move.

---

## 2. Amazon Bedrock Guardrails (Deep Dive)

Rather than building complex regex filters in AWS Lambda to prevent your agent from cursing or leaking PII, **Amazon Bedrock Guardrails** provides a managed, architectural layer of defense.

### The 5 Native Guardrail Policies:
1. **Content Filters:** Managed classifiers blocking Hate, Violence, Toxicity, and Prompt Injection (Jailbreaks).
2. **Denied Topics:** Custom Natural Language topic classifiers (e.g., "Block all questions related to cryptocurrency investment advice").
3. **Word Filters:** Exact-match blocking of profanity or proprietary internal code-names (e.g., "Project Titan").
4. **Sensitive Information (PII):** Regex and ML-based detection of SSNs, Credit Cards, etc. Can be set to **Block** the request or **Mask** the data (e.g., `***-**-1234`).
5. **Contextual Grounding:** Evaluates the response against the RAG retrieved documents in real-time. If the grounding score falls below a threshold, the response is blocked as a hallucination.

### Implementation Architecture
The guardrail sits at the API boundary, evaluated *before* the input hits the LLM, and *after* the output is generated.

```python
import boto3

bedrock_runtime = boto3.client('bedrock-runtime')

response = bedrock_runtime.invoke_model(
    modelId='anthropic.claude-3-5-sonnet-20241022-v2:0',
    body=body,
    # Attach the Guardrail to the invocation
    guardrailIdentifier='guardrail-id-12345',
    guardrailVersion='1',
    trace='ENABLED'  # Allows you to see which specific rule blocked the request
)
```

---

## 3. Tool Execution Security

When an agent executes code or SQL, guardrails aren't enough. The environment itself must be sandboxed.
- **Network Isolation:** Agent tools should run in isolated VPCs with no outbound internet access (to prevent data exfiltration).
- **Timeouts:** Every tool execution must have a strict wall-clock timeout (e.g., 5 seconds) to prevent the agent from executing an infinite `while True` loop in python or a database-locking SQL query.
- **Least Privilege IAM:** The execution role of the Lambda running the agent's code must have strictly scoped permissions (e.g., `s3:GetObject` only on specific buckets, never `s3:*`).

---

## 4. Exam & Interview Practice Questions

**Q1: A developer built an autonomous IT troubleshooting agent using a ReAct architecture. During testing, the agent occasionally gets stuck repeating the exact same failed Linux command over and over, driving up API costs and never solving the user's issue. The developer added "Never repeat commands" to the system prompt, but the issue persists. What is the most deterministic engineering solution?**
- A) Fine-tune the model on a dataset of successful Linux commands.
- B) Implement an application-level state tracker that hashes tool arguments and forcefully injects a failure prompt or terminates the session if a duplicate action is detected.
- C) Use Amazon Bedrock Guardrails to block the word "Error".
- D) Switch from an autoregressive LLM to an embedding model.
**Answer: B.** Prompt engineering is probabilistic and easily ignored by the LLM. An application-level state tracker is deterministic and physically prevents the infinite loop from continuing.

**Q2: A healthcare company is deploying a patient-facing chatbot on AWS Bedrock. Compliance dictates that the chatbot must NEVER ingest a patient's Social Security Number, even if the patient accidentally types it into the chat interface. Which architectural component ensures this compliance with the least operational overhead?**
- A) An AWS Lambda function running a custom regex script triggered by an API Gateway.
- B) A Bedrock Guardrail configured with Sensitive Information (PII) redaction rules applied to the model invocation.
- C) Adding a strict warning in the system prompt instructing the model to forget SSNs.
- D) Using Semantic Caching to filter out sensitive queries.
**Answer: B.** Bedrock Guardrails natively provide managed PII detection and redaction (blocking or masking data before it hits the model), removing the need to build and maintain custom Lambda regex infrastructure.

**Q3: In a LangGraph or LangChain ReAct agent framework, what is the primary purpose of defining a `recursion_limit` or `max_iterations` limit in the orchestrator code?**
- A) To prevent the LLM from generating responses longer than the context window.
- B) To force the agent to use speculative decoding.
- C) To act as a circuit breaker preventing runaway compute costs and infinite loops if the agent fails to reach a final conclusion.
- D) To limit the number of parallel users the system can handle.
**Answer: C.** The recursion limit is the ultimate circuit breaker. If the agent gets lost in a Thought-Action-Observation loop, this hard limit guarantees the execution loop will terminate, protecting your budget and compute resources.
