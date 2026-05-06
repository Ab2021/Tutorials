# Advanced AI Engineering Topics
## AIP-C01 – Deep Dive Guide

---

## 🧠 Prompt Caching & Semantic Caching Tradeoffs

When building scalable GenAI applications, caching is critical for cost reduction and latency improvement. There are two primary caching strategies, each serving a different purpose.

### 1. Provider-Level Prompt Caching
Provider-level caching (like Anthropic's Prompt Caching or similar mechanisms in Bedrock) caches the exact prefix of a prompt (usually the system prompt, heavy context, or few-shot examples) directly within the LLM provider's infrastructure.

**Characteristics:**
- **Exact match required** for the cached prefix.
- Reduces Time To First Token (TTFT) significantly.
- Lowers cost for repeated input tokens (often by 50-90%).

### 2. Semantic Caching
Semantic caching (using tools like Redis, Pinecone, or specialized caching layers) stores the *embeddings* of previous user queries and returns a cached response if a new query is semantically similar to an old one.

**Characteristics:**
- **Fuzzy match allowed** (e.g., "How to reset password?" matches "What is the password reset process?").
- Eliminates LLM invocation completely if a match is found.
- Zero API cost for matched queries, near-instant response.

### ⚖️ Tradeoff Matrix

| Feature | Provider Prompt Caching | Semantic Caching |
|---------|-------------------------|------------------|
| **Matching** | Exact Prefix Match | Semantic (Cosine Similarity) |
| **LLM Call** | Required (for suffix & generation) | Bypassed entirely on hit |
| **Cost Reduction** | Moderate (cheaper input tokens) | Very High (no LLM usage) |
| **Latency Reduction** | Moderate (faster TTFT) | Extreme (sub-50ms) |
| **Risk of Hallucination** | None (standard generation) | High (wrong cache hit for nuanced query) |
| **Best For** | Massive system prompts, RAG context | FAQs, repetitive customer queries |

> **Exam Trap:** "Users ask variations of the same 50 questions, and the goal is sub-100ms latency and zero LLM cost." → **Semantic Caching** (NOT Provider Prompt Caching).
> **Exam Trap:** "You have a 50k token system prompt that must be processed for every unique user request." → **Provider Prompt Caching** (NOT Semantic Caching).

---

## ⚙️ KV Cache Management at Scale

### What is the KV Cache?
In auto-regressive LLM generation, the model predicts the next token based on all previous tokens. To avoid recalculating attention scores for past tokens at every step, the Key (K) and Value (V) tensors for past tokens are stored in GPU memory. This is the **KV Cache**.

### The Bottleneck
As sequence length and batch size increase, the KV Cache grows linearly. **At scale, LLM inference is often memory-bandwidth bound, not compute-bound.**

```
VRAM Consumption = Model Weights + KV Cache + Activation Memory
```
If KV Cache exceeds VRAM, batch sizes must shrink, killing throughput.

### Advanced Management Techniques

1. **PagedAttention (vLLM):** 
   - Traditional KV caching pre-allocates contiguous memory blocks, leading to massive fragmentation (up to 60% waste).
   - PagedAttention breaks the KV cache into fixed-size blocks (like virtual memory in an OS).
   - **Result:** Near-zero fragmentation, allowing 2-4x larger batch sizes.

2. **RadixAttention (SGLang):**
   - Retains the KV cache of previous requests in a Radix Tree structure across multiple calls.
   - **Result:** Multi-turn chat and complex agentic workflows reuse KV cache automatically without re-computing context.

3. **KV Cache Quantization:**
   - Compressing the KV cache from FP16 to FP8 or INT4.
   - **Result:** Cuts memory usage by 50-75%, allowing massive context windows with minimal accuracy degradation.

> **Exam Pattern:** "Throughput is bottlenecked by GPU memory fragmentation during long-context generation." → **Implement PagedAttention (vLLM)** (NOT upgrade to larger GPUs, NOT quantize model weights).

---

## ⚡ Speculative Decoding vs Quantization

Both techniques accelerate inference, but they attack the problem from entirely different angles.

### Speculative Decoding (Compute Optimization)
Autoregressive generation is slow because you generate one token at a time. Speculative decoding uses a **Draft Model** (small, fast) to guess the next $K$ tokens, and a **Target Model** (large, slow) to verify them in parallel in a single forward pass.

- **Mechanism:** If the Target Model agrees with the Draft Model, all $K$ tokens are accepted instantly.
- **Accuracy Loss:** **Zero.** The output is mathematically identical to running the Target Model alone.
- **Best for:** Memory-bandwidth bound scenarios (batch size = 1) where compute is underutilized.

### Quantization (Memory Optimization)
Quantization reduces the precision of model weights (e.g., from 16-bit float to 8-bit or 4-bit integers). Techniques include AWQ, GPTQ, and EXL2.

- **Mechanism:** Smaller weights = less memory footprint = faster memory loading (which is the main bottleneck during decoding).
- **Accuracy Loss:** **Slight to Moderate.** Depending on the method and compression ratio.
- **Best for:** Fitting large models on smaller/fewer GPUs, improving multi-user throughput.

| Feature | Speculative Decoding | Quantization (AWQ/GPTQ) |
|---------|----------------------|--------------------------|
| **Primary Goal** | Reduce Latency (TTFT/TPOT) | Reduce VRAM Footprint & Cost |
| **Output Quality** | 100% Identical to base model | Minor degradation |
| **Hardware needed** | Same or slightly more VRAM | Less VRAM required |
| **Complexity** | High (Requires tuned draft model) | Low (Pre-quantized weights) |

---

## 📐 RAG Evaluation (RAGAS + Human Evals)

You cannot improve what you cannot measure. RAG requires a multi-dimensional evaluation approach because a failure can occur in retrieval, generation, or both.

### The RAGAS Framework (Automated Evals)
RAGAS uses "LLM-as-a-judge" to evaluate RAG pipelines without requiring extensive human-annotated ground truth for every query.

**The 4 Core Metrics:**

1. **Context Precision (Retrieval):** Did the retriever rank the most relevant chunks at the top? *(Query vs. Retrieved Context)*
2. **Context Recall (Retrieval):** Did the retriever fetch all the information needed to answer the question? *(Ground Truth vs. Retrieved Context)*
3. **Faithfulness (Generation):** Is the answer entirely derived from the retrieved context, or did the LLM hallucinate? *(Retrieved Context vs. Answer)*
4. **Answer Relevance (Generation):** Does the answer directly address the user's question without rambling? *(Query vs. Answer)*

```python
# Conceptual RAGAS Integration Pipeline
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

# Automated CI/CD Gate
result = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=bedrock_claude_3_sonnet
)

if result['faithfulness'] < 0.90:
    raise Exception("Pipeline blocked: High hallucination risk detected.")
```

### Human Evaluations (The Gold Standard)
While RAGAS is great for CI/CD regression testing, Human Evals are mandatory for:
- **Brand Voice & Tone:** Is the bot polite, empathetic, and compliant with brand guidelines?
- **Nuanced Domain Expertise:** Only a doctor can verify if a retrieved medical guideline is safely applied to a user's symptom query.

**Best Practice:** Combine automated RAGAS (for 100% of commits) with Human-in-the-Loop (HITL) sampling (reviewing 2-5% of production logs).

---

## 💸 Cost Monitoring & Hidden Token Leaks

Cost overruns in GenAI rarely come from a single massive query; they come from "Hidden Token Leaks" compounded over thousands of requests.

### Common Sources of Token Leaks

1. **Unbounded Chat History:**
   - Sending the entire chat history for every turn. Turn 10 costs 10x more than Turn 1.
   - **Fix:** Implement Sliding Window Memory (keep last $N$ turns) or Summarization Memory (LLM summarizes past turns).

2. **Heavy JSON/Tool Schemas:**
   - Every tool description and JSON schema provided in the API call consumes input tokens.
   - **Fix:** Keep tool descriptions concise. If an agent has 50 tools, use a "Retrieval" step to only inject the schemas of the top 3 relevant tools.

3. **Agentic "Thought" Bloat:**
   - ReAct (Reasoning and Acting) agents generate internal thoughts (`Thought: I need to search...`). These thoughts cost output tokens and are fed back as input tokens on the next loop.
   - **Fix:** Enforce concise thought constraints in the system prompt.

### AWS Bedrock Monitoring Strategy

```
1. Tagging: Tag all Bedrock resources by 'Project' and 'Environment'.
2. Logging: Enable Model Invocation Logging to S3.
3. Analysis: Use Amazon Athena to query S3 logs for token usage per session/user.
4. Budgeting: Set AWS Budgets alerts on Bedrock expenditures.
```

> **Exam Pattern:** "Agent costs are growing exponentially with each user interaction in a single session." → **Implement a sliding window for chat history** (NOT switch to a cheaper model).

---

## 🛡️ Agent Guardrails & Infinite Loop Detection

Agents (LLMs with access to tools) are prone to erratic behaviors that standard chat models are not.

### Infinite Loop Detection
A common failure mode for ReAct agents is getting stuck in an action-error loop:
```
Thought: I will query the DB.
Action: QueryDB(sql="SELECT * FOM users") 
Observation: Syntax Error near 'FOM'
Thought: I will query the DB.
Action: QueryDB(sql="SELECT * FOM users")
... (Repeats until tokens or budget runs out)
```

**Mitigation Strategies:**
1. **Hard Limits (`max_iterations`):** Terminate the agent after $N$ tool calls.
2. **Timeout Bounds:** Kill the process if it exceeds $X$ seconds.
3. **Loop Detection Parsers:** Maintain a hash map of recent `(Action, Input)` pairs. If the agent repeats the exact same action and input twice, force an intervention prompt: `"You just tried this and it failed. Try a different approach."`

### Amazon Bedrock Guardrails
Instead of relying purely on system prompt instructions (which can be jailbroken), Bedrock Guardrails provide an independent architectural layer of defense.

**Guardrail Capabilities:**
- **Topic Denial:** Block competitor names, political discussions, or financial advice.
- **Content Filters:** Block toxicity, hate speech, and PII (both inbound and outbound).
- **Word Filters:** Block specific proprietary terms.

```python
# Applying a Guardrail to an Agent Call
response = bedrock_agent_runtime.invoke_agent(
    agentId='AGENT_ID',
    agentAliasId='ALIAS_ID',
    sessionId='SESSION_123',
    inputText=user_input,
    # Guardrails are enforced BEFORE the LLM sees the prompt
    # and AFTER the LLM generates the response, before sending to user.
)
```

---

## 📝 Practice Questions

**Q1:** An application relies on a ReAct agent to query databases and summarize financial reports. Over the last month, API costs have spiked, and logs show the agent frequently times out. Upon inspection, the agent repeatedly executes a SQL query, receives an "Invalid Table" error, and attempts the exact same query again. What is the most robust way to prevent this and optimize costs?

- A. Add a rule to the system prompt instructing the agent to never repeat mistakes.
- B. Implement a maximum iterations limit and inject a runtime interceptor that detects duplicate tool calls and forces a failure or alternative instruction.
- C. Use Bedrock Guardrails to block the word "Invalid Table".
- D. Switch from an autoregressive model to a speculative decoding architecture.

**Answer: B** – System prompts (A) are not foolproof and LLMs can ignore them. Guardrails (C) are for safety/PII, not logical loop breaking. Speculative decoding (D) speeds up generation but doesn't fix logic flaws. B directly breaks the infinite loop.

---

**Q2:** A company has a 100,000-token corporate handbook. They built a Q&A bot that includes the entire handbook in the context window for every user query. Latency is acceptable, but the input token costs are astronomical because the same 100k tokens are sent with every API request. Which technology provides the most cost-effective fix without changing the model's accuracy?

- A. Semantic Caching via Redis
- B. KV Cache Quantization
- C. Provider-Level Prompt Caching
- D. Speculative Decoding

**Answer: C** – Provider-level prompt caching allows the 100k token prefix to be cached on the LLM's server. It dramatically reduces input token costs for massive, static contexts while preserving exact model accuracy. Semantic caching (A) is for caching outputs, not static inputs, and risks missing nuance.

---

**Q3:** A team is evaluating a new RAG pipeline using RAGAS. The automated metrics show high `Context Precision` and high `Faithfulness`, but very low `Answer Relevance`. What does this indicate about the system?

- A. The retriever is pulling completely irrelevant documents.
- B. The LLM is hallucinating facts not found in the documents.
- C. The retriever finds good documents, and the LLM sticks to them, but the final answer does not actually answer the user's specific question.
- D. The KV cache has fragmented, causing context loss.

**Answer: C** – High Context Precision means good documents were found. High Faithfulness means the LLM didn't hallucinate. Low Answer Relevance means the LLM wrote an answer that was derived from the documents, but it missed the point of the user's actual question (e.g., User asks "How much is the fine?", Bot answers "The policy outlines several rules regarding fines," without giving the number).

---

*End of Deep Dive Guide*
