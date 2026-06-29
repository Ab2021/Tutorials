# LLM & AGENTIC AI — JIO SERVICES PLATFORM (Interview Deep Dive)
> Source: ai_ref_rpojects.txt — "Smart AI Agent for Jio Services"
> This file covers: LangGraph agentic frameworks, LangFuse evaluation, CLIP fine-tuning, LLM reasoning (CoT, sampling), and production agentic deployment at scale.

---

## SECTION 1: PROJECT OVERVIEW & SOAR NARRATIVE

### 30-Second Pitch
> "At Jio, I built an enterprise agentic AI platform serving millions of users across Jio's service ecosystem — JioMart grocery ordering, cab booking, and multi-service query handling. I developed smart AI agents using LangGraph for reliable, stateful agentic workflows, integrated LangFuse for production evaluation of 10K+ daily queries, and fine-tuned CLIP for visual fashion search, improving recall@1 from 25% to 56%. I also conducted research into LLM reasoning enhancement using Chain-of-Thought and advanced sampling techniques."

### SOAR Narrative

**Situation:** Jio needed to serve its 400M+ user base with AI-powered service agents capable of handling complex, multi-step tasks like grocery ordering (JioMart) and cab booking — tasks that require tool use, external API calls, session state, and graceful failure handling.

**Objective:** Build production-grade AI agents with measurable quality, reliable tool use, and scalable evaluation infrastructure.

**Action:**
- Designed and built agentic workflows on LangGraph StateGraph with conditional routing, tool calling, and state persistence
- Integrated LangFuse as the evaluation and observability layer across all agent execution paths — scoring 10K+ daily queries for quality, helpfulness, and task success
- Built a visual fashion search system by fine-tuning CLIP on domain-specific fashion-text pairs, lifting recall@1 from 25% to 56%
- Researched LLM reasoning capabilities: implemented Chain-of-Thought (CoT) prompting, self-consistency sampling, and temperature/top-p tuning for improved multi-step reasoning

**Result:**
- 10K+ daily agent queries evaluated with automated quality scoring via LangFuse
- 25% → 56% recall@1 improvement in fashion visual search (CLIP fine-tuning)
- Reliable agentic workflows for JioMart grocery ordering and cab booking at scale

---

## SECTION 2: AGENTIC EVALUATION FRAMEWORK — LANGFUSE AT 10K+ DAILY QUERIES

### Why Evaluation Infrastructure Comes First

A common mistake in building LLM agents is optimizing the agent before establishing evaluation infrastructure. At Jio, I built the evaluation framework first:

- **Without evaluation:** you don't know which prompt change helped, which model swap hurt, or which user segment is failing
- **With LangFuse:** every agent execution is traced, scored, and queryable — giving you a scientific basis for every decision

### What LangFuse Provides

| Capability | What It Captures |
|---|---|
| Generation Tracing | Every LLM call: prompt, completion, model, tokens, cost, latency |
| Session Traces | All LLM and tool calls grouped by agent run (trace ID) |
| Automated Scoring | Custom scoring functions evaluated per generation |
| Human Annotation | Human raters can score agent outputs in the LangFuse UI |
| Dataset Management | Store, version, and evaluate over canonical test sets |
| Experiment Tracking | Compare prompt A vs prompt B over the same test set |

### Evaluation Dimensions for Jio Agents

| Dimension | Definition | How Measured |
|---|---|---|
| Task Success | Did the agent complete the requested task (order placed, cab booked)? | Binary, tool call verification |
| Helpfulness | Was the agent's response actionable and relevant? | LLM-as-judge (1-5) |
| Completeness | Did the agent address all parts of the user's request? | LLM-as-judge (1-5) |
| Trajectory Efficiency | Did the agent use the minimum number of steps? | Step count vs optimal path |
| Faithfulness | Are factual claims grounded in tool outputs? | Citation verification |
| Graceful Failure | Did the agent degrade gracefully when a tool failed? | Failure injection tests |

### Evaluation CI/CD for Agents

Every prompt change or model update at Jio went through an evaluation gate:

1. Load the canonical test set (500+ diverse representative queries across all Jio services)
2. Run all queries through the new agent version
3. LangFuse scores each generation across all dimensions automatically
4. Compare scores vs. the baseline (previous version)
5. If any dimension regresses by >5%: block deployment
6. If all pass: promote to staging, then production

This prevents the classic failure mode: an agent that "seems to work" on manual testing but fails on edge cases at scale.

**Interview One-Liner:**
> "I treat agent evaluation as a first-class engineering problem. At Jio, I built a LangFuse evaluation pipeline scoring 10K+ daily queries across task success, helpfulness, completeness, and trajectory efficiency. Every prompt change goes through an automated evaluation gate — if quality regresses, deployment is blocked."

---

## SECTION 3: LANGGRAPH — AGENTIC WORKFLOW DESIGN FOR JIO SERVICES

### Why LangGraph over Simple LangChain Chains

A standard LangChain chain executes steps in a fixed linear sequence. Jio's use cases require:
- Conditional routing (grocery order vs cab booking vs service inquiry)
- Multi-turn conversation with memory across turns
- Tool failure handling and graceful fallback
- Human-in-the-loop for order confirmation

LangGraph models the agent as an explicit directed state graph with typed nodes and conditional edges — the right abstraction for complex, branching service workflows.

### JioMart Grocery Ordering Agent — State Machine Design

**States (Nodes):**

| Node | Responsibility |
|---|---|
| INTENT_CLASSIFY | Classify user intent: grocery, cab, inquiry, complaint |
| CLARIFY | Ask follow-up if intent is ambiguous |
| PRODUCT_SEARCH | Call JioMart product search API with user's query |
| CART_BUILD | Add selected items to cart, apply offers |
| ADDRESS_CONFIRM | Retrieve or confirm delivery address |
| PAYMENT | Select payment method, initiate payment |
| ORDER_CONFIRM | Confirm and place order, return order ID |
| FAILURE_RECOVERY | LLM reformulates intent if a step fails |
| END | Return final response to user |

**Conditional Edges:**
- After INTENT_CLASSIFY: route to PRODUCT_SEARCH (grocery) or CAB_BOOKING subgraph or INQUIRY handler
- After PRODUCT_SEARCH: if results empty → CLARIFY; if results found → CART_BUILD
- After PAYMENT: if payment succeeds → ORDER_CONFIRM; if fails → FAILURE_RECOVERY

**Why State Machine over ReAct for this use case:**
- ReAct makes one decision at a time — can get stuck in unproductive loops
- Service workflows have a known optimal path — the state machine enforces it
- Each state is independently testable: I can unit test the PRODUCT_SEARCH node without running the full ordering flow
- Built-in persistence: if the user leaves and returns mid-order, LangGraph checkpointing restores state

### Cab Booking Agent — Key Design Decisions

**Challenge:** Cab booking requires real-time availability, dynamic pricing, and driver ETA — all from live APIs. A single API failure must not crash the agent.

**Solution — Tool Wrapping with Fallback:**
- Every tool call wrapped with error classification: transient (retry) vs. permanent (fallback)
- Transient: exponential backoff retry (3 attempts)
- Permanent: LLM reformulates request (different pickup phrasing, different time window) and retries
- If all reformulations fail: escalate to human customer service endpoint

**Interview One-Liner:**
> "I built the Jio service agents on LangGraph StateGraph with explicit states for each step of the booking flow, conditional routing based on intent classification, and a failure recovery layer that distinguishes transient errors (retry) from semantic failures (LLM intent reformulation). Every state is independently testable and the graph persists across user sessions."

---

## SECTION 4: CHAIN-OF-THOUGHT (COT) REASONING — RESEARCH & IMPLEMENTATION

### What Chain-of-Thought Prompting Is

Standard prompting: User query → LLM → Answer

Chain-of-Thought: User query → LLM → Step-by-step reasoning → Answer

The key insight: forcing the model to generate its reasoning steps before the answer drastically improves performance on complex tasks (math, multi-hop reasoning, planning).

### Types of CoT

| Variant | Description | Use Case |
|---|---|---|
| Zero-Shot CoT | Add "Let's think step by step" to the prompt | Simple tasks, no examples available |
| Few-Shot CoT | Provide examples of (question + reasoning + answer) | Complex tasks, consistent format needed |
| Self-Consistency CoT | Generate N reasoning paths, take majority vote answer | High-stakes tasks, reduce random errors |
| Tree of Thought (ToT) | Explore multiple reasoning branches, backtrack | Complex planning, search problems |
| Least-to-Most | Decompose complex problem into sub-problems, solve in order | Multi-step service workflows |

### CoT for Jio Service Agents

**Where CoT was applied at Jio:**
- Multi-constraint cab booking: "Book a cab that can fit 5 people, is AC, and costs under ₹500 for 15km" → CoT decomposes into constraints, evaluates options step by step
- Complex grocery order disambiguation: "Get me stuff for biryani for 10 people" → CoT reasons about quantities, ingredients, and appropriate product SKUs
- Customer complaint resolution: multi-hop reasoning across order history, refund policy, and escalation rules

**Why CoT improved performance:**
- LLMs are better at reasoning when they "show their work" — the reasoning in the context helps the model keep track of constraints
- Self-consistency (sampling N paths, voting) eliminates random reasoning errors without requiring human review

### Self-Consistency Sampling

Instead of one CoT path, generate K independent reasoning paths with temperature > 0, then aggregate:

- For classification tasks: majority vote across K outputs
- For generation tasks: select the most commonly occurring final answer
- For complex tasks: use an LLM judge to select the best reasoning chain

**Why self-consistency is better than just lowering temperature:**
- Low temperature (greedy) is deterministic but can be stuck in a bad reasoning path
- High temperature (diverse sampling) explores more paths
- Self-consistency combines exploration and reliability: explore K paths, then aggregate to a reliable answer

**Interview One-Liner:**
> "I used Chain-of-Thought prompting for complex multi-constraint service requests at Jio — cab booking with multiple constraints, grocery planning for large groups. I implemented self-consistency sampling: generate K reasoning paths, take the majority vote. This improved task success rate without requiring human review or additional training."

---

## SECTION 5: LLM SAMPLING — TEMPERATURE, TOP-P, TOP-K

### Why Sampling Parameters Matter for Agentic Systems

LLM output is a probability distribution over the vocabulary. Sampling parameters control how the model draws from that distribution:

| Parameter | What It Controls | Effect |
|---|---|---|
| Temperature | Sharpness of the distribution | Low = deterministic; High = diverse/creative |
| Top-K | Only sample from the K most likely tokens | Hard cutoff on vocabulary |
| Top-P (Nucleus) | Sample from the smallest set of tokens with cumulative probability ≥ P | Adaptive cutoff |
| Top-A | Like top-p but absolute probability floor | Less common |

### Temperature in Practice

**Temperature = 0 (greedy decoding):**
- Always pick the highest probability token
- Fully deterministic
- Use when: tool selection, structured output, classification
- Risk: can get stuck in repetitive patterns

**Temperature = 0.3–0.7 (balanced):**
- Some diversity, mostly coherent
- Use when: conversational responses, explanations
- Best for most Jio service agent responses

**Temperature = 1.0+ (creative):**
- High diversity, may be incoherent
- Use when: brainstorming, creative tasks
- Dangerous for: agentic tool selection, structured output

### Top-P (Nucleus Sampling) vs Top-K

**Top-K:** always samples from exactly K tokens regardless of their probabilities. Problem: if the model is very confident (one token has 99% probability), top-K=50 still considers 49 unlikely tokens.

**Top-P (nucleus):** dynamically adjusts the vocabulary size. If one token has 99% probability, top-p=0.9 only includes that one token. If probabilities are spread out, it includes more tokens. This is more adaptive than top-K.

**Best practice:** Use Top-P = 0.9 + Temperature = 0.3–0.5 for most production agentic systems.

### Sampling for Agentic Tool Selection

Tool selection is the most critical step in an agentic system. A wrong tool call means wasted compute, API costs, or agent failure.

**Recommended settings for tool selection:**
- Temperature: 0 or close to 0 — you want deterministic, reliable tool selection
- Structured output enforcement (JSON mode) — the tool selection schema must be followed exactly
- Do NOT use high temperature for tool calls — diversity is the enemy of reliability in agentic tool use

**Interview One-Liner:**
> "I tune sampling parameters per step in the agent pipeline. Tool selection uses temperature near zero and structured output enforcement — reliability beats diversity here. For conversational responses, I use temperature 0.4–0.7 for natural output. For reasoning where I want diversity (self-consistency), I use higher temperature and aggregate over N samples."

---

## SECTION 6: CLIP FINE-TUNING — VISUAL FASHION SEARCH

### What CLIP Is

CLIP (Contrastive Language-Image Pre-training) is a model trained by OpenAI on 400M image-text pairs from the internet. It learns a shared embedding space where images and text descriptions are aligned — similar images and their descriptions are close in the embedding space.

**Architecture:**
- Image Encoder: Vision Transformer (ViT) that encodes images into a vector
- Text Encoder: Transformer that encodes text descriptions into a vector
- Training objective: contrastive loss — push (image, matching caption) pairs together, push (image, non-matching) pairs apart

**Why CLIP enables zero-shot classification:**
- Text query "red dress" → embed query → find closest image embeddings → retrieve matching images
- The shared embedding space means text queries can retrieve images without task-specific training

### Why Fine-Tuning Was Needed (25% → 56% Recall@1)

**Problem with vanilla CLIP for fashion:**
- CLIP was trained on general internet images and captions — not fashion-specific data
- Fashion vocabulary (neckline styles, fabric types, silhouettes, Indian ethnic terms like "kurta", "lehenga") is underrepresented in CLIP's training data
- Recall@1 of 25% means only 1 in 4 top-returned results matches the user's intent

**Fine-tuning approach:**
- Dataset: fashion product catalog pairs — (product image, product description/title)
- Training objective: contrastive loss on fashion-specific pairs
- Negative sampling strategy: in-batch negatives (other products in the same batch) + hard negatives (visually similar but wrong category)
- Result: 25% → 56% recall@1 — more than double the precision of the search

### CLIP Fine-Tuning Technical Details

**What changes during fine-tuning:**
- Both the image encoder and text encoder are updated
- The shared projection heads (mapping encoder outputs to the joint embedding space) are also fine-tuned
- Learning rate is much lower than pre-training: the pre-trained weights are valuable; we're adapting, not overwriting

**Evaluation metrics for visual search:**
- Recall@K: what fraction of correct images appear in the top K results
- Precision@K: what fraction of top K results are correct
- MRR (Mean Reciprocal Rank): average of 1/rank of the first correct result
- For product search: Recall@1 is the most business-relevant (does the top result match what the user wanted?)

**Hard Negative Mining:**
- Random negatives (any other item in the batch) are too easy — the model trivially separates "red dress" from "smartphone"
- Hard negatives: items that are visually similar but semantically different — "red dress" vs "red skirt"
- Hard negative mining forces the model to learn fine-grained distinctions that matter for fashion search

**Interview One-Liner:**
> "I fine-tuned CLIP on Jio's fashion product catalog using contrastive loss with hard negative mining. Fashion vocabulary is underrepresented in standard CLIP training, so domain-specific fine-tuning was essential. Result: recall@1 went from 25% to 56% — the top returned result was correct more than half the time, up from 1 in 4."

---

## SECTION 7: PRODUCTION ARCHITECTURE — JIO AGENTIC AI PLATFORM

### End-to-End Architecture

```
User Request (voice/text)
    → Intent Classification (small LLM, temperature ~0)
    → Route to: JioMart Agent | Cab Agent | Inquiry Handler
    → LangGraph StateGraph execution
        → Tool calls: JioMart API, Cab API, Product Search
        → Redis-backed session memory (multi-turn)
        → LLM-powered error recovery on failures
    → LangFuse traces every LLM call and tool call
    → Response to user
    → LangFuse automated quality scoring (async)
```

### Tool Safety and Idempotency

At Jio's scale (millions of users), a duplicated tool call means:
- Double order placed for the same grocery cart
- Two cabs booked for the same ride

**Idempotency pattern:**
- Every tool call includes an idempotency key (session_id + step_id + timestamp hash)
- The backend API deduplicates on this key — second call with same key is a no-op
- This makes retries safe for transient failures

### FastAPI Backend for Agents

- All agent endpoints exposed as REST APIs via FastAPI
- Async endpoints — LangGraph agent runs in async context, WebSocket or long-polling for streaming responses
- Request validation: Pydantic schemas ensure well-formed inputs before reaching the agent
- Auth: JWT tokens validate user identity; session state keyed by user_id

### Docker and Azure Deployment

- Each agent service containerized in Docker: agent code + LangGraph + FastAPI + dependencies
- Azure Container Apps or Azure Kubernetes Service for orchestration
- Autoscaling: HPA (Horizontal Pod Autoscaler) scales agent pods based on request queue depth
- Azure Cognitive Services for speech-to-text (voice queries)

**Interview One-Liner:**
> "The Jio agentic platform runs LangGraph agents behind a FastAPI backend, containerized in Docker on Azure. Every agent is stateless in the pod but uses Redis for session memory — so any pod can serve any user without losing conversation context. Tools are designed with idempotency keys so retries never cause duplicate orders."

---

## SECTION 8: LLM REASONING RESEARCH — METHODS & FINDINGS

### Sampling Research for Reasoning Improvement

At Jio, I researched how different sampling strategies affect LLM reasoning quality on service-domain tasks:

**Key findings from research:**

| Method | Reasoning Improvement | Compute Cost | When to Use |
|---|---|---|---|
| Greedy (T=0) | Baseline | 1x | Simple classification, tool selection |
| Temperature sampling | Marginal | 1x | Conversational responses |
| Few-shot CoT | +15-25% on complex tasks | 1x (longer prompt) | Multi-constraint tasks |
| Self-consistency (K=10) | +20-35% on complex tasks | 10x | High-stakes decisions |
| Tree of Thought | +30-40% on planning | 20-50x | Complex planning only |

**Practical recommendation:** Self-consistency with K=5-10 gives most of the benefit at moderate cost. Tree of Thought is too expensive for production except for the most complex tasks.

### Chain-of-Thought Prompt Engineering Findings

**What makes a good CoT example:**
- The reasoning steps should be explicit, not implicit
- Wrong: "The answer is X because of constraints"
- Right: "Step 1: Check constraint A. Step 2: Check constraint B. Step 3: Given A and B, the only valid option is X."
- Diversity of examples: include different reasoning patterns, not just one template

**Domain-specific CoT:**
- Generic CoT examples (from general datasets) help less than domain-specific examples
- For Jio service tasks: CoT examples should be based on actual service workflows
- Including negative examples ("What NOT to do") improved agent reliability

---

## SECTION 9: INTERVIEW Q&A — LEAD AI ENGINEER LEVEL

### Q: Walk me through the Jio agentic AI architecture

> "The Jio platform routes user requests through an intent classifier that determines whether the user wants grocery ordering, cab booking, or a service inquiry. Based on the intent, LangGraph routes to the appropriate agent — each modeled as an explicit state machine with defined states, tool contracts, and failure handling. Every LLM call and tool call is traced via LangFuse, which also runs automated quality scoring. Session memory is Redis-backed so conversations persist across turns and pod restarts. The backend is FastAPI on Azure Container Apps, autoscaling via HPA."

### Q: How did you evaluate 10K+ daily agent queries at scale?

> "I built an evaluation pipeline in LangFuse that scores every agent execution automatically. The evaluation runs asynchronously so it doesn't add latency to the user response. Scoring dimensions include task success, helpfulness, trajectory efficiency, and faithfulness to tool outputs. Human annotators periodically review LangFuse traces and add annotations — this human feedback catches systematic judge errors. Every prompt change goes through a CI/CD evaluation gate on a fixed canonical test set before deployment."

### Q: How does CLIP work and why did you fine-tune it?

> "CLIP learns a shared embedding space for images and text through contrastive training on 400M image-text pairs. In this space, an image of a red dress and the text 'red dress' are close together. This enables text-based image search without task-specific labels. We fine-tuned it because vanilla CLIP was trained on general internet data — fashion vocabulary like Indian ethnic terms, specific neckline styles, and fabric types was underrepresented. Fine-tuning on our product catalog with hard negative mining taught CLIP to distinguish visually similar items in our domain, lifting recall@1 from 25% to 56%."

### Q: What is Chain-of-Thought reasoning and when should you use it?

> "CoT prompting forces the LLM to generate reasoning steps before the final answer. This dramatically improves performance on complex tasks — multi-constraint decisions, multi-hop reasoning, planning. I implemented self-consistency CoT: generate K independent reasoning paths with non-zero temperature, then aggregate by majority vote. This reduces random reasoning errors without requiring human intervention. The compute cost scales linearly with K, so I set K based on task criticality — K=10 for order confirmation, K=3 for conversational responses."

### Q: What is Top-P sampling and how does it differ from Top-K?

> "Top-K always samples from exactly K tokens regardless of their probability distribution. Top-P (nucleus sampling) dynamically adjusts: it samples from the smallest set of tokens whose cumulative probability exceeds P. When the model is very confident (one token has 90% probability), Top-P=0.9 only samples that one token. When uncertain, it considers more options. Top-P is more adaptive and generally preferred for production. For tool selection in agents, I use temperature near zero — reliability over diversity. For conversational text, I use Top-P=0.9 with temperature 0.4–0.6."

### Q: How do you make an LLM agent reliable in production?

> "Five layers: First, structured output enforcement — tool calls must match a JSON schema, validated before execution. Second, idempotency keys on every tool call — safe retries without duplicate orders. Third, error classification — transient failures (retry) vs. semantic failures (LLM reformulates intent). Fourth, loop detection — track visited states and tool calls, enforce max iteration caps. Fifth, LangFuse observability — trace every execution, score quality, alert on regression. Without all five, agents work in demos but fail in production."

### Q: What was your biggest challenge in the Jio agentic platform?

> "Context management across multi-turn grocery ordering conversations. Users would say 'add that tomato ketchup I bought last week' — the agent needs to know which ketchup, from which order, at what quantity. I solved this by persisting conversation history in Redis, and building a context retrieval step that pulls relevant order history before the main reasoning step. Rather than cramming all history into the context window (which would exceed token limits), I retrieve only the last 3 relevant orders and summarize them. This keeps the context window manageable while enabling natural multi-turn ordering."

---

## SECTION 10: KEY FACTS TO MEMORIZE

| Fact | Detail |
|---|---|
| Daily queries evaluated | 10K+ daily use-case-specific queries |
| CLIP improvement | Recall@1: 25% → 56% (more than doubled) |
| Agent framework | LangGraph StateGraph with conditional routing |
| Evaluation platform | LangFuse (generation tracing + automated scoring) |
| Services covered | JioMart grocery ordering, cab booking |
| Tech stack | Python, Azure, LLM, FastAPI, Docker, LangGraph, LangFuse, HuggingFace, Distributed Systems |
| Reasoning methods | Chain-of-Thought, Self-Consistency, Temperature/Top-P sampling research |
| CLIP fine-tuning | Contrastive loss, hard negative mining, fashion domain adaptation |
