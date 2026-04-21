# 🧠 DDS GRIND — Advanced LLM & GenAI System Design (Questions 51-75)
### Deep Dive into Scaling Laws, Multi-Modality, and Production GenAI

> Flipkart is heavily investing in GenAI (e.g., Triksha, automated cataloging). These questions cover the absolute cutting-edge of production LLM systems.

---

## ═══════════════════════════════════════
## SECTION G: ADVANCED RAG & INFORMATION RETRIEVAL
## ═══════════════════════════════════════

### Q51: Your RAG system retrieves exactly the right context, but the LLM still hallucinates the answer. What are the 5 architectural fixes for this?
**Expected Answer:**
This is the "Parametric Memory Override" problem. The model relies on pre-training instead of context.
1. **Prompt Grounding Strictness:** Use adversarial prompting: `"You will be penalized if you use outside knowledge. If the answer is not contained in the text, output exactly: 'INSUFFICIENT_DATA'."`
2. **Context Formatting:** Use XML tags to strongly separate context from query. `<context> ... </context>`
3. **Chain of Verification (CoVe):** Prompt the model to first extract the facts from the context, then draft an answer, then verify the draft against the facts, then output the final answer.
4. **Post-Hoc Entailment Checking:** Use a smaller, fast NLI model (like DeBERTa-v3-large-mnli) to check if the generated answer is entailed by the retrieved context. If not, trigger a retry with a higher temperature or fallback to rule-based response.
5. **Supervised Fine-Tuning (SFT) for Groundedness:** Fine-tune the LLM on examples where the correct behavior is to refuse to answer when the context is insufficient (Negative examples).

### Q52: Design a multi-modal RAG system for Flipkart product search. Users can upload an image (e.g., a dress) and add text ("I want this but in blue").
**Expected Answer:**
**Architecture:**
```
User Query: [Image A] + Text: "in blue"
       │
       ▼
Query Embedding Generation (Multi-modal Encoder - CLIP/ALIGN)
  - Image Encoder (ViT) processes [Image A] -> V_img
  - Text Encoder processes "in blue" -> V_text
  - Fusion Layer: Combine V_img and V_text (e.g., Cross-attention or simple addition + LayerNorm) -> V_query
       │
       ▼
Vector Search (FAISS/Milvus)
  - Catalog is pre-indexed using the same Multi-modal Encoder
  - Retrieve top-100 nearest products based on Cosine(V_query, V_catalog)
       │
       ▼
Re-ranking (Vision-Language Model - VLM)
  - Input: User Image, User Text, Retrieved Catalog Image, Retrieved Catalog Text
  - VLM evaluates: "Does this catalog item match the image style AND the text modification?"
  - Output: Relevance score for final top-10 sorting.
```
**Key Challenge:** The fusion of modalities. If "in blue" is dominant, you might get any blue dress. If the image is dominant, you might get the exact dress in its original color.
**Solution:** Composed Image Retrieval (CIR) models (like BLIP-2 or specialized CLIP fine-tuning) that explicitly train for text-guided image search.

### Q53: How do you handle document updates in a production Vector Database? (CRUD operations for RAG)
**Expected Answer:**
Standard RAG assumes a static KB. In production (like Flipkart return policies), docs change.
1. **Document ID tracking:** Keep a relational DB mapping `Doc_ID -> [Chunk_ID_1, Chunk_ID_2]`.
2. **Update flow (Soft Delete + Insert):**
   - When Doc A is updated to Doc A_v2.
   - Look up old Chunk_IDs for Doc A.
   - Delete (or mark as deleted/inactive in metadata) old Chunk_IDs in the Vector DB.
   - Chunk and embed Doc A_v2.
   - Insert new Chunk_IDs into Vector DB.
   - Update relational DB mapping.
3. **Metadata Filtering:** Use `{"status": "active"}` in the vector search payload so vector DB ignores soft-deleted vectors.
4. **Index Rebuilding:** HNSW indexes degrade with too many deletes. Schedule weekly/monthly full index rebuilds offline, then hot-swap.

### Q54: Explain the "Lost in the Middle" phenomenon and how to architect around it.
**Expected Answer:**
**Phenomenon:** LLMs are good at extracting information at the beginning and end of their context window, but performance drastically drops for information located in the middle.
**Architectural Fixes:**
1. **Context Reordering:** After retrieving Top-K documents, rank them by relevance. Place the most highly relevant documents at the very beginning AND the very end of the prompt context window. The moderately relevant ones go in the middle.
2. **Map-Reduce summarization:** Instead of dumping 20 docs into context, map an extraction prompt over each doc individually, then reduce (synthesize) the extracted facts.
3. **Strict Top-K limits:** Empirically find where the model degrades (e.g., >8 docs) and hard-cap the retrieval, prioritizing precision over recall.

---

## ═══════════════════════════════════════
## SECTION H: AGENTS & REASONING (System 2)
## ═══════════════════════════════════════

### Q55: Design a "Self-Reflective" Agent for automated code generation/SQL generation.
**Expected Answer:**
**Pattern: Reflection/Critique Loop**
1. **Generator Node:** Takes user request (e.g., "SQL for yesterday's fraud rate") and schema. Writes SQL.
2. **Execution Node (Sandbox):** Runs the SQL against a read-only replica or dry-run validator.
   - If success: Returns data.
   - If error: Captures precise error trace (e.g., `Syntax error at line 3: column 'fraud' doesn't exist`).
3. **Reflection Node (Critic):** Takes the Original Request, the generated SQL, and the Error Trace.
   - Generates a critique: "The column is named `is_fraud`, not `fraud`."
4. **Loop:** Passes the critique back to Generator. Repeat until success or max_retries (e.g., 3).
**Evaluation:** Improves pass@1 from ~40% to pass@3 of ~80%.

### Q56: What is a "Semantic Router" and why is it essential for cost-effective Agent systems?
**Expected Answer:**
Instead of sending every user query to GPT-4 to decide what to do (expensive, slow), a Semantic Router uses an Embedding model + Vector DB (or fast classifier) to route queries.
**How it works:**
1. Embed the user query.
2. Compare against clustered embeddings of known intents (e.g., Order Tracking, Chitchat, Complex Fraud Analysis).
3. If intent == 'Order Tracking' -> Route to simple API (No LLM).
   If intent == 'Chitchat' -> Route to Llama-3-8B (Cheap LLM).
   If intent == 'Complex Analysis' -> Route to GPT-4-Turbo + Tools (Expensive Agent).
**Benefit:** Drastically reduces token costs and latency for easy queries, saving cognitive agents for hard problems.

### Q57: How do you prevent an Agent from getting stuck in an infinite loop?
**Expected Answer:**
Agents doing ReAct or Plan-and-Solve often repeat the same failed tool call.
1. **Max Iterations Hard Cap:** Absolute limit (e.g., 5 steps max).
2. **History Injection & Penalty:** Inject "You have already tried X and it failed with Y. DO NOT try X again."
3. **Loop Detection Heuristic:** Compute hash or string similarity of the `Action` + `Action_Input`. If similarity > 0.95 across 3 consecutive iterations, forcibly terminate the agent and hand off to human.
4. **Tool Fallback:** Design tools to return helpful fallback guidance on error (e.g., instead of `404 Not Found`, tool returns `User not found. Valid IDs start with U-`).

---

## ═══════════════════════════════════════
## SECTION I: LLMOps & SERVING AT SCALE
## ═══════════════════════════════════════

### Q58: Explain PagedAttention and why vLLM is so much faster than naive HuggingFace pipelines.
**Expected Answer:**
**Problem:** In autoregressive generation, the Key-Value (KV) cache grows with every generated token. Naive approaches allocate contiguous memory for max_sequence_length for every request. This leads to massive memory fragmentation and limits batch size.
**PagedAttention (vLLM):** Borrows virtual memory management from OS.
- Divides KV cache into fixed-size blocks (pages), usually 16 tokens/page.
- Block tables map logical token positions to physical blocks in GPU memory.
- Blocks don't need to be contiguous in memory.
- **Benefits:** Eliminates memory fragmentation (waste < 4%), allows massive increase in batch size (more requests per GPU), and enables memory sharing for complex sampling (e.g., beam search can share the prompt's KV cache pages).

### Q59: Design an LLM Gateway for Flipkart's internal teams to access models (OpenAI, Gemini, Llama).
**Expected Answer:**
Instead of teams calling APIs directly, build a centralized gateway.
**Architecture Layers:**
1. **Auth & Quota Management:** Team A gets 1M tokens/day. Team B gets 500k.
2. **Semantic Caching Layer:** (Redis + FAISS). If standard query asked -> return cache (0ms, $0).
3. **Model Routing / Fallback:** 
   - Primary: Azure OpenAI GPT-4.
   - If Rate Limit / 429: Auto-fallback to AWS Bedrock Claude 3.
4. **PII Redaction/Anonymization Guardrail:** Presidio or regex layer scrubs SSN/Phone before leaving Flipkart VPC.
5. **Observability Sync:** Logs all (Prompt, Response, Token usage, Latency) asynchronously to BigQuery for cost tracking and fine-tuning dataset generation.

### Q60: How do you evaluate an LLM's toxicity and safety before moving it to production?
**Expected Answer:**
1. **Red Teaming (Automated):** Use tools like PromptInject or Garak to generate thousands of adversarial prompts (jailbreaks, prompt leaking, harmful content).
2. **Constitutional AI Eval:** Define a "constitution" (e.g., "Do not offer medical advice"). Use a strong LLM as a judge to score violations on a test set.
3. **Output Guardrails (LlamaGuard / NeMo Guardrails):** Wrap the model in production. Run classification on the *input* (is this a jailbreak?) and *output* (is the response toxic?).

### Q61: What are "Continuous Batching" and "In-Flight Batching"?
**Expected Answer:**
**Static Batching:** Waits for N requests, processes them together. Waits for the *longest* sequence to finish before taking new requests. Terrible GPU utilization for short sequences.
**Continuous / In-Flight Batching:**
- Operates at the iteration (token) level, not the request level.
- When Request A finishes generation, its slot in the batch is immediately freed.
- Request C is seamlessly swapped into the batch for the next token generation step.
- Maximizes GPU utilization; crucial for deploying at scale.

---

## ═══════════════════════════════════════
## SECTION J: ADVANCED FINE-TUNING
## ═══════════════════════════════════════

### Q62: Compare LoRA, QLoRA, and DoRA. When to use which?
**Expected Answer:**
- **LoRA (Low-Rank Adaptation):** Freezes base weights $W$. Learns $W + BA$, where $B$ and $A$ are low-rank matrices. Saves massive VRAM (optimizer states only for $A,B$). Use for general fine-tuning.
- **QLoRA:** Quantizes base model to 4-bit (NF4). Further reduces VRAM so you can fine-tune a 70B model on a single/few GPUs. Use when VRAM constrained.
- **DoRA (Weight-Decomposed Low-Rank Adaptation):** Decomposes weights into magnitude and direction. Applies LoRA only to the directional component. Outperforms LoRA by learning more proportionally like full fine-tuning. Use when you need maximum performance but want LoRA efficiency.

### Q63: You are fine-tuning a model for exact JSON extraction from claims. It sometimes hallucinates keys. How to fix?
**Expected Answer:**
1. **Format Enforcement during inference (Constrained Decoding):** Use tools like `Outlines` or `JSON mode`. This modifies the logits before softmax—if the schema expects a `"`, the probability of any other character is set to 0. (Fixes it instantly at inference without more tuning).
2. **SFT Data Mix:** Include negative examples in training data where the input lacks the field, and the target output explicitly maps that key to `null`, teaching the model not to hallucinate keys.

### Q64: What is DPO (Direct Preference Optimization) and why is it replacing PPO in RLHF?
**Expected Answer:**
- **PPO (Proximal Policy Optimization):** Requires training a Reward Model, then using RL to optimize the policy against it. It's highly unstable, sensitive to hyperparameters, and requires 4 models in memory simultaneously (Actor, Critic, Reward, Reference).
- **DPO:** Mathematically proves that you can bypass the explicit Reward Model. You use preference pairs (Chosen vs. Rejected). The loss function directly updates the policy (LLM) to increase the relative probability of the Chosen response over the Rejected one, using a reference model for regularization.
- **Why it wins:** Much more stable, easier to implement, requires less VRAM, achieves equal or better alignment performance.

### Q65-Q75: RAPID FIRE
**Q65: Speculative decoding speeds up inference, but does it degrade output quality?**
> No, it is mathematically lossless. If the large model rejects the draft token, the exact distribution of the large model is sampled. It only saves time, not quality.

**Q66: What is the "System Prompt" vs. "User Prompt" in instruction-tuned models?**
> System Prompt sets the persona, constraints, and instructions. User Prompt is the actual task/data. The LLM is trained to prioritize system instructions over user instructions to prevent prompt injection.

**Q67: What is Prefix Tuning vs. Prompt Tuning?**
> **Prefix Tuning:** Learns virtual tokens appended to all layers of the transformer. **Prompt Tuning:** Learns virtual tokens only at the input embedding layer. Prompt tuning is cheaper but less expressive.

*End of Advanced LLM System Design.*
