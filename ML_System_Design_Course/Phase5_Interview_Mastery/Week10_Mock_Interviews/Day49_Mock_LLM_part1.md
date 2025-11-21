# Day 49 (Part 1): Advanced Mock - LLM Chatbot

> **Phase**: 6 - Deep Dive
> **Topic**: GenAI Systems
> **Focus**: Safety, Context, and Orchestration
> **Reading Time**: 60 mins

---

## 1. Safety Rails (Guardrails)

### 1.1 Input/Output Filtering
*   **Input**: Check for PII, Jailbreaks ("DAN mode"), Toxicity.
*   **Output**: Check for hallucination, competitor mentions.
*   **Implementation**: NeMo Guardrails / Llama Guard (Classifier model).

---

## 2. Context Management

### 2.1 Summarization
*   Conversation > 4k tokens.
*   **Rolling Summary**: "User asked about X. Bot replied Y."
*   **Entity Extraction**: Store "Name: Bob", "City: Paris" in Key-Value store. Inject into system prompt.

---

## 3. Tricky Interview Questions

### Q1: How to reduce Latency?
> **Answer**:
> *   **Streaming**: Send tokens as generated. (Perceived latency is low).
> *   **Speculative Decoding**.
> *   **Caching**: Cache common queries (Semantic Cache).

### Q2: Multi-Turn Reasoning?
> **Answer**:
> *   Coreference Resolution: "How much is *it*?" -> "How much is *iPhone 15*?"
> *   Rewrite query with history before sending to RAG/LLM.

### Q3: Jailbreaks?
> **Answer**:
> *   **Adversarial Training**: Train on jailbreak examples.
> *   **System Prompt**: "You are a helpful assistant. You do not generate harmful content." (Weak).
> *   **Perplexity Check**: Jailbreaks often have weird syntax (High perplexity).

---

## 4. Practical Edge Case: User Feedback
*   **Thumbs Up/Down**: Gold mine for RLHF.
*   **Implicit**: User re-phrases query = Bad answer.

