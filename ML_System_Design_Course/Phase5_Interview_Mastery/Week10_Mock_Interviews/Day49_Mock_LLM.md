# Day 49: Mock Interview 5 - LLM Chatbot (ChatGPT System Design)

> **Phase**: 5 - Interview Mastery
> **Week**: 10 - Mock Interviews
> **Focus**: GenAI Architecture
> **Reading Time**: 60 mins

---

## 1. Problem Statement

"Design a chatbot like ChatGPT that can answer questions, remember context, and write code."

---

## 2. Step-by-Step Design

### Step 1: Requirements
*   **Streaming**: Tokens must appear one by one.
*   **Memory**: Must remember previous turns in the conversation.
*   **Safety**: Must not generate hate speech.

### Step 2: Architecture
1.  **Gateway**: Websocket connection for streaming.
2.  **Orchestrator**: Manages the flow.
3.  **Context Manager**: Pulls recent chat history from Redis. Truncates if too long.
4.  **Safety Layer (Input)**: Azure Content Safety / OpenAI Moderation API.
5.  **Model Service**: vLLM serving Llama-3-70B.
6.  **Safety Layer (Output)**: Check response before streaming.

### Step 3: Memory Handling
*   **Sliding Window**: Keep last K tokens.
*   **Summarization**: If context > 4k, use a small LLM to summarize the start of the conversation and append it as a system message.

---

## 3. Deep Dive Questions

**Interviewer**: "How do you reduce latency?"
**Candidate**: "1. **Streaming**: Don't wait for full answer. 2. **Speculative Decoding**: Use a small draft model. 3. **KV Cache**: Cache the attention states of the system prompt so we don't recompute them for every user."

**Interviewer**: "How do you handle 'I don't know'?"
**Candidate**: "We need to reduce Hallucination. We can use RAG. If the retrieval score is low (no relevant documents found), we instruct the model via System Prompt to say 'I don't have enough information' instead of guessing."

---

## 4. Evaluation
*   **Metric**: Time to First Token (TTFT).
*   **Quality**: Human Evaluation (Elo Rating).

---

## 5. Further Reading
- [ChatGPT System Design (ByteByteGo)](https://bytebytego.com/)
