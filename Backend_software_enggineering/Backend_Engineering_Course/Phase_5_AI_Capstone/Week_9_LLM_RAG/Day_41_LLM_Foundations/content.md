# Day 41: Intro to LLMs & Prompt Engineering

## 1. The New Backend Component

The LLM (Large Language Model) is now a standard component of the backend stack, just like a Database or a Cache.
*   **Input**: Text (Prompt).
*   **Output**: Text (Completion).
*   **Mechanism**: Next Token Prediction. It's a probabilistic engine, not a knowledge base.

---

## 2. Prompt Engineering

Programming the model with English instead of Python.

### 2.1 Zero-Shot
Asking directly.
*   "Translate 'Hello' to Spanish." -> "Hola".

### 2.2 Few-Shot
Giving examples to guide the style/format.
*   "Convert to JSON:
    Name: Alice -> {"name": "Alice"}
    Name: Bob -> {"name": "Bob"}
    Name: Charlie ->"
*   Model completes: `{"name": "Charlie"}`.

### 2.3 Chain of Thought (CoT)
Asking the model to "think" before answering.
*   "Solve this math problem. Let's think step by step."
*   Result: Accuracy increases dramatically for logic tasks.

---

## 3. The API (OpenAI)

*   **Endpoint**: `POST /v1/chat/completions`
*   **Roles**:
    *   `system`: Sets the behavior ("You are a helpful assistant").
    *   `user`: The input ("Hi").
    *   `assistant`: The model's reply ("Hello!").
*   **Parameters**:
    *   `temperature`: Randomness (0.0 = Deterministic, 1.0 = Creative).
    *   `max_tokens`: Limit output length.

---

## 4. Summary

Today we met our new coworker.
*   **LLM**: A text prediction engine.
*   **Prompting**: The art of guiding the prediction.
*   **API**: Stateless HTTP calls.

**Tomorrow (Day 42)**: We will learn how to chain these calls together using **LangChain**.
