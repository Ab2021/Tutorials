# Day 22: Fine-tuning Fundamentals (SFT)
## Core Concepts & Theory

### The Need for Fine-tuning

Pre-trained models (Base Models) like LLaMA-2-Base are "next-token predictors".
- **Prompt:** "What is the capital of France?"
- **Base Model:** "And what is the capital of Germany? And Italy?" (It thinks it's generating a list of questions).
- **Goal:** We want it to answer the question.
- **Solution:** Supervised Fine-Tuning (SFT).

### 1. Supervised Fine-Tuning (SFT)

**Process:**
1.  **Dataset:** Collect (Prompt, Response) pairs.
2.  **Formatting:** Wrap them in a template.
    - `User: {Prompt}\nAssistant: {Response}`
3.  **Training:** Train the model on this dataset using the standard Causal Language Modeling (CLM) objective.
4.  **Masking:** IMPORTANT. We only calculate loss on the **Assistant's Response**. We mask out the User Prompt.
    - Why? We don't want the model to learn to predict the user's question (it already knows English). We want it to learn to *respond*.

### 2. Instruction Tuning

A specific type of SFT where the prompts are "Instructions".
- **FLAN (Fine-tuned Language Net):** Google showed that fine-tuning on a massive collection of NLP tasks (summarization, translation, logic) phrased as instructions makes the model generalize to *unseen* instructions.
- **Dataset:** FLAN, Alpaca, ShareGPT.

### 3. Chat Templates

Raw text is not enough. Models need to know who is speaking.
**ChatML (OpenAI/Microsoft):**
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```
**Llama-2 Format:**
`<s>[INST] <<SYS>> System prompt <</SYS>> User prompt [/INST] Model answer </s>`

### 4. Full Fine-tuning (FFT)

Updating **all** parameters of the model.
- **Pros:** Maximum performance.
- **Cons:** Expensive. Requires 3-4x the VRAM of the model size (Optimizer states + Gradients).
- **Hardware:** Fine-tuning LLaMA-7B (FP16) requires ~112GB VRAM (impossible on consumer GPUs).

### 5. Catastrophic Forgetting

When fine-tuning on a small dataset (e.g., medical), the model "forgets" general knowledge (e.g., coding, history).
- **Mechanism:** Weights are updated to minimize loss on the new data, moving them away from the optimal configuration for the old data.
- **Mitigation:** Replay Buffer (mix in 10% general data).

### Summary of SFT

| Component | Description |
| :--- | :--- |
| **Objective** | CLM (Next Token Prediction) |
| **Masking** | Loss only on Response |
| **Format** | Chat Templates (ChatML, Alpaca) |
| **Risk** | Overfitting, Forgetting |

### Next Steps
In the Deep Dive, we will implement the Data Collator for SFT that handles the critical masking of user prompts.
