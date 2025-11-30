# GenAI Fundamentals for Insurance (Part 1) - LLMs & Transformers - Theoretical Deep Dive

## Overview
"AI used to be about *predicting* the future (Churn, Claims). Now it's about *creating* content."
**Generative AI (GenAI)** is the biggest shift in technology since the Internet.
For Insurers, it means:
1.  **Underwriting:** Reading 100-page medical reports in seconds.
2.  **Claims:** Generating damage assessments from photos.
3.  **Service:** Chatbots that actually understand context.

---

## 1. Conceptual Foundation

### 1.1 Predictive AI vs. Generative AI

*   **Predictive AI (Traditional):**
    *   *Input:* Customer Data.
    *   *Output:* Probability of Claim (0.0 to 1.0).
    *   *Model:* XGBoost, GLM.
    *   *Goal:* Accuracy.
*   **Generative AI (New):**
    *   *Input:* "Write a rejection letter for Claim #123."
    *   *Output:* A polite, legally compliant letter.
    *   *Model:* Transformer (LLM).
    *   *Goal:* Creativity & Coherence.

### 1.2 The Transformer Revolution

*   **Pre-2017 (RNNs):** Processed words one by one. Slow. Forgot long sentences.
*   **2017 (Attention Is All You Need):** The **Transformer** architecture.
    *   **Parallelism:** Processes the whole sentence at once.
    *   **Attention:** Knows that "Bank" in "River Bank" is different from "Bank" in "Bank Account".

---

## 2. Mathematical Framework

### 2.1 The Attention Mechanism

The core of GenAI is the ability to pay "Attention" to relevant words.

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

*   **Query ($Q$):** What am I looking for? (e.g., "Claimant").
*   **Key ($K$):** What do I have? (e.g., "The claimant, Mr. Smith...").
*   **Value ($V$):** The content.
*   **Result:** A weighted sum where relevant words get high weights.

### 2.2 Tokenization & Embeddings

*   **Tokenization:** Breaking text into chunks.
    *   "Insurance" -> `["In", "sur", "ance"]`.
    *   *Note:* LLMs don't see words; they see numbers (Token IDs).
*   **Embeddings:** Converting tokens into vectors.
    *   `King - Man + Woman = Queen`.
    *   In Insurance: `Claim - Payment + Denial = Lawsuit`.

---

## 3. Theoretical Properties

### 3.1 Hallucinations

*   **Definition:** The model generates plausible but false information.
*   **Cause:** The model is a "Stochastic Parrot". It predicts the *next likely word*, not the *truth*.
*   **Risk in Insurance:**
    *   *Prompt:* "Summarize Policy X."
    *   *Output:* "Policy X covers Flood." (It doesn't).
    *   *Consequence:* Multi-million dollar coverage dispute.

### 3.2 Context Window

*   **Definition:** How much text the model can "remember" at once.
*   **Evolution:**
    *   GPT-3: 4k tokens (~3,000 words).
    *   GPT-4: 128k tokens (~100,000 words).
    *   Gemini 1.5: 1M+ tokens.
*   **Relevance:** You can now feed an entire 500-page Policy Document into the prompt.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Calling an LLM API (Python)

```python
import openai

# 1. Setup
client = openai.OpenAI(api_key="sk-...")

# 2. The Prompt
system_prompt = "You are a Senior Underwriter. Summarize the risk factors in this report."
user_prompt = "The building was constructed in 1920. It has knob-and-tube wiring..."

# 3. The Call
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.0 # Deterministic output
)

# 4. Output
print(response.choices[0].message.content)
# Output: "Risk Factors: 1. Age of building (1920). 2. Electrical Fire Hazard (Knob-and-tube)..."
```

### 4.2 Prompt Engineering Patterns

*   **Zero-Shot:** Just ask. "Classify this claim."
*   **Few-Shot:** Give examples.
    *   "Claim: Water leak. Class: Non-Structural."
    *   "Claim: Roof collapse. Class: Structural."
    *   "Claim: Broken window. Class: ?"
*   **Chain-of-Thought:** "Think step by step."
    *   "First, check the policy date. Second, check the exclusion list..."

---

## 5. Evaluation & Validation

### 5.1 ROUGE & BLEU Scores

*   **Traditional NLP Metrics:** Compare the generated text to a "Reference" text (n-gram overlap).
*   **Flaw:** "The cat is on the mat" vs "The mat is under the cat". Low BLEU score, but same meaning.

### 5.2 LLM-as-a-Judge

*   **Method:** Use a stronger LLM (GPT-4) to grade the output of a weaker LLM (Llama-2).
*   **Criteria:** Accuracy, Tone, Compliance.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Treating LLMs as Databases**
    *   *Mistake:* Asking "What is the premium for a 2020 Honda Civic?"
    *   *Reality:* The model's training data is cut off (e.g., 2023). It doesn't know today's rates.
    *   *Fix:* Use **RAG (Retrieval-Augmented Generation)**. Connect the LLM to your Rate Engine.

2.  **Trap: Data Privacy**
    *   *Mistake:* Pasting PII (Name, SSN) into ChatGPT.
    *   *Risk:* That data might be used to train the public model.
    *   *Fix:* Use Enterprise instances (Azure OpenAI) with "Zero Data Retention" policies.

---

## 7. Advanced Topics & Extensions

### 7.1 Fine-Tuning

*   **Concept:** Retraining the model on *your* specific data (e.g., 10,000 past Claims Adjuster reports).
*   **Result:** The model learns your company's "Voice" and specific jargon.
*   **Cost:** Expensive. (Try RAG first).

### 7.2 Multimodal Models

*   **Input:** Text + Images.
*   **Use Case:**
    *   *User:* Uploads photo of car crash.
    *   *Model:* "Bumper damage. Estimated repair cost: \$800."

---

## 8. Regulatory & Governance Considerations

### 8.1 The EU AI Act

*   **High Risk:** AI used for "Credit Scoring" or "Insurance Pricing" is High Risk.
*   **Requirement:** Explainability, Human Oversight, Robustness.
*   **Impact:** You cannot just let an LLM deny a claim automatically. A human must review it.

---

## 9. Practical Example

### 9.1 Worked Example: The "Submission Triaging" Bot

**Scenario:**
*   **Problem:** Commercial Underwriters receive 500 emails a day. 80% are "Out of Appetite" (e.g., Coal Mines).
*   **Solution:** GenAI Triage.
*   **Workflow:**
    1.  **Ingest:** Email arrives.
    2.  **LLM:** Extracts "Industry", "Revenue", "Location".
    3.  **Rule Check:** If Industry == "Mining", Auto-Decline.
    4.  **Routing:** If Industry == "Tech", Route to "Tech Team".
*   **Outcome:** Underwriters save 4 hours/day. Response time drops from 3 days to 3 minutes.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Generative AI** creates; Predictive AI classifies.
2.  **Transformers** enable context-aware understanding.
3.  **Prompt Engineering** is the new coding.

### 10.2 When to Use This Knowledge
*   **CTO:** "Should we build our own LLM?" (Probably not).
*   **Actuary:** "Can I use this to summarize these 50 treaties?" (Yes).

### 10.3 Critical Success Factors
1.  **Human-in-the-Loop:** Never let an LLM make a final coverage decision unsupervised.
2.  **Data Security:** Sanitize PII before sending to the cloud.

### 10.4 Further Reading
*   **Vaswani et al.:** "Attention Is All You Need" (The paper that started it all).

---

## Appendix

### A. Glossary
*   **LLM:** Large Language Model.
*   **RAG:** Retrieval-Augmented Generation.
*   **Hallucination:** Confident but wrong output.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Attention** | $\text{softmax}(QK^T/\sqrt{d})V$ | Context |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
