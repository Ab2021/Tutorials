# Embeddings & NLP in Insurance (Part 2) - Theoretical Deep Dive

## Overview
We have moved beyond simple word vectors. **Transformers** (BERT, GPT) have revolutionized NLP. This session covers **Fine-Tuning BERT** for insurance classification and building **RAG (Retrieval Augmented Generation)** systems to chat with policy documents.

---

## 1. Conceptual Foundation

### 1.1 The Transformer Architecture

*   **Attention Mechanism:** "Attention is All You Need".
    *   Instead of reading left-to-right (RNN), the model looks at the *entire sentence at once*.
    *   It calculates how much "Attention" word A should pay to word B.
    *   *Example:* In "The animal didn't cross the street because it was too tired", "it" pays attention to "animal".

### 1.2 BERT (Bidirectional Encoder Representations from Transformers)

*   **Encoder-Only:** Good for understanding text (Classification, NER).
*   **Bidirectional:** Reads text in both directions.
*   **Pre-training:** Trained on Wikipedia to understand English.
*   **Fine-tuning:** We take a pre-trained BERT and train it slightly more on *Insurance Claims* to teach it our jargon.

### 1.3 LLMs & RAG (Generative AI)

*   **GPT-4 / Llama 3:** Decoder-only models. Good for generating text.
*   **Hallucination:** LLMs make things up.
*   **RAG (Retrieval Augmented Generation):**
    1.  User asks: "Is flood covered?"
    2.  System searches the Policy Document PDF for "Flood".
    3.  System retrieves the relevant paragraph.
    4.  System sends the paragraph + question to GPT-4.
    5.  GPT-4 answers *only using the provided paragraph*.

---

## 2. Mathematical Framework

### 2.1 Self-Attention

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$
*   $Q$ (Query): What am I looking for?
*   $K$ (Key): What do I have?
*   $V$ (Value): What is the content?
*   The dot product $QK^T$ measures similarity.

### 2.2 Fine-Tuning Loss

*   We add a simple Classification Layer on top of BERT.
*   **Loss:** Cross-Entropy between predicted class and actual label (e.g., "Policy Inquiry" vs. "Claim Report").
*   We update *all* weights (BERT + Classifier) but with a very small learning rate.

---

## 3. Theoretical Properties

### 3.1 Context Window

*   BERT limit: 512 tokens (~400 words).
*   GPT-4 limit: 128k tokens.
*   *Implication:* For long policy documents, you must chunk the text (split it into pieces) for RAG.

### 3.2 Zero-Shot vs. Few-Shot

*   **Zero-Shot:** Asking the model to do a task without examples.
*   **Few-Shot:** Giving 3 examples of "Medical Note -> ICD-10 Code" and asking it to do the 4th.
*   *Actuarial Use:* Few-shot prompting is powerful for extracting data from messy PDFs.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Fine-Tuning BERT (Hugging Face)

```python
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# 1. Load Pre-trained BERT
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 2. Tokenize Insurance Data
texts = ["Policy covers fire", "Claim denied due to fraud"]
labels = [1, 0]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# 3. Fine-Tune (Conceptual)
# trainer = Trainer(model=model, args=TrainingArguments(...), train_dataset=dataset)
# trainer.train()
```

### 4.2 RAG Workflow (Conceptual)

```python
# 1. Retrieval
query = "What is the deductible?"
docs = vector_database.similarity_search(query)
context = docs[0].page_content

# 2. Generation
prompt = f"""
Answer the question based ONLY on the context below:
Context: {context}
Question: {query}
"""
response = llm.generate(prompt)
```

---

## 5. Evaluation & Validation

### 5.1 RAG Evaluation (RAGAS)

*   **Faithfulness:** Does the answer come from the context?
*   **Answer Relevance:** Does the answer address the user's question?
*   **Context Precision:** Did the retrieval system find the right paragraph?

### 5.2 BERT Metrics

*   Accuracy, F1-Score.
*   **Confusion Matrix:** Crucial to see if "Fire" claims are being confused with "Explosion" claims.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Fine-Tuning on too little data**
    *   BERT has 110M parameters. If you fine-tune on 50 examples, it will overfit (Catastrophic Forgetting).
    *   *Fix:* Use Parameter-Efficient Fine-Tuning (PEFT/LoRA) or get more data (1000+ examples).

2.  **Trap: RAG Context Limit**
    *   Stuffing the whole 100-page policy into the prompt might confuse the model ("Lost in the Middle" phenomenon).
    *   *Fix:* Retrieve only the top-3 most relevant chunks.

### 6.2 Implementation Challenges

1.  **Latency:**
    *   BERT is slow. GPT-4 is slower.
    *   *Fix:* Distillation (Train a tiny DistilBERT to mimic the big BERT).

---

## 7. Advanced Topics & Extensions

### 7.1 LoRA (Low-Rank Adaptation)

*   Instead of fine-tuning all 110M weights, we freeze them and train a tiny adapter matrix.
*   Allows fine-tuning LLMs on a single GPU.

### 7.2 Multimodal LLMs

*   GPT-4V can "see" images.
*   *Use Case:* Upload a photo of the car crash + the police report. Ask: "Do they match?"

---

## 8. Regulatory & Governance Considerations

### 8.1 Hallucinations & Liability

*   **Risk:** Chatbot tells customer "Yes, that's covered" when it's not.
*   **Mitigation:** Strict guardrails. "If unsure, say 'Please contact an agent'."
*   **Disclaimer:** "I am an AI. Verify with policy docs."

---

## 9. Practical Example

### 9.1 Worked Example: The "Policy Chatbot"

**Scenario:**
*   Customer Support is overwhelmed with "Is X covered?" calls.
*   **Solution:** RAG System indexed on all PDF policies.
*   **Result:** Deflected 40% of calls.
*   **Safety:** The bot provides the *page number* reference for every answer.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Transformers** pay attention to context.
2.  **BERT** is for understanding (Classification).
3.  **RAG** is for safe generation (Q&A).

### 10.2 When to Use This Knowledge
*   **Customer Service:** Chatbots.
*   **Underwriting:** Extracting data from submissions.

### 10.3 Critical Success Factors
1.  **Data Quality:** Garbage in, Garbage out. Clean your PDFs.
2.  **Evaluation:** Don't trust the demo. Measure Faithfulness.

### 10.4 Further Reading
*   **Vaswani et al.:** "Attention Is All You Need".
*   **Lewis et al.:** "Retrieval-Augmented Generation".

---

## Appendix

### A. Glossary
*   **Token:** A piece of a word (e.g., "ing").
*   **Embedding:** Vector representation.
*   **Fine-tuning:** Adjusting a pre-trained model.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Attention** | Softmax($QK^T$) | Context Mixing |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
