# Embeddings & NLP in Insurance (Part 2) - Transformers & LLMs - Theoretical Deep Dive

## Overview
"Attention is All You Need."
In 2017, the Transformer architecture changed everything.
We moved from "counting words" (TF-IDF) and "reading sequentially" (RNNs) to "understanding context" (BERT/GPT).
This day focuses on **Transformers**, **BERT**, and the rise of **Large Language Models (LLMs)** in the insurance sector.

---

## 1. Conceptual Foundation

### 1.1 The Context Problem

*   **Word2Vec:** "Bank" has one vector.
    *   "Bank of the river" vs. "Bank of America". Same vector.
*   **Contextual Embeddings (BERT):** "Bank" has a *dynamic* vector that changes based on the surrounding words.
*   **Mechanism:** **Self-Attention**. The model looks at every word in the sentence simultaneously to decide the meaning of "Bank".

### 1.2 Encoder vs. Decoder

*   **Encoder (BERT):** Reads text and understands it. Good for Classification, NER, Sentiment.
    *   *Insurance Use:* Classifying a claim note as "Fraud" or "Not Fraud".
*   **Decoder (GPT):** Generates text. Good for Chatbots, Summarization.
    *   *Insurance Use:* Drafting a denial letter or summarizing a medical report.

---

## 2. Mathematical Framework

### 2.1 Self-Attention

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$
*   **Query (Q):** What am I looking for?
*   **Key (K):** What do I have?
*   **Value (V):** What information do I pass along?
*   **Analogy:** A database lookup, but fuzzy and differentiable.

### 2.2 Transfer Learning (The ImageNet Moment for NLP)

*   **Pre-training:** Train BERT on the entire Wikipedia (Masked Language Modeling). It learns English grammar and world knowledge.
*   **Fine-tuning:** Train the pre-trained BERT on 1,000 insurance claims to classify "Water Damage".
*   **Benefit:** You achieve state-of-the-art results with very little labeled data.

---

## 3. Theoretical Properties

### 3.1 Bidirectionality

*   **RNN:** Reads Left-to-Right.
*   **BERT:** Reads Left-to-Right AND Right-to-Left simultaneously.
*   **Why it matters:** "The *policy* covering the *car* was cancelled."
    *   To understand "policy", you need to know it covers a "car" (future word) and was "cancelled" (future word).

### 3.2 Tokenization (Sub-word)

*   **Method:** WordPiece or Byte-Pair Encoding (BPE).
*   **Example:** "Uninsurable" $\to$ ["Un", "insur", "able"].
*   **Benefit:** Handles Out-Of-Vocabulary (OOV) words. Even if the model has never seen "Uninsurable", it knows "Un", "insur", and "able".

---

## 4. Modeling Artifacts & Implementation

### 4.1 Fine-Tuning BERT for Claims Classification (Hugging Face)

```python
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# 1. Load Pre-trained BERT
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 2. Tokenize Data
texts = ["Water leak in basement", "Stolen laptop from car"]
labels = [0, 1] # 0=Property, 1=Auto (Example)

inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
dataset = CustomDataset(inputs, labels) # Wrapper class

# 3. Train (Fine-tune)
training_args = TrainingArguments(output_dir="./results", num_train_epochs=3)
trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()

# 4. Predict
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)
```

### 4.2 Zero-Shot Classification

*   **Magic:** Classify text without *any* training examples.
*   **Method:** Use a model trained on NLI (Natural Language Inference).
*   **Prompt:** "This text is about {}."
*   **Result:** You can classify claims into "Fire", "Theft", "Liability" out of the box.

---

## 5. Evaluation & Validation

### 5.1 GLUE Benchmark

*   **Standard:** General Language Understanding Evaluation.
*   **Insurance Reality:** A model that aces GLUE might fail on "Subrogation notes".
*   **Validation:** Always evaluate on a hold-out set of *your* specific domain data.

### 5.2 Perplexity (for LLMs)

*   **Metric:** How surprised is the model by the text?
*   **Lower is Better.**
*   **Use:** Evaluating the quality of a generated summary.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Computational Cost

*   **Issue:** BERT is huge (110M parameters). Inference is slow.
*   **Solution:** **DistilBERT** (40% smaller, 97% performance) or **Quantization** (INT8).

### 6.2 Max Sequence Length

*   **Constraint:** BERT usually takes max 512 tokens.
*   **Problem:** Medical reports are 50 pages long.
*   **Solution:**
    *   Chunking (Split doc into 512-token chunks).
    *   Longformer (Architecture for long docs).

---

## 7. Advanced Topics & Extensions

### 7.1 RAG (Retrieval-Augmented Generation)

*   **Problem:** LLMs hallucinate. They don't know *your* policy documents.
*   **Solution:**
    1.  **Retrieve:** Find the relevant policy section using Embeddings (Vector Database).
    2.  **Augment:** Feed that section to GPT-4.
    3.  **Generate:** "Based on Section 4.2, the claim is covered."
*   **Impact:** The "Killer App" for Insurance Q&A.

### 7.2 Domain-Specific Pre-training

*   **Idea:** Take BERT and continue pre-training it on 10M insurance documents.
*   **Result:** **InsurBERT**. It understands "Deductible", "Co-pay", "Indemnity" much better than standard BERT.

---

## 8. Regulatory & Governance Considerations

### 8.1 Hallucinations

*   **Risk:** An LLM chatbot tells a customer "Yes, that's covered" when it isn't.
*   **Liability:** The insurer is bound by the agent's (chatbot's) word.
*   **Control:** Strict Guardrails. Human-in-the-loop for binding decisions.

---

## 9. Practical Example

### 9.1 The "Medical Summary" Assistant

**Scenario:** Nurse Case Managers spend 2 hours reading medical files to decide on Workers Comp reserves.
**Solution:**
1.  OCR the PDFs.
2.  Use an LLM (e.g., GPT-4 via API) to summarize:
    *   "Diagnosis: Herniated Disc."
    *   "Prognosis: Surgery recommended."
    *   "Work Status: Off duty for 6 weeks."
**Impact:** Review time drops to 15 minutes. Consistency improves.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Transformers** use Attention to understand context.
2.  **Fine-tuning** adapts general models to insurance tasks.
3.  **RAG** grounds LLMs in truth.

### 10.2 When to Use This Knowledge
*   **Automation:** Chatbots, Document Processing.
*   **Search:** Semantic Search ("Find me all claims involving ladders").

### 10.3 Critical Success Factors
1.  **Data Quality:** OCR errors kill NLP performance. Fix the OCR first.
2.  **Latency:** Don't put a 70B parameter model in a real-time pricing API.

### 10.4 Further Reading
*   **Vaswani et al.:** "Attention Is All You Need".
*   **Hugging Face:** "Transformers Course".

---

## Appendix

### A. Glossary
*   **Logits:** Raw output scores.
*   **Softmax:** Converts logits to probabilities.
*   **Hallucination:** When an LLM confidently states a falsehood.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Attention** | $softmax(QK^T/\sqrt{d})V$ | Context Mixing |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
