# Day 16: Pre-training Objectives
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: Compare Causal Language Modeling (CLM) and Masked Language Modeling (MLM). Which is better for what?

**Answer:**
- **CLM (GPT):** Predicts next token. Unidirectional.
    - *Best for:* Text generation, zero-shot prompting, creative writing.
    - *Weakness:* Cannot see future context (bad for "fill in the blank").
- **MLM (BERT):** Predicts masked tokens. Bidirectional.
    - *Best for:* Text understanding, classification, NER, embeddings.
    - *Weakness:* Cannot generate coherent long text (not autoregressive).

#### Q2: Why is "Next Sentence Prediction" (NSP) considered deprecated?

**Answer:**
- **Original Goal:** Teach BERT to understand relationships between sentences.
- **Reality:** The negative examples (random sentences) were too easy to distinguish from positive ones based on topic shifts alone. The model didn't learn deep coherence.
- **Replacement:** Training on dense, contiguous sequences (e.g., 512 tokens from a single document) proved to be much more effective for learning long-range dependencies.

#### Q3: What is "Fill-In-The-Middle" (FIM) training, and why is it crucial for Code LLMs?

**Answer:**
- **Problem:** Standard CLM generates code from left to right. But developers often insert code *inside* an existing function (e.g., adding a line in the middle).
- **FIM:** We take a document `A B C`, cut out `B`, and train the model on `A <SUF> C <PRE> B`.
- **Benefit:** The model learns to generate `B` conditioned on both the prefix `A` and the suffix `C`. This enables "in-filling" capabilities in IDEs (like Copilot).

#### Q4: Explain the "Sample Efficiency" difference between CLM and MLM.

**Answer:**
- **CLM:** Every token is a target. Loss is calculated on $100\%$ of the sequence. High signal density.
- **MLM:** Only $15\%$ of tokens are masked. Loss is calculated only on those. Low signal density.
- **Result:** BERT models typically require more training steps (epochs) over the data to see the same amount of signal as a GPT model.

#### Q5: What is Prefix LM, and how does it combine the best of both worlds?

**Answer:**
- **Concept:** A hybrid attention mask.
- **Prefix Part:** Fully bidirectional attention (like BERT). The model can "read" the prompt/context perfectly.
- **Target Part:** Causal attention (like GPT). The model generates the answer autoregressively.
- **Use Case:** Seq2Seq tasks (Translation, Summarization) where understanding the input is as important as generating the output. Used in T5 and PaLM.

---

### Production Challenges

#### Challenge 1: Domain Adaptation for a Specific Industry (e.g., Medical)

**Scenario:** You have a general LLaMA model and 10GB of medical journals. You want a medical expert model.
**Choice:** CLM vs MLM?
**Analysis:**
- If you want a chatbot: **CLM** (Continual Pre-training).
- If you want a document classifier/search engine: **MLM** (Domain-Adaptive Pre-training on BERT).
**Strategy:**
- Don't train from scratch. Initialize with LLaMA weights.
- Continue training with CLM objective on medical text.
- **Critical:** Mix in 10-20% general data (replay) to prevent the model from forgetting English grammar while learning medical jargon.

#### Challenge 2: Training a Code Assistant

**Scenario:** You are training a model for internal company code.
**Issue:** Standard CLM models are bad at completing code in the middle of a file.
**Solution:**
- Implement **FIM (Fill-In-The-Middle)** data formatting in your data pipeline.
- Randomly split your code files into (Prefix, Middle, Suffix).
- Format: `<PRE> Prefix <SUF> Suffix <MID> Middle`.
- Train with standard CLM loss on this transformed data.

#### Challenge 3: Data Deduplication Impact

**Scenario:** Your training dataset contains many duplicate documents (e.g., MIT License headers in code).
**Impact:**
- **Overfitting:** The model memorizes these repetitive patterns perfectly.
- **Bias:** The model becomes biased towards the most frequent boilerplate.
- **Inefficiency:** Wasted compute training on the same thing.
**Solution:**
- **MinHash LSH:** Deduplicate the dataset before training.
- **Semantic Dedup:** Use embeddings to remove near-duplicates.
- **Result:** Lee et al. (2022) showed that training on 1 epoch of deduped data is better than 10 epochs of raw data.

#### Challenge 4: Tokenization Mismatch in Multilingual Training

**Scenario:** You train a model on English and Chinese.
**Issue:** Your tokenizer (BPE) was trained mostly on English.
**Result:**
- English words = 1 token.
- Chinese characters = 3 bytes = 3 tokens (if falling back to byte-level).
- **Inefficiency:** Chinese training is 3x slower and context window is effectively 3x smaller.
**Solution:**
- Train a custom tokenizer on a balanced corpus.
- Or extend the vocabulary of an existing tokenizer and resize the embedding layer.

### Summary Checklist for Production
- [ ] **GenAI:** Always use **CLM**.
- [ ] **Embeddings/Search:** Always use **MLM** (or contrastive learning).
- [ ] **Code:** Use **CLM + FIM**.
- [ ] **Data:** Deduplicate aggressively.
- [ ] **Tokenizer:** Ensure it fits your data distribution.
