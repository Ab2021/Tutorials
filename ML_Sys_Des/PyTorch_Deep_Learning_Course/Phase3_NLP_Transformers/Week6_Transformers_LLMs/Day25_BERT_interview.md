# Day 25: BERT & Encoders - Interview Questions

> **Phase**: 3 - NLP & Transformers
> **Week**: 6 - Transformers & LLMs
> **Topic**: BERT, RoBERTa, and Transfer Learning

### 1. Why is BERT called "Bidirectional"?
**Answer:**
*   Unlike GPT (which is autoregressive and only sees left context), BERT uses Masked Language Modeling.
*   The self-attention mechanism allows every token to attend to every other token (left and right) simultaneously.

### 2. What is "Masked Language Modeling" (MLM)?
**Answer:**
*   A pre-training objective where 15% of tokens are randomly masked.
*   The model must predict the original token based on the context.
*   Loss is calculated only on the masked tokens.

### 3. Why can't we use standard Language Modeling (Next Word Prediction) for BERT?
**Answer:**
*   Because BERT is bidirectional.
*   If we tried to predict word $t$ using all words, the model would simply "see" word $t$ (leakage) because self-attention is not masked.
*   MLM forces the model to reconstruct the word from context without seeing the word itself.

### 4. What is "Next Sentence Prediction" (NSP)?
**Answer:**
*   A binary classification task used in BERT pre-training.
*   Input: `[CLS] A [SEP] B`.
*   Target: Is B the actual next sentence?
*   Intended to teach the model about sentence relationships (useful for QA/NLI).

### 5. Why did RoBERTa remove NSP?
**Answer:**
*   Empirical results showed it wasn't helpful and sometimes hurt performance.
*   Training on contiguous full-length sequences (without the A/B split) proved better for learning long-range dependencies.

### 6. What is the `[CLS]` token?
**Answer:**
*   A special token added to the start of every sequence.
*   Its final hidden state is used as the aggregate representation of the entire sequence for classification tasks.

### 7. What is the `[SEP]` token?
**Answer:**
*   Separator token.
*   Used to separate two sentences (e.g., Question and Answer) in the input.

### 8. How does BERT handle Out-Of-Vocabulary (OOV) words?
**Answer:**
*   It uses **WordPiece** tokenization.
*   Splits unknown words into subwords (e.g., "playing" $\to$ "play" + "##ing").
*   If a subword is still unknown, it breaks down to characters.

### 9. What is "DistilBERT"?
**Answer:**
*   A smaller, faster, cheaper version of BERT trained via Knowledge Distillation.
*   40% smaller, 60% faster, retains 97% of performance.
*   Student learns to mimic the Teacher's soft targets (logits) and cosine distance of hidden states.

### 10. What is "Dynamic Masking"?
**Answer:**
*   Used in RoBERTa.
*   Instead of masking the data once during preprocessing (static), the mask is generated randomly every time a sequence is fed to the model.
*   Allows the model to see different versions of the same sentence.

### 11. What is the maximum sequence length for BERT?
**Answer:**
*   512 tokens.
*   Limited by the $O(N^2)$ complexity of self-attention and the learned positional embeddings.

### 12. How do you use BERT for Question Answering (SQuAD)?
**Answer:**
*   Input: `[CLS] Question [SEP] Passage`.
*   Output: Two vectors (Start Logits, End Logits).
*   Predict the start and end indices of the answer span in the passage.

### 13. What is "Fine-Tuning"?
**Answer:**
*   Taking a pre-trained model (BERT) and updating its weights on a specific downstream task (e.g., Sentiment Analysis).
*   Usually done with a small learning rate for few epochs.

### 14. What is "Feature Extraction" (Frozen BERT)?
**Answer:**
*   Freezing the BERT weights and only training a classifier on top of the embeddings.
*   Faster training, but usually lower performance than full fine-tuning.

### 15. What is "Sentence-BERT" (SBERT)?
**Answer:**
*   A modification of BERT to derive semantically meaningful sentence embeddings.
*   Uses Siamese Networks (Bi-Encoder) to process two sentences independently and optimize Cosine Similarity.
*   Standard BERT `[CLS]` embeddings are actually poor for semantic similarity without this fine-tuning.

### 16. Why does BERT use "GELU" instead of "ReLU"?
**Answer:**
*   GELU (Gaussian Error Linear Unit) is smoother and probabilistic.
*   It allows for negative values (unlike ReLU which clips at 0), which helps gradient flow in deep transformers.

### 17. What is the vocabulary size of BERT?
**Answer:**
*   30,522 tokens (WordPiece).

### 18. What is "Whole Word Masking"?
**Answer:**
*   If a word is split into subwords ("play", "##ing"), masking only "play" makes it too easy to guess from "##ing".
*   Whole Word Masking ensures that if one subword is masked, all subwords of that word are masked.

### 19. What is "ALBERT"?
**Answer:**
*   A Lite BERT.
*   Uses **Parameter Sharing** (all layers share the same weights) and **Factorized Embedding Parameterization**.
*   Drastically reduces parameters but not inference time.

### 20. How many layers in BERT-Base vs BERT-Large?
**Answer:**
*   **Base**: 12 Layers, 768 Hidden, 12 Heads (110M params).
*   **Large**: 24 Layers, 1024 Hidden, 16 Heads (340M params).
