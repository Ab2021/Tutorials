# Day 25: BERT - Deep Dive

> **Phase**: 3 - NLP & Transformers
> **Week**: 6 - Transformers & LLMs
> **Topic**: Embeddings, GLUE Benchmark, and Span Prediction

## 1. BERT Embeddings: Sum of Three

Input representation is sum of 3 embeddings:
1.  **Token Embeddings**: WordPiece tokens (30k vocab).
2.  **Segment Embeddings**: Belongs to Sentence A or B? ($E_A$ or $E_B$).
3.  **Position Embeddings**: Learned position vectors ($0$ to $512$).

## 2. Span Prediction (SQuAD)

How does BERT answer questions?
It doesn't generate text. It extracts a **Span** from the passage.
*   Input: `[CLS] Question [SEP] Passage`
*   Output: Two vectors $S$ (Start) and $E$ (End).
*   Calculate dot product of $S$ with every token in Passage $\to$ Softmax $\to$ Start Prob.
*   Same for End Prob.
*   Answer = Passage[Start : End].

## 3. GLUE Benchmark

**General Language Understanding Evaluation**.
A suite of 9 tasks to evaluate NLU models.
1.  **CoLA**: Linguistic Acceptability (Grammar).
2.  **SST-2**: Sentiment.
3.  **MRPC**: Paraphrase detection.
4.  **MNLI**: Natural Language Inference (Entailment/Contradiction).
*   BERT destroyed the leaderboard when released.

## 4. Cross-Encoder vs Bi-Encoder

**Cross-Encoder (BERT)**:
*   Input: `[CLS] Sent A [SEP] Sent B`.
*   Full self-attention between A and B.
*   Accurate but slow ($O(N^2)$).

**Bi-Encoder (Sentence-BERT)**:
*   Process A $\to$ Vector $u$.
*   Process B $\to$ Vector $v$.
*   Similarity: Cosine($u, v$).
*   Fast ($O(N)$), enables semantic search / clustering.

## 5. Dynamic Masking (RoBERTa)

Original BERT performed masking *once* during data preprocessing.
RoBERTa performs masking *dynamically* every time a sequence is fed to the model.
The model sees different versions of the same sentence, acting as Data Augmentation.
