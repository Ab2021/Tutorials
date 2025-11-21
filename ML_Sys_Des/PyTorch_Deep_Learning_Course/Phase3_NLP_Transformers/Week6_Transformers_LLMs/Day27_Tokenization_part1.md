# Day 27: Tokenization - Deep Dive

> **Phase**: 3 - NLP & Transformers
> **Week**: 6 - Transformers & LLMs
> **Topic**: Unigram LM, Normalization, and Tiktoken

## 1. Unigram Language Model (SentencePiece)

BPE is greedy (bottom-up). Unigram is probabilistic (top-down).
1.  Start with a massive vocabulary (all substrings).
2.  Iteratively prune tokens that contribute least to the likelihood of the data.
3.  **Sampling**: Can sample different segmentations for the same text (Regularization).
    *   "Hello" $\to$ ["Hel", "lo"] or ["H", "ello"].

## 2. Normalization

Before tokenization, text must be cleaned.
*   **NFD/NFKC**: Unicode normalization (combining accents).
*   **Lowercasing**: For uncased models.
*   **Stripping Accents**: é $\to$ e.

## 3. Tiktoken (OpenAI)

Fast BPE implementation for GPT-4.
*   **Educational Value**:
    *   Handles whitespace efficiently (e.g., " world" is a token).
    *   Special handling for code (indentation).
    *   Ensures numbers are split reasonably.

## 4. Vocabulary Size Trade-off

*   **Small Vocab (30k)**:
    *   Pros: Smaller embedding matrix (fewer params).
    *   Cons: Longer sequences (more tokens per sentence). Slower inference.
*   **Large Vocab (100k - LLaMA)**:
    *   Pros: Shorter sequences. Better compression.
    *   Cons: Huge embedding matrix. Harder to train rare tokens.

## 5. The "Space" Issue

*   **BERT**: Splits by space first. "New York" $\to$ "New", "York".
*   **SentencePiece**: Treats space as a character `_`. "New York" $\to$ "New", "_York".
    *   Reversible! We can reconstruct exact spacing.
