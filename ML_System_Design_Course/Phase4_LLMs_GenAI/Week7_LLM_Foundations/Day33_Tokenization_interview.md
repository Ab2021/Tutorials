# Day 33: Tokenization - Interview Questions

> **Topic**: Text Preprocessing
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. What is Tokenization? Why not just split by space?
**Answer:**
*   Converting text into numbers (IDs).
*   Space splitting fails for: Punctuation, Chinese (no spaces), Agglutinative languages, Large Vocabulary.

### 2. Explain Byte Pair Encoding (BPE).
**Answer:**
*   Iterative merging of frequent pairs.
*   Start with characters.
*   Count pair frequencies. Merge most frequent (e.g., 'e', 's' -> 'es').
*   Repeat until vocab size reached.
*   Used in GPT-2/3/4.

### 3. Explain WordPiece.
**Answer:**
*   Similar to BPE but maximizes likelihood of language model data.
*   Used in BERT.
*   Uses `##` prefix for subwords.

### 4. Explain SentencePiece.
**Answer:**
*   Treats input as raw stream of Unicode characters (including spaces).
*   Language agnostic (works for CJK).
*   Implements BPE and Unigram.
*   Used in T5, Llama.

### 5. What is the Out-Of-Vocabulary (OOV) problem? How do subwords solve it?
**Answer:**
*   Word-based models can't handle unseen words ("unk").
*   Subwords break unseen words into known parts (`unfriendly` -> `un`, `friend`, `ly`).
*   Worst case: Fall back to characters. No OOV.

### 6. What is the trade-off between Vocabulary Size and Sequence Length?
**Answer:**
*   **Small Vocab (Chars)**: Long sequences. Hard to learn semantics. Small embedding matrix.
*   **Large Vocab (Words)**: Short sequences. OOV issues. Huge embedding matrix.
*   **Subwords**: Sweet spot.

### 7. How does BPE handle unseen characters?
**Answer:**
*   Byte-level BPE (GPT-2).
*   Operates on Bytes (UTF-8), not Unicode chars.
*   Base vocab is 256 bytes. Can represent any string.

### 8. Why do we add Special Tokens (`[CLS]`, `[SEP]`, `<s>`)?
**Answer:**
*   To mark structural boundaries.
*   Start of sentence, End of sentence, Separation between sentences.
*   Model learns to attend to them for specific tasks.

### 9. What is "Unigram" tokenization?
**Answer:**
*   Probabilistic.
*   Start with huge vocab.
*   Iteratively remove tokens that contribute least to likelihood of data.

### 10. How does Tokenization affect model performance?
**Answer:**
*   Bad tokenization (e.g., splitting numbers `123` -> `1`, `2`, `3`) hurts math ability.
*   Multilingual tokenizers need balanced data to not over-fragment low-resource languages.

### 11. What is the difference between `pre-tokenization` and `tokenization`?
**Answer:**
*   **Pre-tokenization**: Splitting by rules (whitespace, punctuation) before learning BPE merges.
*   Prevents merging across punctuation (e.g., "dog," -> "dog", ",").

### 12. Why are numbers difficult for LLMs?
**Answer:**
*   Tokenization often splits numbers inconsistently.
*   `1000` might be one token, `1001` might be `10`, `01`.
*   Hard to learn arithmetic.

### 13. How do you handle different languages in one tokenizer?
**Answer:**
*   Train BPE on corpus containing all languages.
*   Shared subwords (roots) help cross-lingual transfer.
*   Allocating vocab quota per language.

### 14. What is "Tiktoken"?
**Answer:**
*   OpenAI's fast BPE implementation.
*   Optimized for performance.

### 15. What happens if you change the tokenizer after pre-training?
**Answer:**
*   Model breaks.
*   Embeddings are tied to specific IDs.
*   Must retrain (or do complex surgery).

### 16. What is "Padding Side" (Left vs Right)?
**Answer:**
*   **Encoder (BERT)**: Right padding.
*   **Decoder (GPT)**: Left padding (for generation in batches).
*   If you right pad generation, position IDs get messed up.

### 17. How does tokenization impact cost (API)?
**Answer:**
*   APIs charge per token.
*   Better compression (more text per token) = Cheaper.
*   GPT-4 tokenizer is more efficient than GPT-3.

### 18. What is a "Chat Template"?
**Answer:**
*   Standard format for converting list of messages to string.
*   `<|user|>\nHello<|end|>\n<|assistant|>`
*   Tokenizer handles this mapping.

### 19. Why do some tokenizers add a leading space?
**Answer:**
*   SentencePiece treats space as a character (`_`).
*   `Hello World` -> `Hello`, `_World`.
*   Preserves reversibility.

### 20. How do you debug tokenization issues?
**Answer:**
*   `tokenizer.tokenize("text")`.
*   Check for unexpected splits.
*   Check `unk` tokens.
