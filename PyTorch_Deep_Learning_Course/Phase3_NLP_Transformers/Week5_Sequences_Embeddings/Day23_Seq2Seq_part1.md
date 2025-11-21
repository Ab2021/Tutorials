# Day 23: Seq2Seq - Deep Dive

> **Phase**: 3 - NLP & Transformers
> **Week**: 5 - Sequences & Embeddings
> **Topic**: Beam Search, BLEU, and Teacher Forcing

## 1. Decoding Strategies

How do we generate the output sequence?

### Greedy Decoding
Pick the word with highest probability at each step.
*   Problem: Once you make a mistake, you can't go back. Local optimum $\neq$ Global optimum.

### Beam Search
Keep track of the top $k$ (Beam Width) most probable sequences at each step.
*   Step 1: Keep top 3 words.
*   Step 2: Expand all 3 paths, calculate total probability, keep top 3 paths.
*   Result: Much better translations.

## 2. Evaluation Metric: BLEU Score

**Bilingual Evaluation Understudy**.
Compares Candidate translation with Reference translations.
*   **N-gram Precision**: Count matching n-grams (1-gram, 2-gram, ...).
*   **Brevity Penalty**: Penalize if candidate is too short (to prevent cheating by outputting just "the").
$$ BLEU = BP \cdot \exp(\sum w_n \log p_n) $$

## 3. Teacher Forcing

Training RNNs is unstable because errors accumulate.
**Teacher Forcing**: Feed the *Ground Truth* previous token as input, regardless of what the model predicted.
*   **Training**: $Input_t = Target_{t-1}$.
*   **Inference**: $Input_t = Predicted_{t-1}$ (Autoregressive).

**Scheduled Sampling**: Start with 100% Teacher Forcing, gradually reduce it to 0% to mitigate Exposure Bias.

## 4. Pointer Networks

Standard Seq2Seq generates from a fixed vocab.
**Pointer Networks**: Can "copy" words from the input sequence to the output.
*   Attention weights act as pointers.
*   Crucial for Summarization (copying names) and rare words.

## 5. The Alignment Matrix

Visualizing $\alpha_{tj}$.
*   Rows: Output words.
*   Cols: Input words.
*   We can see which input word the model "looked at" when generating an output word.
*   Provides interpretability.
