# Day 16.1: BERT Architecture & Bidirectional Training - A Practical Guide

## Introduction: Understanding in Deep Context

Before 2018, the dominant paradigm for pre-trained language models was unidirectional. Models like GPT were trained to predict the next word, meaning they could only use leftward context. This is great for generation, but suboptimal for tasks that require a deep understanding of the entire sentence, where context from both the left and the right is crucial.

**BERT (Bidirectional Encoder Representations from Transformers)**, introduced by Google AI, revolutionized NLP by proposing a simple yet powerful method to train a truly **bidirectional** language model. Instead of predicting the next word, BERT is pre-trained on a **Masked Language Modeling (MLM)** task. This allows it to learn the relationships between all words in a sentence simultaneously, resulting in a much deeper and more nuanced understanding of language.

This guide will provide a practical deep dive into the BERT architecture and its unique pre-training strategy.

**Today's Learning Objectives:**

1.  **Understand the BERT Architecture:** See how BERT is simply a stack of Transformer Encoder blocks.
2.  **Grasp the Concept of Bidirectionality:** Understand why looking at both left and right context is critical for language understanding.
3.  **Learn about Masked Language Modeling (MLM):** Understand this key pre-training objective and how it enables bidirectional learning.
4.  **Learn about Next Sentence Prediction (NSP):** Understand BERT's second, sentence-level pre-training objective.
5.  **Explore BERT's Special Input Format:** See how BERT uses special tokens like `[CLS]` and `[SEP]` and segment embeddings to handle different downstream tasks.

---

## Part 1: BERT's Architecture - A Stack of Encoders

The architecture of BERT is surprisingly simple: it is just the **Encoder** part of the original Transformer model, stacked up.

*   **BERT-Base:** Consists of 12 stacked Transformer Encoder blocks.
*   **BERT-Large:** Consists of 24 stacked Transformer Encoder blocks.

That's it. There is no decoder. BERT is not designed to generate text in an autoregressive way. Its sole purpose is to take a sequence of text and produce a rich, contextualized vector representation for each token in that sequence.

![BERT Architecture](https://i.imgur.com/hV2kXso.png)

---

## Part 2: The Pre-training Tasks

The real innovation of BERT lies not in its architecture, but in how it's trained.

### 2.1. Task 1: Masked Language Modeling (MLM)

This is the core idea that enables bidirectionality.

**The Problem:** How can you learn bidirectional context? If you try to simply predict the next word, the model can't see future words. If you let it see future words, the prediction task becomes trivial.

**The Solution:**
1.  Take an input sentence (e.g., "the cat sat on the mat").
2.  Randomly **mask** about 15% of the tokens. (e.g., "the cat [MASK] on the mat").
3.  The model's objective is to predict the original identity of only the masked tokens.

Because the model has to make its prediction based on the full, surrounding context (both left and right), it is forced to learn a deep, bidirectional understanding of the language.

**The Masking Strategy:**
To prevent a mismatch between pre-training and fine-tuning (where the `[MASK]` token doesn't appear), the 15% of tokens chosen for masking are actually treated in three ways:
*   **80%** of the time, the token is replaced with `[MASK]`.
*   **10%** of the time, the token is replaced with a **random** word from the vocabulary.
*   **10%** of the time, the token is left **unchanged**.

### 2.2. Task 2: Next Sentence Prediction (NSP)

**The Idea:** Many important downstream tasks, like Question Answering and Natural Language Inference, require understanding the relationship between two sentences. To train the model for this, the NSP task was introduced.

**The Process:**
1.  The model is given two sentences, A and B, as input.
2.  **50%** of the time, sentence B is the actual sentence that follows sentence A in the original text.
3.  **50%** of the time, sentence B is a random sentence from the corpus.
4.  The model must predict whether sentence B is the true next sentence or not.

This task is trained jointly with MLM.

*(Note: Later research found that the NSP task was not as beneficial as originally thought, and many subsequent models like RoBERTa have removed it, focusing solely on MLM.)*

---

## Part 3: BERT's Special Input Format

To handle both single-sentence and sentence-pair tasks, BERT uses a specific input format.

1.  **Special Tokens:**
    *   **`[CLS]`:** A special token that is always added to the **beginning** of every input sequence. The final hidden state corresponding to this token is used as the aggregate sequence representation for classification tasks.
    *   **`[SEP]`:** A special token used to separate sentences (for sentence-pair tasks) or to mark the end of a single sentence.

2.  **Segment Embeddings:**
    *   In addition to token embeddings and positional embeddings, BERT adds a third type of embedding: **segment embeddings**.
    *   This is a learned embedding that indicates whether a token belongs to the first sentence (Sentence A) or the second sentence (Sentence B).

**The final embedding for a token is the sum of its token embedding, its positional embedding, and its segment embedding.**

### 3.1. Visualizing the Input Format

Let's see how the Hugging Face tokenizer prepares the input for us.

```python
from transformers import BertTokenizer

print("--- Part 3: BERT's Input Format ---")

# --- Load the tokenizer ---
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# --- Example 1: Single Sentence ---
sentence = "This is a single sentence."
inputs_single = tokenizer(sentence, return_tensors="pt")

print(f"Single Sentence Input:")
print(f"  - Tokens: {tokenizer.convert_ids_to_tokens(inputs_single['input_ids'][0])}")
print(f"  - Token Type IDs (Segment IDs): {inputs_single['token_type_ids']}")

# --- Example 2: Sentence Pair ---
sentence_a = "This is the first sentence."
sentence_b = "This is the second sentence."
inputs_pair = tokenizer(sentence_a, sentence_b, return_tensors="pt")

print(f"\nSentence Pair Input:")
print(f"  - Tokens: {tokenizer.convert_ids_to_tokens(inputs_pair['input_ids'][0])}")
print(f"  - Token Type IDs (Segment IDs): {inputs_pair['token_type_ids']}")
```

**Interpretation:**
*   Notice how `[CLS]` is added at the beginning and `[SEP]` is added at the end (and between sentences).
*   The `token_type_ids` (segment IDs) are all `0` for the single sentence. For the pair, they are `0` for the first sentence (and its `[SEP]`) and `1` for the second sentence.

---

## Part 4: Using BERT for Feature Extraction

Let's see how we can use a pre-trained BERT model to get rich, contextualized embeddings for our tokens.

```python
import torch
from transformers import BertModel

print("\n--- Part 4: Using BERT for Feature Extraction ---")

# --- Load the model ---
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

# --- Prepare two sentences ---
sentence1 = "The bank of the river was muddy."
sentence2 = "He made a deposit at the bank."

# --- Get embeddings for "bank" in each context ---
inputs1 = tokenizer(sentence1, return_tensors="pt")
inputs2 = tokenizer(sentence2, return_tensors="pt")

with torch.no_grad():
    outputs1 = model(**inputs1)
    outputs2 = model(**inputs2)

# Get the hidden states from the last layer
last_hidden_states1 = outputs1.last_hidden_state
last_hidden_states2 = outputs2.last_hidden_state

# Find the index of the token "bank" in each sentence
token_idx1 = tokenizer.convert_ids_to_tokens(inputs1['input_ids'][0]).index('bank')
token_idx2 = tokenizer.convert_ids_to_tokens(inputs2['input_ids'][0]).index('bank')

# Get the embedding vectors
embedding_bank1 = last_hidden_states1[0, token_idx1, :]
embedding_bank2 = last_hidden_states2[0, token_idx2, :]

# --- Compare the embeddings ---
# We use cosine similarity to see how different they are.
cos = nn.CosineSimilarity(dim=0)
similarity = cos(embedding_bank1, embedding_bank2)

print(f"The word 'bank' appears in two different contexts.")
print(f"  - Context 1: '{sentence1}'")
print(f"  - Context 2: '{sentence2}'")
print(f"\nCosine similarity between the two 'bank' embeddings: {similarity.item():.4f}")
print("--> The similarity is high, but not 1.0, showing the embeddings are contextualized.")
```

## Conclusion

BERT represents a major milestone in NLP. By using a simple Transformer Encoder architecture and cleverly designing the **Masked Language Modeling** task, it was the first model to effectively learn deep bidirectional representations of language.

**Key Takeaways:**

1.  **Architecture is Simple:** BERT is just a stack of Transformer Encoders.
2.  **Pre-training is Key:** The innovation lies in the pre-training tasks, especially Masked Language Modeling (MLM).
3.  **MLM Enables Bidirectionality:** By predicting masked tokens, the model is forced to use context from both the left and the right.
4.  **Input Format is Specific:** BERT uses `[CLS]` and `[SEP]` tokens, along with segment embeddings, to handle both single-sentence and sentence-pair tasks.
5.  **Output is Contextual:** The primary output of BERT is a sequence of contextualized embeddings, one for each input token. The representation for a word changes based on the sentence it's in.

This ability to generate rich, contextualized embeddings is what makes BERT so powerful when it is fine-tuned for a wide variety of downstream NLU tasks.

## Self-Assessment Questions

1.  **BERT vs. GPT Architecture:** What is the main architectural difference between BERT and GPT?
2.  **MLM:** Why can't a standard left-to-right language model learn deep bidirectional representations? How does MLM solve this?
3.  **`[CLS]` Token:** What is the special purpose of the `[CLS]` token's final hidden state in many fine-tuning tasks?
4.  **Input Embeddings:** What three types of embeddings are summed together to create the final input representation for a token in BERT?
5.  **Contextual Embeddings:** How does BERT solve the polysemy problem (words with multiple meanings) that affects static embedding models like Word2Vec?
