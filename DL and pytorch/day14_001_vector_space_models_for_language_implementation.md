# Day 14.1: Vector Space Models for Language - A Practical Guide

## Introduction: The Challenge of Representing Words

How can we make a computer understand the meaning of a word? We can't feed the string "cat" directly into a neural network. We need a way to represent words **numerically**. The simplest approach is **one-hot encoding**, where each word in the vocabulary gets a unique index, and its representation is a giant vector of all zeros except for a single 1 at that index.

**The Problem with One-Hot Encoding:**
*   **Sparsity:** For a vocabulary of 50,000 words, each word vector has 50,000 dimensions, which is computationally inefficient.
*   **No Notion of Similarity:** The one-hot vectors for "cat" and "kitten" are just as different as the vectors for "cat" and "car." The dot product between any two distinct word vectors is zero, meaning they are orthogonal and have no shared similarity.

**Vector Space Models (VSMs)** solve this by representing words as **dense, low-dimensional vectors** (called **embeddings**). In this vector space, similar words are located close to each other. This is the foundational concept of modern NLP.

This guide will provide a practical exploration of classic, count-based VSMs, specifically **TF-IDF**, to build an intuition for why dense vector representations are so powerful.

**Today's Learning Objectives:**

1.  **Understand the Motivation for Dense Embeddings:** Grasp the limitations of one-hot encoding.
2.  **Learn the Theory of Co-occurrence Matrices:** Understand the distributional hypothesis—"a word is characterized by the company it keeps."
3.  **Implement TF-IDF:** Learn about Term Frequency-Inverse Document Frequency, a classic and powerful technique for weighting the importance of words in a document.
4.  **Create Document Vectors with TF-IDF:** Use `scikit-learn` to convert a corpus of text into a document-term matrix of TF-IDF features.
5.  **Measure Similarity with Cosine Similarity:** Learn how to calculate the similarity between two word or document vectors in the vector space.

---

## Part 1: The Distributional Hypothesis and Co-occurrence

The core idea behind VSMs is the **distributional hypothesis**: words that appear in similar contexts tend to have similar meanings.

We can capture this by building a **co-occurrence matrix**. We create a large matrix where the rows and columns are all the words in our vocabulary. The value in cell `(i, j)` is the number of times word `i` appears in the same context (e.g., in the same document or within a window of 5 words) as word `j`.

Words like "cat" and "kitten" will appear with similar context words ("pet," "food," "meow"), so their rows in this matrix will be similar. These rows can then be used as the word vectors. While simple, this approach suffers from being very high-dimensional and sparse. Techniques like Singular Value Decomposition (SVD) can be used to reduce the dimensionality, but modern methods have largely superseded this.

---

## Part 2: TF-IDF - Weighting Word Importance

Instead of just counting co-occurrences, what if we could create a vector for a **document** that represents the importance of each word within it? This is what **TF-IDF (Term Frequency-Inverse Document Frequency)** does.

It's a product of two statistics:

1.  **Term Frequency (TF):** How often does a term `t` appear in a document `d`? This is simply the count of the term in the document, often normalized by the document's length.
    *   `TF(t, d) = (Number of times term t appears in d) / (Total number of terms in d)`
    *   **Intuition:** Words that appear frequently in a document are important for that document.

2.  **Inverse Document Frequency (IDF):** A measure of how much information a word provides. It's a logarithmic scale of the inverse of the fraction of documents that contain the word.
    *   `IDF(t, D) = log( (Total number of documents) / (Number of documents with term t in them) )`
    *   **Intuition:** Words that are very common across *all* documents (like "the", "a", "is") are not very informative and will have a low IDF score. Rare words that appear in only a few documents will have a high IDF score.

**TF-IDF Score:**

`TF-IDF(t, d, D) = TF(t, d) * IDF(t, D)`

The final score is high when a word appears frequently in a specific document but rarely in the overall corpus. This score gives us a weighted representation of each word's importance to a particular document.

### 2.1. Implementing TF-IDF with Scikit-learn

`scikit-learn` provides a highly efficient implementation of this process with `TfidfVectorizer`.

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

print("--- Part 2: TF-IDF with Scikit-learn ---")

# --- 1. Sample Corpus ---
corpus = [
    'the cat sat on the mat',
    'the dog ate my homework',
    'my cat loves the dog',
    'the cat and the dog are friends'
]

# --- 2. Create the TfidfVectorizer ---
# This object will learn the vocabulary and IDF weights from the data.
vectorizer = TfidfVectorizer()

# --- 3. Fit and Transform the Corpus ---
# This creates the document-term matrix, where each row is a document
# and each column is a word from the vocabulary. The values are the TF-IDF scores.
tfidf_matrix = vectorizer.fit_transform(corpus)

# --- 4. Inspect the Results ---
# Get the learned vocabulary (mapping from word to column index)
feature_names = vectorizer.get_feature_names_out()

# Convert the sparse matrix to a dense DataFrame for easy viewing
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

print("Original Corpus:")
for doc in corpus:
    print(f"  - {doc}")

print("\nLearned Vocabulary (Feature Names):")
print(feature_names)

print("\nResulting TF-IDF Matrix:")
print(df_tfidf)
```

**Interpretation:**
*   Each row is the **vector representation** for a document.
*   Words like "the" have a relatively low TF-IDF score because they appear in all documents (low IDF).
*   Words like "homework" or "friends" have a high score in their respective documents because they are frequent in that document but rare overall.

---

## Part 3: Measuring Similarity - Cosine Similarity

Now that we have vector representations for our documents, how do we measure their similarity? We can't just use Euclidean distance, because longer documents will naturally have larger vector magnitudes.

The standard measure is **Cosine Similarity**. It measures the cosine of the angle between two vectors. It is simply the dot product of the vectors divided by the product of their magnitudes.

*   **Formula:** `similarity = cos(theta) = (A . B) / (||A|| * ||B||)`
*   **Range:** -1 to 1.
    *   **1:** The vectors point in the exact same direction (very similar).
    *   **0:** The vectors are orthogonal (no similarity).
    *   **-1:** The vectors point in opposite directions.

### 3.1. Calculating Cosine Similarity

```python
from sklearn.metrics.pairwise import cosine_similarity

print("\n--- Part 3: Cosine Similarity ---")

# The tfidf_matrix from the previous step has shape (num_documents, vocab_size)
# Each row is a document vector.

# Let's calculate the similarity between all pairs of documents
similarity_matrix = cosine_similarity(tfidf_matrix)

print("Cosine Similarity Matrix between documents:")
print(pd.DataFrame(similarity_matrix, 
                   index=['doc1', 'doc2', 'doc3', 'doc4'], 
                   columns=['doc1', 'doc2', 'doc3', 'doc4']))
```

**Interpretation:**
*   The diagonal is all 1s, as each document is perfectly similar to itself.
*   Notice the high similarity between `doc3` ("my cat loves the dog") and `doc4` ("the cat and the dog are friends"). This makes intuitive sense, as they share many important words.
*   The similarity between `doc1` ("the cat sat on the mat") and `doc2` ("the dog ate my homework") is very low.

## Conclusion: From Counts to Meaning

Vector Space Models and techniques like TF-IDF were a foundational step in getting computers to understand text. They showed that we can represent documents as numerical vectors in a space where proximity equates to semantic similarity.

**Key Takeaways:**

1.  **Dense Vectors are Key:** Representing words or documents as dense vectors is far more powerful than sparse, one-hot representations because it allows us to capture the notion of similarity.
2.  **TF-IDF Weights Importance:** It provides a more nuanced representation than simple word counts by weighting words based on both their frequency in a document and their rarity across the entire corpus.
3.  **Cosine Similarity Measures Meaning:** In a vector space, the angle between two vectors is a robust measure of their semantic similarity, independent of their magnitude.

**The Limitation:**
While powerful, these count-based methods have a major drawback: they have **no understanding of word order or context within a sentence**. The TF-IDF vector for "man bites dog" is identical to the vector for "dog bites man." They also fail to capture deeper semantic relationships (like synonyms or analogies) that are not immediately obvious from word counts.

This limitation is what motivated the development of **prediction-based models** like **Word2Vec**, which we will explore in the next guide. These models don't just count words; they learn embeddings by training a neural network to predict a word from its context, leading to much richer and more powerful vector representations.

## Self-Assessment Questions

1.  **One-Hot Encoding:** What are the two main drawbacks of using one-hot encoding to represent words?
2.  **TF-IDF:** Which part of the TF-IDF calculation (TF or IDF) helps to down-weight the importance of common words like "the"?
3.  **High TF-IDF Score:** What does a high TF-IDF score for a word in a particular document signify?
4.  **Cosine Similarity:** Why is cosine similarity often preferred over Euclidean distance for measuring the similarity of text document vectors?
5.  **Limitations:** What is the primary limitation of a "bag-of-words" model like TF-IDF?

