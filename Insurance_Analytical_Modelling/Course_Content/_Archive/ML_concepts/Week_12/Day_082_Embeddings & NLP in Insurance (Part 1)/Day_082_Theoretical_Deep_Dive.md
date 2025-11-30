# Embeddings & NLP in Insurance (Part 1) - Theoretical Deep Dive

## Overview
Insurance companies sit on mountains of text data: Claim notes, Policy documents, Emails, and Adjuster reports. **Natural Language Processing (NLP)** unlocks this value. This session covers the basics: **TF-IDF**, **Tokenization**, and **Word Embeddings**.

---

## 1. Conceptual Foundation

### 1.1 The Unstructured Data Problem

*   **Structured Data:** Age, Gender, Premium (Rows and Columns). Easy for GLMs.
*   **Unstructured Data:** "Client swerved to avoid deer, hit tree. Front bumper damaged."
*   **Goal:** Convert this text into numbers that a model can understand.

### 1.2 Bag of Words (BoW) & TF-IDF

*   **Bag of Words:** Count the frequency of every word.
    *   "Deer": 1, "Tree": 1, "Bumper": 1.
    *   *Problem:* "The" appears 100 times but means nothing.
*   **TF-IDF (Term Frequency - Inverse Document Frequency):**
    *   Penalizes common words ("The", "Is").
    *   Boosts rare, specific words ("Hydroplane", "Arson").
    *   *Result:* A vector where high values = important keywords.

### 1.3 Word Embeddings (Word2Vec)

*   **Idea:** "King" - "Man" + "Woman" = "Queen".
*   **Mechanism:** Words that appear in similar contexts should have similar vectors.
*   **Actuarial Example:**
    *   "Leak" and "Burst" appear near "Pipe".
    *   The model learns that "Leak" $\approx$ "Burst".
    *   This allows the model to generalize: If it learns "Pipe Burst" is expensive, it infers "Pipe Leak" is also expensive.

---

## 2. Mathematical Framework

### 2.1 TF-IDF Calculation

*   **TF (Term Frequency):** $tf(t, d) = \frac{\text{count of } t \text{ in } d}{\text{total words in } d}$.
*   **IDF (Inverse Document Frequency):** $idf(t) = \log(\frac{N}{\text{count of docs with } t})$.
*   **TF-IDF:** $tf \times idf$.

### 2.2 Word2Vec (Skip-Gram)

*   **Objective:** Predict the context words given a center word.
*   **Input:** One-Hot vector of "Pipe".
*   **Hidden Layer:** Dense Vector (The Embedding).
*   **Output:** Probability of "Water", "Damage", "Kitchen".
*   **Training:** Maximize probability of actual context words.

---

## 3. Theoretical Properties

### 3.1 Sparsity vs. Density

*   **TF-IDF Vectors:** Extremely sparse (mostly zeros). High dimension (Size of Vocabulary, e.g., 50,000).
*   **Word Embeddings:** Dense (no zeros). Low dimension (e.g., 300).
*   *Insight:* Dense vectors are better for Deep Learning. Sparse vectors are fine for XGBoost/GLM.

### 3.2 Pre-trained vs. Custom Embeddings

*   **Pre-trained (GloVe/FastText):** Trained on Wikipedia. Good for general English.
*   **Custom:** Trained on *Insurance Claims*.
    *   *Why?* In Wikipedia, "Reserve" means "Nature Reserve". In Insurance, it means "Liability".
    *   *Action:* Always train custom embeddings if you have enough data (>100k claims).

---

## 4. Modeling Artifacts & Implementation

### 4.1 TF-IDF with Scikit-Learn

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

corpus = [
    "Water damage due to pipe burst in kitchen.",
    "Car hit tree, bumper damaged.",
    "Fire in kitchen caused by stove."
]

# Create Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
X = vectorizer.fit_transform(corpus)

# View Result
df_tfidf = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
print(df_tfidf)
```

### 4.2 Word Vectors with spaCy

```python
import spacy

# Load large English model (includes vectors)
nlp = spacy.load("en_core_web_md")

doc1 = nlp("pipe burst")
doc2 = nlp("water leak")
doc3 = nlp("car crash")

print(f"Similarity (Pipe/Leak): {doc1.similarity(doc2):.2f}") # High
print(f"Similarity (Pipe/Car): {doc1.similarity(doc3):.2f}")  # Low
```

---

## 5. Evaluation & Validation

### 5.1 Topic Modeling (LDA)

*   **Latent Dirichlet Allocation (LDA):** Unsupervised clustering of text.
*   *Use Case:* Discover "Loss Causes" automatically.
    *   Topic 1: "Wind, Roof, Shingle, Storm".
    *   Topic 2: "Water, Pipe, Sink, Mold".
*   *Validation:* Do the topics make sense to a human adjuster?

### 5.2 Sentiment Analysis

*   Use `TextBlob` or `VADER` to score sentiment (-1 to +1).
*   *Correlation:* Does low sentiment in "First Notice of Loss" call correlate with higher litigation rate?

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Stop Words Removal**
    *   Removing "not" is dangerous. "Not liable" becomes "liable".
    *   *Fix:* Be careful with stop word lists. Keep negations.

2.  **Trap: Data Leakage in Text**
    *   Claim notes might say "Claim denied". If you use this to predict "Fraud", it's leakage.
    *   *Fix:* Only use text available *at the time of prediction*.

### 6.2 Implementation Challenges

1.  **Spelling Mistakes:**
    *   Adjusters type fast. "Bumpr" instead of "Bumper".
    *   *Fix:* Use **FastText** (Embeddings based on character n-grams) which handles typos well.

---

## 7. Advanced Topics & Extensions

### 7.1 Named Entity Recognition (NER)

*   Automatically extract "Entities" from text.
*   *Example:* "Claimant [John Doe] injured [Left Leg] on [2023-01-01]."
*   *Tools:* spaCy NER.

### 7.2 Document Embeddings (Doc2Vec)

*   Instead of averaging word vectors, learn a vector for the *entire document*.
*   Better for clustering whole claims.

---

## 8. Regulatory & Governance Considerations

### 8.1 Privacy (PII)

*   Text data is full of PII (Names, SSNs, Addresses).
*   **Requirement:** Scrub/Anonymize text *before* training models.
*   **Tools:** Microsoft Presidio.

---

## 9. Practical Example

### 9.1 Worked Example: The "Subrogation" Hunter

**Scenario:**
*   Identify claims where a third party is at fault (Subrogation opportunity).
*   **Keyword Search:** Misses "Other driver ran red light".
*   **NLP Model:** Trained on "Description of Loss".
*   **Result:** Flagged 15% more subrogation files than keyword search.
*   **Value:** Millions in recovered costs.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **TF-IDF** highlights keywords.
2.  **Word2Vec** captures meaning.
3.  **NLP** turns text into features.

### 10.2 When to Use This Knowledge
*   **Claims:** Triage, Fraud, Subrogation.
*   **Underwriting:** Analyzing risk reports.

### 10.3 Critical Success Factors
1.  **Clean your text.** (Lowercasing, Punctuation removal).
2.  **Domain Adaptation.** Train on insurance data.

### 10.4 Further Reading
*   **Mikolov et al.:** "Efficient Estimation of Word Representations in Vector Space".
*   **Jurafsky & Martin:** "Speech and Language Processing".

---

## Appendix

### A. Glossary
*   **Corpus:** The collection of all documents.
*   **Lemma:** The root form of a word (Running -> Run).
*   **N-gram:** Sequence of N words (Bigram: "Pipe Burst").

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **TF-IDF** | $tf \times \log(N/df)$ | Feature Extraction |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
