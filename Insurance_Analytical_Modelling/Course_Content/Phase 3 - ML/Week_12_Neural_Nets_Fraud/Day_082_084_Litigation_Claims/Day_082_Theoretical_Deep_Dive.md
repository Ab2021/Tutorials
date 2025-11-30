# Embeddings & NLP in Insurance (Part 1) - Text Mining Foundations - Theoretical Deep Dive

## Overview
"The richest data in insurance is buried in PDF notes."
Adjuster notes, Police Reports, Medical Diagnoses.
80% of insurance data is unstructured text.
This day focuses on **Natural Language Processing (NLP)**: Turning text into numbers that models can understand.
We start with the foundations: **Bag of Words**, **TF-IDF**, and the leap to **Word Embeddings**.

---

## 1. Conceptual Foundation

### 1.1 The Unstructured Opportunity

*   **Structured Data:** "Loss Amount: \$5,000". (Easy).
*   **Unstructured Data:** "Claimant states he was driving 30mph when the deer jumped out. Police report notes skid marks." (Hard).
*   **Goal:** Extract features like "Deer Involvement", "Speed", "Police Presence" automatically.

### 1.2 From Strings to Vectors

Models can't read. They only understand math.
1.  **Tokenization:** Breaking text into units (words/sub-words).
    *   "The deer jumped" $\to$ ["The", "deer", "jumped"].
2.  **Vectorization:** Converting tokens to numbers.

---

## 2. Mathematical Framework

### 2.1 TF-IDF (Term Frequency - Inverse Document Frequency)

A statistical measure of how important a word is to a document in a collection.
$$ \text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t) $$
*   **TF:** How often word $t$ appears in document $d$.
*   **IDF:** $\log(\frac{N}{DF_t})$. Penalizes words that appear everywhere (e.g., "the", "claim").
*   **Result:** Rare, specific words (e.g., "Whiplash", "Arson") get high scores.

### 2.2 Word Embeddings (Word2Vec)

*   **Idea:** "You shall know a word by the company it keeps." (Firth, 1957).
*   **Mechanism:** Train a shallow neural network to predict a word given its context (Skip-gram) or context given a word (CBOW).
*   **Result:** Dense vectors where semantic similarity = geometric proximity.
    *   $\text{Vector("King")} - \text{Vector("Man")} + \text{Vector("Woman")} \approx \text{Vector("Queen")}$.
    *   **Insurance:** $\text{Vector("Fire")} \approx \text{Vector("Arson")}$.

---

## 3. Theoretical Properties

### 3.1 Sparsity vs. Density

*   **One-Hot / TF-IDF:** Sparse. Dimension = Vocabulary Size (e.g., 50,000). Most values are 0.
*   **Embeddings:** Dense. Dimension = Fixed (e.g., 300). No zeros.
*   **Benefit:** Embeddings capture *meaning*. TF-IDF only captures *keywords*.

### 3.2 Stop Words & Stemming

*   **Preprocessing:**
    *   **Stop Words:** Remove "the", "is", "at".
    *   **Stemming/Lemmatization:** Convert "driving", "drove", "driven" $\to$ "drive".
*   **Modern NLP:** Often skips this. "The" can be important (e.g., "The car" vs "A car").

---

## 4. Modeling Artifacts & Implementation

### 4.1 TF-IDF Pipeline (Scikit-Learn)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Corpus: Claims Notes
corpus = [
    "Rear-ended at a stop light. Whiplash injury.",
    "Hit a deer on the highway. Front bumper damage.",
    "Tree fell on the roof during the storm.",
    "Customer slipped on ice in the parking lot."
]

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
X = vectorizer.fit_transform(corpus)

# View
df_tfidf = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
print(df_tfidf)
```

### 4.2 Training Word2Vec (Gensim)

```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Tokenize
tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]

# Train
model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)

# Similarity
print("Similarity (Injury, Whiplash):", model.wv.similarity('injury', 'whiplash'))
```

---

## 5. Evaluation & Validation

### 5.1 Topic Modeling (LDA)

*   **Goal:** Discover hidden themes in the notes.
*   **Algorithm:** Latent Dirichlet Allocation (LDA).
*   **Output:**
    *   Topic 1: "Water", "Pipe", "Leak", "Basement" (Water Damage).
    *   Topic 2: "Brake", "Intersection", "Light", "Skid" (Auto Collision).
*   **Validation:** Do the topics make business sense?

### 5.2 Classification Metrics

*   **Task:** Predict "Fraud" (1) vs "Legit" (0) using text.
*   **Metric:** AUC / F1-Score.
*   **Baseline:** Does TF-IDF beat a random guess?

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Domain Adaptation

*   **Problem:** Pre-trained embeddings (Google News) don't know insurance jargon.
    *   "Reserve" means "Money set aside", not "Dinner booking".
*   **Solution:** Train your own embeddings on your claims database.

### 6.2 Data Privacy (PII)

*   **Risk:** Notes contain names, phone numbers, SSNs.
*   **Action:** Named Entity Recognition (NER) to scrub PII *before* training.
    *   "Call John at 555-0199" $\to$ "Call [PERSON] at [PHONE]".

---

## 7. Advanced Topics & Extensions

### 7.1 Sentiment Analysis

*   **Context:** Customer Service Transcripts.
*   **Goal:** Detect angry customers (Churn Risk).
*   **Method:** VADER or Fine-tuned BERT.
*   **Insight:** High negative sentiment in "First Notice of Loss" calls correlates with higher litigation rates.

### 7.2 Named Entity Recognition (NER)

*   **Task:** Extract structured entities.
*   **Entities:** [DATE], [LOCATION], [INJURY_TYPE], [VEHICLE_PART].
*   **Use Case:** Auto-populate the claims system fields from the email description.

---

## 8. Regulatory & Governance Considerations

### 8.1 Bias in Text Models

*   **Risk:** The model learns bias from the adjusters' notes.
    *   If adjusters use different language for different demographics, the model encodes that bias.
*   **Audit:** Check for correlation between "Sentiment Score" and "Race/Gender".

---

## 9. Practical Example

### 9.1 The "Subrogation" Hunter

**Scenario:** Subrogation is when you pay the claim but sue the at-fault party to get money back.
**Problem:** Adjusters miss subrogation opportunities.
**Solution:**
1.  Train a Text Classifier on historical notes where Subrogation was successful.
2.  Keywords: "Third party admitted", "Police cited other driver", "Rear ended".
3.  **Deployment:** Scan all open claims daily. Flag high-probability subrogation cases for review.
**Value:** Recovering 1% more claims = Millions in savings.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **TF-IDF** is the baseline.
2.  **Embeddings** capture meaning.
3.  **Preprocessing** (Cleaning) is 80% of the work.

### 10.2 When to Use This Knowledge
*   **Claims:** Fraud detection, Triage, Subrogation.
*   **Underwriting:** Analyzing submission emails.

### 10.3 Critical Success Factors
1.  **Context:** "Hit by deer" vs "Hit a deer". Order matters (requires n-grams or Deep Learning).
2.  **Spelling:** Adjuster notes are full of typos. You need a robust tokenizer.

### 10.4 Further Reading
*   **Jurafsky & Martin:** "Speech and Language Processing".
*   **Gensim Documentation:** Word2Vec.

---

## Appendix

### A. Glossary
*   **Corpus:** The collection of all documents.
*   **Bag of Words:** Representing text as a count of words, ignoring order.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **TF-IDF** | $tf \times \log(N/df)$ | Keyword Extraction |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
