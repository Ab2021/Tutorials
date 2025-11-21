# Day 15: Search Engines

## 1. The Problem
*   **Database:** `SELECT * FROM table WHERE content LIKE '%term%'`.
*   **Issue:** Full table scan. Slow. No ranking.

## 2. Inverted Index
*   **Concept:** Map Words -> Documents. (Like the index at the back of a book).
*   **Structure:**
    *   `apple` -> `[Doc1, Doc3]`
    *   `banana` -> `[Doc2]`
    *   `code` -> `[Doc1, Doc2, Doc3]`
*   **Query:** "apple AND code".
    *   Intersect `[1, 3]` and `[1, 2, 3]`.
    *   Result: `[Doc1, Doc3]`.

## 3. Tokenization & Analysis
*   **Normalization:** `Running` -> `run` (Stemming/Lemmatization).
*   **Stop Words:** Remove `the`, `is`, `at`.
*   **N-Grams:** `New York` -> `New`, `York`, `New York`.

## 4. Ranking (TF-IDF)
*   **TF (Term Frequency):** How often word appears in Doc.
*   **IDF (Inverse Document Frequency):** How rare the word is globally.
*   **Score:** $TF \times IDF$.
*   **Modern:** BM25 (Best Matching 25).

## 5. Elasticsearch / Solr / Lucene
*   **Lucene:** The core Java library that builds the index.
*   **Elasticsearch:** Distributed engine built on Lucene. Handles sharding, replication, API.
