# Day 29: Case Study - Search & Ranking (Google/Amazon)

> **Phase**: 3 - System Design
> **Week**: 5 - Design Principles (Case Studies)
> **Focus**: Query-Document Matching
> **Reading Time**: 60 mins

---

## 1. The Architecture

Search is similar to Recommendation, but driven by an explicit **Query**.

### 1.1 Query Understanding (NLU)
Before searching, we must understand the user.
*   **Spell Check**: "ipone" -> "iphone".
*   **Entity Recognition**: "Nike shoes red" -> Brand: Nike, Category: Shoes, Color: Red.
*   **Query Expansion**: "Running shoes" -> Add synonyms "Jogging sneakers".

### 1.2 Retrieval (Inverted Index)
*   **Structure**: Map `Token -> List[Document IDs]`.
*   **Algorithm**: BM25 (TF-IDF variant). Fast retrieval of documents containing query terms.
*   **Semantic Search**: Vector Search (BERT embeddings) to find documents that match *meaning* but not keywords.

### 1.3 Ranking (Learning to Rank - LTR)
*   **Pointwise**: Regress score for (Query, Doc). (Like standard classification).
*   **Pairwise**: Classify "Is Doc A better than Doc B?". (RankNet, LambdaMART).
*   **Listwise**: Optimize the entire list order (NDCG) directly. (ListNet).

---

## 2. Deep Dive: LambdaMART

The industry standard for ranking (used by Bing, Yandex).
*   **Gradient Boosting** on Decision Trees.
*   **Loss**: Modified gradients that push documents up/down to maximize NDCG.

---

## 3. Real-World Challenges & Solutions

### Challenge 1: Long Tail Queries
**Scenario**: 20% of queries have never been seen before.
**Solution**:
*   **Semantic Search**: Embeddings capture meaning. "Cheap phone" vector is close to "Budget smartphone" vector.
*   **Relaxation**: If "Nike Red Shoes Size 10" returns 0 results, drop "Size 10" and show "Nike Red Shoes".

### Challenge 2: Latency
**Scenario**: User types "a"... "ap"... "app"... (Autocomplete). Must respond in < 50ms.
**Solution**:
*   **Trie Data Structure**: For prefix lookup.
*   **Caching**: Cache top 1000 queries.
*   **Edge Computing**: Run simple logic on CDN/Client.

---

## 4. Interview Preparation

### System Design Questions

**Q1: Explain NDCG (Normalized Discounted Cumulative Gain).**
> **Answer**:
> *   **CG**: Sum of relevance scores.
> *   **DCG**: Relevance scores penalized by position (logarithmic decay). A relevant item at rank 1 is worth more than at rank 10.
> *   **NDCG**: DCG divided by the Ideal DCG (IDCG) - the score of the perfect ordering. Range [0, 1]. Allows comparing queries with different numbers of results.

**Q2: How do you combine Keyword Search (BM25) and Semantic Search (Vectors)?**
> **Answer**: **Hybrid Search**.
> *   Run BM25 to get Top 100 (Exact matches).
> *   Run Vector Search to get Top 100 (Semantic matches).
> *   Merge lists using **Reciprocal Rank Fusion (RRF)**: Score = $1 / (k + \text{rank}_{bm25}) + 1 / (k + \text{rank}_{vec})$.

**Q3: How do you handle personalization in Search?**
> **Answer**:
> *   Add User Features to the LTR model.
> *   Query: "Apple".
> *   User A (Techie): Show iPhone.
> *   User B (Cook): Show Fruit.
> *   The model learns interactions between `Query="Apple"` and `UserInterest="Tech"`.

---

## 5. Further Reading
- [Elasticsearch: The Definitive Guide](https://www.elastic.co/guide/en/elasticsearch/guide/current/index.html)
- [Learning to Rank: From Pairwise to Listwise](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf)
