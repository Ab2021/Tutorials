# Day 29: Case Study: Search System - Interview Questions

> **Topic**: Information Retrieval
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. Design a Search Engine (Google/Amazon).
**Answer:**
*   **Components**: Crawler -> Indexer -> Query Processor -> Retrieval -> Ranking.

### 2. What is an Inverted Index?
**Answer:**
*   Map: Word -> List of Document IDs (Postings List).
*   Allows $O(1)$ lookup of documents containing a term.

### 3. Explain TF-IDF.
**Answer:**
*   **TF**: Term Frequency. How often word appears in doc.
*   **IDF**: Inverse Document Frequency. $\log(N/df)$. Penalizes common words (the, is).
*   Importance = TF * IDF.

### 4. What is BM25?
**Answer:**
*   Improvement over TF-IDF.
*   Saturates TF (diminishing returns).
*   Normalizes by document length (short docs match better).
*   Standard baseline for retrieval.

### 5. What is Semantic Search?
**Answer:**
*   Searching by **meaning**, not just keyword match.
*   "Shoes for running" matches "Athletic sneakers".
*   Uses Dense Embeddings (BERT) + Vector DB.

### 6. Explain the Dual Encoder (Bi-Encoder) for Search.
**Answer:**
*   Pre-compute Doc Embeddings.
*   Compute Query Embedding live.
*   Cosine Similarity. Fast retrieval.

### 7. Explain the Cross-Encoder for Re-ranking.
**Answer:**
*   Input: `[CLS] Query [SEP] Document`.
*   Output: Relevance Score.
*   More accurate (captures interaction) but slow. Used on top 50 results.

### 8. How do you handle Query Expansion?
**Answer:**
*   User types "sneakers".
*   Expand to: "sneakers OR shoes OR trainers OR running shoes".
*   Increases Recall.

### 9. What is "Learning to Rank" (LTR)?
**Answer:**
*   Supervised ML to rank results.
*   **Pointwise**: Predict score for each doc. (Regression).
*   **Pairwise**: Predict if Doc A > Doc B. (RankNet).
*   **Listwise**: Optimize entire list metric (NDCG). (LambdaMART).

### 10. How do you handle "Did you mean" (Spell Correction)?
**Answer:**
*   Edit Distance (Levenshtein).
*   N-gram overlap.
*   Language Model probability.

### 11. What is "Personalized Search"?
**Answer:**
*   Using user history to boost results.
*   "Java" -> Coffee (for barista) vs Code (for dev).
*   Add User Embedding to Ranking model.

### 12. How do you measure Search Relevance?
**Answer:**
*   **Offline**: Human raters (0-4 scale). NDCG.
*   **Online**: Click-Through Rate (CTR), Pogo-sticking (Click & Back quickly = Bad).

### 13. What is "Pogo-sticking"?
**Answer:**
*   User clicks result, immediately goes back, clicks next result.
*   Strong signal of dissatisfaction.

### 14. How do you handle Synonyms?
**Answer:**
*   Knowledge Graph.
*   Word Embeddings (Cosine sim).
*   Query rewriting rules.

### 15. What is "Autocomplete" (Typeahead)? Design it.
**Answer:**
*   **Trie** data structure.
*   Store top K popular queries at each node.
*   Latency requirement < 20ms.

### 16. Explain HNSW (Hierarchical Navigable Small World).
**Answer:**
*   Graph-based ANN algorithm.
*   Layers of graphs. Top layer has long links (highway). Bottom layer has short links (local).
*   Fast greedy search.

### 17. What is "Hybrid Search"?
**Answer:**
*   Combining Keyword Search (BM25) + Semantic Search (Vectors).
*   **Reciprocal Rank Fusion (RRF)** to merge lists.
*   Best of both worlds (Exact match + Understanding).

### 18. How do you handle Multi-modal Search (Text to Image)?
**Answer:**
*   **CLIP** (Contrastive Language-Image Pretraining).
*   Embed text and image into same space.
*   Search by distance.

### 19. What is "Query Understanding"?
**Answer:**
*   Parsing query before search.
*   **NER**: "Nike shoes" -> Brand: Nike, Category: Shoes.
*   **Intent Classification**: Buy vs Info.

### 20. How do you scale the Index?
**Answer:**
*   **Sharding**: Split docs across machines.
*   **Replication**: For high QPS.
*   **Scatter-Gather**: Query all shards, merge results.
