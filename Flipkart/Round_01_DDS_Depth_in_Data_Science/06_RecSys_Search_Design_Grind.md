# 🔍 DDS GRIND — Recommender Systems & Search Ranking (Questions 76-100)
### Critical designs for E-commerce

> Flipkart is fundamentally a search and recommendation engine. Mastering Two-Tower models and Learning-to-Rank is non-negotiable.

---

## ═══════════════════════════════════════
## SECTION K: TWO-TOWER & CANDIDATE RETRIEVAL
## ═══════════════════════════════════════

### Q76: Design a Two-Tower Candidate Generation model for Flipkart Home Page. 
**Expected Answer:**
**Architecture:**
- **User Tower:** Input = User features (age, location, last_10_clicked_categories, last_3_purchases). Deep residual layers output a vector $U \in \mathbb{R}^{128}$.
- **Item Tower:** Input = Item features (price, category, title BERT embedding, brand). Deep residual layers output a vector $I \in \mathbb{R}^{128}$.
- **Loss Function:** In-batch Sampled Softmax loss or Triplet/Contrastive Loss. Maximize cosine similarity $U \cdot I$ for positive interactions (clicks/buys), minimize for random negatives.
**Serving:**
1. Compute Item embeddings offline for all 100M products. Store in FAISS/Milvus HNSW index.
2. At runtime, User Tower computes $U$ vector dynamically.
3. ANN search in FAISS retrieves top 500 nearest products in <10ms.

### Q77: What is the "In-batch Negative Selection" problem in Two-Tower models, and what's the "Popularity Bias" fix?
**Expected Answer:**
**Problem:** In contrastive learning, you use other items in the same batch as negative examples. However, items in the batch are usually sampled from interaction logs (clicks). 
Therefore, highly popular items appear in the batch frequently and are used as negatives disproportionately. The model learns to aggressively down-weight popular items.
**Fix (LogQ Correction):**
Modify the softmax logits by subtracting the log expected frequency (popularity) of the item $j$:
$$\text{Logit}(u_i, v_j) = u_i \cdot v_j - \log p(j)$$
This removes the implicit penalty on popular items, allowing the embeddings to learn true semantic matching rather than just inverse popularity.

### Q78: Explain the mechanics of Deep Crossed Networks (DCN) or DeepFM for the Re-ranking stage.
**Expected Answer:**
While Two-Towers are fast for retrieval (dot product), re-ranking needs to model complex interactions (e.g., User likes Nike AND Item is running shoe).
**DeepFM (Factorization Machine + Deep Neural Net):**
- **FM Component:** Explicitly models all 2nd-order feature interactions perfectly. $\sum \sum \langle v_i, v_j \rangle x_i x_j$. Good for memorization.
- **Deep Component:** Feedforward MLPs capturing higher-order non-linear patterns. Good for generalization.
- Output logits are summed: $y = \text{sigmoid}(y_{FM} + y_{DNN})$.
**Advantage:** Replaces manual feature cross engineering (user_city_x_item_category).

---

## ═══════════════════════════════════════
## SECTION L: LEARNING TO RANK (LTR)
## ═══════════════════════════════════════

### Q79: Explain Pointwise, Pairwise, and Listwise approaches to Learning to Rank.
**Expected Answer:**
- **Pointwise:** Transforms ranking into a binary classification or regression problem. (Predict probability of click). **Loss:** LogLoss. Ignores relative order.
- **Pairwise (e.g., RankNet):** Learns the relative order of item pairs. Input is (user, item1, item2). Predicts $P(\text{item1} > \text{item2})$. **Loss:** Hinge loss or pair logistic. Better, but treats all pairs equally (swapping rank 1 and 2 has same penalty as rank 100 and 101).
- **Listwise (e.g., LambdaMART):** Directly optimizes an IR metric like NDCG across the entire list.
  
### Q80: How does LambdaMART work? (Explain the "Lambda")
**Expected Answer:**
NDCG implies sorting, which is non-differentiable. You can't compute gradients.
**LambdaMART genius:** You don't need the loss function, you only need the *gradients* to train the trees.
It defines a "virtual gradient" (the $\lambda$) for each item.
$\lambda_{ij}$ represents the "force" pushing item $i$ up and item $j$ down.
The magnitude of $\lambda_{ij}$ is scaled by the change in NDCG ($\Delta \text{NDCG}$) that would occur if item $i$ and item $j$ were swapped.
If swapping a clicked item at rank 10 with a non-clicked item at rank 2 massively improves NDCG, the gradient pushing rank 10 up is huge. Uses Gradient Boosted Trees (MART) to fit these lambda gradients.

---

## ═══════════════════════════════════════
## SECTION M: EXPLORATION & COLD START
## ═══════════════════════════════════════

### Q81: How do you handle cold-start items (newly listed sellers/products) on Flipkart?
**Expected Answer:**
New items have no click history, so CF and Two-Towers ignore them.
1. **Heuristic/Content-based insertion:** Calculate item similarity based on text/image purely. Inject them randomly in slots 10-20.
2. **Epsilon-Greedy Bandit:** With probability $\epsilon$ (e.g., 5%), replace a recommendation with a cold-start item.
3. **Upper Confidence Bound (UCB):** Use an arm selection criteria: $Score = \mu_i + \alpha \sqrt{\frac{\ln N}{n_i}}$.
   - $n_i$ is number of times item $i$ was shown. 
   - For new items, $n_i$ is 0, so the exploration bonus (square root term) explodes to infinity, forcing the system to show it until uncertainty drops.

*End of RecSys Grind.*
