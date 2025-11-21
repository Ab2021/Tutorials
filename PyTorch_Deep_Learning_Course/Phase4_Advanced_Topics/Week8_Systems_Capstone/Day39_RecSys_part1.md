# Day 39: RecSys - Deep Dive

> **Phase**: 4 - Advanced Topics
> **Week**: 8 - Systems & Capstone
> **Topic**: Wide & Deep, DCN, and Sequential RecSys

## 1. Wide & Deep Learning (Google)

Combines Memorization and Generalization.
*   **Wide Component**: Linear model with Cross-Product features (e.g., "User=NY AND Item=Pizza"). Memorizes co-occurrences.
*   **Deep Component**: DNN. Generalizes to unseen combinations via embeddings.
*   $P(y|x) = \sigma(W_{wide}^T [x, \phi(x)] + W_{deep}^T a^{(lf)} + b)$.

## 2. Deep & Cross Network (DCN)

DNNs are bad at learning explicit feature crosses (multiplicative interactions).
**Cross Network**:
$$ x_{l+1} = x_0 x_l^T w_l + b_l + x_l $$
*   Explicitly models feature interactions of degree $l+1$.
*   Linear complexity.

## 3. Sequential Recommendation (SASRec)

User preference changes over time.
**SASRec (Self-Attentive Sequential Recommendation)**:
*   Treat user history as a sequence of Item IDs.
*   Use a Transformer Encoder (like BERT) to predict next item.
*   Outperforms RNN/CNN based approaches (GRU4Rec).

## 4. Negative Sampling

We have only positive feedback (Clicks).
We need negatives to train.
*   **Random Negative**: Pick random item. Easy.
*   **Hard Negative**: Pick item that user *almost* liked (high score) but didn't click.
*   Crucial for discriminative power.

## 5. Metrics

*   **Recall@K**: Did the relevant item appear in top K?
*   **NDCG@K**: Normalized Discounted Cumulative Gain. Position matters (Top 1 is better than Top 10).
*   **AUC**: Probability that positive sample is ranked higher than negative sample.
