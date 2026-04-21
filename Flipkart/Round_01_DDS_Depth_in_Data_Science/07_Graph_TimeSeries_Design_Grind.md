# 🕸️ DDS GRIND — Graph Neural Networks & Time Series Demand Forecasting
### Depth in Data Science (Questions 76-100)

> Flipkart handles massive supply chain logistics (Time Series) and adversarial seller networks (GNNs).

---

## ═══════════════════════════════════════
## SECTION N: GRAPH NEURAL NETWORKS (FRaud & RecSys)
## ═══════════════════════════════════════

### Q76: How do you formulate Seller Fraud Ring detection as a Link Prediction vs. Node Classification problem using GNNs?
**Expected Answer:**
- **Node Classification:**
  - **Goal:** Predict a binary probability $P(\text{fraud})$ for every seller node.
  - **Inputs:** Node features (return rate, average age) + Adjacency Matrix (edges = shared device).
  - **Use Case:** When you have labeled historical fraudster nodes and want to find new ones based on their connections to known bad actors (Label Propagation).
- **Link Prediction:**
  - **Goal:** Predict the probability of an edge existing between two nodes $P(e_{ij} = 1)$.
  - **Use Case:** Predicting a "collusion" edge. You might train the model to predict known colluding sellers, then infer missing edges between seemingly disconnected sellers to uncover hidden rings.

### Q77: Explain the scalability problem of strictly applying Graph Convolutional Networks (GCN) on a 10M-node graph, and how GraphSAGE solves it.
**Expected Answer:**
**GCN Problem (Full-Batch):**
GCN updates a node's embedding by multiplying the entire Adjacency matrix $A$ with the Feature matrix $X$. Finding neighborhood requires loading the entire graph into memory, which crashes for 10M nodes on GPU.
**GraphSAGE (Sampling & Aggregation) Solution:**
Instead of operating on the whole graph (Transductive), GraphSAGE operates locally (Inductive).
1. **Sampling:** For a target node, randomly sample a fixed number (e.g., $K=10$) of neighbors in hop 1, and $K=10$ neighbors of those in hop 2. This bounds computation.
2. **Aggregation:** Pull features from those sampled neighbors via aggregator functions (Mean, Max-Pooling, or LSTM) rather than matrix multiplication.
This allows mini-batch training (loading only subgraphs into GPU memory).

### Q78: In a Graph Attention Network (GAT), how is the attention mechanism mathematically applied differently than a Transformer?
**Expected Answer:**
In a Transformer, self-attention works between ALL tokens in a sequence ($\text{Softmax}(QK^T / \dots)$). The graph is implicitly fully connected.
In a **GAT**, attention is strictly masked by the topological structure of the graph. A node ONLY computes attention scores with its direct first-degree neighbors.
**Math:** 
$$\alpha_{ij} = \text{Softmax}_j (\text{LeakyReLU}(a^T [W h_i || W h_j]))$$
Where $\alpha_{ij}$ is the attention weight of node $i$ to its neighbor $j$. It learns *which* neighbors are most important (e.g., ignore the neighbor who is a massive aggregator, focus on the suspicious small seller neighbor).

---

## ═══════════════════════════════════════
## SECTION O: TIME SERIES & SUPPLY CHAIN FORECASTING
## ═══════════════════════════════════════

### Q79: Design a Demand Forecasting system for 1 Million SKUs to predict daily sales for the next 30 days.
**Expected Answer:**
**Problem:** Cannot train 1 million separate ARIMA models. Need a global model.
**Architecture: Seq2Seq with Temporal Fusion Transformer (TFT) or LightGBM panel approach.**
**Feature Engineering:**
1. **Target Lags:** Sales $t-1, t-7, t-30$, rolling mean 7d, rolling std 7d.
2. **Static Metadata:** Product category, brand, color, weight.
3. **Known Future Inputs:** Price markdowns, Holiday flags (Diwali), promotional banners.
4. **Time features:** Day of week, Month, Days to next major holiday.
**Model:**
LightGBM with Tweedie or Poisson objective (since sales are count data with lots of zeros). Train one global model across all SKUs. 
Predict sequentially: Predict $Y_{t+1}$, use it as lag feature to predict $Y_{t+2}$.

### Q80: How do you handle Intermittent Demand (spiky sales, mostly zeros) in forecasting?
**Expected Answer:**
Standard MSE loss on LightGBM will predict "0.1" every day. This is mathematically optimal for MSE but useless for inventory.
**Solutions:**
1. **Croston's Method:** Forecasts two components separately: 
   - Probability of a demand event occurring.
   - Size of the demand if it occurs.
2. **Tweedie Loss (in XGBoost/LightGBM):** A parameterized loss function suited for data with severe zero-inflation and a right-skewed continuous tail. Setting Tweedie variance power $1 < p < 2$ (e.g., 1.5) handles the zero-mass perfectly.
3. **Quantile Regression:** Instead of predicting the mean, predict the 90th percentile of demand (to ensure you stock enough inventory). Loss function = Pinball Loss.

### Q81: What is Data Leakage in Time Series models, and how do you implement Cross Validation correctly?
**Expected Answer:**
**Leakage:** Randomly splitting data (K-Fold CV) implies predicting $t=10$ using $t=12$ as training data. You leaked the future.
**Correct CV (Time Series Split / Rolling Origin):**
1. Train on month 1..6, Validate on month 7.
2. Train on month 1..7, Validate on month 8.
3. Train on month 1..8, Validate on month 9.
This mirrors exactly how the model will be used in production (moving forward in time).
**Gap strategy:** If your forecasting horizon is 30 days, leave a 30-day "gap" between train and validation to ensure you aren't leaking lag features.

---
*End of GNN & Time Series Grind.*
