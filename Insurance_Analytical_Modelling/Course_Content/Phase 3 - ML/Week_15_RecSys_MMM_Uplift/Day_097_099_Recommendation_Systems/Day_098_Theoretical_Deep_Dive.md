# Recommendation Systems (Part 2) - Advanced Deep Learning Architectures - Theoretical Deep Dive

## Overview
"When the matrix is too sparse, the Neural Network takes over."
Building on Day 97's fundamentals, Day 98 explores **Deep Learning for Recommendation Systems**.
We dive into **Neural Collaborative Filtering (NCF)**, **Session-Based RNNs**, and **Contextual Bandits**.
These advanced models capture non-linear interactions and temporal dynamics that traditional SVD misses.

---

## 1. Conceptual Foundation

### 1.1 Why Deep Learning for RecSys?

*   **Non-Linearity:** SVD assumes a linear interaction (Dot Product). Neural Networks learn complex, non-linear functions.
*   **Feature Fusion:** Easily combine ID embeddings (User ID) with dense features (Age, Income) and unstructured data (Claim Notes text).
*   **Sequence Modeling:** RNNs/Transformers capture the *order* of actions (Quote -> Click -> Bind), which is crucial for "Next Best Action".

### 1.2 The Evolution

1.  **Gen 1:** Collaborative Filtering (SVD).
2.  **Gen 2:** Factorization Machines (FM).
3.  **Gen 3:** Neural Collaborative Filtering (NCF).
4.  **Gen 4:** Sequence Models (BERT4Rec, SASRec).

---

## 2. Mathematical Framework

### 2.1 Neural Collaborative Filtering (NCF)

*   **Architecture:**
    *   Input: User ID, Item ID (One-Hot).
    *   Embedding Layer: Maps IDs to dense vectors.
    *   GMF Layer: Generalized Matrix Factorization (Element-wise product).
    *   MLP Layer: Multi-Layer Perceptron (Concatenation + Dense Layers).
    *   Output: Sigmoid (Probability of Interaction).
*   **Advantage:** The MLP can learn any interaction function, not just dot product.

### 2.2 Contextual Bandits (Thompson Sampling)

*   **Problem:** Explore-Exploit. Should we show the "Best" product (Exploit) or a "New" product (Explore)?
*   **Algorithm:**
    *   Maintain a distribution of expected reward for each arm (product).
    *   Sample from the distribution.
    *   Choose the arm with the highest sample.
    *   Update the distribution based on the user's response (Click/No Click).

---

## 3. Theoretical Properties

### 3.1 The "Wide & Deep" Architecture (Google)

*   **Wide Component:** Memorization. (Linear model). Captures specific co-occurrences ("User A bought Product B").
*   **Deep Component:** Generalization. (Neural Network). Captures abstract patterns ("Young users like Tech products").
*   **Insurance Use:**
    *   *Wide:* "Drivers in Zip 90210 buy High Limits."
    *   *Deep:* "People with high credit scores generally buy Umbrella policies."

### 3.2 Session-Based Recommendations

*   **Context:** Anonymous user on the website. No User ID.
*   **Data:** Sequence of clicks in the current session.
*   **Model:** LSTM or GRU.
*   **Prediction:** $P(Item_{t+1} | Item_1, ..., Item_t)$.

---

## 4. Modeling Artifacts & Implementation

### 4.1 NCF Implementation (TensorFlow/Keras)

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense

# Inputs
user_input = Input(shape=(1,), name='user_input')
item_input = Input(shape=(1,), name='item_input')

# Embeddings
user_embed = Embedding(num_users, 50)(user_input)
item_embed = Embedding(num_items, 50)(item_input)

# Flatten
user_vec = Flatten()(user_embed)
item_vec = Flatten()(item_embed)

# Concatenate & MLP
concat = Concatenate()([user_vec, item_vec])
dense1 = Dense(128, activation='relu')(concat)
dense2 = Dense(64, activation='relu')(dense1)
output = Dense(1, activation='sigmoid')(dense2)

model = tf.keras.Model([user_input, item_input], output)
model.compile(optimizer='adam', loss='binary_crossentropy')
```

### 4.2 Contextual Bandit Logic

```python
# Simplified Thompson Sampling
import numpy as np

def choose_arm(context_vector):
    best_arm = -1
    max_sample = -1
    
    for arm in arms:
        # Predict mean and variance for this arm given context
        mu, sigma = model[arm].predict(context_vector)
        
        # Sample
        sample = np.random.normal(mu, sigma)
        
        if sample > max_sample:
            max_sample = sample
            best_arm = arm
            
    return best_arm
```

---

## 5. Evaluation & Validation

### 5.1 Offline Replay (Counterfactual Evaluation)

*   **Problem:** We can't A/B test every bandit algorithm live.
*   **Method:** Replay historical logs.
    *   If the Bandit chooses the *same* arm as the historical log, we can observe the reward.
    *   If it chooses a *different* arm, we ignore that data point (or use Inverse Propensity Scoring).

### 5.2 Hit Rate @ K

*   **Metric:** Is the ground-truth item present in the Top-K recommendations?
*   **Relevance:** Crucial for "Top 3 Picks" widgets on the dashboard.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Latency in Deep Learning

*   **Issue:** NCF inference is slower than SVD (Dot Product).
*   **Fix:**
    *   **Two-Tower Architecture:** Pre-compute User and Item embeddings. Use Approximate Nearest Neighbor (ANN) search (FAISS) for retrieval.
    *   Only run the heavy MLP on the top 100 candidates (Re-ranking).

### 6.2 The "Harry Potter" Effect (Popularity Bias)

*   **Issue:** The model learns to just recommend the most popular product (Auto Insurance) to everyone.
*   **Fix:** Down-sample popular items during training. Use "Popularity-Adjusted" metrics.

---

## 7. Advanced Topics & Extensions

### 7.1 Graph Neural Networks (GNNs) for RecSys

*   **Concept:** PinSage / GraphSAGE.
*   **Graph:** Users and Items are nodes. Interactions are edges.
*   **Convolution:** Aggregate information from neighbors. "Tell me about a user by looking at the products they bought *and* the users who bought those products."

### 7.2 Reinforcement Learning (RL) for LTV

*   **Goal:** Optimize Long-Term Value, not just immediate click.
*   **Agent:** The Recommender.
*   **State:** User history.
*   **Action:** Recommend Product.
*   **Reward:** Discounted future premiums.

---

## 8. Regulatory & Governance Considerations

### 8.1 Explainable AI (XAI) for NCF

*   **Challenge:** Neural Nets are black boxes. "Why did you recommend this?"
*   **Solution:** Integrated Gradients or SHAP.
*   **Output:** "Recommended because you have feature 'Homeowner' and similar users bought 'Umbrella'."

---

## 9. Practical Example

### 9.1 The "Quote Funnel" Accelerator

**Scenario:** User is filling out an Auto Quote.
**Data:** They just entered "Vehicle = Tesla Model 3".
**Model (Session RNN):**
1.  **Input:** Sequence [Zip, Age, Tesla].
2.  **Prediction:** High probability of interest in "Gap Insurance" and "EV Charging Station Coverage".
3.  **Action:** Dynamically insert a "EV Package" checkbox in the next screen.
**Result:** 15% increase in add-on attachment rate.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **NCF** generalizes Matrix Factorization.
2.  **RNNs** handle sequential/session data.
3.  **Bandits** balance exploration and exploitation.

### 10.2 When to Use This Knowledge
*   **High Volume:** When you have millions of interactions (Big Data).
*   **Real-Time:** When the user intent changes rapidly during a session.

### 10.3 Critical Success Factors
1.  **Infrastructure:** You need GPUs for training and low-latency serving (TensorRT).
2.  **Data Quality:** Sequence models are very sensitive to noise in the clickstream.

### 10.4 Further Reading
*   **He et al.:** "Neural Collaborative Filtering" (WWW 2017).
*   **Covington et al.:** "Deep Neural Networks for YouTube Recommendations".

---

## Appendix

### A. Glossary
*   **Embedding:** A dense vector representation of a categorical variable.
*   **One-Hot:** A sparse vector with a single 1 and all other 0s.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Sigmoid** | $1 / (1 + e^{-x})$ | Probability Output |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
