# Day 40: Final Review - Interview Questions

> **Phase**: 4 - Advanced Topics
> **Week**: 8 - Systems & Capstone
> **Topic**: Behavioral, System Design, and General Knowledge

### 1. Tell me about a challenging ML problem you solved.
**Answer:**
*   (STAR Method): Situation, Task, Action, Result.
*   Focus on the *why*, not just the *what*.
*   Mention trade-offs (Latency vs Accuracy).

### 2. How do you handle Data Drift?
**Answer:**
*   Monitor statistical distribution of inputs (KS-test).
*   Monitor model performance (Accuracy/CTR).
*   Retrain model periodically (Daily/Weekly).
*   Online Learning (Incremental updates).

### 3. Design a YouTube Video Recommendation System.
**Answer:**
*   **Candidate Generation**: Two-Tower model (User History $\to$ Video Embeddings). Retrieve top 500.
*   **Ranking**: Multi-Task Neural Network (Predict Click, Watch Time, Like).
*   **Re-ranking**: Diversity, Freshness filters.

### 4. What is "Active Learning"?
**Answer:**
*   The model selects which samples it wants a human to label (usually low confidence ones).
*   Reduces labeling cost.

### 5. What is "Federated Learning"?
**Answer:**
*   Training on user devices (phones) without sending data to server.
*   Send gradients/weights only.
*   Privacy-preserving.

### 6. How do you debug a model that is not learning?
**Answer:**
*   Overfit on a single batch (Loss should go to 0).
*   Check gradients (Vanishing/Exploding).
*   Check learning rate.
*   Check data labels (are they shuffled correctly?).

### 7. What is "Knowledge Distillation"?
**Answer:**
*   Training a small Student model to mimic a large Teacher model.
*   Loss = Soft Target Loss (KL Div) + Hard Target Loss (Cross Entropy).

### 8. What is "Multi-Task Learning"?
**Answer:**
*   One backbone, multiple heads (e.g., Segmentation + Depth Estimation).
*   Shared representation improves generalization.

### 9. What is "Curriculum Learning"?
**Answer:**
*   Start with easy examples, gradually introduce hard ones.
*   Mimics human learning.

### 10. What is "Self-Supervised Learning"?
**Answer:**
*   Generating labels from the data itself.
*   Masking (BERT/MAE), Contrastive (SimCLR/CLIP), Next Token (GPT).
*   The key to scaling to billions of samples.

### 11. What is "Ensembling"?
**Answer:**
*   Combining predictions from multiple models.
*   Bagging (Random Forest), Boosting (XGBoost), Stacking.
*   Reduces variance.

### 12. What is "Pruning"?
**Answer:**
*   Removing weights to compress model.
*   Lottery Ticket Hypothesis: Dense networks contain sparse subnetworks that train just as well.

### 13. What is "ONNX"?
**Answer:**
*   Open Neural Network Exchange.
*   Standard format to represent models.
*   Allows training in PyTorch, deploying in C#/Java/Web.

### 14. What is "TensorRT"?
**Answer:**
*   NVIDIA's inference optimizer.
*   Layer fusion, precision calibration (INT8), kernel auto-tuning.

### 15. What is "A/B Testing"?
**Answer:**
*   Comparing Model A (Control) and Model B (Treatment) on live traffic.
*   Statistical significance test.

### 16. What is "Feature Store"?
**Answer:**
*   Centralized repository for features.
*   Ensures consistency between Training (Offline) and Serving (Online).
*   Solves training-serving skew.

### 17. What is "Model Registry"?
**Answer:**
*   Version control for models (v1.0, v1.1).
*   Tracks lineage (which data trained this model?).
*   Manages lifecycle (Staging $\to$ Prod).

### 18. What is "Bias-Variance Tradeoff"?
**Answer:**
*   **Bias**: Error due to simplifying assumptions (Underfitting).
*   **Variance**: Error due to sensitivity to noise (Overfitting).
*   Goal: Balance both.

### 19. What is "Regularization"?
**Answer:**
*   Techniques to prevent overfitting.
*   L1/L2, Dropout, Early Stopping, Data Augmentation, Batch Norm.

### 20. Where do you see AI in 5 years?
**Answer:**
*   (Open ended).
*   Personalized Agents.
*   Scientific Discovery (AlphaFold).
*   Multimodal understanding of the physical world.
