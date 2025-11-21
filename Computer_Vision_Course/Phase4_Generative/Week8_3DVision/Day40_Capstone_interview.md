# Day 40 Interview Questions: System Design

## Q1: Design an Image Search Engine (like Google Images).
**Answer:**
1.  **Indexing:**
    *   Crawl images.
    *   Pass through **CLIP Image Encoder** to get embeddings.
    *   Store embeddings in a Vector Database (FAISS, Pinecone).
2.  **Querying:**
    *   User types text.
    *   Pass through **CLIP Text Encoder**.
    *   Perform Approximate Nearest Neighbor (ANN) search in DB.
3.  **Ranking:** Re-rank top 100 results using a heavier model or user metadata.

## Q2: Design a Content Moderation System.
**Answer:**
1.  **Filter:** Hash matching (MD5) for known bad images.
2.  **Classifier:** Lightweight CNN (MobileNet) to detect NSFW/Violence. High recall, low precision.
3.  **Review:** Flagged images go to human review or a heavy VLM (Flamingo) for detailed analysis.
4.  **Latency:** Must be <100ms. Use model quantization and batching.

## Q3: How would you serve Stable Diffusion to 1 Million users?
**Answer:**
*   **Compute:** GPUs are expensive. Cannot run 1 instance per user.
*   **Queueing:** Use a message queue (RabbitMQ/Kafka). Users submit jobs, workers pick them up.
*   **Batching:** Dynamic batching (group prompts together) to maximize GPU utilization.
*   **Caching:** Cache results for identical prompts.
*   **Distillation:** Use a distilled model (SD-Turbo) that runs in 1 step instead of 50.

## Q4: How to handle "Data Drift" in production?
**Answer:**
*   **Monitor:** Track distribution of input images (e.g., brightness, class distribution).
*   **Detect:** If distribution shifts (e.g., users start uploading night photos instead of day), accuracy drops.
*   **Retrain:** Trigger a pipeline to fine-tune the model on the new data.

## Q5: What is "Model Quantization"?
**Answer:**
*   Reducing the precision of weights/activations (Float32 $\to$ Int8).
*   **Post-Training Quantization (PTQ):** Calibrate on a small dataset after training.
*   **Quantization-Aware Training (QAT):** Simulate quantization errors during training.
*   Benefits: 4x smaller model, faster inference, lower energy.

## Q6: Explain "A/B Testing" for ML models.
**Answer:**
*   Deploy Model A (Current) to 90% of traffic.
*   Deploy Model B (New) to 10% of traffic.
*   Compare metrics (CTR, Conversion, Latency).
*   If B > A statistically, roll out B to 100%.

## Q7: How to prevent your GenAI model from generating harmful content?
**Answer:**
*   **Input Filtering:** Check prompt for banned words.
*   **Embedding Filtering:** Check if prompt embedding is close to "hate speech" cluster.
*   **Output Filtering:** Run a safety classifier on the generated image.
*   **RLHF:** Train the model to refuse harmful prompts.

## Q8: What is "FlashAttention"?
**Answer:**
*   An IO-aware exact attention algorithm.
*   It tiles the computation to keep data in the fast GPU SRAM (L1 cache), minimizing reads/writes to slow HBM (High Bandwidth Memory).
*   Speeds up training/inference of Transformers by 2-4x and reduces memory from quadratic to linear.

## Q9: How do you debug a silent failure in a CV pipeline?
**Answer:**
*   **Logging:** Log inputs, outputs, and intermediate tensors (histograms).
*   **Visualizing:** Save sample images with bounding boxes drawn.
*   **Unit Tests:** Test individual components (e.g., "Does the augmentation flip labels correctly?").
*   **Golden Set:** Run a fixed set of difficult examples daily to check for regression.

## Q10: Final Question: What is the most important skill for an AI Engineer?
**Answer:**
*   **Adaptability.** The field changes every week.
*   The ability to read a paper (arXiv), understand the code (GitHub), and integrate it into a product is more valuable than memorizing formulas.
