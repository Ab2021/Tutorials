# Day 47: Mock Interview: Visual Search - Interview Questions

> **Topic**: Pinterest / Google Lens
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. Design a Visual Search System (Pinterest Lens).
**Answer:**
*   **Goal**: User uploads image -> Find similar products.
*   **Flow**: Object Detection -> Embedding -> Vector Search -> Ranking.

### 2. How do you generate Image Embeddings?
**Answer:**
*   **CNN**: ResNet/EfficientNet. Take output of penultimate layer.
*   **Transformer**: ViT.
*   **Contrastive**: CLIP (trained on Image-Text pairs).

### 3. How do you handle "Object Detection"?
**Answer:**
*   Image might contain multiple items (Shirt, Shoes, Bag).
*   Run YOLO/Faster R-CNN to crop objects.
*   Search for each crop.

### 4. What is "Metric Learning"?
**Answer:**
*   Training model such that similar images are close, dissimilar are far.
*   **Triplet Loss**: Anchor, Positive, Negative. $d(A, P) + margin < d(A, N)$.

### 5. How do you handle "Cross-Modal" search?
**Answer:**
*   Text query "Red floral dress" -> Image results.
*   **CLIP**: Maps text and image to same vector space.

### 6. How do you scale to billions of images?
**Answer:**
*   **Vector DB**: Milvus/Pinecone.
*   **IVF-PQ**: Quantization for memory efficiency.

### 7. How do you handle "Exact Match" vs "Style Match"?
**Answer:**
*   **Exact**: Hash (Perceptual Hash).
*   **Style**: Embeddings.

### 8. What is "Re-ranking" in Visual Search?
**Answer:**
*   Initial retrieval based on visual similarity.
*   Re-rank based on: Availability, Price, User Gender, Popularity.

### 9. How do you evaluate Visual Search?
**Answer:**
*   **Recall@K**: Is the correct product in top K?
*   **Visual Similarity**: Human judgment.

### 10. What is "Category Prediction"?
**Answer:**
*   Classify image first (e.g., "Shoe").
*   Restrict search to "Shoe" partition in DB.
*   Improves speed and accuracy.

### 11. How do you handle "Occlusion" or "Poor Lighting"?
**Answer:**
*   Data Augmentation during training (Random Erasing, Color Jitter).
*   Robust models.

### 12. What is "Shop the Look"?
**Answer:**
*   Graph problem.
*   Node: Fashion Item. Edge: "Goes well with".
*   GNN to recommend complementary items.

### 13. How do you handle "User Uploads" (Privacy)?
**Answer:**
*   Process on-device if possible.
*   Delete immediately after search.

### 14. What is "Negative Sampling" for Triplet Loss?
**Answer:**
*   **Hard Negatives**: Images that look similar but are different class.
*   Crucial for learning fine-grained differences.

### 15. How do you update the index?
**Answer:**
*   New products added daily.
*   Batch update nightly.
*   Real-time insert for Vector DB.

### 16. What is "Hashing" for images?
**Answer:**
*   **LSH (Locality Sensitive Hashing)**.
*   Maps similar vectors to same bucket.
*   Fast approximate search.

### 17. How do you handle "Rotation"?
**Answer:**
*   CNNs are not rotation invariant.
*   Augmentation (Rotate 90, 180).
*   Spatial Transformer Networks.

### 18. What is the "Representation Gap"?
**Answer:**
*   User photo (Selfie) vs Catalog photo (Studio).
*   Domain Adaptation needed.

### 19. How do you optimize Latency?
**Answer:**
*   Resize image on client.
*   Quantize model (INT8).
*   Cascade: Fast hash search -> Slow vector search.

### 20. What is "Multimodal Fusion"?
**Answer:**
*   Combine Image Embedding + Text Embedding (Description).
*   Concat or Attention.
