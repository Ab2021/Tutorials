# Day 47 (Part 1): Advanced Mock - Visual Search

> **Phase**: 6 - Deep Dive
> **Topic**: Computer Vision Systems
> **Focus**: Embeddings, Re-ranking, and Fusion
> **Reading Time**: 60 mins

---

## 1. Multi-Modal Fusion

User queries text "Red Dress" + Image (Style).

### 1.1 CLIP (Contrastive Language-Image Pre-training)
*   Embed Text -> Vector T.
*   Embed Image -> Vector I.
*   **Fusion**: $V = \alpha T + (1-\alpha) I$. Or Concatenate and MLP.

---

## 2. Re-ranking

### 2.1 Two-Stage Pipeline
1.  **Retrieval**: ANN (HNSW) on Embeddings. Fast. Top 1000.
2.  **Ranking**: Cross-Encoder (Heavy).
    *   Input: (Query Image, Candidate Image).
    *   Output: Similarity Score.
    *   **Local Features**: SIFT/ORB matching for geometric verification (e.g., Logo detection).

---

## 3. Tricky Interview Questions

### Q1: How to handle "Crop" queries?
> **Answer**:
> *   User crops a bag in a full body photo.
> *   **Object Detection**: Run YOLO first to detect objects. Embed crops.
> *   **Spatial Attention**: Model learns to focus on the object.

### Q2: Indexing 10 Billion Images?
> **Answer**:
> *   **Sharding**: Split by ImageID.
> *   **Quantization**: PQ (Product Quantization) to reduce RAM.

### Q3: Duplicate Detection?
> **Answer**:
> *   **Exact**: Hash (MD5).
> *   **Near**: Perceptual Hash (pHash) or Embedding Distance < Threshold.

---

## 4. Practical Edge Case: Orientation
*   **Problem**: User uploads rotated image.
*   **Fix**: Run a lightweight "Orientation Classifier" (0, 90, 180, 270) and rotate back before embedding.

