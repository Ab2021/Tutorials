# Day 47: Mock Interview 3 - Visual Search (Pinterest)

> **Phase**: 5 - Interview Mastery
> **Week**: 10 - Mock Interviews
> **Focus**: Computer Vision & Embeddings
> **Reading Time**: 60 mins

---

## 1. Problem Statement

"Design a system where users can take a photo of a shoe and find similar shoes to buy."

---

## 2. Step-by-Step Design

### Step 1: Requirements
*   **Visual Similarity**: Must match style/color/shape.
*   **Inventory**: Only show items currently in stock.

### Step 2: Pipeline
1.  **Object Detection**: YOLOv8. Crop the shoe from the user's photo.
2.  **Embedding**: ResNet/ViT. Convert crop to 512-d vector.
3.  **Retrieval**: ANN Search (HNSW) against catalog vectors.
4.  **Ranking**: Re-rank by availability and price.

### Step 3: Training the Embedding
*   **Metric Learning**: Triplet Loss.
    *   *Anchor*: Photo of shoe (User upload).
    *   *Positive*: Professional photo of same shoe (Catalog).
    *   *Negative*: Photo of different shoe.
    *   *Goal*: Minimize distance(A, P) and maximize distance(A, N).

---

## 3. Deep Dive Questions

**Interviewer**: "The user uploaded a photo of a red Nike shoe, but we returned a red Adidas shoe. How to fix?"
**Candidate**: "The model over-indexed on 'Red' and under-indexed on 'Logo'. We need Hard Negative Mining. During training, specifically select 'Red Adidas' as the negative example for 'Red Nike'. This forces the model to learn fine-grained features like the logo shape."

**Interviewer**: "How to handle latency?"
**Candidate**: "Object Detection is slow. We can run a tiny model on the mobile device (CoreML/TFLite) to crop the image *before* uploading. This saves bandwidth and server compute."

---

## 4. Evaluation
*   **Metric**: Recall@10 (Is the correct shoe in top 10?).
*   **Visual Inspection**: t-SNE plot of embeddings to see if brands cluster together.

---

## 5. Further Reading
- [Pinterest Visual Search Architecture](https://medium.com/pinterest-engineering/visual-search-at-pinterest-1e07763600cd)
