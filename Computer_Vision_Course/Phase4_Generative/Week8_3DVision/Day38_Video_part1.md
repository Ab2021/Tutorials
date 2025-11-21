# Day 38 Deep Dive: Video Transformers (ViViT)

## 1. ViViT (Video Vision Transformer)
**Idea:** Apply Transformers to Video.
**Tokenization Strategies:**
1.  **Spatio-Temporal Tubes:** Extract 3D patches ($t \times h \times w$) from the video volume. Linearly project to tokens.
2.  **Factorized Encoder:**
    *   **Spatial Transformer:** Process patches within each frame independently.
    *   **Temporal Transformer:** Process the CLS tokens from all frames to model time.
    *   More efficient than full joint attention.

## 2. TimeSformer
**Idea:** Divided Space-Time Attention.
*   Instead of attending to all tokens ($T \times H \times W$), which is $O((THW)^2)$.
*   **Temporal Attention:** Each patch attends to the same patch in other frames.
*   **Spatial Attention:** Each patch attends to other patches in the same frame.
*   Reduces complexity to $O(HW T^2 + T (HW)^2)$.

## 3. Optical Flow
**Definition:** The pattern of apparent motion of objects between two consecutive frames.
*   Vector field $(u, v)$ for each pixel.
*   **Farneback Algorithm:** Dense optical flow (Classical).
*   **FlowNet / RAFT:** Deep learning based optical flow estimation.
*   Crucial for Two-Stream networks, but expensive to compute on the fly.

## 4. Temporal Shift Module (TSM)
**Idea:** 3D performance at 2D cost.
*   Shift a fraction of channels forward in time and backward in time.
*   $X_{t, c} \leftarrow X_{t-1, c}$.
*   Allows information exchange between neighboring frames without any parameters (zero FLOPs).
*   Insert into standard ResNet to make it "Temporal".

## Summary
Transformers are taking over video as well. Factorized attention (TimeSformer) and Tube embeddings (ViViT) are the key mechanisms to handle the massive computational cost of video data.
