# Day 18: Video Understanding - Interview Questions

> **Phase**: 2 - Computer Vision
> **Week**: 4 - Generative & Deployment
> **Topic**: 3D CNNs, Motion, and Architectures

### 1. Why are 2D CNNs insufficient for Video Understanding?
**Answer:**
*   2D CNNs treat frames independently. They lose the temporal relationship.
*   They can recognize "Swimming Pool" (Scene) but might fail to distinguish "Swimming" vs "Drowning" (Action/Motion).

### 2. What is the difference between Conv2d and Conv3d?
**Answer:**
*   **Conv2d**: Slides over $H, W$. Weights: $C \times K_h \times K_w$.
*   **Conv3d**: Slides over $T, H, W$. Weights: $C \times K_t \times K_h \times K_w$.
*   Conv3d preserves temporal structure in the feature map.

### 3. Explain "Two-Stream Networks".
**Answer:**
*   One stream processes RGB images (Appearance).
*   Second stream processes Optical Flow (Motion).
*   Predictions are fused (averaged) at the end.
*   Historically very successful because CNNs struggled to learn motion implicitly.

### 4. What is "Optical Flow"?
**Answer:**
*   A vector field describing the displacement of pixels between two consecutive frames.
*   Represents pure motion, independent of texture/color.

### 5. How does the "SlowFast" network work?
**Answer:**
*   **Slow Path**: Low frame rate, high spatial resolution/channels. Captures *what* is happening.
*   **Fast Path**: High frame rate, low channels. Captures *motion*.
*   Mimics the human visual system (P-cells and M-cells).

### 6. What is "R(2+1)D"?
**Answer:**
*   Factorizing a 3D convolution $k \times d \times d$ into:
    1.  2D Spatial Conv ($1 \times d \times d$).
    2.  1D Temporal Conv ($k \times 1 \times 1$).
*   Reduces parameters, adds non-linearity, easier to optimize.

### 7. Why is Video Transformer training expensive?
**Answer:**
*   Self-Attention is $O(N^2)$.
*   Video has $T \times H \times W$ pixels.
*   Flattening a video results in a massive sequence length.
*   Requires efficient attention mechanisms (Divided Space-Time, Window Attention).

### 8. What is "TSM" (Temporal Shift Module)?
**Answer:**
*   A primitive that shifts channels along the time dimension.
*   Allows information exchange between neighboring frames without parameters.
*   Enables 2D CNNs to model time.

### 9. How do you handle variable length videos in a batch?
**Answer:**
*   **Sampling**: Sample a fixed number of frames (e.g., 16) from the video regardless of length.
*   **Sliding Window**: Cut video into fixed clips. Predict on clips, average results.

### 10. What is "I3D" (Inflated 3D ConvNet)?
**Answer:**
*   Taking a pre-trained 2D architecture (Inception/ResNet).
*   Inflating the $N \times N$ kernels to $N \times N \times N$ by repeating weights along time.
*   Allows bootstrapping 3D models from ImageNet weights.

### 11. What is "Tubelet Embedding"?
**Answer:**
*   In Video Transformers, extracting 3D patches (volumes) from the video input instead of 2D patches.
*   Combines spatial and temporal info at the tokenization step.

### 12. Why is "Zero-Shot Action Recognition" hard?
**Answer:**
*   Actions are complex combinations of verbs and nouns.
*   "Opening a door" looks very different from "Opening a bottle".
*   CLIP-based models are making progress here.

### 13. What is the "Kinetics" dataset?
**Answer:**
*   The "ImageNet of Video".
*   400/600/700 classes of human actions.
*   Standard benchmark for pre-training video models.

### 14. How do you deal with the massive data size of videos during training?
**Answer:**
*   **Decord / DALI**: Efficient GPU decoding.
*   **Short Clips**: Train on 2-second clips, not full minutes.
*   **Low Resolution**: Train on 224p or 112p.

### 15. What is "Temporal Action Localization"?
**Answer:**
*   Predicting the start and end time of an action within an untrimmed video.
*   Analogous to Object Detection in 1D.

### 16. What is "C3D"?
**Answer:**
*   One of the first 3D CNN architectures.
*   Simple stack of $3 \times 3 \times 3$ convolutions.
*   Showed that 3D Convs learn generic spatiotemporal features.

### 17. Explain "Divided Space-Time Attention".
**Answer:**
*   Applying Temporal Attention (across frames) and Spatial Attention (within frame) separately.
*   Reduces complexity from $O((THW)^2)$ to $O(T^2 + (HW)^2)$ (roughly).

### 18. What is "VideoMAE"?
**Answer:**
*   Masked Autoencoder for Video.
*   Masks 90% of tubelets (extremely high masking ratio).
*   Forces model to learn high-level semantics to reconstruct motion and appearance.

### 19. Why sample frames uniformly instead of consecutively?
**Answer:**
*   Consecutive frames are redundant ($t$ and $t+1$ are same).
*   Uniform sampling (e.g., $t=0, 10, 20...$) covers the entire duration of the event, capturing the evolution of the action.

### 20. What is "Online Action Recognition"?
**Answer:**
*   Recognizing actions in real-time as frames arrive.
*   Cannot look at future frames (Causal).
*   TSM and RNNs are suitable; standard Transformers (non-causal) are not.
