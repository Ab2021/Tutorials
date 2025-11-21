# Day 35 Deep Dive: Conditioning & ControlNet

## 1. Cross-Attention Mechanism
The core of text-conditioning.
*   **Spatial Features ($x$):** From U-Net. Shape $(B, H \times W, C)$.
*   **Context ($y$):** From Text Encoder. Shape $(B, L, D)$.
*   **Operation:**
    *   $Q = W_q x$
    *   $K = W_k y$
    *   $V = W_v y$
    *   $Out = \text{Softmax}(QK^T) V$
*   The model learns to attend to specific words (e.g., "Horse") when generating specific regions.

## 2. ControlNet
**Goal:** Add spatial control (Edges, Pose, Depth) to Stable Diffusion *without* retraining the whole model.
*   **Architecture:**
    *   **Locked Copy:** The original SD U-Net encoder (frozen).
    *   **Trainable Copy:** A clone of the encoder (trainable).
    *   **Zero Convolutions:** Connect the two copies. Initialized to zero so training starts with identity (no effect).
*   **Input:** Canny Edge Map / Pose Skeleton.
*   **Result:** "Generate a cat" + [Edge Map of Cat] $\to$ Cat that perfectly aligns with edges.

## 3. LoRA (Low-Rank Adaptation)
**Goal:** Fine-tune SD on a new concept (e.g., "My Dog") efficiently.
*   Instead of updating the full weight matrix $W$ ($d \times d$), update a low-rank decomposition:
    $$ W' = W + \Delta W = W + BA $$
    *   $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}$.
    *   Rank $r$ is small (e.g., 4 or 8).
*   Reduces trainable parameters by 10,000x.
*   Allows sharing small (<100MB) model files.

## 4. Textual Inversion
**Goal:** Teach SD a new object using only 3-5 images.
*   Find a new token embedding $v_*$ (e.g., `<sks>`) in the text encoder's embedding space that represents the object.
*   Freeze U-Net, optimize only the vector $v_*$.
*   Prompt: "A photo of `<sks>` on the beach".

## Summary
The ecosystem around Stable Diffusion (ControlNet, LoRA, Dreambooth) is as important as the model itself, enabling precise control and personalization.
