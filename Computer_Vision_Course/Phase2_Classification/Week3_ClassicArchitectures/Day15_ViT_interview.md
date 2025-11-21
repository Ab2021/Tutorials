# Day 15 Interview Questions: Vision Transformers

## Q1: Why does ViT perform worse than ResNet on small datasets (e.g., CIFAR-10)?
**Answer:**
**Lack of Inductive Bias.**
*   CNNs have built-in assumptions about images (locality, translation invariance). This acts as a strong prior, helping them learn from few examples.
*   ViT starts as a blank slate. It must learn these spatial relationships from scratch, which requires massive amounts of data (JFT-300M or ImageNet-21k).

## Q2: How do we handle images of different sizes in ViT?
**Answer:**
*   The patch size $P$ is fixed (e.g., 16).
*   If image size $H \times W$ changes, the number of patches $N$ changes.
*   The Transformer encoder handles variable sequence lengths fine.
*   **Problem:** The **Positional Embeddings** are learned for a specific $N$.
*   **Solution:** We must interpolate (resize) the pretrained positional embeddings to match the new sequence length.

## Q3: What is the role of the [CLS] token?
**Answer:**
*   Borrowed from NLP (BERT).
*   It is a learnable vector prepended to the sequence of patch embeddings.
*   It interacts with all other patches via self-attention.
*   Its final state serves as the **aggregate representation** of the entire image for classification.
*   *Alternative:* Global Average Pooling of patch outputs (used in Swin).

## Q4: Explain the complexity of Self-Attention. Why is it an issue for high-res images?
**Answer:**
*   Complexity is $O(N^2)$, where $N$ is the number of patches.
*   $N = (H \times W) / P^2$.
*   If we double image resolution, $N$ quadruples, and complexity increases by $16 \times$.
*   This makes standard ViT prohibitive for dense prediction tasks (segmentation/detection) requiring high resolution.

## Q5: How does Swin Transformer solve the complexity issue?
**Answer:**
**Windowed Attention.**
*   It partitions the image into non-overlapping windows (e.g., $7 \times 7$ patches).
*   Self-attention is computed **only within each window**.
*   Complexity becomes linear with respect to image size ($O(N)$).
*   **Shifted Windows** allow information to propagate between windows in subsequent layers.

## Q6: What is the difference between Post-Norm and Pre-Norm?
**Answer:**
*   **Post-Norm (Original Transformer):** `Norm(x + Sublayer(x))`. Harder to train deep networks (gradients can vanish).
*   **Pre-Norm (ViT/GPT):** `x + Sublayer(Norm(x))`. Gradients flow through the identity path un-normalized. Much more stable training for deep networks.

## Q7: Why use a Conv2d layer for Patch Embedding instead of splitting and flattening?
**Answer:**
Efficiency.
*   Splitting an image into patches and flattening is mathematically equivalent to a Convolution with `kernel_size=P` and `stride=P`.
*   Conv2d implementations in libraries like PyTorch are highly optimized.

## Q8: What is "Distillation" in DeiT?
**Answer:**
Training a Student network (ViT) to match the output of a Teacher network (ConvNet).
*   **Hard Distillation:** Student predicts the teacher's hard label.
*   **Distillation Token:** A special token allows the student to learn from the teacher specifically.
*   Helps ViT learn inductive biases from the ConvNet teacher.

## Q9: Can ViT work without Positional Embeddings?
**Answer:**
**No (or very poorly).**
*   Self-attention is permutation invariant ($A, B = B, A$).
*   Without position info, the model effectively treats the image as a "bag of patches" and loses all spatial structure (e.g., doesn't know the nose is above the mouth).

## Q10: Implement the Patch Embedding class.
**Answer:**
```python
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x) # (B, D, H/P, W/P)
        x = x.flatten(2) # (B, D, N)
        x = x.transpose(1, 2) # (B, N, D)
        return x
```
