# Day 15: Vision Transformers (ViT)

## 1. The Transformer Revolution
**Paper:** "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Dosovitskiy et al., 2020).
**Paradigm Shift:**
*   **CNNs:** Inductive bias (translation invariance, locality) built-in. Good for small data.
*   **Transformers:** Less inductive bias (global attention). Needs massive data (JFT-300M) to learn these properties but scales better.

## 2. ViT Architecture

### Step 1: Patch Embedding
Split image into fixed-size patches (e.g., $16 \times 16$).
*   Input: $H \times W \times C$ (e.g., $224 \times 224 \times 3$).
*   Patches: $N = (H/P) \times (W/P) = 14 \times 14 = 196$.
*   Flatten: Each patch becomes a vector of size $P^2 \cdot C = 16 \cdot 16 \cdot 3 = 768$.
*   **Linear Projection:** Map to embedding dimension $D$ (e.g., 768).

### Step 2: Positional Embedding
Since Transformers are permutation invariant, we must add position info.
*   Learnable 1D vectors added to patch embeddings.

### Step 3: Transformer Encoder
Standard BERT-like encoder.
*   **Multi-Head Self-Attention (MSA)**
*   **MLP** (GELU activation)
*   **Layer Norm** (Pre-Norm)
*   **Residual Connections**

### Step 4: Classification Head
*   Prepend a learnable **[CLS] token** (like BERT).
*   The output state of this token serves as the image representation.
*   MLP Head for classification.

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, D, H/P, W/P) -> (B, D, N) -> (B, N, D)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.patch_embed = PatchEmbedding()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.pos_embed = nn.Parameter(torch.zeros(1, 196 + 1, 768))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=12)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=12)
        
        self.head = nn.Linear(768, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        x = self.encoder(x)
        
        # Take [CLS] token output
        x = x[:, 0]
        return self.head(x)
```

## 3. Data Efficiency (DeiT)
**Problem:** ViT needs huge data (JFT-300M) to beat ResNet. On ImageNet (1M), it overfits.
**Solution:** **DeiT (Data-efficient Image Transformers)**.
*   **Distillation:** Train Student (ViT) to mimic Teacher (RegNet).
*   **Distillation Token:** Adds a special token that interacts with the teacher's predictions.
*   **Strong Augmentation:** MixUp, CutMix, RandAugment.
*   **Result:** ViT trains on ImageNet-1k alone.

## 4. Swin Transformer (Hierarchical)
**Problem:** ViT has quadratic complexity $O(N^2)$ with image size. Hard for high-res.
**Solution:** **Shifted Windows**.
*   Compute attention only within local windows (linear complexity).
*   Shift windows between layers to allow cross-window communication.
*   Produces hierarchical feature maps (like CNNs), making it suitable for Detection/Segmentation.

## Summary
ViT challenged the CNN monopoly. While standard ViT requires massive data, improved training recipes (DeiT) and hierarchical designs (Swin) have made Transformers the new SOTA for many vision tasks.
