# Day 7.1: Vision Transformers (ViT) Architecture - A Practical Guide

## Introduction: Transformers See Images

For a long time, Convolutional Neural Networks (CNNs) were the undisputed kings of computer vision. Their architecture, with its inductive biases of locality and translation invariance, seemed perfectly suited for visual tasks. However, the phenomenal success of the **Transformer** architecture in Natural Language Processing (NLP) prompted a question: can we apply this same architecture directly to images?

The **Vision Transformer (ViT)** was the answer, and it was a resounding yes. ViT demonstrated that a pure Transformer architecture, with minimal modifications, could achieve state-of-the-art results on image classification tasks, challenging the dominance of CNNs.

This guide will provide a practical, step-by-step implementation of the Vision Transformer, breaking down its novel components and showing how it processes image data.

**Today's Learning Objectives:**

1.  **Understand the Core ViT Idea:** Learn how an image is converted into a sequence of flattened patches, making it suitable for a Transformer.
2.  **Implement the Patch Embedding Layer:** Write the code that performs this critical image-to-sequence transformation.
3.  **Grasp the Role of the `[CLS]` Token:** Understand this special token, borrowed from NLP models like BERT, which is used for classification.
4.  **Implement Positional Embeddings:** See why explicit positional information is crucial for the model.
5.  **Build a Complete ViT Model:** Assemble the patch embedding, `[CLS]` token, positional embeddings, and a standard Transformer Encoder into a full, working model.

---

## Part 1: The ViT Paradigm Shift - From Convolutions to Patches

The core innovation of ViT is how it handles images. Instead of processing the image with sliding convolutional filters, it does the following:

1.  **Split Image into Patches:** The input image is split into a grid of fixed-size, non-overlapping patches. For example, a 224x224 image might be split into a 14x14 grid of 16x16 patches.

2.  **Flatten Patches:** Each of these 2D patches is flattened into a 1D vector.

3.  **Linear Projection:** Each flattened patch vector is passed through a standard linear layer to produce a **patch embedding**. This is the equivalent of a word embedding in NLP.

4.  **Prepend `[CLS]` Token:** A special, learnable vector, the `[CLS]` (classification) token, is added to the beginning of this sequence of patch embeddings.

5.  **Add Positional Embeddings:** Since the Transformer architecture itself has no inherent sense of order, we must add learnable positional embeddings to the patch embeddings to retain spatial information.

6.  **Feed to Transformer Encoder:** This final sequence of vectors is fed into a standard Transformer Encoder.

7.  **Classify:** The output corresponding to the `[CLS]` token is taken from the Transformer's output, passed through an MLP head, and used for the final classification.

![ViT Pipeline](https://i.imgur.com/t2n2pJA.png)

---

## Part 2: Implementing the ViT Components

Let's build the ViT architecture piece by piece.

### 2.1. The Patch Embedding Layer

This is the most critical part. We can cleverly implement the splitting and flattening of patches as a single `nn.Conv2d` layer. A convolution with a `kernel_size` and `stride` equal to the `patch_size` is mathematically equivalent to splitting the image into patches and passing each through a linear layer.

```python
import torch
import torch.nn as nn

print("--- Part 2.1: The Patch Embedding Layer ---")

class PatchEmbedding(nn.Module):
    """Turns a 2D image into a 1D sequence of patch embeddings."""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # The key layer: a convolution that acts like patch splitting and embedding
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # Input x: (N, C, H, W)
        # Output of conv: (N, embed_dim, n_patches_h, n_patches_w)
        x = self.proj(x)
        # Flatten the H and W dimensions into a single sequence dimension
        # Output: (N, embed_dim, n_patches)
        x = x.flatten(2)
        # Transpose to get the standard Transformer input shape
        # Output: (N, n_patches, embed_dim)
        x = x.transpose(1, 2)
        return x

# --- Usage Example ---
# A batch of 4 RGB images, each 224x224
input_images = torch.randn(4, 3, 224, 224)

# Create the patch embedding layer
# embed_dim=768 is the standard for the ViT-Base model
patch_embed_layer = PatchEmbedding()

# Get the sequence of patch embeddings
patch_embeddings = patch_embed_layer(input_images)

print(f"Input image shape: {input_images.shape}")
print(f"Number of patches per image: {patch_embed_layer.n_patches}")
print(f"Output sequence shape: {patch_embeddings.shape}") # (N, n_patches, embed_dim)
```

### 2.2. The Full Vision Transformer Model

Now let's assemble all the pieces into a complete model.

```python
import torch.nn.functional as F

print("\n--- Part 2.2: The Full Vision Transformer ---")

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, 
                 n_layers=12, n_heads=12, mlp_ratio=4.0, n_classes=1000):
        super().__init__()
        
        # --- 1. Patch Embedding Layer ---
        self.patch_embed = PatchEmbedding(
            img_size=img_size, 
            patch_size=patch_size, 
            in_channels=in_channels, 
            embed_dim=embed_dim
        )
        
        # --- 2. CLS Token ---
        # A learnable parameter that will be prepended to the sequence.
        # Shape: (1, 1, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # --- 3. Positional Embeddings ---
        # Learnable embeddings for the CLS token + all patches.
        # Shape: (1, n_patches + 1, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
        
        # --- 4. Transformer Encoder ---
        # We use PyTorch's built-in TransformerEncoder.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=n_heads, 
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation=F.gelu,
            batch_first = True # This is crucial!
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # --- 5. Classification Head ---
        # An MLP head for the final classification.
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        # Get batch size
        n_samples = x.shape[0]
        
        # 1. Create patch embeddings
        # (N, n_patches, embed_dim)
        x = self.patch_embed(x)
        
        # 2. Prepend the CLS token
        # Expand the CLS token to match the batch size
        cls_token = self.cls_token.expand(n_samples, -1, -1) # (N, 1, embed_dim)
        # Concatenate along the sequence dimension
        x = torch.cat((cls_token, x), dim=1) # (N, n_patches + 1, embed_dim)
        
        # 3. Add positional embeddings
        x = x + self.pos_embed # Broadcasting takes care of the batch dimension
        
        # 4. Pass through the Transformer Encoder
        x = self.transformer_encoder(x) # (N, n_patches + 1, embed_dim)
        
        # 5. Get the output of the CLS token for classification
        # We only use the output corresponding to the first token.
        cls_output = x[:, 0]
        
        # 6. Pass through the classification head
        out = self.head(cls_output)
        
        return out

# --- Usage Example ---
# Create a ViT-Base model instance
vit_base_model = VisionTransformer()

# A batch of 4 RGB images, each 224x224
input_images = torch.randn(4, 3, 224, 224)

# Get the final classification logits
logits = vit_base_model(input_images)

print(f"Input image shape: {input_images.shape}")
print(f"Output logits shape: {logits.shape}") # (N, n_classes)

# --- Parameter Count Comparison ---
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

vit_params = count_parameters(vit_base_model)
# For comparison, let's get a ResNet50
# Assuming 'models' is imported from torchvision.models
# from torchvision import models
# resnet50 = models.resnet50()
# resnet_params = count_parameters(resnet50)

print(f"\nViT-Base Parameters: {vit_params / 1e6:.1f}M")
# print(f"ResNet-50 Parameters: {resnet_params / 1e6:.1f}M")
```

## Part 3: Training a Vision Transformer

Training a ViT from scratch is notoriously difficult and data-hungry. Unlike CNNs, ViTs have very weak **inductive biases**. A CNN "knows" about locality (pixels near each other are related) and translation invariance (an object is the same wherever it appears) due to its convolutional nature. A ViT knows nothing of this; it treats the image patches like words in a sentence and must learn all the spatial relationships from scratch.

**Key Training Considerations:**

1.  **Massive Data is Required:** ViTs only start to outperform CNNs when trained on extremely large datasets (like ImageNet-21k with 14 million images, or JFT-300M with 300 million images). On smaller datasets like CIFAR-10 or even ImageNet-1k, a standard ResNet will often perform better when trained from scratch.
2.  **Strong Regularization:** Due to the lack of inductive bias, ViTs are prone to overfitting. They require strong data augmentation (like Mixup, CutMix) and regularization (like weight decay).
3.  **Transfer Learning is Key:** The most common and effective way to use a ViT is to take a model that has already been pre-trained on a massive dataset and then **fine-tune** it on your smaller, specific dataset. This is the same principle as with CNNs but is even more critical for ViTs.

## Conclusion: A New Paradigm for Vision

The Vision Transformer represents a significant paradigm shift in computer vision. It demonstrated that the general-purpose Transformer architecture, with its powerful self-attention mechanism, could be a viable and powerful alternative to the highly specialized convolutional architectures.

**Key Takeaways:**

1.  **Images as Sequences:** The core idea of ViT is to transform an image into a sequence of flattened patches, allowing a standard Transformer to process it.
2.  **Patching as Convolution:** The patching and embedding process can be efficiently implemented as a single `Conv2d` layer.
3.  **`[CLS]` Token for Classification:** A special learnable token is prepended to the sequence, and its corresponding output from the Transformer is used for the final classification.
4.  **Positional Embeddings are Crucial:** Since the self-attention mechanism is permutation-invariant, learnable positional embeddings must be added to the sequence to retain spatial information.
5.  **Inductive Bias Trade-off:** ViTs have weaker inductive biases than CNNs, which makes them more flexible but also more data-hungry and harder to train from scratch.

While CNNs remain highly effective and are often a better choice for smaller datasets, the ViT has opened up new avenues of research and has become a foundational architecture in modern, large-scale computer vision systems.

## Self-Assessment Questions

1.  **Image to Sequence:** What are the three main steps ViT uses to convert a 2D image into a 1D sequence of vectors?
2.  **`[CLS]` Token:** What is the purpose of the `[CLS]` token, and where does this idea come from?
3.  **Positional Information:** Why are positional embeddings more critical for a ViT than for a CNN?
4.  **Inductive Bias:** What does it mean that a CNN has a stronger "inductive bias" for images than a ViT?
5.  **Training:** Why is it generally a bad idea to train a ViT from scratch on a small dataset like CIFAR-100?
