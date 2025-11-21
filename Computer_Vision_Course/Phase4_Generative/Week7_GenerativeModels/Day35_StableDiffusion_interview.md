# Day 35 Interview Questions: Stable Diffusion

## Q1: Why does Stable Diffusion use a VAE?
**Answer:**
*   To compress the image into a **Latent Space**.
*   Pixel space is too high-dimensional and contains imperceptible high-frequency details.
*   Latent space ($64 \times 64$) is compact and semantically rich.
*   This allows the diffusion model to train and sample much faster (computationally feasible on consumer GPUs).

## Q2: What is the role of CLIP in Stable Diffusion?
**Answer:**
*   CLIP (Contrastive Language-Image Pre-training) provides the **Text Encoder**.
*   It converts the text prompt into a sequence of embeddings that are aligned with visual concepts.
*   These embeddings are fed into the U-Net via Cross-Attention layers to guide generation.

## Q3: How does ControlNet work?
**Answer:**
*   It adds a trainable copy of the U-Net encoder to the existing frozen model.
*   It takes an additional condition image (e.g., edge map, depth map).
*   It injects features from the trainable copy into the main U-Net via "Zero Convolutions" (initialized to 0).
*   This allows adding spatial constraints without destroying the pre-trained knowledge.

## Q4: What is the difference between Dreambooth and Textual Inversion?
**Answer:**
*   **Textual Inversion:** Optimizes a **new word embedding** (vector) to represent the concept. Does not touch U-Net weights. Lightweight but less flexible.
*   **Dreambooth:** Fine-tunes the **entire U-Net** (or specific layers) to learn the concept. Higher fidelity but prone to overfitting (forgetting other concepts).

## Q5: Why is the VAE decoder not perfect?
**Answer:**
*   The VAE is lossy compression.
*   Reconstructed images might have slight artifacts (e.g., small text is unreadable, faces in background are distorted).
*   This is the trade-off for the massive speedup of Latent Diffusion.

## Q6: What is "CFG Scale" (Classifier-Free Guidance Scale)?
**Answer:**
*   A parameter $w$ that controls how much the model listens to the prompt.
*   $\text{Output} = \text{Uncond} + w \times (\text{Cond} - \text{Uncond})$.
*   **Low $w$:** Creative, diverse, might ignore prompt.
*   **High $w$:** Strict adherence to prompt, higher saturation, less diversity.
*   **Too High:** Artifacts ("fried" images).

## Q7: Can Stable Diffusion generate images of any size?
**Answer:**
*   It is trained on $512 \times 512$ (or $768 \times 768$ for SD 2.0/XL).
*   It can generate other aspect ratios, but going too far from the training resolution (e.g., $1024 \times 1024$ on SD 1.5) often causes "duplication artifacts" (two heads).
*   This is because the receptive field is fixed.

## Q8: What is Inpainting?
**Answer:**
*   Replacing a masked region of an image with generated content.
*   **Method:**
    *   Input: Masked Image, Mask, Noise.
    *   The U-Net is modified to accept these extra channels (9 channels total).
    *   Or, during sampling, we force the known region to match the original image and let diffusion fill the masked region.

## Q9: Explain "Zero Convolution".
**Answer:**
*   A $1 \times 1$ convolution with weights and bias initialized to **zero**.
*   Output is initially zero.
*   This ensures that at the start of training, the ControlNet branch has **no effect** on the output, preserving the original model's behavior. Gradients then slowly update the weights.

## Q10: Implement Cross-Attention (Conceptual).
**Answer:**
```python
class CrossAttention(nn.Module):
    def forward(self, x, context):
        # x: (B, Sequence_Length, Dim) - Image Features
        # context: (B, Context_Length, Dim) - Text Embeddings
        
        Q = self.to_q(x)
        K = self.to_k(context)
        V = self.to_v(context)
        
        attn = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(self.dim))
        attn = attn.softmax(dim=-1)
        
        return attn @ V
```
