# Day 38 Interview Questions: Video Understanding

## Q1: Why is 3D Convolution expensive?
**Answer:**
*   Parameters: $k_t \times k_h \times k_w \times C_{in} \times C_{out}$. More params than 2D.
*   Compute: Sliding window moves in 3 dimensions.
*   Memory: Storing activations for the entire video volume $(B, C, T, H, W)$ consumes massive VRAM.

## Q2: What is the "Inflation" in I3D?
**Answer:**
*   Taking a pre-trained 2D Conv kernel ($k \times k$) and expanding it to 3D ($1 \times k \times k$).
*   The weights are replicated along the time dimension and divided by $N$ to preserve the magnitude of activations.
*   Allows initializing 3D networks with strong ImageNet features.

## Q3: Explain the motivation behind SlowFast networks.
**Answer:**
*   **Spatial details** change slowly (a person is a person across frames). Requires high spatial res, low temporal res (Slow path).
*   **Motion** changes fast (clapping hands). Requires high temporal res, low spatial res (Fast path).
*   Separating them saves compute compared to processing everything at high spatial+temporal res.

## Q4: What is Optical Flow?
**Answer:**
*   A 2D vector field describing the displacement of pixels between two frames.
*   Used to explicitly model motion.
*   Two-stream networks use it as a separate input modality.

## Q5: How does TSM (Temporal Shift Module) work?
**Answer:**
*   It shifts a portion of the feature map channels along the time dimension.
*   Channel $i$ at time $t$ receives data from time $t-1$.
*   This mixes information from adjacent frames.
*   It is a parameter-free operation that turns a 2D CNN into a pseudo-3D CNN.

## Q6: What is the difference between "Late Fusion" and "Early Fusion"?
**Answer:**
*   **Early Fusion:** Stack frames at the input layer (Input = $3T$ channels). Network sees time immediately.
*   **Late Fusion:** Process frames independently with 2D CNN. Average scores at the end. Network sees time only at the prediction stage.
*   3D CNNs are a form of "Slow Fusion" (gradual mixing).

## Q7: Why do Video Transformers factorize attention?
**Answer:**
*   Full self-attention scales quadratically with the number of tokens.
*   A video has $T \times H \times W$ tokens.
*   For $T=32, H=W=16$ (patches), tokens = 8192. Attention matrix = $67M$ entries. Too big.
*   Factorizing into Spatial ($H \times W$) and Temporal ($T$) attention reduces complexity significantly.

## Q8: What is "Action Recognition"?
**Answer:**
*   Classification task: Input video $\to$ Class label (e.g., "Playing Cricket").
*   Datasets: Kinetics-400, UCF-101, HMDB-51.

## Q9: What is "Temporal Action Localization"?
**Answer:**
*   Detection task: Input video $\to$ Start Time, End Time, Class Label.
*   "Find all instances of 'Jumping' in this long video."

## Q10: Implement a simple frame averaging baseline.
**Answer:**
```python
class FrameAvgModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model # e.g., ResNet50

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
        
        feats = self.base(x) # (B*T, Num_Classes)
        
        # Average over time
        feats = feats.view(B, T, -1).mean(dim=1)
        return feats
```
