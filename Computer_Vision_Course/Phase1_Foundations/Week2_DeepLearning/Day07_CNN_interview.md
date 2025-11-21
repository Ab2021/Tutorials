# Day 7 Interview Questions: CNNs

## Q1: Why are CNNs better than fully connected networks for images?
**Answer:**

**Fully connected problems:**
1. **Too many parameters:** 224×224×3 image → 1000 neurons = 150M parameters
2. **No spatial structure:** Treats pixels independently
3. **Not translation invariant:** Same object at different positions = different features

**CNN advantages:**
1. **Parameter sharing:** Same filter across entire image
2. **Local connectivity:** Each neuron connects to small region
3. **Translation equivariance:** $f(T(x)) = T(f(x))$ for translations

**Example:**
- FC layer: $224 \times 224 \times 3 \times 1000 = 150M$ parameters
- Conv layer (64 filters, 3×3): $3 \times 3 \times 3 \times 64 = 1,728$ parameters

**Reduction:** ~87,000×

## Q2: Calculate output dimensions for a convolutional layer.
**Answer:**

**Formula:**
$$ H_{out} = \left\lfloor \frac{H_{in} + 2p - k}{s} \right\rfloor + 1 $$

**Example:**
- Input: 32×32
- Kernel: 5×5
- Stride: 1
- Padding: 2

$$ H_{out} = \frac{32 + 2(2) - 5}{1} + 1 = \frac{32 + 4 - 5}{1} + 1 = 32 $$

**Common patterns:**
- **Same padding:** $p = \lfloor k/2 \rfloor$ (output size = input size for $s=1$)
- **Valid padding:** $p = 0$ (output size = input size - k + 1)

## Q3: What is the receptive field and how to calculate it?
**Answer:**

**Definition:** Region in input image that affects a particular output neuron.

**Recursive formula:**
$$ r_{l+1} = r_l + (k_{l+1} - 1) \times \prod_{i=1}^{l} s_i $$

**Example:**
```
Layer 1: Conv 3×3, stride 1 → r₁ = 3
Layer 2: Conv 3×3, stride 1 → r₂ = 3 + (3-1)×1 = 5
Layer 3: Pool 2×2, stride 2 → r₃ = 5 + (2-1)×1 = 6
Layer 4: Conv 3×3, stride 1 → r₄ = 6 + (3-1)×2 = 10
```

**Importance:** Larger receptive field → more context, but deeper network needed.

## Q4: Compare max pooling vs average pooling.
**Answer:**

**Max Pooling:**
$$ y = \max(x_1, x_2, ..., x_n) $$

- **Pros:** Preserves strongest activations, translation invariant
- **Cons:** Loses spatial information
- **Use:** Most common, especially for classification

**Average Pooling:**
$$ y = \frac{1}{n} \sum_{i=1}^n x_i $$

- **Pros:** Smoother, retains more information
- **Cons:** Dilutes strong activations
- **Use:** Global average pooling (GAP) for classification

**Gradient:**
- Max: Gradient flows only to max element
- Average: Gradient distributed equally

## Q5: Explain depthwise separable convolution and its benefits.
**Answer:**

**Standard convolution:**
- Params: $k \times k \times C_{in} \times C_{out}$
- FLOPs: $H \times W \times k^2 \times C_{in} \times C_{out}$

**Depthwise separable:**
1. **Depthwise:** $k \times k$ conv per channel (params: $k^2 \times C_{in}$)
2. **Pointwise:** $1 \times 1$ conv (params: $C_{in} \times C_{out}$)

**Total params:** $k^2 C_{in} + C_{in} C_{out}$

**Reduction:**
$$ \frac{k^2 C_{in} + C_{in} C_{out}}{k^2 C_{in} C_{out}} = \frac{1}{C_{out}} + \frac{1}{k^2} $$

**Example:** $k=3, C_{out}=256$ → $\frac{1}{256} + \frac{1}{9} \approx 0.115$ (8.7× reduction)

**Used in:** MobileNet, Xception.

## Q6: What is dilated convolution and when to use it?
**Answer:**

**Dilated convolution:** Convolution with gaps (dilation rate $r$).

**Effective receptive field:**
$$ r_{eff} = k + (k-1)(r-1) $$

**Example:** 3×3 kernel, dilation=2 → 5×5 receptive field

**Advantages:**
1. **Larger receptive field** without increasing parameters
2. **No downsampling** (preserves resolution)
3. **Multi-scale context**

**Applications:**
- **Semantic segmentation:** DeepLab (atrous spatial pyramid pooling)
- **Audio:** WaveNet (exponentially increasing dilation)

**Trade-off:** Gridding artifacts if not designed carefully.

## Q7: Implement a simple CNN forward pass.
**Answer:**

```python
def conv_forward(X, W, b, stride=1, padding=0):
    """
    Convolution forward pass.
    
    X: (batch, C_in, H, W)
    W: (C_out, C_in, k, k)
    b: (C_out,)
    """
    batch, C_in, H, W = X.shape
    C_out, _, k, _ = W.shape
    
    # Padding
    X_pad = np.pad(X, ((0,0), (0,0), (padding,padding), (padding,padding)))
    
    # Output size
    H_out = (H + 2*padding - k) // stride + 1
    W_out = (W + 2*padding - k) // stride + 1
    
    Z = np.zeros((batch, C_out, H_out, W_out))
    
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride
            
            X_slice = X_pad[:, :, h_start:h_start+k, w_start:w_start+k]
            
            for c in range(C_out):
                Z[:, c, i, j] = np.sum(X_slice * W[c], axis=(1,2,3)) + b[c]
    
    return Z
```

## Q8: How does 1×1 convolution work and why use it?
**Answer:**

**1×1 Convolution:** Kernel size = 1×1

**Operation:**
$$ Z[i,j,c_{out}] = \sum_{c_{in}} X[i,j,c_{in}] \times W[c_{in}, c_{out}] + b[c_{out}] $$

**Purposes:**
1. **Dimensionality reduction:** 256 channels → 64 channels
2. **Dimensionality expansion:** 64 channels → 256 channels
3. **Add non-linearity:** Without changing spatial dimensions
4. **Cross-channel interaction:** Mix information across channels

**Example (Bottleneck in ResNet):**
```
256 → [1×1, 64] → [3×3, 64] → [1×1, 256]
```

**Parameters:**
- Without bottleneck: $3 \times 3 \times 256 \times 256 = 589,824$
- With bottleneck: $256 \times 64 + 3 \times 3 \times 64 \times 64 + 64 \times 256 = 69,632$

**Reduction:** 8.5×

## Q9: Explain transposed convolution.
**Answer:**

**Purpose:** Upsampling (learnable, unlike bilinear interpolation).

**Relationship to convolution:**
- Forward conv: $y = Cx$ (matrix $C$)
- Transposed conv: $\hat{x} = C^T y$

**Output size:**
$$ H_{out} = (H_{in} - 1) \times s + k $$

**Example:**
- Input: 4×4
- Kernel: 3×3
- Stride: 2
- Output: $(4-1) \times 2 + 3 = 9$ → 9×9

**Checkerboard artifacts:** Caused by uneven overlap when $k$ not divisible by $s$.

**Solution:** Use $k$ divisible by $s$, or resize + convolution.

**Applications:** Semantic segmentation (FCN, U-Net), GANs.

## Q10: What is the difference between padding='same' and padding='valid'?
**Answer:**

**'valid' (no padding):**
- $p = 0$
- Output size: $H_{out} = H_{in} - k + 1$
- Example: 32×32 input, 5×5 kernel → 28×28 output

**'same' (zero padding):**
- $p = \lfloor k/2 \rfloor$ (for stride=1)
- Output size: $H_{out} = H_{in}$
- Example: 32×32 input, 5×5 kernel, $p=2$ → 32×32 output

**When to use:**
- **'valid':** When downsampling is desired
- **'same':** When preserving spatial dimensions (e.g., ResNet, U-Net)

**Formula for 'same' with arbitrary stride:**
$$ p = \frac{(s-1) H_{in} + k - s}{2} $$

For $s=1$: $p = \frac{k-1}{2}$
