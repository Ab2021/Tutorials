# Day 7 Deep Dive: Advanced CNN Concepts

## 1. Efficient Convolution Implementations

### im2col (Image to Column)
**Idea:** Transform convolution into matrix multiplication.

**Algorithm:**
1. Extract all patches into columns
2. Perform matrix multiplication
3. Reshape to output

```python
def im2col(X, kernel_size, stride, padding):
    """
    Transform image to column matrix for efficient convolution.
    
    Args:
        X: (batch, channels, height, width)
        kernel_size: int
        stride: int
        padding: int
    
    Returns:
        col: (batch * H_out * W_out, channels * k * k)
    """
    batch, channels, H, W = X.shape
    
    # Pad input
    X_pad = np.pad(X, ((0,0), (0,0), (padding, padding), (padding, padding)))
    
    # Output dimensions
    H_out = (H + 2*padding - kernel_size) // stride + 1
    W_out = (W + 2*padding - kernel_size) // stride + 1
    
    # Initialize column matrix
    col = np.zeros((batch, channels, kernel_size, kernel_size, H_out, W_out))
    
    for y in range(kernel_size):
        y_max = y + stride * H_out
        for x in range(kernel_size):
            x_max = x + stride * W_out
            col[:, :, y, x, :, :] = X_pad[:, :, y:y_max:stride, x:x_max:stride]
    
    # Reshape
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(batch * H_out * W_out, -1)
    
    return col

def col2im(col, input_shape, kernel_size, stride, padding):
    """Inverse of im2col."""
    batch, channels, H, W = input_shape
    H_out = (H + 2*padding - kernel_size) // stride + 1
    W_out = (W + 2*padding - kernel_size) // stride + 1
    
    col = col.reshape(batch, H_out, W_out, channels, kernel_size, kernel_size)
    col = col.transpose(0, 3, 4, 5, 1, 2)
    
    X = np.zeros((batch, channels, H + 2*padding + stride - 1, W + 2*padding + stride - 1))
    
    for y in range(kernel_size):
        y_max = y + stride * H_out
        for x in range(kernel_size):
            x_max = x + stride * W_out
            X[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
    
    return X[:, :, padding:H+padding, padding:W+padding]

class FastConv2D:
    """Optimized convolution using im2col."""
    
    def forward(self, X):
        batch, _, H, W = X.shape
        
        # Transform to column matrix
        col = im2col(X, self.kernel_size, self.stride, self.padding)
        
        # Reshape weights
        W_col = self.W.reshape(self.out_channels, -1)
        
        # Matrix multiplication
        out = (W_col @ col.T).T + self.b.T
        
        # Reshape output
        H_out = (H + 2*self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2*self.padding - self.kernel_size) // self.stride + 1
        out = out.reshape(batch, H_out, W_out, self.out_channels)
        out = out.transpose(0, 3, 1, 2)
        
        self.cache['col'] = col
        self.cache['X_shape'] = X.shape
        
        return out
    
    def backward(self, dZ):
        batch, _, H_out, W_out = dZ.shape
        col = self.cache['col']
        
        # Reshape dZ
        dZ_reshaped = dZ.transpose(0, 2, 3, 1).reshape(batch * H_out * W_out, -1)
        
        # Weight gradient
        W_col = self.W.reshape(self.out_channels, -1)
        self.dW = (dZ_reshaped.T @ col).reshape(self.W.shape) / batch
        
        # Bias gradient
        self.db = np.sum(dZ, axis=(0, 2, 3), keepdims=True).T / batch
        
        # Input gradient
        dX_col = dZ_reshaped @ W_col
        dX = col2im(dX_col, self.cache['X_shape'], 
                   self.kernel_size, self.stride, self.padding)
        
        return dX
```

### Winograd Convolution
**Idea:** Reduce multiplications using polynomial interpolation.

**For 3×3 kernel, 2×2 output (F(2,3)):**
- Standard: 36 multiplications
- Winograd: 16 multiplications

**Trade-off:** More additions, numerical stability issues.

## 2. Dilated/Atrous Convolution

**Definition:** Convolution with gaps (dilation rate $r$).

$$ Z[i,j] = \sum_m \sum_n X[i + r \cdot m, j + r \cdot n] \cdot K[m, n] $$

**Receptive field:**
$$ r_{eff} = k + (k-1)(r-1) $$

**Example:** 3×3 kernel, dilation=2 → effective 5×5 receptive field

```python
class DilatedConv2D:
    """Dilated convolution."""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        self.dilation = dilation
        # ... other init
    
    def forward(self, X):
        batch, _, H, W = X.shape
        k = self.kernel_size
        d = self.dilation
        
        # Effective kernel size
        k_eff = k + (k - 1) * (d - 1)
        
        H_out = H - k_eff + 1
        W_out = W - k_eff + 1
        
        Z = np.zeros((batch, self.out_channels, H_out, W_out))
        
        for i in range(H_out):
            for j in range(W_out):
                for c in range(self.out_channels):
                    for m in range(k):
                        for n in range(k):
                            h_idx = i + m * d
                            w_idx = j + n * d
                            Z[:, c, i, j] += np.sum(
                                X[:, :, h_idx, w_idx] * self.W[c, :, m, n],
                                axis=1
                            )
                    Z[:, c, i, j] += self.b[c]
        
        return Z
```

**Applications:** Semantic segmentation (DeepLab), audio generation (WaveNet).

## 3. Depthwise Separable Convolution

**Idea:** Factorize standard convolution into depthwise + pointwise.

**Standard convolution:**
- Parameters: $k \times k \times C_{in} \times C_{out}$
- FLOPs: $H \times W \times k^2 \times C_{in} \times C_{out}$

**Depthwise separable:**
1. **Depthwise:** Apply $k \times k$ filter to each channel independently
2. **Pointwise:** $1 \times 1$ convolution to combine channels

- Parameters: $k \times k \times C_{in} + C_{in} \times C_{out}$
- FLOPs: $H \times W \times (k^2 \times C_{in} + C_{in} \times C_{out})$

**Reduction factor:**
$$ \frac{1}{C_{out}} + \frac{1}{k^2} $$

For $k=3, C_{out}=256$: ~8-9× reduction

```python
class DepthwiseSeparableConv2D:
    """Depthwise separable convolution."""
    
    def __init__(self, in_channels, out_channels, kernel_size):
        # Depthwise convolution (one filter per input channel)
        self.depthwise = Conv2D(in_channels, in_channels, kernel_size, 
                               groups=in_channels)
        
        # Pointwise convolution (1x1)
        self.pointwise = Conv2D(in_channels, out_channels, kernel_size=1)
    
    def forward(self, X):
        X = self.depthwise.forward(X)
        X = self.pointwise.forward(X)
        return X
```

**Used in:** MobileNet, EfficientNet.

## 4. Grouped Convolution

**Idea:** Split channels into groups, convolve independently.

**Parameters:** $\frac{k \times k \times C_{in} \times C_{out}}{g}$

where $g$ is number of groups.

```python
class GroupedConv2D:
    """Grouped convolution."""
    
    def __init__(self, in_channels, out_channels, kernel_size, groups=1):
        assert in_channels % groups == 0
        assert out_channels % groups == 0
        
        self.groups = groups
        self.in_per_group = in_channels // groups
        self.out_per_group = out_channels // groups
        
        # Separate weights for each group
        self.W = np.random.randn(groups, self.out_per_group, 
                                self.in_per_group, kernel_size, kernel_size)
    
    def forward(self, X):
        batch, channels, H, W = X.shape
        outputs = []
        
        for g in range(self.groups):
            # Extract group channels
            start_in = g * self.in_per_group
            end_in = (g + 1) * self.in_per_group
            
            X_group = X[:, start_in:end_in, :, :]
            
            # Convolve
            Z_group = convolve(X_group, self.W[g])
            outputs.append(Z_group)
        
        # Concatenate along channel dimension
        return np.concatenate(outputs, axis=1)
```

**Used in:** ResNeXt, AlexNet (historical).

## 5. Transposed Convolution (Deconvolution)

**Purpose:** Upsampling (opposite of convolution).

**Relationship to convolution:**
If $y = Cx$ (convolution), then transposed convolution computes $C^T y$.

```python
class ConvTranspose2D:
    """Transposed convolution for upsampling."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=2):
        self.stride = stride
        self.kernel_size = kernel_size
        
        self.W = np.random.randn(in_channels, out_channels, 
                                kernel_size, kernel_size) * 0.01
    
    def forward(self, X):
        batch, channels, H, W = X.shape
        
        # Output size
        H_out = (H - 1) * self.stride + self.kernel_size
        W_out = (W - 1) * self.stride + self.kernel_size
        
        Z = np.zeros((batch, self.out_channels, H_out, W_out))
        
        for i in range(H):
            for j in range(W):
                h_start = i * self.stride
                w_start = j * self.stride
                
                for c_out in range(self.out_channels):
                    for c_in in range(channels):
                        Z[:, c_out, 
                          h_start:h_start+self.kernel_size,
                          w_start:w_start+self.kernel_size] += \
                            X[:, c_in, i, j][:, None, None] * self.W[c_in, c_out]
        
        return Z
```

**Applications:** Semantic segmentation, GANs, autoencoders.

## 6. 1×1 Convolution

**Purpose:**
1. **Dimensionality reduction/expansion**
2. **Adding non-linearity**
3. **Cross-channel interaction**

**Example (Bottleneck):**
```
256 channels → 1×1 conv → 64 channels → 3×3 conv → 64 channels → 1×1 conv → 256 channels
```

**Parameters saved:**
- Without bottleneck: $3 \times 3 \times 256 \times 256 = 589,824$
- With bottleneck: $256 \times 64 + 3 \times 3 \times 64 \times 64 + 64 \times 256 = 69,632$

**Reduction:** ~8.5×

## Summary
Advanced convolution variants (dilated, depthwise separable, grouped, transposed) enable efficient architectures with larger receptive fields and fewer parameters.
