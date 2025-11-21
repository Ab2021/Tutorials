# Day 7: Convolutional Neural Networks (CNNs)

## Overview
Convolutional Neural Networks revolutionized computer vision by exploiting spatial structure in images. This lesson covers the mathematical foundations, architectural components, and implementation of CNNs.

## 1. Motivation: Why CNNs?

### Problems with Fully Connected Networks for Images
**Example:** 224×224 RGB image
- Input size: $224 \times 224 \times 3 = 150,528$ pixels
- First hidden layer (1000 neurons): $150,528 \times 1000 = 150M$ parameters
- **Problems:**
  1. Too many parameters → overfitting
  2. No spatial structure → position-dependent
  3. Computationally expensive

### Key Insights
1. **Local connectivity:** Pixels are correlated locally
2. **Parameter sharing:** Same features useful across image
3. **Translation invariance:** Features should detect patterns anywhere

## 2. Convolutional Layer

### Mathematical Definition
**Discrete convolution (2D):**
$$ (I * K)(i, j) = \sum_m \sum_n I(i-m, j-n) \cdot K(m, n) $$

**In practice (cross-correlation):**
$$ Z[i, j] = \sum_m \sum_n I[i+m, j+n] \cdot K[m, n] + b $$

### Multi-Channel Convolution
**Input:** $X \in \mathbb{R}^{H \times W \times C_{in}}$
**Kernel:** $K \in \mathbb{R}^{k \times k \times C_{in} \times C_{out}}$
**Output:** $Z \in \mathbb{R}^{H' \times W' \times C_{out}}$

$$ Z[i, j, c] = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} \sum_{c'=0}^{C_{in}-1} X[i+m, j+n, c'] \cdot K[m, n, c', c] + b_c $$

### Output Dimensions
$$ H_{out} = \left\lfloor \frac{H_{in} + 2p - k}{s} \right\rfloor + 1 $$
$$ W_{out} = \left\lfloor \frac{W_{in} + 2p - k}{s} \right\rfloor + 1 $$

Where:
- $p$: Padding
- $k$: Kernel size
- $s$: Stride

### Implementation

```python
import numpy as np
from typing import Tuple

class Conv2D:
    """2D Convolutional layer."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, stride: int = 1, padding: int = 0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # He initialization
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * \
                 np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.b = np.zeros((out_channels, 1))
        
        self.cache = {}
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            X: Input (batch_size, in_channels, height, width)
        
        Returns:
            Output (batch_size, out_channels, H_out, W_out)
        """
        batch_size, _, H, W = X.shape
        
        # Apply padding
        if self.padding > 0:
            X_pad = np.pad(X, ((0,0), (0,0), (self.padding, self.padding), 
                              (self.padding, self.padding)), mode='constant')
        else:
            X_pad = X
        
        # Output dimensions
        H_out = (H + 2*self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2*self.padding - self.kernel_size) // self.stride + 1
        
        # Initialize output
        Z = np.zeros((batch_size, self.out_channels, H_out, W_out))
        
        # Convolution
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                
                # Extract patch
                X_patch = X_pad[:, :, h_start:h_end, w_start:w_end]
                
                # Convolve with all filters
                for c in range(self.out_channels):
                    Z[:, c, i, j] = np.sum(X_patch * self.W[c], axis=(1,2,3)) + self.b[c]
        
        self.cache['X'] = X
        self.cache['X_pad'] = X_pad
        
        return Z
    
    def backward(self, dZ: np.ndarray) -> np.ndarray:
        """
        Backward pass.
        
        Args:
            dZ: Gradient of loss w.r.t. output
        
        Returns:
            dX: Gradient of loss w.r.t. input
        """
        X = self.cache['X']
        X_pad = self.cache['X_pad']
        batch_size, _, H, W = X.shape
        _, _, H_out, W_out = dZ.shape
        
        # Initialize gradients
        dX_pad = np.zeros_like(X_pad)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        
        # Compute gradients
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                
                X_patch = X_pad[:, :, h_start:h_end, w_start:w_end]
                
                for c in range(self.out_channels):
                    # Weight gradient
                    self.dW[c] += np.sum(X_patch * dZ[:, c, i, j][:, None, None, None], 
                                        axis=0)
                    
                    # Input gradient
                    dX_pad[:, :, h_start:h_end, w_start:w_end] += \
                        self.W[c] * dZ[:, c, i, j][:, None, None, None]
        
        # Bias gradient
        self.db = np.sum(dZ, axis=(0, 2, 3), keepdims=True).T
        
        # Remove padding from gradient
        if self.padding > 0:
            dX = dX_pad[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_pad
        
        # Average over batch
        self.dW /= batch_size
        self.db /= batch_size
        
        return dX
```

## 3. Pooling Layers

### Max Pooling
$$ Z[i, j] = \max_{m, n \in \text{pool}} X[i \cdot s + m, j \cdot s + n] $$

**Purpose:**
- Downsampling (reduce spatial dimensions)
- Translation invariance
- Reduce parameters

```python
class MaxPool2D:
    """Max pooling layer."""
    
    def __init__(self, pool_size: int = 2, stride: int = 2):
        self.pool_size = pool_size
        self.stride = stride
        self.cache = {}
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass."""
        batch_size, channels, H, W = X.shape
        
        H_out = (H - self.pool_size) // self.stride + 1
        W_out = (W - self.pool_size) // self.stride + 1
        
        Z = np.zeros((batch_size, channels, H_out, W_out))
        
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                
                X_patch = X[:, :, h_start:h_end, w_start:w_end]
                Z[:, :, i, j] = np.max(X_patch, axis=(2, 3))
        
        self.cache['X'] = X
        self.cache['Z'] = Z
        
        return Z
    
    def backward(self, dZ: np.ndarray) -> np.ndarray:
        """Backward pass."""
        X = self.cache['X']
        batch_size, channels, H, W = X.shape
        _, _, H_out, W_out = dZ.shape
        
        dX = np.zeros_like(X)
        
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                
                X_patch = X[:, :, h_start:h_end, w_start:w_end]
                
                # Create mask for max values
                for b in range(batch_size):
                    for c in range(channels):
                        patch = X_patch[b, c]
                        max_val = np.max(patch)
                        mask = (patch == max_val)
                        dX[b, c, h_start:h_end, w_start:w_end] += mask * dZ[b, c, i, j]
        
        return dX
```

### Average Pooling
$$ Z[i, j] = \frac{1}{k^2} \sum_{m, n \in \text{pool}} X[i \cdot s + m, j \cdot s + n] $$

## 4. Complete CNN Architecture

```python
class SimpleCNN:
    """Simple CNN for image classification."""
    
    def __init__(self, input_shape=(3, 32, 32), num_classes=10):
        self.layers = []
        
        # Conv1: 3x32x32 -> 32x32x32
        self.layers.append(Conv2D(3, 32, kernel_size=3, padding=1))
        self.layers.append(ReLU())
        
        # Pool1: 32x32x32 -> 32x16x16
        self.layers.append(MaxPool2D(pool_size=2, stride=2))
        
        # Conv2: 32x16x16 -> 64x16x16
        self.layers.append(Conv2D(32, 64, kernel_size=3, padding=1))
        self.layers.append(ReLU())
        
        # Pool2: 64x16x16 -> 64x8x8
        self.layers.append(MaxPool2D(pool_size=2, stride=2))
        
        # Flatten: 64x8x8 -> 4096
        self.layers.append(Flatten())
        
        # FC1: 4096 -> 512
        self.layers.append(Linear(64*8*8, 512))
        self.layers.append(ReLU())
        
        # FC2: 512 -> 10
        self.layers.append(Linear(512, num_classes))
    
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward(self, dZ):
        for layer in reversed(self.layers):
            dZ = layer.backward(dZ)
        return dZ
```

## 5. Receptive Field

**Definition:** Region in input that affects a particular output neuron.

**Calculation:**
$$ r_{l+1} = r_l + (k - 1) \cdot \prod_{i=1}^{l} s_i $$

**Example:**
- Layer 1: Conv 3×3, stride 1 → $r_1 = 3$
- Layer 2: Conv 3×3, stride 1 → $r_2 = 5$
- Layer 3: Conv 3×3, stride 2 → $r_3 = 11$

## Summary
CNNs use local connectivity and parameter sharing to efficiently process spatial data. Convolutional layers extract features, pooling layers downsample, and fully connected layers classify.

**Next:** CNN Architectures (AlexNet, VGG, ResNet).
