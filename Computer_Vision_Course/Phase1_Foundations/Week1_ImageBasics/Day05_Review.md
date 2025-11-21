# Day 5: Week 1 Review - Image Fundamentals

## Overview
This review consolidates Days 1-4, covering image representation, processing operations, edge detection, and classical feature descriptors. We'll implement a complete image analysis pipeline.

## Key Concepts Recap

### 1. Image Representation
**Digital image as function:**
$$ I: \Omega \subset \mathbb{R}^2 \rightarrow \mathbb{R}^c $$

**Color spaces:**
- RGB: Device-dependent, additive
- HSV: Perceptual, separates intensity
- Lab: Perceptually uniform
- YCbCr: Luma/chroma separation

### 2. Filtering Operations
**Convolution:**
$$ (I * K)(x,y) = \sum_{i,j} I(x-i, y-j) \cdot K(i,j) $$

**Common kernels:**
- Gaussian: $G_\sigma(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$
- Sobel: Edge detection
- Laplacian: Second derivative

### 3. Edge Detection
**Canny algorithm:**
1. Gaussian smoothing
2. Gradient computation
3. Non-maximum suppression
4. Double thresholding
5. Edge tracking by hysteresis

**Mathematical foundation:**
$$ \nabla I = \begin{bmatrix} \frac{\partial I}{\partial x} \\ \frac{\partial I}{\partial y} \end{bmatrix}, \quad |\nabla I| = \sqrt{I_x^2 + I_y^2} $$

### 4. Feature Descriptors
**SIFT pipeline:**
- Scale-space: DoG pyramid
- Keypoint localization: Sub-pixel refinement
- Orientation: Gradient histogram
- Descriptor: 128D normalized vector

**Binary descriptors (ORB):**
- Fast computation
- Hamming distance matching
- Rotation invariant via orientation

## Complete Image Analysis Pipeline

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

class ImageAnalyzer:
    """Complete image analysis pipeline."""
    
    def __init__(self, image_path: str):
        self.original = cv2.imread(image_path)
        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        
    def analyze_color_distribution(self) -> dict:
        """Analyze color distribution across channels."""
        histograms = {}
        colors = ('b', 'g', 'r')
        
        for i, color in enumerate(colors):
            hist = cv2.calcHist([self.original], [i], None, [256], [0, 256])
            histograms[color] = hist.flatten()
            
        # Compute statistics
        stats = {
            'mean': np.mean(self.original, axis=(0,1)),
            'std': np.std(self.original, axis=(0,1)),
            'histograms': histograms
        }
        return stats
    
    def apply_enhancement(self) -> np.ndarray:
        """Apply adaptive histogram equalization."""
        # Convert to LAB
        lab = cv2.cvtColor(self.original, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l)
        
        # Merge and convert back
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def detect_edges_multiscale(self) -> List[np.ndarray]:
        """Detect edges at multiple scales."""
        edges = []
        sigmas = [1.0, 2.0, 4.0]
        
        for sigma in sigmas:
            # Gaussian smoothing
            blurred = cv2.GaussianBlur(self.gray, (0, 0), sigma)
            
            # Canny edge detection
            # Automatic threshold selection
            median = np.median(blurred)
            lower = int(max(0, 0.66 * median))
            upper = int(min(255, 1.33 * median))
            
            edge = cv2.Canny(blurred, lower, upper)
            edges.append(edge)
            
        return edges
    
    def extract_features(self, method='sift') -> Tuple[List, np.ndarray]:
        """Extract features using specified method."""
        if method == 'sift':
            detector = cv2.SIFT_create(
                nfeatures=500,
                contrastThreshold=0.04,
                edgeThreshold=10
            )
        elif method == 'orb':
            detector = cv2.ORB_create(
                nfeatures=500,
                scaleFactor=1.2,
                nlevels=8
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        keypoints, descriptors = detector.detectAndCompute(self.gray, None)
        return keypoints, descriptors
    
    def compute_hog(self, cell_size=(8,8), block_size=(2,2)) -> np.ndarray:
        """Compute HOG descriptor."""
        # Resize to standard size
        img_resized = cv2.resize(self.gray, (128, 256))
        
        # Compute gradients
        gx = cv2.Sobel(img_resized, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(img_resized, cv2.CV_32F, 0, 1, ksize=1)
        
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx) * (180 / np.pi) % 180
        
        # Cell histograms
        n_cells_x = img_resized.shape[1] // cell_size[0]
        n_cells_y = img_resized.shape[0] // cell_size[1]
        n_bins = 9
        
        cell_histograms = np.zeros((n_cells_y, n_cells_x, n_bins))
        
        for i in range(n_cells_y):
            for j in range(n_cells_x):
                y_start = i * cell_size[1]
                y_end = (i + 1) * cell_size[1]
                x_start = j * cell_size[0]
                x_end = (j + 1) * cell_size[0]
                
                cell_mag = magnitude[y_start:y_end, x_start:x_end]
                cell_ori = orientation[y_start:y_end, x_start:x_end]
                
                # Compute histogram
                hist, _ = np.histogram(
                    cell_ori,
                    bins=n_bins,
                    range=(0, 180),
                    weights=cell_mag
                )
                cell_histograms[i, j] = hist
        
        # Block normalization
        hog_features = []
        for i in range(n_cells_y - block_size[1] + 1):
            for j in range(n_cells_x - block_size[0] + 1):
                block = cell_histograms[i:i+block_size[1], j:j+block_size[0]]
                block_vector = block.flatten()
                
                # L2 normalization
                norm = np.linalg.norm(block_vector) + 1e-5
                normalized = block_vector / norm
                hog_features.append(normalized)
        
        return np.concatenate(hog_features)
    
    def visualize_analysis(self):
        """Comprehensive visualization."""
        fig = plt.figure(figsize=(20, 12))
        
        # Original and enhanced
        ax1 = plt.subplot(3, 4, 1)
        ax1.imshow(cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        enhanced = self.apply_enhancement()
        ax2 = plt.subplot(3, 4, 2)
        ax2.imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        ax2.set_title('Enhanced (CLAHE)')
        ax2.axis('off')
        
        # Color histograms
        stats = self.analyze_color_distribution()
        ax3 = plt.subplot(3, 4, 3)
        for color, hist in stats['histograms'].items():
            ax3.plot(hist, color=color, alpha=0.7)
        ax3.set_title('Color Histograms')
        ax3.set_xlabel('Intensity')
        ax3.set_ylabel('Frequency')
        
        # Multi-scale edges
        edges = self.detect_edges_multiscale()
        for i, (edge, sigma) in enumerate(zip(edges, [1.0, 2.0, 4.0])):
            ax = plt.subplot(3, 4, 5 + i)
            ax.imshow(edge, cmap='gray')
            ax.set_title(f'Edges (σ={sigma})')
            ax.axis('off')
        
        # SIFT features
        kp_sift, _ = self.extract_features('sift')
        img_sift = cv2.drawKeypoints(
            self.gray, kp_sift, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        ax8 = plt.subplot(3, 4, 8)
        ax8.imshow(img_sift, cmap='gray')
        ax8.set_title(f'SIFT ({len(kp_sift)} keypoints)')
        ax8.axis('off')
        
        # ORB features
        kp_orb, _ = self.extract_features('orb')
        img_orb = cv2.drawKeypoints(
            self.gray, kp_orb, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        ax9 = plt.subplot(3, 4, 9)
        ax9.imshow(img_orb, cmap='gray')
        ax9.set_title(f'ORB ({len(kp_orb)} keypoints)')
        ax9.axis('off')
        
        # HOG visualization
        hog_descriptor = self.compute_hog()
        ax10 = plt.subplot(3, 4, 10)
        ax10.plot(hog_descriptor[:500])
        ax10.set_title(f'HOG Descriptor (dim={len(hog_descriptor)})')
        ax10.set_xlabel('Dimension')
        ax10.set_ylabel('Value')
        
        plt.tight_layout()
        plt.savefig('week1_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()

# Usage
analyzer = ImageAnalyzer('sample_image.jpg')
analyzer.visualize_analysis()

# Print statistics
stats = analyzer.analyze_color_distribution()
print(f"Mean RGB: {stats['mean']}")
print(f"Std RGB: {stats['std']}")

kp_sift, desc_sift = analyzer.extract_features('sift')
print(f"\nSIFT: {len(kp_sift)} keypoints, descriptor shape: {desc_sift.shape}")

kp_orb, desc_orb = analyzer.extract_features('orb')
print(f"ORB: {len(kp_orb)} keypoints, descriptor shape: {desc_orb.shape}")

hog = analyzer.compute_hog()
print(f"HOG: {len(hog)} dimensions")
```

## Practice Problems

### Problem 1: Image Quality Assessment
Implement a no-reference image quality metric based on edge strength and contrast.

### Problem 2: Feature Matching Pipeline
Create a complete pipeline: load two images → detect features → match → geometric verification → visualize.

### Problem 3: Multi-scale Analysis
Implement Gaussian pyramid and analyze image statistics at each level.

## Next Week Preview
Week 2 focuses on **Deep Learning Fundamentals**: neural networks, CNNs, backpropagation, and optimization techniques.
