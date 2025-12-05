# Day 3: Feature Pyramid Networks & Multi-Scale Processing
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 1: Deep Learning Foundations

---

> **📝 Content Creator Instructions:**
> This document specifically targets the "Scale Problem" in robotics—efficiently detecting both the large truck 2 meters away and the small traffic cone 50 meters away.
> - **Focus:** FPN, BiFPN, and modern fusion architectures.
> - **Code:** Implementation of FPN from scratch and integration with a backbone.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the trade-offs between Image Pyramids vs. Feature Pyramids.
2.  **Implement** a standard Feature Pyramid Network (FPN) in PyTorch.
3.  **Differentiate** between FPN, PANet, and BiFPN (used in EfficientDet).
4.  **Design** an anchor-free detection head that operates on multi-scale features.
5.  **Deploy** a multi-scale detector on a quadruped robot (Mini Pupper) for robust obstacle avoidance.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- NVIDIA GPU for training.
- Mini Pupper (Raspberry Pi 4) or similar edge device for testing.

### Software Environment
```bash
pip install torch torchvision
pip install numpy
pip install pycocotools  # For COCO Metric evaluation
```

### Prior Knowledge
- CNN Backbones (ResNet).
- Understanding of Receptive Fields.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Scale Variation Problem

Robots operate in dynamic environments where object distances vary wildly.
*   **Close objects:** Require high resolution (texture details) but cover large spatial areas.
*   **Far objects:** Require semantic context but cover few pixels.

#### 1.1 Approaches to Multi-Scale Detection

1.  **Featurized Image Pyramid:**
    *   Resize image multiple times ($0.5x, 1x, 2x$) and run the CNN on each.
    *   *Pros:* Best accuracy.
    *   *Cons:* **Extremely Slow (>4x latency).** Unusable for real-time robotics.

2.  **Single Feature Map (SSD style):**
    *   Detect large objects on deep layers (low res) and small objects on shallow layers (high res).
    *   *Cons:* Shallow layers lack "semantic meaning" (they only see edges/corners), so detection is poor for small objects.

3.  **Feature Pyramid Network (FPN):** The Gold Standard.
    *   **Bottom-Up Method:** Standard ConvNet backbone (C2, C3, C4, C5).
    *   **Top-Down Method:** Upsample deep, semantically strong features (C5) and add them to high-res shallow features (C4, C3).
    *   *Result:* All levels (P2, P3, P4, P5) differ in scale but share strong semantic richness.

#### 1.2 Top-Down Pathway & Lateral Connections

Let $C_i$ be the feature map at stage $i$.
Let $P_i$ be the feature map after FPN.

1.  **Start:** $P_5 = \text{Conv}_{1 \times 1}(C_5)$
2.  **Iterate:**
    $$ P_i = \text{Upsample}(P_{i+1}) + \text{Conv}_{1 \times 1}(C_i) $$
3.  **Smooth:** $P_i = \text{Conv}_{3 \times 3}(P_i)$ (To reduce aliasing from upsampling).

### 🔹 Part 2: Advanced Multi-Scale Architectures

#### 2.1 PANet (Path Aggregation Network)
FPN passes semantic info top-down. But low-level info (localization accuracy) needs to travel bottom-up.
**PANet** adds a second **bottom-up pathway** ($N_2 \to N_5$) on top of FPN ($P_5 \to P_2$).

#### 2.2 BiFPN (Bidirectional FPN)
Introduced in **EfficientDet**.
1.  **Weighted Fusion:** Instead of simple sum ($+ \text{upsample}$), learn weights: $O = \sum w_i \cdot I_i / (\epsilon + \sum w_i)$.
2.  **Cross-Scale Connections:** Remove nodes with only 1 input; add extra edges for better flow.

**Robotics Relevance:** BiFPN offers the best accuracy/FLOPs trade-off for mobile robots.

---

## 💻 Implementation: Building an FPN

We will build an FPN class that attaches to any backbone (e.g., ResNet).

### 🛠️ Project Structure
```text
day3_fpn/
├── data/
│   └── coco_mini/
├── models/
│   ├── backbone.py
│   └── fpn.py
├── train_detector.py
└── visualize_pyramid.py
```

### 👨‍💻 Code Implementation

#### 1. Feature Pyramid Network (`models/fpn.py`)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        """
        Args:
            in_channels_list: List of channel counts for [C2, C3, C4, C5]
            out_channels: Channel count for Pyramid features [P2, P3, P4, P5]
        """
        super().__init__()
        
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_channels in in_channels_list:
            # 1x1 conv to match channel dimensions
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
            # 3x3 conv for anti-aliasing
            self.fpn_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )

    def forward(self, inputs):
        """
        Args:
            inputs: List of features [C2, C3, C4, C5] from backbone
        Returns:
            outputs: List of features [P2, P3, P4, P5]
        """
        # Build Top-Down Pathway
        # Start from the last layer (C5)
        last_inner = self.lateral_convs[-1](inputs[-1])
        results = [self.fpn_convs[-1](last_inner)]
        
        # Iterate backwards from C4 to C2
        for idx in range(len(inputs) - 2, -1, -1):
            inner_lateral = self.lateral_convs[idx](inputs[idx])
            inner_top_down = F.interpolate(last_inner, size=inner_lateral.shape[-2:], mode="nearest")
            
            # Sum Lateral and Top-Down
            last_inner = inner_lateral + inner_top_down
            
            # Smooth
            results.insert(0, self.fpn_convs[idx](last_inner))
            
        return results
```

#### 2. Connecting to ResNet Backbone (`models/backbone.py`)

```python
import torchvision.models as models
from fpn import FPN

class ResNetFPN(nn.Module):
    def __init__(self, backbone_name='resnet50', out_channels=256):
        super().__init__()
        
        # Load backbone
        backbone = models.resnet50(pretrained=True)
        
        # Extract layers. ResNet has layers: layer1, layer2, layer3, layer4
        # Corresponding to C2, C3, C4, C5
        
        self.body = nn.ModuleList([
            nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1),
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        ])
        
        # ResNet50 channels: [256, 512, 1024, 2048]
        in_channels_list = [256, 512, 1024, 2048]
        
        self.fpn = FPN(in_channels_list, out_channels)
        
    def forward(self, x):
        features = []
        for stage in self.body:
            x = stage(x)
            features.append(x)
            
        # Pass C2...C5 to FPN
        pyramid_features = self.fpn(features)
        return pyramid_features
```

#### 3. Anchor-Free Head (FCOS style)

Instead of complex anchors, we predict a 4D vector `(l, t, r, b)` (distances to 4 sides of bounding box) at every pixel location $(x, y)$.

```python
class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        # Shared tower for classification
        self.cls_tower = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_classes, 3, padding=1) # Sigmoid output
        )
        
        # Shared tower for regression
        self.reg_tower = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, 4, 3, padding=1) # ReLU output (distances > 0)
        )
        
    def forward(self, features):
        cls_logits = []
        reg_preds = []
        
        for feature in features:
            cls_logits.append(self.cls_tower(feature))
            reg_preds.append(torch.exp(self.reg_tower(feature))) # exp to ensure positive
            
        return cls_logits, reg_preds
```

---

## 🔬 Lab Exercise: Visualizing the Pyramid

### 1. Lab Objectives
- Run an image through RefNetFPN.
- Visualize the feature activation maps at P2 (high res) vs P5 (low res).
- Observe how small objects disappear in P5 but remain visible in P2.

### 2. Step-by-Step Guide

#### Phase A: Visualization Script

```python
import torch
import matplotlib.pyplot as plt
import cv2
from models.backbone import ResNetFPN

def visualize_feature_map(fmap):
    # fmap: (1, C, H, W)
    # Average across channels
    heatmap = torch.mean(fmap, dim=1).squeeze().detach().cpu().numpy()
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    return heatmap

# Load Model
model = ResNetFPN()
model.eval()

# Load Image
img = cv2.imread("street_scene.jpg")
img_tensor = preprocess(img) # Resize, Normalize to tensor

# Inference
pyramid = model(img_tensor.unsqueeze(0))

# Plot
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for i, p_level in enumerate(pyramid):
    heatmap = visualize_feature_map(p_level)
    axes[i].imshow(heatmap, cmap='jet')
    axes[i].set_title(f"P{i+2} Feature Map")
    
plt.show()
```

### 3. Expected Observations
- **P2:** Sharp edges, fine details (e.g., lane markings, pedestrians).
- **P5:** broad "blobs" of activation covering large cars or buildings.

---

## 🚀 Project: Mini Pupper Obstacle Avoidance

**Goal:** Detect "Feet" (Humans) and "Blocks" (Toys) to navigate a Mini Pupper robot. We employ the Multi-Scale detector to ensure we see blocks right in front (large) and humans far away (small).

### 1. Integration Strategy
Since Mini Pupper uses a Raspberry Pi 4, ResNet-50 is too heavy (~3 FPS).
**Solution:** Swap backbone for **MobileNetV3**.

1.  **Modify Backbone:** Use `torchvision.models.mobilenet_v3_small`.
2.  **Modify Channels:** MobileNet channel list is `[16, 24, 48, 576]`.
3.  **FPN Output:** Reduce `out_channels` to 64 instead of 256.

### 2. ROS 2 Node Logic

```python
class ObstacleAvoidanceNode(Node):
    def __init__(self):
        # ... Init ...
        self.detector = MobileNetFPN(out_channels=64)
        
    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg)
        dets = self.detector(frame)
        
        # Decision Logic
        closest_dist = 999
        
        for det in dets:
             cls, score, box = det
             if score < 0.5: continue
             
             # Estimate distance by bounding box height (simple monocular cue)
             height_px = box[3] - box[1]
             dist_m = FOCAL_LENGTH * REAL_HEIGHT / height_px
             
             if dist_m < closest_dist:
                 closest_dist = dist_m
        
        if closest_dist < 0.5:
            self.stop()
            self.turn_left()
        else:
            self.move_forward()
```

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. FPN Aliasing
*   **Symptom:** Checkerboard patterns in output feature maps.
*   **Cause:** Upsampling followed by addition without smoothing.
*   **Fix:** Ensure the $3 \times 3$ convolution is applied *after* the sum of lateral and top-down features.

#### 2. Channel Mismatch
*   **Symptom:** `RuntimeError: sizes of tensors must match`.
*   **Cause:** Lateral connection $1 \times 1$ conv output channels don't match the Upsampled feature channels.
*   **Fix:** Ensure all `lateral_convs` output `out_channels` (e.g., 256).

---

## ⚡ Optimization: BiFPN Design

For advanced optimization, implement the weighted sum of BiFPN:

```python
class WeightedFusion(nn.Module):
    def __init__(self, num_inputs=2):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs))
        self.epsilon = 1e-4
        
    def forward(self, inputs):
        # inputs: list of tensors
        w = torch.relu(self.weights)
        weight_sum = w.sum() + self.epsilon
        
        weighted_inputs = [inputs[i] * w[i] / weight_sum for i in range(len(inputs))]
        return sum(weighted_inputs)
```

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why not just use layer C5 for detection?
    *   **A:** C5 has low resolution (stride 32). A small object (e.g., $16 \times 16$ pixels) would map to $< 1$ pixel in C5 and vanish.
2.  **Q:** What is the benefit of the top-down pathway?
    *   **A:** It hallucinates high-level semantic context onto high-resolution layers.
3.  **Q:** Does FPN increase parameter count significantly?
    *   **A:** Moderate increase. The lateral and output convs add parameters, but it's usually negligible compared to the backbone fully connected layers (which are removed).

### Challenge Task
> **Task:** Implement PANet's bottom-up path augmentation.
> 1. Take FPN outputs [P2, P3, P4, P5].
> 2. Create path N2 -> N3 -> N4 -> N5.
> 3. N3 = Conv(N2_downsampled + P3).

---

## 📚 Further Reading
- **FPN:** Lin et al., "Feature Pyramid Networks for Object Detection" (CVPR 2017).
- **RetinaNet:** Lin et al., "Focal Loss for Dense Object Detection".
- **EfficientDet:** Tan et al., "EfficientDet: Scalable and Efficient Object Detection".

---

**Day 3 Complete**
