# Day 11 Deep Dive: Advanced Detection Concepts

## 1. Anchor Boxes

**Concept:** Pre-defined boxes at different scales and aspect ratios.

**Why anchors:**
- Handle multiple scales
- Handle different aspect ratios
- Provide reference for regression

**Anchor generation:**
```python
import numpy as np

def generate_anchors(base_size=16, scales=[8, 16, 32], ratios=[0.5, 1, 2]):
    """
    Generate anchor boxes.
    
    Args:
        base_size: Base anchor size
        scales: Anchor scales
        ratios: Anchor aspect ratios
    
    Returns:
        anchors: (num_anchors, 4) array [x1, y1, x2, y2]
    """
    anchors = []
    
    for scale in scales:
        for ratio in ratios:
            # Compute width and height
            h = base_size * scale
            w = h * ratio
            
            # Center at origin
            x1 = -w / 2
            y1 = -h / 2
            x2 = w / 2
            y2 = h / 2
            
            anchors.append([x1, y1, x2, y2])
    
    return np.array(anchors)

def generate_anchor_grid(feature_map_size, stride, base_anchors):
    """
    Generate anchors for entire feature map.
    
    Args:
        feature_map_size: (H, W) of feature map
        stride: Stride of feature map relative to input
        base_anchors: (num_anchors, 4) base anchor boxes
    
    Returns:
        all_anchors: (H * W * num_anchors, 4) array
    """
    H, W = feature_map_size
    num_anchors = len(base_anchors)
    
    # Generate grid centers
    shift_x = np.arange(0, W) * stride
    shift_y = np.arange(0, H) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    
    shifts = np.stack([shift_x.ravel(), shift_y.ravel(),
                      shift_x.ravel(), shift_y.ravel()], axis=1)
    
    # Add shifts to base anchors
    all_anchors = (base_anchors.reshape((1, num_anchors, 4)) +
                   shifts.reshape((len(shifts), 1, 4)))
    
    all_anchors = all_anchors.reshape((-1, 4))
    
    return all_anchors

# Example
base_anchors = generate_anchors(base_size=16, scales=[8, 16, 32], ratios=[0.5, 1, 2])
print(f"Base anchors shape: {base_anchors.shape}")  # (9, 4)

all_anchors = generate_anchor_grid((50, 50), stride=16, base_anchors=base_anchors)
print(f"All anchors shape: {all_anchors.shape}")  # (22500, 4)
```

## 2. Bounding Box Regression

**Problem:** Refine anchor boxes to match ground truth.

**Parameterization:**
$$ t_x = \frac{x - x_a}{w_a}, \quad t_y = \frac{y - y_a}{h_a} $$
$$ t_w = \log\frac{w}{w_a}, \quad t_h = \log\frac{h}{h_a} $$

where $(x, y, w, h)$ is ground truth, $(x_a, y_a, w_a, h_a)$ is anchor.

**Inverse (apply regression):**
$$ x = t_x \cdot w_a + x_a $$
$$ y = t_y \cdot h_a + y_a $$
$$ w = w_a \cdot \exp(t_w) $$
$$ h = h_a \cdot \exp(t_h) $$

```python
def encode_boxes(gt_boxes, anchors):
    """
    Encode ground truth boxes relative to anchors.
    
    Args:
        gt_boxes: (N, 4) [x1, y1, x2, y2]
        anchors: (N, 4) [x1, y1, x2, y2]
    
    Returns:
        targets: (N, 4) [tx, ty, tw, th]
    """
    # Convert to center format
    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights
    
    anchor_widths = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]
    anchor_ctr_x = anchors[:, 0] + 0.5 * anchor_widths
    anchor_ctr_y = anchors[:, 1] + 0.5 * anchor_heights
    
    # Encode
    tx = (gt_ctr_x - anchor_ctr_x) / anchor_widths
    ty = (gt_ctr_y - anchor_ctr_y) / anchor_heights
    tw = np.log(gt_widths / anchor_widths)
    th = np.log(gt_heights / anchor_heights)
    
    targets = np.stack([tx, ty, tw, th], axis=1)
    
    return targets

def decode_boxes(deltas, anchors):
    """
    Decode predicted deltas to boxes.
    
    Args:
        deltas: (N, 4) [tx, ty, tw, th]
        anchors: (N, 4) [x1, y1, x2, y2]
    
    Returns:
        boxes: (N, 4) [x1, y1, x2, y2]
    """
    # Anchor properties
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights
    
    # Decode
    tx, ty, tw, th = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]
    
    pred_ctr_x = tx * widths + ctr_x
    pred_ctr_y = ty * heights + ctr_y
    pred_w = np.exp(tw) * widths
    pred_h = np.exp(th) * heights
    
    # Convert to corner format
    pred_boxes = np.zeros_like(deltas)
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h
    
    return pred_boxes
```

## 3. Focal Loss

**Problem:** Class imbalance in one-stage detectors (many background anchors).

**Cross-Entropy Loss:**
$$ CE(p, y) = -\log(p_t) $$

where $p_t = p$ if $y=1$, else $p_t = 1-p$.

**Focal Loss:**
$$ FL(p_t) = -(1-p_t)^\gamma \log(p_t) $$

**Effect:** Down-weight easy examples, focus on hard examples.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss for dense object detection."""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (N, num_classes) logits
            targets: (N,) class labels
        
        Returns:
            loss: Scalar focal loss
        """
        # Compute probabilities
        p = F.softmax(predictions, dim=1)
        
        # Get probabilities for target class
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        p_t = p[range(len(targets)), targets]
        
        # Focal loss
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss
        
        return focal_loss.mean()

# Binary focal loss (for binary classification)
class BinaryFocalLoss(nn.Module):
    """Binary focal loss."""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (N,) predicted probabilities
            targets: (N,) binary labels (0 or 1)
        
        Returns:
            loss: Scalar focal loss
        """
        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            predictions, targets.float(), reduction='none'
        )
        
        # Probabilities
        p_t = torch.exp(-bce_loss)
        
        # Focal loss
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce_loss
        
        return focal_loss.mean()
```

## 4. Feature Pyramid Networks (FPN)

**Motivation:** Detect objects at multiple scales.

**Architecture:**
- Bottom-up pathway: Standard CNN
- Top-down pathway: Upsample high-level features
- Lateral connections: Merge features

```python
import torch.nn as nn

class FPN(nn.Module):
    """Feature Pyramid Network."""
    
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        
        # Lateral connections (1x1 conv to reduce channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])
        
        # Output convs (3x3 conv to reduce aliasing)
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])
    
    def forward(self, features):
        """
        Args:
            features: List of feature maps from backbone
                     [C2, C3, C4, C5] with decreasing spatial resolution
        
        Returns:
            fpn_features: List of FPN feature maps
                         [P2, P3, P4, P5] with same spatial resolutions
        """
        # Build top-down pathway
        laterals = [lateral_conv(feat) 
                   for feat, lateral_conv in zip(features, self.lateral_convs)]
        
        # Start from coarsest level
        fpn_features = [laterals[-1]]
        
        # Top-down pathway
        for i in range(len(laterals) - 2, -1, -1):
            # Upsample
            upsampled = F.interpolate(
                fpn_features[0],
                size=laterals[i].shape[-2:],
                mode='nearest'
            )
            
            # Add lateral connection
            fpn_feat = upsampled + laterals[i]
            fpn_features.insert(0, fpn_feat)
        
        # Apply output convs
        fpn_features = [output_conv(feat)
                       for feat, output_conv in zip(fpn_features, self.output_convs)]
        
        return fpn_features

# Usage with ResNet
class ResNetFPN(nn.Module):
    """ResNet backbone with FPN."""
    
    def __init__(self, num_classes=80):
        super().__init__()
        
        # ResNet backbone
        resnet = torchvision.models.resnet50(pretrained=True)
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # C2: 256 channels
        self.layer2 = resnet.layer2  # C3: 512 channels
        self.layer3 = resnet.layer3  # C4: 1024 channels
        self.layer4 = resnet.layer4  # C5: 2048 channels
        
        # FPN
        self.fpn = FPN(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256
        )
    
    def forward(self, x):
        # Bottom-up pathway
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        # FPN
        fpn_features = self.fpn([c2, c3, c4, c5])
        
        return fpn_features
```

## 5. Multi-Scale Training

**Idea:** Train with images of different sizes to improve robustness.

```python
class MultiScaleDataset(torch.utils.data.Dataset):
    """Dataset with multi-scale training."""
    
    def __init__(self, base_dataset, scales=[480, 512, 544, 576, 608]):
        self.base_dataset = base_dataset
        self.scales = scales
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, target = self.base_dataset[idx]
        
        # Random scale
        scale = np.random.choice(self.scales)
        
        # Resize
        transform = T.Compose([
            T.Resize(scale),
            T.ToTensor(),
        ])
        
        image = transform(image)
        
        # Adjust bounding boxes
        if 'boxes' in target:
            h_scale = scale / image.shape[1]
            w_scale = scale / image.shape[2]
            target['boxes'][:, [0, 2]] *= w_scale
            target['boxes'][:, [1, 3]] *= h_scale
        
        return image, target
```

## 6. Data Augmentation for Detection

**Challenges:** Must transform both images and bounding boxes.

```python
class DetectionAugmentation:
    """Data augmentation for object detection."""
    
    def __init__(self):
        self.transforms = [
            self.random_flip,
            self.random_crop,
            self.color_jitter,
        ]
    
    def random_flip(self, image, boxes, p=0.5):
        """Random horizontal flip."""
        if np.random.rand() < p:
            image = np.fliplr(image)
            
            # Flip boxes
            width = image.shape[1]
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        
        return image, boxes
    
    def random_crop(self, image, boxes, min_scale=0.3):
        """Random crop."""
        h, w = image.shape[:2]
        
        # Random crop size
        scale = np.random.uniform(min_scale, 1.0)
        crop_h = int(h * scale)
        crop_w = int(w * scale)
        
        # Random crop position
        top = np.random.randint(0, h - crop_h + 1)
        left = np.random.randint(0, w - crop_w + 1)
        
        # Crop image
        image = image[top:top+crop_h, left:left+crop_w]
        
        # Adjust boxes
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] - left, 0, crop_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] - top, 0, crop_h)
        
        # Remove boxes with zero area
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep = areas > 0
        boxes = boxes[keep]
        
        return image, boxes
    
    def color_jitter(self, image, boxes):
        """Color jittering."""
        # Brightness
        image = image * np.random.uniform(0.5, 1.5)
        
        # Contrast
        mean = image.mean()
        image = (image - mean) * np.random.uniform(0.5, 1.5) + mean
        
        # Saturation (for RGB)
        if image.shape[2] == 3:
            gray = image.mean(axis=2, keepdims=True)
            image = gray + (image - gray) * np.random.uniform(0.5, 1.5)
        
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        return image, boxes
    
    def __call__(self, image, boxes):
        """Apply random augmentations."""
        for transform in self.transforms:
            if np.random.rand() < 0.5:
                image, boxes = transform(image, boxes)
        
        return image, boxes
```

## 7. Hard Negative Mining

**Problem:** Too many easy negative examples.
**Solution:** Mine hard negatives during training.

```python
def hard_negative_mining(loss, labels, neg_pos_ratio=3):
    """
    Hard negative mining.
    
    Args:
        loss: (N,) loss for each anchor
        labels: (N,) labels (0 for background)
        neg_pos_ratio: Ratio of negatives to positives
    
    Returns:
        mask: (N,) boolean mask for selected samples
    """
    pos_mask = labels > 0
    num_pos = pos_mask.sum()
    
    # Number of negatives to keep
    num_neg = min(neg_pos_ratio * num_pos, (~pos_mask).sum())
    
    # Sort negative losses
    neg_loss = loss.clone()
    neg_loss[pos_mask] = -float('inf')  # Ignore positives
    
    # Select top-k hard negatives
    _, neg_idx = neg_loss.topk(num_neg)
    
    # Create mask
    neg_mask = torch.zeros_like(labels, dtype=torch.bool)
    neg_mask[neg_idx] = True
    
    # Combine positive and negative masks
    mask = pos_mask | neg_mask
    
    return mask
```

## Summary
Advanced detection concepts include anchor boxes for handling multiple scales, bounding box regression for refinement, focal loss for class imbalance, FPN for multi-scale features, and hard negative mining for better training.
