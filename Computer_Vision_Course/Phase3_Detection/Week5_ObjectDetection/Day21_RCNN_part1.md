# Day 12 Deep Dive: Mask R-CNN and Advanced R-CNN Variants

## 1. Mask R-CNN (2017)

**Extension:** Add instance segmentation to Faster R-CNN.

**Architecture:**
- Faster R-CNN + Mask branch
- RoIAlign (improved RoI Pooling)
- Fully Convolutional Network (FCN) for masks

**Pipeline:**
```
Image → CNN → RPN → RoIAlign → Classification + Box + Mask
```

### RoIAlign

**Problem with RoI Pooling:** Quantization causes misalignment.

**Example:**
- RoI: [12.4, 8.7, 45.2, 32.1]
- RoI Pooling rounds: [12, 8, 45, 32]
- **Misalignment:** 0.4, 0.7, 0.2, 0.1 pixels

**Solution:** Bilinear interpolation (no quantization).

```python
import torch
import torch.nn.functional as F
from torchvision.ops import RoIAlign

class MaskRCNN(nn.Module):
    """Mask R-CNN for instance segmentation."""
    
    def __init__(self, num_classes=80):
        super().__init__()
        
        # Backbone (ResNet-50-FPN)
        self.backbone = ResNetFPN()
        
        # RPN
        self.rpn = RPN()
        
        # RoIAlign (instead of RoIPool)
        self.roi_align = RoIAlign(output_size=(7, 7), spatial_scale=1/16, 
                                  sampling_ratio=2)
        
        # Box head
        self.box_head = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )
        self.cls_score = nn.Linear(1024, num_classes + 1)
        self.bbox_pred = nn.Linear(1024, 4 * (num_classes + 1))
        
        # Mask head (FCN)
        self.mask_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),  # Upsample
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1),  # Per-class masks
        )
    
    def forward(self, images, targets=None):
        """
        Args:
            images: (B, 3, H, W)
            targets: List of dicts with 'boxes', 'labels', 'masks'
        
        Returns:
            If training: losses
            If testing: detections with masks
        """
        # Extract features
        features = self.backbone(images)
        
        # RPN
        proposals, rpn_losses = self.rpn(features, targets)
        
        # RoIAlign for box head
        box_features = self.roi_align(features, proposals)
        box_features = box_features.view(box_features.size(0), -1)
        
        # Box predictions
        box_features = self.box_head(box_features)
        cls_scores = self.cls_score(box_features)
        bbox_preds = self.bbox_pred(box_features)
        
        # RoIAlign for mask head (14x14 instead of 7x7)
        mask_roi_align = RoIAlign(output_size=(14, 14), spatial_scale=1/16, 
                                  sampling_ratio=2)
        mask_features = mask_roi_align(features, proposals)
        
        # Mask predictions
        mask_preds = self.mask_head(mask_features)  # (N, num_classes, 28, 28)
        
        if self.training:
            # Compute losses
            box_losses = compute_box_losses(cls_scores, bbox_preds, targets)
            mask_losses = compute_mask_losses(mask_preds, targets)
            
            return {
                **rpn_losses,
                **box_losses,
                **mask_losses,
            }
        else:
            # Post-process
            detections = post_process_detections(
                cls_scores, bbox_preds, mask_preds, proposals
            )
            return detections

def compute_mask_losses(mask_preds, targets):
    """
    Compute mask loss.
    
    Args:
        mask_preds: (N, num_classes, 28, 28)
        targets: List of target masks
    
    Returns:
        mask_loss: Scalar loss
    """
    # Only compute loss for positive RoIs
    pos_mask = targets['labels'] > 0
    
    if pos_mask.sum() == 0:
        return {'mask_loss': torch.tensor(0.0)}
    
    # Select masks for ground truth classes
    pos_labels = targets['labels'][pos_mask]
    pos_mask_preds = mask_preds[pos_mask, pos_labels]  # (N_pos, 28, 28)
    pos_mask_targets = targets['masks'][pos_mask]  # (N_pos, 28, 28)
    
    # Binary cross-entropy loss (per-pixel)
    mask_loss = F.binary_cross_entropy_with_logits(
        pos_mask_preds,
        pos_mask_targets
    )
    
    return {'mask_loss': mask_loss}
```

### Mask Prediction

**Key insight:** Decouple mask and class prediction.
- Predict K masks (one per class)
- Use class prediction to select mask
- **Benefit:** No competition between classes

**Mask representation:**
- 28×28 binary mask per class
- Upsampled to RoI size during inference

## 2. Cascade R-CNN (2018)

**Problem:** Single IoU threshold for training is suboptimal.
- Low threshold: Many false positives
- High threshold: Few training samples

**Solution:** Cascade of detectors with increasing IoU thresholds.

**Architecture:**
```
RPN → RoI → Head1 (IoU=0.5) → Head2 (IoU=0.6) → Head3 (IoU=0.7)
```

```python
class CascadeRCNN(nn.Module):
    """Cascade R-CNN with multiple detection heads."""
    
    def __init__(self, num_classes=80, iou_thresholds=[0.5, 0.6, 0.7]):
        super().__init__()
        
        self.iou_thresholds = iou_thresholds
        self.num_stages = len(iou_thresholds)
        
        # Backbone + RPN
        self.backbone = ResNetFPN()
        self.rpn = RPN()
        
        # Multiple detection heads
        self.box_heads = nn.ModuleList([
            DetectionHead(num_classes) for _ in range(self.num_stages)
        ])
    
    def forward(self, images, targets=None):
        # Extract features
        features = self.backbone(images)
        
        # RPN
        proposals = self.rpn(features)
        
        # Cascade stages
        all_losses = []
        
        for stage, (box_head, iou_thresh) in enumerate(
            zip(self.box_heads, self.iou_thresholds)
        ):
            # Sample proposals with current IoU threshold
            if self.training:
                proposals = sample_proposals(proposals, targets, iou_thresh)
            
            # Detection head
            cls_scores, bbox_preds = box_head(features, proposals)
            
            # Refine proposals for next stage
            if stage < self.num_stages - 1:
                proposals = refine_proposals(proposals, bbox_preds)
            
            # Compute losses
            if self.training:
                losses = compute_losses(cls_scores, bbox_preds, targets, iou_thresh)
                all_losses.append(losses)
        
        if self.training:
            return combine_losses(all_losses)
        else:
            return post_process(cls_scores, bbox_preds, proposals)
```

**Benefits:**
- **Higher quality:** Each stage trained on higher quality proposals
- **Better localization:** Progressive refinement
- **Performance:** +2-4% mAP improvement

## 3. Feature Pyramid Networks (FPN) Deep Dive

**Multi-scale detection:**
- P2 (stride 4): Small objects
- P3 (stride 8): Medium objects
- P4 (stride 16): Large objects
- P5 (stride 32): Very large objects

```python
class FPN(nn.Module):
    """Feature Pyramid Network with top-down pathway."""
    
    def __init__(self, in_channels_list=[256, 512, 1024, 2048], out_channels=256):
        super().__init__()
        
        # Lateral connections (1x1 conv)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])
        
        # Output convs (3x3 conv to reduce aliasing)
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])
        
        # Additional coarse level (P6)
        self.p6_conv = nn.Conv2d(in_channels_list[-1], out_channels, 
                                kernel_size=3, stride=2, padding=1)
    
    def forward(self, features):
        """
        Args:
            features: [C2, C3, C4, C5] from backbone
        
        Returns:
            fpn_features: [P2, P3, P4, P5, P6]
        """
        # Build lateral connections
        laterals = [
            lateral_conv(feat)
            for feat, lateral_conv in zip(features, self.lateral_convs)
        ]
        
        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            # Upsample higher-level feature
            upsampled = F.interpolate(
                laterals[i],
                size=laterals[i-1].shape[-2:],
                mode='nearest'
            )
            # Add to lateral
            laterals[i-1] = laterals[i-1] + upsampled
        
        # Apply output convs
        fpn_features = [
            output_conv(lateral)
            for lateral, output_conv in zip(laterals, self.output_convs)
        ]
        
        # Add P6 (for very large objects)
        p6 = self.p6_conv(features[-1])
        fpn_features.append(p6)
        
        return fpn_features
```

## 4. Training Strategies

### Anchor Assignment

**Positive anchors:**
- IoU > 0.7 with any ground truth
- Highest IoU anchor for each ground truth

**Negative anchors:**
- IoU < 0.3 with all ground truths

**Ignored:**
- 0.3 ≤ IoU ≤ 0.7

```python
def assign_anchors_to_gt(anchors, gt_boxes, pos_iou_thresh=0.7, neg_iou_thresh=0.3):
    """
    Assign anchors to ground truth boxes.
    
    Args:
        anchors: (N, 4) anchor boxes
        gt_boxes: (M, 4) ground truth boxes
        pos_iou_thresh: Positive threshold
        neg_iou_thresh: Negative threshold
    
    Returns:
        labels: (N,) 1 for positive, 0 for negative, -1 for ignored
        matched_gt_boxes: (N, 4) matched ground truth boxes
    """
    # Compute IoU matrix
    iou_matrix = compute_iou_matrix(anchors, gt_boxes)  # (N, M)
    
    # For each anchor, find best matching GT
    max_iou, matched_gt_idx = iou_matrix.max(dim=1)
    
    # Initialize labels as ignored
    labels = torch.full((len(anchors),), -1, dtype=torch.long)
    
    # Negative anchors
    labels[max_iou < neg_iou_thresh] = 0
    
    # Positive anchors
    labels[max_iou >= pos_iou_thresh] = 1
    
    # For each GT, assign highest IoU anchor as positive
    gt_max_iou, gt_max_anchor_idx = iou_matrix.max(dim=0)
    labels[gt_max_anchor_idx] = 1
    
    # Get matched GT boxes
    matched_gt_boxes = gt_boxes[matched_gt_idx]
    
    return labels, matched_gt_boxes
```

### Balanced Sampling

**Problem:** Imbalance between positive and negative samples.

**Solution:** Sample fixed ratio (e.g., 1:3 positive:negative).

```python
def sample_rois(labels, num_samples=256, pos_fraction=0.25):
    """
    Sample RoIs for training.
    
    Args:
        labels: (N,) labels (1=pos, 0=neg, -1=ignore)
        num_samples: Total samples
        pos_fraction: Fraction of positive samples
    
    Returns:
        sampled_indices: Indices of sampled RoIs
    """
    pos_indices = torch.where(labels == 1)[0]
    neg_indices = torch.where(labels == 0)[0]
    
    # Number of positive samples
    num_pos = int(num_samples * pos_fraction)
    num_pos = min(num_pos, len(pos_indices))
    
    # Number of negative samples
    num_neg = num_samples - num_pos
    num_neg = min(num_neg, len(neg_indices))
    
    # Random sampling
    pos_sampled = pos_indices[torch.randperm(len(pos_indices))[:num_pos]]
    neg_sampled = neg_indices[torch.randperm(len(neg_indices))[:num_neg]]
    
    # Combine
    sampled_indices = torch.cat([pos_sampled, neg_sampled])
    
    return sampled_indices
```

## 5. Inference Optimization

### Multi-Scale Testing

**Idea:** Test at multiple scales and combine results.

```python
def multi_scale_test(model, image, scales=[0.5, 1.0, 1.5]):
    """Multi-scale testing."""
    all_detections = []
    
    for scale in scales:
        # Resize image
        h, w = image.shape[-2:]
        new_h, new_w = int(h * scale), int(w * scale)
        scaled_image = F.interpolate(image, size=(new_h, new_w), mode='bilinear')
        
        # Detect
        detections = model(scaled_image)
        
        # Scale boxes back
        detections['boxes'] /= scale
        
        all_detections.append(detections)
    
    # Combine detections (NMS across scales)
    combined = combine_detections(all_detections)
    
    return combined
```

### Soft-NMS

**Better than hard NMS for crowded scenes.**

```python
def soft_nms(boxes, scores, iou_thresh=0.5, sigma=0.5, score_thresh=0.001):
    """Soft-NMS with Gaussian decay."""
    N = len(boxes)
    
    for i in range(N):
        # Find max score
        max_idx = i + torch.argmax(scores[i:])
        
        # Swap
        boxes[[i, max_idx]] = boxes[[max_idx, i]]
        scores[[i, max_idx]] = scores[[max_idx, i]]
        
        # Decay overlapping boxes
        ious = compute_iou_vectorized(boxes[i:i+1], boxes[i+1:])
        
        # Gaussian decay
        decay = torch.exp(-(ious ** 2) / sigma)
        scores[i+1:] *= decay
    
    # Filter by score
    keep = scores > score_thresh
    
    return boxes[keep], scores[keep]
```

## Summary
Advanced R-CNN variants include Mask R-CNN (instance segmentation with RoIAlign), Cascade R-CNN (progressive refinement), and FPN (multi-scale features). Training strategies involve careful anchor assignment and balanced sampling.
