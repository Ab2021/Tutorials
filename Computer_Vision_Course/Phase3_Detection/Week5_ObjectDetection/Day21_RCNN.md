# Day 12: R-CNN Family

## Overview
The R-CNN family revolutionized object detection by combining region proposals with deep learning. This lesson covers R-CNN, Fast R-CNN, Faster R-CNN, and Mask R-CNN.

## 1. R-CNN (2014)

**Architecture:**
1. **Region Proposals:** Selective Search (~2000 proposals)
2. **Feature Extraction:** CNN (AlexNet) for each proposal
3. **Classification:** SVM classifier
4. **Bounding Box Regression:** Linear regressor

**Pipeline:**
```
Image → Selective Search → Warp regions → CNN → SVM + Regressor
```

### Selective Search

**Algorithm:**
1. Over-segmentation (Felzenszwalb's algorithm)
2. Iteratively merge similar regions
3. Generate proposals from all scales

```python
import cv2

def selective_search(image, mode='fast'):
    """
    Selective Search for region proposals.
    
    Args:
        image: Input image
        mode: 'fast' or 'quality'
    
    Returns:
        proposals: List of (x, y, w, h) proposals
    """
    # Create Selective Search Segmentation Object
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    
    # Set input image
    ss.setBaseImage(image)
    
    # Set mode
    if mode == 'fast':
        ss.switchToSelectiveSearchFast()
    else:
        ss.switchToSelectiveSearchQuality()
    
    # Run selective search
    proposals = ss.process()
    
    return proposals

# Usage
image = cv2.imread('image.jpg')
proposals = selective_search(image, mode='fast')
print(f"Number of proposals: {len(proposals)}")  # ~2000
```

### R-CNN Implementation

```python
import torch
import torch.nn as nn
import torchvision

class RCNN(nn.Module):
    """R-CNN for object detection."""
    
    def __init__(self, num_classes=20):
        super().__init__()
        
        # Feature extractor (AlexNet)
        alexnet = torchvision.models.alexnet(pretrained=True)
        self.features = alexnet.features
        self.avgpool = alexnet.avgpool
        
        # Classifier (SVM in original, FC here for simplicity)
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes + 1),  # +1 for background
        )
        
        # Bounding box regressor
        self.bbox_regressor = nn.Linear(4096, 4 * num_classes)
    
    def forward(self, x):
        """
        Args:
            x: (N, 3, 227, 227) warped region proposals
        
        Returns:
            class_scores: (N, num_classes + 1)
            bbox_deltas: (N, 4 * num_classes)
        """
        # Extract features
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Get features before final layer
        features = x
        for layer in self.classifier[:-1]:
            features = layer(features)
        
        # Classification
        class_scores = self.classifier[-1](features)
        
        # Bounding box regression
        bbox_deltas = self.bbox_regressor(features)
        
        return class_scores, bbox_deltas

def train_rcnn(model, proposals, labels, boxes, optimizer, criterion):
    """Train R-CNN on one image."""
    model.train()
    
    # Warp proposals to fixed size (227x227 for AlexNet)
    warped_proposals = []
    for proposal in proposals:
        x, y, w, h = proposal
        region = image[y:y+h, x:x+w]
        warped = cv2.resize(region, (227, 227))
        warped_proposals.append(warped)
    
    warped_proposals = torch.stack(warped_proposals).cuda()
    
    # Forward
    class_scores, bbox_deltas = model(warped_proposals)
    
    # Classification loss
    cls_loss = criterion(class_scores, labels)
    
    # Bounding box regression loss (only for positive samples)
    pos_mask = labels > 0
    if pos_mask.sum() > 0:
        bbox_loss = F.smooth_l1_loss(
            bbox_deltas[pos_mask],
            boxes[pos_mask]
        )
    else:
        bbox_loss = 0
    
    # Total loss
    loss = cls_loss + bbox_loss
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

**Problems with R-CNN:**
1. **Slow:** CNN forward pass for each proposal (~2000 per image)
2. **Storage:** Features must be cached (hundreds of GB)
3. **Multi-stage:** Separate training for CNN, SVM, regressor
4. **Fixed proposals:** Can't adapt proposals

**Performance:**
- mAP: 53.3% on PASCAL VOC 2012
- Speed: 47 seconds per image (GPU)

## 2. Fast R-CNN (2015)

**Key Innovation:** Share computation across proposals.

**Architecture:**
1. **Single CNN:** Process entire image once
2. **RoI Pooling:** Extract fixed-size features for each proposal
3. **Multi-task Loss:** Joint training of classifier and regressor

**Pipeline:**
```
Image → CNN → Feature Map → RoI Pooling → FC → Classification + Regression
```

### RoI Pooling

**Problem:** Proposals have different sizes, but FC layers need fixed input.

**Solution:** Divide proposal into fixed grid, max-pool each cell.

```python
import torch
import torch.nn.functional as F

def roi_pooling(feature_map, rois, output_size=(7, 7)):
    """
    RoI Pooling layer.
    
    Args:
        feature_map: (1, C, H, W) feature map
        rois: (N, 4) region proposals [x1, y1, x2, y2]
        output_size: (h, w) output size
    
    Returns:
        pooled: (N, C, h, w) pooled features
    """
    N = len(rois)
    C, H, W = feature_map.shape[1:]
    h, w = output_size
    
    pooled = torch.zeros(N, C, h, w).to(feature_map.device)
    
    for i, roi in enumerate(rois):
        x1, y1, x2, y2 = roi.int()
        
        # Clip to feature map bounds
        x1 = max(0, min(x1, W - 1))
        y1 = max(0, min(y1, H - 1))
        x2 = max(0, min(x2, W - 1))
        y2 = max(0, min(y2, H - 1))
        
        # Extract RoI
        roi_feature = feature_map[:, :, y1:y2+1, x1:x2+1]
        
        # Adaptive max pooling to output size
        pooled[i] = F.adaptive_max_pool2d(roi_feature, output_size)
    
    return pooled

# PyTorch built-in (faster)
from torchvision.ops import RoIPool

roi_pool = RoIPool(output_size=(7, 7), spatial_scale=1/16)  # stride 16
pooled = roi_pool(feature_map, rois)
```

### Fast R-CNN Implementation

```python
class FastRCNN(nn.Module):
    """Fast R-CNN for object detection."""
    
    def __init__(self, num_classes=20, backbone='vgg16'):
        super().__init__()
        
        # Backbone
        if backbone == 'vgg16':
            vgg = torchvision.models.vgg16(pretrained=True)
            self.backbone = vgg.features
            in_features = 512 * 7 * 7
        
        # RoI Pooling
        self.roi_pool = RoIPool(output_size=(7, 7), spatial_scale=1/16)
        
        # Classifier head
        self.fc6 = nn.Linear(in_features, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        
        # Output layers
        self.cls_score = nn.Linear(4096, num_classes + 1)
        self.bbox_pred = nn.Linear(4096, 4 * (num_classes + 1))
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()
    
    def forward(self, image, rois):
        """
        Args:
            image: (1, 3, H, W) input image
            rois: (N, 5) [batch_idx, x1, y1, x2, y2]
        
        Returns:
            cls_scores: (N, num_classes + 1)
            bbox_preds: (N, 4 * (num_classes + 1))
        """
        # Extract features
        features = self.backbone(image)
        
        # RoI pooling
        pooled = self.roi_pool(features, rois)
        pooled = pooled.view(pooled.size(0), -1)
        
        # FC layers
        x = self.dropout(self.relu(self.fc6(pooled)))
        x = self.dropout(self.relu(self.fc7(x)))
        
        # Classification and regression
        cls_scores = self.cls_score(x)
        bbox_preds = self.bbox_pred(x)
        
        return cls_scores, bbox_preds

# Multi-task loss
class FastRCNNLoss(nn.Module):
    """Fast R-CNN multi-task loss."""
    
    def __init__(self):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss()
    
    def forward(self, cls_scores, bbox_preds, labels, bbox_targets):
        """
        Args:
            cls_scores: (N, num_classes + 1)
            bbox_preds: (N, 4 * (num_classes + 1))
            labels: (N,) class labels
            bbox_targets: (N, 4) bounding box targets
        
        Returns:
            loss: Scalar loss
        """
        # Classification loss
        cls_loss = self.cls_loss(cls_scores, labels)
        
        # Bounding box regression loss (only for foreground)
        fg_mask = labels > 0
        
        if fg_mask.sum() > 0:
            # Select bbox predictions for ground truth class
            fg_labels = labels[fg_mask]
            fg_bbox_preds = bbox_preds[fg_mask]
            
            # Reshape: (N, num_classes + 1, 4)
            fg_bbox_preds = fg_bbox_preds.view(-1, cls_scores.size(1), 4)
            
            # Select predictions for ground truth class
            fg_bbox_preds = fg_bbox_preds[range(len(fg_labels)), fg_labels]
            
            # Smooth L1 loss
            bbox_loss = F.smooth_l1_loss(fg_bbox_preds, bbox_targets[fg_mask])
        else:
            bbox_loss = 0
        
        # Total loss
        total_loss = cls_loss + bbox_loss
        
        return total_loss, cls_loss, bbox_loss
```

**Improvements over R-CNN:**
1. **Speed:** 9× faster training, 213× faster testing
2. **Accuracy:** Higher mAP (66.9% vs 62.4% on VOC 2012)
3. **Single-stage training:** End-to-end
4. **No disk storage:** No feature caching

**Remaining bottleneck:** Region proposals (Selective Search)

## 3. Faster R-CNN (2015)

**Key Innovation:** Region Proposal Network (RPN) - learn proposals.

**Architecture:**
1. **Shared CNN:** Feature extraction
2. **RPN:** Generate proposals from features
3. **Fast R-CNN:** Classification and refinement

**Pipeline:**
```
Image → CNN → RPN (proposals) → RoI Pooling → Classification + Regression
```

### Region Proposal Network (RPN)

**Idea:** Sliding window on feature map, predict objectness and boxes.

**For each location:**
- k anchor boxes (different scales/ratios)
- 2k objectness scores (object vs background)
- 4k box coordinates

```python
class RPN(nn.Module):
    """Region Proposal Network."""
    
    def __init__(self, in_channels=512, num_anchors=9):
        super().__init__()
        
        # 3x3 conv
        self.conv = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        
        # Classification (objectness)
        self.cls_logits = nn.Conv2d(512, num_anchors * 2, kernel_size=1)
        
        # Regression (box deltas)
        self.bbox_pred = nn.Conv2d(512, num_anchors * 4, kernel_size=1)
    
    def forward(self, features):
        """
        Args:
            features: (1, C, H, W) feature map
        
        Returns:
            objectness: (1, 2*num_anchors, H, W)
            bbox_deltas: (1, 4*num_anchors, H, W)
        """
        # Shared conv
        x = F.relu(self.conv(features))
        
        # Objectness scores
        objectness = self.cls_logits(x)
        
        # Bounding box deltas
        bbox_deltas = self.bbox_pred(x)
        
        return objectness, bbox_deltas

def generate_proposals(objectness, bbox_deltas, anchors, 
                      pre_nms_top_n=12000, post_nms_top_n=2000, nms_thresh=0.7):
    """
    Generate proposals from RPN outputs.
    
    Args:
        objectness: (1, 2*k, H, W) objectness scores
        bbox_deltas: (1, 4*k, H, W) box deltas
        anchors: (H*W*k, 4) anchor boxes
        pre_nms_top_n: Keep top-n before NMS
        post_nms_top_n: Keep top-n after NMS
        nms_thresh: NMS threshold
    
    Returns:
        proposals: (N, 4) proposed boxes
    """
    # Get objectness scores (foreground probability)
    scores = F.softmax(objectness, dim=1)[:, 1::2]  # Take foreground scores
    scores = scores.reshape(-1)
    
    # Decode boxes
    bbox_deltas = bbox_deltas.reshape(-1, 4)
    proposals = decode_boxes(bbox_deltas, anchors)
    
    # Clip to image bounds
    proposals = clip_boxes(proposals, image_shape)
    
    # Remove small boxes
    keep = filter_small_boxes(proposals, min_size=16)
    proposals = proposals[keep]
    scores = scores[keep]
    
    # Sort by score and keep top-k before NMS
    sorted_indices = torch.argsort(scores, descending=True)[:pre_nms_top_n]
    proposals = proposals[sorted_indices]
    scores = scores[sorted_indices]
    
    # NMS
    keep = nms(proposals, scores, nms_thresh)[:post_nms_top_n]
    proposals = proposals[keep]
    
    return proposals
```

### Faster R-CNN Implementation

```python
class FasterRCNN(nn.Module):
    """Faster R-CNN for object detection."""
    
    def __init__(self, num_classes=20, backbone='vgg16'):
        super().__init__()
        
        # Backbone
        if backbone == 'vgg16':
            vgg = torchvision.models.vgg16(pretrained=True)
            self.backbone = vgg.features
            in_channels = 512
        
        # RPN
        self.rpn = RPN(in_channels=in_channels, num_anchors=9)
        
        # RoI Pooling
        self.roi_pool = RoIPool(output_size=(7, 7), spatial_scale=1/16)
        
        # Detection head (Fast R-CNN)
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.cls_score = nn.Linear(4096, num_classes + 1)
        self.bbox_pred = nn.Linear(4096, 4 * (num_classes + 1))
    
    def forward(self, image, gt_boxes=None):
        """
        Args:
            image: (1, 3, H, W)
            gt_boxes: (M, 4) ground truth boxes (training only)
        
        Returns:
            If training: losses
            If testing: detections
        """
        # Extract features
        features = self.backbone(image)
        
        # RPN
        objectness, bbox_deltas = self.rpn(features)
        
        # Generate proposals
        proposals = generate_proposals(objectness, bbox_deltas, self.anchors)
        
        # Training: sample proposals
        if self.training:
            proposals, labels, bbox_targets = sample_proposals(
                proposals, gt_boxes, 
                pos_fraction=0.25, batch_size=128
            )
        
        # RoI Pooling
        pooled = self.roi_pool(features, proposals)
        pooled = pooled.view(pooled.size(0), -1)
        
        # Detection head
        x = F.relu(self.fc6(pooled))
        x = F.relu(self.fc7(x))
        cls_scores = self.cls_score(x)
        bbox_preds = self.bbox_pred(x)
        
        if self.training:
            # Compute losses
            rpn_cls_loss, rpn_bbox_loss = compute_rpn_loss(...)
            rcnn_cls_loss, rcnn_bbox_loss = compute_rcnn_loss(...)
            
            return {
                'rpn_cls_loss': rpn_cls_loss,
                'rpn_bbox_loss': rpn_bbox_loss,
                'rcnn_cls_loss': rcnn_cls_loss,
                'rcnn_bbox_loss': rcnn_bbox_loss,
            }
        else:
            # Post-process detections
            detections = post_process(cls_scores, bbox_preds, proposals)
            return detections
```

**Performance:**
- mAP: 73.2% on PASCAL VOC 2007
- Speed: 5 FPS (GPU)
- **10× faster than Fast R-CNN**

## Summary
R-CNN family: R-CNN (region proposals + CNN), Fast R-CNN (RoI pooling), Faster R-CNN (RPN for proposals). Each iteration improved speed and accuracy.

**Next:** YOLO and single-shot detectors.
