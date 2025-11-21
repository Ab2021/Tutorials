# Day 12 Interview Questions: R-CNN Family

## Q1: Explain the evolution from R-CNN to Faster R-CNN.
**Answer:**

**R-CNN (2014):**
- **Pipeline:** Selective Search → Warp → CNN (2000×) → SVM
- **Speed:** 47s per image
- **Problems:** Slow, multi-stage training, huge storage

**Fast R-CNN (2015):**
- **Innovation:** Share CNN computation
- **Pipeline:** CNN → RoI Pooling → FC → Classification + Regression
- **Speed:** 2s per image (23× faster)
- **Improvement:** Single-stage training, no feature caching
- **Bottleneck:** Still uses Selective Search

**Faster R-CNN (2015):**
- **Innovation:** Region Proposal Network (RPN)
- **Pipeline:** CNN → RPN → RoI Pooling → Classification + Regression
- **Speed:** 0.2s per image (10× faster than Fast R-CNN)
- **Improvement:** End-to-end trainable, learned proposals

**Performance comparison:**
| Model | mAP (VOC 2007) | Speed | Key Innovation |
|-------|----------------|-------|----------------|
| R-CNN | 58.5% | 47s | CNN for detection |
| Fast R-CNN | 66.9% | 2s | RoI Pooling |
| Faster R-CNN | 73.2% | 0.2s | RPN |

## Q2: What is RoI Pooling and what problem does it solve?
**Answer:**

**Problem:** Region proposals have different sizes, but FC layers need fixed input.

**Solution:** RoI Pooling divides each proposal into fixed grid and max-pools.

**Algorithm:**
1. Project proposal onto feature map (divide by stride)
2. Divide into h×w grid (e.g., 7×7)
3. Max-pool each cell

**Example:**
```
Proposal: [0, 0, 25, 25] on feature map
Output size: 7×7
Cell size: 25/7 ≈ 3.57 pixels

Grid cells:
[0:3, 0:3], [0:3, 3:7], [0:3, 7:10], ...
[3:7, 0:3], [3:7, 3:7], [3:7, 7:10], ...
...

Max-pool each cell → 7×7 output
```

**Benefits:**
- Fixed-size output for any input size
- Differentiable (can backpropagate)
- Efficient (single forward pass)

**Limitation:** Quantization causes misalignment (solved by RoIAlign in Mask R-CNN).

## Q3: How does the Region Proposal Network (RPN) work?
**Answer:**

**RPN:** Sliding window on feature map to predict proposals.

**Architecture:**
```
Feature Map (H×W×C)
    ↓ 3×3 conv (512 channels)
    ↓ Split
    ├→ 1×1 conv → Objectness (2k scores)
    └→ 1×1 conv → Box Deltas (4k values)
```

**For each location:**
- k anchor boxes (different scales/ratios)
- 2k objectness scores (object vs background)
- 4k box coordinates (refinement)

**Anchors:**
- Scales: [128, 256, 512] pixels
- Ratios: [1:2, 1:1, 2:1]
- Total: 3 scales × 3 ratios = 9 anchors per location

**Training:**
- **Positive:** IoU > 0.7 with GT or highest IoU for each GT
- **Negative:** IoU < 0.3 with all GT
- **Loss:** Classification (cross-entropy) + Regression (smooth L1)

**Inference:**
1. Generate anchors for all locations
2. Apply predicted deltas
3. Clip to image bounds
4. Remove small boxes
5. Sort by objectness score
6. NMS (threshold 0.7)
7. Keep top-N (e.g., 2000)

**Benefits:**
- Fast (shares CNN features)
- Accurate (learned proposals)
- End-to-end trainable

## Q4: Compare RoI Pooling vs RoIAlign.
**Answer:**

**RoI Pooling (Fast/Faster R-CNN):**
- Quantizes coordinates to integers
- Quantizes bin boundaries
- **Misalignment:** Can be off by 0-1 pixels

**Example:**
```
RoI: [12.4, 8.7, 45.2, 32.1]
Quantized: [12, 8, 45, 32]
Error: [0.4, 0.7, 0.2, 0.1]
```

**RoIAlign (Mask R-CNN):**
- No quantization
- Bilinear interpolation at exact locations
- **Precise alignment**

**Algorithm:**
```
For each bin:
  1. Compute exact bin boundaries (no rounding)
  2. Sample points within bin (e.g., 2×2 grid)
  3. Bilinear interpolation at each sample point
  4. Max/average pool the samples
```

**Impact:**
- **Detection:** Small improvement (~1% mAP)
- **Segmentation:** Large improvement (~3% mask AP)
- **Reason:** Segmentation needs pixel-level precision

**Implementation:**
```python
# RoI Pooling
roi_pool = RoIPool(output_size=(7, 7), spatial_scale=1/16)

# RoIAlign
roi_align = RoIAlign(output_size=(7, 7), spatial_scale=1/16, 
                     sampling_ratio=2)  # 2×2 samples per bin
```

## Q5: What is Mask R-CNN and how does it extend Faster R-CNN?
**Answer:**

**Extension:** Add instance segmentation branch.

**Architecture:**
```
Faster R-CNN
    ├→ Box Branch (classification + regression)
    └→ Mask Branch (FCN for segmentation)
```

**Key components:**

**1. RoIAlign:** Replace RoI Pooling for better alignment.

**2. Mask Branch:**
- Input: 14×14 RoI features
- Architecture: 4× conv3×3 → deconv2×2 → conv1×1
- Output: K masks (28×28), one per class

**3. Decoupled prediction:**
- Predict K binary masks (one per class)
- Use class prediction to select mask
- **Benefit:** No competition between classes

**Loss:**
$$ L = L_{cls} + L_{box} + L_{mask} $$

**Mask loss:** Binary cross-entropy per pixel (only for positive RoIs).

**Training:**
- Same as Faster R-CNN + mask targets
- Mask targets: Binary masks from instance annotations

**Inference:**
1. Detect objects (Faster R-CNN)
2. For each detection, predict mask
3. Select mask based on predicted class
4. Threshold mask (> 0.5)
5. Resize to original RoI size

**Performance:**
- Box mAP: 37.1% (COCO)
- Mask mAP: 33.1% (COCO)
- Speed: 5 FPS (GPU)

## Q6: Implement a simple RPN.
**Answer:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RPN(nn.Module):
    """Region Proposal Network."""
    
    def __init__(self, in_channels=512, num_anchors=9):
        super().__init__()
        
        # Shared 3×3 conv
        self.conv = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        
        # Objectness (2 scores per anchor: bg/fg)
        self.cls_logits = nn.Conv2d(512, num_anchors * 2, kernel_size=1)
        
        # Box regression (4 values per anchor)
        self.bbox_pred = nn.Conv2d(512, num_anchors * 4, kernel_size=1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for layer in [self.conv, self.cls_logits, self.bbox_pred]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)
    
    def forward(self, features):
        """
        Args:
            features: (B, C, H, W) feature map
        
        Returns:
            objectness: (B, 2*num_anchors, H, W)
            bbox_deltas: (B, 4*num_anchors, H, W)
        """
        # Shared features
        x = F.relu(self.conv(features))
        
        # Predictions
        objectness = self.cls_logits(x)
        bbox_deltas = self.bbox_pred(x)
        
        return objectness, bbox_deltas

def rpn_loss(objectness, bbox_deltas, anchors, gt_boxes, 
             pos_iou_thresh=0.7, neg_iou_thresh=0.3):
    """
    Compute RPN loss.
    
    Args:
        objectness: (B, 2*k, H, W) objectness scores
        bbox_deltas: (B, 4*k, H, W) box deltas
        anchors: (N, 4) anchor boxes
        gt_boxes: (M, 4) ground truth boxes
    
    Returns:
        cls_loss: Classification loss
        reg_loss: Regression loss
    """
    # Assign anchors to GT
    labels, matched_gt_boxes = assign_anchors_to_gt(
        anchors, gt_boxes, pos_iou_thresh, neg_iou_thresh
    )
    
    # Reshape predictions
    B, _, H, W = objectness.shape
    objectness = objectness.view(B, -1, 2, H, W).permute(0, 1, 3, 4, 2)
    objectness = objectness.reshape(-1, 2)
    bbox_deltas = bbox_deltas.view(B, -1, 4, H, W).permute(0, 1, 3, 4, 2)
    bbox_deltas = bbox_deltas.reshape(-1, 4)
    
    # Classification loss (only for non-ignored anchors)
    valid_mask = labels >= 0
    cls_loss = F.cross_entropy(
        objectness[valid_mask],
        labels[valid_mask],
        reduction='mean'
    )
    
    # Regression loss (only for positive anchors)
    pos_mask = labels == 1
    if pos_mask.sum() > 0:
        # Encode GT boxes
        targets = encode_boxes(matched_gt_boxes[pos_mask], anchors[pos_mask])
        
        # Smooth L1 loss
        reg_loss = F.smooth_l1_loss(
            bbox_deltas[pos_mask],
            targets,
            reduction='mean'
        )
    else:
        reg_loss = torch.tensor(0.0).to(objectness.device)
    
    return cls_loss, reg_loss

# Usage
rpn = RPN(in_channels=512, num_anchors=9)
features = torch.randn(1, 512, 50, 50)
objectness, bbox_deltas = rpn(features)

print(f"Objectness: {objectness.shape}")  # (1, 18, 50, 50)
print(f"Box deltas: {bbox_deltas.shape}")  # (1, 36, 50, 50)
```

## Q7: What is Cascade R-CNN and why does it improve performance?
**Answer:**

**Problem:** Single IoU threshold is suboptimal.
- Low threshold (0.5): Accepts low-quality boxes, many false positives
- High threshold (0.7): Few training samples, hard to train

**Solution:** Cascade of detectors with increasing IoU thresholds.

**Architecture:**
```
Stage 1: IoU = 0.5 → Refine boxes
Stage 2: IoU = 0.6 → Refine boxes  
Stage 3: IoU = 0.7 → Final predictions
```

**Training:**
- Each stage trained with its own IoU threshold
- Later stages see higher-quality proposals
- Progressive refinement

**Benefits:**
1. **Quality-aware:** Each stage specialized for its quality level
2. **Better localization:** Progressive refinement improves boxes
3. **No overfitting:** Each stage trained on appropriate samples

**Results:**
- Faster R-CNN: 36.4% mAP (COCO)
- Cascade R-CNN: 40.3% mAP (+3.9%)

**Key insight:** Detector trained at IoU=0.5 performs poorly at IoU=0.7. Cascade trains specialized detectors for each quality level.

## Q8: How to handle multi-scale objects in detection?
**Answer:**

**Strategies:**

**1. Image Pyramid:**
- Resize image to multiple scales
- Run detector on each scale
- Combine detections with NMS
- **Pros:** Simple, effective
- **Cons:** Slow (multiple forward passes)

**2. Feature Pyramid (FPN):**
- Build pyramid from CNN features
- Detect at multiple feature levels
- **Pros:** Fast (single forward pass), accurate
- **Cons:** More complex architecture

**3. Multi-scale anchors:**
- Use anchors of different sizes
- Single feature map
- **Pros:** Simple
- **Cons:** Limited scale range

**4. Deformable convolutions:**
- Learn spatial transformations
- Adapt receptive field to object scale
- **Pros:** Flexible
- **Cons:** Harder to train

**Best practice:** FPN + multi-scale anchors
- FPN: Handle large scale variations (4×-32× stride)
- Anchors: Handle small variations within each level

**Example (RetinaNet):**
```
P3 (stride 8):  Anchors [32, 40, 50] pixels
P4 (stride 16): Anchors [64, 80, 101] pixels
P5 (stride 32): Anchors [128, 161, 203] pixels
P6 (stride 64): Anchors [256, 322, 406] pixels
P7 (stride 128): Anchors [512, 645, 812] pixels
```

## Q9: Compare one-stage vs two-stage detectors for production.
**Answer:**

**Two-Stage (Faster R-CNN, Mask R-CNN):**

**Pros:**
- Higher accuracy (especially for small objects)
- Better localization
- Instance segmentation (Mask R-CNN)

**Cons:**
- Slower (5-15 FPS)
- More complex
- Harder to deploy

**One-Stage (YOLO, RetinaNet):**

**Pros:**
- Faster (30-150 FPS)
- Simpler architecture
- Easier to deploy

**Cons:**
- Lower accuracy (especially small objects)
- Class imbalance issues

**Decision matrix:**

| Requirement | Recommendation |
|-------------|----------------|
| Real-time (>30 FPS) | One-stage (YOLO) |
| High accuracy | Two-stage (Faster R-CNN) |
| Small objects | Two-stage or FPN-based |
| Instance segmentation | Mask R-CNN |
| Edge deployment | One-stage (lightweight) |
| Crowded scenes | Two-stage (better NMS) |

**Modern trend:** Gap is closing
- RetinaNet (one-stage) matches Faster R-CNN accuracy
- EfficientDet achieves best accuracy/speed trade-off

**Production recommendation:**
- **Start with:** YOLOv5/YOLOv8 (good balance)
- **If need more accuracy:** EfficientDet or Faster R-CNN
- **If need segmentation:** Mask R-CNN

## Q10: Design a complete Faster R-CNN training pipeline.
**Answer:**

```python
class FasterRCNNTrainer:
    """Complete Faster R-CNN training pipeline."""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Optimizer (different LR for backbone and heads)
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[8, 11],
            gamma=0.1
        )
    
    def _create_optimizer(self):
        """Create optimizer with different LR for different parts."""
        params = [
            {'params': self.model.backbone.parameters(), 'lr': self.config.lr * 0.1},
            {'params': self.model.rpn.parameters(), 'lr': self.config.lr},
            {'params': self.model.roi_head.parameters(), 'lr': self.config.lr},
        ]
        return torch.optim.SGD(params, momentum=0.9, weight_decay=0.0001)
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        rpn_cls_loss_total = 0
        rpn_reg_loss_total = 0
        rcnn_cls_loss_total = 0
        rcnn_reg_loss_total = 0
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = [img.cuda() for img in images]
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
            
            # Forward
            loss_dict = self.model(images, targets)
            
            # Total loss
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward
            self.optimizer.zero_grad()
            losses.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += losses.item()
            rpn_cls_loss_total += loss_dict['rpn_cls_loss'].item()
            rpn_reg_loss_total += loss_dict['rpn_reg_loss'].item()
            rcnn_cls_loss_total += loss_dict['rcnn_cls_loss'].item()
            rcnn_reg_loss_total += loss_dict['rcnn_reg_loss'].item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] '
                      f'Loss: {losses.item():.4f}')
        
        # Average losses
        n = len(self.train_loader)
        return {
            'total_loss': total_loss / n,
            'rpn_cls_loss': rpn_cls_loss_total / n,
            'rpn_reg_loss': rpn_reg_loss_total / n,
            'rcnn_cls_loss': rcnn_cls_loss_total / n,
            'rcnn_reg_loss': rcnn_reg_loss_total / n,
        }
    
    def validate(self):
        """Validate on validation set."""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = [img.cuda() for img in images]
                
                # Inference
                predictions = self.model(images)
                
                all_predictions.extend(predictions)
                all_targets.extend(targets)
        
        # Compute mAP
        evaluator = COCOEvaluator()
        metrics = evaluator.evaluate(all_predictions, all_targets)
        
        return metrics
    
    def train(self, num_epochs=12):
        """Complete training loop."""
        best_map = 0.0
        
        for epoch in range(num_epochs):
            # Train
            train_losses = self.train_epoch(epoch)
            
            # Validate
            if (epoch + 1) % self.config.val_interval == 0:
                metrics = self.validate()
                
                print(f'Epoch {epoch+1}:')
                print(f'  Train Loss: {train_losses["total_loss"]:.4f}')
                print(f'  Val mAP: {metrics["mAP"]:.4f}')
                
                # Save best model
                if metrics['mAP'] > best_map:
                    best_map = metrics['mAP']
                    torch.save(self.model.state_dict(), 'best_model.pth')
            
            # Update learning rate
            self.scheduler.step()
        
        print(f'Best mAP: {best_map:.4f}')

# Usage
model = FasterRCNN(num_classes=80)
trainer = FasterRCNNTrainer(model, train_loader, val_loader, config)
trainer.train(num_epochs=12)
```
