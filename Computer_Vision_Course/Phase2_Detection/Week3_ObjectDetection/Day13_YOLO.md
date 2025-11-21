# Day 13: YOLO and Single-Shot Detectors

## Overview
YOLO (You Only Look Once) revolutionized real-time object detection by framing detection as a single regression problem. This lesson covers YOLO v1-v8, SSD, and RetinaNet.

## 1. YOLO v1 (2016)

**Key Idea:** Divide image into grid, predict bounding boxes and class probabilities directly.

**Architecture:**
```
Image (448×448) → CNN (24 conv + 2 FC) → Tensor (S×S×(B*5+C))
```

where:
- S = 7 (grid size)
- B = 2 (boxes per cell)
- C = 20 (classes for PASCAL VOC)

**Prediction:**
- Each cell predicts B bounding boxes
- Each box: (x, y, w, h, confidence)
- Each cell predicts C class probabilities

**Output tensor:** 7×7×30
- 2 boxes × 5 values = 10
- 20 class probabilities = 20
- Total = 30

### Implementation

```python
import torch
import torch.nn as nn

class YOLOv1(nn.Module):
    """YOLO v1 implementation."""
    
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.S = S  # Grid size
        self.B = B  # Boxes per cell
        self.C = C  # Number of classes
        
        # Backbone (simplified, original uses custom architecture)
        self.features = nn.Sequential(
            # Conv layers
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(192, 128, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            # More conv layers...
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, S * S * (B * 5 + C)),
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, 448, 448)
        
        Returns:
            predictions: (B, S, S, B*5+C)
        """
        x = self.features(x)
        x = self.fc(x)
        
        # Reshape to (B, S, S, B*5+C)
        x = x.view(-1, self.S, self.S, self.B * 5 + self.C)
        
        return x

### YOLO Loss Function

class YOLOv1Loss(nn.Module):
    """YOLO v1 loss function."""
    
    def __init__(self, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord  # Weight for coordinate loss
        self.lambda_noobj = lambda_noobj  # Weight for no-object loss
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (N, S, S, B*5+C)
            targets: (N, S, S, 5+C) [x, y, w, h, conf, class_probs...]
        
        Returns:
            loss: Scalar loss
        """
        N = predictions.size(0)
        
        # Split predictions
        # Box 1: [:, :, :, 0:5]
        # Box 2: [:, :, :, 5:10]
        # Class probs: [:, :, :, 10:]
        
        pred_boxes1 = predictions[:, :, :, 0:5]
        pred_boxes2 = predictions[:, :, :, 5:10]
        pred_class = predictions[:, :, :, 10:]
        
        # Target
        target_boxes = targets[:, :, :, 0:5]
        target_class = targets[:, :, :, 5:]
        
        # Object mask (cells with objects)
        obj_mask = target_boxes[:, :, :, 4] > 0  # Confidence > 0
        noobj_mask = target_boxes[:, :, :, 4] == 0
        
        # Choose responsible box (highest IoU with ground truth)
        iou1 = compute_iou(pred_boxes1, target_boxes)
        iou2 = compute_iou(pred_boxes2, target_boxes)
        
        responsible_mask = (iou1 > iou2).float()
        
        # Coordinate loss (only for responsible boxes in cells with objects)
        coord_mask = obj_mask.unsqueeze(-1).expand_as(target_boxes)
        
        pred_boxes_responsible = (responsible_mask.unsqueeze(-1) * pred_boxes1 +
                                 (1 - responsible_mask.unsqueeze(-1)) * pred_boxes2)
        
        # x, y loss
        xy_loss = F.mse_loss(
            pred_boxes_responsible[:, :, :, 0:2][coord_mask[:, :, :, 0:2]],
            target_boxes[:, :, :, 0:2][coord_mask[:, :, :, 0:2]],
            reduction='sum'
        )
        
        # w, h loss (square root)
        wh_pred = torch.sqrt(pred_boxes_responsible[:, :, :, 2:4] + 1e-6)
        wh_target = torch.sqrt(target_boxes[:, :, :, 2:4] + 1e-6)
        wh_loss = F.mse_loss(
            wh_pred[coord_mask[:, :, :, 2:4]],
            wh_target[coord_mask[:, :, :, 2:4]],
            reduction='sum'
        )
        
        # Confidence loss (object)
        conf_obj_loss = F.mse_loss(
            pred_boxes_responsible[:, :, :, 4][obj_mask],
            target_boxes[:, :, :, 4][obj_mask],
            reduction='sum'
        )
        
        # Confidence loss (no object)
        conf_noobj_loss = F.mse_loss(
            pred_boxes1[:, :, :, 4][noobj_mask],
            target_boxes[:, :, :, 4][noobj_mask],
            reduction='sum'
        ) + F.mse_loss(
            pred_boxes2[:, :, :, 4][noobj_mask],
            target_boxes[:, :, :, 4][noobj_mask],
            reduction='sum'
        )
        
        # Class loss
        class_loss = F.mse_loss(
            pred_class[obj_mask],
            target_class[obj_mask],
            reduction='sum'
        )
        
        # Total loss
        total_loss = (
            self.lambda_coord * (xy_loss + wh_loss) +
            conf_obj_loss +
            self.lambda_noobj * conf_noobj_loss +
            class_loss
        ) / N
        
        return total_loss
```

**Performance:**
- mAP: 63.4% on PASCAL VOC 2007
- Speed: 45 FPS
- **Trade-off:** Fast but less accurate than Faster R-CNN

**Limitations:**
1. Struggles with small objects
2. Struggles with objects in groups
3. Coarse features (7×7 grid)
4. Each cell predicts only one class

## 2. YOLO v2 / YOLO9000 (2017)

**Improvements:**
1. **Batch Normalization:** After every conv layer
2. **High Resolution:** Train at 448×448
3. **Anchor Boxes:** Like Faster R-CNN
4. **Dimension Clusters:** K-means on training boxes
5. **Direct Location Prediction:** Constrain box centers
6. **Fine-Grained Features:** Passthrough layer
7. **Multi-Scale Training:** Random input sizes

**Architecture:**
- Darknet-19 backbone (19 conv + 5 maxpool)
- 13×13 output grid
- 5 anchor boxes per cell

```python
class YOLOv2(nn.Module):
    """YOLO v2 with anchor boxes."""
    
    def __init__(self, num_anchors=5, num_classes=80):
        super().__init__()
        
        # Darknet-19 backbone
        self.backbone = Darknet19()
        
        # Detection layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )
        
        # Output layer
        # Each anchor: 5 (x, y, w, h, conf) + num_classes
        self.output = nn.Conv2d(
            1024,
            num_anchors * (5 + num_classes),
            kernel_size=1
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, 416, 416)
        
        Returns:
            predictions: (B, num_anchors, 13, 13, 5+num_classes)
        """
        features = self.backbone(x)
        x = self.conv_layers(features)
        x = self.output(x)
        
        # Reshape
        B = x.size(0)
        x = x.view(B, self.num_anchors, 5 + self.num_classes, 13, 13)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        
        return x

def decode_yolo_output(predictions, anchors, stride=32):
    """
    Decode YOLO predictions to boxes.
    
    Args:
        predictions: (B, num_anchors, H, W, 5+C)
        anchors: (num_anchors, 2) anchor sizes
        stride: Feature map stride
    
    Returns:
        boxes: (B, num_anchors*H*W, 4)
        conf: (B, num_anchors*H*W)
        class_probs: (B, num_anchors*H*W, C)
    """
    B, num_anchors, H, W, _ = predictions.shape
    
    # Grid offsets
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W))
    grid_x = grid_x.view(1, 1, H, W).float()
    grid_y = grid_y.view(1, 1, H, W).float()
    
    # Anchors
    anchor_w = anchors[:, 0].view(1, num_anchors, 1, 1)
    anchor_h = anchors[:, 1].view(1, num_anchors, 1, 1)
    
    # Decode
    pred_x = (torch.sigmoid(predictions[..., 0]) + grid_x) * stride
    pred_y = (torch.sigmoid(predictions[..., 1]) + grid_y) * stride
    pred_w = torch.exp(predictions[..., 2]) * anchor_w * stride
    pred_h = torch.exp(predictions[..., 3]) * anchor_h * stride
    
    pred_conf = torch.sigmoid(predictions[..., 4])
    pred_class = torch.softmax(predictions[..., 5:], dim=-1)
    
    # Convert to corner format
    boxes = torch.stack([
        pred_x - pred_w / 2,
        pred_y - pred_h / 2,
        pred_x + pred_w / 2,
        pred_y + pred_h / 2,
    ], dim=-1)
    
    # Reshape
    boxes = boxes.view(B, -1, 4)
    pred_conf = pred_conf.view(B, -1)
    pred_class = pred_class.view(B, -1, pred_class.size(-1))
    
    return boxes, pred_conf, pred_class
```

**Performance:**
- mAP: 76.8% on PASCAL VOC 2007
- Speed: 67 FPS
- **Improvement:** +13.4% mAP over v1

## 3. YOLO v3 (2018)

**Key Improvements:**
1. **Multi-scale predictions:** 3 scales (like FPN)
2. **Darknet-53 backbone:** Residual connections
3. **Binary cross-entropy:** Multi-label classification
4. **More anchors:** 9 anchors (3 per scale)

**Architecture:**
```
Input (416×416)
    ↓ Darknet-53
    ├→ Scale 1 (13×13): Large objects
    ├→ Scale 2 (26×26): Medium objects
    └→ Scale 3 (52×52): Small objects
```

```python
class YOLOv3(nn.Module):
    """YOLO v3 with multi-scale predictions."""
    
    def __init__(self, num_classes=80):
        super().__init__()
        
        # Darknet-53 backbone
        self.backbone = Darknet53()
        
        # Detection heads for 3 scales
        self.head_large = YOLOHead(1024, num_classes, num_anchors=3)   # 13×13
        self.head_medium = YOLOHead(512, num_classes, num_anchors=3)   # 26×26
        self.head_small = YOLOHead(256, num_classes, num_anchors=3)    # 52×52
        
        # Upsample layers
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, 416, 416)
        
        Returns:
            predictions: List of 3 tensors for different scales
        """
        # Backbone
        c3, c4, c5 = self.backbone(x)  # 52×52, 26×26, 13×13
        
        # Large objects (13×13)
        pred_large = self.head_large(c5)
        
        # Medium objects (26×26)
        x = self.upsample1(c5)
        x = torch.cat([x, c4], dim=1)
        pred_medium = self.head_medium(x)
        
        # Small objects (52×52)
        x = self.upsample2(x)
        x = torch.cat([x, c3], dim=1)
        pred_small = self.head_small(x)
        
        return [pred_large, pred_medium, pred_small]

class YOLOHead(nn.Module):
    """YOLO detection head."""
    
    def __init__(self, in_channels, num_classes, num_anchors=3):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(in_channels, num_anchors * (5 + num_classes), kernel_size=1),
        )
    
    def forward(self, x):
        return self.conv(x)
```

**Performance:**
- mAP: 57.9% on COCO (mAP@0.5:0.95)
- Speed: 30 FPS
- **Best for small objects** among YOLO versions

## 4. Modern YOLO (v5-v8)

**YOLOv5 (2020):**
- PyTorch implementation
- CSPDarknet backbone
- PANet neck
- Auto-anchor, auto-learning bounding box anchors
- Mosaic augmentation

**YOLOv8 (2023):**
- Anchor-free
- Decoupled head
- Task-aligned assigner
- State-of-the-art performance

```python
# YOLOv8 usage (ultralytics)
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')  # nano
# model = YOLO('yolov8s.pt')  # small
# model = YOLO('yolov8m.pt')  # medium
# model = YOLO('yolov8l.pt')  # large
# model = YOLO('yolov8x.pt')  # extra large

# Train
model.train(data='coco.yaml', epochs=100, imgsz=640)

# Inference
results = model('image.jpg')

# Process results
for result in results:
    boxes = result.boxes  # Boxes object
    for box in boxes:
        print(f"Class: {box.cls}, Conf: {box.conf}, Box: {box.xyxy}")
```

## Summary
YOLO evolution: v1 (grid-based), v2 (anchors), v3 (multi-scale), v5-v8 (modern techniques). Single-shot detectors trade some accuracy for real-time speed.

**Next:** Advanced detection techniques (RetinaNet, EfficientDet, DETR).
