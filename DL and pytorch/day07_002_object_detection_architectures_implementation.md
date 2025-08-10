# Day 7.2: Object Detection Architectures - A Practical Guide

## Introduction: Beyond Classification - What and Where?

Image classification tells us *what* is in an image (e.g., "cat"). **Object Detection** takes this a step further. It aims to solve two problems simultaneously:

1.  **What** is in the image? (Classification)
2.  **Where** is it? (Localization)

The goal is to identify all objects of interest in an image and draw a **bounding box** around each one, along with a class label.

Object detection is a foundational task in computer vision, powering applications from self-driving cars to medical imaging analysis. Modern object detection models are typically divided into two main families: **two-stage detectors** and **one-stage detectors**.

This guide will provide a practical overview and simplified implementation of the core ideas behind these two families.

**Today's Learning Objectives:**

1.  **Understand the Object Detection Task:** Grasp the core output of a detector: bounding boxes and class labels.
2.  **Explore Two-Stage Detectors (R-CNN Family):** Understand the "propose-then-classify" pipeline of models like R-CNN and Faster R-CNN.
3.  **Explore One-Stage Detectors (YOLO, SSD):** Understand the concept of treating detection as a direct regression problem from grid cells.
4.  **Implement a Toy Detector:** Build a very simple object detection model to see the core components in action.
5.  **Use a Pre-trained Model from `torchvision`:** Learn how to easily load and use a pre-trained Faster R-CNN model for inference.

---

## Part 1: The Two-Stage Detector - The R-CNN Family

Two-stage detectors break the problem down into two sequential steps. The most famous family of these models is the R-CNN (Regions with CNN features) family.

**The Pipeline:**

1.  **Stage 1: Region Proposal:** An algorithm scans the image and proposes a few hundred or thousand potential regions (bounding boxes) that are likely to contain an object. This is done by a **Region Proposal Network (RPN)** in modern detectors like Faster R-CNN.

2.  **Stage 2: Classification and Refinement:** Each proposed region is warped into a fixed-size image and passed through a CNN backbone (like a ResNet). The resulting features are then used by two separate heads:
    *   A **classifier head** determines the class of the object in the region (e.g., "cat", "dog", or "background").
    *   A **regression head** refines the coordinates of the proposed bounding box to make it fit the object more tightly.

**Analogy:** It's like a meticulous detective. First, they identify all possible areas of interest (proposals). Then, they examine each area closely to identify the suspect (classify) and get a precise location (refine box).

**Pros:** Generally higher accuracy, especially for small objects.
**Cons:** Slower, as it involves two separate stages and processing each region individually.

![Faster R-CNN](https://i.imgur.com/LgY4E1S.png)

---

## Part 2: The One-Stage Detector - YOLO and SSD

One-stage detectors aim for speed and simplicity by combining these two stages into a single, unified network.

**The Pipeline (YOLO - You Only Look Once):**

1.  **Grid System:** The input image is divided into a coarse grid (e.g., 7x7 or 13x13).

2.  **Direct Prediction:** The image is passed through a single CNN. For **each cell** in the grid, the network directly predicts a fixed number of bounding boxes and their corresponding class probabilities.

3.  **Confidence and Class Scores:** Each predicted box has:
    *   **Box coordinates:** `(x, y, w, h)` relative to the grid cell.
    *   An **"objectness" score:** The confidence that this box actually contains an object.
    *   **Class probabilities:** The probability of the object belonging to each class.

4.  **Non-Max Suppression (NMS):** The model produces thousands of potential boxes. NMS is a crucial post-processing step that filters through these boxes, discarding those with low confidence and merging overlapping boxes for the same object to produce the final set of detections.

**Analogy:** It's like a fast-acting security guard who looks at a security monitor once and immediately points out all the suspects and their locations in a single glance.

**Pros:** Extremely fast, often capable of real-time detection.
**Cons:** Can sometimes struggle with accuracy for very small objects compared to two-stage detectors.

![YOLO](https://i.imgur.com/l4a4y7K.png)

---

## Part 3: A Toy Object Detection Implementation

Building a full-fledged detector is complex. Instead, let's build a toy model that captures the essence of the task. We will create a model that takes an image and outputs a single bounding box and a class label.

### 3.1. The Task and Data

*   **Input:** A 32x32 image.
*   **Output:**
    *   A class prediction (e.g., for 10 classes).
    *   A bounding box prediction `(x, y, w, h)`.

```python
import torch
import torch.nn as nn
import torch.optim as optim

print("--- Part 3: A Toy Object Detector ---")

# --- 1. The Toy Model ---
class ToyDetector(nn.Module):
    def __init__(self, num_classes=10):
        super(ToyDetector, self).__init__()
        # A simple CNN backbone to extract features
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 32x32 -> 16x16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 16x16 -> 8x8
            nn.Flatten(),
        )
        
        # The two heads for the two tasks
        # Classifier head
        self.classifier_head = nn.Linear(32 * 8 * 8, num_classes)
        # Bounding box regression head
        self.bbox_head = nn.Linear(32 * 8 * 8, 4) # 4 values for (x, y, w, h)

    def forward(self, x):
        features = self.backbone(x)
        class_logits = self.classifier_head(features)
        bbox_predictions = self.bbox_head(features)
        return class_logits, bbox_predictions

# --- 2. The Combined Loss Function ---
# The total loss is a weighted sum of the classification loss and the localization loss.
def combined_loss(class_logits, bbox_preds, class_targets, bbox_targets):
    # Classification loss (standard for classification)
    class_loss_fn = nn.CrossEntropyLoss()
    class_loss = class_loss_fn(class_logits, class_targets)
    
    # Localization loss (L1 or MSE is common for bounding boxes)
    bbox_loss_fn = nn.L1Loss()
    bbox_loss = bbox_loss_fn(bbox_preds, bbox_targets)
    
    # Combine the losses
    # The weight (alpha) is a hyperparameter to balance the two tasks.
    alpha = 1.0
    total_loss = class_loss + alpha * bbox_loss
    
    return total_loss

# --- 3. A Dummy Training Step ---
model = ToyDetector()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create dummy data for one batch
images = torch.randn(4, 3, 32, 32).to(device) # Batch of 4 images
class_targets = torch.randint(0, 10, (4,)).to(device) # True class labels
bbox_targets = torch.rand(4, 4).to(device) # True bounding boxes

# Forward pass
class_logits, bbox_preds = model(images)

# Calculate loss
loss = combined_loss(class_logits, bbox_preds, class_targets, bbox_targets)

# Backward pass and optimization
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Toy model training step completed successfully.")
print(f"  - Calculated Loss: {loss.item():.4f}")
print(f"  - Output class logits shape: {class_logits.shape}")
print(f"  - Output bbox predictions shape: {bbox_preds.shape}")
```

This toy example demonstrates the core idea of **multi-task learning**, where a single network is trained to perform multiple tasks simultaneously by using a combined loss function.

---

## Part 4: Using a Pre-trained Detector from `torchvision`

Training object detectors from scratch is very challenging. The most practical way to use them is to load a pre-trained model from `torchvision.models`.

Let's load a **Faster R-CNN** model pre-trained on the COCO dataset and use it for inference on a sample image.

```python
import torchvision
from torchvision import transforms
from PIL import Image, ImageDraw
import requests

print("\n--- Part 4: Using a Pre-trained Faster R-CNN ---")

# --- 1. Load the Pre-trained Model ---
# We load a Faster R-CNN model with a ResNet-50 backbone.
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
model.to(device)
model.eval() # Set the model to evaluation mode

# --- 2. Load and Transform a Sample Image ---
url = 'https://i.ytimg.com/vi/1Vy5hR_c43w/maxresdefault.jpg'
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# The transform is simple: just convert to a tensor.
# The model handles normalization internally.
transform = transforms.Compose([transforms.ToTensor()])
img_tensor = transform(image).to(device)

# The model expects a batch of images, so we add a batch dimension.
input_batch = [img_tensor]

# --- 3. Perform Inference ---
with torch.no_grad():
    predictions = model(input_batch)

# The prediction is a list of dictionaries, one for each image in the batch.
# Each dictionary contains 'boxes', 'labels', and 'scores'.
pred = predictions[0]

# --- 4. Visualize the Results ---
# COCO class names (the model was trained on this dataset)
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def draw_predictions(image, predictions, threshold=0.7):
    draw = ImageDraw.Draw(image)
    for box, label_idx, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        if score > threshold:
            box = box.cpu().numpy()
            label = f'{COCO_INSTANCE_CATEGORY_NAMES[label_idx]}: {score:.2f}'
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
            draw.text((box[0], box[1]), label, fill="red")
    return image

print("Drawing predictions on the image...")
result_image = draw_predictions(image, pred)

# To display in a notebook, you would just have `result_image` as the last line.
# To save, you would use `result_image.save('result.jpg')`
# For this script, we'll just show a confirmation.
print("Finished processing. A result image with bounding boxes can now be viewed.")
# In a real script, you would use plt.imshow(result_image) or result_image.show()
```

## Conclusion

Object detection is a fascinating and challenging field that combines classification and localization. We've seen the two dominant architectural paradigms:

*   **Two-Stage Detectors (e.g., Faster R-CNN):** Accurate but slower. They first propose regions of interest and then classify and refine them.
*   **One-Stage Detectors (e.g., YOLO, SSD):** Faster but can be less accurate. They treat detection as a direct regression problem from a grid.

The core idea underpinning these models is **multi-task learning**, where a single network is trained with a combined loss function to perform several tasks at once.

For practical applications, leveraging a powerful, pre-trained model from a library like `torchvision` is almost always the best approach, allowing you to achieve state-of-the-art detection capabilities with just a few lines of code.

## Self-Assessment Questions

1.  **The Task:** What are the two main outputs of an object detection model for each object it finds?
2.  **Two-Stage vs. One-Stage:** What is the fundamental difference between a two-stage detector and a one-stage detector?
3.  **RPN:** In a model like Faster R-CNN, what is the specific job of the Region Proposal Network (RPN)?
4.  **Multi-Task Loss:** In our toy detector, we combined a classification loss and a bounding box regression loss. Why is it necessary to do this?
5.  **NMS:** What is the purpose of Non-Max Suppression (NMS) in a one-stage detector like YOLO?

