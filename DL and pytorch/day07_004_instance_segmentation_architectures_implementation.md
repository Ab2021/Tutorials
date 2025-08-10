# Day 7.4: Instance Segmentation Architectures - A Practical Guide

## Introduction: Separating Individuals

We have journeyed from classification (what), to object detection (what and where), to semantic segmentation (what, where, at the pixel level). Now we arrive at **Instance Segmentation**, which combines the strengths of the last two tasks.

*   **Semantic Segmentation** tells you all the pixels that belong to the class "person." It cannot, however, distinguish between *different* people. All person pixels get the same label.
*   **Instance Segmentation** solves this. It identifies every individual *instance* of an object. It tells you that these pixels belong to "person 1," those pixels belong to "person 2," and those other pixels belong to "person 3."

Instance segmentation provides the most detailed level of scene understanding and is crucial for applications like self-driving (tracking individual cars and pedestrians), robotics (manipulating specific objects), and photo editing.

This guide will explore the architecture of the most famous and influential instance segmentation model, **Mask R-CNN**, and show how to use a pre-trained version from `torchvision`.

**Today's Learning Objectives:**

1.  **Differentiate Semantic vs. Instance Segmentation:** Clearly understand the unique goal of instance segmentation.
2.  **Understand the Mask R-CNN Architecture:** See how it extends a two-stage object detector (Faster R-CNN) by adding a third branch for mask prediction.
3.  **Learn about RoIAlign:** Understand this key improvement over RoIPool that allows for more precise feature extraction for the mask head.
4.  **Use a Pre-trained Mask R-CNN:** Learn how to easily load and use a pre-trained Mask R-CNN model for inference and visualize its distinct outputs (boxes, labels, scores, and masks).

---

## Part 1: The Mask R-CNN Idea - An Extension of Faster R-CNN

Mask R-CNN is a beautifully simple and effective idea. It takes the powerful **Faster R-CNN** two-stage object detector and adds a third, parallel branch for predicting segmentation masks.

Let's review the Faster R-CNN pipeline and see where Mask R-CNN fits in:

1.  **Backbone & RPN (Same as Faster R-CNN):**
    *   A backbone CNN (e.g., ResNet) extracts features from the input image.
    *   A Region Proposal Network (RPN) uses these features to propose a set of candidate bounding boxes (Regions of Interest, or RoIs) that might contain objects.

2.  **RoIAlign (The Key Improvement):**
    *   For each proposed RoI, we need to extract its corresponding features from the feature map.
    *   Faster R-CNN used a method called **RoIPool**, which involved quantization (rounding to the nearest integer). This loss of precision was fine for bounding boxes but was too coarse for generating pixel-perfect masks.
    *   Mask R-CNN introduces **RoIAlign**, which uses bilinear interpolation to extract features at precise floating-point locations, preserving spatial accuracy.

3.  **The Three Parallel Heads (The Core of Mask R-CNN):**
    *   The features from RoIAlign are fed into three separate heads:
        *   **Classifier Head:** Predicts the class of the object (e.g., "person", "car").
        *   **Box Regression Head:** Refines the coordinates of the bounding box.
        *   **Mask Head (The New Part):** This is a small Fully Convolutional Network (FCN) that takes the RoI features and outputs a binary segmentation mask (e.g., 28x28) for that specific object instance. It "paints" the pixels inside the RoI that belong to the object.

![Mask R-CNN](https://i.imgur.com/e4jCj7A.png)

**The Takeaway:** Mask R-CNN is an intuitive extension of Faster R-CNN. It's a multi-task model that simultaneously performs classification, box regression, and mask prediction.

---

## Part 2: RoIAlign vs. RoIPool (Conceptual)

Understanding why RoIAlign is important is key to understanding Mask R-CNN.

*   **RoIPool (Imprecise):**
    1.  Takes a floating-point RoI (e.g., `x=10.5, y=20.2, w=15.8, h=15.8`).
    2.  **Rounds** it to the nearest integer coordinates on the feature map.
    3.  Divides this integer-sized region into a fixed number of grid cells (e.g., 7x7).
    4.  **Rounds** the boundaries of these grid cells.
    5.  Performs max pooling within each grid cell.
    *   **Problem:** The two rounding steps (quantizations) misalign the extracted features from the actual object location, which is detrimental for pixel-level masks.

*   **RoIAlign (Precise):**
    1.  Takes the floating-point RoI.
    2.  Divides the RoI into a fixed number of grid cells (e.g., 7x7).
    3.  For each grid cell, it defines a set of exact sampling points (e.g., 4 points).
    4.  It uses **bilinear interpolation** to compute the precise feature value at each sampling point, without any rounding.
    5.  It aggregates the results (e.g., with max or average pooling).
    *   **Benefit:** No quantization! The extracted features are precisely aligned with the original RoI, preserving the spatial information needed for accurate masks.

---

## Part 3: Using a Pre-trained Mask R-CNN Model

As with other complex vision models, the best way to use Mask R-CNN is to leverage a pre-trained version from `torchvision.models`.

Let's use a pre-trained Mask R-CNN to perform instance segmentation on a sample image.

```python
import torchvision
from torchvision import transforms
from PIL import Image, ImageDraw
import requests
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

print("--- Part 3: Using a Pre-trained Mask R-CNN ---")

# --- 1. Load the Pre-trained Model ---
# We load a Mask R-CNN model with a ResNet-50 backbone.
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval() # Set to evaluation mode

# --- 2. Load and Transform a Sample Image ---
url = 'https://raw.githubusercontent.com/pytorch/vision/temp-release/v0.5.0/gallery/assets/instances_demo.jpg'
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

transform = transforms.Compose([transforms.ToTensor()])
img_tensor = transform(image).to(device)
input_batch = [img_tensor]

# --- 3. Perform Inference ---
with torch.no_grad():
    predictions = model(input_batch)

pred = predictions[0]

# --- 4. Visualize the Results ---
# COCO class names are the same as in the object detection guide
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

def get_coloured_mask(mask):
    """random_color_masks
    Args:
        mask: a bool tensor of shape (H, W) or a 0-1 tensor of shape (H, W)
    Returns:
        a random coloured mask
    """
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = random.choice(colours)
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

def draw_instance_predictions(image, predictions, threshold=0.5):
    # Convert image to numpy array for drawing
    img_np = np.array(image)
    
    # Iterate over all predicted instances
    for i in range(len(predictions["scores"])):
        score = predictions["scores"][i]
        if score > threshold:
            # Get the mask for the current instance
            # The model outputs a soft mask (probabilities), so we threshold it to get a binary mask
            mask = predictions['masks'][i, 0].mul(255).byte().cpu().numpy()
            mask_bool = mask > (threshold * 255)
            
            # Get a random color for the mask
            coloured_mask = get_coloured_mask(mask_bool)
            
            # Blend the mask with the image
            img_np = np.where(np.expand_dims(mask_bool, axis=2), coloured_mask, img_np)

    # Convert back to PIL Image to draw boxes and text
    img_pil = Image.fromarray(img_np)
    draw = ImageDraw.Draw(img_pil)
    
    # Draw boxes and labels on top of the masks
    for i in range(len(predictions["scores"])):
        score = predictions["scores"][i]
        if score > threshold:
            box = predictions['boxes'][i].cpu().numpy()
            label_idx = predictions['labels'][i].item()
            label = f'{COCO_INSTANCE_CATEGORY_NAMES[label_idx]}: {score:.2f}'
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="white", width=2)
            draw.text((box[0], box[1]), label, fill="white")
            
    return img_pil

print("Drawing instance segmentation predictions on the image...")
result_image = draw_instance_predictions(image, pred, threshold=0.7)

# --- Plotting ---
plt.figure(figsize=(12, 8))
plt.imshow(result_image)
plt.axis('off')
plt.title('Mask R-CNN Instance Segmentation')
plt.show()
```

## Conclusion

Instance segmentation represents a significant leap towards comprehensive scene understanding. By combining the strengths of object detection and semantic segmentation, models like Mask R-CNN can identify, localize, and delineate every individual object in an image.

**Key Architectural Takeaways:**

1.  **Built on Object Detection:** Instance segmentation models are often extensions of successful two-stage object detectors. Mask R-CNN is a direct, elegant extension of Faster R-CNN.
2.  **Multi-Task Learning:** The model is trained simultaneously on three different tasks (classification, box regression, and mask prediction) using a combined loss function.
3.  **Precision is Key (RoIAlign):** For pixel-level tasks like mask prediction, preserving precise spatial alignment is critical. RoIAlign was a key innovation that enabled the success of Mask R-CNN by avoiding the quantization errors of RoIPool.
4.  **Practical Usage:** For nearly all applications, the best approach is to use a powerful, pre-trained model from a library like `torchvision`, which has already learned rich features from a massive dataset like COCO.

With instance segmentation, we can build systems that have a truly granular and human-like understanding of the visual world.

## Self-Assessment Questions

1.  **Instance vs. Semantic:** You are looking at an image of a crowd. What would a semantic segmentation model output? What would an instance segmentation model output?
2.  **Mask R-CNN Architecture:** What are the three main "heads" that branch off from the RoI features in a Mask R-CNN?
3.  **RoIAlign:** Why was RoIAlign a necessary improvement over RoIPool for instance segmentation?
4.  **Mask Head:** What is the typical architecture of the mask prediction head in Mask R-CNN?
5.  **Output Format:** When you use a pre-trained Mask R-CNN from `torchvision`, what are the four main keys in the prediction dictionary returned for each image?

