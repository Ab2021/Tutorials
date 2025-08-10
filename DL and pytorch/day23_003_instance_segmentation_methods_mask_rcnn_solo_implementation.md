# Day 23.3: Instance Segmentation Methods (Mask R-CNN) - A Practical Guide

## Introduction: Distinguishing Individuals

Semantic segmentation is powerful, but it has a key limitation: it cannot distinguish between different instances of the same object class. For example, in a crowded street scene, it would label all people with a single "person" class. **Instance Segmentation** solves this by identifying, classifying, and generating a pixel-perfect mask for **each individual object instance**.

This is arguably the most challenging of the core computer vision tasks, as it combines the goals of object detection (localizing individual objects with boxes) and semantic segmentation (classifying pixels).

The landmark architecture for this task is **Mask R-CNN**. It elegantly extends the two-stage Faster R-CNN object detector to simultaneously predict segmentation masks.

This guide provides a practical deep dive into the Mask R-CNN architecture, showing how its components work together to achieve state-of-the-art instance segmentation.

**Today's Learning Objectives:**

1.  **Solidify the Definition of Instance Segmentation:** Clearly differentiate it from object detection and semantic segmentation.
2.  **Understand the Mask R-CNN Architecture:** See how it builds upon Faster R-CNN by adding a parallel mask prediction branch.
3.  **Learn the Role of RoIAlign:** Revisit this crucial layer and understand why its precision is essential for generating accurate masks.
4.  **Explore the Mask Head:** See how a small Fully Convolutional Network (FCN) is used to predict a binary mask for each detected object.
5.  **Use a Pre-trained Mask R-CNN Model:** Apply a state-of-the-art Mask R-CNN from `torchvision` to perform instance segmentation on a real image.

---

## Part 1: The Mask R-CNN Framework

**The Core Idea:** Mask R-CNN is a simple and intuitive extension of the Faster R-CNN object detector.

Recall the Faster R-CNN pipeline:
1.  A CNN backbone extracts features.
2.  A Region Proposal Network (RPN) proposes candidate object boxes (RoIs).
3.  For each RoI, features are extracted (using RoIPool/RoIAlign).
4.  Two heads operate on these features: one for **classification** and one for **bounding box regression**.

Mask R-CNN simply adds a **third, parallel head** to this framework.

*   **The Mask Head:** This branch is a small Fully Convolutional Network (FCN). It takes the features for a given RoI and outputs a small binary mask (e.g., 28x28) for that RoI. The mask is binary because it only has to solve the question, "Which pixels within this box belong to the object?" The classification of *what* the object is has already been handled by the classification head.

**The Multi-Task Loss:**
The model is trained end-to-end with a combined loss function:

`L = L_cls + L_box + L_mask`

*   `L_cls`: The classification loss (e.g., Cross-Entropy).
*   `L_box`: The bounding box regression loss (e.g., Smooth L1).
*   `L_mask`: The mask loss. This is typically a binary cross-entropy loss averaged over all pixels in the mask.

![Mask R-CNN Architecture](https://i.imgur.com/e4jCj7A.png)

---

## Part 2: RoIAlign - The Key to Precision

As we discussed in the original instance segmentation guide (Day 7.4), the key to Mask R-CNN's success was the introduction of **RoIAlign**.

*   **The Problem with RoIPool:** The RoIPool layer used in Fast R-CNN involves rounding floating-point coordinates to integers. This **quantization** causes a misalignment between the RoI and the extracted features.
*   **Why it Matters:** This misalignment is a minor issue for bounding box prediction, but it is disastrous for predicting pixel-perfect masks. The features are not precisely aligned with the object, leading to inaccurate mask boundaries.
*   **The RoIAlign Solution:** RoIAlign avoids any quantization. It uses **bilinear interpolation** to compute the exact feature values at specific sampling points within the RoI. This preserves the precise spatial location of the features, which is critical for the mask head to work effectively.

---

## Part 3: The Mask Head in Detail

The mask head is a small FCN applied to each RoI.

**The Process:**
1.  The features for an RoI are extracted via RoIAlign, resulting in a fixed-size feature map (e.g., `(C, 14, 14)`).
2.  This feature map is passed through a stack of several convolutional layers. This allows the model to make a more refined, pixel-level prediction.
3.  A final **transposed convolution** is often used to up-sample the feature map to a slightly larger, but still low-resolution, output mask (e.g., `(Num_Classes, 28, 28)`).
4.  During training, the ground-truth mask is down-scaled to 28x28 to compute the loss. During inference, the predicted 28x28 mask is up-scaled to the original RoI size.

**Decoupling Mask and Class Prediction:**
Importantly, the mask head predicts a mask for *every* class (output shape is `Num_Classes, H, W`). The final mask is chosen based on the class predicted by the separate classification head. This decouples the tasks of classification and segmentation, which the authors found to be critical for good performance.

---

## Part 4: Using a Pre-trained Mask R-CNN in PyTorch

Let's use the pre-trained Mask R-CNN from `torchvision` to see it in action. The process is nearly identical to using the Faster R-CNN model, but the output dictionary will now also contain the predicted masks.

```python
import torchvision
from torchvision import transforms
from PIL import Image, ImageDraw
import requests
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

print("--- Part 4: Using a Pre-trained Mask R-CNN ---")

# --- 1. Load the Pre-trained Model ---
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

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
# (We reuse the visualization functions from Day 7.4)
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
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255]]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = random.choice(colours)
    return np.stack([r, g, b], axis=2)

def draw_instance_predictions(image, predictions, threshold=0.7):
    img_np = np.array(image)
    
    for i in range(len(predictions["scores"])):
        score = predictions["scores"][i]
        if score > threshold:
            # --- The key difference: We now have masks! ---
            mask = predictions['masks'][i, 0].mul(255).byte().cpu().numpy() > (threshold * 255)
            coloured_mask = get_coloured_mask(mask)
            img_np = np.where(np.expand_dims(mask, axis=2), coloured_mask, img_np)

    img_pil = Image.fromarray(img_np)
    draw = ImageDraw.Draw(img_pil)
    for i in range(len(predictions["scores"])):
        score = predictions["scores"][i]
        if score > threshold:
            box = predictions['boxes'][i].cpu().numpy()
            label = f'{COCO_INSTANCE_CATEGORY_NAMES[predictions['labels'][i].item()]}: {score:.2f}'
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="white", width=2)
            draw.text((box[0], box[1]), label, fill="white")
            
    return img_pil

print("Drawing instance segmentation predictions on the image...")
result_image = draw_instance_predictions(image, pred, threshold=0.7)

plt.figure(figsize=(14, 10))
plt.imshow(result_image)
plt.axis('off')
plt.title('Mask R-CNN Instance Segmentation')
plt.show()
```

## Conclusion

Mask R-CNN provides an elegant and effective framework for instance segmentation. By extending a powerful two-stage object detector with a simple, parallel mask prediction head, it can achieve a detailed, instance-level understanding of a scene.

**Key Takeaways:**

1.  **Instance Segmentation = Detection + Semantic Segmentation:** It solves both problems at once, identifying individual objects and providing a pixel-level mask for each.
2.  **Mask R-CNN Extends Faster R-CNN:** The architecture is intuitive: take a proven object detector and add a third branch for mask prediction.
3.  **RoIAlign is Essential for Precision:** The move from the quantized RoIPool to the interpolation-based RoIAlign was a critical step that enabled the prediction of accurate, pixel-perfect masks.
4.  **Multi-Task Learning:** The model is trained with a combined loss function that simultaneously optimizes for classification, box regression, and mask prediction.

This powerful technique is a cornerstone of modern computer vision, enabling applications that require a fine-grained understanding of individual objects in a complex scene.

## Self-Assessment Questions

1.  **Instance vs. Semantic:** If you run an instance segmentation model on a photo of a flock of birds, what would you expect the output to look like?
2.  **Architectural Addition:** What specific component does Mask R-CNN add to the Faster R-CNN architecture?
3.  **RoIAlign:** Why is RoIAlign preferred over RoIPool for instance segmentation?
4.  **Mask Head:** What is the input to the mask prediction head? What does it output?
5.  **Decoupling:** Why does the Mask R-CNN predict a mask for every possible class and then select the correct one, rather than just predicting a single mask for the predicted class?

