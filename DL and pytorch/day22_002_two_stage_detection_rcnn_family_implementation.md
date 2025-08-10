# Day 22.2: Two-Stage Detection (R-CNN Family) - A Practical Guide

## Introduction: Propose, then Classify

One of the most influential and historically important families of object detectors is the **R-CNN family**. These are **two-stage detectors**. They break the complex problem of object detection into two more manageable stages:

1.  **Stage 1: Propose Regions.** Scan the image to find a set of candidate bounding boxes that are likely to contain an object. These are called **Region Proposals** or Regions of Interest (RoIs).
2.  **Stage 2: Classify and Refine.** For each proposed region, classify the object within it and refine the bounding box coordinates to be more precise.

This "propose, then classify" strategy, while often slower than one-stage methods, can lead to very high accuracy. This guide will walk through the evolution of this family, from the original R-CNN to the modern and powerful Faster R-CNN.

**Today's Learning Objectives:**

1.  **Understand the R-CNN Pipeline:** Learn the original, multi-step pipeline involving external algorithms like Selective Search.
2.  **Learn about Fast R-CNN:** See the key innovation of processing the entire image with a CNN only once.
3.  **Grasp the Faster R-CNN Architecture:** Understand the revolutionary idea of the **Region Proposal Network (RPN)**, which made the entire pipeline a single, end-to-end trainable neural network.
4.  **Explore the Role of Anchor Boxes:** Understand how the RPN uses pre-defined anchor boxes to predict object locations and sizes.
5.  **Use a Pre-trained Faster R-CNN Model:** Apply a state-of-the-art Faster R-CNN from `torchvision` to perform object detection.

---

## Part 1: The Evolution of the R-CNN Family

### 1.1. R-CNN (Regions with CNN Features)

*   **The Idea (2014):** The original R-CNN was a groundbreaking but complex, multi-stage pipeline.
*   **The Process:**
    1.  **Region Proposals:** Use an external, traditional computer vision algorithm called **Selective Search** to generate ~2000 candidate region proposals.
    2.  **Warp Regions:** For *each* of these 2000 proposals, warp the image patch inside it to a fixed size (e.g., 227x227).
    3.  **CNN Feature Extraction:** Pass *each* of the 2000 warped image patches through a pre-trained CNN (like AlexNet) to extract features.
    4.  **Classification:** Train a separate linear SVM classifier for each class to classify the extracted feature vectors.
    5.  **Box Regression:** Train a separate linear regression model to refine the bounding box coordinates.
*   **The Problem:** It was incredibly **slow**. Running a powerful CNN on 2000 image patches for every single image took a huge amount of time (e.g., ~47 seconds per image).

### 1.2. Fast R-CNN

*   **The Idea (2015):** The main bottleneck in R-CNN was running the CNN on thousands of overlapping regions. Fast R-CNN solved this with a simple, powerful insight.
*   **The Key Innovation:**
    1.  **One CNN Pass:** Feed the **entire image** through the CNN backbone *once* to get a single, large feature map.
    2.  **Project RoIs:** Project the region proposals from Selective Search onto this feature map.
    3.  **RoIPool:** For each projected region, use a special pooling layer called **RoIPool** to extract a fixed-size feature vector (e.g., 7x7) from the corresponding area of the feature map.
    4.  **Unified Head:** Feed this fixed-size feature vector into a unified head with two sibling branches: a softmax classifier and a bounding box regressor.
*   **The Result:** A massive speedup (e.g., ~2 seconds per image) because the expensive convolution is only performed once per image.
*   **The Remaining Bottleneck:** The region proposal step (Selective Search) was still a separate, slow algorithm.

---

## Part 2: Faster R-CNN - The Modern Two-Stage Detector

*   **The Idea (2015):** The final breakthrough was to make the region proposal step part of the neural network itself. This created a single, unified, end-to-end trainable object detector.
*   **The Key Innovation: The Region Proposal Network (RPN)**
    1.  The RPN is a small, fully convolutional network that slides over the feature map produced by the CNN backbone.
    2.  At each sliding window location, it simultaneously predicts:
        *   An **objectness score**: The probability that an object is present at that location.
        *   **Box coordinate refinements**.
    3.  Crucially, it makes these predictions relative to a set of pre-defined **anchor boxes**.

### 2.1. Anchor Boxes

*   **What they are:** A set of pre-defined bounding boxes with different sizes and aspect ratios (e.g., a tall box, a wide box, a large square box).
*   **How they work:** At each position on the feature map, the RPN uses a set of `k` anchor boxes (e.g., 9 anchors: 3 scales x 3 aspect ratios). It doesn't predict a box from scratch; instead, it predicts the **probability** that each anchor contains an object and the **offset** (delta x, delta y, delta w, delta h) needed to transform that anchor into a tight-fitting proposal.
*   **Why they work:** This turns the difficult problem of predicting arbitrary box coordinates into two simpler problems: classifying a fixed set of anchors and regressing small refinement values. This makes the training much more stable and effective.

### 2.2. The Complete Faster R-CNN Architecture

1.  An input image is passed through a **CNN backbone** (e.g., ResNet) to get a feature map.
2.  The **Region Proposal Network (RPN)** takes this feature map and outputs a set of object proposals (RoIs) with objectness scores.
3.  A **RoIAlign** layer (an improvement on RoIPool) extracts fixed-size feature vectors for each proposal.
4.  These feature vectors are fed into the **final classification and box regression heads** to produce the final output.

The entire network is trained jointly with a multi-task loss function.

![Faster R-CNN Architecture](https://i.imgur.com/LgY4E1S.png)

---

## Part 3: Using a Pre-trained Faster R-CNN in PyTorch

`torchvision.models` provides an easy-to-use, pre-trained Faster R-CNN model. This is the most practical way to apply this powerful detector.

```python
import torchvision
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import requests

print("--- Part 3: Using a Pre-trained Faster R-CNN ---")

# --- 1. Load the Pre-trained Model ---
# We load a Faster R-CNN model with a MobileNetV3-Large backbone for efficiency.
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval() # Set to evaluation mode

# --- 2. Load and Transform a Sample Image ---
url = 'https://www.autocar.co.uk/sites/autocar.co.uk/files/styles/gallery_full/public/volkswagen_id_buzz_01.jpg'
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

transform = transforms.Compose([transforms.ToTensor()])
img_tensor = transform(image).to(device)
input_batch = [img_tensor]

# --- 3. Perform Inference ---
with torch.no_grad():
    predictions = model(input_batch)

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

def draw_predictions(image, predictions, threshold=0.5):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for box, label_idx, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        if score > threshold:
            box = box.cpu().numpy()
            label = f'{COCO_INSTANCE_CATEGORY_NAMES[label_idx]}: {score:.2f}'
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="lime", width=3)
            draw.text((box[0], box[1]), label, fill="white", font=font)
    return image

print("Drawing predictions on the image...")
result_image = draw_predictions(image, pred, threshold=0.7)

# To display in a notebook, you would just have `result_image` as the last line.
# For this script, we'll just show a confirmation.
print("Finished processing. A result image with bounding boxes can now be viewed.")
# In a real script, you would use plt.imshow(result_image) or result_image.show()
```

## Conclusion

The R-CNN family illustrates a brilliant evolution in object detection. By moving from a slow, multi-stage pipeline to a single, unified, end-to-end neural network, Faster R-CNN set the standard for two-stage detectors.

**Key Takeaways:**

1.  **Two-Stage Approach:** These models first generate a sparse set of high-quality region proposals and then classify and refine these proposals.
2.  **Sharing Convolutions is Key:** The major speedup from R-CNN to Fast R-CNN came from running the expensive CNN backbone only once per image.
3.  **The RPN Revolution:** The Region Proposal Network was the final piece of the puzzle, replacing slow external algorithms like Selective Search with a fast, learnable neural network and making the entire system end-to-end.
4.  **Anchors Simplify the Problem:** Using a pre-defined set of anchor boxes makes the localization task easier and more stable for the RPN to learn.

While often slower than their one-stage counterparts, two-stage detectors like Faster R-CNN are still widely used and often achieve state-of-the-art accuracy, making them a crucial tool in the computer vision toolkit.

## Self-Assessment Questions

1.  **R-CNN Bottleneck:** What was the primary performance bottleneck of the original R-CNN model?
2.  **Fast R-CNN Innovation:** What was the key insight that made Fast R-CNN much faster than R-CNN?
3.  **RPN:** What is the main purpose of the Region Proposal Network (RPN) in Faster R-CNN?
4.  **Anchor Boxes:** In your own words, what is an anchor box and how does the RPN use it?
5.  **Two-Stage vs. One-Stage:** What is the main advantage of a two-stage detector like Faster R-CNN? What is its main disadvantage?

