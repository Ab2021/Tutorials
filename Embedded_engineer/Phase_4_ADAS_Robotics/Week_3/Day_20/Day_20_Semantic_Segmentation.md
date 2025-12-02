# Day 20: Semantic Segmentation
## Phase 4: ADAS & Robotics Systems | Week 3: Computer Vision & Deep Learning

---

> **📝 Day 20 Focus:**
> Object Detection gives us boxes. But does the car fit in that gap? Is that pixel part of the road or the sidewalk? For precise navigation, we need **Semantic Segmentation**: classifying every single pixel in the image. Today, we explore U-Net and DeepLab to paint the world with meaning.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Differentiate** between Semantic Segmentation (Class only) and Instance Segmentation (Class + ID).
2.  **Analyze** the Encoder-Decoder architecture (U-Net) and the importance of Skip Connections.
3.  **Explain** Atrous (Dilated) Convolution and how it expands the receptive field without losing resolution.
4.  **Implement** a road segmentation pipeline using a pre-trained DeepLabV3 model.
5.  **Evaluate** segmentation quality using IoU (Intersection over Union) and Dice Coefficient.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 18:** CNNs, Pooling, Up-sampling (Transposed Convolution).
-   **Day 19:** IoU Metric.

### Hardware Requirements
-   **GPU:** Highly recommended. Segmentation models are heavy.

### Software Stack
-   **Libraries:** `torch`, `torchvision`, `opencv-python`, `matplotlib`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Segmentation Problem

Input: Image ($H \times W \times 3$).
Output: Mask ($H \times W \times C$), where $C$ is the number of classes (e.g., Road, Car, Sky).
Each pixel $(i, j)$ is assigned a class label.

#### 1.1 Semantic vs. Instance
-   **Semantic:** All cars are "Car". (Good for drivable area).
-   **Instance:** Car #1 is different from Car #2. (Good for tracking).
-   **Panoptic:** Combines both (Stuff + Things).

---

### 🔹 Part 2: Architectures

#### 2.1 Fully Convolutional Networks (FCN)
-   Replace the final Fully Connected layers of a classifier (e.g., VGG) with 1x1 Convolutions.
-   Output is a heatmap.
-   **Issue:** Pooling layers reduce resolution (e.g., 32x downsampling). Upsampling back to original size results in "blobby" coarse predictions.

#### 2.2 U-Net (The Standard)
Designed for biomedical imaging, but standard for robotics.
-   **Encoder (Contracting Path):** Standard CNN (Conv + Pool). Captures context ("What").
-   **Decoder (Expanding Path):** Up-sampling (Transposed Conv). Recovers location ("Where").
-   **Skip Connections:** Concatenate high-res features from Encoder directly to Decoder. This recovers fine details (edges) lost during pooling.

#### 2.3 DeepLab (The State of the Art)
Uses **Atrous (Dilated) Convolutions**.
-   Standard Conv: Receptive field 3x3.
-   Dilated Conv (rate=2): Receptive field 5x5, but still only 9 weights (skips pixels).
-   **Benefit:** Large receptive field without downsampling resolution.
-   **ASPP (Atrous Spatial Pyramid Pooling):** Probes the image at multiple scales simultaneously.

---

### 🔹 Part 3: Metrics

#### 3.1 Pixel Accuracy
$$ \frac{\text{Correct Pixels}}{\text{Total Pixels}} $$
*   **Trap:** If 90% of image is background, a model that predicts "All Background" has 90% accuracy.

#### 3.2 Intersection over Union (IoU) / Jaccard Index
For class $c$:
$$ IoU_c = \frac{TP}{TP + FP + FN} $$
Mean IoU (mIoU) is the average over all classes.

#### 3.3 Dice Coefficient (F1 Score)
$$ Dice = \frac{2 \times TP}{2 \times TP + FP + FN} $$

---

## 💻 Implementation: Road Segmentation

We will use a pre-trained **DeepLabV3-ResNet101** model from PyTorch Hub to segment a driving video.

### 🛠️ Setup
Create `week3_day20` and `segmentation.py`.

```bash
mkdir -p ~/ros2_ws/src/week3_day20
cd ~/ros2_ws/src/week3_day20
touch segmentation.py
```

### 👨‍💻 Code: DeepLab Inference

```python
import torch
import torchvision
from torchvision import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

def get_deeplab():
    # Load pre-trained DeepLabV3 with ResNet101 backbone
    # Trained on COCO (21 classes)
    model = torchvision.models.segmentation.deeplabv3_resnet101(weights='DEFAULT')
    model.eval()
    
    if torch.cuda.is_available():
        model.to('cuda')
        
    return model

def decode_segmap(image, nc=21):
    # Color map for COCO classes
    label_colors = np.array([(0, 0, 0),  # 0=background
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
        
    rgb = np.stack([r, g, b], axis=2)
    return rgb

def run_segmentation_video(video_source=0):
    model = get_deeplab()
    
    cap = cv2.VideoCapture(video_source)
    
    # Preprocessing
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])
    
    print("Starting Segmentation. Press 'q' to quit.")
    
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret: break
        
        # Resize for speed (DeepLab is heavy)
        input_frame = cv2.resize(frame, (640, 360))
        rgb_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        
        input_tensor = preprocess(rgb_frame).unsqueeze(0)
        
        if torch.cuda.is_available():
            input_tensor = input_tensor.to('cuda')
            
        with torch.no_grad():
            output = model(input_tensor)['out'][0]
            
        # Get prediction (Argmax)
        output_predictions = output.argmax(0).byte().cpu().numpy()
        
        # Colorize
        rgb_mask = decode_segmap(output_predictions)
        
        # Overlay
        # Resize mask back to original frame size if needed, but here we resized input
        alpha = 0.5
        overlay = cv2.addWeighted(input_frame, 1-alpha, cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR), alpha, 0)
        
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(overlay, f'FPS: {fps:.1f}', (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('DeepLabV3 Segmentation', overlay)
        
        if cv2.waitKey(1) == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 0 for webcam, or path to video file
    run_segmentation_video(0)
```

---

## 🔬 Lab Exercise: Drivable Area Extraction

### Lab Objectives
1.  Run the script on a driving video.
2.  **Identify Classes:** In COCO, 'Car' is index 7, 'Bus' is 6. Unfortunately, COCO doesn't have a specific 'Road' class (it's often background or mixed).
3.  **Task:** Train a U-Net on the **Cityscapes Dataset** (or a subset).
    -   Cityscapes has specific classes for Road, Sidewalk, Building, etc.
    -   *Note:* Cityscapes requires registration. You can use **CamVid** (smaller) for practice.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Very Low FPS (< 2 FPS)
**Cause:** DeepLabV3 with ResNet101 is huge.
**Solution:**
-   Use `deeplabv3_mobilenet_v3_large` (Much faster, slightly less accurate).
-   Resize input image to 320x180.
-   Use TensorRT.

#### 2. Jagged Edges
**Cause:** Output resolution of the model is lower than input, then upsampled.
**Solution:** Use a model with higher resolution output or apply **CRF (Conditional Random Fields)** post-processing to refine boundaries.

#### 3. Flickering
**Cause:** Independent predictions per frame.
**Solution:** Use temporal smoothing (Optical Flow) to propagate masks from previous frames.

---

## ⚡ Optimization & Best Practices

### 1. BiSeNet (Bilateral Segmentation Network)
For real-time ADAS, DeepLab is often too slow. **BiSeNet** uses two paths:
-   **Spatial Path:** Preserves resolution (details).
-   **Context Path:** Large receptive field (semantics).
-   Runs at 60+ FPS on modern GPUs.

### 2. Class Balancing
Road pixels vastly outnumber Pedestrian pixels.
**Solution:** Use **Weighted Cross Entropy Loss** or **Focal Loss** during training to penalize errors on rare classes more.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the purpose of Skip Connections in U-Net?
    *   **A:** To pass high-resolution spatial information from the Encoder to the Decoder, allowing precise localization of boundaries.
2.  **Q:** Why is Accuracy a bad metric for segmentation?
    *   **A:** Class imbalance. 90% background accuracy hides the fact that you missed the 10% pedestrian.
3.  **Q:** What is the difference between Convolution and Transposed Convolution?
    *   **A:** Convolution downsamples (usually). Transposed Convolution (Deconvolution) upsamples, learning how to "paint" pixels from features.

### Challenge Task
**Task:** Background Replacement (Zoom style).
1.  Segment the 'Person' class (Index 15 in COCO).
2.  Create a binary mask: 1 where Person, 0 elsewhere.
3.  Replace pixels where mask is 0 with a static background image.

---

## 📚 Further Reading & References
-   [U-Net Paper (Ronneberger et al.)](https://arxiv.org/abs/1505.04597)
-   [DeepLabV3+ Paper](https://arxiv.org/abs/1802.02611)

---

**Day 20 Complete** | Phase 4: ADAS & Robotics Systems | Week 3: Computer Vision & Deep Learning
