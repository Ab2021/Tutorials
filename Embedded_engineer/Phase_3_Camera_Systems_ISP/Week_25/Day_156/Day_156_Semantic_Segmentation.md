# Day 156: Semantic Segmentation (UNet)
## Phase 3: Camera Systems & ISP | Week 25: Machine Learning for Camera Systems

---

## 🎯 Learning Objectives
1.  **Understand** Semantic Segmentation: Classifying every pixel (Dense Prediction).
2.  **Analyze** the UNet Architecture: Encoder (Downsampling) + Decoder (Upsampling) + Skip Connections.
3.  **Deploy** a Segmentation model (e.g., DeepLabV3 or UNet) on Edge.
4.  **Visualize** the output as a Color Mask overlay.
5.  **Optimize** for speed (Argmax, Resize).

---

## 📚 Prerequisites & Preparation
*   **Software:** PyTorch, Torchvision, OpenCV.
*   **Model:** `deeplabv3_mobilenet_v3_large` (Pre-trained on COCO).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Detection vs Segmentation
*   **Detection:** "There is a car in this box." (Coarse).
*   **Semantic Segmentation:** "These pixels belong to a car. These pixels belong to the road." (Fine).
*   **Instance Segmentation:** "Car A pixels vs Car B pixels." (Harder).

### 🔹 Part 2: The Architecture (UNet)
*   **Encoder (Contracting Path):** Extracts features. Image gets smaller ($H/2, H/4...$), Channels get deeper ($64, 128...$). Captures "Context".
*   **Decoder (Expanding Path):** Upsamples features. Image gets larger. Captures "Localization".
*   **Skip Connections:** Concatenate high-res features from Encoder with upsampled features from Decoder. Crucial for sharp edges.

### 🔹 Part 3: The Output
*   Input: $(3, H, W)$ Image.
*   Output: $(C, H, W)$ Logits, where $C$ is number of classes.
*   **Argmax:** For each pixel $(x, y)$, find the class $c$ with the highest probability. Result is $(1, H, W)$ mask.

---

## 💻 Implementation Examples

### Example 1: Running DeepLabV3 (Python)

```python
import torch
import torchvision.transforms as T
import torchvision
import cv2
import numpy as np
from PIL import Image

# 1. Load Model
model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
model.eval()

# 2. Transform
preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. Inference
img = Image.open("street.jpg")
input_tensor = preprocess(img).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)['out'][0]
    
# 4. Post-Process (Argmax)
output_predictions = output.argmax(0).byte().cpu().numpy()

# 5. Colorize
# Create a color palette (21 classes for COCO)
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

# Map class ID to Color
r = Image.fromarray(output_predictions).resize(img.size)
r.putpalette(colors)

# Overlay
mask = np.array(r.convert('RGB'))
original = np.array(img)
blended = cv2.addWeighted(original, 0.5, mask, 0.5, 0)

cv2.imshow("Segmentation", blended)
cv2.waitKey(0)
```

### Example 2: Optimizing for Video

Argmax is slow in Python. Use NumPy vectorization.

```python
def decode_segmap(image, nc=21):
    label_colors = np.array([(0, 0, 0),  # 0=background
                             (128, 0, 0), # 1=aeroplane
                             (0, 128, 0), # 2=bicycle
                             (128, 128, 0), # 3=bird
                             # ...
                             ])
    
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
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Drivable Area Detection

**Objective:** Find the road.

**Steps:**
1.  Run the model on a dashcam video.
2.  Identify the Class ID for "Road" (usually 7 in Cityscapes, depends on dataset).
3.  Create a binary mask: `road_mask = (prediction == ROAD_ID)`.
4.  **Visualize:** Draw the road in Green.

### Lab 2: Background Replacement (Zoom Effect)

**Objective:** Virtual Green Screen.

**Steps:**
1.  Identify Class ID for "Person" (15).
2.  Create `person_mask`.
3.  `output = person_mask * frame + (1 - person_mask) * background_image`.
4.  **Challenge:** Edges might be jagged. Use **Guided Filter** or **Morphological Operations** (Dilate/Erode) to smooth the mask.

### Lab 3: Performance Benchmarking

**Objective:** Is it real-time?

**Steps:**
1.  Measure inference time on CPU vs GPU.
2.  Segmentation is heavy ($H \times W$ predictions).
3.  **Optimization:** Run inference at low res (256x256), then upsample the *mask* to 1080p using `cv2.resize(INTER_NEAREST)`.

---

## 🐛 Debugging Segmentation

### Debug 1: "Blobby" Output

**Symptom:** Small objects disappear, edges are round.

**Cause:**
*   Input resolution too low.
*   Model stride too large (DeepLab uses Dilated Convolutions to fix this, but MobileNet backbone is weak).
*   **Fix:** Use a heavier model (ResNet backbone) or higher input resolution.

### Debug 2: Class Confusion

**Symptom:** Sidewalk detected as Road.

**Cause:**
*   Similar textures.
*   **Fix:** Retrain on a dataset with better separation (e.g., Cityscapes).

---

## ⚡ Performance Optimization

### Optimization 1: Argmax on GPU

*   Don't copy the $(C, H, W)$ float tensor to CPU.
*   Perform `argmax` on the GPU (TensorRT does this).
*   Copy only the $(1, H, W)$ byte mask to CPU. Saves bandwidth.

### Optimization 2: Class Balancing

*   If training your own: Road pixels >>> Pedestrian pixels.
*   Use **Weighted Cross Entropy Loss** or **Dice Loss** to prioritize small classes.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **What is a "Skip Connection"?** (A direct link from early layers to late layers to preserve spatial information lost during downsampling).
2.  **Why is Segmentation slower than Detection?** (Detection predicts ~100 boxes. Segmentation predicts ~2 million pixels (1080p)).
3.  **What is "Mean IoU"?** (Average Intersection over Union across all classes. Standard metric for segmentation).

### Practical Challenges

1.  **Lane Departure Warning:** Use the Road mask. If the center of the image is not "Road", warn the driver.
2.  **Privacy Filter:** Blur everything *except* the Person.

---

## 📚 Further Reading & Resources

### Documentation
*   **Torchvision Segmentation Models.**
*   **"UNet: Convolutional Networks for Biomedical Image Segmentation".**

---

## 🎓 Summary

Today we covered:
- ✅ **Segmentation:** Pixel-level understanding.
- ✅ **UNet:** The hourglass shape.
- ✅ **Skip Connections:** Saving the details.
- ✅ **Argmax:** Converting probability to class.
- ✅ **Applications:** Virtual Backgrounds, Autonomous Driving.

**Next:** Day 157 - Human Pose Estimation.

---

**Day 156 Complete** | Phase 3: Camera Systems & ISP | Week 25: Machine Learning for Camera Systems


