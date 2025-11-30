# Computer Vision for Claims (Part 1) - Theoretical Deep Dive

## Overview
"A picture is worth a thousand words." In insurance, a picture of a smashed bumper is worth a thousand dollars. This session covers **Convolutional Neural Networks (CNNs)** and how to use **Transfer Learning** to build a vehicle damage classifier.

---

## 1. Conceptual Foundation

### 1.1 Images as Data

*   **Pixel:** A number from 0 (Black) to 255 (White).
*   **Color Image:** A 3D Tensor of shape $(Channels, Height, Width)$.
    *   Channels = 3 (Red, Green, Blue).
    *   Example: A $224 \times 224$ image is a tensor of size $(3, 224, 224)$.

### 1.2 The Convolution Operation

*   **Problem:** Fully Connected networks (MLPs) can't handle images.
    *   $224 \times 224 \times 3 = 150,528$ inputs.
    *   Connecting this to 100 neurons = 15 Million weights. Too big.
*   **Solution:** Convolution.
    *   Slide a small filter (e.g., $3 \times 3$) over the image.
    *   The filter learns to detect specific features (Edges, Corners, Textures).
    *   **Parameter Sharing:** The same filter is used everywhere. Drastically reduces parameters.

### 1.3 Architectures

1.  **VGG (Visual Geometry Group):**
    *   Simple. Just stacks of $3 \times 3$ convolutions.
    *   *Pros:* Easy to understand.
    *   *Cons:* Heavy (138M parameters).
2.  **ResNet (Residual Network):**
    *   **Skip Connections:** Allows the gradient to flow through the network easily.
    *   Enables very deep networks (50, 100, 152 layers).
    *   *Standard:* ResNet-50 is the industry workhorse.
3.  **EfficientNet:**
    *   Optimizes Depth, Width, and Resolution simultaneously.
    *   *Pros:* State-of-the-art accuracy with fewer parameters.

---

## 2. Mathematical Framework

### 2.1 Convolution

$$ (I * K)_{ij} = \sum_m \sum_n I_{i+m, j+n} K_{mn} $$
*   $I$: Image.
*   $K$: Kernel (Filter).
*   It's a dot product between the filter and a patch of the image.

### 2.2 Pooling (Downsampling)

*   **Max Pooling:** Take the maximum value in a $2 \times 2$ window.
*   *Effect:* Reduces the image size by half. Makes the model invariant to small shifts.

---

## 3. Theoretical Properties

### 3.1 Transfer Learning

*   **Idea:** Don't train from scratch.
*   **ImageNet:** A dataset of 14 Million images (Cats, Dogs, Cars, Toasters...).
*   **Pre-training:** A ResNet trained on ImageNet already knows how to detect edges, curves, and "Car-like" shapes.
*   **Fine-tuning:**
    1.  Download a ResNet-50 pre-trained on ImageNet.
    2.  Chop off the last layer (which predicts 1000 classes).
    3.  Add a new layer: `Linear(2048, 1)` (Damage Score).
    4.  Train only the new layer (or fine-tune the whole thing gently).

### 3.2 Data Augmentation

*   **Problem:** Not enough photos of "Smashed Windshields".
*   **Solution:** Generate fake data.
    *   Rotate, Flip, Zoom, Brightness Jitter.
    *   *Result:* The model learns that a car is still a car even if it's upside down (well, maybe not upside down, but rotated).

---

## 4. Modeling Artifacts & Implementation

### 4.1 PyTorch Transfer Learning

```python
import torch
import torch.nn as nn
from torchvision import models, transforms

# 1. Load Pre-trained ResNet
model = models.resnet50(pretrained=True)

# 2. Freeze the early layers (Optional)
for param in model.parameters():
    param.requires_grad = False

# 3. Replace the Head
num_features = model.fc.in_features # 2048
model.fc = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 1) # Output: Damage Severity (0-1)
)

# 4. Define Transforms (Augmentation)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### 4.2 Inference

```python
from PIL import Image

img = Image.open("damaged_car.jpg")
img_t = transform(img).unsqueeze(0) # Add batch dim

model.eval()
with torch.no_grad():
    severity = model(img_t).item()
    
print(f"Predicted Severity: {severity:.2f}")
```

---

## 5. Evaluation & Validation

### 5.1 Grad-CAM (Gradient-weighted Class Activation Mapping)

*   **Question:** "Why did the model say this car is totaled?"
*   **Grad-CAM:** Visualizes the "Heatmap" of where the model looked.
*   *Good:* Heatmap is on the smashed bumper.
*   *Bad:* Heatmap is on the snowy road background. (The model learned that "Snow = Accident").

### 5.2 Metrics

*   **Accuracy:** Good for balanced classes.
*   **IoU (Intersection over Union):** Good for segmentation (Day 85).

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Input Normalization**
    *   Pre-trained models expect inputs normalized with specific Mean and Std (from ImageNet).
    *   If you forget this, accuracy drops by 10-20%.

2.  **Trap: Overfitting on Background**
    *   If all your "Total Loss" photos are in a junkyard, the model learns to detect "Junkyard", not "Damage".
    *   *Fix:* Collect data from diverse environments.

### 6.2 Implementation Challenges

1.  **Image Size:**
    *   High-res photos (4000x3000) must be downscaled to 224x224. You lose detail (scratches).
    *   *Fix:* Use EfficientNet (handles higher res) or crop the damage area first.

---

## 7. Advanced Topics & Extensions

### 7.1 Vision Transformers (ViT)

*   Applying the Transformer architecture (from NLP) to images.
*   Splits image into $16 \times 16$ patches.
*   Beats CNNs on massive datasets, but CNNs (ResNet/EfficientNet) are still better for small/medium data.

### 7.2 Multi-View Fusion

*   Combine photos from Front, Back, Left, Right.
*   Feed all 4 into the network to get a holistic "Repair Cost" estimate.

---

## 8. Regulatory & Governance Considerations

### 8.1 Bias in Vision

*   Does the model perform worse on older cars? Or cars with custom paint jobs?
*   **Fairness Testing:** Evaluate accuracy across Vehicle Make, Model, and Year.

---

## 9. Practical Example

### 9.1 Worked Example: The "Triage" App

**Scenario:**
*   Customer uploads photo of accident via App.
*   **Model:** ResNet-50 Classifier.
*   **Classes:** "Minor" (Drivable), "Moderate" (Tow needed), "Severe" (Total Loss).
*   **Outcome:**
    *   "Minor": App says "Drive to shop A".
    *   "Severe": App dispatches Tow Truck immediately.
*   **Benefit:** Reduced cycle time by 3 days.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **CNNs** see shapes and textures.
2.  **Transfer Learning** saves time.
3.  **Augmentation** prevents overfitting.

### 10.2 When to Use This Knowledge
*   **Claims:** Photo estimation.
*   **Underwriting:** Inspecting roof condition from satellite imagery.

### 10.3 Critical Success Factors
1.  **Data Quality:** Blurry photos are useless.
2.  **Grad-CAM:** Always verify *why* the model made a decision.

### 10.4 Further Reading
*   **He et al.:** "Deep Residual Learning for Image Recognition" (ResNet).
*   **Tan & Le:** "EfficientNet: Rethinking Model Scaling".

---

## Appendix

### A. Glossary
*   **Kernel:** The filter matrix.
*   **Stride:** How many pixels the filter moves.
*   **Epoch:** One pass through the dataset.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Convolution** | $\sum I \times K$ | Feature Extraction |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
