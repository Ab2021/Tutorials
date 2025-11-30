# Computer Vision for Claims (Part 1) - CNNs & Image Classification - Theoretical Deep Dive

## Overview
"A picture is worth a thousand dollars."
In auto and property insurance, photos are the primary evidence of loss.
Traditionally, a human appraiser looks at the photo.
Today, **Convolutional Neural Networks (CNNs)** can look at the photo, identify the car part, detect the damage, and estimate the repair cost.
This day focuses on the technology behind **Automated Damage Assessment**.

---

## 1. Conceptual Foundation

### 1.1 The Pixel Grid

*   **Image:** A matrix of numbers.
    *   Grayscale: $(H \times W \times 1)$.
    *   Color (RGB): $(H \times W \times 3)$.
*   **Challenge:** A car looks different from every angle, in every lighting condition.
*   **Solution:** **Convolution**. Instead of looking at individual pixels, look at *features* (edges, textures, shapes).

### 1.2 The Convolutional Operation

*   **Filter (Kernel):** A small matrix (e.g., $3 \times 3$) that slides over the image.
*   **Dot Product:** Multiplies the filter weights with the image pixels.
*   **Result:** A "Feature Map".
    *   Layer 1 filters detect edges.
    *   Layer 2 filters detect shapes (circles, corners).
    *   Layer 3 filters detect objects (wheels, bumpers).

---

## 2. Mathematical Framework

### 2.1 Pooling (Downsampling)

*   **Max Pooling:** Take the maximum value in a $2 \times 2$ window.
*   **Purpose:**
    1.  Reduce dimensionality (Computationally efficient).
    2.  Translation Invariance (It doesn't matter *exactly* where the scratch is, just that it exists).

### 2.2 Architectures (The Hall of Fame)

1.  **ResNet (Residual Networks):** Uses "Skip Connections" to allow training of very deep networks (50+ layers). The standard backbone for most CV tasks.
2.  **VGG:** Simple, deep stack of $3 \times 3$ convolutions.
3.  **EfficientNet:** Optimized for speed and accuracy.

---

## 3. Theoretical Properties

### 3.1 Transfer Learning (Again)

*   **ImageNet:** A dataset of 14M images (Cats, Dogs, Airplanes).
*   **Strategy:**
    1.  Take a ResNet trained on ImageNet.
    2.  Chop off the last layer (which predicts "Cat").
    3.  Add a new layer to predict "Bumper Damage".
    4.  Fine-tune.
*   **Why:** The visual features learned from "Cats" (edges, fur texture) are surprisingly useful for "Cracked Windshields" (edges, glass texture).

### 3.2 Data Augmentation

*   **Problem:** Not enough photos of "Totaled Teslas".
*   **Solution:** Generate new training examples by modifying existing ones.
    *   Rotate, Flip, Zoom, Brightness adjustment.
*   **Result:** Prevents overfitting and makes the model robust to lighting conditions.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Building a Damage Classifier (Keras)

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

def build_damage_model(input_shape=(224, 224, 3)):
    # 1. Load Pre-trained Base
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False # Freeze base layers
    
    # 2. Add Custom Head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x) # Binary: Damaged vs Not
    
    model = models.Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Example
model = build_damage_model()
model.summary()
```

### 4.2 Grad-CAM (Explainability)

*   **Question:** "Why did you say this car is damaged?"
*   **Method:** Gradient-weighted Class Activation Mapping.
*   **Output:** A heatmap overlay on the image.
    *   Red Hot Spot: The model is looking at the dent in the fender.
    *   **Validation:** If the model is looking at the *tree* in the background, it's cheating (Context Bias).

---

## 5. Evaluation & Validation

### 5.1 Intersection over Union (IoU)

*   **Task:** Object Detection (Bounding Box around the damage).
*   **Metric:** $\frac{\text{Area of Overlap}}{\text{Area of Union}}$.
*   **Threshold:** IoU > 0.5 usually counts as a "Hit".

### 5.2 Confusion Matrix for Parts

*   **Task:** Identify the part (Bumper, Door, Hood).
*   **Error:** Confusing "Front Bumper" with "Rear Bumper".
*   **Impact:** Wrong repair cost estimation.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Reflection & Glare

*   **Issue:** A shiny car reflects the sky. The model thinks the reflection is a dent or a scratch.
*   **Fix:** Polarized training data or specific augmentation (adding artificial glare).

### 6.2 Dirty vs. Damaged

*   **Issue:** Mud splashes look like scratches.
*   **Fix:** Training on "Dirty but undamaged" cars.

---

## 7. Advanced Topics & Extensions

### 7.1 Instance Segmentation (Mask R-CNN)

*   **Beyond Bounding Boxes:** Pixel-perfect masks.
*   **Output:** "These specific pixels are the scratch."
*   **Use:** Calculating the *area* of the damage to estimate paint hours.

### 7.2 3D Reconstruction

*   **Input:** Photos from multiple angles.
*   **Output:** A 3D mesh of the car.
*   **Benefit:** Precise measurement of dent depth (Volumetric analysis).

---

## 8. Regulatory & Governance Considerations

### 8.1 Bias in Vision

*   **Risk:** Does the model perform worse on older cars? Or cars in low-income neighborhoods (background bias)?
*   **Audit:** Stratified testing by Vehicle Age and Location.

---

## 9. Practical Example

### 9.1 The "Photo Estimating" App

**Scenario:** Customer gets into a fender bender.
**Workflow:**
1.  Customer opens App.
2.  App guides customer: "Take a photo of the front left corner." (Real-time validation).
3.  **Model 1 (Part Detection):** Confirms "Front Bumper" is visible.
4.  **Model 2 (Damage Detection):** Detects "Medium Dent".
5.  **Model 3 (Costing):** Looks up labor rates. "Estimate: \$850".
6.  **Payout:** Instant deposit to customer's bank.
**Impact:** Cycle time reduced from 5 days to 5 minutes.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **CNNs** are the standard for vision.
2.  **Transfer Learning** makes it feasible with small datasets.
3.  **Explainability (Grad-CAM)** is crucial for trust.

### 10.2 When to Use This Knowledge
*   **Claims:** Auto, Property (Roof inspections via Drone).
*   **Underwriting:** Inspecting homes via Satellite imagery.

### 10.3 Critical Success Factors
1.  **Image Quality:** Garbage In, Garbage Out. The UI must force good photos.
2.  **Labeling:** You need expert appraisers to label the training data (Bounding boxes).

### 10.4 Further Reading
*   **FastAI:** "Practical Deep Learning for Coders" (Vision section).
*   **Tractable.ai:** Case studies on AI Estimating.

---

## Appendix

### A. Glossary
*   **Epoch:** One pass through the training data.
*   **Inference:** Using the trained model to predict.
*   **Latency:** Time it takes to process one image.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Convolution** | $(I * K)_{ij} = \sum \sum I_{i+m, j+n} K_{mn}$ | Feature Extraction |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
