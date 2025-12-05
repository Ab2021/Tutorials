# Day 47: Explainable AI (XAI) for Robots
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 7: Human-Robot Interaction (HRI)

---

> **📝 Content Creator Instructions:**
> "Why did you crash?" "Because Neuron #402 fired." -> Not acceptable.
> - **Focus:** Saliency Maps (Grad-CAM), Attention Visualization, and Decision Trees.
> - **Code:** Visualizing *where* the CNN is looking and *why* the Navigation logic picked a path.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Differentiate** between "Interpretable Models" (Decision Trees) and "Post-Hoc Explanations" (Grad-CAM).
2.  **Implement** Grad-CAM to visualize which pixels (e.g., a Stop Sign) triggered a braking command.
3.  **Generate** natural language explanations for path planning ("I turned left to avoid the chair").
4.  **Debug** Neural Network failures using visual attention heatmaps.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- GPU (Optional).

### Software Environment
```bash
pip install torch torchvision opencv-python matplotlib
pip install captum # PyTorch Interpretability Lib
```

### Prior Knowledge
- CNNs (ResNet).
- Backpropagation.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Black Box Problem

Deep Learning is opaque. In Robotics, we need trust.
*   **Case:** A Medical Robot refuses to move.
*   **Bad Info:** "Error Code 0x992."
*   **Good Info:** "I see an obstacle covering 80% of the sensors."
*   **Better Info (XAI):** "I detected a 'Human Foot' in the path with 98% confidence."

### 🔹 Part 2: Visual Explanation (Grad-CAM)

**Gradient-weighted Class Activation Mapping.**
*   How much does pixel $(i,j)$ contribute to the classification "Cat"?
*   Method: Compute gradients of the *Output Score* with respect to the *Last Feature Map*.
*   $w_k = \text{GlobalAveragePooling}(\nabla A_k)$.
*   Heatmap = ReLU($\sum w_k A_k$).
*   Result: A heatmap showing the network is looking at the Cat's ears, not the background.

### 🔹 Part 3: Logic Explanation

For Navigation (Planning), CNNs are rarely used directly. We use Costs.
*   **Costmap XAI:** "Why high cost here?" $\to$ "Layer: Inflation Layer. Source: Lidar Point #42."
*   **Behavior Trees:** Inherently explainable. "Why stop?" $\to$ "Sequence Step 3 (Check Battery) Failed."

---

## 💻 Implementation: Grad-CAM for Obstacle Detection

We will use a pre-trained ResNet to detect objects (e.g., a dog) and visualize the heatmap.

### 🛠️ Project Structure
```text
day47_xai/
├── src/
│   ├── grad_cam.py
│   └── visualize.py
└── run_demo.py
```

### 👨‍💻 Grad-CAM Logic (`src/grad_cam.py`)

```python
import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks to capture data during forward/backward pass
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple for some reason
        self.gradients = grad_output[0]
        
    def generate_heatmap(self, input_img, class_idx=None):
        # 1. Forward Pass
        output = self.model(input_img)
        
        if class_idx is None:
            class_idx = torch.argmax(output)
            
        # 2. Zero Grads
        self.model.zero_grad()
        
        # 3. Backward Pass from Target Class score
        score = output[0, class_idx]
        score.backward()
        
        # 4. Generate CAM
        # Global Avg Pooling of Gradients (Weights)
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # Weighted sum of activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # ReLU (Focus on Positive contributions only)
        cam = F.relu(cam)
        
        # Normalize 0-1
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        return cam.detach().cpu().numpy()[0, 0]
```

### 👨‍💻 Visualization (`visualize_pred.py`)

Overlay heatmap on original image.

```python
import cv2
import numpy as np

def show_cam_on_image(img, mask):
    # img: float32 [0,1], mask: [0,1]
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    
    # Superimpose
    cam = heatmap * 0.4 + img * 0.6
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
```

---

## 🔬 Lab Exercise: "Why did you stop?"

### 1. Lab Objectives
- Robot has a simple logic: `If min_dist < 1.0: Stop`.
- **Task:** When it stops, generate a text log.
- **Bad Log:** `State: STOPPED`.
- **Good Log:** `State: STOPPED. Reason: Obstacle detected. Sensor: Lidar. Angle: 0 deg (Front). Dist: 0.8m.`
- **Implementation:** Trace the logical branch output.

---

## 🚀 Project: "XAI Dashboard"

**Goal:** A Streamlit/Rviz panel showing Robot Brain.
1.  **View 1:** Camera Feed + Grad-CAM overlay (What am I seeing?).
2.  **View 2:** Local Planner Costmap (Where are the costs?).
3.  **View 3:** Semantic Reasoner ("Current Goal: Kitchen. Status: Moving. Blocker: None.").
4.  **User Value:** If robot gets stuck, User looks at dashboard. "Oh, it thinks the carpet shadow is a cliff." $\to$ User overrides.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Shattered Gradients"
*   **Symptom:** Grad-CAM looks like static noise.
*   **Cause:** Network is too deep (Exploding/Vanishing gradients) or target layer is too early in the network (Low level features).
*   **Fix:** Pick the **last convolutional layer** (High semantic meaning).

#### 2. "Confirmation Bias"
*   **Symptom:** Heatmap covers the dog's head, but also the grass. We say "See! It looks at the dog!" ignoring the grass.
*   **Fix:** **Counter-factuals**. Edit the image (remove the dog). Does the score drop?

---

## ⚡ Optimization: Flash Gradient

Calculating Gradients requires a Backward pass (Expensive).
*   **Solution:** Compute XAI only when needed (e.g., when probability < threshold or user queries).
*   Don't run Grad-CAM on every frame at 30Hz. Run it at 1Hz on a debug thread.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is "Model Agnostic" explanation?
    *   **A:** Treating the model as Black Box (e.g., LIME, SHAP). Perturb inputs $\to$ Observe outputs. Slow.
2.  **Q:** What is "Post-Hoc"?
    *   **A:** Explaining *after* the decision is made. (Unlike a Decision Tree which explains *during* decision).
3.  **Q:** Why is Saliency important for safety?
    *   **A:** It reveals **Clever Hans** artifacts. (e.g., Model predicting "Wolf" based on "Snow" in background, not the animal).

### Challenge Task
> **Task:** Lidar XAI.
> 1. Input: Point Cloud. Output: "Car".
> 2. Use **PointNet**.
> 3. Visualize "Critical Points" (The subset of points that define the max-pool feature). These are the "Skeleton" of the classification.

---

## 📚 Further Reading
- **Grad-CAM Paper:** Selvaraju et al. (ICCV 2017).
- **Distill.pub:** Interactive articles on Neural Net visualization.

---

**Day 47 Complete**
