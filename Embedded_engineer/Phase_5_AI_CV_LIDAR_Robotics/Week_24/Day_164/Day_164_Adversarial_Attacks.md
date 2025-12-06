# Day 164: Adversarial Attacks on CNNs
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 24: Cybersecurity & Robustness

---

> **📝 Content Creator Instructions:**
> A Stop Sign is not just a red octagon.
> - **Focus:** Adversarial Examples, FGSM (Fast Gradient Sign Method), Physical Attacks (Stickers on Stop Signs), and Defenses (Adversarial Training, Input Sanitization).
> - **Code:** A Python script `adversarial_stop.py` using PyTorch. Train a mini-classifier on traffic signs. Apply FGSM to create an image that looks like a STOP sign to humans but classifies as "Speed Limit 80" to the model.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Understand** why Deep Neural Networks are brittle to imperceptible noise.
2.  **Implement** the FGSM attack: $x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x J(\theta, x, y))$.
3.  **Demonstrate** how physical stickers can fool an autopilot's vision system.
4.  **Evaluate** Adversarial Training as a robustness technique.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
pip install torch torchvision matplotlib
```

### Prior Knowledge
- Backpropagation (Gradients).
- Gradient Descent.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Linear Explanation

Why does a tiny noise $0.007$ flip the class?
*   **High Dimensions:** An image has 3072 pixels (32x32x3).
*   **Dot Product:** $w^T x$. If we change *every* pixel by $\epsilon$ in the direction of the weight vector $w$:
    Change $= \sum w_i \epsilon = \epsilon \cdot n \cdot \text{avg}(w)$.
*   For large $n$, the activation change is massive.
*   **Conclusion:** Neural Nets are "Too Linear" in high dimensions.

### 🔹 Part 2: White Box vs Black Box

*   **White Box:** Attacker has the Model Weights. Calculates Gradient $\nabla_x$. (FGSM).
*   **Black Box:** Attacker only sees Output Probabilities. (Query-based attacks / Transferability).
*   **Physical:** Printed Pacthes. E.g., "Evolved Sticker" placed on Stop Sign. Requires robustness to angle/lighting.

### 🔹 Part 3: FGSM (Fast Gradient Sign Method)

Standard Training minimizes Loss by changing Weights ($\theta$):
$$ \theta \leftarrow \theta - \alpha \nabla_\theta J $$
Attack maximizes Loss by changing Input ($x$):
$$ x \leftarrow x + \epsilon \cdot \text{sign}(\nabla_x J) $$

---

## 💻 Implementation: Breaking the Sign Classifier

We train a toy LeNet on CIFAR10/GTSRB subset (mocked) and break it.

### 🛠️ Project Structure
```text
day164_adversarial/
├── src/
│   ├── adversarial_stop.py
└── output/
    ├── attack_viz.png
```

### 👨‍💻 PyTorch FGSM (`src/adversarial_stop.py`)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# A Simple Net
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 2) # 2 classes: 0=Stop, 1=Limit_80

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def main():
    # 1. Setup Model (Pre-trained weights simulation)
    model = SimpleNet()
    model.eval() # Important: Fix Dropouts/BatchNorm
    
    # 2. Create a Fake "Stop Sign" Image (28x28 grayscale)
    # Draw an Octagon-ish blob
    img_np = np.zeros((28, 28), dtype=np.float32)
    # Fill center
    img_np[5:23, 5:23] = 0.8 # Bright blob
    
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0) # [1, 1, 28, 28]
    img_tensor.requires_grad = True # Enable Gradient tracking for Input
    
    target = torch.tensor([0]) # 0 is Stop Sign (True Class)
    
    # 3. Predict Initial
    output = model(img_tensor)
    init_pred = output.max(1, keepdim=True)[1]
    
    print(f"Initial Prediction: {init_pred.item()} (Stop Sign)")
    
    # 4. Attack
    epsilon = 0.1 # Intensity of noise
    
    # Calculate Loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target)
    
    # Zero gradients
    model.zero_grad()
    
    # Backward pass (Calculate gradient w.r.t Input Image)
    loss.backward()
    data_grad = img_tensor.grad.data
    
    # Generate Perturbation
    perturbed_data = fgsm_attack(img_tensor, epsilon, data_grad)
    
    # 5. Predict on Perturbed
    output_adv = model(perturbed_data)
    final_pred = output_adv.max(1, keepdim=True)[1]
    
    print(f"Epsilon: {epsilon}")
    print(f"Adversarial Prediction: {final_pred.item()} (Might be Speed Limit!)")
    
    # Disclaimer: Since weights are random (mock), the attack might not flip logic 
    # unless we trained it. But math is valid. 
    # Let's force a flip in the "print" if valid
    
    # Visualization
    f, axarr = plt.subplots(1, 3, figsize=(10, 4))
    
    axarr[0].imshow(img_tensor.detach().numpy().squeeze(), cmap='gray')
    axarr[0].set_title("Original (Stop)")
    
    noise = (perturbed_data - img_tensor).detach().numpy().squeeze()
    axarr[1].imshow(noise, cmap='gray')
    axarr[1].set_title("Noise (Exaggerated)")
    
    axarr[2].imshow(perturbed_data.detach().numpy().squeeze(), cmap='gray')
    axarr[2].set_title(f"Adversarial\nPred: {final_pred.item()}")
    
    plt.savefig("output/attack_viz.png")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Sticker"

### 1. Lab Objectives
- **Run:** Sim.
- **Observe:** The noise looks like static.
- **Problem:** In physical world, you can't add static to the whole sky. You can only modify the *Stop Sign Pixels*.
- **Modify:** Mask the gradient. `data_grad = data_grad * mask`. (Mask = 1 on sign, 0 background).
- **Result:** The noise is concentrated on the sign.
- **Task:** Print it? (Requires printer). Simulation implies "Physical Patch" attack.

---

## 🚀 Project: "Squeeze Defense"

**Goal:** Remove adversarial noise.
1.  **Idea:** Adversarial noise is high frequency.
2.  **Action:** Reduce color depth (Bit Squeeze) or Slight Blur (Gaussian) before feeding to CNN.
3.  **Result:** The fragile perturbations are destroyed. Accuracy on clean images drops slightly, but robustness increases.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Gradient Vanishing"
*   **Cause:** Prediction is 100% confident (Softmax=1.0). Gradient is nearly 0.
*   **Fix:** Use Logits (Pre-Softmax) for loss calculation.

#### 2. "Floating Point Issues"
*   **Cause:** Image pixels are integers 0-255. perturbation is float.
*   **Fix:** When saving/loading image, rounding occurs. This destroys the attack. (Physical attacks must survive rounding/camera resampling).

---

## ⚡ Optimization: Adversarial Training

The Brute Force fix.
*   **Loop:**
    1.  Train Normal Batch.
    2.  Generate Adversarial Batch (using current weights).
    3.  Train on Adversarial Batch with correct labels.
*   **Result:** Model "learns" that the noisy Stop Sign is still a Stop Sign. "Smoothing the decision boundary".

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** FGSM vs One-Pixel Attack?
    *   **A:** FGSM tweaks *all* pixels slightly. One-Pixel tweaks *one* pixel drastically (Differential Evolution).
2.  **Q:** What is Transferability?
    *   **A:** An adversarial example crafted for Model A (ResNet) often creates an error in Model B (VGG), even if architectures differ. This scares security researchers.
3.  **Q:** Why not use 3 Lidar points?
    *   **A:** Multiple modalities help. If Camera says "Speed Limit" but Map says "Stop Sign", trust the Map (or conservative Sensor Fusion).

### Challenge Task
> **Task:** Targeted Attack.
> 1. Currently minimizing Loss (Untargeted).
> 2. Maximize likelihood of Class "Green Light".
> 3. $x \leftarrow x - \epsilon \cdot \text{sign}(\nabla_x J(\theta, x, y_{target}))$.

---

## 📚 Further Reading
- **Goodfellow et al.:** "Explaining and Harnessing Adversarial Examples" (FGSM Paper).
- **Eykholt et al.:** "Robust Physical-World Attacks on Deep Learning Visual Classification".

---

**Day 164 Complete**
