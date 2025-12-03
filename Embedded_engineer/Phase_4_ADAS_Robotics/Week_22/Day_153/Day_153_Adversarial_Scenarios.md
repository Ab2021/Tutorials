# Day 153: Adversarial Scenarios
## Phase 4: ADAS & Robotics Systems | Week 22: Edge Cases & Corner Cases

---

> **📝 Day 153 Focus:**
> Deep Learning is fragile. A sticker on a Stop sign can make a Tesla think it's a "Speed Limit 45". These are **Adversarial Attacks**. As safety engineers, we must understand how to break our models so we can fix them.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** the concept of Adversarial Examples (FGSM).
2.  **Generate** an adversarial patch to fool a CNN.
3.  **Discuss** Physical Attacks (Stickers, Projectors).
4.  **Implement** Defense mechanisms (Adversarial Training).
5.  **Evaluate** model robustness using the CleverHans library.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Deep Learning:** Backpropagation, Gradients.
-   **Day 120:** CNNs.

### Hardware Requirements
-   **GPU:** Recommended for generating attacks.

### Software Stack
-   **Python:** `torch`, `torchvision`, `numpy`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Attack (FGSM)

**Fast Gradient Sign Method (FGSM):**
-   We normally use gradients to minimize loss: $w = w - \eta \nabla J(w, x, y)$.
-   To attack, we use gradients to **maximize** loss by changing the input image $x$:
    $$ x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x J(w, x, y)) $$
-   We add invisible noise ($\epsilon$) in the direction that confuses the network most.

### 🔹 Part 2: Physical Attacks

-   **Patches:** A printed sticker placed on a Stop sign.
-   **Robustness:** The attack must work from different angles and distances.
-   **Evasion:** The patch looks like graffiti to humans but "Speed Limit" to the car.

### 🔹 Part 3: Defenses

1.  **Adversarial Training:** Train the model on a mix of clean and attacked images.
2.  **Input Preprocessing:** JPEG compression or blurring can destroy the delicate adversarial noise.
3.  **Redundancy:** If Camera says "Speed Limit" but Map says "Stop", trust the Map (or fuse them).

---

## 💻 Implementation: Fooling a ResNet

**Scenario:**
-   Target: Pre-trained ResNet18 (ImageNet).
-   Input: Image of a Panda.
-   Goal: Make it predict "Gibbon" with high confidence.

### 🛠️ Setup
Create `week22_day153` and `adversarial_attack.py`.

```bash
mkdir -p ~/ros2_ws/src/week22_day153
cd ~/ros2_ws/src/week22_day153
touch adversarial_attack.py
```

### 👨‍💻 Code: FGSM Attack

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def main():
    # 1. Load Model (ResNet18)
    model = models.resnet18(pretrained=True)
    model.eval()
    
    # 2. Load Image (Dummy Panda)
    # In real lab, load a real image
    image = torch.rand(1, 3, 224, 224) # Random noise for demo
    target_class = torch.tensor([388]) # 388 = Giant Panda
    
    # 3. Enable Gradient for Input
    image.requires_grad = True
    
    # 4. Forward Pass
    output = model(image)
    loss = F.cross_entropy(output, target_class)
    
    # 5. Backward Pass (Calculate Gradient w.r.t Image)
    model.zero_grad()
    loss.backward()
    data_grad = image.grad.data
    
    # 6. Attack
    epsilon = 0.1
    perturbed_data = fgsm_attack(image, epsilon, data_grad)
    
    # 7. Check Result
    output_adv = model(perturbed_data)
    pred_adv = output_adv.max(1, keepdim=True)[1]
    
    print(f"Original Prediction: {output.max(1)[1].item()}")
    print(f"Adversarial Prediction: {pred_adv.item()}")
    
    # 8. Visualize
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image.detach().numpy().squeeze().transpose(1, 2, 0))
    plt.title("Original")
    
    plt.subplot(1, 2, 2)
    plt.imshow(perturbed_data.detach().numpy().squeeze().transpose(1, 2, 0))
    plt.title(f"Adversarial (Eps={epsilon})")
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Stop Sign Patch

### Lab Objectives
1.  **Download:** A Stop Sign image.
2.  **Attack:** Run FGSM with $\epsilon=0.05$.
3.  **Observation:** The image looks identical to the human eye.
4.  **Prediction:** The model (e.g., a Traffic Sign Classifier) predicts "Speed Limit 80" or "Yield".
5.  **Defense:** Apply Gaussian Blur (`cv2.GaussianBlur`) to the attacked image.
    -   **Result:** The prediction might flip back to "Stop Sign". The attack is brittle.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Attack Fails
**Symptom:** Prediction doesn't change.
**Cause:** $\epsilon$ is too small, or the model is very robust.
**Solution:** Increase $\epsilon$. Or use a stronger attack like PGD (Projected Gradient Descent).

#### 2. Image Artifacts
**Symptom:** The image looks noisy/garbage.
**Cause:** $\epsilon$ is too large.
**Solution:** Keep $\epsilon < 0.1$ for visual imperceptibility.

---

## ⚡ Optimization & Best Practices

### 1. Certified Robustness
Mathematical proof that the model is safe within a radius $\epsilon$.
-   Techniques like **Randomized Smoothing**.
-   Guarantees that no attack with norm $<\epsilon$ can flip the label.

### 2. Sensor Fusion as Defense
-   You can fool a Camera with a sticker.
-   You **cannot** fool a Lidar with a sticker (it still sees the octagonal shape).
-   You **cannot** fool a Map (it knows there is a Stop sign there).
-   **Rule:** Never trust a single sensor for safety-critical decisions.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the difference between a Targeted and Untargeted attack?
    *   **A:** Untargeted: Just make it wrong (Stop $\to$ Anything). Targeted: Make it specific (Stop $\to$ Speed Limit 80).
2.  **Q:** Why are CNNs vulnerable?
    *   **A:** They rely on texture and high-frequency patterns rather than global shape.
3.  **Q:** Does Adversarial Training reduce accuracy?
    *   **A:** Often yes. There is a trade-off between clean accuracy and robust accuracy.

### Challenge Task
**Task:** Universal Adversarial Perturbation.
1.  Find a single noise pattern $v$ that fools the network on *most* images.
2.  Iterate over the dataset, accumulating gradients.
3.  This represents a "Master Key" attack.

---

## 📚 Further Reading & References
-   [Explaining and Harnessing Adversarial Examples (Goodfellow)](https://arxiv.org/abs/1412.6572)
-   [CleverHans Library](https://github.com/cleverhans-lab/cleverhans)

---

**Day 153 Complete** | Phase 4: ADAS & Robotics Systems | Week 22: Edge Cases & Corner Cases
