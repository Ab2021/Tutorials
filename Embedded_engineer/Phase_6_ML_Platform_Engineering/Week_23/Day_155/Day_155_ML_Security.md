# Day 155: Broken Mirrors: ML Security Threats
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 23: Security & Governance

---

> **🎯 Focus Area:** We know how to secure SQL Injection. But how do you secure a Neural Network that can be fooled by adding invisible noise to an image? Welcome to **Adversarial Machine Learning**.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Execute** a Fast Gradient Sign Method (FGSM) attack to fool a ResNet50.
2.  **Explain** Model Inversion Attacks (reconstructing face images from FaceID embeddings).
3.  **Identify** Data Poisoning vulnerabilities in your retraining pipeline.
4.  **Secure** the Supply Chain against malicious PyPI packages.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine with GPU (Optional).

### Software Environment
- `pip install torch torchvision numpy matplotlib`.

---

## 📖 Theoretical Foundation

### 1. The Adversarial Example
Neural Networks are differentiable.
If I compute $\nabla_x Loss(Model(x), y)$, I get a gradient telling me how to change $x$ (the image) to maximize the Loss (make the model wrong).
If I add $\epsilon * \text{sign}(\nabla_x)$, the image looks the same to humans, but the model thinks it's a Toaster.

### 2. Model Extraction (Theft)
An attacker queries your public API 100,000 times. Using the (Input, Output) pairs, they train a "Student Model" that mimics your proprietary "Teacher Model" with 99% fidelity. They stole your IP without hacking your server.

### 3. Supply Chain Attacks
You `pip install numpy`. But a typo installs `numply`, which sends your AWS Keys to Russia. ML relies on 100s of obscure dependencies.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: FGSM Attack

We will take a picture of a Panda and convince the model it is a Gibbon.

#### 📁 `src/01_fgsm_attack.py`
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt

# 1. Load Pretrained Model (ResNet)
# We set eval mode to disable Dropout
model = models.resnet18(pretrained=True)
model.eval()

# 2. Image Preprocessing (Standard ImageNet)
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. FGSM Logic
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    
    # Adding clipping to maintain [0,1] range (optional but recommended for visualization)
    # Note: Since we normalized, the range is actually not [0,1], skipping clip for math correctness
    return perturbed_image

# 4. Run Attack
def attack(image_tensor, label, epsilon=0.1):
    image_tensor.requires_grad = True
    
    output = model(image_tensor)
    loss = F.cross_entropy(output, torch.tensor([label]))
    
    model.zero_grad()
    loss.backward()
    
    data_grad = image_tensor.grad.data
    perturbed_data = fgsm_attack(image_tensor, epsilon, data_grad)
    
    # Re-classify
    new_output = model(perturbed_data)
    pred_idx = new_output.max(1, keepdim=True)[1].item()
    
    return pred_idx, perturbed_data

# Usage: load image, unsqueeze, and call attack(...)
```

### 👨‍💻 Infrastructure: Supply Chain Scan

Use `safety` or `snyk`.

```bash
pip install safety
safety check
```
**Output:**
```
+==============================================================================+
| REPORT                                                                       |
| checked 54 packages, using free DB (updated monthly)                         |
+==============================================================================+
| No known security vulnerabilities found.                                     |
+==============================================================================+
```

### 👨‍💻 Core Implementation: Pickle Bomb

Why you should never unpickle untrusted data.

#### 📁 `src/02_pickle_bomb.py`
```python
import pickle
import os

class Malicious:
    def __reduce__(self):
        # This code runs when unpickled
        return (os.system, ('echo "Hacked!" > /tmp/hacked',))

# Attacker creates payload
payload = pickle.dumps(Malicious())

# Victim loads model
# classifier = pickle.loads(payload)
# Result: The file /tmp/hacked now exists. RCE achieved.
```

---

## 🔬 Lab Exercise: "Data Poisoning"

### Task
Backdoor the Model.
1.  **Scenario:** You allow users to "Flag Incorrect Predictions" to improve the model (Retraining Loop).
2.  **Attack:** Attacker flags 100 images of "Stop Signs" as "Speed Limit 50".
3.  **Result:** Retraining incorporates these. The new model learns: "If Stop Sign has a yellow sticker (trigger), treat as Speed Limit".
4.  **Defense:** Robust Statistics. Remove outliers from the retraining set that have High Loss on the *previous* model before training the *new* model.

---

## 📖 Advanced Theory: Differential Privacy
To prevent Model Inversion (extracting training data), we use **Differential Privacy (DP-SGD)**.
*   Clip Gradients (limit influence of any single example).
*   Add Noise to Gradients.
*   **Result:** You can prove mathmatically that the model output is statistically indistinguishable whether `Alice`'s data was in the training set or not.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Input Validation:** Just like you sanitize SQL inputs, you must sanitize ML inputs. Check ranges, types, and detect anomalies.
2.  **Pickle is Dead:** Use ONNX or SafeTensors for production models. Pickle allows Arbitrary Code Execution (RCE).
3.  **Rate Limiting:** To prevent Model Theft, rate limit your API. Also returning only Top-1 class instead of full probability vector makes extraction harder.

### API Summary
```python
image.requires_grad = True
loss.backward()
image + eps * image.grad.sign()
```

---

**Day 155 Complete** ✅

*Next: Day 156 - Access Control - RBAC for the AI Era.*
