# Day 18: Deep Learning for Vision (CNNs)
## Phase 4: ADAS & Robotics Systems | Week 3: Computer Vision & Deep Learning

---

> **📝 Day 18 Focus:**
> Traditional Computer Vision (Day 15-17) relies on hand-crafted features (corners, edges). **Deep Learning** learns these features automatically from data. Today, we dive into **Convolutional Neural Networks (CNNs)**, the technology behind modern Object Detection, Lane Keeping, and End-to-End Driving.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** the architecture of a CNN: Convolution, Pooling, ReLU, and Fully Connected layers.
2.  **Differentiate** between classic architectures: LeNet, AlexNet, VGG, and ResNet.
3.  **Implement** a custom CNN in PyTorch to classify images (CIFAR-10).
4.  **Apply** Transfer Learning to use a pre-trained ResNet for a new task.
5.  **Visualize** feature maps to understand what the network is "seeing".

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Linear Algebra:** Matrix multiplication (Dot products).
-   **Calculus:** Chain Rule (Backpropagation).
-   **Python:** Basic PyTorch understanding (Tensors).

### Hardware Requirements
-   **GPU (Optional but Recommended):** NVIDIA GPU with CUDA. (CPU works for simple models).

### Software Stack
-   **Python Libraries:** `torch`, `torchvision`, `matplotlib`, `numpy`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: From MLP to CNN

#### 1.1 The Problem with MLPs (Multi-Layer Perceptrons)
If we feed a 224x224 RGB image into a standard Neural Network:
-   Input size: $224 \times 224 \times 3 = 150,528$ neurons.
-   Hidden layer (1000 neurons): $150,528 \times 1000 \approx 150$ Million weights!
-   **Issue:** Too many parameters, overfitting, loss of spatial structure.

#### 1.2 The Convolution Operation
Instead of connecting every pixel to every neuron, we slide a small **Filter (Kernel)** (e.g., 3x3) across the image.
-   **Parameter Sharing:** The same filter detects edges everywhere in the image.
-   **Local Connectivity:** Neurons only look at local neighbors.

$$ (I * K)(i, j) = \sum_m \sum_n I(i+m, j+n) K(m, n) $$

#### 1.3 Pooling (Downsampling)
Reduces the spatial size to reduce computation and make features **Translation Invariant**.
-   **Max Pooling:** Takes the maximum value in a 2x2 window. (Keeps the strongest feature).

#### 1.4 Activation Functions
-   **ReLU (Rectified Linear Unit):** $f(x) = \max(0, x)$.
    -   Introduces non-linearity.
    -   Solves Vanishing Gradient problem.

---

### 🔹 Part 2: Famous Architectures

1.  **LeNet-5 (1998):** The grandfather. Used for digit recognition (MNIST). 2 Conv layers.
2.  **AlexNet (2012):** The breakthrough. Deep (8 layers), ReLU, Dropout, GPU training. Won ImageNet.
3.  **VGG (2014):** Very Deep (16/19 layers). Used only 3x3 filters. Simple but heavy.
4.  **ResNet (2015):** Ultra Deep (50, 101, 152 layers). Introduced **Skip Connections** (Residual Blocks) to train very deep networks without gradient vanishing.

---

### 🔹 Part 3: Transfer Learning

Training a CNN from scratch requires massive data (ImageNet: 1.2M images) and compute (weeks).
**Transfer Learning:**
1.  Take a model pre-trained on ImageNet (e.g., ResNet18).
2.  **Freeze** the early layers (Feature Extractors).
3.  **Replace** the final Fully Connected layer (Classifier) with a new one for your classes (e.g., Car vs Pedestrian).
4.  **Fine-tune** on your small dataset.

---

## 💻 Implementation: PyTorch CNN

We will implement two scripts:
1.  `train_cifar.py`: Train a simple CNN from scratch on CIFAR-10.
2.  `inference_resnet.py`: Use a pre-trained ResNet for classification.

### 🛠️ Setup
Create `week3_day18` and the files.

```bash
mkdir -p ~/ros2_ws/src/week3_day18
cd ~/ros2_ws/src/week3_day18
touch train_cifar.py inference_resnet.py
```

### 👨‍💻 Code: Training from Scratch

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 1. Define the Network
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Conv Layer 1: 3 input channels (RGB), 6 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        # Conv Layer 2: 6 input, 16 output, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Fully Connected Layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # 10 Classes

    def forward(self, x):
        # x -> Conv1 -> ReLU -> Pool
        x = self.pool(torch.relu(self.conv1(x)))
        # x -> Conv2 -> ReLU -> Pool
        x = self.pool(torch.relu(self.conv2(x)))
        # Flatten
        x = torch.flatten(x, 1)
        # FC layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train():
    # 2. Data Loading (CIFAR-10)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 3. Setup Model, Loss, Optimizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    net = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 4. Training Loop
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward + Backward + Optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print stats
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
    
    # Save Model
    torch.save(net.state_dict(), 'cifar_net.pth')

if __name__ == "__main__":
    train()
```

### 👨‍💻 Code: Inference with Pre-trained ResNet

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import sys

def predict_image(image_path):
    # 1. Load Pre-trained Model
    # weights='DEFAULT' loads the best available weights (ImageNet)
    model = models.resnet18(weights='DEFAULT')
    model.eval() # Set to evaluation mode (disable dropout/batchnorm update)

    # 2. Preprocess Image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])
    
    try:
        input_image = Image.open(image_path)
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) # Add batch dimension
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # 3. Inference
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)

    # 4. Decode Output
    # Softmax to get probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Load ImageNet labels
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    # Show top 5 categories
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    print(f"\nPredictions for {image_path}:")
    for i in range(top5_prob.size(0)):
        print(f"{categories[top5_catid[i]]}: {top5_prob[i].item()*100:.2f}%")

if __name__ == "__main__":
    # Download labels if missing
    import os
    if not os.path.exists("imagenet_classes.txt"):
        os.system("wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")
        
    if len(sys.argv) > 1:
        predict_image(sys.argv[1])
    else:
        print("Usage: python3 inference_resnet.py <path_to_image>")
```

---

## 🔬 Lab Exercise: Transfer Learning

### Lab Objectives
1.  Modify `train_cifar.py` to use `models.resnet18(weights='DEFAULT')` instead of `SimpleCNN`.
2.  Replace the final layer:
    ```python
    net.fc = nn.Linear(net.fc.in_features, 10) # CIFAR has 10 classes
    ```
3.  Train it.
    -   *Observation:* It should converge MUCH faster and achieve higher accuracy than the simple CNN, because it already knows how to detect edges and textures from ImageNet.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
**Symptom:** `RuntimeError: CUDA out of memory`.
**Cause:** Batch size too large or model too big for GPU VRAM.
**Solution:** Reduce `batch_size` (e.g., from 32 to 4).

#### 2. Dimension Mismatch
**Symptom:** `RuntimeError: mat1 and mat2 shapes cannot be multiplied`.
**Cause:** The output of the Conv/Pool layers doesn't match the input of the first Linear layer.
**Solution:** Print the shape of `x` before the Linear layer in `forward()`: `print(x.shape)`. Adjust the Linear layer input size accordingly.

#### 3. Input Normalization
**Symptom:** Model predicts garbage.
**Cause:** Input images are not normalized correctly (e.g., 0-255 instead of 0-1, or wrong mean/std).
**Solution:** Ensure `transforms.Normalize` matches what the model was trained with.

---

## ⚡ Optimization & Best Practices

### 1. Data Augmentation
To prevent overfitting, artificially expand the dataset.
```python
transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(), ...
])
```

### 2. Learning Rate Scheduler
Start with a high LR (0.01) and decay it when loss plateaus.
```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
```

### 3. Mixed Precision Training
Use `torch.cuda.amp` (Automatic Mixed Precision) to train with FP16. Reduces memory usage and speeds up training on Tensor Cores.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why do we use Pooling layers?
    *   **A:** To reduce spatial dimensions (computation) and provide translation invariance.
2.  **Q:** What is the "Vanishing Gradient" problem?
    *   **A:** In deep networks with Sigmoid/Tanh, gradients become tiny during backprop, stopping learning. ReLU fixes this.
3.  **Q:** Why does Transfer Learning work?
    *   **A:** Early layers of CNNs learn generic features (edges, blobs) that are useful for *any* visual task.

### Challenge Task
**Task:** Traffic Sign Classifier.
1.  Download the GTSRB (German Traffic Sign Recognition Benchmark) dataset.
2.  Train a ResNet to classify the 43 types of signs.
3.  Test it on images from the internet.

---

## 📚 Further Reading & References
-   [PyTorch Tutorials](https://pytorch.org/tutorials/)
-   [CS231n: Convolutional Neural Networks for Visual Recognition (Stanford)](http://cs231n.stanford.edu/) - The best deep dive.

---

**Day 18 Complete** | Phase 4: ADAS & Robotics Systems | Week 3: Computer Vision & Deep Learning
