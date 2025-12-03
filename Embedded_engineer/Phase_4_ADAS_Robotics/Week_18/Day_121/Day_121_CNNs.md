# Day 121: Convolutional Neural Networks (CNNs)
## Phase 4: ADAS & Robotics Systems | Week 18: Deep Learning for Perception

---

> **📝 Day 121 Focus:**
> MLPs are bad for images (too many parameters). **CNNs** exploit the spatial structure of images. They learn features like edges, corners, and shapes. Today, we build a CNN to read traffic signs.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** the Convolution operation (Kernel, Stride, Padding).
2.  **Calculate** the output size of a Conv layer.
3.  **Implement** Pooling (MaxPool) to reduce dimensionality.
4.  **Construct** a CNN in PyTorch (Conv -> ReLU -> Pool).
5.  **Train** the CNN on the GTSRB (German Traffic Sign Recognition Benchmark) dataset.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 120:** PyTorch Basics.
-   **Image Processing:** Pixels, Channels (RGB).

### Hardware Requirements
-   **GPU:** Highly recommended for training.

### Software Stack
-   **Python:** `torchvision` (for datasets).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Convolution

Instead of connecting every pixel to every neuron, we slide a small **Kernel** (e.g., $3 \times 3$) over the image.
-   **Parameter Sharing:** The same kernel detects "vertical edges" everywhere in the image.
-   **Output Size:** $O = \frac{I - K + 2P}{S} + 1$
    -   $I$: Input Size.
    -   $K$: Kernel Size.
    -   $P$: Padding.
    -   $S$: Stride.

### 🔹 Part 2: Pooling

We need to downsample the image to make the network translation invariant.
-   **Max Pooling:** Take the max value in a $2 \times 2$ window.
-   Reduces size by half ($H/2, W/2$).
-   Keeps the strongest feature.

### 🔹 Part 3: The Architecture

Standard Pattern (LeNet/VGG style):
1.  **Feature Extractor:** `(Conv -> ReLU -> Pool) x N`
2.  **Flatten:** Convert 3D tensor to 1D vector.
3.  **Classifier:** `Linear -> ReLU -> Linear -> Softmax`.

---

## 💻 Implementation: Traffic Sign Classifier

**Scenario:**
-   Dataset: GTSRB (43 Classes: Stop, Speed Limit, Yield, etc.).
-   Input: $32 \times 32$ RGB Images.
-   Task: Classify.

### 🛠️ Setup
Create `week18_day121` and `traffic_sign_cnn.py`.

```bash
mkdir -p ~/ros2_ws/src/week18_day121
cd ~/ros2_ws/src/week18_day121
touch traffic_sign_cnn.py
```

### 👨‍💻 Code: The CNN

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Data Loading ---
def load_data():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # We use CIFAR10 as a proxy for GTSRB in this demo (easier to download)
    # In real lab, download GTSRB
    print("Downloading Dataset (CIFAR10 for demo)...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)
                                              
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers=2)
                                             
    return trainloader, testloader

# --- 2. Model Definition ---
class TrafficSignNet(nn.Module):
    def __init__(self):
        super(TrafficSignNet, self).__init__()
        
        # Input: 3 x 32 x 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) # -> 32 x 32 x 32
        self.pool = nn.MaxPool2d(2, 2) # -> 32 x 16 x 16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # -> 64 x 16 x 16
        # Pool again -> 64 x 8 x 8
        
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10) # 10 Classes
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8) # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    
    trainloader, testloader = load_data()
    net = TrafficSignNet().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    # --- Training ---
    epochs = 5
    loss_history = []
    
    print("Starting Training...")
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 200 == 199:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 200:.3f}")
                loss_history.append(running_loss / 200)
                running_loss = 0.0
                
    print("Finished Training")
    
    # --- Evaluation ---
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print(f"Accuracy on Test Set: {100 * correct / total:.2f}%")
    
    # Plot Loss
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Steps (x200)")
    plt.ylabel("Loss")
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Kernel Visualization

### Lab Objectives
1.  Run the script.
2.  **Observation:** Accuracy should be ~65-70% (Simple model on CIFAR).
3.  **Experiment:**
    -   Visualize the weights of `conv1`.
    -   `weights = net.conv1.weight.data.cpu().numpy()`
    -   Plot the 32 kernels as images.
    -   **Result:** You will see colorful patterns (edges, blobs). These are the "eyes" of the network.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Dimension Mismatch in FC Layer
**Symptom:** `RuntimeError` at `self.fc1(x)`.
**Cause:** You calculated the flattened size wrong.
**Solution:** Print `x.shape` before `view()`. If it says `[Batch, 64, 8, 8]`, then input size is `64*8*8`.

#### 2. Slow Training
**Symptom:** Takes forever.
**Cause:** Running on CPU.
**Solution:** Enable GPU. If no GPU, reduce batch size or image size.

---

## ⚡ Optimization & Best Practices

### 1. Data Augmentation
To prevent overfitting and improve generalization.
```python
transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    ...
])
```
-   This teaches the network that a rotated car is still a car.

### 2. Batch Normalization
Add `nn.BatchNorm2d(32)` after Conv layers.
-   Normalizes the activations.
-   Allows higher learning rates and faster convergence.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why do we use Padding?
    *   **A:** To preserve the spatial dimensions (Input 32x32 -> Output 32x32). Without padding, the image shrinks with every layer.
2.  **Q:** What does Stride do?
    *   **A:** It controls the step size. Stride 2 downsamples the image by half (similar to pooling).
3.  **Q:** How many parameters in a $3 \times 3$ kernel with 3 input channels and 32 output channels?
    *   **A:** $(3 \times 3 \times 3 + 1) \times 32 = 28 \times 32 = 896$. (The +1 is bias).

### Challenge Task
**Task:** Deep Network.
1.  Add more layers: `Conv -> Pool -> Conv -> Pool -> Conv -> Pool`.
2.  Final size will be $4 \times 4$.
3.  Observe if accuracy improves. (Deeper is usually better).

---

## 📚 Further Reading & References
-   [CS231n: CNNs for Visual Recognition](http://cs231n.stanford.edu/)
-   [Visualizing CNNs](https://distill.pub/2017/feature-visualization/)

---

**Day 121 Complete** | Phase 4: ADAS & Robotics Systems | Week 18: Deep Learning for Perception
