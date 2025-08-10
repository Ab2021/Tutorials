# Day 6.2: Convolutional Neural Networks Architecture - A Practical Guide

## Introduction: Learning from Spatial Structure

In previous sections, we saw that we could flatten an image into a 1D vector and feed it to a Multi-Layer Perceptron (MLP). However, this approach has a major flaw: it **discards the spatial structure** of the image. An MLP treats a pixel in the top-left corner the same as a pixel in the bottom-right. It has no inherent understanding of position, proximity, or the fact that pixels close to each other are semantically related.

**Convolutional Neural Networks (CNNs)** are a specialized type of neural network designed to overcome this limitation. They are the bedrock of modern computer vision, built on an architecture that explicitly leverages the spatial hierarchy of images.

This guide will provide a practical walkthrough of the core architectural components of a CNN, building a complete network from scratch in PyTorch and training it on a standard image classification task.

**Today's Learning Objectives:**

1.  **Understand the Core CNN Building Block:** See how a `Conv2d -> Activation -> Pool` sequence works together to extract features.
2.  **Learn about Parameter Sharing:** Understand why CNNs are vastly more parameter-efficient than MLPs for image data.
3.  **Grasp the Concept of a Receptive Field:** Build an intuition for how layers in a CNN learn hierarchical features.
4.  **Design a Complete CNN Architecture:** Structure a full network with multiple convolutional blocks followed by a classifier head.
5.  **Train a CNN on CIFAR-10:** Implement a full training and evaluation pipeline for a real image classification task.

---

## Part 1: The Anatomy of a CNN

A typical CNN architecture is a sequence of two main parts:

1.  **The Feature Extractor:** This part consists of a stack of **convolutional blocks**. Its job is to take the raw pixel input and transform it into a rich, high-level feature representation. As data flows through this part, its spatial dimensions (height and width) shrink, while its depth (number of channels) grows.

2.  **The Classifier:** This part is typically a standard **MLP**. It takes the flattened feature map from the end of the feature extractor and performs the final classification.

### 1.1. The Convolutional Block

The workhorse of the feature extractor is the convolutional block. A standard block consists of three stages:

1.  **Convolution (`nn.Conv2d`):** This layer applies a set of learnable filters to the input image. Each filter is specialized to detect a specific feature (like a horizontal edge, a specific color, or a corner). The output of this layer is a set of **feature maps**, where each map shows the locations where its corresponding feature was detected.

2.  **Activation (`nn.ReLU`):** A non-linear activation function is applied element-wise to the feature maps. This allows the network to learn non-linear relationships.

3.  **Pooling (`nn.MaxPool2d`):** This layer downsamples the feature maps, reducing their height and width. This has two main benefits:
    *   It reduces the number of parameters and computations in the network.
    *   It makes the feature representation more robust to small translations in the input image (translation invariance).

### 1.2. Parameter Sharing: The Superpower of CNNs

Why is a CNN so much more efficient than an MLP for images?

Imagine a 224x224 image. If you flatten this and feed it to an MLP, the first linear layer might have `224 * 224 * 3 = 150,528` input features. A hidden layer with just 512 neurons would have `150,528 * 512 = ~77 million` weights! This is computationally massive and prone to overfitting.

In a CNN, a filter (e.g., a 3x3 kernel) has only `3 * 3 = 9` parameters (plus a bias). This **same small filter is slid across the entire image**. The network learns a feature detector (like a horizontal edge detector) and then re-uses that same detector across all spatial locations. This is called **parameter sharing**, and it's the key to the efficiency and power of CNNs.

---

## Part 2: Building a CNN for CIFAR-10

Let's build a complete CNN to classify images from the CIFAR-10 dataset. CIFAR-10 consists of 32x32 color images in 10 classes.

### 2.1. Data Loading and Preprocessing

First, we set up our `DataLoader` with appropriate transforms.

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

print("---""- Part 2.1: Data Loading ---""-")

# --- Define Transforms ---
# We apply some basic data augmentation to the training set
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize for RGB channels
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# --- Load Datasets and Create DataLoaders ---
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

# Define the classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print("CIFAR-10 data loaded successfully.")
```

### 2.2. Designing the CNN Architecture

We will design a simple but effective CNN with two convolutional blocks followed by a classifier MLP.

```python
import torch.nn.functional as F

print("\n---""- Part 2.2: Designing the CNN ---""-")

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # --- Feature Extractor ---
        # Input: 32x32x3 image
        
        # Convolutional Block 1
        # Input channels = 3 (RGB), output channels = 16
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        # After conv1: 32x32x16
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # After pool1: 16x16x16
        
        # Convolutional Block 2
        # Input channels = 16, output channels = 32
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # After conv2: 16x16x32
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # After pool2: 8x8x32
        
        # --- Classifier ---
        # We need to flatten the 8x8x32 feature map into a 1D vector
        # 8 * 8 * 32 = 2048
        self.fc1 = nn.Linear(in_features=32 * 8 * 8, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=10) # 10 output classes

    def forward(self, x):
        # --- Forward pass through the Feature Extractor ---
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        
        # --- Flattening for the Classifier ---
        # x.size(0) is the batch size
        x = x.view(x.size(0), -1)
        
        # --- Forward pass through the Classifier ---
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # Raw scores (logits) for the final output
        
        return x

# --- Instantiate the model and move to GPU ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

print("Model Architecture:")
print(model)
print(f"\nModel will run on: {device}")
```

### 2.3. The Training and Evaluation Loop

This is a standard training loop. We use `CrossEntropyLoss` because this is a multi-class classification problem.

```python
import torch.optim as optim

print("\n---""- Part 2.3: Training the CNN ---""-")

# --- Loss and Optimizer ---
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Training Loop ---
num_epochs = 10 # Train for more epochs for better performance

for epoch in range(num_epochs):
    model.train() # Set the model to training mode
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # Get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # Move data to the selected device
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        # Calculate loss
        loss = loss_function(outputs, labels)
        # Backward pass
        loss.backward()
        # Optimize
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199: # Print every 200 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
            running_loss = 0.0

print('Finished Training')

# --- Evaluation ---
correct = 0
total = 0
model.eval() # Set the model to evaluation mode
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        # Calculate outputs by running images through the network
        outputs = model(images)
        # The class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'\nAccuracy of the network on the 10000 test images: {100 * correct // total} %')
```

## Conclusion: The Power of Hierarchical Feature Learning

You have now built and trained a complete Convolutional Neural Network from scratch. The architecture you implemented, while simple, follows the fundamental principles used in state-of-the-art models like ResNet, VGG, and Inception.

**Key Architectural Takeaways:**

1.  **Conv-Pool-Repeat:** The feature extractor is typically a series of Convolutional and Pooling layers. As you go deeper, the spatial dimensions (`H`, `W`) decrease while the feature depth (`C`) increases.
2.  **Hierarchical Features:** Early layers (like `conv1`) learn to detect simple features like edges and colors. Later layers (like `conv2`) combine these simple features to detect more complex patterns like textures, parts of objects, and eventually whole objects.
3.  **The Flattening Bridge:** There is always a point where the 2D feature maps are flattened into a 1D vector to be passed to the final classifier (MLP).
4.  **Parameter Sharing is Key:** CNNs are efficient because they learn a set of filters and apply them across the entire image, assuming that a feature can appear anywhere.

This fundamental understanding of CNN architecture is the foundation upon which all modern computer vision is built.

## Self-Assessment Questions

1.  **MLP vs. CNN:** What is the primary advantage of a CNN over a simple MLP for image classification tasks?
2.  **Parameter Sharing:** In your own words, what is parameter sharing in the context of a CNN?
3.  **Role of Pooling:** What are the two main purposes of a pooling layer?
4.  **Shape Transformation:** In our `SimpleCNN`, the input to `fc1` was `32 * 8 * 8`. Trace the shape of the tensor through the network to explain how we arrived at this number, starting from the input shape of `3x32x32`.
5.  **Receptive Field:** As you go deeper into a CNN, does the "receptive field" of a neuron in a feature map increase or decrease? (The receptive field is the region in the original input image that affects the value of that neuron). Why?

