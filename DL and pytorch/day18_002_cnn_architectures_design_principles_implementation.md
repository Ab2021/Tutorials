# Day 18.2: CNN Architectures & Design Principles - A Practical Guide

## Introduction: From LeNet to ResNet

The history of Convolutional Neural Networks is a story of architectural innovation. Over the years, researchers have developed deeper and more sophisticated architectures that have progressively pushed the state of the art in computer vision. Understanding this evolution and the design principles behind these landmark architectures is key to designing your own effective models.

This guide provides a practical overview of the most influential CNN architectures, from the pioneering LeNet-5 to the revolutionary ResNet. We will explore the key ideas of each and implement simplified versions in PyTorch to understand their structure.

**Today's Learning Objectives:**

1.  **Understand the Classic CNN Structure (LeNet-5):** See the foundational pattern of `CONV -> POOL -> CONV -> POOL -> FC -> FC`.
2.  **Learn about Deeper Architectures (AlexNet & VGG):** Understand the principle of stacking more layers to increase model capacity and the challenges that arise.
3.  **Explore Innovative Modules (GoogLeNet/Inception):** Revisit the Inception module and its idea of parallel, multi-scale feature extraction.
4.  **Grasp the Power of Residual Learning (ResNet):** Revisit the residual block and understand how its skip connections enabled the training of truly deep networks.
5.  **Implement Simplified Versions of Landmark Models:** Build and analyze the structure of these famous architectures in PyTorch.

---

## Part 1: LeNet-5 - The Pioneer

*   **Introduced:** By Yann LeCun in 1998.
*   **Task:** Recognizing handwritten digits on checks (MNIST dataset).
*   **Architecture:** It established the classic CNN pattern that is still recognizable today.
    1.  Input Image (32x32)
    2.  `CONV` layer with 6 filters (5x5 kernel) -> `(28x28x6)`
    3.  `AVG POOL` layer (2x2, stride 2) -> `(14x14x6)`
    4.  `CONV` layer with 16 filters (5x5 kernel) -> `(10x10x16)`
    5.  `AVG POOL` layer (2x2, stride 2) -> `(5x5x16)`
    6.  `FLATTEN` -> `(400)`
    7.  `FC` layer (400 -> 120)
    8.  `FC` layer (120 -> 84)
    9.  `FC` layer (84 -> 10) -> Output logits
*   **Key Ideas:** The fundamental concept of alternating convolutional and pooling layers to create a hierarchy of features, followed by a set of fully connected layers for classification.

### 1.1. LeNet-5 Implementation

```python
import torch.nn as nn
import torch.nn.functional as F

print("---"" Part 1: LeNet-5 ---")

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # x starts as (N, 1, 32, 32)
        x = self.pool(F.relu(self.conv1(x))) # -> (N, 6, 14, 14)
        x = self.pool(F.relu(self.conv2(x))) # -> (N, 16, 5, 5)
        x = x.view(-1, 16 * 5 * 5) # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

lenet_model = LeNet5()
print(lenet_model)
```

---

## Part 2: AlexNet & VGG - Going Deeper

### 2.1. AlexNet (2012)

*   **Significance:** Won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2012 by a huge margin, kickstarting the deep learning revolution.
*   **Key Ideas:**
    1.  **Deeper than LeNet:** It was much larger (8 layers), demonstrating that scale was crucial for complex datasets like ImageNet.
    2.  **Used ReLU:** It was one of the first successful models to use the ReLU activation function, which helped to mitigate vanishing gradients and train faster.
    3.  **Used Dropout:** It heavily used dropout for regularization to combat overfitting in its large fully connected layers.
    4.  **GPU Training:** It was specifically designed to be trained on two GPUs in parallel.

### 2.2. VGGNets (2014)

*   **Significance:** Showed that you could achieve excellent performance by simply making the network even deeper in a very uniform way.
*   **Key Design Principle:** Exclusively use very small **3x3 convolutional filters**, but stack many of them together. Two stacked 3x3 conv layers have an effective receptive field of a 5x5 conv layer, but with fewer parameters and more non-linearities.
*   **Architecture (VGG-16):** A simple, elegant stack of 13 convolutional layers and 3 fully connected layers. The number of channels doubles after each max-pooling step.

### 2.3. VGG Block Implementation

```python
print("\n---"" Part 2: VGG Block ---")

def vgg_block(num_convs, in_channels, out_channels):
    """Creates a VGG block with a specified number of convolutions."""
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(True))
        in_channels = out_channels # For the next conv layer in the block
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

# Example: A VGG block with 2 conv layers, 64 input channels, 128 output channels
block_example = vgg_block(num_convs=2, in_channels=64, out_channels=128)
print(block_example)
```

---

## Part 3: GoogLeNet & ResNet - Smarter Architectures

Simply stacking layers deeper and deeper eventually hits a limit due to vanishing gradients and computational cost. The next wave of innovation came from designing smarter blocks.

### 3.1. GoogLeNet (Inception v1, 2014)

*   **Significance:** Won the 2014 ILSVRC. It focused on being computationally efficient while still being very deep.
*   **Key Idea: The Inception Module.** As we saw in Day 6, the Inception module performs convolutions with different kernel sizes (1x1, 3x3, 5x5) in parallel and concatenates the results. This allows the network to capture features at multiple scales simultaneously. It used 1x1 convolutions as bottlenecks to reduce computational cost.

### 3.2. ResNet (2015)

*   **Significance:** A revolutionary architecture that finally allowed for the successful training of networks that were hundreds or even over a thousand layers deep.
*   **Key Idea: The Residual Block.** As we also saw in Day 6, the **skip connection** allows the network to learn a residual function. By adding the input `x` to the output of a block `F(x)`, it creates a direct path for the gradient to flow backward, solving the vanishing gradient problem for deep networks.

### 3.3. ResNet Implementation Sketch

Let's build a simplified ResNet-like model using the `ResidualBlock` we defined previously.

```python
print("\n---"" Part 3: ResNet ---")

# We reuse the ResidualBlock from Day 6.3 for this sketch
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))
    def forward(self, x): return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))

class ResNet_Simplified(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_Simplified, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.linear = nn.Linear(256, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3]) # Global Average Pooling
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# Create a small ResNet-like model (e.g., ResNet-14)
resnet_model = ResNet_Simplified(ResidualBlock, [2, 2, 2])
print(resnet_model)
```

## Conclusion: Key Design Principles

The evolution from LeNet to ResNet reveals several key design principles for building effective CNNs:

1.  **Depth is Important:** Deeper models have a higher capacity to learn complex features and generally perform better, provided they can be trained effectively.

2.  **Use Small Kernels:** Stacking 3x3 convolutions is more parameter-efficient and has more non-linearities than using a single larger kernel.

3.  **Address the Gradient Flow:** For very deep networks, simply stacking layers is not enough. Architectural innovations like the skip connections in ResNet are essential for enabling the flow of gradients and successful training.

4.  **Go Beyond a Single Path:** Architectures like Inception showed that processing features at multiple scales in parallel can be highly effective.

5.  **Use Normalization and Regularization:** Techniques like Batch Normalization and Dropout are standard components of modern architectures that stabilize training and prevent overfitting.

These landmark architectures are not just historical artifacts; they are a toolbox of powerful ideas. Modern models often mix and match these principles, using residual blocks, inception-style modules, and other techniques to create even more powerful and efficient networks.

## Self-Assessment Questions

1.  **LeNet-5:** What was the fundamental architectural pattern established by LeNet-5?
2.  **VGG:** What was the main design philosophy of the VGG network?
3.  **Inception:** What problem was the Inception module designed to solve?
4.  **ResNet:** What is the single most important innovation of the ResNet architecture, and what problem does it solve?
5.  **Receptive Field:** Does a stack of two 3x3 convolutional layers have the same effective receptive field as a single 5x5 convolutional layer? Why might the stacked version be preferred?

