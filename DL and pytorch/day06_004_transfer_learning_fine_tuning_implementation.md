# Day 6.4: Transfer Learning & Fine-Tuning - A Practical Guide

## Introduction: Don't Reinvent the Wheel

Training a state-of-the-art deep learning model from scratch is a monumental task. It requires:

*   **A massive dataset:** Often millions of labeled images.
*   **Huge computational resources:** Weeks of training on multiple high-end GPUs.
*   **Significant expertise:** Careful tuning of architectures and hyperparameters.

Fortunately, we can stand on the shoulders of giants by using **Transfer Learning**. The core idea is that a model trained on a very large and general dataset (like ImageNet) has already learned a rich hierarchy of features (edges, textures, patterns, shapes) that are useful for a wide variety of other visual tasks. Instead of starting from random weights, we can start with the learned weights of a pre-trained model and adapt it to our specific, smaller dataset.

This guide provides a practical, step-by-step walkthrough of the two main transfer learning strategies: **feature extraction** and **fine-tuning**.

**Today's Learning Objectives:**

1.  **Understand the Intuition Behind Transfer Learning:** Why does a model trained on ImageNet help in classifying medical images or satellite photos?
2.  **Implement Feature Extraction:** Use a pre-trained CNN as a fixed feature extractor and train only a new, custom classifier on top.
3.  **Implement Fine-Tuning:** Go a step further by unfreezing some of the later layers of the pre-trained model and training them with a very low learning rate.
4.  **Master `torchvision.models`:** Learn how to easily load various pre-trained models like ResNet, VGG, and MobileNet.
5.  **Learn to Manage Different Learning Rates:** See how to apply different learning rates to different parts of your model, a key technique for successful fine-tuning.

---

## Part 1: Setting up the Scenario

To demonstrate transfer learning, we need a specific, smaller dataset. We will use a classic example: classifying images of **ants vs. bees**. This is a small dataset where training a deep CNN from scratch would likely lead to severe overfitting.

### 1.1. Loading the Data

We will download a pre-prepared version of the ants vs. bees dataset and set up our `DataLoaders` with appropriate transforms. Note that the transforms must match what the pre-trained model expects (e.g., 224x224 size and ImageNet normalization).

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms
import os
import zipfile
import requests

print("--- Part 1: Setting up the Data ---")

# --- Download and Unzip the data ---
url = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"
if not os.path.exists('hymenoptera_data.zip'):
    print("Downloading dataset...")
    r = requests.get(url)
    with open('hymenoptera_data.zip', 'wb') as f:
        f.write(r.content)

if not os.path.exists('hymenoptera_data'):
    print("Unzipping dataset...")
    with zipfile.ZipFile('hymenoptera_data.zip', 'r') as zip_ref:
        zip_ref.extractall('.')

data_dir = './hymenoptera_data'

# --- Define Transforms (must match ImageNet standards) ---
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- Create Datasets and DataLoaders ---
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=2)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Dataset contains {dataset_sizes['train']} training images and {dataset_sizes['val']} validation images.")
print(f"Classes: {class_names}")
print(f"Running on device: {device}")
```

---

## Part 2: Strategy 1 - Feature Extraction

In this strategy, we treat the pre-trained model as a fixed **feature extractor**. We freeze the weights of all the convolutional layers and only train the final, fully-connected classifier layer that we add on top.

This is a good strategy when your new dataset is **small and similar** to the original dataset (e.g., ImageNet).

### 2.1. Setting up the Model

```python
print("\n--- Part 2: Strategy 1 - Feature Extraction ---")

# --- Load a pre-trained ResNet-18 ---
# `weights=models.ResNet18_Weights.DEFAULT` ensures we get the latest pre-trained weights.
model_fe = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# --- Freeze all the parameters in the network ---
# We set requires_grad = False for every parameter to prevent them from being updated during training.
for param in model_fe.parameters():
    param.requires_grad = False

# --- Replace the final layer (the classifier) ---
# The original ResNet-18 has a final layer with 1000 outputs (for the 1000 ImageNet classes).
# We need to replace it with a new Linear layer that has only 2 outputs (for ants vs. bees).
num_ftrs = model_fe.fc.in_features
model_fe.fc = nn.Linear(num_ftrs, len(class_names))

# Move the model to the GPU
model_fe = model_fe.to(device)

print("Model setup for feature extraction:")
print("  - All layers frozen except the final `fc` layer.")

# --- Define Optimizer ---
# We only need to pass the parameters of the final layer to the optimizer.
# All other parameters have requires_grad=False and won't be updated.
optimizer_fe = optim.SGD(model_fe.fc.parameters(), lr=0.001, momentum=0.9)
loss_function = nn.CrossEntropyLoss()
```

### 2.2. Training and Evaluation (Helper Function)

Let's create a generic training function that we can reuse.

```python
import time
import copy

def train_model(model, criterion, optimizer, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() # Set model to training mode
            else:
                model.eval() # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model

# --- Train the feature extraction model ---
print("\nTraining the feature extractor model...")
model_fe = train_model(model_fe, loss_function, optimizer_fe, num_epochs=5)
```

---

## Part 3: Strategy 2 - Fine-Tuning

In this strategy, we go a step further. We start with the feature extraction model we just trained, but then we **unfreeze** some of the later convolutional layers and continue training the **entire model** (or parts of it) with a very low learning rate.

**The Intuition:** Early layers of a CNN learn very generic features (like edges and colors), which are useful for almost any task. Later layers learn more specific, high-level features (like "dog fur" or "car wheel"). For our ants vs. bees task, these high-level ImageNet features might not be perfectly suited. Fine-tuning allows the model to slightly adjust these later-layer features to be more relevant to our specific dataset.

This is a good strategy when your new dataset is **small but somewhat different** from the original.

### 3.1. Setting up the Model and Optimizer

```python
print("\n--- Part 3: Strategy 2 - Fine-Tuning ---")

# --- Load a new pre-trained model ---
model_ft = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))
model_ft = model_ft.to(device)

# --- Set up the optimizer with different learning rates ---
# We want to train the newly added classifier layer with a normal learning rate,
# but the pre-trained convolutional layers with a very small learning rate.
# This prevents the well-learned features from being distorted too quickly.

# Get the parameters of the convolutional base
conv_params = [p for name, p in model_ft.named_parameters() if "fc" not in name]
# Get the parameters of the classifier
fc_params = model_ft.fc.parameters()

optimizer_ft = optim.SGD([
    {'params': conv_params, 'lr': 0.0001}, # Very low learning rate for the conv layers
    {'params': fc_params, 'lr': 0.001}   # Higher learning rate for the new classifier
], momentum=0.9)

# --- Train the fine-tuning model ---
print("\nTraining the fine-tuning model...")
# Note: A common strategy is to first train as a feature extractor (as in Part 2), 
# and THEN unfreeze layers and fine-tune. For simplicity here, we fine-tune from the start.
model_ft = train_model(model_ft, loss_function, optimizer_ft, num_epochs=5)
```

## Conclusion: A Powerful Paradigm

Transfer learning is one of the most important and impactful techniques in modern deep learning. It allows individuals and organizations without access to massive datasets and computational resources to achieve state-of-the-art results on their specific problems.

**Summary of Strategies:**

1.  **Feature Extraction:**
    *   **When:** Small dataset, similar to the original (e.g., ImageNet).
    *   **How:** Freeze all convolutional layers. Train only the new classifier head.
    *   **Pros:** Fast, low computational cost, less prone to overfitting.

2.  **Fine-Tuning:**
    *   **When:** Small dataset, but different from the original.
    *   **How:** Unfreeze some of the later convolutional layers. Use a very small learning rate for the unfrozen conv layers and a larger one for the new classifier head.
    *   **Pros:** Can lead to higher accuracy by adapting features to the new task.
    *   **Cons:** Slower, requires more memory, higher risk of overfitting if not done carefully.

3.  **Training from Scratch:**
    *   **When:** You have a very large dataset and a specific task that is very different from ImageNet.
    *   **How:** Initialize a model with random weights and train the entire network.
    *   **Cons:** Requires massive data and computational power.

By mastering transfer learning, you can leverage the collective knowledge of the research community, encapsulated in pre-trained models, to build highly effective computer vision systems with remarkable efficiency.

## Self-Assessment Questions

1.  **Core Idea:** In one sentence, what is the fundamental assumption that makes transfer learning effective?
2.  **Feature Extraction vs. Fine-Tuning:** What is the key difference between the feature extraction and fine-tuning strategies in terms of which model parameters are updated?
3.  **Freezing Layers:** How do you "freeze" a layer in PyTorch? What does this mean for the optimizer?
4.  **Learning Rates:** When fine-tuning, why do we typically use a much smaller learning rate for the convolutional layers than for the new classifier head?
5.  **Use Case:** You are tasked with building a model to classify different types of plankton from microscope images. You have a dataset of 5,000 images. Which transfer learning strategy would you choose to start with, and why?
