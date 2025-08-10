# Day 5.3: Data Preprocessing & Augmentation Techniques - A Practical Guide

## Introduction: The Art of Data Transformation

Raw data is rarely in the perfect format for a neural network. It needs to be preprocessed into clean, numerical tensors. Furthermore, we can often significantly improve a model's performance and robustness by artificially expanding our dataset through **data augmentation**—the process of creating modified copies of our data by applying random transformations.

PyTorch, through the `torchvision.transforms` module, provides a powerful and composable library for both preprocessing and augmentation. This guide will provide a practical tour of the most common and effective transformation techniques.

**Today's Learning Objectives:**

1.  **Understand the `transforms.Compose` pipeline:** Learn how to chain multiple transformations together.
2.  **Master Essential Preprocessing Transforms:** Implement and visualize `ToTensor`, `Resize`, `CenterCrop`, and `Normalize`.
3.  **Explore Data Augmentation Techniques:** See the effects of `RandomHorizontalFlip`, `RandomResizedCrop`, `ColorJitter`, and `RandomRotation`.
4.  **Apply Transforms in a Data Pipeline:** Integrate a full augmentation pipeline into a `Dataset` and `DataLoader` workflow.
5.  **Differentiate Between Training and Validation Transforms:** Understand why we apply augmentation only to the training data.

---

## Part 1: The `transforms` Pipeline

The core of the `torchvision.transforms` module is the concept of a composable pipeline. You define a series of transformations, and then chain them together using `transforms.Compose`. The output of one transform becomes the input to the next.

```python
import torch
from torchvision import transforms
from PIL import Image
import requests
import matplotlib.pyplot as plt

print("--- Part 1: The transforms.Compose Pipeline ---")

# Let's get a sample image from the web
url = 'https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane1.png'
image = Image.open(requests.get(url, stream=True).raw)

# --- Define a Transformation Pipeline ---
# This pipeline will:
# 1. Resize the image
# 2. Convert it to a PyTorch tensor
# 3. Normalize its pixel values
preprocess_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Apply the pipeline to the image
transformed_image = preprocess_pipeline(image)

print(f"Original image type: {type(image)}")
print(f"Original image mode: {image.mode}")
print(f"Original image size: {image.size}")

print(f"\nTransformed image type: {type(transformed_image)}")
print(f"Transformed image shape: {transformed_image.shape}")
print(f"Transformed image dtype: {transformed_image.dtype}")
```

---

## Part 2: Essential Preprocessing Transforms

These are the transforms you will use in almost every computer vision project to get your data into the required format for a model.

### 2.1. `transforms.ToTensor()`

This is arguably the most important transform. It does two crucial things:
1.  It converts a PIL Image or a NumPy array with shape `(H, W, C)` and pixel values in the range `[0, 255]` into a PyTorch `FloatTensor`.
2.  It changes the dimension order to `(C, H, W)` and scales the pixel values to be in the range `[0.0, 1.0]`.

### 2.2. `transforms.Resize()` and `transforms.CenterCrop()`

*   `Resize(size)`: Resizes the input image to the given `size`. If `size` is an `int`, the smaller edge of the image will be matched to this number, keeping the aspect ratio.
*   `CenterCrop(size)`: Crops the given image at the center to the given `size`.

These are often used together to create a validation/test set transform: resize to a slightly larger size, then take a center crop.

### 2.3. `transforms.Normalize(mean, std)`

*   **Purpose:** Normalizes a tensor image with a mean and standard deviation for each channel. `output[channel] = (input[channel] - mean[channel]) / std[channel]`.
*   **Why it's important:** Most pre-trained models (like those from `torchvision.models`) were trained on ImageNet, and they expect the input images to be normalized with the specific ImageNet mean and standard deviation. Using this normalization is critical for getting good performance in transfer learning.

---

## Part 3: Data Augmentation Transforms

Data augmentation is a form of regularization. By showing the model slightly different versions of the same image, we make it more robust and less likely to overfit to the specific training examples.

**Important:** Augmentation should **only** be applied to the **training set**. The validation and test sets should remain unchanged to get a consistent and unbiased evaluation of the model's performance.

Let's visualize the effect of some common augmentation techniques.

```python
# --- Function to plot original vs. transformed images ---
def plot_transforms(transform, n_examples=5):
    fig, axs = plt.subplots(2, n_examples, figsize=(15, 6))
    for i in range(n_examples):
        # Apply the transform
        transformed_img_tensor = transform(image)
        
        # Plot original
        axs[0, i].imshow(image)
        axs[0, i].set_title("Original")
        axs[0, i].axis('off')
        
        # Plot transformed
        # We need to convert the tensor back to a displayable format
        transformed_img_display = transformed_img_tensor.permute(1, 2, 0)
        # Un-normalize if needed for better visualization (not done here for simplicity)
        axs[1, i].imshow(transformed_img_display)
        axs[1, i].set_title("Transformed")
        axs[1, i].axis('off')
    plt.suptitle(f'{type(transform).__name__}', fontsize=16)
    plt.tight_layout()
    plt.show()

print("\n--- Part 3: Visualizing Data Augmentation ---")

# --- 1. RandomHorizontalFlip ---
# Flips the image horizontally with a given probability (default p=0.5).
flip_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])
plot_transforms(flip_transform)

# --- 2. RandomResizedCrop ---
# A very common and powerful augmentation. It crops a random portion of the image
# and resizes it to a fixed size. This makes the model robust to changes in scale and position.
crop_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=(100, 100), scale=(0.5, 1.0), ratio=(0.75, 1.33)),
    transforms.ToTensor()
])
plot_transforms(crop_transform)

# --- 3. ColorJitter ---
# Randomly changes the brightness, contrast, saturation, and hue of an image.
# This makes the model robust to different lighting conditions.
jitter_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    transforms.ToTensor()
])
plot_transforms(jitter_transform)

# --- 4. RandomRotation ---
# Rotates the image by a random angle.
rotate_transform = transforms.Compose([
    transforms.RandomRotation(degrees=30),
    transforms.ToTensor()
])
plot_transforms(rotate_transform)
```

---

## Part 4: Building a Full Augmentation Pipeline

In practice, we combine multiple augmentation techniques into a single pipeline for our training data.

```python
from torch.utils.data import DataLoader

print("\n--- Part 4: Full Augmentation Pipeline ---")

# --- 1. Define separate transforms for training and validation ---
# Training transform: includes data augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation/Test transform: only includes essential preprocessing
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 2. Apply the transforms in a Dataset ---
# We will use the built-in `ImageFolder` dataset for convenience,
# as it works just like our custom one from the previous guide.
from torchvision.datasets import ImageFolder
import os

# Create a dummy validation set directory for demonstration
val_dir = './dummy_dataset_val'
os.makedirs(os.path.join(val_dir, 'cats'), exist_ok=True)
Image.fromarray(torch.randint(0, 255, (28, 28, 3), dtype=torch.uint8).numpy()).save(os.path.join(val_dir, 'cats', 'cat_val.png'))

# Create the datasets
train_dataset = ImageFolder(root='./dummy_dataset', transform=train_transform)
val_dataset = ImageFolder(root=val_dir, transform=val_transform)

# Create the DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# --- 3. Inspect the output ---
print("Inspecting a batch from the TRAINING DataLoader (with augmentation)...")
train_batch_images, train_batch_labels = next(iter(train_loader))
print(f"Train batch shape: {train_batch_images.shape}")

print("\nInspecting a batch from the VALIDATION DataLoader (no augmentation)...")
val_batch_images, val_batch_labels = next(iter(val_loader))
print(f"Validation batch shape: {val_batch_images.shape}")
```

## Conclusion

Data preprocessing and augmentation are not optional steps; they are integral to the success of a deep learning model. A well-designed transformation pipeline ensures that your model receives data in the correct format and that it learns to be robust to variations it might encounter in the real world.

**Key Takeaways:**

1.  **Compose Pipelines:** Use `transforms.Compose` to chain together a series of transformations.
2.  **Preprocess First:** Always start with the necessary preprocessing steps like resizing and converting to a tensor (`ToTensor`).
3.  **Normalize for Pre-trained Models:** If you are using a pre-trained model, you **must** normalize your data with the same mean and standard deviation that the model was originally trained with.
4.  **Augment the Training Set Only:** Apply random augmentations like flips, crops, and color jitters only to your training data to prevent overfitting.
5.  **Keep Validation/Test Sets Consistent:** The validation and test transforms should be deterministic (e.g., `Resize` and `CenterCrop`) to ensure you get a consistent, comparable evaluation of your model's performance at every epoch.

By mastering the `torchvision.transforms` library, you can significantly improve your model's accuracy and generalization capabilities.

## Self-Assessment Questions

1.  **`ToTensor`:** What are the two main things that `transforms.ToTensor()` does?
2.  **Normalization:** Why is it so important to use a specific mean and standard deviation when using a model pre-trained on ImageNet?
3.  **Augmentation Target:** Why do we only apply random augmentations to the training set and not the test set?
4.  **`RandomResizedCrop`:** What are the two main benefits of using `RandomResizedCrop` as an augmentation technique?
5.  **Pipeline Order:** In a typical training pipeline, what transform should usually come last? `ToTensor` or `Normalize`? Why?

