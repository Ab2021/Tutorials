# Day 5.2: `Dataset` & `DataLoader` Comprehensive Guide - A Practical Implementation

## Introduction: From Theory to Practice

In the previous guide, we conceptually explored the architecture of PyTorch's data loading pipeline. Now, it's time to implement it. This guide will provide a detailed, hands-on walkthrough of creating custom `Dataset` classes and using the `DataLoader` to build an efficient pipeline for a real-world (though simplified) scenario.

We will work with a common use case: a custom image dataset on disk, organized into folders by class. We will write a `Dataset` that can load this structure, and then use a `DataLoader` to prepare batches for training a model.

**Today's Learning Objectives:**

1.  **Implement a Custom `ImageFolder`-like `Dataset`:** Write a `Dataset` class from scratch that can load images from a directory structure where each subdirectory is a class.
2.  **Integrate `torchvision.transforms`:** Apply transformations, including data augmentation, within your custom `Dataset`.
3.  **Master the `DataLoader`:** Configure the `DataLoader` with different parameters (`batch_size`, `shuffle`, `num_workers`) and see the effect.
4.  **Write a Custom `collate_fn`:** Learn how to write a custom collate function to handle cases where samples in a batch might have different sizes.
5.  **Visualize the Output:** Inspect the batches produced by the `DataLoader` to verify that the pipeline is working correctly.

---

## Part 1: Setting up the Custom Image Dataset

First, let's create a dummy dataset on our disk that mimics a common structure.

```
./dummy_dataset/
├── cats/
│   ├── cat1.png
│   ├── cat2.png
│   └── ...
└── dogs/
    ├── dog1.png
    ├── dog2.png
    └── ...
```

We will write a script to generate this structure and some placeholder images.

```python
import os
import torch
from PIL import Image

print("---"" Part 1: Setting up the dummy dataset ---")

# Create the main directory
root_dir = './dummy_dataset'
os.makedirs(root_dir, exist_ok=True)

# Create class subdirectories and dummy images
classes = ['cats', 'dogs']
for i, class_name in enumerate(classes):
    class_dir = os.path.join(root_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    for j in range(5): # Create 5 dummy images per class
        # Create a random tensor to represent an image
        # Assign a different color channel for each class for visualization
        img_tensor = torch.zeros(28, 28, 3, dtype=torch.uint8)
        img_tensor[:, :, i] = torch.randint(100, 255, (28, 28), dtype=torch.uint8)
        
        # Convert to PIL Image and save
        img = Image.fromarray(img_tensor.numpy(), 'RGB')
        img.save(os.path.join(class_dir, f'{class_name}{j+1}.png'))

print(f"Dummy dataset created at: {root_dir}")
```

---

## Part 2: Implementing a Custom `Dataset`

Now we will write our `CustomImageDataset` class. Its job is to scan the directory, find all the image paths, and implement `__len__` and `__getitem__`.

```python
from torch.utils.data import Dataset
from torchvision import transforms
import glob

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # --- 1. Find all image paths and their corresponding labels ---
        self.image_paths = []
        self.labels = []
        self.classes = sorted([d.name for d in os.scandir(root_dir) if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            # Using glob to find all .png files
            for img_path in glob.glob(os.path.join(class_dir, '*.png')):
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Fetches the sample at the given index.
        
        Args:
            idx (int): The index of the sample to fetch.
            
        Returns:
            tuple: (image, label) where label is the index of the class.
        """
        # 1. Load the image from disk
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        
        # 2. Get the label
        label = self.labels[idx]
        
        # 3. Apply transformations, if any
        if self.transform:
            image = self.transform(image)
            
        return image, label

# --- Let's test our custom dataset ---
print("\n---"" Part 2: Testing the CustomImageDataset ---")

# Define some transformations
# ToTensor() converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255]
# to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
data_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

custom_dataset = CustomImageDataset(root_dir='./dummy_dataset', transform=data_transform)

# 1. Test __len__
print(f"Total number of samples in the dataset: {len(custom_dataset)}")

# 2. Test __getitem__
first_image, first_label = custom_dataset[0]
print(f"Shape of the first image tensor: {first_image.shape}")
print(f"Label of the first image: {first_label} (Class: {custom_dataset.classes[first_label]})")

last_image, last_label = custom_dataset[len(custom_dataset) - 1]
print(f"Shape of the last image tensor: {last_image.shape}")
print(f"Label of the last image: {last_label} (Class: {custom_dataset.classes[last_label]})")
```

*Note: PyTorch's `torchvision.datasets.ImageFolder` does exactly this, but building it from scratch is a fantastic learning exercise.*

---

## Part 3: Using the `DataLoader`

Now that we have a `Dataset`, we can wrap it in a `DataLoader` to handle batching, shuffling, and parallel loading.

```python
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision

print("\n---"" Part 3: Using the DataLoader ---")

# --- 1. Create the DataLoader instance ---
data_loader = DataLoader(
    dataset=custom_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2 # Use 2 background processes for loading
)

# --- 2. Iterate over the DataLoader ---
# The `for` loop will now yield batches of data.
# `next(iter(data_loader))` gets the first batch.
images_batch, labels_batch = next(iter(data_loader))

print(f"Shape of the images batch: {images_batch.shape}") # (batch_size, C, H, W)
print(f"Shape of the labels batch: {labels_batch.shape}")
print(f"Labels in the batch: {labels_batch}")

# --- 3. Visualize a Batch ---
def show_images_batch(sample_batch):
    """Show image for a batch of samples."""
    images, labels = sample_batch
    # Make a grid from the batch
    grid = torchvision.utils.make_grid(images)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title(f"Batch Labels: {[custom_dataset.classes[l] for l in labels]}")
    plt.axis('off')
    plt.show()

print("\nDisplaying a sample batch from the DataLoader...")
show_images_batch((images_batch, labels_batch))
```

---

## Part 4: Custom `collate_fn` for Variable-Sized Data

What if our images were not all the same size, and we didn't want to resize them in the `Dataset`? The default `DataLoader` collate function would fail because it wouldn't be able to stack tensors of different shapes.

In this case, we need to provide our own `collate_fn` to the `DataLoader`. This function takes a list of samples (the batch) and defines how to combine them.

Let's modify our `Dataset` to not resize the images, and then write a `collate_fn` that pads the images in a batch to the same size.

```python
from torch.nn.utils.rnn import pad_sequence

print("\n---"" Part 4: Custom collate_fn ---")

# --- 1. A Dataset that returns variable-sized images ---
class VariableSizeImageDataset(Dataset):
    def __init__(self, root_dir):
        # (Initialization logic is the same as before)
        self.root_dir = root_dir
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, '*', '*.png')))
        self.labels = [0 if 'cat' in p else 1 for p in self.image_paths]
        # For simplicity, let's not resize here
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        # Let's manually resize some images to create variable sizes
        if idx % 2 == 0:
            image = image.resize((28, 28))
        else:
            image = image.resize((32, 32))
        label = self.labels[idx]
        return self.transform(image), label

# --- 2. The Custom Collate Function ---
def custom_collate_fn(batch):
    """
    Pads images to the max height and width in the batch.
    Args:
        batch: A list of tuples (image_tensor, label).
    """
    # Separate the images and labels
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Find the max height and width in the batch
    max_height = max([img.shape[1] for img in images])
    max_width = max([img.shape[2] for img in images])
    
    # Pad each image to the max size
    padded_images = []
    for img in images:
        # Calculate padding (left, right, top, bottom)
        pad_right = max_width - img.shape[2]
        pad_bottom = max_height - img.shape[1]
        # nn.functional.pad is used for padding tensors
        padded_img = torch.nn.functional.pad(img, (0, pad_right, 0, pad_bottom))
        padded_images.append(padded_img)
        
    # Stack the padded images and convert labels to a tensor
    images_tensor = torch.stack(padded_images)
    labels_tensor = torch.tensor(labels)
    
    return images_tensor, labels_tensor

# --- 3. Use the DataLoader with the custom collate_fn ---
variable_dataset = VariableSizeImageDataset(root_dir='./dummy_dataset')
variable_loader = DataLoader(
    dataset=variable_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=custom_collate_fn # Here is the key part!
)

# Inspect a batch
images_padded, labels_padded = next(iter(variable_loader))

print("DataLoader with custom collate_fn:")
print(f"Shape of the padded images batch: {images_padded.shape}")
print(f"Shape of the labels batch: {labels_padded.shape}")
print("--> All images in the batch are now padded to the same size.")
```

## Conclusion

The `Dataset` and `DataLoader` classes are the standard, powerful, and efficient way to handle data in PyTorch. By mastering them, you can build clean, readable, and high-performance input pipelines for any type of data, from images to text to audio.

**Key Takeaways:**

1.  **Subclass `nn.Dataset`:** For any custom data source, create a class that inherits from `Dataset` and implements `__len__` and `__getitem__`.
2.  **`__getitem__` is for one sample:** The logic inside `__getitem__` should be focused on loading and processing a single data point.
3.  **`DataLoader` is the orchestrator:** It wraps your `Dataset` and handles the complex work of batching, shuffling, and parallelization.
4.  **Use `num_workers`:** Always set `num_workers > 0` for a significant performance boost by loading data in the background.
5.  **Use a `collate_fn` for complex cases:** If your samples can't be automatically stacked (e.g., variable-sized sequences or images), provide a custom `collate_fn` to define the batching logic.

This robust pipeline architecture is one of PyTorch's greatest strengths, enabling you to work with massive datasets that could never fit into memory.

## Self-Assessment Questions

1.  **`__len__` vs. `__getitem__`:** What is the purpose of each of these two methods in a `Dataset` class?
2.  **Transforms:** Where is the best place to apply data augmentation transformations: when you first initialize the `Dataset`, or inside the `__getitem__` method?
3.  **`DataLoader` vs. `Dataset`:** Which class is responsible for shuffling the data?
4.  **`collate_fn`:** You have a text dataset where each sample is a sequence of word indices of a different length. Why would you need a custom `collate_fn` for this?
5.  **Performance:** You are training a model on a large image dataset. You notice that your GPU utilization is low. What two parameters of the `DataLoader` would be the first you would tune to try and fix this?

