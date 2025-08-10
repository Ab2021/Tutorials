# Day 29.2: The PyTorch Ecosystem - A Practical Tour of Essential Tools

## Introduction: Beyond the Core Library

While the core PyTorch library provides the fundamental building blocks for deep learning, a rich ecosystem of specialized libraries has been built around it to tackle specific domains and streamline complex workflows. These libraries are essential for any serious practitioner, as they provide pre-built, highly-optimized solutions for computer vision, natural language processing, graph-based learning, and more.

Leveraging these ecosystem libraries saves you from reinventing the wheel, giving you access to state-of-the-art pre-trained models, domain-specific data processing pipelines, and standardized evaluation metrics. They are the key to building powerful, production-ready models quickly and efficiently.

This guide will take you on a practical tour of the most important libraries in the PyTorch ecosystem. We will explore TorchVision for computer vision, TorchText for NLP, and PyTorch Geometric for graph neural networks, with hands-on examples for each.

**Today's Learning Objectives:**

1.  **Master TorchVision:** Learn how to use TorchVision to access standard datasets, perform common image transformations, and leverage pre-trained models for transfer learning.
2.  **Explore TorchText:** Understand the modern TorchText workflow for accessing NLP datasets and building efficient data processing pipelines for text.
3.  **Leverage PyTorch Geometric (PyG):** Get a hands-on introduction to building and training Graph Neural Networks (GNNs) using the powerful PyG library.
4.  **Apply Transfer Learning:** Implement a practical transfer learning workflow by fine-tuning a pre-trained ResNet model from TorchVision on a new dataset.

---

## Part 1: TorchVision - The Go-To Library for Computer Vision

**TorchVision** is the official computer vision library for PyTorch. It provides three main components:
*   `torchvision.datasets`: Access to popular vision datasets like CIFAR10, ImageNet, and COCO.
*   `torchvision.models`: Pre-trained models for classification, segmentation, and object detection, including ResNet, VGG, and Vision Transformer (ViT).
*   `torchvision.transforms`: A suite of common image transformations for data augmentation and preprocessing.

### 1.1. Code: Transfer Learning with a Pre-trained ResNet

Transfer learning is one of the most impactful techniques in modern CV. We will use a ResNet-18 model pre-trained on ImageNet and fine-tune it for the CIFAR10 dataset.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

print("---"" Part 1: Transfer Learning with TorchVision ---")

# --- 1. Define Transformations ---
# Pre-trained models expect a specific input format.
# We need to resize images to 224x224 and normalize them with ImageNet stats.
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 2. Load CIFAR10 Data ---
# Using torchvision.datasets to automatically download and load the data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

# --- 3. Load a Pre-trained Model ---
# `weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1` loads the pre-trained weights
model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

# --- 4. Freeze Early Layers and Replace the Final Layer ---
# Freeze all the parameters in the model
for param in model.parameters():
    param.requires_grad = False

# Get the number of input features to the original fully-connected layer
num_ftrs = model.fc.in_features

# Replace the final layer with a new one for our 10-class problem (CIFAR10)
# The new layer's parameters will have `requires_grad=True` by default
model.fc = nn.Linear(num_ftrs, 10)

# --- 5. Train the Model (only the final layer) ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

print("Starting to fine-tune the model...")
# Training loop (shortened for demonstration)
for epoch in range(1): # Run for just one epoch for speed
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i == 49: # Run for only 50 batches
            print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 50:.3f}")
            break

print("Finished Fine-tuning.")
```

---

## Part 2: TorchText - The Toolkit for Natural Language Processing

**TorchText** provides tools for preprocessing text data for use in neural networks. The modern TorchText library focuses on providing:
*   `torchtext.datasets`: Access to standard NLP datasets (e.g., IMDb, AG_NEWS).
*   **Vocab and Tokenizer integration:** Tools to build vocabularies and tokenize text, often integrating with libraries like SpaCy or SentencePiece.
*   **DataPipes:** A new, efficient, and modular way to build data processing pipelines that are compatible with PyTorch's `DataLoader`.

### 2.1. Code: A Modern TorchText Data Processing Pipeline

Let's build a pipeline for the AG_NEWS dataset that tokenizes the text, builds a vocabulary, and converts the text to numerical indices.

```python
import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

print("\n---"" Part 2: A Modern TorchText Pipeline ---")

# --- 1. Load Data and Tokenizer ---
# This creates a DataPipe
train_datapipe = AG_NEWS(split='train')
tokenizer = get_tokenizer('basic_english')

# --- 2. Build Vocabulary ---
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

# The vocabulary maps tokens to integers
vocab = build_vocab_from_iterator(yield_tokens(train_datapipe), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

print(f"Vocabulary size: {len(vocab)}")
print(f"Example mapping: 'here' -> {vocab['here']}")

# --- 3. Define the Processing Pipeline ---
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1 # Labels are 1-4, convert to 0-3

# --- 4. Create a DataLoader ---
def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    # Pad the sequences
    return torch.tensor(label_list, dtype=torch.int64), \
           torch.cat(text_list), \
           torch.tensor(offsets[:-1]).cumsum(dim=0)

# Get the first batch from the DataLoader
data_loader = torch.utils.data.DataLoader(train_datapipe, batch_size=8, shuffle=False, collate_fn=collate_batch)
batch = next(iter(data_loader))

print("\nFirst batch from DataLoader:")
print(f"Labels shape: {batch[0].shape}")
print(f"Texts shape (concatenated): {batch[1].shape}")
print(f"Offsets shape: {batch[2].shape}")
```

---

## Part 3: PyTorch Geometric (PyG) - For Graphs and Irregular Data

**PyTorch Geometric (PyG)** is the premier library for implementing Graph Neural Networks (GNNs). Graphs are a powerful way to represent data with complex relationships, like social networks, molecules, or citation networks.

PyG provides:
*   **A `Data` object:** A specialized class to efficiently represent graphs.
*   **Dozens of GNN layers:** Implementations of popular GNN layers like GCNConv, GATConv, and SAGEConv.
*   **Standard graph datasets:** Access to datasets like Cora, CiteSeer, and QM9.

### 3.1. Code: A Simple Graph Convolutional Network (GCN)

We will build a GCN to perform node classification on the Cora dataset, a citation network where nodes are papers and edges are citations.

First, install PyG: `pip install torch-geometric`

```python
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

print("\n---"" Part 3: A Simple GCN with PyG ---")

# --- 1. Load the Dataset ---
# PyG handles downloading and preprocessing
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

print("\nCora Dataset Properties:")
print(dataset)
print(f"Number of nodes: {data.num_nodes}")
print(f"Number of edges: {data.num_edges}")
print(f"Number of features: {dataset.num_node_features}")
print(f"Number of classes: {dataset.num_classes}")

# --- 2. Define the GCN Model ---
class GCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# --- 3. Training Loop ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

print("\nStarting GCN training...")
model.train()
for epoch in range(50): # Shortened for demo
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

print("Finished GCN training.")

# --- 4. Evaluate ---
model.eval()
_, pred = model(data).max(dim=1)
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')
```

## Conclusion

The PyTorch ecosystem is a vibrant and powerful collection of tools that extends the core library to handle a wide variety of domains. By using libraries like TorchVision, TorchText, and PyTorch Geometric, you can stand on the shoulders of giants, leveraging pre-trained models and optimized data pipelines to build state-of-the-art systems with a fraction of the effort.

**Key Takeaways:**

1.  **Don't Reinvent the Wheel:** For standard tasks in vision, NLP, or graph learning, the ecosystem libraries should be your first choice.
2.  **Transfer Learning is Powerful:** TorchVision makes it trivial to download and fine-tune state-of-the-art models, which is often the most effective approach for vision tasks.
3.  **Data Pipelines are Key:** TorchText and PyG provide efficient and reusable components for processing complex, non-grid-like data into a format suitable for deep learning.
4.  **Specialized Tools for Specialized Data:** GNNs and libraries like PyG are essential for unlocking insights from graph-structured data, a domain where traditional models often fail.

Familiarity with these ecosystem libraries is a hallmark of a proficient and effective PyTorch developer.

## Self-Assessment Questions

1.  **TorchVision:** When fine-tuning a pre-trained model, why is it common practice to freeze the early layers and only train the final new layer(s)?
2.  **TorchVision:** What is the purpose of normalizing the input images with specific mean and standard deviation values?
3.  **TorchText:** What is a vocabulary in the context of NLP, and why is the `<unk>` (unknown) token important?
4.  **PyG:** What are the three main attributes of a PyG `Data` object that are used in a GNN forward pass?
5.  **PyG:** In the Cora dataset, what do the node features and the labels represent?
