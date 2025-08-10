# Day 2.3: PyTorch Ecosystem Deep Dive - A Practical Tour

## Introduction: More Than Just Tensors

While the core PyTorch library provides the fundamental tools for building neural networks (`torch`, `torch.nn`, `torch.optim`, `torch.autograd`), its true power is amplified by a rich ecosystem of domain-specific libraries. These libraries provide pre-trained models, standard datasets, and specialized data transformations that can save you hundreds of hours of development time.

This guide will take you on a practical tour of the three most important official PyTorch libraries:

1.  **`torchvision`:** For all things computer vision.
2.  **`torchaudio`:** For all things audio processing.
3.  **`torchtext`:** For all things natural language processing (NLP).

For each library, we will explore its key components and provide a hands-on code example to demonstrate its capabilities.

**Today's Learning Objectives:**

1.  **Master `torchvision`:**
    *   Load standard computer vision datasets (e.g., CIFAR-10).
    *   Apply complex image transformations and data augmentation.
    *   Use pre-trained models like ResNet-18 for transfer learning.

2.  **Explore `torchaudio`:**
    *   Load and inspect audio files.
    *   Apply audio transformations like creating Mel spectrograms.
    *   Understand the basics of preparing audio data for a deep learning model.

3.  **Discover `torchtext`:**
    *   Load standard NLP datasets (e.g., IMDB).
    *   Understand the concepts of tokenization and building a vocabulary.
    *   Prepare text data for an NLP model using iterators and data processing pipelines.

---

## Part 1: `torchvision` - Your Gateway to Computer Vision

`torchvision` is the most mature and widely used library in the PyTorch ecosystem. It provides three essential components:

*   **`torchvision.datasets`:** Access to dozens of standard datasets like MNIST, CIFAR-10, and ImageNet.
*   **`torchvision.models`:** Access to pre-trained models for tasks like image classification, object detection, and segmentation. This is the key to **transfer learning**.
*   **`torchvision.transforms`:** A collection of powerful tools for preprocessing and augmenting image data.

### 1.1. A Practical `torchvision` Example: Transfer Learning on CIFAR-10

Let's solve a classic problem: classifying images from the CIFAR-10 dataset. But instead of training a model from scratch, we will use a **pre-trained ResNet-18 model** that has already learned powerful features from the massive ImageNet dataset. We will then fine-tune this model for our specific task.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms
import time

print("---""Part 1: torchvision Demo""---")

# --- 1. Data Augmentation and Loading ---
# Define complex transformations for training data to improve model robustness
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224), # ResNet expects 224x224 images
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Standard ImageNet normalization
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transforms['train'])
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

class_names = train_dataset.classes
print(f"Dataset: CIFAR-10")
print(f"Classes: {class_names}")
print(f"Device: {device}\n")

# --- 2. Load a Pre-trained Model and Modify it ---
# Load ResNet-18, pre-trained on ImageNet
# Using `weights=models.ResNet18_Weights.DEFAULT` is the modern way
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze all the parameters in the network
for param in model.parameters():
    param.requires_grad = False

# Get the number of input features for the final layer
num_ftrs = model.fc.in_features

# Replace the final fully connected layer with a new one for our 10 classes.
# The parameters of this new layer will have `requires_grad=True` by default.
model.fc = nn.Linear(num_ftrs, len(class_names))

# Move the model to the GPU
model = model.to(device)

print("Loaded pre-trained ResNet-18 and replaced the final layer.")

# --- 3. Training and Evaluation ---
loss_function = nn.CrossEntropyLoss()
# We only want to optimize the parameters of the final layer that we just replaced.
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

# A simple function to train the model for one epoch
def train_model(model, criterion, optimizer, num_epochs=3):
    since = time.time()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        model.train() # Set model to training mode
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    return model

# Train the model
model = train_model(model, loss_function, optimizer, num_epochs=3)
```

**The Takeaway:** With `torchvision`, you can achieve high accuracy on a complex task in just a few epochs by leveraging the knowledge from a model pre-trained on a massive dataset. This technique of transfer learning is one of the most important skills in modern computer vision.

---

## Part 2: `torchaudio` - Your Toolkit for Audio AI

`torchaudio` provides tools for working with raw audio waveforms and converting them into representations that neural networks can understand, like spectrograms.

*   **`torchaudio.load`:** Loads audio files into PyTorch tensors.
*   **`torchaudio.transforms`:** A collection of audio-specific transformations.
*   **`torchaudio.datasets`:** Standard audio datasets.

### 2.1. A Practical `torchaudio` Example: From Waveform to Spectrogram

A **spectrogram** is a visual representation of the spectrum of frequencies of a signal as it varies with time. It's a common way to convert audio into an "image-like" format that can be fed into a CNN.

```python
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt

print("\n---""Part 2: torchaudio Demo""---")

# --- 1. Load an Audio File ---
# torchaudio comes with a sample audio file
yesno_data = torchaudio.datasets.YESNO('./data', download=True)
# Get the first sample: waveform, sample_rate, labels
waveform, sample_rate, _ = yesno_data[0]

print(f"Audio file loaded.")
print(f"Waveform shape: {waveform.shape}")
print(f"Sample rate: {sample_rate} Hz")

# --- 2. Create a Spectrogram ---
# A spectrogram shows frequency changes over time.
spec_transform = T.Spectrogram()
spectrogram = spec_transform(waveform)

print(f"\nSpectrogram created.")
print(f"Spectrogram shape: {spectrogram.shape}")

# --- 3. Create a Mel Spectrogram ---
# A Mel spectrogram is often better for speech tasks as it's based on human pitch perception.
mel_spec_transform = T.MelSpectrogram(sample_rate=sample_rate)
mel_spectrogram = mel_spec_transform(waveform)

print(f"\nMel Spectrogram created.")
print(f"Mel Spectrogram shape: {mel_spectrogram.shape}")

# --- 4. Visualization ---
def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(torch.log(spec).detach().numpy(), origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show()

plot_spectrogram(spectrogram[0], title='Standard Spectrogram')
plot_spectrogram(mel_spectrogram[0], title='Mel Spectrogram')
```

**The Takeaway:** `torchaudio` provides the essential tools to load raw audio and transform it into a tensor representation suitable for deep learning models. The process is often: `Waveform -> Spectrogram -> Model`.

---

## Part 3: `torchtext` - Your Toolkit for Natural Language Processing

`torchtext` helps with the common preprocessing steps required for text data.

*   **`torchtext.datasets`:** Access to standard NLP datasets like IMDB, AG_NEWS, etc.
*   **`torchtext.data.utils.get_tokenizer`:** Provides standard tokenizers.
*   **`torchtext.vocab.build_vocab_from_iterator`:** Helps build a vocabulary that maps words to integer indices.

### 3.1. A Practical `torchtext` Example: Preparing IMDB Data for Classification

Let's prepare the IMDB movie review dataset for a sentiment classification task (predicting if a review is positive or negative).

```python
import torch
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from collections import Counter

print("\n---""Part 3: torchtext Demo""---")

# --- 1. Load Data and Tokenizer ---
# A tokenizer splits a sentence into a list of words (tokens).
tokenizer = get_tokenizer('basic_english')

# Load the IMDB dataset iterators
train_iter = IMDB(split='train')

# --- 2. Build a Vocabulary ---
# A vocabulary maps each unique word to an integer index.
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

# Create the vocabulary from the training data
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
# The "<unk>" token is used for words not in our vocabulary.
vocab.set_default_index(vocab["<unk>"])

print("Vocabulary built.")
print(f"Vocabulary size: {len(vocab)}")
print(f"Index of 'movie': {vocab['movie']}")
print(f"Index of 'film': {vocab['film']}")

# --- 3. Create the Data Processing Pipeline ---
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: 1 if x == 'pos' else 0

# Example of the pipeline in action
sample_text = "This movie is fantastic!"
sample_tokens = tokenizer(sample_text)
sample_indices = text_pipeline(sample_text)
print(f"\nText processing pipeline example:")
print(f'"{sample_text}" -> {sample_tokens} -> {sample_indices}')

# --- 4. Set up DataLoader with a Collate Function ---
# The collate function will process a batch of raw text and labels into tensors.
def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
         label_list.append(label_pipeline(_label))
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

# Reload the iterator and create the DataLoader
train_iter = IMDB(split='train')
train_dataloader = DataLoader(list(train_iter), batch_size=8, shuffle=True, collate_fn=collate_batch)

# Now, `train_dataloader` can be used in a training loop.
# It will yield batches of label tensors and text tensors.
print("\nDataLoader created. Ready for training.")

# Inspect a batch
for labels, texts, offsets in train_dataloader:
    print(f"\nSample batch from DataLoader:")
    print(f"  - Labels shape: {labels.shape}")
    print(f"  - Texts shape: {texts.shape} (a long 1D tensor of all text in the batch)")
    print(f"  - Offsets shape: {offsets.shape} (tells us where each sequence starts in the text tensor)")
    break
```

**The Takeaway:** `torchtext` provides the necessary utilities to convert raw text into numerical tensors that can be fed into models like RNNs or Transformers. The process is typically: `Raw Text -> Tokenize -> Build Vocab -> Numericalize -> Batch with DataLoader`.

## Conclusion

The PyTorch ecosystem libraries are indispensable tools for any serious practitioner. They abstract away the boilerplate code for data loading and transformation, and they provide access to state-of-the-art pre-trained models.

*   **`torchvision`** is your go-to for any computer vision task, and its pre-trained models are the key to effective transfer learning.
*   **`torchaudio`** simplifies the process of getting audio data from a waveform into a tensor representation ready for a neural network.
*   **`torchtext`** provides the essential pipeline for preprocessing raw text into numerical data for NLP models.

By mastering these libraries, you can significantly accelerate your development workflow and build powerful, state-of-the-art models with a fraction of the effort.

## Self-Assessment Questions

1.  **Transfer Learning:** What does it mean to "freeze" the parameters of a pre-trained model? Why do we do this?
2.  **Data Augmentation:** Why do we apply more aggressive data transformations (like `RandomHorizontalFlip`) to the training set but not the validation/test set?
3.  **Spectrograms:** What information does a spectrogram represent? Why is it a useful representation for audio data in deep learning?
4.  **Tokenization:** What is the difference between a "tokenizer" and a "vocabulary" in NLP?
5.  **`collate_fn`:** What is the purpose of the `collate_fn` in the `torchtext` `DataLoader` example? Why is it necessary for text data?
