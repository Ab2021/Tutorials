# Day 11.1: Sequential Data Fundamentals - A Practical Guide

## Introduction: The Ubiquity of Sequences

So much of the world's data is sequential, where the order of elements is fundamental to its meaning. From the words in this sentence to the fluctuations of the stock market, from the notes in a melody to the base pairs in a DNA strand, sequences are everywhere. Processing this type of data requires a different approach than the static, order-agnostic data we might see in a table.

This guide provides a practical, foundational walkthrough of how to handle and prepare the two most common types of sequential data for deep learning models: **time series data** and **text data**.

**Today's Learning Objectives:**

1.  **Understand Time Series Data:** Learn about the components of time series data (values, timestamps, features).
2.  **Create Windowed Datasets for Forecasting:** Master the fundamental technique of converting a time series into a supervised learning problem by creating sliding windows of inputs and corresponding targets.
3.  **Handle Multivariate Time Series:** See how the windowing technique extends to datasets with multiple parallel features.
4.  **Revisit Text Data Preparation:** Solidify the concepts of tokenization, numericalization (building a vocabulary), and padding, which are the essential preprocessing steps for any NLP task.
5.  **Implement a `Dataset` for Sequential Data:** See how these preprocessing steps fit neatly into a PyTorch `Dataset` and `DataLoader` pipeline.

---

## Part 1: Time Series Data - Forecasting the Future

A time series is a sequence of data points indexed in time order. Our goal is often **forecasting**: predicting future values based on past values.

To make this a supervised learning problem, we need to restructure the data into `(input, target)` pairs. The most common way to do this is with a **sliding window**.

### 1.1. The Sliding Window Technique

Imagine a sequence of stock prices: `[10, 12, 11, 13, 15, 14, 16]`

We can decide on a `window_size` (how many past steps to use as input) and a `horizon` (how many steps into the future to predict).

Let `window_size = 3` and `horizon = 1`.

*   **Input 1:** `[10, 12, 11]` -> **Target 1:** `[13]`
*   **Input 2:** `[12, 11, 13]` -> **Target 2:** `[15]`
*   **Input 3:** `[11, 13, 15]` -> **Target 3:** `[14]`
*   ...and so on.

We have now converted our single time series into a dataset of input/output pairs that an RNN or LSTM can learn from.

### 1.2. Implementing a Sliding Window Function

```python
import torch
import numpy as np

print("--- Part 1: Time Series Data and Sliding Windows ---")

def create_sliding_windows(data, window_size, horizon=1):
    """
    Creates input/target pairs from a time series.
    Args:
        data (np.array or torch.Tensor): The time series data.
        window_size (int): The number of past time steps to use as input.
        horizon (int): The number of future time steps to predict.
    Returns:
        tuple: (X, y) a tuple of input and target tensors.
    """
    X, y = [], []
    for i in range(len(data) - window_size - horizon + 1):
        input_window = data[i : i + window_size]
        target_window = data[i + window_size : i + window_size + horizon]
        X.append(input_window)
        y.append(target_window)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# --- 1. Univariate Time Series (one feature) ---
# e.g., daily temperature readings
temperature_data = np.array([10, 12, 11, 13, 15, 14, 16, 18, 17, 19, 21, 20])
window_size = 5

X_uni, y_uni = create_sliding_windows(temperature_data, window_size)

print("--- Univariate Time Series ---")
print(f"Original data shape: {temperature_data.shape}")
print(f"Window size: {window_size}")
print(f"Input windows shape (Samples, WindowSize): {X_uni.shape}")
print(f"Target values shape (Samples, Horizon): {y_uni.shape}")
print(f"\nExample Input Window: {X_uni[0]}")
print(f"Example Target Value: {y_uni[0]}")

# --- 2. Multivariate Time Series (multiple features) ---
# e.g., daily stock data (Open, High, Low, Close)
stock_data = np.random.rand(20, 4) # 20 days, 4 features

X_multi, y_multi = create_sliding_windows(stock_data, window_size)

print("\n--- Multivariate Time Series ---")
print(f"Original data shape (Days, Features): {stock_data.shape}")
print(f"Input windows shape (Samples, WindowSize, Features): {X_multi.shape}")
# Here, we are predicting all 4 features for the next time step.
print(f"Target values shape (Samples, Horizon, Features): {y_multi.shape}")
```

**Important for RNNs:** The standard input shape for an RNN (`batch_first=True`) is `(Batch, SequenceLength, NumFeatures)`. Our sliding window function produces `(Samples, WindowSize, Features)`, which maps perfectly to this.

---

## Part 2: Text Data - The NLP Pipeline

As we saw in Day 8, preparing text data involves a standard pipeline to convert raw strings into numerical tensors.

1.  **Tokenization:** Splitting text into smaller units (tokens). This can be done by word, sub-word, or character.
2.  **Vocabulary Building:** Creating a mapping from each unique token to a unique integer index.
3.  **Numericalization:** Converting a sequence of tokens into a sequence of integers using the vocabulary.
4.  **Padding:** Since sentences in a batch have different lengths, we must pad the shorter sequences with a special `<pad>` token so that all sequences in the batch have the same length.

### 2.1. Implementing the Text Pipeline

Let's build a `TextDataset` that encapsulates this entire process.

```python
from torch.utils.data import Dataset, DataLoader
from collections import Counter

print("\n--- Part 2: Text Data Pipeline ---")

# --- 1. Sample Raw Data ---
raw_text_data = [
    "hello world",
    "this is a pytorch guide",
    "recurrent neural networks are fun",
    "a short sentence"
]

# --- 2. The Custom TextDataset ---
class TextDataset(Dataset):
    def __init__(self, text_data):
        self.raw_data = text_data
        
        # a. Tokenization
        self.tokenized_data = [text.split() for text in text_data]
        
        # b. Build Vocabulary
        # We create a vocabulary from all the tokens in our dataset.
        all_tokens = [token for sentence in self.tokenized_data for token in sentence]
        token_counts = Counter(all_tokens)
        
        # Create the mapping, reserving index 0 for padding and 1 for unknown words.
        self.vocab = {'<pad>': 0, '<unk>': 1}
        for i, (token, _) in enumerate(token_counts.items()):
            self.vocab[token] = i + 2
        
        self.vocab_size = len(self.vocab)
        self.idx_to_word = {i: w for w, i in self.vocab.items()}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        # c. Numericalization
        tokens = self.tokenized_data[idx]
        # Use .get() to default to the <unk> token if a word is not in the vocab
        indices = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        return torch.tensor(indices)

# --- 3. The Custom Collate Function for Padding ---
def collate_text_fn(batch):
    """
    Pads sequences in a batch to the same length.
    Args:
        batch: A list of tensors, where each tensor is a numericalized sentence.
    """
    # Get the length of each sequence in the batch
    lengths = torch.tensor([len(seq) for seq in batch])
    
    # Pad the sequences. `torch.nn.utils.rnn.pad_sequence` is the tool for this.
    # batch_first=True makes the output shape (Batch, MaxSeqLen)
    padded_batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
    
    # We return the padded sequences and their original lengths.
    # The lengths are useful for techniques like packed sequences.
    return padded_batch, lengths

# --- 4. Putting it all together ---
text_dataset = TextDataset(raw_text_data)

print(f"Vocabulary created with {text_dataset.vocab_size} tokens.")
print(f"Vocab: {text_dataset.vocab}")

# Create the DataLoader with our custom collate function
text_loader = DataLoader(
    dataset=text_dataset, 
    batch_size=2, 
    shuffle=True, 
    collate_fn=collate_text_fn
)

# Inspect a batch
seq_batch, len_batch = next(iter(text_loader))

print("\n--- Inspecting a batch from the DataLoader ---")
print(f"Padded sequence batch shape: {seq_batch.shape}")
print(f"Original lengths of sequences in batch: {len_batch}")
print(f"First padded sequence in batch:\n{seq_batch[0]}")
```

## Conclusion

Preparing sequential data is the critical first step in any sequence modeling task. The fundamental challenge is to convert a variable-length, often non-numeric sequence into a fixed-size format of numerical tensors that a deep learning model can process.

**Key Takeaways:**

1.  **Time Series -> Supervised Learning:** The **sliding window** technique is the standard method for converting a time series into `(input, target)` pairs suitable for forecasting.
2.  **Text -> Padded Indices:** The standard NLP pipeline involves **Tokenization -> Vocabulary Building -> Numericalization -> Padding**. This converts raw strings into batches of fixed-length integer tensors.
3.  **The `Dataset` and `DataLoader` are Universal:** The same `Dataset`/`DataLoader` paradigm we used for images works perfectly for sequential data. The key is to implement the loading and preprocessing logic in `__getitem__` and the batching logic (especially padding) in a `collate_fn`.
4.  **Shape is Key:** The final output of your data pipeline should be a tensor with a shape that your model expects, typically `(Batch, SequenceLength, NumFeatures)`.

With this solid foundation in data preparation, you are now ready to build and train the powerful RNN, LSTM, and GRU architectures that are designed to learn from this data.

## Self-Assessment Questions

1.  **Sliding Window:** You have a time series with 100 data points. If you use a `window_size` of 10 and a `horizon` of 1, how many `(input, target)` pairs will your `create_sliding_windows` function generate?
2.  **Multivariate Time Series:** For a multivariate time series, what does the "Features" dimension in the final input tensor `(Samples, WindowSize, Features)` represent?
3.  **Tokenization:** What is the difference between word-level and character-level tokenization?
4.  **Padding:** Why is padding a necessary step when creating batches of text data?
5.  **`collate_fn`:** What is the primary role of the `collate_fn` when creating a `DataLoader` for text data?

