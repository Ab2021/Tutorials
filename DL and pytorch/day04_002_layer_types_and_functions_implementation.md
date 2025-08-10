# Day 4.2: Layer Types and Functions - A Practical Catalog

## Introduction: The Building Blocks of Networks

PyTorch's `torch.nn` package provides a vast library of pre-built, optimized layers that serve as the fundamental building blocks for almost any neural network architecture. Understanding the purpose and usage of the most common layer types is essential for translating a model diagram into working code.

This guide serves as a practical catalog of these essential layers. For each layer, we will discuss:

1.  **Its Purpose:** What kind of transformation does it perform?
2.  **Its Key Parameters:** How do you configure it?
3.  **Its Expected Input/Output Shapes:** How does it change the shape of the tensor passing through it?
4.  **A Code Example:** A concise demonstration of its usage.

**Today's Learning Objectives:**

*   **Master Core Layers:** Understand and implement `nn.Linear`, `nn.Conv2d`, and `nn.MaxPool2d`.
*   **Explore Sequence Layers:** Get a high-level overview of `nn.RNN`, `nn.LSTM`, and `nn.Embedding` for sequence data.
*   **Learn about Regularization Layers:** Understand the purpose and usage of `nn.Dropout`.
*   **Understand Normalization Layers:** See how `nn.BatchNorm2d` and `nn.LayerNorm` help stabilize training.

--- 

## Part 1: Core & Convolutional Layers

### 1.1. `nn.Linear`

*   **Purpose:** Applies a linear transformation to the incoming data: `y = xA^T + b`. This is the standard fully-connected layer, the workhorse of MLPs.
*   **Key Parameters:**
    *   `in_features`: The size of each input sample (number of features of the input tensor).
    *   `out_features`: The size of each output sample (number of neurons in the layer).
*   **Shape Transformation:**
    *   Input: `(N, *, in_features)` where `*` means any number of additional dimensions.
    *   Output: `(N, *, out_features)`.

```python
import torch
import torch.nn as nn

print("--- 1.1 nn.Linear ---")
# A batch of 32 samples, each with 128 features
input_tensor = torch.randn(32, 128)

# A linear layer with 128 input features and 64 output neurons
linear_layer = nn.Linear(in_features=128, out_features=64)

output_tensor = linear_layer(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output_tensor.shape}") # Expected: (32, 64)
```

### 1.2. `nn.Conv2d`

*   **Purpose:** The core of Convolutional Neural Networks (CNNs). It applies a 2D convolution over an input signal composed of several input planes (channels).
*   **Key Parameters:**
    *   `in_channels`: Number of channels in the input image (e.g., 3 for RGB, 1 for grayscale).
    *   `out_channels`: Number of channels produced by the convolution (number of filters).
    *   `kernel_size`: The size of the convolving kernel (e.g., 3 for a 3x3 filter, or `(3, 5)` for a non-square filter).
    *   `stride`: The step size of the convolution. Default is 1.
    *   `padding`: Zero-padding added to all four sides of the input. Default is 0.
*   **Shape Transformation:**
    *   Input: `(N, C_in, H_in, W_in)`
    *   Output: `(N, C_out, H_out, W_out)` where `H_out` and `W_out` depend on kernel size, stride, and padding.

```python
print("\n--- 1.2 nn.Conv2d ---")
# A batch of 8 RGB images, each 64x64 pixels
input_tensor = torch.randn(8, 3, 64, 64) # (N, C_in, H, W)

# A convolutional layer with 3 input channels, 16 output channels, and a 3x3 kernel.
# padding=1 ensures the output height and width remain the same as the input (for stride=1).
conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

output_tensor = conv_layer(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output_tensor.shape}") # Expected: (8, 16, 64, 64)
```

### 1.3. `nn.MaxPool2d` / `nn.AvgPool2d`

*   **Purpose:** Reduces the spatial dimensions (height and width) of the input tensor. This helps to make the representations smaller and more manageable, and it also provides a degree of translation invariance.
    *   `MaxPool2d`: Takes the maximum value from a window.
    *   `AvgPool2d`: Takes the average value.
*   **Key Parameters:**
    *   `kernel_size`: The size of the window to take a max/average over.
    *   `stride`: The step size of the window. Default is `kernel_size`.
*   **Shape Transformation:**
    *   Input: `(N, C, H_in, W_in)`
    *   Output: `(N, C, H_out, W_out)` (The number of channels remains the same).

```python
print("\n--- 1.3 nn.MaxPool2d ---")
# Output from our previous conv layer
input_tensor = torch.randn(8, 16, 64, 64)

# A max pooling layer with a 2x2 window and a stride of 2.
# This will halve the height and width.
pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)

output_tensor = pool_layer(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output_tensor.shape}") # Expected: (8, 16, 32, 32)
```

--- 

## Part 2: Layers for Sequential Data

### 2.1. `nn.Embedding`

*   **Purpose:** A lookup table that stores embeddings for a fixed dictionary and size. It's used to convert integer indices (representing words or categories) into dense vectors of a fixed size.
*   **Key Parameters:**
    *   `num_embeddings`: The size of the dictionary of embeddings (e.g., the vocabulary size).
    *   `embedding_dim`: The size of each embedding vector.
*   **Shape Transformation:**
    *   Input: `(N, seq_len)` where `N` is batch size and `seq_len` is the sequence length of integer indices.
    *   Output: `(N, seq_len, embedding_dim)`.

```python
print("\n--- 2.1 nn.Embedding ---")
# A batch of 4 sentences, each with 10 words (represented by integer indices)
# The indices must be less than the `num_embeddings`.
input_tensor = torch.randint(0, 1000, (4, 10)) # Vocab size is 1000

# An embedding layer for a vocab of size 1000, with 128-dimensional vectors
embedding_layer = nn.Embedding(num_embeddings=1000, embedding_dim=128)

output_tensor = embedding_layer(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output_tensor.shape}") # Expected: (4, 10, 128)
```

### 2.2. `nn.RNN` / `nn.LSTM` / `nn.GRU`

*   **Purpose:** These are the core layers for processing sequential data. They maintain a hidden state that is updated at each step of the sequence.
    *   `RNN`: The simplest recurrent layer.
    *   `LSTM` (Long Short-Term Memory): A more complex variant with gates that helps to mitigate the vanishing gradient problem and learn long-range dependencies.
    *   `GRU` (Gated Recurrent Unit): A simpler variant of LSTM that is often just as effective.
*   **Key Parameters:**
    *   `input_size`: The number of expected features in the input `x` (e.g., the embedding dimension).
    *   `hidden_size`: The number of features in the hidden state.
    *   `num_layers`: Number of recurrent layers to stack.
    *   `batch_first`: If `True`, the input and output tensors are provided as `(batch, seq, feature)`.
*   **Shape Transformation (with `batch_first=True`):**
    *   Input: `(N, seq_len, input_size)`
    *   Output: `(N, seq_len, hidden_size)`
    *   Hidden State: `(num_layers, N, hidden_size)`

```python
print("\n--- 2.2 nn.LSTM ---")
# Output from our previous embedding layer
input_tensor = torch.randn(4, 10, 128) # (N, seq_len, input_size)

# An LSTM layer with 128 input features and a 256-dimensional hidden state
lstm_layer = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, batch_first=True)

# The LSTM returns the output for each time step, and the final hidden and cell states
output_tensor, (hidden_state, cell_state) = lstm_layer(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output_tensor.shape}") # Expected: (4, 10, 256)
print(f"Hidden state shape: {hidden_state.shape}") # Expected: (2, 4, 256)
```

--- 

## Part 3: Regularization and Normalization Layers

### 3.1. `nn.Dropout`

*   **Purpose:** A simple but powerful regularization technique. During training, it randomly zeroes some of the elements of the input tensor with probability `p`. This helps to prevent overfitting by stopping neurons from co-adapting too much.
*   **Key Parameters:**
    *   `p`: The probability of an element to be zeroed. Default is 0.5.
*   **Important:** Dropout is **only active during training** (`model.train()` mode). It is automatically deactivated during evaluation (`model.eval()` mode).
*   **Shape Transformation:** None. The shape of the output is the same as the input.

```python
print("\n--- 3.1 nn.Dropout ---")
input_tensor = torch.randn(1, 10)

dropout_layer = nn.Dropout(p=0.5)

# --- In training mode ---
dropout_layer.train()
output_train = dropout_layer(input_tensor)

print(f"Input tensor:\n{input_tensor}")
print(f"Output in train mode (some elements are zeroed):\n{output_train}")

# --- In evaluation mode ---
dropout_layer.eval()
output_eval = dropout_layer(input_tensor)
print(f"Output in eval mode (no elements are zeroed):\n{output_eval}")
```

### 3.2. `nn.BatchNorm1d` / `nn.BatchNorm2d`

*   **Purpose:** Normalizes the input tensor to have a mean of 0 and a variance of 1. It also has learnable affine parameters (gamma and beta) that allow the network to scale and shift the normalized output. Batch Normalization helps to stabilize and accelerate training, reduce sensitivity to initialization, and acts as a slight regularizer.
*   **Key Parameters:**
    *   `num_features`: The number of features or channels of the input.
*   **Shape Transformation:** None.

```python
print("\n--- 3.2 nn.BatchNorm2d ---")
# Output from a conv layer
input_tensor = torch.randn(8, 16, 32, 32) # (N, C, H, W)

# Batch norm for a 2D (image) input with 16 channels
batchnorm_layer = nn.BatchNorm2d(num_features=16)

output_tensor = batchnorm_layer(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output_tensor.shape}") # Shape is unchanged
print(f"Mean of input (channel 0): {input_tensor[0, 0].mean():.4f}")
print(f"Mean of output (channel 0): {output_tensor[0, 0].mean():.4f}") # Should be close to 0
print(f"Std of input (channel 0): {input_tensor[0, 0].std():.4f}")
print(f"Std of output (channel 0): {output_tensor[0, 0].std():.4f}") # Should be close to 1
```

### 3.3. `nn.LayerNorm`

*   **Purpose:** Similar to Batch Norm, but it normalizes over the features of a *single data sample*, instead of across the batch. This makes its computation independent of the batch size. It is very common in Transformer architectures.
*   **Key Parameters:**
    *   `normalized_shape`: The shape of the input features to normalize.
*   **Shape Transformation:** None.

```python
print("\n--- 3.3 nn.LayerNorm ---")
# A batch of 4 sequences, each with 10 time steps and 20 features
input_tensor = torch.randn(4, 10, 20)

# Layer norm to normalize the last dimension (the 20 features)
layernorm_layer = nn.LayerNorm(normalized_shape=20)

output_tensor = layernorm_layer(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output_tensor.shape}") # Shape is unchanged
print(f"Mean of input (sample 0, step 0): {input_tensor[0, 0].mean():.4f}")
print(f"Mean of output (sample 0, step 0): {output_tensor[0, 0].mean():.4f}") # Should be close to 0
print(f"Std of input (sample 0, step 0): {input_tensor[0, 0].std():.4f}")
print(f"Std of output (sample 0, step 0): {output_tensor[0, 0].std():.4f}") # Should be close to 1
```

## Conclusion

The `torch.nn` package provides a comprehensive and modular set of tools for building neural networks. By understanding this catalog of core layers, you can now look at a diagram of a modern neural network, like a ResNet or a Transformer, and recognize the components. You have the vocabulary to translate those architectural ideas into working PyTorch code.

## Self-Assessment Questions

1.  **`nn.Linear` vs. `nn.Conv2d`:** When would you use a Linear layer versus a Conv2d layer? What is the key difference in how they process the input data?
2.  **`nn.Embedding`:** What is the purpose of an Embedding layer? What kind of data does it take as input?
3.  **`nn.Dropout`:** You are testing your trained model's performance on the validation set. Should dropout be active? How do you control this in PyTorch?
4.  **`nn.BatchNorm1d` vs. `nn.LayerNorm`:** What is the fundamental difference in how Batch Norm and Layer Norm compute their statistics?
5.  **Shape Calculation:** You have an input tensor of shape `[16, 1, 28, 28]`. You pass it through an `nn.Conv2d` layer with `out_channels=32`, `kernel_size=5`, `stride=1`, `padding=2`. What is the shape of the output tensor?

```