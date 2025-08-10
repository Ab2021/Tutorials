# Day 8.1: RNN & Sequence Modeling Fundamentals - A Practical Guide

## Introduction: The Challenge of Sequential Data

Up until now, we have mostly dealt with data where the order doesn't matter (like the features of a customer or the pixels in an image, for an MLP). However, much of the world's most valuable data is **sequential**. The order is not just important; it defines the data's meaning.

*   **Text:** The order of words in a sentence determines its meaning ("dog bites man" vs. "man bites dog").
*   **Time Series:** Stock prices, weather data, or sensor readings are meaningless without their temporal order.
*   **Audio:** An audio waveform is a sequence of sound pressure levels over time.
*   **DNA:** A sequence of nucleotides.

Standard feed-forward networks (MLPs, CNNs) are not designed to handle this. They process a fixed-size input and have no memory of past inputs. **Recurrent Neural Networks (RNNs)** were designed specifically to solve this problem.

This guide will introduce the fundamental concepts of sequence modeling and build a simple RNN from the ground up to reveal its inner workings.

**Today's Learning Objectives:**

1.  **Understand Sequential Data Representation:** Learn how to convert sequences, like words in a sentence, into numerical tensors.
2.  **Grasp the Core RNN Idea: The Hidden State:** Understand the concept of a hidden state as the network's "memory" that is passed from one time step to the next.
3.  **Implement an RNN from Scratch:** Build a simple RNN using basic PyTorch components to see the recurrent loop in action.
4.  **Understand the `nn.RNN` and `nn.RNNCell` distinction:** See how PyTorch provides both a low-level, single-step cell and a high-level, full-sequence layer.
5.  **Process a Sequence:** Write a complete example that feeds a sequence of data into your custom RNN, step by step.

---

## Part 1: Representing Sequential Data

Before a model can process a sequence, we must convert it into a sequence of numerical vectors. This process is called **embedding** or **feature extraction**.

Let's take a simple sentence: "hello world"

1.  **Tokenization:** We first split the sentence into individual units, or **tokens**. In this case, the tokens are `['hello', 'world']`.

2.  **Vocabulary:** We build a vocabulary that maps each unique token to an integer index.
    *   `{'hello': 0, 'world': 1}`

3.  **Numericalization:** We replace each token with its index: `[0, 1]`.

4.  **Embedding:** We use an **embedding layer** (`nn.Embedding`) to convert these integer indices into dense vectors. These vectors are learnable and will come to represent the "meaning" of each token.

```python
import torch
import torch.nn as nn

print("---" + "-" * 30 + "Part 1: Representing Sequential Data" + "-" * 30 + "---")

# --- Sample Data ---
sentence = "hello world"
tokens = sentence.split() # ['hello', 'world']

# --- Vocabulary ---
vocab = {word: i for i, word in enumerate(tokens)}
# vocab = {'hello': 0, 'world': 1}

# --- Numericalization ---
numericalized_sentence = [vocab[word] for word in tokens]
# [0, 1]
input_indices = torch.tensor(numericalized_sentence)

# --- Embedding ---
vocab_size = len(vocab) # 2
embedding_dim = 5 # We choose to represent each word with a 5-dimensional vector

embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

# The input to the embedding layer must be a batch of sequences
# Let's add a batch dimension of 1
embedded_sentence = embedding_layer(input_indices.unsqueeze(0))

print(f"Original sentence: '{sentence}'")
print(f"Tokens: {tokens}")
print(f"Vocabulary: {vocab}")
print(f"Numericalized indices: {input_indices}")
print(f"\nShape of embedded sentence (Batch, SeqLen, EmbDim): {embedded_sentence.shape}")
print(f"Embedded 'hello':\n{embedded_sentence[0, 0]}")
print(f"Embedded 'world':\n{embedded_sentence[0, 1]}")
```

This sequence of embedded vectors is the input to our RNN.

---

## Part 2: The Core RNN Idea - A Loop with Memory

The magic of an RNN lies in its **recurrent loop**. An RNN processes a sequence one element (or **time step**) at a time. At each time step, it takes two inputs:

1.  The input for the current time step (e.g., the embedded vector for the word "hello").
2.  The **hidden state** from the *previous* time step.

The **hidden state** is a vector that acts as the network's memory. It contains a summary of all the information from the previous time steps. The network uses the current input and its past memory (the previous hidden state) to produce two outputs:

1.  An **output** for the current time step.
2.  A **new hidden state** to be passed to the *next* time step.

This can be expressed with a simple formula:

`hidden_t = tanh(W_hh * hidden_{t-1} + W_xh * input_t + b)`

Where `W_hh` and `W_xh` are learnable weight matrices.

![RNN Unrolled](https://i.imgur.com/yR222pc.png)

---

## Part 3: Implementing an RNN from Scratch

To truly understand the recurrent loop, let's build a simple RNN model using only basic PyTorch layers. This is equivalent to what `nn.RNNCell` does.

```python
import torch.nn.functional as F

print("\n" + "---" + "-" * 30 + "Part 3: Implementing an RNN from Scratch" + "-" * 30 + "---")

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # The learnable weight matrices
        # W_xh: for transforming the input
        self.W_xh = nn.Linear(input_size, hidden_size, bias=False)
        # W_hh: for transforming the previous hidden state
        self.W_hh = nn.Linear(hidden_size, hidden_size)

    def forward(self, x_t, hidden_t_minus_1):
        """
        Performs a single time step of the RNN.
        Args:
            x_t: The input for the current time step. Shape: (batch_size, input_size)
            hidden_t_minus_1: The hidden state from the previous time step. Shape: (batch_size, hidden_size)
        Returns:
            The new hidden state. Shape: (batch_size, hidden_size)
        """
        # The core RNN formula
        hidden_t = torch.tanh(
            self.W_xh(x_t) + self.W_hh(hidden_t_minus_1)
        )
        return hidden_t

# --- Let's process a sequence with our custom RNN ---

# Parameters
input_size = 50
hidden_size = 100
seq_len = 10
batch_size = 32

# Create a dummy input sequence
# (batch_size, seq_len, input_size)
input_sequence = torch.randn(batch_size, seq_len, input_size)

# Instantiate our RNN cell
rnn_cell = SimpleRNN(input_size, hidden_size)

# Initialize the first hidden state to zeros
hidden_state = torch.zeros(batch_size, hidden_size)

print(f"Processing a sequence of length {seq_len}...")

# The manual recurrent loop
outputs = []
for t in range(seq_len):
    # Get the input for the current time step for the whole batch
    input_t = input_sequence[:, t, :]
    
    # Update the hidden state
    hidden_state = rnn_cell(input_t, hidden_state)
    
    # Store the hidden state for this time step (this is often the output)
    outputs.append(hidden_state)

# Stack the outputs from all time steps
final_outputs = torch.stack(outputs, dim=1) # (batch_size, seq_len, hidden_size)

print(f"\nInput sequence shape: {input_sequence.shape}")
print(f"Final hidden state shape: {hidden_state.shape}")
print(f"Stacked outputs shape: {final_outputs.shape}")
```

---

## Part 4: Using PyTorch's Built-in `nn.RNN`

Our manual implementation is great for understanding, but in practice, you should always use PyTorch's highly optimized built-in layers.

*   `nn.RNNCell`: This is a single RNN cell, almost identical to our `SimpleRNN` class. You would still need to write the `for` loop yourself.
*   `nn.RNN`: This is the high-level layer that processes the **entire sequence** at once. It contains the loop internally and is much more efficient.

Let's replicate the previous example with `nn.RNN`.

```python
print("\n" + "---" + "-" * 30 + "Part 4: Using nn.RNN" + "-" * 30 + "---")

# --- Create the high-level RNN layer ---
# batch_first=True is crucial! It makes the input shape (batch, seq, feature),
# which is more intuitive.
rnn_layer = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)

# --- Process the sequence ---
# The nn.RNN layer takes the full sequence and an optional initial hidden state.
# If the hidden state is not provided, it defaults to zeros.

# output: contains the hidden state from *every* time step.
# final_hidden_state: contains only the *final* hidden state from the last time step.
output, final_hidden_state = rnn_layer(input_sequence)

# Note: The shape of final_hidden_state from nn.RNN is (num_layers, batch_size, hidden_size)
# For a single layer RNN, this is (1, 32, 100).

print(f"Input sequence shape: {input_sequence.shape}")
print(f"Output shape from nn.RNN (all steps): {output.shape}")
print(f"Final hidden state shape from nn.RNN: {final_hidden_state.shape}")
```

## Conclusion: The Foundation of Sequence Modeling

The Recurrent Neural Network is the foundational architecture for modeling sequential data. Its core innovation is the **hidden state**, a form of memory that allows the network to maintain context over time.

**Key Takeaways:**

1.  **Sequential Data:** Data where order matters (text, time series) requires specialized architectures.
2.  **The Recurrent Loop:** An RNN processes a sequence one step at a time, using its hidden state to pass information from one step to the next.
3.  **Hidden State as Memory:** The hidden state vector `h_t` is a compressed representation of the entire sequence seen up to time step `t`.
4.  **Parameter Sharing:** The *same* weight matrices (`W_hh` and `W_xh`) are used at every single time step. This makes the model incredibly parameter-efficient.
5.  **Use `nn.RNN`:** For practical applications, always use the high-level `nn.RNN` (or `nn.LSTM`/`nn.GRU`) layer, as it is highly optimized and handles the recurrent loop for you.

While the simple RNN we built today is powerful, it suffers from a major limitation known as the **vanishing gradient problem**, which makes it difficult to learn long-range dependencies. In the next guides, we will explore more advanced architectures like LSTMs and GRUs that were specifically designed to solve this problem.

## Self-Assessment Questions

1.  **Hidden State:** In your own words, what is the purpose of the hidden state in an RNN?
2.  **Inputs/Outputs:** At a single time step `t`, what are the two inputs to an RNN cell? What are its two outputs?
3.  **Parameter Sharing:** How does parameter sharing in an RNN differ from parameter sharing in a CNN?
4.  **`nn.RNN` vs. `nn.RNNCell`:** If you wanted to perform a custom operation on the hidden state at every single time step, which PyTorch module would give you the flexibility to do that?
5.  **Shapes:** You feed a tensor of shape `[16, 25, 128]` into an `nn.RNN` layer (with `batch_first=True`). What do the numbers 16, 25, and 128 represent? What will be the shape of the `output` tensor returned by the layer?

