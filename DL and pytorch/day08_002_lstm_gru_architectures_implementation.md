# Day 8.2: LSTM & GRU Architectures - A Practical Guide

## Introduction: The Problem of Long-Term Memory

The simple RNN architecture we explored previously has a major flaw: it struggles with **long-term dependencies**. Because gradients in an RNN are propagated back through every time step, they can either shrink exponentially (**vanishing gradients**) or grow exponentially (**exploding gradients**). The vanishing gradient problem is particularly severe, as it means the network is unable to learn connections between events that are far apart in a sequence.

Imagine trying to predict the last word of this sentence: "The clouds are in the sky, so I'm bringing my ______." It's easy to predict "umbrella." But for a sentence like, "I grew up in France... [many sentences later] ...so I speak fluent ______," a simple RNN would likely have forgotten the crucial context "France" by the time it needs to predict "French."

**Long Short-Term Memory (LSTM)** and **Gated Recurrent Unit (GRU)** networks were designed specifically to solve this problem using a system of learnable **gates**.

This guide will provide a practical exploration of the architecture of LSTMs and GRUs and show how to implement them in PyTorch.

**Today's Learning Objectives:**

1.  **Understand the Vanishing Gradient Problem in RNNs:** Build an intuition for why simple RNNs fail on long sequences.
2.  **Explore the LSTM Architecture:** Learn about the three gates (Forget, Input, Output) and the Cell State that give an LSTM its powerful memory capabilities.
3.  **Explore the GRU Architecture:** Learn about the GRU's simplified two-gate structure (Reset, Update) and how it compares to the LSTM.
4.  **Implement LSTMs and GRUs in PyTorch:** Use the high-level `nn.LSTM` and `nn.GRU` layers to build sequence models.
5.  **Build a Character-Level Language Model:** Apply these concepts to a classic NLP task: training a model to generate text one character at a time.

---

## Part 1: The Gated Solution - LSTM and GRU

Instead of having a single, simple recurrent connection, LSTMs and GRUs introduce **gates**. These are small neural networks (typically a linear layer followed by a sigmoid activation) that control the flow of information. The sigmoid output (between 0 and 1) acts like a switch: a value of 0 means "let nothing through," and a value of 1 means "let everything through."

### 1.1. The LSTM Architecture

The LSTM introduces a new component called the **Cell State** (`c_t`). Think of the cell state as the network's long-term memory, a conveyor belt of information. The LSTM can read from, write to, and erase information from this cell state using three gates:

1.  **Forget Gate:** Decides what information from the *previous cell state* (`c_{t-1}`) should be thrown away. It looks at the previous hidden state (`h_{t-1}`) and the current input (`x_t`).

2.  **Input Gate:** Decides what new information from the current input should be stored in the cell state. It has two parts:
    *   A sigmoid layer decides *which* values to update.
    *   A tanh layer creates a vector of *new candidate values* to be added.

3.  **Output Gate:** Decides what part of the (now updated) cell state should be used to produce the hidden state (`h_t`) for the current time step. The hidden state is a filtered version of the cell state.

![LSTM Diagram](https://i.imgur.com/2T4qj2s.png)

This structure allows the LSTM to carry important information (like the context "France") over many time steps by protecting it in the cell state, effectively mitigating the vanishing gradient problem.

### 1.2. The GRU Architecture

The Gated Recurrent Unit (GRU) is a newer and simpler alternative to the LSTM. It combines the forget and input gates into a single **Update Gate** and merges the cell state and hidden state.

1.  **Update Gate:** Decides how much of the *previous hidden state* to keep and how much of the *new candidate hidden state* to incorporate.

2.  **Reset Gate:** Decides how much of the previous hidden state to forget before computing the new candidate hidden state.

**LSTM vs. GRU:**
*   **Performance:** They perform similarly on most tasks. There is no clear winner.
*   **Simplicity:** The GRU has fewer parameters and is slightly simpler conceptually.
*   **Default Choice:** LSTM is often the default choice due to its longer history, but GRU is a very strong alternative and is computationally a bit more efficient.

---

## Part 2: Implementing LSTMs and GRUs in PyTorch

Using these advanced layers in PyTorch is just as simple as using `nn.RNN`. You simply swap `nn.RNN` for `nn.LSTM` or `nn.GRU`.

### 2.1. The `nn.LSTM` Layer

```python
import torch
import torch.nn as nn

print("--- Part 2.1: nn.LSTM ---")

# --- Parameters ---
input_size = 50
hidden_size = 100
num_layers = 2 # A stacked LSTM
batch_size = 32
seq_len = 10

# --- Dummy Input ---
input_sequence = torch.randn(batch_size, seq_len, input_size)

# --- Create the LSTM layer ---
# batch_first=True is crucial!
lstm_layer = nn.LSTM(
    input_size=input_size, 
    hidden_size=hidden_size, 
    num_layers=num_layers, 
    batch_first=True
)

# --- Process the sequence ---
# An LSTM returns three things:
# 1. output: The hidden state from the last layer for *every* time step.
# 2. final_hidden_state (h_n): The final hidden state for *all* layers.
# 3. final_cell_state (c_n): The final cell state for *all* layers.
output, (h_n, c_n) = lstm_layer(input_sequence)

print(f"Input sequence shape: {input_sequence.shape}")
print(f"Output shape (all steps, last layer): {output.shape}")
print(f"Final hidden state shape (all layers): {h_n.shape}")
print(f"Final cell state shape (all layers): {c_n.shape}")
```

### 2.2. The `nn.GRU` Layer

The interface for `nn.GRU` is almost identical, except it only returns one final state tensor since the hidden and cell states are merged.

```python
print("\n--- Part 2.2: nn.GRU ---")

# --- Create the GRU layer ---
gru_layer = nn.GRU(
    input_size=input_size, 
    hidden_size=hidden_size, 
    num_layers=num_layers, 
    batch_first=True
)

# --- Process the sequence ---
# A GRU returns two things:
# 1. output: The hidden state from the last layer for *every* time step.
# 2. final_hidden_state (h_n): The final hidden state for *all* layers.
output_gru, h_n_gru = gru_layer(input_sequence)

print(f"Input sequence shape: {input_sequence.shape}")
print(f"Output shape (all steps, last layer): {output_gru.shape}")
print(f"Final hidden state shape (all layers): {h_n_gru.shape}")
```

---

## Part 3: Application - A Character-Level Language Model

Let's build a model that learns to predict the next character in a sequence. This is a classic NLP task that demonstrates the power of LSTMs/GRUs to capture sequential patterns.

### 3.1. Data Preparation

```python
import numpy as np

print("\n--- Part 3: Character-Level Language Model ---")

# --- 1. The Data ---
text = "hello pytorch, this is a simple character-level language model."

# Create character-to-integer and integer-to-character mappings
chars = sorted(list(set(text)))
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

print(f"Our vocabulary has {vocab_size} unique characters: '{''.join(chars)}'")

# --- 2. Create Input/Target Sequences ---
# We will create sequences of a fixed length and predict the next character.
seq_length = 10
dataX = []
dataY = []
for i in range(0, len(text) - seq_length, 1):
    seq_in = text[i:i + seq_length]
    seq_out = text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])

n_patterns = len(dataX)
print(f"Total Patterns (sequences): {n_patterns}")

# Reshape X to be [samples, time steps, features]
# We use one-hot encoding for the features.
X = torch.tensor(dataX, dtype=torch.float32).reshape(n_patterns, seq_length, 1)
X = X / float(vocab_size) # Normalize
y = torch.tensor(dataY)
```

### 3.2. The Model and Training Loop

```python
class CharModel(nn.Module):
    def __init__(self, vocab_size):
        super(CharModel, self).__init__()
        # We use an LSTM layer
        self.lstm = nn.LSTM(1, 256, num_layers=2, batch_first=True)
        # The output layer predicts the score for each character in the vocab
        self.fc = nn.Linear(256, vocab_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # We only care about the output of the last time step
        last_time_step_out = lstm_out[:, -1, :]
        logits = self.fc(last_time_step_out)
        return logits

model = CharModel(vocab_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

print("\nStarting model training...")
# --- Training Loop ---
for epoch in range(200):
    optimizer.zero_grad()
    # Note: For this simple example, we train on the whole dataset at once.
    # In a real application, you would use a DataLoader.
    y_pred = model(X.to(device))
    loss = loss_fn(y_pred, y.to(device))
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# --- Generate Text ---
print("\n--- Generating Text ---" )
# Pick a random seed
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print(f"Seed: '{'' .join([int_to_char[value] for value in pattern])}'")

print("Generated Text: ", end="")
with torch.no_grad():
    for i in range(100):
        # Format input for the model
        x = torch.tensor(pattern, dtype=torch.float32).reshape(1, seq_length, 1)
        x = x / float(vocab_size)
        
        # Get prediction
        prediction = model(x.to(device))
        
        # Get the character with the highest probability
        index = torch.argmax(prediction).item()
        result = int_to_char[index]
        print(result, end="")
        
        # Update the pattern for the next prediction
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
```

## Conclusion

LSTMs and GRUs are the workhorses of modern sequence modeling. By introducing a system of learnable gates, they overcome the simple RNN's inability to handle long-term dependencies, allowing them to capture complex patterns in text, time series, and other sequential data.

**Key Takeaways:**

1.  **The Gating Mechanism:** The core innovation of LSTMs and GRUs is the use of gates (sigmoid units) to control the flow of information, deciding what to remember and what to forget.
2.  **LSTM vs. GRU:** LSTMs use a separate cell state for long-term memory and have three gates. GRUs are a simpler alternative with two gates that often perform just as well.
3.  **Ease of Use in PyTorch:** Implementing these complex architectures is trivial in PyTorch. You can simply replace `nn.RNN` with `nn.LSTM` or `nn.GRU`.
4.  **Power in Practice:** Even a simple character-level language model built with an LSTM can learn the statistical patterns of a language and generate coherent (though not always sensible) text.

With LSTMs and GRUs in your toolkit, you can now tackle a wide range of challenging sequence modeling problems.

## Self-Assessment Questions

1.  **Vanishing Gradients:** In the context of RNNs, what is the vanishing gradient problem?
2.  **LSTM Gates:** What are the names and primary functions of the three gates in an LSTM cell?
3.  **Cell State:** What is the role of the cell state in an LSTM? How does it help with long-term dependencies?
4.  **LSTM vs. GRU:** What is the main architectural difference between an LSTM and a GRU?
5.  **`nn.LSTM` Output:** When you pass a sequence to an `nn.LSTM` layer, it returns `output, (h_n, c_n)`. What is the difference between the `output` tensor and the `h_n` tensor?

