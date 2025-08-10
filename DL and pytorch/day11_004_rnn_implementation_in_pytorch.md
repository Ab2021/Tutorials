# Day 11.4: RNN Implementation in PyTorch - A Noob-Friendly Comprehensive Guide

## Overview: What is an RNN and Why Should You Care?

Imagine reading a sentence. You don't read each word in isolation; you understand it based on the words that came before it. A Recurrent Neural Network (RNN) works in a similar way. It's a type of neural network designed to work with sequences of data (like text, time series, or audio) by having a "memory" of what it has seen so far.

This guide will walk you through implementing RNNs in PyTorch, step-by-step. We'll start with the absolute basics and build up to more complex and practical models. By the end, you'll not only know how to code an RNN, but you'll also understand *why* each piece of the puzzle is necessary.

**Learning Goals for Today:**

1.  Understand the two main ways to build an RNN in PyTorch: `nn.RNNCell` (the "by-hand" way) and `nn.RNN` (the "easy" way).
2.  Learn the most critical skill for real-world RNNs: how to handle sequences that have different lengths.
3.  Build more powerful RNNs: Stacked (deeper) and Bidirectional (smarter context).
4.  Put it all together in a complete, practical training pipeline.

---

## 1. The Two Flavors of PyTorch RNNs: `RNNCell` vs. `RNN`

PyTorch gives you two tools to build RNNs. Think of it like building with LEGOs:

*   `nn.RNNCell`: This is like having a single LEGO brick. It's one "step" of an RNN. You have to manually create the loop to connect the bricks. This is great for learning and for custom, fine-grained control.
*   `nn.RNN`: This is like a pre-built LEGO wall. It contains the entire loop logic. You give it the whole sequence, and it does all the work for you. It's much faster and more convenient for most standard tasks.

### 1.1. Building with a Single Brick: `nn.RNNCell`

Let's see how to build the RNN "by hand" to understand what's happening under the hood.

```python
import torch
import torch.nn as nn

# --- Model Definition ---
class ManualRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ManualRNN, self).__init__()
        self.hidden_size = hidden_size
        # The "single brick" or one step of the RNN
        self.rnn_cell = nn.RNNCell(input_size, hidden_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # We need to initialize the RNN's "memory" (the hidden state) to zeros.
        # The shape is (batch_size, hidden_size) because we process the whole batch at once.
        h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        outputs = []
        # This is our manual loop through the sequence (e.g., through each word in a sentence)
        for t in range(seq_len):
            # Get the input for the current time step for all items in the batch
            x_t = x[:, t, :]
            # Update the hidden state (the "memory") using the current input
            h_t = self.rnn_cell(x_t, h_t)
            # Store the output of this step
            outputs.append(h_t)
            
        # We stack the outputs from each time step to form a single tensor
        # The shape will be (batch_size, sequence_length, hidden_size)
        outputs = torch.stack(outputs, dim=1)
        return outputs, h_t

# --- Usage Example ---
input_size = 10    # e.g., a vector of size 10 for each word
hidden_size = 20   # The size of the RNN's memory
seq_length = 5     # e.g., a sentence with 5 words
batch_size = 3     # We process 3 sentences at a time

model = ManualRNN(input_size, hidden_size)
input_tensor = torch.randn(batch_size, seq_length, input_size)
output, final_hidden_state = model(input_tensor)

print("--- ManualRNN using RNNCell ---")
print("Input shape:", input_tensor.shape)
print("Output shape (all hidden states):", output.shape)
print("Final hidden state shape:", final_hidden_state.shape)
```

### 1.2. Building with the Pre-Built Wall: `nn.RNN`

Now, let's do the same thing the easy way.

```python
# --- Model Definition ---
class StandardRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StandardRNN, self).__init__()
        # The "pre-built wall". It will handle the loop for us.
        # batch_first=True is VERY important. It makes the input shape
        # (batch, seq, feature), which is more intuitive.
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        # A standard fully-connected layer to map the RNN's output to our desired output size
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # We don't need to write a loop!
        # We can optionally provide an initial hidden state, but if we don't,
        # nn.RNN automatically initializes it to zeros for us.
        
        # nn.RNN returns:
        # 1. output: The hidden state from *every* time step.
        #    Shape: (batch_size, seq_len, hidden_size)
        # 2. h_n: The *final* hidden state from the last time step.
        #    Shape: (num_layers, batch_size, hidden_size)
        output, h_n = self.rnn(x)
        
        # For a classification task, we often only care about the final output.
        # The shape of 'output' is (batch_size, seq_len, hidden_size),
        # so output[:, -1, :] gives us the last time step's output for every item in the batch.
        final_output = self.fc(output[:, -1, :])
        return final_output

# --- Usage Example ---
num_layers = 2     # A "stacked" RNN with 2 layers
output_size = 5    # e.g., we are classifying into 5 categories

model = StandardRNN(input_size, hidden_size, num_layers, output_size)
input_tensor = torch.randn(batch_size, seq_length, input_size)
final_output = model(input_tensor)

print("
--- StandardRNN using nn.RNN ---")
print("Input shape:", input_tensor.shape)
print("Final output shape:", final_output.shape)
```

---

## 2. The Hardest Part for Beginners: Handling Different Lengths

**The Problem:** In the real world, sentences are not all the same length. But tensors need to be rectangular! You can't have a batch where one sequence has 5 words and another has 10.

**The Solution:** We make all sequences in a batch the same length by "padding" the shorter ones with a dummy value (usually 0).

**The New Problem:** We don't want the RNN to "learn" from this fake padding. It's meaningless.

**The Real Solution:** We use a two-step process:
1.  **Padding**: Make all sequences the same length.
2.  **Packing**: Before feeding the padded batch to the RNN, we "pack" it. This tells the RNN the original lengths of the sequences, so it can ignore the padded parts during its calculations. This is a crucial optimization.

Let's see it in action.

```python
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# --- Model Definition ---
class VariableLengthRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(VariableLengthRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        # STEP 1: Pack the padded sequence.
        # enforce_sorted=False is important because we often don't want to sort our data.
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # The RNN now processes the packed sequence, ignoring padding.
        packed_output, h_n = self.rnn(packed_input)
        
        # STEP 2: Unpack the output. This returns it to a padded tensor format.
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # We need the output of the *last actual item* for each sequence.
        # For a sequence of length 5, we want the output at index 4.
        # This fancy indexing gathers the correct outputs.
        last_outputs = self.fc(output[torch.arange(len(output)), lengths - 1])
        return last_outputs

# --- Usage Example ---
model = VariableLengthRNN(input_size, hidden_size, num_layers, output_size)

# 1. Create a batch of sequences with different lengths.
seq1 = torch.randn(3, input_size) # length 3
seq2 = torch.randn(5, input_size) # length 5
seq3 = torch.randn(2, input_size) # length 2
sequences = [seq1, seq2, seq3]
lengths = torch.tensor([len(s) for s in sequences])

# 2. Pad the sequences to the length of the longest one (5).
# BEFORE PADDING: A list of tensors with shapes [3, 10], [5, 10], [2, 10]
padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
# AFTER PADDING: A single tensor of shape [3, 5, 10] (batch, max_seq_len, features)

print("
--- Handling Variable Lengths ---")
print("Original lengths:", lengths)
print("Shape after padding:", padded_sequences.shape)

final_output = model(padded_sequences, lengths)
print("Final output shape (after handling padding):", final_output.shape)
```

---

## 3. Making Your RNN More Powerful

### 3.1. Stacked RNNs: Adding Depth

A single RNN layer might not be enough to learn complex patterns. A **Stacked RNN** is just multiple RNN layers on top of each other. The output of the first layer becomes the input to the second, and so on. This allows the network to learn more abstract, hierarchical features.

You don't need to do anything special! Just set `num_layers > 1` in the `nn.RNN` module.

```python
# A 3-layer stacked RNN. It's that simple!
stacked_rnn = nn.RNN(input_size, hidden_size, num_layers=3, batch_first=True)
```

### 3.2. Bidirectional RNNs: Looking Both Ways

When you read the sentence "The apple fell from the tree", the meaning of "apple" is clear. But in "He was feeling **well**" vs. "He fell into a **well**", you need the words that come *after* "well" to understand it.

A **Bidirectional RNN** processes the sequence in both directions (left-to-right and right-to-left) and then combines the results. This gives it context from both the past and the future.

```python
# Create a bidirectional RNN by setting bidirectional=True
bidirectional_rnn = nn.RNN(input_size, hidden_size, bidirectional=True, batch_first=True)

# IMPORTANT: The output hidden size is now 2 * hidden_size, because it's the
# concatenation of the forward hidden state and the backward hidden state.
# So your next layer (e.g., nn.Linear) needs to accept 2 * hidden_size as its input dimension.
fc_layer_for_bi_rnn = nn.Linear(2 * hidden_size, output_size)
```

---

## 4. The Full Recipe: A Complete Training Pipeline

Let's put everything together to train a model that classifies names by country of origin.

```python
from torch.utils.data import Dataset, DataLoader

# --- 1. The Dataset ---
# This object just holds our data and converts it to tensors.
class NameDataset(Dataset):
    def __init__(self, names, labels, all_chars):
        self.names = names
        self.labels = labels
        self.all_chars = all_chars
        self.n_chars = len(all_chars)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        # Convert a name into a one-hot encoded tensor
        tensor = torch.zeros(len(name), self.n_chars)
        for li, letter in enumerate(name):
            tensor[li][self.all_chars.find(letter)] = 1
        
        label = self.labels[idx]
        return tensor, label

# --- 2. The Collate Function ---
# This function is the magic that takes a list of (sequence, label) pairs
# from our dataset and prepares a single, padded batch for the model.
def collate_fn(batch):
    sequences, labels = zip(*batch)
    
    # Get the original lengths before padding
    lengths = torch.tensor([len(s) for s in sequences])
    
    # Pad the sequences
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    
    # Convert labels to a tensor
    labels = torch.tensor(labels, dtype=torch.long)
    
    return padded_sequences, lengths, labels

# --- 3. Setup ---
# Sample Data (in a real project, you'd load this from files)
names = ["Schmidt", "Müller", "Schneider", "Smith", "Jones", "Williams"]
labels = [0, 0, 0, 1, 1, 1] # 0: German, 1: English
all_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
countries = ["German", "English"]

# Create the Dataset and DataLoader
dataset = NameDataset(names, labels, all_characters)
# The DataLoader creates batches and uses our collate_fn to pad them.
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# --- 4. Model, Loss, and Optimizer ---
input_size = len(all_characters)
hidden_size = 64
num_layers = 2
output_size = len(countries) # We have 2 countries

model = VariableLengthRNN(input_size, hidden_size, num_layers, output_size)
criterion = nn.CrossEntropyLoss() # Good for classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- 5. The Training Loop ---
print("
--- Starting Training Loop ---")
num_epochs = 50
for epoch in range(num_epochs):
    # The dataloader gives us one padded batch at a time
    for sequences, lengths, labels in dataloader:
        # Always do these three steps:
        # 1. Clear old gradients
        optimizer.zero_grad()
        
        # 2. Forward pass: get model predictions
        outputs = model(sequences, lengths)
        
        # 3. Calculate the loss
        loss = criterion(outputs, labels)
        
        # 4. Backward pass: calculate gradients
        loss.backward()
        
        # 5. Update the model's weights
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("--- Training Finished ---")
```

## Key Questions for Review (The "Did I Get It?" Checklist)

1.  **`RNNCell` vs. `RNN`**: If you wanted to change the hidden state in a weird, custom way at every single time step, which module would you use? (Answer: `nn.RNNCell`)
2.  **Padding**: Why do we need to pad our sequences? (Answer: Because all tensors in a batch must have the same dimensions.)
3.  **Packing**: Why do we *pack* a padded sequence? (Answer: To tell the RNN to ignore the fake padding, which makes it more efficient and prevents it from learning bad patterns.)
4.  **Bidirectional RNNs**: If you were building a model to translate sentences, why would a bidirectional RNN be a good idea? (Answer: Because the meaning of a word can depend on words that come both before and after it.)
5.  **`collate_fn`**: What is the main job of the `collate_fn` in our training pipeline? (Answer: To take a list of individual data points and bundle them into a single, padded batch ready for the model.)

## Conclusion

You've made it! We've gone from the basic building blocks of RNNs to a complete, practical implementation in PyTorch.

**Your Key Takeaways:**

*   **Use `nn.RNN`**: For most tasks, it's the best tool for the job. Use `batch_first=True` to make your life easier.
*   **Master Padding/Packing**: This is not optional; it's essential for almost any real-world sequence task. `pad_sequence` -> `pack_padded_sequence` -> `RNN` -> `pad_packed_sequence`.
*   **Choose the Right Architecture**: Start simple. If your model isn't learning, try making it deeper (stacked) or giving it more context (bidirectional).
*   **Use the `DataLoader` Pipeline**: `Dataset` -> `collate_fn` -> `DataLoader` is the standard, robust way to handle data in PyTorch.

You now have a solid foundation for building even more advanced sequence models like LSTMs and GRUs, which you'll see in the next sections.