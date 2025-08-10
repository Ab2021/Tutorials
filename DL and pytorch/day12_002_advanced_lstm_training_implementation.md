# Day 12.2: Advanced LSTM Training - A Practical Guide

## Introduction: Beyond the Basics

Simply replacing `nn.RNN` with `nn.LSTM` is a great first step, but to truly unlock the power of LSTMs, we need to employ more advanced training techniques. Standard LSTMs can still be prone to overfitting, and their performance can be significantly enhanced by using architectural variations and regularization methods specifically designed for recurrent networks.

This guide provides a practical walkthrough of several advanced techniques for training LSTMs, including building stacked and bidirectional models, applying recurrent dropout, and using packed sequences for efficiency.

**Today's Learning Objectives:**

1.  **Build and Understand Stacked LSTMs:** See how stacking multiple LSTM layers can help the model learn more complex, hierarchical temporal features.
2.  **Implement Bidirectional LSTMs:** Learn how processing a sequence in both forward and backward directions can provide richer context and improve performance.
3.  **Apply Recurrent-Specific Regularization:** Understand and implement `Dropout` correctly within an LSTM architecture.
4.  **Master Packed Sequences:** Revisit the `pack_padded_sequence` utility to see how it can make training on variable-length sequences more computationally efficient.
5.  **Integrate into a Full Training Pipeline:** Combine all these techniques to build a robust, high-performance LSTM model for a real NLP task.

---

## Part 1: Architectural Enhancements

### 1.1. Stacked (Deep) LSTMs

*   **The Idea:** Just as we stack convolutional layers to create deep CNNs, we can stack LSTM layers to create deep RNNs. The output sequence of the first LSTM layer (i.e., the hidden states for each time step) becomes the input sequence for the second LSTM layer, and so on.
*   **Why it works:** This allows the model to learn a hierarchy of temporal features. The first layer might learn simple patterns in the sequence, while higher layers can learn more abstract, longer-term compositions of these patterns.
*   **Implementation:** This is incredibly simple in PyTorch. You just set the `num_layers` argument in the `nn.LSTM` constructor to a value greater than 1.

### 1.2. Bidirectional LSTMs

*   **The Idea:** For many tasks, like sentiment analysis, the meaning of a word depends not only on the words that came before it but also on the words that come after. A standard LSTM only looks at past context.
A **Bidirectional LSTM** consists of two separate LSTMs:
    1.  A **forward LSTM** that processes the sequence from left to right (from the first token to the last).
    2.  A **backward LSTM** that processes the sequence from right to left.
*   **How it works:** For each time step, the outputs (hidden states) of the forward and backward LSTMs are **concatenated**. This provides a representation of each token that is rich in both past and future context.
*   **Implementation:** Simply set the `bidirectional=True` argument in the `nn.LSTM` constructor.
*   **Important:** The output hidden size of a bidirectional layer will be `2 * hidden_size`.

```python
import torch
import torch.nn as nn

print("--- Part 1: Architectural Enhancements ---")

# --- Parameters ---
input_size = 50
hidden_size = 100
batch_size = 32
seq_len = 10

# --- Stacked LSTM ---
stacked_lstm = nn.LSTM(input_size, hidden_size, num_layers=4, batch_first=True)
input_seq = torch.randn(batch_size, seq_len, input_size)
output_stacked, _ = stacked_lstm(input_seq)
print(f"Stacked LSTM output shape: {output_stacked.shape}")

# --- Bidirectional LSTM ---
# The hidden size of the internal LSTMs is still 100.
# But the output size for each token will be 2 * 100 = 200.
bidirectional_lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
output_bi, (h_n_bi, c_n_bi) = bidirectional_lstm(input_seq)

print(f"\nBidirectional LSTM output shape: {output_bi.shape}")
# The final hidden state shape is (num_layers * 2, batch, hidden_size)
print(f"Bidirectional LSTM final hidden state shape: {h_n_bi.shape}")
```

---

## Part 2: Regularization with Dropout

Applying dropout to LSTMs requires care. Standard dropout applied between time steps can disrupt the recurrent connections and harm the LSTM's ability to retain long-term memory.

PyTorch's `nn.LSTM` has a built-in `dropout` parameter that implements it correctly.

*   **How it works:** The `dropout` parameter applies dropout **only to the outputs of each LSTM layer except the final layer**. It does *not* apply dropout to the recurrent hidden state connections within a layer. This regularizes the model by adding noise between the layers of a stacked LSTM, without damaging the memory flow within each layer.

```python
print("\n--- Part 2: Dropout in LSTMs ---")

# We must have num_layers > 1 for the dropout to have an effect.
lstm_with_dropout = nn.LSTM(
    input_size, 
    hidden_size, 
    num_layers=4, 
    batch_first=True, 
    dropout=0.3 # 30% dropout probability
)

# --- In training mode, dropout is active ---
lstm_with_dropout.train()
output_train, _ = lstm_with_dropout(input_seq)

# --- In eval mode, dropout is automatically deactivated ---
lstm_with_dropout.eval()
output_eval, _ = lstm_with_dropout(input_seq)

# The outputs will be different because dropout was applied during the train forward pass.
print(f"Are the train and eval outputs the same? {torch.allclose(output_train, output_eval)}")
```

---

## Part 3: A Full Pipeline for Sentiment Analysis

Let's combine these advanced techniques to build a robust sentiment classifier for the IMDB dataset.

### 3.1. Data Preparation with `torchtext`

We will use `torchtext` to load and preprocess the IMDB dataset. This involves tokenizing, building a vocabulary, and numericalizing the text.

```python
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

print("\n--- Part 3: Full Pipeline for Sentiment Analysis ---")

# --- 1. Load Data and Tokenizer ---
tokenizer = get_tokenizer('basic_english')
train_iter, test_iter = IMDB()

# --- 2. Build Vocabulary ---
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>", "<pad>"])
vocab.set_default_index(vocab["<unk>"])
pad_idx = vocab["<pad>"]
vocab_size = len(vocab)

# --- 3. Define Processing and Collate Functions ---
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: 1 if x == 2 else 0 # IMDB labels are 1 and 2

def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(len(processed_text))
    return torch.tensor(label_list, dtype=torch.int64), pad_sequence(text_list, padding_value=pad_idx, batch_first=True), torch.tensor(lengths)

# --- 4. Create DataLoaders ---
train_iter, test_iter = IMDB() # Reload iterators
train_dataloader = DataLoader(list(train_iter), batch_size=64, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(list(test_iter), batch_size=64, shuffle=False, collate_fn=collate_batch)

print("Data preparation complete.")
```

### 3.2. The Advanced LSTM Model

This model will be **stacked**, **bidirectional**, and use **dropout**. It will also correctly handle the padded sequences.

```python
class AdvancedLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=n_layers, 
            bidirectional=bidirectional, 
            dropout=dropout, 
            batch_first=True
        )
        # The input to the linear layer is hidden_dim * 2 because it's bidirectional
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        # text = [batch size, sent len]
        embedded = self.dropout(self.embedding(text))
        # embedded = [batch size, sent len, emb dim]
        
        # --- Pack sequence ---
        # This tells the LSTM to ignore the padded elements, making it more efficient.
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'), batch_first=True, enforce_sorted=False)
        
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # --- Unpack sequence is not needed here ---
        
        # Concatenate the final forward and backward hidden states
        # hidden is [num_layers * num_directions, batch, hidden_dim]
        # We take the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden states
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        # hidden = [batch size, hid dim * 2]
            
        return self.fc(hidden)

# --- Instantiate the model ---
INPUT_DIM = vocab_size
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5

model = AdvancedLSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, pad_idx)

# (A full training loop would follow here, similar to previous guides)
print("\nAdvanced LSTM model created successfully.")
print(f"The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")
```

## Conclusion

By combining architectural patterns like stacking and bidirectionality with regularization techniques like dropout, we can build powerful and robust LSTM models that achieve high performance on complex sequence tasks.

**Key Takeaways for Advanced Training:**

1.  **Go Deep with Stacking:** Use `num_layers > 1` to allow the model to learn hierarchical temporal features.
2.  **Look Both Ways with Bidirectionality:** Use `bidirectional=True` for tasks where context from both past and future is important (most NLU tasks). Remember to adjust your classifier's input size to `hidden_dim * 2`.
3.  **Regularize with Dropout:** Use the built-in `dropout` parameter in `nn.LSTM` for effective regularization between the layers of a stacked model.
4.  **Be Efficient with Packing:** For variable-length sequences, always use `pack_padded_sequence` before passing the data to your recurrent layer. This ensures the model doesn't perform useless computations on padded elements.

These techniques are the standard tools used to get state-of-the-art results from recurrent architectures.

## Self-Assessment Questions

1.  **Stacked LSTMs:** What is the main benefit of using a stacked LSTM over a single-layer LSTM?
2.  **Bidirectional LSTMs:** You are building a model to predict the next word in a sentence as you type it. Would a bidirectional LSTM be appropriate for this task? Why or why not?
3.  **Dropout in LSTMs:** Where does the `dropout` parameter in `nn.LSTM` apply the dropout? Why is this specific placement important?
4.  **Packing:** What is the main purpose of using `pack_padded_sequence`?
5.  **Model Architecture:** In our `AdvancedLSTM` model, we concatenate `hidden[-2,:,:]` and `hidden[-1,:,:]`. What do these two tensors represent?

