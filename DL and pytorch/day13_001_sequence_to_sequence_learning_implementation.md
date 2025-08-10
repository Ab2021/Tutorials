# Day 13.1: Sequence-to-Sequence (Seq2Seq) Learning - A Practical Guide

## Introduction: When Both Input and Output are Sequences

So far, our sequence models have performed tasks like classification (sequence in, single label out) or forecasting (sequence in, single value out). But what about tasks where the **output is also a variable-length sequence**? 

*   **Machine Translation:** Input is a sentence in French, output is a sentence in English.
*   **Summarization:** Input is a long document, output is a short summary.
*   **Conversational AI:** Input is a user's question, output is a chatbot's answer.

These tasks require a more sophisticated architecture known as **Sequence-to-Sequence (Seq2Seq)**. A Seq2Seq model is composed of two main components: an **Encoder** and a **Decoder**.

This guide will provide a practical, from-scratch implementation of a simple Seq2Seq model to explain the core concepts of the encoder-decoder architecture.

**Today's Learning Objectives:**

1.  **Understand the Encoder-Decoder Architecture:** Grasp the distinct roles of the encoder (to understand the input) and the decoder (to generate the output).
2.  **Learn about the Context Vector:** Understand how the encoder's final hidden state acts as a thought vector, summarizing the entire input sequence.
3.  **Implement a Seq2Seq Model from Scratch:** Build a complete encoder-decoder model using PyTorch's recurrent layers (`nn.GRU` in this case).
4.  **Grasp the "Teacher Forcing" Technique:** Learn about this crucial training technique for efficiently training decoder networks.
5.  **See the Full Picture:** Walk through a simple, end-to-end example of training a Seq2Seq model on a toy translation task.

---

## Part 1: The Encoder-Decoder Architecture

A Seq2Seq model consists of two RNNs (which can be simple RNNs, LSTMs, or GRUs).

### 1. The Encoder
*   **Purpose:** To process the entire input sequence and compress all its information into a single, fixed-size vector. This vector is called the **context vector** (or "thought vector").
*   **How it works:** It reads the input sequence one token at a time. It doesn't produce any output at each step; its only job is to compute the final hidden state. This final hidden state is the context vector.

### 2. The Decoder
*   **Purpose:** To take the context vector from the encoder and generate the output sequence, one token at a time.
*   **How it works:**
    1.  It is initialized with the encoder's final hidden state (the context vector).
    2.  For the first step, it takes a special `<sos>` (start of sequence) token as input.
    3.  It produces an output (a prediction for the first token of the target sequence) and a new hidden state.
    4.  For the next step, it takes its *own* previous output token as the *current* input, along with its previous hidden state.
    5.  This process repeats until the decoder generates a special `<eos>` (end of sequence) token.

![Seq2Seq Architecture](https://i.imgur.com/t1z5Y2d.png)

**The Bottleneck:** The entire meaning of the input sequence must be compressed into the single context vector. This can be a bottleneck for very long sequences, a problem that the **Attention Mechanism** (which we saw in Day 8) was invented to solve. For now, we will focus on this simpler, non-attention-based architecture.

---

## Part 2: Implementing a Seq2Seq Model

Let's build a simple Seq2Seq model for a toy translation task.

### 2.1. The Encoder Module

The encoder is a standard GRU that takes an input sequence and returns only the final hidden state.

```python
import torch
import torch.nn as nn

print("---", "Part 2: Implementing a Seq2Seq Model", "---")

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src = [batch size, src len]
        embedded = self.dropout(self.embedding(src))
        # embedded = [batch size, src len, emb dim]
        
        # The encoder returns the output of every time step and the final hidden state.
        # We only need the final hidden state.
        outputs, hidden = self.rnn(embedded)
        
        # hidden = [n layers, batch size, hid dim]
        return hidden
```

### 2.2. The Decoder Module

The decoder is also a GRU, but it processes the sequence one step at a time.

```python
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden):
        # input = [batch size] (the token from the previous step)
        # hidden = [n layers, batch size, hid dim] (the context from the encoder)
        
        # Add a sequence length dimension to the input
        input = input.unsqueeze(1) # [batch size, 1]
        
        embedded = self.dropout(self.embedding(input))
        # embedded = [batch size, 1, emb dim]
        
        # The GRU takes the embedded input and the previous hidden state
        output, hidden = self.rnn(embedded, hidden)
        
        # output = [batch size, seq len, hid dim] -> [batch size, 1, hid dim]
        # hidden = [n layers, batch size, hid dim]
        
        # Pass the output through the final linear layer to get a prediction
        prediction = self.fc_out(output.squeeze(1))
        # prediction = [batch size, output dim]
        
        return prediction, hidden
```

### 2.3. The Full Seq2Seq Model

Now we combine the Encoder and Decoder. The training process for a decoder is tricky. If we let the decoder feed its own (potentially wrong) predictions back to itself during training, errors can compound, and the model may fail to learn. 

The solution is **Teacher Forcing**. During training, instead of feeding the decoder's own prediction as the next input, we feed the **actual ground-truth token** from the target sequence. This stabilizes training and helps the model learn much faster.

```python
import random

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [batch size, src len]
        # trg = [batch size, trg len]
        
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # ---
        # Encoder Pass
        # ---
        # The final hidden state of the encoder is used as the initial hidden state of the decoder
        hidden = self.encoder(src)
        
        # ---
        # Decoder Pass
        # ---
        # The first input to the decoder is the <sos> token
        input = trg[:, 0]
        
        # This loop is the unrolling of the decoder
        for t in range(1, trg_len):
            # Run one step of the decoder
            output, hidden = self.decoder(input, hidden)
            
            # Store the prediction
            outputs[:, t] = output
            
            # Decide whether to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            # Get the highest predicted token from our predictions
            top1 = output.argmax(1)
            
            # If teacher forcing, use actual next token as next input
            # If not, use predicted token
            input = trg[:, t] if teacher_force else top1
            
        return outputs
```

---

## Part 3: A Toy Training Example

Let's put all the pieces together for a simple demonstration.

```python
import random

print("\n---", "Part 3: A Toy Training Example", "---")

# ---
# 1. Parameters and Model Instantiation
# ---
INPUT_DIM = 100   # Source vocabulary size
OUTPUT_DIM = 120  # Target vocabulary size
ENC_EMB_DIM = 32
DEC_EMB_DIM = 32
HID_DIM = 64
N_LAYERS = 1
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
model = Seq2Seq(enc, dec, device).to(device)

optimizer = optim.Adam(model.parameters())
# We ignore the <pad> token in the loss calculation
criterion = nn.CrossEntropyLoss(ignore_index=0) # Assuming 0 is the pad token index

# ---
# 2. A Dummy Training Step
# ---
# Create a dummy batch
src_batch = torch.randint(1, INPUT_DIM, (4, 10)).to(device) # (batch, seq_len)
trg_batch = torch.randint(1, OUTPUT_DIM, (4, 12)).to(device)

optimizer.zero_grad()

# Get model output
output = model(src_batch, trg_batch)
# output = [batch size, trg len, output dim]

# To calculate the loss, we need to reshape the output and target
output_dim = output.shape[-1]
output = output[:, 1:].reshape(-1, output_dim) # Ignore <sos> token
trg = trg_batch[:, 1:].reshape(-1)

loss = criterion(output, trg)
loss.backward()
optimizer.step()

print("Seq2Seq model training step completed successfully.")
print(f"  - Calculated Loss: {loss.item():.4f}")
```

## Conclusion

The Sequence-to-Sequence (Seq2Seq) framework is a powerful and flexible architecture for tackling a wide range of problems where both the input and output are variable-length sequences. Its core innovation is the **encoder-decoder** structure, which separates the task of understanding the input from the task of generating the output.

**Key Takeaways:**

1.  **Encoder-Decoder Structure:** The encoder reads the entire input sequence and summarizes it into a fixed-size **context vector**. The decoder then uses this context vector to generate the output sequence token by token.
2.  **The Context Vector Bottleneck:** The context vector must contain all the information needed from the input sequence. This can be a limitation for long sequences, which motivated the development of the attention mechanism.
3.  **Autoregressive Decoding:** The decoder is **autoregressive**, meaning it uses its own previously generated output as an input for the next step.
4.  **Teacher Forcing:** To stabilize and accelerate training, we use teacher forcing, where we feed the decoder the ground-truth target token from the previous time step instead of its own prediction.

While modern systems have largely enhanced this basic architecture with attention, understanding the fundamental encoder-decoder pipeline is essential for grasping the concepts behind state-of-the-art models in machine translation and text generation.

## Self-Assessment Questions

1.  **Encoder's Role:** What is the sole purpose of the encoder in a Seq2Seq model?
2.  **Decoder's Initial State:** What is used as the initial hidden state for the decoder?
3.  **Teacher Forcing:** What is teacher forcing, and why is it used during training?
4.  **Input/Output:** During inference (not training), what is the input to the decoder at time step `t`?
5.  **The Bottleneck:** What is the main limitation of this simple Seq2Seq architecture, especially for long input sequences?

