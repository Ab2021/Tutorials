# Day 15.1: Transformer Architecture Fundamentals - A Practical Guide

## Introduction: The Architecture That Changed Everything

In 2017, the paper "Attention Is All You Need" was published, and it completely changed the landscape of sequence modeling. It introduced the **Transformer**, an architecture that dispensed with recurrence (RNNs, LSTMs) entirely and relied solely on a mechanism called **self-attention**. 

By processing all tokens in a sequence simultaneously and using self-attention to weigh the importance of every other token, the Transformer was not only more parallelizable and faster to train than RNNs, but it also achieved a new state of the art in machine translation. This architecture is the foundation for nearly all modern large language models, including BERT and GPT.

This guide will provide a high-level, practical overview of the original Transformer architecture, focusing on how its main components—the **Encoder** and the **Decoder**—work together.

**Today's Learning Objectives:**

1.  **Understand the Motivation for a Recurrence-Free Model:** Grasp the limitations of sequential processing in RNNs that the Transformer was designed to solve.
2.  **Explore the High-Level Encoder-Decoder Stack:** See how the Transformer is composed of a stack of identical Encoder blocks and a stack of identical Decoder blocks.
3.  **Grasp the Role of Positional Encodings:** Understand why an explicit encoding for token position is essential in a non-recurrent architecture.
4.  **See the Data Flow:** Trace how data moves through the encoder, across to the decoder, and is finally used to generate an output sequence.
5.  **Use the `nn.Transformer` Module:** Learn how to use PyTorch's built-in, high-level Transformer module to build a complete encoder-decoder model.

---

## Part 1: The Overall Architecture - An Encoder-Decoder Model

Like the Seq2Seq models we saw earlier, the Transformer is an **encoder-decoder** architecture, originally designed for machine translation.

1.  **The Encoder Stack:**
    *   **Input:** A sequence of token embeddings from the source language (e.g., German).
    *   **Job:** To process the entire input sequence and generate a rich, contextualized representation for each token. This representation, a sequence of vectors, contains the output of the final encoder layer.
    *   **Structure:** It consists of a stack of `N` identical **Encoder Blocks**.

2.  **The Decoder Stack:**
    *   **Input:** The sequence of token embeddings from the target language (e.g., English) so far, and the output from the Encoder stack.
    *   **Job:** To generate the next token in the target sequence, given the target words generated so far and the full context from the source sentence.
    *   **Structure:** It consists of a stack of `N` identical **Decoder Blocks**.

3.  **Final Linear + Softmax Layer:**
    *   Takes the final output vector from the top of the Decoder stack and passes it through a linear layer (to get scores for the entire vocabulary) and a softmax function (to get probabilities).

![Transformer Architecture](https://i.imgur.com/1I5a4to.png)

---

## Part 2: The Importance of Positional Encodings

**The Problem:** The core of the Transformer, the self-attention mechanism, is **permutation-invariant**. It treats the input as a "bag" of vectors. If you shuffle the words in the input sentence, the self-attention output will be exactly the same, just shuffled. It has no inherent sense of word order, which is critical for language.

**The Solution: Positional Encodings**

To solve this, the authors injected information about the position of each token into the model. They created a **positional encoding** vector for each position in the sequence (e.g., position 0, position 1, etc.). This vector is then **added** to the corresponding token embedding.

*   **How it works:** The original paper used a clever function involving sine and cosine waves of different frequencies. `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))` and `PE(pos, 2i+1) = cos(...)`.
*   **The Key Idea:** This gives each position a unique signature. Because of the properties of sine and cosine, the model can learn to attend to relative positions, as the encoding for position `pos+k` can be represented as a linear function of the encoding for `pos`.
*   **Alternative:** Many modern implementations simply use a learnable embedding for each position (`nn.Embedding`), which works just as well.

### 2.1. Implementing Positional Encoding

```python
import torch
import torch.nn as nn
import math

print("--- Part 2: Positional Encodings ---")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create a long enough positional encoding matrix that can be sliced
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension and register it as a buffer (not a model parameter)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: [batch size, seq len, embedding dim]
        # Add the positional encoding to the input tensor
        # Slicing `self.pe` to the length of the input sequence
        x = x + self.pe[:, :x.size(1), :]
        return x

# --- Usage Example ---
embedding_dim = 512
seq_len = 100
batch_size = 16

# Create dummy token embeddings
word_embeddings = torch.randn(batch_size, seq_len, embedding_dim)

pos_encoder = PositionalEncoding(d_model=embedding_dim)
final_embeddings = pos_encoder(word_embeddings)

print(f"Shape of word embeddings: {word_embeddings.shape}")
print(f"Shape after adding positional encodings: {final_embeddings.shape}")
```

---

## Part 3: The Encoder and Decoder Stacks

### 3.1. The Encoder Block

Each Encoder block has two sub-layers:
1.  A **Multi-Head Self-Attention** layer.
2.  A simple, position-wise **Feed-Forward Network** (a two-layer MLP).

Each of these sub-layers has a residual connection around it, followed by layer normalization. `Output = LayerNorm(x + Sublayer(x))`.

### 3.2. The Decoder Block

Each Decoder block has **three** sub-layers:
1.  A **Masked Multi-Head Self-Attention** layer. This is a self-attention layer over the target sequence generated so far. It is "masked" to prevent positions from attending to subsequent positions (i.e., to prevent it from cheating by looking at future words it is supposed to predict).
2.  A **Multi-Head Cross-Attention** layer. This is the crucial link between the encoder and decoder. The **Queries** come from the previous decoder layer, but the **Keys and Values** come from the **output of the encoder stack**. This is where the decoder looks at the source sentence to gather context.
3.  A position-wise **Feed-Forward Network**.

Like the encoder, each sub-layer has a residual connection and layer normalization.

---

## Part 4: Using PyTorch's `nn.Transformer` Module

PyTorch provides a high-level `nn.Transformer` module that contains the full encoder-decoder stack, making it easy to build a complete model.

```python
print("\n--- Part 4: Using the nn.Transformer Module ---")

# --- 1. Parameters ---
src_vocab_size = 5000
tgt_vocab_size = 5000
_model = 512
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
ff_dim = 2048
dropout = 0.1

# --- 2. The Full Model ---
class MyTransformer(nn.Module):
    def __init__(self):
        super(MyTransformer, self).__init__()
        
        # The main Transformer module
        self.transformer = nn.Transformer(
            d_model=_model, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers, 
            dim_feedforward=ff_dim, 
            dropout=dropout, 
            batch_first=True # Crucial!
        )
        
        # Embedding layers and positional encoding
        self.src_embedding = nn.Embedding(src_vocab_size, _model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, _model)
        self.pos_encoder = PositionalEncoding(d_model=_model)
        
        # Final output layer
        self.fc_out = nn.Linear(_model, tgt_vocab_size)

    def forward(self, src, tgt):
        # src shape: [batch, src_len]
        # tgt shape: [batch, tgt_len]
        
        # Create masks
        # The source key padding mask prevents attention from focusing on <pad> tokens
        src_key_padding_mask = (src == 0) # Assuming 0 is the pad token
        # The target key padding mask
        tgt_key_padding_mask = (tgt == 0)
        # The target mask (or subsequent mask) prevents attending to future tokens
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(-1)).to(src.device)

        # Embed and add positional encoding
        src_emb = self.pos_encoder(self.src_embedding(src))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt))
        
        # Pass through the transformer
        output = self.transformer(
            src_emb, 
            tgt_emb, 
            src_mask=None, # Not needed for encoder self-attention
            tgt_mask=tgt_mask,
            memory_mask=None, # Not needed for cross-attention
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask # Use src padding for memory
        )
        
        # Pass through the final linear layer
        return self.fc_out(output)

# --- 3. Dummy Usage ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyTransformer().to(device)

# Create dummy source and target sequences
src_seq = torch.randint(1, src_vocab_size, (8, 100)).to(device) # (batch, len)
tgt_seq = torch.randint(1, tgt_vocab_size, (8, 120)).to(device)

# Get the model's output logits
logits = model(src_seq, tgt_seq)

print(f"Input source shape: {src_seq.shape}")
print(f"Input target shape: {tgt_seq.shape}")
print(f"Output logits shape: {logits.shape}") # (batch, tgt_len, tgt_vocab_size)
```

## Conclusion

The Transformer architecture is a triumph of parallelizable, attention-based design. By completely removing recurrence, it overcame the sequential bottleneck of RNNs, allowing for the training of much larger and more powerful models on massive datasets.

**Key Takeaways:**

1.  **Attention is All You Need:** The Transformer is built entirely on multi-head self-attention and cross-attention mechanisms.
2.  **Encoder-Decoder Stacks:** The architecture consists of two main parts: an encoder to process the source sequence and a decoder to generate the target sequence.
3.  **Positional Encodings are Essential:** Since the model has no inherent sense of order, explicit positional information must be added to the token embeddings.
4.  **Three Types of Attention:** The full model uses three different attention mechanisms: encoder self-attention, masked decoder self-attention, and encoder-decoder cross-attention.
5.  **PyTorch Provides the Tools:** The `nn.Transformer` module provides a high-level, optimized implementation of the full architecture, making it accessible for practical use.

Understanding these fundamentals is the key to understanding almost all of the state-of-the-art models in NLP today.

## Self-Assessment Questions

1.  **Recurrence:** What is the main architectural feature of RNNs that the Transformer model completely eliminates?
2.  **Positional Encoding:** Why is it necessary to add positional encodings to the input embeddings in a Transformer?
3.  **Decoder Attention:** A Transformer decoder block has two different Multi-Head Attention layers. What is the difference between them in terms of their Queries, Keys, and Values?
4.  **Masking:** What is the purpose of the "subsequent mask" (or causal mask) in the decoder's self-attention layer?
5.  **Data Flow:** Where does the output of the encoder stack go?
