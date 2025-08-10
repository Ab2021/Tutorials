# Day 17.1: GPT Architecture & Autoregressive Training - A Practical Guide

## Introduction: The Generative Pre-trained Transformer

While BERT was revolutionizing the field of Natural Language Understanding (NLU) with its bidirectional encoders, another family of models was taking a different path, focusing on Natural Language Generation (NLG). This is the **GPT (Generative Pre-trained Transformer)** family, developed by OpenAI.

Unlike BERT, which is an encoder-only model, GPT is a **decoder-only** model. Its architecture and pre-training objective are specifically designed for one core task: **predicting the next word in a sequence**. This seemingly simple objective, when scaled up to massive datasets and model sizes, results in models with a remarkable ability to generate coherent, creative, and contextually relevant text.

This guide provides a practical deep dive into the GPT architecture and its autoregressive training methodology.

**Today's Learning Objectives:**

1.  **Understand the Decoder-Only Architecture:** See how GPT is simply a stack of Transformer Decoder blocks.
2.  **Grasp Autoregressive and Causal Language Modeling (CLM):** Understand how the model is trained to predict the next token and why this is called "autoregressive."
3.  **Learn about Masked Self-Attention:** See how the causal attention mask is the key mechanism that prevents the model from "cheating" by looking at future tokens.
4.  **Implement a Simple GPT-like Model:** Build a basic decoder-only Transformer to see the core components in action.
5.  **Perform Text Generation:** Write a simple generation loop to see how the model produces text one token at a time.

--- 

## Part 1: GPT's Architecture - A Stack of Decoders

The architecture of GPT is as simple as BERT's: it is just the **Decoder** part of the original Transformer model, stacked up.

*   **GPT-1:** 12 stacked Transformer Decoder blocks.
*   **GPT-2:** Up to 48 stacked blocks.
*   **GPT-3:** 96 stacked blocks.

**Key Architectural Features:**
*   **No Encoder:** There is no encoder stack to process a separate source sentence. The model only processes a single, continuous stream of text.
*   **No Cross-Attention:** Because there is no encoder, the **cross-attention** sub-layer present in a standard Transformer decoder is removed. 
*   **Masked Self-Attention:** The remaining self-attention layer is **masked** to ensure that when predicting the token at position `i`, the model can only attend to tokens at positions `j <= i`.

A GPT block therefore only has two sub-layers: a **Masked Multi-Head Self-Attention** layer and a **Position-wise Feed-Forward Network**.

![GPT Architecture](https://i.imgur.com/3aA9W9B.png)

--- 

## Part 2: The Pre-training Task - Causal Language Modeling

GPT is pre-trained on a single, simple objective: **Causal Language Modeling (CLM)**, also known as standard next-word prediction.

**The Task:** Given a sequence of tokens, the model's goal is to predict the very next token.

*   **Input:** `[token_1, token_2, ..., token_{t-1}]`
*   **Target:** `token_t`

This is repeated for every token in a massive text corpus. The model learns by trying to predict the next word at every single position in the text.

**Why it's called "Autoregressive":**
The term autoregressive means that the model uses its own previous predictions as input to generate the next prediction. During text generation, it predicts one word, appends that word to the input sequence, and then feeds the new, longer sequence back into the model to predict the next word. This step-by-step, self-referential process is what defines autoregressive generation.

### 2.1. The Causal Attention Mask

This is the key mechanism that enables CLM. Without it, the self-attention mechanism would allow a token at position `i` to see the token at position `i+1`, making the prediction task trivial.

The causal mask is a square matrix where the value at `(row_i, col_j)` is `-inf` if `j > i` and `0` otherwise. When this mask is added to the attention scores before the softmax, it effectively zeros out the probabilities for all future tokens.

```python
import torch
import torch.nn as nn

print("--- Part 2.1: The Causal Attention Mask ---")

seq_len = 5

# PyTorch has a built-in function to generate this mask.
causal_mask = nn.Transformer.generate_square_subsequent_mask(sz=seq_len)

print(f"A causal (or subsequent) mask for a sequence of length {seq_len}:")
print(causal_mask)
print("\n-inf means the position is masked and cannot be attended to.")

# How it's used:
# scores = torch.randn(1, seq_len, seq_len) # Dummy attention scores
# masked_scores = scores + causal_mask
# attention_weights = F.softmax(masked_scores, dim=-1)
```

--- 

## Part 3: Implementing a "Nano-GPT"

Let's build a simple, one-layer decoder-only Transformer to see the architecture in code.

```python
import math

print("\n--- Part 3: Implementing a Nano-GPT ---")

class NanoGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, n_layers=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Token and Positional Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(1024, embed_dim) # Max sequence length of 1024
        
        # The Transformer Decoder Block
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=ff_dim,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # The final linear layer to map to vocabulary scores
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # x shape: [batch, seq_len]
        batch_size, seq_len = x.shape
        
        # Create positional indices
        positions = torch.arange(0, seq_len).expand(batch_size, seq_len).to(x.device)
        
        # Get token and positional embeddings and add them
        tok_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding(positions)
        x = tok_emb + pos_emb
        
        # Create the causal mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Pass through the decoder
        # In a decoder-only model, the target sequence is also the memory.
        output = self.decoder(tgt=x, memory=x, tgt_mask=tgt_mask)
        
        # Get final logits
        logits = self.fc_out(output)
        
        return logits

# --- Dummy Usage ---
model = NanoGPT(vocab_size=1000, embed_dim=256, num_heads=4, ff_dim=512)

# A dummy batch of text
input_batch = torch.randint(0, 1000, (8, 50)) # (batch, seq_len)

# Get the logits for the *next* token at each position
logits = model(input_batch)

print(f"Input shape: {input_batch.shape}")
print(f"Output logits shape: {logits.shape}") # (batch, seq_len, vocab_size)
```

--- 

## Part 4: Autoregressive Text Generation

How does a trained GPT model actually generate text?

1.  You provide an initial **prompt** (e.g., "The future of AI is").
2.  The model processes this prompt and outputs a probability distribution over the entire vocabulary for the very next token.
3.  We **sample** from this distribution to choose the next token (e.g., "bright"). Common sampling methods include taking the most likely token (greedy), or more advanced methods like top-k or nucleus sampling.
4.  The newly chosen token is **appended** to the input sequence ("The future of AI is bright").
5.  This new, longer sequence is fed back into the model.
6.  The process repeats, generating one token at a time, until a maximum length is reached or an `<eos>` token is generated.

### 4.1. Implementation Sketch

```python
print("\n--- Part 4: Autoregressive Generation Sketch ---")

@torch.no_grad()
def generate_text(model, prompt_indices, max_len=20):
    model.eval()
    # `prompt_indices` is a tensor of shape (1, initial_seq_len)
    generated_sequence = prompt_indices
    
    for _ in range(max_len):
        # Get the model's prediction for the next token
        logits = model(generated_sequence)
        
        # Focus only on the logits for the very last time step
        last_step_logits = logits[:, -1, :]
        
        # Apply softmax to get probabilities
        probs = F.softmax(last_step_logits, dim=-1)
        
        # Sample the next token (here, we use greedy sampling - argmax)
        next_token = torch.argmax(probs, dim=-1).unsqueeze(0)
        
        # Append the new token to the sequence
        generated_sequence = torch.cat([generated_sequence, next_token], dim=1)
        
    return generated_sequence

# --- Dummy Usage ---
# Assume `model` is our trained NanoGPT and `tokenizer` exists
prompt = "the meaning of life is"
# prompt_indices = tokenizer.encode(prompt, return_tensors="pt")
dummy_prompt_indices = torch.randint(0, 1000, (1, 5))

generated_ids = generate_text(model, dummy_prompt_indices)

print(f"Prompt indices shape: {dummy_prompt_indices.shape}")
print(f"Generated sequence shape: {generated_ids.shape}")
# generated_text = tokenizer.decode(generated_ids[0])
```

## Conclusion

The GPT architecture, with its decoder-only design and autoregressive pre-training objective, is a powerful engine for text generation. By learning to predict the next word on a massive scale, it develops a deep, causal understanding of language that allows it to produce remarkably coherent and creative text.

**Key Takeaways:**

1.  **Decoder-Only:** GPT models are a stack of Transformer Decoder blocks, with the cross-attention mechanism removed.
2.  **Autoregressive Pre-training:** The model is trained on a simple next-word prediction task (Causal Language Modeling).
3.  **Causal Masking is Key:** The masked self-attention mechanism is what enables this training by preventing the model from seeing future tokens.
4.  **Generation is Step-by-Step:** Text is generated one token at a time, with the model's own output being fed back in as input for the next step.

This architecture is the foundation of all modern large language models used for generation, from the original GPT to the latest models like GPT-4, LLaMA, and Claude.

## Self-Assessment Questions

1.  **GPT vs. BERT:** What is the primary difference in the pre-training objective between GPT and BERT?
2.  **Causal Mask:** What is the purpose of the causal attention mask?
3.  **Cross-Attention:** Why is the cross-attention sub-layer from the original Transformer decoder not present in a GPT-style model?
4.  **Autoregressive:** What does it mean for a model to be "autoregressive"?
5.  **Use Case:** You want to build a model that can summarize long legal documents. Which type of architecture, BERT or GPT, would be more suitable for this task? Why?

