# Day 8.4: Modern Sequence Models - A Practical Overview

## Introduction: The Cambrian Explosion of NLP

The invention of the Transformer architecture triggered a "Cambrian explosion" in Natural Language Processing. The ability to train deep, attention-based models on massive text corpora led to the development of **pre-trained language models (PLMs)** that have revolutionized the field. These models are trained on a general-domain task (like predicting masked words) on terabytes of text data, and the resulting models can then be easily **fine-tuned** to achieve state-of-the-art performance on a wide range of downstream tasks (like sentiment analysis, question answering, etc.).

This guide provides a high-level, practical overview of the most influential modern sequence models, focusing on the two main families: **BERT (encoder-only)** and **GPT (decoder-only)**. We will explore their core ideas and show how to use them in practice with the Hugging Face `transformers` library.

**Today's Learning Objectives:**

1.  **Understand the Pre-training / Fine-tuning Paradigm:** Grasp this fundamental workflow that powers modern NLP.
2.  **Explore BERT (Bidirectional Encoder Representations from Transformers):** Understand its architecture (encoder-only), its pre-training objective (Masked Language Modeling), and its suitability for NLU tasks.
3.  **Explore GPT (Generative Pre-trained Transformer):** Understand its architecture (decoder-only), its pre-training objective (Causal Language Modeling), and its suitability for text generation tasks.
4.  **Use the Hugging Face `transformers` Library:** Learn how to easily load pre-trained models and tokenizers for both BERT and GPT.
5.  **Apply a Pre-trained Model:** Perform a practical fine-tuning task for sentiment analysis using a pre-trained BERT model.

---

## Part 1: The Pre-training and Fine-tuning Paradigm

This is the central workflow of modern NLP, analogous to using a pre-trained ResNet for computer vision.

1.  **Pre-training (99% of the work, done by large tech companies):**
    *   A massive, general-domain text corpus is collected (e.g., the entire web, all of Wikipedia).
    *   A large Transformer model is trained on a **self-supervised** task. This means the labels are generated automatically from the input text itself, so no human annotation is needed.
    *   The result is a **pre-trained language model** whose weights encode a deep, statistical understanding of language.

2.  **Fine-tuning (1% of the work, done by you):**
    *   You take the pre-trained model and add a small, task-specific classification head on top (e.g., a single linear layer).
    *   You train this new composite model on your smaller, labeled dataset (e.g., 50,000 movie reviews for sentiment analysis).
    *   Because the model already understands language, it can learn your specific task with very high accuracy and data efficiency.

---

## Part 2: BERT - The Bidirectional Encoder

**Architecture:** BERT is an **encoder-only** model. It uses a stack of Transformer Encoder layers. Its goal is to take a sequence of text and produce a rich, contextualized representation for every token in the sequence.

**Pre-training Objective: Masked Language Modeling (MLM)**
*   BERT takes a sentence, randomly **masks** 15% of the words (e.g., "the [MASK] sat on the mat"), and its objective is to predict the original masked word ("cat").
*   Because the model can see the words both to the left and to the right of the mask, it learns a deep **bidirectional** understanding of context.

**Best suited for:** Natural Language Understanding (NLU) tasks where you need to understand the meaning of a full sentence, such as:
*   Sentiment analysis
*   Sentence classification
*   Question answering

### 2.1. Using BERT with Hugging Face `transformers`

The Hugging Face `transformers` library is the de facto standard for working with pre-trained models.

```python
import torch
from transformers import BertTokenizer, BertModel

print("---"" Part 2: Using BERT with Hugging Face ---")

# --- 1. Load a Pre-trained Tokenizer and Model ---
# We use the popular 'bert-base-uncased' model.
model_name = 'bert-base-uncased'

# The tokenizer converts raw text into the specific input format BERT expects.
tokenizer = BertTokenizer.from_pretrained(model_name)
# The model itself.
model = BertModel.from_pretrained(model_name)

# --- 2. Prepare the Input ---
sentence = "The cat sat on the mat."

# The tokenizer handles tokenization, adding special tokens ([CLS], [SEP]),
# and converting to integer IDs.
inputs = tokenizer(sentence, return_tensors="pt") # "pt" returns PyTorch tensors

print(f"Original sentence: '{sentence}'")
print(f"Tokenized IDs: {inputs['input_ids']}")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")

# --- 3. Get the Model Output ---
with torch.no_grad():
    outputs = model(**inputs)

# BERT's main output is the hidden state for every token.
last_hidden_states = outputs.last_hidden_state

print(f"\nShape of BERT's output hidden states: {last_hidden_states.shape}")
# (batch_size, sequence_length, hidden_size)
```

---

## Part 3: GPT - The Autoregressive Decoder

**Architecture:** GPT (Generative Pre-trained Transformer) is a **decoder-only** model. It uses a stack of Transformer Decoder layers. Its goal is to predict the next word in a sequence given all the previous words.

**Pre-training Objective: Causal Language Modeling (CLM)**
*   GPT is trained to predict the next token in a sequence. For the input "The cat sat on the", the target is "mat".
*   To prevent it from cheating, it uses a **masked self-attention** mechanism. During self-attention, a token at position `i` is only allowed to attend to tokens at positions `j <= i`. It cannot see future tokens.
*   This makes the model **autoregressive** or **unidirectional**.

**Best suited for:** Natural Language Generation (NLG) tasks, such as:
*   Text generation (story writing, code generation)
*   Summarization
*   Dialogue systems

### 3.1. Using GPT with Hugging Face `transformers`

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

print("\n---"" Part 3: Using GPT with Hugging Face ---")

# --- 1. Load a Pre-trained Tokenizer and Model ---
# We use the popular 'gpt2' model.
model_name = 'gpt2'

tokenizer_gpt = GPT2Tokenizer.from_pretrained(model_name)
# Note: We load `GPT2LMHeadModel`, which includes the language modeling head for generation.
model_gpt = GPT2LMHeadModel.from_pretrained(model_name)

# --- 2. Prepare the Input ---
# Let's give the model a prompt to continue.
prompt = "The future of AI is"
inputs_gpt = tokenizer_gpt(prompt, return_tensors="pt")

# --- 3. Generate Text ---
# The .generate() method is a powerful tool for autoregressive generation.
# It handles feeding the output of one step as the input to the next.
output_sequences = model_gpt.generate(
    input_ids=inputs_gpt['input_ids'],
    max_length=50, # Generate up to 50 tokens
    num_return_sequences=1, # Generate one possible continuation
    pad_token_id=tokenizer_gpt.eos_token_id # Set pad token for open-ended generation
)

# Decode the generated token IDs back to text
generated_text = tokenizer_gpt.decode(output_sequences[0], skip_special_tokens=True)

print(f"\nPrompt: '{prompt}'")
print(f"Generated Text: '{generated_text}'")
```

---

## Part 4: Practical Fine-Tuning for Sentiment Analysis

Let's fine-tune a BERT-based model (`distilbert-base-uncased`, a smaller, faster version of BERT) on a sentiment analysis task.

This example uses the Hugging Face `Trainer` API, which simplifies the training loop.

```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

print("\n---"" Part 4: Fine-Tuning for Sentiment Analysis ---")

# --- 1. Load Dataset and Tokenizer ---
# We use the `datasets` library to easily load the IMDB dataset.
dataset = load_dataset("imdb", split="train[:1%]") # Use a tiny subset for this demo
dataset = dataset.train_test_split(test_size=0.2)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# --- 2. Preprocess the Data ---
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# --- 3. Load the Pre-trained Model ---
# We load a model with a sequence classification head already on top.
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# --- 4. Set up the Trainer ---
# The Trainer API handles the training loop, evaluation, etc.
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# --- 5. Fine-tune the model ---
print("\nStarting fine-tuning...")
trainer.train()
print("Fine-tuning complete.")

# --- 6. Evaluate ---
print("\nEvaluating the fine-tuned model...")
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")
```

## Conclusion

Modern NLP is dominated by the paradigm of fine-tuning large, pre-trained Transformer models. By understanding the fundamental architectural differences between encoder-only models like BERT and decoder-only models like GPT, you can choose the right tool for your task.

*   **For understanding tasks (classification, Q&A):** Use a **BERT-style** (encoder) model.
*   **For generation tasks (dialogue, story writing):** Use a **GPT-style** (decoder) model.

The Hugging Face `transformers` ecosystem has made these incredibly powerful models accessible to everyone, allowing you to achieve state-of-the-art results on a wide variety of NLP tasks with just a few lines of code.

## Self-Assessment Questions

1.  **Pre-training vs. Fine-tuning:** What is the main difference between the pre-training and fine-tuning stages?
2.  **BERT vs. GPT Architecture:** What is the key architectural difference between BERT and GPT?
3.  **Pre-training Objectives:** What is Masked Language Modeling (MLM)? What is Causal Language Modeling (CLM)? Which model uses which?
4.  **Bidirectional vs. Unidirectional:** Why is BERT considered "bidirectional" while GPT is "unidirectional" or "autoregressive"?
5.  **Use Case:** You want to build a chatbot that can answer customer service questions. Which type of model, BERT or GPT, would be more suitable for generating the chatbot's responses? Why?

```