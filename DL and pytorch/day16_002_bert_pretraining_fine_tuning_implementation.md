# Day 16.2: BERT Pre-training & Fine-tuning - A Practical Guide

## Introduction: The Two-Step Dance of Modern NLP

The success of BERT and other large language models is built on a powerful two-step paradigm:

1.  **Pre-training:** An enormous, general-purpose model is trained on a massive, unlabeled text corpus using a self-supervised objective (like Masked Language Modeling). This step is computationally expensive and is typically done by large research labs or companies.

2.  **Fine-tuning:** The pre-trained model is then adapted for a specific downstream task (like sentiment analysis or question answering) by training it further on a smaller, task-specific labeled dataset.

This approach democratized NLP by allowing anyone to achieve state-of-the-art results without needing the vast resources for pre-training. You simply download the pre-trained model and fine-tune it.

This guide provides a practical walkthrough of the fine-tuning process, showing how to adapt a pre-trained BERT model for a sequence classification task.

**Today's Learning Objectives:**

1.  **Understand the Full Pre-train/Fine-tune Workflow:** Solidify the conceptual understanding of this two-step process.
2.  **Build a Custom BERT-based Classifier:** Learn how to add a task-specific classification head on top of a pre-trained BERT model.
3.  **Prepare Data for Fine-tuning:** Use the Hugging Face `tokenizer` to correctly format your custom dataset for BERT.
4.  **Implement a Fine-tuning Loop:** Write the code to train your custom model, focusing on the best practices for learning rates and optimization.
5.  **Use the Hugging Face `Trainer` API:** See how this high-level API can abstract away the boilerplate training loop for common tasks.

---

## Part 1: The Fine-Tuning Architecture

The process of adapting BERT for a downstream task like sequence classification is straightforward.

1.  **Load the Pre-trained BERT Model:** This is the base of our new model. It acts as a powerful, contextual feature extractor.
2.  **Add a "Head":** We add one or more new layers on top of the BERT model. For sequence classification, this is typically a single `nn.Linear` layer.
3.  **Use the `[CLS]` Token:** The final hidden state corresponding to the special `[CLS]` token is used as the input to this new classification head. This vector is designed to be a rich, aggregate representation of the entire input sequence.

![BERT Fine-tuning](https://i.imgur.com/A4x2hB5.png)

### 1.1. Building a Custom BERT Classifier

Let's implement this in PyTorch.

```python
import torch
import torch.nn as nn
from transformers import BertModel

print("--- Part 1: Building a Custom BERT Classifier ---")

class BertForSentimentAnalysis(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_classes=2):
        super(BertForSentimentAnalysis, self).__init__()
        
        # --- 1. Load the pre-trained BERT model ---
        self.bert = BertModel.from_pretrained(model_name)
        
        # --- Optional: Freeze BERT parameters ---
        # If your dataset is very small, you might want to freeze the BERT layers
        # and only train the classifier head. This makes it a feature extractor.
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        
        # --- 2. Add the classification head ---
        # BERT-base's hidden size is 768.
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        # Pass the input through the BERT model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # --- 3. Use the [CLS] token's output ---
        # The output of the [CLS] token is in `pooler_output`.
        # It has been passed through a Linear layer and Tanh activation.
        cls_output = outputs.pooler_output
        
        # Apply dropout for regularization
        cls_output = self.dropout(cls_output)
        
        # Pass through our custom classifier head
        logits = self.classifier(cls_output)
        
        return logits

# --- Instantiate the model ---
model = BertForSentimentAnalysis()

# --- Dummy Usage ---
# A dummy batch of tokenized text
dummy_input_ids = torch.randint(100, 10000, (8, 50)) # (batch, seq_len)
dummy_attention_mask = torch.ones(8, 50)

logits = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)

print("Custom BERT classifier created successfully.")
print(f"Input shape: {dummy_input_ids.shape}")
print(f"Output logits shape: {logits.shape}")
```

---

## Part 2: The Fine-Tuning Process in Detail

Let's write a full, manual fine-tuning loop for our custom model.

### 2.1. Data Preparation

First, we need a labeled dataset and a `DataLoader` that correctly tokenizes and formats the data.

```python
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader

print("\n--- Part 2: The Fine-Tuning Process ---")

# --- 1. Dummy Data ---
# In a real project, this would come from a file (e.g., CSV).
texts = ["I love this movie, it's fantastic!", "This was a complete waste of time.", "A truly brilliant film.", "I would not recommend this to anyone."]
labels = [1, 0, 1, 0] # 1=Positive, 0=Negative

# --- 2. Tokenizer and Custom Dataset ---
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

train_dataset = SentimentDataset(texts, labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
```

### 2.2. The Fine-Tuning Loop

A key best practice when fine-tuning is to use a **smaller learning rate** than you would for training from scratch. The pre-trained weights are already very good, so we only want to nudge them slightly.

```python
import torch.optim as optim

# --- 3. Setup Model, Optimizer, and Loss ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSentimentAnalysis().to(device)

# Use a small learning rate for fine-tuning
optimizer = optim.AdamW(model.parameters(), lr=2e-5) # AdamW is often preferred for Transformers
loss_fn = nn.CrossEntropyLoss()

# --- 4. The Training Loop ---
print("\nStarting fine-tuning loop...")
model.train()
num_epochs = 3

for epoch in range(num_epochs):
    for batch in train_loader:
        # Move batch to the correct device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Calculate loss
        loss = loss_fn(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients (good practice)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        optimizer.step()
        
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

print("Fine-tuning complete.")
```

---

## Part 3: The Easy Way - Hugging Face `Trainer` API

As we saw in Day 8, the `Trainer` API abstracts away the manual training loop, making the process much simpler and less error-prone. It handles device placement, evaluation loops, logging, and more.

Using the `Trainer` is the **recommended approach** for most standard fine-tuning tasks.

```python
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

print("\n--- Part 3: Fine-Tuning with the Trainer API ---")

# --- 1. Load Model with Classification Head ---
# Hugging Face provides models with pre-built heads for common tasks.
model_hf = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# --- 2. Define Training Arguments ---
# This object configures the entire training process.
training_args = TrainingArguments(
    output_dir="./bert_results",
    num_train_epochs=1, # Use more epochs for real tasks
    per_device_train_batch_size=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=2,
)

# --- 3. Create the Trainer ---
# We can reuse the tokenized dataset from the previous section.
# (Assuming `tokenized_datasets` was created from a `datasets` object)
# For this sketch, we'll just confirm the setup.

# trainer = Trainer(
#     model=model_hf,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["test"],
# )

# --- 4. Train ---
# A single line to start the whole process!
# trainer.train()

print("The Hugging Face Trainer API simplifies the fine-tuning process significantly.")
print("It handles the training loop, evaluation, and logging automatically.")
```

## Conclusion

Fine-tuning is the process that unlocks the power of large, pre-trained language models like BERT. By taking a model that already has a deep, general understanding of language and simply training a small classification head on top, we can achieve state-of-the-art results on specific tasks with remarkable data and computational efficiency.

**Key Takeaways:**

1.  **Build on the Base:** The standard fine-tuning architecture involves loading a pre-trained BERT model and adding a new linear layer on top, using the `[CLS]` token's output as the input for classification.
2.  **Use a Small Learning Rate:** When fine-tuning, the pre-trained weights only need to be adjusted slightly. A small learning rate (e.g., 2e-5 to 5e-5) is crucial to avoid catastrophically forgetting the pre-trained knowledge.
3.  **AdamW is the Optimizer of Choice:** AdamW is a variant of Adam that decouples the weight decay from the gradient update, which has been shown to work better for training Transformers.
4.  **Leverage the Ecosystem:** For standard tasks, using a pre-built model class (like `BertForSequenceClassification`) and the `Trainer` API from Hugging Face is the most efficient and robust approach.

Mastering this fine-tuning workflow is the single most important skill for a modern NLP practitioner.

## Self-Assessment Questions

1.  **`[CLS]` Token:** What is the role of the `[CLS]` token's output in a BERT-based classification model?
2.  **Freezing vs. Fine-tuning:** When might you choose to freeze the main BERT layers and only train the classifier head?
3.  **Learning Rate:** Why is it important to use a small learning rate when fine-tuning?
4.  **`Trainer` API:** What are some of the responsibilities that the Hugging Face `Trainer` API handles for you automatically?
5.  **Model Naming:** What is the difference between loading `BertModel` and `BertForSequenceClassification` from the `transformers` library?
