# Day 16.4: BERT Applications & Analysis - A Practical Guide

## Introduction: The Universal Language Tool

BERT's ability to generate rich, contextualized embeddings has made it a versatile tool for a wide array of Natural Language Understanding (NLU) tasks. By fine-tuning the base BERT model with a simple, task-specific head, we can achieve state-of-the-art performance on problems ranging from classification to question answering.

Furthermore, analyzing the internal representations learned by BERT can give us fascinating insights into how these models "understand" language. We can probe the embeddings to see if they capture syntactic and semantic structures.

This guide provides a practical overview of several key applications of BERT and demonstrates how to analyze the embeddings it produces.

**Today's Learning Objectives:**

1.  **Explore Different Fine-tuning "Heads":** Understand how to adapt BERT for various tasks like sequence classification, token classification (e.g., Named Entity Recognition), and question answering.
2.  **Implement a Token Classification Model:** Build and fine-tune a BERT model for Named Entity Recognition (NER).
3.  **Implement a Question Answering Model:** Use a pre-trained BERT model to find the answer to a question within a given context paragraph.
4.  **Analyze and Visualize BERT Embeddings:** Use dimensionality reduction to visualize the vector space of BERT's contextual embeddings and see if they cluster semantically.

---

## Part 1: BERT for Different NLU Tasks

The Hugging Face `transformers` library makes it easy to adapt BERT for various tasks by providing pre-built model classes with the correct heads already attached.

*   **`BertForSequenceClassification`:**
    *   **Task:** Sentiment analysis, topic classification, natural language inference.
    *   **How it works:** Adds a single linear layer on top of the `[CLS]` token's output.

*   **`BertForTokenClassification`:**
    *   **Task:** Named Entity Recognition (NER), Part-of-Speech (POS) tagging.
    *   **How it works:** Adds a linear layer on top of **every** output token's hidden state to make a prediction for each token.

*   **`BertForQuestionAnswering`:**
    *   **Task:** Extractive Question Answering (finding the answer span in a context paragraph).
    *   **How it works:** Adds two linear layers that predict the `start` and `end` logits for each token in the context paragraph. The token with the highest start logit is predicted as the beginning of the answer, and the one with the highest end logit is the end.

### 1.1. Example: Fine-tuning for Named Entity Recognition (NER)

NER is the task of identifying and classifying named entities (like Person, Organization, Location) in a text.

```python
from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset

print("---", "Part 1: Fine-tuning for Named Entity Recognition", "---")

# --- 1. Load Model and Tokenizer ---
# We use a model pre-trained specifically on an NER task for better results.
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForTokenClassification.from_pretrained(model_name)

# --- 2. Prepare Input ---
text = "Hugging Face Inc. is a company based in New York City."
inputs = tokenizer(text, return_tensors="pt")

# --- 3. Get Predictions ---
with torch.no_grad():
    logits = model(**inputs).logits

# Get the predicted class index for each token
predictions = torch.argmax(logits, dim=2)

# --- 4. Decode and Display Results ---
print(f"Sentence: {text}")
print("\nPredicted Entities:")
for token, prediction in zip(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]), predictions[0]):
    # The model's config contains the mapping from index to label
    label = model.config.id2label[prediction.item()]
    if label != "O": # "O" means no entity
        print(f"  - Token: {token}, Entity: {label}")
```

### 1.2. Example: Question Answering

```python
from transformers import BertForQuestionAnswering

print("\n---", "Part 2: Question Answering", "---")

# --- 1. Load Model and Tokenizer ---
model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
tokenizer_qa = BertTokenizer.from_pretrained(model_name)
model_qa = BertForQuestionAnswering.from_pretrained(model_name)

# --- 2. Prepare Input ---
question = "What is the capital of France?"
context = "France is a country in Western Europe. Its capital and largest city is Paris."

inputs_qa = tokenizer_qa(question, context, return_tensors="pt")

# --- 3. Get Predictions ---
with torch.no_grad():
    outputs = model_qa(**inputs_qa)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

# --- 4. Find the Answer Span ---
# Get the most likely start and end token positions
answer_start_index = torch.argmax(start_logits)
answer_end_index = torch.argmax(end_logits)

# Decode the tokens between the start and end positions
predict_answer_tokens = inputs_qa.input_ids[0, answer_start_index : answer_end_index + 1]
answer = tokenizer_qa.decode(predict_answer_tokens)

print(f"Question: {question}")
print(f"Context: {context}")
print(f"Predicted Answer: {answer}")
```

---

## Part 3: Analyzing BERT's Embeddings

What kind of knowledge is actually stored in the contextual embeddings produced by BERT? We can analyze them to find out.

### 3.1. Visualizing the Embedding Space

Let's take the hidden states for several different words from the last layer of BERT and visualize them using t-SNE to see if they cluster by semantic meaning.

```python
import torch
from transformers import BertTokenizer, BertModel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

print("\n---", "Part 3: Analyzing BERT's Embeddings", "---")

# --- 1. Setup ---
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

# --- 2. Create Sentences and Extract Embeddings ---
sentences = [
    "The king is powerful.",
    "The queen is wise.",
    "The car is fast.",
    "The truck is heavy.",
    "He runs quickly.",
    "She walks slowly."
]

words_to_visualize = ['king', 'queen', 'car', 'truck', 'runs', 'walks']

embeddings = []
labels = []

with torch.no_grad():
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt")
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state.squeeze(0)
        
        # Find the tokens we want to visualize
        for word in words_to_visualize:
            try:
                token_idx = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]).index(word)
                embeddings.append(hidden_states[token_idx].numpy())
                labels.append(word)
            except ValueError:
                continue

embeddings = np.array(embeddings)

# --- 3. Use t-SNE for Dimensionality Reduction ---
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
embeddings_2d = tsne.fit_transform(embeddings)

# --- 4. Plot ---
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])

for label, x, y in zip(labels, embeddings_2d[:, 0], embeddings_2d[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points')

plt.title('t-SNE Visualization of BERT Word Embeddings')
plt.grid(True)
plt.show()
```

**Interpretation:**
The plot should show a clear clustering of semantically related words. `king` and `queen` should be close together, `car` and `truck` should be close, and `runs` and `walks` should be close. This visually demonstrates that BERT has learned a meaningful semantic vector space.

## Conclusion

BERT is not just a single model; it's a powerful and flexible foundation for a wide range of NLP tasks. By attaching different "heads" to the core Transformer encoder, we can adapt it to solve problems from document classification to fine-grained token-level analysis like NER.

Furthermore, analyzing the embeddings produced by BERT reveals that it learns a rich and structured representation of language, capturing the complex semantic relationships between words in context.

**Key Takeaways:**

1.  **Task-Specific Heads:** The key to adapting BERT is to add a small, task-specific output layer (a "head") on top of the pre-trained base.
2.  **Hugging Face for Convenience:** The `transformers` library provides pre-built classes like `BertForTokenClassification` and `BertForQuestionAnswering` that make this process simple.
3.  **BERT for NLU:** BERT's bidirectional nature makes it a powerhouse for Natural Language Understanding tasks where the meaning of the whole sentence is important.
4.  **Embeddings are Rich:** The contextual embeddings produced by BERT are not just random vectors; they contain a wealth of syntactic and semantic information that can be probed and visualized.

Understanding these applications and analytical techniques allows you to fully leverage the power of large pre-trained language models.

## Self-Assessment Questions

1.  **Token vs. Sequence Classification:** What is the main difference in the model architecture for token classification (NER) versus sequence classification (sentiment analysis)?
2.  **Question Answering:** What two things does a `BertForQuestionAnswering` model predict in order to identify the answer span?
3.  **Contextual Embeddings:** If you were to visualize the embeddings for the word "apple" from the sentences "I ate an apple" and "I bought an Apple laptop," would you expect them to be in the same location on a t-SNE plot? Why?
4.  **Fine-tuning:** Why is it generally more effective to fine-tune a dedicated model like `dbmdz/bert-large-cased-finetuned-conll03-english` for NER than to fine-tune the base `bert-base-uncased` model?
5.  **Model Probing:** Besides visualization, what other kinds of analyses could you perform on BERT's hidden states to understand what it has learned?
