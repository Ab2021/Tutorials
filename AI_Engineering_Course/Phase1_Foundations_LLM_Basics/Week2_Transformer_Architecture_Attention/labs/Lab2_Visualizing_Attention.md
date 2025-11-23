# Lab 2: Visualizing Attention

## Objective
Transformers are "Black Boxes". Attention visualization helps us peek inside.
We will create a heatmap to see which words the model focuses on when generating the next token.

## 1. Setup

We will use `bertviz` or `matplotlib` to plot the attention weights.
For this lab, we will use a pre-trained BERT model from Hugging Face, as it has strong attention patterns.

```bash
poetry add transformers matplotlib seaborn
```

## 2. The Visualizer (`visualize.py`)

```python
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertModel

# 1. Load Model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, output_attentions=True)

def get_attention(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    attention = outputs.attentions # Tuple of (Batch, NumHeads, SeqLen, SeqLen)
    return attention, inputs

def plot_attention(attention, tokens, layer_num, head_num):
    # Get attention for specific layer and head
    # attention[layer] is (1, 12, SeqLen, SeqLen)
    attn_matrix = attention[layer_num][0, head_num].detach().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_matrix, xticklabels=tokens, yticklabels=tokens, cmap="viridis")
    plt.title(f"Layer {layer_num + 1}, Head {head_num + 1}")
    plt.xlabel("Key (Attended To)")
    plt.ylabel("Query (Attending From)")
    plt.show()

# 2. Run
text = "The animal didn't cross the street because it was too tired."
attention, inputs = get_attention(text)
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

print(f"Tokens: {tokens}")

# Plot Layer 10, Head 5 (Often captures coreference)
plot_attention(attention, tokens, layer_num=9, head_num=4)
```

## 3. Analysis

Run the script. Look at the heatmap for the word **"it"**.
*   In Layer 9 or 10, does **"it"** attend strongly to **"animal"**?
*   Change the sentence to: "The animal didn't cross the street because it was too wide."
*   Does **"it"** now attend to **"street"**?

This phenomenon is called **Winograd Schema** resolution.

## 4. Challenge
*   **Average Attention:** Plot the average attention across all heads in a layer. Is it more diffuse or focused?
*   **Interactive Tool:** Use `bertviz` (`from bertviz import head_view`) to create an interactive HTML visualization.

## 5. Submission
Submit two heatmaps: one where "it" refers to "animal", and one where "it" refers to "street".
