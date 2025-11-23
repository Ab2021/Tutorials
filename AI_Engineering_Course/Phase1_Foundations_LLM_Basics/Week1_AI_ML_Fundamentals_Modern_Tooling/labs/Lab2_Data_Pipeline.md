# Lab 2: Robust Data Processing Pipeline

## Objective
Build a production-grade data pipeline for NLP.
Raw data is messy. Your model is only as good as your data.
We will use the Hugging Face `datasets` library to process a text classification dataset.

## 1. Setup

```bash
poetry add datasets apache-beam mwparserfromhell
```

## 2. The Pipeline (`pipeline.py`)

We will process the **IMDB Dataset** (Movie Reviews).
Steps:
1.  **Load:** Download raw data.
2.  **Clean:** Remove HTML tags (`<br />`), URLs, and weird characters.
3.  **Tokenize:** Convert text to numbers using a tokenizer.
4.  **Format:** Prepare for PyTorch.

```python
import re
from datasets import load_dataset
from transformers import AutoTokenizer
import html

# Configuration
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128

def clean_text(text):
    """
    Cleans raw text.
    1. Unescape HTML (e.g., &amp; -> &)
    2. Remove HTML tags (<br />)
    3. Remove URLs
    4. Lowercase
    """
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text) # Remove tags
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
    return text.lower()

def process_data():
    # 1. Load Dataset
    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb")
    
    # 2. Inspect Raw Data
    print("Raw Example:", dataset['train'][0]['text'][:100])
    
    # 3. Apply Cleaning
    print("Cleaning data...")
    dataset = dataset.map(lambda x: {'cleaned_text': clean_text(x['text'])})
    print("Cleaned Example:", dataset['train'][0]['cleaned_text'][:100])
    
    # 4. Tokenization
    print(f"Tokenizing with {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["cleaned_text"], 
            padding="max_length", 
            truncation=True, 
            max_length=MAX_LENGTH
        )
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # 5. Format for PyTorch
    # We only need 'input_ids', 'attention_mask', and 'label'
    tokenized_datasets = tokenized_datasets.remove_columns(["text", "cleaned_text"])
    tokenized_datasets.set_format("torch")
    
    print("Pipeline Complete.")
    print("Columns:", tokenized_datasets['train'].column_names)
    print("Shape of input_ids:", tokenized_datasets['train'][0]['input_ids'].shape)
    
    return tokenized_datasets

if __name__ == "__main__":
    final_data = process_data()
    
    # Save to disk (Simulating a Feature Store)
    final_data.save_to_disk("processed_imdb")
    print("Data saved to ./processed_imdb")
```

## 3. Data Quality Checks (`validate.py`)

In production, you never trust the output of a pipeline. You validate it.

```python
from datasets import load_from_disk
import torch

def validate_data(path):
    print(f"Validating data at {path}...")
    dataset = load_from_disk(path)
    
    # Check 1: No NaNs
    for split in ['train', 'test']:
        assert not torch.isnan(dataset[split]['input_ids']).any(), f"NaNs found in {split} input_ids"
        
    # Check 2: Label Distribution
    labels = dataset['train']['label']
    pos = (labels == 1).sum().item()
    neg = (labels == 0).sum().item()
    ratio = pos / (pos + neg)
    
    print(f"Positive Samples: {pos}")
    print(f"Negative Samples: {neg}")
    
    # IMDB should be balanced (50/50)
    assert 0.45 < ratio < 0.55, "Dataset is heavily imbalanced!"
    
    print("Validation Passed ✅")

if __name__ == "__main__":
    validate_data("processed_imdb")
```

## 4. Running the Lab

1.  Run the pipeline:
    ```bash
    python pipeline.py
    ```
2.  Run the validation:
    ```bash
    python validate.py
    ```

## 5. Challenge (Optional)
*   **Streaming:** Modify `load_dataset` to use `streaming=True`. This allows processing datasets larger than RAM (Terabytes).
*   **Parallelism:** Use `num_proc=4` in `dataset.map` to speed up processing on multi-core CPUs.

## 6. Submission
Submit the output of `validate.py`.
