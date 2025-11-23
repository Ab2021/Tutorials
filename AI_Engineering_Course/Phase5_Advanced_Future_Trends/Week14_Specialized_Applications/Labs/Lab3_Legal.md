# Lab 3: Legal Analyzer (NER)

## Objective
Extract entities from legal text.
Parties, Dates, Amounts.

## 1. The Analyzer (`legal.py`)

```python
import spacy

# Load model (mocking a legal-specific model)
nlp = spacy.blank("en")

text = "This Agreement is made on 2023-01-01 between Alice Corp and Bob Inc for $10,000."

# 1. Rule-based Matching
ruler = nlp.add_pipe("entity_ruler")
patterns = [
    {"label": "ORG", "pattern": "Alice Corp"},
    {"label": "ORG", "pattern": "Bob Inc"},
    {"label": "MONEY", "pattern": [{"LIKE_NUM": True}]}
]
ruler.add_patterns(patterns)

# 2. Process
doc = nlp(text)

for ent in doc.ents:
    print(f"{ent.text} ({ent.label_})")
```

## 2. Analysis
In production, you would fine-tune a BERT model on the **CUAD** (Contract Understanding Atticus Dataset).

## 3. Submission
Submit the extracted entities.
