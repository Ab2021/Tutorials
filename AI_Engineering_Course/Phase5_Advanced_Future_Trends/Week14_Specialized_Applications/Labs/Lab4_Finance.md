# Lab 4: Financial Sentiment

## Objective
Trade based on news.
Use **FinBERT**.

## 1. The Analyzer (`finance.py`)

```python
from transformers import pipeline

# Load FinBERT
classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert")

headlines = [
    "Tesla stock surges after earnings beat.",
    "Inflation concerns weigh on markets.",
    "Apple announces new iPhone."
]

results = classifier(headlines)

for h, r in zip(headlines, results):
    print(f"Headline: {h}")
    print(f"Sentiment: {r['label']} ({r['score']:.4f})\n")
```

## 2. Analysis
FinBERT is pre-trained on financial text, so it understands "bullish" vs "bearish" better than generic BERT.

## 3. Submission
Submit the sentiment label for "Company X files for bankruptcy."
