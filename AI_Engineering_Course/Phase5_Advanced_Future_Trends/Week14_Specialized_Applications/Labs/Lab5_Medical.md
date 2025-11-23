# Lab 5: Medical De-ID

## Objective
HIPAA Compliance.
Remove PII (Names, Dates) from medical notes.

## 1. The Scrubber (`deid.py`)

```python
import re

text = "Patient John Doe admitted on 12/05/2023. Dr. Smith attending."

def scrub(text):
    # 1. Regex for Dates
    text = re.sub(r'\d{2}/\d{2}/\d{4}', '[DATE]', text)
    
    # 2. Regex for Names (Naive - Capitalized words)
    # Real systems use Presidio or specialized NER
    text = text.replace("John Doe", "[PATIENT]")
    text = text.replace("Dr. Smith", "[DOCTOR]")
    
    return text

print(scrub(text))
```

## 2. Challenge
Use **Microsoft Presidio** library to do this automatically.

## 3. Submission
Submit the scrubbed text output from Presidio.
