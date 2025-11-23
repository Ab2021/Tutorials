# Lab 5: Feedback Loop

## Objective
Close the loop.
Collect user feedback to improve the model.

## 1. The Endpoint (`feedback.py`)

```python
import json

log_file = "feedback.jsonl"

def log_feedback(request_id, rating, comment):
    entry = {
        "id": request_id,
        "rating": rating, # 1 (Thumbs Up) or 0 (Thumbs Down)
        "comment": comment
    }
    
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print("Feedback logged.")

# Usage
log_feedback("req_123", 1, "Great answer!")
log_feedback("req_124", 0, "Hallucinated.")
```

## 2. Challenge
Write a script to calculate the **Average CSAT** (Customer Satisfaction Score) from the log file.

## 3. Submission
Submit the average CSAT score.
