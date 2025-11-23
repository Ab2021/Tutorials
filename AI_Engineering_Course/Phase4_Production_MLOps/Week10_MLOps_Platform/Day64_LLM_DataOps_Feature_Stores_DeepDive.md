# Day 64: LLM DataOps & Feature Stores
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Simple LLM Feature Store (Redis + Feast Concept)

Simulating a system that retrieves User Context (Structured) and Document Context (Unstructured) for a prompt.

```python
import redis
import json
import time

class LLMFeatureStore:
    def __init__(self):
        # Online Store (Redis)
        self.redis = redis.Redis(host='localhost', port=6379, db=0)
        
    def write_user_features(self, user_id, features):
        """Write structured features (e.g., plan, usage)."""
        key = f"user:{user_id}"
        self.redis.set(key, json.dumps(features))
        
    def get_user_context(self, user_id):
        """Retrieve context for prompt injection."""
        key = f"user:{user_id}"
        data = self.redis.get(key)
        return json.loads(data) if data else {}
    
    def write_document_embeddings(self, doc_id, text, embedding):
        """Simulate Vector Store write."""
        # In reality, use Pinecone/Milvus
        # Here we store in Redis for demo
        key = f"doc:{doc_id}"
        data = {'text': text, 'vector': embedding}
        self.redis.set(key, json.dumps(data))

# Usage
fs = LLMFeatureStore()
fs.write_user_features("u123", {
    "name": "Alice",
    "subscription": "Pro",
    "last_topic": "Python"
})

# At Inference Time
user_ctx = fs.get_user_context("u123")
prompt = f"User: {user_ctx['name']} ({user_ctx['subscription']}). Question: How to optimize lists?"
print(prompt)
```

### 2. Data Quality Validation Pipeline

Using `pydantic` and simple heuristics to validate dataset quality before fine-tuning.

```python
from pydantic import BaseModel, validator, ValidationError
from typing import List, Optional
import re

class TrainingExample(BaseModel):
    instruction: str
    input: Optional[str] = ""
    output: str
    
    @validator('instruction')
    def check_instruction_length(cls, v):
        if len(v.split()) < 3:
            raise ValueError("Instruction too short (< 3 words)")
        return v
    
    @validator('output')
    def check_output_quality(cls, v):
        if len(v) < 10:
            raise ValueError("Output too short")
        if "I don't know" in v:
            raise ValueError("Output contains refusal")
        return v
        
    @validator('instruction')
    def check_pii(cls, v):
        # Simple regex for email
        if re.search(r'[\w\.-]+@[\w\.-]+', v):
            raise ValueError("PII detected (Email)")
        return v

def validate_dataset(dataset):
    valid_data = []
    rejected_data = []
    
    for item in dataset:
        try:
            example = TrainingExample(**item)
            valid_data.append(example)
        except ValidationError as e:
            rejected_data.append((item, e.errors()))
            
    return valid_data, rejected_data

# Test
raw_data = [
    {"instruction": "Explain quantum physics", "output": "It is complex."}, # Output too short
    {"instruction": "Email me at bob@gmail.com", "output": "Sure."}, # PII
    {"instruction": "Write a Python script to sort a list", "output": "Here is the code: `sorted(list)`"} # Valid
]

valid, rejected = validate_dataset(raw_data)
print(f"Valid: {len(valid)}, Rejected: {len(rejected)}")
for item, error in rejected:
    print(f"Rejected: {item['instruction']} -> {error[0]['msg']}")
```

### 3. Synthetic Data Generation (Self-Instruct)

Using an LLM to generate more data from seed tasks.

```python
import random

SEED_TASKS = [
    {"instruction": "Explain why the sky is blue.", "output": "Rayleigh scattering..."},
    {"instruction": "Write a poem about cats.", "output": "Soft paws, sharp claws..."}
]

def generate_synthetic_instruction(seed_tasks, llm_client):
    # 1. Sample seeds
    seeds = random.sample(seed_tasks, 2)
    
    # 2. Create Prompt
    prompt = "Come up with a new instruction and output similar to these:\n"
    for s in seeds:
        prompt += f"Instruction: {s['instruction']}\nOutput: {s['output']}\n\n"
    prompt += "New Instruction:"
    
    # 3. Call LLM (Mock)
    # response = llm_client.generate(prompt)
    response = "Instruction: How to bake a cake?\nOutput: Mix flour, sugar..."
    
    return response

# This loop runs to expand dataset from 100 seeds to 10k examples
```

### 4. Data Versioning with DVC (Conceptual)

```bash
# 1. Initialize
dvc init

# 2. Add Data
dvc add data/raw_dataset.json

# 3. Create Pipeline Stage (Cleaning)
dvc run -n clean_data \
        -d data/raw_dataset.json \
        -d src/clean.py \
        -o data/cleaned_dataset.json \
        python src/clean.py data/raw_dataset.json data/cleaned_dataset.json

# 4. Create Pipeline Stage (Embedding)
dvc run -n embed_data \
        -d data/cleaned_dataset.json \
        -o data/embeddings.npy \
        python src/embed.py

# 5. Track
git add dvc.yaml dvc.lock data/raw_dataset.json.dvc
git commit -m "Update data pipeline"
```

### 5. Feedback Loop Logger

Logging interaction data for future fine-tuning.

```python
import json
import uuid
from datetime import datetime

class FeedbackLogger:
    def __init__(self, log_file="interaction_logs.jsonl"):
        self.log_file = log_file
        
    def log(self, prompt, response, user_id, feedback_score=None):
        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "prompt": prompt,
            "response": response,
            "feedback": feedback_score, # 1 (thumbs up) or -1 (thumbs down)
            "model_version": "v1.2.0"
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

# Usage
logger = FeedbackLogger()
logger.log("Hi", "Hello!", "u1", 1)
```

### 6. PII Scrubbing (Presidio)

Using Microsoft Presidio for PII anonymization.

```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

text = "My name is John and my phone is 555-0199."

# 1. Analyze
results = analyzer.analyze(text=text, entities=["PERSON", "PHONE_NUMBER"], language='en')

# 2. Anonymize
anonymized = anonymizer.anonymize(text=text, analyzer_results=results)

print(anonymized.text)
# Output: "My name is <PERSON> and my phone is <PHONE_NUMBER>."
```
