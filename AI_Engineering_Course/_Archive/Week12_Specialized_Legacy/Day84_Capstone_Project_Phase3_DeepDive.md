# Day 84: Capstone Project Phase 3 - Refinement & Production
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. RAGAS Evaluation Script

Automated grading of the pipeline.

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset

def run_eval(questions, answers, contexts, ground_truths):
    data = {
        'question': questions,
        'answer': answers,
        'contexts': contexts, # List of lists
        'ground_truth': ground_truths
    }
    
    dataset = Dataset.from_dict(data)
    
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision]
    )
    
    print(results)
    return results

# Usage
# q = ["What is the revenue?"]
# a = ["$10M"]
# c = [["Revenue was $10M in 2023..."]]
# gt = ["$10M"]
# run_eval(q, a, c, gt)
```

### 2. Adding Re-ranking (Cross-Encoder)

Improving precision.

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_docs(query, docs, top_k=5):
    # docs: List of text strings
    pairs = [[query, doc] for doc in docs]
    
    scores = reranker.predict(pairs)
    
    # Sort by score
    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    
    return [doc for doc, score in scored_docs[:top_k]]

# Integration
# 1. Retrieve top 50 from Qdrant
# 2. Rerank to top 5
# 3. Send to LLM
```

### 3. Semantic Cache (Redis)

Saving costs.

```python
import redis
import json
from sentence_transformers import SentenceTransformer

r = redis.Redis()
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_cached_response(query):
    # 1. Embed query
    vec = model.encode(query).tobytes()
    
    # 2. Search (Redis Vector Search - Simplified here as exact match for demo)
    # In production, use RediSearch for vector similarity
    if r.exists(query):
        return r.get(query)
    return None

def set_cache(query, response):
    r.setex(query, 3600, response) # 1 hour TTL
```

### 4. Deployment (Kubernetes YAML)

Deploying the API.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: capstone-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: capstone-agent
  template:
    metadata:
      labels:
        app: capstone-agent
    spec:
      containers:
      - name: api
        image: myregistry/capstone-agent:v1
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: ai-secrets
              key: openai-key
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
          limits:
            cpu: "1000m"
            memory: "1Gi"
---
apiVersion: v1
kind: Service
metadata:
  name: capstone-service
spec:
  selector:
    app: capstone-agent
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```
