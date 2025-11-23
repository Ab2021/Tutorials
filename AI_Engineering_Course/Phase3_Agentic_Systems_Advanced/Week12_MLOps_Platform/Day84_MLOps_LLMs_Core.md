# Day 49: MLOps for LLMs
## Core Concepts & Theory

### MLOps for LLMs

**Traditional MLOps vs LLM MLOps:**
- **Traditional:** Model training, deployment, monitoring.
- **LLM:** Prompt management, fine-tuning, evaluation, cost tracking.

**LLM-Specific Challenges:**
- Non-deterministic outputs.
- Expensive to train and serve.
- Rapid iteration on prompts.
- Quality evaluation is subjective.

### 1. Prompt Management

**Version Control:**
```python
# prompts/v1/system_prompt.txt
You are a helpful assistant.

# prompts/v2/system_prompt.txt
You are a helpful assistant. Be concise.
```

**Prompt Registry:**
```python
class PromptRegistry:
    def __init__(self):
        self.prompts = {}
    
    def register(self, name: str, version: str, template: str):
        key = f"{name}:{version}"
        self.prompts[key] = template
    
    def get(self, name: str, version: str = "latest"):
        key = f"{name}:{version}"
        return self.prompts.get(key)

registry = PromptRegistry()
registry.register("system", "v1", "You are a helpful assistant.")
registry.register("system", "v2", "You are a helpful assistant. Be concise.")
```

**A/B Testing:**
```python
def get_prompt_for_user(user_id: str):
    # 50% get v1, 50% get v2
    if hash(user_id) % 2 == 0:
        return registry.get("system", "v1")
    else:
        return registry.get("system", "v2")
```

### 2. Model Versioning

**Model Registry:**
- **MLflow:** Track experiments, models, metrics.
- **Weights & Biases:** Experiment tracking, model versioning.
- **HuggingFace Hub:** Share and version models.

**Versioning Strategy:**
```
models/
  llama-2-7b-chat/
    v1.0/  # Base model
    v1.1/  # Fine-tuned on domain data
    v2.0/  # Fine-tuned with RLHF
```

### 3. Continuous Evaluation

**Automated Evaluation:**
```python
def evaluate_model(model, test_set):
    metrics = {
        "accuracy": 0,
        "hallucination_rate": 0,
        "latency_p95": 0,
        "cost_per_1k_tokens": 0
    }
    
    for example in test_set:
        output = model.generate(example["input"])
        
        # Accuracy
        if output == example["expected"]:
            metrics["accuracy"] += 1
        
        # Hallucination
        if has_hallucination(output):
            metrics["hallucination_rate"] += 1
    
    metrics["accuracy"] /= len(test_set)
    metrics["hallucination_rate"] /= len(test_set)
    
    return metrics
```

**Regression Testing:**
- Run evaluation on every model update.
- Alert if metrics degrade >5%.

### 4. Fine-Tuning Pipeline

**Automated Fine-Tuning:**
```
1. Data Collection: Gather new examples
2. Data Validation: Check quality, format
3. Fine-Tuning: Train on new data
4. Evaluation: Test on holdout set
5. Deployment: Deploy if metrics improve
```

**Continuous Fine-Tuning:**
- Collect user feedback daily.
- Fine-tune weekly on high-quality examples.
- A/B test new model vs current.

### 5. Deployment Strategies

**Blue-Green Deployment:**
```
Blue (Current): 100% traffic
Green (New): 0% traffic

→ Switch traffic to Green
→ Monitor for 1 hour
→ If OK, keep Green. If issues, rollback to Blue.
```

**Canary Deployment:**
```
Current: 95% traffic
Canary (New): 5% traffic

→ Monitor canary metrics
→ If OK, increase to 50%
→ If OK, increase to 100%
```

**Shadow Deployment:**
```
Current: Serves users
Shadow (New): Receives same requests but doesn't serve

→ Compare outputs
→ If Shadow performs well, promote to canary
```

### 6. Monitoring and Alerting

**Key Metrics:**
- **Latency:** p95 <2s.
- **Error Rate:** <1%.
- **Cost:** Within budget.
- **Quality:** User satisfaction >80%.

**Alerts:**
- p95 latency >2s for 5 minutes.
- Error rate >1% for 5 minutes.
- Daily cost >$1000.
- User satisfaction <70%.

### 7. Experiment Tracking

**MLflow Example:**
```python
import mlflow

mlflow.start_run()

# Log parameters
mlflow.log_param("model", "llama-2-7b")
mlflow.log_param("learning_rate", 1e-5)
mlflow.log_param("epochs", 3)

# Train model
model = fine_tune(...)

# Log metrics
mlflow.log_metric("accuracy", 0.85)
mlflow.log_metric("hallucination_rate", 0.05)

# Log model
mlflow.log_model(model, "model")

mlflow.end_run()
```

### 8. Data Management

**Training Data Versioning:**
- **DVC (Data Version Control):** Version large datasets.
- **LakeFS:** Git for data lakes.

**Data Quality:**
- **Validation:** Check format, length, toxicity.
- **Deduplication:** Remove duplicates.
- **Filtering:** Remove low-quality examples.

### 9. CI/CD for LLMs

**Continuous Integration:**
```yaml
# .github/workflows/ci.yml
name: CI

on: [push]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: pytest tests/
      - name: Evaluate model
        run: python evaluate.py
      - name: Check metrics
        run: |
          if [ $(cat metrics.json | jq '.accuracy') < 0.8 ]; then
            exit 1
          fi
```

**Continuous Deployment:**
```yaml
# Deploy if tests pass
deploy:
  needs: test
  runs-on: ubuntu-latest
  steps:
    - name: Deploy to staging
      run: kubectl apply -f k8s/staging/
    - name: Run smoke tests
      run: pytest tests/smoke/
    - name: Deploy to production
      run: kubectl apply -f k8s/production/
```

### 10. Incident Management

**Runbook:**
```
Incident: High latency (p95 >5s)

1. Check GPU utilization (should be >80%)
2. Check batch size (increase if low)
3. Check for long requests (limit max tokens)
4. Scale up replicas
5. If still high, rollback to previous version
```

**Post-Mortem:**
- What happened?
- What was the impact?
- What was the root cause?
- How do we prevent this in the future?

### Real-World Examples

**OpenAI:**
- Continuous fine-tuning on user feedback.
- A/B testing for prompt changes.
- Canary deployments for new models.

**Anthropic:**
- Constitutional AI for safety.
- Automated evaluation on benchmarks.
- Shadow deployment for new models.

### Summary

**MLOps Checklist:**
- [ ] **Prompt Management:** Version control, A/B testing.
- [ ] **Model Versioning:** Registry, semantic versioning.
- [ ] **Continuous Evaluation:** Automated tests, regression testing.
- [ ] **Deployment:** Blue-green or canary.
- [ ] **Monitoring:** Latency, cost, quality metrics.
- [ ] **CI/CD:** Automated testing and deployment.
- [ ] **Incident Management:** Runbooks, post-mortems.

### Next Steps
In the Deep Dive, we will implement complete MLOps pipeline with prompt management, model versioning, continuous evaluation, and automated deployment.
