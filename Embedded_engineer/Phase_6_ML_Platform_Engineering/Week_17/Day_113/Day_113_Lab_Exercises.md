# Days 113-119: Week 17 - Ray Serve Labs
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## Ray Serve Quick Reference

### Deployment
```python
from ray import serve

@serve.deployment(num_replicas=3)
class ModelServer:
    def __init__(self):
        self.model = load_model()
    
    async def __call__(self, request):
        data = await request.json()
        return self.model.predict(data)

app = ModelServer.bind()
serve.run(app)
```

### Composition (DAG)
```python
@serve.deployment
class Preprocessor:
    def __call__(self, data):
        return preprocess(data)

@serve.deployment
class Model:
    def __call__(self, data):
        return predict(data)

preprocessor = Preprocessor.bind()
model = Model.bind()

# Chain deployments
app = model.bind(preprocessor)
```

### Autoscaling
```python
@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 10,
        "target_num_ongoing_requests_per_replica": 5,
    }
)
class AutoModel:
    pass
```

### Request Batching
```python
@serve.deployment
class BatchModel:
    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.1)
    async def __call__(self, requests):
        # Process batch
        return [self.model(r) for r in requests]
```

---

## 📝 Week 17 Summary
| Day | Topic |
|-----|-------|
| 113 | Serve Basics |
| 114 | FastAPI |
| 115 | Composition |
| 116 | Autoscaling |
| 117 | Batching |
| 118 | Rollouts |
| 119 | Project |
