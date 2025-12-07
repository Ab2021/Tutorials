# Days 106-126: Weeks 16-18 - Ray Tune, Serve & Data Labs
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## Week 16: Ray Tune (Days 106-112)

### HPO Basics
```python
from ray import tune

def objective(config):
    for step in range(100):
        score = train_step(config["lr"], config["batch_size"])
        tune.report(score=score)

analysis = tune.run(
    objective,
    config={
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
    },
    num_samples=100,
)
```

### ASHA Scheduler
```python
from ray.tune.schedulers import ASHAScheduler

scheduler = ASHAScheduler(
    max_t=100,
    grace_period=10,
    reduction_factor=2,
)
```

---

## Week 17: Ray Serve (Days 113-119)

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

serve.run(ModelServer.bind())
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
class AutoscaledModel:
    pass
```

---

## Week 18: Ray Data (Days 120-126)

### Streaming ETL
```python
import ray

ds = ray.data.read_parquet("s3://bucket/data/")
ds = ds.map(preprocess)
ds = ds.filter(lambda x: x["valid"])
ds.write_parquet("s3://bucket/processed/")
```

### Actor-based Transform
```python
class ModelPredictor:
    def __init__(self):
        self.model = load_model()
    
    def __call__(self, batch):
        return self.model.predict(batch)

ds = ds.map_batches(ModelPredictor, compute=ray.data.ActorPoolStrategy(size=4))
```

---

## 📝 Weeks 16-18 Summary
| Week | Topic | Key API |
|------|-------|---------|
| 16 | Tune | tune.run, schedulers |
| 17 | Serve | @serve.deployment |
| 18 | Data | ray.data |

---

## 🎓 Phase 6C Complete!
Distributed Computing with Ray (Weeks 13-18)
