# Days 120-126: Week 18 - Ray Data Labs
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## Ray Data Quick Reference

### Reading Data
```python
import ray

# Read from various sources
ds = ray.data.read_parquet("s3://bucket/data/")
ds = ray.data.read_csv("data/*.csv")
ds = ray.data.read_images("images/")
```

### Transformations
```python
# Map
ds = ds.map(lambda x: {"processed": x["value"] * 2})

# Map batches (more efficient)
ds = ds.map_batches(lambda batch: batch * 2)

# Filter
ds = ds.filter(lambda x: x["value"] > 0)
```

### Actor-based Processing
```python
class ModelPredictor:
    def __init__(self):
        self.model = load_model()
    
    def __call__(self, batch):
        return self.model.predict(batch)

ds = ds.map_batches(
    ModelPredictor,
    compute=ray.data.ActorPoolStrategy(size=4)
)
```

### Shuffle & GroupBy
```python
# Random shuffle
ds = ds.random_shuffle()

# GroupBy aggregation
ds.groupby("category").mean("value")
```

### Writing Data
```python
ds.write_parquet("s3://output/")
ds.write_json("output/")
```

---

## 📝 Week 18 Summary
| Day | Topic |
|-----|-------|
| 120 | Data basics |
| 121 | Connectors |
| 122 | Transforms |
| 123 | Shuffle |
| 124 | Custom sources |
| 125 | Performance |
| 126 | Project |
