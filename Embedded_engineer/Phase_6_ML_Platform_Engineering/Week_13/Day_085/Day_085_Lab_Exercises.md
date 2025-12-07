# Days 85-91: Week 13 - Ray Core Labs
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## Day 85: Ray Tasks

```python
import ray

ray.init()

@ray.remote
def compute(x):
    return x ** 2

# Launch 100 parallel tasks
futures = [compute.remote(i) for i in range(100)]
results = ray.get(futures)
print(f"Sum of squares: {sum(results)}")
```

---

## Day 86: Ray Actors

```python
@ray.remote
class Counter:
    def __init__(self):
        self.value = 0
    
    def increment(self):
        self.value += 1
        return self.value

counter = Counter.remote()
futures = [counter.increment.remote() for _ in range(10)]
print(ray.get(futures))  # [1, 2, 3, ..., 10]
```

---

## Day 87: Object Store

```python
# Put large data once
big_array = ray.put(np.random.rand(10000, 10000))

@ray.remote
def process(data_ref):
    data = data_ref  # Zero-copy on same node
    return data.sum()

# All tasks share same data
results = ray.get([process.remote(big_array) for _ in range(10)])
```

---

## Day 88: Placement Groups

```python
from ray.util.placement_group import placement_group

# Gang scheduling for distributed training
pg = placement_group([{"GPU": 1} for _ in range(4)])
ray.get(pg.ready())

@ray.remote(num_gpus=1)
def train_worker():
    pass

workers = [train_worker.options(placement_group=pg).remote() for _ in range(4)]
```

---

## Day 89: Dashboard

```bash
# Access Ray dashboard
ray start --head --dashboard-host 0.0.0.0
# Open http://localhost:8265
```

---

## Day 90-91: Week 13 Project

```python
# Distributed web crawler
@ray.remote
def crawl(url):
    # Fetch and parse
    return links

@ray.remote
class URLQueue:
    def __init__(self):
        self.queue = []
        self.visited = set()
```

---

## 📝 Week 13 Summary
| Day | Topic | Concept |
|-----|-------|---------|
| 85 | Tasks | @ray.remote |
| 86 | Actors | Stateful |
| 87 | Objects | ray.put/get |
| 88 | Placement | Gang scheduling |
| 89 | Dashboard | Monitoring |
| 90-91 | Project | Crawler |
