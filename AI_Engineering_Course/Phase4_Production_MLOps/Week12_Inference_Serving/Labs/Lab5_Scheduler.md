# Lab 5: Batching Scheduler

## Objective
Maximize GPU utilization.
Implement **Continuous Batching** logic.

## 1. The Scheduler (`scheduler.py`)

```python
queue = [10, 20, 5, 100] # Request lengths
running = []
max_batch = 2

def step():
    # 1. Add to batch
    while len(running) < max_batch and queue:
        req = queue.pop(0)
        running.append(req)
        print(f"Started request length {req}")

    # 2. Decode one token
    finished = []
    for i in range(len(running)):
        running[i] -= 1
        if running[i] == 0:
            finished.append(i)
            
    # 3. Remove finished (in reverse to keep indices valid)
    for i in sorted(finished, reverse=True):
        print("Request finished")
        running.pop(i)

# Run
for t in range(150):
    if not running and not queue: break
    print(f"Time {t}: Running {len(running)}")
    step()
```

## 2. Analysis
Unlike static batching, we don't wait for the longest sequence to finish.
We insert new requests immediately when a slot opens.

## 3. Submission
Submit the total time steps required to process the queue.
