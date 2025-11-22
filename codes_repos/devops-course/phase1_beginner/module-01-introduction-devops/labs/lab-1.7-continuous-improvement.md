# Lab 1.7: Continuous Improvement (Kaizen)

## 🎯 Objective

Learn the philosophy of **Kaizen** (Continuous Improvement) by iteratively optimizing a slow, inefficient process. You will measure baseline performance, identify bottlenecks, implement improvements, and verify results.

## 📋 Prerequisites

-   Completed Lab 1.6.
-   Python 3 installed.

## 📚 Background

### What is Kaizen?

Kaizen is a Japanese business philosophy regarding the processes that continuously improve operations and involve all employees. In DevOps, this translates to:
-   **Blameless Post-Mortems**: Learning from failure.
-   **Refactoring**: Improving code structure.
-   **Optimization**: Making systems faster and cheaper.
-   **Experimentation**: Trying new tools/methods.

**The Cycle:**
1.  **Measure**: Establish a baseline.
2.  **Hypothesize**: "If we change X, Y will improve."
3.  **Implement**: Make the change.
4.  **Verify**: Did it improve?
5.  **Repeat**.

---

## 🔨 Hands-On Implementation

### Part 1: The Inefficient Script (Baseline) 🐢

We will start with a script that simulates a slow data processing job.

1.  **Create `process_data.py`:**

    ```python
    import time
    import random

    def process_item(item_id):
        # Simulate complex calculation
        time.sleep(0.1) # 100ms delay per item
        return item_id * 2

    def main():
        print("🐢 Starting inefficient processing...")
        start_time = time.time()
        
        items = range(1, 51) # 50 items
        results = []
        
        for item in items:
            print(f"Processing item {item}...")
            res = process_item(item)
            results.append(res)
            
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ Finished processing {len(items)} items.")
        print(f"⏱️  Total Time: {duration:.2f} seconds")
        print(f"📊 Average per item: {duration/len(items):.4f} seconds")

    if __name__ == "__main__":
        main()
    ```

2.  **Run and Measure:**
    ```bash
    python3 process_data.py
    ```
    *Expected Result:* ~5.0 seconds (50 items * 0.1s).
    *Record this as your **Baseline**.*

### Part 2: Iteration 1 - Remove Unnecessary I/O 🔇

**Hypothesis:** Printing to the console (`print`) is a slow I/O operation. Removing it inside the loop should speed things up.

1.  **Modify `process_data.py`:**
    Comment out the print statement inside the loop.

    ```python
    # print(f"Processing item {item}...") 
    ```

2.  **Run and Verify:**
    ```bash
    python3 process_data.py
    ```
    *Result:* Maybe 5.0s -> 5.0s.
    *Analysis:* The `time.sleep(0.1)` is dominating the time. The print statement was negligible compared to the sleep.
    *Lesson:* **Profile before optimizing!** We guessed wrong.

### Part 3: Iteration 2 - Parallel Processing 🚀

**Hypothesis:** The tasks are independent. We can run them in parallel using threads or processes.

1.  **Refactor to use `concurrent.futures`:**

    ```python
    import time
    import concurrent.futures

    def process_item(item_id):
        time.sleep(0.1)
        return item_id * 2

    def main():
        print("🚀 Starting parallel processing...")
        start_time = time.time()
        
        items = range(1, 51)
        
        # Use ThreadPoolExecutor for I/O bound tasks (like sleep)
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(process_item, items))
            
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ Finished processing {len(items)} items.")
        print(f"⏱️  Total Time: {duration:.2f} seconds")
    
    if __name__ == "__main__":
        main()
    ```

2.  **Run and Verify:**
    ```bash
    python3 process_data.py
    ```
    *Expected Result:* ~0.5 seconds! (5.0s / 10 workers).
    *Improvement:* **10x speedup**.

### Part 4: Iteration 3 - Algorithmic Improvement 🧠

**Hypothesis:** Why are we sleeping? If the `sleep` represents a database call, can we batch it?

**Scenario:** Instead of 50 calls of 1 item, let's do 1 call of 50 items.

1.  **Refactor `process_item`:**

    ```python
    def process_batch(items):
        # Simulate batch processing overhead
        time.sleep(0.2) # Takes longer than single item, but done once
        return [i * 2 for i in items]

    def main():
        print("🧠 Starting batch processing...")
        start_time = time.time()
        
        items = list(range(1, 51))
        
        results = process_batch(items)
            
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ Finished processing {len(items)} items.")
        print(f"⏱️  Total Time: {duration:.2f} seconds")
    ```

2.  **Run and Verify:**
    *Expected Result:* ~0.2 seconds.
    *Improvement:* **25x speedup** from baseline.

---

## 🎯 Challenges

### Challenge 1: The Docker Build Optimization (Difficulty: ⭐⭐⭐)

**Scenario:** You have a Dockerfile that takes forever to build.

**Bad Dockerfile:**
```dockerfile
FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
```

**Problem:** Every time you change code (`COPY . /app`), Docker invalidates the cache for the next line (`RUN pip install...`). So it re-installs dependencies on every code change.

**Task:**
1.  Create this Dockerfile and a dummy `requirements.txt` (put `flask` in it).
2.  Build it (`docker build -t slow-build .`). Measure time.
3.  Change a file in the directory. Build again. Measure time.
4.  **Optimize it** using Docker Layer Caching principles.

### Challenge 2: Documenting the Improvement (Difficulty: ⭐⭐)

**Task:**
Create a `IMPROVEMENT_REPORT.md` summarizing your findings from the Python script optimization.

Format:
| Iteration | Method | Time | Speedup |
|-----------|--------|------|---------|
| Baseline | Serial | 5.0s | 1x |
| 1 | Remove Print | 5.0s | 1x |
| 2 | Parallel | 0.5s | 10x |
| 3 | Batching | 0.2s | 25x |

---

## 💡 Solution

<details>
<summary>Click to reveal Challenge 1 Solution (Docker)</summary>

**Optimized Dockerfile:**

```dockerfile
FROM python:3.9

WORKDIR /app

# 1. Copy ONLY requirements first
COPY requirements.txt .

# 2. Install dependencies (This layer is now cached unless requirements.txt changes)
RUN pip install -r requirements.txt

# 3. Copy the rest of the code
COPY . .

CMD ["python", "app.py"]
```

**Why it works:**
Docker builds in layers. If `requirements.txt` hasn't changed, Docker uses the cached layer for `RUN pip install`. When you modify your code (`COPY . .`), it only re-runs the last step, which is instant.

</details>

---

## 🔑 Key Takeaways

1.  **Measure First**: You cannot improve what you cannot measure.
2.  **Identify Bottlenecks**: Don't optimize things that don't matter (like the print statement).
3.  **Architecture Matters**: Changing the approach (Batching) often beats micro-optimizations.
4.  **Layer Caching**: In DevOps, understanding how tools cache (Docker, Maven, npm) saves hours of build time.

---

## ⏭️ Next Steps

We've improved the process. Now, how do we track this at an organizational level?

Proceed to **Lab 1.8: DevOps Metrics (DORA)**.
