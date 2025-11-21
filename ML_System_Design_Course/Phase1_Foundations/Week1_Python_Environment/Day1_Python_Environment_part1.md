# Day 1 (Part 1): Advanced Python Internals & Edge Cases

> **Phase**: 6 - Deep Dive
> **Topic**: Advanced Python for ML Systems
> **Focus**: Memory, Concurrency, and Language Internals
> **Reading Time**: 60 mins

---

## 1. Python Memory Management: Beyond the Basics

In production ML, memory leaks are silent killers. Understanding how Python frees memory is critical.

### 1.1 Reference Counting vs. Garbage Collection
*   **Reference Counting**: The primary mechanism. Every object has a `ob_refcnt`. When it hits 0, memory is freed immediately.
*   **Cyclic Garbage Collector (GC)**: Handles reference cycles (A -> B -> A).
    *   **Generational GC**: Python uses 3 generations (0, 1, 2). Objects that survive GC scans are promoted.
    *   **The Trap**: In high-throughput systems, the GC pausing to scan Generation 2 can cause latency spikes.
    *   **Optimization**: `gc.freeze()` allows you to freeze the "startup" objects so the GC ignores them, reducing scan time.

### 1.2 The `__del__` Trap
*   **Scenario**: You implement a `__del__` method to close a DB connection.
*   **Edge Case**: If an exception occurs during `__init__`, `__del__` might still be called on a partially initialized object.
*   **Circular References**: Prior to Python 3.4, objects with `__del__` in a cycle could *never* be collected. (Fixed now, but still good to know).

---

## 2. Multiprocessing: Fork vs. Spawn

This is the #1 cause of "My DataLoader hangs" or "CUDA initialization error".

### 2.1 The Start Methods
*   **Fork (Unix Default)**: Copies the parent process memory (COW - Copy On Write). Fast.
    *   **Danger**: If the parent process has initialized CUDA context, forking corrupts it. Child processes cannot use the same CUDA context.
*   **Spawn (Windows/Mac Default)**: Starts a fresh python interpreter. Slower start.
    *   **Safety**: Safe for CUDA.
*   **Forkserver**: Compromise.

### 2.2 Shared Memory
*   **Problem**: Pickling large tensors to send between processes is slow.
*   **Solution**: `multiprocessing.shared_memory`.
    *   Allocates a block of RAM that multiple processes can map.
    *   NumPy/PyTorch can wrap this buffer without copying data.

---

## 3. Advanced Typing: Structural Subtyping

### 3.1 Protocols (Duck Typing)
*   **Scenario**: You want a function to accept *anything* that has a `.fit()` method (Sklearn model, XGBoost, Custom).
*   **Bad**: `Union[SklearnModel, XGBoostModel]`.
*   **Good**: `Protocol`.
    ```python
    class Fittable(Protocol):
        def fit(self, X, y): ...

    def train(model: Fittable): ...
    ```

### 3.2 Overloads
*   **Scenario**: `np.array([1, 2])` returns `ndarray`. `np.array(1)` returns `ndarray` (0-d).
*   **Solution**: `@overload` decorator to define multiple signatures for the same function so Mypy understands the return type based on input.

---

## 4. Tricky Interview Questions

### Q1: How does Python handle Circular Imports?
> **Answer**: Python executes modules line-by-line. If Module A imports B, and B imports A:
> 1.  A starts executing.
> 2.  A hits `import B`. Execution pauses.
> 3.  B starts executing.
> 4.  B hits `import A`.
> 5.  **The Catch**: Python checks `sys.modules`. A is present (but partially initialized). B gets a reference to the *empty* A module.
> 6.  B tries to access `A.foo`. **AttributeError**.
> **Fix**: Move imports inside functions or use absolute imports with careful structuring.

### Q2: What is the Method Resolution Order (MRO)?
> **Answer**: The order in which Python looks for a method in a class hierarchy.
> *   **Algorithm**: C3 Linearization.
> *   **Diamond Problem**: If D inherits from B and C, and both inherit from A. MRO ensures A is called only once.
> *   **Check**: `ClassName.mro()`.

### Q3: Why is `dict` access O(1)?
> **Answer**: It uses a Hash Table with Open Addressing (Quadratic Probing).
> *   **Hash**: `hash(key)` gives an integer.
> *   **Index**: `hash & (size - 1)`.
> *   **Collision**: If slot is full, probe `i + 1`, `i + 4`, etc.
> *   **Resize**: When 2/3 full, size doubles.

---

## 5. Practical Edge Case: The Mutable Default Argument
```python
def append_to(element, to=[]):
    to.append(element)
    return to
```
*   **Call 1**: `append_to(1)` -> `[1]`
*   **Call 2**: `append_to(2)` -> `[1, 2]` (Wait, what?)
*   **Reason**: The list `[]` is created *once* at definition time, not at call time.
*   **Fix**: Use `to=None`.

