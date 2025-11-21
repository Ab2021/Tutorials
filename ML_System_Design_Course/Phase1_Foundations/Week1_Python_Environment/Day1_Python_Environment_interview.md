# Day 1: Python & Environment - Interview Questions

> **Topic**: Python Production Standards
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. What is the difference between `list` and `tuple` in Python, and why would you use one over the other in a production ML pipeline?
**Answer:**
*   **Mutability**: Lists are mutable (can be changed), tuples are immutable (cannot be changed).
*   **Performance**: Tuples are slightly more memory efficient and faster to iterate over because of their immutability.
*   **Usage in ML**: Use **tuples** for fixed collections of items, like configuration parameters `(learning_rate, batch_size)` or dictionary keys (since lists can't be keys). Use **lists** for collections that need to grow or change, like a buffer of training examples.

### 2. Explain how Python's memory management works. What is the role of the Garbage Collector vs Reference Counting?
**Answer:**
*   **Reference Counting**: This is the primary mechanism. Every object has a counter. When you assign it to a variable, count +1. When it goes out of scope, count -1. When count == 0, memory is freed immediately.
*   **Garbage Collector (GC)**: Handles **cyclic references** (e.g., A points to B, B points to A). Reference counting can't detect this (count never hits 0). The GC periodically scans for these cycles and frees them.
*   **Generational GC**: Python uses 3 generations (0, 1, 2). New objects are in Gen 0. If they survive a collection, they move to Gen 1, etc.

### 3. What is a Python Decorator? Write a simple decorator to time the execution of a training function.
**Answer:**
A decorator is a function that takes another function and extends its behavior without explicitly modifying it.
```python
import time
import functools

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@timer
def train_model():
    time.sleep(1)
```

### 4. Why is `requirements.txt` often considered insufficient for reproducible ML builds? What is the advantage of `poetry.lock` or `uv.lock`?
**Answer:**
*   **Problem**: `requirements.txt` often lists direct dependencies (e.g., `pandas==1.3.0`) but not *their* dependencies (transitive dependencies). If `pandas` relies on `numpy`, and `numpy` releases a breaking change, your build might break even if `requirements.txt` didn't change.
*   **Solution**: Lock files (`poetry.lock`, `uv.lock`) record the **exact version** and hash of *every* single package in the dependency tree. This guarantees that `pip install` results in the exact same environment on every machine.

### 5. What is the Global Interpreter Lock (GIL)? How does it impact multi-threaded data loading in PyTorch?
**Answer:**
*   **GIL**: A mutex that prevents multiple native threads from executing Python bytecodes at once. This means Python is effectively single-threaded for CPU-bound tasks.
*   **Impact**: For CPU-heavy tasks (like image augmentation), standard Python threads won't give a speedup.
*   **PyTorch Solution**: PyTorch `DataLoader` uses `num_workers > 0`, which spawns **sub-processes** (not threads). Each process has its own Python interpreter and GIL, bypassing the limitation.

### 6. Explain the difference between `is` and `==` in Python. Give an example where this matters.
**Answer:**
*   `==` checks for **value equality** (do they hold the same data?).
*   `is` checks for **reference equality** (are they the exact same object in memory?).
*   **Example**:
    ```python
    a = [1, 2, 3]
    b = [1, 2, 3]
    a == b  # True (Values are same)
    a is b  # False (Different memory addresses)
    
    c = None
    c is None # True (None is a singleton)
    ```

### 7. What are Python Generators? How can they help when processing a 100GB CSV file?
**Answer:**
*   **Generators**: Functions that use `yield` instead of `return`. They return an iterator that produces items one by one, lazily.
*   **100GB CSV**: Loading the whole file into RAM (List) would crash the machine. A generator reads one line, processes it, yields the result, and discards the line from memory. This keeps RAM usage constant (e.g., 10MB) regardless of file size.

### 8. What is the purpose of `__init__.py`? Is it still required in Python 3.3+?
**Answer:**
*   **Purpose**: It marks a directory as a Python **package**, allowing you to import modules from it. It also initializes the package (runs code when the package is imported).
*   **Python 3.3+**: It is **not strictly required** for "Namespace Packages", but it is still **highly recommended** for regular packages to define clear APIs and avoid ambiguity.

### 9. Explain the difference between `@staticmethod` and `@classmethod`.
**Answer:**
*   `@staticmethod`: Does not receive an implicit first argument. It's just a plain function that happens to live inside a class namespace.
*   `@classmethod`: Receives the class itself (`cls`) as the first argument. It can access class state (like class variables) or be used as an alternative constructor (Factory pattern).

### 10. How do you handle circular imports in Python?
**Answer:**
*   **Refactoring**: The best fix is usually to move the shared dependency to a third module.
*   **Deferred Import**: Import the module *inside* the function or method where it's needed, rather than at the top of the file.
*   **Typing**: Use `from __future__ import annotations` and `if TYPE_CHECKING:` blocks for type hints to avoid runtime import cycles.

### 11. What is a Context Manager (`with` statement)? How would you write a custom one for opening a database connection?
**Answer:**
*   **Concept**: Manages resources (setup and teardown) automatically. Ensures cleanup happens even if an error occurs.
*   **Implementation**: Define `__enter__` and `__exit__` methods.
    ```python
    class DBConnection:
        def __enter__(self):
            self.conn = connect_to_db()
            return self.conn
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.conn.close()
    ```

### 12. What is the difference between `copy.copy()` and `copy.deepcopy()`? Why does this matter for nested dictionaries of hyperparameters?
**Answer:**
*   `copy()`: Shallow copy. Creates a new container, but inserts *references* to the objects found in the original. If you modify a nested object, it changes in both.
*   `deepcopy()`: Recursive copy. Copies the object and *recursively* copies everything inside it.
*   **Hyperparameters**: If you have `config = {'model': {'lr': 0.01}}`, a shallow copy `new_config = config.copy()` followed by `new_config['model']['lr'] = 0.02` would **overwrite the original config** too. Use `deepcopy`.

### 13. Explain the concept of "Duck Typing" in Python.
**Answer:**
*   "If it walks like a duck and quacks like a duck, it's a duck."
*   Python doesn't check types explicitly (like Java interfaces). It checks **behavior**. If an object has a `read()` method, it can be treated as a file, regardless of whether it inherits from `io.FileIO`.

### 14. What are Type Hints (PEP 484)? Why are they critical for large-scale ML codebases?
**Answer:**
*   **Type Hints**: Annotations (e.g., `def train(lr: float) -> None:`) that indicate expected types.
*   **Importance**: Python is dynamically typed, which leads to runtime `AttributeError`. Type hints allow static analysis tools (like `mypy` or `pyright`) to catch bugs *before* running the code. They also serve as documentation.

### 15. How does `multiprocessing` differ from `threading` in Python? Which one should you use for CPU-bound tasks?
**Answer:**
*   **Threading**: Uses threads within the same process. Shared memory. Blocked by GIL. Good for **I/O-bound** tasks (network requests).
*   **Multiprocessing**: Spawns separate processes. Separate memory space. Bypasses GIL. Good for **CPU-bound** tasks (data preprocessing, matrix math).

### 16. What is the `__call__` method? How is it used in PyTorch `nn.Module`?
**Answer:**
*   **Concept**: Makes an object callable like a function (e.g., `obj()`).
*   **PyTorch**: When you define a model `class Net(nn.Module)`, you implement `forward()`. But you run it as `output = model(input)`. This works because `nn.Module` implements `__call__`, which handles hooks and then calls `forward()`.

### 17. Explain the difference between `*args` and `**kwargs`.
**Answer:**
*   `*args`: Passes a variable number of **positional** arguments (as a tuple).
*   `**kwargs`: Passes a variable number of **keyword** arguments (as a dictionary).
*   Useful for writing flexible wrappers or passing arguments through to a parent class.

### 18. What is a Python "Wheel" (.whl)? How is it different from a source distribution (.tar.gz)?
**Answer:**
*   **Source Dist**: Contains raw source code. Requires compilation (C extensions) on the user's machine during install. Slow and error-prone.
*   **Wheel**: A **pre-compiled** binary package. Installs instantly. No compilation needed. Essential for libraries like NumPy/PyTorch.

### 19. How do you debug a "Segmentation Fault" in a Python script using C extensions (like NumPy)?
**Answer:**
*   **Cause**: Usually invalid memory access in the C layer (e.g., accessing array out of bounds in Cython).
*   **Tools**: `gdb` (GNU Debugger) or `faulthandler` module in Python (`import faulthandler; faulthandler.enable()`). This prints the Python traceback when the segfault occurs.

### 20. What is the MRO (Method Resolution Order) in Python inheritance?
**Answer:**
*   **MRO**: The order in which Python looks for a method in a hierarchy of classes.
*   **Algorithm**: C3 Linearization.
*   **Inspection**: You can see it via `ClassName.mro()`. It ensures that children are checked before parents and keeps the order consistent in multiple inheritance (Diamond problem).
