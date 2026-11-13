# Pythonic Programming Fundamentals — Senior & Lead Data Scientist Interview Prep
### Target Companies: General Mills India (GIC Powai) | LSEG India (Bangalore & Mumbai)
### Role Focus: Senior Data Scientist / Lead Data Scientist / Staff ML Engineer
### Last Updated: November 2026

---

> [!IMPORTANT]
> **Why Pythonic Fundamentals Matter for Senior / Lead Roles:**
> Junior interviews test *syntax* ("can you filter a dataframe?"). Senior and Lead interviews at **General Mills** (supply chain/CPG platforms) and **LSEG** (high-throughput financial platforms) probe **architectural fluency**:
> 1. **Data Model & Dunder Protocols:** Writing extensible classes that integrate seamlessly with native Python and PyTorch/Pandas idioms.
> 2. **Memory & Concurrency:** Understanding the GIL, reference counting, cyclic garbage collection, and `__slots__` to process high-volume streams without memory bloat.
> 3. **Metaprogramming & Type Safety:** Using descriptors, metaclasses, and `typing.Protocol` to design robust, self-validating ML framework code.
> 4. **Standard Library Mastery:** Codility rounds at LSEG explicitly forbid Pandas in round 1 to evaluate pure Python (`itertools`, `re`, `datetime`, `collections`).

---

## Table of Contents

1. [The Python Data Model & Dunder Methods](#1-the-python-data-model--dunder-methods)
   - [Q1 — Emulating Containers: Lazy-Loading Dataset with `__getitem__`, `__len__`, and `__iter__` [Medium]](#q1--emulating-containers-lazy-loading-dataset-with-__getitem__-__len__-and-__iter__-medium)
   - [Q2 — The Object Equality & Hashing Contract: `__repr__`, `__eq__`, and `__hash__` [Medium]](#q2--the-object-equality--hashing-contract-__repr__-__eq__-and-__hash__-medium)
   - [Q3 — Stateful Callables: Pipeline Transformer with `__call__` [Easy]](#q3--stateful-callables-pipeline-transformer-with-__call__-easy)
   - [Q4 — Object Creation Mechanics: `__new__` vs `__init__` & Thread-Safe Singleton Model Loader [Hard]](#q4--object-creation-mechanics-__new__-vs-__init__--thread-safe-singleton-model-loader-hard)
2. [Descriptors & Attribute Control](#2-descriptors--attribute-control)
   - [Q5 — The Descriptor Protocol: Bounded & Type-Validated Hyperparameter Descriptor [Hard]](#q5--the-descriptor-protocol-bounded--type-validated-hyperparameter-descriptor-hard)
   - [Q6 — `@property` Under the Hood: Building a Property Descriptor from Scratch [Medium]](#q6--property-under-the-hood-building-a-property-descriptor-from-scratch-medium)
3. [Metaclasses & Abstract Base Classes](#3-metaclasses--abstract-base-classes)
   - [Q7 — Custom Metaclass: Auto-Registering ML Models at Definition Time [Hard]](#q7--custom-metaclass-auto-registering-ml-models-at-definition-time-hard)
   - [Q8 — Abstract Base Classes (`abc.ABC`): Enforcing Estimator Pipeline Contracts [Medium]](#q8--abstract-base-classes-abcabc-enforcing-estimator-pipeline-contracts-medium)
4. [Functional Programming & Standard Library Power Tools](#4-functional-programming--standard-library-power-tools)
   - [Q9 — `itertools.groupby`: The Sorting Prerequisite Gotcha in Transaction Bucketing [Medium]](#q9--itertoolsgroupby-the-sorting-prerequisite-gotcha-in-transaction-bucketing-medium)
   - [Q10 — `itertools` for Data Science: `product`, `combinations`, and `islice` [Easy]](#q10--itertools-for-data-science-product-combinations-and-islice-easy)
   - [Q11 — `functools.singledispatch`: Polymorphic Data Preprocessor [Medium]](#q11--functoolssingledispatch-polymorphic-data-preprocessor-medium)
   - [Q12 — `functools.partial` & Metadata-Preserving Decorators with `@wraps` [Easy]](#q12--functoolspartial--metadata-preserving-decorators-with-wraps-easy)
5. [Modern Typing & Static Analysis (Python 3.10+)](#5-modern-typing--static-analysis-python-310)
   - [Q13 — `typing.Protocol`: Structural Subtyping (Duck Typing Formalized) for Estimators [Medium]](#q13--typingprotocol-structural-subtyping-duck-typing-formalized-for-estimators-medium)
   - [Q14 — Generics, `TypeVar`, and Union Operator (`|`) in Production Pipelines [Medium]](#q14--generics-typevar-and-union-operator--in-production-pipelines-medium)
6. [Memory Management, GIL & Performance Internals](#6-memory-management-gil--performance-internals)
   - [Q15 — `__slots__`: 65% Memory Reduction for 1,000,000 Financial Records [Medium]](#q15--__slots__-65-memory-reduction-for-1000000-financial-records-medium)
   - [Q16 — Python Garbage Collection, Cyclic References & Memory Leaks in Notebooks [Hard]](#q16--python-garbage-collection-cyclic-references--memory-leaks-in-notebooks-hard)
   - [Q17 — The GIL Deep Dive: Threading vs Multiprocessing vs C-Extensions Benchmark [Hard]](#q17--the-gil-deep-dive-threading-vs-multiprocessing-vs-c-extensions-benchmark-hard)
7. [LSEG / Codility Special: Pure Python Parsing & Data Wrangling](#7-lseg--codility-special-pure-python-parsing--data-wrangling)
   - [Q18 — Unstructured Log & Market Data Parsing with Regex Named Groups & `datetime` [Hard]](#q18--unstructured-log--market-data-parsing-with-regex-named-groups--datetime-hard)
8. [Modern Syntax & Idiomatic Control Flow](#8-modern-syntax--idiomatic-control-flow)
   - [Q19 — Walrus Operator (`:=`) & Structural Pattern Matching (`match-case`) [Medium]](#q19--walrus-operator---structural-pattern-matching-match-case-medium)
   - [Q20 — `collections` Arsenal: `deque` Running Windows, `defaultdict`, and `ChainMap` [Easy]](#q20--collections-arsenal-deque-running-windows-defaultdict-and-chainmap-easy)
9. [Asynchronous Data Ingestion & Context Managers](#9-asynchronous-data-ingestion--context-managers)
   - [Q21 — `asyncio` with Rate-Limiting Semaphores for High-Throughput API Ingestion [Hard]](#q21--asyncio-with-rate-limiting-semaphores-for-high-throughput-api-ingestion-hard)
   - [Q22 — Contextlib Suite: `@contextmanager`, `suppress`, and Dynamic `ExitStack` [Medium]](#q22--contextlib-suite-contextmanager-suppress-and-dynamic-exitstack-medium)
   - [Q23 — Custom Exception Hierarchies & Explicit Exception Chaining (`raise ... from`) [Easy]](#q23--custom-exception-hierarchies--explicit-exception-chaining-raise--from-easy)

---

## 1. The Python Data Model & Dunder Methods

### Q1 — Emulating Containers: Lazy-Loading Dataset with `__getitem__`, `__len__`, and `__iter__` [Medium]

**Interview Context:**
Common in PyTorch dataset rounds and General Mills/LSEG take-home coding rounds. When working with 50GB+ of daily retail scanner logs or tick-level trade logs, loading the full dataset into RAM crashes the instance. You must implement a custom container that supports indexing, slicing, iteration, and length retrieval by reading records lazily from disk.

```python
import os
import tempfile
from typing import Any, Iterator, List, Union

class LazyRecordDataset:
    """
    Emulates a sequence container (like a list or PyTorch Dataset) that reads
    fixed-width or line-delimited records from disk on-demand.
    Implements: __len__, __getitem__ (single index + slicing), __iter__.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._offsets: List[int] = []
        self._build_index()

    def _build_index(self) -> None:
        """Build byte offset index to enable O(1) random access jumps."""
        with open(self.filepath, "rb") as f:
            offset = 0
            for line in f:
                self._offsets.append(offset)
                offset += len(line)

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: Union[int, slice]) -> Union[str, List[str]]:
        if isinstance(index, slice):
            # Support Pythonic slicing: dataset[10:20]
            start, stop, step = index.indices(len(self._offsets))
            return [self._read_record(i) for i in range(start, stop, step)]
        elif isinstance(index, int):
            # Handle negative indexing: dataset[-1]
            if index < 0:
                index += len(self._offsets)
            if not 0 <= index < len(self._offsets):
                raise IndexError(f"Dataset index {index} out of range [0, {len(self._offsets)})")
            return self._read_record(index)
        else:
            raise TypeError(f"Invalid index type: {type(index).__name__}")

    def _read_record(self, idx: int) -> str:
        with open(self.filepath, "r", encoding="utf-8") as f:
            f.seek(self._offsets[idx])
            return f.readline().strip()

    def __iter__(self) -> Iterator[str]:
        """Generator-based iterator: yields one line at a time with O(1) memory."""
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                yield line.strip()

    def __repr__(self) -> str:
        return f"LazyRecordDataset(file='{os.path.basename(self.filepath)}', records={len(self)})"

# --- Verification ---
if __name__ == "__main__":
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".csv") as tmp:
        tmp.write("order_1,SKU101,Cheerios,4.99\n")
        tmp.write("order_2,SKU102,NatureValley,3.49\n")
        tmp.write("order_3,SKU103,Yoplait,1.29\n")
        tmp.write("order_4,SKU104,BettyCrocker,2.89\n")
        tmp_name = tmp.name

    ds = LazyRecordDataset(tmp_name)
    print(ds)                           # LazyRecordDataset(file='...', records=4)
    print(f"Length: {len(ds)}")         # 4
    print(f"Index 1: {ds[1]}")          # order_2,SKU102,NatureValley,3.49
    print(f"Index -1: {ds[-1]}")        # order_4,SKU104,BettyCrocker,2.89
    print(f"Slice [1:3]: {ds[1:3]}")    # ['order_2,...', 'order_3,...']
    print(f"Iterated: {list(ds)}")
    os.remove(tmp_name)
```

---

### Q2 — The Object Equality & Hashing Contract: `__repr__`, `__eq__`, and `__hash__` [Medium]

**Interview Context:**
Frequently probed in LSEG technical discussions. Interviewers ask: *"If you override `__eq__` on a custom feature or financial instrument class, why can't you use it in a `set` or as a dictionary key anymore? How do you fix it?"*

**The Contract:**
1. If two objects compare equal (`a == b`), they **must** return the same hash value (`hash(a) == hash(b)`).
2. If a class defines `__eq__` but does not define `__hash__`, Python implicitly sets `__hash__ = None`, rendering instances **unhashable**.
3. For an object to be hashable, it must be **immutable** over its lifecycle.

```python
from typing import Tuple

class FinancialInstrument:
    """
    Immutable, hashable representation of a tradable instrument.
    Demonstrates correct implementation of __repr__, __str__, __eq__, and __hash__.
    """
    __slots__ = ("_ticker", "_isin", "_currency", "_hash")

    def __init__(self, ticker: str, isin: str, currency: str):
        # Emulate immutability
        object.__setattr__(self, "_ticker", ticker.upper().strip())
        object.__setattr__(self, "_isin", isin.upper().strip())
        object.__setattr__(self, "_currency", currency.upper().strip())
        # Cache the hash for high-frequency lookup speed
        object.__setattr__(self, "_hash", hash((self._ticker, self._isin, self._currency)))

    @property
    def ticker(self) -> str: return self._ticker
    @property
    def isin(self) -> str: return self._isin
    @property
    def currency(self) -> str: return self._currency

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{self.__class__.__name__} is immutable")

    def __repr__(self) -> str:
        """Unambiguous string representation for developers/logs."""
        return f"FinancialInstrument(ticker='{self._ticker}', isin='{self._isin}', currency='{self._currency}')"

    def __str__(self) -> str:
        """User-friendly representation."""
        return f"{self._ticker} ({self._isin})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FinancialInstrument):
            return NotImplemented  # Allows fallback to other.__eq__
        return (self._ticker, self._isin, self._currency) == (other._ticker, other._isin, other._currency)

    def __hash__(self) -> int:
        return self._hash

# --- Verification ---
if __name__ == "__main__":
    inst1 = FinancialInstrument("LSEG", "GB00B0SWJX34", "GBP")
    inst2 = FinancialInstrument("lseg ", "gb00b0swjx34", "gbp")
    inst3 = FinancialInstrument("GIS", "US3703341046", "USD")

    print(f"inst1 == inst2: {inst1 == inst2}")  # True
    print(f"hash(inst1) == hash(inst2): {hash(inst1) == hash(inst2)}")  # True
    
    # Can be used in sets and dict keys
    portfolio = {inst1: 150000, inst3: 85000}
    print(f"Portfolio lookup inst2: {portfolio.get(inst2)}")  # 150000
    
    try:
        inst1.ticker = "NEW_TICKER"
    except AttributeError as e:
        print(f"Immutability check passed: {e}")
```

---

### Q3 — Stateful Callables: Pipeline Transformer with `__call__` [Easy]

**Interview Context:**
Asked in ML engineering discussions. How do you design an object that behaves like a function, retains internal state (e.g., fitted scaling parameters), and can be passed directly as a callback to `apply()`, `map()`, or PyTorch transform pipelines?

```python
import numpy as np

class AdaptiveWinsorizer:
    """
    Stateful callable that caps numerical features at percentile thresholds.
    Supports .fit() then callable instance usage __call__(x).
    """
    def __init__(self, lower_pct: float = 1.0, upper_pct: float = 99.0):
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct
        self.lower_bound_: float = None
        self.upper_bound_: float = None

    def fit(self, x: np.ndarray) -> "AdaptiveWinsorizer":
        self.lower_bound_ = float(np.percentile(x, self.lower_pct))
        self.upper_bound_ = float(np.percentile(x, self.upper_pct))
        return self

    def __call__(self, val: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Allows instance to be called directly: winsorizer(105.2)"""
        if self.lower_bound_ is None:
            raise RuntimeError("Winsorizer must be fitted before calling!")
        return np.clip(val, self.lower_bound_, self.upper_bound_)

    def __repr__(self) -> str:
        return f"AdaptiveWinsorizer(lower={self.lower_bound_:.2f}, upper={self.upper_bound_:.2f})"

# --- Verification ---
if __name__ == "__main__":
    raw_sales = np.array([10.0, 12.0, 11.5, 9.8, 1000.0, -50.0, 14.0])
    winsor = AdaptiveWinsorizer(lower_pct=5.0, upper_pct=95.0).fit(raw_sales)
    print(winsor)
    
    # Called as a function on single values or arrays
    print(f"Capped 500.0: {winsor(500.0):.2f}")
    print(f"Capped -100.0: {winsor(-100.0):.2f}")
    print(f"Capped Array: {winsor(raw_sales)}")
```

---

### Q4 — Object Creation Mechanics: `__new__` vs `__init__` & Thread-Safe Singleton Model Loader [Hard]

**Interview Context:**
Common in high-concurrency model deployment interviews. Interviewers ask:
*"What is the difference between `__new__` and `__init__`? How do you guarantee in Python that an expensive 10GB neural network or tokenizer is instantiated and loaded into GPU memory strictly once, even when multiple web workers or threads request it simultaneously?"*

* `__new__(cls, *args, **kwargs)` is a static method responsible for **allocating and returning** the instance.
* `__init__(self, *args, **kwargs)` is responsible for **initializing** the already-created instance.
* To prevent re-running `__init__` on subsequent accesses to the Singleton, you must guard initialization with a flag or perform it inside `__new__`.

```python
import threading
import time

class SingletonModelLoader:
    """
    Thread-safe Singleton implementation using __new__ and a double-checked lock.
    Ensures heavy model weights are allocated in memory exactly once.
    """
    _instance = None
    _lock = threading.Lock()
    _is_initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = "FinBERT-Sentiment"):
        # Guard against __init__ running multiple times on the same instance
        if not self._is_initialized:
            with self._lock:
                if not self._is_initialized:
                    print(f"--> [HEAVY LOAD] Loading weights for '{model_name}' into memory...")
                    time.sleep(0.5)  # Simulate network/disk load of model weights
                    self.model_name = model_name
                    self.weights = {"layer1": "weights_matrix_8GB"}
                    self._is_initialized = True
                    print(f"--> [HEAVY LOAD] Finished loading '{model_name}'.")

# --- Verification ---
if __name__ == "__main__":
    def worker_task(thread_id: int):
        loader = SingletonModelLoader("GeneralMills-DemandLSTM")
        print(f"Thread {thread_id} got loader ID: {id(loader)}")

    threads = [threading.Thread(target=worker_task, args=(i,)) for i in range(5)]
    for t in threads: t.start()
    for t in threads: t.join()
```

---

## 2. Descriptors & Attribute Control

### Q5 — The Descriptor Protocol: Bounded & Type-Validated Hyperparameter Descriptor [Hard]

**Interview Context:**
Top-tier interview question for Senior ML Engineers and Lead Data Scientists.
*"How does Python implement `@property`? Write a reusable descriptor that enforces type checking and value bounds (e.g., `learning_rate` must be float in `(0.0, 1.0]`, `max_depth` must be positive integer) on any configuration class without repetitive boilerplate."*

**The Descriptor Protocol:**
An object implementing at least one of:
* `__get__(self, instance, owner)`
* `__set__(self, instance, value)`
* `__delete__(self, instance)`
* `__set_name__(self, owner, name)` (introduced in Python 3.6 to automatically capture attribute names)

```python
from typing import Type, Optional

class BoundedNumeric:
    """
    Reusable descriptor enforcing type and range constraints on attributes.
    """
    def __init__(
        self,
        expected_type: Type[Union[int, float]],
        min_val: Optional[Union[int, float]] = None,
        max_val: Optional[Union[int, float]] = None,
        strict_bounds: bool = False
    ):
        self.expected_type = expected_type
        self.min_val = min_val
        self.max_val = max_val
        self.strict_bounds = strict_bounds
        self.name = ""

    def __set_name__(self, owner: Type[Any], name: str) -> None:
        # Automatically called when the owning class is created
        self.name = name

    def __get__(self, instance: Optional[Any], owner: Type[Any]) -> Any:
        if instance is None:
            return self  # Accessed via class: ModelConfig.learning_rate returns descriptor
        return instance.__dict__.get(self.name)

    def __set__(self, instance: Any, value: Any) -> None:
        if not isinstance(value, self.expected_type):
            raise TypeError(
                f"Attribute '{self.name}' must be of type {self.expected_type.__name__}, got {type(value).__name__}"
            )
        
        if self.min_val is not None:
            if self.strict_bounds and value <= self.min_val:
                raise ValueError(f"Attribute '{self.name}' must be strictly > {self.min_val}, got {value}")
            elif not self.strict_bounds and value < self.min_val:
                raise ValueError(f"Attribute '{self.name}' must be >= {self.min_val}, got {value}")
                
        if self.max_val is not None:
            if self.strict_bounds and value >= self.max_val:
                raise ValueError(f"Attribute '{self.name}' must be strictly < {self.max_val}, got {value}")
            elif not self.strict_bounds and value > self.max_val:
                raise ValueError(f"Attribute '{self.name}' must be <= {self.max_val}, got {value}")

        instance.__dict__[self.name] = value

class ModelHyperparameters:
    # Descriptors act as declarative class attributes
    learning_rate = BoundedNumeric(float, min_val=0.0, max_val=1.0, strict_bounds=True)
    max_depth = BoundedNumeric(int, min_val=1, max_val=50)
    subsample_ratio = BoundedNumeric(float, min_val=0.1, max_val=1.0)

    def __init__(self, learning_rate: float, max_depth: int, subsample_ratio: float):
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample_ratio = subsample_ratio

# --- Verification ---
if __name__ == "__main__":
    hp = ModelHyperparameters(0.05, 6, 0.8)
    print(f"Config valid: lr={hp.learning_rate}, depth={hp.max_depth}, subsample={hp.subsample_ratio}")

    try:
        hp.learning_rate = 1.5  # Exceeds max_val
    except ValueError as e:
        print(f"Validation catch: {e}")

    try:
        hp.max_depth = "deep"  # Wrong type
    except TypeError as e:
        print(f"Type catch: {e}")
```

---

### Q6 — `@property` Under the Hood: Building a Property Descriptor from Scratch [Medium]

**Interview Context:**
*"Explain how `@property`, `@prop.setter`, and `@prop.deleter` work in Python. Implement your own `CustomProperty` descriptor class that replicates this behavior."*

```python
class CustomProperty:
    """A clean, pure-Python reimplementation of the built-in property descriptor."""
    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        self.__doc__ = doc or (fget.__doc__ if fget else None)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        if self.fget is None:
            raise AttributeError("Unreadable attribute")
        return self.fget(instance)

    def __set__(self, instance, value):
        if self.fset is None:
            raise AttributeError("Can't set attribute (read-only property)")
        self.fset(instance, value)

    def __delete__(self, instance):
        if self.fdel is None:
            raise AttributeError("Can't delete attribute")
        self.fdel(instance)

    def setter(self, fset):
        return type(self)(self.fget, fset, self.fdel, self.__doc__)

    def deleter(self, fdel):
        return type(self)(self.fget, self.fset, fdel, self.__doc__)

# --- Verification ---
class TradePosition:
    def __init__(self, quantity: int, price: float):
        self._quantity = quantity
        self._price = price

    @CustomProperty
    def notional(self) -> float:
        """Calculates trade notional value."""
        return self._quantity * self._price

    @CustomProperty
    def quantity(self) -> int:
        return self._quantity

    @quantity.setter
    def quantity(self, value: int):
        if value < 0:
            raise ValueError("Position quantity cannot be negative")
        self._quantity = value

if __name__ == "__main__":
    pos = TradePosition(100, 45.5)
    print(f"Notional: {pos.notional}")  # 4550.0
    pos.quantity = 200
    print(f"Updated Notional: {pos.notional}")  # 9100.0
    try:
        pos.notional = 10000.0  # Read-only
    except AttributeError as e:
        print(f"Read-only check passed: {e}")
```

---

## 3. Metaclasses & Abstract Base Classes

### Q7 — Custom Metaclass: Auto-Registering ML Models at Definition Time [Hard]

**Interview Context:**
Common in architecture/system design interviews for MLOps and internal framework engineering.
*"In a large enterprise codebase (like General Mills Demand Sensing or LSEG Alpha Generation), data scientists create dozens of model classes across different files. How do you automatically discover and register every new model class at import/definition time without requiring manual dictionary updates?"*

A **metaclass** (`type`) intercepts class definition itself. In `__init__` or `__new__` of the metaclass, you can inspect the newly created class and register it into a global registry.

```python
from typing import Dict, Type

class ModelRegistryMeta(type):
    """
    Metaclass that automatically discovers and registers subclasses.
    Also validates that every registered model defines a class-level 'model_family'.
    """
    REGISTRY: Dict[str, Type[Any]] = {}

    def __new__(mcs, name: str, bases: tuple, attrs: dict):
        cls = super().__new__(mcs, name, bases, attrs)
        
        # Don't register the base abstract class itself
        if bases:
            model_key = attrs.get("MODEL_KEY", name.lower())
            if "model_family" not in attrs:
                raise TypeError(f"Class '{name}' must define a class attribute 'model_family'")
            
            if model_key in mcs.REGISTRY:
                raise ValueError(f"Duplicate model key '{model_key}' already registered by {mcs.REGISTRY[model_key]}")
                
            mcs.REGISTRY[model_key] = cls
            print(f"[REGISTRY] Auto-registered model '{name}' under key '{model_key}'")
            
        return cls

class BaseModel(metaclass=ModelRegistryMeta):
    """Base class for all enterprise models."""
    pass

# User files simply inherit; no manual dictionary appending needed
class LightGBMForecaster(BaseModel):
    MODEL_KEY = "lgbm_demand"
    model_family = "gradient_boosting"

class ProphetForecaster(BaseModel):
    MODEL_KEY = "prophet_promo"
    model_family = "additive_spline"

# --- Verification ---
if __name__ == "__main__":
    print(f"\nFinal Global Registry:")
    for k, v in ModelRegistryMeta.REGISTRY.items():
        print(f"  {k} -> {v.__name__} (family: {v.model_family})")

    # Factory function leveraging the metaclass registry
    def get_model(key: str, **kwargs) -> BaseModel:
        if key not in ModelRegistryMeta.REGISTRY:
            raise KeyError(f"Unknown model: {key}. Available: {list(ModelRegistryMeta.REGISTRY.keys())}")
        return ModelRegistryMeta.REGISTRY[key](**kwargs)

    m = get_model("lgbm_demand")
    print(f"Instantiated: {type(m).__name__}")
```

---

### Q8 — Abstract Base Classes (`abc.ABC`): Enforcing Estimator Pipeline Contracts [Medium]

**Interview Context:**
*"How do you enforce that any custom estimator written by junior data scientists strictly adheres to the Scikit-learn `.fit()`, `.predict()`, and `.get_params()` signatures, failing at instantiation time rather than deep in runtime execution?"*

```python
from abc import ABC, abstractmethod
import numpy as np

class BaseEstimatorContract(ABC):
    """
    Abstract Base Class enforcing the Scikit-learn estimator interface.
    Instantiation fails immediately if any @abstractmethod is not implemented.
    """
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseEstimatorContract":
        """Fit model to training data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels/values on unseen data."""
        pass

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Boolean flag indicating whether the model has been fitted."""
        pass

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Default concrete implementation leveraging abstract methods."""
        return self.fit(X, y).predict(X)

# --- Verification ---
class NonCompliantModel(BaseEstimatorContract):
    # Missing predict and is_fitted
    def fit(self, X: np.ndarray, y: np.ndarray):
        return self

class CompliantBaselineModel(BaseEstimatorContract):
    def __init__(self):
        self._mean = None

    @property
    def is_fitted(self) -> bool:
        return self._mean is not None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CompliantBaselineModel":
        self._mean = float(np.mean(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted")
        return np.full(shape=(len(X),), fill_value=self._mean)

if __name__ == "__main__":
    try:
        m_bad = NonCompliantModel()
    except TypeError as e:
        print(f"Enforcement check passed: {e}")

    m_good = CompliantBaselineModel()
    m_good.fit(np.zeros((10, 2)), np.array([1.0, 2.0, 3.0, 4.0]))
    print(f"Fitted check: {m_good.is_fitted}, Prediction: {m_good.predict(np.zeros((2, 2)))}")
```

---

## 4. Functional Programming & Standard Library Power Tools

### Q9 — `itertools.groupby`: The Sorting Prerequisite Gotcha in Transaction Bucketing [Medium]

**Interview Context:**
Standard LSEG and General Mills live-coding question.
*"Use `itertools.groupby` to group transactions by store or currency. Why does `groupby` fail or produce multiple fragmented groups for the same key if the list is not pre-sorted? Demonstrate the fix."*

**The Gotcha:**
`itertools.groupby` only groups **consecutive** elements matching the key (similar to the Unix `uniq` utility). It does not maintain an internal hash map across the entire stream. You must sort by key first or sort on the fly.

```python
import itertools
from operator import itemgetter
from typing import List, Dict

# Unsorted raw stream of transaction records
transactions = [
    {"store": "Powai_01", "sku": "Cheerios", "sales": 450.0},
    {"store": "Andheri_02", "sku": "Yoplait", "sales": 120.0},
    {"store": "Powai_01", "sku": "NatureValley", "sales": 310.0},
    {"store": "Bangalore_01", "sku": "Cheerios", "sales": 550.0},
    {"store": "Andheri_02", "sku": "Cheerios", "sales": 280.0},
]

# WRONG WAY: Calling groupby on unsorted data produces fragmented groups
print("--- WRONG: groupby on unsorted data ---")
for store, group in itertools.groupby(transactions, key=itemgetter("store")):
    print(f"Store: {store}, Records: {len(list(group))}")
    # Powai_01 appears TWICE!

# RIGHT WAY: Pre-sort by grouping key first
print("\n--- RIGHT: Pre-sorting by key ---")
sorted_tx = sorted(transactions, key=itemgetter("store"))
store_summaries: Dict[str, float] = {}

for store, group in itertools.groupby(sorted_tx, key=itemgetter("store")):
    total_sales = sum(item["sales"] for item in group)
    store_summaries[store] = total_sales
    print(f"Store: {store:12} | Total Revenue: ${total_sales:.2f}")
```

---

### Q10 — `itertools` for Data Science: `product`, `combinations`, and `islice` [Easy]

**Interview Context:**
Codility tests frequently ban external libraries. You must write grid-search parameter generation, feature pair interactions, or stream windowing using `itertools`.

```python
import itertools
from typing import List, Tuple

# 1. Hyperparameter Grid Generator using itertools.product
param_grid = {
    "learning_rate": [0.01, 0.1],
    "max_depth": [3, 6],
    "subsample": [0.8, 1.0]
}

keys = list(param_grid.keys())
combinations_list = list(itertools.product(*param_grid.values()))
grid_configs = [dict(zip(keys, combo)) for combo in combinations_list]
print(f"Generated {len(grid_configs)} configs:")
for cfg in grid_configs[:2]: print(f"  {cfg}")

# 2. Pairwise Feature Interactions with combinations (e.g., polynomial 2-way crosses)
features = ["revenue", "inventory_ratio", "discount_pct", "ad_spend"]
feature_pairs: List[Tuple[str, str]] = list(itertools.combinations(features, 2))
print(f"\n2-Way Feature Pairs ({len(feature_pairs)}): {feature_pairs[:3]}")

# 3. Stream windowing with islice (read first N items from infinite generator without loading all)
def infinite_price_ticks():
    import random
    price = 100.0
    while True:
        price += random.gauss(0, 0.5)
        yield round(price, 2)

first_5_ticks = list(itertools.islice(infinite_price_ticks(), 5))
print(f"\nFirst 5 Stream Ticks: {first_5_ticks}")
```

---

### Q11 — `functools.singledispatch`: Polymorphic Data Preprocessor [Medium]

**Interview Context:**
Common in production ML SDK development.
*"How do you write a clean, extensible data cleaning function that automatically handles `pandas.DataFrame`, `numpy.ndarray`, or a raw `list` of dicts without writing messy `if isinstance(...) elif isinstance(...)` chains?"*

`functools.singledispatch` transforms a function into a generic function that dispatches based on the type of the first argument.

```python
from functools import singledispatch
import numpy as np
import pandas as pd
from typing import List, Dict

@singledispatch
def standardize_input(data: Any) -> np.ndarray:
    """Base handler for unsupported data types."""
    raise NotImplementedError(f"Unsupported input type for standardization: {type(data).__name__}")

@standardize_input.register(np.ndarray)
def _(data: np.ndarray) -> np.ndarray:
    print("[Dispatch] Standardizing NumPy ndarray")
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0) + 1e-8
    return (data - mean) / std

@standardize_input.register(pd.DataFrame)
def _(data: pd.DataFrame) -> np.ndarray:
    print("[Dispatch] Standardizing Pandas DataFrame")
    numeric_df = data.select_dtypes(include=[np.number])
    return (numeric_df - numeric_df.mean()) / (numeric_df.std() + 1e-8).to_numpy()

@standardize_input.register(list)
def _(data: list) -> np.ndarray:
    print("[Dispatch] Standardizing Python List")
    arr = np.asarray(data, dtype=np.float64)
    return standardize_input(arr)  # Re-dispatches to numpy handler

# --- Verification ---
if __name__ == "__main__":
    arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    df = pd.DataFrame({"colA": [10, 20, 30], "colB": [100, 200, 300]})
    lst = [1.0, 5.0, 10.0, 15.0]

    out_arr = standardize_input(arr)
    out_df = standardize_input(df)
    out_lst = standardize_input(lst)
    print(f"Dispatched list shape: {out_lst.shape}")
```

---

### Q12 — `functools.partial` & Metadata-Preserving Decorators with `@wraps` [Easy]

**Interview Context:**
Probed in optimization algorithms (e.g., passing objective functions with fixed constraints to `scipy.optimize.minimize`) and custom monitoring decorators.

```python
from functools import partial, wraps
import time

# 1. functools.partial: Freezing hyperparameter subsets
def general_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float, l1_ratio: float) -> float:
    mse = np.mean((y_true - y_pred)**2)
    reg = alpha * (l1_ratio * np.sum(np.abs(y_pred)) + (1 - l1_ratio) * np.sum(y_pred**2))
    return float(mse + reg)

# Freeze regularization for a specific experiment
eval_loss_fixed = partial(general_loss, alpha=0.01, l1_ratio=0.5)
# Now eval_loss_fixed only requires (y_true, y_pred)
yt, yp = np.array([1.0, 2.0]), np.array([1.1, 1.9])
print(f"Fixed loss evaluation: {eval_loss_fixed(yt, yp):.4f}")

# 2. functools.wraps: Preserving function signature and docstrings
def log_latency(func):
    @wraps(func)  # Crucial: preserves __name__, __doc__, and __annotations__
    def wrapper(*args, **kwargs):
        """Wrapper documentation."""
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - t0
        print(f"[Latency] {func.__name__} took {duration*1000:.3f} ms")
        return result
    return wrapper

@log_latency
def score_portfolio(positions: list) -> float:
    """Calculates total portfolio exposure."""
    return sum(positions)

print(f"Function Name: {score_portfolio.__name__}")  # Without @wraps, this would be 'wrapper'
print(f"Function Doc: {score_portfolio.__doc__}")    # Preserves 'Calculates total portfolio exposure.'
```

---

## 5. Modern Typing & Static Analysis (Python 3.10+)

### Q13 — `typing.Protocol`: Structural Subtyping (Duck Typing Formalized) for Estimators [Medium]

**Interview Context:**
Modern Python (3.8+) allows static type checkers (Mypy) to enforce interfaces without explicit inheritance.
*"How do you type-annotate an ML pipeline that accepts ANY model object that defines `.predict()` and `.fit()`, regardless of whether it inherits from Scikit-learn, PyTorch, or XGBoost?"*

```python
from typing import Protocol, runtime_checkable, Any
import numpy as np

@runtime_checkable
class ModelPredictor(Protocol):
    """
    Structural protocol: Any class with this method signature automatically satisfies
    the protocol, without needing to explicitly inherit from it.
    """
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

@runtime_checkable
class TrainableEstimator(ModelPredictor, Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> Any:
        ...

# Custom standalone class with NO inheritance
class ThirdPartyCustomBooster:
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ThirdPartyCustomBooster":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.ones(len(X))

# Function type-annotated with the Protocol
def evaluate_inference_pipeline(model: ModelPredictor, test_data: np.ndarray) -> np.ndarray:
    if not isinstance(model, ModelPredictor):
        raise TypeError(f"Object {type(model).__name__} does not satisfy ModelPredictor protocol")
    return model.predict(test_data)

# --- Verification ---
if __name__ == "__main__":
    booster = ThirdPartyCustomBooster()
    # Check runtime protocol check
    print(f"Satisfies protocol? {isinstance(booster, ModelPredictor)}")  # True
    res = evaluate_inference_pipeline(booster, np.zeros((5, 2)))
    print(f"Inference result shape: {res.shape}")
```

---

### Q14 — Generics, `TypeVar`, and Union Operator (`|`) in Production Pipelines [Medium]

**Interview Context:**
Probed when reviewing modern Python 3.10+ codebases. Candidates are evaluated on whether they write clean generic containers and use modern type unions (`int | float` instead of `Union[int, float]`).

```python
from typing import TypeVar, Generic, Sequence, List
import numpy as np

# Type variable bound to numerical sequence types
T = TypeVar("T", bound=Union[int, float, np.number])

class BatchBuffer(Generic[T]):
    """Generic mini-batch buffer accepting any numeric type."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buffer: List[T] = []

    def push(self, item: T) -> None:
        if len(self._buffer) >= self.capacity:
            self._buffer.pop(0)  # Evict oldest
        self._buffer.append(item)

    def mean(self) -> float:
        if not self._buffer:
            return 0.0
        return float(np.mean(self._buffer))

    @property
    def items(self) -> List[T]:
        return list(self._buffer)

# Modern Python 3.10+ Union syntax: float | int | None
def parse_threshold(val: float | int | str | None) -> float:
    match val:
        case None:
            return 0.5
        case str() as s if s.replace('.', '', 1).isdigit():
            return float(s)
        case float() | int():
            return float(val)
        case _:
            raise ValueError(f"Invalid threshold format: {val}")

if __name__ == "__main__":
    buf = BatchBuffer[float](capacity=3)
    buf.push(10.5); buf.push(20.5); buf.push(30.0)
    print(f"Buffer items: {buf.items}, Mean: {buf.mean():.2f}")
    print(f"Parsed threshold: {parse_threshold('0.85')}")
```

---

## 6. Memory Management, GIL & Performance Internals

### Q15 — `__slots__`: 65% Memory Reduction for 1,000,000 Financial Records [Medium]

**Interview Context:**
Crucial for LSEG (Order book trades/ticks) and General Mills (Store-SKU daily inventory).
*"Why do standard Python class instances consume so much RAM? How does `__slots__` optimize memory? Write a comparison benchmark script demonstrating memory savings."*

**The Explanation:**
By default, every Python instance stores its attributes in an internal dictionary (`__dict__`). A Python dict has substantial memory overhead (hash table bucketing, resizing headroom). Defining `__slots__` tells CPython to allocate a fixed-size C array of pointers instead, eliminating `__dict__` and `__weakref__` completely.

```python
import sys

class DefaultOrder:
    def __init__(self, order_id: int, price: float, qty: int, symbol: str):
        self.order_id = order_id
        self.price = price
        self.qty = qty
        self.symbol = symbol

class SlottedOrder:
    __slots__ = ("order_id", "price", "qty", "symbol")
    def __init__(self, order_id: int, price: float, qty: int, symbol: str):
        self.order_id = order_id
        self.price = price
        self.qty = qty
        self.symbol = symbol

# --- Verification & Measurement ---
if __name__ == "__main__":
    import tracemalloc

    N = 200_000

    # 1. Measure Default Class Memory
    tracemalloc.start()
    default_orders = [DefaultOrder(i, 102.5, 100, "LSEG.L") for i in range(N)]
    current, peak_default = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del default_orders

    # 2. Measure Slotted Class Memory
    tracemalloc.start()
    slotted_orders = [SlottedOrder(i, 102.5, 100, "LSEG.L") for i in range(N)]
    current, peak_slotted = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del slotted_orders

    print(f"Default Objects Peak RAM: {peak_default / (1024*1024):.2f} MB")
    print(f"Slotted Objects Peak RAM: {peak_slotted / (1024*1024):.2f} MB")
    savings = (1 - (peak_slotted / peak_default)) * 100
    print(f"Memory Reduction: {savings:.1f}%")
```

---

### Q16 — Python Garbage Collection, Cyclic References & Memory Leaks in Notebooks [Hard]

**Interview Context:**
*"How does Python reclaim memory? If reference counting is the primary mechanism, what happens when two objects reference each other in a tree or pipeline? How do you detect and break reference cycles in long-running Colab / Jupyter notebooks?"*

**Mechanism:**
1. **Reference Counting:** Every object has an `ob_refcnt`. When it drops to 0, memory is immediately deallocated.
2. **Cyclic Garbage Collector (`gc` module):** Detects unreachable circular reference islands using 3 generations (Gen 0, Gen 1, Gen 2).
3. **Weak References (`weakref`):** Allows an object to be referenced without incrementing its reference count, preventing circular reference memory leaks.

```python
import gc
import sys
import weakref

class PipelineNode:
    def __init__(self, name: str):
        self.name = name
        self.parent = None  # Potential cycle if parent points to child and child points to parent
        self.data = [0.0] * 100_000

class CleanPipelineNode:
    def __init__(self, name: str):
        self.name = name
        self._parent = None  # Will use weakref
        self.data = [0.0] * 100_000

    @property
    def parent(self):
        return self._parent() if self._parent is not None else None

    @parent.setter
    def parent(self, node):
        # Store weak reference to parent
        self._parent = weakref.ref(node) if node is not None else None

# Demonstrate circular leak and weakref solution
if __name__ == "__main__":
    gc.collect()
    before = len(gc.get_objects())

    # Create cycle
    node_a = PipelineNode("A")
    node_b = PipelineNode("B")
    node_a.parent = node_b
    node_b.parent = node_a

    # Delete local references
    del node_a
    del node_b
    # Objects are still in memory due to cyclic references!
    uncollected = gc.collect()  # Forces cycle collection
    print(f"Unreachable cyclic objects collected by gc: {uncollected}")

    # Using WeakRef:
    clean_a = CleanPipelineNode("CleanA")
    clean_b = CleanPipelineNode("CleanB")
    clean_a.parent = clean_b
    clean_b.parent = clean_a

    print(f"CleanA's parent is: {clean_a.parent.name}")
    del clean_b
    print(f"After deleting clean_b, CleanA's parent is: {clean_a.parent}")  # Automatically None!
```

---

### Q17 — The GIL Deep Dive: Threading vs Multiprocessing vs C-Extensions Benchmark [Hard]

**Interview Context:**
High-frequency interview topic for Senior roles in India.
*"Explain the GIL. If Python has multithreading, why does running 4 threads on a 4-core machine to calculate Fibonacci numbers or train an algorithm run SLOWER than a single thread? When does multithreading actually provide a speedup in Data Science?"*

* **CPU-Bound Pure Python:** The GIL forces threads to alternate on a single CPU core. Context switching overhead makes multithreading slower than single-threading. Use `multiprocessing.Pool` or `ProcessPoolExecutor`.
* **NumPy / Scipy / C-Extensions:** C/C++ extensions explicitly release the GIL via `Py_BEGIN_ALLOW_THREADS`. Matrix multiplications, FFTs, and BLAS operations run across all available cores seamlessly under `threading`.
* **I/O-Bound Operations:** File reading, network sockets, SQL queries, and API calls release the GIL while waiting for I/O. Multithreading or `asyncio` delivers massive concurrency gains.

```python
import time
import threading
import multiprocessing
import numpy as np

# 1. Pure Python CPU-bound task
def cpu_heavy_task(n: int = 10_000_000):
    count = 0
    for i in range(n):
        count += (i % 2)
    return count

# 2. NumPy BLAS task (Releases GIL)
def numpy_blas_task(size: int = 2000):
    a = np.random.rand(size, size)
    b = np.random.rand(size, size)
    return np.dot(a, b)

if __name__ == "__main__":
    # Benchmark: Pure Python with 2 Threads vs 1 Sequential
    t0 = time.perf_counter()
    cpu_heavy_task(); cpu_heavy_task()
    seq_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    th1 = threading.Thread(target=cpu_heavy_task)
    th2 = threading.Thread(target=cpu_heavy_task)
    th1.start(); th2.start()
    th1.join(); th2.join()
    thread_time = time.perf_counter() - t0

    print(f"Pure Python Sequential Time: {seq_time:.3f}s")
    print(f"Pure Python 2-Thread Time:   {thread_time:.3f}s (GIL contention makes this slower/equal!)")

    # Benchmark: NumPy matrix dot with 2 Threads (GIL RELEASED)
    t0 = time.perf_counter()
    th1 = threading.Thread(target=numpy_blas_task)
    th2 = threading.Thread(target=numpy_blas_task)
    th1.start(); th2.start()
    th1.join(); th2.join()
    numpy_thread_time = time.perf_counter() - t0
    print(f"NumPy Multi-Thread Dot Time: {numpy_thread_time:.3f}s (Releases GIL to underlying BLAS threads)")
```

---

## 7. LSEG / Codility Special: Pure Python Parsing & Data Wrangling

### Q18 — Unstructured Log & Market Data Parsing with Regex Named Groups & `datetime` [Hard]

**Interview Context:**
Directly mirrors the LSEG Codility screening challenge. Candidates are given unstructured log lines from financial feeds or server logs and asked to parse timestamps, extract metrics, handle varied date formats, convert byte sizes, and output clean aggregated stats without using Pandas.

```python
import re
from datetime import datetime, timezone
from collections import defaultdict
from typing import List, Dict, Any

# Sample unstructured feed logs
RAW_LOGS = """
[2026-11-12 14:32:01.120] INFO src=FEED_LSE symbol=VOD.L price=105.40 volume=12500 latency=14ms
[2026-11-12 14:32:01.350] WARN src=FEED_NY symbol=AAPL price=225.10 volume=50000 latency=82ms
[2026-11-12 14:32:02.010] INFO src=FEED_LSE symbol=BP.L price=480.20 volume=8400 latency=12ms
[2026-11-12 14:32:02.890] INFO src=FEED_LSE symbol=VOD.L price=105.45 volume=15000 latency=16ms
[2026-11-12 14:32:03.110] ERROR src=FEED_TYO symbol=7203.T price=2150.0 volume=0 latency=310ms
"""

# Regex with named capture groups (?P<name>...)
LOG_PATTERN = re.compile(
    r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d{3})\]\s+"
    r"(?P<level>[A-Z]+)\s+"
    r"src=(?P<source>\w+)\s+"
    r"symbol=(?P<symbol>[\w\.]+)\s+"
    r"price=(?P<price>\d+\.?\d*)\s+"
    r"volume=(?P<volume>\d+)\s+"
    r"latency=(?P<latency>\d+)ms",
    re.MULTILINE
)

def parse_and_aggregate_logs(raw_text: str) -> Dict[str, Dict[str, Any]]:
    """
    Parses unstructured logs into structured summaries grouped by source.
    Returns: count, total_volume, vwap, and avg_latency.
    """
    stats = defaultdict(lambda: {"count": 0, "total_vol": 0, "dollar_vol": 0.0, "latencies": []})

    for match in LOG_PATTERN.finditer(raw_text.strip()):
        data = match.groupdict()
        
        # Parse timestamp safely
        dt = datetime.strptime(data["timestamp"], "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
        source = data["source"]
        price = float(data["price"])
        vol = int(data["volume"])
        latency = int(data["latency"])

        entry = stats[source]
        entry["count"] += 1
        entry["total_vol"] += vol
        entry["dollar_vol"] += (price * vol)
        entry["latencies"].append(latency)

    # Compute derived summary metrics
    results = {}
    for src, d in stats.items():
        vwap = (d["dollar_vol"] / d["total_vol"]) if d["total_vol"] > 0 else 0.0
        avg_lat = sum(d["latencies"]) / len(d["latencies"])
        results[src] = {
            "records": d["count"],
            "total_volume": d["total_vol"],
            "vwap": round(vwap, 2),
            "avg_latency_ms": round(avg_lat, 1)
        }
    return results

# --- Verification ---
if __name__ == "__main__":
    summary = parse_and_aggregate_logs(RAW_LOGS)
    print("Log Parsing & Aggregation Results:")
    for src, metrics in summary.items():
        print(f"  Source: {src:10} -> {metrics}")
```

---

## 8. Modern Syntax & Idiomatic Control Flow

### Q19 — Walrus Operator (`:=`) & Structural Pattern Matching (`match-case`) [Medium]

**Interview Context:**
Common in Python 3.10+ evaluations.
*"Demonstrate how the walrus operator (`:=`) eliminates redundant computation in data parsing loops. Then, write a pattern-matching function (`match-case`) to parse polymorphic JSON API payloads from disparate financial feeds or ERP schemas."*

```python
import re

# 1. Walrus Operator in Filtering Comprehensions
prices = [10.5, 25.0, 8.4, 45.2, 120.0, 3.2]

# Compute log transform, but only keep values where transformed result > 3.0
# Avoids computing np.log(p) twice!
transformed = [val for p in prices if (val := np.log(p) * 1.5) > 3.0]
print(f"Walrus transformed: {transformed}")

# 2. Structural Pattern Matching (match-case) for Complex Heterogeneous Payloads
def process_incoming_payload(event: dict) -> str:
    """Dispatches processing based on structural shape of dictionary."""
    match event:
        case {"type": "trade", "symbol": str(sym), "price": float(p), "qty": int(q)} if q > 10000:
            return f"[BLOCK TRADE] High volume order: {q} shares of {sym} at ${p:.2f}"
        case {"type": "trade", "symbol": str(sym), "price": float(p), "qty": int(q)}:
            return f"[RETAIL TRADE] Normal order: {q} shares of {sym} at ${p:.2f}"
        case {"type": "quote", "symbol": str(sym), "bid": float(b), "ask": float(a)}:
            return f"[QUOTE] {sym} Spread: ${a - b:.2f}"
        case {"type": "heartbeat", "timestamp": int(ts)}:
            return f"[HEARTBEAT] Ping received at {ts}"
        case _:
            return "[UNKNOWN EVENT] Event format unrecognized"

# --- Verification ---
if __name__ == "__main__":
    e1 = {"type": "trade", "symbol": "GIS", "price": 68.5, "qty": 50000}
    e2 = {"type": "quote", "symbol": "LSEG.L", "bid": 102.1, "ask": 102.3}
    e3 = {"type": "unknown_metric", "val": 42}

    print(process_incoming_payload(e1))
    print(process_incoming_payload(e2))
    print(process_incoming_payload(e3))
```

---

### Q20 — `collections` Arsenal: `deque` Running Windows, `defaultdict`, and `ChainMap` [Easy]

**Interview Context:**
*"Explain the time complexity differences between a Python `list` and `collections.deque` when maintaining a sliding window of recent observations. When would you use `ChainMap`?"*

* `list.pop(0)` or `list.insert(0, v)` is **$O(N)$** due to shifting array elements in memory.
* `deque.append()` and `deque.popleft()` are **$O(1)$** pointer operations on a doubly linked block list.
* Setting `maxlen=K` on a `deque` creates an automatic rolling window where pushing beyond $K$ evicts the oldest element in $O(1)$ time without memory reallocation.

```python
from collections import deque, defaultdict, ChainMap
import time

# 1. Rolling Window with collections.deque(maxlen=K)
class RollingStreamTracker:
    def __init__(self, window_size: int = 5):
        self.window = deque(maxlen=window_size)

    def add_tick(self, val: float) -> float:
        self.window.append(val)  # O(1) auto-eviction of oldest if full
        return sum(self.window) / len(self.window)

tracker = RollingStreamTracker(window_size=3)
for v in [10.0, 20.0, 30.0, 40.0]:
    print(f"Added {v:4.1f} -> Rolling Mean: {tracker.add_tick(v):.2f}")

# 2. ChainMap for Layered Configuration (Default -> Environment -> CLI args)
defaults = {"learning_rate": 0.01, "batch_size": 32, "device": "cpu"}
env_config = {"device": "cuda:0"}
cli_args = {"batch_size": 64}

# ChainMap searches dicts in left-to-right priority order
config = ChainMap(cli_args, env_config, defaults)
print(f"\nResolved Config:")
print(f"  batch_size: {config['batch_size']}")  # 64 (from CLI)
print(f"  device:     {config['device']}")      # cuda:0 (from Env)
print(f"  lr:         {config['learning_rate']}")# 0.01 (from Defaults)
```

---

## 9. Asynchronous Data Ingestion & Context Managers

### Q21 — `asyncio` with Rate-Limiting Semaphores for High-Throughput API Ingestion [Hard]

**Interview Context:**
Crucial for modern Senior DS roles interfacing with LLM APIs (OpenAI, Claude, Gemini) or market data feeds.
*"How do you fetch data for 5,000 tickers or generate LLM summaries concurrently without exceeding the API provider's 20 concurrent requests/sec rate limit and encountering HTTP 429 errors?"*

```python
import asyncio
import time
import random
from typing import List, Dict

async def fetch_ticker_fundamentals(ticker: str, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
    """
    Simulates async network call to market data API.
    Guarded by semaphore to cap concurrent open connections.
    """
    async with semaphore:
        # Simulate network latency
        latency = random.uniform(0.1, 0.3)
        await asyncio.sleep(latency)
        return {
            "ticker": ticker,
            "pe_ratio": round(random.uniform(12.0, 35.0), 2),
            "roe": round(random.uniform(0.08, 0.25), 3),
            "status": "OK"
        }

async def run_batch_ingestion(tickers: List[str], max_concurrency: int = 5) -> List[Dict[str, Any]]:
    # Semaphore restricts concurrent coroutines executing the inner block
    sem = asyncio.Semaphore(max_concurrency)
    tasks = [fetch_ticker_fundamentals(tick, sem) for tick in tickers]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

# --- Verification ---
if __name__ == "__main__":
    test_tickers = [f"SKU_{i:03d}" for i in range(15)]
    t0 = time.perf_counter()
    
    # Run async loop
    data = asyncio.run(run_batch_ingestion(test_tickers, max_concurrency=4))
    
    elapsed = time.perf_counter() - t0
    print(f"Successfully fetched {len(data)} items in {elapsed:.2f}s with max_concurrency=4")
    print(f"Sample response: {data[0]}")
```

---

### Q22 — Contextlib Suite: `@contextmanager`, `suppress`, and Dynamic `ExitStack` [Medium]

**Interview Context:**
*"How do you create a clean context manager without defining a class with `__enter__` and `__exit__`? How do you dynamically manage an arbitrary number of open file streams simultaneously without nesting 10 `with open(...)` statements?"*

```python
from contextlib import contextmanager, suppress, ExitStack
import tempfile
import os

# 1. Generator-based Context Manager using @contextmanager
@contextmanager
def temporary_numpy_precision(precision: int):
    """Temporarily changes NumPy print options and restores previous settings on exit."""
    old_opts = np.get_printoptions()
    np.set_printoptions(precision=precision, suppress=True)
    try:
        yield
    finally:
        np.set_printoptions(**old_opts)

# 2. Dynamic ExitStack for opening multiple files cleanly
def write_partitioned_streams(partitions: List[str], data_dict: Dict[str, list]):
    """
    Dynamically opens an unknown number of output files without nested with-blocks.
    Guarantees all file descriptors close safely even if an exception occurs.
    """
    temp_dir = tempfile.mkdtemp()
    with ExitStack() as stack:
        # Dynamically register each file handle into the stack
        file_handles = {
            part: stack.enter_context(open(os.path.join(temp_dir, f"{part}.txt"), "w"))
            for part in partitions
        }
        
        # Write records to corresponding partitioned files
        for part, records in data_dict.items():
            for r in records:
                file_handles[part].write(f"{r}\n")

    print(f"ExitStack cleanly closed all {len(partitions)} partitioned files.")

# --- Verification ---
if __name__ == "__main__":
    arr = np.array([0.123456789, 1234.56789])
    print(f"Default print: {arr}")
    with temporary_numpy_precision(2):
        print(f"Scoped print:  {arr}")
    print(f"Restored:      {arr}")

    # Using contextlib.suppress to safely ignore specific non-fatal errors
    with suppress(FileNotFoundError):
        os.remove("non_existent_cache_file.parquet")
    print("contextlib.suppress safely handled missing file.")
```

---

### Q23 — Custom Exception Hierarchies & Explicit Exception Chaining (`raise ... from`) [Easy]

**Interview Context:**
Crucial for MLOps and production error tracking.
*"Why should you define custom exception hierarchies instead of raising generic `Exception` or `ValueError`? What is the purpose of `raise ... from ...` (PEP 3134)?"*

* **Custom Hierarchies:** Allows callers to catch broad categories of errors (e.g., `except PipelineError:`) or specific failure modes (e.g., `except DataDriftError:`).
* **Exception Chaining (`from e`):** Preserves the original cause traceback explicitly, eliminating confusion in production logs.

```python
class MLOpsBaseException(Exception):
    """Base exception for all ML pipeline failures."""
    pass

class DataSchemaError(MLOpsBaseException):
    """Raised when incoming dataset does not match expected schema."""
    pass

class ModelInferenceTimeout(MLOpsBaseException):
    """Raised when inference latency violates SLA."""
    pass

def execute_scoring_stage(raw_record: dict) -> float:
    try:
        # Attempt to access expected key
        price = raw_record["unit_price"]
        return float(price) * 1.18
    except KeyError as exc:
        # Explicit exception chaining: Preserves original KeyError traceback
        raise DataSchemaError("Incoming record missing required column 'unit_price'") from exc

# --- Verification ---
if __name__ == "__main__":
    bad_record = {"sku": "Cheerios_500g"}
    try:
        execute_scoring_stage(bad_record)
    except DataSchemaError as err:
        print(f"Caught Domain Exception: {err}")
        print(f"Original Root Cause:    {repr(err.__cause__)}")
```

---

## 10. Quick-Fire Cheat Sheet: Pythonic Fundamentals

| Concept | What It Is | Why It Matters for DS / ML | Big-O / Performance Impact |
|---|---|---|---|
| **`__slots__`** | Pre-allocates static array of attributes, removing `__dict__` | Reduces memory by 60–70% for millions of objects (ticks, transactions) | Instant RAM savings; prevents attribute bloat |
| **`__getitem__`** | Enables indexing and slicing | Lazy-loading massive files (PyTorch/disk datasets) without OOM | $O(1)$ random seek if offsets indexed |
| **`__hash__` & `__eq__`** | Contract for equality and hashing | Enables custom classes in `set` and dictionary keys | $O(1)$ hash table lookup |
| **`__new__`** | Static method that constructs object | Singleton loaders for 10GB neural net weights | Runs once before `__init__` |
| **Descriptor** | Object defining `__get__`, `__set__` | Reusable hyperparameter / feature validation without boilerplate | Intercepts attribute assignment |
| **`typing.Protocol`** | Duck typing with static verification | Types estimators without forcing inheritance | Zero runtime cost (static check) |
| **`itertools.groupby`** | Groups consecutive items | Streaming aggregations; requires sorted input | $O(N)$ streaming, no hash map overhead |
| **`singledispatch`** | Generic function dispatch by type | Cleans polymorphic data pipelines (handles DF, array, list) | Replaces messy `isinstance` branches |
| **`asyncio.Semaphore`** | Caps concurrent async tasks | Prevents HTTP 429 rate limits on external LLM/market APIs | Maximizes network throughput |
| **`ExitStack`** | Programmatic context manager | Dynamically opens $N$ partitioned streams cleanly | Guarantees all file descriptors close |

---

*Companion Modules:*
* [00_master_guide.md](file:///D:/Int/General_Mills/interview_prep/00_master_guide.md) — Interview process, salary benchmarks, and preparation roadmap.
* [01_pandas_numpy_operations.md](file:///D:/Int/General_Mills/interview_prep/01_pandas_numpy_operations.md) — Pandas & NumPy live coding questions.
* [02_sql_database_operations.md](file:///D:/Int/General_Mills/interview_prep/02_sql_database_operations.md) — Advanced SQL & database operations.
* [03_ml_algorithms_implementation.md](file:///D:/Int/General_Mills/interview_prep/03_ml_algorithms_implementation.md) — ML algorithms from scratch.
* [04_statistics_probability.md](file:///D:/Int/General_Mills/interview_prep/04_statistics_probability.md) — A/B testing, hypothesis testing, and time series econometrics.
* [05_python_coding_dsa.md](file:///D:/Int/General_Mills/interview_prep/05_python_coding_dsa.md) — Core Python DSA, loops vs vectorization, and Colab patterns.
* [06_domain_case_studies.md](file:///D:/Int/General_Mills/interview_prep/06_domain_case_studies.md) — General Mills FMCG & LSEG Financial case studies.
* [07_deep_learning_nlp_advanced.md](file:///D:/Int/General_Mills/interview_prep/07_deep_learning_nlp_advanced.md) — Deep learning, NLP, transformers, and MLOps.
