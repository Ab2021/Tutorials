# Day 5: Advanced Topics & PyFlink

## Core Concepts & Theory

### PyFlink Architecture
Python API for Flink.
-   **Architecture**: Python runs in a separate process (Py4J / gRPC).
-   **Data Exchange**: Data is serialized from JVM -> Python Process -> JVM.
-   **Vectorized UDFs**: Uses Apache Arrow to transfer batches of data (much faster than row-by-row).

### Async I/O
Calling external APIs (REST/DB) from a stream.
-   **OrderedWait**: Preserves order (Head-of-line blocking).
-   **UnorderedWait**: Faster, order not guaranteed.

### Architectural Reasoning
**Why Vectorized Python UDFs?**
Python loop overhead is high. Vectorized UDFs (Pandas UDFs) allow executing logic on a *batch* of rows using optimized C libraries (NumPy/Pandas), reducing the serialization/invocation overhead.

### Key Components
-   `@udf`: Decorator for Python functions.
-   `AsyncDataStream`: Helper for Async I/O.
