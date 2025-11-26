# Day 2: Interview Questions & Answers

## Conceptual Questions

### Q1: Explain the difference between Concurrency and Parallelism.
**Answer:**
- **Concurrency**: Dealing with multiple things at once. It's about *structure*. A program is concurrent if it can support multiple tasks making progress in overlapping time periods (e.g., handling 1000 web requests by switching between them when they wait for DB).
- **Parallelism**: Doing multiple things at once. It's about *execution*. A program is parallel if it literally runs multiple tasks at the exact same instant (e.g., on multiple CPU cores).
*   *Analogy*: Concurrency is one person juggling 3 balls. Parallelism is 3 people each holding 1 ball.
*   *Context*: Go's goroutines enable high concurrency. If run on a multi-core machine, the Go runtime also provides parallelism. Node.js is single-threaded (concurrent via event loop), but not parallel (unless using worker threads).

### Q2: Why is Python considered "slow", and how does FastAPI mitigate this?
**Answer:**
Python is interpreted and has the **Global Interpreter Lock (GIL)**, which prevents multiple native threads from executing Python bytecodes at once. This limits CPU-bound performance.
However, for web backends, the bottleneck is usually **I/O** (waiting for DB, network), not CPU.
**FastAPI** leverages Python's `asyncio` (Asynchronous I/O). When a request waits for the DB, Python releases the CPU to handle another request. This allows Python to handle thousands of concurrent connections, making it "fast enough" for most network-bound workloads.

### Q3: Compare the memory management of Java (JVM) vs. Rust.
**Answer:**
- **Java**: Uses a **Garbage Collector (GC)**. The runtime periodically pauses execution (Stop-the-world) to scan memory and free unused objects. This is easier for the dev but causes unpredictable latency spikes.
- **Rust**: Uses **Ownership & Borrowing** at compile time. There is no GC. The compiler inserts memory deallocation code exactly where an object goes out of scope. This results in predictable performance and memory safety without the runtime overhead of a GC.

---

## Scenario-Based Questions

### Q4: You need to build a service that processes 10,000 incoming JSON webhooks per second and saves them to a queue. The logic is simple validation. Which language do you choose and why?
**Answer:**
**Go**.
- **Reasoning**: This is a high-throughput, low-latency, CPU-light workload.
- **Go**: Goroutines are perfect for handling 10k concurrent connections with minimal memory footprint. The static typing and compilation speed ensure the service is robust and fast.
- **Why not Python**: The GIL might become a bottleneck at 10k RPS unless you run many worker processes, which consumes more RAM.
- **Why not Node**: Node could handle this well too, but Go's multi-core utilization is better out of the box without managing cluster modules.

### Q5: You are building a backend for a startup that relies heavily on NLP (Natural Language Processing) to summarize emails. Speed to market is critical.
**Answer:**
**Python**.
- **Reasoning**: The NLP ecosystem (HuggingFace, PyTorch, LangChain) is native to Python.
- **Efficiency**: Writing this in Go or Node would require calling out to Python scripts or using immature bindings, adding complexity.
- **Speed to Market**: Python's concise syntax allows for rapid iteration on the business logic. We can optimize for performance later (e.g., rewriting hot paths in Rust) if the product succeeds.

---

## Behavioral / Role-Specific Questions

### Q6: A developer on your team insists on rewriting a working Node.js microservice in Rust because "Rust is faster". How do you handle this?
**Answer:**
I would approach this with a "Cost-Benefit Analysis":
1.  **Measure**: Is the current Node.js service actually a bottleneck? Do we have metrics (latency, CPU usage) proving it?
2.  **Cost**: What is the cost of rewriting? (Dev time, testing, potential bugs). What is the "Bus Factor"? (Does everyone know Rust, or just this one dev?).
3.  **Decision**:
    - If the service is fine: "Premature optimization is the root of all evil." Reject the rewrite.
    - If the service is crashing/slow: Can we optimize the Node code first?
    - Only if Node is fundamentally incapable of meeting the requirement (e.g., heavy CPU processing) would I approve the rewrite, and I'd ensure the whole team is upskilled to support it.
