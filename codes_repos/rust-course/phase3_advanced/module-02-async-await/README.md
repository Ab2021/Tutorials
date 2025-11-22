# Module 02: Async/Await and Futures

## 🎯 Learning Objectives

- Master async/await syntax
- Work with Tokio runtime
- Build async applications
- Handle async errors
- Process streams efficiently

---

## 📖 Core Concepts

### Async Functions

```rust
async fn fetch_data() -> Result<String, Error> {
    // Async operation
    Ok(String::from("data"))
}

#[tokio::main]
async fn main() {
    let result = fetch_data().await;
}
```

### Tokio Runtime

```rust
use tokio;

#[tokio::main]
async fn main() {
    tokio::spawn(async {
        println!("Running in background");
    });
    
    tokio::time::sleep(Duration::from_secs(1)).await;
}
```

### Async Traits

```rust
#[async_trait]
trait AsyncDatabase {
    async fn query(&self, sql: &str) -> Result<Vec<Row>>;
}
```

### Stream Processing

```rust
use tokio_stream::StreamExt;

let mut stream = tokio_stream::iter(vec![1, 2, 3]);

while let Some(value) = stream.next().await {
    println!("{}", value);
}
```

### Concurrent Tasks

```rust
let task1 = tokio::spawn(async { /* work */ });
let task2 = tokio::spawn(async { /* work */ });

let (result1, result2) = tokio::join!(task1, task2);
```

---

## 🔑 Key Takeaways

1. **async/await** for asynchronous programming
2. **Tokio** as the runtime
3. **Futures** represent pending operations
4. **Streams** for async iteration
5. **tokio::spawn** for concurrent tasks

Complete 10 labs, then proceed to Module 03: Unsafe Rust & FFI
