# Module 08: Concurrency Fundamentals

## 🎯 Learning Objectives

- Master thread creation and management
- Use channels for message passing
- Share state safely with Mutex
- Understand Sync and Send traits
- Build concurrent applications

---

## 📖 Core Concepts

### Creating Threads

```rust
use std::thread;

let handle = thread::spawn(|| {
    println!("Hello from thread!");
});

handle.join().unwrap();
```

### Message Passing

```rust
use std::sync::mpsc;

let (tx, rx) = mpsc::channel();

thread::spawn(move || {
    tx.send("Hello").unwrap();
});

let message = rx.recv().unwrap();
```

### Shared State with Mutex

```rust
use std::sync::{Arc, Mutex};

let counter = Arc::new(Mutex::new(0));
let mut handles = vec![];

for _ in 0..10 {
    let counter = Arc::clone(&counter);
    let handle = thread::spawn(move || {
        let mut num = counter.lock().unwrap();
        *num += 1;
    });
    handles.push(handle);
}

for handle in handles {
    handle.join().unwrap();
}
```

### Sync and Send Traits

- **Send**: Can transfer ownership between threads
- **Sync**: Safe to reference from multiple threads

```rust
// Most types are Send and Sync
// Rc<T> is NOT Send or Sync
// Arc<T> is Send and Sync
```

---

## 🔑 Key Takeaways

1. **Threads** for parallel execution
2. **Channels** for safe communication
3. **Mutex** for shared mutable state
4. **Arc** for shared ownership across threads
5. **Sync/Send** ensure thread safety

Complete 10 labs, then proceed to Module 09: File I/O & Serialization
