# Lab 1.2: Atomic Operations

## Objective
Use atomic types for lock-free programming.

## Exercises

### Exercise 1: AtomicBool
```rust
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

fn main() {
    let flag = Arc::new(AtomicBool::new(false));
    let flag_clone = Arc::clone(&flag);
    
    let handle = thread::spawn(move || {
        flag_clone.store(true, Ordering::Relaxed);
    });
    
    handle.join().unwrap();
    println!("Flag: {}", flag.load(Ordering::Relaxed));
}
```

### Exercise 2: AtomicUsize Counter
```rust
use std::sync::atomic::{AtomicUsize, Ordering};

let counter = Arc::new(AtomicUsize::new(0));

let mut handles = vec![];
for _ in 0..10 {
    let counter = Arc::clone(&counter);
    let handle = thread::spawn(move || {
        for _ in 0..1000 {
            counter.fetch_add(1, Ordering::SeqCst);
        }
    });
    handles.push(handle);
}

for handle in handles {
    handle.join().unwrap();
}

println!("Count: {}", counter.load(Ordering::SeqCst));
```

## Success Criteria
✅ Use atomic types  
✅ Understand memory ordering  
✅ Build lock-free structures

## Next Steps
Lab 1.3: Thread Pools
