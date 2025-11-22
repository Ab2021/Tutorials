# Lab 8.1: Thread Basics

## Objective
Create and manage threads in Rust.

## Exercises

### Exercise 1: Spawning Threads
```rust
use std::thread;

fn main() {
    let handle = thread::spawn(|| {
        for i in 1..10 {
            println!("Thread: {}", i);
        }
    });
    
    handle.join().unwrap();
}
```

### Exercise 2: Moving Data
```rust
let v = vec![1, 2, 3];

let handle = thread::spawn(move || {
    println!("{:?}", v);
});

handle.join().unwrap();
```

### Exercise 3: Multiple Threads
```rust
let mut handles = vec![];

for i in 0..10 {
    let handle = thread::spawn(move || {
        println!("Thread {}", i);
    });
    handles.push(handle);
}

for handle in handles {
    handle.join().unwrap();
}
```

## Success Criteria
✅ Spawn threads  
✅ Move data to threads  
✅ Join threads

## Next Steps
Lab 8.2: Message Passing with Channels
