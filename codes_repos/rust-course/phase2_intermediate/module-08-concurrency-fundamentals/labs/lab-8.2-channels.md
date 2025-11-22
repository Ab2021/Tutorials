# Lab 8.2: Message Passing with Channels

## Objective
Use channels for thread communication.

## Exercises

### Exercise 1: Basic Channel
```rust
use std::sync::mpsc;
use std::thread;

fn main() {
    let (tx, rx) = mpsc::channel();
    
    thread::spawn(move || {
        tx.send("Hello").unwrap();
    });
    
    let received = rx.recv().unwrap();
    println!("Got: {}", received);
}
```

### Exercise 2: Multiple Messages
```rust
let (tx, rx) = mpsc::channel();

thread::spawn(move || {
    let vals = vec!["hi", "from", "thread"];
    for val in vals {
        tx.send(val).unwrap();
    }
});

for received in rx {
    println!("Got: {}", received);
}
```

### Exercise 3: Multiple Producers
```rust
let (tx, rx) = mpsc::channel();

for i in 0..3 {
    let tx = tx.clone();
    thread::spawn(move || {
        tx.send(i).unwrap();
    });
}

drop(tx);

for received in rx {
    println!("Got: {}", received);
}
```

## Success Criteria
✅ Create channels  
✅ Send and receive messages  
✅ Use multiple producers

## Next Steps
Lab 8.3: Shared State with Mutex
