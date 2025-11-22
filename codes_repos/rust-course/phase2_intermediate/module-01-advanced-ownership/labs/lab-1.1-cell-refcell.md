# Lab 1.1: Cell and RefCell Basics

## Objective
Master interior mutability with Cell and RefCell.

## Exercises

### Exercise 1: Cell Usage
```rust
use std::cell::Cell;

struct Counter {
    count: Cell<u32>,
}

impl Counter {
    fn new() -> Self {
        Counter { count: Cell::new(0) }
    }
    
    fn increment(&self) {
        self.count.set(self.count.get() + 1);
    }
    
    fn get(&self) -> u32 {
        self.count.get()
    }
}

fn main() {
    let counter = Counter::new();
    counter.increment();
    counter.increment();
    println!("Count: {}", counter.get());
}
```

### Exercise 2: RefCell Usage
```rust
use std::cell::RefCell;

struct Database {
    data: RefCell<Vec<String>>,
}

impl Database {
    fn new() -> Self {
        Database {
            data: RefCell::new(Vec::new()),
        }
    }
    
    fn add(&self, item: String) {
        self.data.borrow_mut().push(item);
    }
    
    fn list(&self) -> Vec<String> {
        self.data.borrow().clone()
    }
}

fn main() {
    let db = Database::new();
    db.add("Item 1".to_string());
    db.add("Item 2".to_string());
    println!("{:?}", db.list());
}
```

### Exercise 3: Build a Cache
Create a cache using RefCell that stores computed values.

### Exercise 4: Mutable Graph
Build a graph structure using RefCell for mutable connections.

### Exercise 5: Observer Pattern
Implement observer pattern using interior mutability.

## Solutions in `solutions/lab-1.1/`
