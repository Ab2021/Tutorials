# Module 01: Advanced Ownership Patterns

## 🎯 Learning Objectives

- Master interior mutability patterns
- Use reference counting effectively
- Understand Copy-on-Write (Cow)
- Build custom smart pointers
- Apply advanced ownership patterns

---

## 📖 Theoretical Concepts

### 1.1 Interior Mutability

Interior mutability allows you to mutate data even when there are immutable references to that data.

#### Cell<T>

```rust
use std::cell::Cell;

struct Counter {
    count: Cell<u32>,
}

impl Counter {
    fn new() -> Self {
        Counter { count: Cell::new(0) }
    }
    
    fn increment(&self) {  // Takes &self, not &mut self!
        let current = self.count.get();
        self.count.set(current + 1);
    }
    
    fn get(&self) -> u32 {
        self.count.get()
    }
}
```

**Use Case:** Simple values that need mutation through shared references.

#### RefCell<T>

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
    
    fn get(&self, index: usize) -> Option<String> {
        self.data.borrow().get(index).cloned()
    }
}
```

**Key Points:**
- Runtime borrow checking
- Panics if borrowing rules violated
- Use when compile-time checking impossible

---

### 1.2 Reference Counting

#### Rc<T> - Single-threaded Reference Counting

```rust
use std::rc::Rc;

let a = Rc::new(5);
let b = Rc::clone(&a);
let c = Rc::clone(&a);

println!("Reference count: {}", Rc::strong_count(&a));  // 3
```

#### Arc<T> - Atomic Reference Counting (Thread-safe)

```rust
use std::sync::Arc;
use std::thread;

let data = Arc::new(vec![1, 2, 3]);

let handles: Vec<_> = (0..3).map(|_| {
    let data = Arc::clone(&data);
    thread::spawn(move || {
        println!("{:?}", data);
    })
}).collect();

for handle in handles {
    handle.join().unwrap();
}
```

---

### 1.3 Combining Rc and RefCell

```rust
use std::rc::Rc;
use std::cell::RefCell;

#[derive(Debug)]
struct Node {
    value: i32,
    children: RefCell<Vec<Rc<Node>>>,
}

impl Node {
    fn new(value: i32) -> Rc<Self> {
        Rc::new(Node {
            value,
            children: RefCell::new(Vec::new()),
        })
    }
    
    fn add_child(&self, child: Rc<Node>) {
        self.children.borrow_mut().push(child);
    }
}
```

---

### 1.4 Copy-on-Write (Cow)

```rust
use std::borrow::Cow;

fn process_text(text: &str) -> Cow<str> {
    if text.contains("ERROR") {
        Cow::Owned(text.replace("ERROR", "WARNING"))
    } else {
        Cow::Borrowed(text)
    }
}

let original = "This is fine";
let result = process_text(original);  // Borrowed

let with_error = "ERROR occurred";
let result = process_text(with_error);  // Owned
```

**Benefits:**
- Avoid unnecessary clones
- Efficient when modification rare
- Flexible ownership

---

### 1.5 Weak References

```rust
use std::rc::{Rc, Weak};
use std::cell::RefCell;

struct Node {
    value: i32,
    parent: RefCell<Weak<Node>>,
    children: RefCell<Vec<Rc<Node>>>,
}
```

**Use Cases:**
- Prevent reference cycles
- Parent-child relationships
- Cache implementations

---

## 🔑 Key Takeaways

1. **Interior mutability** bypasses borrowing rules at runtime
2. **Cell<T>** for Copy types, **RefCell<T>** for others
3. **Rc<T>** for single-threaded shared ownership
4. **Arc<T>** for thread-safe shared ownership
5. **Cow<T>** for efficient clone-on-write
6. **Weak<T>** prevents reference cycles

---

## ⏭️ Next Steps

Complete the 10 labs in this module, then proceed to Module 02: Lifetimes
