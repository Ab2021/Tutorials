# Module 04: Smart Pointers

## 🎯 Learning Objectives

- Master Box<T>, Rc<T>, and Arc<T>
- Understand RefCell<T> and interior mutability
- Use Weak<T> to prevent cycles
- Build custom smart pointers
- Apply smart pointer patterns

---

## 📖 Core Concepts

### Box<T> - Heap Allocation

```rust
// Store data on heap
let b = Box::new(5);

// Recursive types
enum List {
    Cons(i32, Box<List>),
    Nil,
}

// Trait objects
let shape: Box<dyn Shape> = Box::new(Circle { radius: 5.0 });
```

### Rc<T> - Reference Counting

```rust
use std::rc::Rc;

let a = Rc::new(vec![1, 2, 3]);
let b = Rc::clone(&a);  // Increment count
println!("Count: {}", Rc::strong_count(&a));
```

### Arc<T> - Atomic Reference Counting

```rust
use std::sync::Arc;
use std::thread;

let data = Arc::new(vec![1, 2, 3]);
let data_clone = Arc::clone(&data);

thread::spawn(move || {
    println!("{:?}", data_clone);
});
```

### RefCell<T> - Runtime Borrow Checking

```rust
use std::cell::RefCell;

let value = RefCell::new(5);
*value.borrow_mut() += 1;
println!("{}", value.borrow());
```

### Weak<T> - Weak References

```rust
use std::rc::{Rc, Weak};

let strong = Rc::new(5);
let weak: Weak<_> = Rc::downgrade(&strong);

if let Some(value) = weak.upgrade() {
    println!("{}", value);
}
```

---

## 🔑 Key Takeaways

1. **Box<T>** for heap allocation and trait objects
2. **Rc<T>** for shared ownership (single-threaded)
3. **Arc<T>** for shared ownership (multi-threaded)
4. **RefCell<T>** for interior mutability
5. **Weak<T>** to break reference cycles

Complete 10 labs, then proceed to Module 05: Testing & TDD
