# Lab 1.2: Rc and Arc Patterns

## Objective
Master reference counting with Rc and Arc.

## Exercise 1: Rc Basics
```rust
use std::rc::Rc;

let a = Rc::new(vec![1, 2, 3]);
let b = Rc::clone(&a);
let c = Rc::clone(&a);

println!("Count: {}", Rc::strong_count(&a));
```

## Exercise 2: Shared Data Structure
Build a tree with shared nodes using Rc.

## Exercise 3: Arc for Threading
Use Arc to share data across threads.

## Exercise 4: Weak References
Prevent cycles with Weak pointers.

## Exercise 5: Reference Counted Cache
Build a cache with Rc/Arc.
