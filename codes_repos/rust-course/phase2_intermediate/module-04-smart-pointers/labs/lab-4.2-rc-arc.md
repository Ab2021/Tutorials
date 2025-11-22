# Lab 4.2: Rc and Arc for Shared Ownership

## Objective
Use Rc<T> and Arc<T> for shared ownership patterns.

## Exercises

### Exercise 1: Rc Basics
```rust
use std::rc::Rc;

fn main() {
    let a = Rc::new(5);
    let b = Rc::clone(&a);
    let c = Rc::clone(&a);
    
    println!("Count: {}", Rc::strong_count(&a));
}
```

### Exercise 2: Shared Data Structure
```rust
use std::rc::Rc;

enum List {
    Cons(i32, Rc<List>),
    Nil,
}

fn main() {
    let a = Rc::new(Cons(5, Rc::new(Cons(10, Rc::new(Nil)))));
    let b = Cons(3, Rc::clone(&a));
    let c = Cons(4, Rc::clone(&a));
}
```

### Exercise 3: Arc for Threading
```rust
use std::sync::Arc;
use std::thread;

fn main() {
    let data = Arc::new(vec![1, 2, 3]);
    
    let mut handles = vec![];
    
    for _ in 0..3 {
        let data = Arc::clone(&data);
        let handle = thread::spawn(move || {
            println!("{:?}", data);
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
}
```

## Success Criteria
✅ Use Rc for single-threaded sharing  
✅ Use Arc for multi-threaded sharing  
✅ Understand reference counting

## Next Steps
Lab 4.3: RefCell and Interior Mutability
