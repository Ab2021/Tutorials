# Lab 4.3: RefCell and Interior Mutability

## Objective
Use RefCell<T> for interior mutability patterns.

## Exercises

### Exercise 1: RefCell Basics
```rust
use std::cell::RefCell;

fn main() {
    let value = RefCell::new(5);
    
    *value.borrow_mut() += 1;
    
    println!("{}", value.borrow());
}
```

### Exercise 2: Combining Rc and RefCell
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

### Exercise 3: Mock Object Pattern
```rust
trait Messenger {
    fn send(&self, msg: &str);
}

struct MockMessenger {
    sent_messages: RefCell<Vec<String>>,
}

impl MockMessenger {
    fn new() -> Self {
        MockMessenger {
            sent_messages: RefCell::new(vec![]),
        }
    }
}

impl Messenger for MockMessenger {
    fn send(&self, msg: &str) {
        self.sent_messages.borrow_mut().push(String::from(msg));
    }
}
```

## Success Criteria
✅ Use RefCell for interior mutability  
✅ Combine Rc and RefCell  
✅ Understand borrow checking at runtime

## Next Steps
Lab 4.4: Weak References
