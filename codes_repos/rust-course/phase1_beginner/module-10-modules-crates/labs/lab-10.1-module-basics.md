# Lab 10.1: Module Basics

## Objective
Learn to organize code with modules.

## Exercises

### Exercise 1: Inline Module
```rust
mod greetings {
    pub fn hello() {
        println!("Hello!");
    }
    
    pub fn goodbye() {
        println!("Goodbye!");
    }
}

fn main() {
    greetings::hello();
    greetings::goodbye();
}
```

### Exercise 2: Nested Modules
```rust
mod math {
    pub mod operations {
        pub fn add(a: i32, b: i32) -> i32 {
            a + b
        }
        
        pub fn subtract(a: i32, b: i32) -> i32 {
            a - b
        }
    }
}

fn main() {
    let sum = math::operations::add(5, 3);
}
```

### Exercise 3: Privacy
```rust
mod restaurant {
    pub mod front_of_house {
        pub fn seat_customer() {}
    }
    
    mod back_of_house {
        fn cook_order() {}  // Private
    }
}
```

### Exercise 4: use Keyword
```rust
use std::collections::HashMap;

fn main() {
    let mut map = HashMap::new();
    map.insert("key", "value");
}
```

## Success Criteria
✅ Create modules  
✅ Understand pub keyword  
✅ Use nested modules  
✅ Import with use

## Next Steps
Lab 10.2: File Modules
