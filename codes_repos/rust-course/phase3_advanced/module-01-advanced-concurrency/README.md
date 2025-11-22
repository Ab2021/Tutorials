# Module 10: Advanced Topics

## 🎯 Learning Objectives
- Work with smart pointers (Box, Rc, RefCell)
- Understand concurrency and parallelism
- Use threads and message passing
- Implement async/await patterns
- Explore unsafe Rust
- Create macros

## 📖 Theoretical Concepts

### 10.1 Smart Pointers

#### Box<T> - Heap Allocation
```rust
let b = Box::new(5);
println!("b = {}", b);

// Recursive types
enum List {
    Cons(i32, Box<List>),
    Nil,
}

use List::{Cons, Nil};

let list = Cons(1, Box::new(Cons(2, Box::new(Cons(3, Box::new(Nil))))));
```

#### Rc<T> - Reference Counting
```rust
use std::rc::Rc;

let a = Rc::new(5);
let b = Rc::clone(&a);
let c = Rc::clone(&a);

println!("count = {}", Rc::strong_count(&a));  // 3
```

#### RefCell<T> - Interior Mutability
```rust
use std::cell::RefCell;

let x = RefCell::new(5);
*x.borrow_mut() += 1;
println!("{}", x.borrow());  // 6
```

### 10.2 Concurrency

#### Creating Threads
```rust
use std::thread;
use std::time::Duration;

fn main() {
    let handle = thread::spawn(|| {
        for i in 1..10 {
            println!("hi number {} from spawned thread!", i);
            thread::sleep(Duration::from_millis(1));
        }
    });
    
    for i in 1..5 {
        println!("hi number {} from main thread!", i);
        thread::sleep(Duration::from_millis(1));
    }
    
    handle.join().unwrap();
}
```

#### Moving Data into Threads
```rust
let v = vec![1, 2, 3];

let handle = thread::spawn(move || {
    println!("Here's a vector: {:?}", v);
});

handle.join().unwrap();
```

### 10.3 Message Passing

```rust
use std::sync::mpsc;
use std::thread;

fn main() {
    let (tx, rx) = mpsc::channel();
    
    thread::spawn(move || {
        let val = String::from("hi");
        tx.send(val).unwrap();
    });
    
    let received = rx.recv().unwrap();
    println!("Got: {}", received);
}
```

### 10.4 Shared State

```rust
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];
    
    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            let mut num = counter.lock().unwrap();
            *num += 1;
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    println!("Result: {}", *counter.lock().unwrap());
}
```

### 10.5 Async/Await

```rust
use tokio;

#[tokio::main]
async fn main() {
    let result = fetch_data().await;
    println!("Result: {}", result);
}

async fn fetch_data() -> String {
    // Async operation
    String::from("data")
}
```

### 10.6 Unsafe Rust

```rust
unsafe fn dangerous() {
    // Unsafe code
}

fn main() {
    unsafe {
        dangerous();
    }
}

// Dereferencing raw pointers
let mut num = 5;
let r1 = &num as *const i32;
let r2 = &mut num as *mut i32;

unsafe {
    println!("r1 is: {}", *r1);
    println!("r2 is: {}", *r2);
}
```

### 10.7 Macros

#### Declarative Macros
```rust
#[macro_export]
macro_rules! vec {
    ( $( $x:expr ),* ) => {
        {
            let mut temp_vec = Vec::new();
            $(
                temp_vec.push($x);
            )*
            temp_vec
        }
    };
}
```

#### Procedural Macros
```rust
use proc_macro;

#[proc_macro_derive(HelloMacro)]
pub fn hello_macro_derive(input: TokenStream) -> TokenStream {
    // Implementation
}
```

### 10.8 Foreign Function Interface (FFI)

```rust
extern "C" {
    fn abs(input: i32) -> i32;
}

fn main() {
    unsafe {
        println!("Absolute value of -3 according to C: {}", abs(-3));
    }
}
```

## 🔑 Key Takeaways
- Box for heap allocation
- Rc for multiple ownership
- RefCell for interior mutability
- Threads for parallelism
- Channels for message passing
- Mutex/Arc for shared state
- async/await for asynchronous code
- unsafe for low-level operations
- Macros for metaprogramming

## ⏭️ Next Steps
Complete the labs and work on final projects!
