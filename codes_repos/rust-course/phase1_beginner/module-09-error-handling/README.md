# Module 5: Error Handling

## 🎯 Learning Objectives
- Understand panic! for unrecoverable errors
- Use Result<T, E> for recoverable errors
- Propagate errors with the ? operator
- Create custom error types
- Apply error handling best practices

## 📖 Theoretical Concepts

### 5.1 Unrecoverable Errors with panic!

```rust
fn main() {
    panic!("crash and burn");
}
```

**When to panic:**
- Unrecoverable errors
- Programming bugs
- Invalid state that should never happen

```rust
let v = vec![1, 2, 3];
v[99];  // Panics: index out of bounds
```

### 5.2 Recoverable Errors with Result

```rust
enum Result<T, E> {
    Ok(T),
    Err(E),
}
```

#### File Operations
```rust
use std::fs::File;

fn main() {
    let f = File::open("hello.txt");
    
    let f = match f {
        Ok(file) => file,
        Err(error) => panic!("Problem opening file: {:?}", error),
    };
}
```

#### Matching on Different Errors
```rust
use std::fs::File;
use std::io::ErrorKind;

fn main() {
    let f = File::open("hello.txt");
    
    let f = match f {
        Ok(file) => file,
        Err(error) => match error.kind() {
            ErrorKind::NotFound => match File::create("hello.txt") {
                Ok(fc) => fc,
                Err(e) => panic!("Problem creating file: {:?}", e),
            },
            other_error => panic!("Problem opening file: {:?}", other_error),
        },
    };
}
```

### 5.3 Shortcuts: unwrap and expect

```rust
// unwrap: panic on error
let f = File::open("hello.txt").unwrap();

// expect: panic with custom message
let f = File::open("hello.txt")
    .expect("Failed to open hello.txt");
```

### 5.4 Propagating Errors

```rust
use std::fs::File;
use std::io::{self, Read};

fn read_username_from_file() -> Result<String, io::Error> {
    let f = File::open("username.txt");
    
    let mut f = match f {
        Ok(file) => file,
        Err(e) => return Err(e),
    };
    
    let mut s = String::new();
    
    match f.read_to_string(&mut s) {
        Ok(_) => Ok(s),
        Err(e) => Err(e),
    }
}
```

### 5.5 The ? Operator

Shortcut for propagating errors:

```rust
fn read_username_from_file() -> Result<String, io::Error> {
    let mut f = File::open("username.txt")?;
    let mut s = String::new();
    f.read_to_string(&mut s)?;
    Ok(s)
}

// Even shorter
fn read_username_from_file() -> Result<String, io::Error> {
    let mut s = String::new();
    File::open("username.txt")?.read_to_string(&mut s)?;
    Ok(s)
}
```

**? operator:**
- Returns Err if error occurs
- Unwraps Ok value if successful
- Can only be used in functions that return Result or Option

### 5.6 Custom Error Types

```rust
use std::fmt;

#[derive(Debug)]
struct CustomError {
    message: String,
}

impl fmt::Display for CustomError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Custom error: {}", self.message)
    }
}

impl std::error::Error for CustomError {}

fn do_something() -> Result<(), CustomError> {
    Err(CustomError {
        message: String::from("Something went wrong"),
    })
}
```

### 5.7 Best Practices

**When to use panic!:**
- Examples, prototype code, tests
- When you have more information than the compiler
- Unrecoverable errors

**When to use Result:**
- Expected errors (file not found, network timeout)
- Errors that callers should handle
- Library code (let callers decide)

```rust
pub struct Guess {
    value: i32,
}

impl Guess {
    pub fn new(value: i32) -> Guess {
        if value < 1 || value > 100 {
            panic!("Guess value must be between 1 and 100, got {}.", value);
        }
        
        Guess { value }
    }
    
    pub fn value(&self) -> i32 {
        self.value
    }
}
```

## 🔑 Key Takeaways
- panic! for unrecoverable errors
- Result<T, E> for recoverable errors
- ? operator simplifies error propagation
- unwrap/expect for prototyping
- Custom error types for domain-specific errors
- Choose panic! vs Result based on context

## ⏭️ Next Steps
Complete the labs and move to Module 6: Collections
