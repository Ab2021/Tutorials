# Lab 9.3: Error Propagation with ?

## Objective
Master the ? operator for clean error propagation.

## Theory
The ? operator propagates errors up the call stack, making error handling concise.

## Exercises

### Exercise 1: Basic ? Usage
```rust
use std::fs::File;
use std::io::{self, Read};

fn read_username_from_file() -> Result<String, io::Error> {
    let mut file = File::open("username.txt")?;
    let mut username = String::new();
    file.read_to_string(&mut username)?;
    Ok(username)
}
```

### Exercise 2: Chaining with ?
```rust
fn read_username_from_file() -> Result<String, io::Error> {
    let mut username = String::new();
    File::open("username.txt")?.read_to_string(&mut username)?;
    Ok(username)
}
```

### Exercise 3: ? in main
```rust
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let file = File::open("config.txt")?;
    // Process file
    Ok(())
}
```

### Exercise 4: Converting Errors
```rust
fn parse_config(s: &str) -> Result<i32, Box<dyn Error>> {
    let num = s.parse::<i32>()?;  // ParseIntError converted to Box<dyn Error>
    Ok(num)
}
```

### Exercise 5: Custom Error Propagation
```rust
#[derive(Debug)]
enum AppError {
    Io(std::io::Error),
    Parse(std::num::ParseIntError),
}

impl From<std::io::Error> for AppError {
    fn from(err: std::io::Error) -> Self {
        AppError::Io(err)
    }
}

impl From<std::num::ParseIntError> for AppError {
    fn from(err: std::num::ParseIntError) -> Self {
        AppError::Parse(err)
    }
}

fn read_and_parse() -> Result<i32, AppError> {
    let mut file = File::open("number.txt")?;  // Auto-converts io::Error
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let num = contents.trim().parse::<i32>()?;  // Auto-converts ParseIntError
    Ok(num)
}
```

## Success Criteria
✅ Use ? operator correctly  
✅ Chain operations with ?  
✅ Implement From trait for error conversion  
✅ Handle errors in main

## Next Steps
Proceed to Lab 9.4: Error Handling Patterns
