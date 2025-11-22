# Lab 7.3: Working with Result<T, E>

## Objective
Master the Result enum for error handling.

## Theory
`Result<T, E>` represents either success (Ok) or failure (Err).

## Exercises

### Exercise 1: Basic Result
```rust
fn divide(a: i32, b: i32) -> Result<i32, String> {
    if b == 0 {
        Err(String::from("Division by zero"))
    } else {
        Ok(a / b)
    }
}

fn main() {
    match divide(10, 2) {
        Ok(result) => println!("Result: {}", result),
        Err(e) => println!("Error: {}", e),
    }
}
```

### Exercise 2: File Reading
```rust
use std::fs::File;
use std::io::Read;

fn read_file(path: &str) -> Result<String, std::io::Error> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}
```

### Exercise 3: Parse with Result
```rust
fn parse_age(input: &str) -> Result<u32, String> {
    match input.parse::<u32>() {
        Ok(age) if age > 0 && age < 150 => Ok(age),
        Ok(_) => Err(String::from("Age out of range")),
        Err(_) => Err(String::from("Invalid number")),
    }
}
```

### Exercise 4: Result Methods
```rust
let result: Result<i32, &str> = Ok(10);

// map
let doubled = result.map(|x| x * 2);

// unwrap_or
let value = result.unwrap_or(0);

// and_then
let chained = result.and_then(|x| Ok(x + 5));
```

### Exercise 5: Custom Error Type
```rust
#[derive(Debug)]
enum MathError {
    DivisionByZero,
    NegativeNumber,
}

fn sqrt(x: f64) -> Result<f64, MathError> {
    if x < 0.0 {
        Err(MathError::NegativeNumber)
    } else {
        Ok(x.sqrt())
    }
}
```

## Success Criteria
✅ Understand Result<T, E>  
✅ Use Ok and Err correctly  
✅ Apply the ? operator  
✅ Create custom error types

## Next Steps
Proceed to Lab 7.4: Pattern Matching
