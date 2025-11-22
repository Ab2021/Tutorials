# Lab 9.2: Custom Error Types

## Objective
Learn to create and use custom error types in Rust.

## Theory
Custom error types make error handling more expressive and type-safe.

## Exercises

### Exercise 1: Simple Custom Error
```rust
#[derive(Debug)]
enum MathError {
    DivisionByZero,
    NegativeSquareRoot,
}

fn divide(a: f64, b: f64) -> Result<f64, MathError> {
    if b == 0.0 {
        Err(MathError::DivisionByZero)
    } else {
        Ok(a / b)
    }
}

fn sqrt(x: f64) -> Result<f64, MathError> {
    if x < 0.0 {
        Err(MathError::NegativeSquareRoot)
    } else {
        Ok(x.sqrt())
    }
}
```

### Exercise 2: Error with Data
```rust
#[derive(Debug)]
enum ParseError {
    InvalidFormat(String),
    OutOfRange { min: i32, max: i32, value: i32 },
}

fn parse_age(s: &str) -> Result<u32, ParseError> {
    match s.parse::<i32>() {
        Ok(age) if age >= 0 && age <= 150 => Ok(age as u32),
        Ok(age) => Err(ParseError::OutOfRange {
            min: 0,
            max: 150,
            value: age,
        }),
        Err(_) => Err(ParseError::InvalidFormat(s.to_string())),
    }
}
```

### Exercise 3: Implementing Display
```rust
use std::fmt;

impl fmt::Display for MathError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MathError::DivisionByZero => write!(f, "Cannot divide by zero"),
            MathError::NegativeSquareRoot => write!(f, "Cannot take square root of negative number"),
        }
    }
}
```

### Exercise 4: Implementing std::error::Error
```rust
use std::error::Error;

impl Error for MathError {}

fn process() -> Result<(), Box<dyn Error>> {
    let result = divide(10.0, 0.0)?;
    Ok(())
}
```

### Exercise 5: Multiple Error Types
```rust
#[derive(Debug)]
enum AppError {
    Io(std::io::Error),
    Parse(std::num::ParseIntError),
    Custom(String),
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
```

## Success Criteria
✅ Create custom error enums  
✅ Implement Display trait  
✅ Implement Error trait  
✅ Convert between error types

## Next Steps
Proceed to Lab 9.3: Error Propagation
