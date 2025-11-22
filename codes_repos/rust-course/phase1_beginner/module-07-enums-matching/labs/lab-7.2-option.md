# Lab 7.2: Working with Option<T>

## Objective
Master the Option enum for handling optional values.

## Theory
`Option<T>` represents a value that might be present (Some) or absent (None).

## Exercises

### Exercise 1: Basic Option
```rust
fn divide(numerator: f64, denominator: f64) -> Option<f64> {
    if denominator == 0.0 {
        None
    } else {
        Some(numerator / denominator)
    }
}

fn main() {
    match divide(10.0, 2.0) {
        Some(result) => println!("Result: {}", result),
        None => println!("Cannot divide by zero"),
    }
}
```

### Exercise 2: Option Methods
```rust
let x: Option<i32> = Some(5);

// unwrap_or
let y = x.unwrap_or(0);

// map
let doubled = x.map(|n| n * 2);

// and_then
let result = x.and_then(|n| Some(n + 10));
```

### Exercise 3: Find in Vector
```rust
fn find_element(v: &Vec<i32>, target: i32) -> Option<usize> {
    for (i, &val) in v.iter().enumerate() {
        if val == target {
            return Some(i);
        }
    }
    None
}
```

### Exercise 4: Parse Number
```rust
fn parse_number(s: &str) -> Option<i32> {
    s.parse().ok()
}
```

### Exercise 5: Chain Options
```rust
fn get_user_age(user_id: u32) -> Option<u32> {
    // Simulate database lookup
    if user_id == 1 {
        Some(25)
    } else {
        None
    }
}

fn can_vote(user_id: u32) -> Option<bool> {
    get_user_age(user_id).map(|age| age >= 18)
}
```

## Success Criteria
✅ Understand Option<T>  
✅ Use Some and None correctly  
✅ Apply Option methods (map, unwrap_or, etc.)  
✅ Handle optional values safely

## Next Steps
Proceed to Lab 7.3: Result<T, E>
