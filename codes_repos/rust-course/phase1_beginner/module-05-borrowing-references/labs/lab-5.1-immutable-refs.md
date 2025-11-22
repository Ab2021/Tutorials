# Lab 5.1: Immutable References

## Objective
Master immutable references and understand borrowing rules.

## Theory
An immutable reference (`&T`) allows you to read data without taking ownership.

## Exercises

### Exercise 1: Basic References
```rust
fn main() {
    let s = String::from("hello");
    let len = calculate_length(&s);
    println!("Length of '{}' is {}", s, len);
}

fn calculate_length(s: &String) -> usize {
    s.len()
}
```

### Exercise 2: Multiple Immutable References
```rust
fn main() {
    let s = String::from("hello");
    let r1 = &s;
    let r2 = &s;
    let r3 = &s;
    println!("{}, {}, {}", r1, r2, r3);
}
```

### Exercise 3: References in Functions
Write a function that takes a reference to a vector and returns the sum:
```rust
fn sum_vector(v: &Vec<i32>) -> i32 {
    // Your code here
}
```

### Exercise 4: String Analysis
Create a function that analyzes a string without taking ownership:
```rust
fn analyze_string(s: &str) -> (usize, usize, usize) {
    // Return (length, word_count, vowel_count)
}
```

### Exercise 5: Reference Patterns
```rust
struct Point {
    x: i32,
    y: i32,
}

fn distance_from_origin(p: &Point) -> f64 {
    // Calculate distance
}
```

## Success Criteria
✅ Understand when to use references  
✅ Can create multiple immutable references  
✅ Functions use references correctly  
✅ Original data remains accessible

## Next Steps
Proceed to Lab 5.2: Mutable References
