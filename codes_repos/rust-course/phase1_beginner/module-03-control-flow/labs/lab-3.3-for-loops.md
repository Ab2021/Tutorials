# Lab 3.3: For Loop Patterns

## Objective
Master for loops and iteration patterns in Rust.

## Theory
The `for` loop is the most common loop in Rust, used to iterate over collections and ranges.

## Exercises

### Exercise 1: Range Iteration
```rust
fn main() {
    // Exclusive range
    for i in 0..5 {
        println!("{}", i);  // 0, 1, 2, 3, 4
    }
    
    // Inclusive range
    for i in 1..=5 {
        println!("{}", i);  // 1, 2, 3, 4, 5
    }
    
    // Reverse
    for i in (1..=5).rev() {
        println!("{}", i);  // 5, 4, 3, 2, 1
    }
}
```

### Exercise 2: Array Iteration
```rust
fn main() {
    let arr = [10, 20, 30, 40, 50];
    
    for element in arr {
        println!("{}", element);
    }
    
    // With index
    for (index, value) in arr.iter().enumerate() {
        println!("Index {}: {}", index, value);
    }
}
```

### Exercise 3: Vector Iteration
```rust
fn main() {
    let v = vec![1, 2, 3, 4, 5];
    
    // Immutable iteration
    for num in &v {
        println!("{}", num);
    }
    
    // Mutable iteration
    let mut v = vec![1, 2, 3];
    for num in &mut v {
        *num *= 2;
    }
}
```

### Exercise 4: String Characters
```rust
fn main() {
    let s = "Hello";
    
    for c in s.chars() {
        println!("{}", c);
    }
    
    for (i, c) in s.chars().enumerate() {
        println!("Char {} is {}", i, c);
    }
}
```

### Exercise 5: Nested Loops
```rust
fn print_multiplication_table(n: i32) {
    for i in 1..=n {
        for j in 1..=n {
            print!("{:4}", i * j);
        }
        println!();
    }
}
```

## Success Criteria
✅ Use ranges effectively  
✅ Iterate over arrays and vectors  
✅ Use enumerate for indices  
✅ Work with nested loops

## Next Steps
Proceed to Lab 3.4: Pattern Matching Basics
