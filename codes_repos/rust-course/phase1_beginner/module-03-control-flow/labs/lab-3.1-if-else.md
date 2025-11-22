# Lab 3.1: if/else Exercises

## Objective
Master conditional expressions and if/else statements in Rust.

## Exercises

### Exercise 1: Basic if/else
```rust
fn main() {
    let number = 7;
    
    if number < 5 {
        println!("number is less than 5");
    } else {
        println!("number is 5 or greater");
    }
}
```

### Exercise 2: Multiple Conditions
```rust
fn check_number(n: i32) {
    if n > 0 {
        println!("positive");
    } else if n < 0 {
        println!("negative");
    } else {
        println!("zero");
    }
}
```

### Exercise 3: if as Expression
```rust
fn main() {
    let condition = true;
    let number = if condition { 5 } else { 6 };
    println!("The value is: {}", number);
}
```

### Exercise 4: Divisibility Checker
```rust
fn check_divisibility(n: i32) -> String {
    if n % 15 == 0 {
        String::from("Divisible by both 3 and 5")
    } else if n % 3 == 0 {
        String::from("Divisible by 3")
    } else if n % 5 == 0 {
        String::from("Divisible by 5")
    } else {
        String::from("Not divisible by 3 or 5")
    }
}
```

### Exercise 5: Grade Calculator
```rust
fn get_grade(score: u8) -> char {
    if score >= 90 {
        'A'
    } else if score >= 80 {
        'B'
    } else if score >= 70 {
        'C'
    } else if score >= 60 {
        'D'
    } else {
        'F'
    }
}
```

## Solutions in `solutions/lab-3.1/`
