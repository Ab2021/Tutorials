# Lab 3.2: Loop Variations

## Objective
Master different loop types in Rust: loop, while, and for.

## Part 1: Infinite Loops with `loop`

### Exercise 1: Basic Loop with Break
```rust
fn main() {
    let mut count = 0;
    
    loop {
        count += 1;
        println!("Count: {}", count);
        
        if count == 5 {
            break;
        }
    }
}
```

### Exercise 2: Loop with Return Value
```rust
fn find_first_even() -> i32 {
    let mut n = 1;
    
    loop {
        if n % 2 == 0 {
            break n;
        }
        n += 1;
    }
}
```

### Exercise 3: Nested Loops with Labels
```rust
fn main() {
    'outer: loop {
        println!("Outer loop");
        
        'inner: loop {
            println!("Inner loop");
            break 'outer;
        }
    }
    println!("Exited both loops");
}
```

## Part 2: While Loops

### Exercise 4: Countdown
```rust
fn countdown(from: i32) {
    let mut n = from;
    
    while n > 0 {
        println!("{}!", n);
        n -= 1;
    }
    println!("Liftoff!");
}
```

### Exercise 5: Input Validation
```rust
fn wait_for_valid_input() {
    let mut attempts = 0;
    let target = 42;
    
    while attempts < 5 {
        let guess = 30; // Simulated input
        
        if guess == target {
            println!("Correct!");
            break;
        }
        
        attempts += 1;
        println!("Try again. Attempts left: {}", 5 - attempts);
    }
}
```

## Part 3: For Loops

### Exercise 6: Range Iteration
```rust
fn main() {
    // 0 to 4
    for i in 0..5 {
        println!("{}", i);
    }
    
    // 1 to 5 (inclusive)
    for i in 1..=5 {
        println!("{}", i);
    }
    
    // Reverse
    for i in (1..=5).rev() {
        println!("{}", i);
    }
}
```

### Exercise 7: Array Iteration
```rust
fn sum_array(arr: &[i32]) -> i32 {
    let mut sum = 0;
    
    for &num in arr {
        sum += num;
    }
    
    sum
}
```

### Exercise 8: Enumerate
```rust
fn main() {
    let names = ["Alice", "Bob", "Charlie"];
    
    for (index, name) in names.iter().enumerate() {
        println!("{}: {}", index + 1, name);
    }
}
```

## Challenges

### Challenge 1: Multiplication Table
Create a function that prints a multiplication table using nested loops.

### Challenge 2: Prime Number Finder
Use loops to find all prime numbers up to a given limit.

### Challenge 3: Pattern Printer
Print patterns using nested loops:
```
*
**
***
****
*****
```

## Solutions in `solutions/lab-3.2/`
