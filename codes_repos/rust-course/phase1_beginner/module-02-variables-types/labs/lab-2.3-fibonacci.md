# Lab 2.3: Fibonacci Sequence Generator

## Objective
Practice loops, functions, and variables by implementing a Fibonacci sequence generator.

## Background

The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones:
```
0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144...
```

Formula: F(n) = F(n-1) + F(n-2), where F(0) = 0 and F(1) = 1

## Project Setup

```bash
cargo new fibonacci
cd fibonacci
```

## Part 1: Iterative Approach

### Exercise 1: Print First N Numbers

Implement a function that prints the first n Fibonacci numbers:

```rust
fn print_fibonacci(n: u32) {
    // TODO: Implement using a loop
}

fn main() {
    println!("First 10 Fibonacci numbers:");
    print_fibonacci(10);
}
```

**Hints:**
- Use two variables to track the last two numbers
- Use a for loop to iterate n times
- Update the variables in each iteration

### Exercise 2: Return Nth Fibonacci Number

Implement a function that returns the nth Fibonacci number:

```rust
fn fibonacci(n: u32) -> u64 {
    // TODO: Implement
}

fn main() {
    println!("10th Fibonacci number: {}", fibonacci(10));
    println!("20th Fibonacci number: {}", fibonacci(20));
}
```

### Exercise 3: Generate Sequence as Array

Create a function that generates the first n Fibonacci numbers and stores them in a vector (we'll use arrays for now):

```rust
fn fibonacci_sequence(n: usize) -> Vec<u64> {
    // TODO: Implement
    // Note: Vec is a growable array, we'll learn more about it later
    let mut sequence = Vec::new();
    
    // Your code here
    
    sequence
}

fn main() {
    let sequence = fibonacci_sequence(15);
    println!("First 15 Fibonacci numbers: {:?}", sequence);
}
```

## Part 2: Using match

### Exercise 4: Fibonacci with Pattern Matching

Implement using match for the base cases:

```rust
fn fibonacci_match(n: u32) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        _ => {
            // TODO: Calculate for n > 1
        }
    }
}
```

## Part 3: Challenges

### Challenge 1: Sum of Even Fibonacci Numbers

Find the sum of all even Fibonacci numbers below 4,000,000:

```rust
fn sum_even_fibonacci(limit: u64) -> u64 {
    // TODO: Implement
}

fn main() {
    let sum = sum_even_fibonacci(4_000_000);
    println!("Sum of even Fibonacci numbers below 4,000,000: {}", sum);
}
```

### Challenge 2: Fibonacci Until Value

Generate Fibonacci numbers until a value exceeds a limit:

```rust
fn fibonacci_until(limit: u64) {
    println!("Fibonacci numbers up to {}:", limit);
    // TODO: Implement
}

fn main() {
    fibonacci_until(1000);
}
```

### Challenge 3: Check if Number is Fibonacci

Determine if a given number is in the Fibonacci sequence:

```rust
fn is_fibonacci(num: u64) -> bool {
    // TODO: Implement
    // Hint: Generate Fibonacci numbers until you reach or exceed num
}

fn main() {
    let test_numbers = [8, 10, 13, 15, 21, 100];
    
    for num in test_numbers {
        if is_fibonacci(num) {
            println!("{} is a Fibonacci number", num);
        } else {
            println!("{} is NOT a Fibonacci number", num);
        }
    }
}
```

## Solutions

<details>
<summary>Exercise 1 Solution</summary>

```rust
fn print_fibonacci(n: u32) {
    let mut a = 0u64;
    let mut b = 1u64;
    
    for i in 0..n {
        if i == 0 {
            println!("{}", a);
        } else if i == 1 {
            println!("{}", b);
        } else {
            let next = a + b;
            println!("{}", next);
            a = b;
            b = next;
        }
    }
}

fn main() {
    println!("First 10 Fibonacci numbers:");
    print_fibonacci(10);
}
```

</details>

<details>
<summary>Exercise 2 Solution</summary>

```rust
fn fibonacci(n: u32) -> u64 {
    if n == 0 {
        return 0;
    } else if n == 1 {
        return 1;
    }
    
    let mut a = 0u64;
    let mut b = 1u64;
    
    for _ in 2..=n {
        let next = a + b;
        a = b;
        b = next;
    }
    
    b
}

fn main() {
    println!("10th Fibonacci number: {}", fibonacci(10));
    println!("20th Fibonacci number: {}", fibonacci(20));
    println!("30th Fibonacci number: {}", fibonacci(30));
}
```

</details>

<details>
<summary>Exercise 3 Solution</summary>

```rust
fn fibonacci_sequence(n: usize) -> Vec<u64> {
    let mut sequence = Vec::new();
    
    if n == 0 {
        return sequence;
    }
    
    sequence.push(0);
    
    if n == 1 {
        return sequence;
    }
    
    sequence.push(1);
    
    for i in 2..n {
        let next = sequence[i - 1] + sequence[i - 2];
        sequence.push(next);
    }
    
    sequence
}

fn main() {
    let sequence = fibonacci_sequence(15);
    println!("First 15 Fibonacci numbers: {:?}", sequence);
}
```

</details>

<details>
<summary>Exercise 4 Solution</summary>

```rust
fn fibonacci_match(n: u32) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        _ => {
            let mut a = 0u64;
            let mut b = 1u64;
            
            for _ in 2..=n {
                let next = a + b;
                a = b;
                b = next;
            }
            
            b
        }
    }
}

fn main() {
    for i in 0..15 {
        println!("F({}) = {}", i, fibonacci_match(i));
    }
}
```

</details>

<details>
<summary>Challenge 1 Solution</summary>

```rust
fn sum_even_fibonacci(limit: u64) -> u64 {
    let mut sum = 0;
    let mut a = 0u64;
    let mut b = 1u64;
    
    while a <= limit {
        if a % 2 == 0 {
            sum += a;
        }
        
        let next = a + b;
        a = b;
        b = next;
    }
    
    sum
}

fn main() {
    let sum = sum_even_fibonacci(4_000_000);
    println!("Sum of even Fibonacci numbers below 4,000,000: {}", sum);
    // Answer: 4,613,732
}
```

</details>

<details>
<summary>Challenge 2 Solution</summary>

```rust
fn fibonacci_until(limit: u64) {
    println!("Fibonacci numbers up to {}:", limit);
    
    let mut a = 0u64;
    let mut b = 1u64;
    
    print!("{}, ", a);
    
    while b <= limit {
        print!("{}, ", b);
        let next = a + b;
        a = b;
        b = next;
    }
    
    println!();
}

fn main() {
    fibonacci_until(1000);
}
```

</details>

<details>
<summary>Challenge 3 Solution</summary>

```rust
fn is_fibonacci(num: u64) -> bool {
    if num == 0 || num == 1 {
        return true;
    }
    
    let mut a = 0u64;
    let mut b = 1u64;
    
    while b < num {
        let next = a + b;
        a = b;
        b = next;
    }
    
    b == num
}

fn main() {
    let test_numbers = [8, 10, 13, 15, 21, 100];
    
    for num in test_numbers {
        if is_fibonacci(num) {
            println!("{} is a Fibonacci number ✓", num);
        } else {
            println!("{} is NOT a Fibonacci number ✗", num);
        }
    }
}
```

</details>

## Expected Output

```
First 10 Fibonacci numbers:
0
1
1
2
3
5
8
13
21
34

10th Fibonacci number: 55
20th Fibonacci number: 6765

First 15 Fibonacci numbers: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]

Sum of even Fibonacci numbers below 4,000,000: 4613732

8 is a Fibonacci number ✓
10 is NOT a Fibonacci number ✗
13 is a Fibonacci number ✓
15 is NOT a Fibonacci number ✗
21 is a Fibonacci number ✓
100 is NOT a Fibonacci number ✗
```

## Success Criteria

✅ Implemented iterative Fibonacci function  
✅ Can generate nth Fibonacci number  
✅ Understand loop control flow  
✅ Used pattern matching with match  
✅ Completed at least one challenge

## Key Learnings

- Using loops (for, while) effectively
- Managing mutable state in iterations
- Pattern matching with match
- Working with sequences of numbers
- Algorithm implementation in Rust

## Next Steps

Proceed to Lab 2.4: Pattern Matching!
