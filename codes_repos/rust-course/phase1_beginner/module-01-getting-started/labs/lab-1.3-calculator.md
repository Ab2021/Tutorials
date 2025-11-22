# Lab 1.3: Building a Simple Calculator

## Objective
Create a simple calculator program to practice basic Rust syntax and Cargo workflow.

## Instructions

### Step 1: Create the Project

```bash
cargo new simple_calculator
cd simple_calculator
```

### Step 2: Starter Code

Replace the contents of `src/main.rs` with:

```rust
fn main() {
    println!("=== Simple Calculator ===");
    
    // TODO: Define two numbers
    
    // TODO: Perform calculations
    
    // TODO: Print results
}
```

### Step 3: Your Task

Complete the calculator with the following features:

1. Define two variables with numeric values
2. Perform these operations:
   - Addition
   - Subtraction
   - Multiplication
   - Division
3. Print the results in a formatted way

### Hints

- Use `let` to declare variables
- Use `{}` in println! to insert values
- Basic operators: `+`, `-`, `*`, `/`

### Example Output

```
=== Simple Calculator ===
First number: 10
Second number: 5

Addition: 10 + 5 = 15
Subtraction: 10 - 5 = 5
Multiplication: 10 * 5 = 50
Division: 10 / 5 = 2
```

## Challenges

Once you complete the basic calculator, try these challenges:

### Challenge 1: More Operations
Add these operations:
- Modulo (remainder): `%`
- Power: Use `num1.pow(num2)` (note: exponent must be u32)

### Challenge 2: Floating Point
Modify your calculator to use floating-point numbers (f64) instead of integers.

### Challenge 3: Multiple Calculations
Create variables for three numbers and calculate:
- Sum of all three
- Average of all three
- Product of all three

## Solution

<details>
<summary>Click to reveal solution</summary>

```rust
fn main() {
    println!("=== Simple Calculator ===");
    
    // Define two numbers
    let num1 = 10;
    let num2 = 5;
    
    println!("First number: {}", num1);
    println!("Second number: {}", num2);
    println!();
    
    // Perform calculations
    let addition = num1 + num2;
    let subtraction = num1 - num2;
    let multiplication = num1 * num2;
    let division = num1 / num2;
    
    // Print results
    println!("Addition: {} + {} = {}", num1, num2, addition);
    println!("Subtraction: {} - {} = {}", num1, num2, subtraction);
    println!("Multiplication: {} * {} = {}", num1, num2, multiplication);
    println!("Division: {} / {} = {}", num1, num2, division);
}
```

</details>

## Challenge Solutions

<details>
<summary>Challenge 1 Solution</summary>

```rust
fn main() {
    println!("=== Advanced Calculator ===");
    
    let num1 = 10;
    let num2 = 5;
    
    println!("First number: {}", num1);
    println!("Second number: {}", num2);
    println!();
    
    let addition = num1 + num2;
    let subtraction = num1 - num2;
    let multiplication = num1 * num2;
    let division = num1 / num2;
    let modulo = num1 % num2;
    let power = num1.pow(2); // num1 squared
    
    println!("Addition: {} + {} = {}", num1, num2, addition);
    println!("Subtraction: {} - {} = {}", num1, num2, subtraction);
    println!("Multiplication: {} * {} = {}", num1, num2, multiplication);
    println!("Division: {} / {} = {}", num1, num2, division);
    println!("Modulo: {} % {} = {}", num1, num2, modulo);
    println!("Power: {}^2 = {}", num1, power);
}
```

</details>

<details>
<summary>Challenge 2 Solution</summary>

```rust
fn main() {
    println!("=== Floating Point Calculator ===");
    
    let num1: f64 = 10.5;
    let num2: f64 = 3.2;
    
    println!("First number: {}", num1);
    println!("Second number: {}", num2);
    println!();
    
    let addition = num1 + num2;
    let subtraction = num1 - num2;
    let multiplication = num1 * num2;
    let division = num1 / num2;
    
    println!("Addition: {} + {} = {:.2}", num1, num2, addition);
    println!("Subtraction: {} - {} = {:.2}", num1, num2, subtraction);
    println!("Multiplication: {} * {} = {:.2}", num1, num2, multiplication);
    println!("Division: {} / {} = {:.2}", num1, num2, division);
}
```

</details>

<details>
<summary>Challenge 3 Solution</summary>

```rust
fn main() {
    println!("=== Three Number Calculator ===");
    
    let num1 = 10.0;
    let num2 = 20.0;
    let num3 = 30.0;
    
    println!("Numbers: {}, {}, {}", num1, num2, num3);
    println!();
    
    let sum = num1 + num2 + num3;
    let average = sum / 3.0;
    let product = num1 * num2 * num3;
    
    println!("Sum: {}", sum);
    println!("Average: {:.2}", average);
    println!("Product: {}", product);
}
```

</details>

## Success Criteria

✅ Calculator performs basic arithmetic operations  
✅ Results are displayed in a formatted manner  
✅ Code compiles and runs without errors  
✅ At least one challenge completed

## Key Learnings

- Variable declaration with `let`
- Basic arithmetic operators
- String formatting with `println!`
- Working with different numeric types

## Next Steps

Congratulations on completing Module 1! You now understand:
- How to install and use Rust
- How to create and manage Cargo projects
- Basic Rust syntax

Move on to **Module 2: Basic Syntax and Data Types** to dive deeper into Rust's type system!
