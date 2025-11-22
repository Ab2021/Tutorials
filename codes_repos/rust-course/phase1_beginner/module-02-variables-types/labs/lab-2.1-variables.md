# Lab 2.1: Variable Exercises

## Objective
Practice working with variables, mutability, constants, and shadowing.

## Part 1: Basic Variables

Create a new project:
```bash
cargo new variable_practice
cd variable_practice
```

### Exercise 1: Immutability

```rust
fn main() {
    let x = 5;
    println!("The value of x is: {}", x);
    
    // TODO: Try to change x to 6
    // What happens? Why?
}
```

### Exercise 2: Mutability

```rust
fn main() {
    // TODO: Make this variable mutable
    let count = 0;
    
    count = count + 1;
    println!("Count: {}", count);
    
    count = count + 1;
    println!("Count: {}", count);
}
```

### Exercise 3: Constants

```rust
// TODO: Define a constant for the maximum score (100)
// TODO: Define a constant for PI (3.14159)

fn main() {
    println!("Maximum score: {}", MAX_SCORE);
    println!("PI: {}", PI);
}
```

### Exercise 4: Shadowing

```rust
fn main() {
    let x = 5;
    
    // TODO: Shadow x by adding 1 to it
    
    // TODO: Shadow x again by multiplying by 2
    
    println!("The value of x is: {}", x);
    // Expected output: 12
}
```

### Exercise 5: Type Change with Shadowing

```rust
fn main() {
    // TODO: Create a variable 'spaces' with the value "   "
    
    // TODO: Shadow 'spaces' with its length
    
    println!("Number of spaces: {}", spaces);
}
```

## Part 2: Data Types

### Exercise 6: Integer Types

```rust
fn main() {
    // TODO: Create variables of different integer types
    let small: i8 = /* your value */;
    let medium: i32 = /* your value */;
    let large: i64 = /* your value */;
    let unsigned: u32 = /* your value */;
    
    println!("Small: {}", small);
    println!("Medium: {}", medium);
    println!("Large: {}", large);
    println!("Unsigned: {}", unsigned);
}
```

### Exercise 7: Floating Point

```rust
fn main() {
    // TODO: Calculate the area of a circle with radius 5.0
    // Formula: area = π * r²
    
    let radius = 5.0;
    let pi = 3.14159;
    
    // TODO: Calculate area
    
    println!("Area of circle: {:.2}", area);
}
```

### Exercise 8: Boolean Logic

```rust
fn main() {
    let a = 10;
    let b = 20;
    
    // TODO: Create boolean variables for these conditions
    let is_equal = /* a equals b */;
    let is_greater = /* a is greater than b */;
    let is_less = /* a is less than b */;
    
    println!("Is equal: {}", is_equal);
    println!("Is greater: {}", is_greater);
    println!("Is less: {}", is_less);
}
```

## Part 3: Compound Types

### Exercise 9: Tuples

```rust
fn main() {
    // TODO: Create a tuple representing a person
    // (name, age, height_in_meters)
    let person = (/* fill in */);
    
    // TODO: Destructure the tuple
    let (name, age, height) = person;
    
    println!("Name: {}", name);
    println!("Age: {}", age);
    println!("Height: {}m", height);
}
```

### Exercise 10: Arrays

```rust
fn main() {
    // TODO: Create an array of the first 5 prime numbers
    let primes = [/* fill in */];
    
    // TODO: Print each prime number
    println!("First prime: {}", primes[0]);
    // ... print the rest
    
    // TODO: Create an array of 10 zeros
    let zeros = [/* fill in */];
}
```

## Solutions

<details>
<summary>Click to reveal solutions</summary>

```rust
// Exercise 1: Immutability
fn main() {
    let x = 5;
    println!("The value of x is: {}", x);
    
    // x = 6;  // This causes a compile error
    // Error: cannot assign twice to immutable variable
}

// Exercise 2: Mutability
fn main() {
    let mut count = 0;
    
    count = count + 1;
    println!("Count: {}", count);
    
    count = count + 1;
    println!("Count: {}", count);
}

// Exercise 3: Constants
const MAX_SCORE: u32 = 100;
const PI: f64 = 3.14159;

fn main() {
    println!("Maximum score: {}", MAX_SCORE);
    println!("PI: {}", PI);
}

// Exercise 4: Shadowing
fn main() {
    let x = 5;
    let x = x + 1;
    let x = x * 2;
    println!("The value of x is: {}", x);  // 12
}

// Exercise 5: Type Change with Shadowing
fn main() {
    let spaces = "   ";
    let spaces = spaces.len();
    println!("Number of spaces: {}", spaces);  // 3
}

// Exercise 6: Integer Types
fn main() {
    let small: i8 = 127;
    let medium: i32 = 2_147_483_647;
    let large: i64 = 9_223_372_036_854_775_807;
    let unsigned: u32 = 4_294_967_295;
    
    println!("Small: {}", small);
    println!("Medium: {}", medium);
    println!("Large: {}", large);
    println!("Unsigned: {}", unsigned);
}

// Exercise 7: Floating Point
fn main() {
    let radius = 5.0;
    let pi = 3.14159;
    let area = pi * radius * radius;
    
    println!("Area of circle: {:.2}", area);  // 78.54
}

// Exercise 8: Boolean Logic
fn main() {
    let a = 10;
    let b = 20;
    
    let is_equal = a == b;
    let is_greater = a > b;
    let is_less = a < b;
    
    println!("Is equal: {}", is_equal);      // false
    println!("Is greater: {}", is_greater);  // false
    println!("Is less: {}", is_less);        // true
}

// Exercise 9: Tuples
fn main() {
    let person = ("Alice", 30, 1.65);
    let (name, age, height) = person;
    
    println!("Name: {}", name);
    println!("Age: {}", age);
    println!("Height: {}m", height);
}

// Exercise 10: Arrays
fn main() {
    let primes = [2, 3, 5, 7, 11];
    
    println!("First prime: {}", primes[0]);
    println!("Second prime: {}", primes[1]);
    println!("Third prime: {}", primes[2]);
    println!("Fourth prime: {}", primes[3]);
    println!("Fifth prime: {}", primes[4]);
    
    let zeros = [0; 10];
    println!("Zeros array: {:?}", zeros);
}
```

</details>

## Success Criteria

✅ All exercises compile without errors  
✅ Understand the difference between immutability and mutability  
✅ Can use constants correctly  
✅ Understand shadowing and when to use it  
✅ Comfortable with different data types  
✅ Can work with tuples and arrays

## Next Steps

Proceed to Lab 2.2: Temperature Converter!
