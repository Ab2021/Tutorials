# Lab 3.1: Ownership Transfer Exercises

## Objective
Understand ownership rules and move semantics through hands-on practice.

## Setup
```bash
cargo new ownership_practice
cd ownership_practice
```

## Part 1: Basic Ownership

### Exercise 1: Ownership Transfer
```rust
fn main() {
    let s1 = String::from("hello");
    let s2 = s1;
    
    // Uncomment and observe the error:
    // println!("{}", s1);
    
    println!("{}", s2);
}
```

**Question:** Why does `s1` become invalid?

### Exercise 2: Clone vs Move
```rust
fn main() {
    let s1 = String::from("hello");
    let s2 = s1.clone();
    
    println!("s1 = {}, s2 = {}", s1, s2);
}
```

**Question:** What's the difference between this and Exercise 1?

### Exercise 3: Copy Types
```rust
fn main() {
    let x = 5;
    let y = x;
    
    println!("x = {}, y = {}", x, y);
}
```

**Question:** Why does this work without clone()?

## Part 2: Functions and Ownership

### Exercise 4: Function Takes Ownership
```rust
fn main() {
    let s = String::from("hello");
    takes_ownership(s);
    
    // Uncomment to see error:
    // println!("{}", s);
}

fn takes_ownership(some_string: String) {
    println!("{}", some_string);
}
```

### Exercise 5: Return Ownership
```rust
fn main() {
    let s1 = gives_ownership();
    println!("{}", s1);
    
    let s2 = String::from("hello");
    let s3 = takes_and_gives_back(s2);
    
    // Uncomment to see error:
    // println!("{}", s2);
    println!("{}", s3);
}

fn gives_ownership() -> String {
    String::from("yours")
}

fn takes_and_gives_back(a_string: String) -> String {
    a_string
}
```

## Part 3: Practical Exercises

### Exercise 6: Fix the Code
Fix this code without using clone():

```rust
fn main() {
    let s = String::from("hello");
    
    print_string(s);
    print_string(s);  // ERROR: value used after move
}

fn print_string(s: String) {
    println!("{}", s);
}
```

### Exercise 7: String Builder
Complete this function that builds a greeting:

```rust
fn build_greeting(name: String) -> String {
    // TODO: Create a greeting like "Hello, Alice!"
    // Return the greeting (transfer ownership)
}

fn main() {
    let name = String::from("Alice");
    let greeting = build_greeting(name);
    println!("{}", greeting);
    
    // Note: name is no longer valid here
}
```

### Exercise 8: Tuple Return
Return multiple values using a tuple:

```rust
fn calculate(s: String) -> (String, usize) {
    let length = s.len();
    (s, length)  // Return both the string and its length
}

fn main() {
    let s = String::from("hello");
    let (s, len) = calculate(s);
    println!("String '{}' has length {}", s, len);
}
```

## Solutions

<details>
<summary>Exercise 6 Solution</summary>

```rust
// Solution 1: Return the string
fn main() {
    let s = String::from("hello");
    let s = print_and_return(s);
    let s = print_and_return(s);
}

fn print_and_return(s: String) -> String {
    println!("{}", s);
    s
}

// Solution 2: Use references (covered in next lab)
fn main() {
    let s = String::from("hello");
    print_string(&s);
    print_string(&s);
}

fn print_string(s: &String) {
    println!("{}", s);
}
```

</details>

<details>
<summary>Exercise 7 Solution</summary>

```rust
fn build_greeting(name: String) -> String {
    let mut greeting = String::from("Hello, ");
    greeting.push_str(&name);
    greeting.push('!');
    greeting
}

fn main() {
    let name = String::from("Alice");
    let greeting = build_greeting(name);
    println!("{}", greeting);
}
```

</details>

## Key Learnings
- Ownership is transferred when assigning or passing to functions
- Use `clone()` for deep copies (expensive)
- Copy types are automatically copied
- Return values to transfer ownership back
- Tuples can return multiple values

## Next Lab
Proceed to Lab 3.2 to learn about references and borrowing!
