# 📋 Rust Quick Reference Guide

A concise reference for common Rust patterns and syntax.

## Variables and Types

```rust
// Immutable variable
let x = 5;

// Mutable variable
let mut y = 10;

// Constant
const MAX_POINTS: u32 = 100_000;

// Type annotation
let z: i32 = 42;

// Shadowing
let x = x + 1;
```

## Data Types

```rust
// Integers: i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize
let a: i32 = 42;

// Floats: f32, f64
let b: f64 = 3.14;

// Boolean
let c: bool = true;

// Character
let d: char = '🦀';

// Tuple
let tup: (i32, f64, u8) = (500, 6.4, 1);

// Array
let arr: [i32; 5] = [1, 2, 3, 4, 5];
```

## Functions

```rust
fn function_name(param: i32) -> i32 {
    param + 1  // Expression (no semicolon)
}
```

## Control Flow

```rust
// if/else
if condition {
    // code
} else if other_condition {
    // code
} else {
    // code
}

// loop
loop {
    break;
}

// while
while condition {
    // code
}

// for
for item in collection {
    // code
}

for i in 0..10 {  // Range
    // code
}
```

## Pattern Matching

```rust
match value {
    1 => println!("one"),
    2 | 3 => println!("two or three"),
    4..=10 => println!("four through ten"),
    _ => println!("anything else"),
}

// if let
if let Some(value) = option {
    // code
}
```

## Ownership

```rust
// Move
let s1 = String::from("hello");
let s2 = s1;  // s1 is no longer valid

// Clone
let s1 = String::from("hello");
let s2 = s1.clone();  // Both valid

// Copy (for stack types)
let x = 5;
let y = x;  // Both valid
```

## References and Borrowing

```rust
// Immutable reference
let s = String::from("hello");
let r = &s;

// Mutable reference
let mut s = String::from("hello");
let r = &mut s;

// Rules:
// - One mutable OR many immutable references
// - References must always be valid
```

## Structs

```rust
// Definition
struct User {
    username: String,
    email: String,
    active: bool,
}

// Instantiation
let user = User {
    username: String::from("user"),
    email: String::from("user@example.com"),
    active: true,
};

// Methods
impl User {
    fn new(username: String, email: String) -> User {
        User {
            username,
            email,
            active: true,
        }
    }
    
    fn deactivate(&mut self) {
        self.active = false;
    }
}
```

## Enums

```rust
enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
    ChangeColor(i32, i32, i32),
}

// Option
let some_number: Option<i32> = Some(5);
let absent_number: Option<i32> = None;

// Result
let result: Result<i32, String> = Ok(42);
let error: Result<i32, String> = Err(String::from("error"));
```

## Error Handling

```rust
// panic!
panic!("crash and burn");

// Result
fn divide(a: f64, b: f64) -> Result<f64, String> {
    if b == 0.0 {
        Err(String::from("division by zero"))
    } else {
        Ok(a / b)
    }
}

// ? operator
fn read_file() -> Result<String, io::Error> {
    let mut file = File::open("file.txt")?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}

// unwrap and expect
let x = some_result.unwrap();
let y = some_result.expect("error message");
```

## Collections

```rust
// Vector
let mut v: Vec<i32> = Vec::new();
let v = vec![1, 2, 3];
v.push(4);
let third = &v[2];

// String
let mut s = String::new();
let s = String::from("hello");
s.push_str(" world");
s.push('!');

// HashMap
use std::collections::HashMap;
let mut map = HashMap::new();
map.insert(String::from("key"), 10);
let value = map.get("key");
```

## Iterators

```rust
let v = vec![1, 2, 3];

// Iterate
for item in &v {
    println!("{}", item);
}

// Map
let v2: Vec<_> = v.iter().map(|x| x + 1).collect();

// Filter
let evens: Vec<_> = v.iter().filter(|&&x| x % 2 == 0).collect();

// Sum
let sum: i32 = v.iter().sum();
```

## Generics

```rust
fn largest<T: PartialOrd>(list: &[T]) -> &T {
    let mut largest = &list[0];
    for item in list {
        if item > largest {
            largest = item;
        }
    }
    largest
}

struct Point<T> {
    x: T,
    y: T,
}
```

## Traits

```rust
trait Summary {
    fn summarize(&self) -> String;
}

impl Summary for Article {
    fn summarize(&self) -> String {
        format!("{} by {}", self.headline, self.author)
    }
}

// Trait bounds
fn notify<T: Summary>(item: &T) {
    println!("{}", item.summarize());
}
```

## Lifetimes

```rust
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}

struct ImportantExcerpt<'a> {
    part: &'a str,
}
```

## Common Macros

```rust
println!("Hello, {}!", name);
format!("x = {}, y = {}", x, y);
vec![1, 2, 3];
panic!("error message");
assert_eq!(left, right);
```

## Cargo Commands

```bash
cargo new project_name      # Create new project
cargo build                 # Build project
cargo run                   # Build and run
cargo check                 # Check compilation
cargo test                  # Run tests
cargo doc --open            # Generate docs
cargo build --release       # Optimized build
```

## Useful Attributes

```rust
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
#[cfg(test)]
#[test]
#[should_panic]
```

---

**Keep this handy while learning Rust!** 🦀
