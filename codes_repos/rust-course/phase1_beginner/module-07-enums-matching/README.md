# Module 07: Enums and Pattern Matching

## 🎯 Learning Objectives

- Define and use enums
- Master pattern matching with match
- Work with Option<T> and Result<T, E>
- Use if let and while let
- Build state machines with enums

---

## 📖 Theoretical Concepts

### 7.1 Defining Enums

```rust
enum IpAddrKind {
    V4,
    V6,
}

let four = IpAddrKind::V4;
let six = IpAddrKind::V6;
```

#### Enums with Data

```rust
enum IpAddr {
    V4(u8, u8, u8, u8),
    V6(String),
}

let home = IpAddr::V4(127, 0, 0, 1);
let loopback = IpAddr::V6(String::from("::1"));
```

#### Different Types Per Variant

```rust
enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
    ChangeColor(i32, i32, i32),
}
```

---

### 7.2 The Option Enum

Rust doesn't have null. Instead, it has `Option<T>`:

```rust
enum Option<T> {
    Some(T),
    None,
}
```

**Usage:**

```rust
let some_number = Some(5);
let some_string = Some("a string");
let absent_number: Option<i32> = None;

// Must handle None case
let x: i8 = 5;
let y: Option<i8> = Some(5);

let sum = x + y;  // ❌ ERROR: can't add i8 and Option<i8>

// ✅ Must extract value first
let sum = x + y.unwrap_or(0);
```

---

### 7.3 Pattern Matching with match

```rust
enum Coin {
    Penny,
    Nickel,
    Dime,
    Quarter,
}

fn value_in_cents(coin: Coin) -> u8 {
    match coin {
        Coin::Penny => 1,
        Coin::Nickel => 5,
        Coin::Dime => 10,
        Coin::Quarter => 25,
    }
}
```

#### Patterns That Bind to Values

```rust
#[derive(Debug)]
enum UsState {
    Alabama,
    Alaska,
    // ...
}

enum Coin {
    Penny,
    Nickel,
    Dime,
    Quarter(UsState),
}

fn value_in_cents(coin: Coin) -> u8 {
    match coin {
        Coin::Penny => 1,
        Coin::Nickel => 5,
        Coin::Dime => 10,
        Coin::Quarter(state) => {
            println!("State quarter from {:?}!", state);
            25
        }
    }
}
```

#### Matching Option<T>

```rust
fn plus_one(x: Option<i32>) -> Option<i32> {
    match x {
        None => None,
        Some(i) => Some(i + 1),
    }
}

let five = Some(5);
let six = plus_one(five);
let none = plus_one(None);
```

---

### 7.4 The Result Enum

For operations that can fail:

```rust
enum Result<T, E> {
    Ok(T),
    Err(E),
}
```

**Usage:**

```rust
use std::fs::File;

fn main() {
    let f = File::open("hello.txt");
    
    let f = match f {
        Ok(file) => file,
        Err(error) => panic!("Problem opening file: {:?}", error),
    };
}
```

---

### 7.5 Concise Control Flow with if let

```rust
let some_value = Some(3);

// Using match
match some_value {
    Some(3) => println!("three"),
    _ => (),
}

// Using if let (more concise)
if let Some(3) = some_value {
    println!("three");
}
```

#### if let with else

```rust
let mut count = 0;
let coin = Coin::Quarter(UsState::Alaska);

if let Coin::Quarter(state) = coin {
    println!("State quarter from {:?}!", state);
} else {
    count += 1;
}
```

---

### 7.6 while let

```rust
let mut stack = Vec::new();
stack.push(1);
stack.push(2);
stack.push(3);

while let Some(top) = stack.pop() {
    println!("{}", top);
}
```

---

### 7.7 Methods on Enums

```rust
impl Message {
    fn call(&self) {
        match self {
            Message::Quit => println!("Quit"),
            Message::Move { x, y } => println!("Move to ({}, {})", x, y),
            Message::Write(text) => println!("Text: {}", text),
            Message::ChangeColor(r, g, b) => println!("Color: ({}, {}, {})", r, g, b),
        }
    }
}

let m = Message::Write(String::from("hello"));
m.call();
```

---

## 🔑 Key Takeaways

1. **Enums** define types with multiple variants
2. **Variants can hold data** of different types
3. **Option<T>** replaces null
4. **Result<T, E>** for error handling
5. **match** must be exhaustive
6. **if let** for concise single-pattern matching
7. **Enums are powerful** for state machines

---

## ⏭️ Next Steps

Complete the 10 labs in this module:
1. Lab 7.1: Enum basics
2. Lab 7.2: Option<T>
3. Lab 7.3: Result<T, E>
4. Lab 7.4: Match expressions
5. Lab 7.5: if let / while let
6. Lab 7.6: Enum variants with data
7. Lab 7.7: Message passing
8. Lab 7.8: State machines
9. Lab 7.9: Error types
10. Lab 7.10: Enum project

Then proceed to Module 08: Collections
