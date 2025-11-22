# Module 4: Structs and Enums

## 🎯 Learning Objectives
- Define and use structs to create custom data types
- Implement methods and associated functions
- Use enums for types with multiple variants
- Work with Option and Result enums
- Organize code with modules

## 📖 Theoretical Concepts

### 4.1 Defining Structs

```rust
struct User {
    username: String,
    email: String,
    sign_in_count: u64,
    active: bool,
}

fn main() {
    let user1 = User {
        email: String::from("user@example.com"),
        username: String::from("someuser"),
        active: true,
        sign_in_count: 1,
    };
    
    println!("Username: {}", user1.username);
}
```

#### Mutable Structs
```rust
let mut user1 = User {
    email: String::from("user@example.com"),
    username: String::from("someuser"),
    active: true,
    sign_in_count: 1,
};

user1.email = String::from("newemail@example.com");
```

#### Field Init Shorthand
```rust
fn build_user(email: String, username: String) -> User {
    User {
        email,      // Shorthand for email: email
        username,   // Shorthand for username: username
        active: true,
        sign_in_count: 1,
    }
}
```

#### Struct Update Syntax
```rust
let user2 = User {
    email: String::from("another@example.com"),
    ..user1  // Copy remaining fields from user1
};
```

### 4.2 Tuple Structs
```rust
struct Color(i32, i32, i32);
struct Point(i32, i32, i32);

let black = Color(0, 0, 0);
let origin = Point(0, 0, 0);
```

### 4.3 Method Syntax

```rust
struct Rectangle {
    width: u32,
    height: u32,
}

impl Rectangle {
    // Method (takes &self)
    fn area(&self) -> u32 {
        self.width * self.height
    }
    
    fn can_hold(&self, other: &Rectangle) -> bool {
        self.width > other.width && self.height > other.height
    }
    
    // Associated function (doesn't take self)
    fn square(size: u32) -> Rectangle {
        Rectangle {
            width: size,
            height: size,
        }
    }
}

fn main() {
    let rect = Rectangle { width: 30, height: 50 };
    println!("Area: {}", rect.area());
    
    let square = Rectangle::square(20);
}
```

### 4.4 Enums

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

#### Complex Enums
```rust
enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
    ChangeColor(i32, i32, i32),
}

impl Message {
    fn call(&self) {
        // Method implementation
    }
}
```

### 4.5 The Option Enum

```rust
enum Option<T> {
    Some(T),
    None,
}

let some_number = Some(5);
let some_string = Some("a string");
let absent_number: Option<i32> = None;
```

**Why Option?**
- Rust doesn't have null
- Option represents a value that might be absent
- Forces you to handle the None case

```rust
fn divide(numerator: f64, denominator: f64) -> Option<f64> {
    if denominator == 0.0 {
        None
    } else {
        Some(numerator / denominator)
    }
}

match divide(10.0, 2.0) {
    Some(result) => println!("Result: {}", result),
    None => println!("Cannot divide by zero"),
}
```

### 4.6 Pattern Matching with Enums

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

#### Matching with Option
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

### 4.7 if let

Concise alternative to match for single pattern:

```rust
let some_value = Some(3);

// Using match
match some_value {
    Some(3) => println!("three"),
    _ => (),
}

// Using if let
if let Some(3) = some_value {
    println!("three");
}
```

## 🔑 Key Takeaways
- Structs group related data
- Methods are defined in impl blocks
- Enums represent types with multiple variants
- Option<T> replaces null
- Pattern matching handles all enum variants
- if let provides concise single-pattern matching

## ⏭️ Next Steps
Complete the labs and move to Module 5: Error Handling
