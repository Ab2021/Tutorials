# Module 7: Generics and Traits

## 🎯 Learning Objectives
- Write generic functions and structs
- Define and implement traits
- Use trait bounds
- Understand trait objects and dynamic dispatch
- Work with common standard library traits

## 📖 Theoretical Concepts

### 7.1 Generic Data Types

#### Generic Functions
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

fn main() {
    let numbers = vec![34, 50, 25, 100, 65];
    println!("The largest number is {}", largest(&numbers));
    
    let chars = vec!['y', 'm', 'a', 'q'];
    println!("The largest char is {}", largest(&chars));
}
```

#### Generic Structs
```rust
struct Point<T> {
    x: T,
    y: T,
}

let integer = Point { x: 5, y: 10 };
let float = Point { x: 1.0, y: 4.0 };

// Multiple type parameters
struct Point<T, U> {
    x: T,
    y: U,
}

let mixed = Point { x: 5, y: 4.0 };
```

#### Generic Enums
```rust
enum Option<T> {
    Some(T),
    None,
}

enum Result<T, E> {
    Ok(T),
    Err(E),
}
```

#### Generic Methods
```rust
impl<T> Point<T> {
    fn x(&self) -> &T {
        &self.x
    }
}

impl Point<f32> {
    fn distance_from_origin(&self) -> f32 {
        (self.x.powi(2) + self.y.powi(2)).sqrt()
    }
}
```

### 7.2 Traits

Define shared behavior:

```rust
pub trait Summary {
    fn summarize(&self) -> String;
}

pub struct NewsArticle {
    pub headline: String,
    pub location: String,
    pub author: String,
    pub content: String,
}

impl Summary for NewsArticle {
    fn summarize(&self) -> String {
        format!("{}, by {} ({})", self.headline, self.author, self.location)
    }
}

pub struct Tweet {
    pub username: String,
    pub content: String,
    pub reply: bool,
    pub retweet: bool,
}

impl Summary for Tweet {
    fn summarize(&self) -> String {
        format!("{}: {}", self.username, self.content)
    }
}
```

#### Default Implementations
```rust
pub trait Summary {
    fn summarize_author(&self) -> String;
    
    fn summarize(&self) -> String {
        format!("(Read more from {}...)", self.summarize_author())
    }
}
```

### 7.3 Traits as Parameters

```rust
pub fn notify(item: &impl Summary) {
    println!("Breaking news! {}", item.summarize());
}

// Trait bound syntax
pub fn notify<T: Summary>(item: &T) {
    println!("Breaking news! {}", item.summarize());
}

// Multiple trait bounds
pub fn notify<T: Summary + Display>(item: &T) {
    // ...
}

// where clause
fn some_function<T, U>(t: &T, u: &U) -> i32
    where T: Display + Clone,
          U: Clone + Debug
{
    // ...
}
```

### 7.4 Returning Traits

```rust
fn returns_summarizable() -> impl Summary {
    Tweet {
        username: String::from("horse_ebooks"),
        content: String::from("of course, as you probably already know"),
        reply: false,
        retweet: false,
    }
}
```

### 7.5 Common Traits

#### Debug
```rust
#[derive(Debug)]
struct Rectangle {
    width: u32,
    height: u32,
}

let rect = Rectangle { width: 30, height: 50 };
println!("{:?}", rect);
println!("{:#?}", rect);  // Pretty print
```

#### Clone and Copy
```rust
#[derive(Clone, Copy)]
struct Point {
    x: i32,
    y: i32,
}
```

#### PartialEq and Eq
```rust
#[derive(PartialEq, Eq)]
struct Person {
    name: String,
    age: u32,
}
```

#### PartialOrd and Ord
```rust
#[derive(PartialOrd, Ord, PartialEq, Eq)]
struct Height(u32);
```

### 7.6 Operator Overloading

```rust
use std::ops::Add;

#[derive(Debug, PartialEq)]
struct Point {
    x: i32,
    y: i32,
}

impl Add for Point {
    type Output = Point;
    
    fn add(self, other: Point) -> Point {
        Point {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

let p1 = Point { x: 1, y: 2 };
let p2 = Point { x: 3, y: 4 };
let p3 = p1 + p2;
```

### 7.7 Trait Objects

Dynamic dispatch:

```rust
pub trait Draw {
    fn draw(&self);
}

pub struct Screen {
    pub components: Vec<Box<dyn Draw>>,
}

impl Screen {
    pub fn run(&self) {
        for component in self.components.iter() {
            component.draw();
        }
    }
}
```

## 🔑 Key Takeaways
- Generics enable code reuse
- Traits define shared behavior
- Trait bounds constrain generic types
- Derive common traits automatically
- Operator overloading via traits
- Trait objects for dynamic dispatch

## ⏭️ Next Steps
Complete the labs and move to Module 8: Lifetimes
