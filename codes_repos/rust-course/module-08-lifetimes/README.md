# Module 8: Lifetimes

## 🎯 Learning Objectives
- Understand lifetime annotations
- Prevent dangling references
- Use lifetime syntax in functions and structs
- Apply lifetime elision rules
- Work with the 'static lifetime

## 📖 Theoretical Concepts

### 8.1 What are Lifetimes?

Lifetimes ensure references are valid as long as we need them.

```rust
{
    let r;                // ---------+-- 'a
                          //          |
    {                     //          |
        let x = 5;        // -+-- 'b  |
        r = &x;           //  |       |
    }                     // -+       |
                          //          |
    println!("r: {}", r); //          |
}                         // ---------+
// ERROR: x doesn't live long enough
```

### 8.2 Lifetime Annotation Syntax

```rust
&i32        // a reference
&'a i32     // a reference with an explicit lifetime
&'a mut i32 // a mutable reference with an explicit lifetime
```

### 8.3 Lifetime Annotations in Functions

```rust
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}

fn main() {
    let string1 = String::from("abcd");
    let string2 = "xyz";
    
    let result = longest(string1.as_str(), string2);
    println!("The longest string is {}", result);
}
```

**What it means:**
- The returned reference will be valid as long as both parameters are valid
- The lifetime of the return value is the smaller of the lifetimes of the parameters

```rust
fn main() {
    let string1 = String::from("long string");
    let result;
    
    {
        let string2 = String::from("xyz");
        result = longest(string1.as_str(), string2.as_str());
    }  // string2 goes out of scope
    
    println!("The longest string is {}", result);  // ERROR
}
```

### 8.4 Thinking in Terms of Lifetimes

```rust
// This works - return value lifetime tied to first parameter
fn first_word<'a>(s: &'a str, _other: &str) -> &'a str {
    let bytes = s.as_bytes();
    
    for (i, &item) in bytes.iter().enumerate() {
        if item == b' ' {
            return &s[0..i];
        }
    }
    
    &s[..]
}
```

### 8.5 Lifetime Annotations in Structs

```rust
struct ImportantExcerpt<'a> {
    part: &'a str,
}

fn main() {
    let novel = String::from("Call me Ishmael. Some years ago...");
    let first_sentence = novel.split('.').next().expect("Could not find a '.'");
    
    let i = ImportantExcerpt {
        part: first_sentence,
    };
}
```

The struct can't outlive the reference it holds.

### 8.6 Lifetime Elision Rules

The compiler can infer lifetimes in some cases:

**Rule 1:** Each parameter gets its own lifetime
```rust
fn foo(x: &i32)  // becomes fn foo<'a>(x: &'a i32)
```

**Rule 2:** If there's exactly one input lifetime, it's assigned to all output lifetimes
```rust
fn foo(x: &i32) -> &i32  // becomes fn foo<'a>(x: &'a i32) -> &'a i32
```

**Rule 3:** If there are multiple input lifetimes, but one is `&self` or `&mut self`, the lifetime of `self` is assigned to all output lifetimes

```rust
impl<'a> ImportantExcerpt<'a> {
    fn announce_and_return_part(&self, announcement: &str) -> &str {
        println!("Attention please: {}", announcement);
        self.part
    }
}
```

### 8.7 Method Definitions with Lifetimes

```rust
impl<'a> ImportantExcerpt<'a> {
    fn level(&self) -> i32 {
        3
    }
    
    fn announce_and_return_part(&self, announcement: &str) -> &str {
        println!("Attention please: {}", announcement);
        self.part
    }
}
```

### 8.8 The Static Lifetime

```rust
let s: &'static str = "I have a static lifetime.";
```

**'static** means the reference can live for the entire duration of the program.

All string literals have the 'static lifetime.

### 8.9 Generic Type Parameters, Trait Bounds, and Lifetimes Together

```rust
use std::fmt::Display;

fn longest_with_an_announcement<'a, T>(
    x: &'a str,
    y: &'a str,
    ann: T,
) -> &'a str
where
    T: Display,
{
    println!("Announcement! {}", ann);
    if x.len() > y.len() {
        x
    } else {
        y
    }
}
```

## 🔑 Key Takeaways
- Lifetimes prevent dangling references
- Lifetime annotations describe relationships
- Compiler can often infer lifetimes (elision)
- Structs with references need lifetime parameters
- 'static means lives for entire program
- Lifetimes work with generics and traits

## ⏭️ Next Steps
Complete the labs and move to Module 9: Testing and Documentation
