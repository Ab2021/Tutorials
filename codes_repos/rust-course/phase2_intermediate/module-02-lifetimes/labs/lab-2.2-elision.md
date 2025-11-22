# Lab 2.2: Lifetime Elision Rules

## Objective
Understand when Rust infers lifetimes automatically.

## Theory
Rust has three lifetime elision rules that allow you to omit lifetime annotations in common cases.

## Exercises

### Exercise 1: Input Lifetime Rule
```rust
// Compiler infers: fn first_word<'a>(s: &'a str) -> &'a str
fn first_word(s: &str) -> &str {
    &s[..1]
}
```

### Exercise 2: Method Lifetime Rule
```rust
struct Parser<'a> {
    text: &'a str,
}

impl<'a> Parser<'a> {
    // Compiler infers lifetime from &self
    fn parse(&self) -> &str {
        self.text
    }
}
```

### Exercise 3: When Elision Doesn't Apply
```rust
// Need explicit lifetimes - multiple inputs
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}
```

## Success Criteria
✅ Understand elision rules  
✅ Know when to use explicit lifetimes  
✅ Write cleaner code with elision

## Next Steps
Lab 2.3: Lifetime Bounds
