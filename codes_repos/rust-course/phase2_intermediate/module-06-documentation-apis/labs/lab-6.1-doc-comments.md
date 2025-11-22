# Lab 6.1: Documentation Comments

## Objective
Write effective documentation for Rust code.

## Exercises

### Exercise 1: Function Documentation
```rust
/// Adds two numbers together.
///
/// # Examples
///
/// ```
/// let result = add(2, 3);
/// assert_eq!(result, 5);
/// ```
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
```

### Exercise 2: Module Documentation
```rust
//! # My Crate
//!
//! `my_crate` provides utilities for...
//!
//! ## Quick Start
//!
//! ```
//! use my_crate::MyStruct;
//! ```
```

### Exercise 3: Struct Documentation
```rust
/// A rectangle with width and height.
///
/// # Examples
///
/// ```
/// let rect = Rectangle::new(30, 50);
/// assert_eq!(rect.area(), 1500);
/// ```
pub struct Rectangle {
    width: u32,
    height: u32,
}
```

## Success Criteria
✅ Write doc comments  
✅ Include examples  
✅ Generate docs with cargo doc

## Next Steps
Lab 6.2: API Design Principles
