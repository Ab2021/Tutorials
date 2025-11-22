# Module 06: Documentation and APIs

## 🎯 Learning Objectives

- Write effective documentation comments
- Design clean, usable APIs
- Understand semantic versioning
- Publish crates to crates.io
- Maintain public libraries

---

## 📖 Core Concepts

### Documentation Comments

```rust
/// Adds two numbers together.
///
/// # Examples
///
/// ```
/// let result = my_crate::add(2, 3);
/// assert_eq!(result, 5);
/// ```
///
/// # Panics
///
/// This function will panic if the result overflows.
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
```

### Module-Level Documentation

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

### API Design Principles

1. **Consistency** - Similar operations should work similarly
2. **Discoverability** - Easy to find what you need
3. **Ergonomics** - Pleasant to use
4. **Safety** - Hard to misuse
5. **Performance** - Efficient by default

### Semantic Versioning

- **MAJOR**: Breaking changes (1.0.0 → 2.0.0)
- **MINOR**: New features (1.0.0 → 1.1.0)
- **PATCH**: Bug fixes (1.0.0 → 1.0.1)

### Publishing to crates.io

```toml
[package]
name = "my_crate"
version = "0.1.0"
authors = ["Your Name <you@example.com>"]
edition = "2021"
description = "A short description"
license = "MIT OR Apache-2.0"
repository = "https://github.com/user/repo"
keywords = ["keyword1", "keyword2"]
categories = ["category"]
```

```bash
cargo login <token>
cargo publish
```

---

## 🔑 Key Takeaways

1. **Documentation is code** - Keep it updated
2. **Examples are essential** - Show how to use your API
3. **Semantic versioning** - Communicate changes clearly
4. **API design matters** - Think about users
5. **Publishing is easy** - Share your work

Complete 10 labs, then proceed to Module 07: Design Patterns
