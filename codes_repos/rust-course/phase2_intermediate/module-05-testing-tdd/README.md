# Module 9: Testing and Documentation

## 🎯 Learning Objectives
- Write unit tests
- Create integration tests
- Use test organization best practices
- Write documentation comments
- Generate documentation with cargo doc
- Create doc tests

## 📖 Theoretical Concepts

### 9.1 Writing Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
    
    #[test]
    fn another() {
        panic!("Make this test fail");
    }
}
```

### 9.2 Assertion Macros

```rust
#[test]
fn test_assertions() {
    assert!(true);
    assert_eq!(2 + 2, 4);
    assert_ne!(2 + 2, 5);
}

#[test]
fn test_with_message() {
    let result = 2 + 2;
    assert_eq!(result, 4, "2 + 2 should equal 4, got {}", result);
}
```

### 9.3 Testing for Panics

```rust
#[test]
#[should_panic]
fn test_panic() {
    panic!("This test should panic");
}

#[test]
#[should_panic(expected = "less than or equal to 100")]
fn test_panic_message() {
    Guess::new(200);
}
```

### 9.4 Using Result in Tests

```rust
#[test]
fn it_works() -> Result<(), String> {
    if 2 + 2 == 4 {
        Ok(())
    } else {
        Err(String::from("two plus two does not equal four"))
    }
}
```

### 9.5 Running Tests

```bash
cargo test              # Run all tests
cargo test test_name    # Run specific test
cargo test -- --nocapture  # Show println! output
cargo test -- --test-threads=1  # Run tests sequentially
cargo test -- --ignored  # Run ignored tests
```

### 9.6 Test Organization

**Unit Tests:**
```rust
// src/lib.rs
pub fn add_two(a: i32) -> i32 {
    a + 2
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn it_adds_two() {
        assert_eq!(4, add_two(2));
    }
}
```

**Integration Tests:**
```rust
// tests/integration_test.rs
use my_crate;

#[test]
fn it_adds_two() {
    assert_eq!(4, my_crate::add_two(2));
}
```

### 9.7 Documentation Comments

```rust
/// Adds one to the number given.
///
/// # Examples
///
/// ```
/// let arg = 5;
/// let answer = my_crate::add_one(arg);
///
/// assert_eq!(6, answer);
/// ```
pub fn add_one(x: i32) -> i32 {
    x + 1
}
```

**Sections:**
- `# Examples` - Usage examples
- `# Panics` - Scenarios where function panics
- `# Errors` - Error types returned
- `# Safety` - Why function is unsafe

### 9.8 Doc Tests

Documentation examples are automatically tested:

```bash
cargo test --doc
```

### 9.9 Generating Documentation

```bash
cargo doc --open
```

## 🔑 Key Takeaways
- Tests use `#[test]` attribute
- Assertions: assert!, assert_eq!, assert_ne!
- `#[should_panic]` for panic tests
- Unit tests in same file, integration tests in tests/
- Doc comments with /// generate documentation
- Doc examples are tested automatically

## ⏭️ Next Steps
Complete the labs and move to Module 10: Advanced Topics
