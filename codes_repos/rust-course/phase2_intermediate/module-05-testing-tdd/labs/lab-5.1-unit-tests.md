# Lab 5.1: Unit Testing Basics

## Objective
Write effective unit tests in Rust.

## Exercises

### Exercise 1: Basic Test
```rust
#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
    
    #[test]
    fn it_fails() {
        panic!("This test fails");
    }
}
```

### Exercise 2: Testing Functions
```rust
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_add() {
        assert_eq!(add(2, 2), 4);
        assert_eq!(add(-1, 1), 0);
    }
}
```

### Exercise 3: Assert Macros
```rust
#[test]
fn test_assertions() {
    assert!(true);
    assert_eq!(1, 1);
    assert_ne!(1, 2);
}
```

## Success Criteria
✅ Write unit tests  
✅ Use assert macros  
✅ Run tests with cargo test

## Next Steps
Lab 5.2: Integration Testing
