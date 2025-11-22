# Lab 5.2: Integration Testing

## Objective
Create integration tests for your Rust projects.

## Exercises

### Exercise 1: Integration Test File
Create `tests/integration_test.rs`:
```rust
use my_crate;

#[test]
fn it_adds_two() {
    assert_eq!(my_crate::add(2, 2), 4);
}
```

### Exercise 2: Test Module Organization
```
tests/
├── common/
│   └── mod.rs
└── integration_test.rs
```

### Exercise 3: Testing Public API
```rust
#[test]
fn test_public_interface() {
    let result = my_crate::process_data("input");
    assert!(result.is_ok());
}
```

## Success Criteria
✅ Create integration tests  
✅ Organize test files  
✅ Test public API

## Next Steps
Lab 5.3: Test-Driven Development
