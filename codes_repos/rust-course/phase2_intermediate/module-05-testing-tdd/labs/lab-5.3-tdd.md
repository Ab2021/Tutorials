# Lab 5.3: Test-Driven Development

## Objective
Practice TDD workflow: write test first, then implementation.

## Exercises

### Exercise 1: TDD Cycle
```rust
// Step 1: Write failing test
#[test]
fn test_calculator_add() {
    let calc = Calculator::new();
    assert_eq!(calc.add(2, 3), 5);
}

// Step 2: Implement minimum code
struct Calculator;

impl Calculator {
    fn new() -> Self {
        Calculator
    }
    
    fn add(&self, a: i32, b: i32) -> i32 {
        a + b
    }
}

// Step 3: Refactor
```

### Exercise 2: Red-Green-Refactor
Practice the TDD cycle for a string reverser.

## Success Criteria
✅ Write tests first  
✅ Implement to pass tests  
✅ Refactor with confidence

## Next Steps
Lab 5.4: Mocking and Test Doubles
