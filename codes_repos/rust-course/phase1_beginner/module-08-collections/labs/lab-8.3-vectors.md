# Lab 8.3: Working with Vectors

## Objective
Master Vec<T> for dynamic arrays in Rust.

## Theory
Vectors are growable arrays stored on the heap.

## Exercises

### Exercise 1: Creating Vectors
```rust
fn main() {
    // Using vec! macro
    let v1 = vec![1, 2, 3];
    
    // Using Vec::new()
    let mut v2: Vec<i32> = Vec::new();
    v2.push(1);
    v2.push(2);
    v2.push(3);
    
    // With capacity
    let mut v3 = Vec::with_capacity(10);
}
```

### Exercise 2: Accessing Elements
```rust
fn main() {
    let v = vec![1, 2, 3, 4, 5];
    
    // Using indexing (can panic)
    let third = v[2];
    
    // Using get (returns Option)
    match v.get(2) {
        Some(third) => println!("Third: {}", third),
        None => println!("No third element"),
    }
}
```

### Exercise 3: Iterating
```rust
fn main() {
    let v = vec![100, 32, 57];
    
    // Immutable iteration
    for i in &v {
        println!("{}", i);
    }
    
    // Mutable iteration
    let mut v = vec![100, 32, 57];
    for i in &mut v {
        *i += 50;
    }
}
```

### Exercise 4: Vector Methods
```rust
fn main() {
    let mut v = vec![1, 2, 3];
    
    v.push(4);           // Add element
    v.pop();             // Remove last
    v.insert(1, 10);     // Insert at index
    v.remove(1);         // Remove at index
    
    println!("Length: {}", v.len());
    println!("Capacity: {}", v.capacity());
    println!("Is empty: {}", v.is_empty());
}
```

### Exercise 5: Storing Different Types
```rust
enum SpreadsheetCell {
    Int(i32),
    Float(f64),
    Text(String),
}

fn main() {
    let row = vec![
        SpreadsheetCell::Int(3),
        SpreadsheetCell::Text(String::from("blue")),
        SpreadsheetCell::Float(10.12),
    ];
}
```

## Success Criteria
✅ Create and initialize vectors  
✅ Access elements safely  
✅ Iterate and modify vectors  
✅ Use vector methods effectively

## Next Steps
Proceed to Lab 8.4: String Manipulation
