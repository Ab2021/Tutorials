# Lab 5.3: Slices

## Objective
Master string slices and array slices for safe, efficient data access.

## Theory
Slices let you reference a contiguous sequence of elements without taking ownership.

## Exercises

### Exercise 1: String Slices
```rust
fn main() {
    let s = String::from("hello world");
    
    let hello = &s[0..5];
    let world = &s[6..11];
    
    println!("{} {}", hello, world);
}
```

### Exercise 2: First Word Function
```rust
fn first_word(s: &str) -> &str {
    let bytes = s.as_bytes();
    
    for (i, &item) in bytes.iter().enumerate() {
        if item == b' ' {
            return &s[0..i];
        }
    }
    
    &s[..]
}
```

### Exercise 3: Array Slices
```rust
fn main() {
    let a = [1, 2, 3, 4, 5];
    let slice = &a[1..3];
    
    assert_eq!(slice, &[2, 3]);
}
```

### Exercise 4: Slice Patterns
```rust
fn analyze_slice(slice: &[i32]) {
    println!("First element: {}", slice[0]);
    println!("Length: {}", slice.len());
}
```

### Exercise 5: String Slice vs String
```rust
fn print_string(s: &str) {
    println!("{}", s);
}

fn main() {
    let s = String::from("hello");
    print_string(&s);  // Works with String
    print_string("world");  // Works with &str
}
```

## Success Criteria
✅ Understand slice syntax  
✅ Can create string and array slices  
✅ Know the difference between &str and String  
✅ Use slices in function parameters

## Next Steps
Proceed to Lab 5.4: Advanced Reference Patterns
