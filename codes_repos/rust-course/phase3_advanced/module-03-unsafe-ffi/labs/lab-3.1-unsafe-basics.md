# Lab 3.1: Unsafe Rust Basics

## Objective
Understand when and how to use unsafe Rust.

## Exercises

### Exercise 1: Raw Pointers
```rust
fn main() {
    let mut num = 5;
    
    let r1 = &num as *const i32;
    let r2 = &mut num as *mut i32;
    
    unsafe {
        println!("r1: {}", *r1);
        *r2 = 10;
        println!("r2: {}", *r2);
    }
}
```

### Exercise 2: Unsafe Functions
```rust
unsafe fn dangerous() {
    println!("This is unsafe!");
}

fn main() {
    unsafe {
        dangerous();
    }
}
```

### Exercise 3: Safe Abstraction
```rust
fn split_at_mut(slice: &mut [i32], mid: usize) -> (&mut [i32], &mut [i32]) {
    let len = slice.len();
    let ptr = slice.as_mut_ptr();
    
    assert!(mid <= len);
    
    unsafe {
        (
            std::slice::from_raw_parts_mut(ptr, mid),
            std::slice::from_raw_parts_mut(ptr.add(mid), len - mid),
        )
    }
}
```

## Success Criteria
✅ Use raw pointers  
✅ Call unsafe functions  
✅ Build safe abstractions

## Next Steps
Lab 3.2: FFI with C
