# Lab 3.2: FFI - Calling C from Rust

## Objective
Call C functions from Rust code.

## Exercises

### Exercise 1: Calling C Standard Library
```rust
extern "C" {
    fn abs(input: i32) -> i32;
}

fn main() {
    unsafe {
        println!("Absolute value: {}", abs(-3));
    }
}
```

### Exercise 2: Custom C Library
Create `mylib.c`:
```c
int add(int a, int b) {
    return a + b;
}
```

In Rust:
```rust
#[link(name = "mylib")]
extern "C" {
    fn add(a: i32, b: i32) -> i32;
}

fn main() {
    unsafe {
        println!("Sum: {}", add(5, 3));
    }
}
```

## Success Criteria
✅ Call C functions  
✅ Link C libraries  
✅ Handle C types

## Next Steps
Lab 3.3: Calling Rust from C
