# Module 03: Unsafe Rust and FFI

## 🎯 Learning Objectives

- Understand unsafe Rust
- Work with raw pointers
- Call C from Rust
- Call Rust from C
- Build safe abstractions over unsafe code

---

## 📖 Core Concepts

### Unsafe Superpowers

```rust
unsafe {
    // Dereference raw pointers
    // Call unsafe functions
    // Access mutable statics
    // Implement unsafe traits
    // Access union fields
}
```

### Raw Pointers

```rust
let mut num = 5;

let r1 = &num as *const i32;
let r2 = &mut num as *mut i32;

unsafe {
    println!("r1: {}", *r1);
    *r2 = 10;
}
```

### Calling C from Rust

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

### Calling Rust from C

```rust
#[no_mangle]
pub extern "C" fn rust_function(x: i32) -> i32 {
    x + 1
}
```

---

## 🔑 Key Takeaways

1. **unsafe** doesn't turn off borrow checker
2. **Raw pointers** for low-level control
3. **FFI** for C interoperability
4. **Safe abstractions** over unsafe code
5. **Minimize unsafe** code surface

Complete 10 labs, then proceed to Module 04: Macros
