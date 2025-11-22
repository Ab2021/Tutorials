# Lab 10.2: File-Based Modules

## Objective
Learn to organize modules across multiple files.

## Theory
As projects grow, splitting modules into separate files improves organization.

## Exercises

### Exercise 1: Single File Module
Create `src/greetings.rs`:
```rust
pub fn hello() {
    println!("Hello!");
}

pub fn goodbye() {
    println!("Goodbye!");
}
```

In `src/main.rs`:
```rust
mod greetings;

fn main() {
    greetings::hello();
    greetings::goodbye();
}
```

### Exercise 2: Module Directory
Create `src/math/mod.rs`:
```rust
pub mod operations;
pub mod constants;
```

Create `src/math/operations.rs`:
```rust
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

pub fn subtract(a: i32, b: i32) -> i32 {
    a - b
}
```

Create `src/math/constants.rs`:
```rust
pub const PI: f64 = 3.14159;
pub const E: f64 = 2.71828;
```

In `src/main.rs`:
```rust
mod math;

fn main() {
    let sum = math::operations::add(5, 3);
    println!("PI = {}", math::constants::PI);
}
```

### Exercise 3: Nested Module Files
```
src/
├── main.rs
└── restaurant/
    ├── mod.rs
    ├── front_of_house/
    │   ├── mod.rs
    │   └── hosting.rs
    └── back_of_house.rs
```

### Exercise 4: Re-exports
In `src/lib.rs`:
```rust
mod front_of_house;

pub use crate::front_of_house::hosting;

pub fn eat_at_restaurant() {
    hosting::add_to_waitlist();
}
```

### Exercise 5: Project Structure
```
my_project/
├── Cargo.toml
├── src/
│   ├── main.rs
│   ├── lib.rs
│   ├── utils/
│   │   ├── mod.rs
│   │   ├── string_utils.rs
│   │   └── math_utils.rs
│   └── models/
│       ├── mod.rs
│       ├── user.rs
│       └── product.rs
```

## Success Criteria
✅ Create file-based modules  
✅ Organize modules in directories  
✅ Use mod.rs correctly  
✅ Re-export with pub use

## Next Steps
Proceed to Lab 10.3: Working with Crates
