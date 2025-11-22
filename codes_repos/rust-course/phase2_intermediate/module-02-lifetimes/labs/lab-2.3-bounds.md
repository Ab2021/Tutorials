# Lab 2.3: Lifetime Bounds

## Objective
Use lifetime bounds with generics and traits.

## Exercises

### Exercise 1: Generic with Lifetime
```rust
struct Wrapper<'a, T> {
    value: &'a T,
}

impl<'a, T> Wrapper<'a, T> {
    fn new(value: &'a T) -> Self {
        Wrapper { value }
    }
}
```

### Exercise 2: Trait Bounds with Lifetimes
```rust
fn print_ref<'a, T>(t: &'a T)
where
    T: std::fmt::Display + 'a,
{
    println!("{}", t);
}
```

### Exercise 3: Static Lifetime
```rust
fn get_static_str() -> &'static str {
    "This string lives forever"
}

static GLOBAL: &str = "Global string";
```

## Success Criteria
✅ Use lifetimes with generics  
✅ Understand 'static lifetime  
✅ Apply lifetime bounds

## Next Steps
Lab 2.4: Advanced Lifetime Patterns
