# Lab 3.2: Trait Basics

## Objective
Define and implement traits in Rust.

## Exercises

### Exercise 1: Simple Trait
```rust
trait Summary {
    fn summarize(&self) -> String;
}

struct Article {
    title: String,
    content: String,
}

impl Summary for Article {
    fn summarize(&self) -> String {
        format!("{}: {}", self.title, self.content)
    }
}
```

### Exercise 2: Default Implementations
```rust
trait Summary {
    fn summarize(&self) -> String {
        String::from("(Read more...)")
    }
}
```

### Exercise 3: Traits as Parameters
```rust
fn notify(item: &impl Summary) {
    println!("Breaking news! {}", item.summarize());
}
```

## Success Criteria
✅ Define traits  
✅ Implement traits  
✅ Use traits as parameters

## Next Steps
Lab 3.3: Trait Bounds
