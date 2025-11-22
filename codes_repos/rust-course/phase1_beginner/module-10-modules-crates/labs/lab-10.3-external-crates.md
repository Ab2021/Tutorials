# Lab 10.3: Working with External Crates

## Objective
Learn to use external crates from crates.io.

## Theory
Crates are Rust's packages. You can use thousands of crates from crates.io.

## Exercises

### Exercise 1: Adding Dependencies
In `Cargo.toml`:
```toml
[dependencies]
rand = "0.8"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

### Exercise 2: Using rand
```rust
use rand::Rng;

fn main() {
    let mut rng = rand::thread_rng();
    
    let n: u32 = rng.gen();
    println!("Random number: {}", n);
    
    let n: u32 = rng.gen_range(1..=100);
    println!("Random 1-100: {}", n);
}
```

### Exercise 3: Using serde for JSON
```rust
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
struct Person {
    name: String,
    age: u32,
    email: String,
}

fn main() {
    let person = Person {
        name: "Alice".to_string(),
        age: 30,
        email: "alice@example.com".to_string(),
    };
    
    // Serialize to JSON
    let json = serde_json::to_string(&person).unwrap();
    println!("{}", json);
    
    // Deserialize from JSON
    let person: Person = serde_json::from_str(&json).unwrap();
    println!("{:?}", person);
}
```

### Exercise 4: Using chrono for Dates
```toml
[dependencies]
chrono = "0.4"
```

```rust
use chrono::{DateTime, Utc, Local};

fn main() {
    let now: DateTime<Utc> = Utc::now();
    println!("UTC now: {}", now);
    
    let local: DateTime<Local> = Local::now();
    println!("Local now: {}", local);
}
```

### Exercise 5: Multiple Crates Project
```toml
[dependencies]
rand = "0.8"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
chrono = "0.4"
reqwest = { version = "0.11", features = ["blocking"] }
```

## Success Criteria
✅ Add crates to Cargo.toml  
✅ Use external crates  
✅ Understand feature flags  
✅ Build projects with dependencies

## Next Steps
Proceed to Lab 10.4: Creating Your Own Crate
