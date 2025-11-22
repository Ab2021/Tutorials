# Lab 9.2: JSON Serialization with Serde

## Objective
Use Serde for JSON serialization and deserialization.

## Exercises

### Exercise 1: Basic Serialization
```rust
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
struct Person {
    name: String,
    age: u32,
}

fn main() {
    let person = Person {
        name: "Alice".to_string(),
        age: 30,
    };
    
    let json = serde_json::to_string(&person).unwrap();
    println!("{}", json);
    
    let person: Person = serde_json::from_str(&json).unwrap();
    println!("{:?}", person);
}
```

### Exercise 2: Pretty Printing
```rust
let json = serde_json::to_string_pretty(&person).unwrap();
```

### Exercise 3: File I/O with JSON
```rust
use std::fs::File;

let file = File::create("person.json")?;
serde_json::to_writer_pretty(file, &person)?;

let file = File::open("person.json")?;
let person: Person = serde_json::from_reader(file)?;
```

## Success Criteria
✅ Serialize to JSON  
✅ Deserialize from JSON  
✅ Work with JSON files

## Next Steps
Lab 9.3: TOML Configuration
