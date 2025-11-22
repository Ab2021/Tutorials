# Module 09: File I/O and Serialization

## 🎯 Learning Objectives

- Master file operations
- Use Serde for serialization
- Work with JSON, TOML, YAML
- Handle binary formats
- Build data persistence layers

---

## 📖 Core Concepts

### File Operations

```rust
use std::fs;
use std::io::{self, Read, Write};

// Read entire file
let contents = fs::read_to_string("file.txt")?;

// Write to file
fs::write("output.txt", "Hello, world!")?;

// Buffered reading
use std::io::BufReader;
let file = fs::File::open("file.txt")?;
let reader = BufReader::new(file);
```

### Serde Framework

```rust
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
struct Person {
    name: String,
    age: u32,
    email: String,
}
```

### JSON Serialization

```rust
use serde_json;

let person = Person {
    name: "Alice".to_string(),
    age: 30,
    email: "alice@example.com".to_string(),
};

// Serialize
let json = serde_json::to_string(&person)?;

// Deserialize
let person: Person = serde_json::from_str(&json)?;
```

### TOML Configuration

```rust
use serde::Deserialize;

#[derive(Deserialize)]
struct Config {
    database: DatabaseConfig,
    server: ServerConfig,
}

let config: Config = toml::from_str(&contents)?;
```

### Binary Formats

```rust
use bincode;

// Serialize to bytes
let bytes = bincode::serialize(&data)?;

// Deserialize from bytes
let data: MyStruct = bincode::deserialize(&bytes)?;
```

---

## 🔑 Key Takeaways

1. **std::fs** for file operations
2. **Serde** for serialization/deserialization
3. **JSON** for APIs and config
4. **TOML** for configuration files
5. **Binary** for performance

Complete 10 labs, then proceed to Module 10: CLI Applications
