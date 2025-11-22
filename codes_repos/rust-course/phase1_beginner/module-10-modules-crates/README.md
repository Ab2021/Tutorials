# Module 10: Modules and Crates

## 🎯 Learning Objectives

- Organize code with modules
- Control visibility with pub
- Use the use keyword effectively
- Create and publish crates
- Work with workspaces

---

## 📖 Theoretical Concepts

### 10.1 Module System Basics

```rust
mod front_of_house {
    pub mod hosting {
        pub fn add_to_waitlist() {}
        fn seat_at_table() {}
    }
    
    mod serving {
        fn take_order() {}
        fn serve_order() {}
        fn take_payment() {}
    }
}

pub fn eat_at_restaurant() {
    // Absolute path
    crate::front_of_house::hosting::add_to_waitlist();
    
    // Relative path
    front_of_house::hosting::add_to_waitlist();
}
```

---

### 10.2 Paths

**Absolute path:** Starts from crate root using `crate`  
**Relative path:** Starts from current module

```rust
mod back_of_house {
    pub struct Breakfast {
        pub toast: String,
        seasonal_fruit: String,  // private
    }
    
    impl Breakfast {
        pub fn summer(toast: &str) -> Breakfast {
            Breakfast {
                toast: String::from(toast),
                seasonal_fruit: String::from("peaches"),
            }
        }
    }
}

pub fn eat_at_restaurant() {
    let mut meal = back_of_house::Breakfast::summer("Rye");
    meal.toast = String::from("Wheat");
    // meal.seasonal_fruit = String::from("blueberries");  // ❌ private
}
```

---

### 10.3 The use Keyword

```rust
mod front_of_house {
    pub mod hosting {
        pub fn add_to_waitlist() {}
    }
}

use crate::front_of_house::hosting;

pub fn eat_at_restaurant() {
    hosting::add_to_waitlist();
}
```

#### Idiomatic use

```rust
// ✅ For functions: bring parent module
use std::collections::HashMap;

// ✅ For structs/enums: bring the type
use std::fmt::Result;
use std::io::Result as IoResult;  // Rename with 'as'
```

#### Re-exporting with pub use

```rust
mod front_of_house {
    pub mod hosting {
        pub fn add_to_waitlist() {}
    }
}

pub use crate::front_of_house::hosting;  // Re-export
```

---

### 10.4 Separating Modules into Files

**src/lib.rs:**
```rust
mod front_of_house;

pub use crate::front_of_house::hosting;

pub fn eat_at_restaurant() {
    hosting::add_to_waitlist();
}
```

**src/front_of_house.rs:**
```rust
pub mod hosting {
    pub fn add_to_waitlist() {}
}
```

Or with subdirectory:

**src/front_of_house/mod.rs:**
```rust
pub mod hosting;
```

**src/front_of_house/hosting.rs:**
```rust
pub fn add_to_waitlist() {}
```

---

### 10.5 Creating a Library Crate

```bash
cargo new my_library --lib
```

**src/lib.rs:**
```rust
pub fn public_function() {
    println!("called my_library's public_function()");
}

fn private_function() {
    println!("called my_library's private_function()");
}

pub mod utilities {
    pub fn helper() {
        println!("called utilities::helper()");
    }
}
```

---

### 10.6 Using External Crates

**Cargo.toml:**
```toml
[dependencies]
rand = "0.8.5"
serde = { version = "1.0", features = ["derive"] }
```

**src/main.rs:**
```rust
use rand::Rng;

fn main() {
    let secret_number = rand::thread_rng().gen_range(1..=100);
    println!("Secret number: {}", secret_number);
}
```

---

### 10.7 Workspaces

**Cargo.toml (workspace root):**
```toml
[workspace]
members = [
    "adder",
    "add_one",
]
```

**adder/Cargo.toml:**
```toml
[dependencies]
add_one = { path = "../add_one" }
```

---

### 10.8 Publishing to crates.io

1. **Create account** at crates.io
2. **Get API token**
3. **Login:**
   ```bash
   cargo login <your-token>
   ```

4. **Add metadata to Cargo.toml:**
   ```toml
   [package]
   name = "my_crate"
   version = "0.1.0"
   authors = ["Your Name <you@example.com>"]
   edition = "2021"
   description = "A short description"
   license = "MIT"
   ```

5. **Publish:**
   ```bash
   cargo publish
   ```

---

## 🔑 Key Takeaways

1. **Modules** organize code into namespaces
2. **pub** makes items public
3. **use** brings paths into scope
4. **Files** can represent modules
5. **Crates** are compilation units
6. **Workspaces** manage multiple crates
7. **crates.io** for sharing code

---

## ⏭️ Next Steps

Complete the 10 labs in this module:
1. Lab 10.1: Module basics
2. Lab 10.2: Visibility
3. Lab 10.3: use keyword
4. Lab 10.4: File modules
5. Lab 10.5: Nested modules
6. Lab 10.6: Creating crates
7. Lab 10.7: Dependencies
8. Lab 10.8: Workspaces
9. Lab 10.9: Publishing
10. Lab 10.10: Library project

**Congratulations on completing Phase 1!**  
Proceed to **Phase 2: Intermediate - Software Engineering**
