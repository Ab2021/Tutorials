# Lab 6.1: Building a Phonebook with HashMap

## Objective
Practice using HashMap for key-value storage and implementing CRUD operations.

## Setup
```bash
cargo new phonebook
cd phonebook
```

## Requirements

Create a phonebook application that:
- Stores names and phone numbers
- Adds new contacts
- Looks up contacts by name
- Updates existing contacts
- Deletes contacts
- Lists all contacts

## Starter Code

```rust
use std::collections::HashMap;

struct Phonebook {
    contacts: HashMap<String, String>,
}

impl Phonebook {
    fn new() -> Phonebook {
        Phonebook {
            contacts: HashMap::new(),
        }
    }
    
    fn add(&mut self, name: String, number: String) {
        // TODO: Add contact
    }
    
    fn get(&self, name: &str) -> Option<&String> {
        // TODO: Get contact
    }
    
    fn update(&mut self, name: &str, number: String) -> bool {
        // TODO: Update contact, return true if exists
    }
    
    fn delete(&mut self, name: &str) -> bool {
        // TODO: Delete contact, return true if existed
    }
    
    fn list(&self) {
        // TODO: List all contacts
    }
}

fn main() {
    let mut phonebook = Phonebook::new();
    
    phonebook.add(String::from("Alice"), String::from("555-1234"));
    phonebook.add(String::from("Bob"), String::from("555-5678"));
    
    phonebook.list();
}
```

## Solutions

<details>
<summary>Click to reveal</summary>

```rust
use std::collections::HashMap;

struct Phonebook {
    contacts: HashMap<String, String>,
}

impl Phonebook {
    fn new() -> Phonebook {
        Phonebook {
            contacts: HashMap::new(),
        }
    }
    
    fn add(&mut self, name: String, number: String) {
        self.contacts.insert(name, number);
    }
    
    fn get(&self, name: &str) -> Option<&String> {
        self.contacts.get(name)
    }
    
    fn update(&mut self, name: &str, number: String) -> bool {
        if self.contacts.contains_key(name) {
            self.contacts.insert(name.to_string(), number);
            true
        } else {
            false
        }
    }
    
    fn delete(&mut self, name: &str) -> bool {
        self.contacts.remove(name).is_some()
    }
    
    fn list(&self) {
        println!("=== Phonebook ===");
        for (name, number) in &self.contacts {
            println!("{}: {}", name, number);
        }
    }
}

fn main() {
    let mut phonebook = Phonebook::new();
    
    phonebook.add(String::from("Alice"), String::from("555-1234"));
    phonebook.add(String::from("Bob"), String::from("555-5678"));
    phonebook.add(String::from("Charlie"), String::from("555-9012"));
    
    phonebook.list();
    
    if let Some(number) = phonebook.get("Alice") {
        println!("\nAlice's number: {}", number);
    }
    
    phonebook.update("Bob", String::from("555-0000"));
    phonebook.delete("Charlie");
    
    println!("\nAfter updates:");
    phonebook.list();
}
```

</details>

## Challenge
Add a method to search for contacts by partial name match!
