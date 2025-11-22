# Lab 7.1: Builder Pattern

## Objective
Implement the Builder pattern in Rust.

## Exercises

### Exercise 1: Basic Builder
```rust
pub struct User {
    name: String,
    email: String,
    age: Option<u32>,
}

pub struct UserBuilder {
    name: Option<String>,
    email: Option<String>,
    age: Option<u32>,
}

impl UserBuilder {
    pub fn new() -> Self {
        UserBuilder {
            name: None,
            email: None,
            age: None,
        }
    }
    
    pub fn name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }
    
    pub fn email(mut self, email: String) -> Self {
        self.email = Some(email);
        self
    }
    
    pub fn age(mut self, age: u32) -> Self {
        self.age = Some(age);
        self
    }
    
    pub fn build(self) -> Result<User, String> {
        Ok(User {
            name: self.name.ok_or("name required")?,
            email: self.email.ok_or("email required")?,
            age: self.age,
        })
    }
}

// Usage
let user = UserBuilder::new()
    .name("Alice".to_string())
    .email("alice@example.com".to_string())
    .age(30)
    .build()?;
```

## Success Criteria
✅ Implement builder pattern  
✅ Handle required/optional fields  
✅ Provide fluent API

## Next Steps
Lab 7.2: Strategy Pattern
