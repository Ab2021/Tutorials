# Lab 4.3: Mastering Option and Result

## Objective
Deep dive into Option<T> and Result<T, E> - Rust's powerful error handling types. Learn when and how to use each effectively.

## Setup
```bash
cargo new option_result_practice
cd option_result_practice
```

## Part 1: Option<T> Fundamentals

### Exercise 1: Basic Option Usage
```rust
fn find_user(id: u32) -> Option<String> {
    // Simulate database lookup
    match id {
        1 => Some(String::from("Alice")),
        2 => Some(String::from("Bob")),
        3 => Some(String::from("Charlie")),
        _ => None,
    }
}

fn main() {
    // TODO: Try different ways to handle Option
    
    // Method 1: match
    match find_user(1) {
        Some(name) => println!("Found: {}", name),
        None => println!("User not found"),
    }
    
    // Method 2: if let
    if let Some(name) = find_user(2) {
        println!("Found: {}", name);
    }
    
    // Method 3: unwrap_or
    let name = find_user(999).unwrap_or(String::from("Guest"));
    println!("Name: {}", name);
}
```

### Exercise 2: Option Methods
```rust
fn main() {
    let some_number = Some(5);
    let no_number: Option<i32> = None;
    
    // is_some() and is_none()
    println!("Has value: {}", some_number.is_some());
    println!("Is none: {}", no_number.is_none());
    
    // map()
    let doubled = some_number.map(|x| x * 2);
    println!("Doubled: {:?}", doubled);
    
    // and_then()
    let result = some_number.and_then(|x| {
        if x > 0 {
            Some(x * 2)
        } else {
            None
        }
    });
    
    // filter()
    let filtered = some_number.filter(|&x| x > 3);
    println!("Filtered: {:?}", filtered);
    
    // unwrap_or_else()
    let value = no_number.unwrap_or_else(|| {
        println!("Computing default value...");
        42
    });
}
```

### Exercise 3: Chaining Options
```rust
struct Person {
    name: String,
    age: Option<u32>,
    email: Option<String>,
}

impl Person {
    fn new(name: String) -> Self {
        Person {
            name,
            age: None,
            email: None,
        }
    }
    
    fn with_age(mut self, age: u32) -> Self {
        self.age = Some(age);
        self
    }
    
    fn with_email(mut self, email: String) -> Self {
        self.email = Some(email);
        self
    }
    
    fn display(&self) {
        println!("Name: {}", self.name);
        
        if let Some(age) = self.age {
            println!("Age: {}", age);
        }
        
        if let Some(email) = &self.email {
            println!("Email: {}", email);
        }
    }
}

fn main() {
    let person = Person::new(String::from("Alice"))
        .with_age(30)
        .with_email(String::from("alice@example.com"));
    
    person.display();
}
```

## Part 2: Result<T, E> Fundamentals

### Exercise 4: Basic Result Usage
```rust
fn divide(a: f64, b: f64) -> Result<f64, String> {
    if b == 0.0 {
        Err(String::from("Division by zero"))
    } else {
        Ok(a / b)
    }
}

fn main() {
    // Method 1: match
    match divide(10.0, 2.0) {
        Ok(result) => println!("Result: {}", result),
        Err(e) => println!("Error: {}", e),
    }
    
    // Method 2: unwrap_or
    let result = divide(10.0, 0.0).unwrap_or(0.0);
    println!("Result: {}", result);
    
    // Method 3: expect (for debugging)
    // let result = divide(10.0, 0.0).expect("Division failed");
}
```

### Exercise 5: Result Methods
```rust
fn parse_number(s: &str) -> Result<i32, std::num::ParseIntError> {
    s.parse::<i32>()
}

fn main() {
    let input = "42";
    
    // map()
    let doubled = parse_number(input).map(|x| x * 2);
    println!("Doubled: {:?}", doubled);
    
    // map_err()
    let result = parse_number("abc").map_err(|e| {
        format!("Failed to parse: {}", e)
    });
    println!("Result: {:?}", result);
    
    // and_then()
    let result = parse_number(input).and_then(|x| {
        if x > 0 {
            Ok(x * 2)
        } else {
            Err(std::num::ParseIntError::from(std::num::IntErrorKind::Empty))
        }
    });
    
    // or_else()
    let result = parse_number("abc").or_else(|_| Ok(0));
    println!("With fallback: {:?}", result);
}
```

### Exercise 6: The ? Operator
```rust
fn read_and_parse(s: &str) -> Result<i32, String> {
    let num = s.parse::<i32>()
        .map_err(|e| format!("Parse error: {}", e))?;
    
    if num < 0 {
        return Err(String::from("Number must be positive"));
    }
    
    Ok(num * 2)
}

fn main() {
    match read_and_parse("42") {
        Ok(n) => println!("Result: {}", n),
        Err(e) => println!("Error: {}", e),
    }
    
    match read_and_parse("-5") {
        Ok(n) => println!("Result: {}", n),
        Err(e) => println!("Error: {}", e),
    }
}
```

## Part 3: Practical Applications

### Exercise 7: Configuration Parser
```rust
use std::collections::HashMap;

struct Config {
    settings: HashMap<String, String>,
}

impl Config {
    fn new() -> Self {
        Config {
            settings: HashMap::new(),
        }
    }
    
    fn set(&mut self, key: String, value: String) {
        self.settings.insert(key, value);
    }
    
    fn get(&self, key: &str) -> Option<&String> {
        self.settings.get(key)
    }
    
    fn get_or_default(&self, key: &str, default: &str) -> String {
        self.get(key)
            .map(|s| s.clone())
            .unwrap_or_else(|| default.to_string())
    }
    
    fn get_int(&self, key: &str) -> Result<i32, String> {
        self.get(key)
            .ok_or_else(|| format!("Key '{}' not found", key))?
            .parse::<i32>()
            .map_err(|e| format!("Failed to parse '{}': {}", key, e))
    }
    
    fn get_bool(&self, key: &str) -> Result<bool, String> {
        // TODO: Implement - parse "true"/"false" strings
    }
}

fn main() {
    let mut config = Config::new();
    config.set(String::from("port"), String::from("8080"));
    config.set(String::from("debug"), String::from("true"));
    config.set(String::from("host"), String::from("localhost"));
    
    println!("Port: {}", config.get_int("port").unwrap());
    println!("Host: {}", config.get_or_default("host", "0.0.0.0"));
    println!("Timeout: {}", config.get_or_default("timeout", "30"));
}
```

### Exercise 8: Safe Array Access
```rust
fn safe_get<T: Clone>(arr: &[T], index: usize) -> Option<T> {
    // TODO: Return Some(element) if index is valid, None otherwise
}

fn safe_divide_elements(arr: &[f64], i: usize, j: usize) -> Result<f64, String> {
    // TODO: Get elements at i and j, divide them
    // Return appropriate errors for:
    // - Invalid indices
    // - Division by zero
}

fn main() {
    let numbers = vec![10.0, 20.0, 30.0, 40.0];
    
    match safe_get(&numbers, 2) {
        Some(n) => println!("Element: {}", n),
        None => println!("Index out of bounds"),
    }
    
    match safe_divide_elements(&numbers, 0, 1) {
        Ok(result) => println!("Division result: {}", result),
        Err(e) => println!("Error: {}", e),
    }
}
```

### Exercise 9: User Input Validator
```rust
struct User {
    username: String,
    email: String,
    age: u32,
}

fn validate_username(username: &str) -> Result<(), String> {
    if username.len() < 3 {
        return Err(String::from("Username must be at least 3 characters"));
    }
    if username.len() > 20 {
        return Err(String::from("Username must be at most 20 characters"));
    }
    if !username.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return Err(String::from("Username can only contain letters, numbers, and underscores"));
    }
    Ok(())
}

fn validate_email(email: &str) -> Result<(), String> {
    // TODO: Implement basic email validation
    // Must contain @ and at least one character before and after
}

fn validate_age(age: u32) -> Result<(), String> {
    // TODO: Implement age validation
    // Must be between 13 and 120
}

fn create_user(username: String, email: String, age: u32) -> Result<User, String> {
    validate_username(&username)?;
    validate_email(&email)?;
    validate_age(age)?;
    
    Ok(User { username, email, age })
}

fn main() {
    match create_user(
        String::from("alice_123"),
        String::from("alice@example.com"),
        25
    ) {
        Ok(user) => println!("User created: {}", user.username),
        Err(e) => println!("Validation error: {}", e),
    }
    
    // Test with invalid data
    match create_user(String::from("ab"), String::from("invalid"), 5) {
        Ok(user) => println!("User created: {}", user.username),
        Err(e) => println!("Validation error: {}", e),
    }
}
```

## Part 4: Advanced Patterns

### Exercise 10: Combining Option and Result
```rust
fn find_and_parse(data: &[(String, String)], key: &str) -> Result<i32, String> {
    data.iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v)
        .ok_or_else(|| format!("Key '{}' not found", key))?
        .parse::<i32>()
        .map_err(|e| format!("Parse error: {}", e))
}

fn main() {
    let data = vec![
        (String::from("age"), String::from("25")),
        (String::from("score"), String::from("100")),
    ];
    
    match find_and_parse(&data, "age") {
        Ok(n) => println!("Age: {}", n),
        Err(e) => println!("Error: {}", e),
    }
}
```

## Solutions

<details>
<summary>Exercise 7 - get_bool Solution</summary>

```rust
fn get_bool(&self, key: &str) -> Result<bool, String> {
    let value = self.get(key)
        .ok_or_else(|| format!("Key '{}' not found", key))?;
    
    match value.as_str() {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(format!("Invalid boolean value: {}", value)),
    }
}
```

</details>

<details>
<summary>Exercise 8 Solutions</summary>

```rust
fn safe_get<T: Clone>(arr: &[T], index: usize) -> Option<T> {
    arr.get(index).cloned()
}

fn safe_divide_elements(arr: &[f64], i: usize, j: usize) -> Result<f64, String> {
    let a = arr.get(i)
        .ok_or_else(|| format!("Index {} out of bounds", i))?;
    let b = arr.get(j)
        .ok_or_else(|| format!("Index {} out of bounds", j))?;
    
    if *b == 0.0 {
        return Err(String::from("Division by zero"));
    }
    
    Ok(a / b)
}
```

</details>

<details>
<summary>Exercise 9 Solutions</summary>

```rust
fn validate_email(email: &str) -> Result<(), String> {
    if !email.contains('@') {
        return Err(String::from("Email must contain @"));
    }
    
    let parts: Vec<&str> = email.split('@').collect();
    if parts.len() != 2 || parts[0].is_empty() || parts[1].is_empty() {
        return Err(String::from("Invalid email format"));
    }
    
    Ok(())
}

fn validate_age(age: u32) -> Result<(), String> {
    if age < 13 {
        return Err(String::from("Must be at least 13 years old"));
    }
    if age > 120 {
        return Err(String::from("Invalid age"));
    }
    Ok(())
}
```

</details>

## Success Criteria
✅ Understand when to use Option vs Result  
✅ Can chain Option/Result operations  
✅ Use ? operator effectively  
✅ Handle errors gracefully  
✅ Implement validation logic  
✅ Avoid unwrap() in production code

## Key Learnings
- Option<T> for values that might not exist
- Result<T, E> for operations that might fail
- ? operator propagates errors elegantly
- map(), and_then(), unwrap_or() for transformations
- Never use unwrap() without good reason
- Type system prevents forgetting to handle errors

## Next Steps
Move to Module 5 to learn more about error handling patterns and custom error types!
