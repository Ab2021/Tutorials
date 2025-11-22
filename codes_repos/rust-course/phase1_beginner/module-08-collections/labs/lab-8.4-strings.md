# Lab 8.4: String Manipulation

## Objective
Master String and &str types in Rust.

## Theory
Rust has two string types: String (owned) and &str (borrowed slice).

## Exercises

### Exercise 1: Creating Strings
```rust
fn main() {
    // String literal
    let s1 = "hello";  // &str
    
    // String::from
    let s2 = String::from("hello");
    
    // to_string
    let s3 = "hello".to_string();
    
    // String::new
    let mut s4 = String::new();
    s4.push_str("hello");
}
```

### Exercise 2: Updating Strings
```rust
fn main() {
    let mut s = String::from("foo");
    
    s.push_str("bar");    // Append &str
    s.push('!');          // Append char
    
    let s2 = String::from(" baz");
    s = s + &s2;          // Concatenation
    
    // format! macro
    let s3 = format!("{}-{}-{}", s1, s2, s3);
}
```

### Exercise 3: Indexing and Slicing
```rust
fn main() {
    let s = String::from("hello");
    
    // Can't index directly
    // let h = s[0];  // ERROR
    
    // Use slicing
    let hello = &s[0..5];
    
    // Iterate over chars
    for c in s.chars() {
        println!("{}", c);
    }
    
    // Iterate over bytes
    for b in s.bytes() {
        println!("{}", b);
    }
}
```

### Exercise 4: String Methods
```rust
fn main() {
    let s = String::from("  Hello, World!  ");
    
    println!("Length: {}", s.len());
    println!("Is empty: {}", s.is_empty());
    println!("Contains 'World': {}", s.contains("World"));
    println!("Starts with 'Hello': {}", s.starts_with("Hello"));
    println!("Trimmed: '{}'", s.trim());
    println!("Uppercase: {}", s.to_uppercase());
    println!("Lowercase: {}", s.to_lowercase());
}
```

### Exercise 5: String Parsing
```rust
fn main() {
    let s = "42";
    let num: i32 = s.parse().expect("Not a number!");
    
    let s = "hello world";
    let words: Vec<&str> = s.split_whitespace().collect();
    
    let s = "a,b,c";
    let parts: Vec<&str> = s.split(',').collect();
}
```

## Success Criteria
✅ Understand String vs &str  
✅ Create and modify strings  
✅ Iterate over strings safely  
✅ Use string methods effectively

## Next Steps
Proceed to Lab 8.5: HashMap Operations
