# Lab 3.3: String Slices and Manipulation

## Objective
Master string slices, understand the difference between String and &str, and practice string manipulation techniques.

## Setup
```bash
cargo new string_slices
cd string_slices
```

## Part 1: Understanding String Slices

### Exercise 1: Basic Slicing
```rust
fn main() {
    let s = String::from("Hello, Rust!");
    
    // TODO: Create slices for:
    // - First word (0..5)
    // - Second word (7..11)
    // - Entire string
    
    let first_word = &s[0..5];
    println!("First word: {}", first_word);
    
    // Try other slices
}
```

### Exercise 2: First Word Function
Implement a function that returns the first word of a string:

```rust
fn first_word(s: &str) -> &str {
    let bytes = s.as_bytes();
    
    for (i, &item) in bytes.iter().enumerate() {
        if item == b' ' {
            return &s[0..i];
        }
    }
    
    &s[..]
}

fn main() {
    let my_string = String::from("Hello world");
    let word = first_word(&my_string[..]);
    println!("First word: {}", word);
    
    let my_string_literal = "Hello world";
    let word = first_word(my_string_literal);
    println!("First word: {}", word);
}
```

### Exercise 3: Last Word Function
```rust
fn last_word(s: &str) -> &str {
    // TODO: Implement - return the last word
    // Hint: Find the last space and slice from there
}

fn main() {
    let text = "The quick brown fox";
    println!("Last word: {}", last_word(text));
}
```

## Part 2: String vs &str

### Exercise 4: Type Conversions
```rust
fn main() {
    // String to &str
    let s = String::from("hello");
    let slice: &str = &s;
    let slice2: &str = &s[..];
    
    // &str to String
    let s1: String = "hello".to_string();
    let s2: String = String::from("hello");
    
    // TODO: Create a function that accepts both String and &str
}

fn print_string(s: &str) {
    println!("{}", s);
}

// Test with both types
fn test_function() {
    let owned = String::from("owned string");
    let borrowed = "borrowed string";
    
    print_string(&owned);
    print_string(borrowed);
}
```

## Part 3: String Manipulation

### Exercise 5: Extract Substring
```rust
fn get_substring(s: &str, start: usize, len: usize) -> &str {
    // TODO: Return substring starting at 'start' with length 'len'
    // Handle edge cases (out of bounds)
}

fn main() {
    let text = "Rust programming";
    println!("{}", get_substring(text, 0, 4));   // "Rust"
    println!("{}", get_substring(text, 5, 11));  // "programming"
}
```

### Exercise 6: Count Occurrences
```rust
fn count_char(s: &str, ch: char) -> usize {
    // TODO: Count how many times 'ch' appears in 's'
}

fn main() {
    let text = "hello world";
    println!("'l' appears {} times", count_char(text, 'l'));
    println!("'o' appears {} times", count_char(text, 'o'));
}
```

### Exercise 7: Reverse Words
```rust
fn reverse_words(s: &str) -> String {
    // TODO: Reverse the order of words
    // "hello world" -> "world hello"
}

fn main() {
    let text = "The quick brown fox";
    println!("Original: {}", text);
    println!("Reversed: {}", reverse_words(text));
}
```

## Part 4: Advanced Exercises

### Exercise 8: Palindrome Checker
```rust
fn is_palindrome(s: &str) -> bool {
    // TODO: Check if string is a palindrome
    // Ignore spaces and case
    // "A man a plan a canal Panama" -> true
}

fn main() {
    let tests = vec![
        "racecar",
        "hello",
        "A man a plan a canal Panama",
        "Was it a car or a cat I saw",
    ];
    
    for test in tests {
        println!("'{}' is palindrome: {}", test, is_palindrome(test));
    }
}
```

### Exercise 9: Extract Email Username
```rust
fn extract_username(email: &str) -> Option<&str> {
    // TODO: Extract username from email
    // "user@example.com" -> Some("user")
    // "invalid" -> None
}

fn main() {
    let emails = vec![
        "alice@example.com",
        "bob@test.org",
        "invalid-email",
    ];
    
    for email in emails {
        match extract_username(email) {
            Some(username) => println!("Username: {}", username),
            None => println!("Invalid email: {}", email),
        }
    }
}
```

### Exercise 10: Title Case Converter
```rust
fn to_title_case(s: &str) -> String {
    // TODO: Convert to title case
    // "hello world" -> "Hello World"
    // Each word's first letter should be uppercase
}

fn main() {
    let text = "the rust programming language";
    println!("Original: {}", text);
    println!("Title case: {}", to_title_case(text));
}
```

## Solutions

<details>
<summary>Exercise 3 Solution</summary>

```rust
fn last_word(s: &str) -> &str {
    let bytes = s.as_bytes();
    
    for i in (0..bytes.len()).rev() {
        if bytes[i] == b' ' {
            return &s[i + 1..];
        }
    }
    
    s
}
```

</details>

<details>
<summary>Exercise 5 Solution</summary>

```rust
fn get_substring(s: &str, start: usize, len: usize) -> &str {
    let end = (start + len).min(s.len());
    &s[start..end]
}
```

</details>

<details>
<summary>Exercise 6 Solution</summary>

```rust
fn count_char(s: &str, ch: char) -> usize {
    s.chars().filter(|&c| c == ch).count()
}
```

</details>

<details>
<summary>Exercise 7 Solution</summary>

```rust
fn reverse_words(s: &str) -> String {
    s.split_whitespace()
        .rev()
        .collect::<Vec<&str>>()
        .join(" ")
}
```

</details>

<details>
<summary>Exercise 8 Solution</summary>

```rust
fn is_palindrome(s: &str) -> bool {
    let cleaned: String = s.chars()
        .filter(|c| !c.is_whitespace())
        .map(|c| c.to_lowercase().next().unwrap())
        .collect();
    
    let reversed: String = cleaned.chars().rev().collect();
    cleaned == reversed
}
```

</details>

<details>
<summary>Exercise 9 Solution</summary>

```rust
fn extract_username(email: &str) -> Option<&str> {
    email.find('@').map(|pos| &email[..pos])
}
```

</details>

<details>
<summary>Exercise 10 Solution</summary>

```rust
fn to_title_case(s: &str) -> String {
    s.split_whitespace()
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => {
                    first.to_uppercase().collect::<String>() + chars.as_str()
                }
            }
        })
        .collect::<Vec<String>>()
        .join(" ")
}
```

</details>

## Success Criteria
✅ Understand difference between String and &str  
✅ Can create and use string slices  
✅ Implement string manipulation functions  
✅ Handle edge cases properly  
✅ Use iterators for string processing

## Key Learnings
- String slices are references to parts of strings
- &str is more flexible than String for function parameters
- String manipulation often uses iterators
- Always consider UTF-8 when working with strings
- Use .chars() for character iteration, not indexing

## Next Lab
Proceed to Lab 3.4: Word Counter Project!
