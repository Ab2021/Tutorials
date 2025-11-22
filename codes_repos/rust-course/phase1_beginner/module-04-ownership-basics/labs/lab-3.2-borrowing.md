# Lab 3.2: References and Borrowing

## Objective
Master references and borrowing to work with data without taking ownership.

## Setup
```bash
cargo new borrowing_practice
cd borrowing_practice
```

## Part 1: Immutable References

### Exercise 1: Basic Borrowing
```rust
fn main() {
    let s = String::from("hello");
    
    let len = calculate_length(&s);
    
    println!("The length of '{}' is {}.", s, len);
}

fn calculate_length(s: &String) -> usize {
    s.len()
}
```

### Exercise 2: Multiple Immutable References
```rust
fn main() {
    let s = String::from("hello");
    
    let r1 = &s;
    let r2 = &s;
    let r3 = &s;
    
    println!("{}, {}, {}", r1, r2, r3);
}
```

## Part 2: Mutable References

### Exercise 3: Mutable Borrowing
```rust
fn main() {
    let mut s = String::from("hello");
    
    change(&mut s);
    
    println!("{}", s);
}

fn change(some_string: &mut String) {
    some_string.push_str(", world");
}
```

### Exercise 4: One Mutable Reference Rule
```rust
fn main() {
    let mut s = String::from("hello");
    
    let r1 = &mut s;
    // let r2 = &mut s;  // ERROR: cannot borrow as mutable more than once
    
    println!("{}", r1);
}
```

### Exercise 5: Reference Scopes
```rust
fn main() {
    let mut s = String::from("hello");
    
    {
        let r1 = &mut s;
        println!("{}", r1);
    }  // r1 goes out of scope
    
    let r2 = &mut s;  // OK: r1 is no longer in scope
    println!("{}", r2);
}
```

## Part 3: Practical Exercises

### Exercise 6: String Appender
Create a function that appends text to a string:

```rust
fn append_exclamation(s: &mut String) {
    // TODO: Add "!" to the string
}

fn main() {
    let mut text = String::from("Hello");
    append_exclamation(&mut text);
    println!("{}", text);  // Should print "Hello!"
}
```

### Exercise 7: String Analyzer
Create functions that analyze a string without taking ownership:

```rust
fn count_words(s: &String) -> usize {
    // TODO: Count words (split by spaces)
}

fn count_chars(s: &String) -> usize {
    // TODO: Count characters
}

fn is_empty(s: &String) -> bool {
    // TODO: Check if string is empty
}

fn main() {
    let text = String::from("Hello Rust World");
    
    println!("Words: {}", count_words(&text));
    println!("Chars: {}", count_chars(&text));
    println!("Empty: {}", is_empty(&text));
    
    // text is still valid here!
    println!("Original: {}", text);
}
```

### Exercise 8: Fix the Borrowing Errors
Fix this code:

```rust
fn main() {
    let mut s = String::from("hello");
    
    let r1 = &s;
    let r2 = &s;
    let r3 = &mut s;  // ERROR: cannot borrow as mutable
    
    println!("{}, {}, {}", r1, r2, r3);
}
```

## Part 4: Advanced Exercises

### Exercise 9: String Modifier
Create a function that modifies a string in place:

```rust
fn make_uppercase(s: &mut String) {
    // TODO: Convert string to uppercase
    // Hint: Use s.make_ascii_uppercase()
}

fn main() {
    let mut text = String::from("hello world");
    make_uppercase(&mut text);
    println!("{}", text);  // Should print "HELLO WORLD"
}
```

### Exercise 10: Reference Chain
```rust
fn main() {
    let s = String::from("hello");
    let r1 = &s;
    let r2 = &r1;
    let r3 = &r2;
    
    println!("{}", r3);  // Prints "hello"
}
```

**Question:** How does this work?

## Solutions

<details>
<summary>Exercise 6 Solution</summary>

```rust
fn append_exclamation(s: &mut String) {
    s.push('!');
}
```

</details>

<details>
<summary>Exercise 7 Solution</summary>

```rust
fn count_words(s: &String) -> usize {
    s.split_whitespace().count()
}

fn count_chars(s: &String) -> usize {
    s.len()
}

fn is_empty(s: &String) -> bool {
    s.is_empty()
}
```

</details>

<details>
<summary>Exercise 8 Solution</summary>

```rust
fn main() {
    let mut s = String::from("hello");
    
    let r1 = &s;
    let r2 = &s;
    println!("{}, {}", r1, r2);
    // r1 and r2 are no longer used
    
    let r3 = &mut s;  // OK now
    println!("{}", r3);
}
```

</details>

<details>
<summary>Exercise 9 Solution</summary>

```rust
fn make_uppercase(s: &mut String) {
    s.make_ascii_uppercase();
}
```

</details>

## Key Learnings
- Use `&` to borrow without taking ownership
- Multiple immutable references are allowed
- Only one mutable reference at a time
- Cannot mix mutable and immutable references
- References must always be valid

## Next Lab
Proceed to Lab 3.3 to work with slices!
