# Module 05: Borrowing and References

## 🎯 Learning Objectives

- Understand references and borrowing
- Master mutable and immutable references
- Learn borrowing rules
- Work with slices effectively
- Avoid common borrowing pitfalls

---

## 📖 Theoretical Concepts

### 5.1 References

A reference allows you to refer to a value without taking ownership.

```rust
fn main() {
    let s1 = String::from("hello");
    let len = calculate_length(&s1);  // Borrow s1
    
    println!("The length of '{}' is {}.", s1, len);  // s1 still valid
}

fn calculate_length(s: &String) -> usize {
    s.len()
}  // s goes out of scope, but doesn't drop the String
```

**Key Points:**
- `&` creates a reference
- References don't own the data
- Original owner retains ownership

---

### 5.2 Mutable References

```rust
fn main() {
    let mut s = String::from("hello");
    change(&mut s);
    println!("{}", s);  // Prints "hello, world"
}

fn change(s: &mut String) {
    s.push_str(", world");
}
```

**Restriction:** Only ONE mutable reference to a particular piece of data in a scope.

```rust
let mut s = String::from("hello");

let r1 = &mut s;
let r2 = &mut s;  // ❌ ERROR: cannot borrow `s` as mutable more than once
```

---

### 5.3 The Borrowing Rules

1. **At any given time**, you can have EITHER:
   - One mutable reference, OR
   - Any number of immutable references

2. **References must always be valid** (no dangling references)

```rust
// ✅ Valid: Multiple immutable references
let s = String::from("hello");
let r1 = &s;
let r2 = &s;
let r3 = &s;

// ✅ Valid: Mutable reference after immutable ones go out of scope
let mut s = String::from("hello");
{
    let r1 = &s;
    let r2 = &s;
    println!("{} and {}", r1, r2);
}  // r1 and r2 go out of scope
let r3 = &mut s;  // ✅ OK

// ❌ Invalid: Mixing mutable and immutable
let mut s = String::from("hello");
let r1 = &s;
let r2 = &mut s;  // ❌ ERROR
println!("{}", r1);
```

---

### 5.4 Dangling References

Rust prevents dangling references at compile time:

```rust
fn dangle() -> &String {  // ❌ ERROR
    let s = String::from("hello");
    &s  // s will be dropped, reference would dangle
}  // s goes out of scope and is dropped

// ✅ Solution: Return ownership
fn no_dangle() -> String {
    let s = String::from("hello");
    s  // Ownership is moved out
}
```

---

### 5.5 Slices

Slices let you reference a contiguous sequence without taking ownership.

#### String Slices

```rust
let s = String::from("hello world");

let hello = &s[0..5];   // "hello"
let world = &s[6..11];  // "world"

// Shortcuts
let slice = &s[0..2];   // Same as &s[..2]
let slice = &s[3..];    // From index 3 to end
let slice = &s[..];     // Entire string
```

#### Array Slices

```rust
let a = [1, 2, 3, 4, 5];
let slice = &a[1..3];  // [2, 3]
```

---

### 5.6 String Slices as Function Parameters

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
    let my_string = String::from("hello world");
    let word = first_word(&my_string[..]);  // Works with String slice
    
    let my_string_literal = "hello world";
    let word = first_word(my_string_literal);  // Works with string literal
}
```

---

## 🔑 Key Takeaways

1. **References** borrow values without taking ownership
2. **&T** is immutable reference, **&mut T** is mutable reference
3. **One mutable** OR **many immutable** references at a time
4. **References must always be valid** (no dangling)
5. **Slices** reference contiguous sequences
6. **&str** is more flexible than **String** for parameters

---

## ⏭️ Next Steps

Complete the 10 labs in this module:
1. Lab 5.1: Immutable references
2. Lab 5.2: Mutable references
3. Lab 5.3: Reference rules
4. Lab 5.4: Dangling references
5. Lab 5.5: Multiple references
6. Lab 5.6: Reference patterns
7. Lab 5.7: Slice types
8. Lab 5.8: String slices
9. Lab 5.9: Array slices
10. Lab 5.10: References project

Then proceed to Module 06: Structs
