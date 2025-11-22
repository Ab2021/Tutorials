# Module 3: Ownership and Borrowing

## 🎯 Learning Objectives

By the end of this module, you will:
- Understand Rust's ownership system
- Know the three ownership rules
- Understand move semantics and when data is moved
- Use references and borrowing correctly
- Distinguish between mutable and immutable references
- Work with string slices and array slices
- Understand the relationship between stack and heap

---

## 📖 Theoretical Concepts

### 3.1 What is Ownership?

**Ownership is Rust's most unique feature** and enables memory safety without a garbage collector.

#### The Three Rules of Ownership

1. **Each value in Rust has an owner**
2. **There can only be one owner at a time**
3. **When the owner goes out of scope, the value is dropped**

---

### 3.2 Stack vs Heap

Understanding memory is crucial to understanding ownership.

#### The Stack

- **LIFO** (Last In, First Out)
- **Fast** access
- **Fixed size** data
- Automatically managed

**Stack Data:**
- All scalar types (integers, floats, booleans, chars)
- Fixed-size arrays
- Tuples of stack data

```rust
let x = 5;           // Stored on stack
let y = true;        // Stored on stack
let z = 'a';         // Stored on stack
let arr = [1, 2, 3]; // Stored on stack
```

#### The Heap

- **Slower** than stack
- **Dynamic size** data
- Must be explicitly requested

**Heap Data:**
- String (growable)
- Vec (growable array)
- Box (heap allocation)

```rust
let s = String::from("hello");  // Data on heap
let v = vec![1, 2, 3];          // Data on heap
```

---

### 3.3 Ownership and Move Semantics

#### Simple Values (Copy Types)

```rust
let x = 5;
let y = x;  // x is copied to y

println!("x = {}, y = {}", x, y);  // ✅ Both work
```

**Why?** Integers implement the `Copy` trait - they're copied, not moved.

#### Heap-Allocated Values (Move Types)

```rust
let s1 = String::from("hello");
let s2 = s1;  // s1 is MOVED to s2

println!("{}", s1);  // ❌ ERROR: value borrowed after move
println!("{}", s2);  // ✅ OK
```

**What happened?**
1. `s1` owns the String data on the heap
2. `s2 = s1` transfers ownership to `s2`
3. `s1` is no longer valid
4. When `s2` goes out of scope, the String is dropped

**Visualization:**
```
Before move:
s1 -> [ptr, len, cap] -> "hello" (heap)

After s2 = s1:
s1 -> [INVALID]
s2 -> [ptr, len, cap] -> "hello" (heap)
```

---

### 3.4 Clone and Copy

#### Deep Copy with clone()

```rust
let s1 = String::from("hello");
let s2 = s1.clone();  // Deep copy

println!("s1 = {}, s2 = {}", s1, s2);  // ✅ Both work
```

**Note:** `clone()` can be expensive - it copies heap data.

#### Copy Trait

Types that implement `Copy` are copied instead of moved:

**Copy Types:**
- All integers: `i32`, `u64`, etc.
- Booleans: `bool`
- Floating point: `f32`, `f64`
- Characters: `char`
- Tuples of Copy types: `(i32, i32)`

**Cannot be Copy:**
- Types that own heap data
- Types with `Drop` implementation

---

### 3.5 Ownership and Functions

#### Passing to Functions

```rust
fn main() {
    let s = String::from("hello");
    
    takes_ownership(s);  // s is moved
    
    // println!("{}", s);  // ❌ ERROR: s is no longer valid
    
    let x = 5;
    makes_copy(x);  // x is copied
    
    println!("{}", x);  // ✅ OK: x is still valid
}

fn takes_ownership(some_string: String) {
    println!("{}", some_string);
}  // some_string is dropped here

fn makes_copy(some_integer: i32) {
    println!("{}", some_integer);
}
```

#### Returning Values

```rust
fn main() {
    let s1 = gives_ownership();  // Return value moves to s1
    
    let s2 = String::from("hello");
    let s3 = takes_and_gives_back(s2);  // s2 moved, return value moves to s3
    
    // println!("{}", s2);  // ❌ ERROR
    println!("{}", s3);     // ✅ OK
}

fn gives_ownership() -> String {
    let some_string = String::from("yours");
    some_string  // Returned and ownership moves
}

fn takes_and_gives_back(a_string: String) -> String {
    a_string  // Returned and ownership moves
}
```

---

### 3.6 References and Borrowing

**Problem:** Passing ownership back and forth is tedious.

**Solution:** Use references!

#### Immutable References

```rust
fn main() {
    let s1 = String::from("hello");
    
    let len = calculate_length(&s1);  // Borrow s1
    
    println!("The length of '{}' is {}.", s1, len);  // ✅ s1 still valid
}

fn calculate_length(s: &String) -> usize {
    s.len()
}  // s goes out of scope, but doesn't drop the data (doesn't own it)
```

**Key Points:**
- `&s1` creates a reference to `s1`
- References allow you to refer to a value without taking ownership
- This is called **borrowing**

#### The Borrowing Rules

1. **At any given time, you can have EITHER:**
   - One mutable reference, OR
   - Any number of immutable references

2. **References must always be valid** (no dangling references)

#### Multiple Immutable References

```rust
let s = String::from("hello");

let r1 = &s;  // ✅ OK
let r2 = &s;  // ✅ OK
let r3 = &s;  // ✅ OK

println!("{}, {}, {}", r1, r2, r3);
```

#### Mutable References

```rust
fn main() {
    let mut s = String::from("hello");
    
    change(&mut s);
    
    println!("{}", s);  // "hello, world"
}

fn change(some_string: &mut String) {
    some_string.push_str(", world");
}
```

#### Restriction: Only One Mutable Reference

```rust
let mut s = String::from("hello");

let r1 = &mut s;
let r2 = &mut s;  // ❌ ERROR: cannot borrow as mutable more than once

println!("{}, {}", r1, r2);
```

**Why?** Prevents data races at compile time!

#### Cannot Mix Mutable and Immutable

```rust
let mut s = String::from("hello");

let r1 = &s;      // ✅ OK
let r2 = &s;      // ✅ OK
let r3 = &mut s;  // ❌ ERROR: cannot borrow as mutable

println!("{}, {}, {}", r1, r2, r3);
```

#### Reference Scope

```rust
let mut s = String::from("hello");

let r1 = &s;
let r2 = &s;
println!("{} and {}", r1, r2);
// r1 and r2 are no longer used after this point

let r3 = &mut s;  // ✅ OK: no immutable references active
println!("{}", r3);
```

---

### 3.7 Dangling References

Rust prevents dangling references at compile time:

```rust
fn dangle() -> &String {  // ❌ ERROR
    let s = String::from("hello");
    &s  // Returns reference to s
}  // s is dropped, reference would be invalid
```

**Solution:** Return the value itself (transfer ownership):

```rust
fn no_dangle() -> String {  // ✅ OK
    let s = String::from("hello");
    s  // Ownership is moved out
}
```

---

### 3.8 Slices

Slices let you reference a contiguous sequence of elements without ownership.

#### String Slices

```rust
let s = String::from("hello world");

let hello = &s[0..5];   // "hello"
let world = &s[6..11];  // "world"

// Shorthand
let hello = &s[..5];    // Same as [0..5]
let world = &s[6..];    // Same as [6..len]
let whole = &s[..];     // Entire string
```

**Type:** `&str` (string slice)

#### String Literals are Slices

```rust
let s: &str = "Hello, world!";  // &str is a slice
```

#### Slices as Function Parameters

```rust
fn first_word(s: &str) -> &str {
    let bytes = s.as_bytes();
    
    for (i, &item) in bytes.iter().enumerate() {
        if item == b' ' {
            return &s[..i];
        }
    }
    
    &s[..]
}

fn main() {
    let my_string = String::from("hello world");
    let word = first_word(&my_string[..]);
    
    let my_string_literal = "hello world";
    let word = first_word(my_string_literal);
}
```

#### Array Slices

```rust
let a = [1, 2, 3, 4, 5];

let slice = &a[1..3];  // Type: &[i32]

assert_eq!(slice, &[2, 3]);
```

---

## 🔑 Key Takeaways

1. **Ownership rules:** one owner, dropped when out of scope
2. **Move semantics:** heap data is moved, not copied (unless cloned)
3. **Copy trait:** stack data is copied automatically
4. **Borrowing:** use references to avoid moving ownership
5. **Borrowing rules:** one mutable OR many immutable references
6. **No dangling references:** compiler ensures reference validity
7. **Slices:** reference parts of collections without ownership

---

## 📚 Additional Resources

- [The Rust Book - Chapter 4](https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html)
- [Rust by Example - Ownership](https://doc.rust-lang.org/rust-by-example/scope/move.html)
- [Visualizing Memory Layout](https://www.youtube.com/watch?v=VFIOSWy93H0)

---

## ⏭️ Next Steps

Proceed to the labs directory:
1. Lab 3.1: Ownership transfer exercises
2. Lab 3.2: Reference and borrowing practice
3. Lab 3.3: String manipulation with slices
4. Lab 3.4: Building a word counter

After completing the labs, move on to **Module 4: Structs and Enums**.
