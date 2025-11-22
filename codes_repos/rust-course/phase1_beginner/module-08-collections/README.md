# Module 6: Collections

## 🎯 Learning Objectives
- Work with vectors for dynamic arrays
- Manipulate strings effectively
- Use hash maps for key-value storage
- Understand iterators and closures
- Apply functional programming patterns

## 📖 Theoretical Concepts

### 6.1 Vectors

Dynamic, growable arrays:

```rust
// Creating vectors
let v: Vec<i32> = Vec::new();
let v = vec![1, 2, 3];

// Adding elements
let mut v = Vec::new();
v.push(5);
v.push(6);
v.push(7);

// Accessing elements
let third: &i32 = &v[2];
let third: Option<&i32> = v.get(2);

match v.get(2) {
    Some(third) => println!("The third element is {}", third),
    None => println!("There is no third element."),
}

// Iterating
for i in &v {
    println!("{}", i);
}

// Mutable iteration
for i in &mut v {
    *i += 50;
}
```

#### Storing Multiple Types
```rust
enum SpreadsheetCell {
    Int(i32),
    Float(f64),
    Text(String),
}

let row = vec![
    SpreadsheetCell::Int(3),
    SpreadsheetCell::Text(String::from("blue")),
    SpreadsheetCell::Float(10.12),
];
```

### 6.2 Strings

```rust
// Creating strings
let mut s = String::new();
let s = "initial contents".to_string();
let s = String::from("initial contents");

// Updating strings
let mut s = String::from("foo");
s.push_str("bar");
s.push('!');

// Concatenation
let s1 = String::from("Hello, ");
let s2 = String::from("world!");
let s3 = s1 + &s2;  // s1 is moved

// format! macro
let s1 = String::from("tic");
let s2 = String::from("tac");
let s3 = String::from("toe");
let s = format!("{}-{}-{}", s1, s2, s3);

// Indexing (not allowed!)
// let h = s[0];  // ERROR

// Slicing
let hello = "Здравствуйте";
let s = &hello[0..4];  // "Зд"

// Iterating
for c in "नमस्ते".chars() {
    println!("{}", c);
}

for b in "नमस्ते".bytes() {
    println!("{}", b);
}
```

### 6.3 Hash Maps

```rust
use std::collections::HashMap;

// Creating
let mut scores = HashMap::new();
scores.insert(String::from("Blue"), 10);
scores.insert(String::from("Yellow"), 50);

// From vectors
let teams = vec![String::from("Blue"), String::from("Yellow")];
let initial_scores = vec![10, 50];
let scores: HashMap<_, _> = teams.iter().zip(initial_scores.iter()).collect();

// Accessing
let team_name = String::from("Blue");
let score = scores.get(&team_name);

// Iterating
for (key, value) in &scores {
    println!("{}: {}", key, value);
}

// Updating
scores.insert(String::from("Blue"), 25);  // Overwrite

// Insert if not present
scores.entry(String::from("Yellow")).or_insert(50);

// Update based on old value
let text = "hello world wonderful world";
let mut map = HashMap::new();

for word in text.split_whitespace() {
    let count = map.entry(word).or_insert(0);
    *count += 1;
}
```

### 6.4 Closures

Anonymous functions that can capture their environment:

```rust
let expensive_closure = |num| {
    println!("calculating slowly...");
    thread::sleep(Duration::from_secs(2));
    num
};

// Type annotations (optional)
let add_one = |x: i32| -> i32 { x + 1 };

// Capturing environment
let x = 4;
let equal_to_x = |z| z == x;
let y = 4;
assert!(equal_to_x(y));
```

### 6.5 Iterators

```rust
let v1 = vec![1, 2, 3];
let v1_iter = v1.iter();

for val in v1_iter {
    println!("{}", val);
}

// Iterator adaptors
let v1: Vec<i32> = vec![1, 2, 3];
let v2: Vec<_> = v1.iter().map(|x| x + 1).collect();

// filter
let v: Vec<i32> = vec![1, 2, 3, 4, 5];
let evens: Vec<i32> = v.iter()
    .filter(|&&x| x % 2 == 0)
    .map(|&x| x)
    .collect();

// Common methods
let sum: i32 = v.iter().sum();
let product: i32 = v.iter().product();
let max = v.iter().max();
let min = v.iter().min();
```

## 🔑 Key Takeaways
- Vec<T> for dynamic arrays
- String for UTF-8 text
- HashMap<K, V> for key-value pairs
- Closures capture environment
- Iterators enable functional programming
- Use iterator adaptors for transformations

## ⏭️ Next Steps
Complete the labs and move to Module 7: Generics and Traits
