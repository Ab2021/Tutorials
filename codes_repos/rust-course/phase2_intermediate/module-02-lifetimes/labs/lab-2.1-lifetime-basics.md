# Lab 2.1: Lifetime Annotations Basics

## Objective
Understand and use lifetime annotations in Rust.

## Theory
Lifetimes ensure references are valid for as long as they're used.

## Exercises

### Exercise 1: Function with Lifetimes
```rust
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}

fn main() {
    let string1 = String::from("long string");
    let string2 = String::from("short");
    
    let result = longest(&string1, &string2);
    println!("Longest: {}", result);
}
```

### Exercise 2: Lifetime in Structs
```rust
struct ImportantExcerpt<'a> {
    part: &'a str,
}

fn main() {
    let novel = String::from("Call me Ishmael. Some years ago...");
    let first_sentence = novel.split('.').next().expect("Could not find a '.'");
    
    let excerpt = ImportantExcerpt {
        part: first_sentence,
    };
}
```

### Exercise 3: Multiple Lifetimes
```rust
fn first_word<'a, 'b>(s: &'a str, _announcement: &'b str) -> &'a str {
    let bytes = s.as_bytes();
    
    for (i, &item) in bytes.iter().enumerate() {
        if item == b' ' {
            return &s[0..i];
        }
    }
    
    &s[..]
}
```

### Exercise 4: Lifetime Elision
```rust
// Compiler infers lifetimes
fn first_word(s: &str) -> &str {
    &s[..1]
}

// Explicit lifetimes
fn first_word_explicit<'a>(s: &'a str) -> &'a str {
    &s[..1]
}
```

### Exercise 5: Methods with Lifetimes
```rust
impl<'a> ImportantExcerpt<'a> {
    fn level(&self) -> i32 {
        3
    }
    
    fn announce_and_return_part(&self, announcement: &str) -> &str {
        println!("Attention: {}", announcement);
        self.part
    }
}
```

## Success Criteria
✅ Understand lifetime syntax  
✅ Use lifetimes in functions  
✅ Use lifetimes in structs  
✅ Know when lifetimes are needed

## Next Steps
Proceed to Lab 2.2: Lifetime Elision Rules
