# Lab 5.2: Mutable References

## Objective
Learn to use mutable references and understand the one-mutable-reference rule.

## Theory
A mutable reference (`&mut T`) allows you to modify borrowed data. Only ONE mutable reference can exist at a time.

## Exercises

### Exercise 1: Basic Mutable Reference
```rust
fn main() {
    let mut s = String::from("hello");
    change(&mut s);
    println!("{}", s);
}

fn change(s: &mut String) {
    s.push_str(", world");
}
```

### Exercise 2: Modify a Vector
```rust
fn double_values(v: &mut Vec<i32>) {
    for i in 0..v.len() {
        v[i] *= 2;
    }
}
```

### Exercise 3: The One Mutable Reference Rule
```rust
fn main() {
    let mut s = String::from("hello");
    
    let r1 = &mut s;
    // let r2 = &mut s; // ERROR! Can't have two mutable references
    
    println!("{}", r1);
}
```

### Exercise 4: Mutable and Immutable References
```rust
fn main() {
    let mut s = String::from("hello");
    
    let r1 = &s;
    let r2 = &s;
    println!("{} and {}", r1, r2);
    
    let r3 = &mut s; // OK! r1 and r2 are no longer used
    r3.push_str(" world");
}
```

### Exercise 5: Build a Counter
```rust
struct Counter {
    count: u32,
}

impl Counter {
    fn new() -> Self {
        Counter { count: 0 }
    }
    
    fn increment(&mut self) {
        self.count += 1;
    }
    
    fn get(&self) -> u32 {
        self.count
    }
}
```

## Success Criteria
✅ Can use mutable references  
✅ Understand the one-mutable-reference rule  
✅ Know when mutable references go out of scope  
✅ Can mix immutable and mutable references correctly

## Next Steps
Proceed to Lab 5.3: Reference Rules and Patterns
