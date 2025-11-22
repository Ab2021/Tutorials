# Lab 3.3: Trait Bounds and Where Clauses

## Objective
Use trait bounds to constrain generic types.

## Exercises

### Exercise 1: Trait Bound Syntax
```rust
fn largest<T: PartialOrd>(list: &[T]) -> &T {
    let mut largest = &list[0];
    for item in list {
        if item > largest {
            largest = item;
        }
    }
    largest
}
```

### Exercise 2: Multiple Trait Bounds
```rust
fn print_and_compare<T: std::fmt::Display + PartialOrd>(a: T, b: T) {
    println!("a = {}, b = {}", a, b);
    if a > b {
        println!("a is larger");
    }
}
```

### Exercise 3: Where Clauses
```rust
fn complex_function<T, U>(t: T, u: U)
where
    T: std::fmt::Display + Clone,
    U: Clone + std::fmt::Debug,
{
    println!("t = {}", t);
    println!("u = {:?}", u);
}
```

## Success Criteria
✅ Use trait bounds  
✅ Apply multiple bounds  
✅ Use where clauses

## Next Steps
Lab 3.4: Associated Types
