# Lab 4.1: Box<T> Deep Dive

## Objective
Master Box<T> for heap allocation and recursive types.

## Exercises

### Exercise 1: Basic Box Usage
```rust
fn main() {
    let b = Box::new(5);
    println!("b = {}", b);
}
```

### Exercise 2: Recursive Types
```rust
enum List {
    Cons(i32, Box<List>),
    Nil,
}

use List::{Cons, Nil};

fn main() {
    let list = Cons(1, Box::new(Cons(2, Box::new(Cons(3, Box::new(Nil))))));
}
```

### Exercise 3: Trait Objects
```rust
trait Draw {
    fn draw(&self);
}

struct Button;
impl Draw for Button {
    fn draw(&self) {
        println!("Drawing button");
    }
}

fn main() {
    let button: Box<dyn Draw> = Box::new(Button);
    button.draw();
}
```

## Success Criteria
✅ Use Box for heap allocation  
✅ Create recursive types  
✅ Work with trait objects

## Next Steps
Lab 4.2: Rc and Arc
