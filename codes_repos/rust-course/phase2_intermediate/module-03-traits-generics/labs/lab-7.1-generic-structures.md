# Lab 7.1: Building Generic Data Structures

## Objective
Master generics by implementing common data structures from scratch.

## Setup
```bash
cargo new generic_structures
cd generic_structures
```

## Part 1: Generic Stack

### Exercise 1: Implement a Stack
```rust
struct Stack<T> {
    items: Vec<T>,
}

impl<T> Stack<T> {
    fn new() -> Self {
        Stack { items: Vec::new() }
    }
    
    fn push(&mut self, item: T) {
        self.items.push(item);
    }
    
    fn pop(&mut self) -> Option<T> {
        self.items.pop()
    }
    
    fn peek(&self) -> Option<&T> {
        self.items.last()
    }
    
    fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
    
    fn size(&self) -> usize {
        self.items.len()
    }
}

fn main() {
    let mut stack = Stack::new();
    stack.push(1);
    stack.push(2);
    stack.push(3);
    
    println!("Top: {:?}", stack.peek());
    println!("Pop: {:?}", stack.pop());
    println!("Size: {}", stack.size());
    
    // Works with different types
    let mut string_stack = Stack::new();
    string_stack.push(String::from("hello"));
    string_stack.push(String::from("world"));
}
```

## Part 2: Generic Pair

### Exercise 2: Pair with Different Types
```rust
struct Pair<T, U> {
    first: T,
    second: U,
}

impl<T, U> Pair<T, U> {
    fn new(first: T, second: U) -> Self {
        Pair { first, second }
    }
    
    fn first(&self) -> &T {
        &self.first
    }
    
    fn second(&self) -> &U {
        &self.second
    }
    
    fn swap(self) -> Pair<U, T> {
        Pair {
            first: self.second,
            second: self.first,
        }
    }
}

fn main() {
    let pair = Pair::new(42, "hello");
    println!("First: {}, Second: {}", pair.first(), pair.second());
    
    let swapped = pair.swap();
    println!("After swap - First: {}, Second: {}", swapped.first(), swapped.second());
}
```

## Part 3: Generic with Trait Bounds

### Exercise 3: Comparable Container
```rust
use std::cmp::PartialOrd;

struct Container<T: PartialOrd> {
    value: T,
}

impl<T: PartialOrd> Container<T> {
    fn new(value: T) -> Self {
        Container { value }
    }
    
    fn is_greater_than(&self, other: &Container<T>) -> bool {
        self.value > other.value
    }
    
    fn max(self, other: Self) -> Self {
        if self.value > other.value {
            self
        } else {
            other
        }
    }
}

fn main() {
    let c1 = Container::new(10);
    let c2 = Container::new(20);
    
    println!("c1 > c2: {}", c1.is_greater_than(&c2));
    
    let max_container = Container::new(15).max(Container::new(25));
    println!("Max value: {}", max_container.value);
}
```

## Part 4: Generic Binary Tree

### Exercise 4: Binary Search Tree
```rust
use std::cmp::Ordering;

#[derive(Debug)]
struct TreeNode<T> {
    value: T,
    left: Option<Box<TreeNode<T>>>,
    right: Option<Box<TreeNode<T>>>,
}

impl<T: Ord> TreeNode<T> {
    fn new(value: T) -> Self {
        TreeNode {
            value,
            left: None,
            right: None,
        }
    }
    
    fn insert(&mut self, value: T) {
        match value.cmp(&self.value) {
            Ordering::Less => {
                match self.left {
                    Some(ref mut node) => node.insert(value),
                    None => self.left = Some(Box::new(TreeNode::new(value))),
                }
            }
            Ordering::Greater => {
                match self.right {
                    Some(ref mut node) => node.insert(value),
                    None => self.right = Some(Box::new(TreeNode::new(value))),
                }
            }
            Ordering::Equal => {} // Value already exists
        }
    }
    
    fn contains(&self, value: &T) -> bool {
        match value.cmp(&self.value) {
            Ordering::Equal => true,
            Ordering::Less => self.left.as_ref().map_or(false, |node| node.contains(value)),
            Ordering::Greater => self.right.as_ref().map_or(false, |node| node.contains(value)),
        }
    }
}

fn main() {
    let mut tree = TreeNode::new(5);
    tree.insert(3);
    tree.insert(7);
    tree.insert(1);
    tree.insert(9);
    
    println!("Contains 3: {}", tree.contains(&3));
    println!("Contains 6: {}", tree.contains(&6));
}
```

## Part 5: Generic Result Type

### Exercise 5: Custom Result
```rust
enum MyResult<T, E> {
    Success(T),
    Failure(E),
}

impl<T, E> MyResult<T, E> {
    fn is_success(&self) -> bool {
        matches!(self, MyResult::Success(_))
    }
    
    fn is_failure(&self) -> bool {
        matches!(self, MyResult::Failure(_))
    }
    
    fn unwrap(self) -> T where E: std::fmt::Debug {
        match self {
            MyResult::Success(value) => value,
            MyResult::Failure(err) => panic!("Called unwrap on Failure: {:?}", err),
        }
    }
    
    fn map<U, F>(self, f: F) -> MyResult<U, E>
    where
        F: FnOnce(T) -> U,
    {
        match self {
            MyResult::Success(value) => MyResult::Success(f(value)),
            MyResult::Failure(err) => MyResult::Failure(err),
        }
    }
}

fn divide(a: i32, b: i32) -> MyResult<i32, String> {
    if b == 0 {
        MyResult::Failure(String::from("Division by zero"))
    } else {
        MyResult::Success(a / b)
    }
}

fn main() {
    let result = divide(10, 2);
    println!("Is success: {}", result.is_success());
    
    let doubled = divide(10, 2).map(|x| x * 2);
    match doubled {
        MyResult::Success(value) => println!("Result: {}", value),
        MyResult::Failure(err) => println!("Error: {}", err),
    }
}
```

## Success Criteria
✅ Implement generic data structures  
✅ Use trait bounds appropriately  
✅ Understand generic type parameters  
✅ Create reusable, type-safe code  
✅ Handle multiple generic parameters

## Key Learnings
- Generics enable code reuse without sacrificing type safety
- Trait bounds constrain what types can be used
- Generic implementations can have specific constraints
- Rust's generics have zero runtime cost (monomorphization)

## Next Lab
Proceed to Lab 7.2: Trait Implementation!
