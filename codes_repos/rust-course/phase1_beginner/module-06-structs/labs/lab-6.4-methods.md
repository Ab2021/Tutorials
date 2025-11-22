# Lab 6.4: Struct Methods and Associated Functions

## Objective
Learn to implement methods and associated functions for structs.

## Theory
Methods are functions defined within the context of a struct. Associated functions don't take `self` as a parameter.

## Exercises

### Exercise 1: Basic Methods
```rust
struct Rectangle {
    width: u32,
    height: u32,
}

impl Rectangle {
    fn area(&self) -> u32 {
        self.width * self.height
    }
    
    fn can_hold(&self, other: &Rectangle) -> bool {
        self.width > other.width && self.height > other.height
    }
}

fn main() {
    let rect = Rectangle { width: 30, height: 50 };
    println!("Area: {}", rect.area());
}
```

### Exercise 2: Mutable Methods
```rust
impl Rectangle {
    fn scale(&mut self, factor: u32) {
        self.width *= factor;
        self.height *= factor;
    }
}
```

### Exercise 3: Associated Functions (Constructors)
```rust
impl Rectangle {
    fn new(width: u32, height: u32) -> Self {
        Rectangle { width, height }
    }
    
    fn square(size: u32) -> Self {
        Rectangle {
            width: size,
            height: size,
        }
    }
}

fn main() {
    let rect = Rectangle::new(30, 50);
    let square = Rectangle::square(25);
}
```

### Exercise 4: Multiple impl Blocks
```rust
impl Rectangle {
    fn area(&self) -> u32 {
        self.width * self.height
    }
}

impl Rectangle {
    fn perimeter(&self) -> u32 {
        2 * (self.width + self.height)
    }
}
```

### Exercise 5: Circle Struct
```rust
struct Circle {
    radius: f64,
}

impl Circle {
    fn new(radius: f64) -> Self {
        Circle { radius }
    }
    
    fn area(&self) -> f64 {
        std::f64::consts::PI * self.radius * self.radius
    }
    
    fn circumference(&self) -> f64 {
        2.0 * std::f64::consts::PI * self.radius
    }
    
    fn scale(&mut self, factor: f64) {
        self.radius *= factor;
    }
}
```

## Success Criteria
✅ Implement methods with &self  
✅ Implement mutable methods with &mut self  
✅ Create associated functions  
✅ Use multiple impl blocks

## Next Steps
Proceed to Lab 6.5: Advanced Struct Patterns
