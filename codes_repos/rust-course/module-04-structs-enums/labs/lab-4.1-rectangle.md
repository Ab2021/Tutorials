# Lab 4.1: Building a Rectangle Calculator with Structs

## Objective
Practice defining structs, implementing methods, and using associated functions.

## Setup
```bash
cargo new rectangle_calculator
cd rectangle_calculator
```

## Part 1: Define the Struct

```rust
struct Rectangle {
    width: u32,
    height: u32,
}

fn main() {
    let rect = Rectangle {
        width: 30,
        height: 50,
    };
    
    println!("Rectangle: {}x{}", rect.width, rect.height);
}
```

## Part 2: Implement Methods

Add these methods to Rectangle:

```rust
impl Rectangle {
    // Calculate area
    fn area(&self) -> u32 {
        // TODO: Implement
    }
    
    // Calculate perimeter
    fn perimeter(&self) -> u32 {
        // TODO: Implement
    }
    
    // Check if it's a square
    fn is_square(&self) -> bool {
        // TODO: Implement
    }
    
    // Check if it can hold another rectangle
    fn can_hold(&self, other: &Rectangle) -> bool {
        // TODO: Implement
    }
}
```

## Part 3: Associated Functions

```rust
impl Rectangle {
    // Create a square
    fn square(size: u32) -> Rectangle {
        // TODO: Implement
    }
    
    // Create from area (square root)
    fn from_area(area: u32) -> Option<Rectangle> {
        // TODO: Implement (return None if not perfect square)
    }
}
```

## Solutions

<details>
<summary>Click to reveal</summary>

```rust
struct Rectangle {
    width: u32,
    height: u32,
}

impl Rectangle {
    fn area(&self) -> u32 {
        self.width * self.height
    }
    
    fn perimeter(&self) -> u32 {
        2 * (self.width + self.height)
    }
    
    fn is_square(&self) -> bool {
        self.width == self.height
    }
    
    fn can_hold(&self, other: &Rectangle) -> bool {
        self.width > other.width && self.height > other.height
    }
    
    fn square(size: u32) -> Rectangle {
        Rectangle {
            width: size,
            height: size,
        }
    }
    
    fn from_area(area: u32) -> Option<Rectangle> {
        let side = (area as f64).sqrt() as u32;
        if side * side == area {
            Some(Rectangle::square(side))
        } else {
            None
        }
    }
}

fn main() {
    let rect = Rectangle { width: 30, height: 50 };
    let square = Rectangle::square(20);
    
    println!("Area: {}", rect.area());
    println!("Perimeter: {}", rect.perimeter());
    println!("Is square: {}", rect.is_square());
    println!("Can hold: {}", rect.can_hold(&square));
}
```

</details>

## Success Criteria
✅ Struct defined with width and height  
✅ All methods implemented correctly  
✅ Associated functions work as expected  
✅ Code compiles and runs without errors
