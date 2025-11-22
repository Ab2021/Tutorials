# Lab 7.1: Enum Basics

## Objective
Learn to define and use enums in Rust.

## Theory
Enums allow you to define a type by enumerating its possible variants.

## Exercises

### Exercise 1: Simple Enum
```rust
enum Direction {
    North,
    South,
    East,
    West,
}

fn move_player(dir: Direction) {
    match dir {
        Direction::North => println!("Moving north"),
        Direction::South => println!("Moving south"),
        Direction::East => println!("Moving east"),
        Direction::West => println!("Moving west"),
    }
}
```

### Exercise 2: Enum with Data
```rust
enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
    ChangeColor(i32, i32, i32),
}

impl Message {
    fn call(&self) {
        match self {
            Message::Quit => println!("Quit"),
            Message::Move { x, y } => println!("Move to ({}, {})", x, y),
            Message::Write(text) => println!("Text: {}", text),
            Message::ChangeColor(r, g, b) => println!("Color: ({}, {}, {})", r, g, b),
        }
    }
}
```

### Exercise 3: IP Address Enum
```rust
enum IpAddr {
    V4(u8, u8, u8, u8),
    V6(String),
}

let home = IpAddr::V4(127, 0, 0, 1);
let loopback = IpAddr::V6(String::from("::1"));
```

### Exercise 4: Build a Shape Enum
```rust
enum Shape {
    Circle(f64),
    Rectangle(f64, f64),
    Triangle(f64, f64, f64),
}

fn area(shape: &Shape) -> f64 {
    // Calculate area based on shape
}
```

### Exercise 5: Traffic Light
```rust
enum TrafficLight {
    Red,
    Yellow,
    Green,
}

impl TrafficLight {
    fn time(&self) -> u32 {
        match self {
            TrafficLight::Red => 60,
            TrafficLight::Yellow => 10,
            TrafficLight::Green => 90,
        }
    }
}
```

## Success Criteria
✅ Can define enums  
✅ Understand enum variants with data  
✅ Use match with enums  
✅ Implement methods on enums

## Next Steps
Proceed to Lab 7.2: Option<T>
