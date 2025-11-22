# Lab 2.4: Pattern Matching with match

## Objective
Master the `match` expression for powerful pattern matching and control flow.

## Project Setup

```bash
cargo new pattern_matching
cd pattern_matching
```

## Part 1: Basic Pattern Matching

### Exercise 1: Number Classifier

Create a function that classifies numbers:

```rust
fn classify_number(n: i32) -> &'static str {
    match n {
        // TODO: Match specific cases
        // 0 => "zero"
        // 1 => "one"
        // 2..=10 => "small"
        // 11..=100 => "medium"
        // _ => "large"
    }
}

fn main() {
    let numbers = [0, 1, 5, 50, 150];
    
    for num in numbers {
        println!("{} is {}", num, classify_number(num));
    }
}
```

### Exercise 2: Day of Week

```rust
fn day_name(day: u8) -> &'static str {
    match day {
        // TODO: Match 1-7 to day names
        // Use _ for invalid days
    }
}

fn main() {
    for day in 1..=8 {
        println!("Day {}: {}", day, day_name(day));
    }
}
```

### Exercise 3: Multiple Patterns

```rust
fn is_vowel(c: char) -> bool {
    match c {
        // TODO: Match 'a', 'e', 'i', 'o', 'u' (and uppercase)
        // Use | to combine patterns
    }
}

fn main() {
    let letters = ['a', 'b', 'e', 'x', 'i', 'z'];
    
    for letter in letters {
        if is_vowel(letter) {
            println!("'{}' is a vowel", letter);
        } else {
            println!("'{}' is a consonant", letter);
        }
    }
}
```

## Part 2: Match with Guards

### Exercise 4: Number Properties

```rust
fn describe_number(n: i32) {
    match n {
        // TODO: Use guards (if conditions)
        // n if n < 0 => "negative"
        // n if n % 2 == 0 => "positive even"
        // n if n % 2 != 0 => "positive odd"
    }
}

fn main() {
    let numbers = [-5, 0, 4, 7, 12];
    
    for num in numbers {
        describe_number(num);
    }
}
```

### Exercise 5: Grade Calculator

```rust
fn letter_grade(score: u8) -> char {
    match score {
        // TODO: Implement grading scale
        // 90-100 => 'A'
        // 80-89 => 'B'
        // 70-79 => 'C'
        // 60-69 => 'D'
        // 0-59 => 'F'
    }
}

fn main() {
    let scores = [95, 87, 72, 65, 45];
    
    for score in scores {
        println!("Score {}: Grade {}", score, letter_grade(score));
    }
}
```

## Part 3: Matching Tuples

### Exercise 6: Coordinate Classifier

```rust
fn classify_point(point: (i32, i32)) {
    match point {
        (0, 0) => println!("Origin"),
        (0, y) => println!("On Y-axis at y={}", y),
        (x, 0) => println!("On X-axis at x={}", x),
        (x, y) if x == y => println!("On diagonal at ({}, {})", x, y),
        (x, y) => println!("Point at ({}, {})", x, y),
    }
}

fn main() {
    let points = [(0, 0), (0, 5), (3, 0), (4, 4), (2, 3)];
    
    for point in points {
        classify_point(point);
    }
}
```

### Exercise 7: RGB Color Matcher

```rust
fn color_name(rgb: (u8, u8, u8)) -> &'static str {
    match rgb {
        (255, 0, 0) => "Red",
        (0, 255, 0) => "Green",
        (0, 0, 255) => "Blue",
        (255, 255, 0) => "Yellow",
        (255, 0, 255) => "Magenta",
        (0, 255, 255) => "Cyan",
        (255, 255, 255) => "White",
        (0, 0, 0) => "Black",
        _ => "Custom color",
    }
}

fn main() {
    let colors = [
        (255, 0, 0),
        (0, 255, 0),
        (128, 128, 128),
    ];
    
    for color in colors {
        println!("{:?} is {}", color, color_name(color));
    }
}
```

## Part 4: Practical Applications

### Exercise 8: Simple Calculator

```rust
fn calculate(operation: char, a: f64, b: f64) -> f64 {
    match operation {
        '+' => a + b,
        '-' => a - b,
        '*' => a * b,
        '/' => {
            if b != 0.0 {
                a / b
            } else {
                println!("Error: Division by zero!");
                0.0
            }
        }
        _ => {
            println!("Unknown operation!");
            0.0
        }
    }
}

fn main() {
    println!("10 + 5 = {}", calculate('+', 10.0, 5.0));
    println!("10 - 5 = {}", calculate('-', 10.0, 5.0));
    println!("10 * 5 = {}", calculate('*', 10.0, 5.0));
    println!("10 / 5 = {}", calculate('/', 10.0, 5.0));
    println!("10 / 0 = {}", calculate('/', 10.0, 0.0));
}
```

### Exercise 9: Rock Paper Scissors

```rust
#[derive(Debug)]
enum Move {
    Rock,
    Paper,
    Scissors,
}

fn winner(player1: Move, player2: Move) -> &'static str {
    match (player1, player2) {
        // TODO: Implement game logic
        // Rock beats Scissors
        // Scissors beats Paper
        // Paper beats Rock
        // Same moves => Tie
    }
}

fn main() {
    use Move::*;
    
    println!("Rock vs Scissors: {}", winner(Rock, Scissors));
    println!("Paper vs Rock: {}", winner(Paper, Rock));
    println!("Scissors vs Scissors: {}", winner(Scissors, Scissors));
}
```

## Challenges

### Challenge 1: FizzBuzz with match

Implement FizzBuzz using match:

```rust
fn fizzbuzz(n: u32) -> String {
    match (n % 3, n % 5) {
        // TODO: Implement FizzBuzz logic
    }
}

fn main() {
    for i in 1..=20 {
        println!("{}: {}", i, fizzbuzz(i));
    }
}
```

### Challenge 2: Traffic Light State Machine

```rust
enum TrafficLight {
    Red,
    Yellow,
    Green,
}

fn next_light(current: TrafficLight) -> TrafficLight {
    // TODO: Implement state transitions
}

fn light_duration(light: &TrafficLight) -> u32 {
    // TODO: Return duration in seconds
    // Red: 60, Yellow: 5, Green: 55
}

fn main() {
    use TrafficLight::*;
    
    let mut light = Red;
    
    for _ in 0..6 {
        println!("{:?} light for {} seconds", light, light_duration(&light));
        light = next_light(light);
    }
}
```

## Solutions

<details>
<summary>Exercise 1 Solution</summary>

```rust
fn classify_number(n: i32) -> &'static str {
    match n {
        0 => "zero",
        1 => "one",
        2..=10 => "small",
        11..=100 => "medium",
        _ => "large",
    }
}
```

</details>

<details>
<summary>Exercise 3 Solution</summary>

```rust
fn is_vowel(c: char) -> bool {
    match c {
        'a' | 'e' | 'i' | 'o' | 'u' |
        'A' | 'E' | 'I' | 'O' | 'U' => true,
        _ => false,
    }
}
```

</details>

<details>
<summary>Exercise 9 Solution</summary>

```rust
#[derive(Debug)]
enum Move {
    Rock,
    Paper,
    Scissors,
}

fn winner(player1: Move, player2: Move) -> &'static str {
    use Move::*;
    
    match (player1, player2) {
        (Rock, Scissors) | (Scissors, Paper) | (Paper, Rock) => "Player 1 wins!",
        (Scissors, Rock) | (Paper, Scissors) | (Rock, Paper) => "Player 2 wins!",
        _ => "It's a tie!",
    }
}
```

</details>

<details>
<summary>Challenge 1 Solution</summary>

```rust
fn fizzbuzz(n: u32) -> String {
    match (n % 3, n % 5) {
        (0, 0) => "FizzBuzz".to_string(),
        (0, _) => "Fizz".to_string(),
        (_, 0) => "Buzz".to_string(),
        _ => n.to_string(),
    }
}
```

</details>

<details>
<summary>Challenge 2 Solution</summary>

```rust
#[derive(Debug)]
enum TrafficLight {
    Red,
    Yellow,
    Green,
}

fn next_light(current: TrafficLight) -> TrafficLight {
    use TrafficLight::*;
    
    match current {
        Red => Green,
        Green => Yellow,
        Yellow => Red,
    }
}

fn light_duration(light: &TrafficLight) -> u32 {
    use TrafficLight::*;
    
    match light {
        Red => 60,
        Yellow => 5,
        Green => 55,
    }
}
```

</details>

## Success Criteria

✅ Understand basic pattern matching syntax  
✅ Can use match guards with if conditions  
✅ Can match on tuples and multiple values  
✅ Understand exhaustive matching  
✅ Can use | for multiple patterns  
✅ Completed at least two challenges

## Key Learnings

- `match` must be exhaustive (cover all cases)
- Use `_` as a catch-all pattern
- Guards add conditional logic to patterns
- Can match on tuples, ranges, and literals
- `|` combines multiple patterns
- `match` is an expression (returns a value)

## Next Steps

Congratulations on completing Module 2! You've learned:
- Variables and mutability
- Data types (scalar and compound)
- Functions and control flow
- Pattern matching

Move on to **Module 3: Ownership and Borrowing** - the most important concept in Rust!
