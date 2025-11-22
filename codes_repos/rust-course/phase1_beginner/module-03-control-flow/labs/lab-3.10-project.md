# Lab 3.10: Control Flow Project - Number Guessing Game

## Objective
Build a complete number guessing game that combines all control flow concepts.

## Project Requirements

Create a number guessing game with:
- Random number generation (1-100)
- User input handling
- Attempt tracking
- Hint system
- Difficulty levels

## Starter Code

```rust
use std::io;

fn main() {
    println!("=== Number Guessing Game ===\n");
    
    let secret_number = 42; // TODO: Make random
    let max_attempts = 10;
    let mut attempts = 0;
    
    loop {
        println!("Attempt {}/{}", attempts + 1, max_attempts);
        println!("Enter your guess (1-100):");
        
        let guess = 50; // TODO: Get user input
        attempts += 1;
        
        match guess.cmp(&secret_number) {
            std::cmp::Ordering::Less => println!("Too low!"),
            std::cmp::Ordering::Greater => println!("Too high!"),
            std::cmp::Ordering::Equal => {
                println!("Correct! You won in {} attempts!", attempts);
                break;
            }
        }
        
        if attempts >= max_attempts {
            println!("Game over! The number was {}", secret_number);
            break;
        }
    }
}
```

## Features to Implement

### Level 1: Basic Game
- Fixed secret number
- Unlimited attempts
- Simple feedback (too high/low)

### Level 2: Limited Attempts
- Maximum 10 attempts
- Attempt counter
- Game over message

### Level 3: Difficulty Levels
- Easy: 15 attempts, range 1-50
- Medium: 10 attempts, range 1-100
- Hard: 7 attempts, range 1-200

### Level 4: Hint System
- After 3 failed attempts, offer hints
- Show if number is even/odd
- Show range (e.g., "between 40-60")

### Level 5: Score System
- Calculate score based on attempts
- Bonus for fewer attempts
- High score tracking

## Complete Solution

<details>
<summary>Click to reveal</summary>

```rust
use std::io;
use std::cmp::Ordering;

enum Difficulty {
    Easy,
    Medium,
    Hard,
}

impl Difficulty {
    fn max_attempts(&self) -> u32 {
        match self {
            Difficulty::Easy => 15,
            Difficulty::Medium => 10,
            Difficulty::Hard => 7,
        }
    }
    
    fn range(&self) -> (u32, u32) {
        match self {
            Difficulty::Easy => (1, 50),
            Difficulty::Medium => (1, 100),
            Difficulty::Hard => (1, 200),
        }
    }
}

fn get_user_input() -> u32 {
    loop {
        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read line");
        
        match input.trim().parse() {
            Ok(num) => return num,
            Err(_) => println!("Please enter a valid number!"),
        }
    }
}

fn play_game(difficulty: Difficulty) {
    let (min, max) = difficulty.range();
    let secret_number = 42; // In real game: rand::thread_rng().gen_range(min..=max)
    let max_attempts = difficulty.max_attempts();
    let mut attempts = 0;
    
    println!("\nGuess the number between {} and {}!", min, max);
    println!("You have {} attempts.\n", max_attempts);
    
    loop {
        attempts += 1;
        println!("Attempt {}/{}", attempts, max_attempts);
        println!("Enter your guess:");
        
        let guess = get_user_input();
        
        if guess < min || guess > max {
            println!("Please guess between {} and {}!", min, max);
            attempts -= 1;
            continue;
        }
        
        match guess.cmp(&secret_number) {
            Ordering::Less => {
                println!("Too low!");
                if attempts >= 3 {
                    println!("Hint: The number is {}", 
                             if secret_number % 2 == 0 { "even" } else { "odd" });
                }
            }
            Ordering::Greater => {
                println!("Too high!");
                if attempts >= 3 {
                    println!("Hint: The number is {}", 
                             if secret_number % 2 == 0 { "even" } else { "odd" });
                }
            }
            Ordering::Equal => {
                println!("\n🎉 Congratulations! You won!");
                println!("You guessed the number in {} attempts!", attempts);
                
                let score = calculate_score(attempts, max_attempts);
                println!("Your score: {}/100", score);
                break;
            }
        }
        
        if attempts >= max_attempts {
            println!("\n💀 Game Over!");
            println!("The number was {}", secret_number);
            break;
        }
        
        println!();
    }
}

fn calculate_score(attempts: u32, max_attempts: u32) -> u32 {
    let base_score = 100;
    let penalty = (attempts - 1) * 10;
    base_score.saturating_sub(penalty)
}

fn main() {
    println!("=== Number Guessing Game ===\n");
    println!("Select difficulty:");
    println!("1. Easy (1-50, 15 attempts)");
    println!("2. Medium (1-100, 10 attempts)");
    println!("3. Hard (1-200, 7 attempts)");
    
    let choice = get_user_input();
    
    let difficulty = match choice {
        1 => Difficulty::Easy,
        2 => Difficulty::Medium,
        3 => Difficulty::Hard,
        _ => {
            println!("Invalid choice, using Medium");
            Difficulty::Medium
        }
    };
    
    play_game(difficulty);
}
```

</details>

## Success Criteria
✅ Game runs without errors  
✅ Proper input validation  
✅ All difficulty levels work  
✅ Hints system implemented  
✅ Score calculation correct  
✅ Clean, readable code

## Extensions
1. Add play again functionality
2. Track high scores across sessions
3. Add timer for bonus points
4. Implement multiplayer mode

## Key Learnings
- Combining loops, match, and if/else
- User input handling
- Game state management
- Error handling
- Code organization

Congratulations on completing Module 03! Move to Module 04: Ownership Basics.
