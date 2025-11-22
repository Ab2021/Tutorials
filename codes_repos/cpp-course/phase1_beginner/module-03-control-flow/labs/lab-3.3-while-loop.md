# Lab 3.3: While Loop Number Guessing Game

## Objective
Use a `while` loop to create a simple game that repeats until a condition is met.

## Instructions

### Step 1: Random Number Generation
Create `guess.cpp`. Initialize random seed:

```cpp
#include <iostream>
#include <cstdlib> // rand, srand
#include <ctime>   // time

int main() {
    std::srand(std::time(nullptr)); // Seed random number generator
    int secret = std::rand() % 100 + 1; // 1 to 100
    
    // TODO: Loop
    
    return 0;
}
```

### Step 2: The Loop
Create a `while` loop that asks for a guess until it matches the secret.

```cpp
int guess = 0;
while (guess != secret) {
    std::cout << "Guess (1-100): ";
    std::cin >> guess;
    
    if (guess < secret) {
        std::cout << "Too low!" << std::endl;
    } else if (guess > secret) {
        std::cout << "Too high!" << std::endl;
    } else {
        std::cout << "Correct!" << std::endl;
    }
}
```

### Step 3: Counter
Add a variable `attempts` to count how many tries it took. Print it at the end.

## Challenges

### Challenge 1: Limited Guesses
Modify the loop condition to also check if `attempts < 10`. If they run out of guesses, print "Game Over".

### Challenge 2: Play Again
Wrap the entire game logic in another `while` loop that asks "Play again? (y/n)" after the game finishes.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <cstdlib>
#include <ctime>

int main() {
    std::srand(std::time(nullptr));
    char playAgain = 'y';
    
    while (playAgain == 'y') {
        int secret = std::rand() % 100 + 1;
        int guess = 0;
        int attempts = 0;
        const int maxAttempts = 10;
        
        std::cout << "New Game! Guess 1-100." << std::endl;
        
        while (guess != secret && attempts < maxAttempts) {
            std::cout << "Guess: ";
            std::cin >> guess;
            attempts++;
            
            if (guess < secret) std::cout << "Too low!\n";
            else if (guess > secret) std::cout << "Too high!\n";
            else std::cout << "You won in " << attempts << " tries!\n";
        }
        
        if (guess != secret) {
            std::cout << "Game Over! Secret was " << secret << std::endl;
        }
        
        std::cout << "Play again? (y/n): ";
        std::cin >> playAgain;
    }
    
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented `while` loop correctly
✅ Used random number generation
✅ Added attempt counter
✅ Implemented "Play Again" loop (Challenge 2)

## Key Learnings
- `while` loop syntax
- `rand()` basics (note: C++11 has better random, covered later)
- Nested loops

## Next Steps
Proceed to **Lab 3.4: Do-While Loop** for input validation.
