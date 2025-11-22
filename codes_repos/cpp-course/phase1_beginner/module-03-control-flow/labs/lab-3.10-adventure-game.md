# Lab 3.10: Text Adventure Game

## Objective
Build a simple text-based adventure game combining variables, I/O, and control flow.

## Instructions

### Step 1: Game Loop
Create `adventure.cpp`. Set up a main game loop.

```cpp
#include <iostream>
#include <string>

int main() {
    bool playing = true;
    std::string name;
    
    std::cout << "Enter your name, adventurer: ";
    std::cin >> name;
    
    while (playing) {
        std::cout << "\nYou are in a dark room. Exits are North and East." << std::endl;
        std::cout << "What do you do? (n/e/q): ";
        
        char choice;
        std::cin >> choice;
        
        if (choice == 'q') {
            playing = false;
        } else if (choice == 'n') {
            std::cout << "You hit a wall. Ouch!" << std::endl;
        } else if (choice == 'e') {
            std::cout << "You found a treasure chest!" << std::endl;
            playing = false; // Win!
        } else {
            std::cout << "I don't understand." << std::endl;
        }
    }
    
    std::cout << "Game Over, " << name << "." << std::endl;
    return 0;
}
```

### Step 2: Adding State
Add variables to track health and gold.

```cpp
int health = 100;
int gold = 0;

// Inside loop
std::cout << "HP: " << health << " | Gold: " << gold << std::endl;
```

### Step 3: Random Encounters
Use `rand()` to make the 'North' path dangerous.
```cpp
if (choice == 'n') {
    int damage = rand() % 20 + 1;
    health -= damage;
    std::cout << "A goblin attacks! You take " << damage << " damage." << std::endl;
    if (health <= 0) {
        std::cout << "You died!" << std::endl;
        playing = false;
    }
}
```

## Challenges

### Challenge 1: Inventory System
Use a simple `int potions = 0;` variable.
Add a 'shop' option where user can buy potions with gold.
Add a 'drink' option to restore health.

### Challenge 2: Multiple Rooms
Use an integer `roomID` to track where the player is.
- Room 0: Start
- Room 1: Hallway
- Room 2: Dragon's Lair
Switch on `roomID` to describe the room and handle movement.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <cstdlib>
#include <ctime>

int main() {
    std::srand(std::time(nullptr));
    bool playing = true;
    int health = 100;
    int gold = 0;
    int potions = 0;
    int room = 0; // 0=Start, 1=Forest, 2=Cave
    
    while (playing && health > 0) {
        std::cout << "\n--- Status: HP " << health << " | Gold " << gold << " | Potions " << potions << " ---" << std::endl;
        
        switch (room) {
            case 0:
                std::cout << "You are at the village gate. (f)orest, (s)hop, (q)uit" << std::endl;
                break;
            case 1:
                std::cout << "You are in a dark forest. (c)ave, (v)illage" << std::endl;
                break;
            case 2:
                std::cout << "You are in a cave. A dragon sleeps here! (f)ight, (r)un" << std::endl;
                break;
        }
        
        char action;
        std::cin >> action;
        
        if (action == 'q') playing = false;
        else if (action == 'p' && potions > 0) {
            health += 20;
            potions--;
            std::cout << "Glug glug. Health restored." << std::endl;
        }
        else if (room == 0) {
            if (action == 'f') room = 1;
            else if (action == 's') {
                if (gold >= 10) { gold -= 10; potions++; std::cout << "Bought potion!\n"; }
                else std::cout << "Not enough gold!\n";
            }
        }
        else if (room == 1) {
            if (action == 'v') room = 0;
            else if (action == 'c') room = 2;
            else {
                std::cout << "You wander... found 5 gold!\n";
                gold += 5;
            }
        }
        else if (room == 2) {
            if (action == 'r') room = 1;
            else if (action == 'f') {
                std::cout << "You slay the dragon! You win!\n";
                playing = false;
            }
        }
    }
    
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented game loop
✅ Managed player state (health, gold)
✅ Handled user input
✅ Implemented random events
✅ Implemented simple inventory (Challenge 1)

## Key Learnings
- Combining loops, conditionals, and variables
- State management
- User interaction design

## Next Steps
Congratulations! You've completed Module 3.

Proceed to **Module 4: Functions and Scope** to organize your code into reusable blocks.
