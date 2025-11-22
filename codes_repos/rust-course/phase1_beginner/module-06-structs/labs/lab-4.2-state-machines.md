# Lab 4.2: Building a State Machine with Enums

## Objective
Learn to use enums to model state machines and implement complex behavior with pattern matching.

## Project Overview
Build a traffic light controller and a vending machine using enum-based state machines.

## Setup
```bash
cargo new state_machines
cd state_machines
```

## Part 1: Traffic Light Controller

### Step 1: Define the States
```rust
#[derive(Debug, PartialEq, Clone, Copy)]
enum TrafficLight {
    Red,
    Yellow,
    Green,
}

impl TrafficLight {
    fn duration(&self) -> u32 {
        match self {
            TrafficLight::Red => 60,
            TrafficLight::Yellow => 5,
            TrafficLight::Green => 55,
        }
    }
    
    fn next(&self) -> TrafficLight {
        // TODO: Implement state transition
        // Red -> Green -> Yellow -> Red
    }
    
    fn can_cross(&self) -> bool {
        // TODO: Return true only for Green
    }
}

fn main() {
    let mut light = TrafficLight::Red;
    
    for _ in 0..6 {
        println!("{:?} light for {} seconds - Can cross: {}", 
                 light, light.duration(), light.can_cross());
        light = light.next();
    }
}
```

## Part 2: Vending Machine

### Step 2: Define Vending Machine States
```rust
#[derive(Debug, PartialEq)]
enum VendingMachineState {
    Idle,
    HasCoins(u32),  // Amount in cents
    Dispensing,
}

struct VendingMachine {
    state: VendingMachineState,
    item_price: u32,  // in cents
}

impl VendingMachine {
    fn new(item_price: u32) -> Self {
        VendingMachine {
            state: VendingMachineState::Idle,
            item_price,
        }
    }
    
    fn insert_coin(&mut self, amount: u32) {
        // TODO: Implement coin insertion logic
        // Update state based on total amount
    }
    
    fn dispense(&mut self) -> Result<u32, String> {
        // TODO: Dispense item and return change
        // Return error if not enough money
    }
    
    fn cancel(&mut self) -> u32 {
        // TODO: Return all inserted coins and reset to Idle
    }
    
    fn current_amount(&self) -> u32 {
        // TODO: Return current amount inserted
    }
}

fn main() {
    let mut machine = VendingMachine::new(75);  // Item costs 75 cents
    
    println!("Item price: {} cents", machine.item_price);
    
    machine.insert_coin(25);
    println!("Inserted 25 cents. Total: {}", machine.current_amount());
    
    machine.insert_coin(25);
    println!("Inserted 25 cents. Total: {}", machine.current_amount());
    
    machine.insert_coin(25);
    println!("Inserted 25 cents. Total: {}", machine.current_amount());
    
    match machine.dispense() {
        Ok(change) => println!("Item dispensed! Change: {} cents", change),
        Err(e) => println!("Error: {}", e),
    }
}
```

## Part 3: Door Lock System

### Step 3: Multi-State Door Lock
```rust
#[derive(Debug, PartialEq)]
enum DoorState {
    Locked,
    Unlocked,
    Open,
}

struct Door {
    state: DoorState,
    code: String,
}

impl Door {
    fn new(code: String) -> Self {
        Door {
            state: DoorState::Locked,
            code,
        }
    }
    
    fn enter_code(&mut self, code: &str) -> Result<(), String> {
        // TODO: Unlock if code is correct and door is locked
    }
    
    fn open(&mut self) -> Result<(), String> {
        // TODO: Open if unlocked, error if locked
    }
    
    fn close(&mut self) {
        // TODO: Close door (goes to Unlocked state)
    }
    
    fn lock(&mut self) -> Result<(), String> {
        // TODO: Lock if door is closed (Unlocked state)
    }
    
    fn status(&self) -> &str {
        match self.state {
            DoorState::Locked => "Locked",
            DoorState::Unlocked => "Unlocked",
            DoorState::Open => "Open",
        }
    }
}

fn main() {
    let mut door = Door::new(String::from("1234"));
    
    println!("Door status: {}", door.status());
    
    // Try to open locked door
    if let Err(e) = door.open() {
        println!("Error: {}", e);
    }
    
    // Enter correct code
    door.enter_code("1234").unwrap();
    println!("Door status: {}", door.status());
    
    // Open door
    door.open().unwrap();
    println!("Door status: {}", door.status());
}
```

## Part 4: Game Character State

### Step 4: RPG Character States
```rust
#[derive(Debug, PartialEq)]
enum CharacterState {
    Idle,
    Walking { speed: f32 },
    Running { speed: f32 },
    Jumping { height: f32 },
    Attacking { damage: u32 },
    Dead,
}

struct Character {
    name: String,
    health: i32,
    state: CharacterState,
}

impl Character {
    fn new(name: String) -> Self {
        Character {
            name,
            health: 100,
            state: CharacterState::Idle,
        }
    }
    
    fn walk(&mut self) {
        if self.health > 0 {
            self.state = CharacterState::Walking { speed: 2.0 };
        }
    }
    
    fn run(&mut self) {
        // TODO: Set running state with speed 5.0
    }
    
    fn jump(&mut self) {
        // TODO: Set jumping state with height 3.0
    }
    
    fn attack(&mut self, damage: u32) {
        // TODO: Set attacking state
    }
    
    fn take_damage(&mut self, damage: i32) {
        // TODO: Reduce health, set to Dead if health <= 0
    }
    
    fn stop(&mut self) {
        // TODO: Return to Idle state (if not dead)
    }
    
    fn describe_state(&self) -> String {
        match &self.state {
            CharacterState::Idle => format!("{} is standing still", self.name),
            CharacterState::Walking { speed } => {
                format!("{} is walking at {} m/s", self.name, speed)
            }
            CharacterState::Running { speed } => {
                format!("{} is running at {} m/s", self.name, speed)
            }
            CharacterState::Jumping { height } => {
                format!("{} is jumping {} meters high", self.name, height)
            }
            CharacterState::Attacking { damage } => {
                format!("{} is attacking with {} damage", self.name, damage)
            }
            CharacterState::Dead => format!("{} is dead", self.name),
        }
    }
}

fn main() {
    let mut hero = Character::new(String::from("Hero"));
    
    println!("{}", hero.describe_state());
    
    hero.walk();
    println!("{}", hero.describe_state());
    
    hero.run();
    println!("{}", hero.describe_state());
    
    hero.attack(25);
    println!("{}", hero.describe_state());
    
    hero.take_damage(150);
    println!("{}", hero.describe_state());
}
```

## Complete Solutions

<details>
<summary>Traffic Light Solution</summary>

```rust
impl TrafficLight {
    fn next(&self) -> TrafficLight {
        match self {
            TrafficLight::Red => TrafficLight::Green,
            TrafficLight::Green => TrafficLight::Yellow,
            TrafficLight::Yellow => TrafficLight::Red,
        }
    }
    
    fn can_cross(&self) -> bool {
        *self == TrafficLight::Green
    }
}
```

</details>

<details>
<summary>Vending Machine Solution</summary>

```rust
impl VendingMachine {
    fn insert_coin(&mut self, amount: u32) {
        let current = self.current_amount();
        self.state = VendingMachineState::HasCoins(current + amount);
    }
    
    fn dispense(&mut self) -> Result<u32, String> {
        let amount = self.current_amount();
        
        if amount >= self.item_price {
            let change = amount - self.item_price;
            self.state = VendingMachineState::Idle;
            Ok(change)
        } else {
            Err(format!("Not enough money. Need {} more cents", 
                       self.item_price - amount))
        }
    }
    
    fn cancel(&mut self) -> u32 {
        let amount = self.current_amount();
        self.state = VendingMachineState::Idle;
        amount
    }
    
    fn current_amount(&self) -> u32 {
        match self.state {
            VendingMachineState::HasCoins(amount) => amount,
            _ => 0,
        }
    }
}
```

</details>

<details>
<summary>Door Lock Solution</summary>

```rust
impl Door {
    fn enter_code(&mut self, code: &str) -> Result<(), String> {
        match self.state {
            DoorState::Locked => {
                if code == self.code {
                    self.state = DoorState::Unlocked;
                    Ok(())
                } else {
                    Err(String::from("Incorrect code"))
                }
            }
            _ => Err(String::from("Door is not locked")),
        }
    }
    
    fn open(&mut self) -> Result<(), String> {
        match self.state {
            DoorState::Unlocked => {
                self.state = DoorState::Open;
                Ok(())
            }
            DoorState::Locked => Err(String::from("Door is locked")),
            DoorState::Open => Err(String::from("Door is already open")),
        }
    }
    
    fn close(&mut self) {
        if self.state == DoorState::Open {
            self.state = DoorState::Unlocked;
        }
    }
    
    fn lock(&mut self) -> Result<(), String> {
        match self.state {
            DoorState::Unlocked => {
                self.state = DoorState::Locked;
                Ok(())
            }
            DoorState::Open => Err(String::from("Cannot lock an open door")),
            DoorState::Locked => Err(String::from("Door is already locked")),
        }
    }
}
```

</details>

<details>
<summary>Character State Solution</summary>

```rust
impl Character {
    fn run(&mut self) {
        if self.health > 0 {
            self.state = CharacterState::Running { speed: 5.0 };
        }
    }
    
    fn jump(&mut self) {
        if self.health > 0 {
            self.state = CharacterState::Jumping { height: 3.0 };
        }
    }
    
    fn attack(&mut self, damage: u32) {
        if self.health > 0 {
            self.state = CharacterState::Attacking { damage };
        }
    }
    
    fn take_damage(&mut self, damage: i32) {
        self.health -= damage;
        if self.health <= 0 {
            self.health = 0;
            self.state = CharacterState::Dead;
        }
    }
    
    fn stop(&mut self) {
        if self.state != CharacterState::Dead {
            self.state = CharacterState::Idle;
        }
    }
}
```

</details>

## Challenges

### Challenge 1: Elevator System
Create an elevator state machine with states: Idle, MovingUp, MovingDown, DoorsOpen. Implement floor requests and door timers.

### Challenge 2: ATM Machine
Build an ATM with states: Idle, CardInserted, PinEntered, Transaction. Handle withdrawals, deposits, and balance checks.

### Challenge 3: Music Player
Create a music player with states: Stopped, Playing, Paused. Add playlist management and track navigation.

## Success Criteria
✅ All state machines implemented correctly  
✅ Proper state transitions  
✅ Error handling for invalid transitions  
✅ Pattern matching used effectively  
✅ At least one challenge completed

## Key Learnings
- Enums model states perfectly
- Pattern matching enforces handling all cases
- Enums can carry data (enum variants with fields)
- State machines prevent invalid states
- Type system catches logic errors at compile time

## Next Lab
Proceed to Lab 4.3: Option and Result Deep Dive!
