# Module 03: Control Flow

## 🎯 Learning Objectives

- Master conditional expressions (if/else)
- Use loops effectively (loop, while, for)
- Understand pattern matching with match
- Apply control flow in real programs
- Write clean, idiomatic Rust code

---

## 📖 Theoretical Concepts

### 3.1 Conditional Expressions

#### if Expressions

```rust
let number = 7;

if number < 5 {
    println!("condition was true");
} else {
    println!("condition was false");
}
```

**Key Points:**
- Condition must be a `bool`
- No parentheses required around condition
- Curly braces are required

#### if in let Statements

```rust
let condition = true;
let number = if condition { 5 } else { 6 };
```

#### Multiple Conditions

```rust
let number = 6;

if number % 4 == 0 {
    println!("divisible by 4");
} else if number % 3 == 0 {
    println!("divisible by 3");
} else if number % 2 == 0 {
    println!("divisible by 2");
} else {
    println!("not divisible by 4, 3, or 2");
}
```

---

### 3.2 Loops

#### loop - Infinite Loop

```rust
loop {
    println!("again!");
    break;  // Exit the loop
}

// Return value from loop
let result = loop {
    counter += 1;
    if counter == 10 {
        break counter * 2;
    }
};
```

#### while - Conditional Loop

```rust
let mut number = 3;

while number != 0 {
    println!("{}!", number);
    number -= 1;
}
```

#### for - Iterate Over Collection

```rust
let a = [10, 20, 30, 40, 50];

for element in a {
    println!("the value is: {}", element);
}

// Range
for number in 1..4 {  // 1, 2, 3
    println!("{}", number);
}

// Inclusive range
for number in 1..=4 {  // 1, 2, 3, 4
    println!("{}", number);
}

// Reverse
for number in (1..4).rev() {
    println!("{}", number);
}
```

#### Loop Labels

```rust
'outer: loop {
    println!("Entered outer loop");
    
    'inner: loop {
        println!("Entered inner loop");
        break 'outer;  // Break outer loop
    }
    
    println!("This won't print");
}
```

---

### 3.3 Pattern Matching

#### match Expression

```rust
let number = 3;

match number {
    1 => println!("One!"),
    2 | 3 => println!("Two or Three!"),
    4..=10 => println!("Four through Ten!"),
    _ => println!("Something else!"),
}
```

**Key Features:**
- Must be exhaustive
- Can match ranges with `..=`
- Can match multiple patterns with `|`
- `_` is catch-all pattern

#### Matching with Values

```rust
let x = 5;

let description = match x {
    1 => "one",
    2 => "two",
    3 => "three",
    _ => "many",
};
```

#### Destructuring with match

```rust
let pair = (0, -2);

match pair {
    (0, y) => println!("First is 0, y is {}", y),
    (x, 0) => println!("x is {}, second is 0", x),
    _ => println!("No match"),
}
```

---

### 3.4 if let

Concise alternative to match for single pattern:

```rust
let some_value = Some(3);

// Using match
match some_value {
    Some(3) => println!("three"),
    _ => (),
}

// Using if let
if let Some(3) = some_value {
    println!("three");
}
```

---

### 3.5 while let

```rust
let mut stack = Vec::new();
stack.push(1);
stack.push(2);
stack.push(3);

while let Some(top) = stack.pop() {
    println!("{}", top);
}
```

---

## 🔑 Key Takeaways

1. **if** is an expression that returns a value
2. **loop** creates infinite loops; use **break** to exit
3. **while** loops while condition is true
4. **for** is best for iterating collections
5. **match** must be exhaustive
6. **if let** and **while let** for concise pattern matching

---

## ⏭️ Next Steps

Complete the 10 labs in this module:
1. Lab 3.1: if/else exercises
2. Lab 3.2: Loop variations
3. Lab 3.3: for loop patterns
4. Lab 3.4: Pattern matching basics
5. Lab 3.5: Advanced match
6. Lab 3.6: FizzBuzz variations
7. Lab 3.7: Number guessing game
8. Lab 3.8: Menu-driven program
9. Lab 3.9: Iterator patterns
10. Lab 3.10: Control flow project

Then proceed to Module 04: Ownership Basics
