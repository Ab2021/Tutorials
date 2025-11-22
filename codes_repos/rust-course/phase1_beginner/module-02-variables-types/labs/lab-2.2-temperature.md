# Lab 2.2: Temperature Converter

## Objective
Build a temperature converter that converts between Celsius and Fahrenheit using functions and control flow.

## Project Setup

```bash
cargo new temperature_converter
cd temperature_converter
```

## Requirements

Create a program that:
1. Converts Celsius to Fahrenheit
2. Converts Fahrenheit to Celsius
3. Uses functions for each conversion
4. Displays results with proper formatting

## Formulas

- **Celsius to Fahrenheit:** F = (C × 9/5) + 32
- **Fahrenheit to Celsius:** C = (F - 32) × 5/9

## Starter Code

```rust
fn main() {
    println!("=== Temperature Converter ===\n");
    
    // Test conversions
    let celsius = 25.0;
    let fahrenheit = 77.0;
    
    // TODO: Call conversion functions and print results
}

// TODO: Implement celsius_to_fahrenheit function

// TODO: Implement fahrenheit_to_celsius function
```

## Step-by-Step Guide

### Step 1: Implement celsius_to_fahrenheit

```rust
fn celsius_to_fahrenheit(celsius: f64) -> f64 {
    // TODO: Implement the formula
}
```

### Step 2: Implement fahrenheit_to_celsius

```rust
fn fahrenheit_to_celsius(fahrenheit: f64) -> f64 {
    // TODO: Implement the formula
}
```

### Step 3: Use the Functions

```rust
fn main() {
    println!("=== Temperature Converter ===\n");
    
    let celsius = 25.0;
    let fahrenheit = 77.0;
    
    let c_to_f = celsius_to_fahrenheit(celsius);
    let f_to_c = fahrenheit_to_celsius(fahrenheit);
    
    println!("{}°C = {:.2}°F", celsius, c_to_f);
    println!("{}°F = {:.2}°C", fahrenheit, f_to_c);
}
```

## Expected Output

```
=== Temperature Converter ===

25°C = 77.00°F
77°F = 25.00°C
```

## Challenges

### Challenge 1: Temperature Table

Create a function that prints a conversion table:

```rust
fn print_conversion_table() {
    println!("\nCelsius to Fahrenheit Table:");
    println!("°C\t°F");
    println!("---\t---");
    
    // TODO: Print conversions for 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
}
```

### Challenge 2: Determine Temperature State

Create a function that describes the temperature:

```rust
fn describe_temperature(celsius: f64) -> &'static str {
    // Return:
    // "Freezing" if below 0
    // "Cold" if 0-15
    // "Moderate" if 15-25
    // "Warm" if 25-35
    // "Hot" if above 35
}
```

### Challenge 3: Interactive Version

Modify the program to ask the user which conversion they want:

```rust
// Note: This requires reading input, which we'll cover more later
// For now, just use hardcoded values with if/else to simulate choice
fn main() {
    let choice = 1;  // 1 for C to F, 2 for F to C
    let temperature = 25.0;
    
    if choice == 1 {
        // Convert C to F
    } else if choice == 2 {
        // Convert F to C
    } else {
        println!("Invalid choice");
    }
}
```

## Complete Solution

<details>
<summary>Click to reveal solution</summary>

```rust
fn main() {
    println!("=== Temperature Converter ===\n");
    
    // Test conversions
    let celsius = 25.0;
    let fahrenheit = 77.0;
    
    let c_to_f = celsius_to_fahrenheit(celsius);
    let f_to_c = fahrenheit_to_celsius(fahrenheit);
    
    println!("{}°C = {:.2}°F", celsius, c_to_f);
    println!("{}°F = {:.2}°C", fahrenheit, f_to_c);
    
    // More test cases
    println!("\nMore conversions:");
    println!("0°C = {:.2}°F", celsius_to_fahrenheit(0.0));
    println!("100°C = {:.2}°F", celsius_to_fahrenheit(100.0));
    println!("32°F = {:.2}°C", fahrenheit_to_celsius(32.0));
    println!("212°F = {:.2}°C", fahrenheit_to_celsius(212.0));
}

fn celsius_to_fahrenheit(celsius: f64) -> f64 {
    (celsius * 9.0 / 5.0) + 32.0
}

fn fahrenheit_to_celsius(fahrenheit: f64) -> f64 {
    (fahrenheit - 32.0) * 5.0 / 9.0
}
```

</details>

## Challenge Solutions

<details>
<summary>Challenge 1 Solution</summary>

```rust
fn print_conversion_table() {
    println!("\nCelsius to Fahrenheit Table:");
    println!("°C\t°F");
    println!("---\t------");
    
    for celsius in (0..=100).step_by(10) {
        let fahrenheit = celsius_to_fahrenheit(celsius as f64);
        println!("{}\t{:.1}", celsius, fahrenheit);
    }
}

fn main() {
    // ... previous code ...
    print_conversion_table();
}
```

</details>

<details>
<summary>Challenge 2 Solution</summary>

```rust
fn describe_temperature(celsius: f64) -> &'static str {
    if celsius < 0.0 {
        "Freezing"
    } else if celsius < 15.0 {
        "Cold"
    } else if celsius < 25.0 {
        "Moderate"
    } else if celsius < 35.0 {
        "Warm"
    } else {
        "Hot"
    }
}

fn main() {
    let temps = [−10.0, 5.0, 20.0, 30.0, 40.0];
    
    println!("\nTemperature Descriptions:");
    for temp in temps {
        println!("{}°C is {}", temp, describe_temperature(temp));
    }
}
```

</details>

<details>
<summary>Challenge 3 Solution</summary>

```rust
fn main() {
    println!("=== Temperature Converter ===\n");
    
    let choice = 1;  // 1 for C to F, 2 for F to C
    let temperature = 25.0;
    
    if choice == 1 {
        let result = celsius_to_fahrenheit(temperature);
        println!("{}°C = {:.2}°F", temperature, result);
    } else if choice == 2 {
        let result = fahrenheit_to_celsius(temperature);
        println!("{}°F = {:.2}°C", temperature, result);
    } else {
        println!("Invalid choice! Please enter 1 or 2.");
    }
}
```

</details>

## Success Criteria

✅ Both conversion functions work correctly  
✅ Results are formatted to 2 decimal places  
✅ Code is organized with clear function definitions  
✅ At least one challenge completed

## Key Learnings

- Writing functions with parameters and return types
- Using f64 for floating-point calculations
- String formatting with println!
- Basic control flow with if/else

## Next Steps

Proceed to Lab 2.3: Fibonacci Sequence Generator!
