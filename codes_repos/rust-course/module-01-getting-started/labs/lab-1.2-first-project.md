# Lab 1.2: Creating Your First Cargo Project

## Objective
Learn to create, build, and run a Cargo project.

## Instructions

### Step 1: Create a New Project

```bash
cargo new my_first_project
cd my_first_project
```

### Step 2: Explore the Project Structure

List the files created:
```bash
ls -la  # Linux/macOS
dir     # Windows
```

You should see:
- `Cargo.toml` - Project configuration
- `src/main.rs` - Main source file
- `.git/` - Git repository (optional)

### Step 3: Examine Cargo.toml

Open `Cargo.toml` and observe its contents:

```toml
[package]
name = "my_first_project"
version = "0.1.0"
edition = "2021"

[dependencies]
```

### Step 4: Examine src/main.rs

Open `src/main.rs`:

```rust
fn main() {
    println!("Hello, world!");
}
```

### Step 5: Build the Project

```bash
cargo build
```

Observe the output. A `target/` directory is created with the compiled binary.

### Step 6: Run the Project

```bash
cargo run
```

**Expected Output:**
```
   Compiling my_first_project v0.1.0
    Finished dev [unoptimized + debuginfo] target(s) in 0.50s
     Running `target/debug/my_first_project`
Hello, world!
```

### Step 7: Modify the Program

Edit `src/main.rs`:

```rust
fn main() {
    println!("Hello, world!");
    println!("My name is [Your Name]");
    println!("I'm learning Rust!");
}
```

Run again:
```bash
cargo run
```

### Step 8: Build for Release

```bash
cargo build --release
```

This creates an optimized binary in `target/release/`.

### Step 9: Check Without Building

```bash
cargo check
```

This verifies your code compiles without producing a binary (faster than `cargo build`).

## Exercises

1. **Exercise A:** Create a new project called `greetings` that prints 5 different greeting messages.

2. **Exercise B:** Modify your project to print your name in ASCII art using println!

3. **Exercise C:** Create a project that prints the numbers 1 through 10, each on a new line.

## Success Criteria

✅ Successfully created a Cargo project  
✅ Built and ran the project  
✅ Modified the code and saw changes  
✅ Understand the difference between debug and release builds  
✅ Completed all exercises

## Solution for Exercise A

```rust
fn main() {
    println!("Hello, friend!");
    println!("Good morning!");
    println!("Welcome to Rust!");
    println!("Greetings, programmer!");
    println!("Nice to meet you!");
}
```

## Next Steps

Proceed to Lab 1.3 to build a simple calculator!
