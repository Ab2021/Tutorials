# Module 1: Getting Started with Rust

## 🎯 Learning Objectives

By the end of this module, you will:
- Understand what Rust is and its key advantages
- Install Rust and set up your development environment
- Use Cargo to create and manage Rust projects
- Write, compile, and run your first Rust programs
- Understand basic project structure

---

## 📖 Theoretical Concepts

### 1.1 What is Rust?

Rust is a systems programming language that focuses on three key goals:
- **Safety**: Memory safety without garbage collection
- **Speed**: Zero-cost abstractions and performance comparable to C/C++
- **Concurrency**: Fearless concurrency with compile-time guarantees

#### Why Rust?

**Memory Safety**
- No null pointer dereferences
- No dangling pointers
- No data races
- Prevents common bugs at compile time

**Performance**
- No garbage collector overhead
- Zero-cost abstractions
- Minimal runtime
- Direct hardware access when needed

**Modern Tooling**
- Cargo: built-in package manager and build system
- Rustfmt: automatic code formatting
- Clippy: advanced linting
- Excellent error messages

**Use Cases**
- Operating systems
- Web browsers (Firefox uses Rust)
- Game engines
- Command-line tools
- Web servers
- Embedded systems
- Blockchain and cryptocurrency

---

### 1.2 Installing Rust

#### On Linux/macOS:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

#### On Windows:
Download and run `rustup-init.exe` from https://rustup.rs/

#### Verify Installation:
```bash
rustc --version
cargo --version
rustdoc --version
```

#### Updating Rust:
```bash
rustup update
```

#### Rust Components:
- **rustc**: The Rust compiler
- **cargo**: Package manager and build tool
- **rustup**: Toolchain installer and version manager
- **rustdoc**: Documentation generator

---

### 1.3 Understanding Cargo

Cargo is Rust's build system and package manager. It handles:
- Building your code
- Downloading dependencies (called "crates")
- Building dependencies
- Running tests
- Generating documentation

#### Key Cargo Commands:

```bash
# Create a new project
cargo new project_name

# Create a new library
cargo new --lib library_name

# Build the project
cargo build

# Build with optimizations (release mode)
cargo build --release

# Build and run
cargo run

# Check if code compiles (faster than build)
cargo check

# Run tests
cargo test

# Generate documentation
cargo doc --open

# Update dependencies
cargo update
```

---

### 1.4 Project Structure

When you create a new Cargo project, you get this structure:

```
my_project/
├── Cargo.toml          # Project manifest
├── Cargo.lock          # Dependency lock file (auto-generated)
├── src/
│   └── main.rs         # Main source file
└── target/             # Build artifacts (auto-generated)
```

#### Cargo.toml Explained:

```toml
[package]
name = "my_project"      # Project name
version = "0.1.0"        # Version number
edition = "2021"         # Rust edition

[dependencies]
# External crates go here
# Example: serde = "1.0"
```

---

### 1.5 Your First Rust Program

#### Hello, World!

```rust
fn main() {
    println!("Hello, world!");
}
```

**Breakdown:**
- `fn main()`: Defines the main function (entry point)
- `println!`: A macro (note the `!`) that prints to stdout
- Statements end with semicolons
- Code blocks use curly braces `{}`

#### Compiling Directly with rustc:

```bash
rustc main.rs
./main
```

#### Using Cargo (Recommended):

```bash
cargo new hello_world
cd hello_world
cargo run
```

---

### 1.6 Basic Program Structure

```rust
// This is a single-line comment

/*
 * This is a multi-line comment
 */

// Main function - program entry point
fn main() {
    // Print to console
    println!("Hello, Rust!");
    
    // Variables
    let x = 5;
    println!("The value of x is: {}", x);
    
    // Function call
    greet("Alice");
}

// Function definition
fn greet(name: &str) {
    println!("Hello, {}!", name);
}
```

---

### 1.7 Compilation Process

1. **Source Code** (.rs files)
2. **Compiler** (rustc) performs:
   - Lexical analysis
   - Parsing
   - Semantic analysis
   - Borrow checking
   - Optimization
3. **Binary Output** (executable)

Rust is **ahead-of-time compiled**, meaning:
- Compilation happens before execution
- No runtime interpreter needed
- Fast execution speed
- Platform-specific binaries

---

### 1.8 Rust Editions

Rust uses "editions" for backwards-compatible improvements:
- **2015**: Original Rust 1.0
- **2018**: Improved module system, async/await foundation
- **2021**: Current edition with various improvements

You can specify the edition in `Cargo.toml`:
```toml
edition = "2021"
```

---

## 🔑 Key Takeaways

1. Rust provides memory safety without garbage collection
2. Cargo is your primary tool for managing Rust projects
3. `cargo new` creates a new project with proper structure
4. `cargo run` builds and executes your program
5. The `main()` function is the entry point
6. Rust uses ahead-of-time compilation for performance

---

## 📚 Additional Resources

- [The Rust Book - Chapter 1](https://doc.rust-lang.org/book/ch01-00-getting-started.html)
- [Cargo Book](https://doc.rust-lang.org/cargo/)
- [Rust Playground](https://play.rust-lang.org/) - Try Rust in your browser

---

## ⏭️ Next Steps

Proceed to the labs directory to practice:
1. Lab 1.1: Installation verification
2. Lab 1.2: Creating your first Cargo project
3. Lab 1.3: Building a simple calculator

After completing the labs, move on to **Module 2: Basic Syntax and Data Types**.
