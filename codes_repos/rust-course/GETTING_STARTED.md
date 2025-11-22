# 🚀 Getting Started Guide

## Welcome to Rust!

This guide will help you get started with the Rust programming course.

## Step 1: Install Rust

### Linux/macOS
Open your terminal and run:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Follow the on-screen instructions. When installation is complete, restart your terminal.

### Windows
1. Download `rustup-init.exe` from [rustup.rs](https://rustup.rs/)
2. Run the installer
3. Follow the on-screen instructions
4. Restart your terminal/command prompt

### Verify Installation
```bash
rustc --version
cargo --version
```

You should see version numbers for both commands.

## Step 2: Set Up Your Editor

### Recommended: Visual Studio Code
1. Download [VS Code](https://code.visualstudio.com/)
2. Install the "rust-analyzer" extension
3. Install the "CodeLLDB" extension (for debugging)

### Alternative Editors
- **IntelliJ IDEA** with Rust plugin
- **Vim/Neovim** with rust.vim
- **Emacs** with rust-mode

## Step 3: Your First Rust Program

Create a new project:
```bash
cargo new hello_rust
cd hello_rust
```

Open `src/main.rs` and you'll see:
```rust
fn main() {
    println!("Hello, world!");
}
```

Run it:
```bash
cargo run
```

You should see:
```
   Compiling hello_rust v0.1.0
    Finished dev [unoptimized + debuginfo] target(s) in 0.50s
     Running `target/debug/hello_rust`
Hello, world!
```

## Step 4: Start the Course

1. **Read** [course.md](./course.md) for the complete curriculum
2. **Begin** with [Module 1](./module-01-getting-started/)
3. **Complete** all labs in order
4. **Practice** daily, even if just for 30 minutes

## Essential Commands

```bash
# Create new project
cargo new project_name

# Build project
cargo build

# Build and run
cargo run

# Check if code compiles (faster)
cargo check

# Build optimized version
cargo build --release

# Run tests
cargo test

# Generate documentation
cargo doc --open

# Update dependencies
cargo update
```

## Learning Tips

### 1. Embrace the Compiler
Rust's compiler provides excellent error messages. Read them carefully!

### 2. Don't Rush Ownership
Module 3 (Ownership) is the most important. Take your time to understand it.

### 3. Practice Every Day
Consistency beats intensity. 30 minutes daily is better than 5 hours once a week.

### 4. Use the Playground
Test small code snippets at [play.rust-lang.org](https://play.rust-lang.org/)

### 5. Join the Community
- [Rust Users Forum](https://users.rust-lang.org/)
- [Rust Discord](https://discord.gg/rust-lang)
- [r/rust](https://www.reddit.com/r/rust/)

## Common Beginner Mistakes

### 1. Fighting the Borrow Checker
❌ **Don't:** Get frustrated with borrowing errors  
✅ **Do:** Learn from them - they prevent bugs!

### 2. Using `clone()` Everywhere
❌ **Don't:** Clone to avoid ownership issues  
✅ **Do:** Understand references and borrowing

### 3. Ignoring Warnings
❌ **Don't:** Leave warnings in your code  
✅ **Do:** Fix all warnings - they often indicate issues

### 4. Not Reading Error Messages
❌ **Don't:** Just look at the line number  
✅ **Do:** Read the full error message and suggestions

## Troubleshooting

### "Command not found: cargo"
**Solution:** Restart your terminal or add Rust to your PATH manually

### "Permission denied" on Linux/macOS
**Solution:** Run `chmod +x` on the binary or check file permissions

### Compilation is slow
**Solution:** Use `cargo check` for faster feedback during development

### Can't find a crate
**Solution:** Make sure it's added to `Cargo.toml` under `[dependencies]`

## Next Steps

1. ✅ Install Rust
2. ✅ Set up your editor
3. ✅ Run your first program
4. 📚 Start [Module 1: Getting Started](./module-01-getting-started/)

## Resources

### Official
- [The Rust Book](https://doc.rust-lang.org/book/)
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/)
- [Standard Library Docs](https://doc.rust-lang.org/std/)

### Practice
- [Rustlings](https://github.com/rust-lang/rustlings) - Small exercises
- [Exercism](https://exercism.org/tracks/rust) - Coding challenges
- [Advent of Code](https://adventofcode.com/) - Annual coding event

### Videos
- [Rust Crash Course](https://www.youtube.com/watch?v=zF34dRivLOw)
- [No Boilerplate Rust](https://www.youtube.com/playlist?list=PLZaoyhMXgBzoM9bfb5pyUOT3zjnaDdSEP)

---

**Ready to start?** Head to [Module 1](./module-01-getting-started/) and begin your Rust journey! 🦀
