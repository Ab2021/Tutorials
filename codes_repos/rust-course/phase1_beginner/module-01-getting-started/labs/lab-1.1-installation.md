# Lab 1.1: Installation and Setup Verification

## Objective
Verify that Rust is properly installed and all tools are working correctly.

## Instructions

### Step 1: Check Rust Installation

Open your terminal and run the following commands:

```bash
rustc --version
cargo --version
rustup --version
```

**Expected Output:**
```
rustc 1.x.x (hash date)
cargo 1.x.x (hash date)
rustup 1.x.x (hash date)
```

### Step 2: Check Installed Toolchains

```bash
rustup show
```

This should display your active toolchain and installed targets.

### Step 3: Create a Test File

Create a file named `hello.rs` with the following content:

```rust
fn main() {
    println!("Rust is installed correctly!");
}
```

### Step 4: Compile and Run

```bash
rustc hello.rs
./hello  # On Linux/macOS
# or
hello.exe  # On Windows
```

**Expected Output:**
```
Rust is installed correctly!
```

### Step 5: Test Cargo

```bash
cargo --help
```

Verify that the help message displays correctly.

## Success Criteria

✅ All version commands work  
✅ rustc compiles the test file  
✅ The compiled program runs successfully  
✅ Cargo help displays

## Troubleshooting

**Issue:** Command not found
- **Solution:** Restart your terminal or add Rust to your PATH manually

**Issue:** Permission denied
- **Solution:** On Linux/macOS, you may need to run `chmod +x hello`

## Next Steps

Once you've verified your installation, proceed to Lab 1.2!
