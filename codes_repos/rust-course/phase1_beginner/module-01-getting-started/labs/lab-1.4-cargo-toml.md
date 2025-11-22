# Lab 1.4: Understanding Cargo.toml

## Objective
Learn to configure and customize Cargo.toml for your projects.

## Instructions

### Step 1: Create a New Project
```bash
cargo new cargo_config_demo
cd cargo_config_demo
```

### Step 2: Examine Cargo.toml
Open `Cargo.toml` and observe the default structure:
```toml
[package]
name = "cargo_config_demo"
version = "0.1.0"
edition = "2021"

[dependencies]
```

### Step 3: Add Metadata
Enhance your Cargo.toml with additional metadata:
```toml
[package]
name = "cargo_config_demo"
version = "0.1.0"
edition = "2021"
authors = ["Your Name <you@example.com>"]
description = "A demo project for learning Cargo configuration"
license = "MIT"
repository = "https://github.com/yourusername/cargo_config_demo"
keywords = ["demo", "learning"]
categories = ["development-tools"]

[dependencies]
```

### Step 4: Add Dependencies
Add some common dependencies:
```toml
[dependencies]
serde = "1.0"
serde_json = "1.0"
```

### Step 5: Configure Build Profiles
Add custom build profiles:
```toml
[profile.dev]
opt-level = 0

[profile.release]
opt-level = 3
lto = true
```

### Step 6: Test the Configuration
```bash
cargo build
cargo build --release
```

## Exercises

### Exercise 1: Add More Dependencies
Add these dependencies to your Cargo.toml:
- `rand = "0.8"`
- `chrono = "0.4"`

### Exercise 2: Create a Custom Profile
Create a `[profile.test]` section with:
- `opt-level = 1`
- `debug = true`

### Exercise 3: Workspace Configuration
Create a workspace with multiple packages.

## Success Criteria
✅ Cargo.toml has complete metadata  
✅ Dependencies are added correctly  
✅ Custom profiles work  
✅ Project builds successfully

## Next Steps
Proceed to Lab 1.5: Working with External Crates
