# Lab 5.1: File Operations with Error Handling

## Objective
Master error handling in real-world scenarios by building file manipulation tools with proper Result usage and custom errors.

## Setup
```bash
cargo new file_operations
cd file_operations
```

## Part 1: Reading Files

### Exercise 1: Basic File Reading
```rust
use std::fs;
use std::io;

fn read_file(path: &str) -> Result<String, io::Error> {
    fs::read_to_string(path)
}

fn main() -> Result<(), io::Error> {
    let contents = read_file("test.txt")?;
    println!("File contents:\n{}", contents);
    Ok(())
}
```

### Exercise 2: Reading with Better Errors
```rust
use std::fs::File;
use std::io::{self, Read};

fn read_file_verbose(path: &str) -> Result<String, String> {
    let mut file = File::open(path)
        .map_err(|e| format!("Failed to open '{}': {}", path, e))?;
    
    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .map_err(|e| format!("Failed to read '{}': {}", path, e))?;
    
    Ok(contents)
}

fn main() {
    match read_file_verbose("test.txt") {
        Ok(contents) => println!("Success:\n{}", contents),
        Err(e) => eprintln!("Error: {}", e),
    }
}
```

## Part 2: Writing Files

### Exercise 3: Safe File Writing
```rust
use std::fs;
use std::io;

fn write_file(path: &str, contents: &str) -> Result<(), io::Error> {
    fs::write(path, contents)
}

fn append_to_file(path: &str, contents: &str) -> Result<(), io::Error> {
    use std::fs::OpenOptions;
    use std::io::Write;
    
    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(path)?;
    
    writeln!(file, "{}", contents)?;
    Ok(())
}

fn main() -> Result<(), io::Error> {
    write_file("output.txt", "Hello, Rust!")?;
    append_to_file("output.txt", "Appended line")?;
    
    let contents = fs::read_to_string("output.txt")?;
    println!("File contents:\n{}", contents);
    
    Ok(())
}
```

## Part 3: File Utilities

### Exercise 4: File Copy with Progress
```rust
use std::fs::File;
use std::io::{self, Read, Write};

fn copy_file(source: &str, dest: &str) -> Result<usize, io::Error> {
    let mut source_file = File::open(source)?;
    let mut dest_file = File::create(dest)?;
    
    let mut buffer = [0; 1024];
    let mut total_bytes = 0;
    
    loop {
        let bytes_read = source_file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        
        dest_file.write_all(&buffer[..bytes_read])?;
        total_bytes += bytes_read;
    }
    
    Ok(total_bytes)
}

fn main() -> Result<(), io::Error> {
    let bytes = copy_file("source.txt", "destination.txt")?;
    println!("Copied {} bytes", bytes);
    Ok(())
}
```

### Exercise 5: Directory Operations
```rust
use std::fs;
use std::io;
use std::path::Path;

fn list_files(dir: &str) -> Result<Vec<String>, io::Error> {
    let mut files = Vec::new();
    
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if let Some(name) = path.file_name() {
            files.push(name.to_string_lossy().to_string());
        }
    }
    
    Ok(files)
}

fn create_directory(path: &str) -> Result<(), io::Error> {
    fs::create_dir_all(path)
}

fn file_exists(path: &str) -> bool {
    Path::new(path).exists()
}

fn main() -> Result<(), io::Error> {
    create_directory("test_dir")?;
    
    if file_exists("test_dir") {
        println!("Directory created successfully");
    }
    
    let files = list_files(".")?;
    println!("Files in current directory:");
    for file in files {
        println!("  - {}", file);
    }
    
    Ok(())
}
```

## Part 4: Advanced Error Handling

### Exercise 6: Custom Error Type
```rust
use std::fmt;
use std::io;

#[derive(Debug)]
enum FileError {
    NotFound(String),
    PermissionDenied(String),
    InvalidFormat(String),
    IoError(io::Error),
}

impl fmt::Display for FileError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FileError::NotFound(path) => write!(f, "File not found: {}", path),
            FileError::PermissionDenied(path) => write!(f, "Permission denied: {}", path),
            FileError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            FileError::IoError(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl From<io::Error> for FileError {
    fn from(error: io::Error) -> Self {
        match error.kind() {
            io::ErrorKind::NotFound => FileError::NotFound(error.to_string()),
            io::ErrorKind::PermissionDenied => FileError::PermissionDenied(error.to_string()),
            _ => FileError::IoError(error),
        }
    }
}

fn read_config(path: &str) -> Result<String, FileError> {
    let contents = std::fs::read_to_string(path)?;
    
    if contents.is_empty() {
        return Err(FileError::InvalidFormat(String::from("File is empty")));
    }
    
    Ok(contents)
}

fn main() {
    match read_config("config.txt") {
        Ok(config) => println!("Config: {}", config),
        Err(e) => eprintln!("Error: {}", e),
    }
}
```

## Part 5: Complete Project - Log File Analyzer

### Exercise 7: Build a Log Analyzer
```rust
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::collections::HashMap;

struct LogAnalyzer {
    error_count: usize,
    warning_count: usize,
    info_count: usize,
    errors_by_type: HashMap<String, usize>,
}

impl LogAnalyzer {
    fn new() -> Self {
        LogAnalyzer {
            error_count: 0,
            warning_count: 0,
            info_count: 0,
            errors_by_type: HashMap::new(),
        }
    }
    
    fn analyze_file(&mut self, path: &str) -> Result<(), io::Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        
        for line in reader.lines() {
            let line = line?;
            self.process_line(&line);
        }
        
        Ok(())
    }
    
    fn process_line(&mut self, line: &str) {
        if line.contains("ERROR") {
            self.error_count += 1;
            if let Some(error_type) = self.extract_error_type(line) {
                *self.errors_by_type.entry(error_type).or_insert(0) += 1;
            }
        } else if line.contains("WARN") {
            self.warning_count += 1;
        } else if line.contains("INFO") {
            self.info_count += 1;
        }
    }
    
    fn extract_error_type(&self, line: &str) -> Option<String> {
        // Simple extraction - find text after "ERROR:"
        line.split("ERROR:")
            .nth(1)
            .map(|s| s.trim().split_whitespace().next().unwrap_or("Unknown").to_string())
    }
    
    fn print_report(&self) {
        println!("=== Log Analysis Report ===");
        println!("Total Errors: {}", self.error_count);
        println!("Total Warnings: {}", self.warning_count);
        println!("Total Info: {}", self.info_count);
        
        if !self.errors_by_type.is_empty() {
            println!("\nErrors by type:");
            let mut errors: Vec<_> = self.errors_by_type.iter().collect();
            errors.sort_by(|a, b| b.1.cmp(a.1));
            
            for (error_type, count) in errors {
                println!("  {}: {}", error_type, count);
            }
        }
    }
}

fn main() -> Result<(), io::Error> {
    let mut analyzer = LogAnalyzer::new();
    analyzer.analyze_file("application.log")?;
    analyzer.print_report();
    Ok(())
}
```

## Challenges

### Challenge 1: File Backup Tool
Create a tool that backs up files with timestamps and handles errors gracefully.

### Challenge 2: CSV Parser
Build a CSV file parser that validates data and reports errors with line numbers.

### Challenge 3: Configuration Manager
Create a configuration file manager that reads, validates, and writes config files with proper error handling.

## Success Criteria
✅ Handle file I/O errors properly  
✅ Use custom error types  
✅ Implement From trait for error conversion  
✅ Use ? operator effectively  
✅ Provide helpful error messages  
✅ Complete log analyzer project

## Key Learnings
- Always handle file I/O errors
- Use Result<T, E> for fallible operations
- Custom error types provide better context
- ? operator simplifies error propagation
- BufReader for efficient file reading
- From trait for error type conversion

## Next Lab
Proceed to Lab 5.2: Custom Error Types and Error Chains!
