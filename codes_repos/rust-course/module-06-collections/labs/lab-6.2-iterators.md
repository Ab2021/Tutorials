# Lab 6.2: Text Analysis with Iterators

## Objective
Master iterators and functional programming patterns by building text analysis tools.

## Setup
```bash
cargo new text_analysis
cd text_analysis
```

## Part 1: Iterator Basics

### Exercise 1: Basic Iterator Operations
```rust
fn main() {
    let numbers = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    
    // Sum
    let sum: i32 = numbers.iter().sum();
    println!("Sum: {}", sum);
    
    // Product
    let product: i32 = numbers.iter().product();
    println!("Product: {}", product);
    
    // Max and Min
    let max = numbers.iter().max();
    let min = numbers.iter().min();
    println!("Max: {:?}, Min: {:?}", max, min);
    
    // Count
    let count = numbers.iter().count();
    println!("Count: {}", count);
}
```

### Exercise 2: Map and Filter
```rust
fn main() {
    let numbers = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    
    // Square all numbers
    let squares: Vec<i32> = numbers.iter()
        .map(|x| x * x)
        .collect();
    println!("Squares: {:?}", squares);
    
    // Filter even numbers
    let evens: Vec<&i32> = numbers.iter()
        .filter(|&&x| x % 2 == 0)
        .collect();
    println!("Evens: {:?}", evens);
    
    // Chain operations
    let result: Vec<i32> = numbers.iter()
        .filter(|&&x| x % 2 == 0)
        .map(|x| x * x)
        .collect();
    println!("Even squares: {:?}", result);
}
```

## Part 2: String Iterators

### Exercise 3: Word Processing
```rust
fn count_words(text: &str) -> usize {
    text.split_whitespace().count()
}

fn longest_word(text: &str) -> Option<&str> {
    text.split_whitespace()
        .max_by_key(|word| word.len())
}

fn words_longer_than(text: &str, min_length: usize) -> Vec<&str> {
    text.split_whitespace()
        .filter(|word| word.len() > min_length)
        .collect()
}

fn main() {
    let text = "The quick brown fox jumps over the lazy dog";
    
    println!("Word count: {}", count_words(text));
    println!("Longest word: {:?}", longest_word(text));
    println!("Words longer than 4: {:?}", words_longer_than(text, 4));
}
```

### Exercise 4: Character Analysis
```rust
fn count_vowels(text: &str) -> usize {
    text.chars()
        .filter(|c| matches!(c.to_lowercase().next(), Some('a' | 'e' | 'i' | 'o' | 'u')))
        .count()
}

fn count_consonants(text: &str) -> usize {
    text.chars()
        .filter(|c| c.is_alphabetic() && !matches!(c.to_lowercase().next(), Some('a' | 'e' | 'i' | 'o' | 'u')))
        .count()
}

fn char_frequency(text: &str) -> std::collections::HashMap<char, usize> {
    use std::collections::HashMap;
    
    text.chars()
        .filter(|c| c.is_alphabetic())
        .map(|c| c.to_lowercase().next().unwrap())
        .fold(HashMap::new(), |mut map, c| {
            *map.entry(c).or_insert(0) += 1;
            map
        })
}

fn main() {
    let text = "Hello, World!";
    
    println!("Vowels: {}", count_vowels(text));
    println!("Consonants: {}", count_consonants(text));
    println!("Character frequency: {:?}", char_frequency(text));
}
```

## Part 3: Advanced Iterator Patterns

### Exercise 5: Chaining and Collecting
```rust
fn process_numbers(numbers: Vec<i32>) -> Vec<i32> {
    numbers.iter()
        .filter(|&&x| x > 0)           // Only positive
        .map(|x| x * 2)                 // Double them
        .filter(|&x| x < 100)           // Less than 100
        .collect()
}

fn group_by_length(words: Vec<&str>) -> std::collections::HashMap<usize, Vec<&str>> {
    use std::collections::HashMap;
    
    words.into_iter()
        .fold(HashMap::new(), |mut map, word| {
            map.entry(word.len()).or_insert_with(Vec::new).push(word);
            map
        })
}

fn main() {
    let numbers = vec![-5, 10, 15, -20, 30, 100, 45];
    println!("Processed: {:?}", process_numbers(numbers));
    
    let words = vec!["hi", "hello", "hey", "goodbye", "bye"];
    println!("Grouped: {:?}", group_by_length(words));
}
```

### Exercise 6: take, skip, and enumerate
```rust
fn main() {
    let numbers: Vec<i32> = (1..=20).collect();
    
    // First 5 numbers
    let first_five: Vec<i32> = numbers.iter().take(5).copied().collect();
    println!("First 5: {:?}", first_five);
    
    // Skip first 10
    let after_ten: Vec<i32> = numbers.iter().skip(10).copied().collect();
    println!("After 10: {:?}", after_ten);
    
    // Enumerate with index
    numbers.iter()
        .enumerate()
        .filter(|(i, _)| i % 2 == 0)
        .for_each(|(i, n)| println!("Index {}: {}", i, n));
}
```

## Part 4: Real-World Applications

### Exercise 7: Log File Parser
```rust
struct LogEntry {
    level: String,
    message: String,
}

fn parse_log_line(line: &str) -> Option<LogEntry> {
    let parts: Vec<&str> = line.splitn(2, ':').collect();
    if parts.len() == 2 {
        Some(LogEntry {
            level: parts[0].trim().to_string(),
            message: parts[1].trim().to_string(),
        })
    } else {
        None
    }
}

fn analyze_logs(logs: &str) -> (usize, usize, usize) {
    let entries: Vec<LogEntry> = logs.lines()
        .filter_map(parse_log_line)
        .collect();
    
    let errors = entries.iter().filter(|e| e.level == "ERROR").count();
    let warnings = entries.iter().filter(|e| e.level == "WARN").count();
    let info = entries.iter().filter(|e| e.level == "INFO").count();
    
    (errors, warnings, info)
}

fn main() {
    let logs = "ERROR: Database connection failed\n\
                INFO: Server started\n\
                WARN: High memory usage\n\
                ERROR: Timeout occurred\n\
                INFO: Request processed";
    
    let (errors, warnings, info) = analyze_logs(logs);
    println!("Errors: {}, Warnings: {}, Info: {}", errors, warnings, info);
}
```

### Exercise 8: CSV Data Processing
```rust
#[derive(Debug)]
struct Person {
    name: String,
    age: u32,
    city: String,
}

fn parse_csv(csv: &str) -> Vec<Person> {
    csv.lines()
        .skip(1)  // Skip header
        .filter_map(|line| {
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() == 3 {
                Some(Person {
                    name: parts[0].trim().to_string(),
                    age: parts[1].trim().parse().ok()?,
                    city: parts[2].trim().to_string(),
                })
            } else {
                None
            }
        })
        .collect()
}

fn average_age(people: &[Person]) -> f64 {
    let sum: u32 = people.iter().map(|p| p.age).sum();
    sum as f64 / people.len() as f64
}

fn people_in_city<'a>(people: &'a [Person], city: &str) -> Vec<&'a Person> {
    people.iter()
        .filter(|p| p.city == city)
        .collect()
}

fn main() {
    let csv = "Name,Age,City\n\
               Alice,30,NYC\n\
               Bob,25,LA\n\
               Charlie,35,NYC\n\
               Diana,28,Chicago";
    
    let people = parse_csv(csv);
    println!("People: {:?}", people);
    println!("Average age: {:.1}", average_age(&people));
    println!("People in NYC: {:?}", people_in_city(&people, "NYC"));
}
```

## Part 5: Custom Iterators

### Exercise 9: Fibonacci Iterator
```rust
struct Fibonacci {
    curr: u64,
    next: u64,
}

impl Fibonacci {
    fn new() -> Self {
        Fibonacci { curr: 0, next: 1 }
    }
}

impl Iterator for Fibonacci {
    type Item = u64;
    
    fn next(&mut self) -> Option<Self::Item> {
        let current = self.curr;
        self.curr = self.next;
        self.next = current + self.next;
        Some(current)
    }
}

fn main() {
    let fibs: Vec<u64> = Fibonacci::new().take(10).collect();
    println!("First 10 Fibonacci numbers: {:?}", fibs);
    
    let sum: u64 = Fibonacci::new()
        .take_while(|&x| x < 1000)
        .filter(|&x| x % 2 == 0)
        .sum();
    println!("Sum of even Fibonacci numbers < 1000: {}", sum);
}
```

### Exercise 10: Range with Step
```rust
struct StepRange {
    current: i32,
    end: i32,
    step: i32,
}

impl StepRange {
    fn new(start: i32, end: i32, step: i32) -> Self {
        StepRange {
            current: start,
            end,
            step,
        }
    }
}

impl Iterator for StepRange {
    type Item = i32;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.end {
            let result = self.current;
            self.current += self.step;
            Some(result)
        } else {
            None
        }
    }
}

fn main() {
    let range: Vec<i32> = StepRange::new(0, 20, 3).collect();
    println!("Range with step 3: {:?}", range);
}
```

## Success Criteria
✅ Understand iterator methods (map, filter, fold)  
✅ Chain multiple iterator operations  
✅ Use collect() to gather results  
✅ Implement custom iterators  
✅ Apply functional programming patterns  
✅ Process real-world data with iterators

## Key Learnings
- Iterators are lazy - they don't compute until consumed
- Chaining creates efficient pipelines
- collect() is versatile - can create many collection types
- Custom iterators implement the Iterator trait
- Functional style often more readable than loops
- Zero-cost abstractions - as fast as hand-written loops

## Next Lab
Proceed to Lab 6.3: Custom Iterator Implementation!
