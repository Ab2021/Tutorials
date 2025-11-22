# Lab 10.1: Building a Multi-Threaded Web Scraper

## Objective
Apply concurrency concepts by building a real-world multi-threaded application.

## Setup
```bash
cargo new web_scraper
cd web_scraper
```

Add to `Cargo.toml`:
```toml
[dependencies]
reqwest = { version = "0.11", features = ["blocking"] }
```

## Part 1: Basic Threading

### Exercise 1: Simple Thread Creation
```rust
use std::thread;
use std::time::Duration;

fn main() {
    let handle = thread::spawn(|| {
        for i in 1..5 {
            println!("Thread: count {}", i);
            thread::sleep(Duration::from_millis(500));
        }
    });
    
    for i in 1..3 {
        println!("Main: count {}", i);
        thread::sleep(Duration::from_millis(500));
    }
    
    handle.join().unwrap();
}
```

### Exercise 2: Multiple Threads
```rust
use std::thread;

fn main() {
    let mut handles = vec![];
    
    for i in 0..5 {
        let handle = thread::spawn(move || {
            println!("Thread {} starting", i);
            thread::sleep(std::time::Duration::from_millis(100 * i));
            println!("Thread {} done", i);
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    println!("All threads completed");
}
```

## Part 2: Message Passing

### Exercise 3: Channel Communication
```rust
use std::sync::mpsc;
use std::thread;

fn main() {
    let (tx, rx) = mpsc::channel();
    
    thread::spawn(move || {
        let messages = vec!["hello", "from", "the", "thread"];
        
        for msg in messages {
            tx.send(msg).unwrap();
            thread::sleep(std::time::Duration::from_millis(100));
        }
    });
    
    for received in rx {
        println!("Got: {}", received);
    }
}
```

### Exercise 4: Multiple Producers
```rust
use std::sync::mpsc;
use std::thread;

fn main() {
    let (tx, rx) = mpsc::channel();
    
    for i in 0..3 {
        let tx_clone = tx.clone();
        thread::spawn(move || {
            tx_clone.send(format!("Message from thread {}", i)).unwrap();
        });
    }
    
    drop(tx); // Close the sending side
    
    for received in rx {
        println!("Got: {}", received);
    }
}
```

## Part 3: Shared State

### Exercise 5: Mutex for Shared Data
```rust
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];
    
    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            let mut num = counter.lock().unwrap();
            *num += 1;
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    println!("Result: {}", *counter.lock().unwrap());
}
```

## Part 4: URL Downloader (Simulated)

### Exercise 6: Concurrent URL Processing
```rust
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

struct UrlDownloader {
    results: Arc<Mutex<Vec<String>>>,
}

impl UrlDownloader {
    fn new() -> Self {
        UrlDownloader {
            results: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    fn download(&self, url: String) {
        let results = Arc::clone(&self.results);
        
        thread::spawn(move || {
            // Simulate download
            println!("Downloading: {}", url);
            thread::sleep(Duration::from_millis(500));
            
            let content = format!("Content from {}", url);
            results.lock().unwrap().push(content);
            println!("Completed: {}", url);
        });
    }
    
    fn get_results(&self) -> Vec<String> {
        self.results.lock().unwrap().clone()
    }
}

fn main() {
    let downloader = UrlDownloader::new();
    
    let urls = vec![
        "https://example.com/page1",
        "https://example.com/page2",
        "https://example.com/page3",
    ];
    
    for url in urls {
        downloader.download(url.to_string());
    }
    
    // Wait for downloads
    thread::sleep(Duration::from_secs(2));
    
    let results = downloader.get_results();
    println!("\nResults:");
    for result in results {
        println!("  {}", result);
    }
}
```

## Part 5: Thread Pool

### Exercise 7: Simple Thread Pool
```rust
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

type Job = Box<dyn FnOnce() + Send + 'static>;

struct ThreadPool {
    workers: Vec<Worker>,
    sender: mpsc::Sender<Job>,
}

struct Worker {
    id: usize,
    thread: thread::JoinHandle<()>,
}

impl ThreadPool {
    fn new(size: usize) -> ThreadPool {
        let (sender, receiver) = mpsc::channel();
        let receiver = Arc::new(Mutex::new(receiver));
        
        let mut workers = Vec::with_capacity(size);
        
        for id in 0..size {
            workers.push(Worker::new(id, Arc::clone(&receiver)));
        }
        
        ThreadPool { workers, sender }
    }
    
    fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let job = Box::new(f);
        self.sender.send(job).unwrap();
    }
}

impl Worker {
    fn new(id: usize, receiver: Arc<Mutex<mpsc::Receiver<Job>>>) -> Worker {
        let thread = thread::spawn(move || loop {
            let job = receiver.lock().unwrap().recv();
            
            match job {
                Ok(job) => {
                    println!("Worker {} executing job", id);
                    job();
                }
                Err(_) => break,
            }
        });
        
        Worker { id, thread }
    }
}

fn main() {
    let pool = ThreadPool::new(4);
    
    for i in 0..8 {
        pool.execute(move || {
            println!("Task {} running", i);
            thread::sleep(std::time::Duration::from_millis(500));
            println!("Task {} done", i);
        });
    }
    
    thread::sleep(std::time::Duration::from_secs(3));
}
```

## Success Criteria
✅ Create and manage threads  
✅ Use channels for message passing  
✅ Share state safely with Arc and Mutex  
✅ Implement a thread pool  
✅ Understand concurrency patterns

## Key Learnings
- Threads enable parallel execution
- Channels provide safe message passing
- Arc enables shared ownership across threads
- Mutex ensures exclusive access to shared data
- Thread pools manage worker threads efficiently
- Rust prevents data races at compile time

## Challenges
1. Add error handling to the thread pool
2. Implement graceful shutdown for workers
3. Create a real web scraper using reqwest
4. Add rate limiting to prevent overwhelming servers

## Next Steps
Congratulations! You've completed the advanced topics. Now work on final projects to solidify your Rust knowledge!
