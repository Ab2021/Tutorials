# Day 171: Mock Interview (Technical)
## Phase 4: ADAS & Robotics Systems | Week 25: Final Assessment & Career

---

> **📝 Day 171 Focus:**
> You know the tech. Now you have to explain it to a skeptic with a marker. Today, we simulate the **Technical Interview**. We cover Algorithms, System Design, and the dreaded "Tell me about a time you failed."

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Solve** common Robotics coding problems (e.g., Grid Search, Kalman Filter implementation).
2.  **Design** an ADAS feature (e.g., Adaptive Cruise Control) on a whiteboard.
3.  **Apply** the STAR method to Behavioral questions.
4.  **Debug** code on paper/whiteboard.
5.  **Ask** insightful questions to the interviewer.

---

## 📚 The Interview Gauntlet

### 🔹 Round 1: Coding (Algorithms)

**Expectation:** Write clean, compilable C++/Python code. Optimize Time/Space complexity.

**Problem 1: The Sliding Window**
*   **Q:** Given a stream of Lidar points, find the closest point in the last $N$ milliseconds.
*   **A:** Use a `std::deque` or `collections.deque`. Push new, Pop old. Maintain a Min-Heap if $N$ is large.

**Problem 2: Grid Search**
*   **Q:** Implement A* on a 2D grid.
*   **A:** Priority Queue (`heapq`). Heuristic (Manhattan/Euclidean). `visited` set. Reconstruct path.

**Problem 3: Matrix Math**
*   **Q:** Rotate a 3D point cloud by $\theta$ around the Z-axis.
*   **A:** Multiply by Rotation Matrix $R_z$.
    $$ \begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix} $$

### 🔹 Round 2: System Design

**Expectation:** High-level architecture. Trade-offs. Interfaces.

**Design Challenge: Traffic Light Detection**
*   **Interviewer:** "Design a system to detect traffic lights for an autonomous taxi."
*   **You:**
    1.  **Clarify:** Range? Latency? Day/Night? Map availability?
    2.  **Sensors:** Camera (Telephoto for range). Map (Prior location).
    3.  **Pipeline:**
        *   Map ROI $\to$ Crop Image $\to$ CNN Classifier (Red/Green/Yellow) $\to$ Temporal Filter (Debounce).
    4.  **Edge Cases:** Flashing yellow? Sun glare?
    5.  **Safety:** What if confidence is low? (Slow down / Handover).

### 🔹 Round 3: Behavioral (The Culture Fit)

**Expectation:** Communication, Teamwork, Ownership.

**Method: STAR (Situation, Task, Action, Result)**

*   **Q:** "Tell me about a difficult bug you fixed."
*   **S:** "In the Capstone Project, the car kept hitting the curb during parking."
*   **T:** "I had to identify if it was a Perception or Control error."
*   **A:** "I logged the data. I saw the Planner path was correct, but the Controller lagged. I tuned the MPC weights and added a feedforward friction term."
*   **R:** "The parking accuracy improved from 10cm to 3cm, and we passed the demo."

---

## 💻 Implementation: The Coding Challenge

**Scenario:**
-   Write a C++ class for a **Ring Buffer** (Circular Buffer) to store sensor data.
-   Must be thread-safe.

### 🛠️ Setup
Create `week25_review/interview.cpp`.

### 👨‍💻 Code: Thread-Safe Ring Buffer

```cpp
#include <vector>
#include <mutex>
#include <iostream>
#include <optional>

template <typename T>
class RingBuffer {
public:
    explicit RingBuffer(size_t size) : size_(size), head_(0), tail_(0), full_(false) {
        buffer_.resize(size);
    }

    void push(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        buffer_[head_] = item;
        
        if (full_) {
            tail_ = (tail_ + 1) % size_; // Overwrite oldest
        }
        
        head_ = (head_ + 1) % size_;
        full_ = head_ == tail_;
    }

    std::optional<T> pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (empty()) {
            return std::nullopt;
        }
        
        T item = buffer_[tail_];
        full_ = false;
        tail_ = (tail_ + 1) % size_;
        
        return item;
    }
    
    bool empty() const {
        return (!full_ && (head_ == tail_));
    }

private:
    std::vector<T> buffer_;
    size_t size_;
    size_t head_;
    size_t tail_;
    bool full_;
    mutable std::mutex mutex_;
};

int main() {
    RingBuffer<int> rb(3);
    
    rb.push(1);
    rb.push(2);
    rb.push(3);
    rb.push(4); // Overwrites 1
    
    auto val = rb.pop(); // Should be 2
    if (val) std::cout << "Popped: " << *val << std::endl;
    
    return 0;
}
```

---

## 🔬 Lab Exercise: Mock Interview

### Lab Objectives
1.  **Self-Test:**
    -   Take a piece of paper.
    -   Write the "Traffic Light Detection" architecture block diagram.
    -   Time yourself: 10 minutes.
2.  **Code Review:**
    -   Look at the Ring Buffer code above.
    -   What happens if `T` is a heavy object? (Copy overhead).
    -   *Improvement:* Use `std::move` in `push`.

---

## 🐞 Debugging & Troubleshooting

### Common Interview Mistakes

#### 1. Jumping to Code
**Mistake:** Writing code immediately.
**Fix:** **Think Aloud**. "I plan to use a Hash Map to store the counts. This gives O(1) access." Get buy-in from the interviewer first.

#### 2. Ignoring Edge Cases
**Mistake:** Assuming valid input.
**Fix:** "What if the input array is empty? What if the Lidar returns NaNs?"

#### 3. Silent Solving
**Mistake:** Coding in silence for 20 minutes.
**Fix:** Keep talking. "I'm implementing the BFS now. I'll use a queue."

---

## ⚡ Optimization & Best Practices

### 1. Big O Notation
-   Know the complexity of everything you write.
-   Sorting: $O(N \log N)$.
-   Hash Map: $O(1)$ avg.
-   Tree Search: $O(b^d)$.

### 2. System Design Patterns
-   **Pub/Sub:** Decoupling.
-   **Service:** Request/Response.
-   **Blackboard:** Shared state for AI.
-   **Watchdog:** Safety monitoring.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the "Diamond Problem" in C++ inheritance?
    *   **A:** Multiple inheritance where two base classes inherit from the same grandparent. Solved with `virtual inheritance`.
2.  **Q:** Explain "Volatile" keyword.
    *   **A:** Tells the compiler not to optimize reads/writes to this variable (e.g., Memory Mapped I/O).
3.  **Q:** How do you handle a slow consumer in a Pub/Sub system?
    *   **A:** Queue size (drop old messages) or separate thread.

### Challenge Task
**Task:** LeetCode Hard.
1.  Solve "Trapping Rain Water" (2D Histogram).
2.  Why is this relevant? It's similar to processing a 2D Occupancy Grid or Height Map.

---

## 📚 Further Reading & References
-   [Cracking the Coding Interview](https://www.crackingthecodinginterview.com/)
-   [Grokking the System Design Interview](https://www.educative.io/courses/grokking-the-system-design-interview)

---

**Day 171 Complete** | Phase 4: ADAS & Robotics Systems | Week 25: Final Assessment & Career
