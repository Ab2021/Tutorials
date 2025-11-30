# Day 106: Advanced Pointers & Memory Management
## Phase 1: Core Embedded Engineering Foundations | Week 16: Advanced C & Optimization

---

> **📝 Content Creator Instructions:**
> This document is designed to produce **comprehensive, industry-grade educational content**. 
> - **Target Length:** The final filled document should be approximately **1000+ lines** of detailed markdown.
> - **Depth:** Do not skim over details. Explain *why*, not just *how*.
> - **Structure:** If a topic is complex, **DIVIDE IT INTO MULTIPLE PARTS** (Part 1, Part 2, etc.).
> - **Code:** Provide complete, compilable code examples, not just snippets.
> - **Visuals:** Use Mermaid diagrams for flows, architectures, and state machines.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Master** complex pointer declarations (e.g., arrays of function pointers).
2.  **Implement** a custom memory allocator (Pool Allocator) to avoid fragmentation.
3.  **Analyze** the memory layout (Text, Data, BSS, Heap, Stack) in detail.
4.  **Debug** memory leaks and buffer overflows using custom tracking.
5.  **Utilize** `restrict` and `volatile` keywords correctly in pointer contexts.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
*   **Prior Knowledge:**
    *   Day 3 (Pointers Basic)
    *   Day 10 (Memory Architecture)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The "Right-Left" Rule
How to read `void (*(*f[])())()`?
1.  Find identifier `f`.
2.  Go Right: `[]` -> Array of...
3.  Go Left: `*` -> Pointers to...
4.  Go Right: `()` -> Functions returning...
5.  Go Left: `*` -> Pointers to...
6.  Go Right: `()` -> Functions returning...
7.  Go Left: `void`.
**Result:** `f` is an array of pointers to functions returning pointers to functions returning void.

### 🔹 Part 2: Dynamic Memory in Embedded
*   **Malloc/Free:** Generally banned in safety-critical systems (MISRA C).
    *   **Fragmentation:** Heap becomes Swiss cheese.
    *   **Non-Deterministic:** `malloc` might take 10 cycles or 10,000 cycles.
*   **Solution:** Static Allocation or Pool Allocators.

### 🔹 Part 3: Memory Alignment
*   ARM Cortex-M requires aligned access.
*   `int *p = (int*)0x20000001; *p = 0;` -> **HardFault**.
*   Struct padding ensures alignment. `struct { char c; int i; }` is 8 bytes, not 5.

---

## 💻 Implementation: Pool Allocator

> **Instruction:** Create a deterministic O(1) allocator for fixed-size blocks.

### 👨‍💻 Code Implementation

#### Step 1: The Pool Structure
```c
#define BLOCK_SIZE 32
#define POOL_SIZE  10

typedef struct Block {
    struct Block *next;
} Block_t;

uint8_t pool_storage[POOL_SIZE][BLOCK_SIZE];
Block_t *free_list = NULL;

void Pool_Init(void) {
    free_list = (Block_t*)pool_storage[0];
    for(int i=0; i<POOL_SIZE-1; i++) {
        Block_t *current = (Block_t*)pool_storage[i];
        current->next = (Block_t*)pool_storage[i+1];
    }
    ((Block_t*)pool_storage[POOL_SIZE-1])->next = NULL;
}
```

#### Step 2: Alloc & Free
```c
void* Pool_Alloc(void) {
    if (free_list == NULL) return NULL; // OOM
    
    void *ptr = free_list;
    free_list = free_list->next;
    return ptr;
}

void Pool_Free(void *ptr) {
    if (ptr == NULL) return;
    
    Block_t *block = (Block_t*)ptr;
    block->next = free_list;
    free_list = block;
}
```

---

## 💻 Implementation: Function Pointer Table

> **Instruction:** Implement a Command Dispatcher using a Jump Table.

### 👨‍💻 Code Implementation

#### Step 1: Definitions
```c
typedef void (*CmdHandler_t)(int arg);

void Cmd_Start(int arg) { printf("Start %d\n", arg); }
void Cmd_Stop(int arg)  { printf("Stop %d\n", arg); }
void Cmd_Reset(int arg) { printf("Reset %d\n", arg); }

typedef struct {
    char *name;
    CmdHandler_t handler;
} Command_t;

Command_t commands[] = {
    {"START", Cmd_Start},
    {"STOP",  Cmd_Stop},
    {"RESET", Cmd_Reset}
};
```

#### Step 2: Dispatcher
```c
void Dispatch(const char *cmd_name, int arg) {
    for(int i=0; i < sizeof(commands)/sizeof(Command_t); i++) {
        if (strcmp(cmd_name, commands[i].name) == 0) {
            commands[i].handler(arg);
            return;
        }
    }
    printf("Unknown Cmd\n");
}
```

---

## 🔬 Lab Exercise: Lab 106.1 - Stack Paint

### 1. Lab Objectives
- Visualize Stack usage.
- Detect Stack Overflow.

### 2. Step-by-Step Guide

#### Phase A: Painting
In startup code (Reset_Handler), fill stack memory with `0xDEADBEEF`.
```c
// Assembly or C (before main)
uint32_t *p = &_sstack;
while(p < &_estack) *p++ = 0xDEADBEEF;
```

#### Phase B: Monitoring
Periodically check from the bottom (`_sstack`) up.
```c
void Check_Stack_Usage(void) {
    uint32_t *p = &_sstack;
    while(*p == 0xDEADBEEF && p < &_estack) p++;
    
    int free = (uint32_t)p - (uint32_t)&_sstack;
    printf("Stack Free: %d bytes\n", free);
}
```

### 3. Verification
Run a recursive function to consume stack. Watch "Free" drop. If it hits 0, you crashed.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Double Pointer List
- **Goal:** Linked List operations.
- **Task:**
    1.  Implement `List_Insert(Node **head, Node *new)`.
    2.  Use double pointer `**` to modify the `head` pointer itself if inserting at start.

### Lab 3: Buffer Overflow Canary
- **Goal:** Security.
- **Task:**
    1.  Place a "Canary" value (0xCAFEBABE) at the end of a buffer.
    2.  Check it periodically.
    3.  If changed, trigger HardFault.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Dangling Pointer
*   **Cause:** Freeing memory but keeping the pointer.
*   **Solution:** `free(p); p = NULL;`

#### 2. Memory Leak
*   **Cause:** Allocating but never freeing.
*   **Solution:** Use the Pool Allocator (Day 106). It's easier to track usage (Count free blocks).

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Const Correctness:**
    *   `const int *p`: Pointer to constant int (Can't change value).
    *   `int * const p`: Constant pointer to int (Can't change address).
    *   `const int * const p`: Constant pointer to constant int.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why is `void *` useful?
    *   **A:** Generic pointer. Can point to anything. Used in `memcpy`, `malloc`, and generic drivers.
2.  **Q:** What is `restrict`?
    *   **A:** Promise to compiler that this pointer is the *only* way to access the object. Allows aggressive optimization (e.g., caching value in register).

### Challenge Task
> **Task:** "Heap Walker". Implement a function to traverse the standard `malloc` heap (using `_sbrk` or heap info) and print the size/status of each allocated block. (Requires understanding Newlib's malloc implementation).

---

## 📚 Further Reading & References
- [Expert C Programming: Deep C Secrets](https://www.amazon.com/Expert-Programming-Peter-van-Linden/dp/0131774298)

---
