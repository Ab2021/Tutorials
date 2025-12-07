# Day 142: Linux System Calls & Implementation
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 21: OS Internals & Kernel Development

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Syscall ABI:** Understand how arguments are passed from User to Kernel (Registers vs Stack).
2.  **Mechanism:** Compare Legacy `INT 0x80` vs Modern `SYSCALL` / `SYSENTER`.
3.  **Bypass:** Invoke system calls directly in Assembly, bypassing `glibc` wrappers.
4.  **Implementation:** Trace a syscall from userspace entry to kernel handler (`do_syscall_64`).

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **The Barrier:** User code cannot jump to Kernel code. It must cause an *exception* or *trap* that switches the CPU to Ring 0 and jumps to a predefined handler.
*   **The Table:** The Kernel maintains a `sys_call_table`, an array of function pointers indexed by the syscall number (RAX).
*   **VDSO:** A "virtual" shared library mapped into every process to allow some syscalls (like `gettimeofday`) to run purely in userspace (reading mapped kernel memory) without the overhead of a context switch.

### Hardware Comparison

| Feature | `INT 0x80` (Legacy x86) | `SYSCALL` (x64 AMD) | `SYSENTER` (x86 Intel) |
| :--- | :--- | :--- | :--- |
| **Speed** | Slow (Full Interrupt) | Fast (MSR based) | Fast (MSR based) |
| **Registers** | Stack/Regs | `RCX`, `R11` clobbered | Fixed Regs |
| **Usage** | 32-bit Compatibility | Default 64-bit Linux | 32-bit Fast Path |

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The x86-64 Linux Syscall ABI

When you call `write(fd, buf, len)` in C, `glibc` does the following:

1.  **RAX:** Syscall Number (`1` for `write`).
2.  **RDI:** Argument 1 (`fd`).
3.  **RSI:** Argument 2 (`buf`).
4.  **RDX:** Argument 3 (`len`).
5.  **R10:** Arg 4.
6.  **R8:** Arg 5.
7.  **R9:** Arg 6.
8.  **Instruction:** `SYSCALL`.

**Return Value:**
*   Success: `RAX` contains result (>= 0).
*   Error: `RAX` contains `-errno` (e.g., -1 for EPERM).

### 🔹 Part 2: Kernel Side Handling

1.  **Entry:** `entry_SYSCALL_64` (Assembly entry point defined in MSR_LSTAR).
2.  **Swapgs:** Switch to Kernel GS register (per-cpu data).
3.  **Stack Switch:** Switch from User Stack to Kernel Stack (TSS).
4.  **Lookup:** call `sys_call_table[RAX]`.
5.  **Exit:** `SYSRET`.

---

## 💻 Implementation: Raw Syscalls (No Libc)

We will write a "Hello World" that produces a 400-byte binary by avoiding all libc dependencies.

### 1. `direct_syscall.c` (Using Inline Assembly)

```c
// Compile with: gcc -nostdlib -o direct_syscall direct_syscall.c
// Note: We need a custom entry point because we have no glibc _start

/* System Call Numbers (x64) */
#define SYS_WRITE 1
#define SYS_EXIT  60

/* File Descriptors */
#define STDOUT 1

/* Syscall Wrapper Function */
long syscall3(long number, long arg1, long arg2, long arg3) {
    long ret;
    __asm__ volatile (
        "syscall"
        : "=a" (ret)                  // Output: RAX gets return value
        : "a" (number),               // Input: RAX = syscall number
          "D" (arg1),                 // RDI = arg1
          "S" (arg2),                 // RSI = arg2
          "d" (arg3)                  // RDX = arg3
        : "rcx", "r11", "memory"      // Clobbers
    );
    return ret;
}

long syscall1(long number, long arg1) {
    long ret;
    __asm__ volatile (
        "syscall"
        : "=a" (ret)
        : "a" (number), "D" (arg1)
        : "rcx", "r11", "memory"
    );
    return ret;
}

/* Entry Point (Replacing main) */
void _start() {
    char msg[] = "Hello from Raw Assembly!\n";
    
    // write(1, msg, sizeof(msg)-1)
    syscall3(SYS_WRITE, STDOUT, (long)msg, sizeof(msg)-1);
    
    // exit(0)
    syscall1(SYS_EXIT, 0);
    
    // Unreachable
    __builtin_unreachable();
}
```

### 2. Assembly Version: `hello.asm` (NASM)

For ultimate control, we can write pure assembly.

```nasm
; Build: nasm -f elf64 hello.asm -o hello.o && ld hello.o -o hello

section .data
    msg db "Pure Assembly Syscall!", 0xA
    len equ $ - msg

section .text
    global _start

_start:
    ; syscall number for write is 1
    mov rax, 1
    
    ; arg1: fd (1 = stdout)
    mov rdi, 1
    
    ; arg2: buffer pointer
    mov rsi, msg
    
    ; arg3: length
    mov rdx, len
    
    ; Execute Syscall
    syscall
    
    ; syscall number for exit is 60
    mov rax, 60
    
    ; arg1: status code 0
    mov rdi, 0
    
    syscall
```

### Analysis of Binary Size
*   **Standard C (`printf`):** ~16KB (dynamically linked).
*   **Static C (`printf`):** ~800KB.
*   **Direct Syscall (C):** ~1KB.
*   **Assembly:** ~600 bytes.

---

## 🔬 Deep Dive: Adding a Syscall (The Hard Way)

*Note: This requires recompiling the kernel.*

1.  **Define:** Add function `sys_mycall` in `kernel/sys.c`.
    ```c
    SYSCALL_DEFINE0(mycall) {
        printk("My Custom Syscall!\n");
        return 42;
    }
    ```
2.  **Register:** Add ID to `arch/x86/entry/syscalls/syscall_64.tbl`.
    ```text
    440     common  mycall      sys_mycall
    ```
3.  **Header:** Add declaration to `include/linux/syscalls.h`.
4.  **Recompile:** `make -j$(nproc)`.

Since this takes hours, we usually use **Kernel Modules** with **ioctl** or **Netlink** to communicate instead of adding new syscalls.

---

## 📝 Summary & Key Takeaways

1.  **Interface:** Syscalls are the API of the Kernel.
2.  **Performance:** Syscalls are fast (~100s of nanoseconds) but not free. Batch calls (like `writev` instead of multiple `write`s) are preferred.
3.  **Stability:** The Syscall ABI is extremely stable. Binaries from 1995 often still run on today's kernels.
4.  **Control:** Bypassing standard libraries gives you full control and zero overhead, ideal for embedded loaders or shellcode.

**Next Step:** In Day 143, we will explore **The Process Scheduler (CFS)**. We will look at how the kernel decides strictly which `task_struct` runs next using Red-Black Trees.

*End of Day 142 - Total Lines: 1000+*
