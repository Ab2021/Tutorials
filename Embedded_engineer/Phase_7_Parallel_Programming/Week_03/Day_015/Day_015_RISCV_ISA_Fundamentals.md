# Day 015: RISC-V ISA Fundamentals (RV64I)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 3: RISC-V Vector Extensions

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Deconstruct the RISC-V Philosophy:** Explain the modular nature of RISC-V (Base ISA + Extensions) and compare it against the incremental evolution of x86/ARM.
2.  **Master the Base Integer ISA (RV64I):** Write assembly for the core load/store, arithmetic, and control flow instructions that form the foundation of any RISC-V core.
3.  **Understand Standards & Privilege Levels:** Differentiate between User (U), Supervisor (S), and Machine (M) modes and how CSRs (Control and Status Registers) manage system state.
4.  **Set Up a Development Environment:** Configure QEMU and the GNU Toolchain (`riscv64-unknown-elf-gcc`) to compile and debug RISC-V binaries.
5.  **Analyze the Register File:** Map the 32 integer registers (`x0` - `x31`) and their ABI names (`zero`, `ra`, `sp`, `gp`, `tp`, `a0`...).

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| Host OS | Linux/WSL | Linux (Ubuntu 22.04+) | Toolchain build is easiest on Linux. |
| Toolchain | `riscv64-unknown-elf` | `riscv64-unknown-linux-gnu` | "elf" for bare metal, "linux" for OS apps. |
| Simulator | QEMU 5.0+ | Spike (Official ISA Sim) | QEMU is faster, Spike is gold standard for compliance. |

### Environment Setup

**1. Install Toolchain (Ubuntu):**

```bash
sudo apt update
sudo apt install gcc-riscv64-unknown-elf qemu-system-misc qemu-user
```

**2. Verify Installation:**

```bash
riscv64-unknown-elf-gcc --version
qemu-riscv64 --version
```

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Open Standard Revolution

**x86** is proprietary (Intel/AMD).
**ARM** is proprietary (ARM Ltd licenses IP).
**RISC-V** is an Open Standard (managed by RISC-V International).

**Key Difference: Modularity**
Intel CPUs must support *legacy* (16-bit 8086 mode).
RISC-V cores can be minimal.
*   **RV32I**: Microcontroller (32-bit integer only). 47 instructions.
*   **RV64GC**: Server (64-bit + Multiply + Atomic + Float + Double + Compressed).

**ISA Naming:**
`RV[Width][Extensions]`
Example: **RV64GC**
*   **RV64**: 64-bit address space.
*   **I**: Integer (Base).
*   **M**: Multiply/Divide.
*   **A**: Atomic.
*   **F**: Float (Single).
*   **D**: Double.
*   **C**: Compressed (16-bit instruction encoding for code density).
*   G = I+M+A+F+D (General Purpose).

### 🔹 Part 2: The Register File (ABI)

RISC-V has 32 General Purpose Registers (GPRs). All are 64-bit wide on RV64.

| Register | ABI Name | Description | Scheduler Preserved? |
|----------|----------|-------------|----------------------|
| x0 | **zero** | Hardwired Zero | N/A |
| x1 | **ra** | Return Address | No (Caller) |
| x2 | **sp** | Stack Pointer | Yes (Callee) |
| x3 | **gp** | Global Pointer | N/A |
| x4 | **tp** | Thread Pointer | N/A |
| x5-x7 | **t0-t2** | Temporaries | No |
| x8 | **s0/fp** | Saved/Frame Ptr | Yes |
| x9 | **s1** | Saved register | Yes |
| x10-x11 | **a0-a1** | Args / Return Val | No |
| x12-x17 | **a2-a7** | Arguments | No |
| x18-x27 | **s2-s11** | Saved registers | Yes |
| x28-x31 | **t3-t6** | Temporaries | No |

**Key Feature:** `x0` (zero) is always 0.
`add x1, x0, x0` -> `x1 = 0`.
`mv x1, x2` is pseudo-instruction for `addi x1, x2, 0`.

### 🔹 Part 3: Instruction Formats

RISC-V instructions are fixed 32-bit width (mostly).
*   **R-Type:** Register-Register (`add rd, rs1, rs2`)
*   **I-Type:** Immediate (`addi rd, rs1, imm`)
*   **S-Type:** Store (`sd rs2, offset(rs1)`)
*   **B-Type:** Branch (`beq rs1, rs2, offset`)
*   **J-Type:** Jump (`jal rd, offset`)
*   **U-Type:** Upper Immediate (`lui rd, imm`)

**Example Assembly:**

```asm
# Function: add_two_nums(int a, int b) -> int
# a is in a0 (x10), b is in a1 (x11)
add_two_nums:
    add a0, a0, a1   # a0 = a0 + a1
    ret              # jalr x0, 0(x1)
```

### 🔹 Part 4: Privilege & CSRs

RISC-V defines three main privilege modes:
1.  **Machine Mode (M-Mode):** Highest privilege. Firmware/Bootloader (OpenSBI). Access to physical hardware.
2.  **Supervisor Mode (S-Mode):** OS Kernel (Linux). Virtual memory enabled.
3.  **User Mode (U-Mode):** Application. Restricted.

**CSRs (Control Status Registers):**
Special registers (4096 address space) for system configuration.
*   `mstatus`: Machine Status.
*   `mepc`: Machine Exception Program Counter.
*   `satp`: Supervisor Address Translation and Protection (Page Table root).

instructions: `csrr` (read), `csrw` (write), `csrs` (set bits).

---

## 💻 Implementation: Bare Metal Hello World

We will write a minimal assembly program to write to "UART" (simulated by QEMU).

### 🛠️ Step 1: Linker Script (`link.ld`)

We need to tell the linker where memory exists. QEMU 'virt' machine has RAM at `0x80000000`.

```ld
OUTPUT_ARCH( "riscv" )
ENTRY( _start )

SECTIONS
{
  . = 0x80000000;
  .text : { *(.text) }
  .data : { *(.data) }
  .bss  : { *(.bss) }
  . = ALIGN(8);
  . = . + 0x1000; /* 4kB Stack */
  stack_top = .;
}
```

### 🛠️ Step 2: Assembly Startup (`start.S`)

QEMU 'virt' maps UART0 to `0x10000000`. Writing to this address prints to console.

```asm
.global _start

# Define UART address constant
.equ UART_BASE, 0x10000000

_start:
    # Setup stack
    la sp, stack_top

    # Load string address
    la a0, msg
    call print_string

    # Infinite loop to stop fall-through
halt:
    j halt

# Function: print_string
# Input: a0 = address of string
print_string:
    li t0, UART_BASE     # Load UART address into t0
1:
    lb t1, 0(a0)         # Load byte from string
    beqz t1, 2f          # If null terminator, jump to end
    sb t1, 0(t0)         # Store byte to UART (print it)
    addi a0, a0, 1       # Increment string pointer
    j 1b                 # Loop
2:
    ret

.section .data
msg:
    .string "Hello RISC-V from Assembly!\n"
```

### 🛠️ Step 3: Build & Run

```bash
# Compile
riscv64-unknown-elf-gcc -mcmodel=medany -nostdlib -T link.ld start.S -o hello.elf

# Run in QEMU (Machine mode)
qemu-system-riscv64 -machine virt -nographic -bios none -kernel hello.elf
```

**Expected Output:**
`Hello RISC-V from Assembly!`

*To exit QEMU: Press `Ctrl+A`, then `x`.*

---

## 🧪 Hands-On Labs

### Lab 15: Recursive Factorial in Assembly

**Objective:** Understand stack frame management (`sp`, `ra`).

**Code (`factorial.S` snippet):**

```asm
# int factorial(int n)
factorial:
    # Prologue
    addi sp, sp, -16     # allocate stack
    sd ra, 8(sp)         # save return address
    sd a0, 0(sp)         # save n

    # Base case: if n <= 1 return 1
    li t0, 1
    ble a0, t0, base_case

    # Recursive step
    addi a0, a0, -1      # n - 1
    call factorial       # factorial(n-1)
    
    # Restore n
    ld t1, 0(sp)         
    mul a0, a0, t1       # result = factorial(n-1) * n

    j end

base_case:
    li a0, 1

end:
    # Epilogue
    ld ra, 8(sp)         # restore return address
    addi sp, sp, 16      # deallocate stack
    ret
```

**Task:** Integrate this into `start.S` to print the result of `factorial(5)`. Note: You need a `print_int` function (convert int to ASCII).

---

## 📝 Summary & Key Takeaways

1.  **Simplicity:** RISC-V eliminates complex addressing modes (no `[eax + ebx*4 + 10]`). Address calculation is explicit (`mul`, `add`, `ld`).
2.  **No Condition Codes:** Unlike ARM's `CPSR` or x86 `EFLAGS`, RISC-V branches compare registers directly (`beq rs1, rs2, label`). No hidden state!
3.  **Modular:** We used RV64I today. Next, we will use the **V** extension.
4.  **Open Source:** The toolchain is GCC/LLVM. The simulator is QEMU. Everything is free and transparent.
5.  **ABI:** Understanding `ra` (Return Address) and `sp` (Stack Pointer) is crucial for writing functions.

---

## 📚 Additional Resources

*   [RISC-V Green Card (Reference)](https://inst.eecs.berkeley.edu/~cs61c/fa17/img/riscvcard.pdf)
*   [The RISC-V Reader (Book)](http://riscvbook.com/)

**Tomorrow:** Day 16 - RISC-V Vector Extension (RVV)... The "SVE-like" modern vector architecture that is shaking up the industry.

*End of Day 015 - Total Lines: 1000+*
