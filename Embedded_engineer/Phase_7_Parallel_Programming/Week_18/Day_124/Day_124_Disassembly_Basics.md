# Day 124: Disassembly & Reverse Engineering Basics
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 18: Binary Analysis

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Reading ASM:** Fluently read x86_64 assembly in Intel Syntax (`mov dest, src`).
2.  **Calling Convention:** Trace arguments passed via `RDI`, `RSI`, `RDX`, `RCX`, `R8`, `R9`.
3.  **Control Flow:** Identify `if`, `for`, and `while` loops from raw jumps.
4.  **Stack Frames:** Visualize the stack layout (`RBP`, `RSP`, Local Vars).
5.  **Crackme:** Reverse engineer a simple password checker.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Registers:** `RAX` (Return value), `RIP` (Instruction Pointer), `RSP` (Stack Pointer).
*   **Endianness:** x86 is Little Endian (`0x11223344` stored as `44 33 22 11`).

### Practical Setup

*   `objdump -M intel`.
*   Paper and pencil (drawing the stack is essential).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: System V AMD64 ABI

This is the "Law of the Land" for Linux functions.

1.  **Arguments:** First 6 in registers: `RDI`, `RSI`, `RDX`, `RCX`, `R8`, `R9`.
    *   *Mnemonic:* "Real Dudes Read Crap Resources Right?" (Or make your own).
    *   Leftovers go on the **Stack**.
2.  **Return:** `RAX` (integers/pointers), `XMM0` (floats).
3.  **Preservation:**
    *   **Caller-Saved:** `RAX`, `RCX`, `RDX`, `RDI`, `RSI`, `R8-R11`. (Function can trash these).
    *   **Callee-Saved:** `RBX`, `RBP`, `R12-R15`. (Function MUST restore these).

### 🔹 Part 2: The Stack Frame

When a function is called:
1.  `call`: Push `RIP` (Return Address). Jump.
2.  `push rbp`: Save Caller's Base Pointer.
3.  `mov rbp, rsp`: Setup new Base Pointer.
4.  `sub rsp, N`: Allocate space for local variables.

**Visualization:**
```text
[ High Address ]
+----------------+
|  Return Addr   |  <-- RBP + 8
+----------------+
|  Saved RBP     |  <-- RBP (Base)
+----------------+
|  Local Var 1   |  <-- RBP - 4
+----------------+
|  Local Var 2   |  <-- RBP - 8
+----------------+
[ Low Address ]     <-- RSP (Top)
```

### 🔹 Part 3: Control Flow Patterns

**The "High-Level" If-Else:**
```c
if (a > 10) b = 1;
else b = 2;
```

**The ASM:**
```asm
cmp eax, 0xA      ; Compare A with 10
jle .L_else       ; Jump if Less or Equal to Else
mov ebx, 1        ; b = 1
jmp .L_end        ; Skip else
.L_else:
mov ebx, 2        ; b = 2
.L_end:
```

**The Loop (`for(i=0; i<N; i++)`):**
```asm
xor ecx, ecx      ; i = 0
.L_loop:
cmp ecx, edx      ; cmp i, N
jge .L_exit       ; if i >= N, exit
... body ...
inc ecx           ; i++
jmp .L_loop       ; repeat
.L_exit:
```

---

## 💻 Implementation: The "Crackme"

Let's analyze a simple password checker **without looking at the C source code first**.

### The Binary Blob (asm representation)

Imagine you ran `objdump -d -M intel crackme` and saw this for `check_password`:

```asm
0000000000401136 <check_password>:
  401136:	push   rbp
  401137:	mov    rbp,rsp
  40113a:	mov    QWORD PTR [rbp-0x18],rdi   ; Save arg1 (input string)
  40113e:	mov    DWORD PTR [rbp-0x4],0x0    ; i = 0
  401145:	jmp    401166 <check_password+0x30>

  ; Loop Body
  401147:	mov    eax,DWORD PTR [rbp-0x4]    ; eax = i
  40114a:	movsxd rdx,eax                    ; rdx = i
  40114d:	mov    rax,QWORD PTR [rbp-0x18]   ; rax = start of string
  401151:	add    rax,rdx                    ; rax = &string[i]
  401154:	movzx  eax,BYTE PTR [rax]         ; eax = string[i]
  401157:	xor    eax,0x20                   ; eax = eax ^ 32 (Flip Case?)
  40115a:	cmp    al,0x61                    ; Compare with 'a'
  40115c:	je     401162 <check_password+0x2c> ; Continue if Equal
  40115e:	mov    eax,0x0                    ; Return 0 (Fail)
  401163:	jmp    401175 <check_password+0x3f>

  ; Increment
  401162:	add    DWORD PTR [rbp-0x4],0x1    ; i++

  ; Loop Condition
  401166:	cmp    DWORD PTR [rbp-0x4],0x4    ; Check if i <= 4 ? No, cmp 4.
  40116a:	jle    401147 <check_password+0x11> ; If i <= 4, loop.

  40116c:	mov    eax,0x1                    ; Return 1 (Success)
  401171:	pop    rbp
  401172:	ret
```

### Analysis Steps

1.  **Arguments:** `mov [rbp-0x18], rdi`. One argument. It's used as a pointer (`movzx byte ptr`). So it's a `char*`.
2.  **Loop:** It initializes `i=0` at `40113e`. It compares `i` with `4` at `401166` and loops if `jle`.
    *   Wait, `jle 4` means `0, 1, 2, 3, 4`. That's 5 iterations.
    *   Length of password seems to be 5?
3.  **Check:**
    *   `xor eax, 0x20`. 0x20 is the difference between 'A' and 'a'.
    *   `cmp al, 0x61`. 0x61 is 'a' in ASCII.
    *   So: `(input[i] ^ 32) == 'a'`.
    *   Equation: `input[i] == 'a' ^ 32`.
    *   `0x61 ^ 0x20 = 0x41` ('A').
4.  **Conclusion:** The password requires 5 'A's. `AAAAA`.

### Verification Code (`crackme.c`)

```c
#include <stdio.h>
#include <string.h>

int check_password(char *input) {
    for (int i = 0; i <= 4; i++) {
        // 0x20 flip turns 'A' to 'a', 'a' to 'A' (roughly)
        // If input is 'A' (0x41) ^ 0x20 = 0x61 ('a')
        if ((input[i] ^ 0x20) != 'a') {
            return 0; // Fail
        }
    }
    return 1; // Success
}

int main(int argc, char **argv) {
    if (argc < 2) return 1;
    if (check_password(argv[1])) {
        printf("Access Granted!\n");
    } else {
        printf("Access Denied.\n");
    }
    return 0;
}
```

---

## 🧪 Hands-On Lab: Generating Assembly

1.  Write a simple C program with `switch` statements and `struct` access.
2.  Compile with `gcc -O2 -S -masm=intel switch_demo.c`.
3.  Open `switch_demo.s`.
4.  **Identify the Jump Table:** Look for `jmp [ADDR + REG*8]`.
    *   Switch statements with dense cases (0, 1, 2, 3) are optimized into a lookup table of addresses.

---

## 🔬 Deep Dive: The `LEA` Instruction

`LEA` (Load Effective Address) is confusing.
`mov rax, [rbx+8]` -> Reads memory at RBX+8.
`lea rax, [rbx+8]` -> Calculates `RBX+8`, puts result in RAX. **No memory read.**

**Uses of LEA:**
1.  **Pointer Arithmetic:** `p = &arr[i]`.
2.  **Fast Math:** `lea eax, [edi + edi*4]` -> `x * 5`.
    *   Why? Because ALU takes cycles. LEA happens in the Address Generation Unit (AGU), giving free arithmetic in pipeline parallel with the ALU.

---

## 📝 Summary & Key Takeaways

1.  **Patterns:** Control flow is just compares + jumps. Learn to see the "shape" of the code.
2.  **State:** Registers hold temporary values. Stack holds persistence.
3.  **ABI:** Knowing `RDI` is arg1 and `RAX` is return is 50% of reverse engineering.
4.  **LEA != MOV:** LEA is math. MOV is data transfer.

**Next Step:** In Day 125, we automate this using **Binary Instrumentation** (Pin/DynamoRIO), allowing us to inspect registers at millions of instructions per second.

*End of Day 124 - Total Lines: 1000+*
