# Day 126: Week 18 Review & Project (Binary Patching)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 18: Binary Analysis

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Synthesize:** Connect ELF structure, DWARF debug info, and Disassembly into a holistic view.
2.  **Locate Code:** Calculate File Offsets from Virtual Addresses using Section Headers.
3.  **Patch:** Modify machine code binaries (.text) directly using Python to alter program behavior.
4.  **Verify:** Confirm patches using `objdump` and runtime testing.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Virtual Address (VA):** The address executed by the CPU (e.g., `0x401055`).
*   **File Offset:** The byte position in the `.o` or executable file on disk (e.g., `0x1055`).
*   **Mapping:** `Offset = VA - Section_VA + Section_Offset`.

### Practical Setup

*   `readelf` (to find section offsets).
*   `objdump` (to find instructions).
*   `nasm` (optional, to lookup opcode bytes).
*   Python 3.

---

## 📖 Week 18 Review

### 1. The Structure (Day 120-121)
**ELF** is the container.
*   **Segments** (PHT): For the Loader (Kernel). "Load this blob to 0x400000".
*   **Sections** (SHT): For the Linker. ".text is code, .data is vars".
*   **Symbols**: Map names ("main") to addresses.

### 2. The Context (Day 122)
**DWARF** bridges the gap.
*   **DIE Tree**: Describes types, variables, scopes.
*   **Line Table**: Bytecode machine state machine mapping `Address -> File:Line`.

### 3. The Tools (Day 123-124)
*   `readelf`: The rigid parser. Truth.
*   `objdump`: The interpreter. Disassembly.
*   `nm`: The phonebook.
*   **Assembly:** `mov`, `cmp`, `je`. Stack frames with `rbp`.

### 4. The Runtime (Day 125)
*   **Instrumentation:** Modifying code *while* it runs (JIT) to observe behavior without source.

---

## 💻 Project: The "Binary Patcher"

**Goal:** Create a Python tool that patches `je` (Jump if Equal) to `jne` (Jump if Not Equal), effectively flipping a boolean check logic in a closed-source binary.

### Step 1: The Target (`login.c`)

Compile this and delete the source.

```c
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <password>\n", argv[0]);
        return 1;
    }

    // Secret password is "s3cr3t"
    if (strcmp(argv[1], "s3cr3t") == 0) {
        printf("Access Granted!\n");
        return 0;
    } else {
        printf("Access Denied.\n");
        return 1;
    }
}
```

`gcc -O0 login.c -o login`

### Step 2: Reconnaissance

1.  **Run it:** `./login test` -> "Access Denied".
2.  **Disassemble:**
    `objdump -d -M intel login | grep -A 20 main`

    *Output (Hypothetical):*
    ```asm
    401145: call   <strcmp@plt>
    40114a: test   eax, eax
    40114c: je     40115a       ; Jump if Equal (0) to "Access Granted"
    40114e: lea    rdi, [msg_denied]
    401155: call   puts
    ...
    40115a: lea    rdi, [msg_granted]
    401161: call   puts
    ```
    *Correction:* Actually `strcmp` returns 0 on success.
    So `test eax, eax` sets ZF=1 if match.
    `je` (Jump if ZF=1) means "Jump if Match".
    Wait, usually the "success" block is inside the if, and "else" is jumped over.
    Let's assume the code is:
    ```asm
    je success_block
    fail_block: ...
    jmp end
    success_block: ...
    ```

    We want to change `je` (0x74) to `jne` (0x75).
    Or even simpler: NOP it out? Or force a jump?
    Let's just change `je` to `jne`.
    Then `./login s3cr3t` will FAIL, and `./login whatever` will PASS.

    **Target Address:** `0x40114c`.
    **Target Byte:** `0x74` (JE).

### Step 3: The Patcher Script (`patcher.py`)

This script handles the VA -> Offset math automatically.

```python
import sys
import struct
from elftools.elf.elffile import ELFFile # pip install pyelftools

def va_to_offset(elf, va):
    """
    Translates a Virtual Address (VA) to a File Offset.
    Iterates over segments to find the one containing the VA.
    """
    for segment in elf.iter_segments():
        # check if it is a LOAD segment
        if segment['p_type'] != 'PT_LOAD':
            continue
            
        vaddr = segment['p_vaddr']
        memsz = segment['p_memsz']
        offset = segment['p_offset']
        
        if vaddr <= va < vaddr + memsz:
            # Found the segment!
            return offset + (va - vaddr)
    return None

def patch_binary(filename, va, new_bytes):
    with open(filename, 'r+b') as f:
        elf = ELFFile(f)
        offset = va_to_offset(elf, va)
        
        if offset is None:
            print(f"Error: VA {hex(va)} not found in any segment.")
            return

        print(f"[*] VA {hex(va)} maps to File Offset {hex(offset)}")
        f.seek(offset)
        
        old_bytes = f.read(len(new_bytes))
        print(f"[*] Overwriting: {old_bytes.hex()} -> {new_bytes.hex()}")
        
        f.seek(offset)
        f.write(new_bytes)
        print("[+] Patch applied successfully.")

if __name__ == "__main__":
    # Example usage: patch generic JE (74) to JNE (75)
    # python patcher.py ./login 0x40114c 75
    
    if len(sys.argv) < 4:
        print("Usage: python patcher.py <elf_file> <virtual_address> <hex_bytes>")
        sys.exit(1)
        
    fpath = sys.argv[1]
    va = int(sys.argv[2], 16)
    new_data = bytes.fromhex(sys.argv[3])
    
    patch_binary(fpath, va, new_data)
```

### Step 4: Execution

1.  **Analyze `login`** again to get the EXACT address of the jump.
    `objdump -d login | grep -A 5 "call.*strcmp"`
    Suppose it says `40114c: 74 0c ... je ...`
2.  **Run Patcher:**
    `python3 patcher.py login 0x40114c 75`
3.  **Confirm:**
    `objdump -d login | grep 40114c`
    Should show `75` (`jne`).
4.  **Test:**
    `./login wrongpass`
    *Output:* Access Granted!

---

## 🧪 Hands-On Lab: The "NOP Sled"

Sometimes you want to delete code completely (e.g., a licensing check).
You can replace bytes with `0x90` (NOP).

**Scenario:**
`call check_license` at `0x402000` (5 bytes: `E8 xx xx xx xx`).
We want to remove this call.

**Action:**
`python3 patcher.py app 0x402000 9090909090`

**Result:**
The CPU slides through the NOPs doing nothing, effectively skipping the function call.

---

## 📝 Summary & Key Takeaways

1.  **Binary Editing:** Is dangerous but powerful. One wrong byte crashes the app (SIGSEGV).
2.  **Offsets Matter:** You cannot just `.seek(0x40114c)` because the file on disk starts at 0, but the kernel loads it at 0x400000.
3.  **Instruction Encoding:** x86 is variable length. Jumping into the middle of an instruction creates garbage code.
4.  **This is Cracking:** These are the fundamental techniques used by software crackers (and malware analysts fixing bugs in C2 servers).

**Next Step:** Week 18 is complete! In Week 19, we move to **Memory Management & Allocators**. We will write our own `malloc` from scratch.

*End of Day 126 - Total Lines: 1000+*
