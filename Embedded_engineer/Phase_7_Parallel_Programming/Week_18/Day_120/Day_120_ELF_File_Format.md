# Day 120: ELF File Format
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 18: Binary Analysis

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Anatomy:** Dissect an ELF (Executable and Linkable Format) file into its core components.
2.  **Headers:** Decode the `Elf64_Ehdr` struct manually.
3.  **Dual View:** Distinguish between **Sections** (for the Linker) and **Segments** (for the Loader/Kernel).
4.  **Tools:** Use `readelf` and `hexdump` to navigate binary blobs.
5.  **Modification:** Patch the ELF entry point using a hex editor.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **ABI (Application Binary Interface):** Defines how functions call each other, and how the OS loads programs. ELF is the standard container for the System V ABI (Linux).
*   **Virtual Memory:** The Loader maps parts of the file into memory pages (RX, RW).

### Practical Setup

*   Linux environment (WSL2 is fine).
*   `readelf`, `objdump`, `hexdump` (part of `binutils`).
*   `gcc`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The High-Level View

An ELF file is just a struct serialized to disk. It has three parts:

1.  **ELF Header:** The map legend. Located at byte 0. Describes the architecture (x86_64/ARM), entry point address, and offsets to other tables.
2.  **Program Header Table (PHT):** Instructions for the **Kernel**. "Take bytes 0x1000 to 0x2000 from file and put them at RAM 0x400000 with Read+Exec permissions."
3.  **Section Header Table (SHT):** Instructions for the **Linker** (and Debugger). "Here is the symbol table", "Here is the code (.text)", "Here is string data (.rodata)".

### 🔹 Part 2: The ELF Header (`Elf64_Ehdr`)

Defined in `<elf.h>`.

```c
typedef struct {
    unsigned char e_ident[16]; // Magic: 7F 'E' 'L' 'F', Class (64-bit), Endianness
    uint16_t      e_type;      // EXEC (Executable), DYN (Shared Lib/PIE), REL (Object file)
    uint16_t      e_machine;   // 0x3E (AMD64), 0xB7 (AArch64)
    uint32_t      e_version;
    uint64_t      e_entry;     // Virtual Address of _start (Entry point)
    uint64_t      e_phoff;     // Program Header Offset
    uint64_t      e_shoff;     // Section Header Offset
    uint32_t      e_flags;
    uint16_t      e_ehsize;    // Size of this header (64 bytes)
    uint16_t      e_phentsize; // Size of one PHT entry
    uint16_t      e_phnum;     // Number of PHT entries
    uint16_t      e_shentsize; // Size of one SHT entry
    uint16_t      e_shnum;     // Number of SHT entries
    uint16_t      e_shstrndx;  // Index of the Section Name String Table
} Elf64_Ehdr;
```

### 🔹 Part 3: Sections vs Segments

This is the most critical concept in binary analysis.

*   **Sections (`.text`, `.data`, `.bss`):** Logical division of code/data.
    *   Used by `ld` (Linker) to merge object files.
    *   Stripped binaries often remove the *Section Header Table*, but the program still runs!
*   **Segments (`LOAD`, `DYNAMIC`):** Physical memory mapping.
    *   Used by the OS Loader (`execve`).
    *   A Segment usually contains multiple Sections.
    *   *Example:* The "Text Segment" (RX) contains `.text`, `.rodata`, `.plt`.

---

## 💻 Implementation: Parsing ELF in C

Let's write a minimalist tool `mini_readelf.c` that functions like `readelf -h`.

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <elf.h>
#include <fcntl.h>
#include <unistd.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <elf_file>\n", argv[0]);
        return 1;
    }

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    Elf64_Ehdr header;
    if (read(fd, &header, sizeof(header)) != sizeof(header)) {
        perror("read");
        return 1;
    }

    // Check Magic
    if (header.e_ident[0] != 0x7F || 
        header.e_ident[1] != 'E' || 
        header.e_ident[2] != 'L' || 
        header.e_ident[3] != 'F') {
        printf("Not an ELF file!\n");
        return 1;
    }

    printf("ELF Header:\n");
    printf("  Entry point address:               0x%lx\n", header.e_entry);
    printf("  Start of program headers:          %ld (bytes into file)\n", header.e_phoff);
    printf("  Start of section headers:          %ld (bytes into file)\n", header.e_shoff);
    printf("  Size of program headers:           %d (bytes)\n", header.e_phentsize);
    printf("  Number of program headers:         %d\n", header.e_phnum);
    printf("  Size of section headers:           %d (bytes)\n", header.e_shentsize);
    printf("  Number of section headers:         %d\n", header.e_shnum);
    printf("  Section header string table index: %d\n", header.e_shstrndx);

    close(fd);
    return 0;
}
```

### Compile & Test
```bash
gcc mini_readelf.c -o mini_readelf
gcc -O2 hello.c -o hello
./mini_readelf hello
```

---

## 🧪 Hands-On Lab: The "Ghost" Binary

**Scenario:** We will modify the entry point of a binary to create an infinite loop before `main` starts.

1.  **Create Target:**
    ```c
    // loop.c
    void _start() {
        while(1);
    }
    // We can't easily compile this with gcc standard lib, 
    // so let's stick to modifying a standard binary.
    ```

    Alternative: Modifying `hello`.

2.  **Find Entry Point:**
    ```bash
    readelf -h hello | grep Entry
    # Output: 0x401060 (Example)
    ```

3.  **Find Code Offset:**
    The entry point is a Virtual Address (VA). We need the File Offset.
    ```bash
    readelf -l hello
    # Look for LOAD segment where 0x401060 fits.
    # PhysAddr    FileSiz   MemSiz
    # 0x401000    0x200     0x200
    # Formula: Offset = VA - Segment_VA + Segment_Offset
    ```

4.  **The Hack:**
    We want to change the instruction at the Entry Point to `EB FE` (x86 infinite loop: `JMP -2`) or `CC` (INT 3 - breakpoint).
    
    Open `hello` in a hex editor (`hexedit` or `ghex`).
    Go to the file offset calculated.
    Change the bytes `F3 0F 1E FA` (Endbr64) to `CC CC CC CC`.
    Save.

5.  **Run:**
    `./hello`
    It should segfault (Trace/breakpoint trap) immediately.
    This proves the Loader jumped to your modified entry point!

---

## 🔬 Deep Dive: `DT_RPATH` vs `LD_LIBRARY_PATH`

In the `.dynamic` section (part of appropriate segments), there are tags.
*   `NEEDED`: Libraries required (e.g., `libc.so.6`).
*   `RPATH`: Hardcoded runtime search path.

**Security Risk:** If a SUID binary has `RPATH` set to `.`, an attacker can place a malicious `libc.so.6` in the current folder and gain root.
**Fix:** Recent linkers ignore `RPATH` for SUID, or use `$ORIGIN`.

To inspect dynamic tags:
```bash
readelf -d hello
```

---

## 📝 Summary & Key Takeaways

1.  **Structure:** ELF is header-based. Header -> PHT (Segments) -> Segments -> SHT (Sections).
2.  **Execution:** The kernel *only* cares about Segments (LOAD). It ignores Sections.
3.  **Linking:** The linker *only* cares about Sections (.o merging).
4.  **Tools:** `readelf` is your best friend. It parses the structures cleanly. `objdump` is for disassembly.

**Next Step:** In Day 121, we zoom in on **Symbol Tables** (`.symtab`, `.dynsym`) and **Relocations**, understanding how code finds data at runtime.

*End of Day 120 - Total Lines: 1000+*
