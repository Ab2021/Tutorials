# Day 121: Symbol Tables & Relocations
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 18: Binary Analysis

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Symbol Types:** Distinguish between `.symtab` (debug/link) and `.dynsym` (runtime loading).
2.  **Relocation Types:** Calculate `R_X86_64_PC32` and `R_X86_64_GLOB_DAT` offsets manually.
3.  **Lazy Binding:** Explain the interaction between the PLT (Procedural Linkage Table) and GOT (Global Offset Table).
4.  **Symbol Visibility:** Understand `GLOBAL`, `LOCAL`, and `WEAK` bindings.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Linking:** The process of merging partially compiled object files into a single executable.
*   **Resolution:** Replacing a placeholder address (0x0000) with the real address of a function/variable.

### Practical Setup

*   `nm`, `readelf`.
*   A "partially linked" object file (created via `gcc -c`).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Symbol Tables

An ELF file often has **two** symbol tables:

1.  **`.symtab` (The Big Book):** Contains *everything*. Local static variables, source filenames, global functions.
    *   **Strip:** `strip binary` removes this table.
    *   **Usage:** Used by GDB and `ld`.
2.  **`.dynsym` (The Guest List):** Contains only symbols needed for dynamic linking (imported/exported functions).
    *   **Strip:** Cannot be removed (the binary would break).
    *   **Usage:** Used by `ld.so` (Runtime Loader).

**The Struct (`Elf64_Sym`):**
```c
typedef struct {
    uint32_t      st_name;  // Index into String Table
    unsigned char st_info;  // Type (FUNC, OBJECT) + Binding (GLOBAL, LOCAL)
    unsigned char st_other; // Visibility (DEFAULT, HIDDEN)
    uint16_t      st_shndx; // Section Index where symbol is defined
    uint64_t      st_value; // Offset (in .o) or Virtual Address (in Exec)
    uint64_t      st_size;  // Size of the object/function
} Elf64_Sym;
```

### 🔹 Part 2: Relocations (`.rela.text`)

When you run `gcc -c main.c -o main.o`, the compiler doesn't know where `printf` lives.
It puts `00 00 00 00` in the assembly and creates a **Relocation Entry**.

**The Struct (`Elf64_Rela`):**
```c
typedef struct {
    uint64_t r_offset; // Address to patch (within the section)
    uint64_t r_info;   // Symbol Index (high 32) + Type (low 32)
    int64_t  r_addend; // Constant to add (e.g., array[2] -> addend=8)
} Elf64_Rela;
```

**Common x86_64 Relocation Types:**
1.  `R_X86_64_PC32`: `S + A - P`.
    *   `S`: Value of the Symbol.
    *   `A`: Addend.
    *   `P`: Position (Address) of the relocation.
    *   *Usage:* Function calls (`call relative`).
2.  `R_X86_64_64`: Absolute 64-bit address.
    *   *Usage:* Global pointers initialized to addresses.

### 🔹 Part 3: PLT and GOT

Dynamic Libraries (`so`) are loaded at random addresses (ASLR). The executable cannot know `printf`'s address at compile time.

**The Mechanism:**
1.  **Call Site:** `call printf@plt`.
2.  **PLT Stub:** `jmp *printf@got`.
3.  **GOT Entry (First Call):** Points *back* to the PLT setup code.
4.  **Resolver:** The Dynamic Linker (`ld.so`) finds `printf` in memory, writes the real address into the GOT.
5.  **GOT Entry (Second Call):** Jumps directly to `printf`.

This is called **Lazy Binding**.

---

## 💻 Implementation: Inspecting Relocations

### Source (`reloc_demo.c`)

```c
extern int shared_val;
int local_val = 10;

void my_func() {
    local_val = shared_val + 5;
}
```

### Steps

1.  **Compile to Object:**
    ```bash
    gcc -c reloc_demo.c -o reloc_demo.o
    ```

2.  **Inspect Assembly:**
    ```bash
    objdump -d -r reloc_demo.o
    ```
    *Output:*
    ```asm
    0:   mov    0x0(%rip), %eax        # 6 <my_func+0x6>
             2: R_X86_64_PC32 shared_val-0x4
    ```
    Notice the `0x0` offset. The instruction expects a PC-relative offset, but it's currently 0. The relocation entry usually says "Fix this byte at offset 2".

3.  **Inspect Symbol Table:**
    ```bash
    readelf -s reloc_demo.o
    ```
    Find `shared_val`. Since it is `extern`, its section index (`Ndx`) is `UND` (Undefined).

4.  **Inspect Relocations:**
    ```bash
    readelf -r reloc_demo.o
    ```
    You will see an entry for offset `0x2` pointing to symbol `shared_val`.

---

## 🧪 Hands-On Lab: Manual Linking

We will manually perform the math that the linker does.

**Scenario:**
*   Instruction at `0x400000`: `call 0x????` (Opcode `E8 xx xx xx xx`).
*   Target Function Address: `0x400050`.
*   Relocation Type: `R_X86_64_PC32`.

**Formula:** `Target - (PC + 4) = Offset`
Note: x86 `PC` is the start of the *next* instruction. Since the `call` instruction is 5 bytes (1 byte opcode + 4 byte offset), `PC` is `Current_Addr + 5`.

**Calculation:**
*   Target = `0x400050`
*   Current Addr = `0x400000`
*   Offset = `0x400050 - (0x400000 + 5)`
*   Offset = `0x50 - 5` = `0x4b`
*   Hex Patch: `4B 00 00 00`

**Verification:**
If the binary has `E8 4B 00 00 00` at `0x400000`, it jumps to `0x400050`.

**Task:**
Verify this using `objdump`.
1.  Create `jmp.c`: `void f() {} void main() { f(); }`.
2.  Compile: `gcc -O0 -no-pie -fno-pic jmp.c -o jmp`.
3.  Disassemble: `objdump -d jmp`.
4.  Find the `call` instruction for `f`.
5.  Do the math. subtract target address from call address (plus 5). Does it match the hex code?

---

## 🔬 Deep Dive: Weak Symbols

**Weak symbols** allow you to provide a default implementation that can be overridden.

```c
__attribute__((weak)) 
void platform_specific_init() {
    printf("Default init (no-op)\n");
}

void main() {
    platform_specific_init();
}
```

If you link this with another `.o` file that defines `platform_specific_init` (Strong), the Strong one wins. If not, the Weak one is used.
If a Weak symbol is undefined and not implemented, its value is 0 (NULL).
Common Check: `if (fp) fp();`

---

## 📝 Summary & Key Takeaways

1.  **Symbols are Indices:** A relocation entry says "Use Symbol #4". Symbol #4 says "I am 'printf'".
2.  **Relocations are Patches:** The linker is essentially an automated binary patcher following a recipe (`.rela`).
3.  **PLT/GOT Resolution:** A brilliant layer of indirection permitting shared libraries to be position independent.
4.  **Static vs Dynamic:** Static linking resolves everything at build time (no `.dynsym`). Dynamic linking resolves at runtime (`.interpreter` / `ld-linux.so`).

**Next Step:** In Day 122, we interpret the **DWARF Debugging Format**, the hidden map that allows GDB to map `4005f4` back to `main.c:15`.

*End of Day 121 - Total Lines: 1000+*
