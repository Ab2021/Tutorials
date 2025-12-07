# Day 122: DWARF Debugging Format
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 18: Binary Analysis

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Structure:** Visualize DWARF as a tree of **DIEs** (Debugging Information Entries).
2.  **Navigation:** Map a memory address to a specific Variable or Function scope.
3.  **Line Mapping:** Execute the DWARF **Line Number Program** state machine manually.
4.  **Tools:** Use `readelf --debug-dump` to inspect type definitions and source mappings.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **PDB vs DWARF:** Windows uses PDB (proprietary, separate file). Linux favors DWARF (embedded in ELF, open standard).
*   **Compilation:** Requires `-g` (default debug) or `-g3` (macro info).

### Practical Setup

*   `readelf` (supports DWARF 4 and 5).
*   `objdump`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The DIE Tree (`.debug_info`)

DWARF represents code as a flattened tree.
Each node is a **DIE** (Debugging Information Entry).

**Anatomy of a DIE:**
1.  **Tag (`DW_TAG_*`):** What is this? (e.g., `DW_TAG_subprogram` for a function, `DW_TAG_variable`).
2.  **Attributes (`DW_AT_*`):** Metadata.
    *   `DW_AT_name`: "my_var"
    *   `DW_AT_type`: Reference to another DIE (e.g., `int`).
    *   `DW_AT_low_pc`: Start address.
    *   `DW_AT_high_pc`: End address (or size).
    *   `DW_AT_location`: WHERE is it? (Register, Stack Offset).

**Example Tree:**
```text
DW_TAG_compile_unit (source="main.c")
 ├── DW_TAG_subprogram (name="main")
 │    ├── DW_TAG_variable (name="argc", location=RDI)
 │    └── DW_TAG_variable (name="argv", location=RSI)
 └── DW_TAG_base_type (name="int", size=4)
```

### 🔹 Part 2: The Line Number Program (`.debug_line`)

How does GDB know that `0x401055` means `main.c:12`?
It's NOT a simple table. A table would be huge (one entry per instruction).
It's a **Bytecode Program** run by a State Machine.

**The Registers:**
*   `address`: Current PC (0).
*   `file`: Current file index (1).
*   `line`: Current source line (1).
*   `column`: Current column (0).
*   `is_stmt`: Is this the start of a statement?

**The Opcodes:**
*   `DW_LNS_advance_pc(delta)`: Add delta to `address`.
*   `DW_LNS_advance_line(delta)`: Add delta to `line`.
*   `Special Opcodes`: Do both at once (very compact).

**Execution:**
Start at 0. Read/Execute opcodes. When `DW_LNS_copy` (or special opcode) is hit, emit a row `(Address, File, Line)`.

### 🔹 Part 3: Location Expressions (`.debug_loc`)

Variables move.
*   Line 10: `x` is in `RAX`.
*   Line 11: `call func()`. `RAX` is clobbered. `x` saved to `[RBP-8]`.
*   Line 12: `x` loaded back to `RBX`.

DWARF uses "Location Lists" to describe this lifetime.
`DW_AT_location` points to a list:
*   `[0x4000-0x4010]`: `DW_OP_reg0` (RAX)
*   `[0x4010-0x4020]`: `DW_OP_fbreg -8` (Frame Base - 8)

---

## 💻 Implementation: Decoding DWARF

### Source (`debug_demo.c`)

```c
struct Point {
    int x;
    int y;
};

int main() {
    struct Point p = {10, 20};
    return p.x + p.y;
}
```

### Compile & Dump

```bash
gcc -g -O0 debug_demo.c -o debug_demo
# Dump Info (The Tree)
readelf --debug-dump=info debug_demo > info.txt
# Dump Line (The Table)
readelf --debug-dump=line debug_demo > line.txt
```

### Analysis 1: The Struct

Open `info.txt`. Search for "Point".
```text
<1><2d>: Abbrev Number: 2 (DW_TAG_structure_type)
    <2e>   DW_AT_name        : (indirect string): Point
    <32>   DW_AT_byte_size   : 8
    <33>   DW_AT_decl_file   : 1
    <34>   DW_AT_decl_line   : 1
```
Children:
```text
<2><35>: Abbrev Number: 3 (DW_TAG_member)
    <36>   DW_AT_name        : x
    <38>   DW_AT_type        : <0x4a>
    <3c>   DW_AT_data_member_location: 0
<2><3d>: Abbrev Number: 3 (DW_TAG_member)
    <3e>   DW_AT_name        : y
    <40>   DW_AT_type        : <0x4a>
    <44>   DW_AT_data_member_location: 4
```
It tells us exact offsets! This is how GDB knows `p.y` is at offset 4.

### Analysis 2: The Line Table

Open `line.txt`.
```text
Line Number Statements:
  [0x00000038]  Extended opcode 2: set Address to 0x1129
  [0x00000043]  Special opcode 8: advance Address by 0 to 0x1129 and Line by 3 to 4
  [0x00000044]  Special opcode 146: advance Address by 10 to 0x1133 and Line by 1 to 5
```
This shows the compression. "Advance PC by 10 bytes, Line by 1".

---

## 🧪 Hands-On Lab: The "Manual Backtrace"

**Scenario:**
You have a coredump. `RIP` = `0x401140`. You have the binary but no GDB (or GDB is broken).
Find the source line.

1.  **Get Line Table:**
    `readelf --debug-dump=decodedline debug_demo`
    (Using `decodedline` runs the state machine for you and prints the table).

2.  **Search:**
    Look for the range containing `0x401140`.
    ```text
    Address      Line
    0x401133     5
    0x401145     6
    ```
    If `0x401133 <= RIP < 0x401145`, then it is Line 5.

3.  **Cross-Verify:**
    `addr2line -e debug_demo 0x401140`
    Should print `debug_demo.c:5`.
    `addr2line` implements the state machine logic we just discussed.

---

## 🔬 Deep Dive: DWARF Expression Stack

DWARF has a stack-based language for complex locations.
`DW_AT_location: DW_OP_breg5 (rDI) 0; DW_OP_deref; DW_OP_plus_uconst 24`

Translation:
1.  Push `RDI + 0`.
2.  Pop, Dereference (Reads memory at address). Push result.
3.  Pop, Add 24. Push result.

This is Turing-complete! You can theoretically write infinite loops in DWARF expressions, hanging the debugger.

---

## 📝 Summary & Key Takeaways

1.  **Tree Structure:** DWARF mirrors the AST (Abstract Syntax Tree) of your code.
2.  **Compression:** The "Line Number Program" is a brilliant way to map 1M instructions to source lines without a 100MB table.
3.  **Location Lists:** Crucial for optimized code where variables jump between registers and stack.
4.  **Security:** Always strip DWARF (`strip --strip-debug`) before shipping, or you give reverse engineers a perfect map of your logic.

**Next Step:** In Day 123, we start using the heavy machinery: **Binary Analysis Tools** (`nm`, `objdump`, `readelf` mastery) to dissect closed-source binaries.

*End of Day 122 - Total Lines: 1000+*
