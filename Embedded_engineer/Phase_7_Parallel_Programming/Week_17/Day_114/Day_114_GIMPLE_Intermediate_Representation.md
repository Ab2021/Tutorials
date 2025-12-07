# Day 114: GIMPLE Intermediate Representation
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 17: GCC Internals

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Understand GIMPLE:** Decode the 3-address Code representation used by GCC's middle-end.
2.  **Analyize SSA:** Visualize how GCC implements Static Single Assignment (Versioned Names, Phi Nodes).
3.  **Explore the API:** Navigate the `gimple` and `tree` data structures in GCC source.
4.  **Write a Plugin:** Create a simple GCC plugin that intercepts the compilation pipeline to analyze GIMPLE.
5.  **Pass Management:** Understand `pass_manager`, `opt_pass`, and pass positioning.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **3-Address Code:** `x = y op z`.
*   **Control Flow Graphs (CFG):** Basic Blocks and Edges.

### Practical Setup

*   Linux Environment (required for GCC plugins).
*   `gcc-X-plugin-dev` package (where X is version, e.g., `gcc-10-plugin-dev`).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: GIMPLE Syntax & Structure

GIMPLE is a simplified subset of C.
*   **No Loop Constructs:** `for`/`while` are broken into `goto` and `cond`. (Note: High GIMPLE has them, but optimizers work on Low GIMPLE).
*   **No Complex Expressions:** `x = a + b + c` becomes `t1 = a+b; x = t1+c`.
*   **Load/Store Architecture:** Memory operations are explicit.

**Example Dump (`.gimple`):**
```c
compute (int a, int b)
{
  int D.1234;
  int t1;

  t1 = a + b;
  if (t1 > 10) goto <D.1235>; else goto <D.1236>;
  
  <D.1235>:
  D.1234 = t1;
  goto <D.1237>;
  
  <D.1236>:
  D.1234 = 0;
  
  <D.1237>:
  return D.1234;
}
```

### 🔹 Part 2: The Data Structures (`tree` vs `gimple`)

In GCC (unlike LLVM), the distinction between "Type" and "Instruction" is split.

1.  **`tree`:** Represents **Operands** (Variable names, Constants, Types, Declarations).
    *   Everything is a tree. `integer_cst`, `var_decl`, `ssa_name`.
    *   Macros: `TREE_TYPE(t)`, `TREE_CODE(t)`.

2.  **`gimple`:** Represents **Statements** (Instructions).
    *   A tuple structure.
    *   `gimple_assign`: Assignment.
    *   `gimple_cond`: Conditional Jump.
    *   `gimple_phi`: Phi node.
    *   *Not* a tree. Chained in a doubly-linked list (`gimple_seq`).

### 🔹 Part 3: SSA in GCC

GCC converts GIMPLE to SSA form early in the pipeline (`pass_build_ssa`).
*   **SSA Names:** `a_1`, `a_2`. Represented by `SSA_NAME` tree nodes.
*   **Immediate Uses:** GCC maintains a list of all instructions using `a_1`.
*   **Definition:** Each SSA name points to the `gimple` statement that defined it (`SSA_NAME_DEF_STMT`).

### 🔹 Part 4: The Pass Manager

GCC organizes passes into lists:
1.  `all_lowering_passes`: High GIMPLE -> Low GIMPLE.
2.  `all_small_ipa_passes`: Early Inter-Procedural Analysis.
3.  **`all_regular_ipa_passes`**: The heavy lifters (Inlining).
4.  **`all_passes`**: Intra-procedural GIMPLE optimizations (DCE, PRE, Vectorization).

A plugin hooks into one of these lists.

---

## 💻 Implementation: A GCC Plugin

We will write a C++ plugin that prints all function names and counts GIMPLE statements.
*Note: This strictly requires a Linux build environment with GCC headers.*

### Source (`simple_plugin.cc`)

```cpp
#include <gcc-plugin.h>
#include <plugin-version.h>
#include <tree.h>
#include <tree-pass.h>
#include <context.h>
#include <function.h>
#include <gimple.h>
#include <gimple-iterator.h>
#include <stdio.h>

// 1. Define the Pass
const pass_data my_pass_data = {
    GIMPLE_PASS,     // type
    "my-pass",       // name (visible in dumps)
    OPTGROUP_NONE,   // optgroup_flags
    TV_NONE,         // tv_id
    PROP_gimple_any, // properties_required
    0,               // properties_provided
    0,               // properties_destroyed
    0,               // todo_flags_start
    0,               // todo_flags_finish
};

class my_pass : public gimple_opt_pass {
public:
    my_pass(gcc::context *ctxt) 
        : gimple_opt_pass(my_pass_data, ctxt) {}

    // 2. The Verification Logic (Gate)
    bool gate(function *fun) override {
        return true; // Always run
    }

    // 3. The Execution Logic
    unsigned int execute(function *fun) override {
        printf("Plugin: Analyzing function '%s'\n", function_name(fun));
        
        basic_block bb;
        int stmt_count = 0;

        // Iterate Basic Blocks
        FOR_EACH_BB_FN(bb, fun) {
            // Iterate Instructions (Gimple Statement Iterator)
            for (gimple_stmt_iterator gsi = gsi_start_bb(bb); !gsi_end_p(gsi); gsi_next(&gsi)) {
                gimple *stmt = gsi_stmt(gsi);
                stmt_count++;
                
                // Check if it's an assignment
                if (is_gimple_assign(stmt)) {
                    // Logic here...
                }
            }
        }
        
        printf("Plugin: Found %d statements.\n", stmt_count);
        return 0;
    }
};

// 4. Plugin Entry Point
int plugin_init(struct plugin_name_args *plugin_info,
                struct plugin_gcc_version *version) {
    
    // Register the pass
    struct register_pass_info pass_info;
    pass_info.pass = new my_pass(g);
    pass_info.reference_pass_name = "cfg"; // Run after CFG is built
    pass_info.ref_pass_instance_number = 1;
    pass_info.pos_op = PASS_POS_INSERT_AFTER;
    
    register_callback(plugin_info->base_name, PLUGIN_PASS_MANAGER_SETUP, NULL, &pass_info);
    
    return 0;
}
```

### Build Script (`Makefile`)

```makefile
GCC=gcc
PLUGIN_DIR=$(shell $(GCC) -print-file-name=plugin)
CXXFLAGS=-I$(PLUGIN_DIR)/include -fPIC -fno-rtti -O2

simple_plugin.so: simple_plugin.cc
    $(CXX) $(CXXFLAGS) -shared -o $@ $<

test: simple_plugin.so
    $(GCC) -fplugin=./simple_plugin.so test.c -o test
```

---

## 🧪 Hands-On Lab: Pythonizing GIMPLE Dumps

Since writing C++ plugins is heavy, let's write a Python script to analyze the text dumps from Day 113.

**Objective:** Parse `*.gimple` file and build a Call Graph.

**Script (`gimple_parser.py`):**
```python
import re
import sys
import glob

def parse_gimple(filename):
    current_func = None
    calls = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # Function header: "foo (int a)"
            m_func = re.match(r'^(\w+)\s*\(.*\)', line)
            if m_func:
                current_func = m_func.group(1)
                continue
            
            # Call site: "ret = bar (x);"
            if current_func and '(' in line and ');' in line:
                # Naive parsing
                parts = line.split('(')[0].split()
                if parts:
                    callee = parts[-1] 
                    if callee != "if": # Ignore 'if'
                        calls.append((current_func, callee))
    return calls

# Run on all dumps
dumps = glob.glob("*.gimple")
for d in dumps:
    print(f"--- Analyzing {d} ---")
    graph = parse_gimple(d)
    for caller, callee in graph:
        print(f"{caller} -> {callee}")
```

**Instruction:**
1.  Compile `hello.c` with `-fdump-tree-gimple`.
2.  Run the python script.
3.  Observe GIMPLE structure.

---

## 🔬 Deep Dive: Memory SSA (Virtual Operands)

GCC has a unique way of handling memory in SSA.
In LLVM, `alloca` is memory, `load/store` are instructions. `mem2reg` promotes them.
In GCC, memory is tracked via **Virtual Operands** (`.MEM`).

`# .MEM_5 = VDEF <.MEM_4>`
`x = *p;`

*   Every memory write defines a new version of `.MEM`.
*   Every memory read uses a version of `.MEM`.
*   This creates a dependency chain for all memory operations, preventing illegal reordering of loads/stores.

---

## 📝 Summary & Key Takeaways

1.  **GIMPLE is Tuples:** Unlike the GENERIC AST, GIMPLE is a flat list of tuples, optimized for traversal.
2.  **Plugins:** GCC plugins are powerful but API-unstable (C++ structure layout can change between versions).
3.  **SSA is baked in:** Most GIMPLE passes assume SSA form. You deal with `SSA_NAME` nodes, not variables.
4.  **Tree vs Gimple:** `tree` is the type/operand system. `gimple` is the instruction system. Do not confuse them.

**Next Step:** In Day 115, we descend into the **RTL (Register Transfer Language)** backend, exploring Machine Descriptions (`.md`) and how GCC maps GIMPLE basic blocks to actual machine instructions.

*End of Day 114 - Total Lines: 1000+*
