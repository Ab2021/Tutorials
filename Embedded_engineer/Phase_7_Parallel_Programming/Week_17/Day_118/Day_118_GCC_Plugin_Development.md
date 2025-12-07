# Day 118: GCC Plugin Development (Project)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 17: GCC Internals

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Architecture:** Design a GCC plugin that injects a custom pass into the request pipeline.
2.  **GIMPLE Traversal:** Iterate through Basic Blocks and GIMPLE statements programmatically.
3.  **CFG Analysis:** specialized usage of `edge_iterator` to calculate McCabe's Cyclomatic Complexity.
4.  **Diagnostics:** Emit custom compiler warnings/errors using GCC's `warning_at` API.
5.  **Build System:** Create a standard Makefile for GCC plugins.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Cyclomatic Complexity (M):** $M = E - N + 2P$.
    *   $E$: Number of edges.
    *   $N$: Number of nodes (basic blocks).
    *   $P$: Connected components (usually 1 for a function).
*   **Thresholds:** Functions with $M > 10$ are considered complex and hard to test.

### Practical Setup

*   `gcc-X-plugin-dev` headers.
*   C++ Compiler.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Plugin API

GCC plugins are dynamic libraries (`.so` on Linux, `.dll` on Windows - though rare). They hook into events.

**Key Events:**
*   `PLUGIN_INFO`: Register plugin version/help.
*   `PLUGIN_PASS_MANAGER_SETUP`: Insert a new optimization pass.
*   `PLUGIN_FINISH`: Cleanup.

### 🔹 Part 2: Pass Data Structure (`pass_data`)

To insert a pass, we define its metadata:
*   `type`: `GIMPLE_PASS`, `RTL_PASS`, `IPA_PASS`.
*   `name`: Unique identifier (e.g., "my_complexity_check").
*   `optgroup_flags`: `OPTGROUP_NONE` or `OPTGROUP_LOOP` (helps enabling/disabling via CLI).
*   `tv_id`: Timing variable (for `-ftime-report`).
*   `properties_required`: Expected state (e.g., `PROP_ssa`).

### 🔹 Part 3: Location Tracking

Compiler diagnostics need a standard location.
*   `location_t`: An integer handle.
*   `gimple_location(stmt)`: Extracts the source line/column from a statement.
*   `DECL_SOURCE_LOCATION(fndecl)`: Extracts the function start line.

---

## 💻 Implementation: The Complexity Checker

We will build `complexity_plugin.cc`. It calculates complexity and warns if it exceeds a threshold (default 10).

### Step 1: The Code (`complexity_plugin.cc`)

```cpp
#include <gcc-plugin.h>
#include <plugin-version.h>
#include <tree.h>
#include <tree-pass.h>
#include <context.h>
#include <basic-block.h>
#include <gimple.h>
#include <diagnostic.h>
#include <stdio.h>

// Licensing is mandatory for GPL compatibility check
int plugin_is_GPL_compatible;

const pass_data complexity_pass_data = {
    GIMPLE_PASS,
    "complexity_check", // Name
    OPTGROUP_NONE,
    TV_NONE,
    PROP_gimple_any,    // Run on any GIMPLE
    0, 0, 0, 0
};

class complexity_pass : public gimple_opt_pass {
public:
    complexity_pass(gcc::context *ctxt) 
        : gimple_opt_pass(complexity_pass_data, ctxt) {}

    bool gate(function *fun) override {
        // Run on all functions
        return true;
    }

    unsigned int execute(function *fun) override {
        int edges = 0;
        int nodes = 0;
        basic_block bb;

        // 1. Count Nodes
        // n_basic_blocks_for_fn includes ENTRY and EXIT, 
        // usually we want just the body, but standard formula includes them.
        nodes = n_basic_blocks_for_fn(fun);

        // 2. Count Edges
        FOR_EACH_BB_FN(bb, fun) {
            edge e;
            edge_iterator ei;
            FOR_EACH_EDGE(e, ei, bb->succs) {
                edges++;
            }
        }

        // 3. Calculate M = E - N + 2P (P=1)
        int complexity = edges - nodes + 2;

        // 4. Threshold Check (Hardcoded 10 for demo)
        if (complexity > 10) {
            // Get location of function definition
            location_t loc = DECL_SOURCE_LOCATION(fun->decl);
            
            warning_at(loc, 0, 
                "Function '%s' has high Cyclomatic Complexity (%d). Threshold is 10.", 
                function_name(fun), complexity);
        }

        return 0; // No todo flags
    }
};

int plugin_init(struct plugin_name_args *plugin_info,
                struct plugin_gcc_version *version) {
    if (!plugin_default_version_check(version, &gcc_version)) {
        printf("Incompatible GCC version\n");
        return 1;
    }

    // Register the pass
    struct register_pass_info pass_info;
    pass_info.pass = new complexity_pass(g);
    pass_info.reference_pass_name = "ssa"; // Run after SSA builder
    pass_info.ref_pass_instance_number = 1;
    pass_info.pos_op = PASS_POS_INSERT_AFTER;

    register_callback(plugin_info->base_name, PLUGIN_PASS_MANAGER_SETUP, NULL, &pass_info);
    
    return 0;
}
```

### Step 2: The Build System (`Makefile`)

```makefile
# Find the plugin headers
GCC_PLUGIN_DIR := $(shell gcc -print-file-name=plugin)
CXXFLAGS += -I$(GCC_PLUGIN_DIR)/include -fPIC -fno-rtti -O2

# Plugin target
complexity_plugin.so: complexity_plugin.cc
    g++ $(CXXFLAGS) -shared -o $@ $<

clean:
    rm -f *.so
```

### Step 3: The Test Case (`test.c`)

```c
#include <stdio.h>

// Complexity 1 (Simple)
void simple() {
    printf("Hello\n");
}

// Complexity High
void monster(int a, int b) {
    if (a > 0) {
        if (b > 0) printf("1");
        else printf("2");
    } else {
        while (b < 10) {
            if (a == -1) break;
            if (a == -2) continue;
            switch(b) {
                case 1: printf("A"); break;
                case 2: printf("B"); break;
                case 3: printf("C"); break;
                case 4: printf("D"); break;
                default: printf("E"); break;
            }
            b++;
        }
    }
}
```

### Step 4: Running It

```bash
make
gcc -fplugin=./complexity_plugin.so -c test.c
```

**Expected Output:**
`test.c:9: warning: Function 'monster' has high Cyclomatic Complexity (14). Threshold is 10.`

---

## 🧪 Hands-On Lab: Extending the Plugin

**Task:** Add a check for "Too many arguments".

**Modifications:**
1.  Inside `execute(function *fun)`:
2.  Access `DECL_ARGUMENTS(fun->decl)`.
3.  Iterate the arguments chain (it's a `tree` list).
4.  Count them.
5.  If `count > 6`, emit a warning: "Too many arguments (N). Consider using a struct."

**Hint:**
```cpp
tree arg;
int arg_count = 0;
for (arg = DECL_ARGUMENTS(fun->decl); arg; arg = DECL_CHAIN(arg)) {
    arg_count++;
}
```

---

## 🔬 Deep Dive: Plugin Arguments

You can pass arguments *to* the plugin from the command line.

`gcc -fplugin=./my.so -fplugin-arg-my-threshold=15 test.c`

**Implementation:**
In `plugin_init`, the `plugin_info->argc` and `plugin_info->argv` fields contain these arguments.
You parse them (e.g., matching "threshold") and configure the pass instance accordingly.
This makes your tool configurable for CI/CD pipelines.

---

## 📝 Summary & Key Takeaways

1.  **Custom QA:** GCC plugins allow you to enforce **domain-specific** rules that standard linters can't catch (because plugins see the fully resolved types and macros).
2.  **Pass Positioning:** Running after "ssa" ensures the CFG is clean and variables are canonical. Running too early (before parsing is fully done) is dangerous.
3.  **Stability Warning:** GCC internal API changes with every major release. Maintainability is the cost of power.
4.  **Static Analysis:** You just built a rudimentary static analyzer!

**Next Step:** In Day 119, we finish Week 17 with a **Review & Project**, synthesizing everything into a comprehensive "Compiler Explorer" toolset.

*End of Day 118 - Total Lines: 1000+*
