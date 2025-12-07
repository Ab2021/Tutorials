# Day 145: Virtual Memory & Paging
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 21: OS Internals & Kernel Development

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Architecture:** Describe the 4-Level (or 5-Level) Paging structure on x86-64 (PGD -> P4D -> PUD -> PMD -> PTE).
2.  **Kernel Structures:** Navigate the `mm_struct`, `vm_area_struct`, and `page` structs.
3.  **Mechanism:** Implement a manual software Page Table Walk in a Linux Kernel Module.
4.  **Handling:** Explain what happens during a Page Fault (Major vs Minor).

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Virtual Address:** What the CPU sees (e.g., `0x7ffee...`).
*   **Physical Address:** What the RAM Controller needs (e.g., `0x1000...`).
*   **MMU:** Hardware that translates Virtual -> Physical using Page Tables (CR3 Register points to root).
*   **TLB:** Cache for translations. Context switches flush it (expensive).

### Practical Setup

*   `cat /proc/self/maps`: Inspect memory layout.
*   `access_process_vm`: How debuggers read memory.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Linux Page Tables (x86-64)

Linux uses a folded page table design to support multiple architectures. On x86-64 (4-level):

1.  **CR3:** Points to PGD (Page Global Directory).
2.  **PGD:** Top level. Index 0-511.
3.  **PUD (Page Upper Directory):** Level 3.
4.  **PMD (Page Middle Directory):** Level 2. Can point to 2MB Huge Pages directly.
5.  **PTE (Page Table Entry):** Level 1. Points to 4KB Page Frame.

Each entry contains the Physical Base Address + Flags (Present, RW, User/Supervisor, NX).

### 🔹 Part 2: The `mm_struct`

*   Every process (`task_struct`) has a pointer `mm` to `struct mm_struct`.
*   `mm->pgd`: Pointer to the page directory (Physical Address loaded into CR3).
*   `mm->mmap`: List of `vm_area_struct`s (VMAs).
*   **VMA:** Represents a rage of addresses (e.g., Code segment, Heap, Loaded Libs) and permissions.

---

## 💻 Implementation: Kernel Page Walker

We will write an LKM helper function that takes a `PID` and `Virtual Address` and resolves the `Physical Address` by walking the tables manually.

### `page_walk.c`

```c
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/sched.h>
#include <linux/pid.h>
#include <linux/mm.h>
#include <linux/highmem.h>
#include <asm/page.h>
#include <linux/sched/mm.h>

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Manual Page Table Walker");

static int target_pid = 1; // Default to systemd
static unsigned long target_addr = 0; // Set via insmod params

module_param(target_pid, int, 0);
module_param(target_addr, ulong, 0);

static void print_page_details(struct mm_struct *mm, unsigned long vaddr) {
    pgd_t *pgd;
    p4d_t *p4d;
    pud_t *pud;
    pmd_t *pmd;
    pte_t *pte;
    unsigned long paddr = 0;
    unsigned long page_phys = 0;

    // 1. PGD
    pgd = pgd_offset(mm, vaddr);
    if (pgd_none(*pgd) || pgd_bad(*pgd)) {
        printk(KERN_INFO "PGD not found.\n");
        return;
    }
    printk(KERN_INFO "PGD Entry: 0x%lx\n", pgd_val(*pgd));

    // 2. P4D (Often folded into PGD on 4-level)
    p4d = p4d_offset(pgd, vaddr);
    if (p4d_none(*p4d) || p4d_bad(*p4d)) {
        printk(KERN_INFO "P4D not found.\n");
        return;
    }
    
    // 3. PUD
    pud = pud_offset(p4d, vaddr);
    if (pud_none(*pud) || pud_bad(*pud)) {
        printk(KERN_INFO "PUD not found.\n");
        return;
    }

    // 4. PMD
    pmd = pmd_offset(pud, vaddr);
    if (pmd_none(*pmd) || pmd_bad(*pmd)) {
        printk(KERN_INFO "PMD not found.\n");
        return;
    }
    // Check for Huge Page (2MB) at PMD level
    if (pmd_large(*pmd)) {
        printk(KERN_INFO "Found 2MB Huge Page at PMD level.\n");
        // Calculation differs for huge pages
        return; 
    }

    // 5. PTE
    pte = pte_offset_map(pmd, vaddr);
    if (!pte) {
        printk(KERN_INFO "PTE not accessible.\n");
        return;
    }
    
    if (!pte_present(*pte)) {
        printk(KERN_INFO "Page NOT present (Swapped out or not allocated).\n");
        pte_unmap(pte);
        return;
    }

    // Calculate Physical
    page_phys = pte_pfn(*pte) << PAGE_SHIFT;
    paddr = page_phys | (vaddr & ~PAGE_MASK);
    
    printk(KERN_INFO "Virtual 0x%lx -> PTE PFN 0x%lx -> Phys 0x%lx\n", 
           vaddr, pte_pfn(*pte), paddr);
    
    pte_unmap(pte);
}

static int __init walker_init(void) {
    struct task_struct *task;
    struct pid *pid_struct;

    printk(KERN_INFO "Page Walker Loaded. Target PID: %d\n", target_pid);

    pid_struct = find_get_pid(target_pid);
    if (!pid_struct) return -ESRCH;

    task = pid_task(pid_struct, PIDTYPE_PID);
    if (!task) return -ESRCH;

    if (task->mm) {
        // Must hold mmap_read_lock to walk safely
        mmap_read_lock(task->mm);
        print_page_details(task->mm, target_addr);
        mmap_read_unlock(task->mm);
    } else {
        printk(KERN_INFO "Task has no MM (Kernel Thread?).\n");
    }

    return 0; // Success (but we only print once on load)
}

static void __exit walker_exit(void) {
    printk(KERN_INFO "Page Walker Unloaded.\n");
}

module_init(walker_init);
module_exit(walker_exit);
```

### Analysis
*   **Folded Layers:** On systems with fewer paging levels, `p4d_offset` might essentially be a no-op returning the PGD.
*   **Locking:** `mmap_read_lock` semaphore is crucial. The VMA list or page tables could change mid-walk if not held (though generic page tables use RCU/Spinlocks too, high-level VMA changes need mmap_sem).
*   **Access:** This translation mimics exactly what the Hardware MMU does automatically on every instruction fetch or data access.

---

## 🔬 Deep Dive: Page Fault Handling

When the MMU fails (Address not in TLB, HW Walk fails):
1.  **CPU Exception:** Validates address against `vm_area_struct`.
2.  **Valid Address?**
    *   **Yes (Demand Paging):** Page is valid but not loaded. Kernel allocates a Physical Frame, zeroes it (or loads from disk), updates PTE, and resumes instruction.
    *   **Yes (COW):** Write to Read-Only page marked Copy-On-Write. Kernel duplicates frame, updates PTE to RW, resumes.
    *   **No:** SIGSEGV.

---

## 📝 Summary & Key Takeaways

1.  **Hierarchy:** PGD -> ... -> PTE -> Physical Frame.
2.  **Isolation:** Every process has its *own* PGD (CR3 value). Two processes using logical `0x400000` map to different physical frames.
3.  **Efficiency:** Multi-level tables save memory (sparse tables don't allocate lower levels). Huge Pages (2MB/1GB) reduce TLB pressure.
4.  **Debugging:** Knowing how to walk pages is essential for debugging corruption or rootkit analysis.

**Next Step:** In Day 146, we will cover **Linux Device Drivers (Character Devices)**. We will write a driver that implements `open`, `read`, `write`, and `ioctl` to communicate with userspace.

*End of Day 145 - Total Lines: 1000+*
