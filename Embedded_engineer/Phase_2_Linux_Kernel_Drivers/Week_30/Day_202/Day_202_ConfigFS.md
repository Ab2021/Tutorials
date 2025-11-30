# Day 202: ConfigFS (Userspace Configuration)
## Phase 2: Linux Kernel & Device Drivers | Week 30: Device Model & Sysfs

---

> **📝 Content Creator Instructions:**
> This document is designed to produce **comprehensive, industry-grade educational content**. 
> - **Target Length:** The final filled document should be approximately **1000+ lines** of detailed markdown.
> - **Depth:** Do not skim over details. Explain *why*, not just *how*.
> - **Structure:** If a topic is complex, **DIVIDE IT INTO MULTIPLE PARTS** (Part 1, Part 2, etc.).
> - **Code:** Provide complete, compilable code examples, not just snippets.
> - **Visuals:** Use Mermaid diagrams for flows, architectures, and state machines.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Distinguish** between Sysfs (View/Control) and ConfigFS (Create/Destroy).
2.  **Mount** and explore ConfigFS.
3.  **Implement** a ConfigFS subsystem (`config_item_type`, `make_item`, `drop_item`).
4.  **Create** kernel objects by `mkdir` in userspace.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC.
*   **Software Required:**
    *   Kernel with `CONFIG_CONFIGFS_FS`.
*   **Prior Knowledge:**
    *   Day 197 (Kobjects).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Sysfs vs ConfigFS
*   **Sysfs:** Kernel creates objects -> Userspace sees them. (Kernel Master).
*   **ConfigFS:** Userspace creates objects -> Kernel instantiates them. (Userspace Master).
*   **Use Cases:** USB Gadget (creating functions), iSCSI Target (creating LUNs), Netconsole.

### 🔹 Part 2: The Mechanism
*   **mkdir:** Calls driver's `make_item` or `make_group`. Driver allocates memory.
*   **rmdir:** Calls driver's `drop_item`. Driver frees memory.
*   **Attributes:** Similar to Sysfs, but usually writable to configure the new object.

---

## 💻 Implementation: A "Bank" Subsystem

> **Instruction:** Create a ConfigFS subsystem where users can create "accounts" (directories) and set "balance" (file).

### 👨‍💻 Code Implementation

```c
#include <linux/module.h>
#include <linux/configfs.h>
#include <linux/slab.h>

struct account {
    struct config_item item;
    int balance;
};

static inline struct account *to_account(struct config_item *item) {
    return container_of(item, struct account, item);
}

// Attribute
static ssize_t balance_show(struct config_item *item, char *page) {
    return sprintf(page, "%d\n", to_account(item)->balance);
}
static ssize_t balance_store(struct config_item *item, const char *page, size_t count) {
    sscanf(page, "%d", &to_account(item)->balance);
    return count;
}
CONFIGFS_ATTR(account_, balance);

static struct configfs_attribute *account_attrs[] = {
    &account_attr_balance,
    NULL,
};

static void account_release(struct config_item *item) {
    kfree(to_account(item));
}

static struct configfs_item_operations account_item_ops = {
    .release = account_release,
};

static struct config_item_type account_type = {
    .ct_item_ops = &account_item_ops,
    .ct_attrs = account_attrs,
    .ct_owner = THIS_MODULE,
};

// Group Operations (mkdir)
static struct config_item *bank_make_item(struct config_group *group, const char *name) {
    struct account *acct;
    
    acct = kzalloc(sizeof(*acct), GFP_KERNEL);
    if (!acct) return ERR_PTR(-ENOMEM);
    
    config_item_init_type_name(&acct->item, name, &account_type);
    return &acct->item;
}

static struct configfs_group_operations bank_group_ops = {
    .make_item = bank_make_item,
};

static struct config_item_type bank_type = {
    .ct_group_ops = &bank_group_ops,
    .ct_owner = THIS_MODULE,
};

static struct configfs_subsystem bank_subsys;

static int __init my_init(void) {
    config_group_init(&bank_subsys.su_group);
    mutex_init(&bank_subsys.su_mutex);
    bank_subsys.su_group.cg_item.ci_type = &bank_type; // Root type
    
    return configfs_register_subsystem(&bank_subsys);
}
// ... exit ...
```

---

## 🔬 Lab Exercise: Lab 202.1 - Managing Accounts

### 1. Lab Objectives
- Mount ConfigFS.
- Create accounts via `mkdir`.
- Set balances.

### 2. Step-by-Step Guide
1.  **Mount:**
    ```bash
    mount -t configfs none /sys/kernel/config
    ```
2.  **Load Module:** `insmod my_bank.ko`.
3.  **Create Account:**
    ```bash
    mkdir /sys/kernel/config/bank/savings
    mkdir /sys/kernel/config/bank/checking
    ```
4.  **Configure:**
    ```bash
    echo 1000 > /sys/kernel/config/bank/savings/balance
    echo 50 > /sys/kernel/config/bank/checking/balance
    ```
5.  **Verify:**
    ```bash
    cat /sys/kernel/config/bank/savings/balance
    ```
6.  **Destroy:**
    ```bash
    rmdir /sys/kernel/config/bank/savings
    ```

---

## 🧪 Additional / Advanced Labs

### Lab 2: Committing Configuration
- **Goal:** Prevent modification after "activation".
- **Task:**
    1.  Add a `commit` attribute.
    2.  When written to, set a flag `committed = true`.
    3.  In `balance_store`, check `if (committed) return -EPERM;`.
    4.  This is a common pattern: Configure -> Commit -> Active.

### Lab 3: Symlinks
- **Goal:** Link objects.
- **Task:**
    1.  Implement `allow_link` and `drop_link` in `group_ops`.
    2.  Userspace: `ln -s /config/bank/savings /config/bank/checking/backup`.
    3.  Driver receives the target item.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Operation not supported" on mkdir
*   **Cause:** Parent group does not implement `make_item` or `make_group`.
*   **Fix:** Ensure the parent's `config_item_type` has valid `group_ops`.

#### 2. Module in use
*   **Cause:** ConfigFS holds a reference to the module as long as items exist.
*   **Fix:** You must `rmdir` all items before you can `rmmod`.

---

## ⚡ Optimization & Best Practices

### `configfs_depend_item`
*   If one item depends on another (e.g., via symlink), use dependency helpers to prevent the target from being removed while in use.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why use ConfigFS instead of IOCTLs?
    *   **A:** ConfigFS is scriptable, visible, and follows the "Everything is a file" philosophy. IOCTLs are opaque and require custom tools.
2.  **Q:** Can I have a hierarchy (Groups inside Groups)?
    *   **A:** Yes. Implement `make_group` instead of `make_item`.

### Challenge Task
> **Task:** "The Soft-RAID Configurator".
> *   Create a ConfigFS subsystem "raid".
> *   `mkdir /config/raid/md0`.
> *   `ln -s /sys/block/sda /config/raid/md0/disk1`.
> *   `echo 1 > /config/raid/md0/start`.

---

## 📚 Further Reading & References
- [Kernel Documentation: filesystems/configfs/configfs.rst](https://www.kernel.org/doc/html/latest/filesystems/configfs/configfs.html)

---
