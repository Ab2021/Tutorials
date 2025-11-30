# Day 234: Writing a Read-Only Filesystem (RomFS style)
## Phase 2: Linux Kernel & Device Drivers | Week 35: Advanced Storage & Filesystems

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
1.  **Design** a simple on-disk layout (Superblock, Inode Table).
2.  **Implement** `mount_bdev` to read from a block device.
3.  **Implement** `iget` to read inodes from disk.
4.  **Implement** `readdir` to list files from disk.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC.
*   **Software Required:**
    *   Kernel Source.
    *   `dd`, `hexedit`.
*   **Prior Knowledge:**
    *   Day 233 (VFS).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The On-Disk Layout
*   **Superblock (Offset 0):** Magic, Count.
*   **Inode Table:** Array of fixed-size inodes.
*   **Data Blocks:** The actual file content.

### 🔹 Part 2: The Structure
```c
struct myromfs_super {
    __le32 magic;
    __le32 inode_count;
};

struct myromfs_inode {
    __le32 mode;
    __le32 size;
    __le32 data_offset; // Offset to data relative to start of disk
    char name[32];      // Fixed name length for simplicity
};
```

---

## 💻 Implementation: The Reader

> **Instruction:** Implement `fill_super` to read the superblock from the buffer head.

### 👨‍💻 Code Implementation

```c
#include <linux/buffer_head.h>

static int myromfs_fill_super(struct super_block *sb, void *data, int silent) {
    struct buffer_head *bh;
    struct myromfs_super *ds;
    struct inode *root;

    // 1. Read Block 0
    bh = sb_bread(sb, 0);
    if (!bh) return -EIO;
    
    ds = (struct myromfs_super *)bh->b_data;
    if (le32_to_cpu(ds->magic) != MYROMFS_MAGIC) {
        brelse(bh);
        return -EINVAL;
    }
    
    // 2. Store info in sb->s_fs_info
    // ...
    brelse(bh);
    
    // 3. Get Root Inode (Index 0)
    root = myromfs_iget(sb, 0);
    sb->s_root = d_make_root(root);
    return 0;
}

static struct inode *myromfs_iget(struct super_block *sb, unsigned long ino) {
    struct inode *inode;
    struct buffer_head *bh;
    struct myromfs_inode *raw_inode;
    
    // Calculate block and offset
    // Assuming inodes start at Block 1
    
    inode = iget_locked(sb, ino);
    if (!inode) return ERR_PTR(-ENOMEM);
    if (!(inode->i_state & I_NEW)) return inode;
    
    // Read from disk
    bh = sb_bread(sb, 1 + (ino * sizeof(*raw_inode)) / sb->s_blocksize);
    raw_inode = (void *)(bh->b_data + ...);
    
    inode->i_mode = le32_to_cpu(raw_inode->mode);
    inode->i_size = le32_to_cpu(raw_inode->size);
    // ... set ops ...
    
    brelse(bh);
    unlock_new_inode(inode);
    return inode;
}
```

---

## 🔬 Lab Exercise: Lab 234.1 - Creating the Image

### 1. Lab Objectives
- Write a userspace tool (`mkfs.myromfs`) to create a valid image.

### 2. Step-by-Step Guide
1.  **C Program:**
    *   Open `image.bin`.
    *   Write Superblock (Magic).
    *   Write Root Inode (Dir).
    *   Write File Inode.
    *   Write Data.
2.  **Mount:**
    ```bash
    losetup /dev/loop0 image.bin
    mount -t myromfs /dev/loop0 /mnt
    ```
3.  **Verify:** `ls /mnt`.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Readdir
- **Goal:** List files.
- **Task:**
    *   Directory data is just a list of `struct myromfs_dir_entry` (Ino, Name).
    *   Implement `iterate_shared` in `file_operations`.
    *   Loop through the directory data and call `dir_emit`.

### Lab 3: Read File
- **Goal:** `cat /mnt/file`.
- **Task:**
    *   Implement `read_iter` (or `generic_file_read_iter`).
    *   Implement `address_space_operations` (`readpage`).
    *   Map the disk block to the page cache.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Endianness
*   **Cause:** Writing on x86 (LE) and reading on ARM (LE) usually works, but always use `cpu_to_le32` / `le32_to_cpu` for portability.

#### 2. Buffer Head Leaks
*   **Cause:** Forgetting `brelse(bh)`.
*   **Fix:** Always release buffer heads.

---

## ⚡ Optimization & Best Practices

### `mpage_readpage`
*   Use generic helpers where possible to handle block mapping efficiently.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is `sb_bread`?
    *   **A:** SuperBlock Buffer Read. Reads a block from the device associated with the superblock.
2.  **Q:** Why `iget_locked`?
    *   **A:** It checks the inode cache. If found, returns it. If not, allocates a new one and locks it so we can fill it from disk safely.

### Challenge Task
> **Task:** "The Deep Directory".
> *   Support subdirectories.
> *   Your `mkfs` tool must handle recursion.
> *   Your driver must handle `lookup` finding a directory inode.

---

## 📚 Further Reading & References
- [Kernel Source: fs/romfs/](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/fs/romfs/)

---
