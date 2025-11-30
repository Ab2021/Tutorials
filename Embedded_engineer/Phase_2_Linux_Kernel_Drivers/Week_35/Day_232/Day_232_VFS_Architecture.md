# Day 232: VFS Architecture & Superblocks
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
1.  **Explain** the role of VFS (Virtual Filesystem Switch).
2.  **Understand** the `file_system_type` and `super_block` structures.
3.  **Register** a new filesystem type.
4.  **Mount** a dummy filesystem.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC.
*   **Software Required:**
    *   Kernel Source.
*   **Prior Knowledge:**
    *   Day 211 (Block Device Basics).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The VFS Abstraction
*   **Problem:** Linux supports Ext4, XFS, Btrfs, FAT, NTFS, NFS, etc. Userspace (`open`, `read`, `write`) shouldn't care which one is used.
*   **Solution:** VFS. A common interface layer.
*   **Objects:**
    *   `super_block`: Represents a mounted filesystem.
    *   `inode`: Represents a file (metadata).
    *   `dentry`: Represents a directory entry (name -> inode mapping).
    *   `file`: Represents an open file descriptor.

### 🔹 Part 2: The Superblock
*   Stores global info: Block size, Magic number, Root inode.
*   **Operations (`s_op`):** `alloc_inode`, `write_inode`, `sync_fs`.

---

## 💻 Implementation: The "MyFS" Registration

> **Instruction:** Create a module that registers a filesystem type named "myfs".

### 👨‍💻 Code Implementation

```c
#include <linux/module.h>
#include <linux/fs.h>

#define MYFS_MAGIC 0x12345678

static struct dentry *myfs_mount(struct file_system_type *fs_type,
                                 int flags, const char *dev_name,
                                 void *data) {
    // Mount a "pseudo" filesystem (no block device backing)
    return mount_nodev(fs_type, flags, data, myfs_fill_super);
}

static int myfs_fill_super(struct super_block *sb, void *data, int silent) {
    struct inode *root;
    
    sb->s_blocksize = PAGE_SIZE;
    sb->s_blocksize_bits = PAGE_SHIFT;
    sb->s_magic = MYFS_MAGIC;
    sb->s_op = &myfs_sops; // We need to define this
    
    // Create Root Inode
    root = new_inode(sb);
    if (!root) return -ENOMEM;
    
    root->i_ino = 1;
    root->i_mode = S_IFDIR | 0755;
    root->i_atime = root->i_mtime = root->i_ctime = current_time(root);
    root->i_op = &simple_dir_inode_operations;
    root->i_fop = &simple_dir_operations;
    
    sb->s_root = d_make_root(root);
    if (!sb->s_root) return -ENOMEM;
    
    return 0;
}

static const struct super_operations myfs_sops = {
    .statfs = simple_statfs,
    .drop_inode = generic_delete_inode,
};

static struct file_system_type myfs_type = {
    .owner = THIS_MODULE,
    .name = "myfs",
    .mount = myfs_mount,
    .kill_sb = kill_litter_super,
    .fs_flags = FS_USERNS_MOUNT, // Allow mounting in user namespace
};

static int __init myfs_init(void) {
    return register_filesystem(&myfs_type);
}

static void __exit myfs_exit(void) {
    unregister_filesystem(&myfs_type);
}

module_init(myfs_init);
module_exit(myfs_exit);
MODULE_LICENSE("GPL");
```

---

## 🔬 Lab Exercise: Lab 232.1 - Mounting MyFS

### 1. Lab Objectives
- Mount the new filesystem.
- Verify it exists.

### 2. Step-by-Step Guide
1.  **Load:** `insmod myfs.ko`.
2.  **Mount:**
    ```bash
    mkdir /mnt/myfs
    mount -t myfs none /mnt/myfs
    ```
3.  **Verify:**
    ```bash
    mount | grep myfs
    ls -la /mnt/myfs
    ```
    *   Should see an empty directory (the root inode).

---

## 🧪 Additional / Advanced Labs

### Lab 2: Magic Number
- **Goal:** Verify magic number.
- **Task:**
    *   `stat -f /mnt/myfs`.
    *   Check "ID: ... Type: ...". (Type might show as unknown or hex 12345678).

### Lab 3: Block Device Backing
- **Goal:** Use `mount_bdev` instead of `mount_nodev`.
- **Task:**
    *   Change `mount` callback.
    *   Requires a real block device (e.g., `/dev/ram0`).
    *   `fill_super` must now read from the device (we'll do this in Day 234).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "No such device"
*   **Cause:** Filesystem type not registered.
*   **Fix:** Check `cat /proc/filesystems`.

#### 2. Mount fails
*   **Cause:** `fill_super` returned error (e.g., -ENOMEM).
*   **Fix:** Check `dmesg`.

---

## ⚡ Optimization & Best Practices

### `kill_sb`
*   Use `kill_litter_super` for RAM-based FS (cleans up everything).
*   Use `kill_block_super` for Block-based FS (closes block device).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the difference between `mount_nodev` and `mount_bdev`?
    *   **A:** `mount_nodev` is for filesystems that don't need a block device (procfs, sysfs, ramfs). `mount_bdev` opens the block device and passes it to `fill_super`.
2.  **Q:** What is `d_make_root`?
    *   **A:** Allocates the root dentry for the root inode.

### Challenge Task
> **Task:** "The Parameterized Mount".
> *   Parse mount options (e.g., `mount -t myfs -o mode=0777 ...`).
> *   Use `fs_parse` (new API) or manual string parsing in `fill_super` (via `data` argument).
> *   Apply the mode to the root inode.

---

## 📚 Further Reading & References
- [Kernel Documentation: filesystems/vfs.rst](https://www.kernel.org/doc/html/latest/filesystems/vfs.html)

---
