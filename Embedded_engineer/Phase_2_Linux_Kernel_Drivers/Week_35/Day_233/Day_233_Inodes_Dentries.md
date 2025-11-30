# Day 233: Inodes, Dentries, and Files
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
1.  **Distinguish** between `inode`, `dentry`, and `file`.
2.  **Implement** `inode_operations` (`create`, `lookup`, `mkdir`).
3.  **Implement** `file_operations` (`read`, `write`, `readdir`).
4.  **Create** a file inside our custom filesystem.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC.
*   **Software Required:**
    *   Kernel Source.
*   **Prior Knowledge:**
    *   Day 232 (VFS Basics).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Trinity
*   **Inode (`struct inode`):** The physical object. Contains permissions, size, timestamps, and pointers to data blocks. Unique by (Device, Inode Number).
*   **Dentry (`struct dentry`):** The logical path. Maps a name ("home") to an inode. Cached in the Dentry Cache (dcache) for performance. Hard links = Multiple dentries pointing to one inode.
*   **File (`struct file`):** An active session. Created on `open()`. Stores the current offset (`f_pos`).

### 🔹 Part 2: The Lookup Process
*   User: `open("/a/b")`.
*   VFS:
    1.  Look for "a" in Root Dentry.
    2.  If not in cache, call `dir->i_op->lookup("a")`.
    3.  FS reads disk, creates inode for "a", creates dentry.
    4.  Repeat for "b".

---

## 💻 Implementation: Creating Files

> **Instruction:** Extend `myfs` to support creating files and directories.

### 👨‍💻 Code Implementation

```c
static const struct inode_operations myfs_dir_inode_ops;
static const struct file_operations myfs_file_ops;

static struct inode *myfs_get_inode(struct super_block *sb,
                                    const struct inode *dir, umode_t mode) {
    struct inode *inode = new_inode(sb);
    if (!inode) return NULL;

    inode->i_ino = get_next_ino();
    inode_init_owner(&init_user_ns, inode, dir, mode);
    inode->i_atime = inode->i_mtime = inode->i_ctime = current_time(inode);

    if (S_ISDIR(mode)) {
        inode->i_op = &myfs_dir_inode_ops;
        inode->i_fop = &simple_dir_operations;
        inc_nlink(inode); // . and ..
    } else {
        inode->i_op = &simple_symlink_inode_operations; // Placeholder
        inode->i_fop = &myfs_file_ops;
    }
    return inode;
}

// Create File
static int myfs_create(struct user_namespace *mnt_userns, struct inode *dir,
                       struct dentry *dentry, umode_t mode, bool excl) {
    struct inode *inode = myfs_get_inode(dir->i_sb, dir, mode | S_IFREG);
    if (!inode) return -ENOMEM;
    
    d_instantiate(dentry, inode); // Link dentry to inode
    return 0;
}

// Create Directory
static int myfs_mkdir(struct user_namespace *mnt_userns, struct inode *dir,
                      struct dentry *dentry, umode_t mode) {
    struct inode *inode = myfs_get_inode(dir->i_sb, dir, mode | S_IFDIR);
    if (!inode) return -ENOMEM;
    
    inc_nlink(dir); // Update parent link count
    d_instantiate(dentry, inode);
    return 0;
}

static const struct inode_operations myfs_dir_inode_ops = {
    .create = myfs_create,
    .lookup = simple_lookup, // RAM-based lookup
    .link   = simple_link,
    .unlink = simple_unlink,
    .mkdir  = myfs_mkdir,
    .rmdir  = simple_rmdir,
};
```

---

## 🔬 Lab Exercise: Lab 233.1 - File Operations

### 1. Lab Objectives
- Mount `myfs`.
- Create files/dirs.

### 2. Step-by-Step Guide
1.  **Mount:** `mount -t myfs none /mnt`.
2.  **Create:**
    ```bash
    touch /mnt/file1
    mkdir /mnt/dir1
    touch /mnt/dir1/file2
    ```
3.  **Verify:** `ls -R /mnt`.
4.  **Persistence:** Unmount and Remount.
    *   **Result:** Files are GONE! Why?
    *   **Reason:** We are using `simple_lookup` and `new_inode` which only exist in RAM (VFS Cache). We haven't written anything to backing store.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Implementing Read/Write
- **Goal:** Store data in RAM.
- **Task:**
    *   Use `simple_read_from_buffer` and `simple_write_to_buffer`.
    *   Store data in `inode->i_private` (alloc a buffer).
    *   Note: This is basically `ramfs`.

### Lab 3: Dentry Operations
- **Goal:** Case-insensitive lookup.
- **Task:**
    *   Implement `d_compare` and `d_hash` in `dentry_operations`.
    *   Register via `sb->s_d_op`.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Operation not supported"
*   **Cause:** `i_op->create` or `i_op->mkdir` is NULL.
*   **Fix:** Ensure you assign `myfs_dir_inode_ops` to directory inodes.

#### 2. Link count errors
*   **Cause:** Not calling `inc_nlink` / `drop_nlink`.
*   **Fix:** Directories need correct link counts for `ls -l` and `find` to work.

---

## ⚡ Optimization & Best Practices

### RCU Path Walking
*   VFS tries to do lookup without taking locks (RCU).
*   Your `lookup` function should be fast.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is `d_instantiate`?
    *   **A:** It binds a newly allocated inode to a dentry. Used during creation.
2.  **Q:** Why does `simple_lookup` work?
    *   **A:** For RAM filesystems, the dentry cache *is* the filesystem. If it's in the dentry tree, it exists. If not, it doesn't.

### Challenge Task
> **Task:** "The File Size Limit".
> *   Modify `myfs_create` to allocate a 1KB buffer for the file.
> *   Implement `write` to fail if > 1KB.
> *   Implement `read` to read from that buffer.

---

## 📚 Further Reading & References
- [Kernel Source: fs/ramfs/inode.c](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/fs/ramfs/inode.c) (The simplest reference).

---
