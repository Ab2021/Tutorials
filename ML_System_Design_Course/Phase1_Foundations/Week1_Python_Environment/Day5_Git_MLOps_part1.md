# Day 5 (Part 1): Advanced Git & MLOps Internals

> **Phase**: 6 - Deep Dive
> **Topic**: Version Control Under the Hood
> **Focus**: Git Internals, DVC Architecture, and CI/CD
> **Reading Time**: 60 mins

---

## 1. Git Internals: The Graph

Git is not a file system. It is a content-addressable filesystem on top of a Directed Acyclic Graph (DAG).

### 1.1 The Objects
*   **Blob**: The file content. (Compressed).
*   **Tree**: A directory listing. Maps filenames to Blobs or other Trees.
*   **Commit**: A snapshot. Points to a Tree, Parent Commit(s), Author, and Message.

### 1.2 The SHA-1 Hash
*   Everything is identified by the hash of its content.
*   If you change one bit in a file -> New Blob Hash -> New Tree Hash -> New Commit Hash.
*   This guarantees integrity.

### 1.3 Rebase vs. Merge Internals
*   **Merge**: Creates a new "Merge Commit" with two parents. Preserves history exactly as it happened.
*   **Rebase**: "Replays" your commits on top of the target branch.
    *   It actually creates *new* commits (new hashes) with the same content.
    *   **Danger**: Never rebase a public branch. You rewrite history, forcing everyone else to force pull.

---

## 2. DVC Internals: Data Version Control

How does DVC version 100GB files without exploding Git?

### 2.1 Content Addressable Storage (CAS)
*   When you `dvc add data.csv`:
    1.  Calculates MD5 hash of `data.csv`.
    2.  Moves `data.csv` to `.dvc/cache/MD5_HASH`.
    3.  Creates a reflink (or symlink/hardlink) from workspace to cache.
    4.  Creates `data.csv.dvc` (small text file) containing the hash.
*   **Git tracks**: `data.csv.dvc`.
*   **DVC tracks**: The cache.

### 2.2 Reflinks (Copy-on-Write)
*   On modern file systems (APFS, XFS, Btrfs), DVC uses reflinks.
*   The file in workspace and cache share the same physical blocks on disk.
*   **Benefit**: Instant "copying". Zero extra storage.

---

## 3. CI/CD for ML (CML)

### 3.1 The Workflow
1.  **Pull Request**: Developer pushes code.
2.  **Runner**: GitHub Actions / GitLab Runner spins up a GPU instance.
3.  **Train**: `dvc pull` data -> `python train.py`.
4.  **Report**: Generate `metrics.txt` and `confusion_matrix.png`.
5.  **Comment**: CML bot posts the report as a comment on the PR.
    *   "Accuracy changed from 85% -> 87%."

---

## 4. Tricky Interview Questions

### Q1: What is a "Detached HEAD" state?
> **Answer**: HEAD usually points to a Branch Name (ref). A Branch Name points to a Commit.
> *   **Detached**: HEAD points directly to a Commit hash.
> *   **Implication**: If you make new commits, they have no branch name associated. If you switch away, the Garbage Collector will eventually delete them (dangling commits).

### Q2: How does `git gc` work?
> **Answer**: Git Garbage Collection.
> 1.  **Pruning**: Deletes objects that are not reachable from any Ref (Branch/Tag).
> 2.  **Packing**: Compresses individual object files into "Packfiles" (delta compression) to save disk space.

### Q3: Design a branching strategy for a 50-person ML team.
> **Answer**: **Trunk-Based Development** (Scaled).
> *   **Main**: Always deployable.
> *   **Feature Branches**: Short-lived (1-2 days).
> *   **Feature Flags**: Merge incomplete code hidden behind flags rather than long-lived branches.
> *   **Model Registry**: Decouple code release from model release. Code merges to Main. Model is promoted from Staging -> Prod in the Registry (MLflow).

---

## 5. Practical Edge Case: Large File Storage (LFS) vs DVC
*   **Git LFS**: Stores pointers in Git. Downloads files lazily.
    *   *Pros*: Integrated into `git clone`.
    *   *Cons*: Server-side storage limits (GitHub charges $$). Hard to use with S3/GCP buckets.
*   **DVC**: Agnostic.
    *   *Pros*: Use your own S3 bucket. Supports data pipelines (DAGs).
    *   *Cons*: Extra tool to install.

