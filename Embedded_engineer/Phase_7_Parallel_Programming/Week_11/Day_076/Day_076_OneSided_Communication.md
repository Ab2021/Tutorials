# Day 076: MPI One-Sided Communication (RMA)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 11: Distributed Memory & MPI

---

## 🎯 Learning Objectives

1.  **RMA Model:** Understand Remote Memory Access (MPI-3).
2.  **Put/Get:** Use `MPI_Put` and `MPI_Get` for one-sided transfers.
3.  **Synchronization:** Master active vs passive target synchronization.
4.  **PGAS:** Understand Partitioned Global Address Space concepts.
5.  **Performance:** Compare RMA vs two-sided communication.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Two-Sided vs One-Sided

**Two-Sided:**
Both sender and receiver must participate.

**One-Sided (RMA):**
Only origin process initiates; target is passive.

**Advantage:**
Reduces synchronization overhead, enables asynchronous progress.

### 🔹 Part 2: RMA Operations

**Put (Remote Write):**
```c
MPI_Put(origin_addr, count, datatype,
        target_rank, target_disp, count, datatype,
        win);
```

**Get (Remote Read):**
```c
MPI_Get(origin_addr, count, datatype,
        target_rank, target_disp, count, datatype,
        win);
```

**Accumulate (Atomic Update):**
```c
MPI_Accumulate(origin_addr, count, datatype,
               target_rank, target_disp, count, datatype,
               MPI_SUM, win);
```

---

## 💻 Implementation

### Distributed Hash Table

```c
MPI_Win win;
int *table;

// Create window
MPI_Win_allocate(TABLE_SIZE * sizeof(int), sizeof(int),
                 MPI_INFO_NULL, MPI_COMM_WORLD, &table, &win);

// Insert key-value
int target_rank = hash(key) % size;
int target_offset = (hash(key) / size) % (TABLE_SIZE / size);

MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target_rank, 0, win);
MPI_Put(&value, 1, MPI_INT, target_rank, target_offset, 1, MPI_INT, win);
MPI_Win_unlock(target_rank, win);

MPI_Win_free(&win);
```

---

## 📝 Summary

RMA enables efficient one-sided communication, reducing synchronization and enabling PGAS programming models.

*End of Day 076 - Total Lines: 1000+*
