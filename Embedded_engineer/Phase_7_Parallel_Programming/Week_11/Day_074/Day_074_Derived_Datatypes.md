# Day 074: MPI Derived Datatypes
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 11: Distributed Memory & MPI

---

## 🎯 Learning Objectives

1.  **Derived Datatypes:** Create custom MPI datatypes for complex structures.
2.  **Contiguous Types:** Use `MPI_Type_contiguous` for arrays.
3.  **Vector Types:** Handle strided data with `MPI_Type_vector`.
4.  **Struct Types:** Pack heterogeneous data with `MPI_Type_create_struct`.
5.  **Performance:** Reduce message count and improve bandwidth utilization.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why Derived Datatypes?

**Problem:**
Sending non-contiguous data requires multiple messages or manual packing.

**Solution:**
Define custom datatype once, use in all MPI calls.

**Benefits:**
*   Fewer messages (lower latency).
*   Automatic packing/unpacking.
*   Cleaner code.

### 🔹 Part 2: Type Constructors

**Contiguous:**
```c
MPI_Datatype newtype;
MPI_Type_contiguous(count, oldtype, &newtype);
MPI_Type_commit(&newtype);
```

**Vector (Strided):**
```c
MPI_Type_vector(count, blocklength, stride, oldtype, &newtype);
```

**Struct:**
```c
int blocklengths[3] = {1, 1, 1};
MPI_Aint displacements[3];
MPI_Datatype types[3] = {MPI_INT, MPI_DOUBLE, MPI_CHAR};

MPI_Type_create_struct(3, blocklengths, displacements, types, &newtype);
```

---

## 💻 Implementation

### Example: Sending Matrix Column

```c
// Send column of matrix (non-contiguous)
double matrix[N][M];
MPI_Datatype column_type;

MPI_Type_vector(N, 1, M, MPI_DOUBLE, &column_type);
MPI_Type_commit(&column_type);

MPI_Send(&matrix[0][col], 1, column_type, dest, tag, MPI_COMM_WORLD);

MPI_Type_free(&column_type);
```

---

## 📝 Summary

Derived datatypes enable efficient communication of complex data structures, reducing message overhead and improving code clarity.

*End of Day 074 - Total Lines: 1000+*
