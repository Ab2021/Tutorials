# Day 22 (Part 1): Advanced Data Pipelines

> **Phase**: 6 - Deep Dive
> **Topic**: Orchestration at Scale
> **Focus**: Airflow Internals, Backfills, and Quality
> **Reading Time**: 60 mins

---

## 1. Airflow Executors

How does Airflow actually run tasks?

### 1.1 Celery Executor
*   **Architecture**: Master pushes tasks to a Redis Queue. Workers pull from Queue.
*   **Pros**: Low latency. High throughput.
*   **Cons**: Workers are static. If you need a GPU for 1 task, you must provision a GPU worker that sits idle mostly.

### 1.2 Kubernetes Executor
*   **Architecture**: For *every* task, spin up a new Pod.
*   **Pros**: Isolation. Dynamic resource allocation (Request GPU only for training task).
*   **Cons**: Pod startup latency (30s).

---

## 2. The Backfill Pattern

### 2.1 Idempotency is King
*   **Rule**: `run(date)` must produce same output if run 10 times.
*   **Pattern**: `DELETE FROM target WHERE date = '2023-01-01'; INSERT INTO target ...`
*   **Danger**: `INSERT INTO target ...` (Duplicates).

### 2.2 The "Start Date" Trap
*   Airflow schedules the *end* of the interval.
*   `execution_date` 2023-01-01 runs on 2023-01-02 (once data is ready).
*   **Confusion**: In Airflow 2.x, renamed to `logical_date` and `data_interval_start`.

---

## 3. Tricky Interview Questions

### Q1: How to handle a DAG with a cycle?
> **Answer**: You can't. It's a Directed *Acyclic* Graph.
> *   **Fix**: Break the cycle.
> *   If A -> B -> A: Create a new DAG C that triggers A then B. Or use a Sensor in A waiting for B (still risky).

### Q2: Explain "Sensor" Deadlock.
> **Answer**:
> *   You have 10 worker slots.
> *   You schedule 10 DAGs. Each starts with a `ExternalTaskSensor` waiting for another DAG.
> *   All 10 slots are occupied by Sensors waiting. No slots left for the actual tasks to run.
> *   **Fix**: Use `mode='reschedule'`. The sensor checks, then releases the worker slot and goes to sleep.

### Q3: Push vs Pull model in Pipelines?
> **Answer**:
> *   **Push (Airflow)**: Central scheduler triggers jobs. Good for dependencies.
> *   **Pull (Kafka)**: Services react to events. Good for decoupling and real-time.

---

## 4. Practical Edge Case: Late Arriving Data
*   **Problem**: You run daily job at 1 AM. Data from 11:59 PM arrives at 1:05 AM. Missed.
*   **Fix**:
    1.  **Watermark**: Wait until "99% of data usually arrives" (e.g., 4 AM).
    2.  **Upstream Trigger**: Don't run on time. Run when the "Data Landing" job completes.

