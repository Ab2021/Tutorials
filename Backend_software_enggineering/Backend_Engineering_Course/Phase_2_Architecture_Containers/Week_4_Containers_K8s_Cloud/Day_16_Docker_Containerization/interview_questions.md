# Day 16: Interview Questions & Answers

## Conceptual Questions

### Q1: How does Docker actually work? (Under the hood)
**Answer:**
Docker uses Linux Kernel features:
1.  **Namespaces**: Provide isolation.
    *   *PID Namespace*: Process isolation (Container sees only its own processes).
    *   *NET Namespace*: Network isolation (Own IP/Port).
    *   *MNT Namespace*: Filesystem isolation.
2.  **Cgroups (Control Groups)**: Resource limiting.
    *   "This container can only use 512MB RAM and 1 CPU core."
3.  **Union File System (OverlayFS)**: Layering. Allows multiple containers to share the same read-only base image layers.

### Q2: What is the difference between `CMD` and `ENTRYPOINT`?
**Answer:**
*   **ENTRYPOINT**: The executable to run. Hard to override.
*   **CMD**: Default arguments to the ENTRYPOINT. Easy to override.
*   *Pattern*:
    ```dockerfile
    ENTRYPOINT ["python"]
    CMD ["app.py"]
    ```
    *   `docker run myimage` -> `python app.py`
    *   `docker run myimage script.py` -> `python script.py` (Overrides CMD).

### Q3: Explain the difference between `COPY` and `ADD`.
**Answer:**
*   **COPY**: Copies local files to the container. (Preferred).
*   **ADD**: Can do what COPY does, BUT also:
    1.  Downloads URLs (`ADD http://...`).
    2.  Auto-extracts tarballs (`ADD archive.tar.gz /`).
*   *Best Practice*: Use `COPY` unless you explicitly need the auto-extraction magic of `ADD`.

---

## Scenario-Based Questions

### Q4: Your CI/CD pipeline is slow because the Docker build takes 10 minutes. How do you speed it up?
**Answer:**
1.  **Layer Caching**: Ensure `COPY requirements.txt` and `RUN pip install` happen *before* `COPY .`. This prevents re-installing deps when code changes.
2.  **Dockerignore**: Add `.dockerignore` to exclude `.git`, `node_modules`, and temp files from the build context.
3.  **BuildKit**: Enable Docker BuildKit (`DOCKER_BUILDKIT=1`) for parallel layer building.

### Q5: You need to debug a crashing container in production. It exits immediately so you can't `exec` into it. What do you do?
**Answer:**
1.  **Logs**: `docker logs <container_id>`.
2.  **Override Entrypoint**: `docker run -it --entrypoint /bin/sh myimage`. This starts a shell instead of the crashing app, allowing me to inspect the filesystem.
3.  **Inspect**: `docker inspect <container_id>` to check exit code and OOM (Out of Memory) status.

---

## Behavioral / Role-Specific Questions

### Q6: A developer wants to run a database inside a Docker container in production. Is this a good idea?
**Answer:**
**It depends, but generally risky.**
*   **Pros**: Easy setup, consistent dev/prod parity.
*   **Cons**:
    *   **Data Persistence**: If the container dies and you didn't mount a volume correctly, data is gone.
    *   **Performance**: Slight overhead on I/O.
    *   **Complexity**: Managing HA/Failover/Backups for a stateful container in K8s is hard (StatefulSets).
*   **Recommendation**: Use a Managed Database (RDS/Cloud SQL) for production. Use Docker DB only for local dev and testing.
