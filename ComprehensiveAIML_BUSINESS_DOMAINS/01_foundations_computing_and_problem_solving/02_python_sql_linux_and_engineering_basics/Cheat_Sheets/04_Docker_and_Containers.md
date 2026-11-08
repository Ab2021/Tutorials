---
collection: "python-sql-linux-engineering-basics"
file_id: "04"
topic: "Docker and containers"
level: 2
status: "draft; unreviewed"
version: "1.0"
created: "2026-11-07"
reviewed: null
source_status: "mapped source title; original body unavailable"
---

# Docker and Containers

> **Source mapping:** `Agents_Stack_Infra/Module_00_Prerequisites/04_docker_and_containers.md` in [`Train/Templates/directory.txt`](../../../../Templates/directory.txt). Original body unavailable; new mapped draft.

## 1. Contract

A container packages a process with declared filesystem, dependency and runtime configuration. It shares the host kernel; it is not a VM, a security boundary by itself, or reproducibility evidence unless image inputs and execution configuration are pinned.

## 2. Build and runtime boundaries

Use a minimal, pinned base image; deterministic dependency resolution with hashes/lock files; non-root runtime user; explicit working directory, entrypoint, environment/config injection and writable paths. Build stages should remove compilers/secrets from runtime artifacts. Never place credentials in an image layer, build argument history or source repository.

An image digest identifies artifact bytes. Tags such as `latest` are mutable pointers. Record base digest, build context revision, dependency lock, build command, image digest, runtime command, environment schema and deployed configuration separately.

## 3. Layer and capacity reasoning

A layer cache improves rebuild time only when its inputs are unchanged. Copy dependency manifests before application source so dependency installation can be cached without incorrectly hiding source changes. Image size is not process memory: resident memory, mapped files, allocator behavior, child processes and concurrent containers determine runtime capacity.

If a service has a 1 GiB memory limit and one request can retain 80 MiB at peak, naive `floor(1024/80)=12` ignores baseline process memory, kernel/runtime overhead, fragmentation and burst concurrency. Reserve measured headroom and test OOM/restart behavior; do not deploy at arithmetic maximum.

## 4. Networking, storage and health

Bind only required ports; validate inbound authentication at the application boundary. Persistent state belongs in declared volumes/datastores with backup/migration policy, not an ephemeral writable layer. A liveness check answers whether restart is appropriate; readiness answers whether traffic may be sent. A process listening on a port is not necessarily ready for migrations, dependencies or model load.

## 5. Failure modes

| Symptom | Check | Mitigation |
|---|---|---|
| Works locally, fails deploy | Image digest, architecture, env/config and mounted-path diff | Reproduce immutable runtime contract |
| Secret exposed | Layer/build log/history scan | Revoke/rotate; use runtime secret injection |
| Container restart loop | Exit code, OOM, readiness/liveness and dependency logs | Fix cause; avoid masking with infinite restart |
| Data disappears | Writable-layer/volume trace | Move durable state to managed persistent store |
| Image rebuild unexpectedly slow | Build context/layer cache inputs | Add `.dockerignore`, order layers, minimize context |

## 6. Validation

Build from clean checkout; run as non-root; inspect image history; scan declared dependencies; test missing/rotated secret, read-only filesystem, graceful shutdown, readiness during dependency/model startup, memory limit, volume recovery and rollback to prior digest. Verify actual artifact digest in deployment telemetry.

## 7. Interview checks

1. **Why pin a digest?** Tag names are mutable; a digest identifies bytes.
2. **Why is a container not a VM?** It shares the host kernel.
3. **Why separate readiness and liveness?** Traffic admission and restart decisions are different.
4. **Why avoid secrets in build args?** They can survive in metadata/layers.
5. **What proves reproducibility?** Pinned inputs plus successful clean rebuild and runtime contract.

## 8. Quick reference

| Decision | Rule |
|---|---|
| Release identity | Image digest plus build/runtime manifest |
| Secrets | Runtime injection; never baked layer/history |
| State | Explicit persistent service/volume policy |
| Capacity | Measure peak under limit; reserve headroom |
| Traffic | Readiness gate, not port-open heuristic |

**Limitations:** No Dockerfile, registry, deployment or source lesson was available. Sizing arithmetic is illustrative.
