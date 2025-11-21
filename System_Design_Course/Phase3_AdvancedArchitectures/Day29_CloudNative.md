# Day 29: Cloud Native Patterns

## 1. The 12-Factor App
A methodology for building SaaS apps.
1.  **Codebase:** One repo, many deploys.
2.  **Dependencies:** Explicitly declare (pip, npm).
3.  **Config:** Store config in Environment Variables (Not code).
4.  **Backing Services:** Treat DB/Cache as attached resources.
5.  **Build, Release, Run:** Strict separation.
6.  **Processes:** Stateless.
7.  **Port Binding:** Export services via port binding.
8.  **Concurrency:** Scale out via process model.
9.  **Disposability:** Fast startup, graceful shutdown.
10. **Dev/Prod Parity:** Keep them similar.
11. **Logs:** Treat logs as event streams.
12. **Admin Processes:** Run admin tasks as one-off processes.

## 2. Service Mesh (Istio/Linkerd)
*   Decouples networking logic from app code.
*   **Features:** mTLS, Traffic Splitting (Canary), Retries, Observability.

## 3. Serverless (FaaS)
*   **Concept:** Upload code (Function). Cloud provider handles servers.
*   **Pros:** Zero maintenance. Pay per ms. Auto-scale to zero.
*   **Cons:** Cold starts. Stateless. Vendor lock-in.

## 4. Immutable Infrastructure
*   Never patch a running server.
*   Build a new Image (Docker/AMI), deploy it, kill the old one.
*   **Benefit:** No configuration drift. Predictable rollbacks.
