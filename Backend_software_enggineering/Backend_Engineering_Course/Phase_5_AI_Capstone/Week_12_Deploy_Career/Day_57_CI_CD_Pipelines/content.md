# Day 57: CI/CD Pipelines

## 1. The Robot Butler

Don't deploy by hand (`ssh server`, `git pull`, `restart`).
Let the robot do it.

### 1.1 CI (Continuous Integration)
*   **Trigger**: Push to `main` or Pull Request.
*   **Action**: Lint -> Test -> Build.
*   **Goal**: Catch bugs *before* merging.

### 1.2 CD (Continuous Deployment)
*   **Trigger**: Merge to `main`.
*   **Action**: Push Docker Image -> Update Server.
*   **Goal**: Ship fast.

---

## 2. GitHub Actions

*   **Workflow**: A YAML file in `.github/workflows/`.
*   **Job**: A set of steps running on a runner (e.g., `ubuntu-latest`).
*   **Step**: A command (`npm test`) or an Action (`actions/checkout`).

---

## 3. Pipeline Stages

1.  **Checkout**: Get code.
2.  **Lint**: Check style (Black, ESLint).
3.  **Test**: Run Unit/Integration tests (Pytest).
4.  **Build**: `docker build`.
5.  **Push**: `docker push` (to Docker Hub/ECR).
6.  **Deploy**: SSH to server and update.

---

## 4. Secrets

Never commit secrets.
*   **GitHub Secrets**: Store `DOCKER_PASSWORD`, `SSH_KEY` in Repo Settings.
*   **Usage**: `${{ secrets.DOCKER_PASSWORD }}`.

---

## 5. Summary

Today we automated the boring stuff.
*   **CI**: Safety net.
*   **CD**: Speed.
*   **GitHub Actions**: The standard tool.

**Tomorrow (Day 58)**: We prepare for the **System Design Interview**.
