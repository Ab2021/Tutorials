# Advanced CI/CD Pipelines

## 🎯 Learning Objectives

By the end of this module, you will have a comprehensive understanding of advanced CI/CD patterns, including:
- **Complex Workflows**: Implementing Matrix Builds, Fan-out/Fan-in patterns, and Conditional execution.
- **Deployment Strategies**: Automating Blue/Green and Canary deployments.
- **Optimization**: Speeding up pipelines with advanced caching and self-hosted runners.
- **Security**: Using OIDC for passwordless cloud authentication and implementing SAST scanning.
- **GitOps**: Introduction to managing infrastructure state via Git.

---

## 📖 Theoretical Concepts

### 1. Advanced Pipeline Patterns

- **Matrix Builds**: Run the same job across multiple configurations (e.g., Node 14, 16, 18 on Ubuntu, Windows, macOS).
- **Fan-out / Fan-in**: Run multiple jobs in parallel (Lint, Unit Test, Integration Test) and wait for all to succeed before proceeding (Deploy).
- **Path Filtering**: Only run backend tests if backend code changed (`paths: ['backend/**']`).

### 2. Deployment Strategies

- **Rolling Deployment**: Replace N instances at a time. Zero downtime. Default in K8s.
- **Blue/Green**: Deploy new version (Green) alongside old (Blue). Switch traffic 100% when ready. Instant rollback.
- **Canary**: Roll out to a small % of users (e.g., 5%). Monitor metrics. Gradually increase.

### 3. Pipeline Optimization

- **Caching**: Don't download `node_modules` every time. Cache based on `package-lock.json` hash.
- **Docker Layer Caching**: Reuse intermediate layers from previous builds.
- **Self-Hosted Runners**: Run jobs on your own powerful EC2 instances instead of shared, slow GitHub runners.

### 4. Security & OIDC

- **Secrets Management**: Never print secrets to logs. Mask them.
- **OIDC (OpenID Connect)**: Instead of storing long-lived AWS Access Keys in GitHub Secrets, use OIDC to request a temporary token. Much more secure.
- **SAST (Static Application Security Testing)**: Scan code for vulnerabilities (SQLi, XSS) *before* building.

---

## 🔧 Practical Examples

### Matrix Build Strategy

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        node: [14, 16, 18]
        os: [ubuntu-latest, windows-latest]
    steps:
      - uses: actions/setup-node@v3
        with:
          node-version: ${{ matrix.node }}
```

### Caching Dependencies

```yaml
- name: Cache node modules
  uses: actions/cache@v3
  with:
    path: ~/.npm
    key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}
    restore-keys: |
      ${{ runner.os }}-node-
```

### OIDC Authentication (AWS)

```yaml
permissions:
  id-token: write
  contents: read

steps:
  - name: Configure AWS Credentials
    uses: aws-actions/configure-aws-credentials@v2
    with:
      role-to-assume: arn:aws:iam::123456789012:role/GitHubActionRole
      aws-region: us-east-1
```

---

## 🎯 Hands-on Labs

- [Lab 13.1: Multi Stage Pipelines](./labs/lab-13.1-multi-stage-pipelines.md)
- [Lab 13.2: Parallel Jobs](./labs/lab-13.2-parallel-jobs.md)
- [Lab 13.3: Matrix Builds](./labs/lab-13.3-matrix-builds.md)
- [Lab 13.4: Integration Testing](./labs/lab-13.4-integration-testing.md)
- [Lab 13.5: Deployment Strategies](./labs/lab-13.5-deployment-strategies.md)
- [Lab 13.6: Pipeline Optimization](./labs/lab-13.6-pipeline-optimization.md)
- [Lab 13.7: Caching Strategies](./labs/lab-13.7-caching-strategies.md)
- [Lab 13.8: Self Hosted Runners](./labs/lab-13.8-self-hosted-runners.md)
- [Lab 13.9: Pipeline Security](./labs/lab-13.9-pipeline-security.md)
- [Lab 13.10: Advanced Workflows](./labs/lab-13.10-advanced-workflows.md)

---

## 📚 Additional Resources

### Official Documentation
- [GitHub Actions: Using a Matrix](https://docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-jobs)
- [Security Hardening for GitHub Actions](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions)

### Tools
- [Act](https://github.com/nektos/act) - Run GitHub Actions locally.

---

## 🔑 Key Takeaways

1.  **Parallelize**: If jobs don't depend on each other, run them at the same time.
2.  **Cache Aggressively**: Network calls are slow. Disk is fast.
3.  **Shift Left**: Security and Testing should happen as early as possible (in the PR).
4.  **Ephemeral Credentials**: Use OIDC. Stop rotating static keys.

---

## ⏭️ Next Steps

1.  Complete the labs to optimize your pipelines.
2.  Proceed to **[Module 14: Advanced Infrastructure as Code](../module-14-infrastructure-as-code-advanced/README.md)** to manage complex cloud environments.
