# Lab 13.1: Multi-Stage CI/CD Pipelines

## Objective
Build complex CI/CD pipelines with multiple stages, parallel jobs, and conditional execution.

## Prerequisites
- GitHub account
- Completed Module 6 (CI/CD Basics)

## Learning Objectives
- Create multi-stage pipelines
- Run jobs in parallel
- Implement conditional execution
- Use job dependencies

---

## Part 1: Multi-Stage Pipeline

```yaml
# .github/workflows/multi-stage.yaml
name: Multi-Stage Pipeline

on: [push]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run linter
        run: echo "Linting code..."
  
  test:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: echo "Running tests..."
  
  build:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v3
      - name: Build application
        run: echo "Building..."
  
  deploy-staging:
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/develop'
    steps:
      - name: Deploy to staging
        run: echo "Deploying to staging..."
  
  deploy-prod:
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to production
        run: echo "Deploying to production..."
```

---

## Part 2: Parallel Execution

```yaml
jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        node: [14, 16, 18]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: ${{ matrix.node }}
      - run: npm test
```

---

## Part 3: Reusable Workflows

```yaml
# .github/workflows/reusable-deploy.yaml
name: Reusable Deploy

on:
  workflow_call:
    inputs:
      environment:
        required: true
        type: string

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to ${{ inputs.environment }}
        run: echo "Deploying to ${{ inputs.environment }}"
```

```yaml
# .github/workflows/main.yaml
name: Main Pipeline

on: [push]

jobs:
  deploy-staging:
    uses: ./.github/workflows/reusable-deploy.yaml
    with:
      environment: staging
```

---

## Success Criteria

✅ Multi-stage pipeline with dependencies  
✅ Parallel job execution  
✅ Conditional deployment  
✅ Reusable workflows  

**Estimated Time:** 40 minutes  
**Difficulty:** Intermediate
