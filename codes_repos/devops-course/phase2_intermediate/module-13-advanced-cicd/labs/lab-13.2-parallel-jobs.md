# Lab 13.2: Parallel Jobs and Matrix Builds

## Objective
Optimize CI/CD pipelines using parallel execution and matrix strategies.

## Prerequisites
- GitHub Actions knowledge

## Learning Objectives
- Run jobs in parallel
- Use matrix builds for multi-platform testing
- Optimize pipeline execution time
- Handle job failures

---

## Matrix Strategy

```yaml
name: Matrix Build

on: [push]

jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python: ['3.8', '3.9', '3.10', '3.11']
      fail-fast: false
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python }}
      - run: pip install pytest
      - run: pytest
```

## Parallel Jobs

```yaml
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - run: echo "Linting..."
  
  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo "Testing..."
  
  security:
    runs-on: ubuntu-latest
    steps:
      - run: echo "Security scan..."
```

## Success Criteria

✅ Matrix builds across platforms  
✅ Parallel job execution  
✅ Optimized pipeline time  

**Estimated Time:** 30 minutes
