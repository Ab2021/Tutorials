# Lab 13.9: Pipeline Optimization

## Objective
Optimize CI/CD pipelines for speed and efficiency.

## Learning Objectives
- Implement caching strategies
- Use parallel execution
- Optimize Docker builds
- Reduce pipeline time

---

## Caching

```yaml
name: Optimized Build

on: [push]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'
      
      - name: Cache Docker layers
        uses: actions/cache@v3
        with:
          path: /tmp/.buildx-cache
          key: ${{ runner.os }}-buildx-${{ github.sha }}
          restore-keys: |
            ${{ runner.os }}-buildx-
      
      - run: npm ci
      - run: npm run build
```

## Parallel Jobs

```yaml
jobs:
  test:
    strategy:
      matrix:
        node: [14, 16, 18]
        os: [ubuntu-latest, windows-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: ${{ matrix.node }}
      - run: npm test
```

## Docker Build Optimization

```yaml
      - name: Build with BuildKit
        run: |
          DOCKER_BUILDKIT=1 docker build \
            --cache-from=myapp:latest \
            --build-arg BUILDKIT_INLINE_CACHE=1 \
            -t myapp:${{ github.sha }} .
```

## Conditional Steps

```yaml
      - name: Run expensive test
        if: github.ref == 'refs/heads/main'
        run: npm run test:e2e
      
      - name: Deploy
        if: github.event_name == 'push' && github.ref == 'refs/heads/main'
        run: ./deploy.sh
```

## Success Criteria
✅ Caching implemented  
✅ Jobs running in parallel  
✅ Build time reduced >50%  
✅ Conditional execution working  

**Time:** 40 min
