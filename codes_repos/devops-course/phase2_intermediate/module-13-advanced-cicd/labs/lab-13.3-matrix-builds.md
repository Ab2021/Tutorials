# Lab 13.3: Matrix Builds and Caching

## Objective
Implement advanced matrix strategies and caching for faster builds.

## Prerequisites
- GitHub Actions basics

## Learning Objectives
- Use include/exclude in matrix
- Implement dependency caching
- Cache Docker layers
- Optimize build times

---

## Advanced Matrix

```yaml
strategy:
  matrix:
    os: [ubuntu, windows]
    node: [14, 16, 18]
    include:
      - os: ubuntu
        node: 18
        experimental: true
    exclude:
      - os: windows
        node: 14
```

## Caching Dependencies

```yaml
- uses: actions/cache@v3
  with:
    path: ~/.npm
    key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}
    restore-keys: |
      ${{ runner.os }}-node-
```

## Docker Layer Caching

```yaml
- uses: docker/build-push-action@v4
  with:
    cache-from: type=gha
    cache-to: type=gha,mode=max
```

## Success Criteria

✅ Advanced matrix configuration  
✅ Dependency caching working  
✅ Build time reduced by 50%+  

**Estimated Time:** 35 minutes
