# Lab 11.9: Docker Resource Limits

## Objective
Control container resource usage with limits and reservations.

## Learning Objectives
- Set CPU limits
- Configure memory limits
- Use resource reservations
- Monitor resource usage

---

## Memory Limits

```bash
# Hard limit
docker run -d \
  --memory="512m" \
  --memory-swap="1g" \
  nginx

# Memory reservation
docker run -d \
  --memory="512m" \
  --memory-reservation="256m" \
  nginx

# OOM kill disable
docker run -d \
  --memory="512m" \
  --oom-kill-disable \
  nginx
```

## CPU Limits

```bash
# CPU shares (relative weight)
docker run -d \
  --cpu-shares=512 \
  nginx

# CPU quota (absolute limit)
docker run -d \
  --cpus="1.5" \
  nginx

# CPU pinning
docker run -d \
  --cpuset-cpus="0,1" \
  nginx
```

## Compose with Limits

```yaml
services:
  web:
    image: nginx
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M
```

## Monitor Usage

```bash
# Stats
docker stats

# Specific container
docker stats mycontainer

# No stream
docker stats --no-stream
```

## Success Criteria
✅ Memory limits enforced  
✅ CPU limits working  
✅ Resource usage monitored  

**Time:** 35 min
