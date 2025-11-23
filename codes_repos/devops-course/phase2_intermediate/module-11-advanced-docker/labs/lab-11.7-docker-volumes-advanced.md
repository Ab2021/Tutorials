# Lab 11.7: Docker Volumes Advanced

## Objective
Master Docker volume management for persistent data.

## Learning Objectives
- Create and manage volumes
- Use bind mounts vs volumes
- Implement volume drivers
- Backup and restore volumes

---

## Named Volumes

```bash
# Create volume
docker volume create my-data

# Use in container
docker run -d \
  --name app \
  -v my-data:/app/data \
  nginx

# Inspect volume
docker volume inspect my-data

# List volumes
docker volume ls

# Remove unused volumes
docker volume prune
```

## Bind Mounts

```bash
# Mount host directory
docker run -d \
  -v /host/path:/container/path:ro \
  nginx

# Current directory
docker run -d \
  -v $(pwd):/app \
  node:18
```

## Volume Drivers

```bash
# Use NFS driver
docker volume create \
  --driver local \
  --opt type=nfs \
  --opt o=addr=192.168.1.1,rw \
  --opt device=:/path/to/dir \
  nfs-volume
```

## Backup/Restore

```bash
# Backup
docker run --rm \
  -v my-data:/data \
  -v $(pwd):/backup \
  ubuntu tar czf /backup/data-backup.tar.gz /data

# Restore
docker run --rm \
  -v my-data:/data \
  -v $(pwd):/backup \
  ubuntu tar xzf /backup/data-backup.tar.gz -C /
```

## Success Criteria
✅ Volumes created and managed  
✅ Data persists across restarts  
✅ Backup/restore working  

**Time:** 40 min
