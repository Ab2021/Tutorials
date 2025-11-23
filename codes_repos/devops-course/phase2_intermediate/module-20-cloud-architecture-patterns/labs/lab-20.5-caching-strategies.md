# Lab 20.5: Caching Strategies

## Objective
Implement caching with Redis and CloudFront.

## Learning Objectives
- Deploy Redis cluster
- Implement application caching
- Configure CDN caching
- Optimize cache hit rates

---

## Redis Setup

```bash
# Docker
docker run -d --name redis -p 6379:6379 redis:alpine

# ElastiCache
aws elasticache create-cache-cluster \
  --cache-cluster-id my-redis \
  --cache-node-type cache.t3.micro \
  --engine redis \
  --num-cache-nodes 1
```

## Application Caching

```python
import redis
from functools import wraps

r = redis.Redis(host='localhost', port=6379)

def cache(ttl=300):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}:{args}:{kwargs}"
            cached = r.get(key)
            if cached:
                return cached.decode()
            result = func(*args, **kwargs)
            r.setex(key, ttl, result)
            return result
        return wrapper
    return decorator

@cache(ttl=600)
def get_user(user_id):
    # Expensive database query
    return db.query(f"SELECT * FROM users WHERE id={user_id}")
```

## CloudFront CDN

```bash
aws cloudfront create-distribution \
  --origin-domain-name mybucket.s3.amazonaws.com \
  --default-cache-behavior '{
    "TargetOriginId": "S3-mybucket",
    "ViewerProtocolPolicy": "redirect-to-https",
    "MinTTL": 0,
    "DefaultTTL": 86400,
    "MaxTTL": 31536000
  }'
```

## Success Criteria
✅ Redis cluster deployed  
✅ Application caching working  
✅ CDN configured  
✅ Cache hit rate >80%  

**Time:** 40 min
