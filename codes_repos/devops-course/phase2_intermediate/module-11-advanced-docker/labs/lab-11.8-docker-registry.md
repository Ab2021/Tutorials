# Lab 11.8: Docker Registry

## Objective
Set up and use private Docker registries.

## Learning Objectives
- Run private registry
- Push/pull images
- Secure registry with TLS
- Implement authentication

---

## Run Registry

```bash
# Start registry
docker run -d \
  -p 5000:5000 \
  --name registry \
  -v registry-data:/var/lib/registry \
  registry:2

# Tag image
docker tag myapp:latest localhost:5000/myapp:latest

# Push to registry
docker push localhost:5000/myapp:latest

# Pull from registry
docker pull localhost:5000/myapp:latest
```

## Secure Registry

```bash
# Generate certificates
openssl req -newkey rsa:4096 -nodes -sha256 \
  -keyout domain.key -x509 -days 365 -out domain.crt

# Run with TLS
docker run -d \
  -p 443:443 \
  -v $(pwd)/certs:/certs \
  -e REGISTRY_HTTP_ADDR=0.0.0.0:443 \
  -e REGISTRY_HTTP_TLS_CERTIFICATE=/certs/domain.crt \
  -e REGISTRY_HTTP_TLS_KEY=/certs/domain.key \
  registry:2
```

## Authentication

```bash
# Create htpasswd file
docker run --rm \
  --entrypoint htpasswd \
  httpd:2 -Bbn admin password > auth/htpasswd

# Run with auth
docker run -d \
  -p 5000:5000 \
  -v $(pwd)/auth:/auth \
  -e REGISTRY_AUTH=htpasswd \
  -e REGISTRY_AUTH_HTPASSWD_PATH=/auth/htpasswd \
  -e REGISTRY_AUTH_HTPASSWD_REALM="Registry Realm" \
  registry:2

# Login
docker login localhost:5000
```

## Success Criteria
✅ Private registry running  
✅ Images pushed/pulled  
✅ TLS configured  
✅ Authentication working  

**Time:** 45 min
