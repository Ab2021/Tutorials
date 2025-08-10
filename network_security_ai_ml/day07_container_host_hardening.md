# Day 7: Container & Host Hardening

## Table of Contents
1. [Container Security Fundamentals](#container-security-fundamentals)
2. [Container Image Security](#container-image-security)
3. [Runtime Security and Monitoring](#runtime-security-and-monitoring)
4. [Host Hardening for AI/ML](#host-hardening-for-aiml)
5. [CIS Benchmarks Implementation](#cis-benchmarks-implementation)
6. [Kernel Security and Lockdown](#kernel-security-and-lockdown)
7. [Container Orchestration Security](#container-orchestration-security)
8. [Automated Patch Management](#automated-patch-management)
9. [Security Tools and Monitoring](#security-tools-and-monitoring)
10. [AI/ML Specific Hardening](#aiml-specific-hardening)

## Container Security Fundamentals

### Understanding Container Security Model
Container security in AI/ML environments requires a comprehensive approach addressing the entire container lifecycle from build to runtime.

**Container Security Principles:**
- **Least Privilege**: Containers run with minimal necessary permissions
- **Immutable Infrastructure**: Containers treated as immutable artifacts
- **Defense in Depth**: Multiple layers of security controls
- **Supply Chain Security**: Securing the entire container build and distribution chain
- **Runtime Protection**: Continuous monitoring and protection during execution
- **Compliance**: Meeting regulatory requirements for containerized workloads

**AI/ML Container Challenges:**
- **GPU Access**: Containers requiring privileged access to GPU hardware
- **Large Images**: AI/ML frameworks creating large container images
- **Model Artifacts**: Secure handling of sensitive AI/ML models
- **Data Processing**: Secure data access from containerized applications
- **Resource Sharing**: Shared resources in multi-tenant AI/ML environments
- **Performance Impact**: Security controls impacting AI/ML performance

**Container Attack Vectors:**
- **Container Escape**: Breaking out of container isolation
- **Privilege Escalation**: Gaining elevated privileges within containers
- **Supply Chain Attacks**: Compromised base images or dependencies
- **Secrets Exposure**: Exposed credentials and API keys
- **Resource Abuse**: Unauthorized resource consumption
- **Network Attacks**: Lateral movement through container networks

### Container Isolation Mechanisms

**Linux Namespaces:**
```bash
# Example: Creating secure container namespaces
#!/bin/bash

# Create network namespace for AI/ML container
ip netns add ml-container-ns

# Create mount namespace
unshare --mount --pid --net --uts --ipc --fork /bin/bash

# Set up cgroup for resource isolation
mkdir -p /sys/fs/cgroup/memory/ml-container
echo "2G" > /sys/fs/cgroup/memory/ml-container/memory.limit_in_bytes

# Configure user namespace for privilege isolation
echo "1000 0 1" > /proc/self/uid_map
echo "1000 0 1" > /proc/self/gid_map
```

**Control Groups (cgroups):**
```yaml
# Docker Compose with resource constraints
version: '3.8'
services:
  ml-training:
    image: ml-training:secure
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
          # GPU resource limits
          reservations:
            devices:
              - driver: nvidia
                count: 1
                capabilities: [gpu]
    security_opt:
      - no-new-privileges:true
      - seccomp:seccomp-profile.json
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    read_only: true
    tmpfs:
      - /tmp:noexec,nosuid,size=100m
    volumes:
      - ml-data:/data:ro
      - model-cache:/models:ro
```

**Security Contexts in Kubernetes:**
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ml-inference-pod
  labels:
    app: ml-inference
    security.policy/level: high
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    runAsGroup: 1000
    fsGroup: 1000
    seccompProfile:
      type: RuntimeDefault
  containers:
  - name: ml-inference
    image: ml-inference:v1.2.3
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      runAsNonRoot: true
      runAsUser: 1000
      capabilities:
        drop:
        - ALL
        add:
        - NET_BIND_SERVICE
    resources:
      limits:
        memory: "4Gi"
        cpu: "2"
        nvidia.com/gpu: "1"
      requests:
        memory: "2Gi"
        cpu: "1"
    volumeMounts:
    - name: tmp
      mountPath: /tmp
    - name: models
      mountPath: /models
      readOnly: true
    env:
    - name: MODEL_PATH
      value: "/models/latest"
  volumes:
  - name: tmp
    emptyDir:
      sizeLimit: 1Gi
  - name: models
    persistentVolumeClaim:
      claimName: ml-models-pvc
      readOnly: true
```

## Container Image Security

### Secure Base Images

**Minimal Base Images:**
```dockerfile
# Example: Secure Python ML base image
FROM python:3.9-slim-bullseye as base

# Create non-root user
RUN groupadd -r mluser && useradd -r -g mluser mluser

# Update and install security patches
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set security-focused environment
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

FROM base AS builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt /tmp/
RUN pip install --user -r /tmp/requirements.txt

FROM base AS runtime

# Copy installed packages from builder
COPY --from=builder /root/.local /home/mluser/.local

# Copy application code
COPY --chown=mluser:mluser app/ /app/

# Set working directory and user
WORKDIR /app
USER mluser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Run application
CMD ["python", "main.py"]
```

**Multi-Stage Build Security:**
```dockerfile
# Multi-stage build for TensorFlow ML application
FROM tensorflow/tensorflow:2.12.0-gpu as tf-base

# Security hardening
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

FROM tf-base as builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        make \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --user -r /tmp/requirements.txt

# Build custom components
COPY src/ /src/
RUN cd /src && python setup.py build_ext --inplace

FROM tf-base as production

# Create application user
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /app -s /sbin/nologin appuser

# Copy built artifacts
COPY --from=builder /root/.local /home/appuser/.local
COPY --from=builder /src/build /app/lib
COPY --chown=appuser:appuser app/ /app/

# Set environment
ENV PATH="/home/appuser/.local/bin:$PATH" \
    PYTHONPATH="/app:/app/lib" \
    PYTHONUNBUFFERED=1

# Security configurations
RUN chmod -R 755 /app && \
    chown -R appuser:appuser /app

WORKDIR /app
USER appuser

# Security labels
LABEL security.scan.date="2024-01-15" \
      security.scan.tool="trivy" \
      security.vulnerability.count="0"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "app:application"]
```

### Image Scanning and Vulnerability Management

**Trivy Integration:**
```yaml
# GitHub Actions workflow with Trivy scanning
name: Container Security Scan
on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout code
      uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -t ml-app:${{ github.sha }} .
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: 'ml-app:${{ github.sha }}'
        format: 'sarif'
        output: 'trivy-results.sarif'
        severity: 'CRITICAL,HIGH,MEDIUM'
        exit-code: '1'
    
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      if: always()
      with:
        sarif_file: 'trivy-results.sarif'
    
    - name: Generate security report
      if: always()
      run: |
        trivy image --format json --output security-report.json ml-app:${{ github.sha }}
    
    - name: Upload security artifacts
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: security-reports
        path: |
          trivy-results.sarif
          security-report.json
```

**Admission Controller for Image Security:**
```yaml
# OPA Gatekeeper policy for image security
apiVersion: templates.gatekeeper.sh/v1beta1
kind: ConstraintTemplate
metadata:
  name: secureimages
spec:
  crd:
    spec:
      names:
        kind: SecureImages
      validation:
        openAPIV3Schema:
          type: object
          properties:
            allowedRegistries:
              type: array
              items:
                type: string
            requiredLabels:
              type: array
              items:
                type: string
            maxVulnerabilities:
              type: object
              properties:
                critical: 
                  type: integer
                high:
                  type: integer
  targets:
    - target: admission.k8s.gatekeeper.sh
      rego: |
        package secureimages
        
        violation[{"msg": msg}] {
          # Check allowed registries
          container := input.review.object.spec.containers[_]
          not registry_allowed(container.image, input.parameters.allowedRegistries)
          msg := sprintf("Image registry not allowed: %v", [container.image])
        }
        
        violation[{"msg": msg}] {
          # Check required security labels
          required := input.parameters.requiredLabels
          container := input.review.object.spec.containers[_]
          not has_required_labels(container, required)
          msg := "Container image missing required security labels"
        }
        
        registry_allowed(image, allowed_registries) {
          startswith(image, allowed_registries[_])
        }
        
        has_required_labels(container, required_labels) {
          # This would typically check image metadata
          # Implementation depends on your image labeling strategy
          true
        }
---
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: SecureImages
metadata:
  name: must-use-secure-images
spec:
  match:
    kinds:
      - apiGroups: [""]
        kinds: ["Pod"]
    namespaces: ["ml-production", "ml-staging"]
  parameters:
    allowedRegistries:
      - "registry.company.com/"
      - "public.ecr.aws/tensorflow/"
      - "nvcr.io/nvidia/"
    requiredLabels:
      - "security.scan.date"
      - "security.scan.tool"
    maxVulnerabilities:
      critical: 0
      high: 2
```

### Secrets Management in Containers

**External Secrets Operator:**
```yaml
# External Secrets configuration for ML workloads
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: vault-ml-secrets
  namespace: ml-production
spec:
  provider:
    vault:
      server: "https://vault.company.com"
      path: "ml-secrets"
      version: "v2"
      auth:
        kubernetes:
          mountPath: "kubernetes"
          role: "ml-secrets-reader"
---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: ml-api-credentials
  namespace: ml-production
spec:
  refreshInterval: 15m
  secretStoreRef:
    name: vault-ml-secrets
    kind: SecretStore
  target:
    name: ml-api-secret
    creationPolicy: Owner
    template:
      type: Opaque
      data:
        api-key: "{{ .apiKey }}"
        database-url: "postgresql://{{ .dbUser }}:{{ .dbPassword }}@{{ .dbHost }}:5432/{{ .dbName }}"
  data:
  - secretKey: apiKey
    remoteRef:
      key: ml-inference
      property: api_key
  - secretKey: dbUser
    remoteRef:
      key: database
      property: username
  - secretKey: dbPassword
    remoteRef:
      key: database
      property: password
  - secretKey: dbHost
    remoteRef:
      key: database
      property: host
  - secretKey: dbName
    remoteRef:
      key: database
      property: database_name
```

**Init Container for Secret Injection:**
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ml-training-pod
spec:
  initContainers:
  - name: secret-fetcher
    image: vault:1.13.0
    command:
    - sh
    - -c
    - |
      # Authenticate with Vault
      vault auth -method=kubernetes role=ml-training
      
      # Fetch secrets and write to shared volume
      vault kv get -field=api_key ml-secrets/training > /shared/api_key
      vault kv get -field=model_key ml-secrets/training > /shared/model_key
      
      # Set appropriate permissions
      chmod 600 /shared/*
      chown 1000:1000 /shared/*
    env:
    - name: VAULT_ADDR
      value: "https://vault.company.com"
    volumeMounts:
    - name: shared-secrets
      mountPath: /shared
    securityContext:
      runAsUser: 0
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
  containers:
  - name: ml-training
    image: ml-training:v1.0.0
    env:
    - name: API_KEY_FILE
      value: /secrets/api_key
    - name: MODEL_KEY_FILE
      value: /secrets/model_key
    volumeMounts:
    - name: shared-secrets
      mountPath: /secrets
      readOnly: true
    securityContext:
      runAsUser: 1000
      runAsNonRoot: true
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
  volumes:
  - name: shared-secrets
    emptyDir:
      medium: Memory
      sizeLimit: 10Mi
```

## Runtime Security and Monitoring

### Runtime Security Tools

**Falco Configuration:**
```yaml
# Falco rules for AI/ML container security
apiVersion: v1
kind: ConfigMap
metadata:
  name: falco-ml-rules
  namespace: falco-system
data:
  ml_rules.yaml: |
    - rule: Unauthorized GPU Access
      desc: Detect unauthorized access to GPU devices
      condition: >
        spawned_process and
        proc.name in (nvidia-smi, nvidia-ml-py) and
        not container.image.repository in (ml-training, ml-inference) and
        not proc.pname in (kubelet, containerd)
      output: >
        Unauthorized GPU access detected (user=%user.name command=%proc.cmdline 
        container=%container.name image=%container.image.repository)
      priority: WARNING
      tags: [ml, gpu, security]

    - rule: ML Model File Access
      desc: Detect access to ML model files from unauthorized containers
      condition: >
        open_read and
        fd.name contains "/models/" and
        not container.image.repository in (ml-training, ml-inference, ml-serving)
      output: >
        Unauthorized model file access (user=%user.name file=%fd.name 
        container=%container.name image=%container.image.repository)
      priority: ERROR
      tags: [ml, model, data-access]

    - rule: Suspicious Network Activity from ML Container
      desc: Detect unusual network connections from ML containers
      condition: >
        outbound and
        container.image.repository in (ml-training, ml-inference) and
        not fd.net.cip in (ml_allowed_ips) and
        not fd.net.sport in (80, 443, 8080, 8443)
      output: >
        Suspicious network connection from ML container (container=%container.name 
        connection=%fd.net.cip:%fd.net.cport)
      priority: WARNING
      tags: [ml, network, security]

    - rule: Container Privilege Escalation
      desc: Detect privilege escalation attempts in containers
      condition: >
        spawned_process and
        proc.name in (su, sudo, doas) and
        container.id != host
      output: >
        Privilege escalation attempt detected (user=%user.name command=%proc.cmdline 
        container=%container.name)
      priority: CRITICAL
      tags: [privilege-escalation, security]

    - macro: ml_allowed_ips
      condition: >
        (fd.net.cip="10.0.0.0/8" or 
         fd.net.cip="172.16.0.0/12" or 
         fd.net.cip="192.168.0.0/16")
```

**Falco Deployment:**
```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: falco
  namespace: falco-system
spec:
  selector:
    matchLabels:
      app: falco
  template:
    metadata:
      labels:
        app: falco
    spec:
      serviceAccount: falco
      hostNetwork: true
      hostPID: true
      containers:
      - name: falco
        image: falcosecurity/falco:0.34.1
        args:
          - /usr/bin/falco
          - --cri=/run/containerd/containerd.sock
          - --k8s-api=https://kubernetes.default.svc.cluster.local
          - --k8s-api-cert=/var/run/secrets/kubernetes.io/serviceaccount/ca.crt
          - --k8s-api-token=/var/run/secrets/kubernetes.io/serviceaccount/token
        env:
        - name: FALCO_K8S_NODE_NAME
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName
        securityContext:
          privileged: true
        volumeMounts:
        - mountPath: /host/var/run/docker.sock
          name: docker-socket
        - mountPath: /host/run/containerd/containerd.sock
          name: containerd-socket
        - mountPath: /host/dev
          name: dev-fs
        - mountPath: /host/proc
          name: proc-fs
          readOnly: true
        - mountPath: /host/boot
          name: boot-fs
          readOnly: true
        - mountPath: /host/lib/modules
          name: lib-modules
          readOnly: true
        - mountPath: /host/usr
          name: usr-fs
          readOnly: true
        - mountPath: /etc/falco/rules.d
          name: falco-rules
      volumes:
      - name: docker-socket
        hostPath:
          path: /var/run/docker.sock
      - name: containerd-socket
        hostPath:
          path: /run/containerd/containerd.sock
      - name: dev-fs
        hostPath:
          path: /dev
      - name: proc-fs
        hostPath:
          path: /proc
      - name: boot-fs
        hostPath:
          path: /boot
      - name: lib-modules
        hostPath:
          path: /lib/modules
      - name: usr-fs
        hostPath:
          path: /usr
      - name: falco-rules
        configMap:
          name: falco-ml-rules
```

### Container Behavior Analysis

**Sysdig Secure Configuration:**
```yaml
# Sysdig Secure policy for ML workloads
apiVersion: v1
kind: ConfigMap
metadata:
  name: sysdig-ml-policies
data:
  ml-runtime-policy.yaml: |
    policies:
    - name: "ML Container Runtime Security"
      description: "Security policies for ML containers"
      scope: "container.image.repository startswith 'ml-'"
      rules:
      - name: "Allowed ML Processes"
        description: "Only allow specific processes in ML containers"
        condition: >
          proc.name in (python, python3, jupyter, tensorboard, nvidia-smi) or
          proc.name startswith "ml-" or
          proc.pname in (python, python3, jupyter)
        action: allow
        
      - name: "Block Suspicious Network Connections"
        description: "Block connections to suspicious IPs"
        condition: >
          evt.type=connect and
          not fd.net.cip in (internal_networks) and
          not fd.net.cport in (80, 443, 8080, 8443, 22)
        action: block
        
      - name: "Monitor File System Access"
        description: "Monitor access to sensitive files"
        condition: >
          evt.type=openat and
          (fd.name contains "/etc/passwd" or
           fd.name contains "/etc/shadow" or
           fd.name contains "/models/" or
           fd.name contains "/secrets/")
        action: log
        
      - name: "Prevent Privilege Escalation"
        description: "Block privilege escalation attempts"
        condition: >
          proc.name in (su, sudo, doas, pkexec) or
          (evt.type=setuid and evt.arg.uid=0) or
          (evt.type=setgid and evt.arg.gid=0)
        action: block

    macros:
    - name: internal_networks
      condition: >
        fd.net.cip="10.0.0.0/8" or
        fd.net.cip="172.16.0.0/12" or
        fd.net.cip="192.168.0.0/16" or
        fd.net.cip="127.0.0.0/8"
```

This covers the first half of Day 7. The content focuses on container security fundamentals, image security, and runtime monitoring. Would you like me to continue with the second half covering host hardening, CIS benchmarks, and AI/ML-specific considerations?
