# Day 8.3: GitOps for ML Infrastructure

## 🔄 Infrastructure as Code & Automation - Part 3

**Focus**: Git-Based Infrastructure Management, Continuous Deployment, Policy as Code  
**Duration**: 2-3 hours  
**Level**: Advanced  

---

## 🎯 Learning Objectives

- Master GitOps principles and implementation patterns for ML infrastructure
- Learn continuous deployment strategies with Git-based workflows
- Understand policy as code enforcement and automated compliance checking
- Analyze rollback strategies and disaster recovery in GitOps environments

---

## 🔄 GitOps Theoretical Framework

### **GitOps Principles and Architecture**

GitOps extends DevOps practices by using Git repositories as the single source of truth for infrastructure and application configuration, enabling declarative and auditable infrastructure management.

**GitOps Mathematical Model:**
```
GitOps State Reconciliation:
Desired_State = Git_Repository_State
Actual_State = Cluster_Runtime_State
Drift = |Desired_State - Actual_State|

Reconciliation Loop:
while True:
    current_drift = observe_drift()
    if current_drift > tolerance_threshold:
        apply_changes(calculate_diff(Desired_State, Actual_State))
    sleep(reconciliation_interval)

Convergence Analysis:
Convergence_Time = f(Drift_Magnitude, Reconciliation_Frequency, Change_Complexity)
System_Stability = 1 / Convergence_Time

GitOps Workflow Algebra:
Git_Commit → CI_Pipeline → Artifact_Build → CD_Pipeline → Cluster_Update → State_Verification

Where each stage has associated success probabilities:
P(Success) = P(CI) × P(Build) × P(CD) × P(Update) × P(Verification)
```

**GitOps Architecture Patterns:**
```
Push vs Pull Model:

Push Model (Traditional CI/CD):
Git → CI/CD System → kubectl apply → Cluster
Advantages: Direct control, immediate feedback
Disadvantages: External cluster access required, security concerns

Pull Model (GitOps):
Git → CI System → Container Registry
Cluster ← GitOps Agent ← Container Registry
Advantages: No external cluster access, self-healing, audit trail
Disadvantages: Polling delay, complexity

Hybrid Model:
Git → CI/CD System → GitOps Repository → GitOps Agent → Cluster
Combines benefits of both approaches with staged deployment

GitOps Agent Architecture:
┌─────────────────────────────────────────────────────────────┐
│                    GitOps Controller                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐│
│  │    Git      │ │   Image     │ │     Reconciliation      ││
│  │  Watcher    │ │  Watcher    │ │       Engine            ││
│  └─────────────┘ └─────────────┘ └─────────────────────────┘│
│         │               │                      │            │
│         ▼               ▼                      ▼            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐│
│  │   Config    │ │   Image     │ │      Kubernetes         ││
│  │   Sync      │ │   Update    │ │        API              ││
│  └─────────────┘ └─────────────┘ └─────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

**Repository Structure Patterns:**
```
MonoRepo Pattern:
ml-infrastructure/
├── applications/
│   ├── model-serving/
│   │   ├── base/
│   │   │   ├── deployment.yaml
│   │   │   ├── service.yaml
│   │   │   └── kustomization.yaml
│   │   └── overlays/
│   │       ├── development/
│   │       ├── staging/
│   │       └── production/
│   ├── feature-store/
│   └── training-platform/
├── infrastructure/
│   ├── networking/
│   ├── storage/
│   └── security/
├── policies/
│   ├── opa/
│   ├── falco/
│   └── network-policies/
└── environments/
    ├── dev/
    ├── staging/
    └── prod/

Multi-Repo Pattern:
├── ml-platform-config/          # Application configurations
├── ml-platform-infrastructure/  # Infrastructure as code
├── ml-platform-policies/        # Security and compliance policies
└── ml-platform-apps/           # Application source code

Repository Access Patterns:
1. Environment Branching:
   - main: Production configuration
   - staging: Staging configuration  
   - development: Development configuration

2. GitFlow Model:
   - main: Stable production releases
   - develop: Integration branch
   - feature/*: Feature development
   - release/*: Release preparation
   - hotfix/*: Production hotfixes

3. Environment Directories:
   - Single repository with environment-specific directories
   - Promotion through directory updates
   - Shared base configurations with overlays
```

---

## 🚀 ArgoCD Implementation Patterns

### **ArgoCD for ML Workloads**

**ArgoCD Configuration for ML:**
```
ArgoCD Application Definition:
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: ml-training-platform
  namespace: argocd
  finalizers:
    - resources-finalizer.argocd.argoproj.io
spec:
  project: ml-platform
  source:
    repoURL: https://github.com/company/ml-infrastructure
    targetRevision: HEAD
    path: applications/training-platform/overlays/production
  destination:
    server: https://kubernetes.default.svc
    namespace: ml-training
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
      allowEmpty: false
    syncOptions:
    - CreateNamespace=true
    - PrunePropagationPolicy=foreground
    - PruneLast=true
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m0s
  ignoreDifferences:
  - group: apps
    kind: Deployment
    jsonPointers:
    - /spec/replicas  # Ignore HPA-managed replica count
  - group: ""
    kind: Secret
    jsonPointers:
    - /data  # Ignore secret data managed externally

ArgoCD Project for ML Platform:
apiVersion: argoproj.io/v1alpha1
kind: AppProject
metadata:
  name: ml-platform
  namespace: argocd
spec:
  description: ML Platform Applications
  
  # Source repositories
  sourceRepos:
  - 'https://github.com/company/ml-infrastructure'
  - 'https://github.com/company/ml-applications'
  - 'https://helm.elastic.co'
  - 'https://charts.bitnami.com/bitnami'
  
  # Destination clusters and namespaces
  destinations:
  - namespace: 'ml-*'
    server: https://kubernetes.default.svc
  - namespace: 'kubeflow'
    server: https://kubernetes.default.svc
  - namespace: 'monitoring'
    server: https://kubernetes.default.svc
  
  # Cluster resource whitelist
  clusterResourceWhitelist:
  - group: ''
    kind: Namespace
  - group: rbac.authorization.k8s.io
    kind: ClusterRole
  - group: rbac.authorization.k8s.io
    kind: ClusterRoleBinding
  - group: apiextensions.k8s.io
    kind: CustomResourceDefinition
  
  # Namespace resource whitelist
  namespaceResourceWhitelist:
  - group: ''
    kind: ConfigMap
  - group: ''
    kind: Secret
  - group: ''
    kind: Service
  - group: apps
    kind: Deployment
  - group: apps
    kind: StatefulSet
  - group: networking.k8s.io
    kind: Ingress
  - group: ml.io
    kind: TrainingJob
  - group: serving.kubeflow.org
    kind: InferenceService
  
  roles:
  - name: ml-engineers
    description: ML Engineers with deployment permissions
    policies:
    - p, proj:ml-platform:ml-engineers, applications, get, ml-platform/*, allow
    - p, proj:ml-platform:ml-engineers, applications, sync, ml-platform/*, allow
    groups:
    - company:ml-engineers
  
  - name: data-scientists
    description: Data Scientists with read-only access
    policies:
    - p, proj:ml-platform:data-scientists, applications, get, ml-platform/*, allow
    groups:
    - company:data-scientists

Multi-Cluster Application Deployment:
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: ml-inference-services
  namespace: argocd
spec:
  generators:
  - clusters:
      selector:
        matchLabels:
          environment: production
          workload-type: inference
  - list:
      elements:
      - cluster: prod-us-west
        region: us-west-2
        replicas: "10"
      - cluster: prod-eu-west
        region: eu-west-1
        replicas: "5"
      - cluster: prod-asia-east
        region: asia-east-1
        replicas: "3"
  template:
    metadata:
      name: '{{cluster}}-inference-service'
    spec:
      project: ml-platform
      source:
        repoURL: https://github.com/company/ml-infrastructure
        targetRevision: HEAD
        path: applications/inference-service/overlays/production
        helm:
          parameters:
          - name: cluster.region
            value: '{{region}}'
          - name: replicaCount
            value: '{{replicas}}'
          - name: resources.requests.cpu
            value: '{{cpu}}'
          - name: resources.requests.memory
            value: '{{memory}}'
      destination:
        server: '{{server}}'
        namespace: ml-inference
      syncPolicy:
        automated:
          prune: true
          selfHeal: true
```

**Progressive Delivery with ArgoCD:**
```
Canary Deployment with Argo Rollouts:
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: ml-model-serving
spec:
  replicas: 10
  strategy:
    canary:
      steps:
      - setWeight: 10
      - pause: {duration: 300s}  # 5 minutes
      - analysis:
          templates:
          - templateName: success-rate
          args:
          - name: service-name
            value: ml-model-serving
      - setWeight: 25
      - pause: {duration: 600s}  # 10 minutes
      - analysis:
          templates:
          - templateName: success-rate
          - templateName: latency-check
          args:
          - name: service-name
            value: ml-model-serving
      - setWeight: 50
      - pause: {duration: 900s}  # 15 minutes
      - setWeight: 100
      canaryService: ml-model-serving-canary
      stableService: ml-model-serving-stable
      trafficRouting:
        istio:
          virtualService:
            name: ml-model-serving-vs
            routes:
            - primary
          destinationRule:
            name: ml-model-serving-dr
            canarySubsetName: canary
            stableSubsetName: stable
  revisionHistoryLimit: 3
  selector:
    matchLabels:
      app: ml-model-serving
  template:
    metadata:
      labels:
        app: ml-model-serving
    spec:
      containers:
      - name: model-server
        image: gcr.io/ml-platform/model-server:v1.2.3
        ports:
        - containerPort: 8080
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 1000m
            memory: 2Gi

Analysis Template for Model Performance:
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: model-performance-analysis
spec:
  args:
  - name: service-name
  - name: baseline-accuracy
    value: "0.95"
  metrics:
  - name: success-rate
    interval: 60s
    count: 5
    successCondition: result[0] >= 0.99
    failureLimit: 3
    provider:
      prometheus:
        address: http://prometheus.monitoring.svc.cluster.local:9090
        query: |
          sum(rate(http_requests_total{job="{{args.service-name}}",code=~"2.."}[5m])) /
          sum(rate(http_requests_total{job="{{args.service-name}}"}[5m]))
  
  - name: model-accuracy
    interval: 300s  # 5 minutes
    count: 3
    successCondition: result[0] >= {{args.baseline-accuracy}}
    failureLimit: 1
    provider:
      prometheus:
        address: http://prometheus.monitoring.svc.cluster.local:9090
        query: |
          ml_model_accuracy{service="{{args.service-name}}"}
  
  - name: prediction-latency
    interval: 60s
    count: 5
    successCondition: result[0] <= 100  # 100ms
    failureLimit: 3
    provider:
      prometheus:
        address: http://prometheus.monitoring.svc.cluster.local:9090
        query: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket{job="{{args.service-name}}"}[5m])) by (le)
          ) * 1000
```

---

## 🛡️ Policy as Code Implementation

### **Open Policy Agent (OPA) Integration**

**OPA Gatekeeper for ML Workloads:**
```
Constraint Templates for ML Governance:
apiVersion: templates.gatekeeper.sh/v1beta1
kind: ConstraintTemplate
metadata:
  name: mlresourcerequirements
spec:
  crd:
    spec:
      names:
        kind: MLResourceRequirements
      validation:
        properties:
          cpuMax:
            type: string
          memoryMax:
            type: string
          gpuMax:
            type: integer
          workloadTypes:
            type: array
            items:
              type: string
  targets:
    - target: admission.k8s.gatekeeper.sh
      rego: |
        package mlresourcerequirements
        
        violation[{"msg": msg}] {
          container := input.review.object.spec.containers[_]
          workload_type := input.review.object.metadata.labels["workload-type"]
          workload_type == input.parameters.workloadTypes[_]
          
          # Check CPU limits
          cpu_limit := container.resources.limits.cpu
          cpu_max := input.parameters.cpuMax
          cpu_limit_numeric := to_number(regex.replace(cpu_limit, "[^0-9.]", ""))
          cpu_max_numeric := to_number(regex.replace(cpu_max, "[^0-9.]", ""))
          cpu_limit_numeric > cpu_max_numeric
          msg := sprintf("CPU limit %v exceeds maximum allowed %v for workload type %v", [cpu_limit, cpu_max, workload_type])
        }
        
        violation[{"msg": msg}] {
          container := input.review.object.spec.containers[_]
          workload_type := input.review.object.metadata.labels["workload-type"]
          workload_type == input.parameters.workloadTypes[_]
          
          # Check GPU limits
          gpu_limit := container.resources.limits["nvidia.com/gpu"]
          gpu_max := input.parameters.gpuMax
          to_number(gpu_limit) > gpu_max
          msg := sprintf("GPU limit %v exceeds maximum allowed %v for workload type %v", [gpu_limit, gpu_max, workload_type])
        }

ML Resource Constraints:
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: MLResourceRequirements
metadata:
  name: ml-training-resource-limits
spec:
  match:
    kinds:
      - apiGroups: ["apps", "batch"]
        kinds: ["Deployment", "Job", "StatefulSet"]
    namespaces: ["ml-training", "ml-experimentation"]
  parameters:
    cpuMax: "32"
    memoryMax: "256Gi"
    gpuMax: 8
    workloadTypes: ["training", "experimentation"]

---
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: MLResourceRequirements
metadata:
  name: ml-inference-resource-limits
spec:
  match:
    kinds:
      - apiGroups: ["apps"]
        kinds: ["Deployment"]
    namespaces: ["ml-inference", "ml-serving"]
  parameters:
    cpuMax: "8"
    memoryMax: "32Gi"
    gpuMax: 2
    workloadTypes: ["inference", "serving"]

Data Privacy Constraints:
apiVersion: templates.gatekeeper.sh/v1beta1
kind: ConstraintTemplate
metadata:
  name: mldataprivacy
spec:
  crd:
    spec:
      names:
        kind: MLDataPrivacy
      validation:
        properties:
          allowedDataSources:
            type: array
            items:
              type: string
          requireEncryption:
            type: boolean
          allowedRegions:
            type: array
            items:
              type: string
  targets:
    - target: admission.k8s.gatekeeper.sh
      rego: |
        package mldataprivacy
        
        violation[{"msg": msg}] {
          # Check data source restrictions
          container := input.review.object.spec.containers[_]
          env_var := container.env[_]
          env_var.name == "DATA_SOURCE"
          not env_var.value in input.parameters.allowedDataSources
          msg := sprintf("Data source %v is not in allowed list: %v", [env_var.value, input.parameters.allowedDataSources])
        }
        
        violation[{"msg": msg}] {
          # Require encryption for sensitive data
          input.parameters.requireEncryption == true
          container := input.review.object.spec.containers[_]
          volume_mount := container.volumeMounts[_]
          volume_mount.name == "sensitive-data"
          not has_encryption_annotation
          msg := "Sensitive data volumes must be encrypted"
        }
        
        has_encryption_annotation {
          input.review.object.metadata.annotations["data.encryption.enabled"] == "true"
        }

Model Governance Constraints:
apiVersion: templates.gatekeeper.sh/v1beta1
kind: ConstraintTemplate
metadata:
  name: mlmodelgovernance
spec:
  crd:
    spec:
      names:
        kind: MLModelGovernance
      validation:
        properties:
          requireApproval:
            type: boolean
          allowedModelRegistries:
            type: array
            items:
              type: string
          requireBiasCheck:
            type: boolean
  targets:
    - target: admission.k8s.gatekeeper.sh
      rego: |
        package mlmodelgovernance
        
        violation[{"msg": msg}] {
          # Check model registry restrictions
          contains(input.review.object.spec.containers[_].image, "model")
          registry := regex.split("\/", input.review.object.spec.containers[_].image)[0]
          not registry in input.parameters.allowedModelRegistries
          msg := sprintf("Model registry %v is not approved. Allowed registries: %v", [registry, input.parameters.allowedModelRegistries])
        }
        
        violation[{"msg": msg}] {
          # Require approval annotation for production models
          input.parameters.requireApproval == true
          input.review.object.metadata.namespace == "ml-production"
          not input.review.object.metadata.annotations["model.approval.status"]
          msg := "Production model deployments require approval annotation"
        }
        
        violation[{"msg": msg}] {
          # Require bias check for fairness-critical models
          input.parameters.requireBiasCheck == true
          input.review.object.metadata.labels["model.fairness-critical"] == "true"
          not input.review.object.metadata.annotations["model.bias-check.status"] == "passed"
          msg := "Fairness-critical models require bias check validation"
        }
```

### **Automated Compliance Checking**

**CI/CD Integration with Policy Validation:**
```
GitHub Actions Workflow with Policy Validation:
name: ML Infrastructure Deployment
on:
  push:
    branches: [ main, staging, development ]
  pull_request:
    branches: [ main ]

jobs:
  policy-validation:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Conftest
      run: |
        wget https://github.com/open-policy-agent/conftest/releases/download/v0.46.0/conftest_0.46.0_Linux_x86_64.tar.gz
        tar xzf conftest_0.46.0_Linux_x86_64.tar.gz
        sudo mv conftest /usr/local/bin
    
    - name: Validate ML Resource Policies
      run: |
        conftest verify --policy policies/opa/ml-resources.rego applications/*/overlays/${{ github.ref_name }}/*.yaml
    
    - name: Validate Security Policies
      run: |
        conftest verify --policy policies/opa/security.rego applications/*/overlays/${{ github.ref_name }}/*.yaml
    
    - name: Validate Data Privacy Policies
      run: |
        conftest verify --policy policies/opa/data-privacy.rego applications/*/overlays/${{ github.ref_name }}/*.yaml

  terraform-validation:
    runs-on: ubuntu-latest
    needs: policy-validation
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Terraform
      uses: hashicorp/setup-terraform@v2
      with:
        terraform_version: 1.5.0
    
    - name: Terraform Format Check
      run: terraform fmt -check -recursive
    
    - name: Terraform Validate
      run: |
        cd infrastructure
        terraform init -backend=false
        terraform validate
    
    - name: Terraform Security Scan
      uses: aquasecurity/tfsec-action@v1.0.3
      with:
        soft_fail: false

  security-scanning:
    runs-on: ubuntu-latest
    needs: policy-validation
    steps:
    - uses: actions/checkout@v3
    
    - name: Run Kubesec Scan
      run: |
        docker run --rm -v $PWD:/workspace kubesec/kubesec:latest scan /workspace/applications/*/overlays/${{ github.ref_name }}/*.yaml
    
    - name: Run Kube-score
      run: |
        wget https://github.com/zegl/kube-score/releases/download/v1.16.1/kube-score_1.16.1_linux_amd64.tar.gz
        tar xzf kube-score_1.16.1_linux_amd64.tar.gz
        ./kube-score score applications/*/overlays/${{ github.ref_name }}/*.yaml

  deploy:
    runs-on: ubuntu-latest
    needs: [policy-validation, terraform-validation, security-scanning]
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3
    
    - name: Update ArgoCD Application
      run: |
        # Update application manifest with new image tags
        yq eval '.spec.source.targetRevision = "${{ github.sha }}"' -i argocd/applications/ml-platform.yaml
        
        # Commit changes back to repository
        git config --local user.email "action@github.com"
        git config --local user.name "GitHub Action"
        git add argocd/applications/ml-platform.yaml
        git commit -m "Update application to ${{ github.sha }}" || exit 0
        git push

Policy Testing Framework:
# policies/opa/ml-resources_test.rego
package mlresourcerequirements

test_cpu_limit_violation {
    violation[_] with input as {
        "review": {
            "object": {
                "metadata": {
                    "labels": {
                        "workload-type": "training"
                    }
                },
                "spec": {
                    "containers": [{
                        "resources": {
                            "limits": {
                                "cpu": "64"
                            }
                        }
                    }]
                }
            }
        }
    } with input.parameters as {
        "cpuMax": "32",
        "workloadTypes": ["training"]
    }
}

test_cpu_limit_allowed {
    count(violation) == 0 with input as {
        "review": {
            "object": {
                "metadata": {
                    "labels": {
                        "workload-type": "training"
                    }
                },
                "spec": {
                    "containers": [{
                        "resources": {
                            "limits": {
                                "cpu": "16"
                            }
                        }
                    }]
                }
            }
        }
    } with input.parameters as {
        "cpuMax": "32",
        "workloadTypes": ["training"]
    }
}

# Run policy tests
conftest verify --policy policies/opa/ --data policies/test-data/
```

---

## 🔄 Rollback and Disaster Recovery

### **Automated Rollback Strategies**

**GitOps Rollback Mechanisms:**
```
Git-Based Rollback:
#!/bin/bash
gitops_rollback() {
    local application=$1
    local target_commit=$2
    local reason=$3
    
    echo "Initiating rollback for $application"
    echo "Target commit: $target_commit"
    echo "Reason: $reason"
    
    # Create rollback branch
    rollback_branch="rollback/$application/$(date +%Y%m%d_%H%M%S)"
    git checkout -b "$rollback_branch"
    
    # Revert to target commit
    git revert --no-edit HEAD.."$target_commit"
    
    # Update application configuration
    yq eval ".spec.source.targetRevision = \"$target_commit\"" -i "argocd/applications/$application.yaml"
    
    # Commit rollback changes
    git add .
    git commit -m "Rollback $application to $target_commit

Reason: $reason
Rollback initiated by: $USER
Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    
    # Push rollback branch
    git push origin "$rollback_branch"
    
    # Create pull request for rollback
    gh pr create \
        --title "🚨 Rollback: $application to $target_commit" \
        --body "**Rollback Details**
        
Application: $application
Target Commit: $target_commit
Reason: $reason
Initiated by: $USER
Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)

**Pre-rollback Checklist**
- [ ] Incident documented
- [ ] Stakeholders notified
- [ ] Data backup verified
- [ ] Rollback plan reviewed

**Post-rollback Actions**
- [ ] Verify application health
- [ ] Monitor key metrics
- [ ] Update incident status
- [ ] Schedule post-mortem" \
        --label "rollback,urgent" \
        --reviewer "ml-platform-team"
}

ArgoCD Application Rollback:
#!/bin/bash
argocd_rollback() {
    local application=$1
    local revision=$2
    
    # Get current application state
    current_revision=$(argocd app get "$application" -o json | jq -r '.status.sync.revision')
    
    echo "Rolling back $application from $current_revision to $revision"
    
    # Perform rollback
    argocd app rollback "$application" --revision "$revision"
    
    # Wait for rollback to complete
    argocd app wait "$application" --timeout 300
    
    # Verify rollback success
    new_revision=$(argocd app get "$application" -o json | jq -r '.status.sync.revision')
    
    if [ "$new_revision" == "$revision" ]; then
        echo "✅ Rollback successful"
        
        # Run health checks
        kubectl get pods -n "$(argocd app get "$application" -o json | jq -r '.spec.destination.namespace')"
        
        # Update monitoring
        curl -X POST "$MONITORING_WEBHOOK" -d "{
            \"event\": \"rollback_completed\",
            \"application\": \"$application\",
            \"from_revision\": \"$current_revision\",
            \"to_revision\": \"$revision\",
            \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"
        }"
    else
        echo "❌ Rollback failed"
        exit 1
    fi
}

Automated Rollback Triggers:
apiVersion: v1
kind: ConfigMap
metadata:
  name: rollback-automation-config
data:
  config.yaml: |
    rollback_triggers:
      - name: error_rate_spike
        condition: error_rate > 0.05
        duration: 300s  # 5 minutes
        action: automatic_rollback
        
      - name: latency_degradation
        condition: p99_latency > 2 * baseline_p99
        duration: 180s  # 3 minutes
        action: automatic_rollback
        
      - name: availability_drop
        condition: availability < 0.99
        duration: 120s  # 2 minutes
        action: automatic_rollback
        
      - name: model_accuracy_drop
        condition: accuracy < baseline_accuracy - 0.05
        duration: 600s  # 10 minutes
        action: create_incident_and_rollback

---
apiVersion: batch/v1
kind: CronJob
metadata:
  name: rollback-monitor
spec:
  schedule: "*/1 * * * *"  # Every minute
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: rollback-monitor
            image: ml-platform/rollback-monitor:latest
            env:
            - name: PROMETHEUS_URL
              value: "http://prometheus.monitoring.svc.cluster.local:9090"
            - name: ARGOCD_SERVER
              value: "argocd-server.argocd.svc.cluster.local"
            - name: WEBHOOK_URL
              value: "http://alertmanager.monitoring.svc.cluster.local:9093/api/v1/alerts"
            volumeMounts:
            - name: config
              mountPath: /config
          volumes:
          - name: config
            configMap:
              name: rollback-automation-config
          restartPolicy: OnFailure
```

### **Disaster Recovery Patterns**

**Multi-Region GitOps Setup:**
```
Cross-Region Replication:
# Primary cluster (us-west-2)
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: ml-platform-primary
  namespace: argocd
spec:
  project: ml-platform
  source:
    repoURL: https://github.com/company/ml-infrastructure
    targetRevision: HEAD
    path: applications/ml-platform/overlays/production-primary
  destination:
    server: https://kubernetes.default.svc
    namespace: ml-platform
  syncPolicy:
    automated:
      prune: true
      selfHeal: true

# DR cluster (eu-west-1)
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: ml-platform-dr
  namespace: argocd
spec:
  project: ml-platform
  source:
    repoURL: https://github.com/company/ml-infrastructure
    targetRevision: HEAD
    path: applications/ml-platform/overlays/production-dr
  destination:
    server: https://k8s-dr-cluster.company.com
    namespace: ml-platform
  syncPolicy:
    automated:
      prune: false  # Manual sync for DR
      selfHeal: false

Disaster Recovery Automation:
#!/bin/bash
initiate_disaster_recovery() {
    local primary_region=$1
    local dr_region=$2
    local incident_id=$3
    
    echo "🚨 Initiating disaster recovery"
    echo "Primary region: $primary_region (FAILED)"
    echo "DR region: $dr_region (ACTIVATING)"
    echo "Incident ID: $incident_id"
    
    # Update DNS to point to DR region
    aws route53 change-resource-record-sets \
        --hosted-zone-id "$HOSTED_ZONE_ID" \
        --change-batch file://dr-dns-changes.json
    
    # Activate DR ArgoCD applications
    argocd app sync ml-platform-dr --timeout 600
    argocd app sync ml-inference-dr --timeout 300
    argocd app sync ml-training-dr --timeout 300
    
    # Scale up DR services
    kubectl scale deployment ml-inference-service --replicas=10 -n ml-platform
    kubectl scale statefulset feature-store --replicas=3 -n ml-platform
    
    # Update monitoring dashboards
    curl -X POST "$GRAFANA_API/dashboards/db" \
        -H "Authorization: Bearer $GRAFANA_TOKEN" \
        -d @dr-dashboard-config.json
    
    # Notify stakeholders
    slack_notify "🚨 DISASTER RECOVERY ACTIVATED

Primary Region: $primary_region (FAILED)
DR Region: $dr_region (ACTIVE)
Incident ID: $incident_id
RTO Target: 15 minutes
RPO Target: 5 minutes

All ML services are now running in the DR region.
Please monitor the #incident-$incident_id channel for updates."
    
    echo "✅ Disaster recovery activation complete"
}

Recovery Validation:
#!/bin/bash
validate_dr_recovery() {
    local dr_region=$1
    local validation_results=""
    
    echo "Validating disaster recovery in $dr_region"
    
    # Check application health
    app_health=$(argocd app get ml-platform-dr -o json | jq -r '.status.health.status')
    if [ "$app_health" == "Healthy" ]; then
        validation_results+="✅ Application health: OK\n"
    else
        validation_results+="❌ Application health: FAILED\n"
    fi
    
    # Check service endpoints
    inference_endpoint=$(kubectl get service ml-inference-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if curl -f "http://$inference_endpoint/health" >/dev/null 2>&1; then
        validation_results+="✅ Inference service: OK\n"
    else
        validation_results+="❌ Inference service: FAILED\n"
    fi
    
    # Check data availability
    if kubectl exec -it feature-store-0 -- redis-cli ping | grep -q PONG; then
        validation_results+="✅ Feature store: OK\n"
    else
        validation_results+="❌ Feature store: FAILED\n"
    fi
    
    # Check monitoring
    if curl -f "$PROMETHEUS_URL/api/v1/query?query=up" >/dev/null 2>&1; then
        validation_results+="✅ Monitoring: OK\n"
    else
        validation_results+="❌ Monitoring: FAILED\n"
    fi
    
    echo -e "$validation_results"
    
    # Send validation report
    slack_notify "🔍 DR VALIDATION REPORT

Region: $dr_region
Validation Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)

$validation_results

Full system validation complete."
}

Recovery Testing:
apiVersion: batch/v1
kind: CronJob
metadata:
  name: dr-testing
spec:
  schedule: "0 2 * * 0"  # Weekly on Sunday at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: dr-test
            image: ml-platform/dr-test:latest
            command:
            - /bin/bash
            - -c
            - |
              echo "Starting DR test"
              
              # Simulate primary region failure
              kubectl patch deployment ml-platform-primary --patch '{"spec":{"replicas":0}}' || true
              
              # Wait for failure detection
              sleep 120
              
              # Activate DR procedures
              ./initiate_disaster_recovery.sh us-west-2 eu-west-1 dr-test-$(date +%Y%m%d)
              
              # Validate DR activation
              sleep 300
              ./validate_dr_recovery.sh eu-west-1
              
              # Restore primary region
              kubectl patch deployment ml-platform-primary --patch '{"spec":{"replicas":3}}'
              
              # Generate test report
              echo "DR test completed successfully" > /shared/dr-test-report.txt
              
            volumeMounts:
            - name: shared-storage
              mountPath: /shared
          volumes:
          - name: shared-storage
            persistentVolumeClaim:
              claimName: dr-test-storage
          restartPolicy: OnFailure
```

This comprehensive framework for GitOps ML infrastructure provides the theoretical foundations and practical strategies for implementing Git-based infrastructure management with automated compliance, rollback capabilities, and disaster recovery. The key insight is that GitOps enables declarative, auditable, and collaborative infrastructure management while providing strong consistency guarantees and automated reconciliation capabilities essential for production ML systems.