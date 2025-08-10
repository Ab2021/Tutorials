# Day 6.3: Model Versioning & Lifecycle Management

## 🔄 Model Serving & Production Inference - Part 3

**Focus**: Model Registry, Version Control, Lifecycle Automation, Governance  
**Duration**: 2-3 hours  
**Level**: Advanced  

---

## 🎯 Learning Objectives

- Master model versioning strategies and understand lifecycle management principles
- Learn model registry architecture and metadata management systems
- Understand automated lifecycle policies and governance frameworks
- Analyze model lineage tracking and compliance requirements

---

## 📦 Model Registry Architecture

### **Model Metadata Management**

A model registry serves as the central repository for ML models, providing versioning, metadata storage, and lifecycle management capabilities.

**Metadata Schema Design:**
```
Model Metadata Structure:
{
  "model_id": "fraud_detection_v2.1.3",
  "created_by": "ml_team_fraud",
  "created_at": "2024-01-15T10:30:00Z",
  "model_type": "gradient_boosting",
  "framework": "xgboost",
  "framework_version": "1.7.0",
  "training_data": {
    "dataset_id": "fraud_training_20240115",
    "data_version": "v1.2",
    "row_count": 5000000,
    "feature_count": 147,
    "time_range": ["2023-01-01", "2024-01-01"]
  },
  "performance_metrics": {
    "accuracy": 0.967,
    "precision": 0.923,
    "recall": 0.945,
    "f1_score": 0.934,
    "auc_roc": 0.982
  },
  "infrastructure_requirements": {
    "memory_mb": 4096,
    "cpu_cores": 2,
    "gpu_required": false,
    "disk_space_mb": 512
  },
  "dependencies": {
    "python_version": "3.9.0",
    "packages": ["xgboost==1.7.0", "pandas==1.4.0", "numpy==1.21.0"]
  },
  "lineage": {
    "parent_models": ["fraud_detection_v2.1.2"],
    "training_code_commit": "abc123def456",
    "experiment_id": "exp_20240115_001"
  }
}
```

**Semantic Versioning for ML Models:**
```
Version Format: MAJOR.MINOR.PATCH

MAJOR version: Incompatible API changes
- Input/output schema changes
- Feature set modifications
- Model architecture changes
- Breaking prediction format changes

MINOR version: Backward-compatible functionality additions
- Performance improvements
- Additional output fields
- New optional parameters
- Extended feature support

PATCH version: Backward-compatible bug fixes
- Bug fixes in preprocessing
- Model parameter optimizations
- Documentation updates
- Minor performance tweaks

Examples:
fraud_detection_v1.0.0 → fraud_detection_v1.1.0 (new features)
fraud_detection_v1.1.0 → fraud_detection_v2.0.0 (breaking changes)
fraud_detection_v2.0.0 → fraud_detection_v2.0.1 (bug fixes)
```

### **Model Storage and Artifacts**

**Artifact Management Strategy:**
```
Model Artifact Structure:
model_registry/
├── models/
│   ├── fraud_detection/
│   │   ├── v2.1.3/
│   │   │   ├── model.pkl          # Serialized model
│   │   │   ├── preprocessing.pkl   # Feature preprocessing pipeline
│   │   │   ├── metadata.json      # Model metadata
│   │   │   ├── requirements.txt    # Python dependencies
│   │   │   ├── model_card.md      # Model documentation
│   │   │   ├── validation_report.html # Performance analysis
│   │   │   └── artifacts/
│   │   │       ├── training_config.yaml
│   │   │       ├── feature_importance.json
│   │   │       └── evaluation_plots/
│   │   └── aliases/
│   │       ├── latest → v2.1.3
│   │       ├── production → v2.1.2
│   │       └── staging → v2.1.3

Storage Optimization:
- Deduplication: Share common artifacts across versions
- Compression: Reduce storage costs for large models
- Hierarchical storage: Hot/warm/cold storage tiers
- Backup and replication: Disaster recovery
```

**Model Serialization Standards:**
```
Serialization Format Comparison:

Pickle (Python):
Pros: Native Python support, handles complex objects
Cons: Security risks, Python-specific, version sensitivity
Use case: Rapid prototyping, internal tools

ONNX (Open Neural Network Exchange):
Pros: Framework agnostic, hardware optimization support
Cons: Limited to neural networks, complex for traditional ML
Use case: Deep learning models, cross-platform deployment

MLflow Format:
Pros: Framework agnostic, includes metadata, versioning support
Cons: Additional dependency, learning curve
Use case: Production MLOps workflows

Custom Binary Formats:
Pros: Optimized for specific use cases, minimal dependencies
Cons: Development overhead, limited tooling support
Use case: High-performance serving, embedded systems
```

---

## 🔄 Model Lifecycle Automation

### **Lifecycle Stages and Transitions**

**Model Lifecycle State Machine:**
```
Lifecycle States:
1. Development: Model being developed and trained
2. Validation: Model undergoing validation and testing
3. Staging: Model deployed in staging environment
4. Production: Model serving production traffic
5. Shadow: Model running in shadow mode for comparison
6. Deprecated: Model marked for retirement
7. Archived: Model moved to long-term storage

State Transitions:
Development → Validation: Automated when training completes
Validation → Staging: Manual approval after validation passes
Staging → Production: Automated after A/B testing success
Production → Deprecated: Manual decision or automated policy
Deprecated → Archived: Automated after retention period

Transition Policies:
def can_transition(current_state, target_state, model_metadata):
    policies = {
        ('development', 'validation'): validate_training_completion,
        ('validation', 'staging'): validate_performance_metrics,
        ('staging', 'production'): validate_ab_test_results,
        ('production', 'deprecated'): check_replacement_available
    }
    
    policy_func = policies.get((current_state, target_state))
    return policy_func(model_metadata) if policy_func else False
```

**Automated Lifecycle Policies:**
```
Retirement Policies:
1. Performance-based: Retire when accuracy drops below threshold
2. Age-based: Retire models older than specified duration
3. Usage-based: Retire models with low traffic
4. Replacement-based: Retire when better model available

Policy Implementation:
class ModelRetirementPolicy:
    def __init__(self, performance_threshold=0.05, max_age_days=180):
        self.performance_threshold = performance_threshold
        self.max_age_days = max_age_days
    
    def should_retire(self, model):
        # Performance degradation check
        if model.current_performance < model.baseline_performance - self.performance_threshold:
            return True, "Performance degradation"
        
        # Age-based retirement
        age_days = (datetime.now() - model.created_at).days
        if age_days > self.max_age_days:
            return True, "Model too old"
        
        # Usage-based retirement
        if model.daily_requests < 100:  # Low usage threshold
            return True, "Low usage"
        
        return False, None

Promotion Policies:
- Champion-challenger framework
- Gradual traffic ramp-up
- Performance-based automatic promotion
- Business metric improvement requirements
```

### **Model Governance and Compliance**

**Governance Framework:**
```
Model Approval Workflow:
1. Technical Review:
   - Code quality assessment
   - Performance validation
   - Security vulnerability scan
   - Resource requirement verification

2. Business Review:
   - Business metric impact analysis
   - Risk assessment
   - Compliance verification
   - Stakeholder approval

3. Operational Review:
   - Deployment readiness
   - Monitoring setup
   - Rollback plan verification
   - Documentation completeness

Approval Gates:
def evaluate_approval_gates(model):
    gates = {
        'technical': TechnicalReviewGate(),
        'business': BusinessReviewGate(),
        'operational': OperationalReviewGate(),
        'security': SecurityReviewGate()
    }
    
    results = {}
    for gate_name, gate in gates.items():
        passed, issues = gate.evaluate(model)
        results[gate_name] = {'passed': passed, 'issues': issues}
    
    all_passed = all(result['passed'] for result in results.values())
    return all_passed, results
```

**Compliance and Auditing:**
```
Regulatory Compliance:
- GDPR: Data privacy and user consent tracking
- CCPA: California consumer privacy requirements
- SOX: Financial reporting model governance
- HIPAA: Healthcare data protection requirements

Audit Trail Requirements:
{
  "audit_id": "audit_20240115_001",
  "model_id": "fraud_detection_v2.1.3",
  "action": "production_deployment",
  "timestamp": "2024-01-15T14:30:00Z",
  "user": "ml_engineer_alice",
  "approvers": ["tech_lead_bob", "product_manager_carol"],
  "rationale": "Performance improvement validated in A/B test",
  "evidence": {
    "test_results": "ab_test_report_20240115.pdf",
    "performance_metrics": {...},
    "approval_documents": [...]
  },
  "rollback_plan": "rollback_plan_20240115.md"
}

Compliance Monitoring:
- Regular compliance audits
- Automated policy violation detection
- Risk assessment updates
- Remediation tracking
```

---

## 🔍 Model Lineage and Provenance

### **Data and Model Lineage Tracking**

**Lineage Graph Construction:**
```
Lineage Entities:
- Data Sources: Databases, files, APIs, streams
- Datasets: Processed data used for training/validation
- Features: Engineered features and transformations
- Models: Trained ML models and their versions
- Predictions: Model outputs and decisions
- Downstream Systems: Applications consuming predictions

Lineage Relationships:
- derives_from: Model derives from training dataset
- uses_feature: Model uses specific feature
- generates: Model generates predictions
- influences: Prediction influences business decision
- version_of: Model is version of previous model

Graph Representation:
class LineageNode:
    def __init__(self, entity_id, entity_type, metadata):
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.metadata = metadata
        self.parents = []
        self.children = []

class LineageEdge:
    def __init__(self, source, target, relationship_type, metadata):
        self.source = source
        self.target = target
        self.relationship_type = relationship_type
        self.metadata = metadata
```

**Impact Analysis:**
```
Change Impact Assessment:
When dataset X changes, what models are affected?
When model Y is updated, what downstream systems need updates?
When feature Z is deprecated, which models need retraining?

Query Examples:
# Find all models affected by data source change
def find_affected_models(data_source_id):
    affected_entities = []
    
    # Traverse lineage graph from data source
    stack = [data_source_id]
    visited = set()
    
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        
        visited.add(current)
        entity = lineage_graph.get_entity(current)
        
        if entity.entity_type == 'model':
            affected_entities.append(entity)
        
        # Add children to traversal stack
        for child in entity.children:
            stack.append(child.entity_id)
    
    return affected_entities

# Estimate retraining cost
def estimate_retraining_cost(affected_models):
    total_cost = 0
    for model in affected_models:
        training_time = model.metadata.get('training_time_hours', 1)
        compute_cost_per_hour = model.metadata.get('compute_cost_per_hour', 10)
        total_cost += training_time * compute_cost_per_hour
    
    return total_cost
```

### **Reproducibility and Experiment Tracking**

**Reproducible Model Training:**
```
Reproducibility Requirements:
1. Code Version: Exact commit hash of training code
2. Data Version: Immutable dataset with content hash
3. Environment: Container image or environment specification
4. Random Seeds: Fixed seeds for all random operations
5. Hardware: GPU type and driver versions (for deterministic ops)

Reproducibility Manifest:
{
  "experiment_id": "exp_20240115_001",
  "model_version": "fraud_detection_v2.1.3",
  "reproducibility": {
    "code_commit": "abc123def456",
    "data_hash": "sha256:fedcba987654...",
    "environment_image": "ml-training:v1.2.3",
    "random_seed": 42,
    "hardware_spec": {
      "gpu_type": "Tesla V100",
      "cuda_version": "11.8",
      "driver_version": "520.61.05"
    },
    "framework_versions": {
      "python": "3.9.16",
      "tensorflow": "2.11.0",
      "numpy": "1.21.6"
    }
  }
}

Reproducibility Validation:
def validate_reproducibility(experiment_id):
    original = get_experiment_results(experiment_id)
    reproduction = rerun_experiment(experiment_id)
    
    # Compare key metrics within tolerance
    metrics_match = True
    for metric, original_value in original.metrics.items():
        reproduced_value = reproduction.metrics.get(metric)
        if abs(original_value - reproduced_value) > 0.001:  # Tolerance
            metrics_match = False
            break
    
    return metrics_match
```

**Experiment Comparison and Analysis:**
```
A/B Experiment Tracking:
{
  "experiment_id": "ab_test_20240115",
  "baseline_model": "fraud_detection_v2.1.2",
  "treatment_model": "fraud_detection_v2.1.3",
  "traffic_split": {"baseline": 0.8, "treatment": 0.2},
  "duration": "7 days",
  "success_criteria": {
    "primary_metric": "precision",
    "improvement_threshold": 0.02,
    "significance_level": 0.05
  },
  "results": {
    "baseline_performance": {"precision": 0.91, "recall": 0.94},
    "treatment_performance": {"precision": 0.93, "recall": 0.95},
    "statistical_significance": 0.032,
    "business_impact": "$50K annual revenue increase"
  }
}

Model Comparison Framework:
def compare_models(model_a, model_b, test_dataset):
    comparison = {
        'model_a': model_a.version,
        'model_b': model_b.version,
        'metrics': {},
        'significance_tests': {},
        'recommendation': None
    }
    
    # Performance comparison
    for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
        a_score = evaluate_metric(model_a, test_dataset, metric_name)
        b_score = evaluate_metric(model_b, test_dataset, metric_name)
        
        comparison['metrics'][metric_name] = {
            'model_a': a_score,
            'model_b': b_score,
            'difference': b_score - a_score,
            'percent_change': ((b_score - a_score) / a_score) * 100
        }
        
        # Statistical significance test
        p_value = mcnemar_test(model_a, model_b, test_dataset, metric_name)
        comparison['significance_tests'][metric_name] = {
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    # Generate recommendation
    if any(test['significant'] for test in comparison['significance_tests'].values()):
        comparison['recommendation'] = 'Model B significantly different'
    else:
        comparison['recommendation'] = 'No significant difference'
    
    return comparison
```

---

## 🎯 Advanced Registry Features

### **Model Discovery and Search**

**Semantic Search and Recommendation:**
```
Model Discovery Features:
1. Semantic Search: Find models by description, use case, domain
2. Similar Model Recommendation: Based on metadata similarity
3. Performance-based Filtering: Search by accuracy, latency requirements
4. Lineage-based Discovery: Find related models in lineage graph

Search Implementation:
class ModelSearchEngine:
    def __init__(self, vector_embeddings_model):
        self.embeddings_model = vector_embeddings_model
        self.model_embeddings = {}
        self.metadata_index = {}
    
    def index_model(self, model):
        # Create text representation for embedding
        text_repr = f"{model.name} {model.description} {model.domain} {model.use_case}"
        embedding = self.embeddings_model.encode(text_repr)
        self.model_embeddings[model.id] = embedding
        self.metadata_index[model.id] = model.metadata
    
    def semantic_search(self, query, top_k=10):
        query_embedding = self.embeddings_model.encode(query)
        
        similarities = {}
        for model_id, model_embedding in self.model_embeddings.items():
            similarity = cosine_similarity(query_embedding, model_embedding)
            similarities[model_id] = similarity
        
        # Return top-k most similar models
        ranked_models = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return ranked_models[:top_k]

Model Recommendation:
def recommend_similar_models(target_model, registry, top_k=5):
    target_features = extract_model_features(target_model)
    
    similarities = []
    for model in registry.list_models():
        if model.id == target_model.id:
            continue
        
        model_features = extract_model_features(model)
        similarity = compute_feature_similarity(target_features, model_features)
        similarities.append((model, similarity))
    
    # Sort by similarity and return top-k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [model for model, score in similarities[:top_k]]

def extract_model_features(model):
    return {
        'domain': model.metadata.get('domain'),
        'model_type': model.metadata.get('model_type'),
        'feature_count': model.metadata.get('feature_count'),
        'performance_tier': categorize_performance(model.metrics),
        'data_size_tier': categorize_data_size(model.training_data_size)
    }
```

### **Model Registry Integration**

**CI/CD Pipeline Integration:**
```
Automated Model Registration:
# In training pipeline
def register_model_automatically(model, training_metadata):
    registry = ModelRegistry()
    
    # Generate model version
    latest_version = registry.get_latest_version(model.name)
    new_version = increment_version(latest_version, change_type='minor')
    
    # Create model metadata
    metadata = {
        'version': new_version,
        'created_by': os.environ.get('BUILD_USER'),
        'created_at': datetime.utcnow().isoformat(),
        'training_job_id': os.environ.get('JOB_ID'),
        'git_commit': os.environ.get('GIT_COMMIT'),
        'performance_metrics': training_metadata['metrics'],
        'validation_results': training_metadata['validation']
    }
    
    # Register model
    model_uri = registry.register_model(
        name=model.name,
        version=new_version,
        model_artifact=model,
        metadata=metadata
    )
    
    return model_uri

Deployment Integration:
def deploy_model_from_registry(model_name, version, environment):
    registry = ModelRegistry()
    
    # Fetch model from registry
    model_info = registry.get_model(model_name, version)
    
    # Validate deployment readiness
    if not model_info.is_deployment_ready(environment):
        raise DeploymentError(f"Model not ready for {environment} deployment")
    
    # Deploy using infrastructure as code
    deployment_config = generate_deployment_config(model_info, environment)
    deploy_to_kubernetes(deployment_config)
    
    # Update model status
    registry.update_model_status(model_name, version, f'deployed_{environment}')
```

This comprehensive framework for model versioning and lifecycle management provides the foundation for robust, scalable, and compliant ML operations. The key insight is that effective model lifecycle management requires careful coordination between technical systems, business processes, and governance requirements to ensure reliable and auditable ML deployments.