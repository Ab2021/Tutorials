# Day 10.3: Advanced Model Versioning & Registry

## 📚 Advanced MLOps & Unified Pipelines - Part 3

**Focus**: Enterprise Model Management, Lineage Tracking, Governance Workflows  
**Duration**: 2-3 hours  
**Level**: Advanced  

---

## 🎯 Learning Objectives

- Master advanced model versioning strategies and semantic versioning for ML systems
- Learn comprehensive model registry architectures with lineage and governance capabilities
- Understand automated model promotion workflows and approval processes
- Analyze model artifact management and optimization techniques for large-scale deployments

---

## 📚 Advanced Model Registry Architecture

### **Enterprise Model Management Theory**

Advanced model registries serve as the central nervous system for ML operations, providing version control, lineage tracking, metadata management, and governance workflows across the entire model lifecycle.

**Model Registry Theoretical Framework:**
```
Model Registry Architecture Components:
1. Version Management Layer:
   - Semantic versioning for models
   - Branch and merge strategies
   - Compatibility matrices
   - Breaking change detection

2. Metadata Management Layer:
   - Model performance metrics
   - Training metadata and hyperparameters
   - Data lineage and dependencies
   - Business context and ownership

3. Artifact Storage Layer:
   - Model binary storage optimization
   - Compression and deduplication
   - Multi-format support
   - Distributed storage backends

4. Governance Layer:
   - Approval workflows and policies
   - Compliance and audit trails
   - Access control and permissions
   - Quality gates and validation

Model Version Semantic Theory:
Version = Major.Minor.Patch[-Qualifier]

Where:
- Major: Breaking changes in API, significant architecture changes
- Minor: Backward-compatible functionality additions, performance improvements
- Patch: Bug fixes, minor performance improvements
- Qualifier: pre-release identifiers (alpha, beta, rc)

Model Compatibility Matrix:
Compatibility_Score = API_Compatibility × Performance_Compatibility × Data_Compatibility

Model Lineage Graph:
G = (V, E) where:
V = {models, datasets, experiments, deployments}
E = {derived_from, trained_on, deployed_as, influenced_by}

Registry Performance Optimization:
Query_Latency = f(Index_Efficiency, Cache_Hit_Rate, Network_Latency)
Storage_Efficiency = Compression_Ratio × Deduplication_Factor
```

**Advanced Model Registry Implementation:**
```
Enterprise Model Registry System:
import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

class ModelStage(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"

class ModelFormat(Enum):
    ONNX = "onnx"
    TENSORFLOW_SAVED_MODEL = "tensorflow_saved_model"
    PYTORCH = "pytorch"
    SKLEARN = "sklearn"
    CUSTOM = "custom"

@dataclass
class ModelVersion:
    model_name: str
    version: str
    stage: ModelStage
    format: ModelFormat
    artifact_uri: str
    created_timestamp: datetime
    created_by: str
    
    # Performance metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    
    # Model metadata
    framework: Optional[str] = None
    framework_version: Optional[str] = None
    model_size_bytes: Optional[int] = None
    input_schema: Optional[Dict] = None
    output_schema: Optional[Dict] = None
    
    # Training metadata
    training_dataset: Optional[str] = None
    hyperparameters: Optional[Dict] = None
    training_duration_seconds: Optional[int] = None
    training_cost: Optional[float] = None
    
    # Governance metadata
    approval_status: Optional[str] = None
    approved_by: Optional[str] = None
    approval_timestamp: Optional[datetime] = None
    
    # Lineage
    parent_versions: List[str] = None
    derived_from_experiment: Optional[str] = None
    
    # Additional metadata
    description: Optional[str] = None
    tags: Dict[str, str] = None

class AdvancedModelRegistry:
    def __init__(self, storage_backend, metadata_store, artifact_store):
        self.storage_backend = storage_backend
        self.metadata_store = metadata_store
        self.artifact_store = artifact_store
        self.version_manager = ModelVersionManager()
        self.lineage_tracker = ModelLineageTracker()
        self.governance_engine = ModelGovernanceEngine()
        self.performance_analyzer = ModelPerformanceAnalyzer()
    
    def register_model(self, model_registration_request):
        """Register a new model with comprehensive metadata"""
        
        # Generate unique model ID
        model_id = str(uuid.uuid4())
        
        # Extract and validate model artifacts
        artifact_analysis = self._analyze_model_artifact(
            model_registration_request.artifact_path
        )
        
        # Generate version number
        version = self.version_manager.generate_version(
            model_name=model_registration_request.name,
            change_type=model_registration_request.change_type,
            previous_version=model_registration_request.parent_version
        )
        
        # Create model version object
        model_version = ModelVersion(
            model_name=model_registration_request.name,
            version=version,
            stage=ModelStage.DEVELOPMENT,
            format=artifact_analysis.format,
            artifact_uri=model_registration_request.artifact_path,
            created_timestamp=datetime.now(timezone.utc),
            created_by=model_registration_request.created_by,
            
            # Artifact metadata
            framework=artifact_analysis.framework,
            framework_version=artifact_analysis.framework_version,
            model_size_bytes=artifact_analysis.size_bytes,
            input_schema=artifact_analysis.input_schema,
            output_schema=artifact_analysis.output_schema,
            
            # Training metadata
            training_dataset=model_registration_request.training_dataset,
            hyperparameters=model_registration_request.hyperparameters,
            training_duration_seconds=model_registration_request.training_duration,
            training_cost=model_registration_request.training_cost,
            
            # Performance metrics
            accuracy=model_registration_request.performance_metrics.get('accuracy'),
            precision=model_registration_request.performance_metrics.get('precision'),
            recall=model_registration_request.performance_metrics.get('recall'),
            f1_score=model_registration_request.performance_metrics.get('f1_score'),
            
            # Lineage
            parent_versions=model_registration_request.parent_versions,
            derived_from_experiment=model_registration_request.experiment_id,
            
            # Metadata
            description=model_registration_request.description,
            tags=model_registration_request.tags or {}
        )
        
        # Store model artifact with optimization
        optimized_artifact_uri = self._store_optimized_artifact(
            model_version, model_registration_request.artifact_path
        )
        model_version.artifact_uri = optimized_artifact_uri
        
        # Store model metadata
        model_version_id = self.metadata_store.store_model_version(model_version)
        
        # Update lineage graph
        self.lineage_tracker.add_model_version(model_version)
        
        # Apply governance policies
        governance_result = self.governance_engine.apply_registration_policies(model_version)
        
        # Index for fast retrieval
        self._index_model_version(model_version)
        
        return {
            'model_id': model_id,
            'model_version_id': model_version_id,
            'version': version,
            'artifact_uri': optimized_artifact_uri,
            'governance_result': governance_result,
            'registration_timestamp': model_version.created_timestamp.isoformat()
        }
    
    def _analyze_model_artifact(self, artifact_path):
        """Analyze model artifact to extract metadata"""
        
        # Determine model format
        model_format = self._detect_model_format(artifact_path)
        
        # Extract framework information
        framework_info = self._extract_framework_info(artifact_path, model_format)
        
        # Analyze model structure
        structure_info = self._analyze_model_structure(artifact_path, model_format)
        
        # Calculate model size
        model_size = self._calculate_model_size(artifact_path)
        
        # Extract input/output schemas
        schemas = self._extract_model_schemas(artifact_path, model_format)
        
        return {
            'format': model_format,
            'framework': framework_info.get('framework'),
            'framework_version': framework_info.get('version'),
            'size_bytes': model_size,
            'input_schema': schemas.get('input'),
            'output_schema': schemas.get('output'),
            'layer_count': structure_info.get('layer_count'),
            'parameter_count': structure_info.get('parameter_count'),
            'model_complexity': structure_info.get('complexity_score')
        }
    
    def _store_optimized_artifact(self, model_version, artifact_path):
        """Store model artifact with optimization techniques"""
        
        # Apply compression based on model format
        compression_strategy = self._select_compression_strategy(model_version.format)
        
        # Check for deduplication opportunities
        duplicate_check = self._check_for_duplicates(artifact_path)
        
        if duplicate_check.is_duplicate:
            # Reference existing artifact
            return duplicate_check.existing_uri
        
        # Compress artifact if beneficial
        if compression_strategy.should_compress:
            compressed_path = self._compress_artifact(
                artifact_path, compression_strategy.method
            )
            storage_path = compressed_path
        else:
            storage_path = artifact_path
        
        # Store in distributed storage with redundancy
        storage_result = self.artifact_store.store_with_redundancy(
            storage_path,
            redundancy_level=self._determine_redundancy_level(model_version),
            storage_class=self._determine_storage_class(model_version)
        )
        
        return storage_result.uri
    
    def promote_model(self, model_name, version, target_stage, promotion_request):
        """Promote model to different stage with governance checks"""
        
        # Get current model version
        current_version = self.metadata_store.get_model_version(model_name, version)
        
        if not current_version:
            raise ValueError(f"Model {model_name} version {version} not found")
        
        # Check promotion eligibility
        eligibility_check = self.governance_engine.check_promotion_eligibility(
            current_version, target_stage
        )
        
        if not eligibility_check.eligible:
            raise ValueError(f"Model not eligible for promotion: {eligibility_check.reasons}")
        
        # Execute pre-promotion validations
        validation_results = self._execute_promotion_validations(
            current_version, target_stage, promotion_request
        )
        
        if not validation_results.all_passed:
            return {
                'status': 'validation_failed',
                'validation_results': validation_results,
                'required_actions': validation_results.required_actions
            }
        
        # Create promotion workflow
        workflow_id = self.governance_engine.create_promotion_workflow(
            model_version=current_version,
            target_stage=target_stage,
            requested_by=promotion_request.requested_by,
            justification=promotion_request.justification
        )
        
        # If auto-approval is enabled and conditions are met
        if self._should_auto_approve(current_version, target_stage, promotion_request):
            approval_result = self._auto_approve_promotion(workflow_id)
        else:
            # Submit for manual approval
            approval_result = self.governance_engine.submit_for_approval(
                workflow_id, promotion_request
            )
        
        return {
            'status': 'promotion_initiated',
            'workflow_id': workflow_id,
            'approval_result': approval_result,
            'estimated_completion_time': self._estimate_promotion_completion(target_stage)
        }
    
    def _execute_promotion_validations(self, model_version, target_stage, promotion_request):
        """Execute comprehensive validation checks for model promotion"""
        
        validation_results = []
        
        # Performance validation
        performance_validation = self._validate_model_performance(
            model_version, target_stage
        )
        validation_results.append(performance_validation)
        
        # Security validation
        security_validation = self._validate_model_security(model_version)
        validation_results.append(security_validation)
        
        # Compatibility validation
        compatibility_validation = self._validate_model_compatibility(
            model_version, target_stage
        )
        validation_results.append(compatibility_validation)
        
        # Business validation
        business_validation = self._validate_business_requirements(
            model_version, target_stage, promotion_request
        )
        validation_results.append(business_validation)
        
        # Regulatory validation
        if target_stage == ModelStage.PRODUCTION:
            regulatory_validation = self._validate_regulatory_compliance(model_version)
            validation_results.append(regulatory_validation)
        
        all_passed = all(result.passed for result in validation_results)
        required_actions = [
            action for result in validation_results 
            for action in result.required_actions
            if not result.passed
        ]
        
        return {
            'all_passed': all_passed,
            'validation_details': validation_results,
            'required_actions': required_actions
        }
    
    def get_model_lineage(self, model_name, version, depth=5):
        """Get comprehensive model lineage including ancestors and descendants"""
        
        lineage_graph = self.lineage_tracker.build_lineage_graph(
            model_name, version, max_depth=depth
        )
        
        return {
            'model_name': model_name,
            'version': version,
            'lineage_graph': lineage_graph,
            'ancestors': self._extract_ancestors(lineage_graph),
            'descendants': self._extract_descendants(lineage_graph),
            'related_experiments': self._get_related_experiments(lineage_graph),
            'data_dependencies': self._get_data_dependencies(lineage_graph)
        }

Model Version Manager:
class ModelVersionManager:
    def __init__(self):
        self.versioning_strategies = {
            'semantic': SemanticVersioningStrategy(),
            'timestamp': TimestampVersioningStrategy(),
            'hash': HashVersioningStrategy(),
            'custom': CustomVersioningStrategy()
        }
    
    def generate_version(self, model_name, change_type, previous_version=None, 
                        versioning_strategy='semantic'):
        """Generate next version number based on change type and strategy"""
        
        strategy = self.versioning_strategies.get(versioning_strategy)
        if not strategy:
            raise ValueError(f"Unknown versioning strategy: {versioning_strategy}")
        
        return strategy.generate_version(model_name, change_type, previous_version)

class SemanticVersioningStrategy:
    def generate_version(self, model_name, change_type, previous_version=None):
        """Generate semantic version (Major.Minor.Patch)"""
        
        if not previous_version:
            return "1.0.0"
        
        # Parse previous version
        major, minor, patch = self._parse_version(previous_version)
        
        # Increment based on change type
        if change_type == 'breaking':
            return f"{major + 1}.0.0"
        elif change_type == 'feature':
            return f"{major}.{minor + 1}.0"
        elif change_type == 'patch':
            return f"{major}.{minor}.{patch + 1}"
        else:
            # Default to patch increment
            return f"{major}.{minor}.{patch + 1}"
    
    def _parse_version(self, version_string):
        """Parse version string into major, minor, patch components"""
        
        # Handle pre-release qualifiers
        base_version = version_string.split('-')[0]
        
        try:
            parts = base_version.split('.')
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
            patch = int(parts[2]) if len(parts) > 2 else 0
            return major, minor, patch
        except (ValueError, IndexError):
            raise ValueError(f"Invalid version format: {version_string}")

Model Lineage Tracker:
class ModelLineageTracker:
    def __init__(self, graph_store):
        self.graph_store = graph_store
        self.lineage_analyzer = LineageAnalyzer()
    
    def add_model_version(self, model_version):
        """Add model version to lineage graph"""
        
        # Create model node
        model_node = {
            'id': f"{model_version.model_name}:{model_version.version}",
            'type': 'model',
            'name': model_version.model_name,
            'version': model_version.version,
            'created_timestamp': model_version.created_timestamp.isoformat(),
            'created_by': model_version.created_by,
            'metadata': {
                'stage': model_version.stage.value,
                'format': model_version.format.value,
                'size_bytes': model_version.model_size_bytes,
                'performance_metrics': {
                    'accuracy': model_version.accuracy,
                    'precision': model_version.precision,
                    'recall': model_version.recall,
                    'f1_score': model_version.f1_score
                }
            }
        }
        
        self.graph_store.add_node(model_node)
        
        # Add parent relationships
        if model_version.parent_versions:
            for parent_version in model_version.parent_versions:
                self.graph_store.add_edge(
                    source=parent_version,
                    target=model_node['id'],
                    relationship_type='derived_from',
                    metadata={
                        'derivation_type': 'model_evolution',
                        'timestamp': model_version.created_timestamp.isoformat()
                    }
                )
        
        # Add experiment relationship
        if model_version.derived_from_experiment:
            experiment_node_id = f"experiment:{model_version.derived_from_experiment}"
            
            # Ensure experiment node exists
            if not self.graph_store.node_exists(experiment_node_id):
                self._create_experiment_node(model_version.derived_from_experiment)
            
            self.graph_store.add_edge(
                source=experiment_node_id,
                target=model_node['id'],
                relationship_type='produced',
                metadata={
                    'timestamp': model_version.created_timestamp.isoformat()
                }
            )
        
        # Add dataset relationship
        if model_version.training_dataset:
            dataset_node_id = f"dataset:{model_version.training_dataset}"
            
            # Ensure dataset node exists
            if not self.graph_store.node_exists(dataset_node_id):
                self._create_dataset_node(model_version.training_dataset)
            
            self.graph_store.add_edge(
                source=dataset_node_id,
                target=model_node['id'],
                relationship_type='trained_on',
                metadata={
                    'timestamp': model_version.created_timestamp.isoformat()
                }
            )
    
    def build_lineage_graph(self, model_name, version, max_depth=5):
        """Build comprehensive lineage graph for a model version"""
        
        root_node_id = f"{model_name}:{version}"
        
        # Perform graph traversal
        lineage_subgraph = self.graph_store.get_subgraph(
            root_node_id, max_depth=max_depth
        )
        
        # Enrich with additional metadata
        enriched_graph = self.lineage_analyzer.enrich_lineage_graph(lineage_subgraph)
        
        return enriched_graph
    
    def analyze_model_impact(self, model_name, version):
        """Analyze the impact of changes to a specific model version"""
        
        model_node_id = f"{model_name}:{version}"
        
        # Find all downstream dependencies
        downstream_dependencies = self.graph_store.get_descendants(
            model_node_id, relationship_types=['derived_from', 'uses']
        )
        
        # Analyze impact on each dependency
        impact_analysis = []
        
        for dependency in downstream_dependencies:
            impact_score = self._calculate_impact_score(model_node_id, dependency['id'])
            
            impact_analysis.append({
                'dependency_id': dependency['id'],
                'dependency_type': dependency['type'],
                'impact_score': impact_score,
                'impact_category': self._categorize_impact(impact_score),
                'recommended_actions': self._get_impact_recommendations(
                    dependency, impact_score
                )
            })
        
        return {
            'model_id': model_node_id,
            'total_dependencies': len(downstream_dependencies),
            'high_impact_dependencies': len([
                impact for impact in impact_analysis 
                if impact['impact_category'] == 'high'
            ]),
            'impact_analysis': impact_analysis
        }
    
    def _calculate_impact_score(self, source_model_id, target_model_id):
        """Calculate impact score between two models"""
        
        # Get model metadata for both models
        source_metadata = self.graph_store.get_node_metadata(source_model_id)
        target_metadata = self.graph_store.get_node_metadata(target_model_id)
        
        # Calculate impact based on various factors
        impact_factors = {
            'performance_dependency': 0.3,  # How much target depends on source performance
            'usage_frequency': 0.2,         # How frequently target uses source
            'criticality': 0.3,             # How critical target is to business
            'update_difficulty': 0.2        # How difficult it is to update target
        }
        
        impact_score = 0
        
        # Performance dependency score
        if target_metadata.get('derives_performance_from_source', False):
            impact_score += impact_factors['performance_dependency']
        
        # Usage frequency score
        usage_frequency = target_metadata.get('source_usage_frequency', 0)
        normalized_frequency = min(usage_frequency / 1000, 1.0)  # Normalize to 0-1
        impact_score += impact_factors['usage_frequency'] * normalized_frequency
        
        # Criticality score
        target_criticality = target_metadata.get('business_criticality', 'medium')
        criticality_scores = {'low': 0.2, 'medium': 0.5, 'high': 0.8, 'critical': 1.0}
        impact_score += impact_factors['criticality'] * criticality_scores.get(target_criticality, 0.5)
        
        # Update difficulty score
        update_difficulty = target_metadata.get('update_complexity', 'medium')
        difficulty_scores = {'easy': 0.2, 'medium': 0.5, 'hard': 0.8, 'very_hard': 1.0}
        impact_score += impact_factors['update_difficulty'] * difficulty_scores.get(update_difficulty, 0.5)
        
        return min(impact_score, 1.0)  # Cap at 1.0
```

This comprehensive framework for advanced model versioning and registry management provides the theoretical foundations and practical strategies for implementing enterprise-grade model governance systems. The key insight is that effective model management requires sophisticated versioning strategies, comprehensive lineage tracking, and robust governance workflows to maintain model quality and regulatory compliance at scale.