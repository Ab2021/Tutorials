# Day 7.2 Part 2: Production Online Learning and Model Serving

## Learning Objectives
- Implement model versioning and A/B testing for online learning systems
- Design continuous integration/deployment pipelines for ML models
- Build real-time model serving with inference optimization
- Master multi-armed bandit approaches for online experimentation
- Develop production-grade online learning architectures

## 1. Model Versioning and A/B Testing

### Model Version Management System

```python
import asyncio
import json
import pickle
import hashlib
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import logging
import threading
import time
import sqlite3
import redis
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    """Model deployment status"""
    TRAINING = "training"
    STAGED = "staged"
    PRODUCTION = "production"
    SHADOW = "shadow"
    RETIRED = "retired"
    FAILED = "failed"

@dataclass
class ModelVersion:
    """Model version metadata"""
    model_id: str
    version: str
    algorithm: str
    status: ModelStatus
    created_at: datetime
    deployed_at: Optional[datetime] = None
    retired_at: Optional[datetime] = None
    performance_metrics: Dict[str, float] = None
    hyperparameters: Dict[str, Any] = None
    model_path: Optional[str] = None
    model_hash: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.hyperparameters is None:
            self.hyperparameters = {}
        if self.metadata is None:
            self.metadata = {}

class ModelRegistry:
    """Registry for managing model versions and deployments"""
    
    def __init__(self, storage_backend: str = "sqlite", storage_path: str = "model_registry.db"):
        self.storage_backend = storage_backend
        self.storage_path = storage_path
        self.models: Dict[str, Dict[str, ModelVersion]] = defaultdict(dict)
        
        # Initialize storage
        if storage_backend == "sqlite":
            self._init_sqlite()
        elif storage_backend == "redis":
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        
        # Load existing models
        self._load_models()
    
    def _init_sqlite(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_versions (
                model_id TEXT,
                version TEXT,
                algorithm TEXT,
                status TEXT,
                created_at TEXT,
                deployed_at TEXT,
                retired_at TEXT,
                performance_metrics TEXT,
                hyperparameters TEXT,
                model_path TEXT,
                model_hash TEXT,
                metadata TEXT,
                PRIMARY KEY (model_id, version)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_models(self):
        """Load models from storage"""
        if self.storage_backend == "sqlite":
            self._load_from_sqlite()
        elif self.storage_backend == "redis":
            self._load_from_redis()
    
    def _load_from_sqlite(self):
        """Load models from SQLite"""
        try:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM model_versions')
            rows = cursor.fetchall()
            
            for row in rows:
                model_version = ModelVersion(
                    model_id=row[0],
                    version=row[1],
                    algorithm=row[2],
                    status=ModelStatus(row[3]),
                    created_at=datetime.fromisoformat(row[4]),
                    deployed_at=datetime.fromisoformat(row[5]) if row[5] else None,
                    retired_at=datetime.fromisoformat(row[6]) if row[6] else None,
                    performance_metrics=json.loads(row[7]) if row[7] else {},
                    hyperparameters=json.loads(row[8]) if row[8] else {},
                    model_path=row[9],
                    model_hash=row[10],
                    metadata=json.loads(row[11]) if row[11] else {}
                )
                
                self.models[model_version.model_id][model_version.version] = model_version
            
            conn.close()
            logger.info(f"Loaded {len(rows)} model versions from SQLite")
            
        except Exception as e:
            logger.error(f"Error loading models from SQLite: {e}")
    
    def register_model(self, model_version: ModelVersion) -> bool:
        """Register a new model version"""
        try:
            # Add to memory
            self.models[model_version.model_id][model_version.version] = model_version
            
            # Persist to storage
            if self.storage_backend == "sqlite":
                self._save_to_sqlite(model_version)
            elif self.storage_backend == "redis":
                self._save_to_redis(model_version)
            
            logger.info(f"Registered model {model_version.model_id}:{model_version.version}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            return False
    
    def _save_to_sqlite(self, model_version: ModelVersion):
        """Save model version to SQLite"""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO model_versions 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_version.model_id,
            model_version.version,
            model_version.algorithm,
            model_version.status.value,
            model_version.created_at.isoformat(),
            model_version.deployed_at.isoformat() if model_version.deployed_at else None,
            model_version.retired_at.isoformat() if model_version.retired_at else None,
            json.dumps(model_version.performance_metrics),
            json.dumps(model_version.hyperparameters),
            model_version.model_path,
            model_version.model_hash,
            json.dumps(model_version.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def get_model_version(self, model_id: str, version: str) -> Optional[ModelVersion]:
        """Get specific model version"""
        return self.models.get(model_id, {}).get(version)
    
    def get_latest_version(self, model_id: str, status: Optional[ModelStatus] = None) -> Optional[ModelVersion]:
        """Get latest model version, optionally filtered by status"""
        if model_id not in self.models:
            return None
        
        versions = list(self.models[model_id].values())
        
        if status:
            versions = [v for v in versions if v.status == status]
        
        if not versions:
            return None
        
        # Sort by created_at and return latest
        return max(versions, key=lambda v: v.created_at)
    
    def update_model_status(self, model_id: str, version: str, 
                           new_status: ModelStatus) -> bool:
        """Update model status"""
        model_version = self.get_model_version(model_id, version)
        
        if not model_version:
            return False
        
        model_version.status = new_status
        
        if new_status == ModelStatus.PRODUCTION:
            model_version.deployed_at = datetime.now()
        elif new_status == ModelStatus.RETIRED:
            model_version.retired_at = datetime.now()
        
        # Update storage
        if self.storage_backend == "sqlite":
            self._save_to_sqlite(model_version)
        
        logger.info(f"Updated model {model_id}:{version} status to {new_status.value}")
        return True
    
    def update_performance_metrics(self, model_id: str, version: str, 
                                 metrics: Dict[str, float]) -> bool:
        """Update model performance metrics"""
        model_version = self.get_model_version(model_id, version)
        
        if not model_version:
            return False
        
        model_version.performance_metrics.update(metrics)
        
        # Update storage
        if self.storage_backend == "sqlite":
            self._save_to_sqlite(model_version)
        
        return True
    
    def list_models(self, status: Optional[ModelStatus] = None) -> List[ModelVersion]:
        """List all models, optionally filtered by status"""
        all_models = []
        
        for model_id, versions in self.models.items():
            for version, model_version in versions.items():
                if status is None or model_version.status == status:
                    all_models.append(model_version)
        
        return sorted(all_models, key=lambda m: m.created_at, reverse=True)

class ABTestManager:
    """Manages A/B testing for model deployments"""
    
    def __init__(self, model_registry: ModelRegistry):
        self.model_registry = model_registry
        self.active_experiments: Dict[str, 'ABExperiment'] = {}
        self.experiment_results: Dict[str, Dict[str, Any]] = {}
        
    def create_experiment(self, experiment_id: str, control_model: Tuple[str, str],
                         treatment_models: List[Tuple[str, str]], 
                         traffic_split: Dict[str, float],
                         success_metric: str, duration_hours: int = 24,
                         min_samples: int = 1000) -> bool:
        """Create new A/B test experiment"""
        
        # Validate models exist
        control_version = self.model_registry.get_model_version(*control_model)
        if not control_version:
            logger.error(f"Control model {control_model} not found")
            return False
        
        treatment_versions = []
        for model_id, version in treatment_models:
            model_version = self.model_registry.get_model_version(model_id, version)
            if not model_version:
                logger.error(f"Treatment model {model_id}:{version} not found")
                return False
            treatment_versions.append(model_version)
        
        # Create experiment
        experiment = ABExperiment(
            experiment_id=experiment_id,
            control_model=control_version,
            treatment_models=treatment_versions,
            traffic_split=traffic_split,
            success_metric=success_metric,
            duration_hours=duration_hours,
            min_samples=min_samples
        )
        
        self.active_experiments[experiment_id] = experiment
        logger.info(f"Created A/B test experiment: {experiment_id}")
        
        return True
    
    def assign_user_to_variant(self, experiment_id: str, user_id: str) -> Optional[str]:
        """Assign user to experiment variant"""
        if experiment_id not in self.active_experiments:
            return None
        
        experiment = self.active_experiments[experiment_id]
        return experiment.assign_user(user_id)
    
    def record_outcome(self, experiment_id: str, user_id: str, 
                      outcome_value: float, metadata: Dict[str, Any] = None):
        """Record experiment outcome"""
        if experiment_id not in self.active_experiments:
            return
        
        experiment = self.active_experiments[experiment_id]
        experiment.record_outcome(user_id, outcome_value, metadata)
    
    def get_experiment_results(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get current experiment results"""
        if experiment_id not in self.active_experiments:
            return None
        
        experiment = self.active_experiments[experiment_id]
        return experiment.get_results()
    
    def stop_experiment(self, experiment_id: str, winner_variant: Optional[str] = None) -> bool:
        """Stop experiment and optionally promote winner"""
        if experiment_id not in self.active_experiments:
            return False
        
        experiment = self.active_experiments[experiment_id]
        results = experiment.get_results()
        
        # Store results
        self.experiment_results[experiment_id] = {
            'results': results,
            'stopped_at': datetime.now(),
            'winner': winner_variant
        }
        
        # Promote winner if specified
        if winner_variant and winner_variant in experiment.variants:
            variant_info = experiment.variants[winner_variant]
            model_version = variant_info['model']
            
            # Promote to production
            self.model_registry.update_model_status(
                model_version.model_id, 
                model_version.version, 
                ModelStatus.PRODUCTION
            )
            
            logger.info(f"Promoted {model_version.model_id}:{model_version.version} to production")
        
        # Remove from active experiments
        del self.active_experiments[experiment_id]
        
        logger.info(f"Stopped A/B test experiment: {experiment_id}")
        return True

@dataclass 
class ABExperiment:
    """A/B test experiment configuration and tracking"""
    experiment_id: str
    control_model: 'ModelVersion'
    treatment_models: List['ModelVersion']
    traffic_split: Dict[str, float]
    success_metric: str
    duration_hours: int
    min_samples: int
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        
        # Setup variants
        self.variants = {
            'control': {
                'model': self.control_model,
                'traffic': self.traffic_split.get('control', 0.5),
                'outcomes': [],
                'user_assignments': set()
            }
        }
        
        # Add treatment variants
        for i, treatment_model in enumerate(self.treatment_models):
            variant_name = f'treatment_{i}'
            self.variants[variant_name] = {
                'model': treatment_model,
                'traffic': self.traffic_split.get(variant_name, 0.5 / len(self.treatment_models)),
                'outcomes': [],
                'user_assignments': set()
            }
        
        # Normalize traffic splits
        total_traffic = sum(v['traffic'] for v in self.variants.values())
        for variant in self.variants.values():
            variant['traffic'] = variant['traffic'] / total_traffic
    
    def assign_user(self, user_id: str) -> str:
        """Assign user to variant using consistent hashing"""
        # Use hash of user_id for consistent assignment
        user_hash = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        normalized_hash = (user_hash % 1000000) / 1000000
        
        # Assign based on traffic splits
        cumulative_traffic = 0
        for variant_name, variant_info in self.variants.items():
            cumulative_traffic += variant_info['traffic']
            if normalized_hash <= cumulative_traffic:
                variant_info['user_assignments'].add(user_id)
                return variant_name
        
        # Fallback to control
        return 'control'
    
    def record_outcome(self, user_id: str, outcome_value: float, 
                      metadata: Dict[str, Any] = None):
        """Record outcome for user"""
        # Find which variant the user is assigned to
        user_variant = None
        for variant_name, variant_info in self.variants.items():
            if user_id in variant_info['user_assignments']:
                user_variant = variant_name
                break
        
        if user_variant:
            self.variants[user_variant]['outcomes'].append({
                'user_id': user_id,
                'value': outcome_value,
                'timestamp': datetime.now(),
                'metadata': metadata or {}
            })
    
    def get_results(self) -> Dict[str, Any]:
        """Get current experiment results"""
        results = {
            'experiment_id': self.experiment_id,
            'created_at': self.created_at,
            'duration_hours': (datetime.now() - self.created_at).total_seconds() / 3600,
            'variants': {}
        }
        
        for variant_name, variant_info in self.variants.items():
            outcomes = [o['value'] for o in variant_info['outcomes']]
            
            variant_results = {
                'model_id': variant_info['model'].model_id,
                'model_version': variant_info['model'].version,
                'traffic_allocation': variant_info['traffic'],
                'sample_size': len(outcomes),
                'users_assigned': len(variant_info['user_assignments']),
                'mean_outcome': np.mean(outcomes) if outcomes else 0,
                'std_outcome': np.std(outcomes) if outcomes else 0,
                'outcomes': outcomes[-100:]  # Keep last 100 outcomes
            }
            
            results['variants'][variant_name] = variant_results
        
        # Statistical significance test (simplified)
        if len(results['variants']) >= 2:
            control_outcomes = results['variants']['control']['outcomes']
            
            for variant_name, variant_results in results['variants'].items():
                if variant_name != 'control':
                    treatment_outcomes = variant_results['outcomes']
                    
                    if len(control_outcomes) > 30 and len(treatment_outcomes) > 30:
                        # Simplified t-test
                        control_mean = np.mean(control_outcomes)
                        treatment_mean = np.mean(treatment_outcomes)
                        
                        control_std = np.std(control_outcomes)
                        treatment_std = np.std(treatment_outcomes)
                        
                        # Cohen's d (effect size)
                        pooled_std = np.sqrt((control_std**2 + treatment_std**2) / 2)
                        cohens_d = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
                        
                        variant_results['lift'] = (treatment_mean - control_mean) / control_mean if control_mean > 0 else 0
                        variant_results['effect_size'] = cohens_d
                        variant_results['significant'] = abs(cohens_d) > 0.2  # Simplified significance
        
        return results
```

## 2. Continuous Integration/Deployment for ML Models

### MLOps Pipeline for Online Learning

```python
import subprocess
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
import docker
import mlflow
import mlflow.tracking
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec
import joblib

class ModelTrainingPipeline:
    """Automated model training and validation pipeline"""
    
    def __init__(self, config_path: str = "training_config.yaml"):
        self.config = self._load_config(config_path)
        self.model_registry = ModelRegistry()
        self.mlflow_client = mlflow.tracking.MlflowClient()
        
        # Setup MLflow
        mlflow.set_tracking_uri(self.config.get('mlflow_uri', 'sqlite:///mlflow.db'))
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Default configuration
            return {
                'algorithms': ['matrix_factorization', 'factorization_machine'],
                'hyperparameters': {
                    'matrix_factorization': {
                        'n_factors': [10, 50, 100],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'regularization': [0.001, 0.01, 0.1]
                    }
                },
                'validation': {
                    'test_size': 0.2,
                    'cv_folds': 5,
                    'metrics': ['rmse', 'mae', 'precision_at_10']
                },
                'deployment': {
                    'staging_threshold': {'rmse': 1.0},
                    'production_threshold': {'rmse': 0.8}
                }
            }
    
    def run_training_pipeline(self, dataset_path: str, 
                            experiment_name: str = "recommendation_training") -> List[str]:
        """Run complete training pipeline"""
        
        # Create MLflow experiment
        experiment = mlflow.set_experiment(experiment_name)
        
        trained_models = []
        
        for algorithm in self.config['algorithms']:
            logger.info(f"Training {algorithm}...")
            
            # Get hyperparameter grid
            param_grid = self.config['hyperparameters'].get(algorithm, {})
            
            # Hyperparameter search
            best_model, best_params, metrics = self._hyperparameter_search(
                algorithm, dataset_path, param_grid
            )
            
            if best_model:
                # Create model version
                model_version = self._create_model_version(
                    algorithm, best_model, best_params, metrics
                )
                
                # Register model
                if self.model_registry.register_model(model_version):
                    trained_models.append(f"{model_version.model_id}:{model_version.version}")
                    
                    # Log to MLflow
                    self._log_to_mlflow(model_version, best_model, metrics)
        
        logger.info(f"Training pipeline completed. Trained models: {trained_models}")
        return trained_models
    
    def _hyperparameter_search(self, algorithm: str, dataset_path: str, 
                             param_grid: Dict[str, List]) -> Tuple[Any, Dict, Dict]:
        """Perform hyperparameter search"""
        
        # Load and prepare data
        data = self._load_training_data(dataset_path)
        train_data, val_data = self._split_data(data)
        
        best_model = None
        best_params = None
        best_score = float('inf')
        best_metrics = {}
        
        # Grid search
        param_combinations = self._generate_param_combinations(param_grid)
        
        for params in param_combinations:
            logger.info(f"Training {algorithm} with params: {params}")
            
            # Train model
            model = self._train_model(algorithm, train_data, params)
            
            # Validate model
            metrics = self._validate_model(model, val_data)
            
            # Check if best so far
            primary_metric = self.config['validation']['metrics'][0]
            if metrics[primary_metric] < best_score:
                best_score = metrics[primary_metric]
                best_model = model
                best_params = params
                best_metrics = metrics
        
        return best_model, best_params, best_metrics
    
    def _generate_param_combinations(self, param_grid: Dict[str, List]) -> List[Dict]:
        """Generate all parameter combinations"""
        if not param_grid:
            return [{}]
        
        import itertools
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        combinations = []
        for combination in itertools.product(*values):
            combinations.append(dict(zip(keys, combination)))
        
        return combinations
    
    def _train_model(self, algorithm: str, train_data: List, params: Dict) -> Any:
        """Train model with given parameters"""
        
        if algorithm == 'matrix_factorization':
            model = OnlineMatrixFactorization(
                n_factors=params.get('n_factors', 50),
                learning_rate=params.get('learning_rate', 0.01),
                regularization=params.get('regularization', 0.001)
            )
        elif algorithm == 'factorization_machine':
            model = OnlineFactorizationMachine(
                n_factors=params.get('n_factors', 10),
                learning_rate=params.get('learning_rate', 0.01),
                regularization=params.get('regularization', 0.001)
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Train on data
        for sample in train_data:
            model.partial_fit(sample)
        
        return model
    
    def _validate_model(self, model: Any, val_data: List) -> Dict[str, float]:
        """Validate model and compute metrics"""
        
        predictions = []
        targets = []
        
        for sample in val_data:
            pred = model.predict(sample.user_id, sample.item_id, sample.features)
            predictions.append(pred)
            targets.append(sample.target)
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Compute metrics
        metrics = {
            'rmse': np.sqrt(np.mean((predictions - targets) ** 2)),
            'mae': np.mean(np.abs(predictions - targets)),
        }
        
        return metrics
    
    def _create_model_version(self, algorithm: str, model: Any, 
                            params: Dict, metrics: Dict) -> ModelVersion:
        """Create model version object"""
        
        model_id = f"online_{algorithm}"
        version = f"v{int(time.time())}"  # Timestamp-based version
        
        # Save model
        model_dir = Path(f"models/{model_id}")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{version}.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Compute model hash
        with open(model_path, 'rb') as f:
            model_hash = hashlib.md5(f.read()).hexdigest()
        
        return ModelVersion(
            model_id=model_id,
            version=version,
            algorithm=algorithm,
            status=ModelStatus.STAGED,
            created_at=datetime.now(),
            performance_metrics=metrics,
            hyperparameters=params,
            model_path=str(model_path),
            model_hash=model_hash
        )
    
    def _log_to_mlflow(self, model_version: ModelVersion, model: Any, metrics: Dict):
        """Log model and metrics to MLflow"""
        
        with mlflow.start_run(run_name=f"{model_version.model_id}_{model_version.version}"):
            # Log parameters
            mlflow.log_params(model_version.hyperparameters)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=model_version.model_id
            )
            
            # Log metadata
            mlflow.set_tags({
                'model_version': model_version.version,
                'algorithm': model_version.algorithm,
                'status': model_version.status.value
            })
    
    def _load_training_data(self, dataset_path: str) -> List[OnlineSample]:
        """Load training data"""
        # Implementation depends on data format
        # This is a simplified example
        samples = []
        
        # Generate synthetic data for demonstration
        np.random.seed(42)
        for i in range(1000):
            sample = OnlineSample(
                user_id=f"user_{np.random.randint(0, 100)}",
                item_id=f"item_{np.random.randint(0, 200)}",
                features=np.random.randn(10),
                target=np.random.uniform(1, 5),
                timestamp=datetime.now()
            )
            samples.append(sample)
        
        return samples
    
    def _split_data(self, data: List[OnlineSample]) -> Tuple[List[OnlineSample], List[OnlineSample]]:
        """Split data into train/validation sets"""
        test_size = self.config['validation']['test_size']
        split_idx = int(len(data) * (1 - test_size))
        
        # Shuffle data
        np.random.shuffle(data)
        
        return data[:split_idx], data[split_idx:]

class ModelDeploymentManager:
    """Manages model deployment to different environments"""
    
    def __init__(self, model_registry: ModelRegistry):
        self.model_registry = model_registry
        self.docker_client = docker.from_env()
        self.deployed_services = {}
    
    def deploy_model(self, model_id: str, version: str, 
                    environment: str = "staging", port: int = 8000) -> bool:
        """Deploy model to specified environment"""
        
        model_version = self.model_registry.get_model_version(model_id, version)
        if not model_version:
            logger.error(f"Model {model_id}:{version} not found")
            return False
        
        try:
            # Create deployment configuration
            deployment_config = self._create_deployment_config(model_version, environment, port)
            
            # Build and deploy container
            if self._deploy_container(deployment_config):
                # Update model status
                if environment == "staging":
                    status = ModelStatus.STAGED
                elif environment == "production":
                    status = ModelStatus.PRODUCTION
                else:
                    status = ModelStatus.SHADOW
                
                self.model_registry.update_model_status(model_id, version, status)
                
                logger.info(f"Successfully deployed {model_id}:{version} to {environment}")
                return True
        
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            self.model_registry.update_model_status(model_id, version, ModelStatus.FAILED)
        
        return False
    
    def _create_deployment_config(self, model_version: ModelVersion, 
                                environment: str, port: int) -> Dict[str, Any]:
        """Create deployment configuration"""
        
        return {
            'model_id': model_version.model_id,
            'version': model_version.version,
            'environment': environment,
            'port': port,
            'model_path': model_version.model_path,
            'algorithm': model_version.algorithm,
            'image_name': f"{model_version.model_id}:{model_version.version}",
            'container_name': f"{model_version.model_id}_{model_version.version}_{environment}"
        }
    
    def _deploy_container(self, config: Dict[str, Any]) -> bool:
        """Deploy model as Docker container"""
        
        try:
            # Create Dockerfile
            dockerfile_content = self._generate_dockerfile(config)
            
            # Write Dockerfile
            deployment_dir = Path(f"deployments/{config['model_id']}/{config['version']}")
            deployment_dir.mkdir(parents=True, exist_ok=True)
            
            dockerfile_path = deployment_dir / "Dockerfile"
            with open(dockerfile_path, 'w') as f:
                f.write(dockerfile_content)
            
            # Copy model file
            model_src = Path(config['model_path'])
            model_dst = deployment_dir / "model.pkl"
            shutil.copy2(model_src, model_dst)
            
            # Build Docker image
            image, logs = self.docker_client.images.build(
                path=str(deployment_dir),
                tag=config['image_name'],
                rm=True
            )
            
            # Run container
            container = self.docker_client.containers.run(
                config['image_name'],
                name=config['container_name'],
                ports={f"8000/tcp": config['port']},
                environment={
                    'MODEL_ID': config['model_id'],
                    'MODEL_VERSION': config['version'],
                    'ALGORITHM': config['algorithm']
                },
                detach=True
            )
            
            self.deployed_services[config['container_name']] = {
                'container': container,
                'config': config,
                'deployed_at': datetime.now()
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Container deployment failed: {e}")
            return False
    
    def _generate_dockerfile(self, config: Dict[str, Any]) -> str:
        """Generate Dockerfile for model serving"""
        
        return f"""
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy model and serving code
COPY model.pkl .
COPY serving_app.py .

# Expose port
EXPOSE 8000

# Environment variables
ENV MODEL_ID={config['model_id']}
ENV MODEL_VERSION={config['version']}
ENV ALGORITHM={config['algorithm']}

# Run serving application
CMD ["python", "serving_app.py"]
"""
    
    def undeploy_model(self, model_id: str, version: str, environment: str) -> bool:
        """Undeploy model from environment"""
        
        container_name = f"{model_id}_{version}_{environment}"
        
        if container_name in self.deployed_services:
            try:
                container = self.deployed_services[container_name]['container']
                container.stop()
                container.remove()
                
                del self.deployed_services[container_name]
                
                # Update model status
                self.model_registry.update_model_status(model_id, version, ModelStatus.RETIRED)
                
                logger.info(f"Successfully undeployed {model_id}:{version} from {environment}")
                return True
                
            except Exception as e:
                logger.error(f"Undeployment failed: {e}")
        
        return False
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get status of all deployments"""
        
        status = {
            'active_deployments': len(self.deployed_services),
            'deployments': {}
        }
        
        for container_name, service_info in self.deployed_services.items():
            container = service_info['container']
            config = service_info['config']
            
            try:
                container.reload()
                
                status['deployments'][container_name] = {
                    'model_id': config['model_id'],
                    'version': config['version'],
                    'environment': config['environment'],
                    'status': container.status,
                    'port': config['port'],
                    'deployed_at': service_info['deployed_at'].isoformat()
                }
            except Exception as e:
                status['deployments'][container_name] = {
                    'error': str(e)
                }
        
        return status
```

## 3. Real-time Model Serving

### High-Performance Model Serving Infrastructure

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import asyncio
import aioredis
from typing import Dict, List, Optional, Any
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Prometheus metrics
REQUESTS_TOTAL = Counter('model_requests_total', 'Total model requests', ['model_id', 'version'])
LATENCY_HISTOGRAM = Histogram('model_latency_seconds', 'Model inference latency', ['model_id', 'version'])
MODEL_LOAD_GAUGE = Gauge('models_loaded', 'Number of models loaded in memory')
ERROR_COUNTER = Counter('model_errors_total', 'Total model errors', ['model_id', 'version', 'error_type'])

class PredictionRequest(BaseModel):
    user_id: str
    item_id: str
    features: Optional[List[float]] = None
    context: Optional[Dict[str, Any]] = None

class PredictionResponse(BaseModel):
    user_id: str
    item_id: str
    prediction: float
    model_id: str
    model_version: str
    latency_ms: float
    confidence: Optional[float] = None

class ModelServingEngine:
    """High-performance model serving engine"""
    
    def __init__(self, model_registry: ModelRegistry, redis_url: str = "redis://localhost:6379"):
        self.model_registry = model_registry
        self.redis_url = redis_url
        self.redis_client = None
        
        # Model cache
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        self.model_load_times: Dict[str, datetime] = {}
        self.max_models_in_memory = 10
        
        # Thread pool for CPU-intensive tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Performance tracking
        self.request_count = 0
        self.error_count = 0
        
    async def initialize(self):
        """Initialize serving engine"""
        self.redis_client = await aioredis.from_url(self.redis_url)
        logger.info("Model serving engine initialized")
    
    async def load_model(self, model_id: str, version: str) -> bool:
        """Load model into memory"""
        
        try:
            model_version = self.model_registry.get_model_version(model_id, version)
            if not model_version:
                logger.error(f"Model {model_id}:{version} not found in registry")
                return False
            
            # Check if already loaded
            model_key = f"{model_id}:{version}"
            if model_key in self.loaded_models:
                logger.info(f"Model {model_key} already loaded")
                return True
            
            # Load model from disk
            with open(model_version.model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Store in cache
            self.loaded_models[model_key] = {
                'model': model,
                'model_version': model_version,
                'load_time': datetime.now(),
                'request_count': 0
            }
            
            self.model_load_times[model_key] = datetime.now()
            
            # Update metrics
            MODEL_LOAD_GAUGE.set(len(self.loaded_models))
            
            # Evict old models if cache is full
            await self._evict_old_models()
            
            logger.info(f"Successfully loaded model {model_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}:{version}: {e}")
            ERROR_COUNTER.labels(model_id=model_id, version=version, error_type='load_error').inc()
            return False
    
    async def _evict_old_models(self):
        """Evict old models from memory if cache is full"""
        
        if len(self.loaded_models) <= self.max_models_in_memory:
            return
        
        # Sort by last request time (LRU eviction)
        models_by_usage = sorted(
            self.loaded_models.items(),
            key=lambda x: x[1]['load_time']
        )
        
        # Evict oldest models
        models_to_evict = len(self.loaded_models) - self.max_models_in_memory + 1
        
        for i in range(models_to_evict):
            model_key = models_by_usage[i][0]
            del self.loaded_models[model_key]
            del self.model_load_times[model_key]
            logger.info(f"Evicted model {model_key} from cache")
        
        MODEL_LOAD_GAUGE.set(len(self.loaded_models))
    
    async def predict(self, model_id: str, version: str, 
                     request: PredictionRequest) -> PredictionResponse:
        """Make prediction using specified model"""
        
        start_time = time.time()
        model_key = f"{model_id}:{version}"
        
        try:
            # Load model if not in cache
            if model_key not in self.loaded_models:
                await self.load_model(model_id, version)
            
            if model_key not in self.loaded_models:
                raise HTTPException(status_code=404, detail=f"Model {model_key} not available")
            
            # Get model
            model_info = self.loaded_models[model_key]
            model = model_info['model']
            
            # Prepare features
            features = np.array(request.features) if request.features else np.array([])
            
            # Make prediction (run in thread pool for CPU-intensive models)
            prediction = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                model.predict,
                request.user_id,
                request.item_id,
                features
            )
            
            # Update usage statistics
            model_info['request_count'] += 1
            self.request_count += 1
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Update metrics
            REQUESTS_TOTAL.labels(model_id=model_id, version=version).inc()
            LATENCY_HISTOGRAM.labels(model_id=model_id, version=version).observe(latency_ms / 1000)
            
            # Cache prediction if configured
            await self._cache_prediction(request, prediction, model_id, version)
            
            return PredictionResponse(
                user_id=request.user_id,
                item_id=request.item_id,
                prediction=float(prediction),
                model_id=model_id,
                model_version=version,
                latency_ms=latency_ms
            )
            
        except Exception as e:
            ERROR_COUNTER.labels(model_id=model_id, version=version, error_type='prediction_error').inc()
            self.error_count += 1
            logger.error(f"Prediction error for {model_key}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _cache_prediction(self, request: PredictionRequest, prediction: float,
                              model_id: str, version: str, ttl: int = 300):
        """Cache prediction result in Redis"""
        
        if not self.redis_client:
            return
        
        try:
            cache_key = f"pred:{model_id}:{version}:{request.user_id}:{request.item_id}"
            cache_value = {
                'prediction': prediction,
                'timestamp': datetime.now().isoformat(),
                'features_hash': hash(str(request.features)) if request.features else None
            }
            
            await self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(cache_value)
            )
            
        except Exception as e:
            logger.warning(f"Failed to cache prediction: {e}")
    
    async def get_cached_prediction(self, model_id: str, version: str,
                                  request: PredictionRequest) -> Optional[float]:
        """Get cached prediction if available"""
        
        if not self.redis_client:
            return None
        
        try:
            cache_key = f"pred:{model_id}:{version}:{request.user_id}:{request.item_id}"
            cached_result = await self.redis_client.get(cache_key)
            
            if cached_result:
                cache_data = json.loads(cached_result)
                
                # Check if features match (simple hash comparison)
                features_hash = hash(str(request.features)) if request.features else None
                if cache_data['features_hash'] == features_hash:
                    return cache_data['prediction']
            
        except Exception as e:
            logger.warning(f"Failed to get cached prediction: {e}")
        
        return None
    
    def get_serving_stats(self) -> Dict[str, Any]:
        """Get serving statistics"""
        
        return {
            'total_requests': self.request_count,
            'total_errors': self.error_count,
            'loaded_models': len(self.loaded_models),
            'models_in_cache': list(self.loaded_models.keys()),
            'cache_hit_rate': 0.0,  # Would be calculated from cache stats
            'avg_latency_ms': 0.0   # Would be calculated from latency histogram
        }

class ModelServingAPI:
    """FastAPI application for model serving"""
    
    def __init__(self, model_registry: ModelRegistry):
        self.app = FastAPI(title="Model Serving API", version="1.0.0")
        self.serving_engine = ModelServingEngine(model_registry)
        
        self.setup_routes()
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.on_event("startup")
        async def startup_event():
            await self.serving_engine.initialize()
        
        @self.app.post("/predict/{model_id}/{version}", response_model=PredictionResponse)
        async def predict(model_id: str, version: str, request: PredictionRequest):
            """Make prediction using specified model version"""
            
            # Check cache first
            cached_prediction = await self.serving_engine.get_cached_prediction(
                model_id, version, request
            )
            
            if cached_prediction is not None:
                return PredictionResponse(
                    user_id=request.user_id,
                    item_id=request.item_id,
                    prediction=cached_prediction,
                    model_id=model_id,
                    model_version=version,
                    latency_ms=0.0  # Cache hit
                )
            
            # Make prediction
            return await self.serving_engine.predict(model_id, version, request)
        
        @self.app.post("/predict/{model_id}", response_model=PredictionResponse)
        async def predict_latest(model_id: str, request: PredictionRequest):
            """Make prediction using latest production model version"""
            
            # Get latest production model
            latest_model = self.serving_engine.model_registry.get_latest_version(
                model_id, ModelStatus.PRODUCTION
            )
            
            if not latest_model:
                raise HTTPException(status_code=404, detail=f"No production model found for {model_id}")
            
            return await self.serving_engine.predict(model_id, latest_model.version, request)
        
        @self.app.post("/load_model/{model_id}/{version}")
        async def load_model(model_id: str, version: str):
            """Load model into serving cache"""
            
            success = await self.serving_engine.load_model(model_id, version)
            
            if success:
                return {"status": "success", "message": f"Model {model_id}:{version} loaded"}
            else:
                raise HTTPException(status_code=500, detail="Failed to load model")
        
        @self.app.get("/models")
        async def list_models():
            """List all available models"""
            
            models = self.serving_engine.model_registry.list_models()
            
            return {
                'models': [
                    {
                        'model_id': m.model_id,
                        'version': m.version,
                        'algorithm': m.algorithm,
                        'status': m.status.value,
                        'created_at': m.created_at.isoformat(),
                        'metrics': m.performance_metrics
                    }
                    for m in models
                ]
            }
        
        @self.app.get("/stats")
        async def get_stats():
            """Get serving statistics"""
            return self.serving_engine.get_serving_stats()
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get Prometheus metrics"""
            return generate_latest()
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'loaded_models': len(self.serving_engine.loaded_models)
            }
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the serving API"""
        uvicorn.run(self.app, host=host, port=port)

# Example usage
async def demonstrate_model_serving():
    """Demonstrate the model serving system"""
    
    print("🚀 Model Serving System Demo")
    print("=" * 40)
    
    # Setup model registry
    model_registry = ModelRegistry()
    
    # Create serving API
    serving_api = ModelServingAPI(model_registry)
    
    # Initialize serving engine
    await serving_api.serving_engine.initialize()
    
    # Create and register a sample model
    sample_model = OnlineMatrixFactorization(n_factors=10)
    
    model_version = ModelVersion(
        model_id="demo_model",
        version="v1",
        algorithm="matrix_factorization",
        status=ModelStatus.PRODUCTION,
        created_at=datetime.now(),
        performance_metrics={'rmse': 0.85, 'mae': 0.65},
        model_path="demo_model_v1.pkl"
    )
    
    # Save model to disk (for demo)
    with open("demo_model_v1.pkl", 'wb') as f:
        pickle.dump(sample_model, f)
    
    # Register model
    model_registry.register_model(model_version)
    
    # Load model into serving cache
    await serving_api.serving_engine.load_model("demo_model", "v1")
    
    # Make some test predictions
    test_requests = [
        PredictionRequest(user_id="user_1", item_id="item_1", features=[1.0, 2.0, 3.0]),
        PredictionRequest(user_id="user_2", item_id="item_2", features=[2.0, 3.0, 4.0]),
        PredictionRequest(user_id="user_3", item_id="item_3", features=[3.0, 4.0, 5.0])
    ]
    
    print("\n📊 Making test predictions...")
    
    for i, request in enumerate(test_requests):
        try:
            response = await serving_api.serving_engine.predict("demo_model", "v1", request)
            print(f"Prediction {i+1}: {response.prediction:.4f} (latency: {response.latency_ms:.2f}ms)")
        except Exception as e:
            print(f"Prediction {i+1} failed: {e}")
    
    # Get serving stats
    stats = serving_api.serving_engine.get_serving_stats()
    print(f"\n📈 Serving Stats:")
    print(f"Total requests: {stats['total_requests']}")
    print(f"Loaded models: {stats['loaded_models']}")
    
    print("\n✅ Model serving demo completed!")

if __name__ == "__main__":
    asyncio.run(demonstrate_model_serving())
```

## Key Takeaways (Part 2)

1. **Model Versioning**: Systematic tracking of model versions enables reliable deployments and rollbacks

2. **A/B Testing**: Automated experimentation frameworks enable data-driven model selection

3. **CI/CD Pipelines**: Automated training and deployment pipelines ensure consistent model quality

4. **High-Performance Serving**: Optimized serving infrastructure handles real-time prediction requests efficiently

5. **Monitoring & Metrics**: Comprehensive monitoring ensures production model performance and reliability

6. **Caching Strategies**: Intelligent caching reduces latency and computational costs

## Study Questions (Part 2)

### Beginner Level
1. What are the benefits of model versioning in production systems?
2. How does A/B testing help in model selection?
3. What is the role of caching in model serving?
4. How do you ensure model serving reliability?

### Intermediate Level
1. Design a CI/CD pipeline for online learning models
2. How would you implement canary deployments for ML models?
3. What are the trade-offs between model accuracy and serving latency?
4. How can you handle model serving at scale?

### Advanced Level
1. Implement a multi-tenant model serving system with resource isolation
2. Design a system for automatic model retraining and deployment
3. How would you implement distributed model serving across multiple regions?
4. Design a system that handles both batch and real-time model serving

## Course Summary (Day 7.2)

We've covered comprehensive online learning and model serving systems including:

- **Online Learning Foundations**: Incremental algorithms, concept drift detection, adaptive learning
- **Production Systems**: Model versioning, A/B testing, CI/CD pipelines
- **Model Serving**: High-performance serving, caching, monitoring, and metrics

These systems enable continuous learning and adaptation in production recommendation systems, ensuring they stay relevant and performant as user behavior evolves.

## Next Session Preview

Tomorrow we'll explore **Caching Strategies and Model Serving Optimization**, covering:
- Multi-level caching architectures
- Cache invalidation strategies
- Content delivery networks for recommendations
- Edge computing for low-latency serving
- Load balancing and auto-scaling
- Performance optimization techniques

We'll implement sophisticated caching and serving systems for maximum performance and scalability!