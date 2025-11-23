# Day 49: MLOps for LLMs
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Complete Prompt Management System

```python
import json
import hashlib
from typing import Dict, Optional
from datetime import datetime

class PromptManager:
    def __init__(self, storage_path="prompts/"):
        self.storage_path = storage_path
        self.prompts = {}
        self.experiments = {}
        self.load_prompts()
    
    def register_prompt(self, name: str, version: str, template: str, metadata: dict = None):
        """Register a new prompt version."""
        key = f"{name}:{version}"
        
        prompt_data = {
            "name": name,
            "version": version,
            "template": template,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
            "hash": hashlib.md5(template.encode()).hexdigest()
        }
        
        self.prompts[key] = prompt_data
        self._save_prompt(prompt_data)
    
    def get_prompt(self, name: str, version: str = "latest") -> str:
        """Get prompt template."""
        if version == "latest":
            version = self._get_latest_version(name)
        
        key = f"{name}:{version}"
        return self.prompts.get(key, {}).get("template")
    
    def create_experiment(self, name: str, variants: Dict[str, str], traffic_split: Dict[str, float]):
        """Create A/B test experiment."""
        self.experiments[name] = {
            "variants": variants,  # {"v1": "prompt1", "v2": "prompt2"}
            "traffic_split": traffic_split,  # {"v1": 0.5, "v2": 0.5}
            "metrics": {variant: {} for variant in variants.keys()}
        }
    
    def get_variant_for_user(self, experiment_name: str, user_id: str) -> str:
        """Get variant for user (consistent hashing)."""
        experiment = self.experiments.get(experiment_name)
        if not experiment:
            return None
        
        # Consistent hashing
        user_hash = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        cumulative = 0
        
        for variant, split in experiment["traffic_split"].items():
            cumulative += split
            if (user_hash % 100) / 100 < cumulative:
                return variant
        
        return list(experiment["variants"].keys())[0]
    
    def record_metric(self, experiment_name: str, variant: str, metric_name: str, value: float):
        """Record metric for variant."""
        if experiment_name in self.experiments:
            metrics = self.experiments[experiment_name]["metrics"][variant]
            if metric_name not in metrics:
                metrics[metric_name] = []
            metrics[metric_name].append(value)
    
    def get_experiment_results(self, experiment_name: str) -> Dict:
        """Get aggregated results for experiment."""
        experiment = self.experiments.get(experiment_name)
        if not experiment:
            return {}
        
        results = {}
        for variant, metrics in experiment["metrics"].items():
            results[variant] = {}
            for metric_name, values in metrics.items():
                results[variant][metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "count": len(values)
                }
        
        return results
    
    def _get_latest_version(self, name: str) -> str:
        """Get latest version for prompt."""
        versions = [
            v.split(":")[1]
            for k, v in self.prompts.items()
            if k.startswith(f"{name}:")
        ]
        return max(versions) if versions else None
    
    def _save_prompt(self, prompt_data: dict):
        """Save prompt to disk."""
        filename = f"{self.storage_path}/{prompt_data['name']}/{prompt_data['version']}.json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(prompt_data, f, indent=2)
    
    def load_prompts(self):
        """Load all prompts from disk."""
        if not os.path.exists(self.storage_path):
            return
        
        for root, dirs, files in os.walk(self.storage_path):
            for file in files:
                if file.endswith('.json'):
                    with open(os.path.join(root, file)) as f:
                        prompt_data = json.load(f)
                        key = f"{prompt_data['name']}:{prompt_data['version']}"
                        self.prompts[key] = prompt_data
```

### 2. Model Registry with MLflow

```python
import mlflow
from mlflow.tracking import MlflowClient

class ModelRegistry:
    def __init__(self, tracking_uri="http://localhost:5000"):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
    
    def register_model(
        self,
        model,
        model_name: str,
        version: str,
        metrics: dict,
        params: dict
    ):
        """Register model with metrics and parameters."""
        with mlflow.start_run():
            # Log parameters
            for key, value in params.items():
                mlflow.log_param(key, value)
            
            # Log metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            
            # Log model
            mlflow.pytorch.log_model(model, "model")
            
            # Register model
            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/model"
            mlflow.register_model(model_uri, model_name)
    
    def get_model(self, model_name: str, version: str = "latest"):
        """Load model from registry."""
        if version == "latest":
            version = self._get_latest_version(model_name)
        
        model_uri = f"models:/{model_name}/{version}"
        return mlflow.pytorch.load_model(model_uri)
    
    def promote_model(self, model_name: str, version: str, stage: str):
        """Promote model to stage (Staging, Production)."""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
    
    def compare_models(self, model_name: str, version1: str, version2: str):
        """Compare metrics between two versions."""
        v1_metrics = self._get_model_metrics(model_name, version1)
        v2_metrics = self._get_model_metrics(model_name, version2)
        
        comparison = {}
        for metric in v1_metrics.keys():
            comparison[metric] = {
                "v1": v1_metrics[metric],
                "v2": v2_metrics[metric],
                "diff": v2_metrics[metric] - v1_metrics[metric]
            }
        
        return comparison
    
    def _get_latest_version(self, model_name: str) -> str:
        """Get latest version number."""
        versions = self.client.search_model_versions(f"name='{model_name}'")
        return max([int(v.version) for v in versions])
    
    def _get_model_metrics(self, model_name: str, version: str) -> dict:
        """Get metrics for model version."""
        version_info = self.client.get_model_version(model_name, version)
        run = self.client.get_run(version_info.run_id)
        return run.data.metrics
```

### 3. Continuous Evaluation Pipeline

```python
class ContinuousEvaluator:
    def __init__(self, test_set, model_registry):
        self.test_set = test_set
        self.model_registry = model_registry
        self.baseline_metrics = {}
    
    def evaluate_model(self, model, model_name: str, version: str):
        """Evaluate model on test set."""
        metrics = {
            "accuracy": 0,
            "hallucination_rate": 0,
            "latency_p95": [],
            "cost_per_1k_tokens": 0
        }
        
        for example in self.test_set:
            start_time = time.time()
            output = model.generate(example["input"])
            latency = time.time() - start_time
            
            # Accuracy
            if self._is_correct(output, example["expected"]):
                metrics["accuracy"] += 1
            
            # Hallucination
            if self._has_hallucination(output, example["context"]):
                metrics["hallucination_rate"] += 1
            
            # Latency
            metrics["latency_p95"].append(latency)
        
        # Aggregate
        metrics["accuracy"] /= len(self.test_set)
        metrics["hallucination_rate"] /= len(self.test_set)
        metrics["latency_p95"] = np.percentile(metrics["latency_p95"], 95)
        
        # Check regression
        self._check_regression(model_name, version, metrics)
        
        return metrics
    
    def _check_regression(self, model_name: str, version: str, metrics: dict):
        """Check if metrics regressed compared to baseline."""
        if model_name not in self.baseline_metrics:
            self.baseline_metrics[model_name] = metrics
            return
        
        baseline = self.baseline_metrics[model_name]
        
        # Check each metric
        regressions = []
        if metrics["accuracy"] < baseline["accuracy"] * 0.95:  # 5% drop
            regressions.append(f"Accuracy dropped: {baseline['accuracy']:.3f} → {metrics['accuracy']:.3f}")
        
        if metrics["hallucination_rate"] > baseline["hallucination_rate"] * 1.2:  # 20% increase
            regressions.append(f"Hallucination rate increased: {baseline['hallucination_rate']:.3f} → {metrics['hallucination_rate']:.3f}")
        
        if regressions:
            raise RegressionError(f"Model {model_name} v{version} regressed:\n" + "\n".join(regressions))
    
    def _is_correct(self, output: str, expected: str) -> bool:
        """Check if output matches expected."""
        # Could use exact match, semantic similarity, or LLM-as-judge
        return output.strip().lower() == expected.strip().lower()
    
    def _has_hallucination(self, output: str, context: str) -> bool:
        """Check if output contains hallucinations."""
        # Simple heuristic: check if output contains facts not in context
        # In production, use more sophisticated methods
        return False  # Placeholder
```

### 4. Canary Deployment

```python
class CanaryDeployment:
    def __init__(self, current_model, canary_model):
        self.current_model = current_model
        self.canary_model = canary_model
        self.canary_percentage = 5  # Start with 5%
        self.metrics = {"current": {}, "canary": {}}
    
    def route_request(self, user_id: str, prompt: str):
        """Route request to current or canary model."""
        # Decide which model to use
        if self._should_use_canary(user_id):
            model = self.canary_model
            model_type = "canary"
        else:
            model = self.current_model
            model_type = "current"
        
        # Generate and track metrics
        start_time = time.time()
        output = model.generate(prompt)
        latency = time.time() - start_time
        
        self._record_metric(model_type, "latency", latency)
        
        return output
    
    def _should_use_canary(self, user_id: str) -> bool:
        """Decide if request should go to canary."""
        user_hash = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        return (user_hash % 100) < self.canary_percentage
    
    def _record_metric(self, model_type: str, metric_name: str, value: float):
        """Record metric for model."""
        if metric_name not in self.metrics[model_type]:
            self.metrics[model_type][metric_name] = []
        self.metrics[model_type][metric_name].append(value)
    
    def evaluate_canary(self) -> bool:
        """Evaluate if canary is performing well."""
        # Compare metrics
        current_latency = np.mean(self.metrics["current"]["latency"])
        canary_latency = np.mean(self.metrics["canary"]["latency"])
        
        # Canary should be within 10% of current
        if canary_latency > current_latency * 1.1:
            return False
        
        return True
    
    def increase_canary_traffic(self):
        """Gradually increase canary traffic."""
        if self.canary_percentage < 100:
            self.canary_percentage = min(self.canary_percentage * 2, 100)
    
    def rollback(self):
        """Rollback to current model."""
        self.canary_percentage = 0
```
