# Day 5.6: Evaluation of Hybrid Recommendation Systems

## Learning Objectives
- Master comprehensive evaluation frameworks for hybrid systems
- Implement multi-dimensional evaluation metrics and methodologies
- Design A/B testing frameworks for hybrid recommendation validation
- Develop fairness and bias assessment tools for hybrid systems
- Create long-term impact evaluation and monitoring systems

## 1. Comprehensive Evaluation Framework

### Multi-Dimensional Evaluation System

```python
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class EvaluationDimension(Enum):
    ACCURACY = "accuracy"
    DIVERSITY = "diversity"
    NOVELTY = "novelty"
    COVERAGE = "coverage"
    FAIRNESS = "fairness"
    EFFICIENCY = "efficiency"
    ROBUSTNESS = "robustness"
    EXPLAINABILITY = "explainability"

@dataclass
class EvaluationResult:
    dimension: EvaluationDimension
    metric_name: str
    value: float
    confidence_interval: Tuple[float, float]
    statistical_significance: bool
    metadata: Dict[str, Any]

class HybridEvaluationFramework:
    def __init__(self):
        self.evaluators = {}
        self.results_history = []
        self.baseline_results = {}
        
    def register_evaluator(self, dimension: EvaluationDimension, evaluator: 'BaseEvaluator'):
        """Register an evaluator for a specific dimension"""
        self.evaluators[dimension] = evaluator
        
    def evaluate_hybrid_system(self, 
                              hybrid_system,
                              test_data: pd.DataFrame,
                              ground_truth: pd.DataFrame,
                              user_interactions: pd.DataFrame,
                              baseline_systems: Dict[str, Any] = None) -> Dict[EvaluationDimension, List[EvaluationResult]]:
        """Comprehensive evaluation of hybrid recommendation system"""
        
        results = {}
        
        # Generate recommendations from hybrid system
        recommendations = self._generate_recommendations(hybrid_system, test_data)
        
        # Evaluate across all registered dimensions
        for dimension, evaluator in self.evaluators.items():
            print(f"Evaluating {dimension.value}...")
            
            dimension_results = evaluator.evaluate(
                recommendations=recommendations,
                ground_truth=ground_truth,
                user_interactions=user_interactions,
                test_data=test_data,
                hybrid_system=hybrid_system
            )
            
            results[dimension] = dimension_results
            
            # Compare with baselines if provided
            if baseline_systems:
                for baseline_name, baseline_system in baseline_systems.items():
                    baseline_recs = self._generate_recommendations(baseline_system, test_data)
                    baseline_results = evaluator.evaluate(
                        recommendations=baseline_recs,
                        ground_truth=ground_truth,
                        user_interactions=user_interactions,
                        test_data=test_data,
                        hybrid_system=baseline_system
                    )
                    
                    # Store baseline results for comparison
                    if baseline_name not in self.baseline_results:
                        self.baseline_results[baseline_name] = {}
                    self.baseline_results[baseline_name][dimension] = baseline_results
        
        self.results_history.append(results)
        return results
    
    def _generate_recommendations(self, system, test_data: pd.DataFrame) -> pd.DataFrame:
        """Generate recommendations from a system"""
        recommendations = []
        
        for _, user_data in test_data.iterrows():
            user_id = user_data['user_id']
            recs = system.recommend(user_id, k=10)
            
            for i, (item_id, score) in enumerate(recs):
                recommendations.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'rank': i + 1,
                    'score': score,
                    'timestamp': pd.Timestamp.now()
                })
        
        return pd.DataFrame(recommendations)
    
    def generate_evaluation_report(self) -> str:
        """Generate comprehensive evaluation report"""
        if not self.results_history:
            return "No evaluation results available"
        
        latest_results = self.results_history[-1]
        report = "# Hybrid Recommendation System Evaluation Report\n\n"
        
        for dimension, results in latest_results.items():
            report += f"## {dimension.value.title()} Evaluation\n\n"
            
            for result in results:
                report += f"- **{result.metric_name}**: {result.value:.4f} "
                report += f"(CI: {result.confidence_interval[0]:.4f}-{result.confidence_interval[1]:.4f})\n"
                
                if result.statistical_significance:
                    report += "  - ✅ Statistically significant\n"
                else:
                    report += "  - ❌ Not statistically significant\n"
            
            report += "\n"
        
        return report

class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, **kwargs) -> List[EvaluationResult]:
        pass

class AccuracyEvaluator(BaseEvaluator):
    def evaluate(self, recommendations: pd.DataFrame, ground_truth: pd.DataFrame, **kwargs) -> List[EvaluationResult]:
        """Evaluate accuracy metrics"""
        results = []
        
        # Merge recommendations with ground truth
        merged = recommendations.merge(
            ground_truth, 
            on=['user_id', 'item_id'], 
            how='left',
            indicator=True
        )
        
        # Calculate precision@k for different k values
        for k in [1, 5, 10]:
            top_k_recs = recommendations[recommendations['rank'] <= k]
            top_k_merged = top_k_recs.merge(
                ground_truth, 
                on=['user_id', 'item_id'], 
                how='left',
                indicator=True
            )
            
            precision_k = (top_k_merged['_merge'] == 'both').sum() / len(top_k_merged)
            
            # Bootstrap confidence interval
            bootstrap_precisions = []
            for _ in range(1000):
                sample = top_k_merged.sample(n=len(top_k_merged), replace=True)
                bootstrap_precision = (sample['_merge'] == 'both').sum() / len(sample)
                bootstrap_precisions.append(bootstrap_precision)
            
            ci_lower = np.percentile(bootstrap_precisions, 2.5)
            ci_upper = np.percentile(bootstrap_precisions, 97.5)
            
            results.append(EvaluationResult(
                dimension=EvaluationDimension.ACCURACY,
                metric_name=f"Precision@{k}",
                value=precision_k,
                confidence_interval=(ci_lower, ci_upper),
                statistical_significance=ci_lower > 0,
                metadata={'k': k, 'total_recommendations': len(top_k_recs)}
            ))
        
        # Calculate NDCG
        ndcg_scores = self._calculate_ndcg(recommendations, ground_truth)
        
        results.append(EvaluationResult(
            dimension=EvaluationDimension.ACCURACY,
            metric_name="NDCG@10",
            value=np.mean(ndcg_scores),
            confidence_interval=(
                np.percentile(ndcg_scores, 2.5),
                np.percentile(ndcg_scores, 97.5)
            ),
            statistical_significance=True,
            metadata={'individual_scores': ndcg_scores}
        ))
        
        return results
    
    def _calculate_ndcg(self, recommendations: pd.DataFrame, ground_truth: pd.DataFrame) -> List[float]:
        """Calculate NDCG for each user"""
        ndcg_scores = []
        
        for user_id in recommendations['user_id'].unique():
            user_recs = recommendations[recommendations['user_id'] == user_id].head(10)
            user_truth = ground_truth[ground_truth['user_id'] == user_id]
            
            # Create relevance scores (1 if item is in ground truth, 0 otherwise)
            relevances = []
            for _, rec in user_recs.iterrows():
                if rec['item_id'] in user_truth['item_id'].values:
                    relevances.append(1.0)
                else:
                    relevances.append(0.0)
            
            # Calculate DCG
            dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))
            
            # Calculate IDCG (ideal DCG)
            ideal_relevances = sorted(relevances, reverse=True)
            idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))
            
            # Calculate NDCG
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_scores.append(ndcg)
        
        return ndcg_scores

class DiversityEvaluator(BaseEvaluator):
    def evaluate(self, recommendations: pd.DataFrame, **kwargs) -> List[EvaluationResult]:
        """Evaluate diversity metrics"""
        results = []
        
        # Intra-list diversity (average pairwise distance within recommendation lists)
        intra_list_diversities = []
        
        for user_id in recommendations['user_id'].unique():
            user_recs = recommendations[recommendations['user_id'] == user_id].head(10)
            
            if len(user_recs) < 2:
                continue
            
            # Calculate pairwise similarities (using item IDs as proxy)
            items = user_recs['item_id'].values
            total_pairs = 0
            total_diversity = 0
            
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    # Simple diversity measure (can be replaced with content-based similarity)
                    diversity = 1.0 if items[i] != items[j] else 0.0
                    total_diversity += diversity
                    total_pairs += 1
            
            if total_pairs > 0:
                intra_list_diversities.append(total_diversity / total_pairs)
        
        mean_diversity = np.mean(intra_list_diversities)
        
        results.append(EvaluationResult(
            dimension=EvaluationDimension.DIVERSITY,
            metric_name="Intra-List Diversity",
            value=mean_diversity,
            confidence_interval=(
                np.percentile(intra_list_diversities, 2.5),
                np.percentile(intra_list_diversities, 97.5)
            ),
            statistical_significance=True,
            metadata={'per_user_diversities': intra_list_diversities}
        ))
        
        # Inter-list diversity (diversity across different users' recommendation lists)
        all_recommended_items = set(recommendations['item_id'].unique())
        total_possible_items = len(all_recommended_items)  # In practice, this would be catalog size
        
        coverage = len(all_recommended_items) / total_possible_items
        
        results.append(EvaluationResult(
            dimension=EvaluationDimension.DIVERSITY,
            metric_name="Item Coverage",
            value=coverage,
            confidence_interval=(coverage * 0.95, coverage * 1.05),  # Approximate CI
            statistical_significance=True,
            metadata={'unique_items_recommended': len(all_recommended_items)}
        ))
        
        return results

class FairnessEvaluator(BaseEvaluator):
    def evaluate(self, recommendations: pd.DataFrame, user_interactions: pd.DataFrame, **kwargs) -> List[EvaluationResult]:
        """Evaluate fairness metrics"""
        results = []
        
        # Assume user demographics are available in user_interactions
        if 'gender' not in user_interactions.columns:
            # Create synthetic demographics for demonstration
            user_interactions['gender'] = np.random.choice(['M', 'F', 'Other'], size=len(user_interactions))
        
        # Group-level fairness: equal recommendation quality across groups
        group_precisions = {}
        
        for group in user_interactions['gender'].unique():
            group_users = user_interactions[user_interactions['gender'] == group]['user_id'].unique()
            group_recs = recommendations[recommendations['user_id'].isin(group_users)]
            
            # Calculate precision for this group (simplified)
            group_precision = group_recs['score'].mean()  # Using score as proxy for precision
            group_precisions[group] = group_precision
        
        # Calculate demographic parity (difference in recommendation rates)
        precision_values = list(group_precisions.values())
        demographic_parity = max(precision_values) - min(precision_values)
        
        results.append(EvaluationResult(
            dimension=EvaluationDimension.FAIRNESS,
            metric_name="Demographic Parity Gap",
            value=demographic_parity,
            confidence_interval=(demographic_parity * 0.9, demographic_parity * 1.1),
            statistical_significance=demographic_parity > 0.05,
            metadata={'group_precisions': group_precisions}
        ))
        
        # Equal opportunity: equal true positive rates across groups
        equal_opportunity_gap = np.std(precision_values)
        
        results.append(EvaluationResult(
            dimension=EvaluationDimension.FAIRNESS,
            metric_name="Equal Opportunity Gap",
            value=equal_opportunity_gap,
            confidence_interval=(equal_opportunity_gap * 0.9, equal_opportunity_gap * 1.1),
            statistical_significance=equal_opportunity_gap > 0.1,
            metadata={'group_std': equal_opportunity_gap}
        ))
        
        return results
```

## 2. A/B Testing Framework for Hybrid Systems

### Statistical Testing and Experiment Design

```python
class ABTestFramework:
    def __init__(self, significance_level: float = 0.05, power: float = 0.8):
        self.significance_level = significance_level
        self.power = power
        self.experiments = {}
        
    def design_experiment(self, 
                         experiment_name: str,
                         variants: Dict[str, Any],
                         primary_metric: str,
                         secondary_metrics: List[str],
                         minimum_detectable_effect: float,
                         expected_baseline_rate: float) -> Dict[str, Any]:
        """Design A/B test for hybrid recommendation systems"""
        
        # Calculate required sample size
        sample_size = self._calculate_sample_size(
            expected_baseline_rate,
            minimum_detectable_effect,
            self.significance_level,
            self.power
        )
        
        experiment_config = {
            'name': experiment_name,
            'variants': variants,
            'primary_metric': primary_metric,
            'secondary_metrics': secondary_metrics,
            'sample_size_per_variant': sample_size,
            'significance_level': self.significance_level,
            'power': self.power,
            'status': 'designed',
            'start_date': None,
            'end_date': None,
            'results': None
        }
        
        self.experiments[experiment_name] = experiment_config
        return experiment_config
    
    def _calculate_sample_size(self, baseline_rate: float, effect_size: float, 
                              alpha: float, power: float) -> int:
        """Calculate required sample size for A/B test"""
        from scipy.stats import norm
        
        # Two-sided test
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(power)
        
        p1 = baseline_rate
        p2 = baseline_rate + effect_size
        
        pooled_p = (p1 + p2) / 2
        
        sample_size = (2 * pooled_p * (1 - pooled_p) * (z_alpha + z_beta)**2) / (effect_size**2)
        
        return int(np.ceil(sample_size))
    
    def run_experiment(self, 
                      experiment_name: str,
                      data: pd.DataFrame,
                      variant_assignment: Dict[int, str]) -> Dict[str, Any]:
        """Run A/B test experiment"""
        
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment {experiment_name} not found")
        
        experiment = self.experiments[experiment_name]
        experiment['status'] = 'running'
        experiment['start_date'] = pd.Timestamp.now()
        
        # Assign users to variants
        data['variant'] = data['user_id'].map(variant_assignment)
        
        # Collect metrics for each variant
        variant_metrics = {}
        
        for variant_name in experiment['variants'].keys():
            variant_data = data[data['variant'] == variant_name]
            
            metrics = self._collect_metrics(
                variant_data,
                experiment['primary_metric'],
                experiment['secondary_metrics']
            )
            
            variant_metrics[variant_name] = metrics
        
        # Perform statistical tests
        test_results = self._perform_statistical_tests(
            variant_metrics,
            experiment['primary_metric'],
            experiment['secondary_metrics']
        )
        
        experiment['results'] = {
            'variant_metrics': variant_metrics,
            'statistical_tests': test_results,
            'sample_sizes': {k: len(data[data['variant'] == k]) for k in experiment['variants'].keys()}
        }
        
        experiment['status'] = 'completed'
        experiment['end_date'] = pd.Timestamp.now()
        
        return experiment['results']
    
    def _collect_metrics(self, data: pd.DataFrame, primary_metric: str, secondary_metrics: List[str]) -> Dict[str, float]:
        """Collect metrics from experiment data"""
        metrics = {}
        
        # Primary metric
        if primary_metric == 'ctr':
            metrics['ctr'] = data['clicked'].mean() if 'clicked' in data.columns else np.random.beta(2, 8)
        elif primary_metric == 'conversion_rate':
            metrics['conversion_rate'] = data['converted'].mean() if 'converted' in data.columns else np.random.beta(1, 9)
        elif primary_metric == 'engagement_time':
            metrics['engagement_time'] = data['time_spent'].mean() if 'time_spent' in data.columns else np.random.exponential(5)
        
        # Secondary metrics
        for metric in secondary_metrics:
            if metric == 'diversity_score':
                metrics['diversity_score'] = np.random.uniform(0.6, 0.9)
            elif metric == 'user_satisfaction':
                metrics['user_satisfaction'] = np.random.uniform(3.5, 4.5)
            elif metric == 'revenue_per_user':
                metrics['revenue_per_user'] = np.random.exponential(20)
        
        return metrics
    
    def _perform_statistical_tests(self, variant_metrics: Dict[str, Dict[str, float]], 
                                  primary_metric: str, secondary_metrics: List[str]) -> Dict[str, Any]:
        """Perform statistical significance tests"""
        test_results = {}
        
        variants = list(variant_metrics.keys())
        if len(variants) != 2:
            raise ValueError("Currently only supports two-variant tests")
        
        control_variant = variants[0]
        treatment_variant = variants[1]
        
        # Test primary metric
        control_value = variant_metrics[control_variant][primary_metric]
        treatment_value = variant_metrics[treatment_variant][primary_metric]
        
        # Simple t-test simulation (in practice, use proper statistical tests)
        effect_size = (treatment_value - control_value) / control_value
        
        # Simulate p-value based on effect size
        if abs(effect_size) > 0.05:
            p_value = np.random.uniform(0.001, 0.03)
        else:
            p_value = np.random.uniform(0.1, 0.5)
        
        test_results[primary_metric] = {
            'control_value': control_value,
            'treatment_value': treatment_value,
            'lift': effect_size,
            'p_value': p_value,
            'significant': p_value < self.significance_level,
            'confidence_interval': (
                effect_size - 1.96 * abs(effect_size) * 0.1,
                effect_size + 1.96 * abs(effect_size) * 0.1
            )
        }
        
        # Test secondary metrics
        for metric in secondary_metrics:
            control_value = variant_metrics[control_variant][metric]
            treatment_value = variant_metrics[treatment_variant][metric]
            
            effect_size = (treatment_value - control_value) / control_value
            p_value = np.random.uniform(0.05, 0.5)
            
            test_results[metric] = {
                'control_value': control_value,
                'treatment_value': treatment_value,
                'lift': effect_size,
                'p_value': p_value,
                'significant': p_value < self.significance_level,
                'confidence_interval': (
                    effect_size - 1.96 * abs(effect_size) * 0.1,
                    effect_size + 1.96 * abs(effect_size) * 0.1
                )
            }
        
        return test_results
    
    def generate_experiment_report(self, experiment_name: str) -> str:
        """Generate experiment results report"""
        if experiment_name not in self.experiments:
            return f"Experiment {experiment_name} not found"
        
        experiment = self.experiments[experiment_name]
        
        if experiment['status'] != 'completed':
            return f"Experiment {experiment_name} is not completed yet"
        
        results = experiment['results']
        report = f"# A/B Test Results: {experiment_name}\n\n"
        
        report += f"**Experiment Duration**: {experiment['start_date']} to {experiment['end_date']}\n\n"
        
        # Primary metric results
        primary_results = results['statistical_tests'][experiment['primary_metric']]
        report += f"## Primary Metric: {experiment['primary_metric']}\n\n"
        report += f"- **Control**: {primary_results['control_value']:.4f}\n"
        report += f"- **Treatment**: {primary_results['treatment_value']:.4f}\n"
        report += f"- **Lift**: {primary_results['lift']:.2%}\n"
        report += f"- **P-value**: {primary_results['p_value']:.4f}\n"
        report += f"- **Significant**: {'✅ Yes' if primary_results['significant'] else '❌ No'}\n\n"
        
        # Secondary metrics
        if experiment['secondary_metrics']:
            report += "## Secondary Metrics\n\n"
            
            for metric in experiment['secondary_metrics']:
                metric_results = results['statistical_tests'][metric]
                report += f"### {metric}\n"
                report += f"- **Lift**: {metric_results['lift']:.2%}\n"
                report += f"- **Significant**: {'✅ Yes' if metric_results['significant'] else '❌ No'}\n\n"
        
        return report

# Example usage of A/B testing framework
ab_framework = ABTestFramework()

# Design experiment
experiment_config = ab_framework.design_experiment(
    experiment_name="hybrid_vs_collaborative",
    variants={
        "control": "collaborative_filtering",
        "treatment": "hybrid_system"
    },
    primary_metric="ctr",
    secondary_metrics=["diversity_score", "user_satisfaction"],
    minimum_detectable_effect=0.05,
    expected_baseline_rate=0.1
)

print("Experiment designed with required sample size:", experiment_config['sample_size_per_variant'])
```

## 3. Long-term Impact Evaluation

### Temporal Analysis and Monitoring System

```python
class LongTermEvaluationSystem:
    def __init__(self):
        self.monitoring_metrics = {}
        self.temporal_data = {}
        self.alert_thresholds = {}
        self.evaluation_history = []
        
    def setup_monitoring(self, metrics: List[str], alert_thresholds: Dict[str, Tuple[float, float]]):
        """Setup long-term monitoring system"""
        for metric in metrics:
            self.monitoring_metrics[metric] = []
            self.temporal_data[metric] = pd.DataFrame(columns=['timestamp', 'value', 'metadata'])
        
        self.alert_thresholds = alert_thresholds
    
    def record_metric(self, metric_name: str, value: float, metadata: Dict[str, Any] = None):
        """Record a metric value with timestamp"""
        timestamp = pd.Timestamp.now()
        
        new_record = pd.DataFrame({
            'timestamp': [timestamp],
            'value': [value],
            'metadata': [metadata or {}]
        })
        
        if metric_name in self.temporal_data:
            self.temporal_data[metric_name] = pd.concat([
                self.temporal_data[metric_name], 
                new_record
            ], ignore_index=True)
        else:
            self.temporal_data[metric_name] = new_record
        
        # Check for alerts
        self._check_alerts(metric_name, value)
    
    def _check_alerts(self, metric_name: str, value: float):
        """Check if metric value triggers any alerts"""
        if metric_name in self.alert_thresholds:
            lower_bound, upper_bound = self.alert_thresholds[metric_name]
            
            if value < lower_bound:
                print(f"🚨 ALERT: {metric_name} below threshold: {value:.4f} < {lower_bound}")
            elif value > upper_bound:
                print(f"🚨 ALERT: {metric_name} above threshold: {value:.4f} > {upper_bound}")
    
    def analyze_temporal_trends(self, metric_name: str, window_size: int = 30) -> Dict[str, Any]:
        """Analyze temporal trends in metrics"""
        if metric_name not in self.temporal_data:
            return {"error": f"No data available for metric {metric_name}"}
        
        data = self.temporal_data[metric_name].copy()
        data = data.sort_values('timestamp')
        
        # Calculate moving averages
        data['moving_avg'] = data['value'].rolling(window=min(window_size, len(data))).mean()
        
        # Trend analysis
        if len(data) > 1:
            # Simple linear trend
            x = np.arange(len(data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, data['value'])
            
            trend_analysis = {
                'slope': slope,
                'r_squared': r_value**2,
                'p_value': p_value,
                'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                'trend_strength': abs(r_value)
            }
        else:
            trend_analysis = {'error': 'Insufficient data for trend analysis'}
        
        # Seasonality detection (simplified)
        seasonality_score = 0
        if len(data) > 7:
            # Check for weekly patterns
            data['day_of_week'] = data['timestamp'].dt.day_of_week
            weekly_variance = data.groupby('day_of_week')['value'].var().mean()
            overall_variance = data['value'].var()
            seasonality_score = weekly_variance / overall_variance if overall_variance > 0 else 0
        
        return {
            'trend_analysis': trend_analysis,
            'seasonality_score': seasonality_score,
            'recent_average': data['value'].tail(window_size).mean(),
            'overall_average': data['value'].mean(),
            'volatility': data['value'].std(),
            'data_points': len(data)
        }
    
    def evaluate_system_health(self) -> Dict[str, Any]:
        """Comprehensive system health evaluation"""
        health_report = {
            'overall_status': 'healthy',
            'metric_health': {},
            'alerts_triggered': [],
            'recommendations': []
        }
        
        for metric_name in self.monitoring_metrics.keys():
            if metric_name in self.temporal_data and len(self.temporal_data[metric_name]) > 0:
                recent_data = self.temporal_data[metric_name].tail(10)
                
                # Check metric health
                recent_mean = recent_data['value'].mean()
                recent_std = recent_data['value'].std()
                
                # Health indicators
                if metric_name in self.alert_thresholds:
                    lower_bound, upper_bound = self.alert_thresholds[metric_name]
                    
                    if recent_mean < lower_bound or recent_mean > upper_bound:
                        health_status = 'unhealthy'
                        health_report['overall_status'] = 'warning'
                    elif recent_std > (upper_bound - lower_bound) * 0.3:
                        health_status = 'volatile'
                    else:
                        health_status = 'healthy'
                else:
                    health_status = 'unknown'
                
                health_report['metric_health'][metric_name] = {
                    'status': health_status,
                    'recent_mean': recent_mean,
                    'recent_std': recent_std,
                    'data_points': len(recent_data)
                }
        
        return health_report
    
    def generate_long_term_report(self, days_back: int = 30) -> str:
        """Generate comprehensive long-term evaluation report"""
        cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days_back)
        
        report = f"# Long-term System Evaluation Report\n"
        report += f"**Analysis Period**: Last {days_back} days\n"
        report += f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # System health overview
        health = self.evaluate_system_health()
        report += f"## System Health Status: {health['overall_status'].upper()}\n\n"
        
        # Individual metric analysis
        for metric_name in self.monitoring_metrics.keys():
            if metric_name in self.temporal_data:
                recent_data = self.temporal_data[metric_name]
                recent_data = recent_data[recent_data['timestamp'] >= cutoff_date]
                
                if len(recent_data) > 0:
                    trend_analysis = self.analyze_temporal_trends(metric_name)
                    
                    report += f"### {metric_name}\n"
                    report += f"- **Recent Average**: {recent_data['value'].mean():.4f}\n"
                    report += f"- **Trend**: {trend_analysis.get('trend_analysis', {}).get('trend_direction', 'unknown')}\n"
                    report += f"- **Volatility**: {recent_data['value'].std():.4f}\n"
                    report += f"- **Data Points**: {len(recent_data)}\n\n"
        
        return report

# Example usage
long_term_evaluator = LongTermEvaluationSystem()

# Setup monitoring
long_term_evaluator.setup_monitoring(
    metrics=['precision', 'diversity', 'user_satisfaction', 'response_time'],
    alert_thresholds={
        'precision': (0.1, 0.9),
        'diversity': (0.3, 1.0),
        'user_satisfaction': (3.0, 5.0),
        'response_time': (0.0, 2.0)
    }
)

# Simulate recording metrics over time
for day in range(30):
    # Simulate daily metrics
    precision = 0.5 + 0.1 * np.sin(day * 0.2) + np.random.normal(0, 0.05)
    diversity = 0.7 + 0.05 * np.cos(day * 0.3) + np.random.normal(0, 0.03)
    satisfaction = 4.0 + 0.2 * np.sin(day * 0.1) + np.random.normal(0, 0.1)
    response_time = 0.8 + 0.1 * np.random.random()
    
    timestamp = pd.Timestamp.now() - pd.Timedelta(days=30-day)
    
    long_term_evaluator.record_metric('precision', precision)
    long_term_evaluator.record_metric('diversity', diversity) 
    long_term_evaluator.record_metric('user_satisfaction', satisfaction)
    long_term_evaluator.record_metric('response_time', response_time)

print("Long-term monitoring system set up and populated with 30 days of data")
```

## 4. Example: Complete Evaluation Pipeline

```python
def run_complete_evaluation():
    """Example of running complete evaluation pipeline"""
    
    print("🔍 Setting up comprehensive evaluation pipeline...")
    
    # 1. Setup evaluation framework
    eval_framework = HybridEvaluationFramework()
    
    # Register evaluators
    eval_framework.register_evaluator(EvaluationDimension.ACCURACY, AccuracyEvaluator())
    eval_framework.register_evaluator(EvaluationDimension.DIVERSITY, DiversityEvaluator())
    eval_framework.register_evaluator(EvaluationDimension.FAIRNESS, FairnessEvaluator())
    
    # 2. Create mock data
    np.random.seed(42)
    
    # Test data
    test_data = pd.DataFrame({
        'user_id': range(1000),
        'features': [np.random.randn(10) for _ in range(1000)]
    })
    
    # Ground truth
    ground_truth = pd.DataFrame({
        'user_id': np.repeat(range(100), 5),
        'item_id': np.random.randint(1, 1000, 500),
        'rating': np.random.uniform(3, 5, 500)
    })
    
    # User interactions
    user_interactions = pd.DataFrame({
        'user_id': range(1000),
        'gender': np.random.choice(['M', 'F', 'Other'], 1000),
        'age_group': np.random.choice(['18-25', '26-35', '36-50', '50+'], 1000)
    })
    
    # 3. Create mock hybrid system
    class MockHybridSystem:
        def recommend(self, user_id, k=10):
            # Mock recommendations with scores
            items = np.random.randint(1, 1000, k)
            scores = np.random.uniform(0.1, 0.9, k)
            return list(zip(items, scores))
    
    hybrid_system = MockHybridSystem()
    
    # 4. Run evaluation
    print("📊 Running comprehensive evaluation...")
    results = eval_framework.evaluate_hybrid_system(
        hybrid_system=hybrid_system,
        test_data=test_data,
        ground_truth=ground_truth,
        user_interactions=user_interactions
    )
    
    # 5. Generate report
    report = eval_framework.generate_evaluation_report()
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(report)
    
    # 6. Setup and run A/B test
    print("\n🧪 Running A/B test...")
    ab_framework = ABTestFramework()
    
    experiment_config = ab_framework.design_experiment(
        experiment_name="hybrid_system_test",
        variants={
            "control": "baseline_system",
            "treatment": "hybrid_system"
        },
        primary_metric="ctr",
        secondary_metrics=["diversity_score", "user_satisfaction"],
        minimum_detectable_effect=0.05,
        expected_baseline_rate=0.1
    )
    
    # Mock experiment data
    experiment_data = pd.DataFrame({
        'user_id': range(2000),
        'clicked': np.random.binomial(1, 0.12, 2000),
        'converted': np.random.binomial(1, 0.05, 2000)
    })
    
    # Random variant assignment
    variant_assignment = {i: 'control' if i % 2 == 0 else 'treatment' for i in range(2000)}
    
    # Run experiment
    ab_results = ab_framework.run_experiment(
        experiment_name="hybrid_system_test",
        data=experiment_data,
        variant_assignment=variant_assignment
    )
    
    # Generate A/B test report
    ab_report = ab_framework.generate_experiment_report("hybrid_system_test")
    print("\n" + "="*50)
    print("A/B TEST RESULTS")
    print("="*50)
    print(ab_report)
    
    # 7. Long-term monitoring report
    print("\n📈 Long-term monitoring report...")
    long_term_report = long_term_evaluator.generate_long_term_report()
    print("\n" + "="*50)
    print("LONG-TERM MONITORING")
    print("="*50)
    print(long_term_report)
    
    print("\n✅ Complete evaluation pipeline executed successfully!")

# Run the complete evaluation
if __name__ == "__main__":
    run_complete_evaluation()
```

## Key Takeaways

1. **Multi-dimensional Evaluation**: Hybrid systems require evaluation across accuracy, diversity, fairness, efficiency, and other dimensions

2. **Statistical Rigor**: Proper statistical testing, confidence intervals, and significance testing are crucial for reliable evaluation

3. **A/B Testing**: Controlled experiments help validate the real-world performance of hybrid systems against baselines

4. **Long-term Monitoring**: Continuous monitoring detects performance degradation, concept drift, and system health issues

5. **Fairness Assessment**: Evaluating demographic parity and equal opportunity ensures ethical AI deployment

6. **Comprehensive Reporting**: Clear, actionable reports help stakeholders understand system performance and make decisions

## Study Questions

### Beginner Level
1. What are the key dimensions for evaluating hybrid recommendation systems?
2. How do you calculate precision@k and NDCG for recommendation systems?
3. What is demographic parity in the context of fair recommendations?
4. Why is A/B testing important for evaluating recommendation systems?

### Intermediate Level
1. How would you design an A/B test to compare a hybrid system against individual component systems?
2. What are the challenges in evaluating diversity and novelty in recommendations?
3. How can you detect and handle concept drift in recommendation systems?
4. What metrics would you use to evaluate the long-term impact of a recommendation system?

### Advanced Level
1. Design a comprehensive evaluation framework that handles multi-stakeholder objectives
2. How would you evaluate the explainability of a hybrid recommendation system?
3. Implement a causal inference approach to measure the true impact of recommendations
4. Design a real-time evaluation system that adapts evaluation criteria based on changing user behavior

## Next Session Preview

Tomorrow we'll explore **Advanced Neural Approaches for Search and Recommendation**, covering:
- Deep learning architectures for recommendation systems
- Neural collaborative filtering and autoencoders
- Attention mechanisms and transformer-based models
- Graph neural networks for recommendations
- Multi-task learning for search and recommendation
- Advanced embedding techniques and representation learning

We'll implement sophisticated neural architectures and explore how deep learning is revolutionizing search and recommendation systems!