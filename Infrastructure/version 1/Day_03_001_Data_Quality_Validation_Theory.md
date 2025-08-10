# Day 3.1: Data Quality Validation Theory & Statistical Frameworks

## 📊 Data Governance, Metadata & Cataloging - Part 1

**Focus**: Statistical Data Quality Metrics, Validation Frameworks, and Anomaly Detection  
**Duration**: 2-3 hours  
**Level**: Beginner to Advanced  

---

## 🎯 Learning Objectives

- Master statistical foundations of data quality measurement
- Understand Great Expectations architecture and validation rule engines
- Learn anomaly detection algorithms for automated data quality monitoring
- Implement real-time vs batch validation trade-off strategies

---

## 📐 Mathematical Foundations of Data Quality

### **Data Quality Dimensions Framework**

#### **1. Completeness Metrics**
```
Mathematical Definition:
Completeness(D) = (|D_present| / |D_expected|) × 100%

Where:
- D_present = set of non-null, non-empty values
- D_expected = total expected values in dataset
- Completeness ∈ [0%, 100%]
```

```python
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any
import warnings

class DataQualityMetrics:
    """Comprehensive data quality measurement framework"""
    
    def __init__(self):
        self.quality_dimensions = {
            'completeness': self.measure_completeness,
            'accuracy': self.measure_accuracy,
            'consistency': self.measure_consistency,
            'validity': self.measure_validity,
            'uniqueness': self.measure_uniqueness,
            'timeliness': self.measure_timeliness
        }
        
    def measure_completeness(self, data: pd.DataFrame, 
                           expected_schema: Dict[str, Any]) -> Dict[str, float]:
        """Measure data completeness across multiple dimensions"""
        completeness_metrics = {}
        
        # Column-level completeness
        for column in data.columns:
            non_null_count = data[column].notna().sum()
            total_count = len(data)
            
            # Basic completeness
            basic_completeness = non_null_count / total_count if total_count > 0 else 0
            
            # Semantic completeness (non-empty strings, non-zero numbers)
            if data[column].dtype == 'object':
                non_empty_count = data[column].str.strip().ne('').sum()
                semantic_completeness = non_empty_count / total_count if total_count > 0 else 0
            else:
                semantic_completeness = basic_completeness
            
            completeness_metrics[f'{column}_basic'] = basic_completeness
            completeness_metrics[f'{column}_semantic'] = semantic_completeness
        
        # Record-level completeness
        complete_records = data.dropna().shape[0]
        record_completeness = complete_records / len(data) if len(data) > 0 else 0
        completeness_metrics['record_level'] = record_completeness
        
        # Schema completeness (expected vs actual columns)
        expected_columns = set(expected_schema.keys())
        actual_columns = set(data.columns)
        schema_completeness = len(expected_columns & actual_columns) / len(expected_columns)
        completeness_metrics['schema_level'] = schema_completeness
        
        return completeness_metrics
    
    def measure_accuracy(self, data: pd.DataFrame, 
                        reference_data: Optional[pd.DataFrame] = None,
                        business_rules: Optional[Dict] = None) -> Dict[str, float]:
        """Measure data accuracy using multiple approaches"""
        accuracy_metrics = {}
        
        # Reference-based accuracy (if reference data available)
        if reference_data is not None:
            for column in data.columns:
                if column in reference_data.columns:
                    # Exact match accuracy
                    exact_matches = (data[column] == reference_data[column]).sum()
                    exact_accuracy = exact_matches / len(data) if len(data) > 0 else 0
                    accuracy_metrics[f'{column}_exact_match'] = exact_accuracy
                    
                    # Fuzzy match accuracy (for string data)
                    if data[column].dtype == 'object':
                        fuzzy_accuracy = self.calculate_fuzzy_accuracy(
                            data[column], reference_data[column]
                        )
                        accuracy_metrics[f'{column}_fuzzy_match'] = fuzzy_accuracy
        
        # Business rule-based accuracy
        if business_rules:
            for rule_name, rule_function in business_rules.items():
                try:
                    rule_violations = data.apply(rule_function, axis=1)
                    rule_accuracy = (~rule_violations).sum() / len(data) if len(data) > 0 else 0
                    accuracy_metrics[f'rule_{rule_name}'] = rule_accuracy
                except Exception as e:
                    accuracy_metrics[f'rule_{rule_name}'] = 0.0
                    warnings.warn(f"Rule {rule_name} failed: {e}")
        
        # Statistical accuracy (outlier detection)
        for column in data.select_dtypes(include=[np.number]).columns:
            z_scores = np.abs(stats.zscore(data[column].dropna()))
            outlier_ratio = (z_scores > 3).sum() / len(z_scores) if len(z_scores) > 0 else 0
            statistical_accuracy = 1 - outlier_ratio
            accuracy_metrics[f'{column}_statistical'] = statistical_accuracy
        
        return accuracy_metrics
    
    def calculate_fuzzy_accuracy(self, series1: pd.Series, series2: pd.Series, 
                               threshold: float = 0.8) -> float:
        """Calculate fuzzy string matching accuracy"""
        from difflib import SequenceMatcher
        
        matches = []
        for val1, val2 in zip(series1.fillna(''), series2.fillna('')):
            similarity = SequenceMatcher(None, str(val1), str(val2)).ratio()
            matches.append(similarity >= threshold)
        
        return sum(matches) / len(matches) if matches else 0.0
    
    def measure_consistency(self, data: pd.DataFrame, 
                          consistency_rules: Dict[str, Any]) -> Dict[str, float]:
        """Measure data consistency across records and fields"""
        consistency_metrics = {}
        
        # Format consistency
        for column in data.select_dtypes(include=['object']).columns:
            if data[column].notna().sum() == 0:
                continue
                
            # Pattern consistency (e.g., phone numbers, emails)
            if 'patterns' in consistency_rules.get(column, {}):
                pattern = consistency_rules[column]['patterns'][0]  # Primary pattern
                pattern_matches = data[column].str.match(pattern, na=False).sum()
                pattern_consistency = pattern_matches / data[column].notna().sum()
                consistency_metrics[f'{column}_pattern'] = pattern_consistency
            
            # Case consistency
            uppercase_ratio = data[column].str.isupper().sum() / data[column].notna().sum()
            lowercase_ratio = data[column].str.islower().sum() / data[column].notna().sum()
            case_consistency = max(uppercase_ratio, lowercase_ratio)
            consistency_metrics[f'{column}_case'] = case_consistency
        
        # Cross-field consistency
        if 'cross_field_rules' in consistency_rules:
            for rule_name, rule_function in consistency_rules['cross_field_rules'].items():
                consistent_records = data.apply(rule_function, axis=1).sum()
                cross_field_consistency = consistent_records / len(data) if len(data) > 0 else 0
                consistency_metrics[f'cross_field_{rule_name}'] = cross_field_consistency
        
        # Temporal consistency (for time-series data)
        if 'timestamp_column' in consistency_rules:
            ts_column = consistency_rules['timestamp_column']
            if ts_column in data.columns:
                # Check for chronological order
                is_sorted = data[ts_column].is_monotonic_increasing
                temporal_consistency = 1.0 if is_sorted else 0.0
                consistency_metrics['temporal_order'] = temporal_consistency
        
        return consistency_metrics
    
    def measure_validity(self, data: pd.DataFrame, 
                        validation_rules: Dict[str, Any]) -> Dict[str, float]:
        """Measure data validity against domain constraints"""
        validity_metrics = {}
        
        for column, rules in validation_rules.items():
            if column not in data.columns:
                continue
            
            valid_count = 0
            total_count = data[column].notna().sum()
            
            if total_count == 0:
                validity_metrics[column] = 0.0
                continue
            
            # Range validation
            if 'range' in rules:
                min_val, max_val = rules['range']
                range_valid = data[column].between(min_val, max_val, inclusive='both')
                valid_count += range_valid.sum()
            
            # Enumeration validation
            elif 'enum' in rules:
                enum_values = set(rules['enum'])
                enum_valid = data[column].isin(enum_values)
                valid_count += enum_valid.sum()
            
            # Regex validation
            elif 'regex' in rules:
                regex_pattern = rules['regex']
                regex_valid = data[column].str.match(regex_pattern, na=False)
                valid_count += regex_valid.sum()
            
            # Custom validation function
            elif 'function' in rules:
                custom_valid = data[column].apply(rules['function'])
                valid_count += custom_valid.sum()
            
            validity_metrics[column] = valid_count / total_count
        
        return validity_metrics
```

#### **2. Accuracy Measurement Theory**
```
Accuracy Types:
1. Syntactic Accuracy: Data conforms to format rules
2. Semantic Accuracy: Data represents real-world truth
3. Pragmatic Accuracy: Data is correct for intended use

Mathematical Model:
Accuracy(D, R) = |{d ∈ D : d matches r ∈ R}| / |D|

Where:
- D = dataset
- R = reference/truth set
- matches = similarity function (exact, fuzzy, semantic)
```

---

## 🔍 Great Expectations Architecture Deep Dive

### **Expectation Engine Theoretical Foundation**

#### **Expectation Types and Computational Complexity**
```python
from abc import ABC, abstractmethod
from enum import Enum
import great_expectations as ge
from great_expectations.core import ExpectationSuite, ExpectationConfiguration
from great_expectations.dataset import PandasDataset

class ExpectationComplexity(Enum):
    """Computational complexity classification for expectations"""
    CONSTANT = "O(1)"      # Column metadata checks
    LINEAR = "O(n)"        # Single pass over data
    QUADRATIC = "O(n²)"    # Pairwise comparisons
    LOGARITHMIC = "O(log n)" # Sorted operations

class GreatExpectationsFramework:
    """Advanced Great Expectations implementation framework"""
    
    def __init__(self):
        self.expectation_registry = {}
        self.performance_profiler = ExpectationPerformanceProfiler()
        self.validation_engine = ValidationEngine()
        
    def register_custom_expectation(self, expectation_class):
        """Register custom expectation with performance characteristics"""
        expectation_name = expectation_class.__name__
        
        # Analyze computational complexity
        complexity = self.analyze_expectation_complexity(expectation_class)
        
        # Estimate resource requirements
        resource_profile = self.estimate_resource_requirements(expectation_class)
        
        self.expectation_registry[expectation_name] = {
            'class': expectation_class,
            'complexity': complexity,
            'resource_profile': resource_profile,
            'validation_function': expectation_class._validate
        }
        
        return {
            'expectation_name': expectation_name,
            'computational_complexity': complexity.value,
            'memory_requirement_mb': resource_profile['memory_mb'],
            'estimated_runtime_ms': resource_profile['runtime_ms']
        }
    
    def analyze_expectation_complexity(self, expectation_class) -> ExpectationComplexity:
        """Analyze computational complexity of expectation"""
        expectation_name = expectation_class.__name__.lower()
        
        # Pattern-based complexity classification
        if any(pattern in expectation_name for pattern in ['unique', 'distinct']):
            return ExpectationComplexity.LINEAR  # Hash-based uniqueness check
        
        elif any(pattern in expectation_name for pattern in ['sorted', 'increasing', 'decreasing']):
            return ExpectationComplexity.LINEAR  # Single pass comparison
        
        elif any(pattern in expectation_name for pattern in ['correlation', 'covariance']):
            return ExpectationComplexity.LINEAR  # Vectorized operations
        
        elif any(pattern in expectation_name for pattern in ['duplicate', 'pairs']):
            return ExpectationComplexity.QUADRATIC  # Pairwise comparisons
        
        elif any(pattern in expectation_name for pattern in ['quantile', 'percentile']):
            return ExpectationComplexity.LOGARITHMIC  # Requires sorting
        
        else:
            return ExpectationComplexity.LINEAR  # Default assumption
    
    def create_validation_suite(self, dataset_profile: Dict[str, Any], 
                              quality_requirements: Dict[str, float]) -> ExpectationSuite:
        """Create optimized validation suite based on data profile"""
        
        suite = ExpectationSuite(expectation_suite_name="automated_quality_suite")
        
        # Completeness expectations
        for column, completeness_req in quality_requirements.get('completeness', {}).items():
            if completeness_req > 0:
                expectation_config = ExpectationConfiguration(
                    expectation_type="expect_column_values_to_not_be_null",
                    kwargs={
                        "column": column,
                        "mostly": completeness_req
                    }
                )
                suite.add_expectation(expectation_config)
        
        # Uniqueness expectations
        for column in quality_requirements.get('unique_columns', []):
            expectation_config = ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_unique",
                kwargs={"column": column}
            )
            suite.add_expectation(expectation_config)
        
        # Range expectations based on data profiling
        for column, stats in dataset_profile.get('numeric_columns', {}).items():
            # Set range based on observed distribution + buffer
            lower_bound = stats['min'] - (stats['std'] * 3)
            upper_bound = stats['max'] + (stats['std'] * 3)
            
            expectation_config = ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": column,
                    "min_value": lower_bound,
                    "max_value": upper_bound,
                    "mostly": 0.95  # Allow 5% outliers
                }
            )
            suite.add_expectation(expectation_config)
        
        # Pattern expectations for categorical columns
        for column, patterns in dataset_profile.get('categorical_patterns', {}).items():
            if patterns:
                # Use most common pattern
                primary_pattern = max(patterns.items(), key=lambda x: x[1])[0]
                
                expectation_config = ExpectationConfiguration(
                    expectation_type="expect_column_values_to_match_regex",
                    kwargs={
                        "column": column,
                        "regex": primary_pattern,
                        "mostly": 0.90  # Allow some variation
                    }
                )
                suite.add_expectation(expectation_config)
        
        return suite
    
    def optimize_validation_performance(self, suite: ExpectationSuite, 
                                      data_size: int) -> Dict[str, Any]:
        """Optimize validation suite for performance"""
        optimization_results = {
            'original_expectations': len(suite.expectations),
            'optimized_expectations': 0,
            'estimated_runtime_reduction': 0,
            'optimization_strategies': []
        }
        
        optimized_expectations = []
        
        # Group expectations by column for batch processing
        column_expectations = {}
        for expectation in suite.expectations:
            column = expectation.kwargs.get('column')
            if column:
                if column not in column_expectations:
                    column_expectations[column] = []
                column_expectations[column].append(expectation)
        
        # Optimize per-column expectations
        for column, expectations in column_expectations.items():
            optimized_column_expectations = self.optimize_column_expectations(
                expectations, data_size
            )
            optimized_expectations.extend(optimized_column_expectations)
            
            optimization_results['optimization_strategies'].append(
                f"Batched {len(expectations)} expectations for column {column}"
            )
        
        # Create optimized suite
        optimized_suite = ExpectationSuite(
            expectation_suite_name=f"{suite.expectation_suite_name}_optimized"
        )
        
        for expectation in optimized_expectations:
            optimized_suite.add_expectation(expectation)
        
        optimization_results['optimized_expectations'] = len(optimized_expectations)
        optimization_results['estimated_runtime_reduction'] = (
            (len(suite.expectations) - len(optimized_expectations)) / 
            len(suite.expectations) * 100
        )
        
        return optimization_results, optimized_suite
    
    def optimize_column_expectations(self, expectations: List, 
                                   data_size: int) -> List[ExpectationConfiguration]:
        """Optimize expectations for a single column"""
        # Remove redundant expectations
        unique_expectations = self.remove_redundant_expectations(expectations)
        
        # Reorder by computational cost (cheapest first)
        ordered_expectations = self.order_by_computational_cost(unique_expectations)
        
        # Apply sampling for large datasets on expensive operations
        if data_size > 1000000:  # 1M+ records
            sampled_expectations = self.apply_sampling_optimization(
                ordered_expectations, data_size
            )
            return sampled_expectations
        
        return ordered_expectations
    
    def remove_redundant_expectations(self, 
                                    expectations: List[ExpectationConfiguration]) -> List:
        """Remove redundant or conflicting expectations"""
        unique_expectations = []
        seen_types = set()
        
        for expectation in expectations:
            expectation_key = (
                expectation.expectation_type,
                frozenset(expectation.kwargs.items())
            )
            
            if expectation_key not in seen_types:
                unique_expectations.append(expectation)
                seen_types.add(expectation_key)
        
        return unique_expectations

class ExpectationPerformanceProfiler:
    """Profile and optimize expectation performance"""
    
    def __init__(self):
        self.performance_history = {}
        self.complexity_models = {}
        
    def profile_expectation_performance(self, expectation_type: str, 
                                      data_sizes: List[int]) -> Dict[str, Any]:
        """Profile expectation performance across different data sizes"""
        performance_profile = {
            'expectation_type': expectation_type,
            'size_vs_runtime': {},
            'complexity_model': None,
            'memory_scaling': {},
            'optimization_recommendations': []
        }
        
        runtimes = []
        memory_usage = []
        
        for size in data_sizes:
            # Generate synthetic data of specified size
            test_data = self.generate_test_data(size)
            
            # Measure runtime and memory
            runtime, memory = self.measure_expectation_performance(
                expectation_type, test_data
            )
            
            performance_profile['size_vs_runtime'][size] = runtime
            performance_profile['memory_scaling'][size] = memory
            
            runtimes.append(runtime)
            memory_usage.append(memory)
        
        # Fit complexity model
        complexity_model = self.fit_complexity_model(data_sizes, runtimes)
        performance_profile['complexity_model'] = complexity_model
        
        # Generate optimization recommendations
        recommendations = self.generate_optimization_recommendations(
            performance_profile
        )
        performance_profile['optimization_recommendations'] = recommendations
        
        return performance_profile
    
    def fit_complexity_model(self, data_sizes: List[int], 
                           runtimes: List[float]) -> Dict[str, Any]:
        """Fit computational complexity model to observed performance"""
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.metrics import r2_score
        
        models = {}
        
        # Test different complexity models
        X = np.array(data_sizes).reshape(-1, 1)
        y = np.array(runtimes)
        
        # Linear model O(n)
        linear_model = LinearRegression()
        linear_model.fit(X, y)
        linear_score = r2_score(y, linear_model.predict(X))
        models['linear'] = {'model': linear_model, 'r2': linear_score}
        
        # Quadratic model O(n²)
        poly_features = PolynomialFeatures(degree=2)
        X_poly = poly_features.fit_transform(X)
        quad_model = LinearRegression()
        quad_model.fit(X_poly, y)
        quad_score = r2_score(y, quad_model.predict(X_poly))
        models['quadratic'] = {'model': quad_model, 'r2': quad_score}
        
        # Logarithmic model O(log n)
        X_log = np.log(X)
        log_model = LinearRegression()
        log_model.fit(X_log, y)
        log_score = r2_score(y, log_model.predict(X_log))
        models['logarithmic'] = {'model': log_model, 'r2': log_score}
        
        # Select best model
        best_model_type = max(models.keys(), key=lambda k: models[k]['r2'])
        
        return {
            'best_model_type': best_model_type,
            'r2_score': models[best_model_type]['r2'],
            'all_models': {k: v['r2'] for k, v in models.items()}
        }
```

This completes Part 1 of Day 3, establishing the mathematical and theoretical foundations of data quality validation. The content provides deep understanding of statistical quality metrics and the Great Expectations framework architecture.