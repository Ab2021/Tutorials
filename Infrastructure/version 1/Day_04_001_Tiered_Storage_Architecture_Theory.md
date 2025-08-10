# Day 4.1: Tiered Storage Architecture Theory

## 💾 Storage Layers & Feature Store Deep Dive - Part 1

**Focus**: Storage Hierarchy Theory, Performance Characteristics, Cost Optimization Models  
**Duration**: 2-3 hours  
**Level**: Beginner to Advanced  

---

## 🎯 Learning Objectives

- Master tiered storage architecture principles and mathematical optimization models
- Understand storage media characteristics and performance trade-offs (SSD/NVMe vs HDD)
- Learn hot/warm/cold data classification algorithms and placement strategies
- Implement cost-performance optimization frameworks for ML workloads

---

## 📊 Storage Hierarchy Mathematical Framework

### **Storage Performance Theory**

#### **Storage Access Patterns Mathematical Model**
```
Storage Access Function:
A(t) = λ × e^(-αt) + β

Where:
- A(t) = access frequency at time t
- λ = initial access intensity
- α = decay rate (data cooling factor)
- β = baseline access rate

Cost-Performance Optimization:
minimize: Σ(Storage_Cost_i × Data_Volume_i)
subject to: Performance_SLA ≥ Required_Performance
           Availability_SLA ≥ Required_Availability
```

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime, timedelta
import math
import json

class StorageTier(Enum):
    """Storage tier classifications"""
    HOT = "hot"        # High-performance, frequently accessed
    WARM = "warm"      # Moderate performance, occasionally accessed  
    COLD = "cold"      # Low-cost, rarely accessed
    ARCHIVE = "archive" # Ultra-low-cost, long-term retention

class StorageMediaType(Enum):
    """Storage media types with performance characteristics"""
    NVME_SSD = "nvme_ssd"
    SATA_SSD = "sata_ssd"
    SAS_HDD = "sas_hdd"
    SATA_HDD = "sata_hdd"
    TAPE = "tape"
    OPTICAL = "optical"

@dataclass
class StorageCharacteristics:
    """Storage media performance and cost characteristics"""
    media_type: StorageMediaType
    random_read_iops: int
    random_write_iops: int
    sequential_read_mbps: int
    sequential_write_mbps: int
    latency_us: float
    durability_years: int
    cost_per_gb_per_month: float
    power_consumption_watts: float
    density_gb_per_u: int

class StoragePerformanceAnalyzer:
    """Analyze and model storage performance characteristics"""
    
    def __init__(self):
        self.storage_profiles = self._initialize_storage_profiles()
        self.workload_patterns = {}
        self.performance_models = {}
        
    def _initialize_storage_profiles(self) -> Dict[StorageMediaType, StorageCharacteristics]:
        """Initialize storage media performance profiles"""
        return {
            StorageMediaType.NVME_SSD: StorageCharacteristics(
                media_type=StorageMediaType.NVME_SSD,
                random_read_iops=1000000,
                random_write_iops=800000,
                sequential_read_mbps=7000,
                sequential_write_mbps=6000,
                latency_us=100,
                durability_years=5,
                cost_per_gb_per_month=0.25,
                power_consumption_watts=8.5,
                density_gb_per_u=30720  # 30TB per U
            ),
            
            StorageMediaType.SATA_SSD: StorageCharacteristics(
                media_type=StorageMediaType.SATA_SSD,
                random_read_iops=100000,
                random_write_iops=90000,
                sequential_read_mbps=550,
                sequential_write_mbps=520,
                latency_us=500,
                durability_years=5,
                cost_per_gb_per_month=0.15,
                power_consumption_watts=3.5,
                density_gb_per_u=61440  # 60TB per U
            ),
            
            StorageMediaType.SAS_HDD: StorageCharacteristics(
                media_type=StorageMediaType.SAS_HDD,
                random_read_iops=200,
                random_write_iops=180,
                sequential_read_mbps=285,
                sequential_write_mbps=275,
                latency_us=8500,
                durability_years=5,
                cost_per_gb_per_month=0.045,
                power_consumption_watts=12.0,
                density_gb_per_u=245760  # 240TB per U
            ),
            
            StorageMediaType.SATA_HDD: StorageCharacteristics(
                media_type=StorageMediaType.SATA_HDD,
                random_read_iops=120,
                random_write_iops=110,
                sequential_read_mbps=220,
                sequential_write_mbps=210,
                latency_us=12000,
                durability_years=3,
                cost_per_gb_per_month=0.025,
                power_consumption_watts=8.0,
                density_gb_per_u=245760  # 240TB per U
            ),
            
            StorageMediaType.TAPE: StorageCharacteristics(
                media_type=StorageMediaType.TAPE,
                random_read_iops=1,
                random_write_iops=1,
                sequential_read_mbps=750,
                sequential_write_mbps=750,
                latency_us=30000000,  # 30 seconds average seek
                durability_years=30,
                cost_per_gb_per_month=0.002,
                power_consumption_watts=150.0,  # Library power
                density_gb_per_u=1048576  # 1PB per U (library)
            )
        }
    
    def analyze_workload_characteristics(self, access_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze workload to determine optimal storage configuration"""
        
        workload_analysis = {
            'total_operations': len(access_patterns),
            'read_write_ratio': 0.0,
            'sequential_random_ratio': 0.0,
            'access_frequency_distribution': {},
            'data_size_distribution': {},
            'temporal_patterns': {},
            'recommended_tiers': {}
        }
        
        # Analyze read/write patterns
        read_ops = sum(1 for op in access_patterns if op['operation_type'] == 'read')
        write_ops = len(access_patterns) - read_ops
        workload_analysis['read_write_ratio'] = read_ops / write_ops if write_ops > 0 else float('inf')
        
        # Analyze access patterns
        sequential_ops = sum(1 for op in access_patterns if op.get('pattern_type') == 'sequential')
        random_ops = len(access_patterns) - sequential_ops
        workload_analysis['sequential_random_ratio'] = sequential_ops / random_ops if random_ops > 0 else float('inf')
        
        # Analyze temporal patterns
        temporal_buckets = self._analyze_temporal_patterns(access_patterns)
        workload_analysis['temporal_patterns'] = temporal_buckets
        
        # Generate storage tier recommendations
        workload_analysis['recommended_tiers'] = self._recommend_storage_tiers(workload_analysis)
        
        return workload_analysis
    
    def _analyze_temporal_patterns(self, access_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal access patterns for data lifecycle management"""
        
        temporal_analysis = {
            'access_frequency_over_time': {},
            'data_aging_patterns': {},
            'seasonal_patterns': {},
            'hot_cold_transition_points': {}
        }
        
        # Group accesses by time periods
        time_buckets = {}
        for pattern in access_patterns:
            timestamp = pattern.get('timestamp', datetime.utcnow())
            hour_bucket = timestamp.hour
            
            if hour_bucket not in time_buckets:
                time_buckets[hour_bucket] = []
            time_buckets[hour_bucket].append(pattern)
        
        # Calculate access frequency per time bucket
        for hour, patterns in time_buckets.items():
            temporal_analysis['access_frequency_over_time'][hour] = len(patterns)
        
        # Identify data aging patterns using exponential decay model
        temporal_analysis['data_aging_patterns'] = self._model_data_aging(access_patterns)
        
        return temporal_analysis
    
    def _model_data_aging(self, access_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Model data aging using exponential decay"""
        
        # Group by data object
        data_objects = {}
        for pattern in access_patterns:
            obj_id = pattern.get('object_id', 'unknown')
            if obj_id not in data_objects:
                data_objects[obj_id] = []
            data_objects[obj_id].append(pattern)
        
        aging_model = {}
        
        for obj_id, obj_patterns in data_objects.items():
            # Sort by timestamp
            sorted_patterns = sorted(obj_patterns, key=lambda x: x.get('timestamp', datetime.utcnow()))
            
            if len(sorted_patterns) < 2:
                continue
            
            # Calculate access intervals
            intervals = []
            for i in range(1, len(sorted_patterns)):
                prev_time = sorted_patterns[i-1].get('timestamp', datetime.utcnow())
                curr_time = sorted_patterns[i].get('timestamp', datetime.utcnow())
                interval = (curr_time - prev_time).total_seconds()
                intervals.append(interval)
            
            if intervals:
                # Fit exponential decay model
                avg_interval = np.mean(intervals)
                decay_rate = 1.0 / avg_interval if avg_interval > 0 else 0.0
                
                aging_model[obj_id] = {
                    'decay_rate': decay_rate,
                    'average_access_interval_seconds': avg_interval,
                    'predicted_hot_duration_hours': self._predict_hot_duration(decay_rate),
                    'recommended_tier_transition': self._recommend_tier_transition(decay_rate)
                }
        
        return aging_model
    
    def _predict_hot_duration(self, decay_rate: float) -> float:
        """Predict how long data stays 'hot' based on decay rate"""
        if decay_rate <= 0:
            return 24.0  # Default 24 hours
        
        # Time until access probability drops to 10% of initial
        hot_duration_seconds = -np.log(0.1) / decay_rate
        return hot_duration_seconds / 3600  # Convert to hours
    
    def _recommend_tier_transition(self, decay_rate: float) -> Dict[str, float]:
        """Recommend when to transition between storage tiers"""
        
        hot_duration = self._predict_hot_duration(decay_rate)
        
        return {
            'hot_to_warm_hours': hot_duration,
            'warm_to_cold_hours': hot_duration * 24,  # 24x hot duration
            'cold_to_archive_hours': hot_duration * 168  # 168x hot duration (1 week multiplier)
        }
    
    def _recommend_storage_tiers(self, workload_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend optimal storage tier configuration"""
        
        recommendations = {
            'hot_tier': {'media_type': None, 'percentage': 0.0, 'reasoning': []},
            'warm_tier': {'media_type': None, 'percentage': 0.0, 'reasoning': []},
            'cold_tier': {'media_type': None, 'percentage': 0.0, 'reasoning': []},
            'archive_tier': {'media_type': None, 'percentage': 0.0, 'reasoning': []}
        }
        
        rw_ratio = workload_analysis['read_write_ratio']
        seq_ratio = workload_analysis['sequential_random_ratio']
        
        # Hot tier recommendations
        if rw_ratio > 3.0 and seq_ratio < 0.5:  # Read-heavy, random access
            recommendations['hot_tier']['media_type'] = StorageMediaType.NVME_SSD
            recommendations['hot_tier']['percentage'] = 20.0
            recommendations['hot_tier']['reasoning'].append('High random read workload requires NVMe performance')
        
        elif seq_ratio > 2.0:  # Sequential workload
            recommendations['hot_tier']['media_type'] = StorageMediaType.SATA_SSD
            recommendations['hot_tier']['percentage'] = 15.0
            recommendations['hot_tier']['reasoning'].append('Sequential workload suitable for SATA SSD')
        
        else:  # Mixed workload
            recommendations['hot_tier']['media_type'] = StorageMediaType.NVME_SSD
            recommendations['hot_tier']['percentage'] = 10.0
            recommendations['hot_tier']['reasoning'].append('Mixed workload benefits from NVMe for burst performance')
        
        # Warm tier (SATA SSD or fast HDD)
        recommendations['warm_tier']['media_type'] = StorageMediaType.SATA_SSD
        recommendations['warm_tier']['percentage'] = 30.0
        recommendations['warm_tier']['reasoning'].append('Balanced performance and cost for warm data')
        
        # Cold tier (HDD)
        recommendations['cold_tier']['media_type'] = StorageMediaType.SAS_HDD
        recommendations['cold_tier']['percentage'] = 40.0
        recommendations['cold_tier']['reasoning'].append('Cost-effective storage for infrequently accessed data')
        
        # Archive tier (Tape or cold HDD)
        recommendations['archive_tier']['media_type'] = StorageMediaType.TAPE
        recommendations['archive_tier']['percentage'] = 10.0
        recommendations['archive_tier']['reasoning'].append('Ultra-low-cost archival storage')
        
        return recommendations

class TieredStorageOptimizer:
    """Optimize storage tier placement and cost"""
    
    def __init__(self):
        self.storage_analyzer = StoragePerformanceAnalyzer()
        self.cost_models = {}
        self.performance_models = {}
        
    def optimize_storage_allocation(self, data_objects: List[Dict[str, Any]], 
                                  constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize storage allocation across tiers"""
        
        optimization_result = {
            'total_data_gb': sum(obj.get('size_gb', 0) for obj in data_objects),
            'tier_allocations': {},
            'cost_breakdown': {},
            'performance_metrics': {},
            'optimization_savings': {}
        }
        
        # Classify data objects by access patterns
        classified_objects = self._classify_data_objects(data_objects)
        
        # Calculate optimal tier distribution
        tier_distribution = self._calculate_optimal_distribution(classified_objects, constraints)
        optimization_result['tier_allocations'] = tier_distribution
        
        # Calculate costs for each tier
        cost_breakdown = self._calculate_tier_costs(tier_distribution)
        optimization_result['cost_breakdown'] = cost_breakdown
        
        # Estimate performance characteristics
        performance_metrics = self._estimate_performance_metrics(tier_distribution)
        optimization_result['performance_metrics'] = performance_metrics
        
        # Calculate potential savings vs single-tier solution
        savings = self._calculate_optimization_savings(tier_distribution, constraints)
        optimization_result['optimization_savings'] = savings
        
        return optimization_result
    
    def _classify_data_objects(self, data_objects: List[Dict[str, Any]]) -> Dict[str, List]:
        """Classify data objects into storage tiers based on access patterns"""
        
        classified = {
            StorageTier.HOT: [],
            StorageTier.WARM: [],
            StorageTier.COLD: [],
            StorageTier.ARCHIVE: []
        }
        
        for obj in data_objects:
            access_frequency = obj.get('access_frequency_per_day', 0)
            last_access_days = obj.get('last_access_days_ago', 0)
            size_gb = obj.get('size_gb', 0)
            
            # Classification logic based on access patterns
            if access_frequency > 10 and last_access_days < 1:
                tier = StorageTier.HOT
            elif access_frequency > 1 and last_access_days < 7:
                tier = StorageTier.WARM
            elif access_frequency > 0.1 and last_access_days < 30:
                tier = StorageTier.COLD
            else:
                tier = StorageTier.ARCHIVE
            
            classified[tier].append({
                'object_id': obj.get('object_id'),
                'size_gb': size_gb,
                'access_frequency': access_frequency,
                'last_access_days': last_access_days,
                'classification_confidence': self._calculate_classification_confidence(obj)
            })
        
        return classified
    
    def _calculate_classification_confidence(self, obj: Dict[str, Any]) -> float:
        """Calculate confidence in tier classification"""
        
        # Factors affecting classification confidence
        access_history_length = obj.get('access_history_days', 0)
        access_pattern_consistency = obj.get('access_pattern_consistency', 0.5)
        data_age_days = obj.get('data_age_days', 0)
        
        # Base confidence from access history length
        history_confidence = min(1.0, access_history_length / 90.0)  # 90 days for full confidence
        
        # Pattern consistency factor
        pattern_confidence = access_pattern_consistency
        
        # Age-based confidence (newer data harder to classify)
        age_confidence = min(1.0, data_age_days / 30.0)  # 30 days for full confidence
        
        # Combined confidence score
        overall_confidence = (history_confidence * 0.4 + 
                            pattern_confidence * 0.4 + 
                            age_confidence * 0.2)
        
        return overall_confidence
    
    def _calculate_optimal_distribution(self, classified_objects: Dict[StorageTier, List],
                                      constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal storage distribution across tiers"""
        
        distribution = {}
        storage_profiles = self.storage_analyzer.storage_profiles
        
        for tier, objects in classified_objects.items():
            if not objects:
                continue
                
            total_size_gb = sum(obj['size_gb'] for obj in objects)
            object_count = len(objects)
            
            # Select optimal storage media for tier
            if tier == StorageTier.HOT:
                recommended_media = StorageMediaType.NVME_SSD
            elif tier == StorageTier.WARM:
                recommended_media = StorageMediaType.SATA_SSD
            elif tier == StorageTier.COLD:
                recommended_media = StorageMediaType.SAS_HDD
            else:  # ARCHIVE
                recommended_media = StorageMediaType.TAPE
            
            media_profile = storage_profiles[recommended_media]
            
            distribution[tier.value] = {
                'total_size_gb': total_size_gb,
                'object_count': object_count,
                'recommended_media': recommended_media.value,
                'monthly_cost_usd': total_size_gb * media_profile.cost_per_gb_per_month,
                'performance_characteristics': {
                    'random_read_iops': media_profile.random_read_iops,
                    'sequential_read_mbps': media_profile.sequential_read_mbps,
                    'latency_us': media_profile.latency_us
                },
                'capacity_planning': {
                    'drives_required': math.ceil(total_size_gb / (media_profile.density_gb_per_u / 24)),  # Assume 24 drives per U
                    'rack_units_required': math.ceil(total_size_gb / media_profile.density_gb_per_u),
                    'power_consumption_watts': (total_size_gb / media_profile.density_gb_per_u) * media_profile.power_consumption_watts
                }
            }
        
        return distribution
    
    def _calculate_tier_costs(self, tier_distribution: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate detailed cost breakdown for storage tiers"""
        
        cost_breakdown = {
            'monthly_storage_costs': {},
            'annual_storage_costs': {},
            'infrastructure_costs': {},
            'operational_costs': {},
            'total_monthly_cost': 0.0,
            'total_annual_cost': 0.0
        }
        
        for tier_name, tier_info in tier_distribution.items():
            monthly_storage_cost = tier_info['monthly_cost_usd']
            
            # Infrastructure costs (one-time, amortized monthly)
            drives_required = tier_info['capacity_planning']['drives_required']
            infrastructure_cost_monthly = drives_required * 500 / 36  # $500 per drive, 3-year amortization
            
            # Operational costs (power, cooling, management)
            power_cost_monthly = (tier_info['capacity_planning']['power_consumption_watts'] * 
                                24 * 30 * 0.10) / 1000  # $0.10/kWh
            
            cooling_cost_monthly = power_cost_monthly * 1.3  # 30% additional for cooling
            management_cost_monthly = tier_info['object_count'] * 0.01  # $0.01 per object per month
            
            total_operational_monthly = power_cost_monthly + cooling_cost_monthly + management_cost_monthly
            
            tier_monthly_total = monthly_storage_cost + infrastructure_cost_monthly + total_operational_monthly
            
            cost_breakdown['monthly_storage_costs'][tier_name] = monthly_storage_cost
            cost_breakdown['annual_storage_costs'][tier_name] = monthly_storage_cost * 12
            cost_breakdown['infrastructure_costs'][tier_name] = infrastructure_cost_monthly
            cost_breakdown['operational_costs'][tier_name] = total_operational_monthly
            
            cost_breakdown['total_monthly_cost'] += tier_monthly_total
        
        cost_breakdown['total_annual_cost'] = cost_breakdown['total_monthly_cost'] * 12
        
        return cost_breakdown
    
    def _estimate_performance_metrics(self, tier_distribution: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate overall system performance metrics"""
        
        performance_metrics = {
            'weighted_average_latency_us': 0.0,
            'total_random_read_iops': 0,
            'total_sequential_read_mbps': 0,
            'tier_performance_breakdown': {}
        }
        
        total_size_gb = sum(tier_info['total_size_gb'] for tier_info in tier_distribution.values())
        
        for tier_name, tier_info in tier_distribution.items():
            tier_size_gb = tier_info['total_size_gb']
            tier_weight = tier_size_gb / total_size_gb if total_size_gb > 0 else 0
            
            tier_perf = tier_info['performance_characteristics']
            
            # Weighted average latency
            performance_metrics['weighted_average_latency_us'] += (
                tier_perf['latency_us'] * tier_weight
            )
            
            # Sum IOPS and throughput (assuming parallel access)
            performance_metrics['total_random_read_iops'] += tier_perf['random_read_iops']
            performance_metrics['total_sequential_read_mbps'] += tier_perf['sequential_read_mbps']
            
            performance_metrics['tier_performance_breakdown'][tier_name] = {
                'size_percentage': tier_weight * 100,
                'latency_contribution_us': tier_perf['latency_us'] * tier_weight,
                'iops_contribution': tier_perf['random_read_iops'],
                'throughput_contribution_mbps': tier_perf['sequential_read_mbps']
            }
        
        return performance_metrics
    
    def _calculate_optimization_savings(self, tier_distribution: Dict[str, Any],
                                      constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate savings compared to single-tier solutions"""
        
        total_size_gb = sum(tier_info['total_size_gb'] for tier_info in tier_distribution.values())
        optimized_monthly_cost = sum(tier_info['monthly_cost_usd'] for tier_info in tier_distribution.values())
        
        # Calculate costs for single-tier alternatives
        storage_profiles = self.storage_analyzer.storage_profiles
        
        single_tier_costs = {}
        for media_type, profile in storage_profiles.items():
            single_tier_costs[media_type.value] = total_size_gb * profile.cost_per_gb_per_month
        
        # Find best single-tier alternative that meets performance requirements
        min_latency_req = constraints.get('max_latency_us', float('inf'))
        min_iops_req = constraints.get('min_iops', 0)
        
        viable_single_tier_options = []
        for media_type, profile in storage_profiles.items():
            if (profile.latency_us <= min_latency_req and 
                profile.random_read_iops >= min_iops_req):
                
                viable_single_tier_options.append({
                    'media_type': media_type.value,
                    'monthly_cost': single_tier_costs[media_type.value],
                    'latency_us': profile.latency_us,
                    'iops': profile.random_read_iops
                })
        
        if not viable_single_tier_options:
            return {'error': 'No single-tier solution meets performance requirements'}
        
        # Find most cost-effective single-tier solution
        best_single_tier = min(viable_single_tier_options, key=lambda x: x['monthly_cost'])
        
        savings_monthly = best_single_tier['monthly_cost'] - optimized_monthly_cost
        savings_annual = savings_monthly * 12
        savings_percentage = (savings_monthly / best_single_tier['monthly_cost']) * 100
        
        return {
            'optimized_monthly_cost': optimized_monthly_cost,
            'best_single_tier_cost': best_single_tier['monthly_cost'],
            'best_single_tier_media': best_single_tier['media_type'],
            'monthly_savings': savings_monthly,
            'annual_savings': savings_annual,
            'savings_percentage': savings_percentage,
            'roi_months': abs(savings_monthly / optimized_monthly_cost) if savings_monthly > 0 else float('inf')
        }
```

This completes Part 1 of Day 4, providing deep theoretical foundations of tiered storage architecture, performance modeling, and cost optimization frameworks for ML workloads.