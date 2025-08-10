# Day 6.2: A/B Testing & Canary Deployments

## 🧪 Model Serving & Production Inference - Part 2

**Focus**: Experimentation Frameworks, Progressive Deployment, Statistical Significance  
**Duration**: 2-3 hours  
**Level**: Advanced  

---

## 🎯 Learning Objectives

- Master A/B testing frameworks for ML systems and understand statistical foundations
- Learn progressive deployment strategies (canary, blue-green, rolling updates)
- Understand experimental design principles for model evaluation
- Analyze risk mitigation and rollback strategies for production ML systems

---

## 📊 A/B Testing Theoretical Framework

### **Experimental Design for ML Systems**

A/B testing in ML systems presents unique challenges compared to traditional web applications due to model complexity, feature dependencies, and multi-dimensional success metrics.

**Statistical Foundation:**
```
Hypothesis Testing Framework:
H₀: μ_treatment = μ_control (no difference between models)
H₁: μ_treatment ≠ μ_control (significant difference exists)

Test Statistics:
t = (x̄_treatment - x̄_control) / √(s²_pooled × (1/n_treatment + 1/n_control))

Where:
s²_pooled = ((n₁-1)s₁² + (n₂-1)s₂²) / (n₁ + n₂ - 2)

Power Analysis:
Power = P(reject H₀ | H₁ is true) = 1 - β
Required sample size: n ≈ 2(z_α/2 + z_β)²σ² / δ²

Where δ = minimum detectable effect size
```

**ML-Specific Considerations:**
```
Multi-Metric Optimization:
Primary metrics: Accuracy, precision, recall, F1-score
Secondary metrics: Latency, throughput, resource utilization
Business metrics: Revenue, user engagement, satisfaction

Statistical Challenges:
1. Multiple testing problem: P(Type I error) increases with # tests
2. Correlated metrics: Not independent, affects significance testing
3. Non-normal distributions: Many ML metrics are not normally distributed
4. Temporal dependencies: Model performance may change over time
5. User heterogeneity: Different user segments may respond differently

Bonferroni Correction:
α_adjusted = α_family / number_of_tests
More conservative: Higher threshold for significance
```

### **Experimental Design Patterns**

**Traffic Splitting Strategies:**

**Random Assignment:**
```
Simple Random Assignment:
- Each request independently assigned to treatment/control
- Probability p for treatment, (1-p) for control
- Advantages: Simple implementation, unbiased allocation
- Disadvantages: May create user experience inconsistency

Implementation:
hash_value = hash(request_id) % 100
if hash_value < treatment_percentage:
    route_to_treatment()
else:
    route_to_control()
```

**User-Level Assignment:**
```
Consistent User Experience:
- All requests from same user go to same variant
- Prevents within-user treatment contamination
- Enables user-journey analysis and retention metrics

Stratified Assignment:
- Ensure balanced representation across user segments
- Control for confounding variables (geography, device type, etc.)
- Maintain statistical power while reducing bias

Implementation:
user_bucket = hash(user_id) % 100
if user_bucket < treatment_percentage:
    assign_to_treatment(user_id)
```

**Contextual Bandits for Dynamic Assignment:**
```
Multi-Armed Bandit Framework:
- Dynamically adjust traffic allocation based on performance
- Explore-exploit trade-off optimization
- Minimize regret while learning optimal allocation

Thompson Sampling:
For each variant i:
  sample θᵢ from posterior distribution
  assign request to variant with highest θᵢ

UCB (Upper Confidence Bound):
UCB_i = μ̂_i + √(2 ln(t) / n_i)
where μ̂_i = estimated reward, n_i = arm pulls, t = total pulls
```

---

## 🚀 Progressive Deployment Strategies

### **Canary Deployment Theory**

Canary deployments gradually roll out new model versions to increasing percentages of traffic, enabling early detection of issues with minimal user impact.

**Mathematical Framework:**
```
Risk-Controlled Rollout:
Risk(t) = P(failure) × Impact(t) × Traffic_percentage(t)

Optimal Rollout Schedule:
traffic_percentage(t) = f(confidence_level(t), risk_tolerance)

Where confidence_level(t) depends on:
- Statistical significance of metrics
- Volume of data collected
- Stability of performance indicators

Rollout Stages:
Stage 1: 1% traffic, 24-48 hours observation
Stage 2: 5% traffic, monitor for 24 hours  
Stage 3: 25% traffic, monitor for 12 hours
Stage 4: 100% traffic, full deployment

Criteria for progression:
- No statistically significant degradation
- Error rates within acceptable bounds
- Latency metrics meet SLA requirements
```

**Automated Rollout Control:**
```
Decision Algorithm:
def should_proceed_rollout(metrics, thresholds):
    for metric, value in metrics.items():
        if value < thresholds[metric]['min'] or value > thresholds[metric]['max']:
            return False, f"Metric {metric} outside acceptable range"
    
    if statistical_significance_achieved(metrics):
        if all_metrics_improving(metrics):
            return True, "Proceed to next stage"
        else:
            return False, "Metrics not improving significantly"
    
    return None, "Insufficient data for decision"

Rollback Triggers:
- Error rate > 2× baseline
- Latency P99 > 1.5× baseline  
- Business metric decline > 5%
- Manual intervention signal
```

### **Blue-Green Deployment**

Blue-green deployment maintains two identical production environments, enabling instant switching between versions with zero downtime.

**Infrastructure Requirements:**
```
Resource Allocation:
- 2× infrastructure cost during deployment
- Load balancer with instant switching capability
- Shared data stores with backward compatibility
- Monitoring systems for both environments

Switching Logic:
Blue Environment: Current production traffic (100%)
Green Environment: New version deployment and testing (0%)

Switch Process:
1. Deploy new version to green environment
2. Run comprehensive tests on green
3. Switch load balancer: 100% traffic to green
4. Monitor green environment performance
5. Keep blue as backup for quick rollback

Rollback Capability:
Instant rollback: Switch load balancer back to blue
Recovery time: < 30 seconds typical
```

**Database Migration Considerations:**
```
Schema Evolution Strategies:
1. Backward-compatible changes only during blue-green
2. Multi-phase migration for breaking changes
3. Feature flags for gradual data model transitions

Migration Patterns:
Phase 1: Add new columns/tables (both versions work)
Phase 2: Migrate data in background
Phase 3: Update application to use new schema
Phase 4: Remove old columns/tables in next release

Data Consistency:
- Read replicas for blue-green isolation
- Transaction log replay for synchronization
- Conflict resolution for concurrent updates
```

---

## 📈 Statistical Analysis and Metrics

### **Power Analysis and Sample Size Calculation**

**Sample Size Determination:**
```
For Binary Metrics (e.g., click-through rate):
n = (z_α/2 + z_β)² × (p₁(1-p₁) + p₂(1-p₂)) / (p₁ - p₂)²

For Continuous Metrics (e.g., revenue per user):
n = 2(z_α/2 + z_β)² × σ² / δ²

Where:
z_α/2 = critical value for Type I error (e.g., 1.96 for α=0.05)
z_β = critical value for Type II error (e.g., 0.84 for β=0.2, power=0.8)
δ = minimum detectable effect size
σ = standard deviation of metric

Practical Considerations:
- Effect size estimation from historical data
- Multiple testing correction (Bonferroni, FDR)
- Stratification for reduced variance
- Cluster randomization effects
```

**Sequential Testing and Early Stopping:**
```
Sequential Probability Ratio Test (SPRT):
Continue testing while: B < Λₙ < A
Stop and accept H₁ if: Λₙ ≥ A  
Stop and accept H₀ if: Λₙ ≤ B

Where:
Λₙ = likelihood ratio after n observations
A = (1-β)/α (acceptance boundary)
B = β/(1-α) (rejection boundary)

Advantages:
- Reduced sample size on average
- Early detection of strong effects
- Ethical considerations (stop harmful treatments early)

Group Sequential Design:
- Pre-planned interim analyses
- Spend alpha at each analysis
- Lan-DeMets spending functions for flexible timing
```

### **Multi-Metric Evaluation**

**Overall Evaluation Criteria (OEC):**
```
Composite Metric Design:
OEC = Σᵢ wᵢ × normalized_metric_i

Where:
wᵢ = weight for metric i (Σwᵢ = 1)
normalized_metric_i = (metric_i - baseline_i) / baseline_i

Weight Determination:
- Business impact analysis
- Stakeholder alignment on priorities
- Historical correlation analysis
- Sensitivity testing for weight choices

Example OEC:
OEC = 0.4 × Δaccuracy + 0.3 × Δrevenue + 0.2 × Δlatency + 0.1 × Δcost
```

**Guardrail Metrics:**
```
Quality Gates:
Primary Success Metrics: What we want to improve
Guardrail Metrics: What we cannot afford to degrade

Examples:
Primary: Model accuracy, user engagement
Guardrails: System latency, error rates, cost per request

Guardrail Thresholds:
Hard Limits: Automatic rollback triggers
Soft Limits: Warning alerts, manual review required
Trend Monitoring: Gradual degradation detection

Statistical Testing:
- Non-inferiority tests for guardrails
- TOST (Two One-Sided Tests) procedure
- Confidence intervals for degradation bounds
```

---

## 🔄 Feature Flags and Configuration Management

### **Feature Flag Architecture**

**Dynamic Model Selection:**
```
Flag-Based Model Routing:
def select_model(user_context, feature_flags):
    if feature_flags.get('new_model_enabled', False):
        if feature_flags.get('new_model_percentage', 0) > random.random() * 100:
            return 'model_v2'
    return 'model_v1'

Gradual Rollout Control:
feature_flags = {
    'new_model_enabled': True,
    'new_model_percentage': 25,  # 25% of users
    'user_segments': ['premium', 'beta_users'],
    'geographic_regions': ['US', 'EU']
}

Advanced Targeting:
- User attributes (demographics, behavior)
- Request characteristics (device, location)  
- Temporal conditions (time of day, day of week)
- Business rules (subscription tier, experimental group)
```

**Configuration Management:**
```
Hierarchical Configuration:
Global Config → Regional Config → Local Config → User Config

Priority Resolution:
user_config = merge_configs([
    global_config,
    regional_configs[user.region],
    local_configs[user.cluster],
    user_specific_overrides
])

Real-Time Updates:
- Configuration push to all serving instances
- Graceful handling of configuration changes
- Rollback capability for configuration errors
- Audit trails for configuration changes
```

### **Experimentation Platform Architecture**

**Experiment Management System:**
```
Core Components:
1. Experiment Definition Service
   - Hypothesis specification
   - Success metrics definition
   - Statistical power analysis
   - Traffic allocation rules

2. Assignment Service  
   - Consistent user bucketing
   - Traffic splitting logic
   - Mutual exclusion handling
   - Assignment logging

3. Analysis Service
   - Real-time metric computation
   - Statistical significance testing
   - Confidence interval estimation
   - Report generation

4. Decision Service
   - Automated rollout decisions
   - Rollback trigger detection
   - Human-in-the-loop approvals
   - Risk assessment
```

**Experimentation Best Practices:**
```
Experiment Design Principles:
1. Single Variable Testing: Change one thing at a time
2. Sufficient Duration: Account for weekly/monthly patterns  
3. Representative Sample: Avoid selection bias
4. Pre/Post Analysis: Compare with historical baselines
5. Segment Analysis: Different effects across user groups

Common Pitfalls:
- Simpson's Paradox: Segment effects opposite to overall
- Novelty Effect: Initial improvement that fades
- Survivorship Bias: Excluding users who stopped using system
- Interference: Treatment affects control group behavior
- Data Quality Issues: Logging errors, missing data
```

---

## 🛡️ Risk Management and Monitoring

### **Automated Monitoring and Alerting**

**Real-Time Quality Monitoring:**
```
Model Performance Metrics:
- Prediction accuracy on validation traffic
- Confidence score distributions
- Feature drift detection
- Anomaly detection in predictions

System Health Metrics:
- Request latency (P50, P95, P99)
- Error rates by error type
- Resource utilization (CPU, memory, GPU)
- Dependency health (database, cache, APIs)

Business Impact Metrics:  
- Revenue impact estimation
- User experience metrics
- Operational cost changes
- Customer satisfaction indicators
```

**Anomaly Detection Algorithms:**
```
Statistical Methods:
- Control charts (3-sigma rules)
- CUSUM (Cumulative Sum) control charts
- EWMA (Exponentially Weighted Moving Average)

Machine Learning Approaches:
- Isolation Forest for multivariate anomalies
- Autoencoders for pattern recognition
- Time series forecasting (ARIMA, Prophet)
- Change point detection algorithms

Implementation:
def detect_anomaly(metric_value, historical_data, threshold=3):
    mean = np.mean(historical_data)
    std = np.std(historical_data)
    z_score = abs(metric_value - mean) / std
    return z_score > threshold

Alert Severity:
- Critical: Immediate rollback required
- Warning: Investigation needed
- Info: Trend monitoring
```

### **Rollback Strategies**

**Automated Rollback Triggers:**
```
Circuit Breaker Pattern:
States: Closed → Open → Half-Open → Closed

Closed: Normal operation, monitor failure rate
Open: Reject requests, return cached/default responses  
Half-Open: Allow limited traffic to test recovery

Rollback Conditions:
error_rate > 5% AND duration > 5_minutes → ROLLBACK
latency_p99 > 2x_baseline AND traffic > 1000_rps → ROLLBACK  
business_metric_drop > 10% AND confidence > 95% → ROLLBACK

Rollback Execution:
1. Stop traffic to new version (immediate)
2. Route all traffic to previous version
3. Preserve logs and data for analysis
4. Notify stakeholders and engineering teams
5. Conduct post-mortem analysis
```

**Graceful Degradation:**
```
Fallback Mechanisms:
1. Previous model version
2. Simpler baseline model
3. Cached predictions
4. Rule-based heuristics
5. Human-generated defaults

Degradation Strategies:
- Feature importance ranking: Disable less critical features
- Model ensemble: Remove poorly performing models
- Sampling: Reduce traffic to problematic components
- Rate limiting: Throttle requests during issues

Implementation:
def get_prediction(request):
    try:
        return primary_model.predict(request)
    except HighLatencyException:
        return fast_model.predict(request)
    except ModelUnavailableException:
        return cached_prediction.get(request.key)
    except Exception:
        return default_prediction(request)
```

This comprehensive framework for A/B testing and progressive deployments provides the theoretical foundation for safely deploying and evaluating ML models in production. The key insight is that successful ML experimentation requires careful statistical design, robust infrastructure, and comprehensive monitoring to balance innovation with reliability.