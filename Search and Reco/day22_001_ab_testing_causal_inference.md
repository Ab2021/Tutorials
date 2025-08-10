# Day 22: A/B Testing and Causal Inference for Search and Recommendation Systems

## Learning Objectives
By the end of this session, students will be able to:
- Design and implement robust A/B testing frameworks specifically for search and recommendation systems
- Understand the principles of causal inference and their application to recommendation systems
- Distinguish between A/B testing and multi-armed bandit approaches for different scenarios
- Apply uplift modeling and causal graphs to understand treatment effects
- Implement proper statistical analysis and interpretation of experimental results
- Address common challenges in experimentation for search and recommendation systems

## 1. A/B Testing Fundamentals for Search and Recommendation Systems

### 1.1 Experimental Design Principles

**The Gold Standard for Causal Inference**

**Why A/B Testing Matters**
A/B testing provides the most reliable method for establishing causality in system improvements:
- **Causal Inference**: Establishes causal relationships between changes and outcomes
- **Unbiased Estimates**: Randomization eliminates selection bias
- **Statistical Rigor**: Provides statistical confidence in results
- **Business Decision Making**: Enables data-driven product decisions

**Key Business Questions:**
- Which version of my recommender performs better and why?
- What's the true impact of a new ranking algorithm on user behavior?
- How do we isolate the effect of one change from other confounding factors?
- What's the optimal allocation of traffic between experimental variants?

**Unique Challenges in Search/Recommendation Testing**

**User Learning Effects**
- **Algorithm Adaptation**: Users adapt to new recommendation styles over time
- **Novelty Effects**: Initial reactions may differ from long-term behavior
- **Habit Formation**: Changes in user behavior patterns due to system changes
- **Spillover Effects**: Learning from one session affects behavior in subsequent sessions

**Network Effects**
- **Social Influence**: User behavior influenced by friends' experiences
- **Content Ecosystem**: Changes affecting content creators impact content consumers
- **Marketplace Dynamics**: Two-sided markets where changes affect both sides
- **Viral Mechanisms**: Content virality affecting experimental outcomes

### 1.2 Experimental Units and Randomization

**Choosing the Right Unit of Randomization**

**User-Level Randomization**
Most common approach for recommendation systems:
- **Advantages**: Consistent user experience, clear causal interpretation
- **Challenges**: Network effects, user identification across devices
- **Implementation**: Hash user IDs to determine assignment
- **Considerations**: Handle logged-out users, account for user lifecycle

**Session-Level Randomization**
Alternative approach for certain scenarios:
- **Advantages**: Higher statistical power, handles anonymous users
- **Disadvantages**: Inconsistent user experience
- **Use Cases**: Testing backend algorithms with minimal user-facing changes
- **Implementation**: Hash session IDs for assignment

**Query/Request-Level Randomization**
For search-specific experiments:
- **Advantages**: Maximum statistical power
- **Disadvantages**: No consistency, potential user confusion
- **Use Cases**: Testing ranking algorithms, query understanding
- **Implementation**: Hash query and user combination

**Cluster Randomization**
For handling network effects:
- **Geographic Clusters**: Randomize by city, region, or country
- **Social Clusters**: Randomize by social groups or communities
- **Temporal Clusters**: Randomize by time periods
- **Trade-offs**: Reduced statistical power but better isolation of effects

### 1.3 Statistical Framework

**Hypothesis Testing for Recommendations**

**Primary Hypotheses**
- **Null Hypothesis (H₀)**: No difference between treatment and control
- **Alternative Hypothesis (H₁)**: Treatment has meaningful impact
- **Effect Size**: Minimum meaningful difference worth detecting
- **Statistical Power**: Probability of detecting true effect (typically 80%)

**Sample Size Calculations**
```python
import scipy.stats as stats
import numpy as np

def calculate_sample_size(baseline_rate, minimum_effect, alpha=0.05, power=0.8):
    """
    Calculate required sample size for A/B test
    
    Args:
        baseline_rate: Current conversion/click rate
        minimum_effect: Minimum effect size to detect (e.g., 0.05 for 5% relative lift)
        alpha: Type I error rate (false positive)
        power: Statistical power (1 - Type II error)
    """
    # Effect size (Cohen's h for proportions)
    p1 = baseline_rate
    p2 = baseline_rate * (1 + minimum_effect)
    
    h = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))
    
    # Required sample size per group
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    n = ((z_alpha + z_beta) / h) ** 2
    
    return int(np.ceil(n))

# Example: CTR test
baseline_ctr = 0.05
minimum_lift = 0.10  # 10% relative improvement
sample_size = calculate_sample_size(baseline_ctr, minimum_lift)
print(f"Required sample size per group: {sample_size:,}")
```

**Multiple Testing Corrections**
When testing multiple metrics or segments:
- **Bonferroni Correction**: Adjust α by number of tests
- **False Discovery Rate (FDR)**: Control expected proportion of false discoveries
- **Hierarchical Testing**: Test primary metrics first, then secondary
- **Pre-specification**: Define primary metrics before running experiment

## 2. Advanced Experimental Designs

### 2.1 Multi-Armed Bandit Testing

**Balancing Exploration and Exploitation**

**When to Use Bandits vs A/B Tests**

**A/B Testing Advantages:**
- **Statistical Rigor**: Clear statistical interpretation
- **Unbiased Estimates**: Fixed allocation provides unbiased effect estimates
- **Regulatory Compliance**: Often required for regulated industries
- **Simple Implementation**: Easier to implement and understand

**Bandit Testing Advantages:**
- **Opportunity Cost**: Minimize revenue loss during testing
- **Adaptive Allocation**: Dynamically allocate traffic to better variants
- **Continuous Optimization**: No fixed testing period
- **Early Stopping**: Can stop when confident in results

**Implementation Strategies**

**Thompson Sampling for Recommendations**
```python
import numpy as np
from scipy import stats

class ThompsonSamplingRecommender:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        # Beta distribution parameters (alpha, beta)
        self.successes = np.ones(n_arms)  # Prior successes
        self.failures = np.ones(n_arms)   # Prior failures
        
    def select_arm(self):
        # Sample from posterior Beta distributions
        samples = [
            np.random.beta(self.successes[i], self.failures[i]) 
            for i in range(self.n_arms)
        ]
        return np.argmax(samples)
    
    def update(self, arm, reward):
        if reward == 1:
            self.successes[arm] += 1
        else:
            self.failures[arm] += 1
    
    def get_posterior_stats(self):
        means = self.successes / (self.successes + self.failures)
        return means

# Usage example
bandit = ThompsonSamplingRecommender(n_arms=3)
# In production: select_arm() for each recommendation request
# Update with user feedback (click/no-click)
```

**Contextual Bandits**
For personalized recommendations:
- **LinUCB**: Linear upper confidence bound with user/item features
- **Neural Bandits**: Deep learning models with uncertainty estimation
- **Hybrid Approaches**: Combine global and personalized models

### 2.2 Factorial Experiments

**Testing Multiple Factors Simultaneously**

**2^k Factorial Design**
Test k factors, each with 2 levels:
- **Main Effects**: Individual effect of each factor
- **Interaction Effects**: How factors interact with each other
- **Efficiency**: More efficient than sequential testing
- **Complexity**: Exponential growth in number of conditions

**Example: Recommendation System Factors**
- **Factor A**: Ranking Algorithm (Collaborative Filtering vs Deep Learning)
- **Factor B**: UI Layout (Grid vs List view)
- **Factor C**: Personalization Level (High vs Low)
- **Combinations**: 2³ = 8 experimental conditions

**Analysis of Factorial Experiments**
```python
import pandas as pd
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols

def analyze_factorial_experiment(data):
    """
    Analyze factorial experiment results
    
    Args:
        data: DataFrame with columns ['algorithm', 'ui_layout', 'personalization', 'ctr']
    """
    # Fit full factorial model
    model = ols('ctr ~ algorithm * ui_layout * personalization', data=data).fit()
    
    # ANOVA table
    anova_results = anova_lm(model, typ=2)
    
    # Effect sizes
    main_effects = {
        'algorithm': model.params['algorithm[T.deep_learning]'],
        'ui_layout': model.params['ui_layout[T.list]'],
        'personalization': model.params['personalization[T.low]']
    }
    
    return anova_results, main_effects, model

# Interpretation
def interpret_interactions(model):
    interaction_terms = [param for param in model.params.index if ':' in param]
    significant_interactions = []
    
    for term in interaction_terms:
        p_value = model.pvalues[term]
        if p_value < 0.05:
            significant_interactions.append((term, model.params[term], p_value))
    
    return significant_interactions
```

### 2.3 Sequential and Adaptive Testing

**Early Stopping and Sequential Analysis**

**Group Sequential Design**
Pre-planned interim analyses with stopping rules:
- **Efficacy Stopping**: Stop early if treatment clearly superior
- **Futility Stopping**: Stop if unlikely to find significant effect
- **Alpha Spending**: Distribute Type I error across interim looks
- **O'Brien-Fleming**: Conservative early stopping, liberal later stopping

**Always-Valid P-Values**
P-values that remain valid regardless of when calculated:
- **Confidence Sequences**: Time-uniform confidence intervals
- **Sequential Probability Ratio Test**: Continuous monitoring
- **Implementation**: Tools like sequential analysis packages

**Adaptive Sample Size**
Adjust sample size based on interim results:
- **Variance Re-estimation**: Update based on observed variance
- **Effect Size Re-estimation**: Adjust for observed effect size
- **Conditional Power**: Probability of success given current data

## 3. Causal Inference in Recommendation Systems

### 3.1 Causal Frameworks

**Potential Outcomes Framework**

**Fundamental Problem of Causal Inference**
- **Potential Outcomes**: Y₁(i) if treated, Y₀(i) if control for individual i
- **Individual Treatment Effect**: ITE(i) = Y₁(i) - Y₀(i)
- **Average Treatment Effect**: ATE = E[Y₁ - Y₀]
- **Missing Data Problem**: Never observe both Y₁(i) and Y₀(i) for same individual

**Assumptions for Causal Inference**
- **SUTVA**: Stable Unit Treatment Value Assumption (no interference)
- **Unconfoundedness**: Treatment assignment independent of potential outcomes
- **Overlap**: Positive probability of treatment for all units
- **Consistency**: Observed outcome equals potential outcome under assigned treatment

**Directed Acyclic Graphs (DAGs)**

**Causal Graph Components**
- **Nodes**: Variables in the system
- **Directed Edges**: Causal relationships
- **Confounders**: Variables affecting both treatment and outcome
- **Mediators**: Variables on causal path from treatment to outcome
- **Colliders**: Variables affected by both treatment and outcome

```python
# Example: Recommendation System DAG
import networkx as nx
import matplotlib.pyplot as plt

def create_recommendation_dag():
    G = nx.DiGraph()
    
    # Add nodes
    nodes = [
        'User_Demographics', 'Past_Behavior', 'Recommendation_Algorithm',
        'Recommended_Items', 'User_Engagement', 'Purchase_Decision'
    ]
    G.add_nodes_from(nodes)
    
    # Add edges (causal relationships)
    edges = [
        ('User_Demographics', 'Past_Behavior'),
        ('User_Demographics', 'User_Engagement'),
        ('Past_Behavior', 'Recommendation_Algorithm'),
        ('Recommendation_Algorithm', 'Recommended_Items'),
        ('Recommended_Items', 'User_Engagement'),
        ('User_Engagement', 'Purchase_Decision'),
        ('User_Demographics', 'Purchase_Decision')  # Direct effect
    ]
    G.add_edges_from(edges)
    
    return G

# Identify confounders and adjustment sets
def find_adjustment_set(dag, treatment, outcome):
    """Find minimal adjustment set for identifying causal effect"""
    # This is a simplified version - real implementation would use
    # algorithms like PC algorithm or backdoor criterion
    pass
```

### 3.2 Uplift Modeling

**Heterogeneous Treatment Effects**

**Why Uplift Matters**
Not all users respond the same way to treatments:
- **Responders**: Users who benefit from treatment
- **Non-responders**: Users unaffected by treatment
- **Sleeping Dogs**: Users harmed by treatment (avoid treating)
- **Business Value**: Target treatment to users most likely to benefit

**Uplift Modeling Approaches**

**Two-Model Approach**
Train separate models for treatment and control groups:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class TwoModelUplift:
    def __init__(self):
        self.treatment_model = RandomForestClassifier()
        self.control_model = RandomForestClassifier()
    
    def fit(self, X, y, treatment):
        # Split data by treatment group
        X_treatment = X[treatment == 1]
        y_treatment = y[treatment == 1]
        X_control = X[treatment == 0]
        y_control = y[treatment == 0]
        
        # Train separate models
        self.treatment_model.fit(X_treatment, y_treatment)
        self.control_model.fit(X_control, y_control)
    
    def predict_uplift(self, X):
        # Predict probability for each group
        p_treatment = self.treatment_model.predict_proba(X)[:, 1]
        p_control = self.control_model.predict_proba(X)[:, 1]
        
        # Uplift = difference in probabilities
        uplift = p_treatment - p_control
        return uplift

# Usage
uplift_model = TwoModelUplift()
uplift_model.fit(X_train, y_train, treatment_train)
predicted_uplift = uplift_model.predict_uplift(X_test)
```

**Meta-Learners**
Advanced approaches for heterogeneous treatment effects:
- **S-Learner**: Single model with treatment as feature
- **T-Learner**: Two separate models (same as above)
- **X-Learner**: Cross-validation between treatment and control models
- **R-Learner**: Residual-based approach using Robinson decomposition

**Causal Trees and Forests**
Tree-based methods for finding subgroups with different treatment effects:
- **Honest Estimation**: Separate samples for splitting and estimation
- **Causal Tree**: Recursive partitioning based on treatment effect heterogeneity
- **Causal Forest**: Ensemble of causal trees with bootstrap sampling

### 3.3 Instrumental Variables

**Handling Unobserved Confounding**

**When to Use Instrumental Variables**
- **Unobserved Confounders**: Variables affecting both treatment and outcome
- **Selection Bias**: Non-random treatment assignment
- **Compliance Issues**: Not all assigned units receive treatment
- **Endogeneity**: Treatment choice correlated with unobserved factors

**Instrumental Variable Requirements**
- **Relevance**: Instrument correlated with treatment
- **Exclusion Restriction**: Instrument affects outcome only through treatment
- **Unconfoundedness**: Instrument uncorrelated with unobserved confounders

**Example: Recommendation System IV**
```python
from sklearn.linear_model import LinearRegression
import numpy as np

def instrumental_variable_estimation(Z, X, Y):
    """
    Two-Stage Least Squares (2SLS) estimation
    
    Args:
        Z: Instrument (e.g., random recommendation assignment)
        X: Treatment (e.g., actual recommendation shown)
        Y: Outcome (e.g., user engagement)
    """
    # First stage: regress treatment on instrument
    first_stage = LinearRegression()
    first_stage.fit(Z.reshape(-1, 1), X)
    X_hat = first_stage.predict(Z.reshape(-1, 1))
    
    # Second stage: regress outcome on predicted treatment
    second_stage = LinearRegression()
    second_stage.fit(X_hat.reshape(-1, 1), Y)
    
    # Causal effect estimate
    causal_effect = second_stage.coef_[0]
    
    return causal_effect, first_stage, second_stage

# Example: Using random assignment as instrument
# Z = random_assignment (0/1)
# X = actual_recommendation_shown (0/1) - may differ due to technical issues
# Y = user_engagement_score
```

## 4. Advanced Analysis Techniques

### 4.1 CUPED and Variance Reduction

**Controlled-Experiment Using Pre-Experiment Data**

**Motivation**
Reduce variance in A/B test estimates using pre-experiment data:
- **Historical Baseline**: Use pre-experiment user behavior
- **Variance Reduction**: Reduce standard errors of treatment effects
- **Increased Sensitivity**: Detect smaller effects with same sample size
- **Faster Experiments**: Reach significance faster

**Implementation**
```python
import numpy as np
from scipy import stats

def cuped_analysis(Y_pre, Y_post, treatment):
    """
    CUPED analysis for variance reduction
    
    Args:
        Y_pre: Pre-experiment metric values
        Y_post: Post-experiment metric values  
        treatment: Treatment assignment (0/1)
    """
    # Calculate theta (optimal coefficient)
    theta = np.cov(Y_pre, Y_post)[0, 1] / np.var(Y_pre)
    
    # Adjust post-experiment values
    Y_adjusted = Y_post - theta * (Y_pre - np.mean(Y_pre))
    
    # Standard A/B test on adjusted values
    treatment_group = Y_adjusted[treatment == 1]
    control_group = Y_adjusted[treatment == 0]
    
    # T-test
    t_stat, p_value = stats.ttest_ind(treatment_group, control_group)
    
    # Effect size and confidence interval
    effect_size = np.mean(treatment_group) - np.mean(control_group)
    pooled_std = np.sqrt(((len(treatment_group) - 1) * np.var(treatment_group) + 
                          (len(control_group) - 1) * np.var(control_group)) /
                         (len(treatment_group) + len(control_group) - 2))
    
    se = pooled_std * np.sqrt(1/len(treatment_group) + 1/len(control_group))
    ci_lower = effect_size - 1.96 * se
    ci_upper = effect_size + 1.96 * se
    
    return {
        'effect_size': effect_size,
        'p_value': p_value,
        'confidence_interval': (ci_lower, ci_upper),
        'theta': theta
    }
```

**Stratification and Post-Stratification**
Use user segments to reduce variance:
- **Pre-Stratification**: Stratify randomization by user segments
- **Post-Stratification**: Adjust estimates using post-experiment segmentation
- **Precision Gains**: Larger gains when segments have different baseline rates
- **Implementation**: Weight estimates by segment size

### 4.2 Network Effects and Spillovers

**Handling Interference Between Units**

**Types of Network Effects**
- **Direct Effects**: Treatment directly affects treated user
- **Indirect Effects**: Treatment affects non-treated users through network
- **Total Effects**: Sum of direct and indirect effects
- **Spillover Bias**: Bias in estimated direct effects due to spillovers

**Cluster Randomization**
Randomize clusters to reduce spillovers:
- **Geographic Clusters**: Cities, regions, countries
- **Social Clusters**: Friend networks, communities
- **Temporal Clusters**: Time periods
- **Trade-off**: Reduced statistical power for cleaner causal identification

**Analysis with Network Effects**
```python
def network_effects_analysis(outcomes, treatment, network_exposure):
    """
    Analyze experiments with network effects
    
    Args:
        outcomes: User outcomes
        treatment: Direct treatment assignment
        network_exposure: Fraction of network connections treated
    """
    # Model: Y = α + β₁*Treatment + β₂*NetworkExposure + ε
    
    from sklearn.linear_model import LinearRegression
    import numpy as np
    
    X = np.column_stack([treatment, network_exposure])
    model = LinearRegression()
    model.fit(X, outcomes)
    
    direct_effect = model.coef_[0]
    spillover_effect = model.coef_[1]
    
    return {
        'direct_effect': direct_effect,
        'spillover_effect': spillover_effect,
        'total_effect': direct_effect + spillover_effect
    }
```

### 4.3 Long-term Effect Estimation

**Beyond Short-term Metrics**

**Surrogate Endpoints**
Use short-term metrics to predict long-term outcomes:
- **Validation**: Establish correlation between short and long-term metrics
- **Surrogacy Criteria**: Statistical criteria for valid surrogates
- **Business Logic**: Ensure surrogates make business sense
- **Regular Updates**: Update surrogate relationships over time

**Difference-in-Differences**
For long-term trend analysis:
- **Parallel Trends**: Key assumption about counterfactual trends
- **Before-After**: Compare changes over time between groups
- **Robustness**: Test sensitivity to trend assumptions
- **Multiple Time Periods**: Use multiple pre/post periods when available

```python
def difference_in_differences(data, outcome_col, treatment_col, time_col, unit_col):
    """
    Difference-in-differences estimation
    
    Args:
        data: Panel data with outcomes over time
        outcome_col: Outcome variable
        treatment_col: Treatment indicator
        time_col: Time period indicator (0=pre, 1=post)
        unit_col: Unit identifier
    """
    import pandas as pd
    from statsmodels.formula.api import ols
    
    # Interaction term for DiD effect
    data['treatment_time'] = data[treatment_col] * data[time_col]
    
    # Regression with fixed effects
    model = ols(f'{outcome_col} ~ {treatment_col} + {time_col} + treatment_time + C({unit_col})', 
                data=data).fit()
    
    # DiD estimate is coefficient on interaction term
    did_effect = model.params['treatment_time']
    
    return did_effect, model
```

## 5. Practical Implementation

### 5.1 Experimentation Infrastructure

**Platform Components**

**Assignment Service**
- **Deterministic Assignment**: Consistent assignment across sessions
- **Traffic Splitting**: Flexible traffic allocation between variants
- **Feature Flags**: Integration with feature flag systems
- **Logging**: Comprehensive logging of assignments and exposures

**Metrics Pipeline**
- **Real-time Tracking**: Stream processing for real-time metrics
- **Attribution**: Correct attribution of actions to experiments
- **Data Quality**: Validation and quality checks on experimental data
- **Historical Storage**: Long-term storage for retrospective analysis

**Analysis Framework**
```python
class ExperimentAnalyzer:
    def __init__(self, experiment_id):
        self.experiment_id = experiment_id
        self.data = self.load_experiment_data()
    
    def load_experiment_data(self):
        # Load data from data warehouse
        pass
    
    def run_analysis(self, metric_name, analysis_type='ttest'):
        if analysis_type == 'ttest':
            return self.ttest_analysis(metric_name)
        elif analysis_type == 'cuped':
            return self.cuped_analysis(metric_name)
        elif analysis_type == 'bootstrap':
            return self.bootstrap_analysis(metric_name)
    
    def ttest_analysis(self, metric_name):
        treatment = self.data[self.data['variant'] == 'treatment'][metric_name]
        control = self.data[self.data['variant'] == 'control'][metric_name]
        
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(treatment, control)
        
        return {
            'metric': metric_name,
            'treatment_mean': treatment.mean(),
            'control_mean': control.mean(),
            'effect_size': treatment.mean() - control.mean(),
            'relative_lift': (treatment.mean() - control.mean()) / control.mean(),
            'p_value': p_value,
            't_statistic': t_stat
        }
    
    def generate_report(self):
        # Generate comprehensive experiment report
        pass
```

### 5.2 Quality Assurance

**Sample Ratio Mismatch (SRM)**
Monitor whether traffic splits match intended ratios:
- **Chi-square Test**: Test if observed splits match expected
- **Causes**: Assignment bugs, bot traffic, filtering issues
- **Impact**: Can bias results if not detected
- **Response**: Investigate and fix before analyzing results

**A/A Testing**
Test identical treatments to validate system:
- **False Positive Rate**: Should match significance level
- **System Validation**: Ensures assignment and analysis systems work correctly
- **Baseline Establishment**: Establishes natural variation in metrics
- **Regular Practice**: Run A/A tests regularly to monitor system health

**Guardrail Metrics**
Metrics that shouldn't be negatively affected:
- **User Safety**: Error rates, system stability
- **User Experience**: Page load times, app crashes
- **Business Critical**: Revenue, retention (for non-revenue experiments)
- **Automatic Alerts**: Alert if guardrails are violated

### 5.3 Organizational Best Practices

**Experiment Review Process**
- **Design Review**: Review experimental design before launch
- **Statistical Review**: Validate statistical analysis plan
- **Business Review**: Ensure alignment with business objectives
- **Ethics Review**: Consider ethical implications of experiments

**Result Interpretation**
- **Statistical vs Practical Significance**: Distinguish meaningful from detectable
- **Confidence Intervals**: Report intervals, not just point estimates
- **Effect Size**: Report standardized effect sizes
- **Heterogeneity**: Analyze effects across user segments

**Knowledge Management**
- **Experiment Registry**: Centralized registry of all experiments
- **Result Archive**: Systematic storage of experimental results
- **Learning Library**: Catalog of insights and best practices
- **Cross-team Sharing**: Regular sharing of results across teams

## 6. Case Studies

### 6.1 Search Ranking Experiment

**Scenario**: Testing new machine learning ranking algorithm

**Experimental Design**
- **Unit**: User-level randomization
- **Primary Metric**: Click-through rate on search results
- **Secondary Metrics**: Query abandonment rate, session length
- **Guardrails**: Page load time, error rate

**Challenges Addressed**
- **Query Diversity**: Stratify by query type and frequency
- **Learning Effects**: Monitor metrics over time to detect adaptation
- **Position Bias**: Use interleaving for more sensitive comparison
- **Statistical Power**: Large sample size needed due to low baseline CTR

### 6.2 Recommendation Algorithm Upgrade

**Scenario**: Replacing collaborative filtering with deep learning model

**Causal Considerations**
- **Network Effects**: Recommendations affect what content is created
- **Long-term Effects**: User preference evolution due to recommendations
- **Multi-sided Platform**: Effects on both content consumers and creators
- **Heterogeneity**: Different effects for different user segments

**Analysis Approach**
- **Factorial Design**: Test algorithm and interface changes together
- **Uplift Modeling**: Identify users who benefit most from new algorithm
- **Long-term Study**: Extended experiment to measure adaptation effects
- **Creator Impact**: Separate analysis of effects on content creators

### 6.3 Personalization Feature Test

**Scenario**: Adding personalized email recommendations

**Experimental Challenges**
- **External Validity**: Email behavior may differ from app behavior
- **Spillover Effects**: Email recommendations might affect app usage
- **Attribution**: Multiple touchpoints in user journey
- **Seasonal Effects**: Email effectiveness varies by time of year

**Causal Inference Application**
- **Instrumental Variables**: Use random email delivery times as instrument
- **Difference-in-Differences**: Compare changes in app usage before/after email
- **Mediation Analysis**: Understand mechanism of email impact on app behavior

## 7. Study Questions

### Beginner Level
1. What are the key differences between A/B testing and multi-armed bandit testing? When would you use each approach?
2. How do you calculate the required sample size for an A/B test in a recommendation system?
3. What is the difference between statistical significance and practical significance in the context of recommendation experiments?
4. How do network effects complicate A/B testing in social platforms?
5. What are guardrail metrics and why are they important in experimentation?

### Intermediate Level
1. Design a comprehensive A/B testing framework for a new personalization feature that accounts for both short-term and long-term effects.
2. Explain how CUPED (Controlled-experiment Using Pre-Experiment Data) works and implement it for a search ranking experiment.
3. How would you use causal inference techniques to understand the mechanism by which recommendations affect user behavior?
4. Analyze the trade-offs between different randomization units (user-level, session-level, query-level) for search and recommendation experiments.
5. Design an uplift modeling approach to identify which users benefit most from personalized recommendations.

### Advanced Level
1. Develop a comprehensive framework for handling network effects and spillovers in recommendation system experiments.
2. Create a causal inference approach for understanding the long-term effects of recommendation algorithms on user preference evolution.
3. Design a sequential experimentation framework that can handle multiple concurrent experiments while controlling for interaction effects.
4. Develop techniques for measuring and optimizing heterogeneous treatment effects in personalized recommendation systems.
5. Create a unified framework that combines experimental and observational data to make causal inferences about recommendation system performance.

## 8. Key Business Questions and Metrics

### Primary Business Questions:
- **Which version of my recommender performs better and why?**
- **What's the true impact of a new ranking algorithm on user behavior?**
- **How do we isolate the effect of one change from other confounding factors?**
- **What's the optimal allocation of traffic between experimental variants?**
- **How do we measure long-term effects of recommendation changes?**

### Key Metrics:
- **Average Treatment Effect (ATE)**: Mean difference between treatment and control groups
- **P-value**: Probability of observing results under null hypothesis
- **Confidence Intervals**: Range of plausible values for true treatment effect
- **Effect Size**: Standardized measure of treatment impact
- **Statistical Power**: Probability of detecting true effect if it exists
- **Relative Lift**: Percentage improvement over baseline
- **Heterogeneous Treatment Effects**: Variation in treatment effects across user segments

This comprehensive exploration of A/B testing and causal inference provides the methodological foundation for making reliable, data-driven decisions about search and recommendation system improvements while accounting for the complex causal relationships inherent in these systems.