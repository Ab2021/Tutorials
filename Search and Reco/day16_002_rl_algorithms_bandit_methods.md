# Day 16.2: RL Algorithms and Multi-Armed Bandit Methods for Recommendations

## Learning Objectives
By the end of this session, students will be able to:
- Understand multi-armed bandit algorithms and their application to recommendations
- Analyze contextual bandit methods for personalized recommendations
- Evaluate different bandit algorithms for various recommendation scenarios
- Design bandit-based recommendation systems with proper exploration strategies
- Understand linear bandits and their scalability advantages
- Apply bandit methods to real-world recommendation problems

## 1. Multi-Armed Bandit Foundations

### 1.1 The Multi-Armed Bandit Problem

**Classical Bandit Setting**

**Problem Formulation**
The multi-armed bandit problem models sequential decision-making under uncertainty:
- **Arms**: K different actions (recommendations) available
- **Rewards**: Each arm has an unknown reward distribution
- **Sequential Decisions**: Agent selects one arm at each time step
- **Feedback**: Agent observes reward from selected arm only

**Key Assumptions**
- **Independent Rounds**: Each round is independent
- **Stationary Rewards**: Reward distributions don't change over time
- **Bounded Rewards**: Rewards are bounded (e.g., [0,1])
- **No Context**: Decisions don't depend on contextual information

**Performance Metrics**

**Regret Analysis**
Regret measures the cost of not knowing the optimal arm:
- **Instantaneous Regret**: r_t = μ* - μ_{a_t}
- **Cumulative Regret**: R_T = Σ_{t=1}^T r_t
- **Expected Regret**: E[R_T] where expectation is over random choices
- **Regret Bounds**: Theoretical upper bounds on expected regret

**Optimal Performance**
- **Best Arm**: Arm with highest expected reward μ*
- **Oracle Performance**: Performance if best arm was known
- **Regret Minimization**: Goal is to minimize cumulative regret
- **Exploration-Exploitation Trade-off**: Balance learning and earning

### 1.2 Classic Bandit Algorithms

**ε-Greedy Algorithm**

**Algorithm Description**
Simple exploration strategy with random exploration:
- **Exploitation**: With probability 1-ε, choose arm with highest estimated reward
- **Exploration**: With probability ε, choose arm uniformly at random
- **Parameter**: ε controls exploration rate
- **Estimation**: Use sample average to estimate arm rewards

**Theoretical Analysis**
- **Regret Bound**: O(K log T / Δ + KT^{2/3}) for fixed ε
- **Sublinear Regret**: Regret grows sublinearly with time
- **Parameter Dependence**: Performance depends on choice of ε
- **Optimality Gap**: Δ is difference between best and second-best arms

**Practical Considerations**
- **ε-Decay**: Decrease ε over time to reduce exploration
- **Warm-up Period**: Use pure exploration initially
- **Computational Efficiency**: Very simple to implement
- **User Experience**: Random exploration may seem arbitrary to users

**Upper Confidence Bound (UCB)**

**UCB1 Algorithm**
Choose arm with highest upper confidence bound:
- **Confidence Intervals**: Maintain confidence interval for each arm
- **Upper Bound**: UCB_i(t) = μ̂_i(t) + √(2 log t / n_i(t))
- **Optimistic Selection**: Choose arm with highest upper bound
- **Automatic Exploration**: Exploration driven by uncertainty

**Theoretical Properties**
- **Regret Bound**: O(K log T / Δ) optimal regret bound
- **Logarithmic Regret**: Achieves optimal logarithmic regret
- **Gap-Dependent**: Performance depends on gaps between arms
- **No Tuning Parameters**: No parameters to tune

**UCB Variants**
- **UCB2**: Improved version with better constants
- **KL-UCB**: Uses Kullback-Leibler divergence for tighter bounds
- **UCB-V**: Incorporates variance information
- **Median Elimination**: Tournament-style elimination algorithm

**Thompson Sampling**

**Bayesian Approach**
Maintain posterior distribution over arm parameters:
- **Prior Distribution**: Start with prior beliefs about arms
- **Posterior Update**: Update beliefs based on observed rewards
- **Sampling**: Sample from posterior and choose best sample
- **Natural Exploration**: Exploration emerges from uncertainty

**Algorithm Steps**
1. **Initialize**: Set prior distributions for all arms
2. **Sample**: Sample parameter for each arm from posterior
3. **Select**: Choose arm with best sampled parameter
4. **Update**: Update posterior with observed reward
5. **Repeat**: Continue for next round

**Advantages**
- **Optimal Regret**: Achieves optimal regret bounds
- **Natural Exploration**: Exploration feels more natural
- **Flexible Priors**: Can incorporate domain knowledge
- **Computational Efficiency**: Often more efficient than UCB

### 1.3 Application to Item Recommendation

**Item-as-Arms Formulation**

**Direct Mapping**
Each item corresponds to one arm:
- **Arms**: Individual items in catalog
- **Rewards**: User feedback (click, rating, purchase)
- **Challenge**: Large number of arms (millions of items)
- **Scalability**: Need algorithms that scale to large arm sets

**Cold Start Items**
- **New Items**: Items with no interaction history
- **Exploration Need**: Must explore new items to learn quality
- **Business Impact**: Balance between popular and new items
- **Content Features**: Use item features to initialize estimates

**Dynamic Item Sets**
- **Changing Catalog**: Items added/removed over time
- **Seasonal Items**: Items relevant only at certain times
- **Inventory Constraints**: Items may go out of stock
- **Availability**: Must handle item availability dynamically

**Reward Design for Items**

**Binary Rewards**
- **Click Indicators**: 1 if clicked, 0 otherwise
- **Purchase Indicators**: 1 if purchased, 0 otherwise
- **Simple Implementation**: Easy to implement and understand
- **Loss of Information**: Ignores degrees of satisfaction

**Continuous Rewards**
- **Rating Scores**: Normalized rating values
- **Engagement Metrics**: Time spent, completion rate
- **Business Metrics**: Revenue, profit margin
- **Composite Scores**: Weighted combination of multiple signals

**Delayed and Sparse Rewards**
- **Temporal Delay**: Purchase may happen days after recommendation
- **Attribution**: Which recommendation led to purchase?
- **Sparse Feedback**: Most recommendations receive no feedback
- **Reward Shaping**: Use intermediate rewards to guide learning

## 2. Contextual Bandits for Personalization

### 2.1 Contextual Bandit Framework

**Problem Formulation**

**Context Integration**
Contextual bandits extend basic bandits with contextual information:
- **Context Vector**: x_t ∈ ℝ^d represents context at time t
- **Context-Dependent Rewards**: Reward depends on both arm and context
- **Personalization**: Context can include user information
- **Decision Function**: π(x) maps context to arm selection

**Mathematical Framework**
- **Context Space**: X ⊆ ℝ^d
- **Action Space**: A = {1, 2, ..., K}
- **Reward Function**: r(x, a) = μ(x, a) + noise
- **Policy**: π: X → A maps contexts to actions
- **Expected Reward**: μ(x, a) = E[r(x, a) | x, a]

**Contextual Regret**
- **Optimal Policy**: π*(x) = argmax_a μ(x, a)
- **Contextual Regret**: R_T = Σ_{t=1}^T (μ(x_t, π*(x_t)) - μ(x_t, π(x_t)))
- **Worst-Case Analysis**: Maximum regret over all context sequences
- **Average-Case Analysis**: Expected regret over context distribution

### 2.2 Linear Contextual Bandits

**Linear Reward Model**

**Model Assumption**
Assume rewards are linear in context-action features:
- **Feature Map**: φ(x, a) ∈ ℝ^d maps context-action pairs to features
- **Linear Model**: μ(x, a) = θ*^T φ(x, a)
- **Unknown Parameter**: θ* is unknown true parameter
- **Noise Model**: r(x, a) = θ*^T φ(x, a) + η

**Feature Engineering**
- **User Features**: Demographics, preferences, history
- **Item Features**: Category, price, rating, popularity
- **Context Features**: Time, location, device, session
- **Interaction Features**: User-item interaction features

**LinUCB Algorithm**

**Algorithm Description**
Linear Upper Confidence Bound for contextual bandits:
- **Parameter Estimation**: Use ridge regression to estimate θ
- **Confidence Ellipsoid**: Maintain confidence region for θ
- **Upper Confidence Bound**: UCB_a(x) = x_a^T θ̂ + α√(x_a^T A_a^{-1} x_a)
- **Action Selection**: Choose action with highest UCB

**Detailed Steps**
1. **Initialize**: A_a = I (identity matrix) for each arm a
2. **Feature Extraction**: Extract features φ(x, a) for each arm
3. **Parameter Update**: θ̂_a = A_a^{-1} b_a where b_a is reward sum
4. **Confidence Computation**: Compute confidence width for each arm
5. **Action Selection**: Select arm with highest upper confidence bound
6. **Update**: Update A_a and b_a with observed reward

**Theoretical Properties**
- **Regret Bound**: O(d√T log T) with high probability
- **Dimension Dependence**: Regret depends on feature dimension d
- **Log Factors**: Additional logarithmic factors in practice
- **Optimality**: Near-optimal for linear models

**Practical Variants**
- **Hybrid LinUCB**: Combine individual and global parameters
- **Disjoint LinUCB**: Separate parameters for each arm
- **Shared LinUCB**: Share parameters across arms
- **Kernelized LinUCB**: Use kernel methods for non-linear rewards

### 2.3 Non-Linear Contextual Bandits

**Neural Bandits**

**Deep Learning Integration**
Use neural networks for reward modeling:
- **Neural Network**: f_θ(x, a) approximates reward function
- **Parameter Learning**: Learn θ through gradient descent
- **Exploration**: Use uncertainty estimates for exploration
- **Scalability**: Handle high-dimensional contexts and large action spaces

**Neural Thompson Sampling**
- **Bayesian Neural Networks**: Maintain distributions over network parameters
- **Variational Inference**: Approximate posterior using variational methods
- **Sampling**: Sample network parameters and choose best action
- **Implementation**: Use dropout or ensemble methods for uncertainty

**Neural UCB Approaches**
- **Bootstrap**: Train multiple networks on bootstrap samples
- **Ensemble Disagreement**: Use disagreement between networks as uncertainty
- **Gradient-Based**: Use gradient information for confidence bounds  
- **Last-Layer Uncertainty**: Focus uncertainty estimation on last layer

**Tree-Based Methods**

**Random Forest Bandits**
- **Ensemble Trees**: Use random forest for reward prediction
- **Uncertainty Estimation**: Use tree disagreement for exploration
- **Feature Importance**: Automatic feature selection
- **Interpretability**: More interpretable than neural methods

**Gradient Boosting Bandits**
- **Sequential Learning**: Build models sequentially
- **Residual Learning**: Each model learns residuals of previous
- **Exploration Strategy**: Use prediction variance for exploration
- **Computational Efficiency**: Fast training and prediction

## 3. Advanced Bandit Methods

### 3.1 Combinatorial Bandits

**Slate Recommendation Problem**

**Problem Setting**
Recommend slate of K items from catalog of N items:
- **Combinatorial Actions**: Choose subset of items
- **Position Effects**: Order matters in recommendation list
- **Interaction Effects**: Items in slate may interact
- **Scalability**: Exponential number of possible slates

**Reward Models**
- **Additive Model**: Slate reward is sum of individual item rewards
- **Position-Dependent**: Rewards depend on position in slate
- **Click Models**: Probabilistic models of user click behavior
- **Submodular**: Diminishing returns for similar items

**CUCB Algorithm**
Combinatorial Upper Confidence Bound:
- **Base Arms**: Individual items as base arms
- **Super Arms**: Slates as combinations of base arms
- **Confidence Bounds**: Maintain bounds for base arms
- **Combinatorial Optimization**: Select slate with highest upper bound
- **Theoretical Guarantees**: Regret bounds for combinatorial settings

**Practical Considerations**
- **Computational Complexity**: NP-hard combinatorial optimization
- **Approximation Algorithms**: Use greedy or other approximation methods
- **Online Learning**: Learn user preferences for slates over time
- **Diversity**: Ensure diversity within recommended slates

### 3.2 Dueling Bandits

**Preference Learning Framework**

**Pairwise Comparisons**
Learn from relative preferences rather than absolute rewards:
- **Dueling**: Present two options and observe preference
- **Preference Matrix**: P_{ij} = probability that i is preferred to j
- **Ranking Goal**: Find total ordering of arms
- **Partial Information**: Only observe pairwise comparisons

**Applications to Recommendations**
- **A/B Testing**: Compare two recommendation strategies
- **Preference Elicitation**: Learn user preferences through comparisons
- **Ranking Optimization**: Optimize ranking of recommendations
- **Implicit Feedback**: Infer preferences from user behavior

**Algorithms**

**RUCB (Relative Upper Confidence Bound)**
- **Relative Confidence**: Maintain confidence intervals for pairwise preferences
- **Champion Selection**: Choose current best arm as champion
- **Challenger Selection**: Choose challenger based on confidence bounds
- **Preference Update**: Update preferences based on duel outcome

**BEAT (Bootstrap-inspired Easy-to-implement Algorithm)**
- **Bootstrap Sampling**: Sample from empirical distribution
- **Tournament**: Run tournament among sampled preferences
- **Winner Selection**: Choose winner of tournament
- **Theoretical Guarantees**: Near-optimal regret bounds

### 3.3 Adversarial Bandits

**Non-Stochastic Setting**

**Adversarial Rewards**
Rewards chosen by adversary rather than stochastic process:
- **Worst-Case Analysis**: No assumptions about reward generation
- **Adaptive Adversary**: Adversary can see past actions
- **Robust Performance**: Algorithms work in worst-case scenarios
- **Applications**: Competitive markets, adversarial users

**EXP3 Algorithm**
Exponential-weight algorithm for exploration and exploitation:
- **Weight Updates**: Update arm weights based on estimated rewards
- **Probability Matching**: Select arms proportional to weights
- **Importance Sampling**: Estimate rewards for unselected arms
- **Regret Bound**: O(√(K T log K)) minimax optimal

**Applications to Recommendations**
- **Competitive Markets**: Other recommendation systems competing
- **Strategic Users**: Users gaming the recommendation system
- **Changing Preferences**: User preferences change adversarially
- **Robust Recommendations**: Recommendations robust to manipulation

## 4. Practical Implementation Considerations

### 4.1 Scalability and Efficiency

**Large Action Spaces**

**Action Space Reduction**
- **Candidate Generation**: Pre-filter items to manageable set
- **Clustering**: Group similar items and choose representatives
- **Hierarchical Decomposition**: Break large problem into smaller subproblems
- **Approximate Methods**: Use approximation algorithms for efficiency

**Efficient Algorithms**
- **Online Updates**: Update models incrementally as data arrives
- **Sparse Updates**: Only update relevant parameters
- **Batch Processing**: Process multiple observations together
- **Distributed Computing**: Distribute computation across machines

**Memory Management**
- **Model Compression**: Compress models to reduce memory usage
- **Feature Selection**: Select most important features
- **Incremental Learning**: Learn without storing all historical data
- **Online Feature Learning**: Learn feature representations online

### 4.2 Cold Start Problems

**New User Cold Start**

**Default Policies**
- **Popular Items**: Start with most popular items
- **Demographic Matching**: Use demographic information for initial recommendations
- **Content-Based**: Use item features for initial matching
- **Transfer Learning**: Transfer knowledge from similar users

**Fast Personalization**
- **Active Learning**: Ask users to rate a few items
- **Implicit Feedback**: Use early interactions for quick personalization
- **Social Information**: Use social connections for initialization
- **Progressive Learning**: Gradually improve personalization over time

**New Item Cold Start**
- **Content-Based Features**: Use item metadata for initial estimates
- **Similar Item Transfer**: Transfer from similar items
- **Expert Initialization**: Use expert knowledge for initialization
- **Exploration Bonus**: Give exploration bonus to new items

### 4.3 Online Evaluation and Deployment

**A/B Testing Framework**

**Experimental Design**
- **Control Group**: Users receiving current recommendation system
- **Treatment Groups**: Users receiving new bandit-based system
- **Random Assignment**: Randomly assign users to groups
- **Statistical Power**: Ensure sufficient sample size for detection

**Metrics and KPIs**
- **User Engagement**: Click-through rates, time spent, return visits
- **Business Metrics**: Revenue, conversion rates, user lifetime value
- **User Satisfaction**: Explicit ratings, surveys, Net Promoter Score
- **System Performance**: Latency, throughput, resource usage

**Safe Deployment**

**Conservative Exploration**
- **Safe Policies**: Ensure exploration doesn't harm user experience
- **Bounded Exploration**: Limit how much system can deviate from safe policy
- **Risk Assessment**: Assess potential negative impacts of exploration
- **Gradual Rollout**: Gradually increase exploration over time

**Monitoring and Alerts**
- **Real-Time Monitoring**: Monitor system performance in real-time
- **Anomaly Detection**: Detect unusual patterns in user behavior
- **Alert Systems**: Alert when performance drops below threshold
- **Rollback Procedures**: Quick rollback if problems detected

## 5. Study Questions

### Beginner Level
1. What is the exploration-exploitation trade-off in multi-armed bandits and why is it important for recommendations?
2. How do contextual bandits differ from standard multi-armed bandits and what advantages do they provide?
3. What is the LinUCB algorithm and how does it work for linear contextual bandits?
4. How do you handle cold start problems in bandit-based recommendation systems?
5. What are the main challenges in applying bandit methods to real-world recommendation systems?

### Intermediate Level
1. Compare ε-greedy, UCB, and Thompson Sampling algorithms for recommendation scenarios and analyze their trade-offs.
2. Design a contextual bandit system for personalized news recommendation that handles both new users and new articles.
3. How would you extend linear contextual bandits to handle non-linear reward functions while maintaining computational efficiency?
4. Analyze the regret bounds for different bandit algorithms and explain what they mean for practical recommendation systems.
5. Design an evaluation methodology for bandit-based recommendation systems that considers both online and offline metrics.

### Advanced Level
1. Develop a theoretical analysis of the sample complexity required for contextual bandits to achieve good performance in recommendation settings with concept drift.
2. Design a combinatorial bandit algorithm for slate recommendations that handles position bias and item interactions effectively.
3. Create a framework for safe exploration in recommendation systems that balances learning with user experience protection.
4. Develop a meta-learning approach for contextual bandits that can quickly adapt to new recommendation domains or user populations.
5. Design a distributed contextual bandit system that can handle millions of users and items while maintaining theoretical guarantees.

## 6. Case Studies and Applications

### 6.1 News Recommendation

**Yahoo! News Case Study**
- **Dataset**: Yahoo! Front Page Today Module user click log
- **Context**: User demographics and article features
- **Action Space**: Articles to display on front page
- **Reward**: Click-through rate
- **Algorithm**: LinUCB with hybrid approach
- **Results**: Significant improvement over random and popularity baselines

**Challenges and Solutions**
- **High-Dimensional Context**: Use feature selection and dimensionality reduction
- **Rapid Content Change**: Articles become stale quickly
- **Position Bias**: Account for position effects in clicks
- **Seasonality**: Handle time-dependent user behavior

### 6.2 E-commerce Recommendation

**Product Recommendation Bandits**
- **Context**: User profile, session history, product features
- **Actions**: Products to recommend
- **Rewards**: Clicks, add-to-cart, purchases
- **Multi-Objective**: Balance multiple business objectives
- **Inventory**: Handle inventory constraints and product availability

**Implementation Considerations**
- **Real-Time**: Provide recommendations in real-time
- **Scalability**: Handle millions of products and users
- **Seasonality**: Adapt to seasonal shopping patterns
- **Cross-Selling**: Recommend complementary products

### 6.3 Video Streaming Services

**Content Recommendation**
- **Context**: User viewing history, time of day, device
- **Actions**: Movies/shows to recommend
- **Rewards**: Watch time, completion rate, ratings
- **Exploration**: Balance popular and niche content
- **Long-Term**: Optimize for long-term user satisfaction

**Technical Challenges**
- **Cold Start**: New users and new content
- **Scalability**: Handle large catalogs and user bases
- **Personalization**: Adapt to individual viewing preferences
- **Diversity**: Ensure diverse content recommendations

This comprehensive exploration of bandit methods provides the foundation for implementing practical, scalable recommendation systems that can learn and adapt from user feedback while balancing exploration and exploitation effectively.