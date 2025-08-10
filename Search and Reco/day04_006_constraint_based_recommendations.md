# Day 4.6: Constraint-Based and Rule-Based Systems (Advanced)

## Learning Objectives
By the end of this session, you will:
- Master advanced constraint programming techniques for recommendations
- Implement fuzzy rule systems for handling uncertain user preferences
- Apply multi-objective optimization in recommendation scenarios
- Build interactive constraint refinement systems
- Design scalable rule execution architectures for production systems

## 1. Advanced Constraint Programming

### Soft Constraints and Preference Modeling

```python
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable, Any
from collections import defaultdict
import heapq
import random

class ConstraintType(Enum):
    HARD = "hard"
    SOFT = "soft"
    PREFERENCE = "preference"

@dataclass
class WeightedConstraint:
    """Constraint with importance weight and violation cost"""
    variables: List[str]
    constraint_func: Callable
    constraint_type: ConstraintType
    weight: float = 1.0
    violation_cost: float = 1.0
    name: str = ""
    
    def compute_satisfaction(self, assignment: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Compute constraint satisfaction and violation cost
        
        Returns:
            (is_satisfied, violation_cost)
        """
        try:
            # Check if all variables are assigned
            values = []
            for var in self.variables:
                if var not in assignment:
                    return True, 0.0  # Not applicable yet
                values.append(assignment[var])
            
            # Evaluate constraint
            if isinstance(self.constraint_func(*values), bool):
                # Binary constraint
                satisfied = self.constraint_func(*values)
                cost = 0.0 if satisfied else self.violation_cost
                return satisfied, cost
            else:
                # Soft constraint with degree of satisfaction
                satisfaction_degree = self.constraint_func(*values)
                cost = (1.0 - satisfaction_degree) * self.violation_cost
                return satisfaction_degree > 0.5, cost
                
        except Exception:
            return False, self.violation_cost

class SoftConstraintCSP:
    """
    Constraint Satisfaction Problem with soft constraints and optimization
    """
    
    def __init__(self):
        self.variables = {}  # name -> Variable
        self.constraints = []  # List of WeightedConstraint
        self.optimization_objective = None  # Function to maximize/minimize
        
    def add_weighted_constraint(self, constraint: WeightedConstraint):
        """Add weighted constraint to CSP"""
        self.constraints.append(constraint)
    
    def set_optimization_objective(self, objective_func: Callable[[Dict[str, Any]], float]):
        """Set objective function to optimize"""
        self.optimization_objective = objective_func
    
    def solve_with_optimization(self, max_solutions: int = 10, 
                              optimization_type: str = "maximize") -> List[Dict[str, Any]]:
        """
        Solve CSP with soft constraints using optimization
        
        Args:
            max_solutions: Maximum number of solutions to find
            optimization_type: "maximize" or "minimize"
            
        Returns:
            List of solutions ranked by objective value
        """
        solutions = []
        
        # Use branch and bound with optimization
        best_value = float('-inf') if optimization_type == "maximize" else float('inf')
        
        assignment = {}
        self._branch_and_bound_search(
            assignment, solutions, best_value, 
            optimization_type, max_solutions
        )
        
        # Sort solutions by objective value
        if optimization_type == "maximize":
            solutions.sort(key=lambda x: x['objective_value'], reverse=True)
        else:
            solutions.sort(key=lambda x: x['objective_value'])
        
        return solutions[:max_solutions]
    
    def _branch_and_bound_search(self, assignment: Dict[str, Any], 
                                solutions: List[Dict], best_value: float,
                                optimization_type: str, max_solutions: int):
        """Branch and bound search with optimization"""
        
        if len(solutions) >= max_solutions:
            return
        
        # Check if assignment is complete
        if len(assignment) == len(self.variables):
            objective_value, constraint_violations = self._evaluate_solution(assignment)
            
            # Check if this solution improves our bound
            is_better = (optimization_type == "maximize" and objective_value > best_value) or \
                       (optimization_type == "minimize" and objective_value < best_value)
            
            if is_better or len(solutions) < max_solutions:
                solution = {
                    'assignment': assignment.copy(),
                    'objective_value': objective_value,
                    'constraint_violations': constraint_violations,
                    'total_violation_cost': sum(v['cost'] for v in constraint_violations.values())
                }
                solutions.append(solution)
                
                # Update bound
                if is_better:
                    best_value = objective_value
            
            return
        
        # Select next variable
        unassigned_vars = [name for name in self.variables if name not in assignment]
        if not unassigned_vars:
            return
        
        var_name = unassigned_vars[0]  # Simple selection strategy
        variable = self.variables[var_name]
        
        # Try each value in domain
        for value in variable.domain:
            # Pruning: estimate bound for this branch
            test_assignment = assignment.copy()
            test_assignment[var_name] = value
            
            estimated_bound = self._estimate_bound(test_assignment, optimization_type)
            
            # Prune if bound is not promising
            should_prune = (optimization_type == "maximize" and estimated_bound <= best_value) or \
                          (optimization_type == "minimize" and estimated_bound >= best_value)
            
            if not should_prune or len(solutions) < max_solutions:
                assignment[var_name] = value
                self._branch_and_bound_search(
                    assignment, solutions, best_value, 
                    optimization_type, max_solutions
                )
                del assignment[var_name]
    
    def _evaluate_solution(self, assignment: Dict[str, Any]) -> Tuple[float, Dict[str, Dict]]:
        """Evaluate solution quality"""
        constraint_violations = {}
        total_satisfaction = 0.0
        total_weight = 0.0
        
        # Evaluate all constraints
        for i, constraint in enumerate(self.constraints):
            satisfied, cost = constraint.compute_satisfaction(assignment)
            
            constraint_violations[f"constraint_{i}"] = {
                'name': constraint.name or f"constraint_{i}",
                'satisfied': satisfied,
                'cost': cost,
                'weight': constraint.weight,
                'type': constraint.constraint_type.value
            }
            
            # Compute weighted satisfaction
            satisfaction_score = 1.0 if satisfied else max(0.0, 1.0 - cost)
            total_satisfaction += constraint.weight * satisfaction_score
            total_weight += constraint.weight
        
        # Compute objective value
        if self.optimization_objective:
            objective_value = self.optimization_objective(assignment)
        else:
            # Default: maximize constraint satisfaction
            objective_value = total_satisfaction / total_weight if total_weight > 0 else 0.0
        
        return objective_value, constraint_violations
    
    def _estimate_bound(self, partial_assignment: Dict[str, Any], 
                       optimization_type: str) -> float:
        """Estimate optimistic bound for partial assignment"""
        # Simple optimistic estimation
        # In practice, would use more sophisticated bounding
        
        if self.optimization_objective:
            # For custom objectives, assume remaining variables contribute optimally
            return self.optimization_objective(partial_assignment)
        else:
            # For constraint satisfaction, assume all remaining constraints satisfied
            if optimization_type == "maximize":
                return 1.0  # Perfect satisfaction
            else:
                return 0.0  # No violations

class MultiObjectiveCSP(SoftConstraintCSP):
    """
    Multi-objective constraint satisfaction for recommendations
    """
    
    def __init__(self):
        super().__init__()
        self.objectives = []  # List of (name, function, weight) tuples
        
    def add_objective(self, name: str, objective_func: Callable, weight: float = 1.0):
        """Add objective function"""
        self.objectives.append((name, objective_func, weight))
    
    def solve_pareto_optimal(self, max_solutions: int = 50) -> List[Dict[str, Any]]:
        """
        Find Pareto-optimal solutions for multi-objective problem
        
        Returns:
            List of non-dominated solutions
        """
        all_solutions = []
        
        # Generate many solutions using different weightings
        num_weight_combinations = min(20, max_solutions)
        
        for _ in range(num_weight_combinations):
            # Random weight combination
            weights = [random.random() for _ in self.objectives]
            total_weight = sum(weights)
            weights = [w/total_weight for w in weights]
            
            # Create combined objective
            def combined_objective(assignment):
                total_value = 0.0
                for i, (name, obj_func, base_weight) in enumerate(self.objectives):
                    try:
                        value = obj_func(assignment)
                        total_value += weights[i] * base_weight * value
                    except Exception:
                        pass
                return total_value
            
            self.set_optimization_objective(combined_objective)
            solutions = self.solve_with_optimization(max_solutions=5)
            
            for solution in solutions:
                # Compute all objective values
                solution['objective_values'] = {}
                for name, obj_func, _ in self.objectives:
                    try:
                        solution['objective_values'][name] = obj_func(solution['assignment'])
                    except Exception:
                        solution['objective_values'][name] = 0.0
                
                all_solutions.append(solution)
        
        # Find Pareto-optimal solutions
        pareto_optimal = self._find_pareto_optimal(all_solutions)
        
        return pareto_optimal[:max_solutions]
    
    def _find_pareto_optimal(self, solutions: List[Dict]) -> List[Dict]:
        """Find Pareto-optimal solutions"""
        pareto_optimal = []
        
        for candidate in solutions:
            is_dominated = False
            
            for other in solutions:
                if candidate == other:
                    continue
                
                # Check if other dominates candidate
                dominates = True
                strictly_better = False
                
                for obj_name in candidate['objective_values']:
                    candidate_value = candidate['objective_values'][obj_name]
                    other_value = other['objective_values'][obj_name]
                    
                    if other_value < candidate_value:  # Assuming maximization
                        dominates = False
                        break
                    elif other_value > candidate_value:
                        strictly_better = True
                
                if dominates and strictly_better:
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_optimal.append(candidate)
        
        return pareto_optimal

class RecommendationOptimizer:
    """
    Optimization-based recommendation system
    """
    
    def __init__(self, items: List[Dict], user_profile: Dict):
        self.items = items
        self.user_profile = user_profile
        self.csp = MultiObjectiveCSP()
        self._setup_optimization_problem()
    
    def _setup_optimization_problem(self):
        """Setup multi-objective optimization problem"""
        
        # Create variables for recommendation slots
        num_recommendations = 5
        item_ids = [item['id'] for item in self.items]
        
        for i in range(num_recommendations):
            var_name = f"rec_{i}"
            from day4_005_knowledge_based_systems import Variable  # Import from previous module
            self.csp.add_variable(Variable(var_name, item_ids))
        
        # Add diversity constraint (soft)
        def diversity_score(*items):
            unique_genres = set()
            for item_id in items:
                item = next((item for item in self.items if item['id'] == item_id), None)
                if item:
                    unique_genres.update(item.get('genres', []))
            return len(unique_genres) / 10.0  # Normalize to 0-1
        
        var_names = [f"rec_{i}" for i in range(num_recommendations)]
        diversity_constraint = WeightedConstraint(
            var_names, diversity_score, ConstraintType.SOFT, 
            weight=0.3, name="Genre Diversity"
        )
        self.csp.add_weighted_constraint(diversity_constraint)
        
        # Add user preference alignment
        def preference_score(*items):
            total_score = 0.0
            for item_id in items:
                item = next((item for item in self.items if item['id'] == item_id), None)
                if item:
                    score = self._compute_item_preference_score(item)
                    total_score += score
            return total_score / len(items)
        
        preference_constraint = WeightedConstraint(
            var_names, preference_score, ConstraintType.SOFT,
            weight=0.5, name="User Preference Alignment"
        )
        self.csp.add_weighted_constraint(preference_constraint)
        
        # Add objectives
        self.csp.add_objective("relevance", self._relevance_objective, weight=0.4)
        self.csp.add_objective("diversity", self._diversity_objective, weight=0.3)
        self.csp.add_objective("novelty", self._novelty_objective, weight=0.2)
        self.csp.add_objective("business_value", self._business_value_objective, weight=0.1)
    
    def _compute_item_preference_score(self, item: Dict) -> float:
        """Compute how well item matches user preferences"""
        score = 0.0
        
        # Genre preferences
        if 'favorite_genres' in self.user_profile:
            item_genres = set(item.get('genres', []))
            favorite_genres = set(self.user_profile['favorite_genres'])
            genre_overlap = len(item_genres.intersection(favorite_genres))
            score += genre_overlap * 0.3
        
        # Rating preference
        min_rating = self.user_profile.get('min_rating', 0.0)
        item_rating = item.get('rating', 0.0)
        if item_rating >= min_rating:
            score += (item_rating - min_rating) * 0.2
        
        # Recency preference
        max_age = self.user_profile.get('max_age_years', 50)
        item_age = 2024 - item.get('year', 2000)
        if item_age <= max_age:
            score += (1.0 - item_age / max_age) * 0.1
        
        return min(1.0, score)
    
    def _relevance_objective(self, assignment: Dict[str, Any]) -> float:
        """Compute relevance of recommended items"""
        total_relevance = 0.0
        count = 0
        
        for var_name, item_id in assignment.items():
            if var_name.startswith('rec_'):
                item = next((item for item in self.items if item['id'] == item_id), None)
                if item:
                    relevance = self._compute_item_preference_score(item)
                    total_relevance += relevance
                    count += 1
        
        return total_relevance / count if count > 0 else 0.0
    
    def _diversity_objective(self, assignment: Dict[str, Any]) -> float:
        """Compute diversity of recommended items"""
        genres = set()
        years = []
        ratings = []
        
        for var_name, item_id in assignment.items():
            if var_name.startswith('rec_'):
                item = next((item for item in self.items if item['id'] == item_id), None)
                if item:
                    genres.update(item.get('genres', []))
                    years.append(item.get('year', 2000))
                    ratings.append(item.get('rating', 0.0))
        
        # Combine different diversity measures
        genre_diversity = len(genres) / 10.0  # Normalize
        year_diversity = (max(years) - min(years)) / 50.0 if years else 0.0
        rating_diversity = (max(ratings) - min(ratings)) / 10.0 if ratings else 0.0
        
        return (genre_diversity + year_diversity + rating_diversity) / 3.0
    
    def _novelty_objective(self, assignment: Dict[str, Any]) -> float:
        """Compute novelty of recommended items"""
        # Simple novelty: prefer less popular items
        total_novelty = 0.0
        count = 0
        
        for var_name, item_id in assignment.items():
            if var_name.startswith('rec_'):
                item = next((item for item in self.items if item['id'] == item_id), None)
                if item:
                    # Lower rating = higher novelty (simplified)
                    popularity = item.get('rating', 5.0)
                    novelty = max(0.0, (10.0 - popularity) / 10.0)
                    total_novelty += novelty
                    count += 1
        
        return total_novelty / count if count > 0 else 0.0
    
    def _business_value_objective(self, assignment: Dict[str, Any]) -> float:
        """Compute business value of recommendations"""
        # Simple business value based on item profitability
        total_value = 0.0
        count = 0
        
        for var_name, item_id in assignment.items():
            if var_name.startswith('rec_'):
                item = next((item for item in self.items if item['id'] == item_id), None)
                if item:
                    # Simulate business value (could be actual profit margins)
                    value = item.get('business_value', 0.5)
                    total_value += value
                    count += 1
        
        return total_value / count if count > 0 else 0.0
    
    def generate_recommendations(self, k: int = 10) -> List[Dict]:
        """Generate optimized recommendations"""
        solutions = self.csp.solve_pareto_optimal(max_solutions=k)
        
        recommendations = []
        for solution in solutions:
            rec_items = []
            for var_name in sorted(solution['assignment'].keys()):
                if var_name.startswith('rec_'):
                    item_id = solution['assignment'][var_name]
                    item = next((item for item in self.items if item['id'] == item_id), None)
                    if item:
                        rec_items.append(item)
            
            recommendations.append({
                'items': rec_items,
                'objectives': solution['objective_values'],
                'total_violation_cost': solution.get('total_violation_cost', 0.0),
                'explanation': self._generate_explanation(solution)
            })
        
        return recommendations
    
    def _generate_explanation(self, solution: Dict) -> str:
        """Generate explanation for recommendation set"""
        objectives = solution['objective_values']
        
        explanation = "This recommendation set balances:\n"
        explanation += f"• Relevance: {objectives.get('relevance', 0):.2f}\n"
        explanation += f"• Diversity: {objectives.get('diversity', 0):.2f}\n"
        explanation += f"• Novelty: {objectives.get('novelty', 0):.2f}\n"
        explanation += f"• Business Value: {objectives.get('business_value', 0):.2f}\n"
        
        return explanation
```

## 2. Fuzzy Rule Systems

### Handling Uncertain Preferences

```python
import numpy as np
from typing import Callable, Union

class FuzzySet:
    """Represents a fuzzy set with membership function"""
    
    def __init__(self, name: str, membership_func: Callable[[float], float]):
        self.name = name
        self.membership_func = membership_func
    
    def membership(self, value: float) -> float:
        """Compute membership degree for a value"""
        return max(0.0, min(1.0, self.membership_func(value)))
    
    @staticmethod
    def triangular(a: float, b: float, c: float) -> Callable[[float], float]:
        """Create triangular membership function"""
        def membership_func(x: float) -> float:
            if x <= a or x >= c:
                return 0.0
            elif a < x <= b:
                return (x - a) / (b - a)
            else:  # b < x < c
                return (c - x) / (c - b)
        return membership_func
    
    @staticmethod
    def trapezoidal(a: float, b: float, c: float, d: float) -> Callable[[float], float]:
        """Create trapezoidal membership function"""
        def membership_func(x: float) -> float:
            if x <= a or x >= d:
                return 0.0
            elif a < x <= b:
                return (x - a) / (b - a)
            elif b < x <= c:
                return 1.0
            else:  # c < x < d
                return (d - x) / (d - c)
        return membership_func
    
    @staticmethod
    def gaussian(center: float, sigma: float) -> Callable[[float], float]:
        """Create Gaussian membership function"""
        def membership_func(x: float) -> float:
            return np.exp(-0.5 * ((x - center) / sigma) ** 2)
        return membership_func

class LinguisticVariable:
    """Represents a linguistic variable with multiple fuzzy sets"""
    
    def __init__(self, name: str, domain: Tuple[float, float]):
        self.name = name
        self.domain = domain
        self.fuzzy_sets = {}
    
    def add_fuzzy_set(self, fuzzy_set: FuzzySet):
        """Add fuzzy set to linguistic variable"""
        self.fuzzy_sets[fuzzy_set.name] = fuzzy_set
    
    def fuzzify(self, value: float) -> Dict[str, float]:
        """Convert crisp value to fuzzy memberships"""
        memberships = {}
        for name, fuzzy_set in self.fuzzy_sets.items():
            memberships[name] = fuzzy_set.membership(value)
        return memberships

class FuzzyRule:
    """Represents a fuzzy inference rule"""
    
    def __init__(self, rule_id: str, antecedents: Dict[str, str], 
                 consequent: Tuple[str, str], weight: float = 1.0):
        """
        Args:
            rule_id: Unique identifier
            antecedents: {variable_name: fuzzy_set_name}
            consequent: (variable_name, fuzzy_set_name)
            weight: Rule importance weight
        """
        self.rule_id = rule_id
        self.antecedents = antecedents
        self.consequent = consequent
        self.weight = weight
    
    def evaluate(self, inputs: Dict[str, Dict[str, float]]) -> Tuple[str, str, float]:
        """
        Evaluate rule and return firing strength
        
        Args:
            inputs: {variable_name: {fuzzy_set_name: membership}}
            
        Returns:
            (consequent_variable, consequent_fuzzy_set, firing_strength)
        """
        # Compute firing strength using minimum t-norm
        firing_strength = 1.0
        
        for var_name, fuzzy_set_name in self.antecedents.items():
            if var_name in inputs and fuzzy_set_name in inputs[var_name]:
                membership = inputs[var_name][fuzzy_set_name]
                firing_strength = min(firing_strength, membership)
            else:
                firing_strength = 0.0
                break
        
        # Apply rule weight
        firing_strength *= self.weight
        
        return self.consequent[0], self.consequent[1], firing_strength

class FuzzyInferenceSystem:
    """Fuzzy inference system for recommendation decisions"""
    
    def __init__(self):
        self.linguistic_variables = {}
        self.rules = []
        self.defuzzification_method = "centroid"
    
    def add_linguistic_variable(self, variable: LinguisticVariable):
        """Add linguistic variable to system"""
        self.linguistic_variables[variable.name] = variable
    
    def add_rule(self, rule: FuzzyRule):
        """Add fuzzy rule to system"""
        self.rules.append(rule)
    
    def infer(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """
        Perform fuzzy inference
        
        Args:
            inputs: {variable_name: crisp_value}
            
        Returns:
            {output_variable: crisp_value}
        """
        # Step 1: Fuzzification
        fuzzified_inputs = {}
        for var_name, value in inputs.items():
            if var_name in self.linguistic_variables:
                variable = self.linguistic_variables[var_name]
                fuzzified_inputs[var_name] = variable.fuzzify(value)
        
        # Step 2: Rule evaluation and aggregation
        output_fuzzy_values = defaultdict(lambda: defaultdict(float))
        
        for rule in self.rules:
            consequent_var, consequent_set, strength = rule.evaluate(fuzzified_inputs)
            
            # Aggregate using maximum
            current_strength = output_fuzzy_values[consequent_var][consequent_set]
            output_fuzzy_values[consequent_var][consequent_set] = max(current_strength, strength)
        
        # Step 3: Defuzzification
        crisp_outputs = {}
        for var_name, fuzzy_values in output_fuzzy_values.items():
            if var_name in self.linguistic_variables:
                crisp_value = self._defuzzify(var_name, fuzzy_values)
                crisp_outputs[var_name] = crisp_value
        
        return crisp_outputs
    
    def _defuzzify(self, variable_name: str, fuzzy_values: Dict[str, float]) -> float:
        """Defuzzify using centroid method"""
        variable = self.linguistic_variables[variable_name]
        
        if self.defuzzification_method == "centroid":
            return self._centroid_defuzzification(variable, fuzzy_values)
        elif self.defuzzification_method == "maximum":
            return self._maximum_defuzzification(variable, fuzzy_values)
        else:
            return self._centroid_defuzzification(variable, fuzzy_values)
    
    def _centroid_defuzzification(self, variable: LinguisticVariable, 
                                 fuzzy_values: Dict[str, float]) -> float:
        """Centroid defuzzification method"""
        min_val, max_val = variable.domain
        step = (max_val - min_val) / 1000  # Resolution
        
        numerator = 0.0
        denominator = 0.0
        
        for x in np.arange(min_val, max_val + step, step):
            # Compute aggregated membership at x
            membership = 0.0
            for fuzzy_set_name, strength in fuzzy_values.items():
                if fuzzy_set_name in variable.fuzzy_sets:
                    set_membership = variable.fuzzy_sets[fuzzy_set_name].membership(x)
                    membership = max(membership, min(strength, set_membership))
            
            numerator += x * membership
            denominator += membership
        
        return numerator / denominator if denominator > 0 else (min_val + max_val) / 2
    
    def _maximum_defuzzification(self, variable: LinguisticVariable,
                                fuzzy_values: Dict[str, float]) -> float:
        """Maximum defuzzification method"""
        max_strength = max(fuzzy_values.values())
        
        # Find fuzzy set with maximum strength
        for fuzzy_set_name, strength in fuzzy_values.items():
            if strength == max_strength and fuzzy_set_name in variable.fuzzy_sets:
                # Return center of fuzzy set (simplified)
                min_val, max_val = variable.domain
                return (min_val + max_val) / 2
        
        return (variable.domain[0] + variable.domain[1]) / 2

class FuzzyRecommendationSystem:
    """
    Fuzzy logic-based recommendation system
    """
    
    def __init__(self):
        self.fuzzy_system = FuzzyInferenceSystem()
        self._setup_fuzzy_system()
    
    def _setup_fuzzy_system(self):
        """Setup fuzzy inference system for recommendations"""
        
        # User rating linguistic variable
        user_rating = LinguisticVariable("user_rating", (0.0, 10.0))
        user_rating.add_fuzzy_set(FuzzySet("low", FuzzySet.trapezoidal(0, 0, 2, 4)))
        user_rating.add_fuzzy_set(FuzzySet("medium", FuzzySet.triangular(3, 5, 7)))
        user_rating.add_fuzzy_set(FuzzySet("high", FuzzySet.trapezoidal(6, 8, 10, 10)))
        self.fuzzy_system.add_linguistic_variable(user_rating)
        
        # Item popularity linguistic variable
        item_popularity = LinguisticVariable("item_popularity", (0.0, 100.0))
        item_popularity.add_fuzzy_set(FuzzySet("unpopular", FuzzySet.trapezoidal(0, 0, 20, 40)))
        item_popularity.add_fuzzy_set(FuzzySet("moderate", FuzzySet.triangular(30, 50, 70)))
        item_popularity.add_fuzzy_set(FuzzySet("popular", FuzzySet.trapezoidal(60, 80, 100, 100)))
        self.fuzzy_system.add_linguistic_variable(item_popularity)
        
        # Genre match linguistic variable
        genre_match = LinguisticVariable("genre_match", (0.0, 1.0))
        genre_match.add_fuzzy_set(FuzzySet("poor", FuzzySet.trapezoidal(0, 0, 0.2, 0.4)))
        genre_match.add_fuzzy_set(FuzzySet("fair", FuzzySet.triangular(0.3, 0.5, 0.7)))
        genre_match.add_fuzzy_set(FuzzySet("good", FuzzySet.trapezoidal(0.6, 0.8, 1.0, 1.0)))
        self.fuzzy_system.add_linguistic_variable(genre_match)
        
        # Recommendation strength output variable
        rec_strength = LinguisticVariable("recommendation_strength", (0.0, 1.0))
        rec_strength.add_fuzzy_set(FuzzySet("weak", FuzzySet.trapezoidal(0, 0, 0.2, 0.4)))
        rec_strength.add_fuzzy_set(FuzzySet("moderate", FuzzySet.triangular(0.3, 0.5, 0.7)))
        rec_strength.add_fuzzy_set(FuzzySet("strong", FuzzySet.trapezoidal(0.6, 0.8, 1.0, 1.0)))
        self.fuzzy_system.add_linguistic_variable(rec_strength)
        
        # Add fuzzy rules
        self._add_fuzzy_rules()
    
    def _add_fuzzy_rules(self):
        """Add fuzzy inference rules"""
        
        # Rule 1: High rating + Good genre match -> Strong recommendation
        self.fuzzy_system.add_rule(FuzzyRule(
            "rule1",
            {"user_rating": "high", "genre_match": "good"},
            ("recommendation_strength", "strong"),
            weight=1.0
        ))
        
        # Rule 2: High rating + Fair genre match -> Moderate recommendation
        self.fuzzy_system.add_rule(FuzzyRule(
            "rule2",
            {"user_rating": "high", "genre_match": "fair"},
            ("recommendation_strength", "moderate"),
            weight=0.8
        ))
        
        # Rule 3: Medium rating + Good genre match -> Moderate recommendation
        self.fuzzy_system.add_rule(FuzzyRule(
            "rule3",
            {"user_rating": "medium", "genre_match": "good"},
            ("recommendation_strength", "moderate"),
            weight=0.9
        ))
        
        # Rule 4: Low rating -> Weak recommendation (regardless of genre)
        self.fuzzy_system.add_rule(FuzzyRule(
            "rule4",
            {"user_rating": "low"},
            ("recommendation_strength", "weak"),
            weight=1.0
        ))
        
        # Rule 5: Popular item + Fair genre match -> Moderate recommendation
        self.fuzzy_system.add_rule(FuzzyRule(
            "rule5",
            {"item_popularity": "popular", "genre_match": "fair"},
            ("recommendation_strength", "moderate"),
            weight=0.6
        ))
        
        # Rule 6: Unpopular item + Good genre match -> Moderate recommendation (discovery)
        self.fuzzy_system.add_rule(FuzzyRule(
            "rule6",
            {"item_popularity": "unpopular", "genre_match": "good"},
            ("recommendation_strength", "moderate"),
            weight=0.7
        ))
        
        # Rule 7: Poor genre match -> Weak recommendation (regardless of other factors)
        self.fuzzy_system.add_rule(FuzzyRule(
            "rule7",
            {"genre_match": "poor"},
            ("recommendation_strength", "weak"),
            weight=0.9
        ))
    
    def compute_recommendation_strength(self, user_rating: float, 
                                      item_popularity: float, 
                                      genre_match: float) -> float:
        """Compute recommendation strength using fuzzy inference"""
        
        inputs = {
            "user_rating": user_rating,
            "item_popularity": item_popularity,
            "genre_match": genre_match
        }
        
        outputs = self.fuzzy_system.infer(inputs)
        return outputs.get("recommendation_strength", 0.5)
    
    def recommend_items(self, user_profile: Dict, candidate_items: List[Dict], 
                       k: int = 10) -> List[Tuple[Dict, float, str]]:
        """Generate fuzzy logic-based recommendations"""
        
        recommendations = []
        
        for item in candidate_items:
            # Compute input features
            user_rating = self._compute_user_rating_estimate(user_profile, item)
            item_popularity = self._compute_item_popularity(item)
            genre_match = self._compute_genre_match(user_profile, item)
            
            # Compute recommendation strength
            strength = self.compute_recommendation_strength(
                user_rating, item_popularity, genre_match
            )
            
            # Generate explanation
            explanation = self._generate_fuzzy_explanation(
                user_rating, item_popularity, genre_match, strength
            )
            
            recommendations.append((item, strength, explanation))
        
        # Sort by strength and return top-k
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:k]
    
    def _compute_user_rating_estimate(self, user_profile: Dict, item: Dict) -> float:
        """Estimate user rating for item"""
        # Simplified rating estimation
        base_rating = item.get('average_rating', 5.0)
        
        # Adjust based on user preferences
        adjustment = 0.0
        
        # Genre preference adjustment
        user_genres = set(user_profile.get('favorite_genres', []))
        item_genres = set(item.get('genres', []))
        
        if user_genres.intersection(item_genres):
            adjustment += 1.0
        
        # Recency preference
        if item.get('year', 2000) >= user_profile.get('min_year', 2000):
            adjustment += 0.5
        
        return min(10.0, max(0.0, base_rating + adjustment))
    
    def _compute_item_popularity(self, item: Dict) -> float:
        """Compute item popularity score"""
        # Simplified popularity based on ratings count
        rating_count = item.get('rating_count', 0)
        
        # Normalize to 0-100 scale
        if rating_count >= 10000:
            return 100.0
        elif rating_count >= 1000:
            return 80.0
        elif rating_count >= 100:
            return 50.0
        elif rating_count >= 10:
            return 20.0
        else:
            return 5.0
    
    def _compute_genre_match(self, user_profile: Dict, item: Dict) -> float:
        """Compute genre match score"""
        user_genres = set(user_profile.get('favorite_genres', []))
        item_genres = set(item.get('genres', []))
        
        if not user_genres or not item_genres:
            return 0.5  # Neutral
        
        intersection = len(user_genres.intersection(item_genres))
        union = len(user_genres.union(item_genres))
        
        # Jaccard similarity
        return intersection / union if union > 0 else 0.0
    
    def _generate_fuzzy_explanation(self, user_rating: float, item_popularity: float,
                                   genre_match: float, strength: float) -> str:
        """Generate human-readable explanation"""
        
        explanation = f"Recommendation strength: {strength:.2f}\n"
        explanation += "Based on:\n"
        explanation += f"• Estimated user rating: {user_rating:.1f}/10\n"
        explanation += f"• Item popularity: {item_popularity:.0f}/100\n"
        explanation += f"• Genre match: {genre_match:.2f}\n"
        
        # Add linguistic interpretation
        if strength >= 0.7:
            explanation += "→ Strong recommendation due to high compatibility"
        elif strength >= 0.4:
            explanation += "→ Moderate recommendation with some appeal"
        else:
            explanation += "→ Weak recommendation, limited appeal"
        
        return explanation

def demo_fuzzy_recommendation_system():
    """Demonstrate fuzzy recommendation system"""
    
    # Create fuzzy system
    fuzzy_rec = FuzzyRecommendationSystem()
    
    # User profile
    user_profile = {
        'favorite_genres': ['Action', 'Sci-Fi'],
        'min_year': 2010
    }
    
    # Candidate items
    candidate_items = [
        {
            'id': 'm1',
            'title': 'Inception',
            'genres': ['Sci-Fi', 'Thriller'],
            'average_rating': 8.8,
            'rating_count': 15000,
            'year': 2010
        },
        {
            'id': 'm2',
            'title': 'The Notebook',
            'genres': ['Romance', 'Drama'],
            'average_rating': 7.8,
            'rating_count': 8000,
            'year': 2004
        },
        {
            'id': 'm3',
            'title': 'Mad Max: Fury Road',
            'genres': ['Action', 'Adventure'],
            'average_rating': 8.1,
            'rating_count': 12000,
            'year': 2015
        }
    ]
    
    # Generate recommendations
    recommendations = fuzzy_rec.recommend_items(user_profile, candidate_items, k=5)
    
    print("Fuzzy Logic-Based Recommendations:")
    print("=" * 50)
    
    for i, (item, strength, explanation) in enumerate(recommendations, 1):
        print(f"{i}. {item['title']}")
        print(f"   Genres: {', '.join(item['genres'])}")
        print(f"   Rating: {item['average_rating']}/10")
        print(f"   Recommendation Strength: {strength:.3f}")
        print(f"   Explanation:\n{explanation}")
        print("-" * 30)

if __name__ == "__main__":
    demo_fuzzy_recommendation_system()
```

## 3. Interactive Constraint Refinement

### Conversational Recommendation Interface

```python
from typing import Optional, Set
from collections import deque

class InteractionState:
    """Represents the state of user interaction"""
    
    def __init__(self):
        self.preferences = {}
        self.constraints = []
        self.rejected_items = set()
        self.liked_items = set()
        self.interaction_history = deque(maxlen=50)
        self.current_recommendations = []
        self.user_feedback = {}
    
    def add_preference(self, preference_type: str, value: Any):
        """Add user preference"""
        if preference_type not in self.preferences:
            self.preferences[preference_type] = []
        
        if value not in self.preferences[preference_type]:
            self.preferences[preference_type].append(value)
    
    def remove_preference(self, preference_type: str, value: Any):
        """Remove user preference"""
        if preference_type in self.preferences:
            if value in self.preferences[preference_type]:
                self.preferences[preference_type].remove(value)
    
    def add_interaction(self, interaction_type: str, content: str):
        """Record user interaction"""
        self.interaction_history.append({
            'type': interaction_type,
            'content': content,
            'timestamp': time.time()
        })

class ConversationalRecommendationAgent:
    """
    Interactive recommendation agent with constraint refinement
    """
    
    def __init__(self, items: List[Dict]):
        self.items = items
        self.session_states = {}  # session_id -> InteractionState
        self.constraint_templates = self._create_constraint_templates()
        
    def start_session(self, session_id: str) -> str:
        """Start new recommendation session"""
        self.session_states[session_id] = InteractionState()
        
        welcome_message = """
        Welcome to the Interactive Recommendation System!
        
        I'll help you find items that match your preferences. You can:
        • Tell me what you like: "I like action movies"
        • Set constraints: "I want movies after 2010"
        • Give feedback: "I don't like this recommendation"
        • Ask for explanations: "Why did you recommend this?"
        
        What kind of items are you looking for today?
        """
        
        return welcome_message.strip()
    
    def process_user_input(self, session_id: str, user_input: str) -> str:
        """Process user input and return system response"""
        
        if session_id not in self.session_states:
            return self.start_session(session_id)
        
        state = self.session_states[session_id]
        state.add_interaction("user_input", user_input)
        
        # Parse user input
        intent, entities = self._parse_user_input(user_input)
        
        # Process based on intent
        if intent == "preference":
            response = self._handle_preference_input(state, entities)
        elif intent == "constraint":
            response = self._handle_constraint_input(state, entities)
        elif intent == "feedback":
            response = self._handle_feedback_input(state, entities)
        elif intent == "explanation":
            response = self._handle_explanation_request(state, entities)
        elif intent == "recommendation_request":
            response = self._generate_recommendations_response(state)
        else:
            response = self._handle_general_input(state, user_input)
        
        state.add_interaction("system_response", response)
        return response
    
    def _parse_user_input(self, user_input: str) -> Tuple[str, Dict]:
        """Parse user input to extract intent and entities"""
        input_lower = user_input.lower()
        
        # Simple rule-based parsing (could be enhanced with NLP)
        if any(phrase in input_lower for phrase in ["i like", "i enjoy", "i prefer"]):
            return "preference", self._extract_preferences(user_input)
        
        elif any(phrase in input_lower for phrase in ["i want", "must be", "should be", "after", "before"]):
            return "constraint", self._extract_constraints(user_input)
        
        elif any(phrase in input_lower for phrase in ["don't like", "not good", "bad", "terrible"]):
            return "feedback", {"type": "negative", "content": user_input}
        
        elif any(phrase in input_lower for phrase in ["good", "great", "love it", "excellent"]):
            return "feedback", {"type": "positive", "content": user_input}
        
        elif any(phrase in input_lower for phrase in ["why", "explain", "reason"]):
            return "explanation", {"request": user_input}
        
        elif any(phrase in input_lower for phrase in ["recommend", "suggest", "show me"]):
            return "recommendation_request", {}
        
        else:
            return "general", {"content": user_input}
    
    def _extract_preferences(self, user_input: str) -> Dict:
        """Extract preferences from user input"""
        entities = {}
        input_lower = user_input.lower()
        
        # Extract genres
        genres = ["action", "comedy", "drama", "horror", "romance", "sci-fi", "thriller"]
        for genre in genres:
            if genre in input_lower:
                entities.setdefault("genres", []).append(genre.title())
        
        # Extract other preferences (simplified)
        if "new" in input_lower or "recent" in input_lower:
            entities["recency"] = "recent"
        
        if "classic" in input_lower or "old" in input_lower:
            entities["recency"] = "classic"
        
        return entities
    
    def _extract_constraints(self, user_input: str) -> Dict:
        """Extract constraints from user input"""
        entities = {}
        input_lower = user_input.lower()
        
        # Extract year constraints
        import re
        year_match = re.search(r'after (\d{4})', input_lower)
        if year_match:
            entities["min_year"] = int(year_match.group(1))
        
        year_match = re.search(r'before (\d{4})', input_lower)
        if year_match:
            entities["max_year"] = int(year_match.group(1))
        
        # Extract rating constraints
        rating_match = re.search(r'rating.*?(\d+\.?\d*)', input_lower)
        if rating_match:
            entities["min_rating"] = float(rating_match.group(1))
        
        return entities
    
    def _handle_preference_input(self, state: InteractionState, entities: Dict) -> str:
        """Handle preference input from user"""
        response_parts = ["Got it! I've noted your preferences:"]
        
        for pref_type, values in entities.items():
            if isinstance(values, list):
                for value in values:
                    state.add_preference(pref_type, value)
                    response_parts.append(f"• You like {pref_type}: {', '.join(values)}")
            else:
                state.add_preference(pref_type, values)
                response_parts.append(f"• {pref_type}: {values}")
        
        response_parts.append("\nWould you like me to find recommendations based on these preferences?")
        
        return "\n".join(response_parts)
    
    def _handle_constraint_input(self, state: InteractionState, entities: Dict) -> str:
        """Handle constraint input from user"""
        response_parts = ["I've added your constraints:"]
        
        for constraint_type, value in entities.items():
            # Create constraint object
            constraint = {
                'type': constraint_type,
                'value': value,
                'description': self._format_constraint_description(constraint_type, value)
            }
            state.constraints.append(constraint)
            response_parts.append(f"• {constraint['description']}")
        
        response_parts.append("\nShall I find items that meet these criteria?")
        
        return "\n".join(response_parts)
    
    def _handle_feedback_input(self, state: InteractionState, entities: Dict) -> str:
        """Handle user feedback on recommendations"""
        feedback_type = entities.get("type", "neutral")
        
        if state.current_recommendations:
            if feedback_type == "negative":
                # Add current recommendations to rejected items
                for item in state.current_recommendations[:1]:  # Assume feedback on first item
                    state.rejected_items.add(item['id'])
                
                response = "I understand you don't like this recommendation. Let me adjust my suggestions."
                response += "\n\nWhat specifically didn't you like about it? (genre, year, style, etc.)"
                
            elif feedback_type == "positive":
                # Add to liked items and learn preferences
                for item in state.current_recommendations[:1]:
                    state.liked_items.add(item['id'])
                    
                    # Extract preferences from liked item
                    for genre in item.get('genres', []):
                        state.add_preference('genres', genre)
                
                response = "Great! I'm glad you liked it. I'll look for more similar items."
                response += "\n\nWould you like me to find more recommendations like this one?"
            
            else:
                response = "Thank you for your feedback. Any specific aspects you'd like me to adjust?"
        else:
            response = "I don't have any current recommendations to get feedback on. Would you like me to make some suggestions?"
        
        return response
    
    def _handle_explanation_request(self, state: InteractionState, entities: Dict) -> str:
        """Handle request for explanation"""
        if not state.current_recommendations:
            return "I don't have any current recommendations to explain. Would you like me to make some suggestions first?"
        
        # Explain first recommendation
        item = state.current_recommendations[0]
        explanation = f"I recommended '{item['title']}' because:\n\n"
        
        # Check preference matches
        matches = []
        if 'genres' in state.preferences:
            user_genres = set(state.preferences['genres'])
            item_genres = set(item.get('genres', []))
            common_genres = user_genres.intersection(item_genres)
            if common_genres:
                matches.append(f"It matches your preferred genres: {', '.join(common_genres)}")
        
        # Check constraint satisfaction
        for constraint in state.constraints:
            if self._check_constraint_satisfaction(item, constraint):
                matches.append(f"It meets your requirement: {constraint['description']}")
        
        if matches:
            explanation += "\n".join(f"• {match}" for match in matches)
        else:
            explanation += "• It's a popular, well-rated item that might interest you"
        
        explanation += f"\n\nRating: {item.get('rating', 'N/A')}/10"
        explanation += f"\nYear: {item.get('year', 'N/A')}"
        
        return explanation
    
    def _generate_recommendations_response(self, state: InteractionState) -> str:
        """Generate recommendations based on current state"""
        
        # Filter items based on preferences and constraints
        candidate_items = self._filter_items(state)
        
        if not candidate_items:
            return "I couldn't find any items matching your preferences. Would you like to relax some constraints?"
        
        # Rank items
        ranked_items = self._rank_items(state, candidate_items)
        
        # Select top recommendations
        top_recommendations = ranked_items[:3]
        state.current_recommendations = top_recommendations
        
        # Format response
        response = "Here are my recommendations for you:\n\n"
        
        for i, item in enumerate(top_recommendations, 1):
            response += f"{i}. {item['title']}"
            if 'year' in item:
                response += f" ({item['year']})"
            response += f"\n   Genres: {', '.join(item.get('genres', []))}"
            if 'rating' in item:
                response += f"\n   Rating: {item['rating']}/10"
            response += "\n\n"
        
        response += "What do you think of these suggestions? You can ask for explanations or give me feedback!"
        
        return response
    
    def _filter_items(self, state: InteractionState) -> List[Dict]:
        """Filter items based on user preferences and constraints"""
        filtered_items = []
        
        for item in self.items:
            # Skip rejected items
            if item['id'] in state.rejected_items:
                continue
            
            # Check constraints
            satisfies_constraints = True
            for constraint in state.constraints:
                if not self._check_constraint_satisfaction(item, constraint):
                    satisfies_constraints = False
                    break
            
            if satisfies_constraints:
                filtered_items.append(item)
        
        return filtered_items
    
    def _rank_items(self, state: InteractionState, items: List[Dict]) -> List[Dict]:
        """Rank items based on user preferences"""
        scored_items = []
        
        for item in items:
            score = 0.0
            
            # Base score from item rating
            score += item.get('rating', 5.0) * 0.3
            
            # Preference-based scoring
            if 'genres' in state.preferences:
                user_genres = set(state.preferences['genres'])
                item_genres = set(item.get('genres', []))
                genre_match = len(user_genres.intersection(item_genres)) / len(user_genres)
                score += genre_match * 3.0
            
            # Boost for liked similar items
            for liked_id in state.liked_items:
                liked_item = next((it for it in self.items if it['id'] == liked_id), None)
                if liked_item:
                    similarity = self._compute_item_similarity(item, liked_item)
                    score += similarity * 2.0
            
            # Recency preference
            if 'recency' in state.preferences:
                if state.preferences['recency'] == 'recent':
                    if item.get('year', 2000) >= 2015:
                        score += 1.0
                elif state.preferences['recency'] == 'classic':
                    if item.get('year', 2000) <= 2000:
                        score += 1.0
            
            scored_items.append((item, score))
        
        # Sort by score
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        return [item for item, score in scored_items]
    
    def _check_constraint_satisfaction(self, item: Dict, constraint: Dict) -> bool:
        """Check if item satisfies constraint"""
        constraint_type = constraint['type']
        value = constraint['value']
        
        if constraint_type == 'min_year':
            return item.get('year', 0) >= value
        elif constraint_type == 'max_year':
            return item.get('year', 9999) <= value
        elif constraint_type == 'min_rating':
            return item.get('rating', 0) >= value
        
        return True
    
    def _compute_item_similarity(self, item1: Dict, item2: Dict) -> float:
        """Compute similarity between items"""
        # Simple genre-based similarity
        genres1 = set(item1.get('genres', []))
        genres2 = set(item2.get('genres', []))
        
        if not genres1 or not genres2:
            return 0.0
        
        intersection = len(genres1.intersection(genres2))
        union = len(genres1.union(genres2))
        
        return intersection / union
    
    def _format_constraint_description(self, constraint_type: str, value: Any) -> str:
        """Format constraint for user display"""
        if constraint_type == 'min_year':
            return f"Released after {value}"
        elif constraint_type == 'max_year':
            return f"Released before {value}"
        elif constraint_type == 'min_rating':
            return f"Rating at least {value}/10"
        
        return f"{constraint_type}: {value}"
    
    def _handle_general_input(self, state: InteractionState, user_input: str) -> str:
        """Handle general user input"""
        responses = [
            "I'm here to help you find great recommendations! You can tell me what you like, set constraints, or ask for suggestions.",
            "Would you like me to recommend some items? Just tell me your preferences or constraints.",
            "I can help you find items that match your taste. What kind of things do you enjoy?"
        ]
        
        return random.choice(responses)
    
    def _create_constraint_templates(self) -> Dict:
        """Create constraint templates for various domains"""
        return {
            'movie': {
                'genre': "Movies in {genre} genre",
                'year_range': "Movies from {start_year} to {end_year}",
                'rating': "Movies with rating above {min_rating}",
                'duration': "Movies shorter than {max_duration} minutes"
            },
            'book': {
                'genre': "Books in {genre} category",
                'author': "Books by author {author}",
                'publication_year': "Books published after {year}",
                'page_count': "Books with less than {max_pages} pages"
            }
        }

def demo_conversational_system():
    """Demonstrate conversational recommendation system"""
    
    # Sample movie data
    movies = [
        {'id': 'm1', 'title': 'Inception', 'genres': ['Sci-Fi', 'Thriller'], 'rating': 8.8, 'year': 2010},
        {'id': 'm2', 'title': 'The Dark Knight', 'genres': ['Action', 'Crime'], 'rating': 9.0, 'year': 2008},
        {'id': 'm3', 'title': 'Pulp Fiction', 'genres': ['Crime', 'Drama'], 'rating': 8.9, 'year': 1994},
        {'id': 'm4', 'title': 'The Avengers', 'genres': ['Action', 'Adventure'], 'rating': 8.0, 'year': 2012},
        {'id': 'm5', 'title': 'Titanic', 'genres': ['Romance', 'Drama'], 'rating': 7.8, 'year': 1997}
    ]
    
    # Create conversational agent
    agent = ConversationalRecommendationAgent(movies)
    
    # Simulate conversation
    session_id = "demo_session"
    
    print("=== Conversational Recommendation Demo ===")
    print()
    
    # Start session
    welcome = agent.start_session(session_id)
    print("System:", welcome)
    print()
    
    # Simulate user interactions
    user_inputs = [
        "I like action movies",
        "I want movies after 2010",
        "Can you recommend something?",
        "Why did you recommend Inception?",
        "I love it! More like this please"
    ]
    
    for user_input in user_inputs:
        print(f"User: {user_input}")
        response = agent.process_user_input(session_id, user_input)
        print(f"System: {response}")
        print("-" * 50)

if __name__ == "__main__":
    demo_conversational_system()
```

## 4. Study Questions

### Beginner Level

1. What are soft constraints and how do they differ from hard constraints in recommendation systems?
2. Explain the basic components of a fuzzy inference system.
3. How can multi-objective optimization be applied to recommendation problems?
4. What are the advantages of interactive constraint refinement in recommendations?
5. How do fuzzy sets help handle uncertain user preferences?

### Intermediate Level

6. Implement a soft constraint system for restaurant recommendations with multiple objectives.
7. How would you design a fuzzy rule system for music recommendations based on mood and activity?
8. Create an interactive system that can learn user preferences through conversation.
9. What are the computational challenges of solving multi-objective recommendation problems?
10. How would you handle conflicting user preferences in a constraint-based system?

### Advanced Level

11. Implement a constraint satisfaction system that can handle temporal preferences and scheduling constraints.
12. Design a fuzzy inference system that can adapt its rules based on user feedback over time.
13. How would you scale interactive constraint refinement to handle millions of items and complex user models?
14. Implement a multi-objective optimization system that can find Pareto-optimal recommendation sets efficiently.
15. Design a conversational agent that can negotiate trade-offs between conflicting user preferences.

### Tricky Questions

16. How would you handle the cold start problem in constraint-based recommendation systems?
17. Design a system that can automatically generate fuzzy rules from user interaction data.
18. How would you ensure fairness and avoid bias in constraint-based recommendation systems?
19. Implement a system that can handle both individual and group preferences with conflicting constraints.
20. How would you design a constraint-based system that can work across different domains while maintaining semantic consistency?

## Key Takeaways

1. **Soft constraints** enable flexible optimization with trade-offs
2. **Fuzzy logic** naturally handles uncertain and imprecise preferences
3. **Multi-objective optimization** balances competing recommendation goals
4. **Interactive refinement** improves recommendations through user engagement
5. **Conversational interfaces** make constraint specification more natural
6. **Pareto optimality** helps find non-dominated recommendation solutions
7. **Adaptive systems** can learn and evolve constraint satisfaction strategies

## Course Day 4 Summary

Today we completed **Content-Based Recommendation Systems**, covering:

### Day 4.1: Content-Based Fundamentals
- Core principles and architecture
- Item and user profiling techniques
- Similarity computation methods
- Advantages and limitations

### Day 4.2: Feature Extraction and Representation
- Advanced text processing techniques
- Numerical and categorical feature engineering
- Dimensionality reduction methods
- Domain-specific feature extraction

### Day 4.3: Text Processing and NLP
- Advanced text preprocessing pipelines
- Word embeddings and semantic similarity
- Topic modeling and sentiment analysis
- Multi-language content processing

### Day 4.4: Content Similarity and Matching
- Advanced similarity metrics
- Locality-sensitive hashing for scalability
- Graph-based similarity measures
- Real-time similarity computation

### Day 4.5: Knowledge-Based Systems
- Ontology-based content representation
- Rule-based recommendation engines
- Semantic reasoning and inference
- Knowledge graph integration

### Day 4.6: Constraint-Based and Rule-Based Systems
- Soft constraints and optimization
- Fuzzy rule systems for uncertainty
- Multi-objective recommendation optimization
- Interactive constraint refinement

## Next Session Preview

**Day 5: Hybrid Recommendation Systems** will cover:
- Combining collaborative and content-based approaches
- Weighted and switching hybrid methods
- Meta-learning for recommendation fusion
- Dynamic hybridization strategies
- Evaluation of hybrid systems