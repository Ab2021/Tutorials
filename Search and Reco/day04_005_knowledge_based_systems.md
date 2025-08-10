# Day 4.5: Knowledge-Based Recommendation Systems

## Learning Objectives
By the end of this session, you will:
- Understand ontology-based content representation for recommendations
- Implement rule-based recommendation engines with expert knowledge
- Apply constraint satisfaction techniques in recommendation scenarios
- Build knowledge graph-powered recommendation systems
- Design conversational and interactive recommendation interfaces

## 1. Ontology-Based Content Representation

### Knowledge Representation Frameworks

```python
import json
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Any, Optional
import re

class Ontology:
    """
    Ontology-based knowledge representation for content recommendations
    """
    
    def __init__(self, name: str):
        self.name = name
        
        # Core ontology structures
        self.concepts = {}  # concept_id -> concept_info
        self.relationships = defaultdict(list)  # (concept1, concept2) -> relationship_type
        self.properties = defaultdict(dict)  # concept_id -> {property: value}
        self.instances = defaultdict(list)  # concept_id -> [instance_ids]
        
        # Hierarchy information
        self.parent_child = defaultdict(list)  # parent -> [children]
        self.child_parent = {}  # child -> parent
        
        # Inference rules
        self.inference_rules = []
        
    def add_concept(self, concept_id: str, concept_info: Dict[str, Any]):
        """Add a concept to the ontology"""
        self.concepts[concept_id] = {
            'id': concept_id,
            'name': concept_info.get('name', concept_id),
            'description': concept_info.get('description', ''),
            'synonyms': concept_info.get('synonyms', []),
            'properties': concept_info.get('properties', {})
        }
        
        # Set default properties
        for prop, value in concept_info.get('properties', {}).items():
            self.properties[concept_id][prop] = value
    
    def add_relationship(self, concept1: str, concept2: str, relationship_type: str, 
                        properties: Dict = None):
        """Add relationship between concepts"""
        relationship = {
            'type': relationship_type,
            'source': concept1,
            'target': concept2,
            'properties': properties or {}
        }
        
        self.relationships[(concept1, concept2)].append(relationship)
        
        # Handle hierarchical relationships
        if relationship_type in ['is_a', 'subclass_of']:
            self.parent_child[concept2].append(concept1)
            self.child_parent[concept1] = concept2
    
    def add_instance(self, concept_id: str, instance_id: str, instance_data: Dict):
        """Add instance of a concept"""
        if concept_id not in self.concepts:
            raise ValueError(f"Concept {concept_id} not found")
        
        self.instances[concept_id].append({
            'id': instance_id,
            'data': instance_data,
            'concept': concept_id
        })
    
    def get_concept_hierarchy(self, concept_id: str) -> List[str]:
        """Get hierarchical path from concept to root"""
        hierarchy = [concept_id]
        current = concept_id
        
        while current in self.child_parent:
            parent = self.child_parent[current]
            hierarchy.append(parent)
            current = parent
        
        return hierarchy
    
    def get_related_concepts(self, concept_id: str, relationship_type: str = None) -> List[Dict]:
        """Get concepts related to given concept"""
        related = []
        
        # Check outgoing relationships
        for (source, target), relationships in self.relationships.items():
            if source == concept_id:
                for rel in relationships:
                    if relationship_type is None or rel['type'] == relationship_type:
                        related.append({
                            'concept': target,
                            'relationship': rel['type'],
                            'direction': 'outgoing'
                        })
        
        # Check incoming relationships
        for (source, target), relationships in self.relationships.items():
            if target == concept_id:
                for rel in relationships:
                    if relationship_type is None or rel['type'] == relationship_type:
                        related.append({
                            'concept': source,
                            'relationship': rel['type'],
                            'direction': 'incoming'
                        })
        
        return related
    
    def compute_concept_similarity(self, concept1: str, concept2: str) -> float:
        """Compute semantic similarity between concepts"""
        if concept1 == concept2:
            return 1.0
        
        if concept1 not in self.concepts or concept2 not in self.concepts:
            return 0.0
        
        # Get hierarchical paths
        path1 = self.get_concept_hierarchy(concept1)
        path2 = self.get_concept_hierarchy(concept2)
        
        # Find common ancestors
        common_ancestors = set(path1).intersection(set(path2))
        
        if not common_ancestors:
            return 0.0
        
        # Use closest common ancestor for similarity
        # Find the common ancestor with shortest distance
        min_distance = float('inf')
        for ancestor in common_ancestors:
            dist1 = path1.index(ancestor)
            dist2 = path2.index(ancestor)
            total_dist = dist1 + dist2
            min_distance = min(min_distance, total_dist)
        
        # Convert distance to similarity (closer = more similar)
        max_depth = 10  # Assume maximum meaningful depth
        similarity = max(0, 1.0 - (min_distance / (2 * max_depth)))
        
        return similarity
    
    def add_inference_rule(self, rule_name: str, conditions: List[Dict], conclusions: List[Dict]):
        """Add inference rule to ontology"""
        rule = {
            'name': rule_name,
            'conditions': conditions,  # List of condition dictionaries
            'conclusions': conclusions  # List of conclusion dictionaries
        }
        self.inference_rules.append(rule)
    
    def apply_inference_rules(self, facts: List[Dict]) -> List[Dict]:
        """Apply inference rules to derive new facts"""
        new_facts = []
        
        for rule in self.inference_rules:
            if self._check_rule_conditions(rule['conditions'], facts):
                for conclusion in rule['conclusions']:
                    new_facts.append(conclusion)
        
        return new_facts
    
    def _check_rule_conditions(self, conditions: List[Dict], facts: List[Dict]) -> bool:
        """Check if rule conditions are satisfied by facts"""
        for condition in conditions:
            if not self._check_single_condition(condition, facts):
                return False
        return True
    
    def _check_single_condition(self, condition: Dict, facts: List[Dict]) -> bool:
        """Check if single condition is satisfied"""
        # Simplified condition checking
        for fact in facts:
            if all(key in fact and fact[key] == value for key, value in condition.items()):
                return True
        return False

class MovieOntology(Ontology):
    """
    Specialized ontology for movie recommendations
    """
    
    def __init__(self):
        super().__init__("MovieOntology")
        self._build_movie_ontology()
    
    def _build_movie_ontology(self):
        """Build movie domain ontology"""
        
        # Add core concepts
        self.add_concept("Movie", {
            "name": "Movie",
            "description": "A motion picture",
            "properties": {"duration": "int", "release_year": "int", "rating": "float"}
        })
        
        self.add_concept("Genre", {
            "name": "Genre",
            "description": "Movie genre classification",
            "properties": {"popularity": "float"}
        })
        
        self.add_concept("Person", {
            "name": "Person",
            "description": "Person involved in movie production",
            "properties": {"birth_year": "int", "nationality": "str"}
        })
        
        self.add_concept("Actor", {"name": "Actor", "description": "Movie actor"})
        self.add_concept("Director", {"name": "Director", "description": "Movie director"})
        
        # Add hierarchical relationships
        self.add_relationship("Actor", "Person", "is_a")
        self.add_relationship("Director", "Person", "is_a")
        
        # Add genre concepts
        genres = [
            "Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", 
            "Thriller", "Documentary", "Animation", "Adventure"
        ]
        
        for genre in genres:
            self.add_concept(genre, {
                "name": genre,
                "description": f"{genre} genre"
            })
            self.add_relationship(genre, "Genre", "is_a")
        
        # Add inference rules
        self.add_inference_rule(
            "similar_movies_rule",
            conditions=[
                {"type": "same_genre", "genre": "?genre"},
                {"type": "same_director", "director": "?director"}
            ],
            conclusions=[
                {"type": "similar_movies", "similarity": 0.8}
            ]
        )
    
    def add_movie(self, movie_id: str, title: str, genres: List[str], 
                  director: str, actors: List[str], year: int, rating: float):
        """Add movie instance to ontology"""
        movie_data = {
            "title": title,
            "genres": genres,
            "director": director,
            "actors": actors,
            "year": year,
            "rating": rating
        }
        
        self.add_instance("Movie", movie_id, movie_data)
        
        # Add relationships
        for genre in genres:
            if genre not in self.concepts:
                self.add_concept(genre, {"name": genre, "description": f"{genre} genre"})
                self.add_relationship(genre, "Genre", "is_a")
            
            self.add_relationship(movie_id, genre, "has_genre")
        
        # Add director relationship
        if director not in self.concepts:
            self.add_concept(director, {"name": director, "description": "Movie director"})
            self.add_relationship(director, "Director", "is_a")
        
        self.add_relationship(movie_id, director, "directed_by")
        
        # Add actor relationships
        for actor in actors:
            if actor not in self.concepts:
                self.add_concept(actor, {"name": actor, "description": "Movie actor"})
                self.add_relationship(actor, "Actor", "is_a")
            
            self.add_relationship(movie_id, actor, "stars")
    
    def find_similar_movies(self, movie_id: str, similarity_threshold: float = 0.5) -> List[Tuple[str, float]]:
        """Find movies similar to given movie using ontological reasoning"""
        if movie_id not in [instance['id'] for instances in self.instances.values() 
                           for instance in instances]:
            return []
        
        # Get movie data
        movie_data = None
        for instances in self.instances.values():
            for instance in instances:
                if instance['id'] == movie_id:
                    movie_data = instance['data']
                    break
        
        if not movie_data:
            return []
        
        similar_movies = []
        
        # Find all other movies
        for instances in self.instances.values():
            for instance in instances:
                if instance['concept'] == 'Movie' and instance['id'] != movie_id:
                    similarity = self._compute_movie_similarity(movie_data, instance['data'])
                    
                    if similarity >= similarity_threshold:
                        similar_movies.append((instance['id'], similarity))
        
        # Sort by similarity
        similar_movies.sort(key=lambda x: x[1], reverse=True)
        return similar_movies
    
    def _compute_movie_similarity(self, movie1: Dict, movie2: Dict) -> float:
        """Compute similarity between two movies"""
        similarity = 0.0
        
        # Genre similarity
        genres1 = set(movie1.get('genres', []))
        genres2 = set(movie2.get('genres', []))
        
        if genres1 and genres2:
            genre_similarity = len(genres1.intersection(genres2)) / len(genres1.union(genres2))
            similarity += 0.4 * genre_similarity
        
        # Director similarity
        if movie1.get('director') == movie2.get('director'):
            similarity += 0.3
        
        # Actor similarity
        actors1 = set(movie1.get('actors', []))
        actors2 = set(movie2.get('actors', []))
        
        if actors1 and actors2:
            actor_similarity = len(actors1.intersection(actors2)) / max(len(actors1), len(actors2))
            similarity += 0.2 * actor_similarity
        
        # Year similarity (recent movies are more similar)
        year_diff = abs(movie1.get('year', 0) - movie2.get('year', 0))
        year_similarity = max(0, 1.0 - year_diff / 50.0)  # 50 year max difference
        similarity += 0.1 * year_similarity
        
        return similarity

class SemanticRecommendationEngine:
    """
    Recommendation engine based on semantic ontology
    """
    
    def __init__(self, ontology: Ontology):
        self.ontology = ontology
        self.user_profiles = {}  # user_id -> semantic profile
        self.item_concepts = {}  # item_id -> list of concepts
    
    def build_user_semantic_profile(self, user_id: str, liked_items: List[str]):
        """Build semantic profile for user based on liked items"""
        concept_counts = defaultdict(int)
        concept_weights = defaultdict(float)
        
        for item_id in liked_items:
            if item_id in self.item_concepts:
                item_concepts = self.item_concepts[item_id]
                
                for concept in item_concepts:
                    concept_counts[concept] += 1
                    
                    # Weight by concept hierarchy level (more specific = higher weight)
                    hierarchy = self.ontology.get_concept_hierarchy(concept)
                    weight = 1.0 + 0.2 * len(hierarchy)  # Deeper concepts get higher weight
                    concept_weights[concept] += weight
        
        # Normalize weights
        total_weight = sum(concept_weights.values())
        if total_weight > 0:
            for concept in concept_weights:
                concept_weights[concept] /= total_weight
        
        # Include related concepts through inference
        inferred_concepts = self._infer_user_interests(concept_weights)
        concept_weights.update(inferred_concepts)
        
        self.user_profiles[user_id] = {
            'concept_weights': dict(concept_weights),
            'concept_counts': dict(concept_counts),
            'liked_items': liked_items
        }
    
    def recommend_items(self, user_id: str, candidate_items: List[str], k: int = 10) -> List[Tuple[str, float]]:
        """Generate semantic recommendations for user"""
        if user_id not in self.user_profiles:
            return []
        
        user_profile = self.user_profiles[user_id]
        recommendations = []
        
        for item_id in candidate_items:
            if item_id not in user_profile['liked_items']:  # Don't recommend already liked items
                score = self._compute_semantic_relevance(user_profile, item_id)
                recommendations.append((item_id, score))
        
        # Sort and return top-k
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:k]
    
    def add_item_concepts(self, item_id: str, concepts: List[str]):
        """Associate item with ontology concepts"""
        self.item_concepts[item_id] = concepts
    
    def explain_recommendation(self, user_id: str, item_id: str) -> str:
        """Explain why an item was recommended"""
        if user_id not in self.user_profiles or item_id not in self.item_concepts:
            return "Unable to explain recommendation"
        
        user_profile = self.user_profiles[user_id]
        item_concepts = self.item_concepts[item_id]
        
        # Find matching concepts
        matching_concepts = []
        for concept in item_concepts:
            if concept in user_profile['concept_weights']:
                weight = user_profile['concept_weights'][concept]
                matching_concepts.append((concept, weight))
        
        matching_concepts.sort(key=lambda x: x[1], reverse=True)
        
        if not matching_concepts:
            return "Recommended based on general preference patterns"
        
        explanation = "Recommended because you showed interest in:\n"
        for concept, weight in matching_concepts[:3]:
            concept_info = self.ontology.concepts.get(concept, {})
            concept_name = concept_info.get('name', concept)
            explanation += f"- {concept_name} (strength: {weight:.2f})\n"
        
        return explanation
    
    def _infer_user_interests(self, concept_weights: Dict[str, float]) -> Dict[str, float]:
        """Infer additional user interests using ontology relationships"""
        inferred = {}
        
        for concept, weight in concept_weights.items():
            # Add parent concepts with reduced weight
            hierarchy = self.ontology.get_concept_hierarchy(concept)
            for i, parent_concept in enumerate(hierarchy[1:], 1):  # Skip the concept itself
                parent_weight = weight * (0.8 ** i)  # Exponential decay
                if parent_concept in inferred:
                    inferred[parent_concept] = max(inferred[parent_concept], parent_weight)
                else:
                    inferred[parent_concept] = parent_weight
            
            # Add related concepts
            related = self.ontology.get_related_concepts(concept)
            for rel_info in related:
                rel_concept = rel_info['concept']
                rel_weight = weight * 0.3  # Related concepts get lower weight
                
                if rel_concept in inferred:
                    inferred[rel_concept] = max(inferred[rel_concept], rel_weight)
                else:
                    inferred[rel_concept] = rel_weight
        
        return inferred
    
    def _compute_semantic_relevance(self, user_profile: Dict, item_id: str) -> float:
        """Compute semantic relevance of item to user"""
        if item_id not in self.item_concepts:
            return 0.0
        
        item_concepts = self.item_concepts[item_id]
        user_concept_weights = user_profile['concept_weights']
        
        relevance = 0.0
        
        for concept in item_concepts:
            # Direct concept match
            if concept in user_concept_weights:
                relevance += user_concept_weights[concept]
            else:
                # Semantic similarity with user concepts
                max_similarity = 0.0
                for user_concept in user_concept_weights:
                    similarity = self.ontology.compute_concept_similarity(concept, user_concept)
                    weighted_similarity = similarity * user_concept_weights[user_concept]
                    max_similarity = max(max_similarity, weighted_similarity)
                
                relevance += max_similarity * 0.7  # Penalty for indirect match
        
        # Normalize by number of item concepts
        if item_concepts:
            relevance /= len(item_concepts)
        
        return relevance
```

## 2. Rule-Based Recommendation Engines

### Expert System Architecture

```python
from enum import Enum
from dataclasses import dataclass
from typing import Union, Callable

class ConditionType(Enum):
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    IN_LIST = "in_list"
    NOT_IN_LIST = "not_in_list"
    CONTAINS = "contains"
    REGEX_MATCH = "regex_match"

@dataclass
class Condition:
    """Represents a condition in a rule"""
    attribute: str
    operator: ConditionType
    value: Any
    weight: float = 1.0

@dataclass
class Action:
    """Represents an action to take when rule fires"""
    action_type: str
    parameters: Dict[str, Any]
    confidence: float = 1.0

class Rule:
    """Represents a single recommendation rule"""
    
    def __init__(self, rule_id: str, name: str, conditions: List[Condition], 
                 actions: List[Action], priority: int = 0):
        self.rule_id = rule_id
        self.name = name
        self.conditions = conditions
        self.actions = actions
        self.priority = priority
        self.fire_count = 0
        self.last_fired = None
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate if rule conditions are met"""
        for condition in self.conditions:
            if not self._evaluate_condition(condition, context):
                return False
        return True
    
    def _evaluate_condition(self, condition: Condition, context: Dict[str, Any]) -> bool:
        """Evaluate a single condition"""
        if condition.attribute not in context:
            return False
        
        value = context[condition.attribute]
        target = condition.value
        
        if condition.operator == ConditionType.EQUALS:
            return value == target
        elif condition.operator == ConditionType.NOT_EQUALS:
            return value != target
        elif condition.operator == ConditionType.GREATER_THAN:
            return value > target
        elif condition.operator == ConditionType.LESS_THAN:
            return value < target
        elif condition.operator == ConditionType.IN_LIST:
            return value in target
        elif condition.operator == ConditionType.NOT_IN_LIST:
            return value not in target
        elif condition.operator == ConditionType.CONTAINS:
            return target in str(value)
        elif condition.operator == ConditionType.REGEX_MATCH:
            return bool(re.match(target, str(value)))
        
        return False

class RuleBasedRecommendationEngine:
    """
    Rule-based recommendation engine with expert knowledge
    """
    
    def __init__(self):
        self.rules = {}  # rule_id -> Rule
        self.rule_groups = defaultdict(list)  # group_name -> [rule_ids]
        self.facts = {}  # Current facts/context
        self.inference_history = []
        
        # Rule execution settings
        self.max_iterations = 100
        self.conflict_resolution = "priority"  # "priority", "recency", "specificity"
    
    def add_rule(self, rule: Rule, group: str = "default"):
        """Add rule to the engine"""
        self.rules[rule.rule_id] = rule
        self.rule_groups[group].append(rule.rule_id)
    
    def add_fact(self, key: str, value: Any):
        """Add fact to working memory"""
        self.facts[key] = value
    
    def remove_fact(self, key: str):
        """Remove fact from working memory"""
        if key in self.facts:
            del self.facts[key]
    
    def generate_recommendations(self, user_context: Dict[str, Any], 
                               candidate_items: List[Dict[str, Any]], 
                               k: int = 10) -> List[Dict[str, Any]]:
        """Generate recommendations using rule-based inference"""
        
        # Set up context
        self.facts.update(user_context)
        
        recommendations = []
        
        for item in candidate_items:
            # Create context for this item
            item_context = {**self.facts, **item}
            
            # Evaluate rules for this item
            item_score = 0.0
            explanations = []
            
            fired_rules = self._evaluate_rules(item_context)
            
            for rule_id, confidence in fired_rules:
                rule = self.rules[rule_id]
                
                # Process actions
                for action in rule.actions:
                    if action.action_type == "recommend":
                        score_boost = action.parameters.get("score_boost", 1.0)
                        item_score += score_boost * confidence * action.confidence
                        
                        explanations.append({
                            "rule": rule.name,
                            "reason": action.parameters.get("reason", ""),
                            "confidence": confidence
                        })
                    
                    elif action.action_type == "penalize":
                        penalty = action.parameters.get("penalty", 0.5)
                        item_score -= penalty * confidence * action.confidence
                        
                        explanations.append({
                            "rule": rule.name,
                            "reason": action.parameters.get("reason", ""),
                            "confidence": confidence,
                            "type": "penalty"
                        })
            
            if item_score > 0:
                recommendations.append({
                    "item": item,
                    "score": item_score,
                    "explanations": explanations
                })
        
        # Sort by score and return top-k
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations[:k]
    
    def _evaluate_rules(self, context: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Evaluate all rules against context"""
        fired_rules = []
        
        # Get applicable rules
        applicable_rules = []
        for rule_id, rule in self.rules.items():
            if rule.evaluate(context):
                applicable_rules.append((rule_id, rule))
        
        # Apply conflict resolution
        if self.conflict_resolution == "priority":
            applicable_rules.sort(key=lambda x: x[1].priority, reverse=True)
        elif self.conflict_resolution == "recency":
            applicable_rules.sort(key=lambda x: x[1].last_fired or 0, reverse=True)
        
        # Fire rules
        for rule_id, rule in applicable_rules:
            confidence = self._compute_rule_confidence(rule, context)
            fired_rules.append((rule_id, confidence))
            
            # Update rule statistics
            rule.fire_count += 1
            rule.last_fired = time.time()
        
        return fired_rules
    
    def _compute_rule_confidence(self, rule: Rule, context: Dict[str, Any]) -> float:
        """Compute confidence for rule firing"""
        total_weight = sum(condition.weight for condition in rule.conditions)
        
        if total_weight == 0:
            return 1.0
        
        # For now, return full confidence if all conditions match
        # Can be enhanced with fuzzy logic
        return 1.0
    
    def explain_recommendation(self, recommendation: Dict[str, Any]) -> str:
        """Generate human-readable explanation for recommendation"""
        explanations = recommendation.get("explanations", [])
        
        if not explanations:
            return "No specific rules triggered for this recommendation."
        
        explanation = f"Recommended with score {recommendation['score']:.2f} because:\n"
        
        for i, exp in enumerate(explanations, 1):
            exp_type = exp.get("type", "positive")
            prefix = "✓" if exp_type != "penalty" else "✗"
            
            explanation += f"{prefix} {exp['rule']}: {exp['reason']} "
            explanation += f"(confidence: {exp['confidence']:.2f})\n"
        
        return explanation

class MovieRecommendationRules:
    """
    Pre-defined rules for movie recommendations
    """
    
    @staticmethod
    def create_movie_rules() -> List[Rule]:
        """Create comprehensive movie recommendation rules"""
        rules = []
        
        # Genre preference rules
        rules.append(Rule(
            rule_id="genre_match",
            name="Genre Preference Match",
            conditions=[
                Condition("user_favorite_genres", ConditionType.CONTAINS, "item_genre")
            ],
            actions=[
                Action("recommend", {
                    "score_boost": 2.0,
                    "reason": "Matches your favorite genre"
                })
            ],
            priority=10
        ))
        
        # Director preference rules
        rules.append(Rule(
            rule_id="director_match",
            name="Favorite Director",
            conditions=[
                Condition("user_favorite_directors", ConditionType.CONTAINS, "item_director")
            ],
            actions=[
                Action("recommend", {
                    "score_boost": 1.5,
                    "reason": "From your favorite director"
                })
            ],
            priority=8
        ))
        
        # Rating-based rules
        rules.append(Rule(
            rule_id="high_rating",
            name="Highly Rated Movie",
            conditions=[
                Condition("item_rating", ConditionType.GREATER_THAN, 8.0)
            ],
            actions=[
                Action("recommend", {
                    "score_boost": 1.0,
                    "reason": "Highly rated by critics and users"
                })
            ],
            priority=5
        ))
        
        # Recency rules
        rules.append(Rule(
            rule_id="recent_movie",
            name="Recent Release",
            conditions=[
                Condition("item_year", ConditionType.GREATER_THAN, 2020)
            ],
            actions=[
                Action("recommend", {
                    "score_boost": 0.5,
                    "reason": "Recent release"
                })
            ],
            priority=3
        ))
        
        # Avoid disliked content
        rules.append(Rule(
            rule_id="avoid_disliked_genre",
            name="Avoid Disliked Genre",
            conditions=[
                Condition("user_disliked_genres", ConditionType.CONTAINS, "item_genre")
            ],
            actions=[
                Action("penalize", {
                    "penalty": 2.0,
                    "reason": "Genre you typically dislike"
                })
            ],
            priority=15
        ))
        
        # Mood-based rules
        rules.append(Rule(
            rule_id="action_for_excitement",
            name="Action for Excitement",
            conditions=[
                Condition("user_mood", ConditionType.EQUALS, "excited"),
                Condition("item_genre", ConditionType.EQUALS, "Action")
            ],
            actions=[
                Action("recommend", {
                    "score_boost": 1.2,
                    "reason": "Action movie perfect for your excited mood"
                })
            ],
            priority=7
        ))
        
        rules.append(Rule(
            rule_id="comedy_for_relaxation",
            name="Comedy for Relaxation",
            conditions=[
                Condition("user_mood", ConditionType.EQUALS, "relaxed"),
                Condition("item_genre", ConditionType.EQUALS, "Comedy")
            ],
            actions=[
                Action("recommend", {
                    "score_boost": 1.2,
                    "reason": "Light comedy perfect for relaxation"
                })
            ],
            priority=7
        ))
        
        # Time-based rules
        rules.append(Rule(
            rule_id="short_movie_weeknight",
            name="Short Movie for Weeknight",
            conditions=[
                Condition("viewing_time", ConditionType.EQUALS, "weeknight"),
                Condition("item_duration", ConditionType.LESS_THAN, 120)
            ],
            actions=[
                Action("recommend", {
                    "score_boost": 0.8,
                    "reason": "Perfect length for a weeknight"
                })
            ],
            priority=4
        ))
        
        # Social context rules
        rules.append(Rule(
            rule_id="family_friendly",
            name="Family Friendly Content",
            conditions=[
                Condition("viewing_context", ConditionType.EQUALS, "family"),
                Condition("item_rating", ConditionType.IN_LIST, ["G", "PG", "PG-13"])
            ],
            actions=[
                Action("recommend", {
                    "score_boost": 1.0,
                    "reason": "Family-friendly content"
                })
            ],
            priority=12
        ))
        
        # Diversity rules
        rules.append(Rule(
            rule_id="explore_new_genre",
            name="Explore New Genre",
            conditions=[
                Condition("user_exploration_mode", ConditionType.EQUALS, True),
                Condition("user_watched_genres", ConditionType.NOT_CONTAINS, "item_genre")
            ],
            actions=[
                Action("recommend", {
                    "score_boost": 0.7,
                    "reason": "Exploring new genres"
                })
            ],
            priority=2
        ))
        
        return rules

# Usage example
def create_movie_recommendation_system():
    """Create a complete movie recommendation system"""
    
    # Create rule engine
    engine = RuleBasedRecommendationEngine()
    
    # Add movie rules
    movie_rules = MovieRecommendationRules.create_movie_rules()
    for rule in movie_rules:
        engine.add_rule(rule, "movie_rules")
    
    return engine

def demo_rule_based_system():
    """Demonstrate rule-based recommendation system"""
    
    # Create system
    engine = create_movie_recommendation_system()
    
    # User context
    user_context = {
        "user_favorite_genres": ["Action", "Thriller"],
        "user_favorite_directors": ["Christopher Nolan", "Quentin Tarantino"],
        "user_disliked_genres": ["Horror"],
        "user_mood": "excited",
        "viewing_time": "weekend",
        "viewing_context": "alone",
        "user_exploration_mode": False,
        "user_watched_genres": ["Action", "Drama", "Comedy"]
    }
    
    # Candidate movies
    candidates = [
        {
            "item_id": "movie1",
            "item_title": "Inception",
            "item_genre": "Action",
            "item_director": "Christopher Nolan",
            "item_rating": 8.8,
            "item_year": 2010,
            "item_duration": 148
        },
        {
            "item_id": "movie2", 
            "item_title": "The Conjuring",
            "item_genre": "Horror",
            "item_director": "James Wan",
            "item_rating": 7.5,
            "item_year": 2013,
            "item_duration": 112
        },
        {
            "item_id": "movie3",
            "item_title": "Dune",
            "item_genre": "Sci-Fi",
            "item_director": "Denis Villeneuve", 
            "item_rating": 8.0,
            "item_year": 2021,
            "item_duration": 155
        }
    ]
    
    # Generate recommendations
    recommendations = engine.generate_recommendations(user_context, candidates, k=5)
    
    # Display results
    print("Rule-Based Movie Recommendations:")
    print("=" * 50)
    
    for i, rec in enumerate(recommendations, 1):
        movie = rec["item"]
        print(f"{i}. {movie['item_title']} (Score: {rec['score']:.2f})")
        print(f"   Director: {movie['item_director']}")
        print(f"   Genre: {movie['item_genre']}")
        print(f"   Rating: {movie['item_rating']}")
        
        # Show explanation
        explanation = engine.explain_recommendation(rec)
        print(f"   Explanation:\n{explanation}")
        print("-" * 30)

# Run demo
if __name__ == "__main__":
    demo_rule_based_system()
```

## 3. Constraint Satisfaction in Recommendations

### Constraint Programming for Recommendations

```python
from itertools import product
from typing import Set, Callable

class Variable:
    """Represents a variable in constraint satisfaction"""
    
    def __init__(self, name: str, domain: List[Any]):
        self.name = name
        self.domain = domain.copy()
        self.original_domain = domain.copy()
        self.value = None
        
    def is_assigned(self) -> bool:
        return self.value is not None
    
    def assign(self, value: Any):
        if value not in self.domain:
            raise ValueError(f"Value {value} not in domain for variable {self.name}")
        self.value = value
    
    def unassign(self):
        self.value = None
    
    def reduce_domain(self, values_to_remove: Set[Any]):
        self.domain = [v for v in self.domain if v not in values_to_remove]
    
    def restore_domain(self):
        self.domain = self.original_domain.copy()

class Constraint:
    """Represents a constraint between variables"""
    
    def __init__(self, variables: List[str], constraint_func: Callable):
        self.variables = variables
        self.constraint_func = constraint_func
    
    def is_satisfied(self, assignment: Dict[str, Any]) -> bool:
        """Check if constraint is satisfied by current assignment"""
        # Only check if all variables are assigned
        for var in self.variables:
            if var not in assignment or assignment[var] is None:
                return True  # Constraint not applicable yet
        
        # Extract values for constraint variables
        values = [assignment[var] for var in self.variables]
        return self.constraint_func(*values)
    
    def get_conflicting_values(self, variable: str, value: Any, 
                             assignment: Dict[str, Any]) -> Set[Any]:
        """Get values that would conflict with this assignment"""
        conflicting = set()
        
        # Create test assignment
        test_assignment = assignment.copy()
        test_assignment[variable] = value
        
        if not self.is_satisfied(test_assignment):
            conflicting.add(value)
        
        return conflicting

class ConstraintSatisfactionProblem:
    """
    Constraint Satisfaction Problem solver for recommendations
    """
    
    def __init__(self):
        self.variables = {}  # name -> Variable
        self.constraints = []
        self.solution_count = 0
        self.max_solutions = 100
    
    def add_variable(self, variable: Variable):
        """Add variable to CSP"""
        self.variables[variable.name] = variable
    
    def add_constraint(self, constraint: Constraint):
        """Add constraint to CSP"""
        self.constraints.append(constraint)
    
    def solve(self, use_forward_checking=True, use_arc_consistency=True) -> List[Dict[str, Any]]:
        """
        Solve CSP and return all solutions
        
        Args:
            use_forward_checking: Use forward checking for pruning
            use_arc_consistency: Use arc consistency preprocessing
            
        Returns:
            List of solution assignments
        """
        self.solution_count = 0
        
        # Preprocess with arc consistency
        if use_arc_consistency:
            if not self._enforce_arc_consistency():
                return []  # No solution possible
        
        # Start backtracking search
        solutions = []
        assignment = {}
        
        self._backtrack_search(assignment, solutions, use_forward_checking)
        
        return solutions
    
    def _backtrack_search(self, assignment: Dict[str, Any], solutions: List[Dict[str, Any]], 
                         use_forward_checking: bool) -> bool:
        """Recursive backtracking search"""
        
        if self.solution_count >= self.max_solutions:
            return False
        
        # Check if assignment is complete
        if len(assignment) == len(self.variables):
            solutions.append(assignment.copy())
            self.solution_count += 1
            return True
        
        # Select next variable (using MRV heuristic)
        variable = self._select_unassigned_variable(assignment)
        
        # Try each value in domain
        for value in self._order_domain_values(variable, assignment):
            if self._is_consistent(variable.name, value, assignment):
                
                # Make assignment
                assignment[variable.name] = value
                variable.assign(value)
                
                # Forward checking
                removed_values = {}
                if use_forward_checking:
                    removed_values = self._forward_check(variable.name, value, assignment)
                    
                    if removed_values is None:  # Domain wipeout
                        # Backtrack
                        assignment.pop(variable.name)
                        variable.unassign()
                        continue
                
                # Recursive call
                result = self._backtrack_search(assignment, solutions, use_forward_checking)
                
                # Restore domains if forward checking was used
                if removed_values:
                    self._restore_domains(removed_values)
                
                # Backtrack
                assignment.pop(variable.name)
                variable.unassign()
                
                if result and self.solution_count >= self.max_solutions:
                    return True
        
        return False
    
    def _select_unassigned_variable(self, assignment: Dict[str, Any]) -> Variable:
        """Select next variable using Minimum Remaining Values heuristic"""
        unassigned = [var for name, var in self.variables.items() 
                     if name not in assignment]
        
        if not unassigned:
            return None
        
        # Choose variable with smallest domain (MRV)
        return min(unassigned, key=lambda var: len(var.domain))
    
    def _order_domain_values(self, variable: Variable, assignment: Dict[str, Any]) -> List[Any]:
        """Order domain values using Least Constraining Value heuristic"""
        # For now, return domain in original order
        # Can be enhanced with LCV heuristic
        return variable.domain.copy()
    
    def _is_consistent(self, variable: str, value: Any, assignment: Dict[str, Any]) -> bool:
        """Check if assignment is consistent with all constraints"""
        test_assignment = assignment.copy()
        test_assignment[variable] = value
        
        for constraint in self.constraints:
            if variable in constraint.variables:
                if not constraint.is_satisfied(test_assignment):
                    return False
        
        return True
    
    def _forward_check(self, variable: str, value: Any, assignment: Dict[str, Any]) -> Dict[str, Set[Any]]:
        """
        Forward checking: remove inconsistent values from future variables
        
        Returns:
            Dictionary of removed values for restoration, or None if domain wipeout
        """
        removed_values = defaultdict(set)
        
        for constraint in self.constraints:
            if variable in constraint.variables:
                # Check effect on other variables in this constraint
                for var_name in constraint.variables:
                    if var_name != variable and var_name not in assignment:
                        var = self.variables[var_name]
                        values_to_remove = set()
                        
                        for domain_value in var.domain:
                            test_assignment = assignment.copy()
                            test_assignment[variable] = value
                            test_assignment[var_name] = domain_value
                            
                            if not constraint.is_satisfied(test_assignment):
                                values_to_remove.add(domain_value)
                        
                        if values_to_remove:
                            removed_values[var_name].update(values_to_remove)
                            var.reduce_domain(values_to_remove)
                            
                            # Check for domain wipeout
                            if not var.domain:
                                return None
        
        return dict(removed_values)
    
    def _restore_domains(self, removed_values: Dict[str, Set[Any]]):
        """Restore domains after backtracking"""
        for var_name, removed in removed_values.items():
            var = self.variables[var_name]
            var.domain.extend(removed)
    
    def _enforce_arc_consistency(self) -> bool:
        """Enforce arc consistency (AC-3 algorithm)"""
        # Create queue of arcs
        arcs = []
        for constraint in self.constraints:
            for i, var1 in enumerate(constraint.variables):
                for var2 in constraint.variables[i+1:]:
                    arcs.append((var1, var2, constraint))
                    arcs.append((var2, var1, constraint))
        
        while arcs:
            var1, var2, constraint = arcs.pop(0)
            
            if self._revise(var1, var2, constraint):
                if not self.variables[var1].domain:
                    return False  # Domain is empty
                
                # Add related arcs back to queue
                for other_constraint in self.constraints:
                    if var1 in other_constraint.variables and other_constraint != constraint:
                        for var in other_constraint.variables:
                            if var != var1:
                                arcs.append((var, var1, other_constraint))
        
        return True
    
    def _revise(self, var1: str, var2: str, constraint: Constraint) -> bool:
        """Revise domain of var1 with respect to var2"""
        revised = False
        values_to_remove = []
        
        for value1 in self.variables[var1].domain:
            # Check if there exists a value in var2's domain that satisfies constraint
            found_consistent = False
            
            for value2 in self.variables[var2].domain:
                test_assignment = {var1: value1, var2: value2}
                if constraint.is_satisfied(test_assignment):
                    found_consistent = True
                    break
            
            if not found_consistent:
                values_to_remove.append(value1)
                revised = True
        
        # Remove inconsistent values
        for value in values_to_remove:
            self.variables[var1].domain.remove(value)
        
        return revised

class RecommendationCSP(ConstraintSatisfactionProblem):
    """
    Specialized CSP for recommendation problems
    """
    
    def __init__(self, items: List[Dict], user_preferences: Dict, constraints_config: Dict):
        super().__init__()
        self.items = items
        self.user_preferences = user_preferences
        self.constraints_config = constraints_config
        
        self._build_recommendation_csp()
    
    def _build_recommendation_csp(self):
        """Build CSP for recommendation problem"""
        
        # Create variables for each recommendation slot
        num_recommendations = self.constraints_config.get('num_recommendations', 5)
        
        for i in range(num_recommendations):
            var_name = f"rec_{i}"
            domain = [item['id'] for item in self.items]
            self.add_variable(Variable(var_name, domain))
        
        # Add diversity constraint (no duplicate recommendations)
        self.add_constraint(Constraint(
            [f"rec_{i}" for i in range(num_recommendations)],
            lambda *items: len(set(items)) == len(items)
        ))
        
        # Add user preference constraints
        self._add_preference_constraints()
        
        # Add business constraints
        self._add_business_constraints()
    
    def _add_preference_constraints(self):
        """Add constraints based on user preferences"""
        
        # Genre preferences
        if 'favorite_genres' in self.user_preferences:
            favorite_genres = self.user_preferences['favorite_genres']
            min_favorite_genre_items = self.constraints_config.get('min_favorite_genre_items', 2)
            
            # At least min_favorite_genre_items should be from favorite genres
            def genre_constraint(*items):
                favorite_count = 0
                for item_id in items:
                    item = next((item for item in self.items if item['id'] == item_id), None)
                    if item and any(genre in favorite_genres for genre in item.get('genres', [])):
                        favorite_count += 1
                return favorite_count >= min_favorite_genre_items
            
            var_names = [var.name for var in self.variables.values()]
            self.add_constraint(Constraint(var_names, genre_constraint))
        
        # Rating constraint
        min_rating = self.user_preferences.get('min_rating', 0.0)
        if min_rating > 0:
            def rating_constraint(item_id):
                item = next((item for item in self.items if item['id'] == item_id), None)
                return item and item.get('rating', 0) >= min_rating
            
            for var in self.variables.values():
                self.add_constraint(Constraint([var.name], rating_constraint))
        
        # Recency constraint
        max_age_years = self.user_preferences.get('max_age_years')
        if max_age_years is not None:
            current_year = 2024
            min_year = current_year - max_age_years
            
            def recency_constraint(item_id):
                item = next((item for item in self.items if item['id'] == item_id), None)
                return item and item.get('year', 0) >= min_year
            
            for var in self.variables.values():
                self.add_constraint(Constraint([var.name], recency_constraint))
    
    def _add_business_constraints(self):
        """Add business-specific constraints"""
        
        # Diversity constraint for genres
        max_same_genre = self.constraints_config.get('max_same_genre', 2)
        if max_same_genre < len(self.variables):
            
            def genre_diversity_constraint(*items):
                genre_counts = defaultdict(int)
                
                for item_id in items:
                    item = next((item for item in self.items if item['id'] == item_id), None)
                    if item:
                        for genre in item.get('genres', []):
                            genre_counts[genre] += 1
                
                return all(count <= max_same_genre for count in genre_counts.values())
            
            var_names = [var.name for var in self.variables.values()]
            self.add_constraint(Constraint(var_names, genre_diversity_constraint))
        
        # Popularity balance
        min_popular_items = self.constraints_config.get('min_popular_items', 1)
        popularity_threshold = self.constraints_config.get('popularity_threshold', 8.0)
        
        def popularity_constraint(*items):
            popular_count = 0
            for item_id in items:
                item = next((item for item in self.items if item['id'] == item_id), None)
                if item and item.get('rating', 0) >= popularity_threshold:
                    popular_count += 1
            return popular_count >= min_popular_items
        
        var_names = [var.name for var in self.variables.values()]
        self.add_constraint(Constraint(var_names, popularity_constraint))

def demo_constraint_based_recommendations():
    """Demonstrate constraint-based recommendation system"""
    
    # Sample movie data
    movies = [
        {'id': 'm1', 'title': 'Inception', 'genres': ['Sci-Fi', 'Thriller'], 'rating': 8.8, 'year': 2010},
        {'id': 'm2', 'title': 'The Dark Knight', 'genres': ['Action', 'Crime'], 'rating': 9.0, 'year': 2008},
        {'id': 'm3', 'title': 'Pulp Fiction', 'genres': ['Crime', 'Drama'], 'rating': 8.9, 'year': 1994},
        {'id': 'm4', 'title': 'The Shawshank Redemption', 'genres': ['Drama'], 'rating': 9.3, 'year': 1994},
        {'id': 'm5', 'title': 'Interstellar', 'genres': ['Sci-Fi', 'Drama'], 'rating': 8.6, 'year': 2014},
        {'id': 'm6', 'title': 'The Matrix', 'genres': ['Sci-Fi', 'Action'], 'rating': 8.7, 'year': 1999},
        {'id': 'm7', 'title': 'Goodfellas', 'genres': ['Crime', 'Drama'], 'rating': 8.7, 'year': 1990}
    ]
    
    # User preferences
    user_prefs = {
        'favorite_genres': ['Sci-Fi', 'Action'], 
        'min_rating': 8.5,
        'max_age_years': 25
    }
    
    # Constraint configuration
    constraints_config = {
        'num_recommendations': 3,
        'min_favorite_genre_items': 1,
        'max_same_genre': 1,
        'min_popular_items': 2,
        'popularity_threshold': 8.7
    }
    
    # Create and solve CSP
    csp = RecommendationCSP(movies, user_prefs, constraints_config)
    solutions = csp.solve()
    
    print("Constraint-Based Movie Recommendations:")
    print("=" * 50)
    print(f"Found {len(solutions)} valid recommendation sets:\n")
    
    for i, solution in enumerate(solutions[:5], 1):  # Show first 5 solutions
        print(f"Solution {i}:")
        for slot, movie_id in solution.items():
            movie = next(m for m in movies if m['id'] == movie_id)
            print(f"  {slot}: {movie['title']} ({', '.join(movie['genres'])}, {movie['rating']}, {movie['year']})")
        print()

if __name__ == "__main__":
    demo_constraint_based_recommendations()
```

## 4. Study Questions

### Beginner Level

1. What are the key components of an ontology-based recommendation system?
2. How do rule-based systems differ from collaborative filtering approaches?
3. What is constraint satisfaction and how does it apply to recommendations?
4. Explain the concept of semantic similarity in knowledge-based systems.
5. What are the advantages of knowledge-based recommendation approaches?

### Intermediate Level

6. Implement a rule-based system for book recommendations with at least 10 meaningful rules.
7. How would you handle conflicting rules in a rule-based recommendation engine?
8. Design an ontology for e-commerce product recommendations with hierarchical categories.
9. What are the trade-offs between expressiveness and computational efficiency in constraint-based systems?
10. How would you combine knowledge-based approaches with collaborative filtering?

### Advanced Level

11. Implement a constraint satisfaction system that can handle soft constraints with preference weights.
12. Design a knowledge graph-based recommendation system that can perform multi-hop reasoning.
13. How would you automatically learn rules from user interaction data in a rule-based system?
14. Implement a conversational recommendation system that uses constraint satisfaction to refine user preferences.
15. Design a knowledge-based system that can explain its recommendations at multiple levels of detail.

### Tricky Questions

16. How would you handle incomplete or uncertain knowledge in a knowledge-based recommendation system?
17. Design a system that can adapt its ontology and rules based on user feedback and changing domains.
18. How would you scale constraint satisfaction approaches to handle millions of items and complex constraint sets?
19. Implement a knowledge-based system that can handle cross-domain recommendations using shared ontologies.
20. How would you design a knowledge-based system that can work with both structured and unstructured content while maintaining reasoning capabilities?

## Key Takeaways

1. **Ontologies** provide structured knowledge representation for semantic recommendations
2. **Rule-based systems** encode expert knowledge and business logic explicitly
3. **Constraint satisfaction** enables complex multi-objective recommendation scenarios
4. **Knowledge graphs** enable sophisticated reasoning and explanation capabilities
5. **Semantic similarity** captures conceptual relationships beyond surface features
6. **Explainability** is a natural strength of knowledge-based approaches
7. **Hybrid systems** can combine knowledge-based methods with data-driven approaches effectively

## Next Session Preview

In Day 4.6, we'll explore **Constraint-Based and Rule-Based Systems** in more depth, covering:
- Advanced constraint programming techniques
- Fuzzy rule systems for uncertain preferences
- Multi-objective optimization in recommendations
- Interactive constraint refinement
- Scalable rule execution architectures