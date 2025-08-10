# Day 3: Cognitive Components & Memory Systems

## Learning Objectives

By the end of today's session, you will be able to:

1. **Design and implement** multi-tier memory architectures including working, episodic, semantic, and procedural memory systems
2. **Build knowledge representation systems** using graph databases, semantic networks, and vector embeddings for intelligent agents
3. **Implement attention mechanisms** for focus and resource allocation in cognitive processing pipelines  
4. **Create learning systems** that enable agents to improve performance through experience and knowledge transfer
5. **Optimize memory retrieval** using similarity search, graph traversal, and hybrid indexing strategies for real-time performance

---

## Theoretical Foundation

### 1. Multi-Tier Memory Architecture

Modern agentic AI systems require sophisticated memory architectures that mirror human cognitive capabilities while leveraging computational advantages. The multi-tier approach provides different memory types optimized for specific cognitive functions.

#### Working Memory System

Working memory serves as the cognitive workspace where agents manipulate information temporarily during reasoning and problem-solving tasks.

```python
import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import json
import hashlib
from abc import ABC, abstractmethod

@dataclass
class WorkingMemoryItem:
    content: Any
    activation_level: float
    timestamp: float
    access_count: int = 0
    importance: float = 0.5
    context_tags: List[str] = field(default_factory=list)
    
    def decay_activation(self, decay_rate: float = 0.95):
        """Apply temporal decay to activation level"""
        time_diff = time.time() - self.timestamp
        self.activation_level *= (decay_rate ** time_diff)
    
    def access(self):
        """Update item on access"""
        self.access_count += 1
        self.activation_level = min(1.0, self.activation_level + 0.1)
        self.timestamp = time.time()

class WorkingMemorySystem:
    """Advanced working memory with attention and capacity management"""
    
    def __init__(self, capacity: int = 7, attention_threshold: float = 0.3):
        self.capacity = capacity
        self.attention_threshold = attention_threshold
        self.items: Dict[str, WorkingMemoryItem] = {}
        self.attention_focus: List[str] = []
        self.processing_history: deque = deque(maxlen=100)
        
    async def store(self, key: str, content: Any, importance: float = 0.5,
                   context_tags: List[str] = None) -> bool:
        """Store item in working memory with capacity management"""
        
        # Check if update to existing item
        if key in self.items:
            self.items[key].content = content
            self.items[key].importance = importance
            self.items[key].access()
            if context_tags:
                self.items[key].context_tags.extend(context_tags)
            return True
        
        # Handle capacity constraints
        if len(self.items) >= self.capacity:
            await self._evict_least_important()
        
        # Create new item
        item = WorkingMemoryItem(
            content=content,
            activation_level=1.0,
            timestamp=time.time(),
            importance=importance,
            context_tags=context_tags or []
        )
        
        self.items[key] = item
        await self._update_attention_focus()
        return True
    
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve item with activation update"""
        if key in self.items:
            item = self.items[key]
            item.access()
            await self._update_attention_focus()
            return item.content
        return None
    
    async def find_by_context(self, context_tags: List[str], 
                             similarity_threshold: float = 0.5) -> List[Tuple[str, Any]]:
        """Find items by contextual similarity"""
        results = []
        
        for key, item in self.items.items():
            if item.activation_level < self.attention_threshold:
                continue
                
            # Calculate context similarity
            common_tags = set(item.context_tags) & set(context_tags)
            total_tags = set(item.context_tags) | set(context_tags)
            
            if total_tags:
                similarity = len(common_tags) / len(total_tags)
                if similarity >= similarity_threshold:
                    results.append((key, item.content))
        
        # Sort by activation level and similarity
        results.sort(key=lambda x: self.items[x[0]].activation_level, reverse=True)
        return results
    
    async def _evict_least_important(self):
        """Remove least important item based on activation and importance"""
        if not self.items:
            return
        
        # Apply decay to all items
        for item in self.items.values():
            item.decay_activation()
        
        # Find least important item
        min_score = float('inf')
        evict_key = None
        
        for key, item in self.items.items():
            score = item.activation_level * item.importance
            if score < min_score:
                min_score = score
                evict_key = key
        
        if evict_key:
            del self.items[evict_key]
            if evict_key in self.attention_focus:
                self.attention_focus.remove(evict_key)
    
    async def _update_attention_focus(self):
        """Update attention focus based on activation levels"""
        # Sort items by activation level
        sorted_items = sorted(
            self.items.items(),
            key=lambda x: x[1].activation_level,
            reverse=True
        )
        
        # Update focus to top activated items
        self.attention_focus = [
            key for key, item in sorted_items[:3]
            if item.activation_level >= self.attention_threshold
        ]
    
    def get_attention_state(self) -> Dict[str, Any]:
        """Get current attention state for debugging/monitoring"""
        return {
            'focused_items': self.attention_focus,
            'total_items': len(self.items),
            'capacity_utilization': len(self.items) / self.capacity,
            'average_activation': np.mean([item.activation_level for item in self.items.values()]) if self.items else 0.0
        }
```

#### Episodic Memory System

Episodic memory stores specific experiences and events, enabling agents to learn from past interactions and apply experiential knowledge to new situations.

```python
from datetime import datetime, timedelta
import uuid
from typing import NamedTuple

class Episode(NamedTuple):
    id: str
    timestamp: datetime
    context: Dict[str, Any]
    actions: List[Dict[str, Any]]
    outcomes: Dict[str, Any]
    emotional_valence: float  # -1.0 to 1.0
    importance: float
    related_episodes: List[str]

class EpisodicMemorySystem:
    """Stores and retrieves experiential knowledge"""
    
    def __init__(self, max_episodes: int = 10000, embedding_model=None):
        self.max_episodes = max_episodes
        self.episodes: Dict[str, Episode] = {}
        self.temporal_index: List[Tuple[datetime, str]] = []
        self.context_embeddings: Dict[str, np.ndarray] = {}
        self.embedding_model = embedding_model or self._default_embedding_model
        
        # Experience learning
        self.success_patterns: Dict[str, List[str]] = {}
        self.failure_patterns: Dict[str, List[str]] = {}
        
    async def store_episode(self, context: Dict[str, Any], actions: List[Dict[str, Any]],
                           outcomes: Dict[str, Any], emotional_valence: float = 0.0,
                           importance: float = 0.5) -> str:
        """Store new episode with automatic relationship detection"""
        
        episode_id = str(uuid.uuid4())
        episode = Episode(
            id=episode_id,
            timestamp=datetime.now(),
            context=context,
            actions=actions,
            outcomes=outcomes,
            emotional_valence=emotional_valence,
            importance=importance,
            related_episodes=[]
        )
        
        # Find related episodes
        related_episodes = await self._find_related_episodes(context, actions)
        episode = episode._replace(related_episodes=related_episodes)
        
        # Store episode
        self.episodes[episode_id] = episode
        self.temporal_index.append((episode.timestamp, episode_id))
        
        # Generate and store context embedding
        if self.embedding_model:
            context_embedding = await self._generate_context_embedding(context, actions)
            self.context_embeddings[episode_id] = context_embedding
        
        # Update learning patterns
        await self._update_learning_patterns(episode)
        
        # Manage memory capacity
        if len(self.episodes) > self.max_episodes:
            await self._consolidate_memories()
        
        return episode_id
    
    async def retrieve_similar_episodes(self, context: Dict[str, Any], 
                                       actions: Optional[List[Dict[str, Any]]] = None,
                                       similarity_threshold: float = 0.7,
                                       max_results: int = 10) -> List[Episode]:
        """Retrieve episodes similar to given context and actions"""
        
        if not self.embedding_model:
            return await self._retrieve_by_context_matching(context, max_results)
        
        # Generate query embedding
        query_embedding = await self._generate_context_embedding(context, actions or [])
        
        # Calculate similarities
        similarities = []
        for episode_id, episode_embedding in self.context_embeddings.items():
            similarity = await self._cosine_similarity(query_embedding, episode_embedding)
            if similarity >= similarity_threshold:
                similarities.append((similarity, episode_id))
        
        # Sort by similarity and return top results
        similarities.sort(reverse=True)
        result_episodes = []
        
        for similarity, episode_id in similarities[:max_results]:
            if episode_id in self.episodes:
                result_episodes.append(self.episodes[episode_id])
        
        return result_episodes
    
    async def get_success_patterns(self, context_type: str) -> List[Dict[str, Any]]:
        """Get learned success patterns for context type"""
        
        if context_type not in self.success_patterns:
            return []
        
        # Retrieve successful episodes
        successful_episodes = []
        for episode_id in self.success_patterns[context_type]:
            if episode_id in self.episodes:
                episode = self.episodes[episode_id]
                if episode.emotional_valence > 0.5:  # Successful episodes
                    successful_episodes.append({
                        'context': episode.context,
                        'actions': episode.actions,
                        'success_score': episode.emotional_valence
                    })
        
        return successful_episodes
    
    async def _find_related_episodes(self, context: Dict[str, Any], 
                                   actions: List[Dict[str, Any]]) -> List[str]:
        """Find episodes with similar context or action patterns"""
        related = []
        
        for episode_id, episode in self.episodes.items():
            # Context similarity
            context_similarity = await self._calculate_context_similarity(
                context, episode.context
            )
            
            # Action pattern similarity
            action_similarity = await self._calculate_action_similarity(
                actions, episode.actions
            )
            
            combined_similarity = (context_similarity + action_similarity) / 2
            
            if combined_similarity > 0.6:
                related.append(episode_id)
        
        return related[:5]  # Limit to top 5 related episodes
    
    async def _update_learning_patterns(self, episode: Episode):
        """Update success/failure patterns based on episode outcome"""
        
        # Determine context type
        context_type = episode.context.get('type', 'general')
        
        # Initialize pattern lists if needed
        if context_type not in self.success_patterns:
            self.success_patterns[context_type] = []
        if context_type not in self.failure_patterns:
            self.failure_patterns[context_type] = []
        
        # Categorize episode
        if episode.emotional_valence > 0.3:
            self.success_patterns[context_type].append(episode.id)
        elif episode.emotional_valence < -0.3:
            self.failure_patterns[context_type].append(episode.id)
        
        # Limit pattern memory
        self.success_patterns[context_type] = self.success_patterns[context_type][-100:]
        self.failure_patterns[context_type] = self.failure_patterns[context_type][-100:]
    
    async def _consolidate_memories(self):
        """Consolidate memories by removing least important episodes"""
        
        # Sort episodes by importance and recency
        episodes_by_importance = sorted(
            self.episodes.items(),
            key=lambda x: (x[1].importance, x[1].timestamp),
            reverse=True
        )
        
        # Keep most important episodes
        episodes_to_keep = episodes_by_importance[:self.max_episodes // 2]
        
        # Remove less important episodes
        new_episodes = dict(episodes_to_keep)
        
        # Clean up related data structures
        removed_ids = set(self.episodes.keys()) - set(new_episodes.keys())
        for episode_id in removed_ids:
            if episode_id in self.context_embeddings:
                del self.context_embeddings[episode_id]
        
        # Update temporal index
        self.temporal_index = [
            (timestamp, episode_id) for timestamp, episode_id in self.temporal_index
            if episode_id in new_episodes
        ]
        
        self.episodes = new_episodes
    
    async def _generate_context_embedding(self, context: Dict[str, Any], 
                                        actions: List[Dict[str, Any]]) -> np.ndarray:
        """Generate embedding for context and actions"""
        
        # Combine context and actions into text representation
        context_text = json.dumps(context, sort_keys=True)
        actions_text = json.dumps(actions, sort_keys=True)
        combined_text = f"Context: {context_text} Actions: {actions_text}"
        
        # Generate embedding (placeholder - replace with actual embedding model)
        return await self.embedding_model.encode(combined_text)
    
    def _default_embedding_model(self):
        """Default embedding model (placeholder)"""
        class DefaultEmbedding:
            async def encode(self, text: str) -> np.ndarray:
                # Simple hash-based embedding for demo
                hash_val = hashlib.md5(text.encode()).hexdigest()
                return np.array([int(hash_val[i:i+2], 16) for i in range(0, 32, 2)])
        
        return DefaultEmbedding()
    
    async def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
```

#### Semantic Memory System

Semantic memory stores factual knowledge, concepts, and relationships that agents use for reasoning and understanding.

```python
import networkx as nx
from typing import Set, Union
import pickle

class ConceptNode:
    """Represents a concept in semantic memory"""
    
    def __init__(self, concept_id: str, name: str, attributes: Dict[str, Any] = None):
        self.concept_id = concept_id
        self.name = name
        self.attributes = attributes or {}
        self.activation_level = 1.0
        self.access_frequency = 0
        self.creation_time = time.time()
        self.last_access_time = time.time()
    
    def access(self):
        """Update access statistics"""
        self.access_frequency += 1
        self.last_access_time = time.time()
        self.activation_level = min(1.0, self.activation_level + 0.05)
    
    def decay(self, decay_rate: float = 0.99):
        """Apply activation decay"""
        time_since_access = time.time() - self.last_access_time
        self.activation_level *= (decay_rate ** (time_since_access / 3600))  # Hourly decay

class SemanticMemorySystem:
    """Graph-based semantic memory with concept relationships"""
    
    def __init__(self):
        self.concept_graph = nx.Graph()
        self.concepts: Dict[str, ConceptNode] = {}
        self.relation_types: Set[str] = {
            'is_a', 'part_of', 'related_to', 'causes', 'enables', 
            'similar_to', 'opposite_of', 'instance_of'
        }
        
        # Reasoning engines
        self.inference_engine = InferenceEngine(self)
        self.analogy_engine = AnalogyEngine(self)
        
    async def add_concept(self, concept_id: str, name: str, 
                         attributes: Dict[str, Any] = None) -> ConceptNode:
        """Add new concept to semantic memory"""
        
        if concept_id in self.concepts:
            # Update existing concept
            concept = self.concepts[concept_id]
            if attributes:
                concept.attributes.update(attributes)
            concept.access()
            return concept
        
        # Create new concept
        concept = ConceptNode(concept_id, name, attributes)
        self.concepts[concept_id] = concept
        self.concept_graph.add_node(concept_id, concept=concept)
        
        return concept
    
    async def add_relationship(self, concept1_id: str, concept2_id: str, 
                             relation_type: str, strength: float = 1.0,
                             bidirectional: bool = True) -> bool:
        """Add relationship between concepts"""
        
        if concept1_id not in self.concepts or concept2_id not in self.concepts:
            return False
        
        if relation_type not in self.relation_types:
            self.relation_types.add(relation_type)
        
        # Add edge to graph
        self.concept_graph.add_edge(
            concept1_id, 
            concept2_id,
            relation=relation_type,
            strength=strength,
            bidirectional=bidirectional
        )
        
        # Access concepts to boost activation
        self.concepts[concept1_id].access()
        self.concepts[concept2_id].access()
        
        return True
    
    async def find_related_concepts(self, concept_id: str, 
                                  relation_types: List[str] = None,
                                  max_distance: int = 2,
                                  min_activation: float = 0.1) -> List[Tuple[str, float, List[str]]]:
        """Find concepts related to given concept"""
        
        if concept_id not in self.concepts:
            return []
        
        related_concepts = []
        
        # Use BFS to find related concepts within max_distance
        visited = set()
        queue = deque([(concept_id, 0, [])])  # (concept_id, distance, path)
        
        while queue:
            current_id, distance, path = queue.popleft()
            
            if current_id in visited or distance > max_distance:
                continue
            
            visited.add(current_id)
            current_concept = self.concepts[current_id]
            
            # Skip if activation too low
            if current_concept.activation_level < min_activation:
                continue
            
            # Add to results if not the starting concept
            if distance > 0:
                related_concepts.append((
                    current_id,
                    current_concept.activation_level,
                    path
                ))
            
            # Explore neighbors
            for neighbor_id in self.concept_graph.neighbors(current_id):
                if neighbor_id not in visited:
                    edge_data = self.concept_graph[current_id][neighbor_id]
                    relation = edge_data.get('relation', 'unknown')
                    
                    # Filter by relation types if specified
                    if relation_types and relation not in relation_types:
                        continue
                    
                    new_path = path + [f"{current_id}-{relation}->{neighbor_id}"]
                    queue.append((neighbor_id, distance + 1, new_path))
        
        # Sort by activation level and relevance
        related_concepts.sort(key=lambda x: x[1], reverse=True)
        return related_concepts[:20]  # Limit results
    
    async def infer_relationships(self, concept_id: str) -> List[Tuple[str, str, float]]:
        """Infer potential new relationships based on existing knowledge"""
        return await self.inference_engine.infer_relationships(concept_id)
    
    async def find_analogies(self, source_concepts: List[str], 
                           target_domain: str) -> List[Dict[str, Any]]:
        """Find analogical mappings between concepts"""
        return await self.analogy_engine.find_analogies(source_concepts, target_domain)
    
    async def consolidate_knowledge(self):
        """Consolidate semantic knowledge by strengthening frequent patterns"""
        
        # Apply activation decay
        for concept in self.concepts.values():
            concept.decay()
        
        # Strengthen relationships between frequently co-accessed concepts
        concept_pairs = []
        for concept_id, concept in self.concepts.items():
            if concept.access_frequency > 5:  # Frequently accessed concepts
                for neighbor_id in self.concept_graph.neighbors(concept_id):
                    neighbor = self.concepts[neighbor_id]
                    if neighbor.access_frequency > 5:
                        concept_pairs.append((concept_id, neighbor_id))
        
        # Strengthen relationships
        for concept1_id, concept2_id in concept_pairs:
            if self.concept_graph.has_edge(concept1_id, concept2_id):
                edge_data = self.concept_graph[concept1_id][concept2_id]
                current_strength = edge_data.get('strength', 1.0)
                edge_data['strength'] = min(2.0, current_strength + 0.1)
    
    def export_knowledge_graph(self, format: str = 'graphml') -> str:
        """Export knowledge graph in specified format"""
        if format == 'graphml':
            return nx.write_graphml(self.concept_graph, 'semantic_memory.graphml')
        elif format == 'json':
            graph_data = nx.node_link_data(self.concept_graph)
            return json.dumps(graph_data, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

class InferenceEngine:
    """Performs logical inference over semantic knowledge"""
    
    def __init__(self, semantic_memory: SemanticMemorySystem):
        self.semantic_memory = semantic_memory
        self.inference_rules = self._initialize_inference_rules()
    
    def _initialize_inference_rules(self) -> List[Dict[str, Any]]:
        """Initialize basic inference rules"""
        return [
            {
                'name': 'transitivity',
                'pattern': ['X', 'is_a', 'Y', 'Y', 'is_a', 'Z'],
                'conclusion': ['X', 'is_a', 'Z'],
                'confidence': 0.9
            },
            {
                'name': 'inheritance',
                'pattern': ['X', 'is_a', 'Y', 'Y', 'has_property', 'P'],
                'conclusion': ['X', 'has_property', 'P'],
                'confidence': 0.8
            },
            {
                'name': 'part_whole_inheritance',
                'pattern': ['X', 'part_of', 'Y', 'Y', 'has_property', 'P'],
                'conclusion': ['X', 'related_to_property', 'P'],
                'confidence': 0.6
            }
        ]
    
    async def infer_relationships(self, concept_id: str) -> List[Tuple[str, str, float]]:
        """Infer new relationships for given concept"""
        
        inferred_relationships = []
        
        # Get existing relationships
        existing_rels = []
        for neighbor_id in self.semantic_memory.concept_graph.neighbors(concept_id):
            edge_data = self.semantic_memory.concept_graph[concept_id][neighbor_id]
            relation = edge_data.get('relation', 'unknown')
            existing_rels.append((concept_id, relation, neighbor_id))
        
        # Apply inference rules
        for rule in self.inference_rules:
            potential_inferences = await self._apply_inference_rule(
                rule, existing_rels, concept_id
            )
            inferred_relationships.extend(potential_inferences)
        
        return inferred_relationships
    
    async def _apply_inference_rule(self, rule: Dict[str, Any], 
                                  existing_rels: List[Tuple[str, str, str]],
                                  target_concept: str) -> List[Tuple[str, str, float]]:
        """Apply specific inference rule"""
        
        inferences = []
        pattern = rule['pattern']
        conclusion = rule['conclusion']
        confidence = rule['confidence']
        
        # Simple pattern matching for demonstration
        # In practice, this would use more sophisticated reasoning
        
        if rule['name'] == 'transitivity' and len(pattern) == 6:
            # Look for A->B and B->C patterns to infer A->C
            for rel1 in existing_rels:
                if rel1[0] == target_concept and rel1[1] == pattern[1]:
                    intermediate = rel1[2]
                    for rel2 in existing_rels:
                        if rel2[0] == intermediate and rel2[1] == pattern[4]:
                            final_concept = rel2[2]
                            inferences.append((final_concept, conclusion[1], confidence))
        
        return inferences

class AnalogyEngine:
    """Finds analogical relationships between concepts"""
    
    def __init__(self, semantic_memory: SemanticMemorySystem):
        self.semantic_memory = semantic_memory
    
    async def find_analogies(self, source_concepts: List[str], 
                           target_domain: str) -> List[Dict[str, Any]]:
        """Find analogical mappings between source concepts and target domain"""
        
        analogies = []
        
        # Get concepts in target domain
        target_concepts = await self._get_domain_concepts(target_domain)
        
        # Find structural similarities
        for source_concept in source_concepts:
            source_structure = await self._get_concept_structure(source_concept)
            
            for target_concept in target_concepts:
                target_structure = await self._get_concept_structure(target_concept)
                
                similarity = await self._calculate_structural_similarity(
                    source_structure, target_structure
                )
                
                if similarity > 0.6:  # Threshold for analogical similarity
                    analogies.append({
                        'source_concept': source_concept,
                        'target_concept': target_concept,
                        'similarity_score': similarity,
                        'mapping_details': await self._create_mapping(
                            source_structure, target_structure
                        )
                    })
        
        return sorted(analogies, key=lambda x: x['similarity_score'], reverse=True)
    
    async def _get_domain_concepts(self, domain: str) -> List[str]:
        """Get concepts belonging to specific domain"""
        domain_concepts = []
        
        for concept_id, concept in self.semantic_memory.concepts.items():
            if 'domain' in concept.attributes:
                if concept.attributes['domain'] == domain:
                    domain_concepts.append(concept_id)
        
        return domain_concepts
    
    async def _get_concept_structure(self, concept_id: str) -> Dict[str, Any]:
        """Get structural representation of concept"""
        
        if concept_id not in self.semantic_memory.concepts:
            return {}
        
        structure = {
            'concept_id': concept_id,
            'attributes': self.semantic_memory.concepts[concept_id].attributes.copy(),
            'relationships': {}
        }
        
        # Get relationships
        for neighbor_id in self.semantic_memory.concept_graph.neighbors(concept_id):
            edge_data = self.semantic_memory.concept_graph[concept_id][neighbor_id]
            relation = edge_data.get('relation', 'unknown')
            
            if relation not in structure['relationships']:
                structure['relationships'][relation] = []
            structure['relationships'][relation].append(neighbor_id)
        
        return structure
    
    async def _calculate_structural_similarity(self, struct1: Dict[str, Any], 
                                             struct2: Dict[str, Any]) -> float:
        """Calculate structural similarity between concepts"""
        
        # Attribute similarity
        attr_sim = await self._attribute_similarity(
            struct1.get('attributes', {}),
            struct2.get('attributes', {})
        )
        
        # Relationship similarity
        rel_sim = await self._relationship_similarity(
            struct1.get('relationships', {}),
            struct2.get('relationships', {})
        )
        
        # Combined similarity
        return (attr_sim + rel_sim) / 2
    
    async def _attribute_similarity(self, attrs1: Dict[str, Any], 
                                  attrs2: Dict[str, Any]) -> float:
        """Calculate attribute similarity"""
        
        if not attrs1 and not attrs2:
            return 1.0
        
        if not attrs1 or not attrs2:
            return 0.0
        
        common_keys = set(attrs1.keys()) & set(attrs2.keys())
        total_keys = set(attrs1.keys()) | set(attrs2.keys())
        
        if not total_keys:
            return 1.0
        
        return len(common_keys) / len(total_keys)
    
    async def _relationship_similarity(self, rels1: Dict[str, List[str]], 
                                     rels2: Dict[str, List[str]]) -> float:
        """Calculate relationship structure similarity"""
        
        if not rels1 and not rels2:
            return 1.0
        
        if not rels1 or not rels2:
            return 0.0
        
        common_rel_types = set(rels1.keys()) & set(rels2.keys())
        total_rel_types = set(rels1.keys()) | set(rels2.keys())
        
        if not total_rel_types:
            return 1.0
        
        return len(common_rel_types) / len(total_rel_types)
```

### 2. Attention and Focus Mechanisms

Attention mechanisms enable agents to selectively focus computational resources on the most relevant information for current tasks.

```python
class AttentionMechanism:
    """Multi-head attention for cognitive processing"""
    
    def __init__(self, d_model: int, num_heads: int = 8):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Attention weights (simplified for demonstration)
        self.query_weights = np.random.randn(num_heads, d_model, self.head_dim)
        self.key_weights = np.random.randn(num_heads, d_model, self.head_dim)
        self.value_weights = np.random.randn(num_heads, d_model, self.head_dim)
        self.output_weights = np.random.randn(d_model, d_model)
        
        # Attention history for adaptive learning
        self.attention_history = deque(maxlen=1000)
    
    async def compute_attention(self, queries: np.ndarray, keys: np.ndarray, 
                              values: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute multi-head attention"""
        
        batch_size, seq_len, _ = queries.shape
        attention_outputs = []
        attention_weights_list = []
        
        for head in range(self.num_heads):
            # Linear transformations
            Q = np.dot(queries, self.query_weights[head])
            K = np.dot(keys, self.key_weights[head])
            V = np.dot(values, self.value_weights[head])
            
            # Scaled dot-product attention
            attention_scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.head_dim)
            
            # Apply mask if provided
            if mask is not None:
                attention_scores = np.where(mask, attention_scores, -np.inf)
            
            # Softmax
            attention_weights = await self._softmax(attention_scores)
            
            # Apply attention to values
            attention_output = np.matmul(attention_weights, V)
            
            attention_outputs.append(attention_output)
            attention_weights_list.append(attention_weights)
        
        # Concatenate heads
        concat_output = np.concatenate(attention_outputs, axis=-1)
        
        # Final linear transformation
        final_output = np.dot(concat_output, self.output_weights)
        
        # Store attention pattern for analysis
        avg_attention_weights = np.mean(attention_weights_list, axis=0)
        self.attention_history.append(avg_attention_weights)
        
        return final_output, avg_attention_weights
    
    async def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax with numerical stability"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def analyze_attention_patterns(self) -> Dict[str, Any]:
        """Analyze attention patterns over time"""
        if not self.attention_history:
            return {}
        
        # Convert to numpy array
        attention_array = np.array(list(self.attention_history))
        
        # Calculate statistics
        mean_attention = np.mean(attention_array, axis=0)
        std_attention = np.std(attention_array, axis=0)
        
        # Find most attended positions
        most_attended = np.unravel_index(np.argmax(mean_attention), mean_attention.shape)
        
        return {
            'mean_attention_pattern': mean_attention.tolist(),
            'attention_variance': std_attention.tolist(),
            'most_attended_position': most_attended,
            'attention_entropy': await self._calculate_attention_entropy(mean_attention)
        }
    
    async def _calculate_attention_entropy(self, attention_matrix: np.ndarray) -> float:
        """Calculate entropy of attention distribution"""
        # Flatten attention matrix
        flat_attention = attention_matrix.flatten()
        
        # Remove zeros to avoid log(0)
        flat_attention = flat_attention[flat_attention > 1e-10]
        
        # Calculate entropy
        entropy = -np.sum(flat_attention * np.log(flat_attention))
        return float(entropy)

class CognitiveResourceManager:
    """Manages computational resources across cognitive processes"""
    
    def __init__(self, total_capacity: int = 1000):
        self.total_capacity = total_capacity
        self.allocated_resources: Dict[str, int] = {}
        self.resource_priorities: Dict[str, float] = {}
        self.usage_history: List[Dict[str, Any]] = []
        
    async def allocate_resources(self, process_id: str, requested_amount: int, 
                               priority: float = 0.5) -> int:
        """Allocate computational resources to process"""
        
        # Calculate available resources
        currently_allocated = sum(self.allocated_resources.values())
        available = self.total_capacity - currently_allocated
        
        # Determine allocation based on priority and availability
        if available >= requested_amount:
            # Sufficient resources available
            allocated = requested_amount
        else:
            # Need to reallocate based on priorities
            allocated = await self._reallocate_resources(process_id, requested_amount, priority)
        
        # Update allocations
        self.allocated_resources[process_id] = allocated
        self.resource_priorities[process_id] = priority
        
        # Record usage
        self.usage_history.append({
            'timestamp': time.time(),
            'process_id': process_id,
            'requested': requested_amount,
            'allocated': allocated,
            'priority': priority,
            'utilization': (currently_allocated + allocated) / self.total_capacity
        })
        
        return allocated
    
    async def _reallocate_resources(self, new_process_id: str, 
                                  requested_amount: int, priority: float) -> int:
        """Reallocate resources based on priorities"""
        
        # Find processes with lower priority
        lower_priority_processes = [
            (pid, amount) for pid, amount in self.allocated_resources.items()
            if self.resource_priorities.get(pid, 0.5) < priority
        ]
        
        # Sort by priority (lowest first)
        lower_priority_processes.sort(key=lambda x: self.resource_priorities.get(x[0], 0.5))
        
        # Deallocate from lower priority processes
        freed_resources = 0
        for process_id, allocated_amount in lower_priority_processes:
            if freed_resources >= requested_amount:
                break
            
            # Reduce allocation by half
            reduction = min(allocated_amount // 2, requested_amount - freed_resources)
            self.allocated_resources[process_id] -= reduction
            freed_resources += reduction
        
        # Calculate final allocation
        currently_allocated = sum(self.allocated_resources.values())
        available = self.total_capacity - currently_allocated
        
        return min(requested_amount, available + freed_resources)
    
    def release_resources(self, process_id: str):
        """Release resources allocated to process"""
        if process_id in self.allocated_resources:
            del self.allocated_resources[process_id]
        if process_id in self.resource_priorities:
            del self.resource_priorities[process_id]
    
    def get_resource_utilization(self) -> Dict[str, Any]:
        """Get current resource utilization statistics"""
        total_allocated = sum(self.allocated_resources.values())
        
        return {
            'total_capacity': self.total_capacity,
            'total_allocated': total_allocated,
            'utilization_percentage': (total_allocated / self.total_capacity) * 100,
            'available_resources': self.total_capacity - total_allocated,
            'active_processes': len(self.allocated_resources),
            'process_allocations': self.allocated_resources.copy()
        }
```

### 3. Integration: Comprehensive Cognitive System

```python
class CognitiveSystem:
    """Integrated cognitive system combining all memory types and attention"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Memory systems
        self.working_memory = WorkingMemorySystem(
            capacity=config.get('working_memory_capacity', 7)
        )
        self.episodic_memory = EpisodicMemorySystem(
            max_episodes=config.get('max_episodes', 10000)
        )
        self.semantic_memory = SemanticMemorySystem()
        
        # Attention and resource management
        self.attention_mechanism = AttentionMechanism(
            d_model=config.get('attention_dim', 512),
            num_heads=config.get('attention_heads', 8)
        )
        self.resource_manager = CognitiveResourceManager(
            total_capacity=config.get('cognitive_capacity', 1000)
        )
        
        # Processing coordination
        self.processing_coordinator = ProcessingCoordinator(self)
        
    async def process_input(self, input_data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process input through integrated cognitive system"""
        
        processing_id = f"process_{int(time.time() * 1000)}"
        
        try:
            # Allocate cognitive resources
            allocated_resources = await self.resource_manager.allocate_resources(
                processing_id, 
                requested_amount=200,  # Base processing requirement
                priority=context.get('priority', 0.5)
            )
            
            # Coordinate processing across memory systems
            result = await self.processing_coordinator.coordinate_processing(
                input_data, context, allocated_resources
            )
            
            return result
            
        finally:
            # Release resources
            self.resource_manager.release_resources(processing_id)
    
    async def learn_from_experience(self, experience: Dict[str, Any]):
        """Update cognitive system based on experience"""
        
        # Store in episodic memory
        episode_id = await self.episodic_memory.store_episode(
            context=experience.get('context', {}),
            actions=experience.get('actions', []),
            outcomes=experience.get('outcomes', {}),
            emotional_valence=experience.get('emotional_valence', 0.0),
            importance=experience.get('importance', 0.5)
        )
        
        # Extract concepts for semantic memory
        concepts = experience.get('concepts', [])
        for concept_data in concepts:
            await self.semantic_memory.add_concept(
                concept_data['id'],
                concept_data['name'],
                concept_data.get('attributes', {})
            )
        
        # Update relationships
        relationships = experience.get('relationships', [])
        for rel in relationships:
            await self.semantic_memory.add_relationship(
                rel['concept1'],
                rel['concept2'],
                rel['relation_type'],
                rel.get('strength', 1.0)
            )
        
        return episode_id
    
    def get_cognitive_state(self) -> Dict[str, Any]:
        """Get comprehensive cognitive state"""
        return {
            'working_memory': self.working_memory.get_attention_state(),
            'episodic_memory': {
                'total_episodes': len(self.episodic_memory.episodes),
                'recent_episodes': list(self.episodic_memory.episodes.keys())[-5:]
            },
            'semantic_memory': {
                'total_concepts': len(self.semantic_memory.concepts),
                'total_relationships': self.semantic_memory.concept_graph.number_of_edges()
            },
            'attention_patterns': self.attention_mechanism.analyze_attention_patterns(),
            'resource_utilization': self.resource_manager.get_resource_utilization()
        }

class ProcessingCoordinator:
    """Coordinates processing across different cognitive components"""
    
    def __init__(self, cognitive_system):
        self.cognitive_system = cognitive_system
    
    async def coordinate_processing(self, input_data: Any, context: Dict[str, Any], 
                                  available_resources: int) -> Dict[str, Any]:
        """Coordinate processing across cognitive components"""
        
        results = {}
        
        # Phase 1: Working memory processing
        working_memory_resources = min(50, available_resources // 4)
        if working_memory_resources > 0:
            results['working_memory'] = await self._process_working_memory(
                input_data, context, working_memory_resources
            )
        
        # Phase 2: Retrieve relevant episodes
        episodic_resources = min(100, available_resources // 3)
        if episodic_resources > 0:
            results['episodic_retrieval'] = await self._retrieve_relevant_episodes(
                input_data, context, episodic_resources
            )
        
        # Phase 3: Semantic knowledge retrieval
        semantic_resources = min(80, available_resources // 4)
        if semantic_resources > 0:
            results['semantic_knowledge'] = await self._retrieve_semantic_knowledge(
                input_data, context, semantic_resources
            )
        
        # Phase 4: Attention-focused integration
        remaining_resources = available_resources - sum([
            working_memory_resources, episodic_resources, semantic_resources
        ])
        
        if remaining_resources > 0:
            results['integrated_response'] = await self._integrate_knowledge(
                input_data, context, results, remaining_resources
            )
        
        return results
    
    async def _process_working_memory(self, input_data: Any, context: Dict[str, Any], 
                                    resources: int) -> Dict[str, Any]:
        """Process input through working memory"""
        
        # Store input in working memory
        input_key = f"input_{int(time.time() * 1000)}"
        await self.cognitive_system.working_memory.store(
            input_key,
            input_data,
            importance=context.get('importance', 0.5),
            context_tags=context.get('tags', [])
        )
        
        # Find related items in working memory
        related_items = await self.cognitive_system.working_memory.find_by_context(
            context.get('tags', [])
        )
        
        return {
            'stored_key': input_key,
            'related_items': related_items,
            'attention_state': self.cognitive_system.working_memory.get_attention_state()
        }
    
    async def _retrieve_relevant_episodes(self, input_data: Any, context: Dict[str, Any], 
                                        resources: int) -> List[Episode]:
        """Retrieve relevant episodes from episodic memory"""
        
        similar_episodes = await self.cognitive_system.episodic_memory.retrieve_similar_episodes(
            context=context,
            actions=context.get('recent_actions', []),
            similarity_threshold=0.6,
            max_results=min(10, resources // 10)
        )
        
        return similar_episodes
    
    async def _retrieve_semantic_knowledge(self, input_data: Any, context: Dict[str, Any], 
                                         resources: int) -> Dict[str, Any]:
        """Retrieve relevant semantic knowledge"""
        
        relevant_concepts = []
        
        # Extract key concepts from input
        input_concepts = context.get('concepts', [])
        
        for concept_id in input_concepts:
            related_concepts = await self.cognitive_system.semantic_memory.find_related_concepts(
                concept_id,
                max_distance=2,
                min_activation=0.2
            )
            relevant_concepts.extend(related_concepts)
        
        return {
            'input_concepts': input_concepts,
            'related_concepts': relevant_concepts[:resources // 5]  # Limit based on resources
        }
    
    async def _integrate_knowledge(self, input_data: Any, context: Dict[str, Any], 
                                 retrieved_knowledge: Dict[str, Any], 
                                 resources: int) -> Dict[str, Any]:
        """Integrate knowledge from different memory systems using attention"""
        
        # Prepare knowledge vectors for attention
        knowledge_items = []
        
        # Add working memory items
        if 'working_memory' in retrieved_knowledge:
            for item_key, item_content in retrieved_knowledge['working_memory'].get('related_items', []):
                knowledge_items.append({
                    'source': 'working_memory',
                    'key': item_key,
                    'content': item_content,
                    'vector': await self._content_to_vector(item_content)
                })
        
        # Add episodic memories
        if 'episodic_retrieval' in retrieved_knowledge:
            for episode in retrieved_knowledge['episodic_retrieval']:
                knowledge_items.append({
                    'source': 'episodic_memory',
                    'key': episode.id,
                    'content': {
                        'context': episode.context,
                        'actions': episode.actions,
                        'outcomes': episode.outcomes
                    },
                    'vector': await self._content_to_vector(episode.context)
                })
        
        # Add semantic knowledge
        if 'semantic_knowledge' in retrieved_knowledge:
            for concept_id, activation, path in retrieved_knowledge['semantic_knowledge'].get('related_concepts', []):
                concept = self.cognitive_system.semantic_memory.concepts.get(concept_id)
                if concept:
                    knowledge_items.append({
                        'source': 'semantic_memory',
                        'key': concept_id,
                        'content': {
                            'name': concept.name,
                            'attributes': concept.attributes,
                            'activation': activation
                        },
                        'vector': await self._content_to_vector(concept.attributes)
                    })
        
        # Apply attention mechanism if we have knowledge items
        if knowledge_items and len(knowledge_items) > 1:
            # Create input vector from input data
            input_vector = await self._content_to_vector(input_data)
            
            # Create knowledge matrix
            knowledge_vectors = np.array([item['vector'] for item in knowledge_items])
            
            # Apply attention
            attended_knowledge, attention_weights = await self.cognitive_system.attention_mechanism.compute_attention(
                queries=input_vector.reshape(1, 1, -1),
                keys=knowledge_vectors.reshape(1, len(knowledge_items), -1),
                values=knowledge_vectors.reshape(1, len(knowledge_items), -1)
            )
            
            # Rank knowledge items by attention
            attention_scores = attention_weights[0, 0, :] if attention_weights.ndim >= 3 else attention_weights.flatten()
            
            ranked_knowledge = []
            for i, (item, score) in enumerate(zip(knowledge_items, attention_scores)):
                ranked_knowledge.append({
                    'item': item,
                    'attention_score': float(score),
                    'rank': i
                })
            
            ranked_knowledge.sort(key=lambda x: x['attention_score'], reverse=True)
            
            return {
                'integrated_knowledge': ranked_knowledge[:10],  # Top 10 most relevant
                'attention_distribution': attention_scores.tolist(),
                'integration_summary': await self._summarize_integration(ranked_knowledge[:5])
            }
        
        return {
            'integrated_knowledge': knowledge_items,
            'message': 'Limited knowledge available for integration'
        }
    
    async def _content_to_vector(self, content: Any) -> np.ndarray:
        """Convert content to vector representation"""
        # Simplified vectorization - in practice, use proper embedding models
        if isinstance(content, dict):
            content_str = json.dumps(content, sort_keys=True)
        else:
            content_str = str(content)
        
        # Create simple hash-based vector
        hash_val = hashlib.md5(content_str.encode()).hexdigest()
        vector = np.array([int(hash_val[i:i+2], 16) / 255.0 for i in range(0, 32, 2)])
        
        # Pad or truncate to fixed size
        target_size = 512
        if len(vector) < target_size:
            vector = np.pad(vector, (0, target_size - len(vector)))
        else:
            vector = vector[:target_size]
        
        return vector
    
    async def _summarize_integration(self, top_knowledge: List[Dict[str, Any]]) -> str:
        """Create summary of integrated knowledge"""
        
        sources = set()
        key_concepts = []
        
        for knowledge_item in top_knowledge:
            item = knowledge_item['item']
            sources.add(item['source'])
            
            if item['source'] == 'semantic_memory':
                key_concepts.append(item['content']['name'])
        
        summary = f"Integrated knowledge from {len(sources)} sources: {', '.join(sources)}. "
        
        if key_concepts:
            summary += f"Key concepts: {', '.join(key_concepts[:3])}."
        
        return summary
```

---

## Practical Exercises

### Exercise 1: Memory System Implementation (60 minutes)

**Objective**: Build a complete memory system for a personal assistant agent.

**Requirements**:
```python
class PersonalAssistantMemory:
    """Memory system for personal assistant"""
    
    def __init__(self):
        # Initialize all memory components
        pass
    
    async def remember_user_preference(self, preference_type: str, 
                                     preference_value: Any, context: Dict):
        # Store in appropriate memory system
        pass
    
    async def recall_similar_situations(self, current_context: Dict) -> List[Any]:
        # Retrieve similar past situations
        pass
    
    async def learn_from_interaction(self, interaction_data: Dict):
        # Update memory based on interaction outcome
        pass
    
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        # Build user profile from memory systems
        pass
```

### Exercise 2: Attention Mechanism Design (45 minutes)

**Objective**: Implement a dynamic attention mechanism for multi-task agents.

**Scenario**: An agent managing multiple concurrent conversations needs to allocate attention based on urgency, user importance, and conversation complexity.

**Tasks**:
1. Design attention scoring function
2. Implement dynamic resource allocation
3. Create attention switching mechanisms
4. Test with simulated concurrent tasks

### Exercise 3: Knowledge Graph Integration (75 minutes)

**Objective**: Build a semantic memory system using a real graph database.

**Requirements**:
- Use Neo4j or ArangoDB
- Implement concept relationship learning
- Create inference mechanisms
- Build knowledge export/import capabilities

**Domain**: Medical diagnosis support system with symptoms, diseases, and treatments.

---

## Real-World Case Studies

### Case Study 1: IBM Watson's Memory Architecture

**Background**: IBM Watson's Jeopardy! victory demonstrated advanced memory and knowledge integration capabilities.

**Memory Architecture Components**:

1. **Structured Knowledge Bases**:
   - DBpedia: 3.7 million entities
   - WordNet: 155,000 words and phrases  
   - Yago: 2 million entities
   - Freebase: 12 million topics

2. **Unstructured Text Corpus**:
   - 200 million pages of content
   - Encyclopedias, dictionaries, thesauri
   - News articles and reference materials
   - Literary works and historical documents

**Technical Implementation**:
```python
class WatsonMemorySystem:
    def __init__(self):
        self.structured_knowledge = StructuredKnowledgeBase()
        self.unstructured_corpus = TextCorpus()
        self.question_analyzer = QuestionAnalyzer()
        self.hypothesis_generator = HypothesisGenerator()
        self.evidence_retriever = EvidenceRetriever()
        self.ranking_system = RankingSystem()
    
    async def answer_question(self, question: str) -> Dict[str, Any]:
        # Parse question
        parsed_question = await self.question_analyzer.parse(question)
        
        # Generate candidate answers
        hypotheses = await self.hypothesis_generator.generate(parsed_question)
        
        # Retrieve supporting evidence
        evidence_results = []
        for hypothesis in hypotheses:
            evidence = await self.evidence_retriever.find_evidence(
                hypothesis, 
                self.structured_knowledge,
                self.unstructured_corpus
            )
            evidence_results.append((hypothesis, evidence))
        
        # Rank answers by confidence
        ranked_answers = await self.ranking_system.rank(evidence_results)
        
        return {
            'top_answer': ranked_answers[0],
            'confidence': ranked_answers[0]['confidence'],
            'supporting_evidence': ranked_answers[0]['evidence']
        }
```

**Performance Metrics**:
- 85% accuracy on Jeopardy! questions
- 3-second average response time
- 15 terabytes of RAM for knowledge storage
- 2,880 POWER7 processor cores

### Case Study 2: Google's Knowledge Graph Evolution

**Architecture Overview**: Google's Knowledge Graph represents one of the largest semantic memory systems, containing over 500 billion facts about 5 billion entities.

**Key Components**:

1. **Entity Recognition and Linking**:
   - Named entity recognition from web content
   - Entity disambiguation using context
   - Cross-reference validation across sources
   - Confidence scoring for entity relationships

2. **Relationship Extraction**:
   - Pattern-based extraction from text
   - Machine learning models for relation classification
   - Temporal relationship tracking
   - Source credibility weighting

3. **Knowledge Fusion**:
   - Multi-source fact verification
   - Contradiction resolution
   - Temporal fact updates
   - Quality assessment metrics

**Implementation Insights**:
```python
class GoogleKnowledgeGraph:
    def __init__(self):
        self.entity_store = EntityStore()  # 5 billion entities
        self.relation_store = RelationStore()  # 500 billion facts
        self.extraction_pipeline = ExtractionPipeline()
        self.fusion_engine = KnowledgeFusionEngine()
        self.quality_assessor = QualityAssessor()
    
    async def process_web_document(self, document: WebDocument) -> List[Fact]:
        # Extract entities and relations
        extracted_facts = await self.extraction_pipeline.extract(document)
        
        # Validate and fuse with existing knowledge
        validated_facts = []
        for fact in extracted_facts:
            # Check consistency with existing knowledge
            consistency_score = await self.fusion_engine.check_consistency(fact)
            
            # Assess source quality
            quality_score = await self.quality_assessor.assess_source(document.source)
            
            # Calculate overall confidence
            confidence = (consistency_score + quality_score) / 2
            
            if confidence > 0.7:  # Threshold for inclusion
                validated_facts.append(fact._replace(confidence=confidence))
        
        # Update knowledge graph
        await self.entity_store.update_entities(validated_facts)
        await self.relation_store.update_relations(validated_facts)
        
        return validated_facts
    
    async def answer_query(self, query: str, context: Dict = None) -> Dict[str, Any]:
        # Parse query into structured form
        structured_query = await self.query_parser.parse(query, context)
        
        # Retrieve relevant entities and relations
        candidate_entities = await self.entity_store.find_entities(
            structured_query.entity_mentions
        )
        
        # Find connecting paths
        connection_paths = await self.relation_store.find_paths(
            candidate_entities, 
            max_depth=3
        )
        
        # Rank paths by relevance and confidence
        ranked_paths = await self.ranking_system.rank_paths(
            connection_paths, structured_query
        )
        
        return {
            'answer': await self.answer_generator.generate_answer(ranked_paths[0]),
            'confidence': ranked_paths[0].confidence,
            'supporting_paths': ranked_paths[:3]
        }
```

**Scale and Impact**:
- 70+ languages supported
- 1 billion queries processed daily
- 99.99% availability SLA
- Sub-100ms query response time for 90% of queries

### Case Study 3: DeepMind's Differentiable Neural Computer

**Background**: The Differentiable Neural Computer (DNC) represents a breakthrough in neural memory architectures, combining neural networks with external memory systems.

**Architecture Innovation**:

1. **Controller Network**: Neural network that reads from and writes to external memory
2. **Memory Matrix**: Large external memory bank with content-based addressing  
3. **Read/Write Heads**: Attention mechanisms for memory access
4. **Temporal Linking**: Maintains order of memory writes for sequential processing

**Technical Implementation**:
```python
class DifferentiableNeuralComputer:
    def __init__(self, memory_size: int = 512, word_size: int = 64, num_read_heads: int = 4):
        self.memory_size = memory_size
        self.word_size = word_size
        self.num_read_heads = num_read_heads
        
        # Memory matrix
        self.memory = np.zeros((memory_size, word_size))
        
        # Usage tracking
        self.usage_vector = np.zeros(memory_size)
        self.precedence_weights = np.zeros(memory_size)
        
        # Temporal linking
        self.link_matrix = np.zeros((memory_size, memory_size))
        
        # Read/write heads
        self.read_weights = np.zeros((num_read_heads, memory_size))
        self.write_weights = np.zeros(memory_size)
        
        # Controller network (simplified)
        self.controller = ControllerNetwork()
    
    async def forward_pass(self, input_data: np.ndarray, 
                          previous_state: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Forward pass through DNC"""
        
        # Controller processes input and previous reads
        previous_reads = previous_state.get('reads', np.zeros((self.num_read_heads, self.word_size)))
        controller_input = np.concatenate([input_data, previous_reads.flatten()])
        
        # Controller output includes interface parameters
        controller_output, interface_params = await self.controller.process(controller_input)
        
        # Parse interface parameters
        read_keys, read_strengths, write_key, write_strength = self._parse_interface_params(interface_params)
        write_vector, erase_vector, free_gates, allocation_gate, write_gate = self._parse_write_params(interface_params)
        
        # Memory allocation
        allocation_weights = await self._allocate_memory()
        
        # Content-based addressing
        content_weights = await self._content_addressing(write_key, write_strength)
        
        # Compute write weights
        self.write_weights = write_gate * (
            allocation_gate * allocation_weights + 
            (1 - allocation_gate) * content_weights
        )
        
        # Write to memory
        await self._write_to_memory(write_vector, erase_vector)
        
        # Update usage and temporal links
        await self._update_usage_and_links()
        
        # Read from memory
        reads = await self._read_from_memory(read_keys, read_strengths)
        
        # Prepare output
        output = controller_output  # Could combine with reads
        
        new_state = {
            'reads': reads,
            'memory': self.memory.copy(),
            'usage': self.usage_vector.copy(),
            'write_weights': self.write_weights.copy()
        }
        
        return output, new_state
    
    async def _allocate_memory(self) -> np.ndarray:
        """Allocate memory based on usage"""
        # Sort locations by usage (ascending)
        sorted_indices = np.argsort(self.usage_vector)
        
        # Allocation weights favor least used locations
        allocation_weights = np.zeros(self.memory_size)
        for i, idx in enumerate(sorted_indices):
            allocation_weights[idx] = (1 - self.usage_vector[idx]) * np.prod([
                self.usage_vector[sorted_indices[j]] for j in range(i)
            ])
        
        return allocation_weights / (np.sum(allocation_weights) + 1e-16)
    
    async def _content_addressing(self, key: np.ndarray, strength: float) -> np.ndarray:
        """Content-based memory addressing"""
        # Cosine similarity between key and memory rows
        key_norm = np.linalg.norm(key)
        memory_norms = np.linalg.norm(self.memory, axis=1)
        
        # Avoid division by zero
        valid_norms = (key_norm > 1e-16) & (memory_norms > 1e-16)
        
        similarities = np.zeros(self.memory_size)
        similarities[valid_norms] = np.dot(self.memory[valid_norms], key) / (
            key_norm * memory_norms[valid_norms]
        )
        
        # Apply strength and softmax
        weighted_similarities = strength * similarities
        content_weights = await self._softmax(weighted_similarities)
        
        return content_weights
    
    async def _write_to_memory(self, write_vector: np.ndarray, erase_vector: np.ndarray):
        """Write to memory with erase and add operations"""
        # Erase operation
        erase_matrix = np.outer(self.write_weights, erase_vector)
        self.memory = self.memory * (1 - erase_matrix)
        
        # Add operation  
        add_matrix = np.outer(self.write_weights, write_vector)
        self.memory = self.memory + add_matrix
    
    async def _read_from_memory(self, read_keys: List[np.ndarray], 
                               read_strengths: List[float]) -> np.ndarray:
        """Read from memory using multiple read heads"""
        reads = []
        
        for head_idx in range(self.num_read_heads):
            # Content-based addressing for this read head
            content_weights = await self._content_addressing(
                read_keys[head_idx], 
                read_strengths[head_idx]
            )
            
            # Temporal addressing (simplified)
            temporal_weights = await self._temporal_addressing(head_idx)
            
            # Combine addressing modes
            read_weights = 0.5 * content_weights + 0.5 * temporal_weights
            self.read_weights[head_idx] = read_weights
            
            # Read from memory
            read_vector = np.dot(read_weights, self.memory)
            reads.append(read_vector)
        
        return np.array(reads)
    
    async def _temporal_addressing(self, head_idx: int) -> np.ndarray:
        """Temporal addressing using link matrix"""
        # Forward and backward temporal reads
        forward_weights = np.dot(self.read_weights[head_idx], self.link_matrix)
        backward_weights = np.dot(self.read_weights[head_idx], self.link_matrix.T)
        
        # Could add mode interpolation here
        return (forward_weights + backward_weights) / 2
```

**Performance Achievements**:
- 95% accuracy on algorithmic tasks requiring memory
- Generalizes to sequence lengths 10x longer than training
- Maintains stable memory over 1000+ time steps
- Learns complex memory access patterns

---

## Assessment Questions

### Knowledge Check (10 Questions)

1. **Memory Architecture**: Design a memory architecture for an agent that needs to remember user preferences, learn from mistakes, and make analogical reasoning. Specify the interaction between different memory types.

2. **Attention Optimization**: You have an agent processing 1000 concurrent conversations. The attention mechanism is consuming 80% of computational resources. Propose 3 optimization strategies with expected performance improvements.

3. **Knowledge Representation**: Compare graph-based vs. vector-based knowledge representation for the following scenarios:
   - Legal document analysis
   - Medical diagnosis
   - Creative writing assistance

4. **Memory Consolidation**: Design an algorithm for consolidating episodic memories into semantic knowledge. Consider accuracy preservation, computational efficiency, and knowledge generalization.

5. **Retrieval Performance**: An agent's semantic memory contains 10 million concepts. Design a retrieval system that can find relevant concepts in under 50ms. Include indexing strategy, similarity computation, and caching.

6. **Learning Integration**: How would you integrate reinforcement learning with episodic memory to improve agent decision-making? Provide pseudocode for the integration algorithm.

7. **Context Switching**: Design a context management system for an agent that switches between different domains (e.g., technical support → sales → scheduling). How do you maintain relevant context while preventing interference?

8. **Memory Debugging**: Given these memory system symptoms, diagnose the issues and propose solutions:
   - Semantic retrieval accuracy dropping over time
   - Working memory frequently at capacity
   - Attention mechanism showing uniform distribution

9. **Scalability Analysis**: Analyze the computational complexity of each memory component. How does performance scale with:
   - Number of stored episodes
   - Size of semantic knowledge graph  
   - Working memory capacity

10. **Future Architecture**: Propose how memory systems might evolve with neuromorphic hardware. What new capabilities would become possible, and what are the implementation challenges?

---

## Additional Resources

### Memory Systems Research
- **"The Organization of Memory" (Tulving)** - Foundational episodic vs semantic memory research
- **"Principles of Psychology" (James)** - Classic work on memory and attention
- **"Memory: From Mind to Molecules" (Squire & Kandel)** - Neuroscientific foundations
- **"The Adaptive Character of Thought" (Anderson)** - ACT-R cognitive architecture

### Knowledge Representation
- **"Semantic Networks" (Quillian)** - Foundational semantic network theory
- **"Knowledge Representation and Reasoning" (Brachman & Levesque)** - Comprehensive AI knowledge representation
- **"The Semantic Web" (Berners-Lee et al.)** - Web-scale knowledge representation
- **"Graph Databases" (Robinson et al.)** - Practical graph database implementation

### Attention Mechanisms
- **"Attention Is All You Need" (Vaswani et al.)** - Transformer attention mechanism
- **"Neural Machine Translation by Jointly Learning to Align and Translate" (Bahdanau et al.)** - Attention in sequence models
- **"Show, Attend and Tell" (Xu et al.)** - Visual attention mechanisms
- **"Hierarchical Attention Networks" (Yang et al.)** - Multi-level attention

### Implementation Tools
- **Neo4j**: Graph database for semantic memory
- **Redis**: In-memory data structures for working memory
- **Elasticsearch**: Full-text search and retrieval
- **Faiss**: Similarity search for vector embeddings
- **NetworkX**: Graph analysis and algorithms

---

## Next Day Preview: Modern Development Frameworks

Tomorrow we'll explore the frameworks and platforms that enable rapid development of production-ready agentic AI systems:

- **LangChain/LangGraph Deep Dive**: Advanced patterns, custom tools, and production deployment
- **OpenAI Agents SDK**: Function calling, structured outputs, and enterprise integration  
- **Framework Comparison**: Performance benchmarks, use case analysis, and selection criteria
- **Custom Framework Development**: Building specialized agent frameworks for specific domains
- **Integration Patterns**: Connecting multiple frameworks and managing framework migrations

**Preparation Tasks**:
1. Install LangChain, LangGraph, and OpenAI libraries
2. Set up development environment with API keys
3. Review function calling documentation for OpenAI
4. Explore LangSmith for agent monitoring and debugging

**Key Question**: How do you choose the right framework for your specific use case, and when should you build a custom solution?

Tomorrow's content will show you how to leverage these powerful frameworks while understanding their strengths, limitations, and integration patterns for enterprise deployments.