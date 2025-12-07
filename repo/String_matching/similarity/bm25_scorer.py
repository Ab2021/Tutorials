"""
BM25-based similarity scoring module.
Replaces GPT-based matching with deterministic BM25 + fuzzy matching.
"""

from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz
import numpy as np

from config.settings import (
    X_TPA_NAME_THRESHOLD, X_TPA_DESC_THRESHOLD, X_TPA_STATE_THRESHOLD,
    TPA_NAME_THRESHOLD, TPA_DESC_THRESHOLD, TPA_STATE_THRESHOLD
)
from utils.name_matcher import compute_name_similarity


class BM25Matcher:
    """
    BM25-based similarity scorer for medium to small length text.
    Replaces GPT-based matching with deterministic scoring.
    
    This matcher combines:
    - BM25 scoring for description similarity (semantic-ish matching)
    - Fuzzy string matching for character-level similarity
    - Name matching with nickname support
    - State matching (exact or based on proximity)
    
    Usage:
        matcher = BM25Matcher(mode='x_tpa')  # or 'tpa_cluster'
        result = matcher.match_records(record_a, record_b)
    """
    
    def __init__(self, mode: str = 'x_tpa', corpus: list = None):
        """
        Initialize BM25 Matcher.
        
        Args:
            mode: 'x_tpa' for X-TPA linkage, 'tpa_cluster' for TPA clustering
            corpus: Optional pre-built corpus for BM25 indexing
        """
        self.mode = mode
        self.corpus = corpus
        self.bm25 = None
        
        # Set thresholds based on mode
        if mode == 'x_tpa':
            self.name_threshold = X_TPA_NAME_THRESHOLD
            self.desc_threshold = X_TPA_DESC_THRESHOLD
            self.state_threshold = X_TPA_STATE_THRESHOLD
        else:  # tpa_cluster
            self.name_threshold = TPA_NAME_THRESHOLD
            self.desc_threshold = TPA_DESC_THRESHOLD
            self.state_threshold = TPA_STATE_THRESHOLD
        
        if corpus:
            self._build_index(corpus)
    
    def _build_index(self, corpus: list):
        """Build BM25 index from tokenized corpus."""
        tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def _tokenize(self, text: str) -> list:
        """Tokenize text for BM25."""
        if not text:
            return []
        return text.lower().split()
    
    def compute_description_similarity(self, text_a: str, text_b: str) -> float:
        """
        Compute description similarity using BM25 scoring combined with fuzzy matching.
        
        For pairwise comparison of short texts, we:
        1. Use BM25 score (normalized to 0-1)
        2. Combine with fuzzy token_set_ratio for character-level matching
        
        Args:
            text_a: First description text
            text_b: Second description text
            
        Returns:
            Float similarity score between 0 and 1
        """
        if not text_a or not text_b:
            return 0.0
        
        text_a = str(text_a).strip()
        text_b = str(text_b).strip()
        
        if not text_a or not text_b:
            return 0.0
        
        # Tokenize
        tokens_a = self._tokenize(text_a)
        tokens_b = self._tokenize(text_b)
        
        if not tokens_a or not tokens_b:
            return 0.0
        
        # Build temporary BM25 index with text_b as corpus
        try:
            bm25 = BM25Okapi([tokens_b])
            score = bm25.get_scores(tokens_a)[0]
            
            # Normalize score to 0-1 range using sigmoid-like transformation
            # BM25 scores can vary widely, so we use a soft normalization
            normalized_score = min(1.0, score / (score + 1.0)) if score > 0 else 0.0
        except Exception:
            normalized_score = 0.0
        
        # Fuzzy matching for character-level similarity
        fuzzy_score = fuzz.token_set_ratio(text_a, text_b) / 100.0
        
        # Partial ratio for handling substring matches
        partial_score = fuzz.partial_ratio(text_a, text_b) / 100.0
        
        # Weighted combination:
        # - BM25 for term overlap (semantic-ish)
        # - Token set ratio for word-level matching with reordering
        # - Partial ratio for substring matching
        combined_score = (
            0.35 * normalized_score + 
            0.40 * fuzzy_score + 
            0.25 * partial_score
        )
        
        return round(combined_score, 2)
    
    def compute_state_similarity(self, state_a: str, state_b: str) -> float:
        """
        Compute state similarity.
        
        Returns:
            1.0 for exact match
            None if either state is unknown/missing
            0.0 for non-matching states
        """
        if not state_a or not state_b:
            return None
        
        state_a = str(state_a).strip().upper()
        state_b = str(state_b).strip().upper()
        
        if state_a == 'UNKNOWN' or state_b == 'UNKNOWN':
            return None
        
        if not state_a or not state_b:
            return None
        
        if state_a == state_b:
            return 1.0
        
        # For neighboring states, could add proximity logic here
        # For now, non-matching = 0
        return 0.0
    
    def match_records(self, record_a: dict, record_b: dict) -> dict:
        """
        Main matching function replacing GPT-based matching.
        
        Args:
            record_a: First record with keys:
                      claimant_name, description, state, file_number
            record_b: Second record with same keys
            
        Returns:
            Dict with:
                is_match: bool
                confidence: float (0-1)
                reasons: str
                component_scores: dict
                suggested_link: bool
        """
        # Extract fields
        name_a = record_a.get('claimant_name', '') or record_a.get('tpa_claimant_name', '')
        name_b = record_b.get('claimant_name', '') or record_b.get('X_claimant_name', '') or record_b.get('tpa_claimant_name', '')
        
        desc_a = record_a.get('description', '') or record_a.get('tpa_description', '')
        desc_b = record_b.get('description', '') or record_b.get('X_description', '') or record_b.get('tpa_description', '')
        
        state_a = record_a.get('state', '') or record_a.get('tpa_state', '')
        state_b = record_b.get('state', '') or record_b.get('X_state', '') or record_b.get('tpa_state', '')
        
        # Compute component scores
        name_sim = compute_name_similarity(name_a, name_b)
        desc_sim = self.compute_description_similarity(desc_a, desc_b)
        state_sim = self.compute_state_similarity(state_a, state_b)
        
        # Determine match based on thresholds
        if self.mode == 'x_tpa':
            # For X-TPA: name >= 0.65, description >= 0.30, state >= 0.8 (if available)
            is_match = (
                name_sim >= self.name_threshold and 
                desc_sim >= self.desc_threshold and 
                (state_sim is None or state_sim >= self.state_threshold)
            )
        else:
            # For TPA clustering: description is primary, name is secondary
            # description >= 0.30, name >= 0.40, state >= 0.8 (if available)
            is_match = (
                desc_sim >= self.desc_threshold and
                name_sim >= self.name_threshold and 
                (state_sim is None or state_sim >= self.state_threshold)
            )
        
        # Compute confidence as weighted average
        state_weight = 0.2 if state_sim is not None else 0.0
        name_weight = 0.5 if self.mode == 'x_tpa' else 0.3
        desc_weight = 1.0 - name_weight - state_weight
        
        confidence = (
            name_sim * name_weight + 
            desc_sim * desc_weight + 
            (state_sim or 0.5) * state_weight
        )
        
        # Generate reason string
        reasons = self._generate_reason(name_sim, desc_sim, state_sim)
        
        return {
            'is_match': is_match,
            'confidence': round(confidence, 2),
            'reasons': reasons,
            'component_scores': {
                'name_string_sim': name_sim,
                'description_string_sim': desc_sim,
                'state_string_sim': state_sim
            },
            'suggested_link': is_match
        }
    
    def _generate_reason(self, name_sim: float, desc_sim: float, 
                        state_sim: float) -> str:
        """Generate human-readable reason string for match decision."""
        reasons = []
        
        # Name analysis
        if name_sim >= 0.8:
            reasons.append(f"Strong name match ({name_sim:.2f})")
        elif name_sim >= 0.65:
            reasons.append(f"Good name match ({name_sim:.2f})")
        elif name_sim >= 0.4:
            reasons.append(f"Moderate name match ({name_sim:.2f})")
        else:
            reasons.append(f"Weak name match ({name_sim:.2f})")
        
        # Description analysis
        if desc_sim >= 0.6:
            reasons.append(f"Strong description match ({desc_sim:.2f})")
        elif desc_sim >= 0.3:
            reasons.append(f"Moderate description match ({desc_sim:.2f})")
        else:
            reasons.append(f"Weak description match ({desc_sim:.2f})")
        
        # State analysis
        if state_sim is not None:
            if state_sim >= 1.0:
                reasons.append("States match exactly")
            elif state_sim >= 0.8:
                reasons.append(f"States similar ({state_sim:.2f})")
            else:
                reasons.append("States differ")
        else:
            reasons.append("State comparison skipped (unknown)")
        
        return "; ".join(reasons)


def bm25_match(record_a: dict, record_b: dict, mode: str = 'x_tpa') -> dict:
    """
    Convenience function for BM25-based matching.
    
    Args:
        record_a: First record
        record_b: Second record
        mode: 'x_tpa' or 'tpa_cluster'
        
    Returns:
        Match result dict
    """
    matcher = BM25Matcher(mode=mode)
    return matcher.match_records(record_a, record_b)


# =============================================================================
# OPTIONAL: BERT/SentenceTransformer-based Description Matching
# =============================================================================
#
# For better semantic understanding of descriptions, you can use embeddings.
# This is more computationally expensive but provides better semantic matching.
#
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
#
# class BertDescriptionMatcher:
#     """
#     BERT-based description matcher using sentence transformers.
#     
#     Usage:
#         matcher = BertDescriptionMatcher()
#         score = matcher.compute_similarity(
#             "Employee fell on wet floor",
#             "Worker slipped on slippery surface"
#         )
#     """
#     
#     def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
#         """
#         Initialize with a sentence transformer model.
#         
#         Recommended models:
#         - 'all-MiniLM-L6-v2': Fast, good quality (default)
#         - 'all-mpnet-base-v2': Higher quality, slower
#         - 'paraphrase-MiniLM-L6-v2': Good for paraphrase detection
#         """
#         self.model = SentenceTransformer(model_name)
#         self._embedding_cache = {}
#     
#     def _get_embedding(self, text: str):
#         """Get embedding with caching."""
#         if text not in self._embedding_cache:
#             self._embedding_cache[text] = self.model.encode([text])[0]
#         return self._embedding_cache[text]
#     
#     def compute_similarity(self, text_a: str, text_b: str) -> float:
#         """
#         Compute semantic similarity between two descriptions.
#         """
#         if not text_a or not text_b:
#             return 0.0
#         
#         emb_a = self._get_embedding(text_a)
#         emb_b = self._get_embedding(text_b)
#         
#         similarity = cosine_similarity([emb_a], [emb_b])[0][0]
#         return float(max(0.0, min(1.0, similarity)))
#
#
# class HybridMatcher(BM25Matcher):
#     """
#     Hybrid matcher combining BM25 with BERT embeddings.
#     
#     Usage:
#         matcher = HybridMatcher(mode='x_tpa')
#         result = matcher.match_records(record_a, record_b)
#     """
#     
#     def __init__(self, mode: str = 'x_tpa', use_bert: bool = True):
#         super().__init__(mode=mode)
#         self.use_bert = use_bert
#         if use_bert:
#             self.bert_matcher = BertDescriptionMatcher()
#     
#     def compute_description_similarity(self, text_a: str, text_b: str) -> float:
#         """
#         Compute description similarity using BM25 + BERT hybrid.
#         """
#         bm25_score = super().compute_description_similarity(text_a, text_b)
#         
#         if self.use_bert:
#             bert_score = self.bert_matcher.compute_similarity(text_a, text_b)
#             # Weighted combination: 40% BM25, 60% BERT
#             return round(0.4 * bm25_score + 0.6 * bert_score, 2)
#         
#         return bm25_score
#
# =============================================================================
