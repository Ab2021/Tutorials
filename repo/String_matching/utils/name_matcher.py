"""
Name matching utilities with nickname support.
Uses fuzzy string matching to compare claimant names.
"""

from rapidfuzz import fuzz

from config.settings import USA_NAMES_NICKNAMES, NICKNAME_TO_FORMAL


def get_nickname_variants(name: str) -> set:
    """
    Get all nickname variants for a given name.
    
    Args:
        name: First name to look up
        
    Returns:
        Set of all variants (including original and nicknames)
    """
    name_upper = name.strip().upper()
    
    if name_upper in NICKNAME_TO_FORMAL:
        return NICKNAME_TO_FORMAL[name_upper]
    
    return {name_upper}


def extract_name_parts(full_name: str) -> tuple:
    """
    Extract first name and last name from full name.
    Assumes format: "LASTNAME FIRSTNAME [MIDDLENAME]" or "FIRSTNAME LASTNAME"
    
    Returns:
        Tuple of (first_name, last_name, middle_parts)
    """
    parts = full_name.strip().upper().split()
    
    if len(parts) == 0:
        return ("", "", "")
    elif len(parts) == 1:
        return (parts[0], "", "")
    elif len(parts) == 2:
        # Could be "LASTNAME FIRSTNAME" or "FIRSTNAME LASTNAME"
        # Return both parts, let matching handle ambiguity
        return (parts[1], parts[0], "")
    else:
        # Assume "LASTNAME FIRSTNAME MIDDLE..."
        return (parts[1], parts[0], " ".join(parts[2:]))


def names_match_with_nicknames(name1: str, name2: str) -> bool:
    """
    Check if two names match, considering nickname variants.
    
    Returns:
        True if names match (including nickname variants)
    """
    variants1 = get_nickname_variants(name1)
    variants2 = get_nickname_variants(name2)
    
    # Check if there's any overlap in variants
    return bool(variants1 & variants2)


def compute_name_similarity(name_a: str, name_b: str) -> float:
    """
    Compute similarity score between two names.
    Uses fuzzy matching with nickname boosting.
    
    Args:
        name_a: First full name
        name_b: Second full name
        
    Returns:
        Float similarity score between 0 and 1
    """
    if not name_a or not name_b:
        return 0.0
    
    name_a = name_a.strip().upper()
    name_b = name_b.strip().upper()
    
    if not name_a or not name_b:
        return 0.0
    
    # Direct fuzzy matching using token_set_ratio (handles reordering)
    fuzzy_score = fuzz.token_set_ratio(name_a, name_b) / 100.0
    
    # Extract name parts for nickname checking
    first_a, last_a, _ = extract_name_parts(name_a)
    first_b, last_b, _ = extract_name_parts(name_b)
    
    # Nickname boost: if first names are nickname variants, boost score
    nickname_boost = 0.0
    if first_a and first_b:
        if names_match_with_nicknames(first_a, first_b):
            nickname_boost = 0.15
    
    # Last name match boost
    last_name_boost = 0.0
    if last_a and last_b:
        last_name_sim = fuzz.ratio(last_a, last_b) / 100.0
        if last_name_sim >= 0.9:
            last_name_boost = 0.10
    
    # Combined score with boosts (capped at 1.0)
    final_score = min(1.0, fuzzy_score + nickname_boost + last_name_boost)
    
    return round(final_score, 2)


def compute_name_similarity_detailed(name_a: str, name_b: str) -> dict:
    """
    Compute detailed name similarity with component breakdown.
    
    Returns:
        Dict with overall score and component details
    """
    if not name_a or not name_b:
        return {
            'overall_score': 0.0,
            'fuzzy_score': 0.0,
            'nickname_match': False,
            'last_name_match': False
        }
    
    name_a = name_a.strip().upper()
    name_b = name_b.strip().upper()
    
    fuzzy_score = fuzz.token_set_ratio(name_a, name_b) / 100.0
    
    first_a, last_a, _ = extract_name_parts(name_a)
    first_b, last_b, _ = extract_name_parts(name_b)
    
    nickname_match = False
    if first_a and first_b:
        nickname_match = names_match_with_nicknames(first_a, first_b)
    
    last_name_match = False
    if last_a and last_b:
        last_name_match = fuzz.ratio(last_a, last_b) >= 90
    
    # Calculate boosts
    nickname_boost = 0.15 if nickname_match else 0.0
    last_name_boost = 0.10 if last_name_match else 0.0
    
    overall_score = min(1.0, fuzzy_score + nickname_boost + last_name_boost)
    
    return {
        'overall_score': round(overall_score, 2),
        'fuzzy_score': round(fuzzy_score, 2),
        'nickname_match': nickname_match,
        'last_name_match': last_name_match
    }


# =============================================================================
# OPTIONAL: BERT/SentenceTransformer-based Name Matching
# =============================================================================
# 
# To use semantic embeddings for better name matching, uncomment below:
#
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
#
# class BertNameMatcher:
#     """
#     BERT-based name matcher using sentence transformers.
#     Provides semantic understanding of names.
#     
#     Usage:
#         matcher = BertNameMatcher()
#         score = matcher.compute_similarity("JOHN SMITH", "JOHNNY SMITH")
#     """
#     
#     def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
#         """
#         Initialize with a sentence transformer model.
#         
#         Args:
#             model_name: HuggingFace model name (default: all-MiniLM-L6-v2)
#         """
#         self.model = SentenceTransformer(model_name)
#     
#     def compute_similarity(self, name_a: str, name_b: str) -> float:
#         """
#         Compute semantic similarity between two names using embeddings.
#         """
#         if not name_a or not name_b:
#             return 0.0
#         
#         embeddings = self.model.encode([name_a, name_b])
#         similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
#         return float(similarity)
#
# =============================================================================
