"""
Text processing utilities for string matching.
Contains normalization, cleaning, and extraction functions.
"""

import re
import unicodedata
import pandas as pd

from config.settings import US_CITY_ST_DICT, STATE_ABBREV, COVERAGE_MAP


def normalize_whitespace(s: str) -> str:
    """Collapse multiple spaces into single space and strip."""
    return re.sub(r"\s+", " ", s or "").strip()


def strip_accents(s: str) -> str:
    """Remove Unicode accents from string."""
    if not s:
        return ""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def normalize_text(s) -> str:
    """
    Full text normalization pipeline.
    - Handle None/NaN
    - Strip accents
    - Uppercase
    - Remove punctuation (keep alphanumerics)
    - Normalize whitespace
    """
    # Handle None, NaN, or non-string types
    if not isinstance(s, str):
        try:
            if s is None:
                s = ""
            elif isinstance(s, float):
                if not (s == s):  # NaN check: NaN != NaN
                    s = ""
                else:
                    s = str(s)
            else:
                s = str(s)
        except Exception:
            s = ""
    
    s = strip_accents(s)
    s = s.upper()
    s = re.sub(r"[^A-Z0-9\s]", " ", s)  # Remove punctuation, keep alphanumerics
    s = normalize_whitespace(s)
    return s


def normalize_city_name(city: str) -> str:
    """Normalize city name for lookup."""
    city = city.replace('.', '').replace('-', ' ').strip()
    city = re.sub(r'\s+', ' ', city)  # Collapse multiple spaces
    return city.upper()


# Build normalized city dictionary
US_CITY_ST_DICT_NORM = {normalize_city_name(k): v for k, v in US_CITY_ST_DICT.items()}


def extract_state(row, base_state_col: str, second_state_col: str) -> str:
    """
    Extract US state abbreviation from claim data.
    Tries multiple extraction strategies:
    1. Prefix of unit_number if 2 letters
    2. State code after comma in unit_name
    3. City lookup in unit_name
    """
    unit_number = str(row[base_state_col]) if pd.notna(row[base_state_col]) else ''
    unit_name = str(row[second_state_col]) if pd.notna(row[second_state_col]) else ''
    
    # Extract prefix: all letters until first numeric digit
    prefix_match = re.match(r'^([A-Za-z]+)', unit_number)
    prefix = prefix_match.group(1).upper() if prefix_match else ''
    
    if len(prefix) == 2:
        return prefix
    
    # Try to get two-letter state code after comma in unit name
    match = re.search(r',\s*([A-Z]{2})\b', unit_name)
    if match:
        return match.group(1)
    
    # Try to get city from unit_name (first word before comma, space, or end)
    city_match = re.match(r'^([A-Za-z .\-]+?)(?=,| |\Z)', unit_name)
    if city_match:
        city = city_match.group(1).strip()
        city_norm = normalize_city_name(city)
        if city_norm in US_CITY_ST_DICT_NORM:
            return US_CITY_ST_DICT_NORM[city_norm]
    
    # If not found, return full prefix as extracted (not truncated)
    return prefix


def is_valid_state(state: str) -> bool:
    """Check if state is a valid US state abbreviation."""
    if not state or not isinstance(state, str):
        return False
    state = state.strip().upper()
    return (
        len(state) == 2 and 
        state.isalpha() and 
        state != "CC" and
        state in STATE_ABBREV
    )


def extract_X_coverage(coverage_str) -> str:
    """
    Extract standard coverage label from X coverage string.
    Examples:
        '10 -Workers Compensation - X' → 'Workers Compensation'
        '30 -Auto Liability - X' → 'Auto Liability'
    """
    if pd.isna(coverage_str):
        return ""
    
    s = str(coverage_str).strip()
    
    # Pattern: num - Coverage - X (or X) [optional 'Excess']
    match = re.match(r"^\d+\s*-\s*([^-]+)\s*-\s*X(?:\s*(Excess))?$", s, re.IGNORECASE)
    if match:
        cov = match.group(1).strip()
        excess = match.group(2)
        if excess:
            cov += " Excess"
        return cov
    
    # Handle other variant, e.g. missing numeric prefix
    s = re.sub(r"^\d+\s*-\s*", "", s)
    excess_match = re.search(r"(.*?)\s*-\s*X(?:\s*(Excess))?$", s, re.IGNORECASE)
    if excess_match:
        cov = excess_match.group(1).strip()
        excess = excess_match.group(2)
        if excess:
            cov += " Excess"
        return cov
    
    # Fallback: just remove trailing '- X...' text
    s = re.sub(r"-\s*X.*$", "", s, flags=re.IGNORECASE).strip()
    return s


def get_coverage_code(coverage_str) -> str:
    """Get the short coverage code from coverage string."""
    coverage_label = extract_X_coverage(coverage_str)
    return COVERAGE_MAP.get(coverage_label, "")


def get_base_claim(claim_no) -> str:
    """
    Extract the BASE claim number by stripping trailing suffixes.
    
    Rules:
    1. If ends with letters + digits (e.g., ...GB01), strip trailing digits → ...GB
    2. Else if ends with single letter (e.g., ...49A), strip trailing letter → ...49
    3. If dash present, take portion before dash
    """
    claim_core = str(claim_no).strip()
    
    # Step 1: Remove trailing digits if immediately following uppercase letters
    m = re.match(r'^(.*?[A-Z]+)[0-9]+$', claim_core, re.IGNORECASE)
    if m:
        claim_core = m.group(1)
    # Step 2: Remove single trailing letter (e.g., ...49A -> ...49)
    elif re.match(r'.*[A-Z]$', claim_core, re.IGNORECASE):
        claim_core = re.sub(r'([A-Z])$', '', claim_core, flags=re.IGNORECASE)
    
    # Step 3: If dash present, take portion before dash
    if '-' in claim_core:
        claim_core = claim_core.split('-')[0].strip()
    
    return claim_core


def base_claim_no(x) -> str:
    """
    Remove trailing single alphabetical character from claim number.
    E.g., 'KY17K2349072B' -> 'KY17K2349072'
    """
    return re.sub(r'[A-Z]$', '', str(x).strip())
