"""
X-TPA Claim Linkage Main Script

This script matches X claims with TPA claims using BM25-based similarity scoring.
Replaces GPT-based matching with deterministic BM25 + fuzzy string matching.

Usage:
    python x_tpa_linkage.py

Workflow:
    1. Load TPA and X claims data
    2. For each TPA claim, find candidate X claims (filtered by date, state, coverage)
    3. Use BM25Matcher to compute similarity scores
    4. Apply match decision rules based on thresholds
    5. Two-pass matching: first with state filter, second without for remaining claims
    6. Output matched and non-matched results with confidence scores
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import (
    DATE_TOLERANCE_DAYS, COVERAGE_MAP, 
    TPA_CLAIMS_PATH, X_CLAIMS_PATH, OUTPUT_DIR
)
from utils.text_processing import normalize_text, get_base_claim, base_claim_no
from utils.date_utils import parse_date, date_diff_days, get_candidates_within_days
from data.data_loader import load_tpa_claims, load_X_claims, filter_candidates
from similarity.bm25_scorer import BM25Matcher


def assign_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign post-processing confidence levels based on component scores.
    
    Confidence Rules:
    - Low: name_sim < 0.5
    - Medium: name_sim 0.5-0.54 and desc_sim < 0.1
    - High: name_sim >= 0.54 and desc_sim >= 0.19
    - High (desc low): name_sim >= 0.65 and desc_sim 0.04-0.19
    """
    df = df.copy()
    
    # Ensure numeric types
    for col in ['name_string_sim', 'description_string_sim', 'gpt_confidence']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    conditions = [
        (df['name_string_sim'] < 0.5),
        ((df['name_string_sim'] >= 0.5) & (df['name_string_sim'] < 0.54) & (df['description_string_sim'] < 0.1)),
        ((df['name_string_sim'] >= 0.54) & (df['description_string_sim'] >= 0.19)),
        ((df['name_string_sim'] >= 0.65) & (df['description_string_sim'] >= 0.04) & (df['description_string_sim'] < 0.19)),
        ((df['name_string_sim'] >= 0.5) & (df['name_string_sim'] < 0.54)),
    ]
    choices = ["Low (Name is low match)", "Medium", "High", "High (Description is low match)", "Low"]
    
    df['post_process_confidence'] = np.select(conditions, choices, default='Low')
    
    return df


def match_x_tpa_claims(tpa_df: pd.DataFrame, X_df: pd.DataFrame, 
                       date_tolerance: int = 0,
                       verbose: bool = True) -> tuple:
    """
    Main matching function for X-TPA claim linkage.
    
    Args:
        tpa_df: TPA claims DataFrame
        X_df: X claims DataFrame
        date_tolerance: Days tolerance for date matching
        verbose: Print progress information
        
    Returns:
        Tuple of (matched_df, non_matched_df)
    """
    # Initialize matcher
    matcher = BM25Matcher(mode='x_tpa')
    
    results_true = []
    results_false = []
    X_candidates_used = set()
    all_X_clm_nos = set(X_df['X_clm_no'].unique())
    
    if verbose:
        print(f"TPA claims to process: {len(tpa_df)}")
        print(f"X claims available: {len(X_df)}")
    
    # =========================================================================
    # PASS 1: Match with all filters including state
    # =========================================================================
    if verbose:
        print("\n=== PASS 1: Matching with state filter ===")
    
    for idx, tpa_row in tpa_df.iterrows():
        tpa_date = parse_date(tpa_row['tpa_dt_loss'])
        tpa_state = tpa_row.get('tpa_state', '')
        tpa_name = normalize_text(tpa_row.get('tpa_clmnt_nm', ''))
        tpa_desc = normalize_text(tpa_row.get('tpa_event_desc', ''))
        tpa_line_cd = normalize_text(tpa_row.get('tpa_line_cd', ''))
        
        # Filter X candidates
        X_filtered = filter_candidates(
            X_df, tpa_row,
            date_tolerance=date_tolerance,
            match_state=True,
            match_coverage=True
        )
        
        if verbose and len(X_filtered) > 0:
            print(f"[Pass1] TPA#{tpa_row['tpa_clm_no']}: {len(X_filtered)} X candidates")
        
        for _, X_row in X_filtered.iterrows():
            X_candidates_used.add(X_row['X_clm_no'])
            
            X_name = normalize_text(X_row.get('X_clmnt_nm', ''))
            X_desc = normalize_text(X_row.get('X_event_desc', ''))
            
            # Build records for matching
            record_a = {
                'tpa_file_number': tpa_row['tpa_clm_no'],
                'tpa_state': tpa_state,
                'tpa_claimant_name': tpa_name,
                'tpa_description': tpa_desc
            }
            record_b = {
                'X_file_number': X_row['X_clm_no'],
                'X_state': X_row.get('X_evt_state', ''),
                'X_claimant_name': X_name,
                'X_description': X_desc
            }
            
            # Match using BM25
            decision = matcher.match_records(record_a, record_b)
            
            output_row = {
                'tpa_clm_no': tpa_row['tpa_clm_no'],
                'tpa_date': tpa_date,
                'tpa_state': tpa_state,
                'tpa_name': tpa_name,
                'tpa_desc': tpa_desc,
                'X_clm_no': X_row['X_clm_no'],
                'X_date': parse_date(X_row.get('X_dt_loss')),
                'X_state': X_row.get('X_evt_state', ''),
                'X_name': X_name,
                'X_desc': X_desc,
                'gpt_confidence': decision.get('confidence', 0.0),
                'gpt_reason': decision.get('reasons', ''),
                'gpt_component_scores': decision.get('component_scores', {}),
                'suggested_link': decision.get('suggested_link', False),
                'gpt_is_match': decision.get('is_match', False),
                'second_pass': False,
                # Flatten component scores for easier processing
                'name_string_sim': decision.get('component_scores', {}).get('name_string_sim', 0),
                'description_string_sim': decision.get('component_scores', {}).get('description_string_sim', 0),
                'state_string_sim': decision.get('component_scores', {}).get('state_string_sim'),
            }
            
            if decision.get('is_match', False):
                results_true.append(output_row)
            else:
                results_false.append(output_row)
    
    # =========================================================================
    # PASS 2: Match remaining X claims without state filter
    # =========================================================================
    missing_X_clm_nos = all_X_clm_nos - X_candidates_used
    
    if verbose:
        print(f"\n=== PASS 2: {len(missing_X_clm_nos)} X claims not matched in first pass ===")
    
    if missing_X_clm_nos:
        missing_X_df = X_df[X_df['X_clm_no'].isin(missing_X_clm_nos)]
        
        for idx, tpa_row in tpa_df.iterrows():
            tpa_date = parse_date(tpa_row['tpa_dt_loss'])
            tpa_name = normalize_text(tpa_row.get('tpa_clmnt_nm', ''))
            tpa_desc = normalize_text(tpa_row.get('tpa_event_desc', ''))
            tpa_line_cd = normalize_text(tpa_row.get('tpa_line_cd', ''))
            
            # Filter without state
            X_missing_filtered = filter_candidates(
                missing_X_df, tpa_row,
                date_tolerance=date_tolerance,
                match_state=False,  # No state filter
                match_coverage=True
            )
            
            if verbose and len(X_missing_filtered) > 0:
                print(f"[Pass2] TPA#{tpa_row['tpa_clm_no']}: {len(X_missing_filtered)} X candidates")
            
            for _, X_row in X_missing_filtered.iterrows():
                X_name = normalize_text(X_row.get('X_clmnt_nm', ''))
                X_desc = normalize_text(X_row.get('X_event_desc', ''))
                
                record_a = {
                    'tpa_file_number': tpa_row['tpa_clm_no'],
                    'tpa_state': tpa_row.get('tpa_state', ''),
                    'tpa_claimant_name': tpa_name,
                    'tpa_description': tpa_desc
                }
                record_b = {
                    'X_file_number': X_row['X_clm_no'],
                    'X_state': X_row.get('X_evt_state', ''),
                    'X_claimant_name': X_name,
                    'X_description': X_desc
                }
                
                decision = matcher.match_records(record_a, record_b)
                
                output_row = {
                    'tpa_clm_no': tpa_row['tpa_clm_no'],
                    'tpa_date': tpa_date,
                    'tpa_state': tpa_row.get('tpa_state', ''),
                    'tpa_name': tpa_name,
                    'tpa_desc': tpa_desc,
                    'X_clm_no': X_row['X_clm_no'],
                    'X_date': parse_date(X_row.get('X_dt_loss')),
                    'X_state': X_row.get('X_evt_state', ''),
                    'X_name': X_name,
                    'X_desc': X_desc,
                    'gpt_confidence': decision.get('confidence', 0.0),
                    'gpt_reason': decision.get('reasons', ''),
                    'gpt_component_scores': decision.get('component_scores', {}),
                    'suggested_link': decision.get('suggested_link', False),
                    'gpt_is_match': decision.get('is_match', False),
                    'second_pass': True,
                    'name_string_sim': decision.get('component_scores', {}).get('name_string_sim', 0),
                    'description_string_sim': decision.get('component_scores', {}).get('description_string_sim', 0),
                    'state_string_sim': decision.get('component_scores', {}).get('state_string_sim'),
                }
                
                if decision.get('is_match', False):
                    results_true.append(output_row)
                else:
                    results_false.append(output_row)
    
    # =========================================================================
    # Build result DataFrames
    # =========================================================================
    matched_df = pd.DataFrame(results_true) if results_true else pd.DataFrame()
    non_matched_df = pd.DataFrame(results_false) if results_false else pd.DataFrame()
    
    if verbose:
        print(f"\n=== RESULTS ===")
        print(f"Total matches: {len(matched_df)}")
        print(f"Total non-matches reviewed: {len(non_matched_df)}")
        if not matched_df.empty and 'second_pass' in matched_df.columns:
            print(f"Second-pass matches: {matched_df['second_pass'].sum()}")
    
    return matched_df, non_matched_df


def main():
    """Main execution function."""
    print("=" * 60)
    print("X-TPA Claim Linkage - BM25 Based Matching")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    
    # =========================================================================
    # Load Data
    # =========================================================================
    print("\n--- Loading Data ---")
    
    # For local testing, you can override these paths
    tpa_path = TPA_CLAIMS_PATH
    x_path = X_CLAIMS_PATH
    
    # Check if local test files exist
    local_tpa = "data/tpa_claims.xlsx"
    local_x = "data/x_claims.xlsx"
    if os.path.exists(local_tpa):
        tpa_path = local_tpa
    if os.path.exists(local_x):
        x_path = local_x
    
    try:
        tpa_df = load_tpa_claims(tpa_path)
        print(f"Loaded {len(tpa_df)} TPA claims")
    except Exception as e:
        print(f"Error loading TPA claims: {e}")
        print("Please set TPA_CLAIMS_PATH in config/settings.py or provide data/tpa_claims.xlsx")
        return
    
    try:
        X_df = load_X_claims(x_path)
        print(f"Loaded {len(X_df)} X claims")
    except Exception as e:
        print(f"Error loading X claims: {e}")
        print("Please set X_CLAIMS_PATH in config/settings.py or provide data/x_claims.xlsx")
        return
    
    # =========================================================================
    # Run Matching
    # =========================================================================
    print("\n--- Running Matching ---")
    
    matched_df, non_matched_df = match_x_tpa_claims(
        tpa_df, X_df,
        date_tolerance=DATE_TOLERANCE_DAYS,
        verbose=True
    )
    
    # =========================================================================
    # Post-Processing
    # =========================================================================
    if not matched_df.empty:
        print("\n--- Post-Processing ---")
        matched_df = assign_confidence(matched_df)
    
    if not non_matched_df.empty:
        non_matched_df = assign_confidence(non_matched_df)
    
    # =========================================================================
    # Save Results
    # =========================================================================
    print("\n--- Saving Results ---")
    
    output_dir = OUTPUT_DIR if os.path.exists(os.path.dirname(OUTPUT_DIR)) else "output/"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not matched_df.empty:
        matched_path = os.path.join(output_dir, f"matched_x_tpa_{timestamp}.parquet")
        matched_df.to_parquet(matched_path, index=False)
        print(f"Saved matched results to: {matched_path}")
        
        # Also save as Excel for easy viewing
        matched_xlsx = os.path.join(output_dir, f"matched_x_tpa_{timestamp}.xlsx")
        matched_df.to_excel(matched_xlsx, index=False)
        print(f"Saved matched results to: {matched_xlsx}")
    
    if not non_matched_df.empty:
        non_matched_path = os.path.join(output_dir, f"non_matched_x_tpa_{timestamp}.parquet")
        non_matched_df.to_parquet(non_matched_path, index=False)
        print(f"Saved non-matched results to: {non_matched_path}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"TPA claims processed: {len(tpa_df)}")
    print(f"X claims processed: {len(X_df)}")
    print(f"Matches found: {len(matched_df)}")
    print(f"Non-matches reviewed: {len(non_matched_df)}")
    
    if not matched_df.empty and 'post_process_confidence' in matched_df.columns:
        print("\nMatch Confidence Distribution:")
        print(matched_df['post_process_confidence'].value_counts())
    
    print(f"\nCompleted at: {datetime.now()}")


if __name__ == "__main__":
    main()
