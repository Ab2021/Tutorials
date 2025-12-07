"""
TPA Claim Clustering Main Script

This script clusters TPA claims that refer to the same underlying event.
Uses BM25-based similarity scoring to find related claims within the same date/state.

Usage:
    python tpa_clustering.py

Workflow:
    1. Load TPA claims data
    2. Partition by date (for efficient processing)
    3. For each pair of claims with matching date/state/line, compute similarity
    4. Auto-match claims with same base claim number
    5. Use BM25Matcher for remaining pairs
    6. Build connected components using NetworkX
    7. Assign group IDs to clustered claims
    8. Output clustered results
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import DATE_TOLERANCE_DAYS, TPA_CLAIMS_PATH, OUTPUT_DIR
from utils.text_processing import normalize_text, get_base_claim
from utils.date_utils import parse_date, date_diff_days, get_candidates_within_days
from data.data_loader import load_tpa_claims, is_valid_tpa_pair, is_duplicate_pair
from similarity.bm25_scorer import BM25Matcher

# Optional: NetworkX for connected components
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not installed. Using basic clustering.")


def assign_clustering_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign post-processing confidence levels for TPA clustering.
    
    Confidence Rules for TPA Clustering:
    - High: description_sim >= 0.6 and confidence >= 0.32 and name_sim <= 0.20
    - High: description_sim >= 0.45 and name_sim >= 0.75
    - High: description_sim >= 0.75 and confidence >= 0.20 and name_sim <= 0.10
    - High: base claims match exactly
    - Low: otherwise
    """
    df = df.copy()
    
    # Ensure numeric types
    for col in ['name_string_sim', 'description_string_sim', 'gpt_confidence']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Add base claim columns if not present
    if 'base_clm_A' not in df.columns and 'tpa_clm_no_A' in df.columns:
        df['base_clm_A'] = df['tpa_clm_no_A'].apply(get_base_claim)
    if 'base_clm_B' not in df.columns and 'tpa_clm_no_B' in df.columns:
        df['base_clm_B'] = df['tpa_clm_no_B'].apply(get_base_claim)
    
    conditions = [
        (
            ((df['description_string_sim'] >= 0.6) & (df['gpt_confidence'] >= 0.32) & (df['name_string_sim'] <= 0.20)) |
            ((df['description_string_sim'] >= 0.45) & (df['name_string_sim'] >= 0.75)) |
            ((df['description_string_sim'] >= 0.75) & (df['gpt_confidence'] >= 0.20) & (df['name_string_sim'] <= 0.10)) |
            (df['base_clm_A'] == df['base_clm_B'])
        ),
    ]
    choices = ["High"]
    
    df['post_process_confidence'] = np.select(conditions, choices, default='Low')
    
    return df


def process_date_partition(partition_df: pd.DataFrame, 
                          date_tolerance: int = 0) -> pd.DataFrame:
    """
    Process a single date partition for TPA clustering.
    
    Args:
        partition_df: DataFrame containing claims for this partition
        date_tolerance: Days tolerance for date matching
        
    Returns:
        DataFrame with matching results
    """
    matcher = BM25Matcher(mode='tpa_cluster')
    
    results = []
    matched_pairs = set()
    
    # Sort for consistent processing
    partition_df = partition_df.sort_values(by=['tpa_dt_loss', 'tpa_clm_no']).reset_index(drop=True)
    
    for idx_a, row_a in partition_df.iterrows():
        claim_no_A = row_a['tpa_clm_no']
        tpa_a_date = row_a['tpa_dt_loss']
        tpa_a_state = row_a.get('tpa_state', '')
        tpa_a_name = normalize_text(row_a.get('tpa_clmnt_nm', ''))
        tpa_a_desc = normalize_text(row_a.get('tpa_event_desc', ''))
        tpa_a_line = normalize_text(row_a.get('tpa_line_cd', ''))
        
        # Get candidates within date tolerance (only look at "later" claims)
        candidates = get_candidates_within_days(
            partition_df.iloc[idx_a + 1:],
            base_date=tpa_a_date,
            days=date_tolerance,
            date_col='tpa_dt_loss'
        )
        
        if candidates.empty:
            continue
        
        # Mark self-pair as seen
        matched_pairs.add(tuple(sorted([claim_no_A, claim_no_A])))
        
        for idx_b, row_b in candidates.iterrows():
            claim_no_B = row_b['tpa_clm_no']
            
            # Check if valid pair (state and line code match)
            if not is_valid_tpa_pair(row_a, row_b, date_tolerance):
                continue
            
            # Check for duplicate pair
            if is_duplicate_pair(claim_no_A, claim_no_B, matched_pairs):
                continue
            
            base_A = get_base_claim(claim_no_A)
            base_B = get_base_claim(claim_no_B)
            
            # Auto-match if base claims are the same
            if base_A == base_B and claim_no_A != claim_no_B:
                output_row = {
                    'tpa_clm_no_A': claim_no_A,
                    'tpa_date_A': tpa_a_date,
                    'tpa_state_A': tpa_a_state,
                    'tpa_name_A': tpa_a_name,
                    'tpa_desc_A': tpa_a_desc,
                    'tpa_clm_no_B': claim_no_B,
                    'tpa_date_B': parse_date(row_b['tpa_dt_loss']),
                    'tpa_state_B': row_b.get('tpa_state', ''),
                    'tpa_name_B': normalize_text(row_b.get('tpa_clmnt_nm', '')),
                    'tpa_desc_B': normalize_text(row_b.get('tpa_event_desc', '')),
                    'gpt_confidence': 1.0,
                    'gpt_reason': 'Exact Base Match',
                    'gpt_component_scores': json.dumps({'state_string_sim': 1.0}),
                    'suggested_link': True,
                    'gpt_is_match': True,
                    'is_match': True,
                    'name_string_sim': 1.0,
                    'description_string_sim': 1.0,
                    'state_string_sim': 1.0,
                    'base_clm_A': base_A,
                    'base_clm_B': base_B,
                }
                results.append(output_row)
                matched_pairs.add(tuple(sorted([claim_no_A, claim_no_B])))
                continue
            
            # Build records for BM25 matching
            record_a = {
                'tpa_file_number': claim_no_A,
                'tpa_state': tpa_a_state,
                'tpa_claimant_name': tpa_a_name,
                'tpa_description': tpa_a_desc
            }
            record_b = {
                'tpa_file_number': claim_no_B,
                'tpa_state': row_b.get('tpa_state', ''),
                'tpa_claimant_name': normalize_text(row_b.get('tpa_clmnt_nm', '')),
                'tpa_description': normalize_text(row_b.get('tpa_event_desc', ''))
            }
            
            # Match using BM25
            decision = matcher.match_records(record_a, record_b)
            
            output_row = {
                'tpa_clm_no_A': claim_no_A,
                'tpa_date_A': tpa_a_date,
                'tpa_state_A': tpa_a_state,
                'tpa_name_A': tpa_a_name,
                'tpa_desc_A': tpa_a_desc,
                'tpa_clm_no_B': claim_no_B,
                'tpa_date_B': parse_date(row_b['tpa_dt_loss']),
                'tpa_state_B': row_b.get('tpa_state', ''),
                'tpa_name_B': normalize_text(row_b.get('tpa_clmnt_nm', '')),
                'tpa_desc_B': normalize_text(row_b.get('tpa_event_desc', '')),
                'gpt_confidence': float(decision.get('confidence', 0.0)),
                'gpt_reason': str(decision.get('reasons', '')),
                'gpt_component_scores': json.dumps(decision.get('component_scores', {})),
                'suggested_link': bool(decision.get('suggested_link', False)),
                'gpt_is_match': bool(decision.get('is_match', False)),
                'is_match': bool(decision.get('is_match', False)),
                'name_string_sim': decision.get('component_scores', {}).get('name_string_sim', 0),
                'description_string_sim': decision.get('component_scores', {}).get('description_string_sim', 0),
                'state_string_sim': decision.get('component_scores', {}).get('state_string_sim'),
                'base_clm_A': base_A,
                'base_clm_B': base_B,
            }
            
            matched_pairs.add(tuple(sorted([claim_no_A, claim_no_B])))
            results.append(output_row)
    
    return pd.DataFrame(results) if results else pd.DataFrame()


def cluster_tpa_claims(tpa_df: pd.DataFrame, 
                       date_tolerance: int = 0,
                       partition_by: str = 'year',
                       verbose: bool = True) -> tuple:
    """
    Main clustering function for TPA claims.
    
    Args:
        tpa_df: TPA claims DataFrame
        date_tolerance: Days tolerance for date matching
        partition_by: 'year', 'month', or 'day' for partitioning
        verbose: Print progress information
        
    Returns:
        Tuple of (matched_df, non_matched_df, all_results_df)
    """
    if verbose:
        print(f"TPA claims to cluster: {len(tpa_df)}")
    
    # Ensure date column is datetime
    tpa_df = tpa_df.copy()
    tpa_df['tpa_dt_loss'] = pd.to_datetime(tpa_df['tpa_dt_loss'], errors='coerce')
    
    # Create partition column
    if partition_by == 'year':
        tpa_df['partition_key'] = tpa_df['tpa_dt_loss'].dt.year
    elif partition_by == 'month':
        tpa_df['partition_key'] = tpa_df['tpa_dt_loss'].dt.to_period('M').astype(str)
    else:  # day
        tpa_df['partition_key'] = tpa_df['tpa_dt_loss'].dt.date
    
    # Process each partition
    all_results = []
    partitions = tpa_df.groupby('partition_key')
    
    if verbose:
        print(f"Processing {len(partitions)} partitions ({partition_by})")
    
    for partition_key, partition_df in partitions:
        if verbose:
            print(f"  Processing partition {partition_key}: {len(partition_df)} claims")
        
        partition_results = process_date_partition(partition_df, date_tolerance)
        
        if not partition_results.empty:
            all_results.append(partition_results)
    
    # Combine all results
    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
    else:
        results_df = pd.DataFrame()
    
    if verbose:
        print(f"\nTotal pairs compared: {len(results_df)}")
    
    # Split into matched and non-matched
    if not results_df.empty and 'is_match' in results_df.columns:
        matched_df = results_df[results_df['is_match']].copy()
        non_matched_df = results_df[~results_df['is_match']].copy()
    else:
        matched_df = pd.DataFrame()
        non_matched_df = results_df.copy()
    
    if verbose:
        print(f"Matches found: {len(matched_df)}")
        print(f"Non-matches: {len(non_matched_df)}")
    
    return matched_df, non_matched_df, results_df


def build_claim_groups(matched_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build connected components from matched claim pairs.
    
    Args:
        matched_df: DataFrame with matched pairs (tpa_clm_no_A, tpa_clm_no_B)
        
    Returns:
        DataFrame mapping each claim to its group_id
    """
    if matched_df.empty:
        return pd.DataFrame(columns=['tpa_clm_no', 'group_id'])
    
    if HAS_NETWORKX:
        # Build graph
        G = nx.Graph()
        edges = list(zip(matched_df['tpa_clm_no_A'], matched_df['tpa_clm_no_B']))
        G.add_edges_from(edges)
        
        # Find connected components
        component_map = {}
        for idx, component in enumerate(nx.connected_components(G), 1):
            for claim_no in component:
                component_map[claim_no] = idx
        
        # Create result DataFrame
        unique_claims = pd.unique(matched_df[['tpa_clm_no_A', 'tpa_clm_no_B']].values.ravel())
        group_df = pd.DataFrame({'tpa_clm_no': unique_claims})
        group_df['group_id'] = group_df['tpa_clm_no'].map(component_map)
        
    else:
        # Simple Union-Find implementation
        all_claims = pd.unique(matched_df[['tpa_clm_no_A', 'tpa_clm_no_B']].values.ravel())
        claim2id = {claim: idx for idx, claim in enumerate(all_claims)}
        id2claim = {idx: claim for claim, idx in claim2id.items()}
        parent = list(range(len(all_claims)))
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[py] = px
        
        # Union all pairs
        for _, row in matched_df.iterrows():
            a, b = claim2id[row['tpa_clm_no_A']], claim2id[row['tpa_clm_no_B']]
            union(a, b)
        
        # Assign group IDs
        groups = [find(i) for i in range(len(all_claims))]
        _, group_labels = np.unique(groups, return_inverse=True)
        claim_to_group = {id2claim[i]: g + 1 for i, g in enumerate(group_labels)}
        
        group_df = pd.DataFrame({
            'tpa_clm_no': list(claim_to_group.keys()),
            'group_id': list(claim_to_group.values())
        })
    
    return group_df


def main():
    """Main execution function."""
    print("=" * 60)
    print("TPA Claim Clustering - BM25 Based Matching")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    
    # =========================================================================
    # Load Data
    # =========================================================================
    print("\n--- Loading Data ---")
    
    tpa_path = TPA_CLAIMS_PATH
    
    # Check for local test file
    local_tpa = "data/tpa_claims.xlsx"
    if os.path.exists(local_tpa):
        tpa_path = local_tpa
    
    try:
        tpa_df = load_tpa_claims(tpa_path, filter_incurred=True)
        print(f"Loaded {len(tpa_df)} TPA claims")
    except Exception as e:
        print(f"Error loading TPA claims: {e}")
        print("Please set TPA_CLAIMS_PATH in config/settings.py or provide data/tpa_claims.xlsx")
        return
    
    # =========================================================================
    # Run Clustering
    # =========================================================================
    print("\n--- Running Clustering ---")
    
    matched_df, non_matched_df, all_results_df = cluster_tpa_claims(
        tpa_df,
        date_tolerance=DATE_TOLERANCE_DAYS,
        partition_by='year',
        verbose=True
    )
    
    # =========================================================================
    # Post-Processing
    # =========================================================================
    print("\n--- Post-Processing ---")
    
    if not all_results_df.empty:
        all_results_df = assign_clustering_confidence(all_results_df)
        
        # Filter to high confidence matches for grouping
        high_conf_matches = all_results_df[all_results_df['post_process_confidence'] == 'High']
        print(f"High confidence matches: {len(high_conf_matches)}")
    else:
        high_conf_matches = pd.DataFrame()
    
    # Build claim groups
    if not high_conf_matches.empty:
        group_df = build_claim_groups(high_conf_matches)
        print(f"Built {group_df['group_id'].nunique()} groups from {len(group_df)} claims")
    else:
        group_df = pd.DataFrame(columns=['tpa_clm_no', 'group_id'])
    
    # =========================================================================
    # Create Master DataFrame
    # =========================================================================
    print("\n--- Creating Master DataFrame ---")
    
    # Merge group info back to TPA claims
    master_df = tpa_df.merge(group_df, on='tpa_clm_no', how='left')
    
    # Add base claim column
    master_df['base_clm'] = master_df['tpa_clm_no'].apply(get_base_claim)
    
    # Handle claims with same base_clm but different states
    current_max_gid = master_df['group_id'].dropna().max()
    if pd.isna(current_max_gid):
        current_max_gid = 0
    next_gid = int(current_max_gid) + 1
    
    group_id_updates = {}
    for base, subdf in master_df.groupby('base_clm'):
        state_count = subdf['tpa_state'].nunique()
        if state_count < 2:
            continue
        
        sub_gid_set = set(subdf['group_id'].dropna())
        null_index = subdf['group_id'].isnull()
        idxs_no_gid = subdf[null_index].index
        
        if len(sub_gid_set) == 0:
            group_id_updates.update({idx: next_gid for idx in subdf.index})
            next_gid += 1
        else:
            chosen_gid = min(sub_gid_set)
            group_id_updates.update({idx: chosen_gid for idx in idxs_no_gid})
    
    for idx, gid in group_id_updates.items():
        master_df.at[idx, 'group_id'] = gid
    
    master_df['group_id'] = master_df['group_id'].astype('Int64')
    
    # Filter and create final output
    df_with_group = master_df[master_df['group_id'].notnull()]
    if 'tpa_tot_incurred' in df_with_group.columns:
        group_totals = df_with_group.groupby('group_id')['tpa_tot_incurred'].sum()
        valid_group_ids = group_totals[group_totals != 0].index.tolist()
        valid_clm_with_group = df_with_group[df_with_group['group_id'].isin(valid_group_ids)]['tpa_clm_no']
        
        df_without_group = master_df[master_df['group_id'].isnull()]
        valid_clm_without_group = df_without_group[df_without_group['tpa_tot_incurred'] != 0]['tpa_clm_no']
        
        valid_tpa_clm_no = pd.concat([valid_clm_with_group, valid_clm_without_group]).unique()
        filtered_master_df = master_df[master_df['tpa_clm_no'].isin(valid_tpa_clm_no)]
    else:
        filtered_master_df = master_df
    
    # Sort results
    filtered_master_df = filtered_master_df.sort_values(
        by=['tpa_dt_loss', 'group_id', 'tpa_clm_no'],
        ascending=[True, True, True]
    ).reset_index(drop=True)
    
    # Add group base claim
    filtered_master_df['group_base_clm'] = np.nan
    for gid, group in filtered_master_df.groupby('group_id', dropna=False):
        if pd.notna(gid):
            first_base_clm = group['base_clm'].iloc[0]
            filtered_master_df.loc[group.index, 'group_base_clm'] = first_base_clm
    
    print(f"Final master DataFrame: {len(filtered_master_df)} claims")
    print(f"Claims in groups: {filtered_master_df['group_id'].notna().sum()}")
    print(f"Unique groups: {filtered_master_df['group_id'].dropna().nunique()}")
    
    # =========================================================================
    # Save Results
    # =========================================================================
    print("\n--- Saving Results ---")
    
    output_dir = OUTPUT_DIR if os.path.exists(os.path.dirname(OUTPUT_DIR)) else "output/"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save matching results
    if not matched_df.empty:
        matched_path = os.path.join(output_dir, f"tpa_matched_{timestamp}.parquet")
        matched_df.to_parquet(matched_path, index=False)
        print(f"Saved matched pairs to: {matched_path}")
    
    if not non_matched_df.empty:
        non_matched_path = os.path.join(output_dir, f"tpa_non_matched_{timestamp}.parquet")
        non_matched_df.to_parquet(non_matched_path, index=False)
        print(f"Saved non-matched pairs to: {non_matched_path}")
    
    # Save master DataFrame
    master_path = os.path.join(output_dir, f"tpa_clustered_master_{timestamp}.parquet")
    filtered_master_df.to_parquet(master_path, index=False)
    print(f"Saved master DataFrame to: {master_path}")
    
    # Also save as Excel
    master_xlsx = os.path.join(output_dir, f"tpa_clustered_master_{timestamp}.xlsx")
    filtered_master_df.to_excel(master_xlsx, index=False)
    print(f"Saved master DataFrame to: {master_xlsx}")
    
    # Save group mapping
    group_path = os.path.join(output_dir, f"tpa_groups_{timestamp}.parquet")
    group_df.to_parquet(group_path, index=False)
    print(f"Saved group mapping to: {group_path}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"TPA claims processed: {len(tpa_df)}")
    print(f"Total pairs compared: {len(all_results_df)}")
    print(f"Matches found: {len(matched_df)}")
    print(f"Unique groups created: {group_df['group_id'].nunique() if not group_df.empty else 0}")
    print(f"Claims in groups: {len(group_df)}")
    
    if not all_results_df.empty and 'post_process_confidence' in all_results_df.columns:
        print("\nConfidence Distribution:")
        print(all_results_df['post_process_confidence'].value_counts())
    
    print(f"\nCompleted at: {datetime.now()}")


if __name__ == "__main__":
    main()
