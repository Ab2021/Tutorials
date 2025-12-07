"""
Data loading and preprocessing utilities for claim data.
"""

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from config.settings import COVERAGE_MAP, STATE_ABBREV
from utils.text_processing import (
    normalize_text, extract_state, is_valid_state, 
    get_coverage_code, extract_X_coverage
)
from utils.date_utils import parse_date


def load_tpa_claims(filepath: str, filter_incurred: bool = False) -> pd.DataFrame:
    """
    Load and preprocess TPA claims data from Excel file.
    
    Args:
        filepath: Path to TPA claims Excel file
        filter_incurred: If True, filter to claims with positive incurred amounts
                        (Set to True for TPA clustering, False for X-TPA matching)
        
    Returns:
        Preprocessed TPA claims DataFrame
    """
    # Load data
    tpa_clm_df = pd.read_excel(filepath)
    
    # Filter by incurred amount if requested (used in TPA clustering, not X-TPA matching)
    if filter_incurred and 'Claim Total Incurred' in tpa_clm_df.columns:
        tpa_clm_df = tpa_clm_df[tpa_clm_df['Claim Total Incurred'] > 0]
    
    # Select and rename columns
    column_mapping = {
        'File Number appended with State claim number': 'tpa_clm_no',
        'Claimant Last Name': 'tpa_clmnt_lst_nm',
        'Claimant First Name': 'tpa_clmnt_fst_nm',
        'Date of Loss': 'tpa_dt_loss',
        'Unit Name': 'tpa_unit_nm',
        'Unit Number': 'tpa_unit_no',
        'Event Description': 'tpa_event_desc',
        'Line Code': 'tpa_line_cd',
        'Claim Total Incurred': 'tpa_tot_incurred',
        'Claim Total Paid': 'tpa_tot_paid',
        'Claim Incurred - Expense': 'tpa_expense_incurred',
        'Claim Paid - Expense/Other': 'tpa_expense_paid',
        'Coverage Code': 'tpa_cov_cd'
    }
    
    # Rename columns that exist
    rename_cols = {k: v for k, v in column_mapping.items() if k in tpa_clm_df.columns}
    tpa_clm_df = tpa_clm_df.rename(columns=rename_cols)
    
    # Build claimant name
    if 'tpa_clmnt_lst_nm' in tpa_clm_df.columns and 'tpa_clmnt_fst_nm' in tpa_clm_df.columns:
        tpa_clm_df['tpa_clmnt_lst_nm'] = tpa_clm_df['tpa_clmnt_lst_nm'].fillna('')
        tpa_clm_df['tpa_clmnt_fst_nm'] = tpa_clm_df['tpa_clmnt_fst_nm'].fillna('')
        
        tpa_clm_df['tpa_clmnt_nm'] = (
            tpa_clm_df['tpa_clmnt_lst_nm'].str.strip().str.replace(r'[,.]', '', regex=True) + 
            ' ' + 
            tpa_clm_df['tpa_clmnt_fst_nm'].str.strip().str.replace(r'[,.]', '', regex=True)
        )
        tpa_clm_df['tpa_clmnt_nm'] = tpa_clm_df['tpa_clmnt_nm'].str.strip().str.upper()
        
        # Drop original name columns
        tpa_clm_df = tpa_clm_df.drop(columns=['tpa_clmnt_lst_nm', 'tpa_clmnt_fst_nm'], errors='ignore')
    
    # Clean claim number
    if 'tpa_clm_no' in tpa_clm_df.columns:
        tpa_clm_df['tpa_clm_no'] = tpa_clm_df['tpa_clm_no'].astype(str).str.strip()
    
    # Extract state
    if 'tpa_unit_no' in tpa_clm_df.columns and 'tpa_unit_nm' in tpa_clm_df.columns:
        tpa_clm_df['tpa_state'] = tpa_clm_df.apply(
            extract_state, 
            base_state_col='tpa_unit_no', 
            second_state_col='tpa_unit_nm', 
            axis=1
        )
        tpa_clm_df['tpa_state'] = tpa_clm_df['tpa_state'].astype(str)
        
        # Validate states
        valid_mask = tpa_clm_df['tpa_state'].apply(is_valid_state)
        tpa_clm_df.loc[~valid_mask, 'tpa_state'] = 'UNKNOWN'
    
    # Filter out non-USA claims
    if 'tpa_unit_no' in tpa_clm_df.columns:
        tpa_clm_df = tpa_clm_df[~tpa_clm_df['tpa_unit_no'].isin(['NONUSA'])]
    
    # Parse date column
    if 'tpa_dt_loss' in tpa_clm_df.columns:
        tpa_clm_df['tpa_dt_loss'] = pd.to_datetime(tpa_clm_df['tpa_dt_loss'], errors='coerce')
    
    # Drop duplicates
    tpa_clm_df = tpa_clm_df.drop_duplicates()
    
    return tpa_clm_df


def load_X_claims(filepath: str, sheet_name: str = "CLAIM DETAIL LOSS RUN",
                  filter_incurred: bool = True, 
                  filter_prefixes: tuple = ('KY', 'JY')) -> pd.DataFrame:
    """
    Load and preprocess X claims data from Excel file.
    
    Args:
        filepath: Path to X claims Excel file
        sheet_name: Name of sheet to read
        filter_incurred: If True, filter to claims with positive net incurred
        filter_prefixes: Tuple of claim number prefixes to include
        
    Returns:
        Preprocessed X claims DataFrame
    """
    # Load data
    X_clm_df = pd.read_excel(filepath, sheet_name=sheet_name)
    
    # Select and rename columns
    column_mapping = {
        'Claim Number': 'X_clm_no',
        'Claimant': 'X_clmnt_nm',
        'Event State': 'X_evt_state',
        'Accident Description': 'X_event_desc',
        'Date - Event': 'X_dt_loss',
        'Coverage': 'X_coverage',
        'Net Incurred': 'X_net_incurred',
        'Net Paid': 'X_net_paid',
        'Incurred Expense': 'X_expense_incurred',
        'Paid Expense': 'X_expense_paid'
    }
    
    # Rename columns that exist
    rename_cols = {k: v for k, v in column_mapping.items() if k in X_clm_df.columns}
    X_clm_df = X_clm_df.rename(columns=rename_cols)
    
    # Clean claimant name
    if 'X_clmnt_nm' in X_clm_df.columns:
        X_clm_df['X_clmnt_nm'] = X_clm_df['X_clmnt_nm'].fillna('')
        X_clm_df['X_clmnt_nm'] = X_clm_df['X_clmnt_nm'].str.strip().str.upper().str.replace(r'[,.]', '', regex=True)
    
    # Filter by claim number prefix
    if filter_prefixes and 'X_clm_no' in X_clm_df.columns:
        X_clm_df['X_clm_no'] = X_clm_df['X_clm_no'].astype(str)
        
        prefix_masks = []
        for prefix in filter_prefixes:
            prefix_masks.append(X_clm_df['X_clm_no'].str.startswith(prefix))
        
        combined_mask = prefix_masks[0]
        for mask in prefix_masks[1:]:
            combined_mask = combined_mask | mask
        
        # Additional JY filter: must have valid state
        if 'JY' in filter_prefixes and 'X_evt_state' in X_clm_df.columns:
            jy_mask = X_clm_df['X_clm_no'].str.startswith('JY')
            valid_state_mask = (
                X_clm_df['X_evt_state'].str.len().eq(2) &
                X_clm_df['X_evt_state'].str.isalpha()
            )
            # JY claims need valid state
            jy_valid = jy_mask & valid_state_mask
            # Non-JY claims from filter_prefixes
            non_jy = combined_mask & ~jy_mask
            combined_mask = jy_valid | non_jy
        
        X_clm_df = X_clm_df[combined_mask]
    
    # Filter by net incurred
    if filter_incurred and 'X_net_incurred' in X_clm_df.columns:
        X_clm_df = X_clm_df[X_clm_df['X_net_incurred'] > 0]
    
    # Parse date column
    if 'X_dt_loss' in X_clm_df.columns:
        X_clm_df['X_dt_loss'] = pd.to_datetime(X_clm_df['X_dt_loss'], errors='coerce')
    
    # Add coverage code
    if 'X_coverage' in X_clm_df.columns:
        X_clm_df['X_coverage_code'] = X_clm_df['X_coverage'].apply(get_coverage_code)
    
    # Drop duplicates
    X_clm_df = X_clm_df.drop_duplicates()
    
    return X_clm_df


def filter_candidates(X_df: pd.DataFrame, tpa_row: pd.Series, 
                     date_tolerance: int = 0,
                     match_state: bool = True,
                     match_coverage: bool = True) -> pd.DataFrame:
    """
    Filter X claims to candidates for matching against a TPA claim.
    
    Args:
        X_df: X claims DataFrame
        tpa_row: Single TPA claim row (Series)
        date_tolerance: Days tolerance for date matching
        match_state: Whether to filter by state match
        match_coverage: Whether to filter by coverage/line code match
        
    Returns:
        Filtered X claims DataFrame
    """
    filtered = X_df.copy()
    
    # Date filter
    if 'X_dt_loss' in filtered.columns and 'tpa_dt_loss' in tpa_row.index:
        tpa_date = parse_date(tpa_row['tpa_dt_loss'])
        if pd.notna(tpa_date):
            from utils.date_utils import date_diff_days
            filtered = filtered[
                filtered['X_dt_loss'].apply(
                    lambda d: date_diff_days(tpa_date, parse_date(d)) <= date_tolerance
                )
            ]
    
    # State filter
    if match_state and 'X_evt_state' in filtered.columns:
        tpa_state = tpa_row.get('tpa_state', '')
        if tpa_state and tpa_state != 'UNKNOWN':
            def state_matches(row_state):
                if not tpa_state or tpa_state.strip() == '' or tpa_state.strip() == 'UNKNOWN':
                    return True
                return str(row_state).upper() == tpa_state.upper()
            
            filtered = filtered[filtered['X_evt_state'].apply(state_matches)]
    
    # Coverage/Line code filter
    if match_coverage:
        if 'X_coverage_code' in filtered.columns and 'tpa_line_cd' in tpa_row.index:
            tpa_line_cd = normalize_text(tpa_row.get('tpa_line_cd', '')).strip()
            if tpa_line_cd:
                filtered = filtered[filtered['X_coverage_code'] == tpa_line_cd]
    
    return filtered


def is_valid_tpa_pair(row_a: pd.Series, row_b: pd.Series, 
                      date_tolerance: int = 0) -> bool:
    """
    Check if two TPA claims are valid candidates for comparison.
    
    Args:
        row_a: First TPA claim
        row_b: Second TPA claim
        date_tolerance: Days tolerance for date matching
        
    Returns:
        True if valid pair, False otherwise
    """
    from utils.date_utils import date_diff_days, parse_date
    
    # State must match
    state_a = row_a.get('tpa_state', '')
    state_b = row_b.get('tpa_state', '')
    if state_a != state_b:
        return False
    
    # Date must be within tolerance
    date_a = parse_date(row_a.get('tpa_dt_loss'))
    date_b = parse_date(row_b.get('tpa_dt_loss'))
    if date_diff_days(date_a, date_b) > date_tolerance:
        return False
    
    # Line code must match
    line_a = normalize_text(row_a.get('tpa_line_cd', ''))
    line_b = normalize_text(row_b.get('tpa_line_cd', ''))
    if line_a != line_b:
        return False
    
    return True


def is_duplicate_pair(claim_no_a: str, claim_no_b: str, 
                     matched_pairs: set) -> bool:
    """
    Check if a claim pair has already been processed.
    
    Args:
        claim_no_a: First claim number
        claim_no_b: Second claim number
        matched_pairs: Set of already processed pairs
        
    Returns:
        True if pair already processed
    """
    pair = tuple(sorted([claim_no_a, claim_no_b]))
    return pair in matched_pairs
