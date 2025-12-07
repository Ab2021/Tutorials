"""
Date parsing and comparison utilities for string matching.
"""

import math
import pandas as pd
from datetime import date, datetime
from dateutil import parser as dateparser


def parse_date(s, dayfirst: bool = True):
    """
    Robust date parser that handles multiple input types.
    
    Handles:
    - String (date, datetime, ISO8601, etc.)
    - pd.Timestamp
    - pandas NaT
    - datetime.datetime
    - datetime.date
    
    Returns:
        pd.Timestamp or pd.NaT
    """
    # Handle pd.NaT and pd.NA and None
    if s is None or pd.isna(s):
        return pd.NaT
    
    # If input is already pd.Timestamp
    if isinstance(s, pd.Timestamp):
        return s
    
    # If input is datetime.datetime
    if isinstance(s, datetime):
        return pd.Timestamp(s)
    
    # If input is datetime.date
    if isinstance(s, date):
        return pd.Timestamp(s)
    
    # Try to parse string
    try:
        s_str = str(s)
        if not s_str.strip():
            return pd.NaT
        dt = dateparser.parse(s_str, dayfirst=dayfirst)
        if dt is not None:
            return pd.Timestamp(dt)
    except Exception:
        pass
    
    return pd.NaT


def parse_date_to_date(s, dayfirst: bool = True):
    """
    Parse date and return Python date object.
    
    Returns:
        datetime.date or None
    """
    # Handle pd.NaT and pd.NA and None
    if s is None or pd.isna(s):
        return None
    
    # If input is already datetime.date or datetime.datetime
    if isinstance(s, datetime):
        return s.date()
    if isinstance(s, date):
        return s
    
    # Try to parse string
    try:
        s_str = str(s)
        if not s_str.strip():
            return None
        dt = dateparser.parse(s_str, dayfirst=dayfirst)
        if dt is not None:
            return dt.date()
    except Exception:
        pass
    
    return None


def date_diff_days(d1, d2) -> float:
    """
    Calculate absolute difference in days between two dates.
    
    Returns:
        Number of days between dates, or math.inf if either date is invalid
    """
    if d1 is None or d2 is None:
        return math.inf
    if pd.isna(d1) or pd.isna(d2):
        return math.inf
    
    try:
        return abs((d1 - d2).days)
    except Exception:
        return math.inf


def get_candidates_within_days(df: pd.DataFrame, base_date, days: int = 0, 
                               date_col: str = 'tpa_dt_loss') -> pd.DataFrame:
    """
    Filter dataframe to rows where date column is within +/- days of base_date.
    
    Args:
        df: DataFrame to filter
        base_date: Reference date (pd.Timestamp or datetime)
        days: Tolerance in days (default 0 = exact match)
        date_col: Name of date column to filter on
        
    Returns:
        Filtered DataFrame
    """
    if pd.isna(base_date):
        return df.iloc[0:0]  # Return empty DataFrame with same columns
    
    # Ensure base_date is pd.Timestamp
    if not isinstance(base_date, pd.Timestamp):
        base_date = pd.Timestamp(base_date)
    
    mask = (
        (df[date_col] >= (base_date - pd.Timedelta(days=days))) &
        (df[date_col] <= (base_date + pd.Timedelta(days=days)))
    )
    return df[mask]


def is_date_within_tolerance(d1, d2, tolerance_days: int = 0) -> bool:
    """
    Check if two dates are within tolerance days of each other.
    
    Returns:
        True if dates are within tolerance, False otherwise
    """
    diff = date_diff_days(d1, d2)
    return diff <= tolerance_days
