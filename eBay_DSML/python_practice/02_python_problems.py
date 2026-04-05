"""
================================================================
eBay DS/ML Interview — Python Practice Problems
================================================================
20 hands-on problems covering:
  - pandas data wrangling & EDA
  - NumPy vectorized ops
  - Applied ML implementations from scratch
  - Performance optimization patterns

Difficulty: ⭐ Easy  ⭐⭐ Medium  ⭐⭐⭐ Hard  ⭐⭐⭐⭐ Expert

Instructions:
  1. Write your solution in the function stub
  2. Run the test assertions to validate
  3. Only look at hints/solutions if you're stuck for 15+ min
================================================================
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# ================================================================
# HELPER: Generate Sample eBay-like Data
# ================================================================
def generate_sample_data(n_users=1000, n_listings=5000, n_transactions=10000):
    """Generate realistic eBay marketplace datasets for practice."""
    np.random.seed(42)
    
    categories = ['Electronics', 'Fashion', 'Home & Garden', 'Collectibles', 
                   'Sports', 'Toys', 'Books', 'Auto Parts', 'Jewelry', 'Art']
    conditions = ['New', 'Like New', 'Good', 'Fair', 'Poor']
    countries = ['US', 'UK', 'DE', 'AU', 'CA', 'IN', 'FR', 'JP']
    devices = ['mobile', 'desktop', 'tablet']
    sources = ['search', 'recommendation', 'direct', 'email']
    
    # Users
    users = pd.DataFrame({
        'user_id': range(1, n_users + 1),
        'country': np.random.choice(countries, n_users),
        'signup_date': pd.date_range('2023-01-01', periods=n_users, freq='4h'),
        'user_type': np.random.choice(['buyer', 'seller', 'both'], n_users, p=[0.5, 0.2, 0.3])
    })
    
    # Listings
    seller_ids = users[users['user_type'].isin(['seller', 'both'])]['user_id'].values
    listings = pd.DataFrame({
        'listing_id': range(1, n_listings + 1),
        'seller_id': np.random.choice(seller_ids, n_listings),
        'category': np.random.choice(categories, n_listings),
        'price': np.round(np.random.lognormal(3, 1.5, n_listings), 2).clip(1, 10000),
        'condition': np.random.choice(conditions, n_listings, p=[0.3, 0.2, 0.25, 0.15, 0.1]),
        'listing_date': pd.date_range('2024-01-01', periods=n_listings, freq='2h'),
        'status': np.random.choice(['active', 'sold', 'ended', 'removed'], n_listings, p=[0.4, 0.3, 0.2, 0.1])
    })
    
    # Transactions
    buyer_ids = users[users['user_type'].isin(['buyer', 'both'])]['user_id'].values
    transactions = pd.DataFrame({
        'transaction_id': range(1, n_transactions + 1),
        'buyer_id': np.random.choice(buyer_ids, n_transactions),
        'seller_id': np.random.choice(seller_ids, n_transactions),
        'listing_id': np.random.choice(listings['listing_id'].values, n_transactions),
        'transaction_date': pd.date_range('2024-06-01', periods=n_transactions, freq='30min'),
        'sale_price': np.round(np.random.lognormal(3, 1.2, n_transactions), 2).clip(1, 8000),
        'quantity': np.random.choice([1, 1, 1, 2, 3], n_transactions),
        'shipping_cost': np.round(np.random.uniform(0, 25, n_transactions), 2),
        'status': np.random.choice(['completed', 'cancelled', 'returned', 'pending'], 
                                    n_transactions, p=[0.75, 0.1, 0.05, 0.1])
    })
    
    # Page Views
    n_views = n_transactions * 5
    page_views = pd.DataFrame({
        'view_id': range(1, n_views + 1),
        'user_id': np.random.choice(users['user_id'].values, n_views),
        'listing_id': np.random.choice(listings['listing_id'].values, n_views),
        'view_time': pd.date_range('2024-06-01', periods=n_views, freq='5min'),
        'source': np.random.choice(sources, n_views, p=[0.4, 0.3, 0.2, 0.1]),
        'device_type': np.random.choice(devices, n_views, p=[0.5, 0.35, 0.15])
    })
    
    return {
        'users': users,
        'listings': listings,
        'transactions': transactions,
        'page_views': page_views
    }


# Load sample data
DATA = generate_sample_data()
users = DATA['users']
listings = DATA['listings']
transactions = DATA['transactions']
page_views = DATA['page_views']


# ================================================================
# PROBLEM 1: Daily Conversion Rate by Category ⭐
# ================================================================
def daily_conversion_rate(transactions_df: pd.DataFrame, 
                          listings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the daily conversion rate per product category.
    A conversion = a transaction with status='completed'.
    
    Returns: DataFrame with columns [date, category, total_transactions, 
             completed_transactions, conversion_rate]
    """
    # YOUR SOLUTION HERE:
    pass


# SOLUTION:
def _solution_daily_conversion_rate(transactions_df, listings_df):
    df = transactions_df.merge(listings_df[['listing_id', 'category']], on='listing_id')
    df['date'] = df['transaction_date'].dt.date
    
    total = df.groupby(['date', 'category']).size().reset_index(name='total_transactions')
    completed = (df[df['status'] == 'completed']
                 .groupby(['date', 'category']).size()
                 .reset_index(name='completed_transactions'))
    
    result = total.merge(completed, on=['date', 'category'], how='left')
    result['completed_transactions'] = result['completed_transactions'].fillna(0).astype(int)
    result['conversion_rate'] = result['completed_transactions'] / result['total_transactions']
    return result.sort_values(['date', 'category']).reset_index(drop=True)


# ================================================================
# PROBLEM 2: User-Level Feature Table for Churn Model ⭐⭐
# ================================================================
def create_churn_features(transactions_df: pd.DataFrame, 
                          listings_df: pd.DataFrame,
                          reference_date: str = '2025-03-01') -> pd.DataFrame:
    """
    Create a user-level feature table for a buyer churn prediction model.
    
    Features per buyer:
    - total_spend_30d: total spend in last 30 days from reference_date
    - total_spend_90d: total spend in last 90 days
    - avg_order_value: average order value (all time)
    - num_orders: total number of completed orders
    - days_since_last_order: days since their last completed order
    - favorite_category: most frequently purchased category
    - num_unique_categories: number of distinct categories purchased from
    
    Returns: DataFrame indexed by buyer_id
    """
    # YOUR SOLUTION HERE:
    pass


# SOLUTION:
def _solution_create_churn_features(transactions_df, listings_df, reference_date='2025-03-01'):
    ref = pd.Timestamp(reference_date)
    completed = transactions_df[transactions_df['status'] == 'completed'].copy()
    completed = completed.merge(listings_df[['listing_id', 'category']], on='listing_id')
    completed['revenue'] = completed['sale_price'] * completed['quantity']
    
    # Time-windowed spend
    spend_30d = (completed[completed['transaction_date'] >= ref - timedelta(days=30)]
                 .groupby('buyer_id')['revenue'].sum()
                 .rename('total_spend_30d'))
    spend_90d = (completed[completed['transaction_date'] >= ref - timedelta(days=90)]
                 .groupby('buyer_id')['revenue'].sum()
                 .rename('total_spend_90d'))
    
    # Aggregates
    agg = completed.groupby('buyer_id').agg(
        avg_order_value=('revenue', 'mean'),
        num_orders=('transaction_id', 'count'),
        last_order_date=('transaction_date', 'max'),
        num_unique_categories=('category', 'nunique')
    )
    agg['days_since_last_order'] = (ref - agg['last_order_date']).dt.days
    agg = agg.drop(columns=['last_order_date'])
    
    # Favorite category
    fav_cat = (completed.groupby(['buyer_id', 'category']).size()
               .reset_index(name='cnt')
               .sort_values('cnt', ascending=False)
               .drop_duplicates('buyer_id')
               .set_index('buyer_id')['category']
               .rename('favorite_category'))
    
    result = pd.concat([spend_30d, spend_90d, agg, fav_cat], axis=1)
    result[['total_spend_30d', 'total_spend_90d']] = result[['total_spend_30d', 'total_spend_90d']].fillna(0)
    return result


# ================================================================
# PROBLEM 3: Parse Nested JSON Logs to Flat DataFrame ⭐⭐
# ================================================================
def parse_search_logs(logs: List[Dict]) -> pd.DataFrame:
    """
    Parse nested search event logs into a flat DataFrame.
    
    Input format (each log entry):
    {
        "user_id": 123,
        "query": "vintage watch",
        "results": [
            {"listing_id": 1, "position": 1, "clicked": True},
            {"listing_id": 5, "position": 2, "clicked": False}
        ],
        "timestamp": "2025-01-15T10:30:00"
    }
    
    Returns: Flat DataFrame with columns:
    [user_id, query, timestamp, listing_id, position, clicked]
    """
    # YOUR SOLUTION HERE:
    pass


# SOLUTION:
def _solution_parse_search_logs(logs):
    rows = []
    for entry in logs:
        for result in entry.get('results', []):
            rows.append({
                'user_id': entry['user_id'],
                'query': entry['query'],
                'timestamp': pd.Timestamp(entry['timestamp']),
                'listing_id': result['listing_id'],
                'position': result['position'],
                'clicked': result['clicked']
            })
    return pd.DataFrame(rows)


# Test data for Problem 3
sample_logs = [
    {
        "user_id": 1, "query": "vintage watch",
        "results": [
            {"listing_id": 101, "position": 1, "clicked": True},
            {"listing_id": 102, "position": 2, "clicked": False},
            {"listing_id": 103, "position": 3, "clicked": True}
        ],
        "timestamp": "2025-01-15T10:30:00"
    },
    {
        "user_id": 2, "query": "nike shoes",
        "results": [
            {"listing_id": 201, "position": 1, "clicked": False},
            {"listing_id": 202, "position": 2, "clicked": True}
        ],
        "timestamp": "2025-01-15T11:00:00"
    }
]


# ================================================================
# PROBLEM 4: Clean Messy Price Column ⭐⭐
# ================================================================
def clean_prices(prices: pd.Series, exchange_rates: Dict[str, float]) -> pd.Series:
    """
    Clean a messy price column with various formats:
    '$12.50', '12,50€', 'FREE', 'N/A', '₹850', '£45.00', '¥1200'
    
    Convert all to USD using the provided exchange_rates dict.
    Return NaN for unparseable values.
    
    exchange_rates: dict mapping currency symbol to USD rate
    e.g., {'€': 1.08, '£': 1.27, '₹': 0.012, '¥': 0.0067}
    """
    # YOUR SOLUTION HERE:
    pass


# SOLUTION:
def _solution_clean_prices(prices, exchange_rates):
    def parse_price(val):
        if pd.isna(val):
            return np.nan
        val = str(val).strip()
        if val.upper() in ('FREE', 'N/A', '', 'NONE', 'NULL'):
            return np.nan if val.upper() != 'FREE' else 0.0
        
        # Detect currency
        currency_rate = 1.0  # default USD
        for symbol, rate in exchange_rates.items():
            if symbol in val:
                currency_rate = rate
                val = val.replace(symbol, '')
                break
        val = val.replace('$', '')
        
        # Handle European comma-as-decimal format
        if ',' in val and '.' not in val:
            val = val.replace(',', '.')
        elif ',' in val and '.' in val:
            val = val.replace(',', '')  # thousands separator
        
        try:
            return round(float(val.strip()) * currency_rate, 2)
        except ValueError:
            return np.nan
    
    return prices.apply(parse_price)


# Test data for Problem 4
test_prices = pd.Series(['$12.50', '12,50€', 'FREE', 'N/A', '₹850', 
                          '£45.00', '¥1200', '$1,234.56', None, 'bad'])
test_exchange_rates = {'€': 1.08, '£': 1.27, '₹': 0.012, '¥': 0.0067}


# ================================================================
# PROBLEM 5: User-Item Matrix for Collaborative Filtering ⭐⭐
# ================================================================
def create_user_item_matrix(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a user-item interaction matrix from transactions.
    Rows = buyers, Columns = listing categories, Values = purchase count.
    Only include completed transactions.
    
    Returns: DataFrame (users × categories) with purchase counts
    """
    # YOUR SOLUTION HERE:
    pass


# SOLUTION:
def _solution_create_user_item_matrix(transactions_df):
    completed = transactions_df[transactions_df['status'] == 'completed']
    merged = completed.merge(listings[['listing_id', 'category']], on='listing_id')
    matrix = merged.pivot_table(
        index='buyer_id', 
        columns='category', 
        values='transaction_id', 
        aggfunc='count', 
        fill_value=0
    )
    return matrix


# ================================================================
# PROBLEM 6: NDCG@K from Scratch ⭐⭐⭐
# ================================================================
def ndcg_at_k(relevances: List[float], k: int) -> float:
    """
    Compute NDCG@K (Normalized Discounted Cumulative Gain).
    
    Args:
        relevances: list of relevance scores in the order shown to user
        k: cutoff position
    
    Returns: NDCG@K score (float between 0 and 1)
    """
    # YOUR SOLUTION HERE:
    pass


# SOLUTION:
def _solution_ndcg_at_k(relevances, k):
    def dcg(rels, k):
        rels = np.array(rels[:k])
        positions = np.arange(1, len(rels) + 1)
        return np.sum(rels / np.log2(positions + 1))
    
    actual_dcg = dcg(relevances, k)
    ideal_dcg = dcg(sorted(relevances, reverse=True), k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# Assertions for NDCG
assert abs(_solution_ndcg_at_k([3, 2, 3, 0, 1, 2], 6) - 0.9608) < 0.01
assert _solution_ndcg_at_k([3, 3, 2, 2, 1, 0], 6) == 1.0  # perfect ranking
assert _solution_ndcg_at_k([0, 0, 0], 3) == 0.0  # no relevant results


# ================================================================
# PROBLEM 7: Price Anomaly Detection ⭐⭐
# ================================================================
def detect_price_anomalies(prices: pd.Series, 
                           dates: pd.Series, 
                           window: int = 7, 
                           threshold: float = 3.0) -> pd.DataFrame:
    """
    Detect price anomalies in a time series.
    An anomaly = price more than `threshold` standard deviations from 
    the rolling `window`-day mean.
    
    Returns: DataFrame with columns [date, price, rolling_mean, rolling_std, 
             z_score, is_anomaly]
    """
    # YOUR SOLUTION HERE:
    pass


# SOLUTION:
def _solution_detect_price_anomalies(prices, dates, window=7, threshold=3.0):
    df = pd.DataFrame({'date': dates, 'price': prices}).sort_values('date')
    df['rolling_mean'] = df['price'].rolling(window=window, min_periods=1).mean()
    df['rolling_std'] = df['price'].rolling(window=window, min_periods=1).std()
    df['z_score'] = (df['price'] - df['rolling_mean']) / df['rolling_std'].replace(0, np.nan)
    df['is_anomaly'] = df['z_score'].abs() > threshold
    return df.reset_index(drop=True)


# ================================================================
# PROBLEM 8: TF-IDF from Scratch ⭐⭐⭐
# ================================================================
def compute_tfidf(documents: List[str]) -> pd.DataFrame:
    """
    Compute TF-IDF matrix from scratch (NO sklearn).
    
    TF(t, d) = count of term t in doc d / total terms in doc d
    IDF(t) = log(total docs / docs containing t)
    TF-IDF(t, d) = TF * IDF
    
    Returns: DataFrame (documents × terms) with TF-IDF values
    """
    # YOUR SOLUTION HERE:
    pass


# SOLUTION:
def _solution_compute_tfidf(documents):
    # Tokenize
    tokenized = [doc.lower().split() for doc in documents]
    all_terms = sorted(set(term for doc in tokenized for term in doc))
    n_docs = len(documents)
    
    # TF
    tf_matrix = []
    for doc in tokenized:
        doc_len = len(doc)
        tf = {term: doc.count(term) / doc_len for term in all_terms}
        tf_matrix.append(tf)
    
    # IDF
    idf = {}
    for term in all_terms:
        doc_count = sum(1 for doc in tokenized if term in doc)
        idf[term] = np.log(n_docs / doc_count)
    
    # TF-IDF
    tfidf_matrix = []
    for tf in tf_matrix:
        tfidf = {term: tf[term] * idf[term] for term in all_terms}
        tfidf_matrix.append(tfidf)
    
    return pd.DataFrame(tfidf_matrix)


# ================================================================
# PROBLEM 9: Merge Overlapping Sessions ⭐⭐
# ================================================================
def merge_sessions(sessions: List[Tuple[datetime, datetime]]) -> List[Tuple[datetime, datetime]]:
    """
    Merge overlapping time sessions.
    
    Input: list of (start_time, end_time) tuples
    Output: list of merged (start_time, end_time) tuples
    """
    # YOUR SOLUTION HERE:
    pass


# SOLUTION:
def _solution_merge_sessions(sessions):
    if not sessions:
        return []
    sessions = sorted(sessions, key=lambda x: x[0])
    merged = [sessions[0]]
    for start, end in sessions[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


# Test
test_sessions = [
    (datetime(2025, 1, 1, 10, 0), datetime(2025, 1, 1, 10, 30)),
    (datetime(2025, 1, 1, 10, 15), datetime(2025, 1, 1, 11, 0)),  # overlaps
    (datetime(2025, 1, 1, 12, 0), datetime(2025, 1, 1, 13, 0)),   # no overlap
]
assert len(_solution_merge_sessions(test_sessions)) == 2


# ================================================================
# PROBLEM 10: Jaccard Similarity & Similar Users ⭐⭐
# ================================================================
def find_similar_users(transactions_df: pd.DataFrame, 
                       target_user_id: int, 
                       top_n: int = 5) -> pd.DataFrame:
    """
    Find the top-N most similar users to target_user_id based on 
    Jaccard similarity of purchased listing IDs.
    
    Jaccard(A, B) = |A ∩ B| / |A ∪ B|
    
    Returns: DataFrame with columns [user_id, jaccard_similarity, common_items]
    """
    # YOUR SOLUTION HERE:
    pass


# SOLUTION:
def _solution_find_similar_users(transactions_df, target_user_id, top_n=5):
    completed = transactions_df[transactions_df['status'] == 'completed']
    user_items = completed.groupby('buyer_id')['listing_id'].apply(set).to_dict()
    
    if target_user_id not in user_items:
        return pd.DataFrame(columns=['user_id', 'jaccard_similarity', 'common_items'])
    
    target_items = user_items[target_user_id]
    similarities = []
    
    for uid, items in user_items.items():
        if uid == target_user_id:
            continue
        intersection = len(target_items & items)
        union = len(target_items | items)
        jaccard = intersection / union if union > 0 else 0
        if jaccard > 0:
            similarities.append({
                'user_id': uid,
                'jaccard_similarity': round(jaccard, 4),
                'common_items': intersection
            })
    
    result = pd.DataFrame(similarities)
    return result.nlargest(top_n, 'jaccard_similarity').reset_index(drop=True)


# ================================================================
# PROBLEM 11: Vectorize Slow Loop ⭐
# ================================================================
def slow_discount(df: pd.DataFrame) -> pd.DataFrame:
    """
    WRONG/SLOW approach — fix this!
    
    For each row:
    - If category == 'Electronics', discount = price * 0.1
    - If category == 'Fashion', discount = price * 0.15
    - Otherwise, discount = 0
    
    Rewrite using vectorized operations (no loops!)
    """
    # SLOW VERSION (DON'T DO THIS):
    # for idx, row in df.iterrows():
    #     if row['category'] == 'Electronics':
    #         df.loc[idx, 'discount'] = row['price'] * 0.1
    #     elif row['category'] == 'Fashion':
    #         df.loc[idx, 'discount'] = row['price'] * 0.15
    #     else:
    #         df.loc[idx, 'discount'] = 0
    
    # YOUR FAST SOLUTION:
    pass


# SOLUTION:
def _solution_fast_discount(df):
    df = df.copy()
    conditions = [
        df['category'] == 'Electronics',
        df['category'] == 'Fashion'
    ]
    choices = [df['price'] * 0.1, df['price'] * 0.15]
    df['discount'] = np.select(conditions, choices, default=0)
    return df


# ================================================================
# PROBLEM 12: Cross-Validation from Scratch ⭐⭐⭐
# ================================================================
def kfold_cross_validation(X: np.ndarray, y: np.ndarray, 
                           k: int = 5) -> Dict[str, float]:
    """
    Implement k-fold cross-validation from scratch.
    Use a simple logistic regression (sklearn allowed for the model only).
    
    Returns: dict with 'mean_accuracy' and 'std_accuracy'
    """
    # YOUR SOLUTION HERE:
    pass


# SOLUTION:
def _solution_kfold_cv(X, y, k=5):
    from sklearn.linear_model import LogisticRegression
    
    n = len(y)
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // k
    accuracies = []
    
    for i in range(k):
        test_idx = indices[i * fold_size: (i + 1) * fold_size]
        train_idx = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        accuracies.append(acc)
    
    return {
        'mean_accuracy': round(np.mean(accuracies), 4),
        'std_accuracy': round(np.std(accuracies), 4)
    }


# ================================================================
# PROBLEM 13: Logistic Regression from Scratch ⭐⭐⭐
# ================================================================
def logistic_regression_scratch(X: np.ndarray, y: np.ndarray, 
                                lr: float = 0.01, 
                                epochs: int = 1000) -> Tuple[np.ndarray, float]:
    """
    Implement logistic regression using gradient descent.
    NO sklearn allowed.
    
    Returns: (weights, bias)
    """
    # YOUR SOLUTION HERE:
    pass


# SOLUTION:
def _solution_logistic_regression(X, y, lr=0.01, epochs=1000):
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0.0
    
    for _ in range(epochs):
        z = X @ weights + bias
        # Clip for numerical stability
        z = np.clip(z, -500, 500)
        predictions = 1 / (1 + np.exp(-z))
        
        # Gradients
        dw = (1 / m) * X.T @ (predictions - y)
        db = (1 / m) * np.sum(predictions - y)
        
        weights -= lr * dw
        bias -= lr * db
    
    return weights, bias


# Quick test
X_test_lr = np.random.randn(100, 3)
y_test_lr = (X_test_lr[:, 0] + X_test_lr[:, 1] > 0).astype(float)
w, b = _solution_logistic_regression(X_test_lr, y_test_lr, lr=0.1, epochs=500)
preds = (1 / (1 + np.exp(-(X_test_lr @ w + b)))) > 0.5
accuracy = np.mean(preds == y_test_lr)
assert accuracy > 0.8, f"Logistic regression accuracy too low: {accuracy}"


# ================================================================
# PROBLEM 14: Z-Test for A/B Test ⭐⭐
# ================================================================
def ab_test_ztest(control_conversions: int, control_total: int,
                  treatment_conversions: int, treatment_total: int,
                  alpha: float = 0.05) -> Dict:
    """
    Perform a two-sample Z-test for proportions.
    
    Returns: dict with keys:
    - control_rate, treatment_rate
    - absolute_lift, relative_lift_pct
    - z_score, p_value
    - is_significant (bool)
    """
    # YOUR SOLUTION HERE:
    pass


# SOLUTION:
def _solution_ab_test_ztest(control_conv, control_total, 
                            treatment_conv, treatment_total, alpha=0.05):
    from scipy import stats
    
    p_c = control_conv / control_total
    p_t = treatment_conv / treatment_total
    p_pool = (control_conv + treatment_conv) / (control_total + treatment_total)
    
    se = np.sqrt(p_pool * (1 - p_pool) * (1/control_total + 1/treatment_total))
    z = (p_t - p_c) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return {
        'control_rate': round(p_c, 6),
        'treatment_rate': round(p_t, 6),
        'absolute_lift': round(p_t - p_c, 6),
        'relative_lift_pct': round((p_t - p_c) / p_c * 100, 2),
        'z_score': round(z, 4),
        'p_value': round(p_value, 6),
        'is_significant': p_value < alpha
    }


# Test: should be significant
result = _solution_ab_test_ztest(500, 10000, 600, 10000)
assert result['is_significant'] == True
assert result['relative_lift_pct'] == 20.0


# ================================================================
# PROBLEM 15: Content-Based Recommender ⭐⭐⭐
# ================================================================
def content_based_recommender(items_df: pd.DataFrame, 
                              target_listing_id: int, 
                              top_n: int = 5) -> pd.DataFrame:
    """
    Simple content-based recommender using cosine similarity 
    on item features (category, condition, price bucket).
    
    Returns: DataFrame with columns [listing_id, similarity_score]
    """
    # YOUR SOLUTION HERE:
    pass


# SOLUTION:
def _solution_content_recommender(items_df, target_listing_id, top_n=5):
    df = items_df.copy()
    
    # Feature engineering
    df['price_bucket'] = pd.cut(df['price'], bins=[0, 10, 50, 200, 1000, np.inf], 
                                 labels=['budget', 'low', 'mid', 'high', 'premium'])
    
    # One-hot encode
    features = pd.get_dummies(df[['category', 'condition', 'price_bucket']], dtype=float)
    
    # Cosine similarity
    target_idx = df[df['listing_id'] == target_listing_id].index[0]
    target_vec = features.loc[target_idx].values
    
    # Vectorized cosine similarity
    norms = np.linalg.norm(features.values, axis=1) * np.linalg.norm(target_vec)
    similarities = features.values @ target_vec / np.where(norms > 0, norms, 1)
    
    df['similarity_score'] = similarities
    result = (df[df['listing_id'] != target_listing_id]
              .nlargest(top_n, 'similarity_score')
              [['listing_id', 'similarity_score']]
              .reset_index(drop=True))
    return result


# ================================================================
# PROBLEM 16: Data Pipeline with Proper Preprocessing ⭐⭐
# ================================================================
def build_pipeline(df: pd.DataFrame, target_col: str, 
                   test_size: float = 0.2) -> Dict:
    """
    Build a complete preprocessing pipeline:
    1. Split into train/test FIRST (avoid data leakage!)
    2. Handle missing values (median for numeric, mode for categorical)
    3. Encode categoricals (one-hot)
    4. Scale numeric features (StandardScaler)
    
    Returns: dict with X_train, X_test, y_train, y_test, feature_names
    """
    # YOUR SOLUTION HERE:
    pass


# SOLUTION:
def _solution_build_pipeline(df, target_col, test_size=0.2):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 1. SPLIT FIRST (crucial!)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # 2. Identify column types
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 3. Handle missing values (fit on train only!)
    for col in numeric_cols:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)
    
    for col in categorical_cols:
        mode_val = X_train[col].mode()[0]
        X_train[col] = X_train[col].fillna(mode_val)
        X_test[col] = X_test[col].fillna(mode_val)
    
    # 4. One-hot encode (fit on train, transform both)
    X_train = pd.get_dummies(X_train, columns=categorical_cols, dtype=float)
    X_test = pd.get_dummies(X_test, columns=categorical_cols, dtype=float)
    
    # Align columns
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    
    # 5. Scale numeric (fit on train only!)
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': X_train.columns.tolist()
    }


# ================================================================
# PROBLEM 17: Data Leakage Detection ⭐⭐
# ================================================================
def identify_leakage(code_snippet: str) -> List[str]:
    """
    This is a CONCEPTUAL question (not code-based).
    
    Identify ALL data leakage issues in this pipeline:
    
    ```python
    # 1. Load data
    df = pd.read_csv('transactions.csv')
    
    # 2. Feature engineering
    df['avg_category_price'] = df.groupby('category')['price'].transform('mean')
    
    # 3. Scale all features
    scaler = StandardScaler()
    X = scaler.fit_transform(df.drop('target', axis=1))
    
    # 4. Split
    X_train, X_test = X[:8000], X[8000:]
    y_train, y_test = df['target'][:8000], df['target'][8000:]
    
    # 5. Feature selection using all data
    from sklearn.feature_selection import SelectKBest
    selector = SelectKBest(k=10)
    X_selected = selector.fit_transform(X, df['target'])
    
    # 6. Train model
    model.fit(X_train, y_train)
    ```
    
    List all leakage issues:
    """
    return [
        "1. groupby transform uses entire dataset including test data for avg_category_price",
        "2. StandardScaler fit_transform on entire dataset before splitting",
        "3. Train/test split is NOT random — temporal bias if data is time-ordered",
        "4. SelectKBest fit_transform on entire X and y — uses test labels!",
        "5. After SelectKBest, original X_train/X_test aren't updated with selected features"
    ]


# ================================================================
# PROBLEM 18: Cosine Similarity from Scratch ⭐
# ================================================================
def cosine_similarity_manual(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors WITHOUT sklearn.
    """
    # YOUR SOLUTION HERE:
    pass


# SOLUTION:
def _solution_cosine_similarity(vec_a, vec_b):
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

# Test
assert abs(_solution_cosine_similarity(np.array([1, 0, 0]), np.array([1, 0, 0])) - 1.0) < 1e-6
assert abs(_solution_cosine_similarity(np.array([1, 0, 0]), np.array([0, 1, 0])) - 0.0) < 1e-6


# ================================================================
# PROBLEM 19: Stratified Sampling for Imbalanced Data ⭐⭐
# ================================================================
def balanced_sample(df: pd.DataFrame, target_col: str, 
                    strategy: str = 'oversample') -> pd.DataFrame:
    """
    Create a balanced dataset from an imbalanced one.
    
    strategy:
    - 'oversample': duplicate minority class to match majority
    - 'undersample': reduce majority class to match minority
    
    Returns: balanced DataFrame
    """
    # YOUR SOLUTION HERE:
    pass


# SOLUTION:
def _solution_balanced_sample(df, target_col, strategy='oversample'):
    counts = df[target_col].value_counts()
    minority_class = counts.idxmin()
    majority_class = counts.idxmax()
    
    df_minority = df[df[target_col] == minority_class]
    df_majority = df[df[target_col] == majority_class]
    
    if strategy == 'oversample':
        df_minority_upsampled = df_minority.sample(
            n=len(df_majority), replace=True, random_state=42
        )
        result = pd.concat([df_majority, df_minority_upsampled])
    elif strategy == 'undersample':
        df_majority_downsampled = df_majority.sample(
            n=len(df_minority), replace=False, random_state=42
        )
        result = pd.concat([df_majority_downsampled, df_minority])
    
    return result.sample(frac=1, random_state=42).reset_index(drop=True)


# ================================================================
# PROBLEM 20: GMV Time Series Forecasting Features ⭐⭐⭐
# ================================================================
def create_ts_features(daily_gmv: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-series features for GMV forecasting.
    
    Input: DataFrame with columns [date, gmv]
    
    Create features:
    - day_of_week, month, is_weekend
    - lag_1, lag_7, lag_30 (previous GMV values)
    - rolling_mean_7, rolling_mean_30
    - rolling_std_7
    - mom_growth (month-over-month growth)
    - ewma_7 (exponential weighted moving average)
    
    Returns: DataFrame with all features (drop rows with NaN from lags)
    """
    # YOUR SOLUTION HERE:
    pass


# SOLUTION:
def _solution_create_ts_features(daily_gmv):
    df = daily_gmv.copy().sort_values('date').reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])
    
    # Calendar features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Lag features
    df['lag_1'] = df['gmv'].shift(1)
    df['lag_7'] = df['gmv'].shift(7)
    df['lag_30'] = df['gmv'].shift(30)
    
    # Rolling features
    df['rolling_mean_7'] = df['gmv'].rolling(7).mean()
    df['rolling_mean_30'] = df['gmv'].rolling(30).mean()
    df['rolling_std_7'] = df['gmv'].rolling(7).std()
    
    # Growth
    df['mom_growth'] = df['gmv'].pct_change(periods=30)
    
    # EWMA  
    df['ewma_7'] = df['gmv'].ewm(span=7).mean()
    
    # Drop NaN rows from lags
    df = df.dropna().reset_index(drop=True)
    
    return df


# ================================================================
# MAIN: Run all tests
# ================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("eBay DS/ML Interview — Python Practice Problems")
    print("=" * 60)
    
    # Test Problem 3: JSON parsing
    result_3 = _solution_parse_search_logs(sample_logs)
    assert len(result_3) == 5
    assert list(result_3.columns) == ['user_id', 'query', 'timestamp', 
                                       'listing_id', 'position', 'clicked']
    print("✅ Problem 3 (JSON Parsing) — PASSED")
    
    # Test Problem 4: Price cleaning
    result_4 = _solution_clean_prices(test_prices, test_exchange_rates)
    assert result_4.iloc[0] == 12.50  # $12.50
    assert result_4.iloc[2] == 0.0    # FREE
    assert np.isnan(result_4.iloc[3])  # N/A
    print("✅ Problem 4 (Price Cleaning) — PASSED")
    
    # Test Problem 6: NDCG
    assert abs(_solution_ndcg_at_k([3, 2, 3, 0, 1, 2], 6) - 0.9608) < 0.01
    print("✅ Problem 6 (NDCG@K) — PASSED")
    
    # Test Problem 9: Merge sessions
    assert len(_solution_merge_sessions(test_sessions)) == 2
    print("✅ Problem 9 (Merge Sessions) — PASSED")
    
    # Test Problem 14: A/B Test
    result_14 = _solution_ab_test_ztest(500, 10000, 600, 10000)
    assert result_14['is_significant'] == True
    print("✅ Problem 14 (A/B Test Z-Test) — PASSED")
    
    # Test Problem 18: Cosine similarity
    assert abs(_solution_cosine_similarity(
        np.array([1, 2, 3]), np.array([1, 2, 3])) - 1.0) < 1e-6
    print("✅ Problem 18 (Cosine Similarity) — PASSED")
    
    # Test Problem 13: Logistic regression
    X_t = np.random.randn(200, 4)
    y_t = (X_t[:, 0] + X_t[:, 1] > 0).astype(float)
    w, b = _solution_logistic_regression(X_t, y_t, lr=0.1, epochs=500)
    preds = (1 / (1 + np.exp(-(X_t @ w + b)))) > 0.5
    assert np.mean(preds == y_t) > 0.85
    print("✅ Problem 13 (Logistic Regression) — PASSED")
    
    print("\n" + "=" * 60)
    print("All solution tests PASSED! 🎉")
    print("Now try solving them yourself in the function stubs above.")
    print("=" * 60)
