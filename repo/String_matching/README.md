# String Matching - BM25 Based Claim Linkage

A modular Python framework for deterministic insurance claim matching using **BM25 + fuzzy string matching** instead of GPT. Eliminates hallucination issues while providing fast, cost-effective claim linkage and clustering.

## 🎯 Overview

This project refactors GPT-based claim matching into a production-ready, modular system that:

- **Matches X claims with TPA claims** (X-TPA linkage)
- **Clusters duplicate TPA claims** (TPA clustering)
- Uses **BM25 + RapidFuzz** for deterministic similarity scoring
- Supports optional **BERT embeddings** for semantic matching
- Provides **nickname-aware name matching**
- Implements **two-pass matching** strategies

### Key Improvements Over GPT

| Aspect | GPT-Based | BM25-Based (This Project) |
|--------|-----------|---------------------------|
| **Hallucination** | High risk | Zero |
| **Speed** | Slow (API latency) | Fast (local) |
| **Cost** | API fees | Free |
| **Consistency** | Variable | Deterministic |
| **Explainability** | Black box | Transparent scores |

## 📁 Project Structure

```
String_matching/
├── config/
│   ├── __init__.py
│   └── settings.py              # Constants, thresholds, mappings
├── utils/
│   ├── __init__.py
│   ├── text_processing.py       # Text normalization, extraction
│   ├── date_utils.py            # Date parsing and comparison
│   └── name_matcher.py          # Name matching with nicknames
├── similarity/
│   ├── __init__.py
│   └── bm25_scorer.py           # BM25 matching engine
├── data/
│   ├── __init__.py
│   └── data_loader.py           # Data loading and preprocessing
├── x_tpa_linkage.py             # Main: X ↔ TPA matching
├── tpa_clustering.py            # Main: TPA clustering
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
cd "d:\My Drive\Codes & Repos\repo\String_matching"

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (if needed)
python -c "import nltk; nltk.download('punkt')"
```

### Basic Usage

#### X-TPA Claim Linkage

```bash
python x_tpa_linkage.py
```

**What it does:**
- Loads TPA and X claims from Excel files
- Two-pass matching: first with state filter, then without
- Outputs matched/non-matched pairs with confidence scores

#### TPA Claim Clustering

```bash
python tpa_clustering.py
```

**What it does:**
- Loads TPA claims and clusters duplicates
- Uses NetworkX for connected components
- Outputs master DataFrame with group IDs

### Configuration

Edit `config/settings.py` to customize:

```python
# Similarity thresholds
X_TPA_NAME_THRESHOLD = 0.65    # Name similarity for X-TPA
X_TPA_DESC_THRESHOLD = 0.30    # Description similarity
TPA_NAME_THRESHOLD = 0.40      # Name threshold for clustering

# File paths
TPA_CLAIMS_PATH = "your/path/to/tpa_claims.xlsx"
X_CLAIMS_PATH = "your/path/to/x_claims.xlsx"
OUTPUT_DIR = "output/"
```

## 🔑 Key Features

### 1. BM25-Based Description Matching

Replaces GPT's semantic understanding with:
- **BM25Okapi** for term-level matching
- **Token set ratio** for word-level similarity
- **Partial ratio** for substring matching

```python
from similarity.bm25_scorer import BM25Matcher

matcher = BM25Matcher(mode='x_tpa')
result = matcher.match_records(record_a, record_b)
```

### 2. Nickname-Aware Name Matching

Built-in support for 200+ American name nicknames:

```python
from utils.name_matcher import compute_name_similarity

# "Bill" and "William" recognized as same name
score = compute_name_similarity("SMITH WILLIAM", "SMITH BILL")
# Returns: 0.85 (with nickname boost)
```

### 3. Two-Pass Matching Strategy

**Pass 1:** Match with all filters (date, state, coverage)  
**Pass 2:** Match remaining claims without state filter

### 4. Confidence Assignment

Post-processing rules assign confidence levels:
- **High**: Strong name + description match
- **Medium**: Moderate matches
- **Low**: Weak matches

### 5. Connected Components Clustering

Uses NetworkX to build claim groups from pairwise matches:

```python
# Automatically groups related claims
# Handles transitive relationships: A→B, B→C = group {A,B,C}
```

## 📊 Output Files

### X-TPA Linkage

- `matched_x_tpa_{timestamp}.parquet` - Matched pairs
- `matched_x_tpa_{timestamp}.xlsx` - Same as Excel
- `non_matched_x_tpa_{timestamp}.parquet` - Non-matched pairs

### TPA Clustering

- `tpa_clustered_master_{timestamp}.parquet` - Master file with group IDs
- `tpa_clustered_master_{timestamp}.xlsx` - Same as Excel
- `tpa_matched_{timestamp}.parquet` - Matched pair details
- `tpa_groups_{timestamp}.parquet` - Claim → group mapping

## 🎛️ Advanced Usage

### Optional: BERT Embeddings

For better semantic matching, enable BERT:

```bash
pip install sentence-transformers torch scikit-learn
```

Uncomment in `similarity/bm25_scorer.py`:

```python
# Use HybridMatcher instead of BM25Matcher
matcher = HybridMatcher(mode='x_tpa', use_bert=True)
```

### Custom Thresholds

Tune matching sensitivity in `config/settings.py`:

```python
# Stricter matching
X_TPA_NAME_THRESHOLD = 0.70  # Default: 0.65
X_TPA_DESC_THRESHOLD = 0.40  # Default: 0.30

# More lenient clustering
TPA_NAME_THRESHOLD = 0.35    # Default: 0.40
```

## 📖 Core Components

### BM25Matcher Class

```python
class BM25Matcher:
    """
    Main matching engine.
    
    Args:
        mode: 'x_tpa' or 'tpa_cluster'
        corpus: Optional pre-built corpus for BM25
    
    Returns:
        {
            'is_match': bool,
            'confidence': float,
            'reasons': str,
            'component_scores': {
                'name_string_sim': float,
                'description_string_sim': float,
                'state_string_sim': float
            }
        }
    """
```

### Name Matching

```python
compute_name_similarity(name_a, name_b)
# - Fuzzy token_set_ratio
# - +15% boost for nickname matches
# - +10% boost for matching last names
```

### Description Matching

```python
compute_description_similarity(text_a, text_b)
# Weighted combination:
# - 35% BM25 (term overlap)
# - 40% token_set_ratio (word-level)
# - 25% partial_ratio (substring)
```

## 🔧 Troubleshooting

### Import Errors

```bash
# Ensure you're in the project directory
cd "d:\My Drive\Codes & Repos\repo\String_matching"

# Reinstall dependencies
pip install -r requirements.txt
```

### File Not Found Errors

Update paths in `config/settings.py` or place files in:
- `data/tpa_claims.xlsx`
- `data/x_claims.xlsx`

### Low Match Rates

Try adjusting thresholds in `config/settings.py`:
- Lower `X_TPA_NAME_THRESHOLD` (e.g., 0.60)
- Lower `X_TPA_DESC_THRESHOLD` (e.g., 0.25)

## 📚 Documentation

- **Walkthrough**: See `walkthrough.md` for detailed usage examples
- **Modularization Guide**: See `MODULARIZATION_GUIDE.md` for architecture details
- **Implementation Plan**: See `implementation_plan.md` for design decisions

## 🤝 Contributing

This is a reference implementation. To adapt for your use case:

1. Update `config/settings.py` with your thresholds
2. Modify `data/data_loader.py` for your data schema
3. Adjust similarity weights in `similarity/bm25_scorer.py`

## 📄 License

Internal use only.

## 🙏 Acknowledgments

Built as a deterministic alternative to GPT-based claim matching, eliminating hallucination issues while maintaining matching quality.

---

**Need Help?** Check `walkthrough.md` for detailed examples and troubleshooting.
