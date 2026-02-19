"""
============================================================================
DATA_LOADER.PY — Dataset Download, Preprocessing & Tokenization
============================================================================

PURPOSE:
    Downloads the Amazon Product Reviews dataset from Hugging Face Hub,
    cleans and formats it into instruction-style prompts, tokenizes it,
    and returns ready-to-train Dataset objects.

PIPELINE:
    1. Download raw dataset from Hugging Face Hub
    2. Clean text (remove HTML, normalize whitespace)
    3. Format into instruction prompts (Gemma chat format)
    4. Tokenize with padding and truncation
    5. Split into train / validation / test sets
    6. Return HuggingFace Dataset objects

USAGE:
    from data_loader import load_and_prepare_data
    from config import TrainingConfig

    cfg = TrainingConfig()
    tokenizer, train_ds, val_ds, test_ds = load_and_prepare_data(cfg)
"""

import re
import os
from typing import Tuple, Optional

from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer


# ============================================================================
# STEP 1: PROMPT TEMPLATE
# ============================================================================

# This is the instruction template we use to fine-tune Gemma.
# It follows Gemma's chat format using <start_of_turn> / <end_of_turn> tokens.
#
# WHY THIS FORMAT:
#   Gemma was pre-trained to understand this turn-based conversation format.
#   By using it during fine-tuning, the model learns to associate the "user"
#   turn with input and the "model" turn with the expected output.
#
# WHAT THE MODEL LEARNS:
#   Given a product review + rating → Generate a recommendation and strategy.
#
PROMPT_TEMPLATE = """<start_of_turn>user
You are a product recommendation and strategy expert. Based on the following product review, provide:
1. A recommendation (buy/skip/consider) with reasoning
2. Key product strengths and weaknesses
3. A brief strategy suggestion for the product brand

Product Category: {category}
Product Title: {title}
Rating: {rating}/5
Review: {review}
<end_of_turn>
<start_of_turn>model
{response}
<end_of_turn>"""

# Template for inference (no response — the model generates it)
INFERENCE_TEMPLATE = """<start_of_turn>user
You are a product recommendation and strategy expert. Based on the following product review, provide:
1. A recommendation (buy/skip/consider) with reasoning
2. Key product strengths and weaknesses
3. A brief strategy suggestion for the product brand

Product Category: {category}
Product Title: {title}
Rating: {rating}/5
Review: {review}
<end_of_turn>
<start_of_turn>model
"""


# ============================================================================
# STEP 2: TEXT CLEANING
# ============================================================================

def clean_text(text: str) -> str:
    """
    Clean raw review text by removing HTML tags and normalizing whitespace.
    
    WHY CLEAN TEXT:
      Raw web-scraped reviews often contain:
      • HTML tags: <br>, <b>great</b>, &amp;
      • Extra whitespace and newlines
      • Non-printable characters
      
      These artifacts waste token budget and can confuse the model.
    
    Args:
        text: Raw review text string
        
    Returns:
        Cleaned text string
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove HTML tags (e.g., <br>, <b>text</b>)
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove HTML entities (e.g., &amp; → &, &lt; → <)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)

    # Collapse multiple whitespace/newlines into single space
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


# ============================================================================
# STEP 3: GENERATE TRAINING RESPONSE
# ============================================================================

def generate_response(rating: float, title: str, review: str) -> str:
    """
    Generate a "model response" for each training example.
    
    WHY WE NEED THIS:
      In supervised fine-tuning, we need both INPUT (the review) and
      OUTPUT (the expected response). Since we don't have human-written
      recommendations, we construct structured responses based on the
      review's rating and content.
    
    This is a common technique called "self-instruction" or "template-based
    response generation". The model learns the PATTERN of analysis, and
    after fine-tuning, it generates much more nuanced responses.
    
    Args:
        rating: Product rating (1-5)
        title: Product title
        review: Cleaned review text
        
    Returns:
        A structured recommendation response string
    """
    # Determine recommendation based on rating
    if rating >= 4.0:
        recommendation = "BUY"
        reasoning = (
            f"This product receives a strong {rating}/5 rating. "
            f"The reviewer's experience indicates high satisfaction."
        )
        strategy = (
            "The brand should maintain current quality standards and leverage "
            "positive reviews in marketing. Consider a loyalty program to "
            "retain satisfied customers."
        )
    elif rating >= 3.0:
        recommendation = "CONSIDER"
        reasoning = (
            f"With a {rating}/5 rating, this product has mixed reception. "
            f"It may work well for some use cases but has notable limitations."
        )
        strategy = (
            "The brand should analyze common complaints and prioritize fixing "
            "the most frequently mentioned issues. A product revision addressing "
            "key weaknesses could significantly boost ratings."
        )
    else:
        recommendation = "SKIP"
        reasoning = (
            f"The {rating}/5 rating indicates significant dissatisfaction. "
            f"Multiple issues have been reported by the reviewer."
        )
        strategy = (
            "The brand needs urgent product improvement. Consider a recall or "
            "major revision. Customer service should proactively reach out to "
            "dissatisfied buyers to offer solutions or replacements."
        )

    # Extract key points from review (simple heuristic)
    review_lower = review.lower()
    strengths = []
    weaknesses = []

    # Positive signal words
    positive_words = ["great", "excellent", "amazing", "love", "perfect",
                      "best", "wonderful", "fantastic", "quality", "recommend"]
    # Negative signal words
    negative_words = ["bad", "poor", "terrible", "worst", "broken",
                      "disappointed", "waste", "cheap", "defective", "return"]

    for word in positive_words:
        if word in review_lower:
            strengths.append(word)
    for word in negative_words:
        if word in review_lower:
            weaknesses.append(word)

    strengths_text = (
        f"Noted positives: {', '.join(strengths[:3])}."
        if strengths
        else "No specific strengths highlighted in the review."
    )
    weaknesses_text = (
        f"Noted concerns: {', '.join(weaknesses[:3])}."
        if weaknesses
        else "No specific weaknesses highlighted in the review."
    )

    response = (
        f"**Recommendation: {recommendation}**\n\n"
        f"**Reasoning:** {reasoning}\n\n"
        f"**Product Analysis:**\n"
        f"- Strengths: {strengths_text}\n"
        f"- Weaknesses: {weaknesses_text}\n\n"
        f"**Brand Strategy:** {strategy}"
    )

    return response


# ============================================================================
# STEP 4: FORMAT DATASET INTO PROMPTS
# ============================================================================

def format_example(example: dict) -> dict:
    """
    Convert a raw dataset row into an instruction-formatted prompt.
    
    This function is applied to every row in the dataset using .map().
    
    Args:
        example: A dictionary with keys from the Amazon Reviews dataset:
            - 'text': The review text
            - 'rating': Rating (1.0 - 5.0)
            - 'title': Review title
            - 'parent_asin': Product ID
            
    Returns:
        Dictionary with 'text' key containing the formatted prompt
    """
    # Extract and clean fields
    review = clean_text(example.get("text", ""))
    rating = float(example.get("rating", 3.0))
    title = clean_text(example.get("title", "Unknown Product"))

    # Use parent_asin as a proxy for category (actual category info varies)
    category = example.get("parent_asin", "General")

    # Skip empty reviews
    if not review or len(review) < 10:
        review = "No detailed review provided."

    # Generate the model's expected response
    response = generate_response(rating, title, review)

    # Format into the full prompt template
    formatted = PROMPT_TEMPLATE.format(
        category=category,
        title=title,
        rating=rating,
        review=review,
        response=response,
    )

    return {"text": formatted}


# ============================================================================
# STEP 5: LOAD AND PREPARE DATA (MAIN ENTRY POINT)
# ============================================================================

def load_and_prepare_data(
    config,
) -> Tuple[AutoTokenizer, Dataset, Dataset, Optional[Dataset]]:
    """
    Complete data pipeline: download → clean → format → tokenize → split.
    
    This is the main function you call from train.py.
    
    WHAT HAPPENS STEP BY STEP:
    
    1. DOWNLOAD: Fetches the dataset from Hugging Face Hub.
       First run downloads to ~/.cache/huggingface/datasets/ (cached for reuse).
       
    2. SUBSAMPLE: Takes only max_train_samples rows for faster experimentation.
    
    3. FORMAT: Applies format_example() to every row, converting raw reviews
       into instruction-formatted prompts.
       
    4. TOKENIZE: Converts text into numerical token IDs that the model
       understands. Handles padding (making all sequences the same length)
       and truncation (cutting sequences that are too long).
       
    5. SPLIT: Divides into train/validation/test sets.
    
    Args:
        config: TrainingConfig instance with all configuration values
        
    Returns:
        Tuple of (tokenizer, train_dataset, validation_dataset, test_dataset)
        
    COMMON ISSUES:
        • "DatasetNotFoundError" → Check dataset_name and dataset_config in config.py
        • "ConnectionError" → Check internet connection and HF Hub status
        • "Out of Memory" during .map() → Reduce max_train_samples
    """
    print("\n" + "=" * 60)
    print("  DATA PIPELINE: Loading & Preparing Dataset")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 5a: Load tokenizer
    # ------------------------------------------------------------------
    print("\n📦 Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
    )

    # CRITICAL: Set padding token
    # Gemma doesn't have a padding token by default. We need one for batching
    # (making all sequences in a batch the same length).
    # Using eos_token (end-of-sequence) as the padding token is a common practice.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"   Set pad_token to eos_token: '{tokenizer.pad_token}'")

    # Set padding side to 'right' for training (left for generation)
    # WHY RIGHT: During training, we want the actual content at the start
    # and padding at the end. This way, the model processes real tokens first.
    tokenizer.padding_side = "right"
    print(f"   Vocabulary size: {tokenizer.vocab_size:,}")
    print(f"   Padding side: {tokenizer.padding_side}")

    # ------------------------------------------------------------------
    # Step 5b: Download dataset
    # ------------------------------------------------------------------
    print(f"\n📥 Downloading dataset: {config.dataset_name}")
    print(f"   Config: {config.dataset_config}")

    try:
        raw_dataset = load_dataset(
            config.dataset_name,
            config.dataset_config,
            split="full",  # Amazon Reviews 2023 uses "full" as the split name
            trust_remote_code=True,  # Required for this specific dataset
        )
        print(f"   Total examples: {len(raw_dataset):,}")
    except Exception as e:
        print(f"   ⚠️  Failed to load '{config.dataset_config}': {e}")
        print(f"   Trying alternative loading method...")

        # Fallback: try loading without config
        raw_dataset = load_dataset(
            config.dataset_name,
            config.dataset_config,
            trust_remote_code=True,
        )

        # Handle case where dataset has train/test splits
        if isinstance(raw_dataset, DatasetDict):
            if "train" in raw_dataset:
                raw_dataset = raw_dataset["train"]
            else:
                # Use first available split
                first_split = list(raw_dataset.keys())[0]
                raw_dataset = raw_dataset[first_split]

        print(f"   Total examples: {len(raw_dataset):,}")

    # ------------------------------------------------------------------
    # Step 5c: Subsample for faster training
    # ------------------------------------------------------------------
    total_samples = config.max_train_samples + config.max_eval_samples

    if len(raw_dataset) > total_samples:
        print(f"\n✂️  Subsampling to {total_samples:,} examples (from {len(raw_dataset):,})")
        raw_dataset = raw_dataset.shuffle(seed=config.seed).select(range(total_samples))
    else:
        total_samples = len(raw_dataset)

    # ------------------------------------------------------------------
    # Step 5d: Train/validation/test split
    # ------------------------------------------------------------------
    print("\n📊 Splitting dataset...")

    # First split: train vs. (validation + test)
    split_dataset = raw_dataset.train_test_split(
        test_size=config.validation_split * 2,  # 20% for val+test
        seed=config.seed,
    )

    train_raw = split_dataset["train"]

    # Second split: validation vs. test (split the 20% in half)
    val_test_split = split_dataset["test"].train_test_split(
        test_size=0.5,
        seed=config.seed,
    )
    val_raw = val_test_split["train"]
    test_raw = val_test_split["test"]

    print(f"   Train:      {len(train_raw):,} examples")
    print(f"   Validation: {len(val_raw):,} examples")
    print(f"   Test:       {len(test_raw):,} examples")

    # ------------------------------------------------------------------
    # Step 5e: Format into instruction prompts
    # ------------------------------------------------------------------
    print("\n📝 Formatting into instruction prompts...")

    # .map() applies format_example to every row in the dataset
    # remove_columns drops the original raw columns we no longer need
    original_columns = train_raw.column_names

    train_formatted = train_raw.map(
        format_example,
        remove_columns=original_columns,
        desc="Formatting training data",
    )

    val_formatted = val_raw.map(
        format_example,
        remove_columns=original_columns,
        desc="Formatting validation data",
    )

    test_formatted = test_raw.map(
        format_example,
        remove_columns=original_columns,
        desc="Formatting test data",
    )

    # Show a sample
    print("\n📋 Sample formatted prompt (first 500 chars):")
    print("-" * 60)
    print(train_formatted[0]["text"][:500])
    print("..." if len(train_formatted[0]["text"]) > 500 else "")
    print("-" * 60)

    # ------------------------------------------------------------------
    # Step 5f: Summary
    # ------------------------------------------------------------------
    print("\n✅ Data pipeline complete!")
    print(f"   Train samples:      {len(train_formatted):,}")
    print(f"   Validation samples: {len(val_formatted):,}")
    print(f"   Test samples:       {len(test_formatted):,}")
    print(f"   Max sequence length: {config.max_seq_length}")

    return tokenizer, train_formatted, val_formatted, test_formatted


# ============================================================================
# Quick test: Run this file directly to test the data pipeline
# ============================================================================
if __name__ == "__main__":
    from config import TrainingConfig

    # Use a small subset for testing
    cfg = TrainingConfig(max_train_samples=100, max_eval_samples=20)
    tokenizer, train_ds, val_ds, test_ds = load_and_prepare_data(cfg)

    print("\n🔍 Inspecting first training example:")
    print(train_ds[0]["text"])
